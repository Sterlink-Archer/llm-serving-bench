#!/usr/bin/env python3
"""Matrix runner: one spec (bench/presets/*.json) -> many traffic.py cells against one live endpoint, then a summary
with the knee per series. Stdlib only; runs inside the sandbox next to the server, or anywhere for a dry run.

Spec:
  {"name": "core", "defaults": {"duration": 60, "warmup": 10, ...traffic.py options...},
   "slo": {"ttft_p95_s": 2.0, "ttft_p95_s_long": 6.0, "long_input_tokens": 4000, "tpot_p50_ms": 50},
   "series": [{"id": "s1k1k", "workload": "synthetic", "params": {"input_tokens": 1024, "output_tokens": 1024},
               "sweep": {"concurrency": [1, 16, 64]}, "stop_past_knee": true, "duration": 60}, ...]}
  sweep keys: "concurrency", "rate", "max_inflight", or "params.<name>" (overrides one workload param). Several keys =
  their Cartesian product, in the listed order. Anything else in a series overrides the defaults for that series.

Knee report, per series that sweeps concurrency (or rate):
  peak        highest output tok/s in the series;
  optimal     smallest C (rate) reaching >= 95% of peak: the most throughput before latency is spent for nothing;
  slo_best    highest output tok/s among cells meeting the SLO (TTFT p95, TPOT p50);
  for rate series also: max_sustained = highest offered rate whose achieved req/s >= 95% of it, no drops, SLO met.
stop_past_knee: after a cell that adds < 3% throughput over the best so far while TTFT p95 doubled, the series' higher
levels are skipped (recorded as skipped) to save GPU time.

  python3 bench/matrix.py --spec bench/presets/core.json --model gemma --out out/matrix_core [--metrics URL]
         [--usd-per-hour 3.67] [--max-minutes 45] [--only REGEX] [--scale-duration 0.2] [--list]
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traffic as T  # noqa: E402
import workloads as W  # noqa: E402

RUN_KEYS = {"concurrency", "rate", "max_inflight", "duration", "warmup", "max_requests", "temperature", "extra", "seed", "no_think"}


def cells_of(spec):
    out = []
    for s in spec["series"]:
        sweep = s.get("sweep", {})
        keys = list(sweep)
        for combo in itertools.product(*[sweep[k] for k in keys]) if keys else [()]:
            c = dict(series=s["id"], workload=s.get("workload", "synthetic"), params=json.loads(json.dumps(s.get("params", {}))),
                     run={k: v for k, v in {**spec.get("defaults", {}), **s}.items() if k in RUN_KEYS},
                     stop_past_knee=s.get("stop_past_knee", False), sweep_keys=keys)
            tag = []
            for k, v in zip(keys, combo):
                if k.startswith("params."):
                    c["params"][k[7:]] = v
                else:
                    c["run"][k] = v
                tag.append(f"{k.replace('params.', '')}={v}")
            c["id"] = s["id"] + ("__" + "_".join(tag) if tag else "")
            out.append(c)
    return out


def est_s(c, scale):
    return c["run"].get("duration", 60) * scale + 8


def cell_args(a, c, cal, scale):
    p = T.parser()
    argv = ["--model", a.model, "--base", a.base, "--workload", c["workload"], "--params", json.dumps(c["params"]),
            "--out", os.path.join(a.out, "cells", c["id"] + ".json"), "--label", c["id"],
            "--cpt", str(cal["cpt_prose"]), "--cpt-table", str(cal["cpt_table"])]
    if a.metrics:
        argv += ["--metrics", a.metrics]
    r = c["run"]
    for k in ("concurrency", "rate", "max_inflight", "max_requests", "temperature", "seed"):
        if k in r:
            argv += [f"--{k.replace('_', '-')}", str(r[k])]
    argv += ["--duration", str(r.get("duration", 60) * scale)]
    if "warmup" in r:
        argv += ["--warmup", str(r["warmup"] * scale)]
    if r.get("extra"):
        argv += ["--extra", json.dumps(r["extra"]) if not isinstance(r["extra"], str) else r["extra"]]
    if r.get("no_think") is False:
        argv += ["--think"]
    return T.finish_args(p.parse_args(argv))


def slo_ok(s, slo, long_in):
    ttft = (s.get("ttft_s") or {}).get("p95")
    tpot = (s.get("tpot_s") or {}).get("p50")
    lim = slo.get("ttft_p95_s_long", 6.0) if long_in else slo.get("ttft_p95_s", 2.0)
    return ttft is not None and tpot is not None and ttft <= lim and tpot * 1000 <= slo.get("tpot_p50_ms", 50) and not s["errors"]


def row(c, d, usd_h):
    s = d["summary"]
    r = dict(cell=c["id"], series=c["series"], workload=c["workload"], mode=s["mode"], C=s["concurrency"], rate=s["offered_rate"],
             in_p50=(s["prompt_tokens"] or {}).get("p50"), out_p50=(s["completion_tokens"] or {}).get("p50"),
             req_s=s["req_per_s"], out_tok_s=s["output_tok_per_s"], in_tok_s=s["input_tok_per_s"], cached_in_tok_s=s["cached_input_tok_per_s"],
             cache_hit=s["cache_hit_ratio"], srv_out_tok_s=(s.get("server_metrics") or {}).get("gen_tok_per_s"),
             srv_pc_hit=(s.get("server_metrics") or {}).get("prefix_cache_hit_ratio"),
             spec_accept=(s.get("server_metrics") or {}).get("spec_acceptance_rate"),
             preemptions=(s.get("server_metrics") or {}).get("preemptions"),
             ttft_p50=(s["ttft_s"] or {}).get("p50"), ttft_p95=(s["ttft_s"] or {}).get("p95"), ttft_p99=(s["ttft_s"] or {}).get("p99"),
             tpot_p50=(s["tpot_s"] or {}).get("p50"), tpot_p95=(s["tpot_s"] or {}).get("p95"), itl_p99=(s["itl_s"] or {}).get("p99"),
             e2e_p50=(s["e2e_s"] or {}).get("p50"), e2e_p95=(s["e2e_s"] or {}).get("p95"),
             ttft_cold_p50=(s["ttft_cold_prefix_s"] or {}).get("p50"), ttft_warm_p50=(s["ttft_warm_prefix_s"] or {}).get("p50"),
             errors=s["errors"], dropped=s["dropped"], completed=s["completed"], peak_inflight=s["peak_inflight"])
    fixed = c["workload"] in ("synthetic", "longdoc_qa") and c["params"].get("ignore_eos", True)
    if fixed:   # every request yields exactly output_tokens, so the steady request rate is the token rate / that
        r["req_s"] = s["output_tok_per_s"] / c["params"].get("output_tokens", 256)
        r["req_s_basis"] = "out tok/s / output_tokens"
    else:
        r["req_s_basis"] = "completed in window"
    if usd_h and s["output_tok_per_s"]:
        r["usd_per_m_out_all"] = usd_h / 3600 / s["output_tok_per_s"] * 1e6
        r["usd_per_req"] = (usd_h / 3600 / r["req_s"]) if r["req_s"] else None
    return r


def knees(rows, slo):
    out = {}
    for sid in sorted({r["series"] for r in rows}):
        rs = [r for r in rows if r["series"] == sid and not r.get("skipped")]
        if len(rs) < 2:
            continue
        by = "rate" if rs[0]["mode"] == "open" else "C"
        if len({r[by] for r in rs}) < 2:
            continue
        rs.sort(key=lambda r: r[by])
        peak = max(rs, key=lambda r: r["out_tok_s"])
        opt = next(r for r in rs if r["out_tok_s"] >= 0.95 * peak["out_tok_s"])
        ok = [r for r in rs if r.get("slo_ok")]
        k = dict(by=by, peak={by: peak[by], "out_tok_s": peak["out_tok_s"]}, optimal={by: opt[by], "out_tok_s": opt["out_tok_s"],
                 "ttft_p95": opt["ttft_p95"], "tpot_p50": opt["tpot_p50"]},
                 slo_best=({by: max(ok, key=lambda r: r["out_tok_s"])[by], "out_tok_s": max(r["out_tok_s"] for r in ok)} if ok else None))
        if by == "rate":
            sus = [r for r in rs if r["req_s"] >= 0.95 * r["rate"] and not r["dropped"] and r.get("slo_ok")]
            k["max_sustained_rate"] = max((r["rate"] for r in sus), default=None)
        out[sid] = k
    return out


def fmt(x, n=2):
    return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def md(spec, meta, rows, kn):
    L = [f"# Matrix `{spec['name']}`", "", f"endpoint model `{meta['model']}`, started {meta['t_start']}, calibration {json.dumps(meta['calibration'])}",
         f"SLO: {json.dumps(spec.get('slo', {}))}", "",
         "| cell | mode | in p50 | out p50 | req/s | out tok/s (server) | in tok/s | cache hit (server) | TTFT p50/p95/p99 s | TPOT p50/p95 ms | cold/warm TTFT p50 s | $/1M out | err | SLO |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("skipped"):
            L.append(f"| {r['cell']} | skipped: {r['skipped']} |||||||||||||")
            continue
        mode = f"R={r['rate']}" if r["mode"] == "open" else f"C={r['C']}"
        L.append(f"| {r['cell']} | {mode} | {fmt(r['in_p50'], 0)} | {fmt(r['out_p50'], 0)} | {r['req_s']:.2f} | {r['out_tok_s']:.0f} ({fmt(r['srv_out_tok_s'], 0)}) | "
                 f"{r['in_tok_s']:.0f} | {fmt(r['cache_hit'])} ({fmt(r['srv_pc_hit'])}) | {fmt(r['ttft_p50'])}/{fmt(r['ttft_p95'])}/{fmt(r['ttft_p99'])} | "
                 f"{fmt((r['tpot_p50'] or 0) * 1000, 1)}/{fmt((r['tpot_p95'] or 0) * 1000, 1)} | {fmt(r['ttft_cold_p50'])}/{fmt(r['ttft_warm_p50'])} | "
                 f"{fmt(r.get('usd_per_m_out_all'), 3)} | {r['errors']} | {'ok' if r.get('slo_ok') else '-'} |")
    L += ["", "## Knees", "", "| series | by | peak out tok/s @ | optimal (>=95% of peak) @ | SLO-best @ | max sustained rate |", "|---|---|---|---|---|---|"]
    for sid, k in kn.items():
        by = k["by"]
        L.append(f"| {sid} | {by} | {k['peak']['out_tok_s']:.0f} @ {k['peak'][by]} | {k['optimal']['out_tok_s']:.0f} @ {k['optimal'][by]} | "
                 f"{(fmt(k['slo_best']['out_tok_s'], 0) + ' @ ' + str(k['slo_best'][by])) if k['slo_best'] else 'none'} | {fmt(k.get('max_sustained_rate'))} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--metrics", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--usd-per-hour", type=float, default=0.0)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--scale-duration", type=float, default=1.0)
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    cells = [c for c in cells_of(spec) if not a.only or re.search(a.only, c["id"])]
    total = sum(est_s(c, a.scale_duration) for c in cells)
    if a.list:
        for c in cells:
            print(f"{c['id']:60s} {c['workload']:13s} run={json.dumps(c['run'])} params={json.dumps(c['params'])}")
        print(f"{len(cells)} cells, ~{total / 60:.1f} min")
        return 0
    os.makedirs(os.path.join(a.out, "cells"), exist_ok=True)
    t0 = time.time()
    if a.no_calibrate:
        cal = dict(cpt_prose=4.0, cpt_table=3.0, given=True)
    else:
        ns = T.finish_args(T.parser().parse_args(["--model", a.model, "--base", a.base, "--out", "/dev/null"]))
        cal = T.calibrate(ns, W.Text())
    meta = dict(spec=a.spec, spec_name=spec["name"], model=a.model, base=a.base, t_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                calibration=cal, usd_per_hour=a.usd_per_hour, scale_duration=a.scale_duration, cells_planned=len(cells), est_minutes=total / 60)
    print(f"[matrix {spec['name']}] {len(cells)} cells, ~{total / 60:.1f} min; calibration {cal}", flush=True)
    rows, slo = [], spec.get("slo", {})
    stopped = set()
    best = {}
    for c in cells:
        if c["series"] in stopped:
            rows.append(dict(cell=c["id"], series=c["series"], skipped="past knee"))
            continue
        if a.max_minutes and (time.time() - t0 + est_s(c, a.scale_duration)) / 60 > a.max_minutes:
            rows.append(dict(cell=c["id"], series=c["series"], skipped="time budget"))
            continue
        ns = cell_args(a, c, cal, a.scale_duration)
        try:
            d = asyncio.run(T.run(ns))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(cell=c["id"], series=c["series"], skipped=f"error {e!r}"[:200]))
            print(f"[{c['id']}] ERROR {e!r}", flush=True)
            continue
        json.dump(d, open(ns.out, "w"), indent=1)
        print(T.line(d), flush=True)
        r = row(c, d, a.usd_per_hour)
        long_in = (r["in_p50"] or 0) > slo.get("long_input_tokens", 4000)
        r["slo_ok"] = slo_ok(d["summary"], slo, long_in)
        rows.append(r)
        if c["stop_past_knee"]:
            prev = best.get(c["series"])
            if prev and r["out_tok_s"] < 1.03 * prev["out_tok_s"] and (r["ttft_p95"] or 0) > 2 * (prev["ttft_p95"] or 1e9):
                stopped.add(c["series"])
            if not prev or r["out_tok_s"] > prev["out_tok_s"]:
                best[c["series"]] = r
        json.dump(dict(meta=meta, rows=rows), open(os.path.join(a.out, "matrix_summary.json"), "w"), indent=1)
    kn = knees(rows, slo)
    meta["t_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["wall_minutes"] = (time.time() - t0) / 60
    json.dump(dict(meta=meta, rows=rows, knees=kn), open(os.path.join(a.out, "matrix_summary.json"), "w"), indent=1)
    open(os.path.join(a.out, "matrix_summary.md"), "w").write(md(spec, meta, rows, kn))
    print(f"[matrix {spec['name']}] done in {meta['wall_minutes']:.1f} min -> {a.out}/matrix_summary.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
