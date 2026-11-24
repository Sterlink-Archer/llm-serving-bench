#!/usr/bin/env python3
"""Price every measured matrix cell against the bars.

Primary bar: OpenRelay's own published price for Gemma 4 31B (research/openrelay_prices_<stamp>.json, parsed from the
saved https://openrelay.inc/inference-pricing page): Sync, Sync with cached input, and Batch.
Second reference: the cheapest live OpenRouter provider (research/openrouter_bar_<model>_<stamp>.json; Modal excluded).

Inputs: results/<label>/<stamp>/{config.json, m_<server>_<preset>/cells/*.json} written by run.py + bench/matrix.py.

Definitions (also in WRITEUP.md):
  box $/h          GPU spot price x count + vCPU + RAM + disk (infra/daytona.hourly_usd, Daytona pricing page).
  req/s            fixed-shape cells (ignore_eos): output tok/s in the steady window / output tokens per request;
                   natural-length cells: requests completed inside the window / window.
  tokens/request   mean server-reported prompt tokens; output = the fixed length, or the mean completion tokens.
  our $/request    box $/h / 3600 / req/s: the box saturated at that load (100% utilisation).
  bar $/request    in x input price + out x output price at the bar's rates. "cached" applies OpenRelay's cached-input
                   rate to the share of input the server reported as cached (assumes their cache would hit as often).
  ratio            our / bar. < 1 = we are cheaper. It is also the break-even utilisation: the share of wall time the box
                   must be serving at that load to match the bar (an idle box still bills).
SLO "interactive": TTFT p95 <= 2 s (<= 6 s when input > 4k tokens) and TPOT p50 <= 50 ms (>= 20 tok/s per user).

  python3 -I analysis/cost.py [--results results] [--md out.md] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "infra"))
import daytona as dt  # noqa: E402
sys.path.insert(0, str(ROOT / "bench"))
import pricebar as PB  # noqa: E402  (the one rule for which bar a config is priced against)
sys.path.insert(0, str(ROOT / "analysis"))
import runfiles as RF  # noqa: E402  (which protocol dirs belong to which server)

MODEL_OPENRELAY = "openrelay/gemma-4-31b"
# OpenRelay's own GPU rental list prices ($/GPU-hour, host + default disk included), https://openrelay.inc/gpus saved
# research/raw/openrelay/gpus_20261122T032357Z.html: "what this config would cost on OpenRelay's own rented GPUs".
OPENRELAY_GPU_USD_H = {"H100": 2.60, "B200": 6.30, "B300": 8.49, "MI355X": 6.30}


def latest(pattern):
    xs = sorted(glob.glob(str(ROOT / pattern)))
    return xs[-1] if xs else None


def openrouter_cheapest(bar, i, o, exclude=("Modal",)):
    best = None
    for e in bar["endpoints"]:
        if (e["status"] or 0) < 0 or e["provider"] in exclude:
            continue
        c = (i * e["in_usd_per_m"] + o * e["out_usd_per_m"]) / 1e6
        if best is None or c < best[0]:
            best = (c, e)
    return best


def tier_cost(t, i, o, cached=0.0):
    return ((i - cached) * t["in_usd_per_m"] + cached * (t.get("cached_in_usd_per_m") or t["in_usd_per_m"]) + o * t["out_usd_per_m"]) / 1e6


def cells(results_dir):
    for cfgf in sorted(glob.glob(f"{results_dir}/*/*/config.json")):
        rd = pathlib.Path(cfgf).parent
        cfg = json.loads(pathlib.Path(cfgf).read_text())
        snap = cfg["snapshot"]
        pph = dt.hourly_usd(snap["gpu"], snap.get("gpus", 1), snap["cpu"], snap["mem"], snap["disk"])["total"]
        for mdir in sorted(list(rd.glob("m_*")) + list(rd.glob("out/m_*")) + list(rd.glob("protocol_*"))):   # out/m_*: the pilot's layout
            if not mdir.is_dir():
                continue
            pre = "protocol_" if mdir.name.startswith("protocol_") else "m_"
            names = [s["name"] for s in cfg["servers"]]
            if pre == "protocol_":
                # the shared rule (analysis/runfiles.py): run.json steps first, then the longest server prefix. A prefix-only
                # match left Inkling's protocol_b300_* / protocol_rep_* cells with server None and crashed report.py.
                run = json.loads((rd / "run.json").read_text()) if (rd / "run.json").exists() else {}
                server = next((n for n in names if mdir in RF.protocol_dirs(rd, n, names, run)), None)
            else:
                server = max((n for n in names if mdir.name.startswith(pre + n)), key=len, default=None)
            for cf in sorted((mdir / "cells").glob("*.json")):
                if pre == "protocol_":   # only our client's cells (schema traffic/1); vllm-bench rungs are priced in protocol_rows.json
                    try:
                        if json.loads(cf.read_text()).get("schema") != "traffic/1":
                            continue
                    except (json.JSONDecodeError, OSError):
                        continue
                yield rd, cfg, snap, pph, mdir.name, server, cf


def rows(results_dir, orl, orb):
    out = []
    for rd, cfg, snap, pph, mname, server, cf in cells(results_dir):
        bt = PB.bar_tiers(cfg, orl)           # the config's own bar: price_model / price_bar / default Gemma 4 31B
        sync, batch = bt["sync"], bt["batch"]
        d = json.loads(cf.read_text())
        s, p = d["summary"], d.get("params") or {}
        if not s.get("output_tok_per_s"):
            continue
        reqs = [r for r in d["requests"] if r.get("prompt_tokens") and r.get("status") == 200]
        mean_in = sum(r["prompt_tokens"] for r in reqs) / len(reqs) if reqs else 0
        mean_cached = sum(r.get("cached_tokens") or 0 for r in reqs) / len(reqs) if reqs else 0
        fixed_out = d["args"]["workload"] in ("synthetic", "longdoc_qa") and p.get("ignore_eos", True)
        if fixed_out:
            o = p.get("output_tokens", 256)
            rps = s["output_tok_per_s"] / o
        else:
            o = (s.get("completion_tokens") or {}).get("mean") or 0
            rps = s["req_per_s"]
        our = pph / 3600 / rps if rps else None
        b_sync, b_cached = tier_cost(sync, mean_in, o), tier_cost(sync, mean_in, o, mean_cached)
        b_batch = tier_cost(batch, mean_in, o, mean_cached) if batch else None
        orc = openrouter_cheapest(orb, mean_in, o) if bt["is_openrelay_gemma"] else None   # the OpenRouter bar file is Gemma 4 31B
        ttft95, tpot50 = (s.get("ttft_s") or {}).get("p95"), (s.get("tpot_s") or {}).get("p50")
        r = dict(run=rd.parent.name, stamp=rd.name, matrix=mname, server=server, cell=d["label"], gpu=f"{snap.get('gpus', 1)}x{snap['gpu']}",
                 engine=snap["image"].split("/")[-1], workload=d["args"]["workload"], mode=s["mode"], C=s.get("concurrency"), rate=s.get("offered_rate"),
                 in_mean=mean_in, cached_mean=mean_cached, out_per_req=o, req_s=rps, out_tok_s=s["output_tok_per_s"],
                 srv_out_tok_s=(s.get("server_metrics") or {}).get("gen_tok_per_s"), ttft_p50=(s.get("ttft_s") or {}).get("p50"),
                 ttft_p95=ttft95, tpot_p50=tpot50, errors=s["errors"], usd_h=pph, our_usd_req=our,
                 openrelay_sync=b_sync, openrelay_sync_cached=b_cached, openrelay_batch=b_batch,
                 price_bar_name=bt["name"], price_model=bt["model"], price_bar_is_openrelay_gemma=bt["is_openrelay_gemma"],
                 openrouter=orc[0] if orc else None, openrouter_who=f"{orc[1]['provider']} ({orc[1]['quant']})" if orc else None,
                 usd_per_m_out_all=pph / 3600 / s["output_tok_per_s"] * 1e6, file=str(cf.relative_to(ROOT)) if cf.is_relative_to(ROOT) else str(cf))
        alt = OPENRELAY_GPU_USD_H.get(snap["gpu"])
        if alt and rps and b_sync:
            r["usd_h_openrelay_rental"] = alt * snap.get("gpus", 1)
            r["ratio_sync_on_openrelay_rental"] = (r["usd_h_openrelay_rental"] / 3600 / rps) / b_sync
        if our:
            r["ratio_sync"] = our / b_sync
            r["ratio_sync_cached"] = our / b_cached
            r["ratio_batch"] = our / b_batch if b_batch else None
            r["ratio_openrouter"] = our / orc[0] if orc else None
            r["quote_in_usd_m"] = r["ratio_sync"] * sync["in_usd_per_m"]
            r["quote_out_usd_m"] = r["ratio_sync"] * sync["out_usd_per_m"]
        lim = 6.0 if mean_in > 4000 else 2.0
        r["slo_interactive"] = bool(ttft95 is not None and tpot50 is not None and ttft95 <= lim and tpot50 <= 0.050 and not s["errors"])
        out.append(r)
    return out


def fmt(x, n=2):
    return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--openrelay", default=latest("research/openrelay_prices_*.json"))
    ap.add_argument("--openrouter", default=latest("research/openrouter_bar_google_gemma-4-31b-it_*.json"))
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    orl = json.loads(pathlib.Path(a.openrelay).read_text())
    orb = json.loads(pathlib.Path(a.openrouter).read_text())
    rs = rows(a.results, orl, orb)
    t = orl["models"][MODEL_OPENRELAY]["tiers"]
    L = [f"Bars: OpenRelay `{MODEL_OPENRELAY}` Sync ${t['Sync']['in_usd_per_m']} in / ${t['Sync']['cached_in_usd_per_m']} cached / "
         f"${t['Sync']['out_usd_per_m']} out per 1M, Batch half ({pathlib.Path(a.openrelay).name}); OpenRouter cheapest live provider "
         f"({pathlib.Path(a.openrouter).name}, Modal excluded). ratio < 1 = we are cheaper = break-even utilisation. A config with "
         f"`price_model` or `price_bar` (GLM, Inkling) is priced against ITS bar, named in the bar column; its OpenRouter column is blank.", "",
         "| run / server | gpu | cell | in / cached / out per req | req/s | out tok/s (server) | TTFT p95 s | TPOT p50 ms | $/h | our $/req | vs OR-Sync | vs OR-Sync cached | vs OR-Batch | vs OpenRouter | vs OR-Sync on OR's own GPU rental | bar | SLO |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        L.append(f"| {r['run']} / {r['server']} | {r['gpu']} | {r['cell']} | {r['in_mean']:.0f} / {r['cached_mean']:.0f} / {r['out_per_req']:.0f} | "
                 f"{r['req_s']:.2f} | {r['out_tok_s']:.0f} ({fmt(r['srv_out_tok_s'], 0)}) | {fmt(r['ttft_p95'])} | {fmt((r['tpot_p50'] or 0) * 1000, 1)} | "
                 f"{r['usd_h']:.2f} | {fmt(r['our_usd_req'], 6)} | {fmt(r.get('ratio_sync'))} | {fmt(r.get('ratio_sync_cached'))} | "
                 f"{fmt(r.get('ratio_batch'))} | {fmt(r.get('ratio_openrouter'))} | {fmt(r.get('ratio_sync_on_openrelay_rental'))} | {r['price_bar_name']} | {'interactive' if r['slo_interactive'] else 'batch'} |")
    txt = "\n".join(L)
    print(txt)
    if a.md:
        pathlib.Path(a.md).write_text(txt + "\n")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rs, indent=1))


if __name__ == "__main__":
    main()
