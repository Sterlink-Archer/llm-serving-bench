#!/usr/bin/env python3
"""The Phase 2 paired table for the GEMM-headroom track, read from the mxG2_* runs' own files: on each box the same GPU
served our recipe (server mx_recipe) and our recipe + MX_PS_FP4=1 (server mx_ps: M >= 2048 MXFP4 GEMMs on aiter
v0.1.30's persistent 256x256 kernel). Their client's rungs, prefill probe, GSM8K gate, MTP acceptance, the box's GFX
clock under load, and whether the ps arm's EngineCore actually routed GEMMs (server log lines from mx_ps_fp4).

  python3 -I recipes/mi355x/tools/ps_paired_table.py [tag=G2]
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import gpumon_amd  # noqa: E402
import screen  # noqa: E402

ARMS = (("mx_recipe", "ours"), ("mx_ps", "ours+ps"))


def fmt(x, n=2):
    return "-" if x is None else f"{x:.{n}f}"


def routed(rd):
    p = rd / "server_mx_ps.log"
    if not p.exists():
        return "no log"
    t = p.read_text(errors="replace")
    first = len(re.findall(r"mx_ps_fp4 .*routed first", t))
    counts = [int(x) for x in re.findall(r"routed (\d+) GEMMs so far", t)]
    fb = len(re.findall(r"mx_ps_fp4 .*fallback", t))
    dis = "DISABLED" in t
    return f"{first} shapes routed, >= {max(counts) if counts else 0} launches logged, {fb} fallback kinds" + (", DISABLED" if dis else "")


def main(tag="G2"):
    rows, notes = [], []
    for rd in sorted(glob.glob(str(ROOT / "results" / f"mx{tag}_*" / "2*"))):
        rd = pathlib.Path(rd)
        run = f"{rd.parent.name}/{rd.name}"
        _, _, clk = gpumon_amd.summarise(rd / "gpumon.csv") if (rd / "gpumon.csv").exists() else (0, None, None)
        notes.append(f"- {run}: ps arm {routed(rd)}")
        for srv, who in ARMS:
            g = sorted(rd.glob(f"gate_{srv}_*.json"))
            gate = json.loads(g[0].read_text()).get("gate") if g else None
            al, _ = screen.prom(rd / f"metrics_{srv}.prom")
            for ps in sorted(rd.glob(f"protocol_{srv}/protocol_summary.json")):
                s = json.loads(ps.read_text())
                pre = (s.get("prefill") or {}).get("prefill_tok_s")
                for r in s.get("ladder", []):
                    for sd, x in (r.get("results") or {}).items():
                        rows.append(dict(run=run, who=who, seed=int(sd), rate=r["rate"], p95=x.get("ttft_p95_s"), p50=x.get("ttft_p50_s"),
                                         tpot=x.get("tpot_p50_s"), ok=x.get("slo_pass"), err=x.get("errors"), pre=pre,
                                         gsm=gate and f"{gate.get('passed')}/{gate.get('n')}", accept=al, clk=clk))
    if not rows:
        print(f"no results for mx{tag}_*")
        return
    print("| box (run) | GFX MHz | engine | seed | rate | p95 TTFT s | p50 TTFT s | TPOT p50 ms | pass | errors | prefill tok/s | GSM8K | accept/step |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in sorted(rows, key=lambda r: (r["seed"], r["run"], r["rate"], r["who"])):
        print(f"| {x['run']} | {x['clk'] and round(x['clk'])} | {x['who']} | {x['seed']} | {x['rate']} | {fmt(x['p95'])} | {fmt(x['p50'])} | "
              f"{fmt((x['tpot'] or 0) * 1000, 1) if x['tpot'] is not None else '-'} | {'PASS' if x['ok'] else 'fail'} | {x['err']} | "
              f"{x['pre'] and round(x['pre'])} | {x['gsm']} | {fmt(x['accept'])} |")
    print()
    print("\n".join(notes))
    print()
    for who in ("ours", "ours+ps"):
        rs = [r for r in rows if r["who"] == who]
        okr = [rate for rate in sorted({r["rate"] for r in rs}) if all(r["ok"] for r in rs if r["rate"] == rate)]
        pre = [r["pre"] for r in rs if r["pre"]]
        print(f"- {who}: passes on every box and seed up to {okr[-1] if okr else 'none'}; prefill probe "
              f"{', '.join(str(round(p)) for p in sorted(set(pre)))} tok/s")
    # paired deltas per box and rate
    print()
    for run in sorted({r["run"] for r in rows}):
        for rate in sorted({r["rate"] for r in rows if r["run"] == run}):
            a = next((r for r in rows if r["run"] == run and r["rate"] == rate and r["who"] == "ours"), None)
            b = next((r for r in rows if r["run"] == run and r["rate"] == rate and r["who"] == "ours+ps"), None)
            if a and b and a["p95"] and b["p95"]:
                print(f"- {run} @ {rate}: p95 TTFT {a['p95']:.2f} -> {b['p95']:.2f} s; TPOT p50 "
                      f"{fmt((a['tpot'] or 0) * 1000, 1)} -> {fmt((b['tpot'] or 0) * 1000, 1)} ms; prefill {a['pre'] and round(a['pre'])} -> "
                      f"{b['pre'] and round(b['pre'])} tok/s")


if __name__ == "__main__":
    main(*sys.argv[1:])
