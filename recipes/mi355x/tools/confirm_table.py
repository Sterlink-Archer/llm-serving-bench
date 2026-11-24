#!/usr/bin/env python3
"""The paired confirm table, read from the confirm runs' own files (never typed): for every mx<tag>_* box, every server
(ours = mx_recipe, stacked = mx_stack (C2: + MX_PS_FP4), theirs = or_recipe) with their client's rungs (column A), our real-text client's rungs (column B),
prefill probe, GSM8K gate, MTP acceptance, and the box's median GFX clock under load.

  python3 -I recipes/mi355x/tools/confirm_table.py [tag=C1]   # markdown to stdout
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


SERVERS = (("mx_recipe", "ours"), ("mx_stack", "ours+ps"), ("or_recipe", "theirs"))


def fmt(x, n=2):
    return "-" if x is None else f"{x:.{n}f}"


def main(tag="C1"):
    rows = []
    for rd in sorted(glob.glob(str(ROOT / "results" / f"mx{tag}_*" / "2*"))):
        rd = pathlib.Path(rd)
        run = f"{rd.parent.name}/{rd.name}"
        _, _, clk = gpumon_amd.summarise(rd / "gpumon.csv") if (rd / "gpumon.csv").exists() else (0, None, None)
        for srv, who in SERVERS:
            g = sorted(rd.glob(f"gate_{srv}_*.json"))
            gate = json.loads(g[0].read_text()).get("gate") if g else None
            al, ar = screen.prom(rd / f"metrics_{srv}.prom")
            for ps in sorted(rd.glob(f"protocol_{srv}*/protocol_summary.json")):
                s = json.loads(ps.read_text())
                col = "A" if s.get("client") == "vllmbench" else "B"
                pre = (s.get("prefill") or {}).get("prefill_tok_s")
                for r in s.get("ladder", []):
                    for sd, x in (r.get("results") or {}).items():
                        rows.append(dict(run=run, who=who, col=col, seed=int(sd), rate=r["rate"], p95=x.get("ttft_p95_s"),
                                         p50=x.get("ttft_p50_s"), tpot=x.get("tpot_p50_s"), ok=x.get("slo_pass"), err=x.get("errors"),
                                         ach=x.get("achieved_rps"), pre=pre if col == "A" else None,
                                         gsm=gate and f"{gate.get('passed')}/{gate.get('n')}", accept=al, clk=clk))
    if not rows:
        print(f"no results for mx{tag}_*")
        return
    print(f"| box (run) | GFX MHz | engine | column | seed | rate | p95 TTFT s | p50 TTFT s | TPOT p50 ms | pass | errors | prefill tok/s | GSM8K | accept tokens/step (run) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in sorted(rows, key=lambda r: (r["col"], r["seed"], r["run"], r["rate"], r["who"])):
        print(f"| {x['run']} | {x['clk'] and round(x['clk'])} | {x['who']} | {x['col']} | {x['seed']} | {x['rate']} | {fmt(x['p95'])} | "
              f"{fmt(x['p50'])} | {fmt((x['tpot'] or 0) * 1000, 1) if x['tpot'] is not None else '-'} | {'PASS' if x['ok'] else 'fail'} | "
              f"{x['err']} | {x['pre'] and round(x['pre'])} | {x['gsm']} | {fmt(x['accept'])} |")
    # verdict per engine x column: highest rate at which every (box, seed) row passes
    print()
    for who in [w for _, w in SERVERS]:
        for col in ("A", "B"):
            rs = [r for r in rows if r["who"] == who and r["col"] == col]
            if not rs:
                continue
            ok = []
            for rate in sorted({r["rate"] for r in rs}):
                at = [r for r in rs if r["rate"] == rate]
                if all(r["ok"] for r in at):
                    ok.append((rate, len(at), sorted({r["seed"] for r in at}), max(r["p95"] for r in at)))
            best = ok[-1] if ok else None
            print(f"- {who}, column {col}: " + (f"every box and seed passes up to {best[0]} req/s ({best[1]} cells, seeds {best[2]}, "
                                                f"worst p95 TTFT {best[3]:.2f} s)" if best else "no rate passes on every box and seed"))


if __name__ == "__main__":
    main(*sys.argv[1:])
