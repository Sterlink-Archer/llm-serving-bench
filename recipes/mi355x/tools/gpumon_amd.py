#!/usr/bin/env python3
"""Summarise run.py's raw AMD gpumon.csv (`amd-smi metric --usage --power --mem-usage --clock --temperature --csv`, header
stripped) for MI355X runs: median socket power and median GFX clock (mean of the 8 XCDs) over samples with GFX activity
>= 90%, next to the run's prefill probe. (run.py's gpumon_norm.csv only parses nvidia-smi rows, so it is empty on AMD.)
Field positions were read off a live row (2026-11-22): [1] gfx_activity %, [9] socket power W, [15,20,..,50] GFX clock
MHz per XCD (each followed by min 500 / max 2400).

  python3 -I recipes/mi355x/tools/gpumon_amd.py [label-glob ...]
"""
from __future__ import annotations

import csv
import glob
import json
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]


def summarise(path):
    util, pw, clk = [], [], []
    for r in csv.reader(open(path, errors="replace")):
        if len(r) < 55 or r[0] != "0":
            continue
        try:
            u = float(r[1])
            if u < 90:
                continue
            p = float(r[9])
            c = st.mean(float(r[i]) for i in range(15, 51, 5))
        except ValueError:
            continue
        util.append(u), pw.append(p), clk.append(c)
    return len(util), (st.median(pw) if pw else None), (st.median(clk) if clk else None)


def main(globs):
    for g in globs:
        for rd in sorted(glob.glob(str(ROOT / "results" / g / "2*"))):
            rd = pathlib.Path(rd)
            f = rd / "gpumon.csv"
            if not f.exists():
                continue
            n, p, c = summarise(f)
            pre = None
            for ps in rd.glob("protocol_*/protocol_summary.json"):
                pre = pre or (json.loads(ps.read_text()).get("prefill") or {}).get("prefill_tok_s")
            sb = (json.loads((rd / "run.json").read_text()).get("sandbox_id") or "")[:8] if (rd / "run.json").exists() else ""
            print(f"{rd.parent.name}/{rd.name} sb={sb} busy_samples={n} power_med={p} W gfx_clk_med={c and round(c)} MHz prefill={pre and round(pre)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["s2_mi355x*", "mx*"])
