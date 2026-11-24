#!/usr/bin/env python3
"""gemm_table.py: the per-shape GEMM-headroom table from the mxG* runs (box/gemm_headroom.py JSONL rows).

Reads only run files: results/<label>/<stamp>/gemm/*.jsonl. Every speedup is the engine path (aiter.gemm_a4w4 with the
image's tuned CSV, exactly as vLLM calls it) over a variant, both timed in the SAME phase on the SAME box (median over
repeated rounds where a phase repeats), so box-to-box and phase-to-phase clock differences never enter a ratio. A variant
counts only if its output matches the engine's (relative L2 error < 0.03). Absolute rates are put against the clock-adjusted
dense MXFP4 peak at the GFX clock measured during that same measurement.

  python3 -I recipes/mi355x/tools/gemm_table.py                      # latest mxG1_asm, mxG1_alt, mxG1b_ab
  python3 -I recipes/mi355x/tools/gemm_table.py --json out.json run_dir [run_dir ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics as st
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[3]
SHAPES = [("qkv_s", 16384, 5376, 50), ("qkv_g", 20480, 5376, 10), ("o_s", 5376, 8192, 50), ("o_g", 5376, 16384, 10),
          ("gate_up", 43008, 5376, 60), ("down", 5376, 21504, 60)]
SQUARES = [("sq8k", 8192, 8192, 0), ("sq16k", 16384, 16384, 0)]
ERR_OK = 0.03
LABELS = ("mxG1_asm", "mxG1_alt", "mxG1b_ab")


def latest(label):
    d = sorted(glob.glob(str(ROOT / "results" / label / "2*")))
    return pathlib.Path(d[-1]) if d else None


def rel(p):
    p = pathlib.Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def load(run_dirs):
    rows = []
    for rd in run_dirs:
        for f in sorted(pathlib.Path(rd).glob("gemm/*.jsonl")):
            for line in f.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r["_run"] = rel(rd)
                rows.append(r)
    return rows


def ok(r):
    return "us" in r and not r.get("error") and r.get("err_vs_engine", 9) < ERR_OK


def family(v):
    return ("engine" if v == "engine" else "ps" if v.startswith("ps_") else "ck" if v.startswith("ck_") else
            "triton" if v.startswith("triton") else "ctypes" if v.startswith("ctypes") else "asm" if v.startswith("asm") else v)


def med(rs, key):
    xs = [r[key] for r in rs if r.get(key) is not None]
    return st.median(xs) if xs else None


def groups(rows):
    """(run, phase, shape, M) -> variant -> [rows]"""
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "shape" in r and "M" in r and "variant" in r and ok(r):
            g[(r["_run"], r["phase"], r["shape"], r["M"])][r["variant"]].append(r)
    return g


def summarize(rows):
    g = groups(rows)
    pts = defaultdict(lambda: dict(engine=[], fams={}))
    for (run, phase, shape, M), vs in g.items():
        if "engine" not in vs:
            continue
        e_us = med(vs["engine"], "us")
        p = pts[(shape, M)]
        p["engine"].append(dict(run=run, phase=phase, us=e_us, pflops=med(vs["engine"], "pflops"), pct=med(vs["engine"], "pct_peak_at_clk"),
                                clk=med(vs["engine"], "clk_mhz"), n=len(vs["engine"]), cfg=vs["engine"][0].get("engine_cfg")))
        for v, rs in vs.items():
            fam = family(v)
            if fam in ("engine", "ctypes"):
                continue
            us = med(rs, "us")
            cand = dict(variant=v, run=run, phase=phase, us=us, speedup=e_us / us, pflops=med(rs, "pflops"),
                        pct=med(rs, "pct_peak_at_clk"), clk=med(rs, "clk_mhz"), n=len(rs))
            key = f"{fam}:{run}:{phase}"
            if key not in p["fams"] or cand["speedup"] > p["fams"][key]["speedup"]:
                p["fams"][key] = cand
    return pts


def best_of(p, fam):
    c = [v for k, v in p["fams"].items() if k.startswith(fam + ":")]
    return max(c, key=lambda x: x["speedup"]) if c else None


def all_of(p, fam):
    return [v for k, v in p["fams"].items() if k.startswith(fam + ":")]


def kname(cfg):
    if cfg is None:
        return "heuristic (no row)"
    n = (cfg.get("kernelName") or "")
    n = n.split("Fp4_BpreShuffle_")[-1].rstrip("E") if "Fp4_" in n else ("CK " + n.split("_")[2] if n.startswith("a4w4") else n)
    return n + (f" sk{cfg.get('splitK')}" if cfg.get("splitK") else "")


def f(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def md_table(pts, Ms):
    out = ["| shape (N x K) | M | engine kernel | engine PF | % peak@clk | clk MHz | best existing kernel (no ps) | x engine | "
           "v0.1.30 persistent 256x256 | x engine | ps PF | ps % peak@clk |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, N, K, _ in SHAPES + SQUARES:
        for M in Ms:
            p = pts.get((name, M))
            if not p or not p["engine"]:
                continue
            e = next((x for x in p["engine"] if x["phase"] == "sweep"), p["engine"][0])
            ex = max([best_of(p, fam) for fam in ("asm", "ck", "triton") if best_of(p, fam)], key=lambda x: x["speedup"], default=None)
            ps = all_of(p, "ps")
            psx = " / ".join(f"{x['speedup']:.3f}" for x in sorted(ps, key=lambda x: x["run"])) if ps else "-"
            psb = max(ps, key=lambda x: x["speedup"]) if ps else None
            out.append(f"| {name} ({N} x {K}) | {M} | {kname(e['cfg'])} | {f(e['pflops'])} | {f(e['pct'], 1)} | {f(e['clk'], 0)} | "
                       f"{(ex['variant'] if ex else '-')} | {f(ex['speedup'], 3) if ex else '-'} | {'ps_256x256' if ps else '-'} | {psx} | "
                       f"{f(psb['pflops']) if psb else '-'} | {f(psb['pct'], 1) if psb else '-'} |")
    return "\n".join(out)


def compact(pts, ab_run, sweep_run, Ms=(8192, 6912, 4096, 2048)):
    """The note's per-shape table: engine path vs the v0.1.30 persistent kernel from the interleaved A/B box (median of its
    rounds), the same ratio from the first box's ps phase, and the best kernel the image already has (asm sweep, CK,
    Triton; each vs its own same-phase engine path)."""
    out = ["| shape (N x K), FLOP share | M | engine path kernel | engine PF | % peak@clk | best kernel already in the image | x | "
           "persistent 256x256 PF | % peak@clk | x engine (A/B box) | x engine (box A) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    share = {"qkv_s": 15.0, "qkv_g": 3.7, "o_s": 7.5, "o_g": 3.0, "gate_up": 47.2, "down": 23.6}
    for name, N, K, _ in SHAPES:
        for M in Ms:
            p = pts.get((name, M))
            if not p:
                continue
            e = next((x for x in p["engine"] if x["run"] == ab_run), None)
            ps_ab = next((v for k, v in p["fams"].items() if k.startswith("ps:" + ab_run)), None)
            ps_a = next((v for k, v in p["fams"].items() if k.startswith("ps:" + sweep_run)), None)
            ex = [v for fam in ("asm", "ck", "triton") for v in all_of(p, fam) if v["run"] != ab_run]
            exb = max(ex, key=lambda x: x["speedup"]) if ex else None
            if not e:
                continue
            out.append(f"| {name} ({N} x {K}), {share[name]}% | {M} | {kname(e['cfg'])} | {f(e['pflops'])} | {f(e['pct'], 1)} | "
                       f"{exb['variant'] if exb else '-'} | {f(exb['speedup'], 3) if exb else '-'} | {f(ps_ab['pflops']) if ps_ab else '-'} | "
                       f"{f(ps_ab['pct'], 1) if ps_ab else '-'} | **{f(ps_ab['speedup'], 3) if ps_ab else '-'}** | {f(ps_a['speedup'], 3) if ps_a else '-'} |")
    for name, N, K, _ in SQUARES:
        p = pts.get((name, N))
        if not p:
            continue
        for e in p["engine"]:
            ps = next((v for k, v in p["fams"].items() if k.startswith("ps:" + e["run"] + ":" + e["phase"])), None)
            if ps:
                out.append(f"| {name} ({N} x {K}) ceiling, {e['run'].split('/')[1]} {e['phase']} | {N} | {kname(e['cfg'])} | {f(e['pflops'])} | "
                           f"{f(e['pct'], 1)} | - | - | {f(ps['pflops'])} | {f(ps['pct'], 1)} | {f(ps['speedup'], 3)} | |")
    return "\n".join(out)


def layer_weighted(pts, M, fam_sets, run=None):
    """Time of one forward's 240 MXFP4 GEMMs at this M (per-shape engine us x layer count) and the saving if every shape ran
    its best variant from `fam_sets`, using each variant's same-phase speedup. With `run`, engine times and speedups come
    from that one box only (no max across boxes)."""
    te = tb = 0.0
    for name, _, _, n in SHAPES:
        p = pts.get((name, M))
        if not p or not p["engine"]:
            return None
        es = [x for x in p["engine"] if run is None or x["run"] == run]
        if not es:
            return None
        e = next((x for x in es if x["phase"] == "sweep"), es[0])
        cands = [b["speedup"] for fam in fam_sets for b in all_of(p, fam) if run is None or b["run"] == run]
        if run is not None and not cands:
            return None          # this box did not time that family at this point: no figure, not 0%
        sp = max([1.0] + cands)
        te += n * e["us"]
        tb += n * e["us"] / sp
    return dict(M=M, engine_ms=round(te / 1e3, 3), best_ms=round(tb / 1e3, 3), saved_pct=round(100 * (1 - tb / te), 2))


def selftest():
    """Every ratio pairs a variant with the engine path of the SAME run and phase; a per-box figure needs that box's data."""
    def row(run, phase, v, us):
        return dict(_run=run, phase=phase, shape="o_s", M=8192, variant=v, us=us, pflops=1.0, err_vs_engine=0.0, clk_mhz=1700,
                    pct_peak_at_clk=60.0)
    rows = [row("A", "sweep", "engine", 100.0), row("A", "ps", "engine", 120.0), row("A", "ps", "ps_256x256", 100.0),
            row("C", "ab", "engine", 110.0), row("C", "ab", "engine", 110.0), row("C", "ab", "ps_256x256", 100.0)]
    pts = summarize(rows)
    ps = sorted(round(x["speedup"], 3) for x in all_of(pts[("o_s", 8192)], "ps"))
    assert ps == [1.1, 1.2], ps          # 120/100 within A's ps phase, 110/100 within C; never 100 (A sweep) / 100
    assert layer_weighted(pts, 8192, ("ps",), run="A") is None   # A has no other shapes: no figure, not 0%
    print("gemm_table selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--compact", action="store_true", help="the note's per-shape table only")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    runs = [pathlib.Path(r) if pathlib.Path(r).is_absolute() else ROOT / r for r in a.runs] or [x for x in map(latest, LABELS) if x]
    rows = load(runs)
    pts = summarize(rows)
    Ms = sorted({M for (_, M) in pts})
    if a.compact:
        ab = next((rel(r) for r in runs if "mxG1b_ab" in str(r)), "")
        sw = next((rel(r) for r in runs if "mxG1_asm" in str(r)), "")
        print(compact(pts, ab, sw))
        return
    print("Runs: " + ", ".join(rel(r) for r in runs) + "\n")
    print(md_table(pts, Ms))
    print("\n| M | 240 GEMMs, engine (ms) | best existing kernels | saved | ps where better, else engine | saved | best of all | saved |")
    print("|---|---|---|---|---|---|---|---|")
    lw = []
    for M in Ms:
        x = layer_weighted(pts, M, ("asm", "ck", "triton"))
        y = layer_weighted(pts, M, ("ps",))
        z = layer_weighted(pts, M, ("asm", "ck", "triton", "ps"))
        if x:
            lw.append(dict(M=M, existing=x, ps=y, all=z))
            print(f"| {M} | {x['engine_ms']} | {x['best_ms']} | {x['saved_pct']}% | {y['best_ms']} | {y['saved_pct']}% | {z['best_ms']} | {z['saved_pct']}% |")
    print("\nPer box, persistent kernel where it beats the engine path, else the engine path (no max across boxes):\n")
    print("| M | " + " | ".join(rel(r) for r in runs if any(x.get("_run") == rel(r) and x.get("phase") in ("ps", "ab") for x in rows)) + " |")
    pr = [rel(r) for r in runs if any(x.get("_run") == rel(r) and x.get("phase") in ("ps", "ab") for x in rows)]
    print("|---|" + "---|" * len(pr))
    for M in Ms:
        cells = [layer_weighted(pts, M, ("ps",), run=r) for r in pr]
        if any(cells):
            print(f"| {M} | " + " | ".join(f"{c['engine_ms']} -> {c['best_ms']} ms ({c['saved_pct']}%)" if c else "-" for c in cells) + " |")
    print()
    for r in rows:
        if r.get("phase") in ("mfma",) and "us" in r:
            print(f"mfma {r['kernel']} {r['operands']} hold {r['hold_s']} s: {r['pflops']:.3f} PF at {r.get('clk_mhz')} MHz "
                  f"({r.get('pct_peak_at_clk')}% of clock-adjusted peak), {r.get('power_w')} W  [{r['_run']}]")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(dict(runs=[rel(r) for r in runs],
                                                        points={f"{k[0]}:{k[1]}": v for k, v in pts.items()}, layer_weighted=lw),
                                                   indent=1, default=str))


if __name__ == "__main__":
    main()
