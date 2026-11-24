"""aiter_tune.py: tune AITER's BF16 GEMMs for the exact prefill shapes Inkling-2-Mini runs on an MI355X, then drop the
merged table where AITER merges model tables (aiter/configs/model_configs/inkling2mini_bf16_tuned_gemm.csv), so the NEXT
server start picks it up.

Shapes: box/inkling2mini_bf16_untuned_gemm.csv, the 117 distinct M>=1024 shapes SGLang logged as "not found tuned config
... will use default config! using torch solution:0" in ik4_mi355x_aiter's base server (N=8600 K=2304 = fused q/k/v/rel
projection, N=2304 K=3072 = output projection; M = SGLang's prefill-graph token buckets, up to 3 x 16384).

Tuner: AITER's csrc/gemm_a16w16/gemm_tuner.py, one backend at a time, PRECOMPILED backends only:
  torch     (the default AITER falls back to: the baseline us per shape),
  asm       (AITER's prebuilt gfx950 assembly kernels),
  hipblaslt (--with-hipblaslt; hipBLASLt's solution list).
ik4's first attempt ran the default backend set (flydsl/triton/opus JIT-compile thousands of variants on 8 vCPUs: one
worker at 100% CPU, GPU at 1%, nothing written after 26 min), so JIT backends are excluded here. Each backend has its own
timeout and streams its log; the merge keeps the lowest-us row per shape across backends, so a backend that times out
costs only its own time.

  python3 aiter_tune.py --untuned /root/or/ik/inkling2mini_bf16_untuned_gemm.csv --out /root/or/out   # tune (GPU must be free)
  python3 aiter_tune.py --untuned ... --out /root/or/out --install-only                                # later: install
Writes <out>/aiter_tuned_<lib>.csv, <out>/aiter_tune_<lib>.log, <out>/aiter_tuned_inkling.csv (merged, installed),
<out>/aiter_tune_summary.json (per-backend rc/time/rows, per-shape speedup of the winner vs torch).
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

KEY = ("M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle")
# ik7 (with SGLang holding the GPU): torch 34/117 rows, hipBLASLt and asm 0 rows, all CUDA OOM ("0 bytes free"); asm also
# hit a tuner assertion (shape_grouped: group count != in_datas count). ik9 runs this with NO engine on the GPU (stub
# server), cheap backends first, flydsl (JIT, what AITER's own model_configs/inkling_bf16_tuned_gemm.csv uses) last.
PLAN = (("torch", [], 600), ("hipblaslt", ["--with-hipblaslt"], 1500), ("asm", [], 300), ("flydsl", [], 2400))


def rows_of(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--untuned", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma list of backends to run (default: torch,hipblaslt,asm,flydsl)")
    ap.add_argument("--install", action="store_true", help="also copy the merged table into aiter/configs/model_configs/")
    ap.add_argument("--install-only", action="store_true", help="only install <out>/aiter_tuned_inkling.csv (no tuning)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    adir = os.path.dirname(importlib.util.find_spec("aiter").origin)
    dst = os.path.join(adir, "configs", "model_configs", "inkling2mini_bf16_tuned_gemm.csv")
    if a.install_only:
        src = os.path.join(a.out, "aiter_tuned_inkling.csv")
        n = len(rows_of(src))
        if n:
            shutil.copy(src, dst)
        print(f"[aiter_tune] install-only: {n} rows -> {dst if n else 'NOTHING (no merged table)'}", flush=True)
        return 0 if n else 4
    root = os.path.dirname(adir)
    tuner = os.path.join(root, "csrc", "gemm_a16w16", "gemm_tuner.py")
    summ = dict(t_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), tuner=tuner, untuned=a.untuned,
                shapes=len(rows_of(a.untuned)), backends={})
    print(f"[aiter_tune] {summ['shapes']} shapes, tuner {tuner}", flush=True)
    want = set(a.only.split(",")) if a.only else None
    per_lib = {}
    for lib, extra, tmo in PLAN:
        if want and lib not in want:
            continue
        tf = os.path.join(a.out, f"aiter_tuned_{lib}.csv")
        lf = os.path.join(a.out, f"aiter_tune_{lib}.log")
        cmd = [sys.executable, tuner, "--input_file", a.untuned, "--tuned_file", tf, "--libtype", lib] + extra
        t0 = time.time()
        with open(lf, "w") as log:
            log.write(" ".join(cmd) + "\n")
            log.flush()
            try:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=tmo, cwd=os.path.dirname(tuner)).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        rows = rows_of(tf)
        per_lib[lib] = rows
        summ["backends"][lib] = dict(rc=rc, secs=round(time.time() - t0, 1), rows=len(rows), timeout_s=tmo)
        print(f"[aiter_tune] {lib}: rc={rc} {time.time() - t0:.0f}s rows={len(rows)}", flush=True)
    best, base = {}, {}
    for lib, rows in per_lib.items():
        for r in rows:
            try:
                us = float(r.get("us") or "inf")
            except ValueError:
                continue
            if us <= 0 or us != us:
                continue
            k = tuple(str(r.get(x)) for x in KEY)
            if lib == "torch":
                base[k] = us
            if k not in best or us < float(best[k]["us"]):
                best[k] = r
    merged = os.path.join(a.out, "aiter_tuned_inkling.csv")
    if best:
        header = list(next(iter(best.values())).keys())
        with open(merged, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for k in sorted(best, key=lambda k: (int(k[1]), int(k[2]), int(k[0]))):
                w.writerow(best[k])
        if a.install:
            shutil.copy(merged, dst)
            summ["installed_as"] = dst
        sp = [base[k] / float(best[k]["us"]) for k in best if k in base]
        summ["winner_libtype"] = {lt: sum(1 for r in best.values() if r.get("libtype") == lt) for lt in {r.get("libtype") for r in best.values()}}
        summ["speedup_vs_torch"] = dict(n=len(sp), median=round(statistics.median(sp), 3) if sp else None,
                                        min=round(min(sp), 3) if sp else None, max=round(max(sp), 3) if sp else None)
        summ["us_total_torch"] = round(sum(base.values()), 1)
        summ["us_total_best"] = round(sum(float(best[k]["us"]) for k in best if k in base), 1)
    summ["merged_rows"] = len(best)
    summ["t_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(summ, open(os.path.join(a.out, "aiter_tune_summary.json"), "w"), indent=1)
    print(f"[aiter_tune] merged {len(best)} rows; speedup vs torch {summ.get('speedup_vs_torch')}; installed {summ.get('installed_as')}",
          flush=True)
    return 0 if best else 4


if __name__ == "__main__":
    sys.exit(main())
