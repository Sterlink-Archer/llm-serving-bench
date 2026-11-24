#!/usr/bin/env python3
"""glue_table.py: the glue-headroom tables from the mxGL1b run files (box/glue_headroom.py JSONL rows), never typed.

Per (layer type, M): the engine replica (Inductor + vllm reshape_and_cache_flash, or Inductor sandwich norms + aiter
per_1x32_f4_quant_hip) against each aiter candidate, both timed on the same box in the same phase, with the output checks.
Per forward: savings x layers (50 sliding + 10 global for attention prep; 2 sandwich sites x 60 layers), and % of a
forward estimated as 23.8 us per token (the knee profile's pure prefill: 713.6 ms of kernels for 3 x 10K tokens,
notes/profile_knee_or_recipe.md).

  python3 -I recipes/mi355x/tools/glue_table.py [run_dir]       # default: latest results/mxGL1b_glue
"""
from __future__ import annotations

import collections
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
US_PER_TOKEN = 713.6e3 / 30000          # pure-prefill kernel time per token (profile_knee_or_recipe.md)
LAYERS = {"sliding": 50, "global": 10}


def load(rd, name):
    p = pathlib.Path(rd) / "glue" / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def main(rd=None):
    rd = pathlib.Path(rd) if rd else pathlib.Path(sorted(glob.glob(str(ROOT / "results" / "mxGL1b_glue" / "2*")))[-1])
    print(f"Run: {rd.relative_to(ROOT) if rd.is_relative_to(ROOT) else rd}\n")
    prep = collections.defaultdict(dict)
    for r in load(rd, "glue_prep"):
        if "us" in r:
            prep[(r["layer"], r["M"])][r["variant"]] = r
    print("| layer | M | engine us (compute + KV write) | rope_kv (vLLM fuse_rope_kvcache) | qkv_fused (+ v_norm) | hip_fused (d256) | "
          "output check vs engine (rel. L2: q / k / v) |")
    print("|---|---|---|---|---|---|---|")
    save = collections.defaultdict(float)
    for (lt, M), v in sorted(prep.items(), key=lambda kv: (kv[0][0] != "sliding", kv[0][1])):
        e = v["engine"]["us"]

        def cell(k, v=v, e=e):
            return f"{v[k]['us']:.1f} ({e / v[k]['us']:.3f}x)" if k in v else "-"
        best = max((e - v[k]["us"] for k in ("rope_kv", "qkv_fused", "hip_fused") if k in v), default=0.0)
        save[M] += max(best, 0.0) * LAYERS[lt]
        chk = "; ".join(f"{k} {v[k].get('err_q')} / {v[k].get('err_k')} / {v[k].get('err_v')}" for k in ("rope_kv", "qkv_fused", "hip_fused") if k in v)
        print(f"| {lt} | {M} | {e:.1f} ({v['engine_compute']['us']:.1f} + {v['engine_cache_write']['us']:.1f}) | {cell('rope_kv')} | "
              f"{cell('qkv_fused')} | {cell('hip_fused')} | {chk} |")
    norm = collections.defaultdict(dict)
    for r in load(rd, "glue_norm"):
        if "us" in r:
            norm[(r["site"], r["M"])][r["variant"]] = r
    print("\n| sandwich site | M | engine us (norms + quant) | fused_rms_mxfp4_quant | v0.1.30 fused_rmsnorm_add_rmsnorm + quant | "
          "FP4 bytes equal to engine (rms_quant / sandwich) |")
    print("|---|---|---|---|---|---|")
    for (site, M), v in sorted(norm.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        e = v["engine"]["us"]

        def cell(k, v=v, e=e):
            return f"{v[k]['us']:.1f} ({e / v[k]['us']:.3f}x)" if k in v else "-"
        best = max((e - v[k]["us"] for k in ("rms_quant", "sandwich_v130") if k in v), default=0.0)
        save[M] += max(best, 0.0) * 60
        print(f"| {site} | {M} | {e:.1f} ({v['engine_norms']['us']:.1f} + {v['engine_quant']['us']:.1f}) | {cell('rms_quant')} | "
              f"{cell('sandwich_v130')} | {v['rms_quant'].get('fp4_bytes_equal')} / {v.get('sandwich_v130', {}).get('fp4_bytes_equal', '-')} |")
    print("\n| M | best candidate everywhere it wins, saved per forward | % of a prefill-sized forward (23.8 us/token; not meaningful at decode M) |")
    print("|---|---|---|")
    for M in sorted(save):
        print(f"| {M} | {save[M] / 1e3:.2f} ms | {100 * save[M] / (US_PER_TOKEN * M):.2f}% |")


if __name__ == "__main__":
    main(*sys.argv[1:])
