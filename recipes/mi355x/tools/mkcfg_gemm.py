#!/usr/bin/env python3
"""Generate the MXFP4 GEMM-headroom configs (recipes/mi355x/configs/mxG*.json) for run.py.

Phase 1 of the GEMM-headroom track (notes/gemm_headroom.md): kernel microbenchmarks on OpenRelay's image with NO engine
on the GPU. run.py needs a server per config, so the "server" here is a stdlib http.server answering /health on :8000
(allow_rejected: it has no startup facts by design); the steps run box/gemm_headroom.py phases.

Two boxes run in parallel (Maks's standing rule: independent boxes never run one after another):
  mxG1_asm  inventory, MFMA ceiling, aiter asm sweep (+ synthetic squares), v0.1.30 persistent kernel, confirm
  mxG1_alt  inventory, hipBLASLt, Triton preshuffle grid, CK a4w4 blockscale (JIT build)
Each phase times the engine path (aiter.gemm_a4w4 as vLLM calls it) on its own box, so cross-box clock spread does not
enter a ratio; absolute rates are put against the clock measured during that same measurement.

  python3 -I recipes/mi355x/tools/mkcfg_gemm.py G1
  python3 -I run.py recipes/mi355x/configs/mxG1_asm.json --dry-run
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mkcfg import ENV_OR, OUT, SNAP_OR, best_flags, gate, proto, server, setup_cmd, write  # noqa: E402

BENCH = "python3 /root/or/mx/gemm_headroom.py"
HEALTH = ["mkdir -p /root/or/hc /root/or/out/gemm", "echo ok > /root/or/hc/health"]


def step(name, args, est_s):
    return {"type": "sh", "name": name, "cmd": f"{BENCH} {args} --out /root/or/out/gemm", "outputs": ["out/gemm/"], "est_s": est_s}


def idle_server(steps):
    return {"name": "nogpu", "weights": [], "served_name": "none", "env": {}, "allow_rejected": True,
            "launch": "python3 -m http.server 8000 --bind 127.0.0.1 --directory /root/or/hc",
            "ready_timeout_s": 120, "steps": steps, "delete_weights_after": False}


def config(label, note, steps, max_minutes, budget):
    return {"label": label, "stage": "mxG1-gemm-headroom", "note": note, "snapshot": SNAP_OR, "max_minutes": max_minutes,
            "budget_usd": budget, "setup_cmd": setup_cmd(["gemm_headroom.py"], HEALTH), "servers": [idle_server(steps)]}


def g1():
    asm = config(
        "mxG1_asm",
        "GEMM headroom phase 1, box A (no engine on the GPU): versions + resolved tuned rows; pure-MFMA FP4 ceiling (hipcc loop of "
        "v_mfma_scale_f32_32x32x64_f8f6f4) at the clock it holds; every aiter asm f4gemm kernel x log2 split-K vs the engine path "
        "(aiter.gemm_a4w4 + tuned CSV) at Gemma 4 31B's six (N, K) and M = 16..384 (graph sizes) and 1K..16K (prefill chunks), "
        "plus 8192^3 and 16384^3; aiter v0.1.30's persistent 256x256 kernel via a ctypes launcher; confirm rounds. Clock and power "
        "logged per measurement (box/gemm_headroom.py).",
        [step("gemm_inventory", "inventory", 120), step("gemm_mfma", "mfma", 120),
         step("gemm_sweep", "sweep --min-s 0.1 --shape-timeout 900", 1500), step("gemm_ps", "ps --min-s 0.12", 420),
         step("gemm_confirm", "confirm --rounds 2 --min-s 0.25", 600)],
        max_minutes=70, budget=6.0)
    alt = config(
        "mxG1_alt",
        "GEMM headroom phase 1, box B (no engine on the GPU): the other backends against the same engine path on the same box: "
        "hipBLASLt via torch._scaled_mm (+ hipblaslt-bench all solutions if present), aiter Triton gemm_afp4wfp4_preshuffle with a "
        "10-config grid at M=8192, CK a4w4 blockscale (21 kernelIds x splitK 0-3; JIT build time recorded).",
        [step("gemm_inventory", "inventory", 120), step("gemm_hipblaslt", "hipblaslt --min-s 0.15", 400),
         step("gemm_triton", "triton --min-s 0.12 --budget-s 600", 700), step("gemm_ck", "ck --min-s 0.08 --budget-s 900", 1800)],
        max_minutes=60, budget=5.0)
    return [asm, alt]


PS_CO_URL = ("https://raw.githubusercontent.com/ROCm/aiter/77ba72b7061a3c76c4f726a50e8edfd2fe2a80e7/hsa/gfx950/f4gemm/"
             "f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ps.co")   # aiter tag v0.1.30
PS_CO = "/root/or/mx/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ps.co"
PS_SHA = "8b541b78bff2de71f3cb48db89572f1eb4adc403e09a675be9a8a546a60ff938"
PS_FETCH = [f"curl -sfL --retry 3 -o {PS_CO} {PS_CO_URL}", f"echo '{PS_SHA}  {PS_CO}' | sha256sum -c -"]


def g1b():
    """Second box, interleaved A/B (the mxG1_asm confirm phase aborted in glibc): engine path vs the v0.1.30 persistent
    256x256 kernel vs the best asm tiles, 3 rounds with the order reversed every other round, M from 1024 to 8192."""
    return [config(
        "mxG1b_ab",
        "GEMM headroom phase 1, box C (no engine on the GPU): interleaved A/B of the engine path (aiter.gemm_a4w4 + tuned CSV) vs "
        "aiter v0.1.30's persistent 256x256 f4gemm kernel (ctypes launcher, sha256-pinned .co) vs asm 256x256/160x384/128x384, at "
        "M = 1024..8192 on Gemma 4 31B's six (N, K) plus 8192^3 / 16384^3; 3 rounds, order reversed every other round; clock and "
        "power per measurement. Replicates mxG1_asm's ps phase on another box and sets the MX_PS_MIN_M routing threshold.",
        [step("gemm_inventory", "inventory", 120),
         step("gemm_ab", "ab --rounds 3 --min-s 0.15 --ms 1024,1536,2048,3072,4096,5120,6144,6912,7680,8192 --shape-timeout 600", 900)],
        max_minutes=35, budget=3.0)]


def g2():
    """Phase 2, paired: on each box the SAME MI355X serves our recipe and our recipe + MX_PS_FP4=1 (box/patch_ps_fp4.py:
    M >= 2048 MXFP4 GEMMs on aiter v0.1.30's persistent 256x256 kernel) back to back, weights kept. Seed 7 box runs the ps
    arm first, seed 8 box the baseline first. Their client at 3.5 and 3.75 req/s, 300 s per rate, + prefill probe;
    GSM8K test[:250] on the ps arm, [:100] on the baseline (re-gate of a known config)."""
    out = []
    env = dict(ENV_OR, MX_DRAFT_COMPACT="1")
    for seed, ps_first in ((7, True), (8, False)):
        base = server("mx_recipe", best_flags(), env, [gate(100), proto([3.5, 3.75], [seed])],
                      claim_group="mx_recipe_G2", origin="ours")
        ps = server("mx_ps", best_flags(), dict(env, MX_PS_FP4="1", MX_PS_MIN_M="2048"), [gate(250), proto([3.5, 3.75], [seed])],
                    claim_group="mx_recipe_ps_G2", origin="ours")
        ps["launch"] = "bash /root/or/mx/serve_ps.sh " + ps["launch"]
        order = [ps, base] if ps_first else [base, ps]
        c = {"label": f"mxG2_s{seed}", "stage": "mxG2-gemm-ps-paired",
             "note": (f"PHASE 2 (paired), seed {seed}: the same MI355X serves {'ours + MX_PS_FP4=1, then ours' if ps_first else 'ours, then ours + MX_PS_FP4=1'}. "
                      "Ours = OpenRelay's image by digest + MTP k=3 + --max-num-seqs 96 + graphs to 384 + MX_DRAFT_COMPACT=1. "
                      "MX_PS_FP4=1 = box/patch_ps_fp4.py: MXFP4 GEMMs with M >= 2048 launch aiter v0.1.30's persistent 256x256 kernel "
                      "(sha256-pinned .co) instead of the image's non-persistent asm kernels; decode steps (M <= 384, CUDA graphs) unchanged. "
                      "Their client (vllm bench serve, random tokens, 300 s per rate) at 3.5/3.75 + prefill probe; GSM8K test[:250] on the ps arm."),
             "snapshot": SNAP_OR, "max_minutes": 80, "budget_usd": 7.0,
             "setup_cmd": setup_cmd(["patch_draft_compact.py", "patch_ps_fp4.py", "serve_ps.sh"],
                                    ["python3 /root/or/mx/patch_draft_compact.py"] + PS_FETCH + ["python3 /root/or/mx/patch_ps_fp4.py"]),
             "servers": order}
        out.append(c)
    return out


def gl1():
    """Glue headroom phase 1 (lead, 11-22 ~10:1xZ): aiter fused kernels for Gemma 4's non-GEMM glue vs Inductor replicas of
    the engine path (box/glue_headroom.py): attention prep (q/k/v norm + RoPE + fp8 KV write) per layer type, and the two
    sandwich-norm sites + MXFP4 activation quant. Output checks against the replica on every variant."""
    glue = "python3 /root/or/mx/glue_headroom.py"
    steps = [step("gemm_inventory", "inventory", 120),
             {"type": "sh", "name": "glue_inventory", "cmd": f"{glue} inventory --out /root/or/out/glue", "outputs": ["out/glue/"], "est_s": 120},
             {"type": "sh", "name": "glue_prep", "cmd": f"{glue} prep --out /root/or/out/glue", "outputs": ["out/glue/"], "est_s": 900},
             {"type": "sh", "name": "glue_norm", "cmd": f"{glue} norm --out /root/or/out/glue", "outputs": ["out/glue/"], "est_s": 600}]
    c = config("mxGL1_glue",
               "Glue headroom phase 1 (no engine on the GPU): Gemma 4 31B's attention prep (split + q/k/v RMSNorm + RoPE + fp8 "
               "paged KV write; sliding d256 and global d512 layers) and sandwich norms + MXFP4 activation quant, M = 256-8192. "
               "Engine path = torch.compile replicas of vLLM 0.30's Gemma 4 code + vllm reshape_and_cache_flash + aiter "
               "per_1x32_f4_quant_hip; candidates = aiter fused_qk_rope_reshape_and_cache (vLLM fuse_rope_kvcache), "
               "fused_qkv_split_qk_norm_rope_cache, HIP fused_qk_norm_rope_cache_pts_quant_shuffle (vLLM fuse_qk_norm_rope_kvcache, "
               "d256), fused_rms_mxfp4_quant, and v0.1.30's fused_rmsnorm_add_rmsnorm (inlined). Output checks on every variant.",
               steps, max_minutes=45, budget=4.0)
    c["stage"] = "mxGL1-glue-headroom"
    c["setup_cmd"] = setup_cmd(["gemm_headroom.py", "glue_headroom.py"], HEALTH)
    return [c]


def gl1b():
    """Re-run of mxGL1_glue with a faithful engine replica (token dim dynamic only, vLLM's combo-kernel Inductor settings),
    per-component timings, the 1 + w weight convention for fused_qkv_split_qk_norm_rope_cache, and M = 6912."""
    c = gl1()[0]
    c["label"] = "mxGL1b_glue"
    c["note"] = c["note"].replace("Glue headroom phase 1", "Glue headroom phase 1, re-run with a faithful engine replica")
    return [c]


def bf1():
    """BF16 GEMM headroom (lead, 11-22 ~10:5xZ): target lm_head and MTP-drafter linears, BF16 hipBLASLt vs FP8 (per-tensor and
    per-token x per-channel, hipBLASLt and aiter CK), activation quant included, output error (box/bf16_headroom.py)."""
    steps = [step("gemm_inventory", "inventory", 120),
             {"type": "sh", "name": "bf16_fp8", "cmd": "python3 /root/or/mx/bf16_headroom.py --out /root/or/out/bf16",
              "outputs": ["out/bf16/"], "est_s": 600}]
    c = config("mxBF1_fp8",
               "BF16 GEMM headroom phase 1 (no engine on the GPU): Gemma 4 31B's lm_head (262144 x 5376) at M = 16-384 sampled "
               "positions and the MTP drafter's linears (hidden 1024; q/o/gate_up/down, in/out projections, 2048-centroid head) "
               "at M = 36 / 144, BF16 F.linear vs FP8 e4m3 (per-tensor and per-token x per-channel; torch._scaled_mm and aiter "
               "gemm_a8w8), activation quant included, relative error and lm_head top-1/top-5 agreement.",
               steps, max_minutes=30, budget=2.5)
    c["stage"] = "mxBF1-bf16-headroom"
    c["setup_cmd"] = setup_cmd(["gemm_headroom.py", "bf16_headroom.py"], HEALTH)
    return [c]


def bf1b():
    """Drafter shapes again, one process per shape (mxBF1_fp8 segfaulted in hipBLASLt FP8 graph replay at the first one);
    torch._scaled_mm timed eagerly next to an eager BF16 row."""
    shapes = "d_q_sliding d_q_global d_o_sliding d_o_global d_gate_up d_down d_in_proj d_out_proj d_centroids"
    cmd = (f"for s in {shapes}; do python3 /root/or/mx/bf16_headroom.py --out /root/or/out/bf16 --shapes $s "
           "|| echo \"shape $s rc=$?\"; done")
    steps = [step("gemm_inventory", "inventory", 120),
             {"type": "sh", "name": "bf16_drafter", "cmd": cmd, "outputs": ["out/bf16/"], "est_s": 600}]
    c = config("mxBF1b_drafter", "BF16 GEMM headroom phase 1b: the MTP drafter's linears only (hidden 1024 shapes at M = 36 / 144), "
               "BF16 vs FP8, one process per shape; hipBLASLt FP8 timed eagerly (graph replay segfaulted in mxBF1_fp8).",
               steps, max_minutes=25, budget=2.0)
    c["stage"] = "mxBF1-bf16-headroom"
    c["setup_cmd"] = setup_cmd(["gemm_headroom.py", "bf16_headroom.py"], HEALTH)
    return [c]


WAVES = {"G1": g1, "G1B": g1b, "G2": g2, "GL1": gl1, "GL1B": gl1b, "BF1": bf1, "BF1B": bf1b}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for w in (sys.argv[1:] or list(WAVES)):
        for c in WAVES[w]():
            print(write(c))
