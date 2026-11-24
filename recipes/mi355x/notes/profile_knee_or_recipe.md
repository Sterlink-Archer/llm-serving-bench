# Profile at the knee: OpenRelay's recipe on a Daytona MI355X

Source: `results/mx1_or_profile/20261122T050153Z/prof_summary.txt` (run `mx1_or_profile/20261122T050153Z`, 2026-11-22 05:06-05:12 UTC).
Method: OpenRelay's own scripts (inference-recipes @6767b4d, `bench/profreq.py`, `bench/analysis/{ktrace,steps,stepkern,gaps}.py`,
copied unchanged to `../box/`), driven by `../box/profile_knee.sh`. The engine is their tuned image and flags plus `--profiler-config`
(torch, record_shapes, no stack).
- **Prefill:** 3 concurrent 10K-token prompts.
- **Load:** a 12 s window of open-loop load at 3.25 req/s (their random-token shape) after 90 s of warm-up.

The profiler slows the engine, so these are shares of GPU time, not throughput.

## Pure prefill (3 x 10K tokens, 713.6 ms of GPU kernels = 42K tok/s)

| Slice | ms | share |
|---|---|---|
| MXFP4 GEMM (`f4gemm ... 256x256`, `160x384`, `64x1024`) | 357.8 | 50.1% |
| Global attention, d512 (Triton `unified_attention_2d`, BLOCK_M 128, TILE 64) | 111.5 | 15.6% |
| Sliding attention, d256 (Triton 2D) | 55.8 | 7.8% |
| GELU(tanh)*up + MXFP4 quant (`_act_mul_and_dynamic_mxfp4_quant_kernel`) | 51.3 | 7.2% |
| q/k split + norm + RoPE (`triton_red_fused_2`, Inductor) | 34.8 | 4.9% |
| RMSNorms (Inductor fused add+norm) | 37.3 | 5.2% |
| FP4 activation quant for qkv/o (`dynamic_per_group_scaled_quant`) | 15.4 | 2.2% |
| KV-cache write (`reshape_and_cache_flash`) | 14.8 | 2.1% |
| q FP8 quant (`scaled_quant`) | 11.0 | 1.5% |

The MXFP4 GEMMs run 30K tokens x 2 x ~30.7B linear parameters (about 1.84 PFLOP) in 357.8 ms. That is ~5.1 PFLOP/s, about half
the MI355X's dense MXFP4 peak.

Global attention does ~0.1 PFLOP of causal QK^T + PV (10 layers, 32 heads, d512, 10K tokens, x3) in 111.5 ms. That is ~0.9 PFLOP/s,
under a fifth of the FP8 MFMA peak, so it is the least efficient large slice.

## Under load at 3.25 req/s (12 s window, 12.2 s of kernels)

| Step type | steps | GPU ms | avg per step | per step |
|---|---|---|---|---|
| mixed (prefill chunk + decode) | 54 | 8,660 | 160 ms | 6.7K prefill tokens + 28.7 decoding requests |
| decode-only | 160 | 2,537 | 15.9 ms | 29.8 requests x 3 tokens (MTP k=2) |

Decode-only steps are **21% of the window**. About 1 s of kernel time (~8%) falls outside both step windows: the MTP drafter's BF16
GEMMs (hipBLASLt `Cijk_*`), the `_topk_topp_kernel` sampler (the model's generation_config applies top_k 64 and top_p 0.95,
because `vllm bench serve` no longer sends temperature 0) and the rejection sampler.

Whole-window shares:
- MXFP4 GEMM: 50.7%;
- global attention: 19.4%, split between the 2D prefill kernel (13.9%) and the 3D split-KV decode kernel (5.6%);
- sliding attention: 13.5%, split between the prefill tile (7.3%) and the decode tile (4.3%);
- norms: 6.1%;
- BF16 GEMM (drafter, lm_head): 2.9%.

Inside a decode-only step (16.3 ms of kernels):
- global split-KV attention: 3.48 ms, ~1.7 TB/s over ~195 MB of FP8 KV per request;
- sliding decode attention: 3.12 ms, ~4 TB/s over ~420 MB per request (50 layers x 1024-token window);
- MXFP4 GEMMs: ~5.5 ms, ~3 TB/s weight streaming at M = 48-256, mostly untuned M rows;
- per-group FP4 activation quant: 0.79 ms over 180 small launches.

## What this says about levers

1. **Cap and scheduling.** Little's law at 3.5 req/s gives ~58 decoding requests (TPOT ~45 ms x 380 tokens). That runs into
   `--max-num-seqs 64`. New prompts then wait for a slot while the GPU runs decode-only steps, which cost ~16 ms each and do no
   prefill. Lifting the cap needs graphs up to seqs x 3 tokens. KV room is ~977K tokens, about 98 average requests. → `mx3_or_seqs96`,
   `mx3_or_seqs128`.
2. **Chunk size.** 8K-token chunks make mixed steps ~160-200 ms, so TPOT ≈ step / 2.8. Smaller chunks (`mx1_or_b4k`, `mx1_or_b6k`)
   shorten decode residency, and so the concurrency the cap sees.
3. **Global d512 prefill attention** is the least efficient large kernel (13.9-15.6%). A 1.5x faster tile would give ~5%.
4. **Decode attention** is ~10% of the knee: global split-KV at ~1.7 TB/s, plus decode requests inside mixed steps riding the
   prefill-shaped 2D tile.
5. **MTP k=3** (`mx1_or_mtp3`) cuts verify steps per output token, if acceptance on their random-token prompts holds.
6. **Not worth chasing.** Host-side gaps in mixed steps are 0.0%: the GPU is 100% busy inside mixed steps. Their compile-storm
   fix holds on our box.

## Second profile: k=3 + `--max-num-seqs 96` at 3.5 req/s, the drafter's share

Source: `results/mx7_prof_mtp3_seqs96/20261122T054335Z/prof_summary.txt` (2026-11-22 05:47-05:53 UTC). Same method, plus
`../box/outside.py`, which attributes kernels that fall outside the target model's `execute_context_*` step windows.

| Step type | steps | GPU ms | avg per step |
|---|---|---|---|
| mixed | 60 | 9,873 | 164.5 ms (6.8K prefill tokens + 34.9 decoding requests x 4 tokens) |
| decode-only | 74 | 1,440 | 19.5 ms (37.6 requests) |
| outside the target forward | (134 steps) | 1,000 (8.1%) | 7.46 ms per step |

Outside the target forward:
- **d512 2D attention** (`kernel_unified_attention_2d ... BLOCK_SIZE_128 TILE_SIZE_64 HEAD_SIZE_512`), prefill-shaped: 174.9 ms
  over **60 calls**, one per mixed step, ~2.9 ms each.
- **Drafter decode passes:** 3D attention ×342 (141 ms) and sliding 2D ×984 (99.6 ms).
- **Drafter BF16 GEMMs** (hipBLASLt `Cijk_*`): ~220 ms in total, the biggest being the MT256x256 tile ×372 (67 ms).
- **Samplers:** `_topk_topp_kernel` (49 ms) and `_resample_kernel`.

The one-per-mixed-step prefill-shaped call is the drafter's first pass running over the whole mixed batch. vLLM 0.30's
`AutoRegressiveSpeculator.propose` reuses the target's batch and attention metadata for draft position 0, but Gemma 4's MTP head
is Q-only: `Gemma4Speculator.advance_draft_positions` is False, it reads the target's KV and writes none. Only the
`last_token_indices` rows are used. `../box/patch_draft_compact.py` runs that pass on one token per request instead.

The patch, on the same config:
- seed 8 at 3.5: p95 TTFT 4.16 → 3.06 s and TPOT p50 43.3 → 36.7 ms;
- acceptance 3.62 → 3.61 tokens per verify step;
- runs `mx5_or_mtp3_seqs96` → `mx10_best_dc_s8`, journal steps 110 → 119.
