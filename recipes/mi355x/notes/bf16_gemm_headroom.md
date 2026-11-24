# BF16 GEMM headroom: Gemma 4 31B's lm_head and MTP drafter in FP8 on one MI355X

Task (lead, 2026-11-22 ~10:5x UTC): the knee profile lists "BF16 GEMM (drafter, lm_head)". Can the MTP drafter's linears
and/or the target lm_head run in FP8 with acceptable quality? Phase 2 only if it is worth >= 1% of a forward. Measured
2026-11-22 10:55-11:01 UTC (4:55-5:01 AM Central), 2 spot boxes, ~$0.39.

**Verdict: no Phase 2.**
- **The slice is small:** on our current recipe the BF16 GEMMs are **1.7% of the knee window**. That is 202.6 of 12,130 ms
  of kernels in `results/mxP_ours_profile/20261122T091537Z/prof_summary.txt`, our recipe with the drafter compaction at
  3.5 req/s. The 2.9% figure predates the compaction patch.
- **The bound is under the bar:** these GEMMs are weight- or latency-bound, so FP8's best case is halving the bytes.
  Even 2x on every BF16 GEMM would save ~0.8% of the window.
- **FP8 helps one GEMM, and only a little:**
  - **lm_head:** FP8 runs 1.84x faster at the sampled M=144 per step (511 -> 277 us), saving ~0.23 ms per step. That is
    ~0.28% of the knee window: 145 steps in the 12 s window, and 1.2% of a decode-only step. It changes the target's
    logits: 3.75% relative error, and the top-1 token agrees with BF16 on 81-93% of rows (random hidden states).
  - **Drafter:** FP8 is 3-5x *slower*. Its linears are tiny latency-bound GEMMs (5-15 us at M=36), where the activation
    quant plus untuned FP8 kernels cost more than the halved weights save.
- **The identity and GSM8K checks** the lead asked for apply only to a shipped lm_head change. None is proposed, so they
  were not run.

## Shapes

- **Target lm_head:** 262144 x 5376 BF16 (1.41 B params, 2.82 GB, tied to the embedding). It runs once per step on the
  sampled positions: k+1 = 4 per decoding request, so M ~ 4 x 36.7 ~ 147 at the knee and up to 96 x 4 = 384.
- **MTP drafter** (`google/gemma-4-31B-it-assistant` @627c5ec):
  - 4 decoder layers at hidden 1024, all KV-shared (q projection only; they read the target's KV);
  - 32 q heads x 256 (sliding) / 512 (global), intermediate 8192;
  - a projection from the 5376-wide backbone, and a 2048-centroid head (top-32), so no full-vocab head.
  - Each draft pass runs at M = requests (~36). k = 3 passes per step, plus the compacted first pass.
- **In the profile** almost all of it sits outside the target's forward. Hundreds to thousands of hipBLASLt `Cijk_*`
  calls at 7-94 us each come to 130.8 ms in the outside-the-forward list. The lm_head appears as `MT256x144` /
  `MT256x160` calls at ~0.47 ms each.

## Method

`box/bf16_headroom.py` (reuses `box/gemm_headroom.py`'s clock sampler and graph timing), on OpenRelay's image by digest,
with no engine on the GPU. Configs are `tools/mkcfg_gemm.py BF1` and `BF1B`.
- **Engine path:** BF16 `F.linear` (hipBLASLt).
- **Candidates:** FP8 e4m3 weights with dynamic activation quant, timed *including* that quant:
  - `fp8_tensor`: per-tensor scales, `torch._scaled_mm` (hipBLASLt FP8);
  - `fp8_rowwise`: per-token x per-channel scales, `torch._scaled_mm`;
  - `aiter_a8w8`: per-token x per-channel, aiter CK `gemm_a8w8`.
- **Weights** rotate over >= 768 MB per variant, so every call streams from HBM.
- **Error vs BF16:** relative L2. For lm_head, also top-1 agreement and whether the FP8 argmax is in BF16's top 5.
  Hidden states are random normal, so the top-1 figures understate the margin real logits have. They still show FP8
  changes which token wins on a material share of rows.

## lm_head (`results/mxBF1_fp8/20261122T105550Z`, CUDA-graph timed)

| M (sampled positions) | BF16 us | fp8_tensor us (x) | fp8_rowwise us (x) | aiter_a8w8 us (x) | rel. error | top-1 agree (tensor / rowwise) |
|---|---|---|---|---|---|---|
| 16 | 431.0 | 250.3 (1.72x) | 245.0 (1.76x) | 321.0 (1.34x) | 3.75% | 87.5% / 81.3% |
| 64 | 423.4 | 244.6 (1.73x) | 263.8 (1.61x) | 288.0 (1.47x) | 3.75% | 92.2% / 87.5% |
| 144 | 510.6 | 277.1 (1.84x) | 279.1 (1.83x) | 433.1 (1.18x) | 3.75% | 88.9% / 85.4% |
| 256 | 603.9 | 334.9 (1.80x) | 373.6 (1.62x) | 481.9 (1.25x) | 3.75% | 91.0% / 93.0% |
| 384 | 1023.0 | 494.8 (2.07x) | 598.0 (1.71x) | 667.8 (1.53x) | 3.75% | 89.3% / 90.4% |

- **Bandwidth:** BF16 streams the 2.82 GB at 5.5-6.7 TB/s up to M=144, near HBM bandwidth. FP8 halves the bytes and gets
  1.7-1.8x. The FP8 argmax was in BF16's top 5 on >= 99.7% of rows.
- **Per step at the knee** (M ~ 144): 0.23 ms saved.
  - 0.14% of a 170 ms mixed step and 1.2% of a 19.5 ms decode-only step.
  - Over the profile's 12 s window (57 mixed + 88 decode-only steps), ~34 ms of 12,130 ms = **0.28%**.
  - That is what moves req/s, so it does not reach the 1% bar.
- **Cost of the change:** it alters the target's logits, so outputs are no longer token-identical to the recipe and the
  quality gates would need re-running. That cost is out of proportion to a 0.28% gain.

## Drafter (`results/mxBF1b_drafter/20261122T105819Z`, one process per shape)

BF16 at M=36 (graph-timed):

| shape | q 8192x1024 | q 16384x1024 | o 1024x8192 | o 1024x16384 | gate_up 16384x1024 | down 1024x8192 | in 1024x6400 | out 5376x1024 | centroids 2048x1024 |
|---|---|---|---|---|---|---|---|---|---|
| BF16 us | 7.8 | 9.7 | 11.4 | 14.3 | 9.7 | 11.3 | 14.5 | 7.6 | 5.3 |

Where FP8 ran:
- **o_proj 1024 x 16384:** M=36 BF16 14.3 us vs aiter_a8w8 55.9 us and `_scaled_mm` (eager) 63-64 us, against 17.0 us
  for eager BF16. M=144 BF16 21.3 us vs 65-116 us.
- **in_proj 1024 x 6400:** M=36 BF16 14.5 us vs aiter_a8w8 42.3 us.

**FP8 is 3-5x slower on the drafter.** These GEMMs read 2-34 MB of weights and run at 0.8-3.5 TB/s, so they are latency-
not bandwidth-bound. FP8 adds an activation-quant kernel and lands on kernels untuned for these shapes.

Two crash notes, recorded because they block FP8 here regardless of speed:
- hipBLASLt FP8 (`torch._scaled_mm`) segfaulted in CUDA-graph replay at 8192 x 1024, M=36
  (`results/mxBF1_fp8/20261122T105550Z/bf16_fp8.log`).
- 8 of 9 drafter processes died with rc 139 (`mxBF1b_drafter` log). Seven died in aiter's CK `gemm_a8w8`, the variant
  right after the BF16 row. d_in_proj died later, in the eager hipBLASLt FP8 variant.

Both printed hipBLASLt "Latency not found ... BFloat8Float8_fnuz" warnings first. This image's FP8 tables look incomplete
for OCP fp8 on gfx950 at these shapes. The drafter runs inside CUDA graphs in vLLM, so these crashes would surface
in-engine too.

## Files and runs

| path | what |
|---|---|
| `box/bf16_headroom.py` | the microbenchmark (`--shapes` to isolate shapes) |
| `tools/mkcfg_gemm.py` | configs `mxBF1_fp8`, `mxBF1b_drafter` |
| `results/mxBF1_fp8/20261122T105550Z` | lm_head complete; died at the first drafter shape; 1.7 min, ~$0.14 |
| `results/mxBF1b_drafter/20261122T105819Z` | drafter shapes, one process each; 3.2 min, ~$0.25 |
| `results/mxP_ours_profile/20261122T091537Z` | the recipe session's profile of our recipe: the 1.7% share |
