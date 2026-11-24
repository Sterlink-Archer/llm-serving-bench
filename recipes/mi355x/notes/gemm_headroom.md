# MXFP4 GEMM headroom: Gemma 4 31B on one MI355X

Task (lead, 2026-11-22): GEMMs are ~half the GPU time at our recipe's knee. How much faster can they run on one MI355X,
and does any real gain become a measured recipe improvement? Measured 2026-11-22 08:50-09:38 UTC (2:50-3:38 AM Central), 5 spot boxes, ~$7.2.

**Verdict (Phase 1).** The practical ceiling for MXFP4 work on this box is **~5.8-6.2 PFLOP/s, i.e. ~86% of the dense peak
at the clock the GPU actually holds under FP4 load (~1.6-1.8 GHz at 1,340-1,390 W)**, not ~8.4 PF at 2.0 GHz.
- **The engine path sits 16-30% below that ceiling at prefill M.** It runs 4.3-4.9 PF at M=8192, 60-72% of clock-adjusted
  peak at the clock measured in the same window.
- **No backend the image already has closes the gap.** Every other aiter assembly kernel and split-K, CK a4w4 blockscale,
  hipBLASLt and Triton: the best of them saves 0.7-1.7% of a forward's GEMM time at M=4096, 5120, 7680 and 8192, and 2.8-5.4%
  at 6144/6912, where tiles tuned at 8192 are reused.
- **aiter v0.1.30's new persistent 256x256 kernel does.** It is not in the image. Its output matches the engine path (relative error
  < 5e-6) and it is 1.07-1.15x faster at M=8192 on all six shapes, on two boxes, at the same power cap and clock. That is ~8% of the prefill-chunk
  GEMM time, or roughly +3.5-4% throughput if it carries into the engine. Phase 2 tests exactly that, paired.

**Verdict (Phase 2, paired, 2 boxes).** Routing MXFP4 GEMMs with M >= 2048 to the persistent kernel
(`MX_PS_FP4=1`, `box/patch_ps_fp4.py`) is a measured, quality-neutral gain on the same GPUs:
- **Prefill probe:** +4.2% and +3.9% (44,218 -> 46,069 and 43,730 -> 45,437 tok/s).
- **TPOT p50:** 10-30% lower at 3.5 and 3.75 req/s.
- **p95 TTFT at 3.75 req/s:** 4.45 -> **2.74 s (passes)** on seed 7 and 7.06 -> 4.17 s on seed 8 (still fails, 0.17 s over).
- **GSM8K test[:250]:** 247/250 on both boxes.

So 3.75 passes on 1 of 2 seeds, against 0 of 2 here and 0 of 4 in the C1 confirm. **The recipe's claim-grade rate stays
3.5 req/s, with more margin.** Lifting it to 3.75 needs the claim-grade replicate (4 boxes, both seeds, column B). Of the
"+7% for 70% of peak" sizing, the kernel delivered ~4% at the engine level, as predicted from the microbenchmark (GEMMs
~48% of a mixed step x ~8% GEMM time).

## Why it matters

`notes/profile_knee_or_recipe.md`: MXFP4 GEMMs are 50.7% of kernel time under load. In the k=3 + 96-seq profile
(`results/mx7_prof_mtp3_seqs96/20261122T054335Z/prof_summary.txt`), per mixed step (164.5 ms) the f4gemm kernels take
~79 ms (256x256 57.3, 160x384 16.0, the rest <2 each) and per decode-only step (19.5 ms) ~6.3 ms. Mixed steps carry 87% of
the step windows (9,873 of 11,313 ms), so **the prefill-chunk GEMMs (M in the thousands) are the lever; decode-size GEMMs are ~4% of the window.**

FLOP accounting correction: the MXFP4 linears are 29.40 B parameters per forward (qkv 16384/20480 x 5376, o 5376 x
8192/16384, gate_up 43008 x 5376, down 5376 x 21504 over 50 sliding + 10 global layers). lm_head (1.41 B) is BF16. The
earlier "~30.7 B" counted it. So the pure-prefill profile's 357.8 ms is 1.764 PFLOP at **4.93 PF/s**, not 5.1. Share of
FLOPs: gate_up 47.2%, down 23.6%, qkv_s 15.0%, o_s 7.5%, qkv_g 3.7%, o_g 3.0%.

**M is at most 8192 in our recipe.** vLLM v0.30.0's scheduler (`vllm/v1/core/sched/scheduler.py` L581, L655-665) starts the
step budget at `max_num_batched_tokens` and charges running requests their `num_tokens_with_spec`. So decode and draft tokens
come out of the same 8192, and the target forward's M never exceeds 8192 (the brief's 8.2-8.6K does not occur). The k=3
profile's mixed steps average 6,800 prefill + 139 decode tokens, so M ~ 6.9K on average with full chunks at 8192.

## Method

`box/gemm_headroom.py`, run on OpenRelay's image by digest (vLLM 0.30.0, aiter 0.1.21.post2, ROCm 7.2.3 hipcc), no engine
on the GPU, through `run.py` with an idle stdlib http "server" (configs from `tools/mkcfg_gemm.py`).
- **Timing.** Each variant is captured into a HIP graph of back-to-back calls that rotate over >= 768 MB of weight copies,
  3x the 256 MB Infinity Cache, so every call streams its weights from HBM as in the engine, and 2 activation copies. The
  graph is replayed for >= 0.1-0.15 s and timed with events. Data: random normal, quantized with aiter's own per-1x32
  quant (random data is the realistic power case; zeros are not, see below).
- **Same-path reference.** "Engine" = `aiter.gemm_a4w4(..., bpreshuffle=True)`, exactly as vLLM calls it, so it resolves
  the image's tuned CSV (OpenRelay's 44 rows merged with aiter's) with `get_padded_m`, or falls to the C++ heuristic.
  Every variant's output is compared with it (relative L2; accepted < 0.03). Every speedup is engine/variant timed in the
  same phase on the same box.
- **Clock and power per measurement.** amdsmi `gpu_metrics` at 20 Hz (mean GFX clock of the 8 XCDs, the definition
  `tools/gpumon_amd.py` uses, and socket power), median over the measurement window. "% of peak@clk" = achieved /
  (10.066 PF x clk/2400). 10.066 PF = 256 CU x 16,384 FLOP/clk x 2.4 GHz, AMD's dense MXFP4 figure.
- **Shapes.** The six (N, K) above, at M = 16..384 (the CUDA-graph sizes our recipe captures) and 1,024..8,192 (prefill
  chunks), plus 8,576 and 16,384 for reference, and 8192^3 / 16384^3 squares.

Runs (all spot, project=openrelay, every number below cites one):

| run | what | GPU time / cost |
|---|---|---|
| `results/mxG1_asm/20261122T085007Z` | inventory, MFMA ceiling, asm sweep (36 kernels x split-K, 132 points), persistent kernel | 11.7 min, ~$0.93 |
| `results/mxG1_alt/20261122T085009Z` | hipBLASLt, Triton grid, CK a4w4 blockscale (21 ids x split-K, JIT build) | 8.8 min, ~$0.69 |
| `results/mxG1b_ab/20261122T090635Z` | second box: interleaved A/B, engine vs persistent vs best tiles, 3 rounds, M=1024-8192 | 6.8 min, ~$0.54 |
| `results/mxG2_s7/20261122T090631Z`, `results/mxG2_s8/20261122T090633Z` | Phase 2 paired knee (ours vs ours + `MX_PS_FP4=1`) | 31.8 min each, ~$2.52 each |

The confirm phase of `mxG1_asm` aborted in glibc (`free(): invalid pointer`) before any row. `mxG1b_ab` replaces it, with
faulthandler on and one process per shape.

## 1. The ceiling of this box

`results/mxG1_asm/20261122T085007Z/gemm/mfma.jsonl` holds a hipcc-built loop of `v_mfma_scale_f32_32x32x64_f8f6f4`
(FP4 x FP4, e8m0 scales), 2,048 workgroups, 4 independent accumulators per wave.

| operands | held | PF | GFX clock | socket power | % of peak at that clock |
|---|---|---|---|---|---|
| random bits | 3 s | **6.21** | **1,725 MHz** | 1,217 W | 85.8% |
| all zero | 3 s | 8.74 | 2,377 MHz | 880 W | 87.6% |
| random, 16x16x128 MFMA | 3 s | 5.64 | 1,999 MHz | 1,307 W | 67.3% |

- **FP4 MFMA work with real-looking data is power-limited.** The same instruction stream runs at 2.38 GHz on zeros and
  1.72 GHz on random bits. The scaled 32x32x64 MFMA also issues at ~86-88% of nominal per clock even in a tight loop, so
  **~86% of clock-adjusted peak is the instruction-level ceiling, at ~1.7 GHz.**
- **Best GEMM on a large square:** the persistent 256x256 kernel on 16384^3 reaches **5.82 PF at 1,618 MHz = 85.8%**. That
  equals the MFMA loop, so it is the practical ceiling. The image's own 256x256 kernel does 5.66-5.69 PF there
  (`gemm/ps.jsonl`, `gemm/sweep.jsonl`).
- **Under sustained GEMM load the GPU holds ~1.6-1.8 GHz at 1,340-1,390 W** (large-M rows in `gemm/ps.jsonl`, `gemm/ab.jsonl`). The "61% of the ~2.0 GHz clock-adjusted peak" framing
  used the serving-average clock. At the clock the prefill GEMMs actually run at, the engine path is at ~60-72% and the
  ceiling is ~86%.

## 2. Per-shape table

Generated by `python3 -I tools/gemm_table.py --compact` from `results/mxG1_asm/20261122T085007Z` (box A: sweep, ps
phase), `results/mxG1_alt/20261122T085009Z` (box B: CK, Triton, hipBLASLt) and `results/mxG1b_ab/20261122T090635Z` (box C:
3 interleaved rounds, order reversed every other round; engine PF and % of clock-adjusted peak come from box C).
"Best kernel already in the image" is the best of every asm tile/split-K (box A) and CK/Triton (box B), each against its
own same-phase engine path. The persistent kernel's output matched the engine path at every point on both boxes
(relative L2 error < 5e-6).

| shape (N x K), FLOP share | M | engine path kernel | engine PF | % peak@clk | best kernel already in the image | x | persistent 256x256 PF | % peak@clk | x engine (A/B box) | x engine (box A) |
|---|---|---|---|---|---|---|---|---|---|---|
| qkv_s (16384 x 5376), 15.0% | 8192 | heuristic (no row) | 4.59 | 67.5 | asm_256x256 | 0.999 | 4.95 | 72.8 | **1.079** | 1.092 |
| qkv_s (16384 x 5376), 15.0% | 6912 | heuristic (no row) | 4.51 | 65.3 | asm_256x256 | 1.009 | 4.72 | 66.9 | **1.046** | 1.055 |
| qkv_s (16384 x 5376), 15.0% | 4096 | heuristic (no row) | 4.69 | 67.7 | asm_256x256 | 1.007 | 4.79 | 66.5 | **1.021** | 1.041 |
| qkv_s (16384 x 5376), 15.0% | 2048 | heuristic (no row) | 4.59 | 65.4 | asm_default | 1.013 | 4.60 | 62.1 | **1.002** | 1.022 |
| qkv_g (20480 x 5376), 3.7% | 8192 | heuristic (no row) | 4.53 | 66.5 | asm_256x256 | 1.003 | 4.93 | 73.4 | **1.088** | 1.102 |
| qkv_g (20480 x 5376), 3.7% | 6912 | heuristic (no row) | 4.43 | 63.3 | asm_256x256 | 1.007 | 4.76 | 67.4 | **1.073** | 1.087 |
| qkv_g (20480 x 5376), 3.7% | 4096 | heuristic (no row) | 4.57 | 66.9 | asm_256x256 | 1.004 | 4.77 | 68.4 | **1.042** | 1.052 |
| qkv_g (20480 x 5376), 3.7% | 2048 | heuristic (no row) | 4.15 | 51.9 | asm_256x256 | 1.016 | 4.31 | 57.9 | **1.039** | 1.015 |
| o_s (5376 x 8192), 7.5% | 8192 | 256x256 | 4.31 | 60.5 | asm_160x384 | 1.029 | 4.95 | 70.8 | **1.149** | 1.133 |
| o_s (5376 x 8192), 7.5% | 6912 | 256x256 | 4.08 | 52.0 | asm_128x384 | 1.075 | 4.50 | 58.2 | **1.104** | 1.108 |
| o_s (5376 x 8192), 7.5% | 4096 | heuristic (no row) | 3.79 | 49.8 | asm_128x384 | 1.081 | 4.25 | 51.8 | **1.122** | 1.136 |
| o_s (5376 x 8192), 7.5% | 2048 | heuristic (no row) | 3.72 | 48.4 | asm_default | 1.014 | 4.11 | 48.3 | **1.105** | 1.129 |
| o_g (5376 x 16384), 3.0% | 8192 | 256x256 | 4.67 | 69.3 | asm_160x384 | 1.015 | 5.21 | 74.5 | **1.115** | 1.082 |
| o_g (5376 x 16384), 3.0% | 6912 | 256x256 | 4.38 | 58.5 | asm_128x384 | 1.040 | 4.72 | 62.9 | **1.078** | 1.092 |
| o_g (5376 x 16384), 3.0% | 4096 | 128x384 | 4.50 | 62.8 | asm_128x384 | 1.026 | 4.47 | 56.5 | **0.993** | 1.001 |
| o_g (5376 x 16384), 3.0% | 2048 | 128x384 | 4.23 | 57.6 | asm_default | 1.002 | 4.39 | 54.3 | **1.036** | 1.041 |
| gate_up (43008 x 5376), 47.2% | 8192 | heuristic (no row) | 4.55 | 67.3 | asm_default | 1.002 | 4.88 | 71.9 | **1.072** | 1.078 |
| gate_up (43008 x 5376), 47.2% | 6912 | heuristic (no row) | 4.63 | 66.0 | asm_default | 1.003 | 4.91 | 70.3 | **1.061** | 1.070 |
| gate_up (43008 x 5376), 47.2% | 4096 | heuristic (no row) | 4.44 | 63.6 | asm_256x256 | 1.007 | 4.68 | 68.8 | **1.054** | 1.083 |
| gate_up (43008 x 5376), 47.2% | 2048 | heuristic (no row) | 4.23 | 55.6 | asm_default | 1.005 | 4.41 | 61.6 | **1.044** | 1.050 |
| down (5376 x 21504), 23.6% | 8192 | 160x384 | 4.78 | 71.9 | asm_160x384 | 1.015 | 5.23 | 75.2 | **1.094** | 1.089 |
| down (5376 x 21504), 23.6% | 6912 | 160x384 | 4.34 | 58.1 | asm_128x384 | 1.082 | 4.76 | 63.5 | **1.097** | 1.161 |
| down (5376 x 21504), 23.6% | 4096 | heuristic (no row) | 4.24 | 61.0 | asm_128x384 | 1.021 | 4.53 | 56.9 | **1.070** | 1.073 |
| down (5376 x 21504), 23.6% | 2048 | 128x384 | 4.29 | 58.9 | asm_default | 1.011 | 4.42 | 53.9 | **1.030** | 1.042 |
| sq8k (8192 x 8192) ceiling, mxG1_asm ps | 8192 | 256x256 | 5.16 | 73.5 | - | - | 5.52 | 80.6 | 1.071 | |
| sq8k (8192 x 8192) ceiling, mxG1b_ab ab | 8192 | 256x256 | 5.03 | 73.1 | - | - | 5.28 | 76.1 | 1.049 | |
| sq16k (16384 x 16384) ceiling, mxG1_asm ps | 16384 | 256x256 | 5.62 | 82.2 | - | - | 5.82 | 85.8 | 1.036 | |
| sq16k (16384 x 16384) ceiling, mxG1b_ab ab | 16384 | 256x256 | 5.38 | 82.4 | - | - | 5.58 | 83.9 | 1.037 | |

**Practical ceiling per shape:** ~86% of the clock-adjusted peak. That is the pure-MFMA loop (85.8% at 1,725 MHz) and the
persistent kernel on 16384^3 (83.9-85.8%). In absolute terms it is 5.6-6.2 PF at the 1.6-1.7 GHz the GPU holds under
FP4 load.
- **Engine path at M=8192:** 4.31-4.78 PF, 60-72% of clock-adjusted peak. That is 14-26 points (16-30%) below the ceiling,
  so every shape clears the brief's 10-15% bar.
- **With the persistent kernel:** 4.88-5.23 PF, 71-75%, closing roughly half of the gap.
- **The full per-(shape, M) table**, including decode M = 16-384 and M = 8576/16384, is `python3 -I tools/gemm_table.py`.

**Layer-weighted** (one forward's 240 MXFP4 GEMMs, per-shape time x layer count, per box), the persistent kernel where
it wins, else the engine path:

| M | box A (`mxG1_asm` ps phase) | box C (`mxG1b_ab`, interleaved) |
|---|---|---|
| 1536 | - | 23.797 -> 21.611 ms (9.19%) |
| 2048 | 28.136 -> 26.832 ms (4.64%) | 28.358 -> 27.294 ms (3.75%) |
| 3072 | - | 39.531 -> 37.049 ms (6.28%) |
| 4096 | 53.731 -> 49.977 ms (6.99%) | 55.024 -> 52.095 ms (5.32%) |
| 5120 | - | 67.157 -> 62.413 ms (7.06%) |
| 6144 | - | 81.632 -> 72.477 ms (11.21%) |
| 6912 | 88.368 -> 80.866 ms (8.49%) | 90.722 -> 84.657 ms (6.69%) |
| 7680 | - | 99.543 -> 92.692 ms (6.88%) |
| 8192 | 101.08 -> 92.921 ms (8.07%) | 104.842 -> 96.577 ms (7.88%) |

So the routed kernel saves **~7-8% of a full chunk's GEMM time** (M=8192), 5-11% at M=4096-7680 and ~4% at M=2048.
At M=1024 it loses on the N=5376 shapes (0.68-0.70x: 84 tiles for 256 CUs), so the route starts at M >= 1536.
Mixed steps are ~48% GEMM, so the expectation for the engine is **~3.5-4% shorter prefill-heavy steps**. Phase 2
measures it.

## 3. Backends, each against the engine path on the same box

| backend | best at M=8192 vs engine | notes |
|---|---|---|
| every aiter asm f4gemm kernel (35 tiles, + log2 split-K 1-3 on 256x256 / 128x512) | 1.000-1.029x (o_s 160x384 1.03x) | the engine path already lands on the best tile at 8192, except N=5376 rows tuned at 8192 being reused at other M (below) |
| **aiter v0.1.30 persistent 256x256 (`_ps`)** | **1.078-1.133x on all six** | output matches (rel. error < 5e-6); 1.07-1.15x on box C's interleaved rounds; not in the image; needs v0.1.30's 376-byte kernarg (num_cu at offset 372), so it cannot be enabled by a CSV row |
| CK a4w4 blockscale (21 kernelIds x splitK 0-3) | 0.69-0.71x | ties only gate_up at M <= 64 (0.999-1.012x); JIT module built in 219 s (`results/mxG1_alt/20261122T085009Z/gemm/ck.jsonl`) |
| aiter Triton `gemm_afp4wfp4_preshuffle` (10 configs) | 0.59-0.76x (best 256x256x256, 8 warps) | BLOCK_K 512 does not compile for K=5376 |
| hipBLASLt (`torch._scaled_mm`, MXFP4 e8m0 1x32) | 0.40-0.66x (1.9-3.2 PF; vs box A's engine path, box B timed no engine in this phase) | `hipblaslt-bench` is not in the image, so no all-solutions sweep |

## 4. The three hypotheses

**Wave quantization on N=5376 (o_proj, down; 34% of FLOPs).** Partly right, but not at M=8192.
- At M=8192 the tuned rows already sit on 2.6-2.8-wave tiles and no existing tile beats them by more than 3%.
- The loss is at other M. Rows tuned at M=8192 are reused for every M in (4096, 8192] by `get_padded_m`, and the best tile
  changes with M. down at M=6144: the row's 160x384 is 2.13 waves against 256x256's 1.97, so 256x256 is 1.24x faster.
  o_g at M=3072 (4096 row's 128x384): 1.27x. o_s/down at M=6912: 1.08x.
- Layer-weighted over all 240 GEMMs of a forward, M-exact tiles save 1.0-5.4% at M=3072-6144 and 0.7% at 8192.
- Split-K helps only down/o_g at M=16-32 (256x256 ks3: 1.09-1.25x). Stream-K (fractional tail waves) does not exist in
  aiter's FP4 asm kernels. The new `_ps` kernel is persistent but does not split the tail (`grid = min(tiles, CUs)`;
  K slices are dropped until tiles fit). Its gain at M=8192 is not a wave effect: it holds on the squares too.

**Decode-step GEMMs (M = 16-384).** Weight streaming is slow, and tuning does not fix it. At M=128-160 the engine path
streams qkv at 2.6-3.2 TB/s, gate_up 3.6-4.1 TB/s, and **o_proj/down at only 1.5-1.8 TB/s** (~8 TB/s HBM). The best
existing kernel moves those by <1-8%. Most decode M do resolve to tuned rows via the padded lookup (exceptions: qkv_g at
M=96/128 and the large-N shapes at prefill M have no rows). Layer-weighted, the best existing kernels save 0.7-5.7% of
decode-step GEMM time at M=128-160. Decode-only steps are ~13% of the knee window and GEMMs ~32% of them, so that is <0.3%
of throughput. The cause is too few workgroups for a 5376-wide output (32x128 tiles: 42 workgroups on 256 CUs). That is a
kernel to write, not a row to tune (section 6).

**Newer aiter.** All 35 FP4 assembly kernels in v0.1.30 are byte-identical to the image's (git blob hashes, tags
v0.1.21.post2 vs v0.1.30), and the CK instance list is unchanged (21). The one new kernel is
`f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ps.co` (sha256 8b541b78...). v0.1.30 also adds Triton/FlyDSL rows to the
tuned-CSV dispatch, and FlyDSL a4w4 for gfx1250 only. The persistent kernel is the only usable new piece. It cannot ride
the tuned CSV: the 0.1.21 C++ dispatcher has no `ps` column and passes a 372-byte kernarg. So Phase 2 launches it through
a small ctypes path (`box/patch_ps_fp4.py`).

## 5. Phase 2: the persistent kernel in the recipe, paired

**Setup** (`tools/mkcfg_gemm.py G2`, configs `configs/mxG2_s7.json`, `configs/mxG2_s8.json`). On each box the same MI355X
served two arms back to back, weights kept:
- `mx_recipe`: our recipe as in the C1 confirm. OpenRelay's image by digest + MTP k=3 + `--max-num-seqs 96` + graphs to
  384 + `MX_DRAFT_COMPACT=1`.
- `mx_ps`: the same + `MX_PS_FP4=1 MX_PS_MIN_M=2048`.

The seed-7 box ran the ps arm first, the seed-8 box the baseline first. Their client (`vllm bench serve`, random tokens,
300 s per rate) at 3.5 and 3.75 + the prefill probe; GSM8K test[:250] on the ps arm, [:100] on the baseline.

**What the patch does** (`box/patch_ps_fp4.py`, applied at setup; the .co fetched from the v0.1.30 tag and checked by
sha256):
- A 5-line route at the top of `aiter.gemm_a4w4`'s gfx950 path sends calls with M >= `MX_PS_MIN_M` to aiter v0.1.30's
  `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ps`. The launch goes through ctypes with v0.1.30's 376-byte kernarg
  (num_cu at offset 372), and the route uses the operand checks from v0.1.30's `ps_supported()`.
- Anything else falls through unchanged. Decode steps (M <= 384, CUDA graphs) never route. The output buffer is padded to
  256 rows, so a 256-aligned profile run allocates the same.
- `box/serve_ps.sh` refuses to start the ps arm unless the patch reported `applied`.
- Both ps EngineCores logged the route for all six (N, K) at the profile run (M=8192, activation scales padded to 256
  rows), at least 696,000 routed launches, and zero fallbacks (`server_mx_ps.log`).

**Result** (`python3 -I tools/ps_paired_table.py G2`, from the run files: `results/mxG2_s7/20261122T090631Z`,
`results/mxG2_s8/20261122T090633Z`; GFX clock = box median under load, `tools/gpumon_amd.py`):

| box (run) | GFX MHz | engine | seed | rate | p95 TTFT s | p50 TTFT s | TPOT p50 ms | pass | errors | prefill tok/s | GSM8K | accept/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mxG2_s7/20261122T090631Z | 2023 | ours | 7 | 3.5 | 2.27 | 0.93 | 29.8 | PASS | 0 | 44218 | 99/100 | 3.58 |
| mxG2_s7/20261122T090631Z | 2023 | ours+ps | 7 | 3.5 | 3.22 | 0.89 | 26.9 | PASS | 0 | 46069 | 247/250 | 3.59 |
| mxG2_s7/20261122T090631Z | 2023 | ours | 7 | 3.75 | 4.45 | 2.44 | 54.5 | fail | 0 | 44218 | 99/100 | 3.58 |
| mxG2_s7/20261122T090631Z | 2023 | ours+ps | 7 | 3.75 | 2.74 | 1.19 | 39.2 | PASS | 0 | 46069 | 247/250 | 3.59 |
| mxG2_s8/20261122T090633Z | 1985 | ours | 8 | 3.5 | 3.06 | 1.10 | 37.5 | PASS | 0 | 43730 | 99/100 | 3.59 |
| mxG2_s8/20261122T090633Z | 1985 | ours+ps | 8 | 3.5 | 2.13 | 0.75 | 26.3 | PASS | 1 | 45437 | 247/250 | 3.60 |
| mxG2_s8/20261122T090633Z | 1985 | ours | 8 | 3.75 | 7.06 | 3.18 | 55.2 | fail | 0 | 43730 | 99/100 | 3.59 |
| mxG2_s8/20261122T090633Z | 1985 | ours+ps | 8 | 3.75 | 4.17 | 1.39 | 44.4 | fail | 0 | 45437 | 247/250 | 3.60 |

- **Prefill** (pure GEMM-heavy steps): **+4.2% / +3.9%**, against ~4% predicted from the microbenchmark.
- **At 3.75 req/s** the ps arm cuts p95 TTFT by 38-41% and TPOT p50 by 20-28%. It passes on seed 7 (2.74 s) and misses
  on seed 8 by 0.17 s (4.17 s). The baseline fails both (4.45 / 7.06 s), in line with C1's 4.19-6.84 s at 3.75.
- **At 3.5:** seed 8 p95 3.06 -> 2.13 s. On seed 7 the ps arm's p95 is higher (2.27 -> 3.22 s, both pass) with lower
  TPOT (29.8 -> 26.9 ms) and p50 (0.93 -> 0.89 s). One seed-7 cell at 300 s is a single Poisson draw: C1 put ours at
  2.23-2.69 s at 3.5 across four boxes. This cell is the one adverse point in the pair, and the claim-grade replicate
  should settle it.
- **Quality:** GSM8K test[:250] 247/250 (98.8%) on both ps arms; baseline 99/100 on both. MTP acceptance unchanged
  (3.58-3.60 tokens per verify step). Errors: 1 failed request in one ps cell (~1 in 1,000, the known random-token
  detokenization failure that hits both recipes).
- **Cost:** 31.8 min per box, ~$2.52 each; the track total is ~$7.2 (Phase 1 $2.16 + Phase 2 $5.04, the runs'
  `cost_measured.json`).

**What it is and is not.** It is a vLLM-0.30-image patch that uses one kernel from an aiter release cut today (v0.1.30;
the `_ps` .co is absent from every earlier tag, v0.1.22 through v0.1.29.post1, so vLLM 0.34's aiter 0.1.28.post1 does not
have it either). It is not yet claim-grade: 2 boxes, one seed each, random-token client only.

**Threshold.** `mxG1b_ab` shows the kernel already wins from M=1536 (o_s 1.62x, layer-weighted 9.2% of GEMM time at
1536) and loses at 1,024 on the N=5376 shapes. The replicate should use `MX_PS_MIN_M=1536`.

## 6. Phase 3 (report only): what a custom kernel would need

After the persistent kernel, prefill-chunk GEMMs run at 71-75% of clock-adjusted peak, against a ceiling of ~86%. The
ceiling binds even for a bare MFMA loop: the scaled 32x32x64 MFMA does not issue at its nominal rate, and the clock is
held by the ~1,380-1,400 W cap. Decode-size GEMMs stream weights at 1.4-4.3 TB/s against ~8 TB/s HBM. What would close
each part, largest first:

1. **Fuse GELU(tanh)*up + MXFP4 quant into the gate_up epilogue** (gate_up is 47% of the FLOPs). Today the M=8192 gate_up
   call writes 705 MB of bf16 (8192 x 43008 x 2 B). `_act_mul_and_dynamic_mxfp4_quant_kernel` then reads it back and
   writes FP4; that kernel alone is 7.2% of pure-prefill GPU time (`notes/profile_knee_or_recipe.md`).
   - The epilogue needs each 256-column output tile to hold matching gate and up columns: interleave them in the
     preshuffled weight at load time. It then emits FP4 + e8m0 scales directly.
   - Bound: most of that 7.2% plus less HBM traffic under the power cap, so a higher clock for everything else.
   - This is a new gfx950 assembly/HIP epilogue on top of the `_ps` kernel. OpenRelay's Triton fusion of the same op
     (`patch_g4_fuse.py`) only removed the elementwise kernel, not the bf16 round trip.
2. **Stream-K tail for the N=5376 shapes** (o_proj, down: 34% of the FLOPs). At M=8192 the 256x256 grid is 672 tiles =
   2.625 waves on 256 CUs, and `_ps` does not split the last 0.625 wave. Splitting the tail tiles' K across the idle CUs,
   with a fix-up reduction, is worth up to 3/2.625 = 1.14x on those shapes, i.e. ~4% of GEMM time. The persistent loop
   is the natural base. aiter has no stream-K FP4 kernel.
3. **Split-K weight streaming for decode** (M <= 384, inside CUDA graphs). o_proj/down launch 42 workgroups (32x128
   tiles on N=5376) and stream 1.4-1.8 TB/s. A GEMV-style FP4 kernel would use 256-512 workgroups, each streaming a K
   slice of the preshuffled weight, reduced in fp32. At ~5 TB/s on all six shapes, a decode-only step's 240 GEMMs
   (6.3 ms at M=160) would take ~3 ms. That is ~16% of a decode-only step, ~2% of the knee window, and a lower TPOT.
4. **Not worth it:** RMSNorm + quant prologue fusion (OpenRelay measured 0-3%, slower at decode); Triton or hipBLASLt
   rewrites (0.4-0.76x today); M-exact tuned rows once `_ps` is routed (it beats every tile at M >= 1536 except o_g at
   M=4096, 0.99x).

**Realistic total beyond Phase 2:** prefill GEMMs at ~80-82% of clock-adjusted peak (items 1-2) is another ~8-10% of GEMM
time, so ~4-5% throughput, plus ~1-2% from decode (item 3). These are hand-written gfx950 kernels with scaled-MFMA
instructions and a weight-layout change. That is weeks of kernel work, not a weekend change; nothing in aiter v0.1.30
provides them.

## Files

| path | what |
|---|---|
| `box/gemm_headroom.py` | the microbenchmark (phases inventory, mfma, sweep, ps, confirm, ab, hipblaslt, triton, ck; `selftest` runs locally) |
| `box/patch_ps_fp4.py`, `box/serve_ps.sh` | Phase 2: env-gated route of M >= 2048 MXFP4 GEMMs to the v0.1.30 persistent kernel; the serve guard |
| `tools/mkcfg_gemm.py` | configs `mxG1_asm`, `mxG1_alt`, `mxG1b_ab`, `mxG2_s7`, `mxG2_s8` |
| `tools/gemm_table.py` | the per-shape table (`--compact` = section 2) and layer-weighted savings, from the run files only |
| `tools/ps_paired_table.py` | the Phase 2 paired table, incl. whether the ps arm actually routed (server log) |
| `research/raw/openrelay_recipe/full/bench/kernels/` | OpenRelay's `gemm_bench.py`, `hipblaslt_fp4.py`, `g4_a4w4_untuned.csv`, `fuse_bench2.py` @6767b4d (fetched 2026-11-22) |

## Definition of done

DoD gemm-headroom: BASELINE: the C1 paired confirm (ours 3.5 pass / 3.75 fail at 4.19-6.84 s; G2's baseline arms 3.5 pass,
3.75 fail at 4.45 / 7.06 s) + OpenRelay's kernels.md (asm 4.1-4.7 PF, hipBLASLt 1.7-2.4, Triton 1.7-1.9 at M=16384; here
engine 4.4-4.9, hipBLASLt 1.9-3.2, tuned Triton 2.2-3.6) | GAP-HUNT: gemm_table.py --selftest, ps_paired_table.py G2,
regenerated configs vs launched config.json, every note number re-derived from the run files (6 corrected) | GAPS-FILED:
t018 (claim-grade replicate), t020 (confirm-phase glibc abort), t021 (Phase 3 kernels)

Ledger: t017 (done), t019 (journal steps 400-404, done), t018 / t020 / t021 open.
