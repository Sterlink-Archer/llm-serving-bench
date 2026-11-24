# Playbook: Minimizing Cost per Token for Gemma 4 31B and GLM‑5.3‑Flash on Rented GPUs (state as of Nov 21, 2026)

The highest-leverage path is: serve **Gemma 4 31B** on vLLM (≥0.30) with a 4‑bit checkpoint matched to the hardware (NVFP4 on Blackwell, MXFP4 + AITER on MI355X, W4A16 QAT elsewhere), FP8 KV cache, a context cap well below 256K for the bulk pool, fixed attention-backend routing (the 512‑dim global heads are the #1 hidden performance trap), and the official MTP drafter tuned *per concurrency level*; then serve **GLM‑5.3‑Flash** on 4×B200/B300 or 4–8×H200 (or MI355X) with the native FP8 or NVFP4 checkpoint, its built‑in MTP head, prefix caching, and a properly sized KDA‑state/KV split. GLM‑5.3‑Flash is not a viable production target on RTX 4090s, and Gemma 4 31B on a single 4090 is memory-starved for 128-way concurrency.

## TL;DR

- **Gemma 4 31B:** the fast path exists but is not the default. Make sure the 512‑dim global attention layers do not fall back to Triton (vLLM PR #47547 / FlashInfer ≥0.6.16, FA4 on Blackwell, AITER tile tables on MI355X). Then stack a 4‑bit checkpoint, FP8 KV, `--limit-mm-per-prompt` zeroed for text pools, `--async-scheduling`, CUDA-graph sizes that cover batch×(1+draft), and MTP (2–2.5× at low concurrency, only ~1.09× at C100). OpenRelay's Sep 29, 2026 blog reports "1.5x traffic per GPU at the same latency target" on one MI355X in one evening: 2.0 to 3.0 req/s per GPU at p95 TTFT under 4 s, TPOT 40–43 → 22 ms, prefill 27.8K → 41.9K tok/s, GSM8K 98.8% before and after. Reproduce that open-loop harness (300 s per rate, worse of two seeds) first.
- **GLM‑5.3‑Flash:** it is a 321B/18B‑active hybrid (KDA linear attention + NoPE sparse MLA/DSA, 288 experts top‑8, 1 MTP layer, mHC). Supported recipes exist for vLLM (nightly/≥0.29 + PR #62424, Rust frontend), SGLang (≥0.5.20), TokenSpeed, NVIDIA Dynamo 1.5.1, NIM 2.1.4 and TRT‑LLM (NVFP4 PR #20127). Weights are ~306 GiB at FP8 and ~184 GiB at NVFP4, so the smallest sane nodes are 4×H200/B200/B300, 4×MI300X/MI355X, or 2–4×RTX PRO 6000 with 4‑bit experts. Hopper must run BF16 KV for this model. The biggest tunables are the KDA-state vs KV pool split (+26–36% on B300 in SGLang's cookbook), MTP depth vs concurrency, DP+EP "Mega MoE" on H200 (+22–44%), and PD disaggregation.
- **Economics:** for agentic, long-prompt traffic, revenue per GPU-hour is dominated by *input* tokens, so prefill throughput and cache-hit accounting matter as much as decode tok/s. On OpenRelay's MI355X Gemma workload (9.5K in / 380 out, 3.0 req/s per GPU), uncached input is ~85% of list-price revenue.\[1\] Benchmark goodput at a TTFT/TPOT SLO with the real input/cached/output mix, not raw output tok/s.

---

## 0. Ground truth for the agent: model facts and break-even framing

**Gemma 4 31B (google/gemma-4-31B-it).** 60 layers, hidden size 5,376.\[2\] There are 50 sliding-window layers with head_dim 256 (32 query / 16 KV heads) and 10 global layers with head_dim 512 (32 query / 4 KV heads, i.e., 8 query heads per KV head).\[1\] Native context is 262,144.\[2\]\[3\] The heterogeneous head dims (256/512) are the root of most Gemma 4 serving pathologies: FlashAttention‑2 tops out at head_dim 256.\[4\]\[5\]

**GLM‑5.3‑Flash (zai-org/GLM-5.3-Flash).** ~321B total / 18B active.\[6\] 45 language layers combine KDA linear attention with NoPE sparse MLA (DSA), with 288 routed experts of which 8 are active.\[7\] There is one MTP draft layer, a 24‑layer vision encoder, and a declared max context of 1,048,576.\[6\]\[7\]\[8\] The default repo *is the native FP8 checkpoint* (~306 GiB of weights); the BF16 variant is `zai-org/GLM-5.3-Flash-BF16` and needs roughly twice the weight memory.\[7\] Thinking is always on, with `reasoning_effort` low/high/max and max as the default.\[7\]\[9\]\[10\] That matters for cost: output-token volume per request is set by the client's effort level, and output is the expensive token class.

**Break-even math (aggregate output, from the brief):** at $0.30/M output, a GPU must produce ~926 output tok/s per $1/hr; at $0.18, ~1,543; at $0.09, ~3,086. **Always add input revenue.** On OpenRelay's MI355X Gemma workload (9.5K input / 380 output per request), 3.0 req/s/GPU is 28.5K input tok/s ($7.18/hr at $0.07/M uncached) plus 1,140 output tok/s ($1.23/hr at $0.30/M).\[1\] Input is ~85% of revenue at 0% cache hits. At high cache-hit rates input revenue falls ~10× (cached input is $0.007), but so does prefill cost. Track revenue per GPU-hour as Σ(tokens × price) for {uncached input, cached input, output}.

---

## 1. Serving engines

| Engine | Gemma 4 31B | GLM‑5.3‑Flash | Notes / versions |
|---|---|---|---|
| **vLLM** (V1, Model Runner V2) | **Supported**: recipe requires ≥0.19.1; MTP via `google/gemma-4-31B-it-assistant`. The recipe page still says MTP is nightly-only (images `gemma4-0505-cu129/cu130`); field reports show MTP working on 0.30. NVIDIA Hopper/Blackwell, AMD MI300X–MI355X, TPU, Intel XPU.\[3\]\[11\]\[12\] | **Supported (new)**: recipe says "Requires vLLM ≥ 0.29.0 (nightly)" plus native GLM5Next PR #62424 and FlashInfer ≥0.6.17/0.6.18. Hopper+ on NVIDIA, gfx950 on AMD (model support PR #53906), Ascend 950PR.\[7\] | v0.29.0 (Sep 9, 2026) made Model Runner V2 the default and added `--per-request-spec-decode-metrics`.\[13\]\[14\] The latest stable is v0.32.0 (Oct 18, 2026) per a secondary source; verify.\[15\] **Rust frontend** (`VLLM_USE_RUST_FRONTEND=1`, `--tokenizer-mode glm5_next`) is the GLM default but is *text-only*; multimodal needs the Python frontend.\[7\] |
| **SGLang** | Supported (0.5.20 tested). Upstream EAGLE3 for Gemma 4 needed a fork (`tails-mpt/sglang`). DFlash fixes in PR #52081.\[16\]\[17\]\[18\] | **Supported day‑0**, ≥v0.5.20. Cookbook covers H100/H200/B200/B300/GB200/GB300/MI300X/MI325X/MI355X/TPU7x/Ascend. ROCm image `lmsysorg/sglang:v0.5.22-rocm10-mi30x`. AMD enablement PR #36607 (GSM8K 97.35% MI300X, 97.65% MI355X).\[8\]\[19\]\[20\] | SM120 (RTX PRO 6000) needs community image `ghcr.io/ormandj/sglang-glm53-flash-sm120`. The official SGLang 0.5.20 has no DSA backend on RTX PRO Blackwell (per NIM docs).\[21\]\[22\] |
| **TensorRT‑LLM** | Not verified in this pass, so treat as *unknown/build*. | Partial: glm5_next FP8 supported; NVFP4 loading added in PR #20127 (GSM8K 92.12 NVFP4 vs 92.91 FP8, 4×B200 TP4/EP4 MTP3).\[23\] | Check the merge status of #20127 before use. |
| **NVIDIA Dynamo** | Generic vLLM/SGLang backends; no Gemma-specific recipe found. | **Supported**: v1.5.1 recipes for GB200 (TP4 agg/disagg over MNNVL) and H200 (TP8, MTP7, DeepGEMM, FLASHMLA_SPARSE, KV-aware routing, NIXL).\[24\] | GB200 disagg uses UCX over TCP because the image's UCX lacks MNNVL IPC.\[24\] |
| **NVIDIA NIM** | Not verified. | Supported via `sglang-model-free-nim:2.1.4-variant-glm-5.3-flash` (NVFP4). No DSA on RTX PRO 6000/4500.\[22\] | — |
| **TokenSpeed** (lightseek) | Not verified. | Supported day‑0: `ts serve ... --tensor-parallel-size 8 --enable-expert-parallel --moe-backend flashinfer_trtllm --kv-cache-dtype fp8 --speculative-algorithm MTP --speculative-num-steps 2 --speculative-num-draft-tokens 3`.\[25\]\[26\] | C++ control plane; worth an A/B against vLLM/SGLang on GLM. |
| **AMD ATOM** | Supported on MI355X (ROCm/ATOM PR #546: standalone + vLLM plugin; MXFP4-alllinear GSM8K ≈0.95).\[27\] | Not verified, so treat as unknown. | ATOM is an AITER-centric engine and a vLLM OOT plugin.\[28\] |
| **KTransformers** | n/a | Supported (CPU–GPU hybrid; SM89/SM120; ≥350 GB host RAM).\[10\]\[29\] | Not a throughput-at-SLO engine; skip for production. |
| **llama.cpp** | Supported, including the MTP drafter (`--spec-type draft-mtp`, PR #23398 merged 2026‑06‑07).\[30\] | **Not upstream**: GGUFs need the Unsloth fork or unmerged PRs #27752/#27754/#27773.\[31\]\[32\] | Weak batching; use only as a baseline. |
| **ExLlamaV3 / TabbyAPI** | Not verified. | Community EXL3 2–3 bpw runs (2×RTX 5090: ~26 tok/s generation at 3.05 bpw).\[33\] | Single-user/local only. |
| **vLLM production‑stack, llm‑d, LMDeploy, TGI, MLC‑LLM** | **Not verified in this pass.** | **Not verified.** | The agent must check model support. TGI's README says "text-generation-inference is now in maintenance mode" and recommends vLLM or SGLang; Hugging Face dates maintenance mode to Dec 11, 2025. |

**Field comparison (Gemma 4 31B, TP4, agricultural chat):** vLLM 0.30 + MTP4 gave the best median TTFT of 474 ms at c5, vs 1,020 ms for SGLang 0.5.20 and 912 ms for plain vLLM 0.30. SGLang + DFlash improved TTFT but worsened overall latency.\[18\] Default to vLLM for Gemma and A/B SGLang/TokenSpeed for GLM.

---

## 2. Kernels and attention backends

**Gemma 4 attention (most important item).**
- vLLM 0.19.0 forced `TRITON_ATTN` for *all* layers because of the 256/512 head dims (issue #38887; ~9 tok/s for E4B on an RTX 4090).\[34\]\[35\]\[36\]
- PR #38891 introduced per-layer selection: sliding layers get FlashAttention, global layers stay on Triton.\[36\]
- PR #47547 (open since Jul 3, 2026) removes the forced fallback so FlashInfer serves the 512‑dim global layers when FA4 is unavailable. It requires FlashInfer #3739, which ships in every FlashInfer release from v0.6.16 (Jul 31, 2026).\[37\]
- Community A100 measurement (vLLM 0.28, W4A16 + MTP, 30K prompt): forced Triton gave 31.50 tok/s with MTP and 43.85 without. The selector path (FA2 + FlashInfer) gave 66.09 with MTP and 48.95 without. At 50K, Triton collapses to 15.75 vs 41.35 (issue #52049).\[37\]
- Caveats: multimodal-enabled deployments still exclude FlashInfer at validation until #46558. Clear `~/.cache/vllm/torch_compile_cache` when toggling `limit_mm_per_prompt` (#50891 crash).\[37\]
- **Action:** grep startup logs for `Using AttentionBackendEnum...` on every config. Treat any TRITON_ATTN on global layers on Ada/Hopper as a bug to fix (patch in #47547 or force `--attention-backend FLASHINFER` where supported).
- Support by hardware: Ada (4090/L40S) and Hopper need the FlashInfer h512 path; Blackwell has an FA4 path; MI355X uses AITER unified attention, whose tile table lacked a prefill entry for 512‑dim heads (OpenRelay's fix gave attention 2.6× and prefill +33%, ROCm/aiter#5926).\[1\]

**AMD AITER (Gemma 4 on MI355X):**
- AITER PRs from OpenRelay: #5926 (512‑dim prefill tile, 2.7×), #5929 (split‑KV for spec decode, up to 3.8×), #5928 (Gemma GEMM rows, up to 1.8×), #5927 (no recompile per batch size).\[1\]
- vLLM #59153: fused GELU+MXFP4 quant, +1.8% prefill.\[1\]
- Set `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1` on vLLM after v0.30.0. Without it, MXFP4 GEMMs fall back to Triton and prefill drops from 27.7K to 18.9K tok/s.\[1\]
- Didn't help: Triton/hipBLASLt MXFP4 GEMMs (2.5× slower than AITER asm), vLLM QK‑norm/RoPE fusion passes (layout mismatch).\[1\]

**GLM‑5.3‑Flash kernels:**
- NVIDIA: NoPE sparse MLA needs FlashInfer ≥0.6.17 (0.6.18 to avoid init errors).\[7\] Hopper uses FLASHMLA_SPARSE + DeepGEMM (Dynamo H200 recipe).\[24\]
- `--moe-backend deep_gemm_mega_moe` (vLLM #66812) fuses FP8 routed and shared experts into one DeepGEMM kernel. It needs DP+EP with TP1 and is text-only; +22–44% output tok/s on 8×H200.\[7\]
- GB300 only: `--additional-config '{"decode_backend":"latency_specialized"}'` (vLLM #63102) uses fused CuTe DSL decode for steps ≤64 tokens. Median ITL fell from 5.15 to 2.93 ms at c=1 and from 13.92 to 12.00 ms at c=64; GSM8K 0.918 vs 0.914.\[7\]
- SGLang knobs: `--dsa-prefill-backend/--dsa-decode-backend` (tilelang | trtllm), `--linear-attn-backend triton`, `--moe-runner-backend` (flashinfer_trtllm on Blackwell, DeepGEMM on Hopper, aiter on MI355X), and `--enable-linear-replayssm-spec` for spec decode through KDA. Keep KV dtype and DSA backend paired; one SM90-only backend from PR #41790 forbids FP8 KV.\[8\]
- AMD:
  - `VLLM_ROCM_USE_AITER=1`; on gfx950 also `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` and `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1`, plus `--attention-backend ROCM_AITER_MLA_SPARSE`.\[7\]
  - For ≥64K prompts on gfx950, add `--sparse-indexer-topk-backend aiter`.\[7\]
  - SGLang on MI300X: `SGLANG_AITER_GFX942_BLOCKSCALE_USE_CK=1` gave +5–17% throughput per GPU and −11–23% TPOT (default after #43815). Use `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`. gfx942 needs `SGLANG_OPT_FUSE_MHC_POST_PRE=0`, `SGLANG_DSA_FUSE_TOPK=0` and `SGLANG_OPT_USE_TOPK_V2=0`.\[8\]
- SM120 (RTX PRO 6000/5090): there is no upstream NoPE sparse‑MLA path. The community kernel is `VLLM_GLM53_CUDA_SPARSE_MLA=1` (Libertai repo). The NVFP4 MoE `w13_input_scale` bug zeroes expert outputs ("locklocklock"); the workaround is `VLLM_GLM53_MOE_INPUT_SCALE=1.0`.\[38\]

**CUDA graphs and compile:**
- Capture sizes must cover max batch × (1 + num_draft). OpenRelay found 63% GPU idle per decode step when graphs stopped at 64 tokens with 2 draft tokens; capturing to 256 gave a 2.7× shorter step (+19% req/s), at a cost of 7 s startup and 3.1 GiB.\[1\]
- Graphs to 16K tokens bought nothing for 14 GiB.\[1\]
- SGLang: `--cuda-graph-max-bs-decode` ≈ 2× concurrency. `--cuda-graph-backend-prefill breakable` (PR #38522) is NVIDIA only.\[8\]
- **Not verified in this pass, so the agent should inventory:** FA3/FA4 version matrix, Marlin/Machete, DeepEP/other EP comm kernels, custom all-reduce/all-reduce fusion flags, torch.compile fusion passes for Gemma's layout.

---

## 3. Quantization

**Existing Gemma 4 31B checkpoints:**
- BF16, ~59 GB.\[39\]
- FP8: `RedHatAI/gemma-4-31B-it-FP8-dynamic` (Hopper/Blackwell; vision tower stays BF16)\[3\] and `RedHatAI/gemma-4-31B-it-FP8-block`.
- INT4 QAT: `google/gemma-4-31B-it-qat-w4a16-ct`, ~19.8 GB, auto-detected by vLLM.\[40\]\[41\]\[42\]
- NVFP4: `nvidia/gemma-4-31B-it-NVFP4` (Blackwell)\[3\] and `RedHatAI` NVFP4.
- MXFP4: `amd/gemma-4-31B-it-MXFP4` (MI355X).\[1\]
- AutoRound: `Intel/gemma-4-31B-it-int4-AutoRound`.\[43\]
- Community AWQ/NVFP4 variants.

**Measured on 1× RTX PRO 6000** (1K in / 200 out, FP8 KV; third-party, self-promoting card): batched decode was 494 tok/s for BF16, 913 for nvidia NVFP4, 1,244 for the "turbo" NVFP4 repack, and 757 for AWQ/compressed-tensors (Marlin path).\[44\] Takeaway: on Blackwell, use FP4 tensor-core kernels, not Marlin W4A16. The Marlin path wins single-stream (64 vs 39 tok/s) but loses at batch.\[44\]

**Existing GLM‑5.3‑Flash checkpoints:**
- Native FP8 block-scale (default).
- BF16.
- `nvidia/GLM-5.3-Flash-NVFP4`: ModelOpt 0.47; experts + dense MLP quantized, FP8 KV cast; GPQA 0.9189; reported within ~2 points of BF16.\[6\]\[45\]
- `RedHatAI/GLM-5.3-Flash-NVFP4`: experts only, MTP kept FP8; GSM8K‑Platinum 97.74%, GPQA 90.57%; ~184 GiB.\[46\]\[47\]\[48\]
- `nvidia/GLM-5.3-Flash-FP4-Local`: QAD, ~166 GiB.\[49\]
- `OneNexus/GLM-5.3-Flash-MXFP4` (MI355X).\[8\]
- `ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO` (SM120 TP2).\[21\]
- Ascend W8A8 / W8A8‑MXFP8.\[8\]
- Unsloth GGUF (UD‑Q4_K_XL ~200 GB; IQ1_S 93 GB, top‑1 agreement 70.9%) and EXL3.\[6\]\[31\]

**KV cache:**
- Gemma: `--kv-cache-dtype fp8` saves ~50% of KV memory.\[3\] One tuned H100 recipe uses FP8 e4m3 KV + prefix caching + 0.95 utilization.\[2\]\[50\]
- GLM on Blackwell/MI355X: FP8 KV is the default.\[8\]
- GLM on Hopper: **FP8 KV not supported; run BF16 KV.**\[7\]
- GLM on SM120: an FP8 KV route was added by community PRs; it gives a 1.88× larger pool (7.80M vs 4.14M tokens on 3×PRO 6000).\[51\]
- **Build-our-own:** calibrated FP8 KV scales for Gemma (the default is unit scales) and FP4 KV. Gate both with long-context needle tests.

**Quality gates:**
- OpenRelay's own gate is GSM8K ≥96.5%; its MI355X run read 98.8% before and after.\[1\]
- Add an lm-eval subset (GSM8K, MMLU-Pro sample, IFEval), a long-context retrieval probe at 32K/128K/256K, and greedy reference-agreement (token match rate vs BF16 on 500 prompts).
- For GLM, also tool-call parse success: the `glm47` parser silently swallows calls if misconfigured.\[52\]

**Produce ourselves:**
- ModelOpt NVFP4/FP8 for Gemma variants (e.g., NVFP4 attention + MLP with FP8 KV scales).
- AMD Quark MXFP4 "all‑linear" Gemma: AMD-internal, validated in ATOM PR #546 but weights not published.\[27\]
- llm-compressor W4A16/W8A8 and AutoRound for Ada (Marlin path).
- W4A8 or INT8 for GLM experts on Hopper/Ada: no checkpoint exists, so this would have to be built.

---

## 4. Speculative decoding

**Gemma 4 31B drafts available:**

| Draft | Method | Measured |
|---|---|---|
| `google/gemma-4-31B-it-assistant` (~0.5B, 4 layers, Q‑only, shares target KV)\[53\]\[54\] | MTP | Callstack, 2×RTX PRO 6000 TP2 BF16 (vLLM 0.22): 2.48× at C1, 1.92× at C10, **1.09× at C100** with worse p99 TPOT; accepted length 3.1–3.3. AMD MI355X/MI300X vLLM blog: 2.00× GSM8K, 1.99× MBPP.\[55\] H100 c=1: 40→125 tok/s.\[56\]\[57\] |
| `RedHatAI/gemma-4-31B-it-speculator.eagle3` (+ NVFP4 draft variant) | EAGLE‑3 | Above baseline on all 4 AMD datasets.\[55\]\[58\] Thoughtworks SGLang-fork EAGLE3: 1.72× MT‑Bench.\[59\] |
| `z-lab/gemma-4-31B-it-DFlash`, `RedHatAI/gemma-4-31B-it-speculator.dflash2` | DFlash / DFlash2 | AMD: 2.34× MATH500, 2.05× HumanEval.\[55\] JarvisLabs H100 C16: baseline 375, DFlash 725, MTP 953 output tok/s. vLLM #42068: the forced TRITON_ATTN propagates to DFlash drafters and breaks them.\[60\]\[61\] |
| Red Hat DSpark | DSpark | Above baseline on AMD datasets.\[55\] |

**GLM‑5.3‑Flash:**
- It has **one native MTP layer**. vLLM uses `{"method":"mtp","num_speculative_tokens":N}` (recipes use 5; Dynamo H200 uses 7; GB300 low-latency uses 3). SGLang uses `--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`.\[7\]\[8\]\[24\]
- SGLang's "High Throughput" strategy turns speculation **off**. SGLang states that neither MTP nor DFlash2 runs with DP‑attention.\[8\]
- `incoai/GLM-5.3-Flash-DFlash2` (gated): on 2× DGX Spark it gave +44% on structured, +22% on code and ≈0 on prose vs MTP‑4.\[8\]\[62\]
- No public EAGLE‑3 draft is known. A DSpark training recipe exists (inclusionAI/ASpec PR #36: FP8 TP4 target, 5‑layer BF16 draft).\[63\]\[64\]
- Reported MTP acceptance: ~2.5 tokens general, ~6 on math (SM120 repo).\[21\] GLM‑5 paper: shared‑parameter MTP accept length 2.76 at 4 steps.\[65\]

**Concurrency rule:** speculation pays when decode is memory-bound and idle compute exists. As batch grows, verification cost approaches the cost of real tokens. Measure at C1/8/32/64/128/256 with num_spec ∈ {0,1,2,3,4} and pick per pool. OpenRelay found a third draft token gave the same capacity as two.\[1\] vLLM #64808: MTP halves prefix-cache hits when `--prefix-cache-retention-interval` is set; the workaround is `"disable_eagle_block_drop": true`.\[66\]

**Draft training:** SpecForge (used for DFlash drafts), the ASpec DSpark recipe, the Red Hat speculators format, NVIDIA ModelOpt speculative decoding. The cost of training a GLM EAGLE3/DSpark draft is **not published**. Rough planning estimate (our assumption, not sourced): generating hidden states from the 4‑GPU FP8 target dominates, so budget hundreds of GPU-hours.

---

## 5. Scheduling and batching

**vLLM:**
- The live vLLM recipe table (recipes.vllm.ai/Google/gemma-4-31B-it) lists "Max throughput | 1-2 | 256-512 | Best tok/s per GPU" (TP 1–2 with `--max-num-seqs 256–512`); balanced uses TP2 / 128; min latency uses TP 4–8 / 8–16.
- Tuned H100 Gemma recipe: `--max-num-seqs 128 --max-num-batched-tokens 16384`, chunked prefill, prefix caching.\[2\] OpenRelay MI355X used `--max-num-batched-tokens 8192`.\[1\]
- 128 concurrent sequences doubled time per token for only 1–3% more throughput on MI355X, so cap max-num-seqs at the SLO knee rather than maximizing it.\[1\]
- `--async-scheduling` is recommended by the recipe.\[3\]\[56\]
- On CUDA/ROCm the recipe now advises leaving `--gpu-memory-utilization` unset (defaults to 1.0 with warmup-based sizing) unless the GPU is shared.\[3\]

**GLM in vLLM:**
- `VLLM_ENGINE_READY_TIMEOUT_S=3600`.\[7\]
- `--disable-flashinfer-autotune` in the official serve line.\[7\]
- Dynamo H200: max seqs 32 / batched tokens 8,192 aggregated; decode pool 128 seqs / 2,048 tokens.\[24\]
- AMD TP4: `--max-num-seqs 512`. Mega MoE on 4 GPUs: `--max-num-seqs 948`.\[7\]

**SGLang GLM (B300 TP4, AgentX, C64):** `--prefill-decode-interval 1`, `--mamba-max-states-per-path 2` and `--mamba-full-memory-ratio 0.5` gave +26.3% total and +35.6% output throughput. TTFT p50 went from 1,744 to 642 ms and TPOT p99 from 163 to 66 ms. That benchmark pinned acceptance with `SGLANG_SIMULATE_ACC_LEN=3.5`, so re-measure with real acceptance.\[8\]

**Frontend overhead:** use vLLM's Rust frontend for text GLM pools (multimodal not supported there).\[7\] For Gemma, test multiple API-server processes. Measure structured-output (xgrammar) and sampling overhead explicitly; neither was benchmarked in sources found.

---

## 6. Caching and memory

**Context cap:**
- Set `--max-model-len` to the product cap (OpenRelay offers 256K) **only on a long-context pool**.
- Run the bulk pool at 16–32K. The Gemma model card discussion and the vLLM recipe both flag full-256K KV preallocation as the main concurrency killer.\[50\]\[67\]
- On one RTX PRO 6000 at 0.70 utilization, NVFP4 Gemma gets 542K KV tokens, about 2.07 concurrent 262K‑token requests.\[54\]

**Gemma KV sizing (our estimate):**
- Global layers: 10 × 4 KV heads × 512 dim × (K+V) ≈ 80 KiB/token in BF16, ≈40 KiB in FP8. Sliding layers add a window-bounded amount per sequence.
- On a 24 GB RTX 4090 with the 19.8 GB QAT checkpoint, only ~3–4 GB is left. That is roughly 40–100K total cached tokens, not 128 × 2K = 256K.
- **A single 4090 cannot host 128 concurrent Gemma conversations.** Use TP2 4090 pairs or larger GPUs.

**Prefix caching:**
- vLLM APC and SGLang RadixAttention. The hybrid KV manager handles sliding + full layers (Gemma) and KDA + MLA (GLM; `--no-disable-hybrid-kv-cache-manager` in PD recipes).\[7\]
- SGLang GLM defaults: `--mamba-full-memory-ratio 0.25` with `--mamba-radix-cache-strategy extra_buffer_lazy`. High-throughput recipes pin 0.9 because many short requests need more KDA states; use the `compute-mamba-ratio` skill.\[8\]
- AMD TP4 FP8 GLM: 14.92M-token KV pool, ~113.81× max concurrency at 128K context.\[7\]

**Offload:**
- vLLM KV offloading for GLM (PR #57145) was validated on 4×H200 with CPU + filesystem tiers.\[7\]
- LMCache 0.5.4 + vLLM validated hybrid-state recovery for GLM NVFP4 (arXiv 2609.15030).\[68\]
- **LMCache does not support Gemma 4's multi-group KV layout** (issue #3106: reshape error), so this would have to be built.\[69\]
- SGLang HiCache (`--enable-hierarchical-cache`) was used for MI300X/MI325X GLM at higher concurrency.\[8\]

**Multimodal:**
- Gemma vision budget: `--mm-processor-kwargs '{"max_soft_tokens": 560}'` (70–1120 tokens/image).\[3\]
- For text pools, use `--limit-mm-per-prompt '{"image":0,"audio":0}'` to skip encoder profiling and free KV memory.\[2\]\[3\]
- GLM: `--compilation-config '{"cudagraph_mm_encoder": true}'`. SGLang encoder disaggregation exists. H200 Mega‑MoE DP fails on image input.\[7\]

---

## 7. Parallelism

**Gemma 4 31B (dense):**
- Prefer **many small replicas** (TP1 on ≥48 GB cards with 4‑bit, TP1–2 on 80–96 GB, TP2 on 24–32 GB consumer cards) over wide TP.
- On an 8‑GPU PCIe RTX PRO 6000 host, Callstack recommends four TP2 replicas over one TP8.\[56\]
- H100 scaling (inferencebench): 855 (1×) → 1,472 (2×) → 2,355 (4×) → 3,050 tok/s (8×), i.e., severe per-GPU efficiency loss.\[70\]

**GLM‑5.3‑Flash (MoE):**
- vLLM strategies: `single_node_tp`, `single_node_tep`, `single_node_dep`, multi-node TP/TP+PP/TEP/DEP, `pd_cluster`.\[7\]
- PD routers: vllm / dynamo / smg; connectors: nixl / mooncake / moriio.\[7\]
- PD requires identical `VLLM_SSM_CONV_STATE_LAYOUT=DS` and `VLLM_KV_CACHE_LAYOUT=HND` and identical `num_speculative_tokens` on both pools.\[7\]
- DP+EP with Mega MoE is the measured throughput winner on H200.\[7\]
- On SM120: TP2+EP2 on 2×PRO 6000 gave C4/8K aggregate 965.6 vs 875.6 tok/s (TP2), but needs unmerged patches (sglang #45298, flashinfer #6824).\[71\]
- **8×RTX 5090:** TP8 means 8 heads per rank and padded MMA ("half the tensor core is wasted"), with no NVLink across NUMA islands. Avoid.\[38\]
- EPLB/redundant experts and DeepEP for GLM: **not verified**. The agent should check `--enable-eplb` support for Glm5Next before testing.

---

## 8. Routing

- **Dynamo KV router** is shipped in the GLM GB200/H200 recipes, with prefix-cache match unit 128 on H200.\[24\]
- **vllm-router** is used in the PD recipe with `--policy round_robin --vllm-pd-disaggregation`.\[7\]
- **SGLang router**, **llm‑d inference scheduler**, **production-stack router:** not verified for these models.
- Implement prefix-hash affinity with a cache-hit/queue-depth tie-break as the demo's "route repeat prompts to replicas that hold their KV".
- Route by length into a ≤32K pool and a 256K pool.
- Normalize heterogeneous GPUs by measured goodput, not FLOPs.
- Track the cached-input share per route, since it changes revenue 10×.

---

## 9. Hardware-specific tuning

| Family | Gemma 4 31B | GLM‑5.3‑Flash |
|---|---|---|
| **RTX 4090 (Ada, 24 GB)** | 4‑bit only (QAT W4A16/AWQ via Marlin); FP8 compute exists on Ada, but no FP4. KV-starved; TP2 pairs or skip.\[40\]\[72\] Verify the attention backend (#38887).\[41\] | **Not viable** (NVFP4 needs ~184 GiB; 8×24 GB = 192 GB with no KV room; no sparse-MLA path).\[29\]\[73\]\[74\]\[75\] |
| **RTX 5090 / RTX PRO 6000 (SM120)** | NVFP4 on FP4 tensor cores (CUDA 13 / cu130 images). PRO 6000 is the sweet spot (96 GB).\[44\]\[76\] | Community-only (ormandj SGLang image; Libertai vLLM patches). 2×PRO 6000 W4A16: C4 aggregate 347–966 tok/s. No high-concurrency data.\[21\]\[71\] |
| **L40S / A100** | W4A16 or FP8 (L40S); FlashInfer h512 path needed (A100 data above). | Not supported upstream (A800 port required custom FP8 Marlin MoE).\[77\] |
| **H100/H200/GH200** | BF16/FP8. H100 ~855–1,260 tok/s single GPU.\[70\] | FP8 weights, **BF16 KV only**. H200 Mega MoE DP8+EP; TP8 + MTP7 Dynamo.\[7\]\[24\] |
| **B200/B300/GB200/GB300** | NVFP4, FA4 path. | NVFP4/FP8 + FP8 KV; `latency_specialized` decode (GB300); `--numa-bind` on Grace (saves ~2.4 ms/step at c=1).\[7\] |
| **MI300X/MI325X** | BF16/FP8 (recipe lists 1× MI300X).\[12\] | SGLang TP4·EP4 InferenceX-measured recipe; CK block-FP8 GEMM.\[8\] |
| **MI355X** | MXFP4 + AITER (+ tuned tables). OpenRelay measured 3.0 req/s/GPU at p95 TTFT <4 s.\[1\] | vLLM TP4 AITER sparse MLA + MTP (#55239); SGLang MXFP4.\[7\]\[8\] |

**Not verified in this pass, so the agent should measure:** power limits/clocks (perf per watt on 4090/5090), NCCL P2P settings on PCIe consumer boxes (`NCCL_P2P_DISABLE` behavior), NUMA pinning on x86, TunableOp/hipBLASLt.

---

## 10. Benchmark methodology

- **Harness:** copy OpenRelay's: open-loop arrivals, 300 s per rate, two seeds, report the worse seed, SLO p95 TTFT <4 s, and a GSM8K gate before any number counts.\[1\]
- **Throughput–interactivity curve:** plot InferenceX-style curves of tok/s/GPU vs tok/s/user.\[78\]\[79\]
- **Tools:** `vllm bench serve` (it now warns on warm prefix cache for random datasets),\[80\] SGLang `bench_serving`, AIPerf/genai-perf, GuideLLM.
- **Workloads:**
  1. Chat: ShareGPT-like.
  2. Agentic: 9.5K in / 380 out, multi-turn with growing shared prefixes (AgentX-like).
  3. Long-context: 64K/128K/256K.
  4. Batch: offline max throughput for the $0.035/$0.15 tier.
- **Cost line:** revenue/GPU‑hr from the three-price mix vs rent; break-even achieved at SLO (goodput), not peak.
- **Correctness:** GSM8K ≥ baseline −0.5 pt, greedy token-agreement vs reference, tool-call parse rate, long-context needle.

---

## 11. Published baselines and competitor prices

**Gemma 4 31B throughput:**

| Source | Setup | Result |
|---|---|---|
| OpenRelay | MI355X, MXFP4 + MTP, vLLM 0.30 ROCm | 2.0 → 3.0 req/s/GPU at p95 TTFT <4 s; TPOT 40–43 → 22 ms; prefill 27.8K → 41.9K tok/s\[1\] |
| Callstack | 2×PRO 6000 TP2 BF16 C100 | 1,396 output tok/s, or 1,524 with MTP\[56\] |
| inferencebench | 1×H100 BF16 | 855 tok/s (sweep), 1,260 tok/s at C128\[70\] |
| hexgridcloud | 1×PRO 6000 FP8 | 548 output / 1,168 total tok/s at C≤21\[81\] |
| AMD vLLM blog | MI355X | MATH500 baseline 1,365 tok/s → 3,006 with MTP N=4\[55\] |

**GLM‑5.3‑Flash throughput:**
- SGLang B300 TP4 C64: 1,642.5 output tok/s across 4 GPUs (≈411 per GPU), 227K total tok/s including cached input.\[8\]
- GB300 ITL table (see §2).
- SM120 community numbers at C≤4.
- Artificial Analysis's GLM‑5.3‑Flash page lists an output speed of 59 tok/s per user and 86.50 s time to first answer token at the default "max" effort, on $0.15/$0.50 first-party pricing; secondary sources had cited ~49 tok/s.

**Competitor prices:**
- *Gemma 4 31B (OpenRouter):* the OpenRouter model page lists $0.10 input / $0.34 output per 1M across 16 providers; the cheapest listed provider, KT, charges $0.100 / $0.340 / $0.100 cache read, and NextBit charges $0.125 / $0.340 / $0.075 cached. OpenRelay's $0.07/$0.30 undercuts both.
- *GLM‑5.3‑Flash list:* $0.15/$0.50 (Z.ai launch discount $0.075/$0.25).\[82\]
- *GLM‑5.3‑Flash on OpenRouter:* low providers Relace $0.0399/$0.0784 and Wafer $0.042/$0.184; batch Together $0.023/$0.068.\[83\]\[84\] OpenRelay's $0.045/$0.18 sync is mid-pack, and someone is pricing below plausible cost.
- Aggregators conflict and refresh daily; re-check on the day of the demo.

---

## 12. Build-our-own list

| Item | Why / payoff | How |
|---|---|---|
| Gemma h512 attention routing on Ada/Hopper (cherry-pick #47547 + FlashInfer ≥0.6.16) | 1.1–2.1× decode at long context; MTP flips from loss to gain | Patch config.py, verify backend logs, run needle tests |
| AITER tile/GEMM tables for Gemma on MI300X (OpenRelay did MI355X) | Up to 2.6× attention prefill\[1\] | Reuse the OpenRelay recipe repo; profile; add table entries |
| LMCache multi-group KV for Gemma | Cross-replica prefix reuse | Implement per-group chunking (#3106) |
| EAGLE‑3/DSpark draft for GLM | MTP is limited to 1 layer; DFlash2 is gated | ASpec PR #36 recipe / SpecForge |
| Calibrated FP8 KV scales (Gemma, GLM) | Safer FP8 KV, enables FP4 KV | ModelOpt/llm-compressor KV calibration |
| NVFP4 attention-inclusive Gemma, MXFP4‑all‑linear Gemma | Smaller weights, more KV | ModelOpt / AMD Quark + GSM8K gate |
| GLM W4A16/W4A8 experts for Hopper | Fits 4×H100 with KV headroom | llm-compressor; Marlin/Machete MoE |
| SM120 GLM upstreaming (sparse MLA, FP8 KV, EP2) | Opens PRO 6000 nodes | Port Libertai/ormandj patches |
| Concurrency-adaptive spec length | Avoids the C100 regression | Disable/shrink num_spec above a batch threshold |
| KV-/prefix-aware router + length pools | Directly the job's "route repeat prompts" project | Dynamo KV router or a custom prefix-hash router |

---

## Prioritized test order (limited GPU budget)

**Gemma 4 31B (days 1–3):**
1. **1×H100 or 1×RTX PRO 6000, vLLM ≥0.30.** Baseline BF16/FP8 at 16K/32K max-len, text-only, with the OpenRelay-style harness. Log the attention backend.
2. **Fix routing:** FlashInfer/FA4 for global layers; re-measure 8K/32K/64K prompts.
3. **Quantization sweep:** FP8‑dynamic vs NVFP4 (Blackwell) vs QAT W4A16, plus FP8 KV. Apply the GSM8K + agreement gate.
4. **Scheduler sweep:** max-num-seqs {64,128,256}, max-num-batched-tokens {8K,16K}, CUDA-graph sizes to 256+, async scheduling.
5. **MTP sweep:** num_spec {0,1,2,4} × concurrency {1,8,32,64,128}; also a DFlash/EAGLE3 spot check.
6. **Replica topology:** TP1 ×N vs TP2 on the same box. Run 2×RTX 4090 TP2 W4A16 only if OpenRelay's 4090 fleet is the demo target.
7. **MI355X:** reproduce OpenRelay's recipe and confirm the 3.0 req/s point. Optional.
8. **Router demo:** prefix-affinity routing across 2+ replicas with a multi-turn workload; report cache-hit and revenue mix.

**GLM‑5.3‑Flash (days 4–7):**
1. **4×B200 (or 4×H200 with BF16 KV), vLLM nightly recipe** FP8 + MTP5, max-len 32K vs 256K pools.
2. **NVFP4 (nvidia vs RedHat) on Blackwell** with the quality gate.
3. **SGLang cookbook A/B:** Low-Latency (MTP 5/1/6) vs High-Throughput (spec off); tune `mamba-full-memory-ratio` / `max-states-per-path` / `prefill-decode-interval`.
4. **MTP depth sweep** {0,1,3,5,7} vs concurrency.
5. **H200 Mega MoE DP+EP** (text-only) if Hopper is the economic choice.
6. **PD disaggregation (NIXL)** 1P1D on 8 GPUs vs 2× aggregated TP4. Keep PD only if goodput at SLO improves.
7. **TokenSpeed A/B;** MI355X vLLM/SGLang MXFP4 if available.

## Caveats

- Many "latest" facts are days old and some sources conflict. The vLLM GLM recipe says ≥0.29.0 is unreleased while release trackers date v0.29.0 to Sep 9, 2026;\[13\]\[85\] trust the live recipe page and test images, not version numbers.
- Several key numbers come from vendors or self-promoting third parties (Callstack, LilaRest), or used simulated acceptance (SGLang B300). Reproduce before quoting them in the demo.
- No source gave 32–256 concurrency numbers for Gemma on RTX 4090/5090 or for GLM on any consumer GPU. Those are open measurements and a good original contribution for the application.

## Sources

1. [Gemma 4 Optimization with Agentic Kernel Engineering](https://openrelay.inc/blog/gemma-4-agentic-kernel-engineering)
2. [neural-nova/gemma-4-31B-it-optimized · Hugging Face](https://huggingface.co/neural-nova/gemma-4-31B-it-optimized)
3. [gemma-4-31B-it on vLLM — Serve command for H100, MI300X, MI325X, MI355X, TPU v6e (Trillium), TPU7x, Arc Pro B60, Arc Pro B70](https://recipes.vllm.ai/Google/gemma-4-31B-it)
4. [Gemma 4 e4b latency optimisations - Models - Hugging Face Forums](https://discuss.huggingface.co/t/gemma-4-e4b-latency-optimisations/175910)
5. [Google’s Gemma 4 is Weirder than you Realize](https://machine-learning-made-simple.medium.com/googles-gemma-4-is-weirder-than-you-realize-17d00d95b0d5)
6. [GLM-5.3 Local Deployment: Requirements, Size, Flash](https://vdf.ai/blog/glm-5-3-local-deployment/)
7. <https://recipes.vllm.ai/zai-org/GLM-5.3-Flash>
8. [GLM-5.3-Flash - SGLang Documentation](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash)
9. [zai-org/GLM-5.3-Flash · Hugging Face](https://huggingface.co/zai-org/GLM-5.3-Flash)
10. [Can You Run GLM-5.3-Flash Locally? The Hardware Reality](https://shop.zimaspace.com/blogs/tech-ai-hub/glm-5-3-flash-locally-hardware-ram-vram-requirements)
11. [How to use Gemma 4 with the new MTP drafters? - General - vLLM Forums](https://discuss.vllm.ai/t/how-to-use-gemma-4-with-the-new-mtp-drafters/2619)
12. [recipes/Google/Gemma4.md at main · vllm-project/recipes · GitHub](https://github.com/vllm-project/recipes/blob/main/Google/Gemma4.md)
13. [vLLM changelog — v0.29.0, Sep 9, 2026 · What's New](https://whatsnew.fyi/product/vllm)
14. [vLLM v0.29.0 — Model Runner V2 becomes the…](https://ai-tldr.dev/releases/vllm-v0-29-0/)
15. [VLLM](https://en.wikipedia.org/wiki/VLLM)
16. [thoughtworks/Gemma-4-31B-Eagle3 · Hugging Face](https://huggingface.co/thoughtworks/Gemma-4-31B-Eagle3)
17. [\[Fix\] gemma4: Make DFlash drafts work on a Gemma-4 target by Noridom1 · Pull Request #52081 · sgl-project/sglang · GitHub](https://github.com/sgl-project/sglang/pull/52081)
18. [Improving inference · Issue #444 · OpenAgriNet/amul-oan-api · GitHub](https://github.com/OpenAgriNet/amul-oan-api/issues/444)
19. [SGLang on X: "GLM-5.3-Flash (320B-A18B) from @Zai\_org drops today with day-0 support in SGLang. You may know it as ox-alpha from the past few days! 👀 It's the first native multimodal model in the GLM-5 series, able to review its own output visually and fix what's wrong. 🚀 It outperforms GLM-5.2… / X](https://x.com/sgl_project/status/2092616826853503263)
20. [\[AMD\] Enable GLM-5.3-Flash on gfx942 and gfx950 by andyluo7 · Pull Request #36607 · sgl-project/sglang · GitHub](https://github.com/sgl-project/sglang/pull/36607)
21. [GitHub - ormandj/sglang-glm53-flash-sm120: SGLang for GLM-5.3-Flash on two or four RTX PRO 6000 Blackwell GPUs (SM120), with TP2 W4A16 and TP4 NVIDIA NVFP4 profiles. · GitHub](https://github.com/ormandj/sglang-glm53-flash-sm120)
22. [Get Started with GLM-5.3-Flash — NVIDIA NIM for LLM and VLM](https://docs.nvidia.com/nim/large-language-models/2.1.4/get-started/advanced/get-started-glm-5.3-flash.html)
23. [\[None\]\[feat\] Support NVFP4 GLM-5.3-Flash checkpoints by ruocheng-nv · Pull Request #20127 · NVIDIA/TensorRT-LLM · GitHub](https://github.com/NVIDIA/TensorRT-LLM/pull/20127)
24. [GLM-5.3-Flash | NVIDIA Dynamo Documentation](https://docs.nvidia.com/dynamo/v1.5.1/recipes/glm-5-3-flash)
25. [Model recipes](https://lightseek.org/tokenspeed/recipes/models)
26. [TokenSpeed LLM Inference Engine Review: Architecture, Setup, and Limitations](https://hysenlabs.com/projects/lightseekorg-tokenspeed)
27. [feat: add Gemma4 31B support for standalone and vLLM plugin mode on MI355X by ClementLinCF · Pull Request #546 · ROCm/ATOM · GitHub](https://github.com/ROCm/ATOM/pull/546)
28. [Serve Kimi-K2.5-MXFP4 on MI355X with ATOM - ROCm™ Blogs](https://rocm.blogs.amd.com/software-tools-optimization/kimi-k25-mxfp4-atom/README.html)
29. [Run GLM-5.3-Flash Locally: RAM, VRAM & Setup Guide - CometAPI](https://www.cometapi.com/how-to-run-glm-5-3-flash-locally/)
30. [Gemma 4 MTP drafter (gemma4-assistant) is indexed but never offered in the speculative decoding draft model dropdown on macOS · Issue #2044 · lmstudio-ai/lmstudio-bug-tracker · GitHub](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2044)
31. [How to Run GLM-5.3-Flash Locally: Hardware & Setup](https://codersera.com/blog/how-to-run-glm-5-3-flash-locally-2026/)
32. [How to Run GLM-5.3-Flash Locally - Atomic Chat](https://atomic.chat/blog/guides/how-to-run-glm-5-3-flash-locally)
33. [GLM-5.3 benchmarks: tokens per second by GPU](https://llamaperf.com/model/glm-5-3)
34. [\[Bug\]: Gemma 4 E4B extremely slow on v0.19.0 forced TRITON\_ATTN fallback yields \~9 tok/s on RTX 4090 (vs \~100+ tok/s for comparable Llama 3B) · Issue #38887 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/issues/38887)
35. [coolthor/Gemma-4-E4B-it-NVFP4A16 · Hugging Face](https://huggingface.co/coolthor/Gemma-4-E4B-it-NVFP4A16)
36. [\[Gemma4\] Allow per-layer attention backend selection for heterogeneou… by CunXin1 · Pull Request #38891 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/pull/38891)
37. [\[Model\] Allow Gemma4 to use FlashInfer when FA4 is unavailable](https://github.com/vllm-project/vllm/pull/47547)
38. [GitHub - Libertai/glm53-flash-vllm-gb10: LibertAI Labs: running GLM-5.3-Flash-NVFP4 under vLLM on 2x GB10 (sm\_121). Hand-written sparse-MLA CUDA kernel for NoPE MLA, plus a fix for vLLM's uninitialised NVFP4 MoE activation scale.](https://github.com/Libertai/glm53-flash-vllm-gb10)
39. [Gemma 4 on vLLM vs Ollama: Benchmarks on a 96 GB Blackwell GPU](https://allenkuo.medium.com/gemma-4-on-vllm-vs-ollama-benchmarks-on-a-96-gb-blackwell-gpu-804ca4845a21)
40. [Deploy Gemma 4 31B QAT on a Single RTX 4090 with vLLM](https://markaicode.com/howto/how-to-optimize-gemma-4-inference/)
41. [Ollama vs vLLM Gemma 4: RTX 4090 Benchmark Guide 2026](https://markaicode.com/benchmarks/cuda-gemma-benchmark/)
42. [Gemma 4 Usage Guide - vLLM Recipes](https://translate.google.com/translate?u=https%3A%2F%2Fdocs.vllm.ai%2Fprojects%2Frecipes%2Fen%2Flatest%2FGoogle%2FGemma4.html&hl=pt&sl=en&tl=pt&client=srp)
43. [Gemma4 draft models are now available - DGX Spark / GB10 Projects - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/gemma4-draft-models-are-now-available/369114)
44. [LilaRest/gemma-4-31B-it-NVFP4-turbo · Hugging Face](https://huggingface.co/LilaRest/gemma-4-31B-it-NVFP4-turbo)
45. [nvidia/GLM-5.3-Flash-NVFP4 · Hugging Face](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
46. [RedHatAI/GLM-5.3-Flash-NVFP4 · Hugging Face](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)
47. [recipes/models/zai-org/GLM-5.3-Flash.yaml at main · vllm-project/recipes · GitHub](https://github.com/vllm-project/recipes/blob/main/models/zai-org/GLM-5.3-Flash.yaml)
48. [GitHub - himorishige/glm53-flash-2x-dgx-spark-recipe: GLM-5.3-Flash (320B-A18B, NVFP4) on 2x NVIDIA DGX Spark with vLLM TP=2: config-as-code, measured numbers, pitfalls · GitHub](https://github.com/himorishige/glm53-flash-2x-dgx-spark-recipe)
49. [nvidia/GLM-5.3-Flash-FP4-Local · Hugging Face](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local)
50. [Gemma 4 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html)
51. [dsa: run GLM-5.3 NoPE fp8 KV through FlashInfer sparse MLA on SM120 by Thireus · Pull Request #16 · Thireus/sglang · GitHub](https://github.com/Thireus/sglang/pull/16)
52. [LibertAIDAI/GLM-5.3-Flash-NVFP4 · Hugging Face](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
53. [The Local Model That Doesn't Sleep: Gemma 4 + MTP as a Marathon Engine - DEV Community](https://dev.to/gde/the-local-model-that-doesnt-sleep-gemma-4-mtp-as-a-marathon-engine-4c9)
54. [GitHub - MiaAI-Lab/Gemma-4-31B-IT-NVFP4-Recipe: Serve nvidia/Gemma-4-31B-IT-NVFP4 via vLLM with MTP speculative decoding, tool calling, and thinking/reasoning support · GitHub](https://github.com/MiaAI-Lab/Gemma-4-31B-IT-NVFP4-Recipe)
55. [Exploring Speculative Decoding in vLLM on AMD GPUs](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus)
56. [Optimizing Self-Hosted Gemma for Production Inference](https://www.callstack.com/blog/optimizing-self-hosted-gemma-for-production-inference)
57. [Gemma 4 31B H100 Benchmark: vLLM Throughput & Cost](https://markaicode.com/benchmarks/hugging-face-gemma-4-h100-latency-benchmark/)
58. [AEON-7/gemma-4-31B-it-speculator.eagle3-NVFP4 · Hugging Face](https://huggingface.co/AEON-7/gemma-4-31B-it-speculator.eagle3-NVFP4)
59. [Google Released Gemma-4 Four Days Ago. We Already Made It 1.72× Faster.](https://huggingface.co/blog/lujangusface/tw-eagle3-gemma4)
60. [Gemma 4 + DFlash incompatible: MTP-specific backend propagation forces TRITON\_ATTN on independent (DFlash) drafters · Issue #42068 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/issues/42068)
61. [Benchmarking Gemma 4 MTP vs DFlash on a Single H100](https://jarvislabs.ai/blog/gemma-4-mtp-vs-dflash-benchmark)
62. [GitHub - Deep-AI-Evo/GLM-5.3-Flash-DSpark-2x-DGX-Spark: GLM-5.3-Flash-DSpark-2x-DGX-Spark · GitHub](https://github.com/Deep-AI-Evo/GLM-5.3-Flash-DSpark-2x-DGX-Spark)
63. [feat(dspark): support GLM-5.3-Flash and selectable losses by zhouaihui · Pull Request #36 · inclusionAI/ASpec · GitHub](https://github.com/inclusionAI/ASpec/pull/36)
64. [Wayfinder map: DSpark-style speculative decoding for GLM-5.3-Flash to serve real-time phone agents · Issue #1 · PhillipChaffee/GLM-5 · GitHub](https://github.com/PhillipChaffee/GLM-5/issues/1)
65. [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/pdf/2602.15763)
66. [\[Bug\]: Gemma-4 MTP halves prefix-cache hits with \`--prefix-cache-retention-interval\` · Issue #64808 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/issues/64808)
67. [google/gemma-4-31B-it · This model is so good, but the KV cache ruins it.](https://huggingface.co/google/gemma-4-31B-it/discussions/100)
68. [Validating Hybrid-State Cache Recovery for GLM-5.3-Flash with vLLM and LMCache](https://arxiv.org/pdf/2609.15030)
69. [\[Feature\] Support multi-group KV layouts (hybrid sliding + full attention, e.g. Gemma-4) · Issue #3106 · LMCache/LMCache · GitHub](https://github.com/LMCache/LMCache/issues/3106)
70. [Gemma 4 31B on H100: The Complete Inference Benchmark](https://inferencebench.io/blog/gemma-4-31b-h100-complete-inference-benchmark/)
71. [GitHub - adubkov/glm53-flash-rtx-pro-6000-tp2-ep2: GLM-5.3-Flash (W4A16 NVFP4-K32) on 2× RTX PRO 6000 Blackwell 96 GB (SM120) over PCIe: TP2 + EP2 — 288 routed experts split 144/rank.](https://github.com/adubkov/glm53-flash-rtx-pro-6000-tp2-ep2)
72. [Gemma 4 RTX 4090 Benchmark: Tokens/Sec & VRAM (2026)](https://vantaige.io/blog/gemma-4-rtx-4090-tokens-per-sec-vram-benchmark)
73. [GLM-5.3-Flash: GGUF size, RAM & VRAM requirements · localmodel.run](https://localmodel.run/model/glm-5.3-flash)
74. [GLM-5.3-Flash GPU Requirements: Serving a 320B MoE Model](https://neevcloud.com/blogs/glm-5-3-flash-gpu-requirements-serving-a-320b-moe-model)
75. [GLM-5.3 Hardware Requirements: 433 GB VRAM, Flash Needs 185 GB](https://willitrunai.com/blog/glm-5-3-hardware-requirements)
76. [README.md · LilaRest/gemma-4-31B-it-NVFP4-turbo at main](https://huggingface.co/LilaRest/gemma-4-31B-it-NVFP4-turbo/blob/main/README.md)
77. [Automatic Model-Hardware Co-Adaptation for Heterogeneous AI Accelerators](https://arxiv.org/pdf/2607.12875)
78. [InferenceMAX: Open Source Inference Benchmarking](https://inferencex.semianalysis.com/blog/inferencemax-open-source-inference-benchmarking)
79. [Ultra-High Interactivity on NVIDIA GPUs? TileRT on InferenceX](https://inferencex.semianalysis.com/blog/ultra-high-interactivity-on-nvidia)
80. [Release v0.29.0 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)
81. [Gemma-4 31B + vLLM on RTX 6000 PRO : A Real-Load Benchmark](https://huggingface.co/blog/hexgridcloud/gemma-4-31b-vllm-on-rtx-6000-pro)
82. [GLM 5.3 Flash on OpenRouter: Model ID, Provider Prices & API Setup - GLM 5](https://glm5.app/blog/glm-5-3-flash-openrouter)
83. [GLM 5.3 Flash - API Pricing & Benchmarks](https://openrouter.ai/z-ai/glm-5.3-flash-20260826)
84. [GLM 5.3 Flash (batch) - API Pricing & Benchmarks](https://openrouter.ai/z-ai/glm-5.3-flash:batch)
85. [vLLM 0.29.0](https://freedom.tech/posts/2026-09-09-vllm-0-29-0/)
