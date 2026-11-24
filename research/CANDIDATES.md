# Which model: Gemma 4 31B (picked) vs GLM-5.3-Flash

**The bar is OpenRelay's own published price** (Maks, 11-21 evening: "cheaper than openrelay"). OpenRouter's
cheapest provider is kept as the second reference. OpenRelay lists BOTH candidates, so either could be used.

**Verdict: Gemma 4 31B-it.** OpenRelay charges $0.07 in / $0.30 out per 1M tokens for it, with $0.007 for
cached input and a Batch tier at half that. It fits on one GPU of every type Daytona rents, and every quantization
format we want to test already exists for it. To beat OpenRelay's sync price on 1k-in/1k-out traffic, a $3.67/h H100
box has to sustain 2.69 req/s (~2,750 output tok/s). That is hard but plausible for a tuned FP8/FP4 dense 31B.
GLM-5.3-Flash is 321B total / 18B active and needs at least 229 GB of VRAM. OpenRelay serves it "on our own GPUs" at
$0.045 / $0.18. Matching that would take about 8k output tok/s on one B300 (0 spot B300 free when checked), or about
10k on 2x MI355X: one of OpenRelay's own specialities (they run MoE models on MI355X).

## The bar: OpenRelay's published prices, USD per 1M tokens

Source: https://openrelay.inc/inference-pricing, saved as `raw/openrelay/inference-pricing_20261122T032303Z.html`
(fetched **2026-11-22 03:23 UTC = 9:23 PM Central Sat 11-21**), parsed by `openrelay_prices.py` into
`openrelay_prices_20261122T032303Z.json`. The same Gemma and GLM rates appear on /ai-inference-platform and
/inference-catalog, saved in the same folder at the same time. (A web-search summary quoted other figures,
$0.14/$0.40 and $0.99/$1.49. The live pages do not say that, and only the saved pages are used.)

| model (OpenRelay id) | tier | input | cached input | output |
|---|---|---|---|---|
| Gemma 4 31B (`openrelay/gemma-4-31b`, 256K) | Sync | **$0.07** | $0.007 | **$0.30** |
| | Batch (24 h, JSONL) | $0.035 | $0.0035 | $0.15 |
| GLM 5.3 Flash (`openrelay/glm-5.3-flash`, 256K) | Sync | $0.045 | $0.009 | $0.18 |
| | Async within 1 h | $0.036 | $0.0072 | $0.144 |
| | Async within 24 h | $0.0225 | $0.0045 | $0.09 |

OpenRelay also rents GPUs (https://openrelay.inc/gpus, saved `raw/openrelay/gpus_20261122T032357Z.html`):
H100 SXM $2.60/h, B200 $6.30/h, B300 $8.49/h, MI355X $6.30/h, A100 80GB $1.10/h.

Per-request bar for the workload shapes we test (Sync / Batch / OpenRouter cheapest, Modal excluded):

| tokens in / out | OpenRelay Sync | OpenRelay Batch | OpenRouter cheapest |
|---|---|---|---|
| 1024 / 1024 | $0.000379 | $0.000189 | $0.000451 |
| 1024 / 256 | $0.000148 | $0.000074 | $0.000189 |
| 8192 / 1024 | $0.000881 | $0.000440 | $0.001167 |

## Side by side

| | Gemma 4 31B-it | GLM-5.3-Flash |
|---|---|---|
| OpenRelay, $/1M in / out | **$0.07 / $0.30** sync; $0.035 / $0.15 batch | $0.045 / $0.18 sync; $0.0225 / $0.09 async 24 h |
| OpenRouter cheapest, $/1M in / out | **$0.10 / $0.34** (KT, CoreWeave fp4); fp8 floor $0.16 / $0.32 (DeepInfra); bf16 floor $0.14 / $0.40 (Crusoe, Novita) | **$0.0399 / $0.0784** (Relace fp4); Z.AI first-party $0.15 / $0.50; batch tier $0.023 / $0.068 |
| OpenRouter providers listed | 18 | 41 |
| Architecture | dense, 31.27B params; 60 layers (50 sliding-window 1024 + 10 global); 256K ctx | MoE 321B total / 18B active, 288 experts top-8; hybrid KDA linear attention (34 layers) + sparse MLA (11); 1M ctx |
| Weights (official) | BF16, 62.6 GB | FP8, 328.4 GB (BF16 variant 772 GB of VRAM) |
| Min VRAM (vLLM recipe) | 75 GB BF16 / 38 GB FP8 / 19 GB NVFP4 | 386 GB FP8 / 229 GB NVFP4 |
| Smallest Daytona fit | 1× L40S (FP8/INT4), 1× RTX PRO 6000 (any), 1× H100 (FP8) | 1× B300 NVFP4 (tight; 0 spot B300 free at 03:05Z), else 2× MI355X / 4× H100 FP8 |
| License | Apache-2.0 (HF card, `license_link` ai.google.dev/gemma/docs/gemma_4_license) | MIT |
| HF repo, pinned rev | `google/gemma-4-31B-it` @ `842da379…` (not gated) | `zai-org/GLM-5.3-Flash` @ `46a050be…` |

## Quantizations that exist for Gemma 4 31B (HF API, 2026-11-22 03:0x UTC)

| format | repo @ rev | size | runs on |
|---|---|---|---|
| BF16 (reference) | google/gemma-4-31B-it @ 842da379 | 62.6 GB | any 80 GB+ GPU |
| FP8 dynamic (W8A8) | RedHatAI/gemma-4-31B-it-FP8-dynamic @ d4ab4f57 | 33.3 GB | Hopper, Blackwell, Ada (L40S) |
| FP8 block | RedHatAI/gemma-4-31B-it-FP8-block @ d7242548 | 33.3 GB | Hopper, Blackwell |
| NVFP4 (ModelOpt) | nvidia/Gemma-4-31B-IT-NVFP4 @ 4135a98a | 32.7 GB (half the linear weights kept BF16) | Blackwell only |
| NVFP4 (compressed-tensors) | RedHatAI/gemma-4-31B-it-NVFP4 @ 5df75cca | 23.3 GB | Blackwell only |
| W4A16 QAT (Google) | google/gemma-4-31B-it-qat-w4a16-ct @ 52f3f65b | 23.3 GB | any GPU |
| AWQ 4-bit | cyankiwi/gemma-4-31B-it-AWQ-4bit @ a6b1af08 | 20.9 GB | any GPU |
| MXFP4 (AMD Quark) | amd/gemma-4-31B-it-MXFP4 | — | MI355X |
| GGUF Q2…Q8, BF16 | unsloth/gemma-4-31B-it-GGUF @ c1ac76e9 (30 files) | 465 GB total | llama.cpp |
| MTP drafter (spec. decoding) | google/gemma-4-31B-it-assistant @ 627c5ec1 | 0.47B, 1.0 GB | vLLM, llama.cpp |
| EAGLE3 drafter | RedHatAI/gemma-4-31B-it-speculator.eagle3 @ 28a1c8b4 | 2.2B, 4.5 GB | vLLM |

GLM-5.3-Flash has the same breadth: NVFP4 from nvidia, RedHatAI and RadixArk; MXFP4 from amd and Intel; many GGUFs; EXL3.
Format breadth is not what separates the two models.

## Engine support (source trees read 2026-11-22 03:1x UTC)

| engine (pinned version) | Gemma 4 | GLM-5.3-Flash |
|---|---|---|
| vLLM v0.33.0 (image pushed 2026-11-02) | `gemma4.py`, `gemma4_mm.py`, `gemma4_mtp.py`, `gemma4_dspark.py`; reasoning parser `gemma4`; official recipe verified on H100 / MI300X / MI325X / MI355X | `vllm/models/glm5next` (registry: `Glm5NextForConditionalGeneration`, MTP); recipe verified on H100 / B200 / GB200 / MI355X; min vLLM 0.29.0 |
| SGLang v0.5.23 (2026-10-30) | `gemma4_causal.py`, `gemma4_mtp.py` | `glm5_next.py`, `glm5_next_nextn.py` |
| llama.cpp (daily builds) | `gemma4.cpp`, `gemma4-assistant.cpp` (MTP drafter) | `glm5-next.cpp` |
| TensorRT-LLM v1.3.0 / 1.4.0rc | `modeling_gemma4.py` (PyTorch backend) | `modeling_glm5_next.py` |

## Why not GLM-5.3-Flash

0. **Against OpenRelay itself:** GLM-5.3-Flash at $0.045/$0.18 is $0.000230 per 1k/1k request. A 1x B300
   spot box (~$6.6/h with host) needs ~8 req/s to match; 2x MI355X (~$8.7/h) needs ~10.5 req/s. OpenRelay runs it on
   its own fleet, which is where it is strongest.
1. **The OpenRouter floor is very low.** The 1k-in / 1k-out request costs $0.000121 at the cheapest provider. A 1× B300 NVFP4
   box costs about $5.17/h for the GPU plus host. To match the bar it would have to sustain about 12 requests/s of
   1k/1k on one GPU: roughly 12k output tok/s from an 18B-active MoE, alongside 12k tok/s of prefill. That is at
   the edge of what the hardware can do, and the 0.0784 price is most likely subsidised.
2. **Spot capacity.** At 03:05Z the GPU-capacity API showed 0 spot B300 and no B200 at all. The FP8 model would need
   2× MI355X or 4× H100. That makes every experiment 2–4× the price and turns a single-GPU sweep into a multi-GPU one.
3. **Cold starts.** Daytona GPU sandboxes are ephemeral, so each run downloads the weights again. That is 328 GB for
   GLM against 23–63 GB for Gemma, and every one of our experiments pays it.
4. **Breadth.** Gemma lets one budget cover 6+ GPU types, 6 quantizations and 4 engines. GLM would cover 1–2 of each.

## OpenRouter reference for Gemma 4 31B (second column) (from `openrouter_bar_google_gemma-4-31b-it_20261122T030326Z.md`)

| workload, tokens in/out per request | cheapest provider $/request | fp8 floor | bf16 floor |
|---|---|---|---|
| 1024 / 1024 | $0.000451 (KT; CoreWeave fp4 same) | $0.000492 | $0.000553 |
| 1024 / 256 | $0.000189 | $0.000246 | $0.000246 |
| 8192 / 1024 | $0.001167 | $0.001638 | $0.001556 |
| 1024 / 8192 | $0.001167 (Modal — lists $0.34 in / **$0.10 out**, probably swapped; next is DeepInfra fp8 $0.002785) | $0.002785 | $0.003420 |

Break-even throughput on a 1× H100 spot box (8 vCPU / 48 GiB / 150 GB = $3.67/h, Daytona pricing page 03:05Z) for
1024/1024 traffic: 3.67 / 3600 / 0.000451 = **2.26 req/s**, about 2,300 output tok/s plus 2,300 input tok/s, sustained.

Break-even against **OpenRelay Sync** on the same box: 3.67 / 3600 / 0.000379 = **2.69 req/s** (about 2,750 output
tok/s). Against OpenRelay Batch: 5.38 req/s.
