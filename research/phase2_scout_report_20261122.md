<!-- Phase-2 scouting report written by a Grok agent (grok-4.7) on 2026-11-22 ~07:10Z for the build session.
     Verified by the build session against primary sources saved in research/raw/phase2_*: Inkling-2-Mini upload commit
     2026-11-20T22:02:14Z, config (24 layers, 24/8 heads, head_dim 128, top-8 experts, loop 6-18 x3, NVFP4), OpenRouter
     prices (PrimeIntellect 0.075/0.30, Io Net 0.08/0.35), OpenRelay catalog (7 models, none new), recipe pages HTTP 200.
     Other facts are Grok's, with its sources inline. -->
# Phase 2 scout: open-weight LLMs, weights since 2026-11-19

**Window:** Sunday 2026-11-22 (UTC). Included only if the weights themselves became available on or after **2026-11-19 00:00 UTC** (about 72 hours), even if the model was announced earlier. A repo `createdAt` before that date does not count when the weight-upload commit is inside the window. The reverse also holds: a Nov 21 write-up of a Nov 17 upload is a near-miss.

**Checked this session:** Hugging Face model API and commit history; OpenRouter `GET https://openrouter.ai/api/v1/models` (476 models) plus `/endpoints`; OpenRelay `https://openrelay.inc/inference-pricing` (full rate table). Org scan: the 15 most recently modified repos for tencent, thinkingmachines, OVHaiLLM, Metric-AI, baselabs, nvidia, Qwen, deepseek-ai, moonshotai, MiniMaxAI, inclusionAI, microsoft, allenai, ibm-granite, LiquidAI, Cloudflare, sensenova, RadixArk, Agnes-AI, sarvamai, open-athena. Qwen, DeepSeek, Mistral, Google/Gemma, Zhipu, Moonshot, MiniMax, NVIDIA, IBM Granite, AllenAI/OLMo, Cohere, OpenAI, Baidu/ERNIE, StepFun, Xiaomi/MiMo had **no new generative-LLM repo** in that slice.

**One-GPU test used below:** one AMD MI355X at **288 GB** HBM, or one NVIDIA B300 at **~275 GB**, as given in the task. Those capacities were not re-measured. “Fits” means the weight bytes leave tens of GB for KV and runtime, not that a 1M-token batch was benchmarked on that chip.

**OpenRelay price bar:** none of the in-window models appear on the OpenRelay rate table fetched this session. Listed sync models there are Gemma 4 31B ($0.07 / $0.30), GLM 5.3 ($0.079 / $3.00), GLM 5.3 Flash ($0.045 / $0.18), DeepSeek V4.1-Flash ($0.035 / $0.33), Kimi K3.1 Pro ($0.24 / $7.00), DeepSeek-OCR 2 ($0.039 / $0.039), and BGE-M3 ($0.013 input). Source: https://openrelay.inc/inference-pricing

---

## 1. Inkling 2 Mini — Thinking Machines Lab

The only in-window model that is a general LLM, fits one of the target GPUs, and already has a vLLM recipe and an SGLang cookbook (including an MI355X image).

| Field | Value | Source |
| --- | --- | --- |
| Name | Inkling 2 Mini (also called Loopling in secondary catalogues) | https://huggingface.co/thinkingmachines/Inkling-2-Mini |
| HF repo | `thinkingmachines/Inkling-2-Mini` | same |
| Weights appeared | Commit `upload model` at **2026-11-20 22:02:14 UTC**. Card: “Date of release: November 20, 2026”. Repo `createdAt` is 2026-11-17 (placeholder; not the weight drop). | HF commits API `…/thinkingmachines/Inkling-2-Mini/commits/main`; card README |
| Sibling same minute | `thinkingmachines/TML-d24-varloop-research`, commit `upload model` **2026-11-20 22:02:20 UTC**, Apache-2.0, same safetensors total (26,388,646,440). Variable-loop research checkpoint, not a second architecture family. | HF API + commits |
| Total / active | **26.4B total, 3.4B active.** Three loop passes add about 1.0B each, so **5.4B effective** active parameters per token. HF safetensors total: 26,388,646,440. | Card; HF model API `safetensors.total` |
| Architecture | Looped MoE decoder. 24 stored layers; layers 6–17 run **3 passes** (48 layer visits). Each layer routes to **8 of 128** experts, no shared expert. Hybrid attention: **20 sliding-window layers (window 1,024) and 4 full layers**. **GQA: 24 query heads, 8 KV heads, head_dim 128.** Hidden size 2,304. Experts of layers 0–19 are NVFP4; layers 20–23 are MXFP8. 23-layer ViT. 3-layer DSpark/MTP draft in the checkpoint. `loop_config.is_varloop` is false on the Mini repo. | `config.json` on the repo; card; https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini ; https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini |
| Context | **1,048,576** (`model_max_length` and the card). | `config.json`; card |
| Modalities | **Text + image in, text out.** | Card |
| License | **Apache-2.0.** Card also links a Model Acceptable Use Policy. | Card front-matter `license: apache-2.0`; https://www.apache.org/licenses/LICENSE-2.0 ; https://thinkingmachines.ai/model-acceptable-use-policy/ |
| Official checkpoint | One mixed **NVFP4 + MXFP8** checkpoint, about **20 GB** (SGLang cookbook says about **18 GB**). `quantization_config.quant_algo` is `NVFP4` (`quant_method: modelopt`). **No official BF16.** | Card “Required hardware”; SGLang cookbook; `config.json` |
| Community quants (all created 2026-11-20 or 2026-11-21) | `unsloth/Inkling-2-Mini-GGUF` (created 2026-11-20 13:09 UTC), `ggml-org/Inkling-2-Mini-GGUF`, `AtomicChat/Inkling-2-Mini-GGUF`, `lmstudio-community/Inkling-2-Mini-GGUF`, `mlx-community/Inkling-2-Mini-4bit`, `mlx-community/Inkling-2-Mini-oQ4e-mtp`, `nativ-community/Inkling-2-Mini-nvfp4`. No RedHatAI, NVIDIA, or AMD quant repo turned up in the download-sorted HF search for `Inkling-2-Mini`. | HF search API, this session |
| vLLM | **Yes, not in a numbered release yet.** Recipe requires **vLLM ≥ 0.35.0 (nightly)**; image `vllm/vllm-openai:inkling-2-mini`. Blog linked from the recipe: https://vllm.ai/blog/2026-11-20-inkling2-mini . One-GPU **TP1** serve line. Verified hardware listed: GB200, **B300**, GB300, H100, H200, DGX Spark. The quoted “full 1,048,576 context, ~1.8 concurrent 1M requests” figure is for **one GB200**. H100 and DGX Spark are capped at 262,144 in that recipe. **AMD is not in the vLLM hardware list.** PR number: unknown (not on the recipe page). | https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini |
| SGLang | **Yes, day 0, not in a release wheel.** Install from **PR https://github.com/sgl-project/sglang/pull/51926** or image `lmsysorg/sglang:inkling-2-mini`. **MI350X / MI355X: image `lmsysorg/sglang:inkling-2-mini-mi35x`.** Other ROCm GPUs refuse at launch. Signed-off MI355X recipe is **TP4/EP4** (half of an 8-GPU node), not TP1. TP1 is signed off on RTX PRO 6000, RTX 5090, and DGX Spark. **B300 / GB300: yes**, image `lmsysorg/sglang:inkling-2-mini`. | https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini |
| OpenRouter | **Yes.** `thinkingmachines/inkling-2-mini`, `created` **2026-11-21 02:19 UTC**, context 1,048,576. Cheapest endpoint: **PrimeIntellect, $0.075 / $0.30 per 1M** input/output. Next: Io Net, $0.080 / $0.350. Only two endpoints. Prices are `pricing.prompt` / `pricing.completion` × 1e6. | https://openrouter.ai/api/v1/models and `…/thinkingmachines/inkling-2-mini/endpoints` |
| OpenRelay | **Not listed.** | https://openrelay.inc/inference-pricing |
| One MI355X or one B300? | **Yes.** Shipped weights are ~18–20 GB, so a 288 GB or ~275 GB GPU has on the order of 250 GB left. vLLM documents TP1 on B300-class hardware and a full 1M context on one GB200. SGLang’s signed-off MI355X recipe is TP4; single-GPU MI355X is implied by the 18 GB weight size and is **not** the signed-off parallelism. | Card; both engine pages above |

---

## 2. Hy4 — Tencent Hy Team

Flagship weight drop of the window. Does **not** fit one MI355X or one B300 at FP8 or 4-bit.

| Field | Value | Source |
| --- | --- | --- |
| Name | Hy4 (instruct), plus Hy4-FP8 and Hy4 base | https://huggingface.co/tencent/Hy4 |
| HF repos | `tencent/Hy4` (BF16 instruct), `tencent/Hy4-FP8`, `tencent/Hy4-Base` (same id as `tencent/Hy4-base`) | HF model API |
| Also listed by Tencent | ModelScope `Tencent-Hunyuan/Hy4`, GitCode `tencent_hunyuan/Hy4` | Model card |
| Weights appeared | HF commit `update` at **2026-11-20 03:05:49 UTC** (`Hy4`), 03:05:53 (`Hy4-FP8`), 03:05:54 (`Hy4-Base`). Tencent Hy posted “releasing the open weights” at **2026-11-20 03:52:40 UTC**. Repos existed earlier (`Hy4` created 2026-10-29; FP8 and Base created 2026-11-18) as shells. This is not Hy4-preview (August, text-only, Apache-2.0) and not Hy4-flash (weights 17–18 Nov). | HF commits; https://x.com/TencentHunyuan/status/2123625549147070538 ; card |
| Total / active | Language backbone **770B total, 49B active per text token**, plus **1.23B vision** when an image is used. ViT 1.23B, DFly2 drafter **3.3B** (5 layers). HF safetensors total on `tencent/Hy4`: **774,434,002,142**. | Card table; HF `safetensors.total` |
| Architecture | MoE. **78 layers: 1 dense + 77 MoE.** **256 routed experts + 1 shared, top-8.** Expert intermediate 2,048; dense FFN intermediate 18,432; hidden 6,144. Attention: **Gated DeepSeek Sparse Attention (top-2,048) with IndexCache, formulated as MLA.** Card: 64 heads, `q_lora_rank` 2,048, `kv_lora_rank` 512, indexer 32 heads × 128. Config also has `qk_head_dim` 256 (nope 192 + RoPE 64), `v_head_dim` 256, `num_key_value_heads` 8. Residual: **iHC**, 4 residual streams. Vocab 120,832. | Card; `config.json` |
| Context | **1,048,576.** | Card; `max_position_embeddings` |
| Modalities | **Text, image, and video in; text out.** vLLM recipe’s default serve line turns video off (`"video":0`) and allows one image. SGLang cookbook documents images and sampled-frame video. | Card; https://recipes.vllm.ai/tencent/Hy4 ; https://docs.sglang.io/cookbook/autoregressive/Tencent/Hy4 |
| License | **Tencent Hy License, Version 1.0.** Derived from Apache-2.0; the file says it is **not** Apache-2.0. Section 10: a MaaS business needs a separate written license if worldwide gross revenue exceeds **US$50 million** in any 12 months, **or** a product exceeds **100 million MAU**. Internal use is exempt. Contact `<email redacted>`. HF tag is `license:other`. | https://huggingface.co/tencent/Hy4/raw/main/LICENSE |
| Official checkpoints | BF16 instruct `tencent/Hy4` (~**1.55 TB**). Block FP8 instruct `tencent/Hy4-FP8` (~**786 GB** on disk; e4m3 128×128, activations group-128). BF16 base `tencent/Hy4-Base` (HF total 771,142,535,581). FP8 leaves vision, first-layer attention, indexer, shared experts, and `lm_head` in BF16. | Card; SGLang cookbook; vLLM recipe; HF safetensors |
| GGUF | `AngelSlim/Hy4-GGUF`: commit “Upload Hy4-Q4_K_M.gguF” at **2026-11-20 06:29:37 UTC** (repo created 2026-11-16). Tencent’s announcement links a GGUF; that short URL was **not resolved**, so this repo is not confirmed to be the linked artifact. File byte size: unknown. No RedHatAI / NVIDIA / AMD quant repo in the HF `Hy4` search. | HF commits for `AngelSlim/Hy4-GGUF`; announcement post above |
| vLLM | **Yes, nightly only.** **vLLM ≥ 0.35.0**, not released; v0.34.0 does not register the arch. Nightly wheels **from 2026-11-20**. Image `vllm/vllm-openai:hy4`. Default recipe: **8× B300, TP8, FP8**, `FLASHMLA_SPARSE`, DFly2, parsers `hy_v4`. FP8 aggregate GPU memory cited as **~944 GB**, BF16 **~1,859 GB**. **“AMD Instinct is not supported”** (no vision encoder, no DFly2 on the ROCm port). Hardware switch on the recipe includes H100, H200, B200, GB200, B300, GB300. PR number: unknown. | https://recipes.vllm.ai/tencent/Hy4 |
| SGLang | **Yes, day-0 image, release version not stated.** Image `lmsysorg/sglang:hy4`. Announcement post 2026-11-20 04:20 UTC. FP8 recipes: **TP8 on 8× H200/B200**, or **TP4 on 4× B300/GB300**. BF16 has **no recipe** (does not fit those shapes). DFly2 is **CUDA-only**. FP8 KV cache and decode-context-parallel are Blackwell (SM100/SM103/SM107: B200/B300/GB300), refused on H200. No MI300X/MI355X note on this cookbook page. The older PR https://github.com/sgl-project/sglang/pull/36805 (merged 2026-09-05) is **Hy4-preview**, not this checkpoint. | https://docs.sglang.io/cookbook/autoregressive/Tencent/Hy4 ; https://x.com/sgl_project/status/2123632546022039909 |
| OpenRouter | **Yes, but the listing predates the weights.** `tencent/hy4`, `created` **2026-11-04 04:20 UTC** (API launch), context 1,048,576. Cheapest endpoint: **Tencent, $0.417 / $1.2505 per 1M**. DeepInfra and Together: $0.834 / $2.501. | OpenRouter `/models` and `…/tencent/hy4/endpoints` |
| OpenRelay | **Not listed.** | https://openrelay.inc/inference-pricing |
| One MI355X or one B300? | **No.** FP8 checkpoint is ~786 GB on disk and the vLLM recipe asks ~944 GB aggregate. A uniform 4-bit packing of 770B weights is about **385 GB** (770×10⁹ × 0.5 bytes), still above 288 GB and ~275 GB before KV. SGLang’s smallest signed-off shape is **4× B300**. | SGLang cookbook; vLLM recipe; parameter count on the card |

---

## 3. Olala v1.0 mini 7A1B — OVHai LLM / OVHcloud

Small European MoE. Fits one GPU easily. Engine support is a vLLM fork, not upstream, and there is no price bar.

| Field | Value | Source |
| --- | --- | --- |
| Name | Olala-v1.0-mini-7A1B | https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B |
| HF repos | Final: `OVHaiLLM/Olala-v1.0-mini-7A1B`. Also `…-base`, `…-SFT`, `Olala-v1.0-summarise`, `Olala-v1.0-translate`, `…-GGUF`, `…-MLX`. | Card |
| Weights appeared | Commit `Olala-v1.0-mini-7A1B v1.0` at **2026-11-19 10:13:13 UTC**. Repo `createdAt` is 2026-10-30 (existed before the public v1.0 commit). | HF commits |
| Total / active | **6.8B total, about 1B active.** HF safetensors: 6,773,124,144 (BF16 6,756,461,568 + F32 16,662,576). | Card; HF API |
| Architecture | Hybrid **Mamba-3 + attention MoE**. **36 layers: 29 Mamba-3 and 7 attention** (“one in five”), each followed by MoE: **256 routed, 6 active, plus 1 shared.** Figure caption calls the attention layers **differential attention**. Config: `head_dim` 128, `num_attention_heads` 48, `num_key_value_heads` **0** (not a normal GQA label), `sliding_window_size` 65,536, `mamba_headdim` 64, `mamba_d_state` 128. Text only. | Card; `config.json` |
| Context | Card: **32K tokens (input + output)**. Official serve flag: **`--max-model-len 32688`**. Config `max_position_embeddings` is 100,000, which is higher than the published context. A 128K version is described as not yet released. | Card; serve block in the README; `config.json` |
| Modalities | **Text only.** | Card |
| License | **Apache-2.0.** | Card |
| Checkpoints / quants | Official BF16 on the main repo. Official GGUF (`OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF`, card: **6 to 14 GB**). Official MLX (**5.4 GB**). No FP8, NVFP4, AWQ, RedHatAI, NVIDIA, or AMD quant found. | Card |
| vLLM | **Fork, not upstream.** `github.com/OVHai-LLM/vllm`, `VLLM_VERSION_OVERRIDE=0.33.0`, precompiled wheel commit `4135590ffe0119dab806d78a7eee6dff3be2618e`. Image `ovhaillm/olala-vllm` (CUDA 13.0, host driver ≥ 580). Must set `--mamba-ssm-cache-dtype float32`. Card: stock vLLM cannot load `model_type: olala`. Tested on L4 and L40S. **ROCm / MI300X / MI355X: unknown** (not mentioned). | Card serve section |
| SGLang | **Unknown / not mentioned** on the model card. Card says stock llama.cpp, mlx-lm, Ollama, and LM Studio cannot load it yet; the GGUF path is their own build. | Card |
| OpenRouter | **Not in the 476-model catalog** fetched this session (no `olala` id). | https://openrouter.ai/api/v1/models |
| OpenRelay | **Not listed.** | https://openrelay.inc/inference-pricing |
| One GPU? | **Yes.** BF16 weights are about **13.5 GB** (6.756×10⁹ BF16 params × 2 bytes). A 24 GB NVIDIA GPU is the card’s stated minimum. KV for a 32,688 context is negligible next to 275–288 GB. | HF safetensors; card |

---

## 4. WeMM-Video — Tencent Hunyuan

Omni understanding models (text out). Weights uploaded 2026-11-20. The 35B-A3B fits one target GPU. Serving is patched forks, and there is no API price bar.

Technical report: https://arxiv.org/abs/2611.26409 (HF paper page dated 2026-11-20). License for the family: **Apache-2.0** (35B card). Context in every text config read: **262,144**. Modalities: **text, image, audio, and video in; text out.**

| Repo | HF safetensors total | Architecture (from that repo’s `config.json` / card) | BF16 upload commit |
| --- | --- | --- | --- |
| `tencent/WeMM-Video-2B` | 3,283,624,264 | Dense Qwen3.5-style: 24 layers, hidden 2,048, **GQA 8 / 2**, head_dim 256, full attention every 4th layer, the rest `linear_attention` | **2026-11-20 01:41:08 UTC** |
| `tencent/WeMM-Video-4B` | 5,611,749,896 | Dense: 32 layers, hidden 2,560, **GQA 16 / 4**, head_dim 256, same hybrid | **2026-11-20 04:08:42 UTC** |
| `tencent/WeMM-Video-9B` | 10,488,603,384 | Dense: 32 layers, hidden 4,096, **GQA 16 / 4**, head_dim 256. Card calls it 9B. | Upload commit **not in the newest four** (those are card edits 2026-11-20 through 2026-11-21). Initial commit 2026-11-17. Treated as the same release; exact file time unknown. |
| `tencent/WeMM-Video-35BA3B` | 36,177,564,536 | Card: **35B total / 3B active**. Config: 40 layers, hidden 2,048, **256 experts, top-8**, shared-expert intermediate 512, **GQA 16 query / 2 KV**, head_dim 256, linear attn key/value head_dim 128 (16 key heads, 32 value heads), full attention every 4 layers | **2026-11-20 03:57:36 UTC** |

Sources: HF model API; HF commits; `config.json` on each repo; https://huggingface.co/tencent/WeMM-Video-35BA3B

| Field | Value | Source |
| --- | --- | --- |
| Official quants | 35B: `…-FP8` (upload **2026-11-20 05:53:01 UTC**), `…-AWQ`, `…-GGUF`, `…-NVFP4`. Same four suffixes exist for 9B, 4B, and 2B under `tencent/`. All official, not RedHatAI / NVIDIA / AMD. Per-file quant upload times other than the 35B FP8 commit were not opened. | HF author listing; 35B-FP8 commits; 35B card links |
| vLLM | **Patched, not upstream.** 35B card: patched **vLLM 0.18.0** for the printed command, and patched **vLLM 0.31.0** for BF16. Recipe: https://github.com/Tencent/WeMM-Video/blob/main/docs/serving.md . Load check cited on **H20**. Example uses **TP across two GPUs**. | 35B card |
| SGLang | **Patched.** Card: patched **SGLang 0.5.16 or 0.5.19**, BF16, `--attention-backend triton --linear-attn-backend triton`. | 35B card |
| AMD / Blackwell | **Unknown.** The card does not mention MI300X, MI355X, B200, or B300. | 35B card |
| OpenRouter | **Not in the catalog** fetched this session. | https://openrouter.ai/api/v1/models |
| OpenRelay | **Not listed.** | https://openrelay.inc/inference-pricing |
| One GPU? | **Yes for every size here**, including 35B BF16. 36.178×10⁹ BF16 params × 2 bytes ≈ **72.4 GB**, leaving ~200 GB on a 288 GB MI355X and ~200 GB on a ~275 GB B300 before the audio/vision encoders’ activations. Official FP8 and NVFP4 are smaller. Long video, not the text weights, is the likely memory risk; the card’s example caps frames. | HF safetensors; 35B card |

---

## 5. Ayb 0.5 — Metric (Yerevan)

Armenian adaptation of Qwen3.5. Public blog post is inside the window. Small, gated, non-commercial. Not a general cost-per-request candidate.

| Field | Value | Source |
| --- | --- | --- |
| Name | Ayb 0.5 family: Ayb-0.5-0.8B, Ayb-0.5-4B, Ayb-0.5-9B | https://huggingface.co/blog/Metric-AI/ayb-05-intro (published **November 20, 2026**) |
| HF repos | `Metric-AI/Ayb-0.5-0.8B`, `Metric-AI/Ayb-0.5-4B`, `Metric-AI/Ayb-0.5-9B` | Blog |
| When weights appeared | Blog says the family is introduced that day. HF `createdAt`: **9B 2026-11-19 04:50 UTC**, **4B 2026-11-19 10:24 UTC**, **0.8B 2026-11-16 12:02 UTC**. Commit history returned **401** (gated), so whether the 0.8B repo was public before 19 Nov is **unknown**. 4B and 9B repo creation is inside the window. | HF model API; blog |
| Size | HF safetensors, all BF16: **0.8B = 875,929,152**; **4B = 4,666,091,008**; **9B = 9,673,027,312**. Dense (no MoE tag). | HF API |
| Architecture | **Qwen3.5 continued pretrain** with an Armenian tokenizer. Vision encoder and projector left unchanged, so image input remains. Head counts, sliding window, and context length: **unknown** (gated `config.json` was not read). Bases recorded on the cards: `Qwen/Qwen3.5-0.8B-Base`, `Qwen/Qwen3.5-4B-Base`, `Qwen/Qwen3.5-9B-Base`. | Blog; HF `cardData.base_model` |
| Modalities | Text, and image via the untouched Qwen3.5 vision tower. Pipeline tag `image-text-to-text`. | Blog; HF API |
| License | **CC BY-NC 4.0.** HF `gated: manual`. | HF `cardData.license` |
| Quants | **None found.** HF search for `Ayb-0.5` returned only the three official repos. | HF search |
| vLLM / SGLang | **Unknown.** Not stated on the blog. A Qwen3.5 checkpoint often loads in existing Qwen3.5 paths; that was **not verified**. | Blog |
| AMD / Blackwell | **Unknown.** | — |
| OpenRouter / OpenRelay | **Not listed** in either catalog fetched this session. | OpenRouter `/models`; OpenRelay pricing page |
| One GPU? | **Yes** on size alone. 9B BF16 ≈ **19.3 GB**. | HF safetensors |

---

## 6. In the window, not generative LLMs

Counted so they are not mistaken for missing LLM drops. They return calibrated decisions, not text.

| Model | HF | Weights | What it is | License on HF | Price | One GPU |
| --- | --- | --- | --- | --- | --- | --- |
| bison-1 | `baselabs/bison-1` | Created **2026-11-20 04:49:38 UTC**. Weight copy commits 04:56 UTC. | Decision model post-trained from **Gemma 4 31B**. HF total 31,273,086,512. Pipeline `text-classification`. Context on the Baseten API: 128k / 131,072. Text + image in. | HF tag **`other`**. Exact license name was not read from the file. | Baseten **$0.10 per 1M tokens** (not split input/output). **Not on OpenRouter. Not on OpenRelay.** | BF16 ~62 GB, so yes on 288 / ~275 GB. |
| groundhog-1 | `baselabs/groundhog-1` | Created **2026-11-20 04:50:20 UTC**. | Decision model from **Gemma 4 26B-A4B**. HF total 25,805,933,872. Baseten: 26B MoE, ~4B active. | HF tag **`other`**. Exact name not read. | Baseten **$0.035 per 1M tokens**. Not on OpenRouter or OpenRelay. | Yes. |

Sources: HF model API and commits; https://www.baseten.co/blog/meet-bison-1-and-groundhog-1/ ; https://huggingface.co/baselabs/bison-1 is linked from https://x.com/baselabs/status/2123887799057387902

---

## Near-misses (open weights earlier in November, or just outside the window)

Not candidates. Dates are the weight event, not a later blog post.

| Model | Weight timing | Why it is outside | Notes |
| --- | --- | --- | --- |
| **Hy4-flash** | First HF commit **2026-11-17 13:19 UTC**. Tencent post “now open-weight” **2026-11-18 04:39 UTC**. | Before 2026-11-19. A Nov 21 briefing repeated it. | `tencent/Hy4-flash`, `…-FP8`, `…-flash-base`. HF total 220,251,201,958. Talking Fingers: 220B of which 64B is Engram, 12B active, 1M context. License: Tencent Hy License 1.0. OpenRouter `tencent/hy4-flash` created **2026-11-16 14:18 UTC**, only endpoint Tencent **$0.035 / $0.1405 per 1M**. SGLang image `lmsysorg/sglang:hy4-flash`, not in a release wheel (branch cited as PR 51159). vLLM image `vllm/vllm-openai:hy4-flash`. **Not on OpenRelay.** |
| **Clairic / Clairic-flash** | HF lastModified **2026-11-18**. llm-releases.com dates the release Nov 18. | One day outside. | `Cloudflare/clairic` 27,781,427,952 params, base `Qwen/Qwen3.8-27B`, Apache-2.0. `Cloudflare/clairic-flash` 9,653,104,368, base `Qwen/Qwen3.5-9B`, Apache-2.0. Extraction models. |
| **DaisyRose-27B** | HF created **2026-11-18 05:31 UTC**. | Outside. | `RadixArk/DaisyRose-27B` plus FP8, NVFP4, GGUF. 27,781,427,952 BF16. Base Qwen3.8-27B. Apache-2.0. OpenRouter `radixark/daisyrose-27b` created 2026-11-21 00:49 UTC at **$0.15 / $1.875 per 1M**, context 262,144. |
| **SenseNova-Loop-9B** | HF created **2026-11-17 10:16 UTC**. | Outside. | `sensenova/SenseNova-Loop-9B`, Apache-2.0, ~9.66B stored, AWQ and GGUF repos created Nov 18. |
| **Aion-1.0-Instruct-2.3B** | HF created **2026-11-18 17:05 UTC**. | Outside. | `microsoft/Aion-1.0-Instruct-2.3B`, MIT, 2,288,831,488 BF16. |
| **Sarvam-105B-1711** | HF created **2026-11-15**. Changelog: Nov 17. | Outside. | `sarvamai/sarvam-105b-1711` (and `-bf16`). Apache-2.0. HF total 56,102,772,608. |
| **Inkling 2 Small** | nlp.com.ai: **November 12, 2026**. | Outside. | `thinkingmachines/Inkling-2-Small` and `…-MXFP8`. 594B / 17B, Apache-2.0. OpenRouter since 2026-11-09 at $0.30 / $1.00 per 1M. |
| **Kimi K3.1 Pro** | nlp.com.ai: weights **November 4, 2026**. | Outside. Heise’s Nov 19 piece is a recap. | 2.8T / 104B. On OpenRelay at **$0.24 / $7.00** sync (`openrelay/kimi-k3.1-pro`). |
| **LongCat-2.5** | nlp.com.ai: **November 9, 2026**. | Outside. | 1.6T / ~48B, MIT. |
| **Agnes 2.5 Pro Theta** | nlp.com.ai: **November 15, 2026**. | Outside. | ~187.8B, Apache-2.0. |
| **Marin Mochi 535B-A23B** | nlp.com.ai: **November 17, 2026**. HF lastModified 2026-11-20 was **not** confirmed as a new weight upload. | Outside on the catalogue date. | `open-athena/Marin-Mochi-535B-A23B`, OpenMDW 1.1, BF16 total 535,477,106,688. Base model, not instruct. |
| **Qwen4-Max** | API general release **November 16, 2026**. | Closed weights. | Not an open-weight drop. Smaller Qwen4 open weights were promised with no date. |

Near-miss sources: HF model API (createdAt / safetensors / license); https://www.llm-releases.com/changelog ; https://nlp.com.ai/models/ ; https://x.com/TencentHunyuan/status/2122919100729135385 (Hy4-flash post, 2026-11-18 04:39 UTC); OpenRouter `/models`; https://openrelay.inc/inference-pricing ; https://talkingfingers.net/topics/models-compute/2026/11/ (the Nov 21 Hy4-flash item is the late briefing).

---

## Recommendation

**No in-window model is on OpenRelay**, so none can be benchmarked against an OpenRelay price for that same model. The three below are the ones worth a cost-per-request run anyway. Compare their measured $/request on one MI355X or one B300 with (a) OpenRouter’s price where it exists and (b) OpenRelay’s nearest served rates, Gemma 4 31B at $0.07 / $0.30 and DeepSeek V4.1-Flash at $0.035 / $0.33.

1. **Inkling 2 Mini** (`thinkingmachines/Inkling-2-Mini`). It is the only drop that fits one B300 under a published vLLM TP1 recipe and has an official SGLang MI355X image, with a live price (PrimeIntellect $0.075 / $0.30 per 1M) sitting next to OpenRelay’s Gemma 4 31B rate.
2. **Olala-v1.0-mini-7A1B**. Apache-2.0 and about 13.5 GB BF16, so one GPU is not the constraint; use it as the small-model floor against OpenRelay’s cheapest token rates. Budget time for the OVH vLLM fork. There is no SGLang path and no hosted price.
3. **WeMM-Video-35BA3B**, only if the benchmark includes audio and video. BF16 is ~72 GB, and official FP8 / NVFP4 exist, but vLLM and SGLang are patched forks and neither OpenRouter nor OpenRelay lists it.

**Do not put Hy4 on a one-GPU bench.** FP8 is ~786 GB and the smallest signed-off SGLang shape is 4× B300. It does have a real OpenRouter price ($0.417 / $1.2505 via Tencent) if the comparison is allowed to use more than one GPU.

---

## Method limits

- Hugging Face commit pages return the newest page only. WeMM-Video-9B’s weight-upload commit was not on that page.
- Ayb commit history is gated (HTTP 401). Context length and head layout were not read.
- Tencent’s GGUF short link in the Nov 20 posts was not expanded.
- vLLM PR numbers for Inkling 2 Mini and Hy4 were not on the recipe pages.
- OpenRouter `created` times are the API’s unix timestamps converted to UTC in this session, not a separate announcement clock.
- Negative results for Qwen, DeepSeek, Mistral, Google, Zhipu, Moonshot, MiniMax, NVIDIA, Granite, OLMo, Cohere, OpenAI, ERNIE, StepFun, and MiMo are “no new generative repo in the 15 most recently modified models for that org,” not a proof that no side org published something.
