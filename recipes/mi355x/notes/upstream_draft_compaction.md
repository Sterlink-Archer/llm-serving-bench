# Draft (not filed): run the Gemma 4 MTP drafter's first pass on one token per request

Status: a DRAFT write-up for a possible vLLM issue/PR. Nothing has been filed; filing is Maks's call.

**Problem.** In vLLM 0.30 (`vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`), `AutoRegressiveSpeculator.propose`
runs draft position 0 (`_prefill`) over every token the target scheduled in the step. It reuses the target's attention metadata
and slot mappings, then keeps only `last_token_indices` (one row per request). For EAGLE-style drafters that write their own KV,
this is required. Gemma 4's MTP head (`Gemma4Speculator`) is Q-only: it reads the target's KV cache through cross-model KV
sharing, writes none, and `advance_draft_positions` is False. Every non-last row of the first pass is discarded work. In a mixed
step that is the whole prefill chunk: thousands of tokens through 4 draft layers, including a full-attention head_dim-512 layer.

**Measured** (one MI355X, `amd/gemma-4-31B-it-MXFP4` + `google/gemma-4-31B-it-assistant`, k=3, `--max-num-seqs 96`, vLLM 0.30
ROCm):
- **Profile at 3.5 req/s** (9.5K-in / 380-out random-token mix): the first pass costs ~2.9 ms of head_dim-512 prefill attention
  plus the drafter's BF16 MLPs, per ~165 ms mixed step.
- **Compaction to one token per request**, applied only to steps that contain a prefill, using the decode passes' own position and
  seq_len rule:
  - p95 TTFT at 3.5 req/s 4.16 → 3.06 s (seed 8) and passes on seed 7 (2.30 s);
  - TPOT p50 43.3 → 36.7 ms;
  - acceptance 3.62 → 3.61 tokens per verify step;
  - GSM8K test[:100] 99%.
- All-decode steps keep the FULL-graph path unchanged.

**Patch:** `recipes/mi355x/box/patch_draft_compact.py` (env-gated `MX_DRAFT_COMPACT=1`). An upstream version would key on the
speculator, not an env var: e.g. a `first_pass_needs_all_tokens` property, False for Q-only KV-sharing heads. It would also capture
the 1-token-per-request first pass in the decode CUDA-graph manager instead of running it eagerly.

## Novelty check (2026-11-22 ~09:3x UTC, read-only: source at tags/main + GitHub search; nothing filed or commented)

**1. Does vLLM 0.34.0 / current main still run draft position 0 over every scheduled token for Gemma 4 MTP? Yes.**
- **Where it lives now.** The 11-03 refactor (#60989, "Refactor AR speculators") moved the code to
  `vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py`, class `TargetDependentARSpeculator`.
- **0.34.0 and main are byte-identical** (724 lines, `diff` empty).
- **The waste.** `propose()` calls `self._prefill(num_reqs, prefill_batch_desc.num_tokens, ...)`, which runs the drafter over the
  target's padded batch (`self.hidden_states[:num_tokens_padded].copy_(hidden_states)`). It keeps only
  `last_token_indices[:num_reqs]` for sampling. The FULL-graph replay is taken only when the batch is a uniform decode batch
  (`input_batch.decode_graph_eligible`). A mixed step with a prefill chunk takes the piecewise/eager path over the whole chunk.
- **The drafter class.** `vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py`: `class Gemma4Speculator(TargetDependentARSpeculator)`
  with `advance_draft_positions = False` ("Gemma4 MTP is Q-only and reads K/V from the target's existing cache"). Unchanged in
  0.34.0 and main.
- **Porting.** Our patch targets 0.30's `AutoRegressiveSpeculator.propose`. On 0.34/main the same compaction belongs in
  `TargetDependentARSpeculator.propose` (gated on `not self.advance_draft_positions` and a prefill in the batch), after
  `_build_drafter_attn_metadata`.

**2. Is the code path shared with NVIDIA? Yes.**
- `vllm/v1/worker/gpu/` is the platform-agnostic GPU model runner (Model Runner V2) used by CUDA and ROCm alike. No ROCm-specific
  branch exists in `propose()`/`_prefill()`.
- Gemma 4 MTP deployments on H100/H200/B200/B300 pay the same waste in every step that carries a prefill chunk. #68347 below
  validates this drafter class on an NVIDIA H200.
- How much the waste costs on NVIDIA is not measured here. It scales with prefill-chunk tokens per step, the drafter's 4 layers
  (one full-attention head_dim-512 layer) and its BF16 MLP.

**3. Existing reports or fixes (GitHub search over vllm-project/vllm issues and PRs: "Gemma4 MTP", "Gemma4Speculator",
"TargetDependentAR", "last_token_indices", "draft prefill", "MTP prefill overhead"):**
- [vllm#68347](https://github.com/vllm-project/vllm/pull/68347) (open, 2026-11-18), "[Spec Decode][PD] Skip target-KV-only drafters
  on P/D prefill". It names exactly this drafter class ("Gemma 4 Assistant — Q-only attention with cross-model KV sharing") and
  states such drafters "can avoid draft-side prompt computation entirely". It applies only to **prefill/decode-disaggregated**
  serving: the prefill instance skips the drafter altogether. It does not touch aggregated serving, where one engine runs mixed
  prefill+decode steps and the drafter must still run, but only on each request's last token. That case is ours, and nothing
  covers it.
- [vllm#67104](https://github.com/vllm-project/vllm/pull/67104) (merged 2026-11-12), "[GLM-5.3-Flash Perf] Skip unused MTP
  post-attention rows". Same family of idea for GLM-5.3-Flash's MTP module: it binds `last_token_indices` so norm/MoE/final norm
  run on the needed rows only. Attention still runs on all rows (its words: "attention/KV on 24 rows -> select 4"), it is
  GLM-specific, and it does not apply to Gemma 4's `TargetDependentARSpeculator` path.
- [vllm#53420](https://github.com/vllm-project/vllm/issues/53420) (open): the cost of the draft "prefill" forward kept as KV sync
  when dynamic K resolves to 0. It is related (that forward is measurable overhead) but about a different case.
- Nothing found reports or fixes the full-chunk first draft pass for Q-only KV-sharing drafters in aggregated serving.

**Verdict.** Novel as a fix. The underlying observation, that target-KV-only drafters need no prompt computation, is shared with
#68347 (P/D prefill skip), and #67104 is a partial row-pruning analogue for GLM. A fair write-up line: "#68347 skips this drafter on
disaggregated prefill nodes; we show the same waste costs aggregated serving ~2-4% of mixed-step time and, removed, moves the
MI355X knee from 3.25 to 3.5 req/s together with k=3 and 96 sequences; the code path is shared with NVIDIA."
