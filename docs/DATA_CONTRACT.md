# Data contract v1 (FROZEN 2026-11-22) — what `dashboard/` and `analysis/` may read

Owner: the build session. Readers: the dashboard session (`dashboard/` only) and `analysis/`. Any change is
announced to the dashboard session before it lands and gets a new version line at the bottom. Fields marked
**NEW** were requested by the dashboard proposal v0. They are being built now; until a run produces them, readers
show "awaiting data" and never fill them in.

## 0. Identity keys
- `run_id` = `<label>/<UTC stamp>`: the results dir relative to `results/`, e.g. `pilot_h100_vllm_fp8/20261122T034304Z`.
- `config_id` = `<label>:<server name>`, e.g. `pilot_h100_vllm_fp8:fp8dyn`.
- `cell_id` = `<dir>/<cell label>`: `<dir>` is a matrix dir `m_<server>_<preset>` or a protocol dir `protocol_<server>`.
- `gpu` = the Daytona type string, a key of `infra/daytona.GPU_SPOT_USD_H` (`H100`, `RTX-PRO-6000`, `MI355X`, …).
- Times are UTC ISO `YYYY-MM-DDTHH:MM:SSZ`.

## 1. `infra/ledger.jsonl` (append-only, one JSON object per line, key `event`)
- `create` (frozen): `t, event, id, run, snapshot, gpu, gpus, cpu, mem, disk, spot, usd_per_h{gpu,cpu,mem,disk,total}, ttl_min, deadline`.
  **NEW** `run_id`.
- `delete` (frozen): `t, event, id, why`. **NEW** `run_id` when deleted by run.py; the reaper writes `run_id: null`.
- **NEW** `phase`: `t, event, run_id, config_id|null, cell_id|null, sandbox_id, gpu, gpus, image, image_digest, model_repo|null,
  model_rev|null, phase, t_start, t_end, dur_s, usd_per_h, usd, nested_in|null, ok, detail{}`.
  - `phase` is one of `create | image_pull | weight_download | engine_start | torch_compile | graph_capture | warmup |
    steady | eval | teardown | reap_confirm | failed_create | eviction | idle`.
  - `usd = dur_s/3600 × usd_per_h` is an estimate. Metered dollars come from `spend`.
  - `torch_compile` and `graph_capture` have `nested_in: "engine_start"`. Never stack a nested phase on top of its parent.
  - `detail`:
    - `weight_download {bytes, source:"huggingface_hub|range", mb_s}`
    - `image_pull {cached: bool|null}`, null when Daytona does not say
    - `graph_capture {sizes:[int]}`
    - `steady {cell_ids:[...], requests_completed}`
    - `eviction {spotEvictedAt}`
- **NEW** `spend`: `t, event, est_total_usd, metered_total_usd|null, metered_file`. Written after each reconcile with
  Daytona's metered usage.
- **NEW** `evicted` (Maks 11-22: spot preemption is outside our control; never part of a benchmark number):
  `t, event, run_id, sandbox_id, gpu, gpus, state, spotEvictedAt, evicted_at, last_good_t, interrupted {server, step}|null,
  minutes_lost, usd_lost, usd_per_h, label:"spot preemption, outside our control"`. The interrupted step's record in
  run.json gets `excluded: "evicted"`; run.py re-runs the config on a fresh box (new run_id, `rerun_of_evicted` in run.json).
  Report evictions as their own line (count, minutes, $ per GPU type), never folded into a cell or a $/request.
- **NEW** `milestone`: `t, event, threshold_usd, total_usd, basis:"metered"|"estimate"`. Written when the project total
  first crosses 50/100/200/300/400; the lead is told at the same moment.

## 2. Traffic cells (frozen)
`results/<run_id>/m_<server>_<preset>/cells/<cell>.json` and `.../matrix_summary.json` (`meta`, `rows`, `knees`).
The protocol runner writes the same cell format under `results/<run_id>/protocol_<server>/cells/`.
Fields: as listed in the proposal §2. **NEW** top-level `schema: "traffic/1"`. Times inside `summary` are in seconds.

## 3. `results/<run_id>/run.json` (frozen)
As listed in the proposal §3. Coarse fallback only; `phase` events are the profile.

## 4. **NEW** `results/<run_id>/startup_facts_<server>.json`
```
{schema:"startup_facts/1", run_id, config_id, engine, engine_version, image, image_digest, model_repo, model_rev,
 attention_backend: {"sliding": str|null, "global_512": str|null}, attention_backend_lines: [raw log lines],
 kv_cache_tokens, kv_cache_gib, kv_cache_dtype, max_concurrency_at_max_len, cuda_graph_sizes: [int], max_model_len,
 quantization: {weights, detail}, spec_decode: null | {method, draft_model, num_speculative_tokens},
 gpu_mem_after_load_mib, source_log, missing: [names], rejected: bool}
```
`rejected` is true when `attention_backend.global_512`, `kv_cache_tokens` or `cuda_graph_sizes` is missing. It is
also true when the global layers run on `TRITON_ATTN`, which is a PLAN §4 stop rule. A rejected config gets no price.

## 5. **NEW** `results/<run_id>/protocol_<server>/protocol_summary.json`
Shape as in the proposal §5, plus `seeds:[7,8]` and `kept: "worse"`.
- `goodput_rps`: completed requests per second over the steady window of one (rate, seed) cell. It counts only when
  that cell meets the SLO: p95 TTFT < 4 s, errors < 1%, no client-side drops. Otherwise it is 0. Achieved and
  actually-arrived rates are reported beside it, not used as criteria.
- `max_rate_at_slo_rps`: the highest offered rate on the ladder whose worse seed meets the SLO. It is the headline
  req/s per GPU, comparable to OpenRelay's 3.0.
- `prefill`: 16 concurrent 10K-token prompts, `prefill_tok_s` = the sum of prompt tokens ÷ (last first-token time −
  first send time).

## 6. Cost
- Read `results/cost_rows.json`, written by `analysis/report.py` from `analysis/cost.py`. Do not import the module;
  the JSON is the contract.
- **NEW** `results/<run_id>/cost_measured.json`: shape as in the proposal §6. `basis` says over which steady requests
  the cold start is amortised. `metered_usd` is added when Daytona usage is available.

## 7. Eval (frozen + NEW)
`results/<run_id>/eval_*.json` and `results/<run_id>/gate_*.json` (same shape; gate files are written by the gate step;
the OpenRelay gate has task `gsm8kor`, their prompt/scoring on openai/grade-school-math test, first N problems): `label, summary.<task>.{n, correct, acc, no_answer, errors, truncated}`.
**NEW** `split` (e.g. `"gsm8k_200[:30]"`) and `gate {task:"gsm8k", threshold:0.965, acc, pass}`.

## 7b. Protocol rungs and prices (v1.2)
- A `ladder[]` rung in protocol_summary.json: `{multiple, rate, seeds: {"<seed>": "<path relative to the protocol dir>"},
  results: {"<seed>": {client, achieved_rps, arrived_rps, goodput_rps, ttft_p50_s, ttft_p95_s, ttft_p99_s, tpot_p50_s, tpot_p95_s,
  e2e_p50_s, errors, requests_sent, dropped, slo_pass, out_tok_s, in_tok_s, ...}}, kept_seed, <the kept seed's fields copied up>}`
  or `{multiple, rate, skipped}`. With client "vllmbench" the seed path points at vLLM's own `vllm bench serve` result JSON
  (not traffic/1; big per-request lists trimmed); with client "traffic" it points at a traffic/1 cell.
- `results/protocol_rows.json` (analysis/screen_report.py, also run by analysis/report.py): one row per (rung, seed) with
  `run_id, config_id, protocol, gpu, client, multiple, offered_rps, seed, kept, slo_pass, achieved_rps, goodput_rps, ttft_p95_s,
  tpot_p50_s, tpot_p95_s, errors, usd_per_h, usd_per_req_at_offered, usd_per_req_at_achieved, openrelay_sync_usd_per_req,
  ratio_sync_at_offered, file, summary_file` (paths relative to the repo root). This prices vllm-bench rungs.
  v1.4 adds (additive, may be null until Daytona reports the sandbox's usage): `usd_per_h_metered, metered_read_at,
  ratio_sync_at_offered_metered`. `usd_per_h` stays Daytona's PUBLISHED rates (the headline, conservative).
- **v1.5** protocol_rows also carry `claim_group` (the key that merges twin runs of one config, e.g. s2_mi355x_or_recipe_seed7 and
  _seed8 into `s2_mi355x_or_recipe`; set by `claim_group` in the config, else the label without `_text` / `_seedN`) and `column`
  ("A" = OpenRelay's client, random tokens; "B" = our real-text client), plus `duration_s, prompt_tok_xcheck_ratio,
  prompt_tok_xcheck_pass, spec_accept_len, spec_acceptance_rate, prompt_tokens_mean, stratified_lengths, exact_prompt_tokens`.
  Merge rule (results/claims.json does it already): per (claim_group, column, offered_rps) a rate passes only if every seed
  passes; price on the worst seed's achieved_rps. Read results/claims.json for the claim; do not re-derive it.
- **v1.8 scoring window + cross-check basis (Grok review of the C1 claim):** traffic.py (column B) cells are scored by ONE rule,
  bench/protocol.py `score_cell`. It takes the LAST 300 s of the cell (new cells run 360 s with 60 s warmup; old 300 s cells
  are re-scored over 0-300 s, warmup included) and treats a request with no first token by the window end as +inf for
  pass/fail. protocol_rows carry the re-scored values in `achieved_rps, ttft_p95_s, slo_pass` plus `scored_window_s,
  ttft_p95_lower_s` (censored requests at their lower bound), `ttft_censored`, and the old 240 s judge values in
  `window_ttft_p95_s, window_achieved_rps`. Old runs' protocol_summary.json still hold the 240 s values: protocol_rows and
  claims.json are authoritative. `duration_s` on protocol_rows is now the SCORED window (claim rule: >= 300 s).
  The prompt-token cross-check is whole-cell (counter before the first request vs after all finished and settled), not the
  steady-window delta: vLLM counts at prefill, SGLang at completion with a variable lag (bench/counter_probe.py).
  `prompt_tok_window_ratio` is informational. vllm bench cells write their full per-request arrays to
  `<cell>.per_request.json` (the summary file stays trimmed).
- **v1.7 price bars:** a config may set `price_bar` {name, in_usd_per_m, out_usd_per_m, cached_usd_per_m?, source} (phase 2:
  OpenRouter PrimeIntellect's Inkling 2 Mini price); without it the bar is OpenRelay's own Gemma 4 31B sync price. One rule:
  bench/pricebar.py. protocol_rows gain `price_bar_name, price_bar_usd_per_req, price_bar_source`; every `ratio_*` field is vs
  that bar, and `openrelay_sync_usd_per_req` is null when the bar is not OpenRelay's Gemma price. claims rows gain
  `price_bar_name, price_bar_usd_per_req, ratio_vs_bar, ratio_vs_bar_metered`; `ratio_vs_openrelay_sync*` stay, null for other
  bars. Configs may also set `startup_rules` {require_global_512, triton_ok} (model-aware startup rejection; Gemma defaults).
  Protocol dirs: a ladder dir belongs to the server named in its summary's `config_id` (`<label>:<server>`); that field is
  authoritative, not the dir name (SGLang runs name dirs after the step).
- **NEW v1.6 — the headline source: `results/claims.json`** (analysis/screen_report.py, also run by analysis/report.py; CLAIMS.md
  is its table). `{schema: "claims/1", openrelay_sync_usd_per_req, rules, claims: [row]}`, one row per (claim group, column):
  `group, column ("A"|"B"), column_name, origin ("openrelay_recipe" = an exact reproduction of OpenRelay's recipe, set by
  `origin` in the config; every other config, including variants of their recipe, is "ours"; "mixed:..." if a group mixes),
  gpu, engine ("vllm"|"sglang"|"llama.cpp"), rates_tested, runs, max_rate_all_seeds_pass, seeds ["run_id:sN"],
  worst_seed_achieved_rps, worst_seed_ttft_p95_s, prompt_tokens_mean_range, spec_accept_len[], usd_per_h, usd_per_req
  (= usd_per_h / 3600 / worst_seed_achieved_rps), ratio_vs_openrelay_sync, usd_per_h_metered, usd_per_req_metered,
  ratio_vs_openrelay_sync_metered, claim_grade (bool), not_claim_grade_because []}`. Rows with no passing rate carry
  max_rate_all_seeds_pass null, claim_grade false. Claim-grade = >= 2 seeds, 300 s cells, server-vs-client prompt tokens within 1%,
  and for column B tokenizer-exact, stratified prompts with a realized mean within 9.5K +-2%. Column B is the claim; A sits beside it.
- **NEW v1.4** `results/metered_usd.json` (infra/spend.py): `{schema: "metered/1", read_at, window, source, metered_total_usd,
  sandboxes: {<sandbox_id>: {run, run_id, gpu, cpu, mem, disk, metered_usd, closed_intervals, open_intervals, metered_hours,
  metered_usd_per_h, published_usd_per_h, our_estimate_usd}}}`. metered_usd_per_h = sum(price)/sum(hours) over closed intervals.
- `results/cost_rows.json` (analysis/report.py) covers `m_*`, `out/m_*` AND `protocol_*/cells/*.json` cells of schema traffic/1;
  `file` relative to the repo root.

## 7c. Experiment journey: `results/experiments.jsonl` (v1.2; Maks 11-22)
Append-only; one line per decision step; a later line with the same `step` supersedes an earlier one. Written by
`analysis/journal.py add` (metrics computed from the run's files, never typed); rendered to `docs/JOURNAL.md`.
```
{schema:"experiment/1", step:int, t, kind:"baseline"|"change"|"rejection"|"infra"|"calibration", config_id|null,
 parent_step:int|null, parent_config_id|null, change:"<= 80 chars", hypothesis:"one line", why:"one line: what happened / why the decision",
 run_ids:[...], basis:"pilot"|"screen"|"promote"|"ladder"|"calibration"|"infra", decision:"baseline"|"kept"|"reverted"|"rejected"|"info",
 slo_goodput_rps, max_rate_at_slo_rps, breakeven_rps, achieved_rps_at_breakeven, usd_per_req, usd_per_req_at_achieved, ratio_sync,
 ttft_p95_s, tpot_p50_s, prefill_tok_s, kv_tokens, mtp_accept_len, gsm8k_acc, download_mb_s, engine_ready_s, usd_per_h, run_usd_est,
 diff:{gpu|cpu|mem|image|weights:[old,new], flags_added_or_changed:{...}, flags_removed:{...}}, delta_vs_parent:{metric:[parent,this]},
 note?, evidence_files:[...]}
```
Two writers (v1.3): the build session owns steps 0-99, the MI355X recipe session (`recipes/mi355x/`) owns steps 100-199
(`journal.py add --owner mi355x-recipe`). Appends and renders hold an exclusive flock on `results/experiments.jsonl.lock`;
each record is one write() of one line. Readers must tolerate steps out of numeric order in time.

Metric meanings: `usd_per_req` = $/h ÷ 3600 ÷ max_rate_at_slo_rps (null when no rung met the SLO); `ratio_sync` = usd_per_req ÷
OpenRelay sync ($0.000779 at 9.5K/380); `ttft_p95_s`/`tpot_p50_s` are from the highest SLO-passing rung, else the 1.0x rung.
This is the dashboard proposal v0's field set plus Maks's asks (parent, diff, hypothesis, delta, prefill, KV, MTP).

## 8. Price bars (frozen)
As listed in the proposal §8.

## 9. GPU monitor
- `gpumon.csv` (frozen) is raw `nvidia-smi --format=csv,noheader` output, so values carry their units (`87 %`,
  `71234 MiB`, `312.45 W`, `1980 MHz`). On AMD it is raw `amd-smi` text instead.
- **NEW** `gpumon_norm.csv` normalises both vendors. Header:
  `t_utc,gpu_index,util_pct,mem_used_mib,power_w,sm_mhz,temp_c`. Read this one.

## 10. MTP
Acceptance length = `summary.server_metrics.spec_mean_accepted_per_step` = 1 + accepted ÷ drafts (tokens per verify
step). Draft width is in `startup_facts.spec_decode.num_speculative_tokens`.

## Git
The build session owns the repo and pushes. The dashboard session commits only paths under `dashboard/`
(`git commit -- dashboard/…`, never `git add -A`) and does not push. The build session pushes those commits with its
own.

## Versions
- v1 frozen 2026-11-22 ~04:05 UTC (proposal v0 accepted with the amendments above).
- v1.1 2026-11-22 ~04:35 UTC: `evicted` event added; §5 SLO = OpenRelay's (achieved-vs-offered reported, not a criterion);
  protocol `client` field; matrices from the pilot run live under `results/<run_id>/out/m_*` (later runs: `results/<run_id>/m_*`).
- v1.2 2026-11-22 ~04:40 UTC: §7 gate_*.json, §7b protocol rung shape + protocol_rows.json + cost_rows coverage, §7c experiments.jsonl.
- v1.3 2026-11-22 ~04:58 UTC: two journal writers, step ranges, flock.
- v1.4 2026-11-22 ~05:10 UTC: metered $/h (results/metered_usd.json; three nullable fields on protocol_rows). Additive only.
- v1.5 2026-11-22 ~05:40 UTC: protocol_rows `claim_group` + `column` and the claim-rule fields; results/claims.json + CLAIMS.md. Additive only.
- v1.6 2026-11-22 ~05:55 UTC: results/claims.json documented as the headline source; rows gain origin, gpu, engine. Additive only.
- v1.7 2026-11-22 ~07:40 UTC: per-config price bars (ratio_vs_bar), startup_rules, config_id authoritative for ladder dirs.
- v1.8 2026-11-22 ~07:55 UTC: 300 s scored window for traffic cells (re-scored), whole-cell prompt-token cross-check, vllm bench sidecars.
