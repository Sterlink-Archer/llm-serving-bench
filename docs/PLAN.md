# Plan: Gemma 4 31B below OpenRelay's price, measured by OpenRelay's own method

Written 2026-11-22 03:5x UTC (Sat 9:5x PM US Central), after brainstorm round 1. Inputs:
- the build session's harness report;
- an independent Grok critique (kept as `research/brainstorm_r1_grok.md`);
- the playbook (`docs/PLAYBOOK_cost_per_token_gemma4_glm53flash.md`);
- OpenRelay's published post.

This plan changes only by a dated note under "Changes" at the bottom.

## 1. The claim, and the bar

**Headline claim:** self-hosted Gemma 4 31B on a rented spot GPU meets OpenRelay's own latency SLO and quality gate at a fully loaded cost per request below OpenRelay's sync list price. The workload is OpenRelay's agentic mix: 9.5K input / 380 output, cold cache.

**The price bar** (openrelay.inc/inference-pricing, saved 2026-11-22 03:23 UTC):
- Gemma 4 31B: $0.07 / 1M input, $0.30 / 1M output, $0.007 / 1M cached input;
- batch: $0.035 / $0.15.

At 9.5K uncached in / 380 out, list revenue is $0.000779 per request, so that is what each request must cost us at most. OpenRouter's cheapest provider ($0.10 / $0.34) is a second column only.

**The method we copy.** It comes from OpenRelay's post "Gemma 4 Optimization with Agentic Kernel Engineering", https://openrelay.inc/blog/gemma-4-agentic-kernel-engineering, Sep 29 2026. Their recipe repo: https://github.com/OpenRelayInc/inference-recipes/tree/main/gemma-4-31b-mi355x.
- Open-loop arrivals, 300 s per rate, two seeds (7 and 8), report the worse seed.
- SLO: p95 time-to-first-token < 4 s. Median time-per-output-token is reported alongside.
- Requests average 9.5K input / 380 output tokens. Prefill is measured as 16 concurrent 10K-token prompts.
- Quality gate: GSM8K ≥ 96.5%. Theirs read 98.8% before and after.
- Their result, one MI355X, vLLM 0.30 ROCm, MXFP4 + Google's MTP drafter with 2 draft tokens: 2.0 → 3.0 req/s per GPU at the SLO, TPOT 40–43 → 22 ms, prefill 27.8K → 41.9K tok/s.

**Break-even at their sync list** (Daytona spot GPU price + ~$0.81/h for 16 vCPU, 9.5K/380 cold):

| GPU | $/h loaded | req/s to break even | prefill offered at that rate |
|---|---|---|---|
| RTX 5090 | 1.55 | 0.55 | ~5.3K tok/s |
| L40S | 1.95 | 0.70 | ~6.6K tok/s |
| RTX PRO 6000 | 2.76 | 0.98 | ~9.4K tok/s |
| H100 | 3.28 | 1.17 | ~11.1K tok/s |
| MI355X | 4.37 | 1.56 | ~14.8K tok/s |

The batch tier ($0.000390 per request) roughly doubles each bar.

**Fairness rules** (from the critique; all are mandatory):
1. No price is printed before the same checkpoint passes greedy GSM8K ≥ 96.5% on a named split. Use 30 items to screen and 200 on the winner. Forced output length (ignore_eos) is for rate tests only, never for the exam.
2. The headline is the cold, unique-prompt mix. Cached multi-turn results are reported separately, priced at $0.007 for hits and $0.07 for misses. Never blend cache hits into the headline.
3. Report median and p95 TPOT next to goodput. A TTFT pass with a much worse TPOT than their 22 ms is called out, not hidden.
4. Two cost figures:
   - steady-state $/request, from the steady window only;
   - measured $/request, including sandbox create, image pull, weight download, compile, CUDA-graph capture and failed creates.
   Also report "payback minutes", the time it takes to recover the cold start.
   **Spot evictions** (Maks, 11-22): they are outside our control and never enter the benchmark.
   - Each one is recorded in the ledger: sandbox, cell, time, minutes and $ lost.
   - The affected cell is marked `excluded: evicted` and re-run on a fresh box.
   - Evictions are reported as their own line per GPU type: count, time lost, $ lost.
5. Beating their list price is not beating their efficiency. Always show our req/s per GPU and prefill tok/s beside their 3.0 req/s and 41.9K tok/s.
6. A number only counts with its run id in `infra/ledger.jsonl`.

## 2. What gets built

Status at writing: the generator exists and dry-ran clean against a mock. The harness, launcher and reaper exist. The rest is to do.

| Piece | Purpose | Status |
|---|---|---|
| `bench/traffic.py`, `bench/workloads.py`, `bench/matrix.py` | Workload and traffic generator. Closed loop (C) and open-loop Poisson, streaming, token-exact lengths, unique nonce per request, prefix-ratio knob. | built; mock dry run clean; 2 bugs fixed with tests |
| OpenRelay-protocol runner | The headline series: rate ladder of 0.6, 0.9, 1.2, 1.6, 2.0 × break-even; 300 s per rate; seeds 7 and 8; worse kept; steady window only | to build, on top of `traffic.py` |
| Prefill shot | 16 concurrent 10K prompts; tokens/s | to build |
| `infra/daytona.py`, `infra/reap.py`, `infra/spend.py` | Launcher with `spot: true` enforced and checked on the returned sandbox, a deadline label and TTL; a 5-minute reaper (systemd timer); spend tracking | built |
| Spend guard | Before every create, add (GPU+CPU) × TTL to the cumulative spend. After every delete, reconcile with Daytona's metered usage. Report the $50/100/200/300/400 crossings to the lead; re-plan at ~$500. | to finish |
| One-sandbox-many-cells mode | Weights reload on every create, so screen and promote cells run on one live box per GPU type, server restarted between configs | to build |
| `bench/evalrun.py` | GSM8K gate (greedy, named split) plus greedy token agreement against the reference checkpoint | built; to wire into the cell gate |
| `analysis/cost.py` and a report | Prices each cell for sync, sync-with-cache and batch tiers. Plots goodput vs offered rate, median/p95 TPOT, MTP acceptance length, and a stacked $/request (GPU, CPU, cold start, eviction) | partly built |
| `dashboard/`: a local dashboard site | Maks's ask. Polished local dashboards of every metric above, read from the ledger and result files and rebuilt from the repo with one command. **Built by a separate Opus session** (Maks: not Grok). Contents: (1) cost vs OpenRelay bar per GPU/config; (2) goodput vs offered rate with the SLO line; (3) TTFT/TPOT/ITL percentiles; (4) stacked $/request; (5) phase-timing profile per run; (6) running spend with the $50-400 milestones and the ~$500 re-plan line; (7) startup facts per config; (8) MTP acceptance vs concurrency. Every chart cites its run ids. Works offline. | to build (dashboard session) |

## 3. Observability standard (Maks's explicit ask)

Every run appends one record per phase to `infra/ledger.jsonl`. Each record carries the run id, cell id, sandbox id, GPU, image digest and model revision, plus UTC start and end:
- create requested → sandbox ready;
- image pull (cached or not);
- weight download (bytes, source, MB/s);
- engine process start → healthy;
- torch.compile;
- CUDA-graph capture (sizes);
- warmup;
- steady window;
- teardown;
- the reaper's confirmation that the sandbox is gone.

**Startup facts** are captured from the engine log. A cell is rejected if any is missing:
- attention backend per layer type, with the 512-dim global layers named;
- KV-cache capacity in tokens;
- CUDA-graph capture sizes;
- max-model-len;
- spec-decode config;
- quantization in effect.

**Per cell:**
- requests sent, completed and failed;
- TTFT, TPOT, ITL and E2E at p50/p90/p95/p99;
- input, cached and output tokens per second;
- the server's Prometheus counters, cross-checked against the client within 1%;
- MTP acceptance length;
- GPU utilisation and memory, sampled during the steady window.

**Spend:** overhead dollars, steady dollars and the running project total, printed per cell and reconciled with Daytona usage after each sandbox is deleted.

**Profile summary:** a `docs/PROFILE.md` table per stage showing where wall-clock time and dollars went (cold start vs steady vs idle). The lead reads every result file whole before quoting a number. A number without its run id is not repeated.

## 4. What gets tested, in order

Stop rules for every cell (kill and move on):
- TRITON attention on the global layers;
- errors over 1%;
- p95 TTFT over 4 s at the break-even rate;
- the GSM8K smoke under 96.5%.

**Stage 0: pilot (running Sat night).** One spot H100, vLLM. It validates the harness end to end: smoke matrix, our client against `vllm bench serve` at the same shape, a 25-item eval, the quick matrix and the startup-facts capture.

**Stage 1: screen (overnight into Sunday).** One quant per card, MTP off, 120 s at that card's break-even rate on the cold 9.5K/380 mix:
- RTX PRO 6000: NVFP4, FP8 KV, max-len 16K, text-only (mm limits 0);
- RTX 5090: NVFP4. The KV-token count decides whether 10K prompts fit at all;
- H100: FP8 weights, FP8 KV, FlashInfer on the global layers;
- L40S: FP8 or QAT W4A16, only if the log shows FlashInfer (no FP4 on Ada);
- MI355X: MXFP4 with OpenRelay's flags from their recipe, as a calibration point against their 3.0 req/s, not as part of the search.

Keep the two lowest $/request at the SLO.

**Stage 2: promote (Sunday morning), on the top two, same box.**
- MTP: 0 vs 2 draft tokens, at 0.75× and 1.25× break-even;
- `--max-num-seqs`: 32 vs 128;
- CUDA-graph capture sizes to 256 when MTP is on;
- the alternate quantization;
- max-model-len: 16K vs 32K.

Keep a change only if it adds more than 10% SLO goodput or flips the TTFT pass.

**Stage 3: confirm (Sunday afternoon), the headline numbers.** The full OpenRelay protocol on the winner (and the runner-up if budget allows):
- rate ladder 0.6–2.0 × break-even, seeds 7 and 8, 300 s, worse kept;
- 200-item GSM8K;
- the 16 × 10K prefill shot.

**Stage 4: secondary workloads (Sunday afternoon), on the winner.**
- Cached multi-turn: console_chat shape and OpenRelay's growing-prefix pattern, priced with the cached rate;
- ciu_screen and fin_extract shapes (production pipeline lengths, counts only);
- an offline batch run against the batch tier.

**Stage 5: the original contribution, if time.** The full knee curve on one RTX PRO 6000 (or the winner):
- goodput, median TPOT, acceptance length and fully loaded $/request against offered rate, at draft width 0 and 2;
- mark where speculation stops paying.

If time remains, add a concurrency-adaptive draft width (spec on at low load, off at high load).

**Out of scope this weekend.** GLM-5.3-Flash: it needs 4 large GPUs, nightly builds and patches, and an OpenRouter provider already sells it near or below plausible cost. Also out:
- RTX 4090, too little KV room;
- TP2 / multi-GPU;
- DFlash and EAGLE-3;
- router demos.

The just-released model Maks mentioned is phase 2, chosen at brainstorm round 3.

## 5. Budget and timeline (US Central)

- Spend Daytona credit as needed. Report crossings of $50 / $100 / $200 / $300 / $400 to Maks without pausing. Stop and re-plan near $500.
- Rough stage estimates (screening is cheap; the confirm stage is the big one):

  | Stage | Estimate |
  |---|---|
  | pilot | ≤ $5 |
  | screen | $15–30 |
  | promote | $20–40 |
  | confirm | $30–60 per card |
  | secondaries and contribution | $20–50 |

- Sat night: pilot, harness fixes, protocol runner, spend guard, one-box mode.
- Overnight: stage 1 screen.
- Sun morning: brainstorm round 2 outcome to Maks; stage 2.
- Sun afternoon: stages 3–4.
- Sun evening: stage 5 if time; write-up and README results; repo pushed (private).

## 6. Brainstorms

- **Round 1:** done Sat 9:4x PM. Participants: lead, build session, Grok reviewer.
- **Round 2:** when the stage-1 screen finishes (expected overnight). Lead, build session and a fresh Grok reviewer pick the two finalists and the stage-2 cells from the screen data only. A summary goes to Maks by Sun 9:00 AM Central.
- **Round 3:** Sun ~11:00 AM Central, with Maks. Review screen and promote results, fix the confirm cells and the headline claim wording, and pick the phase-2 just-released model.

## Changes
- 2026-11-22 ~07:1x UTC (Maks): **every quantized config gets compared with the original-precision model.**
  - Gemma 4 31B: a self-hosted BF16 reference run (GSM8K 250, MMLU-Pro 200, greedy agreement on ~300-500 fixed prompts). Each quantized config is scored against it: accuracy delta with CI, plus agreement rate. MTP on vs off doubles as a losslessness check.
  - Inkling-2-Mini: the full-precision original if Thinking Machines published one; otherwise this is stated plainly and the official mixed NVFP4/MXFP8 checkpoint is the reference.
  - Cloudflare Workers AI (and quantized OpenRouter providers) as third-party reference rows, labelled with their stated precision or "precision not disclosed".
  - The results go into `results/QUALITY_TABLE.md` and a dashboard Quality view. A price is only quotable beside its quality row.
- 2026-11-22 ~07:0x UTC: Maks approved all round-3 items; phase 2 is the lead's call.
  - **Phase 2 = thinkingmachines/Inkling-2-Mini.** Released Nov 20 2026; 26.4B total / 3.4B active looped MoE; ~20 GB NVFP4 + MXFP8; Apache-2.0. It's a new architecture with immature runtime support, which is room to optimize.
  - **Bar:** OpenRouter's cheapest, PrimeIntellect $0.075 / $0.30 per 1M (cached $0.015). OpenRelay doesn't list it.
  - **Owner:** a dedicated session, `recipes/inkling/`, journal 200-299.
  - **GLM-5.3-Flash** is parked as a possible phase 3 on a single MI355X with MXFP4 (OneNexus checkpoint), so no B300 is needed.
- 2026-11-22 ~06:5x UTC, **brainstorm round 3** (Maks approved; the lead's recommendations are adopted):
  - Headline as in WRITEUP.md: OpenRelay's recipe on Daytona spot = 0.61× their price on real text, 0.53× with their client. The "our recipe beats theirs" sentence is added only once the paired confirm is claim-grade.
  - **Priority 1:** our MI355X recipe (drafter compaction patch + MTP 3 + max-num-seqs 96 + graphs 384) to claim-grade on both columns, via the paired same-GPU confirm.
  - **Priority 2:** B300 + SGLang. Run text-only (prefill graphs back) and use a checkpoint that quantizes more than ModelOpt's.
  - **Phase 2:** the agents pick a just-released open model. The build session scouts and shortlists; no spend until the lead confirms.
  - **Dashboard:** a same-GPU ours-vs-theirs view, plus polish.
  - **Review and publish:** Maks reviews on Sunday afternoon; the repo stays private until then.
- 2026-11-22 05:1x UTC, **first price result** (lead-checked against results/PROTOCOL_TABLE.md): MI355X running OpenRelay's recipe (their image digest, flags and client) passes p95 TTFT < 4 s at 3.25 req/s on both seeds (3.28 s / 2.55 s, 0 errors) and fails at 3.5. Prefill is 41.6-42.8K tok/s; gates 99 and 98/100.
  - At Daytona's published spot rate ($4.76/h incl. host), that's ~$0.000417 per request on achieved throughput (0.535× OpenRelay's sync list); ~0.47× at the metered rate.
  - **Open before it's our claim:** real-text column B; MTP acceptance logged; achieved-throughput pricing.
  - This row is their recipe reproduced, not ours. Our improvements come from the recipe session.
- 2026-11-22 ~05:1x UTC, brainstorm round 2 (interim screen + `research/brainstorm_r2_grok.md`). Decisions:
  1. **Two columns for every MTP result.** Column A is OpenRelay's random-token client (comparable to their 3.0 req/s). Column B is real text. **The price claim must hold on column B.** Clearing break-even at p95 TTFT < 4 s on real text beats their price honestly. Column A is reported beside it for comparability.
  2. **Screen rungs (120 s, one seed) only screen.** They are shorter than a request's lifetime at these loads, so their TTFT seconds don't rank boxes. Ranking and claims use 300 s steady windows on both seeds, with server Prometheus prompt tok/s within 1% of the client.
  3. **MI355X: check the fast path first.** Before calling any req/s "their 3.0", confirm the public image has their kernels. Check the AITER FP4 asm GEMM env and the 512-dim prefill tile, then run their prefill probe (16×64 of 10K, twice, keep the second). Expected tiers: ~41.9K tok/s full path, ~27.7K tile missing, ~18.9K Triton GEMM. This is what the MI355X recipe session starts from.
  4. **PRO 6000.** The FP4 path is confirmed: `FlashInferCutlassNvFp4LinearKernel`, W4A4. Add a 2K/10K/16K prefill length sweep to the 04:53Z profile box. Flat means GEMM-bound (the chip's ceiling); a drop with length means the h512 attention kernel.
  5. **B300.** Run a speculation-off prefill probe with kernel names and CUDA-graph sizes logged before any MTP comparison.
  6. **H200's TPOT is ~2× the H100's** at the same prefill. Attribute it (more KV → more concurrent decodes, or noise) before blaming prefill chunking.
- 2026-11-22 04:4x UTC (Maks):
  - **Dashboard redesign:** React + TanStack Table/Router, Tailwind/shadcn-style, far less text, KPI tiles, offline. A new "Journey" page shows baseline → best, step by step.
  - **Weights staged once** per checkpoint on our own fast storage (Daytona volume or object storage), measured against fixed parallel HF download. Target ~1 min per box.
  - **Experiment journal:** `results/experiments.jsonl` (config diff vs parent, hypothesis, run ids, metrics, delta, kept or reverted) plus `docs/JOURNAL.md`, so the write-up shows where we started, what changed, what worked and where it landed.
- 2026-11-22 04:3x UTC, after the pilot and a read of OpenRelay's bench scripts (OpenRelayInc/inference-recipes @6767b4d, saved in `research/raw/openrelay_recipe/`):
  - **Their exact client is our protocol default.** `vllm bench serve` on /v1/completions with random-token prompts, input uniform 1.9K-17.1K (mean 9.5K), output uniform 76-684 (mean 380), rate × 300 prompts per rate with the whole run counted. Prefill probe: 16 concurrent × 64 prompts of 10K tokens, run twice, the second counts. Gate: GSM8K test[:250] with their prompt and '####' scoring. Our text-prompt client is the cross-check.
  - **Their headline is likely inflated by random-token prompts.** Their own script notes random tokens give "97-98% MTP acceptance per position, far above real text". We report MTP results under both their random-token method (comparable) and real text (honest), side by side.
  - **max-model-len 32768,** not 16K: their inputs reach 17.1K + 684 out.
  - **Screen gate 100 GSM8K items,** not 30: at 98.8% true accuracy, 30 items falsely fail about 5% of good configs. The winner gets their 250-item split.
  - **Screen ladder** 1.0 / 1.5 / 2.0 × break-even, seed 7, 120 s, early stop.
  - **Break-even includes host RAM** ($0.0162/GiB/h). With 8 vCPU and 32-48 GiB hosts: PRO 6000 1.03 req/s, 5090 0.60, H100 1.31, L40S 0.74, MI355X 1.69. Right-sizing host RAM is a lever in itself.
  - **Pilot facts** (H100, vLLM 0.33.0 cu130, FP8 + FP8 KV):
    - the global 512-dim layers run FlashAttention-4, not Triton;
    - KV capacity is only 166,343 tokens;
    - 1k/1k peaks at 1,322 output tok/s at C=64;
    - weight download 33.3 GB at 47.8 MB/s took 697 s (fix: parallel download; 11-19 MI355X reached ~1.43 GB/s).
  - **Stage-1 screen launched 04:24 UTC** in parallel: RTX PRO 6000 NVFP4, RTX 5090 NVFP4, H100 FP8, L40S W4A16-QAT, MI355X with OpenRelay's recipe. H200 is added in the next wave.
- 2026-11-22 04:0x UTC: Maks questioned the H100 pick, the overnight schedule and the sequential runs. All three were wrong for the raised budget:
  - **Parallel.** GPU types now run IN PARALLEL, as soon as the pilot proves the harness. Spot sandboxes don't count against a quota; only free capacity limits us.
  - **Capacity at 04:01Z:** MI355X 242 spot, H100 229, H200 133, RTX PRO 6000 118, RTX 5090 26, L40S 10, B300 0 (B200 not listed).
  - **MI355X first.** Run OpenRelay's own recipe: reproducing their 3.0 req/s would cost ~$0.000405/request, about half their list. Then try to beat it.
  - **The other cards, in parallel:** H200 replaces further H100 time; RTX PRO 6000 and RTX 5090 run alongside. B300 launches when spot capacity appears.
  - **Seeds:** finalists run seeds 7 and 8 on parallel boxes.
  - **Timeline:** first headline numbers in ~1-2 h, not Sunday. Brainstorm round 2 moves to "when the parallel screen is in".
  - **Host cost.** Host memory is billed too: the pilot box at 8 vCPU / 48 GiB costs ~$1.18/h on top of the GPU. Size the host to the minimum and include it in $/request.
- 2026-11-22 ~04:0x UTC: Maks added a local dashboard site. It is built by an Opus session (not Grok), owns `dashboard/` only, and reads the ledger and results through a data contract agreed with the build session.
