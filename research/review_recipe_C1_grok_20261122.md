# Independent review: MI355X recipe claim (mx_recipe_C1 vs or_recipe_paired_C1)

- Reviewer: Grok agent (grok-4.7), read-only, briefed by the lead to try to BREAK the claim.
- Finished 2026-11-22 ~07:5xZ (Sun ~01:5x CST), about 918 s.
- Scope: recompute from the raw run files (`results/mxC1_*`); fairness; protocol fidelity; cherry-picking; drafter-patch quality; price; WRITEUP overstatement.
- Claim under review: `recipes/mi355x/README.md` @ affe3e5 and the `results/CLAIMS.md` rows `mx_recipe_C1` / `or_recipe_paired_C1`.
- Lead's follow-ups are at the bottom of this file.

---

VERDICT: HOLDS WITH CAVEATS

The 3.5 / 3.25 pass-fail split, the worst-seed prices, and the GSM8K scores match the raw cell files. Nothing in those files reverses the headline. Three caveats sit beside it: column B is not a 300 s whole-run measurement, $0.000388 ties OpenRelay's Batch tier, and the drafter patch is not shown to leave outputs unchanged.

## Findings

1. **Caveat.** Column B's scored window is 240 s, not a 300 s whole run. `results/mxC1_s7_r1/20261122T061956Z/protocol_mx_recipe_text/cells/r1.916x_3.25rps_s7.json` has `args.duration` 300, `args.warmup` 60, `summary.window_s` 240. The same 60 s warmup is on every real-text cell. `bench/traffic.py` cancels requests still running at t=300 (`aborted` 97–116 of 975–1050). `docs/PLAN.md` and `results/claims.json` `rules` require a 300 s window; `analysis/screen_report.py` `cell_flaws` checks `duration_s`, which stays at the configured 300, so these cells stay claim-eligible. Putting the warmup requests back into p95 does not flip the claim: ours at 3.25 is 2.76 / 3.19 / 2.14 s; theirs is 3.71 / 4.10 / 3.80 s, still failing `mxC1_s7_r2`. Treating requests with no first token as a lower bound also leaves ours at or under 3.30 s and theirs at 4.05–4.91 s.

2. **Caveat.** $0.000388 is 0.498× the sync list and 0.996× Batch. `research/openrelay_prices_20261122T070022Z.json` Batch is $0.035 / $0.15 per 1M, so 9,500 in / 380 out is $0.0003895, which rounds to the stated $0.000390. Sync is exactly $0.000779. `results/mxC1_s7_r1/20261122T061956Z/run.json` `usd_per_h` is gpu 3.56 + cpu 0.4032 + mem 0.7776 + disk 0.0157 = 4.7565. Those unit prices are on `research/raw/daytona_pricing_20261122T030555Z.html` (MI355X preemptible $3.56/h, vCPU $0.0504/h, GiB $0.0162/h, GiB disk $0.000108/h). Ledger creates for these runs are `spot: true`, 8 vCPU, 48 GiB, 150 GiB. Worst-seed achieved at column A 3.5 is 3.405384 req/s, so 4.7565/3600/3.405384 = $0.00038799 = 0.498× sync. `docs/WRITEUP.md` lines 66–68 says column A matches Batch and column B ($0.000452) does not beat it. `recipes/mi355x/README.md` line 33 prints $0.000388 (0.50×) with no Batch line.

3. **Caveat.** The drafter patch is a different forward, and output identity is not shown. `recipes/mi355x/box/patch_draft_compact.py` `_mx_compact_prefill` gathers the last token, rebuilds attention metadata, and runs the drafter eagerly. It restores `positions` and leaves `input_ids`, `seq_lens`, and `query_start_loc` overwritten. README lines 154–156 say outputs cannot change and acceptance did not. `MX_DRAFT_COMPACT=1` is set only on `mx_recipe` (`recipes/mi355x/configs/mxC1_s7_r1.json`); `or_recipe` leaves it unset. Every box's `inspect.txt` says `PATCH draft_compact: applied`. GSM8K on the patched server, greedy via `bench/evalrun.py` `temperature=0`: the four paired boxes are 245, 246, 246, 247 of 250 (`gate_mx_recipe_gsm8kor250.json`). `mxC1_text_s8` is 99/100 (`gate_mx_recipe_gsm8kor100.json`), not 250. The four 250-item texts are not stable: `mxC1_s7_r1` vs `mxC1_s7_r2` differ on 236/250 strings, and the wrong-item sets differ (`gsm8kor-87`, `125`, `182`). Unpatched k=3 gates (`mx5_or_mtp3_seqs96`, `mx6_or_mtp3_seqs96_s7`) also differ from the patched text on 70–73/100 items, with one correctness flip (`gsm8kor-87`) that already flips between patched boxes. No long-prompt agreement file. Column A acceptance at 3.5 is 3.552–3.594, against the README's single 3.57.

4. **Caveat.** Column B seed 8 is one box. Real-text cells exist on `mxC1_s7_r1`, `mxC1_s7_r2`, and `mxC1_text_s8` only. `mxC1_s8_r1` and `mxC1_s8_r2` have no `protocol_*_text` cells. README lines 35–36 say this. The slow box did run column B: `mxC1_s7_r2` at 1965 MHz, p95 3.30 s. The text box is 2051 MHz.

5. **Nit.** Column A p95 cannot be rebuilt from samples. `bench/protocol.py` lines 128–130 empty `ttfts` and `input_lens` when the list is longer than 50. Every saved `*.json.log` P95 TTFT matches the JSON (`log` mismatch count 0). Lengths are means only.

6. **Nit.** `docs/WRITEUP.md` lines 47–48 attach "3.50 req/s … 0.50× (0.58× on real text)" in one sentence. The 0.581× is the column B price at 3.25 ($0.000452), and the table at lines 54–58 says 3.25. The status line still says ~05:25Z. The table numbers match the cells.

7. **Nit.** Seed 7 column A mean input is 9,690–9,714 tokens (about 2.0–2.3% above 9,500), identical for ours and theirs on each box. Seed 8 mean output at 3.5 is 369.57 against 380. Both recipes share the draw. The command is OpenRelay's (`--random-input-len 9500 --random-range-ratio 0.8 --random-output-len 380`, `num-prompts` = rate×300).

## Recomputed next to claimed

Column A p95 TTFT (s), README table vs `p95_ttft_ms/1000`. Clocks are the median of the mean XCD clock at ≥90% GFX activity, from `gpumon.csv`.

| box | MHz claimed / recomputed | ours 3.25 | ours 3.5 | ours 3.75 | theirs 3.25 | theirs 3.5 |
|---|---|---|---|---|---|---|
| mxC1_s7_r1 | 2033 / 2032.8 | 2.77 / 2.767 | 2.23 / 2.226 | 4.19 / 4.193 | 2.41 / 2.412 | 6.66 / 6.656 |
| mxC1_s7_r2 | 1965 / 1965.0 | 1.80 / 1.803 | 2.69 / 2.686 | 6.84 / 6.836 | 2.97 / 2.974 | 10.01 / 10.015 |
| mxC1_s8_r1 | 2045 / 2045.1 | 1.72 / 1.721 | 2.67 / 2.674 | 6.37 / 6.365 | 2.57 / 2.566 | 9.32 / 9.321 |
| mxC1_s8_r2 | 2025 / 2025.2 | 2.05 / 2.052 | 2.63 / 2.634 | 6.52 / 6.519 | 2.62 / 2.625 | 9.40 / 9.399 |

No cell rounds to a different hundredth than the README. All 3.75 cells fail. All theirs-at-3.5 cells fail.

| quantity | claimed | recomputed |
|---|---|---|
| Ours column A, every seed passes | 3.5 (4/4) | 4/4 pass; worst p95 2.686 s |
| Theirs column A | pass 3.25, fail 3.5 | worst p95 at 3.25 is 2.974 s; at 3.5, 6.66–10.01 s |
| Worst achieved req/s, ours A at 3.5 | 3.405 | 3.405384 (`request_throughput` = completed/duration) |
| $/request at that rate | $0.000388 = 0.498× | $0.00038799 = 0.49806× of $0.000779 |
| Theirs A at 3.25, worst achieved, $/request | 3.198, $0.000413 (0.53×) | 3.198459, $0.00041309 (0.530×) |
| Ours B at 3.25 p95 | 2.04–3.30 s, both seeds | 2.038, 3.298, 2.947 s on seeds 7, 7, 8 |
| Ours B at 3.5 | seed 8 2.95 s; seed 7 6.05 / 7.01 s | 2.950; 6.053 / 7.013 |
| Theirs B at 3.25 | 4.05–4.91 s, fails | 4.049, 4.910, 4.196 s |
| Ours B $/request at 3.25 (CLAIMS.md) | $0.000452 = 0.581× | worst achieved 2.920833 → $0.00045235 = 0.5807× |
| Errors | ~1/1000, under 1% | ours A 3.5 on s8_r1: 1/1050; theirs A 3.25 on s8_r2: 1/975; all other headline cells 0 |
| GSM8K test[:250], ours | 98.0 / 98.4 / 98.4 / 98.8% | 245 / 246 / 246 / 247 of 250 |
| Theirs GSM8K | 98–99% of the first 100 | 98, 99, 99, 98 of 100 |
| Prefill, 2nd of 16×10K | ours 43.6–44.4K; theirs 42.2–42.9K | 43,635–44,437; 42,158–42,922 tok/s |
| TPOT p50 at 3.25 / 3.5 | ours 20–22 / 29–36 ms; theirs 28–38 / 44–45 ms | 20.3–22.3 / 28.9–36.2; 28.1–37.8 / 43.7–45.4 |
| Server vs client prompt tokens | within 1% | column A ratio 1.0 on every cell; column B 0.99994–0.99999 |

Fairness checks that matched the files: each of the four boxes ran both recipes (`server_mx_recipe.log` / `server_or_recipe.log`). Order alternates (s7_r1 and s8_r2 ours first; s7_r2 and s8_r1 theirs first). Theirs on `mxC1_text_s8` still fails 3.25 while running second. Image digest `sha256:f22210e5…6d4cfa` on both. Their flags match `research/raw/openrelay_recipe/bench/engine_args.sh` `TUNED_ARGS` (MTP 2, `--max-num-seqs 64`, graphs through 256, `--max-num-batched-tokens 8192`). Prompt counts are rate×300 on every cell, and mean input tokens match across the two recipes on a box. Column B finished requests all have `finish=length` and `completion_tokens == max_tokens` (`ignore_eos` plus `min_tokens` in `bench/traffic.py`). Column A `ignore_eos` is not in the saved command; per-request lengths were stripped, and the output means match across recipes. Prefix-cache hit ratio is 0 on three boxes and 0.021% (10,624 / ~5.1e7) on the two seed-7 boxes.

No earlier run of this recipe (k=3, seqs 96, graphs to 384, `MX_DRAFT_COMPACT=1`) fails at 3.5. `mx10_best_dc_s7` / `s8` pass at 2.298 s and 3.065 s. The step 118 failures in `results/experiments.jsonl` are the pre-patch config. `claims.json` `excluded_cells` for both C1 groups is empty. `mxH4_*` and `q2_mi355x_recipes` have no cell files.

Column A p95 was checked against vLLM's JSON and stdout, not recomputed from per-request samples. The on-box patched source was not diffed; the evidence is `inspect.txt` plus the repo patch.

---

## Lead follow-ups (2026-11-22 ~07:5xZ)

- **Caveat 1 (column B window)** → build session.
  - Re-score column B over the full 300 s from the saved per-request files. Count the warmup requests, and count requests still running at t=300 at their lower-bound TTFT.
  - Make `cell_flaws` check `summary.window_s`, not the configured duration (the durable guard).
  - Regenerate CLAIMS.
- **Caveat 2 (Batch tie)** → README gets the Batch line (MI355X session). WRITEUP already states it.
- **Caveat 3 (patch output identity)** → MI355X session, as an identity probe on one box.
  - Test: patched vs unpatched, same k=3 config, concurrency 1, temperature 0, with short prompts and 10K-token prompts.
  - Run an unpatched-vs-unpatched control to measure run-to-run noise.
  - Check that the overwritten `input_ids` / `seq_lens` / `query_start_loc` are not read by the target model's next step.
  - Until the probe lands, README wording changes from "outputs cannot change" to "quality gate passes; identity probe pending".
- **Caveat 4 (seed-8 column B on one box)** → disclosed. A second seed-8 text box is optional (about $4).
- **Nit 5** → build session. `bench/protocol.py` keeps the per-request arrays in a sidecar file so column A p95 can be rebuilt.
- **Nit 6** → build session. Split the WRITEUP line 47–48 sentence (0.581× belongs to 3.25 req/s on column B) and refresh the status line.
