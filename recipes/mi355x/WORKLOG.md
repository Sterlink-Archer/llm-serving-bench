# WORKLOG: MI355X recipe (MI355X recipe session, 2026-11-22)

- 04:5x-05:02Z: read the plan, OpenRelay's recipe @6767b4d, run.py and the reproduction. Launched wave 1: chunk 4K/6K, MTP k=3,
  a knee profile, vLLM 0.34.
- 05:05Z: reproduction of their recipe on our boxes is 3.25 req/s on both seeds (journal step 100). The bar becomes 3.5.
- 05:15-05:25Z: 0.34 stock ran all Gemma 4 linears on Triton FP4 (-43% prefill). Forced ASM plus their ported patches gave parity
  only. Chunk sizes 4K/6K/12K were worse or neutral. k=3 and `--max-num-seqs 96` were the first gains.
- 05:4x-06:1xZ: combinations, replicates, block and probabilistic verification, seqs 80/112, k=4. Box clock spread found
  (1940-2060 MHz). The second profile found the drafter's first pass wasted on prefill tokens. The compaction patch passed 3.5 on
  both seeds (step 119).
- 06:20-07:18Z: paired confirm, 4 boxes plus a real-text box. Ours passes 3.5 on all 4 cells; theirs passes 3.25 and fails 3.5 on
  all 4. GSM8K test[:250] 245-247/250 (step 122).
- Spend for this recipe: $61.60 over 32 sandboxes (`tools/spend.py`, 07:18Z). One spot eviction, re-run automatically.

DoD MI355X-recipe (t004): BASELINE: OpenRelay's published knee (research/raw/openrelay_recipe/full/results/knee.md: 3.0, 3.25
borderline) AND their exact recipe re-measured PAIRED on the same 4 boxes (theirs 3.25 / ours 3.5, `tools/confirm_table.py C1`),
plus the build session's claim for their recipe (results/claims.json s2_mi355x_or_recipe: A 3.25, B 3.0) | GAP-HUNT: confirm_table
re-derivation + gate files + README docker-run flags diffed against the measured launch; 3 found (README run block never executed
verbatim -> disclosed in README + t012; claims.json cannot group paired servers -> t007; journal.py picks the wrong gate on paired
runs -> t008), plus not-run levers t009 (EAGLE-3/DFlash2) and t010 (adaptive draft length), and the upstream decision t011 |
GAPS-FILED: t007 t008 t009 t010 t011 t012

## 2026-11-22 07:2x-09:1x UTC: follow-ups (lead and build session asks)
- Small host 4 vCPU / 24 GiB: 3.5 on both seeds, 0.431x sync / 0.86x Batch (claim-grade mx_recipe_c4m24, step 124).
- SGLang track: cannot load MXFP4 (step 125); 41% slower prefill than vLLM on the same FP8 weights (steps 126-127).
- Identity probe: patched == unpatched 60/60 at c1, temperature 0 (mxID_identity).
- Recipe spend now $69.89 (committed configs only).

DoD follow-ups: BASELINE: the C1 paired confirm + OpenRelay's recipe on the same harness | GAP-HUNT: the lead's Grok review
caveats (identity, Batch tie, acceptance range) all resolved in the README; claims re-derived with screen_report.py | GAPS-FILED:
none new (t008 journal gate pick still open with the build session)

## 2026-11-22 09:1x-11:4x UTC: round 3 + 4 (lead's consolidated queue, then "beat it by more")
- c4m24 replication: 3.5 on all 4 small-host cells; cgroup CPU sampler (`box/cgmon.py`) shows 3.2 of 4 quota cores, 0.0-0.4%
  throttled. The first sampler read the 192-CPU host from /proc/stat; it was swapped on the live boxes before the first load cell
  (journal 131). Now the headline price.
- Finer knee, paired: ours 3.6 on 3 of 4 boxes, theirs 3.3 on 2 of 4 (journal 133). 256K paired: ours 3.25 both seeds, theirs fails
  3.25 on seed 8 (journal 132).
- Levers, all no-gain or <2%: sampler 0.46% (128); attention split ~1.5%, decode split-KV tile ~0.2% (129); num_stages-2 tile 1.11x
  but aborts the compiler at >= 63 decodes (130, `box/attn_s2_safety.py`); long-prefill threshold 6144/4096 (134); on top of the
  stack, batched tokens 10K/12K and seqs 80/112 (135-138).
- Three-way confirm C2 (4 boxes, 4 vCPU host): ours + MX_PS_FP4 (GEMM session's patch) 3.5 on every box (worst 2.41 s);
  ours-today 3.25 (misses 3.5 on the 1927 MHz box); theirs fails 3.25 on seed 8. 3.75 on 1 of 4 (journal 139).
- GLM README: honest headline, inverse MTP, why no cheaper GPU, the 0.34 Triton fallback bug class, gate-change timing, quality
  delta not measured.
- Spend: `tools/spend.py` $151.56 over 72 sandboxes at 11:4xZ (all tracked mx* configs, the GEMM session's mxG* included).

DoD round 3+4 (t022): BASELINE: OpenRelay's published knee (3.0) and their exact recipe re-measured PAIRED on the same 4 C2 boxes
(theirs fails 3.25 on seed 8), plus our previous recipe (C1) on the same boxes as the prior version (`tools/confirm_table.py C2`) |
GAP-HUNT: confirm_table re-derivation of every README C2 number, cpu_cells per cell on all cgmon runs, per-request burst analysis of
the one real-text miss, routing-line count on every stacked server; 4 found (host-wide CPU sampler -> cgmon + memory record;
num_stages-2 compiler abort -> rejected + attn_s2_safety.py + memory record; cpu_cells ISO t_start crash -> fixed a6d66d1; the
README "on every box" claim no longer true for ours-today -> reworded to the stack) | GAPS-FILED: t026 (stacked real text seed 7
at 3.25), t027 (claims regeneration, build session), t028 (stacked docker-run verbatim), t029 (GLM FP8 quality reference)

## 2026-11-22 11:4x-13:5x UTC: C2b and the confirm-ladder guard
- Guard (lead): `tools/mkcfg.py` `check_confirm_ladders` refuses any confirm config in which some server, on some column, lacks a
  rung below its declared expected claim rate. `tools/test_ladder_guard.py` shows that all 4 C2 configs would have been
  refused: or_recipe A, mx_stack B, plus or_recipe B and mx_recipe B.
- C2b (journal 140), every seed, 2 slow boxes:
  - column A: stack 3.5, ours-today 3.25, theirs 3.0;
  - column B: stack 3.0, ours-today 3.0, theirs 2.75.
  Stack A costs $0.000335/request (0.43x sync, 0.86x Batch). Every server now has a claimable rate on both columns.
- journal.py takes a multi-run step's metrics from run_ids[0], so 133 and 140 first recorded one run's rate (3.7 / 3.75). Both
  were re-added with the binding run first (ledger t039 for the build session).
- Process slip: I committed 2fa4f33 through a red gate by piping `commit_gate.sh | tail`. A rerun was ALL GREEN and the
  build session kept the commit. Memory rule `never-pipe-commit-gate`; the build session is adding `bin/gated_commit.sh`.

DoD C2b + guard: BASELINE: C2 (the same three engines, same host type), whose missing rungs this re-run closes |
GAP-HUNT: confirm_table C2b re-derivation, the guard's test against the committed C2 configs, journal rows checked against the
every-seed rule; 2 found (journal run[0] metrics -> t039 + re-added rows; piped gate -> memory rule + gated_commit.sh) |
GAPS-FILED: t039
