# WORKLOG: Inkling-2-Mini recipe (phase-2 session, 2026-11-22)

- 07:0x-07:10Z: read the plan, data contract, model card (HF @b85c3cf), engine recipe pages (vLLM, SGLang, TokenSpeed).
  Bar: PrimeIntellect $0.075 / $0.30 = $0.0008265 per 9.5K/380 request. Launched the 6-box screen (ik1).
- 07:15Z: Maks's "compare to the original": no full-precision Inkling-2-Mini exists; Cloudflare serves only inkling-2-small;
  OpenRouter 402 (`raw/reference_check_*.json`). Reference = the vendor recipe on the official checkpoint.
- 07:30-07:50Z: screen. SGLang beats vLLM (KV 1.25M vs 142K tokens); every card passes column A at 1.0-1.5x but only MI355X
  passes column B at 1.0x; the SWA pool fills first on NVIDIA, MI355X is compute-bound (steps 200-205).
- 07:55Z: bench/traffic.py crashed on the B prefill shot (Inkling's swallowed first token): six boxes stopped and relaunched
  ($1.40); fixed upstream with --ttft-basis token (step 215).
- 08:00-09:25Z: levers, one per step, paired on the same box: SWA ratio (neutral), mixed-chunk (tail halved), 16K chunks
  (neutral), DSpark (TPOT halved), FP8 KV (impossible on sm120), the 5090's KV budget (parity at best), right-sized host
  (4 vCPU / 24 GiB, -11% $/h, no slowdown), mixed-chunk + DSpark (passes 1.25x). MI355X AITER tuning: three attempts; with the
  GPU free the defaults are already 1.25-1.37 PFLOPS on the projections (~11% of prefill): closed; profile shows emulated NVFP4
  MoE (~37%) and the controller kernel (12%) (steps 206-222).
- 09:11-10:29Z: claim-grade confirms (300 s, seeds 7 + 8) + the quality reference twice: PRO 6000 mixed-chunk + DSpark passes
  1.5x on both seeds (0.689x) with seed 7 failing 1.35x; mixed-chunk alone 1.25x (0.816x). Quality gates pass vs both reference
  runs (steps 223-225).
- 10:37-10:57Z: replicate on two fresh boxes: seed 7's 1.35x fail reproduces (4.07 s); headline stays the monotonic 1.25x =
  0.83x (step 226).
- 10:07-11:51Z: B300 claim run (queued on the build session's launcher): 1.5x on both seeds, 1.75x fails on both (same-box
  extension via tools/extend_live.py); 0.656x, monotonic (steps 227-228). Sandbox deleted 11:50:49Z.
- Spend for this recipe: $45.29 over 32 sandboxes (`tools/spend.py`, 12:0xZ); every sandbox has a ledger delete event.

DoD Inkling phase 2 (t006): BASELINE: the written brief (method, GPUs, engines, journal, quality gate, bar re-check) AND
PrimeIntellect's live price re-checked 10:59Z (`raw/or_endpoints_inkling-2-mini_20261122T105921Z.json`), with claims read from
`results/claims.json` (analysis/screen_report.py, independent of this recipe's tools) | GAP-HUNT: brief-vs-delivered diff + claims
assertion (column B claim-grade and < 1.0x for ikC_b300_mixed_dspark 0.656, ikC_pro6000_mixed_dspark 0.689,
ikC_pro6000_mixed 0.816) + the knee replicate (step 226); 6 found -> t030 (no NVIDIA kernel profile), t031 (DSpark acceptance
missing from claim rows), t032 (OpenRouter reference rows blocked on credits, Maks), t033 (MI355X native FP4 MoE + controller
kernel), t034 (quality noise floor), t035 (fold into docs/WRITEUP.md, lead) | GAPS-FILED: t030-t035 | GUARDS:
tools/test_agree.py (per-task gate regression), tools/extend_live.py + memory daytona-sandbox-autostop-pause, upstream
cf9e936 (traffic --ttft-basis token + test_traffic_notext, SGLang verify-graph parse) and analysis/runfiles.py (unclaimed protocol
dirs warned, not dropped).
Retro: `agent-comm decision propose` is not installed on this machine (`command -v agent-comm`: absent), so the lesson went to
the session memory instead (inkling-2-mini-serving-facts, daytona-sandbox-autostop-pause): "on a new looped MoE, start from
SGLang + mixed-chunk + DSpark on a right-sized host and judge every lever on real text (column B); random tokens hide both
the MoE/attention tail and the speculation win".
