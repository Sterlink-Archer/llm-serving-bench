# Dashboard: Gemma 4 31B serving bench

This is a local product dashboard for the benchmark: cost, goodput, latency, runs, spend and the experiment journey. It is a static React app, rebuilt from the repo's own data with one Python command, and it works offline, opened straight from disk.

## Run it

```bash
python3 -I dashboard/build.py              # results/ + infra/ledger.jsonl -> dashboard/site/  (Python only; the UI bundle is prebuilt)
python3 -I dashboard/serve.py --watch 60   # http://127.0.0.1:8765, rebuilding every 60 s while runs land
python3 -I dashboard/check.py              # re-derive every number from the source files, render every route (both themes), measure layout at 1366/446/390 px
```

- **Open it:** run `serve.py`, or open `dashboard/site/index.html` from disk. It needs no network.
- **Themes:** add `?theme=dark` or `?theme=light` to force one; the switch is in the top right.
- **Mock build:** `--demo` on all three commands builds from contract-shaped mock data (`dashboard/demo/make_demo.py`). That build carries a banner and is never quoted.

## Reading it in two minutes

1. **Overview.** It follows the write-up's headline order (`docs/WRITEUP.md`):
   - **H1, ours vs theirs on the same boxes:** our recipe against OpenRelay's on the same MI355X boxes and image. It shows the highest rate every box passes, the achieved gain, real text, and p95 TTFT per box against the 4 s SLO.
   - **H2, our cost vs their API price:** labelled as a different question. It shows Sync, the Batch tier and real text, plus the claims chart.
   - **H3, host sizing:** a Daytona lever, never a recipe comparison.
   - **H4, scope:** the recipes OpenRelay published, and 256K context.
   - Every number comes from `results/claims.json` and `results/protocol_rows.json`. A row that is not claim-grade prints its reasons.
2. **Ours vs theirs.** The same MI355X serves OpenRelay's recipe and ours back to back, so box-to-box clock spread cancels.
   - It shows p95 TTFT at each rate for both recipes, plus prefill, TPOT, MTP acceptance and GSM8K, with the deltas per box.
3. **Journey.** Every change from the baseline, as a step chart, with its bet, result and flag diff. It reads **screening** until our recipe has a claim-grade row.
4. **Cost, Goodput, Latency, Runs, Spend, Startup, MTP, Compare** hold the detail. The **Sources** page lists every file read, with its sha256.

The ⓘ on each card explains it, and the **Method** drawer explains the rules. The link icon on each card lists the run ids and files behind it. Clicking a table row opens that run.

## Pages

| page | what |
|---|---|
| Overview | H1 same-box ours vs theirs (rates, achieved gain, real text, p95 TTFT per box), H2 cost vs their API price (Sync, Batch, real text, claims bars), H3 host sizing, H4 scope chips, then spend, runs, evictions, goodput and the journey |
| Quality | `results/quality_rows.json`: each config vs BF16 on GSM8K and MMLU-Pro as a delta with its 95% interval, greedy agreement next to the self-repeat noise floor, MTP on/off pairs (labelled when on different boxes), every config, gate runs with Wilson intervals |
| Journey | baseline to current best (or "screening"): kept-path step chart, stepper with bet / result / flag diff / claim group / runs, full metric table |
| Ours vs theirs | paired confirm boxes: per box p95 TTFT vs rate for both recipes with the 4 s line, deltas, prefill, TPOT, MTP acceptance, GSM8K; column A/B |
| Cost | claims $/request per group vs OpenRelay Sync, Batch and OpenRouter; every cell at its own shape; dollars by phase; measured $/request; priced cells; break-even rates |
| Goodput | ladders merged by claim group (a rate passes only if every seed passes), p95 TTFT, TPOT, closed-loop knees, client cross-check |
| Latency | TTFT / TPOT / ITL / E2E percentile ladders per cell, by run and matrix |
| Runs, run detail | phase Gantt, wall clock by phase, all runs; the detail page has KPIs, phases, GPU monitor, records, cells, log and files |
| Spend | running spend with the $50–$400 milestones and the ~$500 re-plan line, Daytona metered readings, spot evictions (count, minutes, $ per GPU), every sandbox |
| Startup | observed engine facts per config (attention per layer type, KV, graphs, quantization, speculation) with the stop rules |
| MTP | acceptance length vs concurrency, and where speculation stops paying |
| Compare | up to four configs side by side on the cells they share |
| Sources | every input file with sha256, build warnings, the data contract |

## Guarantees

- **Provenance:** every card cites its runs and files, and every tooltip names the run and file of its point.
- **Claims and gates:**
  - the headline is read from `results/claims.json`: priced on the worst seed's achieved req/s, with claim-grade status and reasons;
  - evicted runs never feed a benchmark number or a $/request;
  - the headline is the cold 9.5K/380 mix.
- **Verification:** `check.py` re-derives every number from the source files:
  - cell percentiles, req/s and prices;
  - claims and `protocol_rows` arithmetic (each row against its own price bar), merged ladder points, paired-box numbers (re-scored from `protocol_rows`, contract v1.8), evictions and spend;
  - the H1 same-box pair, the H3 host lever and the H4 recipe listing;
  - `quality_rows.json` deltas and their 95% intervals, and the gate runs' Wilson intervals.

  It also renders every route in both themes, then measures the layout in real Chrome (`layout_check.mjs`, CDP device metrics) at 1366, 446 and 390 px. It fails on:
  - console errors;
  - NaN or undefined text;
  - missing citations or numbers;
  - an undrawn bar;
  - a headline that is not claim-grade but hides its reasons;
  - horizontal page overflow, or a chart drawn narrower than 150 px;
  - two chart labels whose boxes intersect, or a label cut off at its chart's edge;
  - a plot area under 40% of its card's width;
  - a bar or bar segment under 2 px (stack segments that small are folded into a grey "short phases" segment; exact values stay in the tooltip);
  - a clipped KPI value, or a KPI number split across two lines.
- **Colour:** GPU = fixed hue (dataviz palette, validated order); config = marker shape; OpenRelay's recipe and published numbers are drawn in neutral ink.

## Stack (app/)

React 19.3.1, Vite 7.3.7, TypeScript 5.9.3, TanStack Router 1.170.46 (hash history) and Table 8.21.3, Tailwind 4.3.5, Recharts 3.10.1, Radix primitives, lucide.

- Every dependency is pinned exactly, with `package-lock.json` committed.
- Install scripts are off (`.npmrc`). `npm audit` reports 0 vulnerabilities.
- Each version was the newest release at least 14 days old in its line.
- Vite emits one classic script (not an ES module) so the site opens from `file://`.

To change the UI:

```bash
cd dashboard/app && npm ci && npm run typecheck && npm run build   # writes dashboard/ui/ (committed)
```

## Data contract

The dashboard reads only what [docs/DATA_CONTRACT.md](../docs/DATA_CONTRACT.md) lists (owned by the build session) and writes only inside `dashboard/`.
