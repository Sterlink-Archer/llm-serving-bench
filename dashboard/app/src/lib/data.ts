// Typed access to window.BENCH (written by dashboard/build.py from the files in docs/DATA_CONTRACT.md).
/* eslint-disable @typescript-eslint/no-explicit-any */
export type Dist = { n: number; mean: number; min: number; max: number; p50: number; p90: number; p95: number; p99: number } | null;

export interface Cell {
  run_id: string; config_id: string; gpu_key: string | null; matrix: string; cell: string; cell_id: string; file: string; t_start: string | null;
  workload: string; mode: 'open' | 'closed'; C: number | null; rate: number | null; params: Record<string, any>;
  requests_sent: number; completed: number; errors: number; err_frac: number; dropped: number; peak_inflight: number;
  in_mean: number; cached_mean: number; out_per_req: number; req_s: number; req_s_completed: number; out_tok_s: number;
  srv_out_tok_s: number | null; spec_accept_rate: number | null; spec_accept_len: number | null; client_vs_server_out?: number;
  ttft: Dist; tpot: Dist; itl: Dist; e2e: Dist;
  usd_h: number | null; our_usd_req: number | null; bar_sync: number | null; bar_batch: number | null; bar_openrouter: number | null;
  ratio_sync: number | null; ratio_sync_cached: number | null; ratio_batch: number | null; ratio_openrouter: number | null; priced: boolean;
  agentic_shape: boolean; slo_openrelay: boolean; slo_interactive: boolean; protocol: null | 'kept' | 'other_seed';
}
export interface Phase { phase: string; cat: string; label: string; t_start: string; t_end: string; dur_s: number; billed?: boolean; usd: number;
  nested_in?: string | null; detail?: Record<string, any>; source: string; cell_id?: string | null }
export interface Run {
  run_id: string; label: string; stamp: string; config_ids: string[]; gpu_key: string | null; sandbox_id: string | null; status: string;
  t_start: string | null; t_end: string | null; elapsed_s: number | null; est_cost_usd: number | null; usd_per_h: Record<string, number> | null;
  spot_capacity_at_start: number | null; error: string | null; evicted: any; phases: Phase[]; phase_source: string; n_cells: number; files: string[];
  driver_log: string[]; gpumon: { columns: string[]; rows: (number | null)[][]; note: string; file: string } | null; run_json: string | null; servers: Record<string, any>;
}
export interface Config { config_id: string; label: string; server: string; config_file: string; gpu: string; gpus: number; gpu_key: string; cpu: number; mem: number;
  image: string; weights_repo: string; launch: string; declared: Record<string, any>; usd_per_h: Record<string, number> | null; symbol: string; symbol_index: number }
export interface Best { config_id: string; gpu_key: string; run_id: string; cell_id: string; file: string; our_usd_req: number; req_s: number; ttft_p95: number;
  tpot_p50: number; ratio_sync: number; gate: any; stop: string[]; eligible: boolean; protocol: any;
  ratio_sync_metered?: number | null; usd_per_req_metered?: number | null; metered_read_at?: string | null }
export interface Claim { group: string; column: 'A' | 'B'; column_name: string; rates_tested: number[]; runs: string[]; max_rate_all_seeds_pass: number | null;
  seeds?: string[]; worst_seed_achieved_rps?: number; worst_seed_ttft_p95_s?: number; usd_per_h?: number; usd_per_req?: number; ratio_vs_openrelay_sync?: number;
  usd_per_h_metered?: number | null; usd_per_req_metered?: number | null; ratio_vs_openrelay_sync_metered?: number | null; claim_grade: boolean; not_claim_grade_because?: string[];
  gpu_key: string | null; config_ids: string[]; origin: string; gpu?: string; engine?: string;
  bar_name: string | null; bar_short: string | null; bar_usd: number | null; ratio: number | null; ratio_metered: number | null }
export type Column = 'A' | 'B';
export const COLUMN_NAME: Record<Column, string> = { A: "OpenRelay's client, random tokens", B: 'our client, real text' };
/** price bars present in the claims (contract v1.7); the OpenRelay bar first */
export const claimBars = (): string[] => [...new Set((B?.claims?.claims ?? []).filter(r => r.usd_per_req != null && r.bar_short).map(r => r.bar_short as string))]
  .sort((a, b) => Number(!/openrelay/i.test(a)) - Number(!/openrelay/i.test(b)) || a.localeCompare(b));
/** priced claim rows for a column (and bar), cheapest first */
export const claimsFor = (col: Column, bar?: string): Claim[] => (B?.claims?.claims ?? []).filter(r => r.column === col && r.usd_per_req != null && (!bar || r.bar_short === bar))
  .sort((a, b) => (a.usd_per_req as number) - (b.usd_per_req as number));
/** the headline row of a column: claim-grade first, then more seeds, then cheaper (a 1-seed row never outranks a 2-seed row) */
export const headlineFor = (col: Column, bar?: string): Claim | undefined => claimsFor(col, bar ?? claimBars()[0]).slice().sort((a, b) =>
  (Number(b.claim_grade) - Number(a.claim_grade)) || ((b.seeds ?? []).length - (a.seeds ?? []).length) || ((a.usd_per_req as number) - (b.usd_per_req as number)))[0];
export const originLabel = (c: Claim | undefined) => (!c ? '' : c.origin === 'openrelay_recipe' ? "OpenRelay's recipe, reproduced" : String(c.origin).startsWith('mixed:') ? 'mixed: theirs + ours' : 'our recipe');
export interface Experiment { schema?: string; step: number; t?: string; kind?: string; config_id: string; parent_config_id?: string | null; change: string; why?: string;
  run_ids: string[]; basis?: string; slo_goodput_rps: number | null; usd_per_req: number | null; ratio_sync: number | null; ttft_p95_s?: number | null;
  model?: string; bar_name?: string; bar_short?: string | null; bar_usd?: number | null; ratio_bar?: number | null; bar_source?: string | null;
  tpot_p50_s?: number | null; gsm8k_acc?: number | null; decision: 'baseline' | 'kept' | 'reverted' | 'rejected' | 'info'; evidence_files?: string[]; _line?: number;
  hypothesis?: string; parent_step?: number | null; max_rate_at_slo_rps?: number | null; breakeven_rps?: number | null; achieved_rps_at_breakeven?: number | null;
  usd_per_req_at_achieved?: number | null; prefill_tok_s?: number | null; kv_tokens?: number | null; mtp_accept_len?: number | null; usd_per_h?: number | null;
  run_usd_est?: number | null; diff?: Record<string, any>; delta_vs_parent?: Record<string, [any, any]>; note?: string | null; claim_groups?: string[] }

export interface Bench {
  build: { t: string; demo: boolean; contract_version: string; contract_file?: string; cost_rows_file?: string | null; results_dir: string };
  constants: { shape_in: number; shape_out: number; shape_tol: number; slo_ttft_p95_s: number; max_err_frac: number; gsm8k_gate: number; gate_min_n: number;
    milestones: number[]; replan_usd: number; gpu_order: string[]; phase_cats: string[] };
  bars: { per_request: { sync: number; sync_all_cached: number; batch: number; openrouter: number }; openrelay: any; openrouter: any; shape: any };
  refs: { openrelay_mi355x: any };
  gpus: Record<string, { slot: number | null; spot_usd_h: number | null }>;
  configs: Record<string, Config>;
  runs: Run[]; cells: Cell[]; evals: any[]; startup: any[]; protocols: any[]; measured: any[]; xchecks: any[]; best: Best[]; gate: Record<string, any>;
  spend: { points: { t: string; usd: number; live: number }[]; sandboxes: any[]; milestones: { usd: number; crossed: string | null; kind: string }[]; total_est: number;
    metered: any[]; reported: any[]; ledger_file: string };
  journey: { steps: Experiment[]; file: string | null; problems: string[]; superseded?: number };
  protocol_rows: any[];
  claims: { schema: string; openrelay_sync_usd_per_req: number; rules: string; claims: Claim[]; file: string; md_file?: string } | null;
  evictions: { events: any[]; by_gpu: { gpu: string; count: number; minutes_lost: number; usd_lost: number }[]; total_count: number; total_minutes: number; total_usd: number; note: string };
  ledger_counts: Record<string, number>; inputs: { path: string; bytes: number; sha256: string; mtime: string }[]; warnings: string[]; contract: string | null;
}

declare global { interface Window { BENCH: Bench } }
export const B: Bench = window.BENCH;
export const K = B?.constants;
export const BARS = B?.bars?.per_request;
/** this benchmark's model slug (price bar names: openrelay/gemma-4-31b); other models get their own journey */
export const BENCH_MODEL: string = (B?.build as any)?.bench_model ?? 'gemma-4-31b';
/** models in the experiments log, this benchmark's first: one journey per model, never one path across models */
export const journeyModels = (): string[] => [...new Set((B?.journey?.steps ?? []).map((s: any) => (s.model ?? BENCH_MODEL) as string))]
  .sort((a, b) => Number(a !== BENCH_MODEL) - Number(b !== BENCH_MODEL) || a.localeCompare(b));
export const stepsOf = (model?: string) => (B?.journey?.steps ?? []).filter((s: any) => (s.model ?? BENCH_MODEL) === (model ?? journeyModels()[0] ?? BENCH_MODEL)).slice().sort((a: any, b: any) => a.step - b.step);
export const REF = B?.refs?.openrelay_mi355x ?? {};
export const CELLS: Cell[] = B?.cells ?? [];
/** protocol ladders count only the kept (worse) seed (PLAN fairness rule; contract section 5) */
export const CELLS_K: Cell[] = CELLS.filter(c => c.protocol !== 'other_seed');
export const RUNS: Run[] = B?.runs ?? [];
export const CFG: Record<string, Config> = B?.configs ?? {};

export const cfgOf = (id: string | null | undefined): Partial<Config> => (id && CFG[id]) || {};
export const cfgShort = (id: string | null | undefined) => {
  const c = cfgOf(id);
  if (!c.label) return id ?? '–';
  return `${c.label.split('_').slice(0, 2).join('_')} · ${c.server}`;
};
export const runShort = (r: string | null | undefined) => {
  if (!r) return '';
  const m = r.match(/^([^/]+)\/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})\d{2}Z$/);
  return m ? `${m[1].split('_').slice(0, 2).join('_')} · ${m[5]}:${m[6]}Z` : r;
};
export const seriesOf = (c: Cell) => (c.protocol ? 'ladder' : String(c.cell).split('__')[0].replace(/_r[\d.]+(_s\d+)?$/, ''));
export const loadOf = (c: Cell) => (c.mode === 'open' ? `${c.rate} req/s` : `C=${c.C}`);
export const runOf = (id: string) => RUNS.find(r => r.run_id === id);
export const cellsOfRun = (id: string) => CELLS.filter(c => c.run_id === id);
export const gateOf = (cid: string) => (B?.gate ?? {})[cid];
export function groupBy<T>(xs: T[], f: (x: T) => string): Record<string, T[]> {
  const o: Record<string, T[]> = {};
  for (const x of xs) (o[f(x)] ||= []).push(x);
  return o;
}
export const uniq = <T,>(xs: (T | null | undefined)[]) => [...new Set(xs.filter((x): x is T => x != null && (x as any) !== ''))];
export const phasesOf = (r: Run) => (r.phases || []).filter(p => Number.isFinite(p.dur_s));
export function dollarsByCat(r: Run) {
  const out: Record<string, number> = Object.fromEntries((K?.phase_cats ?? []).map(c => [c, 0]));
  for (const p of phasesOf(r)) if (!p.nested_in && p.billed !== false) out[p.cat] = (out[p.cat] || 0) + (p.usd || 0);
  return out;
}
export function secsByCat(r: Run) {
  const out: Record<string, number> = Object.fromEntries((K?.phase_cats ?? []).map(c => [c, 0]));
  for (const p of phasesOf(r)) if (!p.nested_in) out[p.cat] = (out[p.cat] || 0) + (p.dur_s || 0);
  return out;
}
export const evictions = () => B?.evictions?.total_count ?? RUNS.filter(r => r.status === 'evicted' || (r as any).excluded === 'evicted').length;
