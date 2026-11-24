import { useNavigate } from '@tanstack/react-router';
import { B, K, cfgShort } from '../lib/data';
import { isNum, num, pct } from '../lib/format';
import { gpuColor, useTokens } from '../lib/theme';
import { IntervalPlot, Legend, TipBox, type IRow } from '../components/charts';
import { Card, Empty, PageHead, Pill, RunLink, Sources, runParam } from '../components/ui';
import { DataTable, numCol, type Col } from '../components/DataTable';

/* eslint-disable @typescript-eslint/no-explicit-any */
// results/quality_rows.json (schema quality/1, analysis/quality.py): every served config vs the BF16 original. A delta is never
// shown without its 95% interval, and agreement with the reference is shown next to the noise floor (a config repeated
// against itself), as the build session asked. The gate table below is the per-run GSM8K gate (contract section 7).

/** signed percentage points: +0.8 pp, −1.0 pp */
export const pp = (x: number | null | undefined, d = 1) => (isNum(x) ? (x > 0 ? '+' : x < 0 ? '−' : '±') + Math.abs(x * 100).toFixed(d) + ' pp' : '–');
const ci = (c: any) => (c && isNum(c[0]) ? `${pct(c[0])}–${pct(c[1])}` : '–');
const dci = (c: any) => (c && isNum(c[0]) ? `${pp(c[0])} to ${pp(c[1])}` : '–');
const TASKS = [['gsm8kor250', 'GSM8K (250)'], ['mmlupro200', 'MMLU-Pro (200)']] as const;
const runIds = (c: any) => String(c.run_id ?? '').split(',').map((s: string) => s.trim()).filter(Boolean);
const label = (c: any) => `${c.tag} · ${c.gpu}`;
/** files: a {task: path} map in quality/1 (any shape tolerated) */
const filesOf = (c: any): string[] => (Array.isArray(c?.files) ? c.files : c?.files && typeof c.files === 'object' ? Object.values(c.files) : c?.files ? [c.files] : []).filter(Boolean).map(String);

function Deltas({ q, task, title }: { q: any; task: string; title: string }) {
  const t = useTokens();
  const ref = q.configs.find((c: any) => c.is_reference);
  const cs = q.configs.filter((c: any) => !c.is_reference && c[task]);
  const lo = Math.min(0, ...cs.map((c: any) => c[task].delta_ci95?.[0] ?? c[task].delta_vs_ref)) * 100 - 0.6;
  const hi = Math.max(0, ...cs.map((c: any) => c[task].delta_ci95?.[1] ?? c[task].delta_vs_ref)) * 100 + 0.6;
  const rows: IRow[] = cs.map((c: any) => {
    const r = c[task], col = gpuColor(t, c.gpu);
    return {
      key: c.config, label: label(c),
      spans: r.delta_ci95 ? [{ lo: r.delta_ci95[0] * 100, hi: r.delta_ci95[1] * 100, color: col }] : [],
      marks: [{ v: r.delta_vs_ref * 100, color: col, text: pp(r.delta_vs_ref), side: 'right' }],
      tip: <TipBox title={c.config} rows={[{ name: 'Δ vs BF16', value: pp(r.delta_vs_ref) }, { name: '95% interval', value: dci(r.delta_ci95) }, { name: 'accuracy', value: `${pct(r.acc)} (${ci(r.ci95)})` },
        { name: 'BF16', value: pct(ref?.[task]?.acc) }, { name: 'lost / gained', value: `${r.b_lost ?? '–'} / ${r.c_gained ?? '–'}` }, { name: 'McNemar p', value: isNum(r.mcnemar_p) ? num(r.mcnemar_p, 3) : '–' },
        { name: 'n', value: num(r.n) }, { name: 'weights · KV · MTP', value: `${c.weights} · ${c.kv_cache} · ${c.mtp ? 'k=' + c.mtp : 'off'}` }]} foot={filesOf(c).join(' · ') || c.run_id} />,
    };
  });
  return (
    <Card title={`${title} vs BF16`} info={<>Paired difference against the BF16 original ({ref ? label(ref) : 'reference'}, {pct(ref?.[task]?.acc)}) on the same items, with its 95% interval. An interval that spans zero is no measurable change.</>}
      sources={<Sources runs={cs.flatMap(runIds)} files={[q.file, q.md_file, ...cs.flatMap(filesOf)]} />}>
      {!rows.length ? <Empty>No {title} rows yet.</Empty> : (
        <>
          <Legend items={[...new Set<string>(cs.map((c: any) => String(c.gpu)))].map((g: string) => ({ label: g, color: gpuColor(t, g) }))} />
          <IntervalPlot t={t} rows={rows} domain={[lo, hi]} fmt={v => `${v > 0 ? '+' : ''}${num(v, Math.abs(v) < 1 && v !== 0 ? 1 : 0)} pp`} refs={[{ v: 0, label: 'BF16', color: t.ref }]} ariaLabel={`${title} delta vs BF16 with 95% intervals`} />
        </>
      )}
    </Card>
  );
}

function Agreement({ q }: { q: any }) {
  const t = useTokens();
  const ref = q.configs.find((c: any) => c.is_reference);
  const floor = ref?.agree_self_repeat?.exact_match;
  const cs = q.configs.filter((c: any) => c.agree_vs_ref?.exact_match != null || c.agree_self_repeat?.exact_match != null);
  const rows: IRow[] = cs.map((c: any) => {
    const a = c.agree_vs_ref, s = c.agree_self_repeat, col = c.is_reference ? t.ref : gpuColor(t, c.gpu);
    const spans: NonNullable<IRow['spans']> = [], marks: IRow['marks'] = [];
    if (a?.exact_match != null) {
      if (a.exact_ci95) spans.push({ lo: a.exact_ci95[0] * 100, hi: a.exact_ci95[1] * 100, color: col, dy: -5 });
      marks.push({ v: a.exact_match * 100, color: col, dy: -5, text: pct(a.exact_match, 0), side: 'right' });
    }
    if (s?.exact_match != null) {
      if (s.exact_ci95) spans.push({ lo: s.exact_ci95[0] * 100, hi: s.exact_ci95[1] * 100, color: col, dash: '3 3', dy: 6 });
      marks.push({ v: s.exact_match * 100, color: col, hollow: true, glyph: 'diamond', dy: 6, text: pct(s.exact_match, 0), side: 'right' });
    }
    return {
      key: c.config, label: c.is_reference ? `${c.tag} (reference)` : label(c), spans, marks,
      tip: <TipBox title={c.config} rows={[a?.exact_match != null && { name: 'identical to BF16', value: `${pct(a.exact_match)} (${ci(a.exact_ci95)})` }, a?.mean_prefix_frac != null && { name: 'shared prefix', value: pct(a.mean_prefix_frac) },
        s?.exact_match != null && { name: 'identical to its own repeat', value: `${pct(s.exact_match)} (${ci(s.exact_ci95)})` }, { name: 'prompts', value: num(a?.n ?? s?.n) }]} foot={filesOf(c).join(' · ') || c.run_id} />,
    };
  });
  return (
    <Card title="Greedy outputs identical to BF16" info={<>Share of 400 prompts whose greedy output is identical to the BF16 original's (circle), next to the share identical when the same server answers twice (diamond, the noise floor from batching nondeterminism). Prefix share in the tooltip.</>}
      sources={<Sources runs={cs.flatMap(runIds)} files={[q.file]} />}>
      {!rows.length ? <Empty>No agreement runs yet.</Empty> : (
        <>
          <Legend items={[{ label: 'vs BF16', color: t.ink2 }, { label: 'vs its own repeat', color: t.ink2, glyph: 'diamond' }]} />
          <IntervalPlot t={t} rows={rows} rowH={36} domain={[0, 100]} fmt={v => `${num(v)}%`} ariaLabel="greedy agreement vs BF16 and self-repeat"
            refs={isNum(floor) ? [{ v: floor * 100, label: `BF16 repeat ${pct(floor, 0)}`, color: t.ref, dash: '3 3' }] : []} />
        </>
      )}
    </Card>
  );
}

function MtpPairs({ q }: { q: any }) {
  const pairs = q.mtp_lossless ?? [];
  if (!pairs.length) return null;
  return (
    <Card title="MTP on vs off" info="Greedy outputs of the same weights with speculative decoding on and off. Lossless MTP gives identical outputs up to the noise floor."
      sources={<Sources files={[q.file, ...pairs.flatMap((m: any) => m.agree_files ?? [])]} />}>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {pairs.map((m: any) => (
          <div key={m.mtp_on + m.mtp_off} className="min-w-0">
            <div data-kpi-value="" className="text-[26px] font-semibold leading-[1.15] tracking-[-0.02em]">{pct(m.agree?.exact_match)}</div>
            <div className="mt-1 text-[12px] leading-snug text-muted">identical greedy outputs ({ci(m.agree?.exact_ci95)}), shared prefix {pct(m.agree?.mean_prefix_frac)}{isNum(m.gsm_same_answer) ? `, GSM8K same answer ${pct(m.gsm_same_answer)}` : ''}</div>
            <div className="mt-1 font-mono text-[11px] text-muted [overflow-wrap:anywhere]">{m.mtp_on} vs {m.mtp_off}</div>
            <div className="mt-2 flex flex-wrap gap-1">{m.same_box ? <Pill tone="good">same box</Pill> : <Pill tone="warn">different boxes</Pill>}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

export function Quality() {
  const nav = useNavigate();
  const q = (B as any).quality;
  const ref = q?.configs?.find((c: any) => c.is_reference);
  // per-run GSM8K gate: one row per (config, task), the largest n
  const ev = (B.evals ?? []).filter((e: any) => (e.task ?? '').startsWith('gsm8k') && Number.isFinite(e.acc));
  const best: Record<string, any> = {};
  for (const e of ev) for (const cid of e.config_ids?.length ? e.config_ids : [e.run_id]) {
    const k = `${cid}|${e.task}`;
    if (!best[k] || (e.n ?? 0) > (best[k].n ?? 0)) best[k] = { ...e, config_id: cid };
  }
  const gates = Object.values(best).sort((a: any, b: any) => b.acc - a.acc);
  return (
    <div>
      <PageHead title="Quality" actions={ref ? <Pill>reference: {ref.weights} on {ref.gpu}</Pill> : null} />
      {!q ? <Card><Empty>Awaiting results/quality_rows.json (analysis/quality.py).</Empty></Card> : (
        <>
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            {TASKS.map(([k, title]) => <Deltas key={k} q={q} task={k} title={title} />)}
          </div>
          <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
            <Agreement q={q} />
            <MtpPairs q={q} />
          </div>
          <Card className="mt-4" title="Every config" sources={<Sources runs={q.configs.flatMap(runIds)} files={[q.file, q.md_file]} />}>
            <DataTable data={q.configs} search={false} columns={[
              { id: 'config', header: 'Config', accessorFn: (c: any) => c.tag + (c.is_reference ? ' (ref)' : '') },
              { id: 'gpu', header: 'GPU', accessorFn: (c: any) => c.gpu },
              { id: 'weights', header: 'Weights', accessorFn: (c: any) => c.weights, meta: { wrap: true } },
              { id: 'kv', header: 'KV', accessorFn: (c: any) => c.kv_cache },
              { id: 'mtp', header: 'MTP', accessorFn: (c: any) => (c.mtp ? `k=${c.mtp}` : 'off') },
              numCol<any>('gsm', 'GSM8K', c => c.gsm8kor250?.acc, v => pct(v), { cell: (x: any) => <span>{pct(x.row.original.gsm8kor250?.acc)} <span className="text-muted">({ci(x.row.original.gsm8kor250?.ci95)})</span></span> }),
              numCol<any>('dgsm', 'Δ GSM8K', c => c.gsm8kor250?.delta_vs_ref, v => pp(v), { cell: (x: any) => <span>{pp(x.row.original.gsm8kor250?.delta_vs_ref)} <span className="text-muted">({dci(x.row.original.gsm8kor250?.delta_ci95)})</span></span> }),
              numCol<any>('mmlu', 'MMLU-Pro', c => c.mmlupro200?.acc, v => pct(v), { cell: (x: any) => <span>{pct(x.row.original.mmlupro200?.acc)} <span className="text-muted">({ci(x.row.original.mmlupro200?.ci95)})</span></span> }),
              numCol<any>('dmmlu', 'Δ MMLU-Pro', c => c.mmlupro200?.delta_vs_ref, v => pp(v), { cell: (x: any) => <span>{pp(x.row.original.mmlupro200?.delta_vs_ref)} <span className="text-muted">({dci(x.row.original.mmlupro200?.delta_ci95)})</span></span> }),
              numCol<any>('pm', 'p (MMLU)', c => c.mmlupro200?.mcnemar_p, v => (isNum(v) ? num(v, 3) : '–'), { meta: { hide: true } }),
              numCol<any>('agree', 'Same as BF16', c => c.agree_vs_ref?.exact_match, v => pct(v)),
              numCol<any>('self', 'Self-repeat', c => c.agree_self_repeat?.exact_match, v => pct(v)),
              { id: 'run', header: 'Runs', accessorFn: (c: any) => c.run_id, cell: (x: any) => <div className="flex flex-col">{runIds(x.row.original).map((r: string) => <RunLink key={r} id={r} />)}</div> },
            ] as Col<any>[]} />
          </Card>
        </>
      )}
      <Card className="mt-4" title={`Gate runs · GSM8K ≥ ${pct(K.gsm8k_gate)}`} info="The per-run quality gate a price needs (contract section 7): largest-n GSM8K run per config, with its Wilson 95% interval."
        sources={<Sources runs={gates.map((r: any) => r.run_id)} files={gates.map((r: any) => r.file)} />}>
        {!gates.length ? <Empty>No GSM8K gate or eval file yet.</Empty> : (
          <DataTable data={gates} onRowClick={(r: any) => nav({ to: '/run/$runId', params: { runId: runParam(r.run_id) } })} initialSort={[{ id: 'acc', desc: true }]} columns={[
            { id: 'config', header: 'Config', accessorFn: (r: any) => cfgShort(r.config_id) },
            { id: 'task', header: 'Task', accessorFn: (r: any) => r.task, meta: { mono: true } },
            numCol<any>('n', 'n', r => r.n, v => num(v)),
            numCol<any>('acc', 'GSM8K', r => r.acc, v => pct(v), { cell: (x: any) => <span>{pct(x.row.original.acc)} <span className="text-muted">({ci(x.row.original.ci95)})</span></span> }),
            { id: 'gate', header: 'Gate', accessorFn: (r: any) => (r.acc >= K.gsm8k_gate ? 1 : 0), cell: c => (c.row.original.acc >= K.gsm8k_gate ? <Pill tone="good">pass</Pill> : <Pill tone="crit">below</Pill>) },
            { id: 'run', header: 'Run', accessorFn: (r: any) => r.run_id, cell: c => <RunLink id={c.getValue() as string} /> },
          ] as Col<any>[]} />
        )}
      </Card>
    </div>
  );
}
