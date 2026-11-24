import { useState } from 'react';
import { B, BENCH_MODEL, REF, cfgShort, headlineFor, journeyModels, stepsOf, type Experiment } from '../lib/data';
import { compact, isNum, num, pct, ratio, rps, secs, usd, utc } from '../lib/format';
import { useTokens } from '../lib/theme';
import { Card, Empty, PageHead, Pill, RunLink, Segmented, Sources } from '../components/ui';
import { JourneyChart, JourneyLegend, decisionTone, journeyMetrics, type JMetric } from '../components/widgets';
import { DataTable, numCol, type Col } from '../components/DataTable';
import { useNavigate } from '@tanstack/react-router';
import { runParam } from '../components/ui';

/* eslint-disable @typescript-eslint/no-explicit-any */
const fmtDelta = (k: string, v: any) => (v == null ? '—' : k.includes('usd') ? usd(v) : k.includes('ttft') || k.includes('tpot') ? secs(v) : k.includes('tok') || k.includes('kv') ? compact(v) : k.includes('rps') ? rps(v) : typeof v === 'number' ? num(v, v < 10 ? 2 : 0) : String(v));

function Stepper({ steps }: { steps: Experiment[] }) {
  const t = useTokens();
  const dot = (d: string) => (d === 'kept' ? t.s1 : d === 'baseline' ? t.ink2 : d === 'info' ? t.muted : t.s8);
  return (
    <ol className="relative">
      {steps.map((s, i) => {
        const flags = { ...(s.diff?.flags_added_or_changed ?? {}) };
        const other = Object.entries(s.diff ?? {}).filter(([k]) => !k.startsWith('flags')) as [string, any][];
        const removed = Object.keys(s.diff?.flags_removed ?? {});
        return (
          <li key={s.step} className="relative flex gap-4 pb-5 last:pb-0">
            {i < steps.length - 1 && <span className="absolute left-[9px] top-6 bottom-0 w-px bg-line-2" aria-hidden />}
            <span className="relative mt-1 size-[19px] shrink-0 rounded-full border-[3px]" style={{ borderColor: dot(s.decision), background: s.decision === 'kept' || s.decision === 'baseline' ? dot(s.decision) : 'var(--panel)' }} aria-hidden />
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-[11.5px] text-muted">#{s.step}</span>
                <span className="text-[13.5px] font-semibold text-ink">{s.change}</span>
                <Pill tone={decisionTone(s.decision)}>{s.decision}</Pill>
                {s.config_id && <span className="text-[12px] text-muted">{cfgShort(s.config_id)}</span>}
                {(s.claim_groups ?? []).map(g => <span key={g} className="rounded-md border border-line px-1.5 py-0.5 font-mono text-[10.5px] text-ink-2" title="claim group (protocol_rows.json)">{g}</span>)}
              </div>
              {s.hypothesis && <p className="mt-1 text-[12.5px] text-ink-2"><span className="text-muted">Bet · </span>{s.hypothesis}</p>}
              {s.why && <p className="mt-0.5 text-[12.5px] text-ink-2"><span className="text-muted">Result · </span>{s.why}</p>}
              <div className="mt-2 flex flex-wrap gap-1.5">
                {isNum(s.max_rate_at_slo_rps ?? s.slo_goodput_rps) && <Pill tone={(s.max_rate_at_slo_rps ?? s.slo_goodput_rps)! > 0 ? 'good' : 'crit'}>goodput {rps(s.max_rate_at_slo_rps ?? s.slo_goodput_rps)} req/s</Pill>}
                {isNum(s.usd_per_req) && <Pill tone={isNum(s.bar_usd) && s.usd_per_req < s.bar_usd ? 'good' : 'warn'} title={s.bar_name ? `vs ${s.bar_name} ${usd(s.bar_usd)}/req${s.bar_source ? ' · ' + s.bar_source : ''}` : undefined}>{usd(s.usd_per_req)}/req · {ratio(s.ratio_bar ?? s.ratio_sync)} {s.bar_short ?? 'OpenRelay sync'}</Pill>}
                {!isNum(s.usd_per_req) && isNum(s.usd_per_req_at_achieved) && <Pill>{usd(s.usd_per_req_at_achieved)}/req at achieved</Pill>}
                {isNum(s.ttft_p95_s) && <Pill tone={s.ttft_p95_s < 4 ? 'good' : 'crit'}>TTFT p95 {secs(s.ttft_p95_s)}</Pill>}
                {isNum(s.gsm8k_acc) && <Pill>GSM8K {pct(s.gsm8k_acc)}</Pill>}
                {Object.entries(flags).map(([k, v]) => <span key={k} className="rounded-md bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-ink-2">{k}{v === true ? '' : `=${v}`}</span>)}
                {removed.map(k => <span key={k} className="rounded-md bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-muted line-through">{k}</span>)}
                {other.map(([k, v]) => <span key={k} className="rounded-md bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-ink-2">{k}: {Array.isArray(v) ? `${v[0]} → ${v[1]}` : JSON.stringify(v)}</span>)}
              </div>
              {(s.run_ids ?? []).length > 0 && <div className="mt-1.5 flex flex-wrap gap-x-3">{s.run_ids.map(r => <RunLink key={r} id={r} />)}</div>}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

export function Journey() {
  const nav = useNavigate();
  // the recipe the journey ends at is the headline's (H1 same-box group), never the cheapest row: a smaller host is a Daytona
  // lever (H3), not a step of the recipe
  const sb = (B as any).same_box?.[0];
  const ourClaim = sb ? (B.claims?.claims ?? []).find(r => r.group === sb.ours && r.column === 'A' && r.claim_grade)
    : [headlineFor('A'), headlineFor('B')].find(r => r && r.origin === 'ours' && r.claim_grade);
  const models = journeyModels();
  const [model, setModel] = useState<string>(models[0] ?? BENCH_MODEL);
  const steps = stepsOf(model) as Experiment[];
  const bench = model === BENCH_MODEL;
  const bar = steps.find(s => isNum(s.bar_usd));
  const metrics = journeyMetrics(steps);
  const [m, setM] = useState<JMetric>(metrics[0]?.key ?? 'usd_per_req');
  const runs = steps.flatMap(s => s.run_ids ?? []);
  const files = [B.journey?.file, ...steps.flatMap(s => s.evidence_files ?? [])];
  const kept = steps.filter(s => s.decision === 'kept').length;
  const rejected = steps.filter(s => s.decision === 'rejected' || s.decision === 'reverted').length;
  const cols: Col<Experiment>[] = [
    numCol<Experiment>('step', '#', s => s.step, v => num(v)),
    { id: 'change', header: 'Change', accessorFn: s => s.change, meta: { wrap: true } },
    { id: 'decision', header: 'Decision', accessorFn: s => s.decision, cell: c => <Pill tone={decisionTone(c.getValue() as string)}>{c.getValue() as string}</Pill> },
    { id: 'config', header: 'Config', accessorFn: s => cfgShort(s.config_id) },
    { id: 'group', header: 'Claim group', accessorFn: s => (s.claim_groups ?? []).join(' '), meta: { mono: true } },
    numCol<Experiment>('goodput', 'Goodput', s => s.max_rate_at_slo_rps ?? s.slo_goodput_rps, v => (isNum(v) ? `${rps(v)}` : '–')),
    numCol<Experiment>('be', 'Break-even', s => s.breakeven_rps, v => rps(v)),
    numCol<Experiment>('ach', 'Achieved', s => s.achieved_rps_at_breakeven, v => rps(v)),
    numCol<Experiment>('usd', '$/req @SLO', s => s.usd_per_req, usd),
    numCol<Experiment>('vsbar', 'vs its bar', s => s.ratio_bar ?? s.ratio_sync, v => ratio(v)),
    { id: 'bar', header: 'Price bar', accessorFn: s => s.bar_short ?? '', meta: { hide: true } },
    numCol<Experiment>('usda', '$/req achieved', s => s.usd_per_req_at_achieved, usd),
    numCol<Experiment>('ttft', 'TTFT p95', s => s.ttft_p95_s, secs),
    numCol<Experiment>('tpot', 'TPOT p50', s => s.tpot_p50_s, v => secs(v), { meta: { num: true, hide: true } }),
    numCol<Experiment>('prefill', 'Prefill tok/s', s => s.prefill_tok_s, compact, { meta: { num: true, hide: true } }),
    numCol<Experiment>('kv', 'KV tokens', s => s.kv_tokens, compact, { meta: { num: true, hide: true } }),
    numCol<Experiment>('gsm', 'GSM8K', s => s.gsm8k_acc, v => pct(v), { meta: { num: true, hide: true } }),
    numCol<Experiment>('runusd', 'Run $', s => s.run_usd_est, v => (isNum(v) ? '$' + v.toFixed(2) : '–'), { meta: { num: true, hide: true } }),
    { id: 'basis', header: 'Basis', accessorFn: s => s.basis ?? '', meta: { hide: true } },
    { id: 't', header: 'When', accessorFn: s => s.t ?? '', cell: c => utc(c.getValue() as string), meta: { hide: true } },
    { id: 'delta', header: 'Δ vs parent', accessorFn: s => Object.keys(s.delta_vs_parent ?? {}).join(' '), meta: { hide: true, wrap: true },
      cell: c => <span className="text-[11.5px]">{Object.entries(c.row.original.delta_vs_parent ?? {}).map(([k, [a, b]]) => `${k}: ${fmtDelta(k, a)} → ${fmtDelta(k, b)}`).join(' · ')}</span> },
  ];
  return (
    <div>
      <PageHead title="Journey" actions={steps.length ? <>{models.length > 1 && <Segmented value={model} onChange={setModel} options={models.map(x => ({ value: x, label: x }))} />}{bench && <Pill tone={ourClaim ? 'good' : 'warn'} title={ourClaim ? `results/claims.json: ${ourClaim.group} column ${ourClaim.column} is claim-grade` : 'no claim-grade row for our own recipe in results/claims.json yet'}>{ourClaim ? `ends at our recipe · ${ourClaim.group} (claim-grade)` : 'screening'}</Pill>}<Pill tone="good">{kept} kept</Pill><Pill tone="crit">{rejected} rejected</Pill><Pill>{steps.length} steps</Pill></> : null} />
      {!steps.length ? (
        <Card><Empty>The experiments log <code>results/experiments.jsonl</code> drives this page: one line per change, from the baseline config to the current best.</Empty></Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
          <Card className="xl:col-span-3" title={bench ? (ourClaim ? 'From baseline to our recipe' : 'From baseline to now · screening') : `${model} · from baseline to now`} info={`Step line: the kept path of ${model}. Dots: every attempt (filled kept, hollow rejected or reverted). Dashed: its price bar (${bar?.bar_short ?? 'OpenRelay sync'} ${usd(bar?.bar_usd)}/req)${bench ? `, OpenRelay's ${num(REF.req_s_per_gpu_at_slo, 1)} req/s` : ''}, or the 4 s SLO.`}
            actions={metrics.length > 1 ? <Segmented value={m} onChange={setM} options={metrics.map(x => ({ value: x.key, label: x.label }))} /> : null}
            sources={<Sources runs={runs} files={files} />}>
            <JourneyLegend />
            <JourneyChart metric={m} height={300} model={model} />
          </Card>
          <Card className="xl:col-span-2" title="Steps" sources={<Sources runs={runs} files={files} />} bodyClass="max-h-[460px] overflow-auto">
            <Stepper steps={steps} />
          </Card>
          <Card className="xl:col-span-5" title="All metrics" sources={<Sources runs={runs} files={files} />}>
            <DataTable data={steps} columns={cols} initialSort={[{ id: 'step', desc: false }]} maxHeight="none" onRowClick={s => s.run_ids?.[0] && nav({ to: '/run/$runId', params: { runId: runParam(s.run_ids[0]) } })} />
            {(B.journey?.problems ?? []).length > 0 && <ul className="mt-3 list-disc pl-5 text-[12px] text-warn-ink">{B.journey.problems.map(p => <li key={p}>{p}</li>)}</ul>}
          </Card>
        </div>
      )}
    </div>
  );
}
