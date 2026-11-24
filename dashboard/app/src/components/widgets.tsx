import { Bar, BarChart, CartesianGrid, Cell as RCell, ComposedChart, LabelList, Line, ReferenceDot, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Symbols, Tooltip, XAxis, YAxis } from 'recharts';
import { B, BARS, BENCH_MODEL, CELLS_K, K, REF, cfgOf, cfgShort, claimBars, claimsFor, groupBy, originLabel, seriesOf, stepsOf, type Best, type Claim, type Column, type Experiment } from '../lib/data';
import { isNum, ms, num, ratio, rps, secs, usd } from '../lib/format';
import { gpuColor, useTokens, type Tokens } from '../lib/theme';
import { ANIM, FONT, Legend, Measured, TipBox, axisLabel, axisX, axisY, gridProps, minBar, placeLabels, pointShape, refLabel, stackLabel, symbolOf, tip } from './charts';
import { Empty } from './ui';

/* eslint-disable @typescript-eslint/no-explicit-any */

// ---------------------------------------------------------------- cost vs the bar, per config
export function CostVsBar({ rows, height }: { rows: Best[]; height?: number }) {
  const t = useTokens();
  if (!rows.length) return <Empty>No cell at {num(K.shape_in)}/{K.shape_out} meets the SLO with a price yet.</Empty>;
  const data = rows.map(r => ({ ...r, name: cfgShort(r.config_id), color: gpuColor(t, r.gpu_key) }));
  const xmax = Math.max(BARS.openrouter, ...rows.map(r => r.our_usd_req)) * 1.15;
  const marks: [number, string, string, number][] = [[BARS.batch, 'Batch', '2 3', 0], [BARS.sync, 'Sync', '6 4', 1], [BARS.openrouter, 'OpenRouter', '6 4', 0]];
  return (
    <ResponsiveContainer width="100%" height={height ?? Math.max(180, 64 + rows.length * 38)}>
      <BarChart data={data} layout="vertical" margin={{ top: 32, right: 64, bottom: 4, left: 8 }}>
        <CartesianGrid {...gridProps(t, true)} horizontal={false} />
        <XAxis type="number" domain={[0, xmax]} tickFormatter={v => usd(v)} {...axisX(t)} />
        <YAxis type="category" dataKey="name" {...axisY(t, { width: 170, tick: { fill: t.ink2, fontSize: 12 } })} />
        <Tooltip cursor={{ fill: t.grid, opacity: 0.5 }} content={tip(d => (
          <TipBox title={d.config_id} rows={[{ name: '$ / request', value: usd(d.our_usd_req) }, { name: 'vs OpenRelay Sync', value: ratio(d.ratio_sync) },
            { name: 'req/s', value: rps(d.req_s) }, { name: 'TTFT p95', value: secs(d.ttft_p95) }, { name: 'quotable', value: d.eligible ? 'yes' : 'no: ' + (d.stop?.[0] ?? 'GSM8K gate') }]}
            foot={`${d.run_id} · ${d.cell_id}`} />))} />
        {marks.map(([v, l, dash, row]) => <ReferenceLine key={l} x={v} stroke={t.ref} strokeDasharray={dash} strokeOpacity={0.8} label={stackLabel(t, l, row)} ifOverflow="extendDomain" />)}
        <Bar dataKey="our_usd_req" barSize={18} radius={[0, 4, 4, 0]} isAnimationActive={ANIM}>
          {data.map(d => <RCell key={d.config_id} fill={d.color} fillOpacity={d.eligible ? 1 : 0.45} />)}
          <LabelList dataKey="ratio_sync" position="right" formatter={(v: any) => ratio(v)} fill={t.ink} fontSize={11.5} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------- claims (results/claims.json): $/req per claim group
export function ClaimBars({ column, height, bar, gradedOnly }: { column: Column; height?: number; bar?: string; gradedOnly?: boolean }) {
  const t = useTokens();
  const theBar = bar ?? claimBars()[0];
  const all = claimsFor(column, theBar);
  const graded = all.filter(r => r.claim_grade);
  const rows = gradedOnly && graded.length ? graded : all;
  const isOR = /openrelay/i.test(theBar ?? '');
  const barUsd = rows[0]?.bar_usd ?? BARS.sync;
  const barName = rows[0]?.bar_short ?? theBar ?? 'the bar';
  if (!B.claims) return <Empty>Awaiting results/claims.json.</Empty>;
  if (!rows.length) return <Empty>No claim group holds the SLO on every seed in column {column} yet.</Empty>;
  const data = rows.map(r => ({ ...r, name: r.origin === 'openrelay_recipe' ? `${r.group} (theirs)` : r.group, color: r.origin === 'openrelay_recipe' ? t.ref : gpuColor(t, r.gpu_key),
    label: ratio(r.ratio) }));
  // reference lines: short names on the plot (rows chosen so they never collide), prices in the caption
  const marks: { v: number; name: string; price: string; dash: string }[] = isOR
    ? [{ v: BARS.batch, name: 'Batch', price: usd(BARS.batch), dash: '2 3' }, { v: BARS.sync, name: 'Sync', price: usd(BARS.sync), dash: '6 4' }, { v: BARS.openrouter, name: 'OpenRouter', price: usd(BARS.openrouter), dash: '6 4' }]
    : [{ v: barUsd, name: barName, price: usd(barUsd), dash: '6 4' }];
  const top0 = Math.max(...marks.map(m => m.v), ...rows.map(r => r.usd_per_req as number)) * 1.06;
  const e10 = 10 ** Math.floor(Math.log10(top0 / 4));
  const step = [1, 2, 2.5, 5, 10].map(m => m * e10).find(s => s >= top0 / 4) ?? 10 * e10;
  const xmax = Math.ceil(top0 / step) * step;
  const xticks = Array.from({ length: Math.round(xmax / step) + 1 }, (_, i) => +(i * step).toPrecision(6));
  const anyFaded = rows.some(r => !r.claim_grade);
  return (
    <div data-bars={rows.length}>
      <Measured>{w => {
        const M = { left: 4, right: 48, bottom: 4 };
        const plotW = Math.max(40, w - M.left - M.right);
        const px = (v: number) => M.left + (v / xmax) * plotW;
        const sorted = [...marks].sort((x, y) => x.v - y.v);
        const wide = w >= 520;
        const texts = sorted.map(m => (wide ? `${m.name} ${m.price}` : m.name));
        const place = placeLabels(sorted.map((m, i) => ({ x: px(m.v), text: texts[i] })), w);
        const top = 22 + 15 * Math.max(0, ...place.map(q => q.row));
        const rowH = 44;
        const H = height ?? top + rows.length * rowH + 30 + M.bottom;
        const budget = Math.floor((plotW + M.right - 4) / (11.5 * 0.58));
        // the group name sits above its bar, so the plot keeps the card's full width at any size
        const NameAbove = (p: any) => {
          const d = data[p.index];
          if (!d) return null;
          const suffix = d.claim_grade ? '' : ' · not claim-grade';
          const nm = d.name.length + suffix.length > budget ? d.name.slice(0, Math.max(4, budget - suffix.length - 1)) + '…' : d.name;
          return <text x={p.x} y={p.y - 5} fill={t.ink2} fontSize={11.5} fontFamily={FONT} paintOrder="stroke" stroke={t.panel} strokeWidth={3} strokeLinejoin="round">
            {nm}{suffix && <tspan fill={t.muted}>{suffix}</tspan>}</text>;
        };
        return (
          <ResponsiveContainer width="100%" height={H}>
            <BarChart data={data} layout="vertical" margin={{ top, right: M.right, bottom: M.bottom, left: M.left }} barCategoryGap={0}>
              <CartesianGrid {...gridProps(t, true)} horizontal={false} />
              <XAxis type="number" domain={[0, xmax]} ticks={xticks} tickFormatter={v => usd(v)} {...axisX(t)} />
              <YAxis type="category" dataKey="name" hide width={0} />
              <Tooltip cursor={{ fill: t.grid, opacity: 0.5 }} content={tip((d: Claim) => (
                <TipBox title={`${d.group} · column ${d.column}`} rows={[{ name: '$ / request', value: usd(d.usd_per_req) }, { name: `vs ${d.bar_short ?? 'bar'}`, value: ratio(d.ratio) },
                  isNum(d.ratio_metered) && { name: 'metered', value: ratio(d.ratio_metered) },
                  { name: 'rate, every seed passes', value: `${rps(d.max_rate_all_seeds_pass)} req/s` }, { name: 'worst seed achieved', value: `${rps(d.worst_seed_achieved_rps)} req/s` },
                  { name: 'worst p95 TTFT', value: secs(d.worst_seed_ttft_p95_s) }, { name: 'seeds', value: String((d.seeds ?? []).length) },
                  { name: 'claim-grade', value: d.claim_grade ? 'yes' : 'no' }, { name: 'origin', value: originLabel(d) }, d.engine ? { name: 'engine', value: d.engine } : null]}
                  foot={[...(d.not_claim_grade_because ?? []), ...(d.runs ?? [])].join(' · ')} />))} />
              {sorted.map((m, i) => <ReferenceLine key={m.name} x={m.v} stroke={t.ref} strokeDasharray={m.dash} strokeOpacity={0.8} label={stackLabel(t, texts[i], place[i].row, place[i].anchor)} />)}
              <Bar dataKey="usd_per_req" barSize={14} radius={[0, 4, 4, 0]} isAnimationActive={false} minPointSize={minBar}>
                {data.map(d => <RCell key={d.group + d.column} fill={d.color} fillOpacity={d.claim_grade ? 1 : 0.5} stroke={d.claim_grade ? 'none' : d.color} strokeDasharray="3 2" />)}
                <LabelList dataKey="name" content={NameAbove} />
                <LabelList dataKey="label" position="right" fill={t.ink} fontSize={11.5} fontFamily={FONT} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        );
      }}</Measured>
      <p className="mt-1 text-[11.5px] leading-snug text-muted" data-bar-caption="">
        × is $/request ÷ <span className="text-ink-2">{barName} {usd(barUsd)}</span>.{isOR && <> Dashed: OpenRelay Batch {usd(BARS.batch)}, Sync {usd(BARS.sync)}, cheapest OpenRouter {usd(BARS.openrouter)}.</>}{anyFaded && ' Faded, dashed outline: not claim-grade.'}
      </p>
    </div>
  );
}
export function GpuLegend({ gpus, extra = [] }: { gpus: (string | null | undefined)[]; extra?: { label: string; color: string; glyph?: string }[] }) {
  const t = useTokens();
  const gs = [...new Set(gpus.filter(Boolean) as string[])];
  return <Legend items={[...gs.map(g => ({ label: g, color: gpuColor(t, g), key: g })), ...extra]} />;
}

// ---------------------------------------------------------------- goodput vs offered rate
export type GPoint = { key: string; config_id: string; run_id: string; file: string; rate: number; achieved: number | null; arrived?: number | null; goodput: number | null;
  ttft95: number | null; tpot50: number | null; tpot95: number | null; pass: boolean; label: string; client?: string; usd_offered?: number | null; usd_achieved?: number | null; ratio_metered?: number | null;
  group?: string; column?: 'A' | 'B'; run_ids?: string[]; seeds?: string[]; gpu_key?: string | null; censored?: boolean };
export function goodputPoints(): { pts: GPoint[]; source: 'ladder' | 'cells'; note: string } {
  const lp = (B as any).ladder_points as any[] | undefined;
  if (lp && lp.length) {
    const pts: GPoint[] = lp.map(p => ({ key: p.key, config_id: p.config_id, run_id: p.run_ids[0], file: p.files[0], rate: p.rate, achieved: p.achieved, goodput: p.goodput,
      ttft95: p.ttft95, tpot50: p.tpot50, tpot95: p.tpot95, pass: !!p.pass_, label: `${p.group} · ×${p.multiple} · ${p.seeds.length} seed${p.seeds.length > 1 ? 's' : ''}`,
      client: p.column === 'A' ? 'vllmbench' : 'traffic', usd_achieved: p.usd_achieved, ratio_metered: p.ratio_metered, group: p.group, column: p.column, run_ids: p.run_ids, seeds: p.seeds, gpu_key: p.gpu_key, censored: !!p.censored }));
    return { pts, source: 'ladder', note: 'OpenRelay ladder, twin runs merged by claim group: a rate passes only if every seed passes, at the worst seed (protocol_rows.json).' };
  }
  const pts = CELLS_K.filter(c => c.mode === 'open' && isNum(c.rate)).map(c => ({ key: `${c.config_id}|${seriesOf(c)}`, config_id: c.config_id, run_id: c.run_id, file: c.file,
    rate: c.rate as number, achieved: c.req_s_completed, goodput: c.slo_openrelay ? c.req_s_completed : 0, ttft95: c.ttft?.p95 ?? null, tpot50: c.tpot?.p50 ?? null,
    tpot95: c.tpot?.p95 ?? null, pass: c.slo_openrelay, label: c.cell, client: 'traffic' }));
  return { pts, source: 'cells', note: 'No OpenRelay ladder yet: open-loop matrix cells (any shape, one seed).' };
}
const DASHES = ['', '6 4', '2 3', '8 3 2 3'];
const SYMS = ['circle', 'square', 'triangle', 'diamond', 'star', 'cross', 'wye'] as const;
function seriesStyles(t: Tokens, keys: string[]) {
  const per: Record<string, number> = {};
  return keys.map(k => {
    const cid = k.split('|')[0];
    const i = (per[cid] = (per[cid] ?? -1) + 1);
    return { key: k, cid, color: gpuColor(t, cfgOf(cid).gpu_key), sym: SYMS[(SYMS.indexOf(symbolOf(cid)) + i) % SYMS.length], dash: DASHES[i % DASHES.length], label: k.includes('|') ? `${cfgShort(cid)} · ${k.split('|')[1]}` : cfgShort(cid) };
  });
}
export function GoodputPanel({ height = 260, compact, column }: { height?: number; compact?: boolean; column?: Column }) {
  const t = useTokens();
  const all = goodputPoints().pts;
  const pts = column ? all.filter(p => (p.column ?? (p.client === 'vllmbench' ? 'A' : 'B')) === column) : all;
  if (!pts.length) return <Empty>{all.length ? `No ladder rungs in column ${column} yet.` : 'The OpenRelay ladder (0.6–2.0× break-even, seeds 7/8, 300 s) fills this.'}</Empty>;
  const groups = groupBy(pts, p => (column ? (p.group ?? p.config_id) + '|' : p.key));
  const st = seriesStyles(t, Object.keys(groups)).map(x => { const p0 = groups[x.key][0]; return { ...x, cid: p0.config_id, color: gpuColor(t, p0.gpu_key ?? cfgOf(p0.config_id).gpu_key), label: p0.group ?? x.label }; });
  const xmax = Math.max(REF.req_s_per_gpu_at_slo * 1.25, ...pts.map(p => p.rate)) * 1.05;
  const ymaxG = Math.max(REF.req_s_per_gpu_at_slo * 1.15, ...pts.map(p => p.achieved ?? 0)) * 1.08;
  const tt = tip(d => (
    <TipBox title={`${d.group ?? cfgShort(d.config_id)} · ${d.gpu_key ?? cfgOf(d.config_id).gpu_key ?? ''} · offered ${num(d.rate, 2)} req/s`} rows={[{ name: 'goodput', value: d.pass ? rps(d.goodput) : 'SLO missed' }, { name: 'achieved', value: rps(d.achieved) },
      isNum(d.arrived) && { name: 'arrived', value: rps(d.arrived) }, { name: 'TTFT p95', value: (d.censored ? '≥ ' : '') + secs(d.ttft95) }, { name: 'TPOT p50', value: ms(d.tpot50) },
      isNum(d.usd_achieved) && { name: '$/req at achieved', value: usd(d.usd_achieved) },
      isNum(d.ratio_metered) && { name: 'vs Sync (metered $/h)', value: ratio(d.ratio_metered) },
      d.seeds && { name: 'seeds (all must pass)', value: String(d.seeds.length) },
      d.client && { name: 'column', value: d.client === 'vllmbench' ? 'A · vllm bench serve' : 'B · traffic.py, real text' }]} foot={(d.run_ids ?? [d.run_id]).join(' · ')} />));
  const ttftVals = pts.map(p => p.ttft95).filter((v): v is number => isNum(v) && v > 0);
  const logT = ttftVals.length > 1 && Math.max(...ttftVals, K.slo_ttft_p95_s) / Math.min(...ttftVals, K.slo_ttft_p95_s) > 40;
  return (
    <div>
      <GpuLegend gpus={st.map(s => cfgOf(s.cid).gpu_key)} extra={[{ label: 'OpenRelay MI355X (published)', color: t.ref, glyph: 'diamond' }]} />
      <div className="mb-0.5 text-[11px] font-medium text-muted">achieved req/s per GPU</div>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
          <CartesianGrid {...gridProps(t)} />
          <XAxis type="number" dataKey="rate" domain={[0, +xmax.toFixed(1)]} {...axisX(t)} label={axisLabel(t, 'offered req/s', 'insideBottom', { offset: -12 })} />
          <YAxis type="number" dataKey="achieved" domain={[0, +ymaxG.toFixed(1)]} {...axisY(t, { width: 40 })} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: xmax, y: xmax }]} stroke={t.axis} ifOverflow="hidden" />
          <Tooltip content={tt} cursor={false} />
          {st.map(s => <Scatter key={s.key} data={groups[s.key].slice().sort((a, b) => a.rate - b.rate)} line={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }}
            lineJointType="linear" shape={pointShape(s.color, t.panel, s.sym)} isAnimationActive={ANIM} />)}
          <ReferenceDot x={REF.req_s_per_gpu_at_slo} y={REF.req_s_per_gpu_at_slo} r={0} ifOverflow="extendDomain"
            shape={(p: any) => <Symbols cx={p.cx} cy={p.cy} type="diamond" size={110} fill={t.ref} stroke={t.panel} strokeWidth={2} />}
            label={{ value: `OpenRelay ${num(REF.req_s_per_gpu_at_slo, 1)}`, position: 'left', fill: t.ink, fontSize: 11 }} />
        </ScatterChart>
      </ResponsiveContainer>
      <div className="mt-1 mb-0.5 text-[11px] font-medium text-muted">p95 time to first token</div>
      <ResponsiveContainer width="100%" height={compact ? 120 : 170}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
          <CartesianGrid {...gridProps(t)} />
          <XAxis type="number" dataKey="rate" domain={[0, +xmax.toFixed(1)]} hide />
          <YAxis type="number" dataKey="ttft95" scale={logT ? 'log' : 'auto'} domain={logT ? ['auto', 'auto'] : [0, 'auto']} tickFormatter={v => (v >= 10 ? v.toFixed(0) + 's' : v >= 1 ? v.toFixed(1) + 's' : (v * 1000).toFixed(0) + 'ms')} {...axisY(t, { width: 40 })} tickCount={4} />
          <ReferenceLine y={K.slo_ttft_p95_s} stroke={t.crit} strokeDasharray="6 4" strokeWidth={1.5} label={refLabel(t, `SLO ${K.slo_ttft_p95_s} s`, 'insideBottomRight')} ifOverflow="extendDomain" />
          <Tooltip content={tt} cursor={false} />
          {st.map(s => <Scatter key={s.key} data={groups[s.key].filter(p => isNum(p.ttft95)).slice().sort((a, b) => a.rate - b.rate)} line={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }}
            shape={pointShape(s.color, t.panel, s.sym, 56)} isAnimationActive={ANIM} />)}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------- journey step chart
export type JMetric = 'usd_per_req' | 'max_rate_at_slo_rps' | 'usd_per_req_at_achieved' | 'ttft_p95_s';
export const JMETRICS: { key: JMetric; label: string; lower: boolean; f: (v: number) => string }[] = [
  { key: 'usd_per_req', label: '$/req at SLO', lower: true, f: v => usd(v) },
  { key: 'max_rate_at_slo_rps', label: 'Goodput', lower: false, f: v => `${rps(v)} req/s` },
  { key: 'usd_per_req_at_achieved', label: '$/req achieved', lower: true, f: v => usd(v) },
  { key: 'ttft_p95_s', label: 'TTFT p95', lower: true, f: v => secs(v) },
];
const jval = (s: Experiment, m: JMetric) => { const v = (s as any)[m] ?? (m === 'max_rate_at_slo_rps' ? s.slo_goodput_rps : null); return isNum(v) ? v : null; };
/** metrics with >= 2 values; ones whose values all agree (e.g. goodput 0 everywhere) go last */
export const journeyMetrics = (steps: Experiment[]) => {
  const ok = JMETRICS.filter(m => steps.filter(s => jval(s, m.key) != null).length >= 2);
  const flat = (m: (typeof JMETRICS)[number]) => new Set(steps.map(s => jval(s, m.key)).filter(v => v != null)).size < 2;
  return [...ok.filter(m => !flat(m)), ...ok.filter(flat)];
};
export const decisionTone = (d: string) => (d === 'kept' ? 'good' : d === 'rejected' || d === 'reverted' ? 'crit' : 'neutral') as 'good' | 'crit' | 'neutral';

/** Step chart: the kept path from baseline to current best as a step line; every attempt as a dot (filled = kept, hollow = rejected/reverted). */
export function JourneyChart({ metric, height = 240, model }: { metric: JMetric; height?: number; model?: string }) {
  const t = useTokens();
  const steps = stepsOf(model) as Experiment[];
  const bench = (steps[0]?.model ?? BENCH_MODEL) === BENCH_MODEL;
  const barUsd = steps.find(s => isNum(s.bar_usd))?.bar_usd ?? BARS.sync;
  const barShort = steps.find(s => s.bar_short)?.bar_short ?? 'OpenRelay sync';
  const m = JMETRICS.find(x => x.key === metric)!;
  const pts = steps.map(s => ({ name: `#${s.step}`, step: s.step, v: jval(s, metric), s, onPath: s.decision === 'baseline' || s.decision === 'kept' }));
  const withV = pts.filter(p => p.v != null);
  if (withV.length < 2) return <Empty>The experiments log (results/experiments.jsonl) fills this as steps land: baseline to current best, one point per change.</Empty>;
  let run: number | null = null;
  const data = pts.map(p => { if (p.onPath && p.v != null) run = p.v; return { ...p, path: run }; });
  // the reference is this model's own: its price bar for $/req; OpenRelay's published MI355X rate exists for Gemma 4 31B only
  const ref = metric === 'ttft_p95_s' ? { y: K.slo_ttft_p95_s, l: `SLO ${K.slo_ttft_p95_s} s`, c: t.crit } : metric === 'max_rate_at_slo_rps' ? (bench ? { y: REF.req_s_per_gpu_at_slo, l: `OpenRelay ${num(REF.req_s_per_gpu_at_slo, 1)}`, c: t.ref } : null)
    : { y: barUsd, l: `${barShort} ${usd(barUsd)}`, c: t.ref };
  const vals = withV.map(p => p.v as number).concat(ref ? [ref.y] : []);
  const log = metric === 'ttft_p95_s' && Math.max(...vals) / Math.min(...vals) > 20;
  const color = (d: string) => (d === 'kept' ? t.s1 : d === 'baseline' ? t.ink2 : d === 'info' ? t.muted : t.s8);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 16, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid {...gridProps(t)} />
        <XAxis dataKey="name" {...axisX(t)} interval="preserveStartEnd" minTickGap={10} />
        <YAxis scale={log ? 'log' : 'auto'} domain={log ? ['auto', 'auto'] : [0, 'auto']} tickFormatter={v => (metric.startsWith('usd') ? usd(v) : metric === 'ttft_p95_s' ? (v >= 1 ? v.toFixed(v < 10 ? 1 : 0) + 's' : (v * 1000).toFixed(0) + 'ms') : num(v, 2))} {...axisY(t, { width: metric.startsWith('usd') ? 72 : 44 })} allowDataOverflow={false} />
        {ref && <ReferenceLine y={ref.y} stroke={ref.c} strokeDasharray="6 4" label={refLabel(t, ref.l, 'insideBottomLeft')} ifOverflow="extendDomain" />}
        <Tooltip cursor={{ stroke: t.axis }} content={tip((d: any) => (
          <TipBox title={`#${d.step} · ${d.s.change}`} rows={[{ name: m.label, value: d.v != null ? m.f(d.v) : '—' }, { name: 'decision', value: d.s.decision }, d.s.config_id && { name: 'config', value: cfgShort(d.s.config_id) },
            d.s.why && { name: 'why', value: <span className="block max-w-56 whitespace-normal text-right font-normal">{String(d.s.why).slice(0, 140)}{String(d.s.why).length > 140 ? '…' : ''}</span> }]}
            foot={(d.s.run_ids ?? []).join(' · ')} />))} />
        <Line dataKey="path" type="stepAfter" stroke={t.s1} strokeWidth={2} strokeOpacity={0.55} dot={false} isAnimationActive={false} connectNulls />
        <Line dataKey="v" stroke="none" isAnimationActive={false} legendType="none"
          dot={(p: any) => p.payload.v == null ? <g key={p.index} /> : (
            <circle key={p.index} cx={p.cx} cy={p.cy} r={p.payload.onPath ? 6 : 5} fill={p.payload.onPath ? color(p.payload.s.decision) : t.panel}
              stroke={p.payload.onPath ? t.panel : color(p.payload.s.decision)} strokeWidth={2} />)}
          activeDot={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
export function JourneyLegend() {
  const t = useTokens();
  return <Legend items={[{ label: 'kept path', color: t.s1, line: 'solid' }, { label: 'kept', color: t.s1 }, { label: 'rejected / reverted', color: t.s8 }, { label: 'baseline', color: t.ink2 }]} />;
}
