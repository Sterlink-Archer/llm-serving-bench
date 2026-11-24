import { useState } from 'react';
import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import { B, K, REF, type Column } from '../lib/data';
import { isNum, ms, num, pct, rps, secs, signedPct } from '../lib/format';
import { gpuColor, useTokens } from '../lib/theme';
import { Legend, TipBox, axisX, axisY, gridProps, pointShape, refLabel, tip } from '../components/charts';
import { Card, Empty, Kpi, PageHead, Pill, RunLink, Segmented, Sources } from '../components/ui';
import { DataTable, numCol, type Col } from '../components/DataTable';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Rung = { multiple: number; rate: number; skipped?: any; pass_: boolean; ttft95: number | null; tpot50: number | null; accept: number | null; achieved: number | null; seed?: number; censored?: boolean };
type Rec = { server: string; config_id: string; ladders: Partial<Record<Column, Rung[]>>; max_rate: Partial<Record<Column, number | null>>; prefill_tok_s: number | null;
  gate: { acc: number; n: number; split?: string; passed?: boolean; file: string } | null; startup: { spec?: number; graphs?: number; max_graph?: number; kv_tokens?: number } | null; files: string[] };
type Box = { run_id: string; label: string; status: string; excluded?: string | null; gpu_key: string | null; seed: number | null; recipes: { theirs?: Rec; ours?: Rec }; files: string[] };

const at = (rs: Rung[] | undefined, rate: number) => (rs ?? []).find(r => Math.abs(r.rate - rate) < 1e-6 && !r.skipped);
const median = (xs: number[]) => { const v = xs.filter(isNum).sort((a, b) => a - b); if (!v.length) return null; const m = Math.floor(v.length / 2); return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2; };
const complete = (b: Box, col: Column) => !!(b.recipes.theirs?.ladders[col]?.length && b.recipes.ours?.ladders[col]?.length) && !b.excluded;

function BoxChart({ b, col }: { b: Box; col: Column }) {
  const t = useTokens();
  const ours = gpuColor(t, b.gpu_key);
  const th = b.recipes.theirs?.ladders[col] ?? [];
  const ou = b.recipes.ours?.ladders[col] ?? [];
  const pts = (rs: Rung[], who: string) => rs.filter(r => !r.skipped && isNum(r.ttft95)).map(r => ({ ...r, who, pass: r.pass_ }));
  const all = [...pts(th, 'theirs'), ...pts(ou, 'ours')];
  if (!all.length) return <Empty>Column {col} not measured on this box yet.</Empty>;
  const vals = all.map(p => p.ttft95 as number).concat(K.slo_ttft_p95_s);
  const rates = [...new Set(all.map(p => p.rate))].sort((x, y) => x - y);
  const log = Math.max(...vals) / Math.min(...vals) > 20;
  return (
    <ResponsiveContainer width="100%" height={190}>
      <ScatterChart margin={{ top: 6, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid {...gridProps(t)} />
        <XAxis type="number" dataKey="rate" domain={[Math.min(...rates) - 0.1, Math.max(...rates) + 0.1]} ticks={rates} tickFormatter={v => num(v, 2)} {...axisX(t)} />
        <YAxis type="number" dataKey="ttft95" scale={log ? 'log' : 'auto'} domain={log ? ['auto', 'auto'] : [0, 'auto']} tickFormatter={v => (v >= 1 ? v.toFixed(v < 10 ? 1 : 0) + 's' : (v * 1000).toFixed(0) + 'ms')} {...axisY(t, { width: 40 })} />
        <ReferenceLine y={K.slo_ttft_p95_s} stroke={t.crit} strokeDasharray="6 4" strokeWidth={1.5} label={refLabel(t, `SLO ${K.slo_ttft_p95_s} s`, 'insideBottomRight')} ifOverflow="extendDomain" />
        <Tooltip cursor={false} content={tip((d: any) => <TipBox title={`${d.who === 'ours' ? 'Ours' : "OpenRelay's recipe"} · ${num(d.rate, 2)} req/s`} rows={[
          { name: 'p95 TTFT', value: (d.censored ? '≥ ' : '') + secs(d.ttft95) }, { name: 'SLO', value: d.pass_ ? 'met' : 'missed' }, { name: 'achieved', value: `${rps(d.achieved)} req/s` },
          { name: 'TPOT p50', value: ms(d.tpot50) }, isNum(d.accept) && { name: 'MTP accept', value: num(d.accept, 2) }]} foot={b.run_id} />)} />
        <Scatter data={pts(th, 'theirs').sort((a, z) => a.rate - z.rate)} line={{ stroke: t.ref, strokeWidth: 2, strokeDasharray: '6 4' }} shape={pointShape(t.ref, t.panel, 'diamond', 64)} isAnimationActive={false} />
        <Scatter data={pts(ou, 'ours').sort((a, z) => a.rate - z.rate)} line={{ stroke: ours, strokeWidth: 2 }} shape={pointShape(ours, t.panel, 'circle', 64)} isAnimationActive={false} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export function Paired() {
  const t = useTokens();
  const boxes: Box[] = (B as any).paired ?? [];
  const [col, setCol] = useState<Column>('A');
  const done = boxes.filter(b => complete(b, col));
  const holds = (side: 'theirs' | 'ours', rate: number) => done.filter(b => at(b.recipes[side]?.ladders[col], rate)?.pass_).length;
  const top = Math.max(0, ...boxes.flatMap(b => [b.recipes.theirs, b.recipes.ours].flatMap(r => (r?.ladders[col] ?? []).map(x => x.rate))));
  const verdictRate = 3.5;
  const m = (side: 'theirs' | 'ours', f: (r: Rec) => number | null | undefined) => median(done.map(b => f(b.recipes[side] as Rec) as number));
  const mr = { theirs: m('theirs', r => r.max_rate[col]), ours: m('ours', r => r.max_rate[col]) };
  const pf = { theirs: median(boxes.map(b => b.recipes.theirs?.prefill_tok_s as number)), ours: median(boxes.map(b => b.recipes.ours?.prefill_tok_s as number)) };
  const tp = { theirs: m('theirs', r => at(r.ladders[col], 3.25)?.tpot50), ours: m('ours', r => at(r.ladders[col], 3.25)?.tpot50) };
  const ac = { theirs: m('theirs', r => at(r.ladders[col], 3.25)?.accept), ours: m('ours', r => at(r.ladders[col], 3.25)?.accept) };
  const files = boxes.flatMap(b => b.files);
  const runs = boxes.map(b => b.run_id);
  const rows = boxes.flatMap(b => (['theirs', 'ours'] as const).filter(s => b.recipes[s]).map(s => ({ b, side: s, r: b.recipes[s] as Rec })));
  const deltas = boxes.filter(b => b.recipes.theirs && b.recipes.ours).map(b => {
    const th = b.recipes.theirs as Rec, ou = b.recipes.ours as Rec;
    const d = (f: (r: Rec) => number | null | undefined) => { const a = f(th), z = f(ou); return isNum(a) && isNum(z) ? z - a : null; };
    return { b, dRate: d(r => r.max_rate[col]), dTtft: d(r => at(r.ladders[col], 3.25)?.ttft95), dTpot: d(r => at(r.ladders[col], 3.25)?.tpot50),
      dAcc: d(r => at(r.ladders[col], 3.25)?.accept), dPre: isNum(th.prefill_tok_s) && isNum(ou.prefill_tok_s) && th.prefill_tok_s ? ou.prefill_tok_s / th.prefill_tok_s - 1 : null };
  });
  if (!boxes.length) return <div><PageHead title="Same GPU: ours vs theirs" /><Card><Empty>No paired confirm box yet: a run whose config serves OpenRelay's recipe (origin openrelay_recipe) and ours on one sandbox.</Empty></Card></div>;
  return (
    <div>
      <PageHead title="Same GPU: ours vs theirs" actions={<Segmented value={col} onChange={setCol} options={[{ value: 'A', label: "A · OpenRelay's client" }, { value: 'B', label: 'B · real text' }]} />} />
      <div className="-mt-2 mb-4 flex flex-wrap gap-1.5">
        <Pill>{boxes.length} boxes · seeds {[...new Set(boxes.map(b => b.seed).filter(Boolean))].join(' / ')}</Pill>
        <Pill tone={done.length ? 'neutral' : 'warn'}>{done.length} complete in column {col}</Pill>
        {done.length > 0 && top >= verdictRate && <Pill tone={holds('ours', verdictRate) === done.length ? 'good' : 'warn'}>ours holds {verdictRate} on {holds('ours', verdictRate)}/{done.length}</Pill>}
        {done.length > 0 && top >= verdictRate && <Pill tone="neutral">theirs holds {verdictRate} on {holds('theirs', verdictRate)}/{done.length}</Pill>}
        <Pill tone="warn" title="the claim-grade label comes from results/claims.json once the confirm rows land there">screening until claims.json marks it claim-grade</Pill>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-5">
        <Kpi label="Max rate at SLO" value={<>{rps(mr.ours)} <span className="text-[14px] font-medium text-muted">vs {rps(mr.theirs)}</span></>} sub={`req/s · median, ${done.length} boxes · OpenRelay ${num(REF.req_s_per_gpu_at_slo, 1)}`}
          info="Highest offered rate on the box's ladder with p95 TTFT < 4 s, ours vs OpenRelay's recipe on the same GPU. Median across complete boxes." sources={<Sources runs={runs} files={files} />} />
        <Kpi label="Prefill tok/s" value={pf.ours ? `${num(pf.ours / 1000, 1)}K` : '—'} sub={pf.theirs ? `theirs ${num(pf.theirs / 1000, 1)}K · ${signedPct(pf.ours && pf.theirs ? pf.ours / pf.theirs - 1 : null)}` : ''}
          info="16 concurrent 10K-token prompts (OpenRelay's probe.sh), the counted second run. OpenRelay published 41.9K." sources={<Sources runs={runs} files={files} />} />
        <Kpi label="TPOT p50 @3.25" value={ms(tp.ours)} sub={`theirs ${ms(tp.theirs)} · OpenRelay ${REF.tpot_ms} ms`} sources={<Sources runs={runs} files={files} />} />
        <Kpi label="MTP accept @3.25" value={isNum(ac.ours) ? num(ac.ours, 2) : '—'} sub={`theirs ${isNum(ac.theirs) ? num(ac.theirs, 2) : '—'} tokens/step`}
          info="Mean tokens per verify step (1 + accepted ÷ drafts). Ours drafts 3, theirs 2. Random tokens (column A) accept far more than real text (column B)." sources={<Sources runs={runs} files={files} />} />
        <Kpi label="GSM8K" value={pct(median(boxes.map(b => b.recipes.ours?.gate?.acc as number)))} sub={`theirs ${pct(median(boxes.map(b => b.recipes.theirs?.gate?.acc as number)))} · gate ${pct(K.gsm8k_gate)}`}
          sources={<Sources runs={runs} files={boxes.flatMap(b => [b.recipes.ours?.gate?.file, b.recipes.theirs?.gate?.file])} />} />
      </div>
      <div className="mt-4 mb-2"><Legend items={[{ label: 'ours', color: gpuColor(t, boxes[0]?.gpu_key), glyph: 'circle' }, { label: "OpenRelay's recipe", color: t.ref, glyph: 'diamond' }]} /></div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 2xl:grid-cols-3">
        {boxes.map(b => (
          <Card key={b.run_id} title={<span className="font-mono text-[12px]">{b.run_id}</span>} info={`Seed ${b.seed ?? '?'} · ${b.status}${b.excluded ? ' · excluded: ' + b.excluded : ''}. p95 TTFT against offered rate for both recipes on this one GPU; hollow = SLO missed.`}
            sources={<Sources runs={[b.run_id]} files={b.files} />}>
            <div className="mb-1 flex flex-wrap gap-1.5"><Pill>seed {b.seed ?? '?'}</Pill>{b.status === 'running' && <Pill tone="live">running</Pill>}
              {(['ours', 'theirs'] as const).map(s => { const r = b.recipes[s]; const mx = r?.max_rate[col]; return r?.ladders[col]?.length ? <Pill key={s} tone={s === 'ours' ? 'good' : 'neutral'}>{s} {isNum(mx) && mx > 0 ? `holds ${rps(mx)}` : 'misses every rung'}</Pill> : null; })}</div>
            <BoxChart b={b} col={col} />
          </Card>
        ))}
      </div>
      <Card className="mt-4" title="Deltas per box (ours − theirs)" info="Same GPU, so box-to-box clock spread cancels. Rates and latencies at the 3.25 req/s rung, which both recipes ran." sources={<Sources runs={runs} files={files} />}>
        <DataTable data={deltas} search={false} maxHeight="none" columns={[
          { id: 'box', header: 'Box', accessorFn: (x: any) => x.b.run_id, cell: c => <RunLink id={c.getValue() as string} /> },
          numCol<any>('seed', 'Seed', x => x.b.seed, v => num(v)),
          numCol<any>('dr', 'Δ max rate', x => x.dRate, v => (isNum(v) ? (v >= 0 ? '+' : '') + num(v, 2) : '–')),
          numCol<any>('dt', 'Δ p95 TTFT @3.25', x => x.dTtft, v => (isNum(v) ? (v >= 0 ? '+' : '−') + secs(Math.abs(v)) : '–')),
          numCol<any>('dp', 'Δ TPOT @3.25', x => x.dTpot, v => (isNum(v) ? (v >= 0 ? '+' : '−') + ms(Math.abs(v)) : '–')),
          numCol<any>('da', 'Δ MTP accept', x => x.dAcc, v => (isNum(v) ? (v >= 0 ? '+' : '') + num(v, 2) : '–')),
          numCol<any>('dpf', 'Δ prefill', x => x.dPre, signedPct),
        ] as Col<any>[]} />
      </Card>
      <Card className="mt-4" title="Every box and recipe" sources={<Sources runs={runs} files={files} />}>
        <DataTable data={rows} columns={[
          { id: 'box', header: 'Box', accessorFn: (x: any) => x.b.run_id, cell: c => <RunLink id={c.getValue() as string} /> },
          { id: 'side', header: 'Recipe', accessorFn: (x: any) => (x.side === 'ours' ? 'ours' : "OpenRelay's"), cell: c => <Pill tone={c.row.original.side === 'ours' ? 'good' : 'neutral'}>{c.getValue() as string}</Pill> },
          numCol<any>('mra', 'Max rate A', x => x.r.max_rate.A, v => rps(v)),
          numCol<any>('mrb', 'Max rate B', x => x.r.max_rate.B, v => rps(v)),
          numCol<any>('t325', `TTFT @3.25 (${col})`, x => at(x.r.ladders[col], 3.25)?.ttft95, secs),
          numCol<any>('t35', `TTFT @3.5 (${col})`, x => at(x.r.ladders[col], 3.5)?.ttft95, secs),
          numCol<any>('tp', 'TPOT @3.25', x => at(x.r.ladders[col], 3.25)?.tpot50, ms),
          numCol<any>('ac', 'Accept @3.25', x => at(x.r.ladders[col], 3.25)?.accept, v => (isNum(v) ? num(v, 2) : '–')),
          numCol<any>('pf', 'Prefill tok/s', x => x.r.prefill_tok_s, v => num(v)),
          numCol<any>('gs', 'GSM8K', x => x.r.gate?.acc, v => pct(v)),
          numCol<any>('gn', 'GSM8K n', x => x.r.gate?.n, v => num(v), { meta: { num: true, hide: true } }),
          numCol<any>('k', 'Draft k', x => x.r.startup?.spec, v => num(v)),
          numCol<any>('g', 'Max graph', x => x.r.startup?.max_graph, v => num(v), { meta: { num: true, hide: true } }),
        ] as Col<any>[]} />
      </Card>
    </div>
  );
}
