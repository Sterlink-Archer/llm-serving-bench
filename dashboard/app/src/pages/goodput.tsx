import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import { B, CELLS_K, K, REF, cfgOf, cfgShort, groupBy, seriesOf, type Cell } from '../lib/data';
import { compact, isNum, ms, num, rps, secs, signedPct } from '../lib/format';
import { gpuColor, useTokens, type Tokens } from '../lib/theme';
import { ANIM, Legend, TipBox, axisLabel, axisX, axisY, gridProps, pointShape, refLabel, symbolOf, tip } from '../components/charts';
import { Card, Empty, PageHead, Pill, RunLink, Sources } from '../components/ui';
import { GoodputPanel, GpuLegend, goodputPoints, type GPoint } from '../components/widgets';
import { useState } from 'react';
import { Segmented } from '../components/ui';
import type { Column } from '../lib/data';
import { DataTable, numCol, type Col } from '../components/DataTable';
import { useNavigate } from '@tanstack/react-router';
import { runParam } from '../components/ui';

/* eslint-disable @typescript-eslint/no-explicit-any */
const DASH = ['', '6 4', '2 3', '8 3 2 3'];
const pow2 = (lo: number, hi: number) => { const o: number[] = []; for (let v = 1; v <= hi * 1.01; v *= 2) if (v >= lo / 1.01) o.push(v); return o; };

function styles(t: Tokens, keys: string[], name: (k: string) => string) {
  const per: Record<string, number> = {};
  const SY = ['circle', 'square', 'triangle', 'diamond', 'star', 'cross', 'wye'];
  return keys.map(k => { const cid = k.split('|')[0]; const i = (per[cid] = (per[cid] ?? -1) + 1); return { k, cid, color: gpuColor(t, cfgOf(cid).gpu_key), sym: SY[(SY.indexOf(symbolOf(cid)) + i) % SY.length], dash: DASH[i % 4], label: name(k) }; });
}

export function Goodput() {
  const t = useTokens();
  const nav = useNavigate();
  const [col, setCol] = useState<Column>('B');
  const gpAll = goodputPoints();
  const gp = { ...gpAll, pts: gpAll.pts.filter(p => (p.column ?? (p.client === 'vllmbench' ? 'A' : 'B')) === col) };
  const groups = groupBy(gp.pts, p => p.group ?? p.config_id);
  const st = styles(t, Object.keys(groups), k => k).map(x => { const p0 = groups[x.k][0]; return { ...x, cid: p0.config_id, color: gpuColor(t, p0.gpu_key ?? cfgOf(p0.config_id).gpu_key) }; });
  const xmax = Math.max(REF.req_s_per_gpu_at_slo * 1.25, ...gp.pts.map(p => p.rate)) * 1.05;
  const closed = CELLS_K.filter(c => c.mode === 'closed' && isNum(c.C));
  const kg = groupBy(closed, c => `${c.config_id}|${c.run_id}|${seriesOf(c)}`);
  const kkeys = Object.keys(kg).filter(k => new Set(kg[k].map(c => c.C)).size >= 2);
  const kst = styles(t, kkeys, k => `${cfgShort(k.split('|')[0])} · ${k.split('|')[2]}`);
  const kcells = kkeys.flatMap(k => kg[k]);
  const cs = kcells.map(c => c.C as number);
  const ttftK = kcells.map(c => c.ttft?.p95).filter((v): v is number => isNum(v) && v > 0);
  const kTip = tip((d: Cell) => <TipBox title={d.cell} rows={[{ name: 'out tok/s', value: compact(d.out_tok_s) }, { name: 'server', value: compact(d.srv_out_tok_s) }, { name: 'req/s', value: rps(d.req_s) },
    { name: 'TTFT p95', value: secs(d.ttft?.p95) }, { name: 'TPOT p50', value: ms(d.tpot?.p50) }]} foot={`${d.run_id} · ${d.file}`} />);
  const xs = B.xchecks ?? [];
  const xrows = xs.flatMap((x: any) => ['output_throughput', 'request_throughput', 'median_ttft_ms', 'p99_ttft_ms', 'median_tpot_ms', 'median_itl_ms', 'p99_itl_ms'].map(k => ({
    run: x.run_id, metric: k.replace(/_/g, ' '), theirs: x.theirs?.[k], ours: x.ours?.[k], diff: isNum(x.theirs?.[k]) && isNum(x.ours?.[k]) && x.theirs[k] ? x.ours[k] / x.theirs[k] - 1 : null })));
  return (
    <div>
      <PageHead title="Goodput" actions={<><Segmented value={col} onChange={setCol} options={[{ value: 'B', label: 'B · real text' }, { value: 'A', label: "A · OpenRelay's client" }]} /><Pill tone="neutral">SLO p95 TTFT &lt; {K.slo_ttft_p95_s} s</Pill><Pill>OpenRelay {num(REF.req_s_per_gpu_at_slo, 1)} req/s · TPOT {REF.tpot_ms} ms</Pill>
        <Pill tone={gp.source === 'ladder' ? 'good' : 'warn'} title={gp.note}>{gp.source === 'ladder' ? 'OpenRelay ladder' : 'matrix cells (no ladder yet)'}</Pill></>} />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title="Goodput vs offered rate" info={`Completed req/s per GPU meeting the SLO (upper) and p95 TTFT against the ${K.slo_ttft_p95_s} s line (lower). Hollow = missed. ${gp.note}`}
          sources={<Sources runs={gp.pts.map(p => p.run_id)} files={gp.pts.map(p => p.file)} note={gp.note} />}>
          <GoodputPanel height={280} column={col} />
        </Card>
        <Card title="Time per output token" info={`Median (solid) and p95 (faint) against offered rate. OpenRelay reports a ${REF.tpot_ms} ms median; a TTFT pass with a much worse TPOT is called out.`}
          sources={<Sources runs={gp.pts.map(p => p.run_id)} files={gp.pts.map(p => p.file)} />}>
          {!gp.pts.length ? <Empty>No open-loop points yet.</Empty> : (
            <>
              <GpuLegend gpus={st.map(s => cfgOf(s.cid).gpu_key)} />
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
                  <CartesianGrid {...gridProps(t)} />
                  <XAxis type="number" dataKey="rate" domain={[0, +xmax.toFixed(1)]} {...axisX(t)} label={axisLabel(t, 'offered req/s', 'insideBottom', { offset: -12 })} />
                  <YAxis type="number" dataKey="y" domain={[0, 'auto']} tickFormatter={v => ms(v)} {...axisY(t, { width: 56 })} />
                  <ReferenceLine y={REF.tpot_ms / 1000} stroke={t.ref} strokeDasharray="6 4" label={refLabel(t, `OpenRelay ${REF.tpot_ms} ms`)} ifOverflow="extendDomain" />
                  <Tooltip cursor={false} content={tip((d: any) => <TipBox title={`${cfgShort(d.config_id)} · ${num(d.rate, 2)} req/s`} rows={[{ name: 'TPOT p50', value: ms(d.tpot50) }, { name: 'TPOT p95', value: ms(d.tpot95) }]} foot={d.run_id} />)} />
                  {st.map(s => <Scatter key={s.k + '50'} data={groups[s.k].filter(p => isNum(p.tpot50)).map(p => ({ ...p, y: p.tpot50 })).sort((a, b) => a.rate - b.rate)}
                    line={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} shape={pointShape(s.color, t.panel, s.sym)} isAnimationActive={ANIM} />)}
                  {st.map(s => <Scatter key={s.k + '95'} data={groups[s.k].filter(p => isNum(p.tpot95)).map(p => ({ ...p, y: p.tpot95 })).sort((a, b) => a.rate - b.rate)}
                    line={{ stroke: s.color, strokeWidth: 1.25, strokeOpacity: 0.45, strokeDasharray: '4 3' }} shape={pointShape(s.color, t.panel, s.sym, 30)} isAnimationActive={ANIM} />)}
                </ScatterChart>
              </ResponsiveContainer>
            </>
          )}
        </Card>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
        {([['Throughput knee (closed loop)', 'out_tok_s', (v: number) => compact(v), 'output tok/s'], ['Latency under concurrency', 'ttft95', (v: number) => secs(v), 'TTFT p95']] as const).map(([title, key, f, yl]) => (
          <Card key={key} title={title} info={key === 'out_tok_s' ? 'Output tokens per second against fixed concurrency C per sweep. The knee is where more concurrency stops buying throughput.' : 'p95 time to first token for the same sweeps.'}
            sources={<Sources runs={kcells.map(c => c.run_id)} files={kcells.map(c => c.file)} />}>
            {!kkeys.length ? <Empty>No closed-loop concurrency sweep yet.</Empty> : (
              <>
                <Legend items={kst.map(s => ({ label: s.label, color: s.color, glyph: s.sym, key: s.k }))} />
                <div className="mb-0.5 text-[11px] font-medium text-muted">{yl}</div>
                <ResponsiveContainer width="100%" height={260}>
                  <ScatterChart margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
                    <CartesianGrid {...gridProps(t)} />
                    <XAxis type="number" dataKey="C" scale="log" domain={[Math.min(...cs), Math.max(...cs)]} ticks={pow2(Math.min(...cs), Math.max(...cs))} {...axisX(t)} label={axisLabel(t, 'concurrency C', 'insideBottom', { offset: -12 })} />
                    <YAxis type="number" dataKey="y" scale={key === 'ttft95' && ttftK.length > 1 && Math.max(...ttftK) / Math.min(...ttftK) > 40 ? 'log' : 'auto'} domain={key === 'ttft95' ? ['auto', 'auto'] : [0, 'auto']}
                      tickFormatter={v => f(v)} {...axisY(t, { width: 52 })} />
                    <Tooltip cursor={false} content={kTip} />
                    {kst.map(s => <Scatter key={s.k} data={kg[s.k].map(c => ({ ...c, y: key === 'out_tok_s' ? c.out_tok_s : c.ttft?.p95 })).sort((a, b) => (a.C ?? 0) - (b.C ?? 0))}
                      line={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} shape={pointShape(s.color, t.panel, s.sym)} isAnimationActive={ANIM} />)}
                  </ScatterChart>
                </ResponsiveContainer>
              </>
            )}
          </Card>
        ))}
      </div>

      <Card className="mt-4" title={gp.source === 'ladder' ? 'Ladder rungs' : 'Open-loop points'} sources={<Sources runs={gp.pts.map(p => p.run_id)} files={gp.pts.map(p => p.file)} />}>
        <DataTable data={gp.pts} onRowClick={p => nav({ to: '/run/$runId', params: { runId: runParam(p.run_id) } })} initialSort={[{ id: 'rate', desc: false }]} columns={[
          { id: 'config', header: 'Claim group', accessorFn: (p: GPoint) => p.group ?? cfgShort(p.config_id) },
          { id: 'point', header: 'Point', accessorFn: (p: GPoint) => p.label, meta: { mono: true } },
          numCol<GPoint>('rate', 'Offered', p => p.rate, v => num(v, 2)),
          numCol<GPoint>('arrived', 'Arrived', p => p.arrived, v => num(v, 2), { meta: { num: true, hide: gp.source !== 'ladder' } }),
          numCol<GPoint>('achieved', 'Achieved', p => p.achieved, v => num(v, 2)),
          numCol<GPoint>('goodput', 'Goodput', p => p.goodput, v => num(v, 2)),
          numCol<GPoint>('ttft', 'TTFT p95', p => p.ttft95, secs),
          numCol<GPoint>('tpot', 'TPOT p50', p => p.tpot50, ms),
          numCol<GPoint>('seeds', 'Seeds', p => p.seeds?.length ?? 1, v => num(v)),
          { id: 'slo', header: 'SLO', accessorFn: (p: GPoint) => (p.pass ? 1 : 0), cell: c => (c.row.original.pass ? <Pill tone="good">met</Pill> : <Pill tone="crit">missed</Pill>) },
          { id: 'client', header: 'Client', accessorFn: (p: GPoint) => (p.client === 'vllmbench' ? 'vllm bench serve' : 'traffic.py'), meta: { hide: gp.source !== 'ladder' } },
          { id: 'run', header: 'Run', accessorFn: (p: GPoint) => p.run_id, cell: c => <RunLink id={c.getValue() as string} /> },
        ] as Col<GPoint>[]} />
      </Card>

      <Card className="mt-4" title="Client cross-check: traffic.py vs vllm bench serve" info="Same 1024/256, C=32 on the same live server. Throughput and inter-token latency should agree; TTFT differs because vllm bench starts 32 requests at once while traffic.py staggers its workers."
        sources={<Sources runs={xs.map((x: any) => x.run_id)} files={xs.flatMap((x: any) => [x.file, x.ours_file])} />}>
        {!xrows.length ? <Empty>No vllm_bench_*.json next to an xcheck matrix yet.</Empty> : (
          <DataTable data={xrows} search={false} maxHeight="none" columns={[
            { id: 'metric', header: 'Metric', accessorFn: (r: any) => r.metric },
            numCol<any>('theirs', 'vllm bench', r => r.theirs, v => num(v, 1)),
            numCol<any>('ours', 'traffic.py', r => r.ours, v => num(v, 1)),
            numCol<any>('diff', 'Δ', r => r.diff, signedPct),
            { id: 'run', header: 'Run', accessorFn: (r: any) => r.run, cell: c => <RunLink id={c.getValue() as string} /> },
          ]} />
        )}
      </Card>
    </div>
  );
}
