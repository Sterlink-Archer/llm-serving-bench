import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import { useNavigate } from '@tanstack/react-router';
import { B, BARS, CELLS, CELLS_K, CFG, K, RUNS, cfgOf, cfgShort, cellsOfRun, dollarsByCat, groupBy, loadOf, phasesOf, runShort, type Cell, type Config, type Run } from '../lib/data';
import { isNum, num, ratio, rps, secs, usd } from '../lib/format';
import { catColor, gpuColor, useTokens } from '../lib/theme';
import { ANIM, FOLD_PX, Legend, Measured, TipBox, axisLabel, axisX, axisY, catAxis, foldStack, gridProps, logTicks, perPx, pointShape, refLabel, symbolOf, tip } from '../components/charts';
import { Card, Empty, PageHead, Pill, RunLink, Sources, runParam } from '../components/ui';
import { ClaimBars, GpuLegend } from '../components/widgets';
import { useState } from 'react';
import { Segmented, Select } from '../components/ui';
import { claimBars, claimsFor, COLUMN_NAME, type Column } from '../lib/data';
import { DataTable, numCol, type Col } from '../components/DataTable';

/* eslint-disable @typescript-eslint/no-explicit-any */
const sloPill = (c: Cell) => (c.slo_openrelay ? <Pill tone="good">met</Pill> : <Pill tone="crit" title={`p95 TTFT ${secs(c.ttft?.p95)}, errors ${c.errors}`}>missed</Pill>);

export function cellColumns(extra: Col<Cell>[] = []): Col<Cell>[] {
  return [
    { id: 'cell', header: 'Cell', accessorFn: c => c.cell_id, meta: { mono: true } },
    { id: 'config', header: 'Config', accessorFn: c => cfgShort(c.config_id) },
    { id: 'load', header: 'Load', accessorFn: c => loadOf(c), enableSorting: false },
    ...extra,
    { id: 'run', header: 'Run', accessorFn: c => c.run_id, cell: c => <RunLink id={c.getValue() as string} /> },
    { id: 'file', header: 'File', accessorFn: c => c.file, meta: { mono: true, hide: true } },
  ];
}

function measuredStack(): any[] {
  if ((B.measured ?? []).length) {
    return B.measured.filter((m: any) => m.requests_served > 0).map((m: any) => {
      const cold = Object.values(m.cold_start_usd ?? {}).reduce((a: number, b: any) => a + (b || 0), 0);
      const r: any = { run_id: m.run_id, requests: m.requests_served, steady_gpu: m.steady_gpu_usd || 0, steady_host: m.steady_host_usd || 0, cold,
        evict: 0, other: (m.idle_usd || 0) + (m.failed_create_usd || 0), basis: m.basis, file: m.file, provisional: false };
      return r;
    });
  }
  const rows: any[] = [];
  for (const r of RUNS) {
    const ph = phasesOf(r);
    const reqs = cellsOfRun(r.run_id).reduce((a, c) => a + (c.completed || 0), 0);
    if (!ph.length || !reqs || !r.usd_per_h) continue;
    const g = r.usd_per_h.gpu / r.usd_per_h.total;
    let steady = 0, cold = 0, evict = 0, other = 0;
    for (const p of ph) {
      if (p.nested_in || p.billed === false) continue;
      if (p.cat === 'measure') steady += p.usd || 0;
      else if (p.phase === 'eviction') evict += p.usd || 0;   // tracked, never charged to a request
      else if (p.phase === 'failed_create') other += p.usd || 0;
      else if (p.cat === 'eval' || p.phase === 'idle') other += p.usd || 0;
      else if (p.cat !== 'teardown') cold += p.usd || 0;
    }
    rows.push({ run_id: r.run_id, requests: reqs, steady_gpu: steady * g, steady_host: steady * (1 - g), cold, evict, other, file: r.run_json, provisional: true,
      basis: `${num(reqs)} requests completed in ${cellsOfRun(r.run_id).length} cells (dashboard-derived until cost_measured.json)` });
  }
  return rows;
}

export function Cost() {
  const t = useTokens();
  const [col, setCol] = useState<Column>('B');
  const bars = claimBars();
  const [bar, setBar] = useState<string | undefined>(bars[0]);
  const nav = useNavigate();
  const best = B.best ?? [];
  const priced = CELLS_K.filter(c => isNum(c.ratio_sync) && isNum(c.req_s) && c.req_s > 0);
  const byCfg = groupBy(priced, c => c.config_id);
  const cats = K.phase_cats;
  const runsP = RUNS.filter(r => phasesOf(r).length);
  const stack = measuredStack();
  // evictions are spot preemptions outside our control: never part of a $/request (Maks); they are shown on the Spend page
  const parts: [string, string][] = [['steady GPU', 'steady_gpu'], ['steady host', 'steady_host'], ['cold start', 'cold'], ['eval & idle', 'other']];
  const nCells = (id: string) => CELLS.filter(c => c.config_id === id).length;
  const cfgs = Object.values(CFG).filter(c => c.usd_per_h?.total).sort((a, b) => nCells(b.config_id) - nCells(a.config_id) || a.config_id.localeCompare(b.config_id));
  const ratioVals = priced.map(c => c.ratio_sync as number);
  return (
    <div>
      <PageHead title="Cost" actions={<><Pill>Sync {usd(BARS.sync)}</Pill><Pill>Batch {usd(BARS.batch)}</Pill><Pill>OpenRouter {usd(BARS.openrouter)}</Pill></>} />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        <Card className="xl:col-span-2" title="Claims, agentic mix" info={`results/claims.json, column ${col} (${COLUMN_NAME[col]}): $/req at the highest rate every seed holds, on the worst seed's achieved req/s. Faded = not claim-grade.`}
          actions={<>{bars.length > 1 && <Select value={bar ?? bars[0]} onChange={setBar} options={bars.map(b => ({ value: b, label: b }))} label="Price bar" />}<Segmented value={col} onChange={setCol} options={[{ value: 'B', label: 'B' }, { value: 'A', label: 'A' }]} /></>}
          sources={<Sources runs={claimsFor(col, bar).flatMap(r => r.runs)} files={[B.claims?.file]} />}>
          <ClaimBars column={col} bar={bar} />
        </Card>
        <Card className="xl:col-span-3" title="Every cell vs its price bar, at its own shape" info="Our steady $/request ÷ the cell's own price bar (OpenRelay Sync for Gemma 4 31B; OpenRelay's GLM price or OpenRouter's Inkling price for those models) at the same token counts. Below 1× is cheaper. Hollow = missed the SLO."
          sources={<Sources runs={priced.map(c => c.run_id)} files={[...priced.map(c => c.file), B.build.cost_rows_file]} />}>
          {!priced.length ? <Empty>{CELLS.length ? `${CELLS.length} cells measured; none has a row in results/cost_rows.json yet.` : 'No cells yet.'}</Empty> : (
            <>
              <GpuLegend gpus={Object.keys(byCfg).map(cid => cfgOf(cid).gpu_key)} />
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 8, right: 70, bottom: 18, left: 0 }}>
                  <CartesianGrid {...gridProps(t)} />
                  <XAxis type="number" dataKey="req_s" scale="log" domain={[Math.min(...priced.map(c => c.req_s)) * 0.8, Math.max(...priced.map(c => c.req_s)) * 1.25]} ticks={logTicks(Math.min(...priced.map(c => c.req_s)) * 0.8, Math.max(...priced.map(c => c.req_s)) * 1.25)} tickFormatter={v => num(v, v < 1 ? 2 : 0)} {...axisX(t)} label={axisLabel(t, 'req/s (steady)', 'insideBottom', { offset: -12 })} />
                  <YAxis type="number" dataKey="ratio_sync" scale="log" domain={[Math.min(0.4, ...ratioVals) * 0.8, Math.max(2, ...ratioVals) * 1.2]} tickFormatter={v => ratio(v)} {...axisY(t, { width: 52 })} />
                  <ReferenceLine y={1} stroke={t.ref} strokeDasharray="6 4" label={refLabel(t, 'bar 1×', 'right')} />
                  <ReferenceLine y={0.5} stroke={t.ref} strokeDasharray="2 3" label={refLabel(t, 'Batch 0.5×', 'right')} />
                  <Tooltip cursor={false} content={tip(d => <TipBox title={d.cell} rows={[{ name: 'ours $/req', value: usd(d.our_usd_req) }, { name: `${d.bar_name ?? 'OpenRelay sync'} at this shape`, value: usd(d.bar_sync) },
                    { name: 'ratio', value: ratio(d.ratio_sync) }, { name: 'in / out', value: `${num(d.in_mean)} / ${num(d.out_per_req)}` }, { name: 'req/s', value: rps(d.req_s) },
                    { name: 'TTFT p95', value: secs(d.ttft?.p95) }]} foot={`${d.run_id} · ${d.file}`} />)} />
                  {Object.entries(byCfg).map(([cid, cs]) => {
                    const col = gpuColor(t, cfgOf(cid).gpu_key);
                    return <Scatter key={cid} data={cs.map(c => ({ ...c, pass: c.slo_openrelay }))} shape={pointShape(col, t.panel, symbolOf(cid))} isAnimationActive={ANIM} />;
                  })}
                </ScatterChart>
              </ResponsiveContainer>
            </>
          )}
        </Card>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title="Dollars by phase, per run" info="Sandbox seconds × the box's full $/h, by phase. Snapshot builds are free."
          sources={<Sources runs={runsP.map(r => r.run_id)} files={runsP.map(r => r.run_json)} />}>
          {!runsP.length ? <Empty>No phase timings yet.</Empty> : (
            <>
              <Measured>{w => {
                const raw = runsP.map(r => ({ run: runShort(r.run_id), run_id: r.run_id, ...dollarsByCat(r) })) as Record<string, any>[];
                const ax = catAxis(w, raw.map(r => r.run), { max: 150 });
                const M = { top: 4, right: 16, bottom: 4, left: 4 };
                const data = foldStack(raw, cats, FOLD_PX * perPx(raw, cats, w - ax.width - M.left - M.right));
                const short = data.some(d => d._short > 0);
                const open = (d: any) => nav({ to: '/run/$runId', params: { runId: runParam(d.run_id) } });
                return (<>
                  <Legend items={[...cats.map(c => ({ label: c, color: catColor(t, c) })), ...(short ? [{ label: 'short phases', color: t.muted }] : [])]} />
                  <ResponsiveContainer width="100%" height={Math.max(160, 50 + runsP.length * 34)}>
                    <BarChart data={data} layout="vertical" margin={M}>
                      <CartesianGrid {...gridProps(t, true)} horizontal={false} />
                      <XAxis type="number" tickFormatter={v => usd(v)} {...axisX(t)} />
                      <YAxis type="category" dataKey="run" {...axisY(t, { width: ax.width, tickFormatter: ax.clip, tick: { fill: t.ink2, fontSize: 11.5 } })} />
                      <Tooltip cursor={{ fill: t.grid, opacity: 0.5 }} content={tip((d: any) => <TipBox title={d.run_id} rows={cats.filter(c => d._raw[c] > 0).map(c => ({ name: c + (d._shortKeys.includes(c) ? ' (short)' : ''), value: usd(d._raw[c]), color: catColor(t, c) }))} />)} />
                      {cats.map(c => <Bar key={c} dataKey={c} stackId="d" fill={catColor(t, c)} stroke={t.panel} strokeWidth={2} barSize={16} isAnimationActive={ANIM} onClick={open} className="cursor-pointer" />)}
                      <Bar dataKey="_short" stackId="d" fill={t.muted} stroke={t.panel} strokeWidth={2} barSize={16} isAnimationActive={ANIM} onClick={open} className="cursor-pointer" />
                    </BarChart>
                  </ResponsiveContainer>
                </>);
              }}</Measured>
            </>
          )}
        </Card>
        <Card title="Measured $ per request" info={(stack[0]?.provisional ? 'Provisional: derived from run phases until cost_measured.json lands. ' : 'From cost_measured.json. ') + 'Steady, cold start, eval and idle per request. Spot evictions are excluded and shown on the Spend page.'}
          sources={<Sources runs={stack.map(r => r.run_id)} files={stack.map(r => r.file)} />}>
          {!stack.length ? <Empty>No run has both phase timings and completed requests yet.</Empty> : (
            <>
              <Measured>{w => {
                const keys = parts.map(([, k]) => k + '_r');
                const raw = stack.map(r => ({ ...r, run: runShort(r.run_id), ...Object.fromEntries(parts.map(([, k]) => [k + '_r', r[k] / r.requests])) })) as Record<string, any>[];
                const ax = catAxis(w, raw.map(r => r.run), { max: 150 });
                const M = { top: 18, right: 16, bottom: 4, left: 4 };
                const data = foldStack(raw, keys, FOLD_PX * perPx(raw, keys, w - ax.width - M.left - M.right, BARS.sync));
                const short = data.some(d => d._short > 0);
                return (<>
                  <Legend items={[...parts.map(([l], i) => ({ label: l, color: (t as any)['s' + (i + 1)] })), ...(short ? [{ label: 'small parts', color: t.muted }] : [])]} />
                  <ResponsiveContainer width="100%" height={Math.max(160, 50 + stack.length * 34)}>
                    <BarChart data={data} layout="vertical" margin={M}>
                      <CartesianGrid {...gridProps(t, true)} horizontal={false} />
                      <XAxis type="number" tickFormatter={v => usd(v)} {...axisX(t)} />
                      <YAxis type="category" dataKey="run" {...axisY(t, { width: ax.width, tickFormatter: ax.clip, tick: { fill: t.ink2, fontSize: 11.5 } })} />
                      <ReferenceLine x={BARS.sync} stroke={t.ref} strokeDasharray="6 4" label={refLabel(t, 'Sync', 'top')} ifOverflow="extendDomain" />
                      <Tooltip cursor={{ fill: t.grid, opacity: 0.5 }} content={tip((d: any) => <TipBox title={d.run_id} rows={[...parts.map(([l, k], i) => ({ name: l, value: usd(d._raw[k + '_r']), color: (t as any)['s' + (i + 1)] })),
                        { name: 'requests', value: num(d.requests) }]} foot={d.basis} />)} />
                      {keys.map((k, i) => <Bar key={k} dataKey={k} stackId="m" fill={(t as any)['s' + (i + 1)]} stroke={t.panel} strokeWidth={2} barSize={16} isAnimationActive={ANIM} />)}
                      <Bar dataKey="_short" stackId="m" fill={t.muted} stroke={t.panel} strokeWidth={2} barSize={16} isAnimationActive={ANIM} />
                    </BarChart>
                  </ResponsiveContainer>
                </>);
              }}</Measured>
            </>
          )}
        </Card>
      </div>

      <Card className="mt-4" title="Priced cells" sources={<Sources runs={priced.map(c => c.run_id)} files={[B.build.cost_rows_file]} />}>
        <DataTable data={CELLS_K} onRowClick={c => nav({ to: '/run/$runId', params: { runId: runParam(c.run_id) } })} initialSort={[{ id: 'ratio', desc: false }]}
          columns={cellColumns([
            numCol<Cell>('inout', 'In / out', c => c.in_mean, (_v) => null, { cell: c => `${num(c.row.original.in_mean)} / ${num(c.row.original.out_per_req)}` }),
            numCol<Cell>('rps', 'req/s', c => c.req_s, v => num(v, 3)),
            numCol<Cell>('usd_h', 'Box $/h', c => c.usd_h, v => num(v, 3), { meta: { num: true, hide: true } }),
            numCol<Cell>('ours', 'Ours $/req', c => c.our_usd_req, usd),
            { id: 'barname', header: 'Price bar', accessorFn: (c: Cell) => (c as any).bar_name ?? 'OpenRelay sync', meta: { hide: true, wrap: true } },
            numCol<Cell>('bar', 'Bar $/req', c => c.bar_sync, usd),
            numCol<Cell>('ratio', 'vs its bar', c => c.ratio_sync, ratio),
            numCol<Cell>('rb', 'vs Batch', c => c.ratio_batch, ratio, { meta: { num: true, hide: true } }),
            numCol<Cell>('ror', 'vs OpenRouter', c => c.ratio_openrouter, ratio, { meta: { num: true, hide: true } }),
            { id: 'slo', header: 'SLO', accessorFn: c => (c.slo_openrelay ? 1 : 0), cell: c => sloPill(c.row.original) },
          ])} />
      </Card>

      <Card className="mt-4" title="Break-even rates" info={`Rate a box must sustain at ${num(K.shape_in)}/${K.shape_out} to cost exactly the bar: box $/h ÷ 3600 ÷ bar $/request.`}
        sources={<Sources files={[...cfgs.map(c => c.config_file), B.bars.openrelay.file]} note="Box $/h for configs not yet run: infra/daytona.py hourly_usd() at the saved Daytona spot list." />}>
        <DataTable data={cfgs} maxHeight={360}
          columns={[
            { id: 'config', header: 'Config', accessorFn: (c: Config) => cfgShort(c.config_id) },
            { id: 'gpu', header: 'GPU', accessorFn: (c: Config) => c.gpu_key, cell: c => <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-[3px]" style={{ background: gpuColor(t, c.getValue() as string) }} />{c.getValue() as string}</span> },
            numCol<Config>('gpuh', 'GPU $/h', c => c.usd_per_h?.gpu, v => num(v, 2)),
            numCol<Config>('boxh', 'Box $/h', c => c.usd_per_h?.total, v => num(v, 3)),
            numCol<Config>('be', 'Break-even Sync', c => (c.usd_per_h?.total ?? 0) / 3600 / BARS.sync, v => `${rps(v)} req/s`),
            numCol<Config>('beb', 'Break-even Batch', c => (c.usd_per_h?.total ?? 0) / 3600 / BARS.batch, v => `${rps(v)} req/s`),
            numCol<Config>('cells', 'Cells', c => nCells(c.config_id), v => num(v)),
            { id: 'file', header: 'Config file', accessorFn: (c: Config) => c.config_file, meta: { mono: true, hide: true } },
          ] as Col<Config>[]} />
      </Card>
    </div>
  );
}
export type { Run };
