import { useNavigate, useParams } from '@tanstack/react-router';
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { ChevronLeft } from 'lucide-react';
import { K, RUNS, cellsOfRun, cfgShort, phasesOf, runOf, runShort, secsByCat, type Cell, type Phase, type Run } from '../lib/data';
import { both, compact, ms, num, secs, toMs, usd, usd2, utc, utcFull, central, hhmm } from '../lib/format';
import { catColor, gpuColor, useTokens } from '../lib/theme';
import { ANIM, FOLD_PX, Gantt, Legend, Measured, TipBox, axisX, axisY, catAxis, foldStack, gridProps, perPx, tip, type GanttSeg } from '../components/charts';
import { Card, Empty, Kpi, PageHead, Pill, RunLink, Sources, runParam } from '../components/ui';
import { DataTable, numCol, type Col } from '../components/DataTable';
import { Link } from '@tanstack/react-router';

/* eslint-disable @typescript-eslint/no-explicit-any */
export const statusPill = (r: Run) => {
  const m: Record<string, ['good' | 'warn' | 'crit' | 'live' | 'neutral', string]> = { running: ['live', 'running'], done: ['good', 'done'], error: ['crit', 'error'],
    evicted: ['warn', 'evicted'], stale: ['warn', 'stale'] };
  const [tone, l] = m[r.status] ?? ['neutral', r.status];
  return <Pill tone={tone} title={r.error ?? (r.status === 'stale' ? 'no delete recorded and run.json not rewritten for 20 min' : undefined)}>{l}</Pill>;
};
const detailStr = (d?: Record<string, any>) => (d && Object.keys(d).length ? Object.entries(d).filter(([, v]) => typeof v !== 'object' || Array.isArray(v))
  .map(([k, v]) => `${k}=${Array.isArray(v) ? v.length + ' items' : typeof v === 'number' ? (k === 'bytes' ? compact(v) : num(v, k === 'mb_s' ? 1 : 0)) : v}`).join('  ') : '');
const segsOf = (r: Run, row: number | ((p: Phase) => number), t: any): GanttSeg[] => phasesOf(r).filter(p => p.t_start && p.t_end).map(p => ({
  row: typeof row === 'number' ? row : row(p), t0: toMs(p.t_start), t1: toMs(p.t_end), color: catColor(t, p.cat), nested: !!p.nested_in, label: p.label || p.phase,
  dur: p.dur_s, usd: p.billed === false ? null : p.usd, source: `${r.run_id} · ${p.source}`, tStart: p.t_start, tEnd: p.t_end, detail: detailStr(p.detail) }));

export function Runs() {
  const t = useTokens();
  const nav = useNavigate();
  const runsP = RUNS.filter(r => phasesOf(r).length);
  const cats = K.phase_cats;
  const go = (r: Run) => nav({ to: '/run/$runId', params: { runId: runParam(r.run_id) } });
  return (
    <div>
      <PageHead title="Runs" actions={<><Pill>{RUNS.length} runs</Pill><Pill tone="live">{RUNS.filter(r => r.status === 'running').length} running</Pill></>} />
      <Card title="Phase timeline" info="Every run's wall clock, coloured by phase. Click a run to open it. Ledger phase events carry their own times; older runs are laid out from run.json durations."
        sources={<Sources runs={runsP.map(r => r.run_id)} files={runsP.map(r => r.run_json)} />}>
        {!runsP.length ? <Empty>No phase timings yet.</Empty> : (
          <>
            <Legend items={cats.map(c => ({ label: c, color: catColor(t, c) }))} />
            <Gantt t={t} rows={runsP.map(r => runShort(r.run_id))} segs={runsP.flatMap((r, i) => segsOf(r, i, t))} onRow={i => go(runsP[i])} />
          </>
        )}
      </Card>
      <Card className="mt-4" title="Where the wall clock went" info="Minutes per phase category for each run. Cold start is everything before the first measured step."
        sources={<Sources runs={runsP.map(r => r.run_id)} files={runsP.map(r => r.run_json)} />}>
        {!runsP.length ? <Empty>No phase timings yet.</Empty> : (
          <Measured>{w => {
            const raw = runsP.map(r => ({ run: runShort(r.run_id), run_id: r.run_id, ...Object.fromEntries(Object.entries(secsByCat(r)).map(([k, v]) => [k, v / 60])) })) as Record<string, any>[];
            const ax = catAxis(w, raw.map(r => r.run), { max: 150 });
            const M = { top: 4, right: 16, bottom: 4, left: 4 };
            const data = foldStack(raw, cats, FOLD_PX * perPx(raw, cats, w - ax.width - M.left - M.right));
            const short = data.some(d => d._short > 0);
            return (<>
              <Legend items={[...cats.map(c => ({ label: c, color: catColor(t, c) })), ...(short ? [{ label: 'short phases', color: t.muted }] : [])]} />
              <ResponsiveContainer width="100%" height={Math.max(150, 40 + runsP.length * 32)}>
                <BarChart data={data} layout="vertical" margin={M}>
                  <CartesianGrid {...gridProps(t, true)} horizontal={false} />
                  <XAxis type="number" tickFormatter={v => `${num(v)} min`} {...axisX(t)} />
                  <YAxis type="category" dataKey="run" {...axisY(t, { width: ax.width, tickFormatter: ax.clip, tick: { fill: t.ink2, fontSize: 11.5 } })} />
                  <Tooltip cursor={{ fill: t.grid, opacity: 0.5 }} content={tip((d: any) => <TipBox title={d.run_id} rows={cats.filter(c => d._raw[c] > 0).map(c => ({ name: c + (d._shortKeys.includes(c) ? ' (short)' : ''), value: secs(d._raw[c] * 60), color: catColor(t, c) }))} />)} />
                  {cats.map(c => <Bar key={c} dataKey={c} stackId="s" fill={catColor(t, c)} stroke={t.panel} strokeWidth={2} barSize={16} isAnimationActive={ANIM} />)}
                  <Bar dataKey="_short" stackId="s" fill={t.muted} stroke={t.panel} strokeWidth={2} barSize={16} isAnimationActive={ANIM} />
                </BarChart>
              </ResponsiveContainer>
            </>);
          }}</Measured>
        )}
      </Card>
      <Card className="mt-4" title="All runs" sources={<Sources runs={RUNS.map(r => r.run_id)} files={RUNS.map(r => r.run_json)} />}>
        <DataTable data={RUNS} onRowClick={go} initialSort={[{ id: 'start', desc: true }]} columns={[
          { id: 'run', header: 'Run', accessorFn: (r: Run) => r.run_id, meta: { mono: true } },
          { id: 'configs', header: 'Configs', accessorFn: (r: Run) => (r.config_ids ?? []).map(cfgShort).join(', ') },
          { id: 'gpu', header: 'GPU', accessorFn: (r: Run) => r.gpu_key ?? '', cell: c => <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-[3px]" style={{ background: gpuColor(t, c.getValue() as string) }} />{c.getValue() as string}</span> },
          { id: 'status', header: 'Status', accessorFn: (r: Run) => r.status, cell: c => statusPill(c.row.original) },
          { id: 'start', header: 'Started', accessorFn: (r: Run) => r.t_start ?? '', cell: c => both(c.getValue() as string) },
          numCol<Run>('elapsed', 'Elapsed', r => r.elapsed_s, secs),
          numCol<Run>('cost', 'Est. $', r => r.est_cost_usd, usd2),
          numCol<Run>('cells', 'Cells', r => r.n_cells, v => num(v)),
          { id: 'src', header: 'Phase data', accessorFn: (r: Run) => r.phase_source, meta: { hide: true } },
          { id: 'sandbox', header: 'Sandbox', accessorFn: (r: Run) => r.sandbox_id ?? '', meta: { mono: true, hide: true } },
        ] as Col<Run>[]} />
      </Card>
    </div>
  );
}

export function RunDetail() {
  const t = useTokens();
  const { runId } = useParams({ from: '/run/$runId' });
  const id = runId.replace('~', '/');
  const r = runOf(id);
  if (!r) return <div><PageHead title="Run not found" /><Card><Empty>{id} is not in this build.</Empty></Card></div>;
  const ph = phasesOf(r);
  const cats = K.phase_cats.filter(c => ph.some(p => p.cat === c));
  const cells = cellsOfRun(r.run_id);
  const uh = r.usd_per_h ?? {};
  const gm = r.gpumon;
  const ci = (n: string) => gm?.columns.indexOf(n) ?? -1;
  return (
    <div>
      <Link to="/runs" className="mb-2 inline-flex items-center gap-1 text-[12.5px] text-muted hover:text-ink"><ChevronLeft className="size-3.5" /> Runs</Link>
      <PageHead title={r.run_id} actions={<>{statusPill(r)}{(r.config_ids ?? []).map(c => <Pill key={c}>{cfgShort(c)}</Pill>)}</>} />
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
        <Kpi label="GPU" value={<span className="text-[20px]">{r.gpu_key ?? '–'}</span>} accent={gpuColor(t, r.gpu_key)} sub={r.sandbox_id ?? ''} sources={<Sources runs={[r.run_id]} files={[r.run_json]} />} />
        <Kpi label="Box $/h" value={num(uh.total, 3)} sub={`GPU ${num(uh.gpu, 2)} + host ${num((uh.total ?? 0) - (uh.gpu ?? 0), 2)}`} />
        <Kpi label="Elapsed" value={secs(r.elapsed_s)} sub={r.status === 'running' ? 'still running' : utc(r.t_end)} />
        <Kpi label="Est. cost" value={usd2(r.est_cost_usd)} sub="run.py estimate" />
        <Kpi label="Cells" value={r.n_cells} sub={`${cells.filter(c => c.slo_openrelay).length} meet the SLO`} />
      </div>
      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-5">
        <Card className="xl:col-span-3" title="Phases" info={`Source: ${r.phase_source}. Thin bars are sub-phases nested in their parent (compile, graph capture).`} sources={<Sources runs={[r.run_id]} files={[r.run_json]} />}>
          {!ph.length ? <Empty>No phase timings yet.</Empty> : (
            <><Legend items={cats.map(c => ({ label: c, color: catColor(t, c) }))} />
              <Gantt t={t} rows={cats} segs={segsOf(r, p => cats.indexOf(p.cat), t)} rowH={30} /></>
          )}
        </Card>
        <Card className="xl:col-span-2" title="Run facts" sources={<Sources runs={[r.run_id]} files={[r.run_json]} />}>
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-[12.5px] [&_dd]:[overflow-wrap:anywhere]">
            <dt className="text-muted">Started</dt><dd className="tnum">{utcFull(r.t_start)} <span className="text-muted">({central(r.t_start)})</span></dd>
            <dt className="text-muted">Ended</dt><dd className="tnum">{r.status === 'running' ? 'running' : utcFull(r.t_end)}</dd>
            <dt className="text-muted">Spot capacity</dt><dd>{r.spot_capacity_at_start ?? '–'} free at start</dd>
            <dt className="text-muted">Phase data</dt><dd>{r.phase_source}</dd>
            {r.error && <><dt className="text-muted">Error</dt><dd className="font-mono text-[11.5px] text-crit-ink break-all">{r.error}</dd></>}
            {(r as any).evicted_event && <><dt className="text-muted">Evicted</dt><dd>{utcFull((r as any).evicted_event.evicted_at)} · {num((r as any).evicted_event.minutes_lost, 2)} min, {usd2((r as any).evicted_event.usd_lost)} lost · interrupted <span className="font-mono text-[11.5px]">{(r as any).evicted_event.interrupted?.step ?? '–'}</span> · spot preemption, excluded from benchmark numbers</dd></>}
            {r.evicted && !(r as any).evicted_event && <><dt className="text-muted">Evicted</dt><dd className="font-mono text-[11.5px]">{JSON.stringify(r.evicted)}</dd></>}
            {(r as any).rerun_of_evicted && <><dt className="text-muted">Re-run of</dt><dd><RunLink id={typeof (r as any).rerun_of_evicted === 'string' ? (r as any).rerun_of_evicted : (r as any).rerun_of_evicted.run_id} /></dd></>}
          </dl>
        </Card>
      </div>
      {gm && gm.rows.length > 1 && (
        <Card className="mt-4" title="GPU monitor" info={gm.note} sources={<Sources runs={[r.run_id]} files={[gm.file]} />}>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {([['util_pct', 'Utilisation %', (v: number) => num(v), 's1'], ['mem_mib', 'Memory GiB', (v: number) => num(v / 1024, 1), 's2'], ['power_w', 'Power W', (v: number) => num(v), 's3']] as const).map(([n, label, f, col]) => (
              <div key={n} data-chart-cell="">
                <div className="mb-1 text-[12px] text-ink-2">{label}</div>
                <ResponsiveContainer width="100%" height={120}>
                  <AreaChart data={gm.rows.map(x => ({ t: (x[0] as number) * 1000, v: x[ci(n)] }))} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                    <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} tickFormatter={v => hhmm(v)} {...axisX(t, { tickCount: 4, minTickGap: 30 })} />
                    <YAxis tickFormatter={v => f(v)} {...axisY(t, { width: 36 })} />
                    <Tooltip content={tip((d: any) => <TipBox title={hhmm(d.t)} rows={[{ name: label, value: f(d.v) }]} />)} />
                    <Area dataKey="v" stroke={(t as any)[col]} fill={(t as any)[col]} fillOpacity={0.1} strokeWidth={1.5} isAnimationActive={false} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        </Card>
      )}
      {ph.length > 0 && (
        <Card className="mt-4" title="Phase records" sources={<Sources runs={[r.run_id]} files={[r.run_json]} />}>
          <DataTable data={ph} search={false} maxHeight={360} columns={[
            { id: 'phase', header: 'Phase', accessorFn: (p: Phase) => p.label || p.phase },
            { id: 'cat', header: 'Category', accessorFn: (p: Phase) => p.cat, cell: c => <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-[3px]" style={{ background: catColor(t, c.getValue() as string) }} />{c.getValue() as string}</span> },
            { id: 'start', header: 'Start', accessorFn: (p: Phase) => p.t_start, cell: c => utc(c.getValue() as string) },
            numCol<Phase>('dur', 'Duration', p => p.dur_s, secs),
            numCol<Phase>('usd', '$', p => (p.billed === false ? null : p.usd), usd),
            { id: 'detail', header: 'Detail', accessorFn: (p: Phase) => detailStr(p.detail), meta: { mono: true } },
          ] as Col<Phase>[]} />
        </Card>
      )}
      {cells.length > 0 && (
        <Card className="mt-4" title="Cells" sources={<Sources runs={[r.run_id]} files={cells.map(c => c.file)} />}>
          <DataTable data={cells} columns={[
            { id: 'cell', header: 'Cell', accessorFn: (c: Cell) => c.cell_id, meta: { mono: true } },
            numCol<Cell>('rps', 'req/s', c => c.req_s, v => num(v, 2)),
            numCol<Cell>('tok', 'Out tok/s', c => c.out_tok_s, v => num(v)),
            numCol<Cell>('ttft', 'TTFT p95', c => c.ttft?.p95, secs),
            numCol<Cell>('tpot', 'TPOT p50', c => c.tpot?.p50, ms),
            numCol<Cell>('err', 'Errors', c => c.errors, v => num(v)),
            { id: 'slo', header: 'SLO', accessorFn: (c: Cell) => (c.slo_openrelay ? 1 : 0), cell: c => (c.row.original.slo_openrelay ? <Pill tone="good">met</Pill> : <Pill tone="crit">missed</Pill>) },
          ] as Col<Cell>[]} />
        </Card>
      )}
      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
        {r.driver_log?.length > 0 && <Card title="driver.log" info={`Last ${r.driver_log.length} lines.`}><pre className="max-h-72 overflow-auto rounded-xl bg-panel-2 p-3 font-mono text-[11px] leading-relaxed text-ink-2">{r.driver_log.join('\n')}</pre></Card>}
        <Card title={`Result files (${r.files.length})`}><ul className="max-h-72 overflow-auto font-mono text-[11px] leading-relaxed text-ink-2">{r.files.map(f => <li key={f} className="break-all">{f}</li>)}</ul></Card>
      </div>
      <span className="sr-only"><RunLink id={r.run_id} /></span>
    </div>
  );
}
