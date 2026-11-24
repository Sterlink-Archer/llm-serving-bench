import { useState } from 'react';
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell as RCell, ComposedChart, LabelList, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Symbols, Tooltip, XAxis, YAxis } from 'recharts';
import { B, CELLS, CELLS_K, CFG, K, REF, RUNS, cfgOf, cfgShort, gateOf, groupBy, seriesOf, uniq, type Cell, type Config } from '../lib/data';
import { both, compact, isNum, ms, num, pct, ratio, rps, secs, toMs, usd, usd2, utc, utcFull, hhmm } from '../lib/format';
import { gpuColor, useTokens } from '../lib/theme';
import { ANIM, Legend, Measured, TipBox, axisLabel, axisX, axisY, catAxis, gridProps, minBar, pointShape, refLabel, symbolOf, tip } from '../components/charts';
import { Card, Empty, Kpi, Meter, PageHead, Pill, RunLink, Segmented, Select, Sources, runParam } from '../components/ui';
import { useNavigate } from '@tanstack/react-router';
import { DataTable, numCol, type Col } from '../components/DataTable';

/* eslint-disable @typescript-eslint/no-explicit-any */
const pow2 = (lo: number, hi: number) => { const o: number[] = []; for (let v = 1; v <= hi * 1.01; v *= 2) if (v >= lo / 1.01) o.push(v); return o; };

// ================================================================ Spend
export function Spend() {
  const t = useTokens();
  const sp = B.spend;
  const [full, setFull] = useState<'auto' | 'full'>('auto');
  const next = sp.milestones.find(m => !m.crossed);
  const open = sp.sandboxes.filter((s: any) => s.open);
  const met = (sp.metered ?? []).filter((m: any) => isNum(m.metered_total_usd));
  const lastMet = met[met.length - 1];
  const snap = (sp as any).metered_snapshot;
  const tot = sp.total_est;
  const ymax = full === 'full' ? K.replan_usd * 1.06 : Math.max(tot * 1.15, (next ? next.usd : tot) * 1.12, 1);
  const data = sp.points.map(p => ({ t: toMs(p.t), usd: p.usd, live: p.live, iso: p.t }));
  const byGpu = groupBy(sp.sandboxes, (s: any) => (s.gpus > 1 ? `${s.gpus}x${s.gpu}` : s.gpu));
  const gpuRows = Object.entries(byGpu).map(([g, ss]) => ({ gpu: g, usd: ss.reduce((a: number, s: any) => a + s.est_usd, 0), n: ss.length, hours: ss.reduce((a: number, s: any) => a + s.hours, 0) })).sort((a, b) => b.usd - a.usd);
  return (
    <div>
      <PageHead title="Spend" actions={<Segmented value={full} onChange={setFull} options={[{ value: 'auto', label: 'Zoom' }, { value: 'full', label: `To $${K.replan_usd}` }]} />} />
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <Kpi label="Spend (estimate)" value={usd2(tot)} sub="list $/h × ledger wall time" sources={<Sources icon files={[sp.ledger_file]} />}>
          <Meter frac={tot / K.replan_usd} ticks={K.milestones.map(m => m / K.replan_usd)} />
        </Kpi>
        <Kpi label="Daytona metered" value={lastMet ? usd2(lastMet.metered_total_usd) : '—'} sub={lastMet ? `read ${utc(lastMet.t)}${snap ? ' · per-sandbox detail below' : ''}` : 'no reconcile yet'} sources={<Sources icon files={met.map((m: any) => m.file)} />} />
        <Kpi label="Next milestone" value={next ? `$${next.usd}` : '—'} sub={next ? `${usd2(next.usd - tot)} to go` : 'all crossed'}
          info="Crossings of $50/100/200/300/400 are reported to Maks without pausing; near $500 the plan is revisited." />
        <Kpi label="Open sandboxes" value={open.length} sub={open.length ? open.map((s: any) => s.gpu).join(', ') : 'none'} sources={<Sources files={[sp.ledger_file]} />} />
      </div>
      <Card className="mt-4" title="Running spend" info="Cumulative estimate from every create and delete in the ledger. Hairlines: reporting milestones; red dashed: the ~$500 re-plan line. Triangles mark milestone reports; diamonds Daytona's metered total."
        sources={<Sources files={[sp.ledger_file, ...met.map((m: any) => m.file)]} />}>
        {!data.length ? <Empty>The ledger has no sandboxes yet.</Empty> : (
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={data} margin={{ top: 10, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid {...gridProps(t)} />
              <XAxis dataKey="t" type="number" scale="time" domain={['dataMin', 'dataMax']} tickFormatter={v => hhmm(v)} {...axisX(t, { minTickGap: 40 })} />
              <YAxis domain={[0, +ymax.toFixed(2)]} tickFormatter={v => usd(v)} allowDataOverflow {...axisY(t)} />
              {sp.milestones.filter(m => m.usd <= ymax).map(m => <ReferenceLine key={m.usd} y={m.usd} stroke={m.kind === 'replan' ? t.crit : t.axis} strokeDasharray={m.kind === 'replan' ? '6 4' : undefined}
                label={refLabel(t, m.kind === 'replan' ? `re-plan $${m.usd}` : `$${m.usd}${m.crossed ? ' · ' + utc(m.crossed) : ''}`, 'insideTopLeft')} />)}
              <Tooltip content={tip((d: any) => d.metered ? <TipBox title={both(d.iso)} rows={[{ name: 'Daytona metered', value: usd2(d.usd), color: t.ref }]} foot={d.file} />
                : <TipBox title={both(d.iso)} rows={[{ name: 'cumulative estimate', value: usd2(d.usd), color: t.s1 }, { name: 'open sandboxes', value: d.live }]} foot={sp.ledger_file} />)} />
              <Area dataKey="usd" type="linear" stroke={t.s1} strokeWidth={2} fill={t.s1} fillOpacity={0.1} isAnimationActive={ANIM} dot={false} />
              {met.length > 0 && <Scatter data={met.map((m: any) => ({ t: toMs(m.t), usd: m.metered_total_usd, iso: m.t, file: m.file, metered: true }))} dataKey="usd" isAnimationActive={false}
                shape={(p: any) => <Symbols cx={p.cx} cy={p.cy} type="diamond" size={48} fill={t.ref} stroke={t.panel} strokeWidth={1.5} />} />}
            </ComposedChart>
          </ResponsiveContainer>
        )}
        {(sp.reported.length > 0 || met.length > 0) && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {sp.reported.map((m: any) => <Pill key={m.threshold_usd} tone="good" title={`${both(m.t)} · total ${usd2(m.total_usd)} (${m.basis})`}>${m.threshold_usd} reported</Pill>)}
            {lastMet && <Pill title={`${met.length} metered readings; latest ${both(lastMet.t)} · ${lastMet.file}`}>◆ metered {usd2(lastMet.metered_total_usd)} @ {utc(lastMet.t)} ({met.length} readings)</Pill>}
          </div>
        )}
      </Card>
      <Card className="mt-4" title="Spot evictions" info={B.evictions?.note ?? 'Spot preemptions, outside our control.'}
        sources={<Sources runs={(B.evictions?.events ?? []).map((e: any) => e.run_id)} files={[sp.ledger_file]} />}>
        {!(B.evictions?.events ?? []).length ? <Empty title="No evictions">No spot preemption recorded in the ledger.</Empty> : (
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
            <div className="xl:col-span-2">
              <DataTable data={B.evictions.by_gpu} search={false} maxHeight="none" columns={[
                { id: 'gpu', header: 'GPU', accessorFn: (g: any) => g.gpu, cell: c => <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-[3px]" style={{ background: gpuColor(t, c.getValue() as string) }} />{c.getValue() as string}</span> },
                numCol<any>('n', 'Evictions', g => g.count, v => num(v)),
                numCol<any>('min', 'Minutes lost', g => g.minutes_lost, v => num(v, 1)),
                numCol<any>('usd', '$ lost', g => g.usd_lost, usd2),
              ]} />
            </div>
            <div className="xl:col-span-3">
              <DataTable data={B.evictions.events} search={false} maxHeight={260} columns={[
                { id: 'run', header: 'Run', accessorFn: (e: any) => e.run_id, cell: c => <RunLink id={c.getValue() as string} /> },
                { id: 'at', header: 'Evicted', accessorFn: (e: any) => e.evicted_at, cell: c => utc(c.getValue() as string) },
                { id: 'step', header: 'Interrupted', accessorFn: (e: any) => e.interrupted ? `${e.interrupted.server} · ${e.interrupted.step}` : '–', meta: { mono: true } },
                numCol<any>('min', 'Min lost', e => e.minutes_lost, v => num(v, 2)),
                numCol<any>('usd', '$ lost', e => e.usd_lost, usd2),
                { id: 'rerun', header: 'Re-run', accessorFn: (e: any) => (e.rerun_run_ids ?? []).join(' '), cell: c => <div className="flex flex-col">{(c.row.original.rerun_run_ids ?? []).map((r: string) => <RunLink key={r} id={r} />)}</div> },
              ]} />
            </div>
          </div>
        )}
      </Card>
      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-5">
        <Card className="xl:col-span-2" title="By GPU" sources={<Sources files={[sp.ledger_file]} />}>
          {!gpuRows.length ? <Empty>No sandboxes yet.</Empty> : (
            <Measured>{w => { const ax = catAxis(w, gpuRows.map(r => r.gpu), { max: 100, font: 12 }); return (
            <ResponsiveContainer width="100%" height={Math.max(140, 40 + gpuRows.length * 34)}>
              <BarChart data={gpuRows} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 4 }}>
                <XAxis type="number" tickFormatter={v => usd(v)} {...axisX(t)} />
                <YAxis type="category" dataKey="gpu" {...axisY(t, { width: ax.width, tickFormatter: ax.clip, tick: { fill: t.ink2, fontSize: 12 } })} />
                <Tooltip cursor={{ fill: t.grid, opacity: 0.5 }} content={tip((d: any) => <TipBox title={d.gpu} rows={[{ name: 'estimate', value: usd2(d.usd) }, { name: 'sandboxes', value: d.n }, { name: 'hours', value: num(d.hours, 2) }]} />)} />
                <Bar dataKey="usd" barSize={16} radius={[0, 4, 4, 0]} isAnimationActive={ANIM} minPointSize={minBar}>
                  {gpuRows.map(r => <RCell key={r.gpu} fill={gpuColor(t, r.gpu)} />)}
                  <LabelList dataKey="usd" position="right" formatter={(v: any) => usd2(v)} fill={t.ink} fontSize={11.5} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>); }}</Measured>
          )}
        </Card>
        <Card className="xl:col-span-3" title="Sandboxes" sources={<Sources files={[sp.ledger_file]} />}>
          <DataTable data={[...sp.sandboxes].reverse()} maxHeight={320} columns={[
            { id: 'id', header: 'Sandbox', accessorFn: (s: any) => s.id, meta: { mono: true, hide: true } },
            { id: 'run', header: 'Run', accessorFn: (s: any) => s.run_id ?? s.run, cell: c => (c.row.original.run_id ? <RunLink id={c.row.original.run_id} /> : c.getValue()) },
            { id: 'gpu', header: 'GPU', accessorFn: (s: any) => s.gpu },
            { id: 'created', header: 'Created', accessorFn: (s: any) => s.created, cell: c => utc(c.getValue() as string) },
            { id: 'deleted', header: 'Deleted', accessorFn: (s: any) => s.deleted ?? '', cell: c => (c.getValue() ? utc(c.getValue() as string) : <Pill tone="live">open</Pill>) },
            numCol<any>('hours', 'Hours', s => s.hours, v => num(v, 2)),
            numCol<any>('rate', '$/h', s => s.usd_per_h, v => num(v, 3)),
            numCol<any>('usd', 'Est. $', s => s.est_usd, usd2),
            numCol<any>('met', 'Metered $', s => s.metered_usd, usd2),
            numCol<any>('meth', 'Metered $/h', s => s.metered_usd_per_h, v => num(v, 3), { meta: { num: true, hide: true } }),
            { id: 'why', header: 'Why deleted', accessorFn: (s: any) => s.why ?? '', meta: { hide: true } },
          ]} />
        </Card>
      </div>
    </div>
  );
}

// ================================================================ Startup
const inferQuant = (repo?: string) => { const r = (repo ?? '').toLowerCase(); for (const [k, v] of [['nvfp4', 'nvfp4'], ['mxfp4', 'mxfp4'], ['fp8', 'fp8'], ['awq', 'awq'], ['w4a16', 'w4a16'], ['gguf', 'gguf']]) if (r.includes(k)) return v; return 'bf16'; };
export function Startup() {
  const t = useTokens();
  const nav = useNavigate();
  const facts = B.startup ?? [];
  const factOf = (id: string) => facts.filter((f: any) => f.config_id === id).pop();
  const withData = new Set([...CELLS.map(c => c.config_id), ...RUNS.flatMap(r => r.config_ids ?? [])]);
  const cfgs = Object.values(CFG).sort((a, b) => (Number(!!factOf(b.config_id)) - Number(!!factOf(a.config_id))) || (Number(withData.has(b.config_id)) - Number(withData.has(a.config_id))) || a.config_id.localeCompare(b.config_id));
  const specTxt = (s: any) => (!s ? 'off' : typeof s === 'string' ? s : `${s.method ?? 'spec'}${s.num_speculative_tokens ? ' ×' + s.num_speculative_tokens : ''}`);
  const kv = facts.filter((f: any) => isNum(f.kv_cache_tokens));
  const gs = facts.filter((f: any) => Array.isArray(f.cuda_graph_sizes) && f.cuda_graph_sizes.length);
  const attn = (f: any, k: string) => { const be = f?.attention_backend?.[k]; if (!f) return <span className="text-muted">?</span>; if (!be) return <Pill tone="warn">missing</Pill>;
    return /TRITON/i.test(be) && k === 'global_512' ? <Pill tone="crit" title="PLAN §4 stop rule">{be}</Pill> : <span className="font-mono text-[11.5px]">{be}</span>; };
  return (
    <div>
      <PageHead title="Startup" actions={<><Pill tone="good">{facts.filter((f: any) => !f.rejected).length} clean</Pill><Pill tone="crit">{facts.filter((f: any) => f.rejected).length} rejected</Pill></>} />
      <Card title="Engine facts per config" info="Declared = the launch flags; observed = what the engine log said at startup (startup_facts, contract §4). A config is rejected when global_512 attention, KV capacity or graph sizes are missing, or when the global layers run on TRITON."
        sources={<Sources runs={facts.map((f: any) => f.run_id)} files={facts.map((f: any) => f.file)} />}>
        <DataTable data={cfgs} onRowClick={c => { const r = factOf(c.config_id)?.run_id; if (r) nav({ to: '/run/$runId', params: { runId: runParam(r) } }); }} columns={[
          { id: 'config', header: 'Config', accessorFn: (c: Config) => cfgShort(c.config_id) },
          { id: 'gpu', header: 'GPU', accessorFn: (c: Config) => c.gpu_key, cell: c => <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-[3px]" style={{ background: gpuColor(t, c.getValue() as string) }} />{c.getValue() as string}</span> },
          { id: 'status', header: 'Status', accessorFn: (c: Config) => { const f = factOf(c.config_id); return f ? (f.rejected ? 'rejected' : 'clean') : withData.has(c.config_id) ? 'missing' : 'not run'; },
            cell: c => { const v = c.getValue() as string; const f = factOf(c.row.original.config_id); return <Pill tone={v === 'clean' ? 'good' : v === 'rejected' ? 'crit' : v === 'missing' ? 'warn' : 'neutral'} title={f?.reject_reason ?? (f?.missing?.length ? 'missing: ' + f.missing.join(', ') : undefined)}>{v}</Pill>; } },
          { id: 'sliding', header: 'Sliding attn', accessorFn: (c: Config) => factOf(c.config_id)?.attention_backend?.sliding ?? '', cell: c => attn(factOf(c.row.original.config_id), 'sliding') },
          { id: 'global', header: 'Global-512 attn', accessorFn: (c: Config) => factOf(c.config_id)?.attention_backend?.global_512 ?? '', cell: c => attn(factOf(c.row.original.config_id), 'global_512') },
          numCol<Config>('kv', 'KV tokens', c => factOf(c.config_id)?.kv_cache_tokens, v => (isNum(v) ? compact(v) : '?')),
          numCol<Config>('kvg', 'KV GiB', c => factOf(c.config_id)?.kv_cache_gib, v => (isNum(v) ? num(v, 1) : '?'), { meta: { num: true, hide: true } }),
          numCol<Config>('conc', 'Conc. at max-len', c => factOf(c.config_id)?.max_concurrency_at_max_len, v => (isNum(v) ? num(v, 1) : '?')),
          { id: 'quant', header: 'Quant', accessorFn: (c: Config) => factOf(c.config_id)?.quantization?.weights ?? `${c.declared?.quantization ?? inferQuant(c.weights_repo)} (declared)` },
          { id: 'kvd', header: 'KV dtype', accessorFn: (c: Config) => factOf(c.config_id)?.kv_cache_dtype ?? c.declared?.kv_cache_dtype ?? 'auto', meta: { hide: true } },
          { id: 'mml', header: 'Max len', accessorFn: (c: Config) => factOf(c.config_id)?.max_model_len ?? c.declared?.max_model_len ?? '', meta: { hide: true } },
          { id: 'graphs', header: 'Graphs', accessorFn: (c: Config) => { const s = factOf(c.config_id)?.cuda_graph_sizes; return Array.isArray(s) ? `${s.length} · max ${Math.max(...s)}` : '?'; } },
          { id: 'spec', header: 'Spec decode', accessorFn: (c: Config) => (factOf(c.config_id) ? specTxt(factOf(c.config_id).spec_decode) : `${specTxt(c.declared?.spec)} (declared)`) },
          { id: 'weights', header: 'Weights', accessorFn: (c: Config) => c.weights_repo ?? '', meta: { mono: true, hide: true } },
          { id: 'run', header: 'Run', accessorFn: (c: Config) => factOf(c.config_id)?.run_id ?? '', cell: c => (c.getValue() ? <RunLink id={c.getValue() as string} /> : '') },
        ] as Col<Config>[]} />
      </Card>
      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title="KV-cache capacity" info={`Tokens of KV room after weights; the line is the prefill shot (16 × 10K = 160K tokens).`} sources={<Sources runs={kv.map((f: any) => f.run_id)} files={kv.map((f: any) => f.file)} />}>
          {!kv.length ? <Empty>Awaiting startup facts (kv_cache_tokens).</Empty> : (
            <Measured>{w => { const ax = catAxis(w, kv.map((f: any) => cfgShort(f.config_id)), { max: 130, font: 12 }); return (
            <ResponsiveContainer width="100%" height={Math.max(140, 50 + kv.length * 34)}>
              <BarChart data={kv.map((f: any) => ({ ...f, name: cfgShort(f.config_id) }))} layout="vertical" margin={{ top: 18, right: 48, bottom: 4, left: 4 }}>
                <CartesianGrid {...gridProps(t, true)} horizontal={false} />
                <XAxis type="number" tickFormatter={v => compact(v)} {...axisX(t)} />
                <YAxis type="category" dataKey="name" {...axisY(t, { width: ax.width, tickFormatter: ax.clip, tick: { fill: t.ink2, fontSize: 12 } })} />
                <ReferenceLine x={160000} stroke={t.ref} strokeDasharray="6 4" label={refLabel(t, '16 × 10K', 'top')} ifOverflow="extendDomain" />
                <Tooltip cursor={{ fill: t.grid, opacity: 0.5 }} content={tip((d: any) => <TipBox title={d.config_id} rows={[{ name: 'KV tokens', value: num(d.kv_cache_tokens) }, { name: 'KV GiB', value: num(d.kv_cache_gib, 1) },
                  { name: 'concurrent 9.9K requests', value: num(d.kv_cache_tokens / (K.shape_in + K.shape_out), 1) }]} foot={d.file} />)} />
                <Bar dataKey="kv_cache_tokens" barSize={16} radius={[0, 4, 4, 0]} isAnimationActive={ANIM} minPointSize={minBar}>
                  {kv.map((f: any) => <RCell key={f.config_id} fill={gpuColor(t, cfgOf(f.config_id).gpu_key)} fillOpacity={f.rejected ? 0.45 : 1} />)}
                  <LabelList dataKey="kv_cache_tokens" position="right" formatter={(v: any) => compact(v)} fill={t.ink} fontSize={11.5} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>); }}</Measured>
          )}
        </Card>
        <Card title="CUDA-graph capture sizes" info="Batch sizes captured as CUDA graphs. With MTP on, captures should reach 256." sources={<Sources runs={gs.map((f: any) => f.run_id)} files={gs.map((f: any) => f.file)} />}>
          {!gs.length ? <Empty>Awaiting startup facts (cuda_graph_sizes).</Empty> : (
            <Measured>{w => { const ax = catAxis(w, gs.map((f: any) => cfgShort(f.config_id)), { max: 170, font: 12 }); return (
            <ResponsiveContainer width="100%" height={Math.max(140, 50 + gs.length * 30)}>
              <ScatterChart margin={{ top: 8, right: 16, bottom: 18, left: 4 }}>
                <CartesianGrid {...gridProps(t, true)} />
                <XAxis type="number" dataKey="x" scale="log" domain={[1, 'dataMax']} ticks={pow2(1, Math.max(...gs.flatMap((f: any) => f.cuda_graph_sizes)))} {...axisX(t)} label={axisLabel(t, 'batch size', 'insideBottom', { offset: -12 })} />
                <YAxis type="category" dataKey="y" allowDuplicatedCategory={false} {...axisY(t, { width: ax.width, tickFormatter: ax.clip, tick: { fill: t.ink2, fontSize: 12 } })} />
                {gs.map((f: any) => <Scatter key={f.config_id} data={f.cuda_graph_sizes.map((s: number) => ({ x: s, y: cfgShort(f.config_id) }))} fill={gpuColor(t, cfgOf(f.config_id).gpu_key)} shape="circle" isAnimationActive={false} />)}
              </ScatterChart>
            </ResponsiveContainer>); }}</Measured>
          )}
        </Card>
      </div>
    </div>
  );
}

// ================================================================ MTP
export function Mtp() {
  const t = useTokens();
  const specOf = (id: string) => (B.startup ?? []).filter((f: any) => f.config_id === id).pop()?.spec_decode ?? cfgOf(id).declared?.spec ?? null;
  const loadX = (c: Cell) => (c.mode === 'closed' ? c.C : c.peak_inflight) as number;
  const acc = CELLS_K.filter(c => isNum(c.spec_accept_len));
  const ag = groupBy(acc, c => `${c.config_id}|${seriesOf(c)}`);
  const pairs: { c: Cell; b: Cell; x: number; speed: number; key: string }[] = [];
  for (const c of CELLS_K) {
    if (!specOf(c.config_id)) continue;
    const lab = cfgOf(c.config_id).label;
    const b = CELLS_K.find(x => x.cell === c.cell && x.config_id !== c.config_id && cfgOf(x.config_id).label === lab && !specOf(x.config_id) && x.out_tok_s > 0);
    if (b) pairs.push({ c, b, x: loadX(c), speed: c.out_tok_s / b.out_tok_s, key: `${c.config_id}|${seriesOf(c)}` });
  }
  const pg = groupBy(pairs, p => p.key);
  const widths = uniq(Object.keys(ag).map(k => specOf(k.split('|')[0])?.num_speculative_tokens)).filter(isNum);
  const xs = [...acc.map(loadX), ...pairs.map(p => p.x)].filter(isNum);
  const lo = Math.max(1, Math.min(...(xs.length ? xs : [1]))), hi = Math.max(...(xs.length ? xs : [2]));
  const style = (k: string, i: number) => { const cid = k.split('|')[0]; return { color: gpuColor(t, cfgOf(cid).gpu_key), sym: symbolOf(cid), dash: ['', '6 4', '2 3'][i % 3], label: `${cfgShort(cid)} · ${k.split('|')[1]}` }; };
  return (
    <div>
      <PageHead title="Speculative decoding" actions={<Pill>OpenRelay: MTP, 2 draft tokens</Pill>} />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title="Acceptance length vs concurrency" info="Tokens emitted per verify step with the drafter on (1 + accepted ÷ drafts). The ceiling is 1 + draft width."
          sources={<Sources runs={acc.map(c => c.run_id)} files={acc.map(c => c.file)} />}>
          {!acc.length ? <Empty>No cell has spec-decode counters yet.</Empty> : (
            <>
              <Legend items={Object.keys(ag).map((k, i) => ({ ...style(k, i), glyph: style(k, i).sym, key: k }))} />
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
                  <CartesianGrid {...gridProps(t)} />
                  <XAxis type="number" dataKey="x" scale="log" domain={[lo, hi]} ticks={pow2(lo, hi)} {...axisX(t)} label={axisLabel(t, 'concurrency', 'insideBottom', { offset: -12 })} />
                  <YAxis type="number" dataKey="y" domain={[0.8, Math.max(2, ...widths.map(w => 1 + w)) + 0.3]} {...axisY(t, { width: 40 })} />
                  <ReferenceLine y={1} stroke={t.axis} label={refLabel(t, 'no gain')} />
                  {widths.map(w => <ReferenceLine key={w} y={1 + w} stroke={t.ref} strokeDasharray="6 4" label={refLabel(t, `ceiling, ${w} drafts`)} />)}
                  <Tooltip cursor={false} content={tip((d: any) => <TipBox title={d.cell} rows={[{ name: 'acceptance length', value: num(d.y, 2) }, { name: 'acceptance rate', value: pct(d.spec_accept_rate) }, { name: 'out tok/s', value: compact(d.out_tok_s) }]} foot={`${d.run_id} · ${d.file}`} />)} />
                  {Object.keys(ag).map((k, i) => { const s = style(k, i); return <Scatter key={k} data={ag[k].map(c => ({ ...c, x: loadX(c), y: c.spec_accept_len })).sort((a, b) => a.x - b.x)} line={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} shape={pointShape(s.color, t.panel, s.sym)} isAnimationActive={ANIM} />; })}
                </ScatterChart>
              </ResponsiveContainer>
            </>
          )}
        </Card>
        <Card title="Where speculation stops paying" info="Output tok/s with the drafter ÷ the same cell without it, on the same box and config file. Above 1× it pays."
          sources={<Sources runs={pairs.map(p => p.c.run_id)} files={pairs.flatMap(p => [p.c.file, p.b.file])} />}>
          {!pairs.length ? <Empty>Needs spec-on and spec-off servers measured on the same cells.</Empty> : (
            <>
              <Legend items={Object.keys(pg).map((k, i) => ({ ...style(k, i), glyph: style(k, i).sym, key: k }))} />
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
                  <CartesianGrid {...gridProps(t)} />
                  <XAxis type="number" dataKey="x" scale="log" domain={[lo, hi]} ticks={pow2(lo, hi)} {...axisX(t)} label={axisLabel(t, 'concurrency', 'insideBottom', { offset: -12 })} />
                  <YAxis type="number" dataKey="speed" domain={[0, 'auto']} tickFormatter={v => ratio(v)} {...axisY(t, { width: 48 })} />
                  <ReferenceLine y={1} stroke={t.ref} strokeDasharray="6 4" label={refLabel(t, 'break-even')} />
                  <Tooltip cursor={false} content={tip((d: any) => <TipBox title={d.c.cell} rows={[{ name: 'ratio', value: ratio(d.speed) }, { name: 'spec on', value: compact(d.c.out_tok_s) }, { name: 'spec off', value: compact(d.b.out_tok_s) }]} foot={`${d.c.run_id}`} />)} />
                  {Object.keys(pg).map((k, i) => { const s = style(k, i); return <Scatter key={k} data={pg[k].slice().sort((a, b) => a.x - b.x).map(p => ({ ...p, pass: p.speed >= 1 }))} line={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} shape={pointShape(s.color, t.panel, s.sym)} isAnimationActive={ANIM} />; })}
                </ScatterChart>
              </ResponsiveContainer>
              <div className="mt-1 flex flex-wrap gap-1.5">{Object.keys(pg).map(k => { const c = pg[k].slice().sort((a, b) => a.x - b.x).find(p => p.speed < 1); return <Pill key={k} tone={c ? 'warn' : 'good'}>{cfgShort(k.split('|')[0])}: {c ? `stops paying at C=${c.x}` : 'pays at every load measured'}</Pill>; })}</div>
            </>
          )}
        </Card>
      </div>
      {(acc.length > 0 || pairs.length > 0) && (
        <Card className="mt-4" title="Spec-decode cells" sources={<Sources runs={acc.map(c => c.run_id)} files={acc.map(c => c.file)} />}>
          <DataTable data={acc} columns={[
            { id: 'cell', header: 'Cell', accessorFn: (c: Cell) => c.cell_id, meta: { mono: true } },
            { id: 'config', header: 'Config', accessorFn: (c: Cell) => cfgShort(c.config_id) },
            numCol<Cell>('load', 'Load', c => loadX(c), v => num(v)),
            numCol<Cell>('al', 'Accept len', c => c.spec_accept_len, v => num(v, 2)),
            numCol<Cell>('ar', 'Accept rate', c => c.spec_accept_rate, v => pct(v)),
            numCol<Cell>('tok', 'Out tok/s', c => c.out_tok_s, v => num(v)),
            numCol<Cell>('sp', 'vs spec off', c => pairs.find(p => p.c === c)?.speed, ratio),
            { id: 'run', header: 'Run', accessorFn: (c: Cell) => c.run_id, cell: c => <RunLink id={c.getValue() as string} /> },
          ] as Col<Cell>[]} />
        </Card>
      )}
    </div>
  );
}

// ================================================================ Compare
export function Compare() {
  const t = useTokens();
  const counts = groupBy(CELLS_K, c => c.config_id);
  const avail = Object.keys(counts).sort((a, b) => counts[b].length - counts[a].length);
  const [sel, setSel] = useState<string[]>(avail.slice(0, Math.min(3, avail.length)));
  if (!avail.length) return <div><PageHead title="Compare" /><Card><Empty>Needs measured cells.</Empty></Card></div>;
  const ids = uniq(sel);
  const best = (id: string) => (B.best ?? []).find(b => b.config_id === id);
  const prot = (id: string) => (B.protocols ?? []).filter((p: any) => p.config_id === id).pop();
  const rows: { m: string; f: (id: string) => any }[] = [
    { m: 'GPU', f: id => cfgOf(id).gpu_key }, { m: 'Box $/h', f: id => num(cfgOf(id).usd_per_h?.total, 3) },
    { m: 'Best $/req (agentic, SLO)', f: id => (best(id) ? `${usd(best(id)!.our_usd_req)} · ${ratio(best(id)!.ratio_sync)}` : '–') },
    { m: 'Max req/s at SLO', f: id => (prot(id) ? rps(prot(id).max_rate_at_slo_rps) : '–') },
    { m: 'GSM8K gate', f: id => { const g = gateOf(id); return g ? <Pill tone={g.passed ? 'good' : g.too_small ? 'warn' : 'crit'}>{pct(g.acc)} n={g.n}</Pill> : '–'; } },
    { m: 'Cells', f: id => (counts[id] ?? []).length }, { m: 'Weights', f: id => <span className="font-mono text-[11px]">{cfgOf(id).weights_repo}</span> },
  ];
  const shared = uniq(CELLS_K.filter(c => ids.includes(c.config_id)).map(c => c.cell)).filter(cl => ids.filter(id => CELLS_K.some(c => c.config_id === id && c.cell === cl)).length >= Math.min(2, ids.length));
  const cellOf = (id: string, cl: string) => CELLS_K.filter(c => c.config_id === id && c.cell === cl).pop();
  const metrics: [string, (c: Cell) => number | null | undefined, (v: number) => string][] = [['Output tok/s', c => c.out_tok_s, v => compact(v)], ['TTFT p95', c => c.ttft?.p95, secs], ['TPOT p50', c => c.tpot?.p50, ms], ['$/req vs Sync', c => c.ratio_sync, ratio]];
  return (
    <div>
      <PageHead title="Compare" actions={[0, 1, 2, 3].map(i => <Select key={i} label={`#${i + 1}`} value={sel[i] ?? ''} onChange={v => setSel(s => { const n = [...s]; n[i] = v; return n.filter(Boolean); })}
        options={[{ value: '', label: '—' }, ...avail.map(a => ({ value: a, label: `${cfgShort(a)} (${counts[a].length})` }))]} />)} />
      <Card title="Side by side" sources={<Sources runs={ids.flatMap(id => (counts[id] ?? []).map(c => c.run_id))} files={ids.map(id => cfgOf(id).config_file)} />}>
        <DataTable data={rows} search={false} maxHeight="none" columns={[
          { id: 'm', header: 'Metric', accessorFn: (r: any) => r.m, enableSorting: false, cell: c => <span className="text-ink-2">{c.getValue() as string}</span> },
          ...ids.map(id => ({ id, header: cfgShort(id), enableSorting: false, accessorFn: (r: any) => String(r.m), cell: (c: any) => c.row.original.f(id) })),
        ] as Col<any>[]} />
      </Card>
      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
        {metrics.map(([title, f, fmt]) => {
          const used = shared.filter(cl => ids.some(id => isNum(f(cellOf(id, cl) ?? ({} as Cell)))));
          return (
            <Card key={title} title={title} info="Cells measured by at least two of the selected configs." sources={<Sources runs={used.flatMap(cl => ids.map(id => cellOf(id, cl)?.run_id))} files={used.flatMap(cl => ids.map(id => cellOf(id, cl)?.file))} />}>
              {!used.length ? <Empty>No shared cells with this metric.</Empty> : (
                <>
                  <Legend items={ids.map(id => ({ label: cfgShort(id), color: gpuColor(t, cfgOf(id).gpu_key), glyph: symbolOf(id), key: id }))} />
                  <Measured>{w => { const ax = catAxis(w, used as string[], { max: 170, font: 11 }); return (
                  <ResponsiveContainer width="100%" height={Math.max(160, 50 + used.length * 26)}>
                    <ScatterChart margin={{ top: 4, right: 16, bottom: 4, left: 4 }}>
                      <CartesianGrid {...gridProps(t, true)} />
                      <XAxis type="number" dataKey="x" domain={['auto', 'auto']} tickFormatter={v => fmt(v)} {...axisX(t)} />
                      <YAxis type="category" dataKey="y" allowDuplicatedCategory={false} {...axisY(t, { width: ax.width, tickFormatter: ax.clip, tick: { fill: t.ink2, fontSize: 11, fontFamily: 'ui-monospace, Menlo, monospace' } })} />
                      <Tooltip cursor={false} content={tip((d: any) => <TipBox title={d.y} rows={[{ name: cfgShort(d.cid), value: fmt(d.x), color: gpuColor(t, cfgOf(d.cid).gpu_key) }]} foot={d.run_id} />)} />
                      {ids.map(id => <Scatter key={id} data={used.map(cl => { const c = cellOf(id, cl); return c && isNum(f(c)) ? { x: f(c), y: cl, cid: id, run_id: c.run_id } : null; }).filter(Boolean) as any[]}
                        shape={pointShape(gpuColor(t, cfgOf(id).gpu_key), t.panel, symbolOf(id), 80)} isAnimationActive={ANIM} />)}
                    </ScatterChart>
                  </ResponsiveContainer>); }}</Measured>
                </>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}

// ================================================================ Sources
export function SourcesPage() {
  const b = B.build;
  return (
    <div>
      <PageHead title="Sources" actions={<><Pill tone={b.demo ? 'warn' : 'good'}>{b.demo ? 'demo build' : 'real data'}</Pill><Pill>{b.contract_version}</Pill><Pill tone={B.warnings.length ? 'warn' : 'good'}>{B.warnings.length} warnings</Pill></>} />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card title="This build">
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-[12.5px] [&_dd]:[overflow-wrap:anywhere]">
            <dt className="text-muted">Built</dt><dd className="tnum">{utcFull(b.t)}</dd>
            <dt className="text-muted">Results</dt><dd className="font-mono text-[11.5px]">{b.results_dir}</dd>
            <dt className="text-muted">Contract</dt><dd className="font-mono text-[11.5px]">{b.contract_file}</dd>
            <dt className="text-muted">Prices</dt><dd className="font-mono text-[11.5px]">{b.cost_rows_file ?? 'results/cost_rows.json (missing)'}</dd>
            <dt className="text-muted">Ledger</dt><dd>{Object.entries(B.ledger_counts ?? {}).map(([k, v]) => `${k} ${v}`).join(' · ') || 'empty'}</dd>
            <dt className="text-muted">Rebuild</dt><dd className="font-mono text-[11.5px]">python3 -I dashboard/build.py</dd>
            <dt className="text-muted">Verify</dt><dd className="font-mono text-[11.5px]">python3 -I dashboard/check.py</dd>
          </dl>
        </Card>
        <Card className="xl:col-span-2" title="Warnings" info="Data-integrity checks the build ran. Each is a gap in the data, not in the dashboard.">
          {B.warnings.length ? <ul className="space-y-1 text-[12.5px] text-ink-2 [overflow-wrap:anywhere]">{B.warnings.map(w => <li key={w} className="flex gap-2"><span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-warn" />{w}</li>)}</ul> : <p className="text-[12.5px] text-muted">None.</p>}
        </Card>
      </div>
      <Card className="mt-4" title="Input files" info="Every file this build read, fingerprinted. If a number disagrees with a file, the hash says which version was read.">
        <DataTable data={B.inputs} maxHeight={520} columns={[
          { id: 'path', header: 'Path', accessorFn: (f: any) => f.path, meta: { mono: true } },
          numCol<any>('bytes', 'Bytes', f => f.bytes, v => num(v)),
          { id: 'mtime', header: 'Modified', accessorFn: (f: any) => f.mtime, cell: c => utcFull(c.getValue() as string) },
          { id: 'sha', header: 'sha256', accessorFn: (f: any) => f.sha256, cell: c => <span className="font-mono text-[11px] text-ink-2" title={c.getValue() as string}>{(c.getValue() as string).slice(0, 16)}…</span> },
        ]} />
      </Card>
      {B.contract && (
        <details className="mt-4 min-w-0 rounded-2xl border border-line bg-panel p-4">
          <summary className="cursor-pointer text-[13.5px] font-semibold">Data contract ({b.contract_file})</summary>
          <pre className="mt-3 max-h-[520px] overflow-auto whitespace-pre-wrap rounded-xl bg-panel-2 p-3 font-mono text-[11px] leading-relaxed text-ink-2">{B.contract}</pre>
        </details>
      )}
    </div>
  );
}
export const _unused = { REF, Area, toMs };
