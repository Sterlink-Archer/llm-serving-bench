import { useMemo, useState } from 'react';
import { CELLS, K, REF, cfgShort, loadOf, uniq, type Cell, type Dist } from '../lib/data';
import { isNum, ms, num, secs } from '../lib/format';
import { useTokens, type Tokens } from '../lib/theme';
import { FONT, TipBox, textW, useWidth } from '../components/charts';
import { Card, Empty, PageHead, Select, Sources } from '../components/ui';
import { DataTable, numCol, type Col } from '../components/DataTable';
import { useNavigate } from '@tanstack/react-router';
import { runParam } from '../components/ui';

type Metric = 'ttft' | 'tpot' | 'itl' | 'e2e';
const PCT = ['p50', 'p90', 'p95', 'p99'] as const;

/** percentile ladder per cell: p50→p99 dots on a span line (log scale when the range is wide) */
function RangeDots({ cells, metric, f, refv, refl, t }: { cells: Cell[]; metric: Metric; f: (v: number) => string; refv?: number; refl?: string; t: Tokens }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ c: Cell; y: number } | null>(null);
  const rowH = 26;
  const left = Math.min(220, Math.max(110, w * 0.34));
  const right = 16;
  const top = refv ? 18 : 6;
  const h = top + cells.length * rowH + 24;
  const vals = cells.flatMap(c => PCT.map(p => (c[metric] as Dist)?.[p])).filter((v): v is number => isNum(v) && v > 0);
  if (refv) vals.push(refv);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const log = hi / lo > 40;
  const [a, b] = log ? [Math.log10(lo * 0.8), Math.log10(hi * 1.25)] : [0, hi * 1.1];
  const x = (v: number) => left + ((log ? Math.log10(v) : v) - a) / (b - a) * Math.max(10, w - left - right);
  const ticks = log ? (() => { const o: number[] = []; for (let e = Math.floor(a); e <= Math.ceil(b); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (Math.log10(v) >= a && Math.log10(v) <= b) o.push(v); } return o; })()
    : Array.from({ length: 5 }, (_, i) => (b / 4) * i);
  // drop a tick whose label would touch the previous one (1-2-5 log ticks crowd at phone width)
  const shown: number[] = [];
  for (const v of ticks) { const last = shown[shown.length - 1]; if (last == null || x(v) - x(last) >= (textW(f(v)) + textW(f(last))) / 2 + 6) shown.push(v); }
  const nameChars = Math.max(6, Math.floor((left - 14) / 6.7));
  const refRight = refv != null && refl ? x(refv) + 4 + textW(refl) > w - 2 : false;
  const cols = [t.o1, t.o2, t.o3, t.o4];
  return (
    <div ref={ref} className="relative" onMouseLeave={() => setHover(null)}>
      {w > 0 && (
        <svg width={w} height={h} role="img" aria-label={`${metric} percentiles per cell`}>
          {shown.map(v => { const tw = textW(f(v)); const an = x(v) + tw / 2 > w - 1 ? 'end' : 'middle';
            return <g key={v}><line x1={x(v)} x2={x(v)} y1={top} y2={h - 20} stroke={t.grid} /><text x={an === 'end' ? w - 1 : x(v)} y={h - 6} fontSize={10.5} fill={t.muted} textAnchor={an} fontFamily={FONT}>{f(v)}</text></g>; })}
          {refv && <g><line x1={x(refv)} x2={x(refv)} y1={top - 4} y2={h - 20} stroke={metric === 'ttft' ? t.crit : t.ref} strokeDasharray="6 4" strokeWidth={1.25} />
            <text x={refRight ? x(refv) - 4 : x(refv) + 4} y={top - 6} fontSize={10.5} fill={t.ink2} fontFamily={FONT} textAnchor={refRight ? 'end' : 'start'}>{refl}</text></g>}
          {cells.map((c, i) => {
            const d = c[metric] as Dist;
            const y = top + i * rowH + rowH / 2;
            return (
              <g key={c.cell_id} onMouseEnter={() => setHover({ c, y })}>
                <rect x={0} y={y - rowH / 2} width={w} height={rowH} fill={hover?.c === c ? t.grid : 'transparent'} opacity={0.5} />
                <text x={left - 10} y={y + 4} fontSize={11} fill={t.ink2} textAnchor="end" fontFamily="ui-monospace, Menlo, monospace">{c.cell.length > nameChars ? c.cell.slice(0, nameChars - 1) + '…' : c.cell}</text>
                {d && isNum(d.p50) && isNum(d.p99) && <line x1={x(d.p50)} x2={x(d.p99)} y1={y} y2={y} stroke={t.axis} strokeWidth={2} strokeLinecap="round" />}
                {d && PCT.map((p, j) => (isNum(d[p]) ? <circle key={p} cx={x(d[p])} cy={y} r={4.5} fill={cols[j]} stroke={t.panel} strokeWidth={1.5} /> : null))}
              </g>
            );
          })}
        </svg>
      )}
      {hover && (
        <div className="absolute z-10" style={{ left: Math.max(0, Math.min(left, w - 300)), top: hover.y + 14 }}>
          <TipBox title={hover.c.cell} rows={[...PCT.map((p, j) => ({ name: p, value: f((hover.c[metric] as any)?.[p]), color: cols[j] })), { name: 'samples', value: num((hover.c[metric] as any)?.n) }, { name: 'load', value: loadOf(hover.c) }]}
            foot={`${hover.c.run_id} · ${hover.c.file}`} />
        </div>
      )}
    </div>
  );
}

export function Latency() {
  const t = useTokens();
  const nav = useNavigate();
  const runIds = uniq(CELLS.map(c => c.run_id));
  const byCount = [...runIds].sort((a, b) => CELLS.filter(c => c.run_id === b).length - CELLS.filter(c => c.run_id === a).length);
  const [run, setRun] = useState(byCount[0] ?? '');
  const [mat, setMat] = useState('');
  const mats = useMemo(() => uniq(CELLS.filter(c => c.run_id === run).map(c => c.matrix)), [run]);
  const cells = CELLS.filter(c => c.run_id === run && (!mat || c.matrix === mat));
  const metrics: [Metric, string, (v: number) => string, number | undefined, string | undefined][] = [
    ['ttft', 'Time to first token', secs, K.slo_ttft_p95_s, `SLO ${K.slo_ttft_p95_s} s`], ['tpot', 'Time per output token', ms, REF.tpot_ms / 1000, `OpenRelay ${REF.tpot_ms} ms`],
    ['itl', 'Inter-token latency', ms, undefined, undefined], ['e2e', 'End-to-end', secs, undefined, undefined]];
  const pc = (m: Metric, p: (typeof PCT)[number], f: (v: number | null | undefined) => string, hide = false) =>
    numCol<Cell>(`${m}${p}`, `${m.toUpperCase()} ${p}`, c => (c[m] as Dist)?.[p], f, { meta: { num: true, hide } });
  if (!runIds.length) return <div><PageHead title="Latency" /><Card><Empty>No cells yet.</Empty></Card></div>;
  return (
    <div>
      <PageHead title="Latency" actions={<>
        <Select label="Run" value={run} onChange={v => { setRun(v); setMat(''); }} options={runIds.map(r => ({ value: r, label: r }))} />
        <Select label="Matrix" value={mat} onChange={setMat} options={[{ value: '', label: `all (${mats.length})` }, ...mats.map(m => ({ value: m, label: m }))]} />
      </>} />
      <div className="mb-3 flex items-center gap-4 text-[12px] text-ink-2">
        {PCT.map((p, j) => <span key={p} className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-full" style={{ background: [t.o1, t.o2, t.o3, t.o4][j] }} />{p}</span>)}
      </div>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {metrics.map(([m, title, f, refv, refl]) => {
          const cs = cells.filter(c => c[m] && isNum((c[m] as Dist)!.p50));
          return (
            <Card key={m} title={title} info={`${m.toUpperCase()} p50, p90, p95, p99 per cell (steady window).${refl ? ' Reference: ' + refl + '.' : ''}`} sources={<Sources runs={[run]} files={cs.map(c => c.file)} />}>
              {cs.length ? <RangeDots cells={cs} metric={m} f={f} refv={refv} refl={refl} t={t} /> : <Empty>No samples in this selection.</Empty>}
            </Card>
          );
        })}
      </div>
      <Card className="mt-4" title="All percentiles" sources={<Sources runs={[run]} files={cells.map(c => c.file)} />}>
        <DataTable data={cells} onRowClick={c => nav({ to: '/run/$runId', params: { runId: runParam(c.run_id) } })} columns={[
          { id: 'cell', header: 'Cell', accessorFn: (c: Cell) => c.cell, meta: { mono: true } },
          { id: 'config', header: 'Config', accessorFn: (c: Cell) => cfgShort(c.config_id), meta: { hide: true } },
          { id: 'load', header: 'Load', accessorFn: (c: Cell) => loadOf(c) },
          numCol<Cell>('n', 'n', c => c.ttft?.n, v => num(v)),
          pc('ttft', 'p50', secs), pc('ttft', 'p95', secs), pc('ttft', 'p99', secs), pc('ttft', 'p90', secs, true),
          pc('tpot', 'p50', ms), pc('tpot', 'p95', ms), pc('tpot', 'p90', ms, true), pc('tpot', 'p99', ms, true),
          pc('itl', 'p50', ms), pc('itl', 'p99', ms), pc('itl', 'p90', ms, true), pc('itl', 'p95', ms, true),
          pc('e2e', 'p50', secs), pc('e2e', 'p95', secs), pc('e2e', 'p90', secs, true), pc('e2e', 'p99', secs, true),
          numCol<Cell>('err', 'Errors', c => c.errors, v => num(v)),
          { id: 'file', header: 'File', accessorFn: (c: Cell) => c.file, meta: { mono: true, hide: true } },
        ] as Col<Cell>[]} />
      </Card>
    </div>
  );
}
