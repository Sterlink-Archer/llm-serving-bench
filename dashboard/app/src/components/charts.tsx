import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Symbols } from 'recharts';
import type { Tokens } from '../lib/theme';
import { cfgOf } from '../lib/data';
import { both, secs, toMs, usd, hhmm } from '../lib/format';
import { cn } from './ui';

/* eslint-disable @typescript-eslint/no-explicit-any */
// headless Chrome never advances Recharts' bar animation under virtual time, so screenshots and check.py render static
// Animation off everywhere: a bar chart rendered empty for the lead (animation never completed); static renders are deterministic
export const ANIM = false;
export const FONT = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';

export const axisX = (t: Tokens, extra: any = {}) => ({
  tick: { fill: t.muted, fontSize: 11, fontFamily: FONT }, axisLine: { stroke: t.axis }, tickLine: false, tickMargin: 6, ...extra,
});
export const axisY = (t: Tokens, extra: any = {}) => ({
  tick: { fill: t.muted, fontSize: 11, fontFamily: FONT }, axisLine: false, tickLine: false, tickMargin: 4, width: 56, ...extra,
});
export const gridProps = (t: Tokens, vertical = false) => ({ stroke: t.grid, vertical, strokeDasharray: undefined });
export const axisLabel = (t: Tokens, value: string, position: string = 'insideBottom', extra: any = {}): any =>
  ({ value, position, fill: t.ink2, fontSize: 11, fontFamily: FONT, offset: -2, ...extra });
/** reference-line label stacked above the plot in rows, so close lines never print over each other */
export const stackLabel = (t: Tokens, value: string, row: number, anchor: 'start' | 'middle' | 'end' = 'middle'): any => (p: any) => {
  const vb = p?.viewBox ?? {};
  return <text x={vb.x} y={(vb.y ?? 0) - 6 - row * 15} textAnchor={anchor} fill={t.ink2} fontSize={10.5} fontFamily={FONT}>{value}</text>;
};
/** rough width of an 11 px-ish label (the layout guard in check.py measures the real boxes) */
export const textW = (s: string, font = 10.5) => s.length * font * 0.6 + 4;
/** greedy rows for labels at pixel x (sorted left to right): each goes in the lowest row where it clears its neighbours, and is
 *  anchored to stay inside [0, W] */
export function placeLabels(items: { x: number; text: string }[], W: number, font = 10.5) {
  const ends: number[] = [];
  return items.map(it => {
    const w = textW(it.text, font);
    const anchor: 'start' | 'middle' | 'end' = it.x + w / 2 > W - 2 ? 'end' : it.x - w / 2 < 2 ? 'start' : 'middle';
    const l = anchor === 'start' ? it.x : anchor === 'end' ? it.x - w : it.x - w / 2;
    let row = 0;
    while (ends[row] != null && ends[row] > l - 6) row++;
    ends[row] = l + w;
    return { row, anchor };
  });
}
/** category-axis width for a horizontal bar chart drawn at container width w: wide enough for the labels, never more than
 *  `frac` of the width (so the plot keeps the rest), and a clip to that many characters (full names stay in the tooltip) */
export function catAxis(w: number, labels: string[], { min = 64, max = 170, frac = 0.3, font = 11.5 } = {}) {
  const want = Math.max(0, ...labels.map(l => textW(l, font))) + 8;
  const width = Math.round(Math.max(min, Math.min(max, want, w * frac)));
  const chars = Math.max(5, Math.floor((width - 12) / (font * 0.64)));
  return { width, clip: (s: string) => (s.length > chars ? s.slice(0, chars - 1) + '…' : s) };
}
/** a single (unstacked) bar is never thinner than 2 px; zero stays zero */
export const minBar = (v: any) => (typeof v === 'number' && v > 0 ? 2 : 0);
/** stacked bars: a segment narrower than `minV` (about 3 px at this scale) is folded into `_short`, so no sliver is drawn; the
 *  exact values stay on the row under `_raw` for the tooltip. If the folded remainder is still under `minV`, it is not drawn. */
export function foldStack<T extends Record<string, any>>(rows: T[], keys: string[], minV: number) {
  return rows.map(r => {
    const o: Record<string, any> = { ...r, _raw: Object.fromEntries(keys.map(k => [k, r[k] ?? 0])), _short: 0, _shortKeys: [] as string[] };
    for (const k of keys) { const v = r[k]; if (typeof v === 'number' && v > 0 && v < minV) { o._short += v; o[k] = 0; o._shortKeys.push(k); } }
    if (o._short < minV) o._short = 0;
    return o as T & { _raw: Record<string, number>; _short: number; _shortKeys: string[] };
  });
}
/** the scale behind foldStack: value per pixel for a horizontal stacked chart (domain padded the way Recharts pads it) */
export const perPx = (rows: Record<string, any>[], keys: string[], plotW: number, extraMax = 0) =>
  (Math.max(1e-12, extraMax, ...rows.map(r => keys.reduce((a, k) => a + (typeof r[k] === 'number' ? r[k] : 0), 0))) * 1.5) / Math.max(40, plotW);
/** the folded-segment threshold: 4 px of geometry, so at least 2 px of colour shows inside the 2 px surface gap */
export const FOLD_PX = 4;

/** render-prop wrapper that hands its children the measured width (charts that size axes and labels to the card) */
export function Measured({ children, className }: { children: (w: number) => ReactNode; className?: string }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  return <div ref={ref} className={cn('w-full min-w-0', className)}>{w > 0 ? children(w) : null}</div>;
}
/** 1-2-5 ticks for a log axis */
export function logTicks(lo: number, hi: number) {
  const o: number[] = [];
  if (!(lo > 0) || !(hi > 0)) return o;
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) o.push(+v.toPrecision(3)); }
  return o;
}
export const refLabel = (t: Tokens, value: string, position: string = 'insideTopRight'): any =>
  ({ value, position, fill: t.ink2, fontSize: 11, fontFamily: FONT });

/** config marker: the config's fixed symbol within its GPU colour */
const SYM = ['circle', 'square', 'triangle', 'diamond', 'star', 'cross', 'wye'] as const;
export const symbolOf = (cid: string | null | undefined) => SYM[(cfgOf(cid).symbol_index ?? 0) % SYM.length];
export const GLYPH: Record<string, string> = { circle: '●', square: '■', triangle: '▲', diamond: '◆', star: '★', cross: '✚', wye: '✱' };

/** scatter point: filled when it met the SLO, hollow when it missed */
export function pointShape(color: string, surface: string, sym: string, size = 70) {
  return (p: any) => {
    const pass = p?.payload?.pass !== false;
    return <Symbols cx={p.cx} cy={p.cy} type={sym as any} size={size} fill={pass ? color : surface} stroke={pass ? surface : color} strokeWidth={pass ? 1.5 : 2} />;
  };
}

export type TipRow = { name: ReactNode; value: ReactNode; color?: string; dash?: boolean } | null | false | undefined;
export function TipBox({ title, rows, foot }: { title: ReactNode; rows: TipRow[]; foot?: ReactNode }) {
  return (
    <div className="pointer-events-none min-w-44 max-w-80 rounded-xl border border-line bg-panel px-3 py-2 text-[12px] shadow-xl">
      <div className="mb-1 font-semibold text-ink">{title}</div>
      {rows.filter(Boolean).map((r, i) => {
        const row = r as Exclude<TipRow, null | false | undefined>;
        return (
          <div key={i} className="flex items-center justify-between gap-4 leading-6">
            <span className="flex items-center gap-1.5 text-ink-2">
              {row.color && <i className="inline-block w-3 border-t-2" style={{ borderColor: row.color, borderStyle: row.dash ? 'dashed' : 'solid' }} />}
              {row.name}
            </span>
            <span className="font-semibold text-ink tnum">{row.value}</span>
          </div>
        );
      })}
      {foot && <div className="mt-1.5 border-t border-line pt-1.5 font-mono text-[10.5px] leading-snug text-muted break-all">{foot}</div>}
    </div>
  );
}
/** Recharts <Tooltip content={tip(d => ...)} />: render from the hovered datum */
export const tip = (render: (d: any, payload: any[]) => ReactNode) =>
  function Tip({ active, payload }: any) {
    if (!active || !payload?.length) return null;
    const d = payload[0]?.payload;
    if (!d) return null;
    return <>{render(d, payload)}</>;
  };

export function Legend({ items, className }: { items: { label: ReactNode; color: string; glyph?: string; line?: 'solid' | 'dashed' | 'dotted'; key?: string }[]; className?: string }) {
  if (items.length < 2) return null;
  return (
    <div className={cn('mb-2 flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-ink-2', className)}>
      {items.map((it, i) => (
        <span key={it.key ?? i} className="inline-flex items-center gap-1.5">
          {it.line ? <i className="inline-block w-4 border-t-2" style={{ borderColor: it.color, borderStyle: it.line }} />
            : it.glyph ? <span style={{ color: it.color }} className="text-[11px] leading-none">{GLYPH[it.glyph] ?? '●'}</span>
              : <span className="size-2.5 rounded-[3px]" style={{ background: it.color }} />}
          {it.label}
        </span>
      ))}
    </div>
  );
}

export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(es => setW(es[0].contentRect.width));
    ro.observe(ref.current);
    setW(ref.current.clientWidth);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/** Gantt of phases (custom SVG: Recharts has no time-range bars). Nested phases draw as a thin bar under their parent. */
export type GanttSeg = { row: number; t0: number; t1: number; color: string; nested?: boolean; label: string; dur: number; usd: number | null; source: string; tStart: string; tEnd: string; detail?: string };
export function Gantt({ rows, segs, t, rowH = 34, onRow }: { rows: string[]; segs: GanttSeg[]; t: Tokens; rowH?: number; onRow?: (i: number) => void }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ s: GanttSeg; x: number; y: number } | null>(null);
  const left = Math.min(150, Math.max(90, w * 0.18));
  const nameChars = Math.max(6, Math.floor((left - 14) / 7.6));
  const right = 12;
  const top = 6;
  const h = top + rows.length * rowH + 26;
  const tmin = Math.min(...segs.map(s => s.t0));
  const tmax = Math.max(...segs.map(s => s.t1));
  const span = Math.max(1, tmax - tmin);
  const x = (v: number) => left + ((v - tmin) / span) * Math.max(10, w - left - right);
  const step = [60e3, 120e3, 300e3, 600e3, 900e3, 1800e3, 3600e3, 7200e3, 10800e3, 21600e3].find(s => span / s <= Math.max(3, (w - left) / 90)) ?? 43200e3;
  const ticks: number[] = [];
  for (let v = Math.ceil(tmin / step) * step; v <= tmax; v += step) ticks.push(v);
  return (
    <div ref={ref} className="relative w-full" onMouseLeave={() => setHover(null)}>
      {w > 0 && segs.length > 0 && (
        <svg width={w} height={h} role="img" aria-label="Phase timeline">
          {ticks.map(v => { const atEnd = x(v) + textW(hhmm(v), 11) / 2 > w - 1;
            return <g key={v}><line x1={x(v)} x2={x(v)} y1={top} y2={h - 22} stroke={t.grid} /><text x={atEnd ? w - 1 : x(v)} y={h - 8} fontSize={11} fill={t.muted} textAnchor={atEnd ? 'end' : 'middle'} fontFamily={FONT}>{hhmm(v)}</text></g>; })}
          {rows.map((r, i) => (
            <text key={r} x={left - 10} y={top + i * rowH + rowH / 2 + 4} fontSize={11.5} fill={t.ink2} textAnchor="end" fontFamily={FONT}
              className={onRow ? 'cursor-pointer' : undefined} onClick={onRow ? () => onRow(i) : undefined}><title>{r}</title>{r.length > nameChars ? r.slice(0, nameChars - 1) + '…' : r}</text>
          ))}
          {segs.map((s, i) => {
            const bh = s.nested ? rowH * 0.16 : rowH * 0.5;
            const y = top + s.row * rowH + (rowH - rowH * 0.5) / 2 + (s.nested ? rowH * 0.56 : 0);
            return <rect key={i} x={x(s.t0)} y={y} width={Math.max(2, x(s.t1) - x(s.t0))} height={bh} rx={3} fill={s.color} stroke={t.panel} strokeWidth={1.5}
              onMouseMove={e => { const b = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect(); setHover({ s, x: e.clientX - b.left, y: e.clientY - b.top }); }} />;
          })}
        </svg>
      )}
      {hover && (
        <div className="absolute z-10" style={{ left: Math.min(hover.x + 12, Math.max(0, w - 300)), top: hover.y + 12 }}>
          <TipBox title={hover.s.label} rows={[{ name: 'duration', value: secs(hover.s.dur) }, { name: 'dollars', value: hover.s.usd == null ? 'not billed' : usd(hover.s.usd) },
            { name: 'start', value: both(hover.s.tStart) }, hover.s.detail ? { name: 'detail', value: hover.s.detail } : null]} foot={hover.s.source} />
        </div>
      )}
    </div>
  );
}
export const msOf = toMs;

/** Dot-and-whisker rows on one shared x scale: 95% intervals, deltas vs a reference, two recipes on one box. Custom SVG so the
 *  label gutter, tick count and value labels are sized to the card (the layout guard in check.py measures every text box). */
export type IMark = { v: number; color: string; hollow?: boolean; glyph?: 'circle' | 'diamond'; dy?: number; text?: string; side?: 'left' | 'right' };
export type IRow = { key: string; label: string; spans?: { lo: number; hi: number; color: string; dash?: string; dy?: number }[]; marks: IMark[]; tip: ReactNode };
export function IntervalPlot({ rows, t, domain, fmt, refs = [], rowH = 30, ariaLabel }: {
  rows: IRow[]; t: Tokens; domain: [number, number]; fmt: (v: number) => string; refs?: { v: number; label: string; color?: string; dash?: string }[]; rowH?: number; ariaLabel: string;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ r: IRow; y: number } | null>(null);
  const left = Math.round(Math.min(170, Math.max(84, w * 0.3)));
  const right = 14;
  const chars = Math.max(6, Math.floor((left - 14) / (11.5 * 0.64)));
  const plotW = Math.max(40, w - left - right);
  const [d0, d1] = domain;
  const x = (v: number) => left + ((Math.min(d1, Math.max(d0, v)) - d0) / (d1 - d0 || 1)) * plotW;
  const sRefs = [...refs].sort((a, b) => a.v - b.v);
  const place = placeLabels(sRefs.map(r => ({ x: x(r.v), text: r.label })), w);
  const top = sRefs.length ? 20 + 15 * Math.max(0, ...place.map(p => p.row)) : 8;
  const h = top + rows.length * rowH + 24;
  const span = d1 - d0 || 1;
  const e10 = 10 ** Math.floor(Math.log10(span / 4));
  const fit = Math.max(2, Math.floor(plotW / 64));
  const step = [1, 2, 2.5, 5, 10, 20].map(m => m * e10).find(s => span / s <= fit) ?? 20 * e10;
  const ticks: number[] = [];
  for (let v = Math.ceil(d0 / step - 1e-9) * step; v <= d1 + 1e-9; v += step) ticks.push(+v.toPrecision(6));
  // a value label sits past the end of its own interval (never on the whisker), flipping sides when it would leave the plot
  const markText = (m: IMark, cy: number, i: number, r: IRow) => {
    if (!m.text) return null;
    const tw = textW(m.text, 11);
    const same = (r.spans ?? []).filter(s => (s.dy ?? 0) === (m.dy ?? 0));
    const xr = Math.max(x(m.v), ...same.map(s => x(s.hi))) + 8, xl = Math.min(x(m.v), ...same.map(s => x(s.lo))) - 8;
    const fitsR = xr + tw <= w - 1, fitsL = xl - tw >= left - 4;
    const want = m.side ?? 'right';
    // preferred side past the interval; else the other side past the interval; else inside, next to the mark (halo keeps it legible)
    // with another mark on the line (a dumbbell), the far side belongs to that mark's label: fall back to inside, next to this mark
    const solo = r.marks.filter(o => (o.dy ?? 0) === (m.dy ?? 0)).length === 1;
    const [tx, anchor] = want === 'right' ? (fitsR ? [xr, 'start'] : solo && fitsL ? [xl, 'end'] : [x(m.v) - 9, 'end'])
      : (fitsL ? [xl, 'end'] : solo && fitsR ? [xr, 'start'] : [x(m.v) + 9, 'start']);
    return <text key={'t' + i} x={tx} y={cy + 4} textAnchor={anchor as 'start' | 'end'} fontSize={11} fill={t.ink2} fontFamily={FONT} paintOrder="stroke" stroke={t.panel} strokeWidth={3} strokeLinejoin="round">{m.text}</text>;
  };
  return (
    <div ref={ref} className="relative w-full" onMouseLeave={() => setHover(null)}>
      {w > 0 && (
        <svg width={w} height={h} role="img" aria-label={ariaLabel}>
          {ticks.map(v => { const tw = textW(fmt(v), 10.5); const an = x(v) + tw / 2 > w - 1 ? 'end' : x(v) - tw / 2 < left - 6 ? 'start' : 'middle';
            return <g key={v}><line x1={x(v)} x2={x(v)} y1={top} y2={h - 20} stroke={t.grid} /><text x={an === 'end' ? w - 1 : x(v)} y={h - 6} fontSize={10.5} fill={t.muted} textAnchor={an} fontFamily={FONT}>{fmt(v)}</text></g>; })}
          {sRefs.map((r, i) => <g key={r.label}><line x1={x(r.v)} x2={x(r.v)} y1={top - 4} y2={h - 20} stroke={r.color ?? t.ref} strokeDasharray={r.dash ?? '6 4'} strokeWidth={1.25} />
            <text x={x(r.v)} y={top - 8 - place[i].row * 15} textAnchor={place[i].anchor} fontSize={10.5} fill={t.ink2} fontFamily={FONT}>{r.label}</text></g>)}
          {rows.map((r, i) => {
            const y = top + i * rowH + rowH / 2;
            return (
              <g key={r.key} onMouseEnter={() => setHover({ r, y })}>
                <rect x={0} y={y - rowH / 2} width={w} height={rowH} fill={hover?.r === r ? t.grid : 'transparent'} opacity={0.5} />
                <text x={left - 10} y={y + 4} fontSize={11.5} fill={t.ink2} textAnchor="end" fontFamily={FONT}><title>{r.label}</title>{r.label.length > chars ? r.label.slice(0, chars - 1) + '…' : r.label}</text>
                {(r.spans ?? []).map((s, j) => <line key={'s' + j} x1={x(s.lo)} x2={x(s.hi)} y1={y + (s.dy ?? 0)} y2={y + (s.dy ?? 0)} stroke={s.color} strokeWidth={2} strokeDasharray={s.dash} strokeLinecap="round" />)}
                {r.marks.map((m, j) => {
                  const cx = x(m.v), cy = y + (m.dy ?? 0);
                  const fill = m.hollow ? t.panel : m.color;
                  return <g key={'m' + j}>{m.glyph === 'diamond'
                    ? <path d={`M${cx},${cy - 5.5}L${cx + 5.5},${cy}L${cx},${cy + 5.5}L${cx - 5.5},${cy}Z`} fill={fill} stroke={m.hollow ? m.color : t.panel} strokeWidth={m.hollow ? 2 : 1.5} />
                    : <circle cx={cx} cy={cy} r={5} fill={fill} stroke={m.hollow ? m.color : t.panel} strokeWidth={m.hollow ? 2 : 1.5} />}{markText(m, cy, j, r)}</g>;
                })}
              </g>
            );
          })}
        </svg>
      )}
      {hover && <div className="pointer-events-none absolute z-10" style={{ left: Math.max(0, Math.min(left, w - 300)), top: hover.y + 14 }}>{hover.r.tip}</div>}
    </div>
  );
}
