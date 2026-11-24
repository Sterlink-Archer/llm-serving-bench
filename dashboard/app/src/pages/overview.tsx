import { useState, type ReactNode } from 'react';
import { Link } from '@tanstack/react-router';
import { B, BARS, BENCH_MODEL, COLUMN_NAME, K, RUNS, claimBars, claimsFor, evictions, headlineFor, stepsOf, type Claim, type Column, type Experiment } from '../lib/data';
import { isNum, num, ratio, rps, secs, signedPct, usd, usd2 } from '../lib/format';
import { gpuColor, useTokens } from '../lib/theme';
import { IntervalPlot, Legend, TipBox, type IRow } from '../components/charts';
import { Card, Empty, Kpi, Meter, PageHead, Pill, Segmented, Sources, runParam } from '../components/ui';
import { ClaimBars, GoodputPanel, JourneyChart, JourneyLegend, goodputPoints, journeyMetrics, type JMetric } from '../components/widgets';

/* eslint-disable @typescript-eslint/no-explicit-any */
// The Overview mirrors the write-up's headline order (docs/WRITEUP.md, 97ad059): H1 our deployment vs theirs on the same boxes,
// H2 our cost vs their API price (a different question), H3 host sizing (a Daytona lever), H4 scope. Every number is read from
// build output that dashboard/check.py re-derives from results/claims.json and results/protocol_rows.json.

/** a big number with its caption; data-kpi-value lets the layout guard catch a clipped or split value */
function Stat({ value, label, children }: { value: ReactNode; label: ReactNode; children?: ReactNode }) {
  return (
    <div className="min-w-0">
      <div data-kpi-value="" className="text-[26px] font-semibold leading-[1.15] tracking-[-0.02em]">{value}</div>
      <div className="mt-1 text-[12px] leading-snug text-muted">{label}</div>
      {children}
    </div>
  );
}
const Unit = ({ children }: { children: ReactNode }) => <span className="ml-1 text-[13px] font-medium text-muted">{children}</span>;
const Nw = ({ children, className }: { children: ReactNode; className?: string }) => <span className={'mr-3 inline-block whitespace-nowrap ' + (className ?? '')}>{children}</span>;

/** not_claim_grade_because, visible next to the ratio (never a bare number when the row is not claim-grade) */
function Reasons({ rows }: { rows: (Claim | undefined)[] }) {
  const items = rows.filter((r): r is Claim => !!r && !r.claim_grade && (r.not_claim_grade_because ?? []).length > 0);
  if (!items.length) return null;
  return (
    <ul className="mt-2 space-y-0.5 text-[11.5px] leading-snug text-muted" data-reasons="">
      {items.flatMap(r => (r.not_claim_grade_because ?? []).map((why, i) => <li key={r.column + i} className="flex gap-1.5"><span className="font-semibold text-warn-ink">{r.column}</span><span>{why}</span></li>))}
    </ul>
  );
}

// ---------------------------------------------------------------- H1: ours vs theirs on the same boxes
function SameBox() {
  const t = useTokens();
  const sb = (B as any).same_box?.[0];
  if (!sb) return <Card title="Ours vs theirs, same boxes"><Empty>No paired confirm (our recipe and OpenRelay's on the same boxes) in results/claims.json yet.</Empty></Card>;
  const ours = gpuColor(t, sb.gpu);
  const boxes: any[] = sb.boxes;
  const hi = Math.max(K.slo_ttft_p95_s, ...boxes.flatMap(b => [b.ours.ttft95, b.theirs.ttft95].filter(isNum))) * 1.12;
  const lab = (x: any) => (x.censored ? '≥ ' : '') + secs(x.ttft95);
  const rows: IRow[] = boxes.map(b => ({
    key: b.run_id, label: b.run_id.split('/')[0],
    spans: [{ lo: Math.min(b.ours.ttft95, b.theirs.ttft95), hi: Math.max(b.ours.ttft95, b.theirs.ttft95), color: t.axis }],
    marks: [{ v: b.ours.ttft95, color: ours, hollow: !b.ours.pass_, text: lab(b.ours), side: 'left' }, { v: b.theirs.ttft95, color: t.ref, hollow: !b.theirs.pass_, glyph: 'diamond', text: lab(b.theirs), side: 'right' }],
    tip: <TipBox title={`${b.run_id} · ${rps(sb.rate)} req/s`} rows={[{ name: 'ours, p95 TTFT', value: lab(b.ours), color: ours }, { name: 'theirs, p95 TTFT', value: lab(b.theirs), color: t.ref },
      { name: 'ours achieved', value: `${rps(b.ours.achieved)} req/s` }, { name: 'theirs achieved', value: `${rps(b.theirs.achieved)} req/s` }, { name: 'seed', value: String(b.ours.seed ?? '–') }]}
      foot={[b.ours.file, b.theirs.file].filter(Boolean).join(' · ')} />,
  }));
  const same = [sb.same_image && 'same image', sb.mml?.[0] && sb.mml[0] === sb.mml[1] && `max-model-len ${num(sb.mml[0])}`, "their client (A)"].filter(Boolean) as string[];
  return (
    <Card title={`Ours vs theirs · same ${boxes.length} ${sb.gpu} boxes`}
      info={<>Claim groups {sb.ours} (ours) and {sb.theirs} (OpenRelay's recipe), served back to back on each box: box-to-box clock spread cancels. The ladder steps by 0.25 req/s, so the throughput gap is step-quantised; the latency gap at one rate is the sharper evidence.</>}
      actions={<>{same.map(s => <Pill key={s}>{s}</Pill>)}</>}
      sources={<Sources runs={sb.runs} files={[B.claims?.file, ...sb.files]} />}>
      <div className="grid grid-cols-1 gap-x-8 gap-y-5 lg:grid-cols-5">
        <div className="grid grid-cols-1 content-start gap-5 sm:grid-cols-3 lg:col-span-2 lg:grid-cols-1">
          <Stat value={<><Nw>{rps(sb.ours_rate)}<Unit>ours</Unit></Nw> <Nw className="text-[20px] text-ink-2">{rps(sb.theirs_rate)}<Unit>theirs</Unit></Nw></>}
            label="req/s per GPU: the highest rate every box passes" />
          <Stat value={<Nw>{signedPct(sb.gain)}</Nw>} label={<>achieved at that rate, worst box: {num(sb.ours_achieved, 3)} vs {num(sb.theirs_achieved, 3)} req/s</>} />
          <Stat value={<><Nw>{isNum(sb.b.ours_rate) ? rps(sb.b.ours_rate) : '–'}<Unit>ours</Unit></Nw> <Nw className="text-[20px] text-ink-2">{isNum(sb.b.theirs_rate) ? rps(sb.b.theirs_rate) : 'fails'}<Unit>theirs</Unit></Nw></>}
            label={<>real text (B): {sb.b.theirs_rate == null ? `theirs: ${sb.b.theirs_because.join('; ') || 'no passing rate'}` : 'highest rate every seed passes'}</>} />
        </div>
        <div className="min-w-0 lg:col-span-3">
          <div className="mb-1 text-[12px] font-medium text-ink-2">p95 time to first token at {rps(sb.rate)} req/s, per box</div>
          <Legend items={[{ label: 'ours', color: ours }, { label: "theirs (OpenRelay's recipe)", color: t.ref, glyph: 'diamond' }]} />
          <IntervalPlot t={t} rows={rows} domain={[0, hi]} fmt={v => `${num(v, v < 10 && v % 1 ? 1 : 0)} s`} refs={[{ v: K.slo_ttft_p95_s, label: `SLO ${K.slo_ttft_p95_s} s`, color: t.crit }]}
            rowH={30} ariaLabel="p95 TTFT per box, ours vs theirs" />
          <p className="mt-1 text-[11.5px] text-muted">Hollow: misses the SLO.</p>
        </div>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------- H2: our cost vs their API price
function CostVsPrice({ className }: { className?: string }) {
  const bar = claimBars()[0];
  const [col, setCol] = useState<Column>('A');
  const hA = headlineFor('A', bar);
  const hB = headlineFor('B', bar);
  const batch = BARS.batch;
  const vsBatch = (u?: number | null) => (!isNum(u) ? null : Math.abs(u / batch - 1) < 0.01 ? 'ties Batch' : u < batch ? 'below Batch' : 'above Batch');
  return (
    <Card className={className} title="Our cost vs their API price"
      info={<>Daytona spot rental against OpenRelay's published per-token price, priced on the worst seed's achieved req/s (results/claims.json). Their price includes their margin, so this answers "is self-hosting cheaper than their API", not "is our deployment better than theirs".</>}
      actions={<><Pill>a different question</Pill><Segmented value={col} onChange={setCol} options={[{ value: 'A', label: 'A' }, { value: 'B', label: 'B' }]} /></>}
      sources={<Sources runs={[...(hA?.runs ?? []), ...(hB?.runs ?? [])]} files={[B.claims?.file, B.claims?.md_file, B.bars.openrelay.file]} />}>
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
        <Stat value={hA ? <Nw className={hA.claim_grade ? '' : 'text-ink-2'}>{ratio(hA.ratio)}</Nw> : '–'} label={<>{hA?.bar_short ?? 'OpenRelay'} · their client (A) · {usd(hA?.usd_per_req)}</>}>
          {hA && <div className="mt-1.5"><Pill tone={hA.claim_grade ? 'good' : 'warn'}>A {hA.claim_grade ? 'claim-grade' : 'not claim-grade'}</Pill></div>}
        </Stat>
        <Stat value={<Nw>{vsBatch(hA?.usd_per_req) ?? '–'}</Nw>} label={<>their Batch tier {usd(batch)} vs ours {usd(hA?.usd_per_req)} (A)</>} />
        <Stat value={hB ? <Nw className={hB.claim_grade ? '' : 'text-ink-2'}>{ratio(hB.ratio)}</Nw> : '–'} label={<>{hB?.bar_short ?? 'OpenRelay'} · real text (B) · {usd(hB?.usd_per_req)}</>}>
          {hB && <div className="mt-1.5"><Pill tone={hB.claim_grade ? 'good' : 'warn'}>B {hB.claim_grade ? 'claim-grade' : 'not claim-grade'}</Pill></div>}
        </Stat>
      </div>
      <Reasons rows={[hA, hB]} />
      <div className="mt-4 text-[12px] font-medium text-ink-2">$/request per claim group · column {col} ({COLUMN_NAME[col]})</div>
      <ClaimBars column={col} bar={bar} gradedOnly />
    </Card>
  );
}

// ---------------------------------------------------------------- H3: host sizing, a Daytona lever
function HostLever() {
  const lv = (B as any).host_levers?.[0];
  if (!lv) return null;
  const A = lv.cols.A, Bc = lv.cols.B;
  return (
    <Card title="Host sizing"
      info={<>The same recipe and image on a smaller Daytona host. Their recipe would get the same saving, so this is a lever of the rental, never a comparison with their recipe.</>}
      actions={<Pill>Daytona lever</Pill>}
      sources={<Sources runs={lv.runs} files={[B.claims?.file, ...lv.files]} />}>
      <Stat value={<>{A && <Nw>{ratio(A.ratio)}<Unit>A</Unit></Nw>} {Bc && <Nw className="text-[20px]">{ratio(Bc.ratio)}<Unit>B</Unit></Nw>}</>}
        label={<>{lv.group}: {lv.host[0]} vCPU / {lv.host[1]} GiB at {usd2(lv.usd_per_h)}/h, vs {lv.base_host[0]} / {lv.base_host[1]} at {usd2(lv.base_usd_per_h)}/h ({ratio(lv.base_cols.A?.ratio)} A, {ratio(lv.base_cols.B?.ratio)} B). × {A?.bar_short ?? 'the bar'}.</>}>
        <div className="mt-2 flex flex-wrap gap-1">
          {A && <Pill tone={A.claim_grade ? 'good' : 'warn'}>A {A.claim_grade ? 'claim-grade' : 'not claim-grade'}</Pill>}
          {Bc && <Pill tone={Bc.claim_grade ? 'good' : 'warn'}>B {Bc.claim_grade ? 'claim-grade' : 'not claim-grade'}</Pill>}
          <Pill tone="neutral">not a recipe comparison</Pill>
        </div>
      </Stat>
    </Card>
  );
}

// ---------------------------------------------------------------- H4: scope
function Scope() {
  const sc = (B as any).scope;
  if (!sc) return null;
  const n = (sc.recipes ?? []).length;
  return (
    <div className="mt-4 flex flex-wrap items-center gap-1.5" data-scope="">
      <span className="mr-1 text-[12px] font-medium text-muted">Scope</span>
      {sc.recipes_file && <Pill title={`OpenRelayInc/inference-recipes: ${(sc.recipes ?? []).join(', ')} (${sc.recipes_file})`}>OpenRelay published {n} recipe{n === 1 ? '' : 's'}: {(sc.recipes ?? []).join(', ')}</Pill>}
      <Pill tone={sc.long_context === 'done' ? 'good' : 'warn'} title={`${sc.api_context ? `Their API advertises ${sc.api_context} (${sc.pricing_file}); ` : ''}the recipes ran max-model-len ${num(sc.served_context)}`}>
        {sc.api_context ?? '256K'} context: {sc.long_context === 'done' ? 'measured' : 'PENDING'}</Pill>
      <Sources icon files={[sc.recipes_file, sc.pricing_file]} />
    </div>
  );
}

export function Overview() {
  const jsteps = stepsOf(BENCH_MODEL) as Experiment[];   // the headline model's journey only
  const jms = journeyMetrics(jsteps);
  const [jm, setJm] = useState<JMetric>(jms[0]?.key ?? 'usd_per_req');
  const [gcol, setGcol] = useState<Column>('A');
  const sp = B.spend;
  const met = (sp.metered ?? []).filter((m: any) => Number.isFinite(m.metered_total_usd)).pop();
  const st = (s: string) => RUNS.filter(r => r.status === s).length;
  const ev = evictions();
  const gp = goodputPoints();
  return (
    <div>
      <PageHead title="Overview" />
      <SameBox />
      <div className="mt-4 grid grid-cols-1 items-start gap-4 xl:grid-cols-3">
        <CostVsPrice className="xl:col-span-2" />
        <HostLever />
      </div>
      <Scope />

      <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-3">
        <Kpi label="Spend" value={usd2(met?.metered_total_usd ?? sp.total_est)} sub={met ? `metered · est ${usd2(sp.total_est)}` : `estimate · re-plan near $${K.replan_usd}`}
          info="List $/h × sandbox wall time from the ledger, reconciled with Daytona's metered usage when available. Ticks mark the $50–$400 reporting milestones."
          sources={<Sources icon files={[sp.ledger_file, met?.file]} />}>
          <Meter frac={(met?.metered_total_usd ?? sp.total_est) / K.replan_usd} ticks={K.milestones.map(m => m / K.replan_usd)} />
        </Kpi>
        <Kpi label="Runs" value={st('done')} sub={`${st('running')} running · ${st('error') + st('stale')} failed`} sources={<Sources icon runs={RUNS.map(r => r.run_id)} />} />
        <Kpi className="col-span-2 md:col-span-1" label="Evictions" value={ev} sub={ev ? 'excluded from every number' : 'spot preemptions'}
          info="Spot preemptions are outside our control: shown on their own and never folded into a cell or a $/request."
          sources={<Sources runs={(B.evictions?.events ?? []).map((e: any) => e.run_id)} files={[B.spend.ledger_file]} />}>
          {(B.evictions?.events ?? []).slice(0, 2).map((e: any) => (
            <Link key={e.run_id + e.t} to="/run/$runId" params={{ runId: runParam(e.run_id) }} className="mt-1.5 block truncate text-[12px] text-accent hover:underline">
              {e.run_id.split('/')[0]} · {String(e.evicted_at ?? e.t).slice(11, 16)}Z · {usd2(e.usd_lost)} lost
            </Link>
          ))}
        </Kpi>
      </div>

      <div className="mt-4 grid grid-cols-1 items-start gap-4 xl:grid-cols-2">
        <Card title="Goodput vs offered rate" info={<>Completed req/s per GPU while meeting the SLO; the lower panel is p95 TTFT against the {K.slo_ttft_p95_s} s line. Hollow points missed it. {gp.note}</>}
          actions={<Segmented value={gcol} onChange={setGcol} options={[{ value: 'A', label: 'A' }, { value: 'B', label: 'B' }]} />}
          sources={<Sources runs={gp.pts.map(p => p.run_id)} files={gp.pts.map(p => p.file)} note={gp.note} />}>
          <GoodputPanel height={220} compact column={gcol} />
        </Card>
        <Card title={(B.claims?.claims ?? []).some(r => r.origin === 'ours' && r.claim_grade && r.usd_per_req != null) ? 'Journey · ends at our recipe' : 'Journey · screening'} info="From the baseline config to the current best: one point per change, kept or reverted. Full detail on the Journey page."
          actions={jms.length > 1 ? <Segmented value={jm} onChange={setJm} options={jms.map(x => ({ value: x.key, label: x.label }))} /> : null}
          sources={<Sources runs={jsteps.flatMap(s => s.run_ids ?? [])} files={[B.journey?.file]} />}>
          <JourneyLegend />
          <JourneyChart metric={jm} height={220} model={BENCH_MODEL} />
        </Card>
      </div>
      {claimsFor('A').length === 0 && claimsFor('B').length === 0 && <p className="mt-4 text-[12px] text-muted">Awaiting results/claims.json.</p>}
    </div>
  );
}
