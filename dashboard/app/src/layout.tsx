import * as Dialog from '@radix-ui/react-dialog';
import { Link, Outlet, useRouterState } from '@tanstack/react-router';
import { BadgeCheck, BookOpen, Columns2, Cpu, DollarSign, FileCheck2, Footprints, Gauge, LayoutDashboard, ListTree, Monitor, Moon, Scale, Sun, Timer, Wallet, X, Zap } from 'lucide-react';
import type { ReactNode } from 'react';
import { B, BARS, K, REF } from './lib/data';
import { num, pct, usd, utcFull } from './lib/format';
import { useTheme } from './lib/theme';
import { cn } from './components/ui';

export const NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/journey', label: 'Journey', icon: Footprints },
  { to: '/paired', label: 'Ours vs theirs', icon: Scale },
  { to: '/quality', label: 'Quality', icon: BadgeCheck },
  { to: '/cost', label: 'Cost', icon: DollarSign },
  { to: '/goodput', label: 'Goodput', icon: Gauge },
  { to: '/latency', label: 'Latency', icon: Timer },
  { to: '/runs', label: 'Runs', icon: ListTree },
  { to: '/spend', label: 'Spend', icon: Wallet },
  { to: '/startup', label: 'Startup', icon: Cpu },
  { to: '/mtp', label: 'MTP', icon: Zap },
  { to: '/compare', label: 'Compare', icon: Columns2 },
  { to: '/sources', label: 'Sources', icon: FileCheck2 },
] as const;

function ThemeSwitch() {
  const { mode, setMode } = useTheme();
  const opts = [{ v: 'light', i: Sun }, { v: 'system', i: Monitor }, { v: 'dark', i: Moon }] as const;
  return (
    <div className="inline-flex rounded-lg border border-line bg-panel p-0.5" role="radiogroup" aria-label="Theme">
      {opts.map(o => (
        <button key={o.v} type="button" role="radio" aria-checked={mode === o.v} aria-label={o.v} onClick={() => setMode(o.v)}
          className={cn('rounded-md p-1.5 text-muted transition-colors hover:text-ink', mode === o.v && 'bg-panel-2 text-ink')}>
          <o.i className="size-3.5" />
        </button>
      ))}
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return <section className="border-t border-line py-4 first:border-t-0 first:pt-0"><h3 className="mb-1.5 text-[13px] font-semibold text-ink">{title}</h3><div className="space-y-1.5 text-[12.8px] leading-relaxed text-ink-2">{children}</div></section>;
}

export function MethodDrawer() {
  return (
    <Dialog.Root>
      <Dialog.Trigger asChild>
        <button type="button" className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 text-[12.5px] font-medium text-ink-2 hover:text-ink">
          <BookOpen className="size-3.5" /> Method
        </button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/30 backdrop-blur-[2px]" />
        <Dialog.Content className="fixed inset-y-0 right-0 z-50 w-[min(480px,100vw)] overflow-y-auto border-l border-line bg-panel p-6 shadow-2xl outline-none">
          <div className="mb-4 flex items-center">
            <Dialog.Title className="text-[16px] font-semibold">Method</Dialog.Title>
            <Dialog.Close className="ml-auto rounded-md p-1 text-muted hover:bg-panel-2 hover:text-ink" aria-label="Close"><X className="size-4" /></Dialog.Close>
          </div>
          <Dialog.Description className="sr-only">How every number on this dashboard is defined.</Dialog.Description>
          <Section title="The bar">
            <p>OpenRelay sells Gemma 4 31B at ${B.bars.openrelay.sync.in_usd_per_m} / 1M input and ${B.bars.openrelay.sync.out_usd_per_m} / 1M output (cached input ${B.bars.openrelay.sync.cached_in_usd_per_m}; Batch half). At their agentic mix of {num(K.shape_in)} in / {K.shape_out} out that is <b className="text-ink">{usd(BARS.sync)}</b> per request (Batch {usd(BARS.batch)}). The cheapest OpenRouter provider ({B.bars.openrouter.provider}) is the second reference at {usd(BARS.openrouter)}.</p>
            <p className="font-mono text-[11px] text-muted">{B.bars.openrelay.file} · {B.bars.openrouter.file}</p>
          </Section>
          <Section title="Cost per request">
            <p>Steady cost = box $/h (GPU + vCPU + RAM + disk at Daytona spot list) ÷ 3600 ÷ requests per second in the steady window: the box saturated at that load. Ratio &lt; 1 means cheaper than the bar; the ratio is also the break-even utilisation.</p>
            <p>Measured cost adds cold start (create, image pull, weights, compile, graph capture, warmup), evictions and failed creates, amortised over the run's steady requests. Prices come only from <code>results/cost_rows.json</code>.</p>
          </Section>
          <Section title="SLO and goodput">
            <p>OpenRelay's SLO: p95 time to first token &lt; {K.slo_ttft_p95_s} s, plus errors &lt; 1% and no client-side drops. Goodput counts completed requests per second only for cells that meet it. The headline req/s per GPU is the highest offered rate on the ladder whose worse seed (7 vs 8, 300 s each) meets the SLO. Theirs: {num(REF.req_s_per_gpu_at_slo, 1)} req/s on one MI355X, median TPOT {REF.tpot_ms} ms.</p>
          </Section>
          <Section title="When a price is quotable">
            <p>Only when the same checkpoint passed GSM8K ≥ {pct(K.gsm8k_gate)} on at least {K.gate_min_n} items, its startup facts are complete, and the global attention layers are not on TRITON. Ladders count only the kept (worse) seed. The headline is the cold {num(K.shape_in)}/{K.shape_out} mix; cached results never blend in.</p>
          </Section>
          <Section title="Reading the charts">
            <p>Colour is the GPU (fixed slot per type); marker shape is the config on that GPU; several series of one config differ by dash. Black dashed lines and the black diamond are OpenRelay's published numbers. Hollow points missed the SLO.</p>
          </Section>
          <Section title="Provenance">
            <p>Every card's link icon lists the run ids and files behind it. The Sources page fingerprints every file read (sha256). <code>python3 -I dashboard/check.py</code> re-derives every number from the source files and renders every page.</p>
            <p className="text-muted">Contract: {B.build.contract_file} ({B.build.contract_version}). Built {utcFull(B.build.t)}.</p>
          </Section>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export function Layout() {
  const path = useRouterState({ select: s => s.location.pathname });
  const active = (to: string) => (to === '/' ? path === '/' : path.startsWith(to) || (to === '/runs' && path.startsWith('/run/')));
  return (
    <div className="min-h-full">
      {B.build.demo && (
        <div className="bg-demo px-4 py-1.5 text-center text-[12px] text-ink" role="alert">
          <b>Demo build: mock data, not measurements.</b> Never quote a number from this build.
        </div>
      )}
      <div className="flex">
        <aside className="no-print sticky top-0 hidden h-screen w-56 shrink-0 flex-col border-r border-line bg-panel/60 px-3 py-4 lg:flex">
          <Link to="/" className="mb-6 flex items-center gap-2.5 px-2">
            <span className="size-6 rounded-lg bg-[linear-gradient(135deg,var(--s1)_0_50%,var(--s3)_50%_100%)]" aria-hidden />
            <span className="leading-tight"><span className="block text-[13.5px] font-semibold">Gemma 4 31B</span><span className="block text-[11.5px] text-muted">serving bench</span></span>
          </Link>
          <nav className="flex flex-col gap-0.5" aria-label="Dashboards">
            {NAV.map(n => (
              <Link key={n.to} to={n.to} className={cn('flex items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-[13px] font-medium text-ink-2 transition-colors hover:bg-panel-2 hover:text-ink',
                active(n.to) && 'bg-panel-2 text-ink')}>
                <n.icon className="size-4" strokeWidth={1.9} /> {n.label}
              </Link>
            ))}
          </nav>
          <div className="mt-auto px-2 text-[11px] leading-relaxed text-muted">
            <div>Built {B.build.t.slice(11, 16)}Z</div>
            <div>{B.inputs.length} files · {B.warnings.length} warnings</div>
          </div>
        </aside>
        <div className="min-w-0 flex-1">
          <header className="no-print sticky top-0 z-30 border-b border-line bg-bg/85 backdrop-blur-md">
            <div className="flex items-center gap-2 px-4 py-2.5 lg:px-8">
              <Link to="/" className="flex items-center gap-2 lg:hidden">
                <span className="size-5 rounded-md bg-[linear-gradient(135deg,var(--s1)_0_50%,var(--s3)_50%_100%)]" aria-hidden />
                <span className="text-[13.5px] font-semibold">Gemma 4 31B</span>
              </Link>
              <span className="hidden text-[12px] text-muted lg:inline">Self-hosted vs OpenRelay's {usd(BARS.sync)} per request</span>
              <div className="ml-auto flex items-center gap-2"><MethodDrawer /><ThemeSwitch /></div>
            </div>
            <nav className="scrollbar-none flex gap-1 overflow-x-auto px-3 pb-2 lg:hidden" aria-label="Dashboards">
              {NAV.map(n => (
                <Link key={n.to} to={n.to} className={cn('whitespace-nowrap rounded-full px-3 py-1 text-[12.5px] font-medium text-ink-2', active(n.to) && 'bg-panel text-ink shadow-sm ring-1 ring-line')}>{n.label}</Link>
              ))}
            </nav>
          </header>
          <main className="mx-auto max-w-[1400px] px-4 py-6 lg:px-8"><Outlet /></main>
        </div>
      </div>
    </div>
  );
}
