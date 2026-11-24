import * as Tooltip from '@radix-ui/react-tooltip';
import * as Popover from '@radix-ui/react-popover';
import { Link } from '@tanstack/react-router';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { FileText, Info, Link2 } from 'lucide-react';
import { cloneElement, isValidElement, type ReactElement, type ReactNode } from 'react';
import { uniq } from '../lib/data';

export const cn = (...xs: ClassValue[]) => twMerge(clsx(xs));
export const runParam = (runId: string) => runId.replace('/', '~');

export function InfoTip({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <Tooltip.Root delayDuration={120}>
      <Tooltip.Trigger asChild>
        <button type="button" aria-label="What is this?" className={cn('inline-flex text-muted hover:text-ink transition-colors', className)}>
          <Info className="size-3.5" strokeWidth={2} />
        </button>
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content sideOffset={6} collisionPadding={12}
          className="z-50 max-w-80 rounded-lg border border-line bg-panel px-3 py-2 text-[12.5px] leading-relaxed text-ink-2 shadow-lg">
          {children}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

export function Hint({ label, children }: { label: ReactNode; children: ReactNode }) {
  return (
    <Tooltip.Root delayDuration={150}>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content sideOffset={6} collisionPadding={12} className="z-50 max-w-80 rounded-lg border border-line bg-panel px-3 py-2 text-[12.5px] leading-relaxed text-ink-2 shadow-lg">
          {label}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

export function RunLink({ id, short }: { id: string; short?: boolean }) {
  return (
    <Link to="/run/$runId" params={{ runId: runParam(id) }} className="font-mono text-[11.5px] text-accent hover:underline" onClick={e => e.stopPropagation()}>
      {short ? id.split('/')[0] : id}
    </Link>
  );
}

/** Citation: every card names the run ids and files behind it. Visible as a popover; also present in the DOM for check.py. */
export function Sources({ runs = [], files = [], note, icon }: { runs?: (string | null | undefined)[]; files?: (string | null | undefined)[]; note?: ReactNode; icon?: boolean }) {
  const rs = uniq(runs);
  const fs = uniq(files);
  return (
    <>
      <Popover.Root>
        <Popover.Trigger asChild>
          <button type="button" className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11.5px] text-muted hover:bg-panel-2 hover:text-ink"
            aria-label={`Sources: ${rs.length} runs, ${fs.length} files`}>
            <Link2 className="size-3.5" />{icon ? null : ' '}{icon ? null : rs.length || fs.length ? `${rs.length ? rs.length + ' run' + (rs.length > 1 ? 's' : '') : ''}${rs.length && fs.length ? ' · ' : ''}${fs.length ? fs.length + ' file' + (fs.length > 1 ? 's' : '') : ''}` : 'no sources'}
          </button>
        </Popover.Trigger>
        <Popover.Portal>
          <Popover.Content align="end" sideOffset={6} collisionPadding={12}
            className="z-50 max-h-96 w-[min(560px,92vw)] overflow-auto rounded-xl border border-line bg-panel p-3 text-[12px] shadow-xl">
            {rs.length > 0 && <div className="mb-2"><div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">Runs</div>
              <ul className="space-y-0.5">{rs.map(r => <li key={r}><RunLink id={r} /></li>)}</ul></div>}
            {fs.length > 0 && <div><div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">Files</div>
              <ul className="space-y-0.5">{fs.map(f => <li key={f} className="flex gap-1.5 font-mono text-[11px] text-ink-2 break-all"><FileText className="mt-0.5 size-3 shrink-0" />{f}</li>)}</ul></div>}
            {note && <p className="mt-2 border-t border-line pt-2 text-ink-2">{note}</p>}
            {!rs.length && !fs.length && !note && <p className="text-muted">No source rows.</p>}
          </Popover.Content>
        </Popover.Portal>
      </Popover.Root>
      <span className="sr-only" data-sources="">{[...rs, ...fs].join(' ')}</span>
    </>
  );
}

export function Card({ title, info, sources, actions, children, className, bodyClass }: {
  title?: ReactNode; info?: ReactNode; sources?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; bodyClass?: string;
}) {
  return (
    <section className={cn('min-w-0 rounded-2xl border border-line bg-panel shadow-[0_1px_2px_rgba(0,0,0,0.03)]', className)}>
      {(title || actions || sources) && (
        <header className="flex flex-wrap items-center gap-x-2 gap-y-1.5 px-4 pt-3.5 pb-1">
          {title && <h2 className="min-w-0 truncate text-[13.5px] font-semibold tracking-[-0.005em]">{title}</h2>}
          {info && <InfoTip>{info}</InfoTip>}
          <div className="ml-auto flex flex-wrap items-center justify-end gap-1">{actions}{sources}</div>
        </header>
      )}
      <div className={cn('px-4 pb-4 pt-2', bodyClass)}>{children}</div>
    </section>
  );
}

type Tone = 'good' | 'warn' | 'crit' | 'live' | 'neutral';
const TONE: Record<Tone, string> = {
  good: 'text-good-ink before:bg-good', warn: 'text-warn-ink before:bg-warn', crit: 'text-crit-ink before:bg-crit',
  live: 'text-ink-2 before:bg-accent before:[animation:pulse-dot_1.6s_ease-in-out_infinite]', neutral: 'text-ink-2 before:bg-muted',
};
/** status pill: dot + label, never colour alone */
export function Pill({ tone = 'neutral', children, title }: { tone?: Tone; children: ReactNode; title?: ReactNode }) {
  const el = (
    <span className={cn("inline-flex h-6 items-center gap-1.5 whitespace-nowrap rounded-full border border-line bg-panel px-2.5 text-[11.5px] font-medium before:size-1.5 before:rounded-full before:content-['']", TONE[tone])}>
      {children}
    </span>
  );
  return title ? <Hint label={title}>{el}</Hint> : el;
}

export function Kpi({ label, value, sub, info, sources, accent, children, className }: {
  label: ReactNode; value: ReactNode; sub?: ReactNode; info?: ReactNode; sources?: ReactNode; accent?: string; children?: ReactNode; className?: string;
}) {
  return (
    <section className={cn('flex min-w-0 flex-col rounded-2xl border border-line bg-panel p-4', className)}>
      <div className="flex items-center gap-1.5 text-[12px] font-medium text-ink-2">
        {accent && <span className="size-2 rounded-[3px]" style={{ background: accent }} />}
        <span className="truncate" data-kpi-label="">{label}</span>{info && <InfoTip>{info}</InfoTip>}
        <span className="ml-auto">{isValidElement(sources) ? cloneElement(sources as ReactElement<{ icon?: boolean }>, { icon: true }) : sources}</span>
      </div>
      <div data-kpi-value="" className="mt-2 min-w-0 text-[26px] font-semibold leading-[1.15] tracking-[-0.02em]">{value}</div>
      {children}
      {sub && <div className="mt-1.5 truncate text-[12px] text-muted">{sub}</div>}
    </section>
  );
}

export function Empty({ title = 'Awaiting data', children }: { title?: ReactNode; children?: ReactNode }) {
  return (
    <div className="flex min-h-36 flex-col items-center justify-center gap-1 rounded-xl border border-dashed border-line-2 bg-panel-2/60 px-6 py-8 text-center">
      <div className="text-[13px] font-semibold text-ink">{title}</div>
      {children && <div className="max-w-md text-[12.5px] text-muted">{children}</div>}
    </div>
  );
}

export function Meter({ frac, ticks = [] }: { frac: number; ticks?: number[] }) {
  return (
    <div className="relative mt-3 h-1.5 overflow-hidden rounded-full bg-[color-mix(in_srgb,var(--s1)_16%,var(--panel-2))]" role="meter"
      aria-valuenow={Math.round(frac * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div className="h-full rounded-full bg-[var(--s1)]" style={{ width: `${Math.min(100, Math.max(0.6, frac * 100))}%` }} />
      {ticks.map(x => <span key={x} className="absolute -top-0.5 -bottom-0.5 w-0.5 bg-panel" style={{ left: `${x * 100}%` }} />)}
    </div>
  );
}

export function Segmented<T extends string>({ value, options, onChange }: { value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void }) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-panel-2 p-0.5">
      {options.map(o => (
        <button key={o.value} type="button" onClick={() => onChange(o.value)}
          className={cn('rounded-md px-2.5 py-1 text-[12px] font-medium text-ink-2 transition-colors', value === o.value && 'bg-panel text-ink shadow-sm')}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Select({ value, onChange, options, label }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[]; label: string }) {
  return (
    <label className="inline-flex items-center gap-2 text-[12px] text-ink-2">
      <span className="text-muted">{label}</span>
      <select value={value} onChange={e => onChange(e.target.value)} aria-label={label}
        className="h-8 max-w-[320px] rounded-lg border border-line bg-panel px-2 text-[12.5px] text-ink outline-none focus:ring-2 focus:ring-accent/30">
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}

export function PageHead({ title, children, actions }: { title: string; children?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end gap-3">
      <div className="min-w-0">
        <h1 className="text-[22px] font-semibold tracking-[-0.02em] [overflow-wrap:anywhere]">{title}</h1>
        {children && <p className="mt-0.5 text-[13px] text-muted">{children}</p>}
      </div>
      <div className="ml-auto flex flex-wrap items-center gap-2">{actions}</div>
    </div>
  );
}
