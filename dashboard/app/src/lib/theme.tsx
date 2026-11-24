import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { B } from './data';

type Mode = 'light' | 'dark' | 'system';
const NAMES = ['ink', 'ink-2', 'muted', 'line', 'grid', 'axis', 'ref', 'panel', 'panel-2', 'accent', 'good', 'warn', 'crit',
  's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 'o1', 'o2', 'o3', 'o4'] as const;
type Strip<S extends string> = S extends `${infer A}-${infer B}` ? `${A}${B}` : S;
export type Tokens = Record<Strip<(typeof NAMES)[number]>, string>;

const read = (): Tokens => {
  const cs = getComputedStyle(document.documentElement);
  return Object.fromEntries(NAMES.map(n => [n.replace('-', ''), cs.getPropertyValue('--' + n).trim()])) as Tokens;
};
const prefersDark = () => matchMedia('(prefers-color-scheme: dark)').matches;

const Ctx = createContext<{ mode: Mode; dark: boolean; t: Tokens; setMode: (m: Mode) => void }>(null!);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const initial = (() => {
    try { return (new URLSearchParams(location.search).get('theme') || localStorage.getItem('bench-theme') || 'system') as Mode; } catch { return 'system'; }
  })();
  const [mode, setModeState] = useState<Mode>(initial);
  const [sys, setSys] = useState(prefersDark());
  const dark = mode === 'dark' || (mode === 'system' && sys);
  const [t, setT] = useState<Tokens>(read);
  useEffect(() => {
    const mq = matchMedia('(prefers-color-scheme: dark)');
    const f = () => setSys(mq.matches);
    mq.addEventListener('change', f);
    return () => mq.removeEventListener('change', f);
  }, []);
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    setT(read());
  }, [dark]);
  const setMode = useCallback((m: Mode) => {
    setModeState(m);
    try { localStorage.setItem('bench-theme', m); } catch { /* storage blocked: lasts this page only */ }
  }, []);
  const v = useMemo(() => ({ mode, dark, t, setMode }), [mode, dark, t, setMode]);
  return <Ctx.Provider value={v}>{children}</Ctx.Provider>;
}
export const useTheme = () => useContext(Ctx);
export const useTokens = () => useContext(Ctx).t;

/** GPU colour: fixed categorical slot per GPU type (never cycled, never by rank). */
export const gpuColor = (t: Tokens, gk: string | null | undefined) => {
  const s = gk ? B?.gpus?.[gk]?.slot : null;
  return s ? (t as Record<string, string>)['s' + s] : t.muted;
};
export const catColor = (t: Tokens, c: string) => {
  const i = (B?.constants?.phase_cats ?? []).indexOf(c);
  return i >= 0 ? (t as Record<string, string>)['s' + (i + 1)] : t.muted;
};
