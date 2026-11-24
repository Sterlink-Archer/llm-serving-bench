export const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);

export function usd(x: number | null | undefined): string {
  if (!isNum(x)) return '–';
  const a = Math.abs(x);
  if (a >= 100) return '$' + x.toLocaleString('en-US', { maximumFractionDigits: 0 });
  if (a >= 1) return '$' + x.toFixed(2);
  if (a >= 0.01) return '$' + x.toFixed(3);
  if (a === 0) return '$0';
  return '$' + Number(x.toPrecision(3)).toFixed(Math.max(0, 2 - Math.floor(Math.log10(a))));
}
export const usd2 = (x: number | null | undefined) => (isNum(x) ? '$' + x.toFixed(2) : '–');
export const ratio = (x: number | null | undefined) => (isNum(x) ? x.toFixed(x < 10 ? 2 : x < 100 ? 1 : 0) + '×' : '–');
export const num = (x: number | null | undefined, d = 0) =>
  isNum(x) ? x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }) : '–';
export const compact = (x: number | null | undefined) =>
  isNum(x) ? new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(x) : '–';
export const pct = (x: number | null | undefined, d = 1) => (isNum(x) ? (x * 100).toFixed(d) + '%' : '–');
export function secs(x: number | null | undefined): string {
  if (!isNum(x)) return '–';
  if (x === 0) return '0';
  if (x < 1) return (x * 1000).toFixed(x < 0.01 ? 2 : x < 0.1 ? 1 : 0) + ' ms';
  if (x < 60) return x.toFixed(2) + ' s';
  if (x < 3600) return (x / 60).toFixed(1) + ' min';
  return (x / 3600).toFixed(2) + ' h';
}
export const ms = (x: number | null | undefined) => (isNum(x) ? (x * 1000).toFixed(x < 0.1 ? 1 : 0) + ' ms' : '–');
export const rps = (x: number | null | undefined) => (isNum(x) ? x.toFixed(x < 10 ? 2 : 1) : '–');
export const toMs = (iso: string | null | undefined) => (iso ? Date.parse(iso) : NaN);
export const utc = (iso: string | null | undefined) => (iso ? iso.slice(11, 16) + 'Z' : '–');
export const utcFull = (iso: string | null | undefined) => (iso ? iso.replace('T', ' ').replace('Z', ' UTC') : '–');
const CT = new Intl.DateTimeFormat('en-US', { timeZone: 'America/Chicago', weekday: 'short', hour: 'numeric', minute: '2-digit' });
export const central = (iso: string | null | undefined) => (iso ? CT.format(new Date(iso)) + ' CT' : '–');
export const both = (iso: string | null | undefined) => (iso ? `${utc(iso)} · ${central(iso)}` : '–');
export const hhmm = (t: number) => new Date(t).toISOString().slice(11, 16);
export const signedPct = (x: number | null | undefined) => (isNum(x) ? (x >= 0 ? '+' : '') + (x * 100).toFixed(1) + '%' : '–');
