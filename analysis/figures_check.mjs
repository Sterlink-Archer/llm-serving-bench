// Layout guard for analysis/figures.py: renders each SVG inline in headless Chrome (CDP device metrics) at 800 and
// 390 px wide and fails on
//   * two <text> boxes that intersect;
//   * a <text> that runs past the figure's edge (cut off);
//   * a <text> that sits on a data mark (bar, point, line) it is not meant to be inside (data-in marks the ones that are).
// It also reports the smallest rendered font size per width (an SVG in <img> scales with the page; < 5 px fails).
// usage: node analysis/figures_check.mjs <svg>...   (FIG_SHOTS=<dir> also saves a 2x screenshot of each at 800 px)
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { createServer } from 'node:net';
import { basename } from 'node:path';

const files = process.argv.slice(2);
const port = await new Promise(r => { const s = createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => r(p)); }); });
const ch = spawn('taskset', ['-c', '8-15', 'nice', '-n', '19', 'chromium', '--headless=new', '--disable-gpu', '--no-sandbox',
  `--remote-debugging-port=${port}`, '--remote-debugging-address=127.0.0.1', '--user-data-dir=' + `/tmp/figcheck-${port}`, 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ver = null;
for (let i = 0; i < 100 && !ver; i++) { try { ver = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json(); } catch { await sleep(100); } }
if (!ver) { console.log('figure guard: chromium did not start'); ch.kill(); process.exit(2); }
const t = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' })).json();
const ws = new WebSocket(t.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r, { once: true }));
let id = 0;
const pending = new Map();
ws.addEventListener('message', e => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async expr => (await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })).result?.result?.value;

const PROBE = `(() => {
  const svg = document.querySelector('#c svg');
  const sr = svg.getBoundingClientRect();
  const vb = svg.viewBox.baseVal;
  const k = sr.width / vb.width;
  const out = { width: Math.round(sr.width), overlaps: [], textOut: [], onMark: [], nText: 0, nMark: 0, minFontPx: null };
  const texts = [...svg.querySelectorAll('text')].filter(x => (x.textContent || '').trim()).map(x => ({ el: x, s: x.textContent.trim(), r: x.getBoundingClientRect(), fs: parseFloat(x.getAttribute('font-size') || '12') }));
  out.nText = texts.length;
  for (const x of texts) { const px = x.fs * k; out.minFontPx = out.minFontPx == null ? px : Math.min(out.minFontPx, px); }
  for (const x of texts) if (x.r.left < sr.left - 0.5 || x.r.right > sr.right + 0.5 || x.r.top < sr.top - 0.5 || x.r.bottom > sr.bottom + 0.5)
    out.textOut.push(x.s.slice(0, 50));
  const inter = (a, b, m) => Math.min(a.right, b.right) - Math.max(a.left, b.left) > m && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > m;
  for (let i = 0; i < texts.length; i++) for (let j = i + 1; j < texts.length; j++)
    if (inter(texts[i].r, texts[j].r, 1)) out.overlaps.push([texts[i].s.slice(0, 40), texts[j].s.slice(0, 40)]);
  const marks = [...svg.querySelectorAll('[data-mark]')];
  out.nMark = marks.length;
  const pts = [];
  for (const m of marks) {
    if (m.dataset.mark !== 'line') continue;
    const ctm = m.getScreenCTM(); const len = m.getTotalLength(); const p = svg.createSVGPoint();
    for (let d = 0; d <= len; d += 1.5) { const q = m.getPointAtLength(d); p.x = q.x; p.y = q.y; const s = p.matrixTransform(ctm); pts.push(s); }
  }
  for (const x of texts) {
    if (x.el.hasAttribute('data-in')) continue;
    const r = { left: x.r.left + 1, right: x.r.right - 1, top: x.r.top + 1, bottom: x.r.bottom - 1 };
    for (const m of marks) {
      if (m.dataset.mark === 'line') continue;
      if (inter(r, m.getBoundingClientRect(), 0.5)) { out.onMark.push([x.s.slice(0, 40), m.dataset.mark]); break; }
    }
    if (pts.some(p => p.x > r.left && p.x < r.right && p.y > r.top && p.y < r.bottom)) out.onMark.push([x.s.slice(0, 40), 'line']);
  }
  out.minFontPx = +out.minFontPx.toFixed(2);
  return out;
})()`;

const shots = process.env.FIG_SHOTS;
if (shots) mkdirSync(shots, { recursive: true });
let checks = 0, failed = 0;
const minPx = {};
for (const f of files) {
  const svg = readFileSync(f, 'utf8');
  const dark = /-dark\.svg$/.test(f);
  for (const vw of [800, 390]) {
    await send('Emulation.setDeviceMetricsOverride', { width: vw, height: 900, deviceScaleFactor: 2, mobile: vw < 600 });
    await send('Page.navigate', { url: 'about:blank' });
    await sleep(80);
    const page = `<meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body{margin:0;background:${dark ? '#0d1117' : '#ffffff'}}#c{margin:16px;max-width:720px}#c svg{display:block;width:100%;height:auto}</style><div id="c">${svg}</div>`;
    await evaluate(`document.open();document.write(${JSON.stringify(page)});document.close();document.fonts.ready.then(()=>1)`);
    await sleep(60);
    const r = await evaluate(PROBE);
    const fails = [];
    if (r.overlaps.length) fails.push(`labels overlap: ${JSON.stringify(r.overlaps)}`);
    if (r.textOut.length) fails.push(`label cut off at the edge: ${JSON.stringify(r.textOut)}`);
    if (r.onMark.length) fails.push(`label on a data mark: ${JSON.stringify(r.onMark)}`);
    if (r.minFontPx < 5) fails.push(`smallest text renders at ${r.minFontPx} px (< 5)`);
    // without the viewport meta, mobile emulation lays out at 980 px and a "390 px" check silently measures 720 px
    if (r.width > vw - 16) fails.push(`figure laid out ${r.width} px wide in a ${vw} px viewport: emulation not applied`);
    checks += 5;
    failed += fails.length;
    minPx[vw] = Math.min(minPx[vw] ?? 99, r.minFontPx);
    console.log(JSON.stringify({ file: basename(f), vw, width: r.width, texts: r.nText, marks: r.nMark, minFontPx: r.minFontPx, ok: !fails.length, fails }));
    if (shots && vw === 800) {
      const box = await evaluate(`(() => { const b = document.querySelector('#c').getBoundingClientRect(); return { x: 0, y: 0, width: b.right + 16, height: b.bottom + 16 }; })()`);
      const shot = await send('Page.captureScreenshot', { format: 'png', clip: { ...box, scale: 1 } });
      writeFileSync(`${shots}/${basename(f, '.svg')}.png`, Buffer.from(shot.result.data, 'base64'));
    }
  }
}
console.log(`figure guard: ${files.length} SVGs x 2 widths (800, 390 px), ${checks} checks, ${checks - failed} passed, ${failed} failed; smallest text ${minPx[800]} px at 800, ${minPx[390]} px at 390`);
ws.close();
ch.kill();
process.exit(failed ? 1 : 0);
