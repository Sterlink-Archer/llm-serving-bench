// Layout guard for check.py (added after the lead saw a collapsed price chart and colliding labels that every
// screenshot-by-eye and DOM check had passed). Real Chrome over CDP, true device metrics, every route:
//   * no two visible SVG <text> boxes intersect inside one chart, and none runs past its chart's edge;
//   * every Recharts plot area (its clipPath rect) is >= 40% of its card's width (a small multiple: its [data-chart-cell]);
//   * every drawn bar (path.recharts-rectangle with area) is >= 2 px on both sides;
//   * no KPI value is clipped (scrollWidth > clientWidth), and no token of one (a number) is split across two lines;
//   * the page has no horizontal overflow and every chart is drawn at width.
// Widths: 1366 and 446 (light), 390 phone (light and dark).
// Prints one JSON line per (width, theme, route).
// usage: node dashboard/layout_check.mjs <site dir> <cdp port> <route>...
const [site, port, ...routes] = process.argv.slice(2);
const t = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' })).json();
const ws = new WebSocket(t.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r, { once: true }));
let id = 0;
const pending = new Map();
ws.addEventListener('message', e => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const sleep = ms => new Promise(r => setTimeout(r, ms));

const PROBE = `(() => {
  const out = { scrollWidth: document.documentElement.scrollWidth, overlaps: [], plots: [], smallBars: [], kpiClipped: [], kpiBroken: [], textOut: [], chartWidths: [],
    nLabels: 0, nBars: 0, minBar: null, nKpi: document.querySelectorAll('[data-kpi-value]').length };
  const vis = el => { const r = el.getBoundingClientRect(); const cs = getComputedStyle(el); return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none' && +cs.opacity !== 0; };
  const titleOf = el => { const s = el.closest('section'); const h = s && s.querySelector('h2'); return (h ? h.textContent : (s ? 'card' : 'page')).trim().slice(0, 60); };
  document.querySelectorAll('main svg').forEach(svg => {
    if (svg.closest('.recharts-tooltip-wrapper')) return;
    const sr = svg.getBoundingClientRect();
    if (sr.width < 80 || sr.height < 50) return;               // icons and glyphs
    const title = titleOf(svg);
    const texts = [...svg.querySelectorAll('text')].filter(x => (x.textContent || '').trim() && vis(x)).map(x => ({ s: x.textContent.trim(), r: x.getBoundingClientRect() }));
    out.nLabels += texts.length;
    // a label running past its chart's edge is cut off (or spills over the card): it must sit inside the svg box
    for (const x of texts) if (x.r.left < sr.left - 1 || x.r.right > sr.right + 1) out.textOut.push({ chart: title, text: x.s.slice(0, 40), by: Math.round(Math.max(sr.left - x.r.left, x.r.right - sr.right)) });
    for (let i = 0; i < texts.length; i++) for (let j = i + 1; j < texts.length; j++) {
      const a = texts[i].r, b = texts[j].r;
      const ix = Math.min(a.right, b.right) - Math.max(a.left, b.left), iy = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
      if (ix > 1.5 && iy > 1.5) out.overlaps.push({ chart: title, a: texts[i].s.slice(0, 40), b: texts[j].s.slice(0, 40) });
    }
    const clip = svg.querySelector('defs clipPath rect');
    const card = svg.closest('[data-chart-cell], section');   // a small multiple's card is its own cell
    if (clip && card) {
      const pw = parseFloat(clip.getAttribute('width')), cw = card.getBoundingClientRect().width;
      if (cw > 0 && pw >= 0) out.plots.push({ chart: title, plot: Math.round(pw), card: Math.round(cw), ratio: +(pw / cw).toFixed(3) });
    }
    svg.querySelectorAll('path.recharts-rectangle').forEach(p => {
      const r = p.getBoundingClientRect();
      if (r.width > 0.01 && r.height > 0.01) { out.nBars++; const m = Math.min(r.width, r.height); out.minBar = out.minBar == null ? m : Math.min(out.minBar, m); }
      if (r.width > 0.01 && r.height > 0.01 && (r.width < 2 || r.height < 2)) out.smallBars.push({ chart: title, w: +r.width.toFixed(2), h: +r.height.toFixed(2), name: p.getAttribute('name') || '' });
    });
  });
  document.querySelectorAll('.recharts-responsive-container').forEach(e => { const w = e.querySelector('.recharts-wrapper'); out.chartWidths.push(w ? w.getBoundingClientRect().width : 0); });
  document.querySelectorAll('[data-kpi-value]').forEach(el => { if (el.scrollWidth > el.clientWidth + 1) out.kpiClipped.push({ kpi: (el.closest('section')?.querySelector('[data-kpi-label]')?.textContent || '').trim().slice(0, 40), text: el.textContent.trim().slice(0, 40) }); });
  // a token of a KPI value (e.g. "0.50×") laid out on two lines is as unreadable as a clipped one
  document.querySelectorAll('[data-kpi-value]').forEach(el => {
    const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    let n;
    while ((n = w.nextNode())) {
      const re = /\\S+/g; let m;
      while ((m = re.exec(n.data))) {
        const r = document.createRange(); r.setStart(n, m.index); r.setEnd(n, m.index + m[0].length);
        const tops = new Set([...r.getClientRects()].filter(q => q.width > 0).map(q => Math.round(q.top)));
        if (tops.size > 1) out.kpiBroken.push({ kpi: (el.closest('section')?.querySelector('[data-kpi-label]')?.textContent || '').trim().slice(0, 40), token: m[0] });
      }
    }
  });
  return JSON.stringify(out);
})()`;

await send('Page.enable');
// 1366 laptop, 446 the side-pane width where the lead saw labels collide, 390 phone (both themes)
for (const [w, h, mobile] of [[1366, 900, false], [446, 900, false], [390, 844, true]]) {
  await send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile });
  for (const theme of mobile ? ['light', 'dark'] : ['light']) {
    for (const r of routes) {
      await send('Page.navigate', { url: `file://${site}/index.html?theme=${theme}#/${r}` });
      await sleep(1600);
      const v = await send('Runtime.evaluate', { expression: PROBE, returnByValue: true });
      const o = JSON.parse(v.result.result.value);
      const cw = o.chartWidths;
      console.log(JSON.stringify({ width: w, theme, route: r || 'overview', scrollWidth: o.scrollWidth, charts: cw.length, minChartWidth: cw.length ? Math.min(...cw) : null,
        overlaps: o.overlaps, plots: o.plots, smallBars: o.smallBars, kpiClipped: o.kpiClipped, kpiBroken: o.kpiBroken, textOut: o.textOut, nLabels: o.nLabels, nBars: o.nBars, minBar: o.minBar, nKpi: o.nKpi }));
    }
  }
}
await fetch(`http://127.0.0.1:${port}/json/close/${t.id}`);
ws.close();
