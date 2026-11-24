#!/usr/bin/env python3
"""Static figures for README.md and docs/WRITEUP.md: how our Gemma 4 31B recipe improves on OpenRelay's.

    python3 -I analysis/figures.py           # write docs/img/<name>-{light,dark}.svg and docs/img/figures.json
    python3 -I analysis/figures.py --check   # write, then render every SVG in headless Chrome at 800 and 390 px
                                              # (analysis/figures_check.mjs) and fail on overlapping or cut-off labels
    python3 -I analysis/figures.py --verify  # write nothing; exit 1 if docs/img differs from a fresh render (a claim
                                              # moved and the figures did not follow), for bin/test_all.sh

Style follows OpenRelay's blog figures (dark #09090b, one blue accent, direct labels, no legend boxes where a direct
label will do), but the figures are ours: OpenRelay's recipe is drawn in grey and ours in the accent. Fonts are the
system stack only, because an SVG loaded through <img> cannot fetch web fonts. Every number in a figure, its alt text
and its caption is computed here from the files below; none is typed. No run ids or hashes appear inside the figures
or captions: provenance is this docstring, and figures.json carries each figure's source files.

Sources, per figure (all paths relative to the repo root):
- knee: results/protocol_rows.json, column A (OpenRelay's client), claim groups mx_recipe_C1 (ours) and
  or_recipe_paired_C1 (theirs): the first paired confirm, journal step 122, runs mxC1_s7_r1, mxC1_s7_r2, mxC1_s8_r1,
  mxC1_s8_r2. Each box served both recipes back to back. Lines follow the worst box at each offered rate (the claim
  rule); dots are single boxes. Claim rates and worst-box achieved req/s come from results/claims.json.
- ablation: results/experiments.jsonl steps 100, 106, 110, 119 (config_id and run_ids) and 139, joined to
  results/protocol_rows.json at 3.5 req/s, column A. Steps 100-119 are one-box screens, seed 8, on different boxes
  (s2_mi355x_or_recipe_seed8, mx1_or_mtp3, mx5_or_mtp3_seqs96, mx10_best_dc_s8). Step 139 is the three-way confirm C2
  (claim group mx_stack_C2, 4 boxes); its bar is the worst of the four boxes.
- gpu-time: torch-profiler summaries results/<run>/<stamp>/prof_summary.txt of mx1_or_profile (their recipe, 3.25 req/s),
  mx7_prof_mtp3_seqs96 (k=3 + 96 seqs, 3.5 req/s, before the drafter fix) and mxP_ours_profile (our recipe C1 with the
  drafter fix, 3.5 req/s); see recipes/mi355x/notes/profile_knee_or_recipe.md and bf16_gemm_headroom.md. Shares are of
  all GPU kernel time in the 12 s load window. Inside the target forward, kernels are binned from the per-step lists of
  the mixed and decode-only sections (f4gemm -> MXFP4 GEMM; unified_attention / reduce_segments -> attention; the rest,
  including any unlisted residual, -> everything else). "Drafter and samplers" is the kernel time outside the target's
  step windows: the OUTSIDE line where the summary has one, else total minus the two step sections (mx1_or_profile).
- drafter-fix: an illustration, not a trace. Its two measured numbers come from mx7_prof_mtp3_seqs96's summary: the
  mixed steps' average prefill and decode tokens (rows through the drafter's first pass) and the prefill-shaped
  attention call outside the forward that runs once per mixed step (ms per call). Mechanism: recipes/mi355x/README.md,
  "The profile that found the drafter waste", and docs/WRITEUP.md "The find".
- price: results/claims.json rows (usd_per_req, ratio_vs_bar, price_bar_*) for the groups bin/build_readme.py quotes:
  Gemma mx_recipe_c4m24 A and mx_recipe_C1 B; Inkling ikC_b300_mixed_dspark and ikC_pro6000_mixed_dspark, A and B;
  GLM glm_best_C, A and B. OpenRelay's Gemma Batch tier comes from bench/pricebar.py (the newest
  research/openrelay_prices_*.json), as build_readme.py does.
"""
from __future__ import annotations

import glob
import html
import inspect
import json
import math
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/img"
sys.path.insert(0, str(ROOT / "bench"))
import pricebar as PB  # noqa: E402

W = 720
_shape = inspect.signature(PB.usd_per_req).parameters          # the request shape every $/request is priced on
IN_TOK, OUT_TOK = _shape["in_tok"].default, _shape["out_tok"].default
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
# a figure or caption must never carry a run id, a UTC stamp or a commit hash (provenance lives in figures.json)
LEAKS = [r"\b\d{8}T\d{6}Z\b", r"\b(?:mx|s\d|ik|glm|or)[A-Za-z0-9]*_[A-Za-z0-9_]+", r"\b[0-9a-f]{7,40}\b"]
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "dark": dict(bg="#09090b", border="#27272a", grid="#27272a", fg="#fafafa", muted="#a1a1aa", faint="#8a8a94",
                 theirs="#8a9099", theirs_bar="#8a9099", accent="#0065f4", accent_soft="#0065f4", seg_attn="#5c6370",
                 seg_else="#aeb3bb", seg_draft="#fafafa", on_dark="#ffffff", on_light="#09090b", empty="#3f3f46"),
    "light": dict(bg="#ffffff", border="#e4e4e7", grid="#e4e4e7", fg="#09090b", muted="#52525b", faint="#71717a",
                  theirs="#868c96", theirs_bar="#9aa0a9", accent="#0065f4", accent_soft="#0065f4", seg_attn="#5c6370",
                  seg_else="#c4c8ce", seg_draft="#18181b", on_dark="#ffffff", on_light="#09090b", empty="#d4d4d8"),
}


# ----------------------------------------------------------------------------------------------------------- data

def rel(p):
    return str(pathlib.Path(p).resolve().relative_to(ROOT))


def load():
    claims_f = ROOT / "results/claims.json"
    C = json.loads(claims_f.read_text())
    prow_f = ROOT / "results/protocol_rows.json"
    P = json.loads(prow_f.read_text())
    exp_f = ROOT / "results/experiments.jsonl"
    E = {}
    for line in exp_f.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            E[e["step"]] = e          # the last entry per step wins (re-recorded steps)
    return dict(claims={(c["group"], c["column"]): c for c in C["claims"]}, claims_doc=C, prow=P, exp=E,
                files=dict(claims=rel(claims_f), prow=rel(prow_f), exp=rel(exp_f)), slo_cache={})


def slo(D, rows):
    """The SLO the cells were judged on (protocol_summary.json slo.ttft_p95_s); every row must agree."""
    vals = set()
    for r in rows:
        f = r["summary_file"]
        if f not in D["slo_cache"]:
            D["slo_cache"][f] = json.loads((ROOT / f).read_text())["slo"]["ttft_p95_s"]
        vals.add(D["slo_cache"][f])
    if len(vals) != 1:
        raise SystemExit(f"figures: rows were judged on different SLOs: {sorted(vals)}")
    return vals.pop()


def words(n):
    return WORDS.get(n, str(n))


def all_boxes(n, digits=False):
    """'both boxes' / 'all four boxes' (or 'all 4 boxes' with digits)."""
    return "both boxes" if n == 2 else f"all {n if digits else words(n)} boxes"


def pooled_counts(D, ours, theirs, rates):
    """Per-box counts across every same-box test (results/box_counts.json, analysis/box_counts.py), column A."""
    f = ROOT / "results/box_counts.json"
    if not f.exists():
        return None
    C = json.loads(f.read_text())["counts"]
    fam = {}
    for who, g in (("ours", ours), ("theirs", theirs)):
        hit = [v for v in C.values() if v["column"] == "A" and g in v["groups"]]
        if len(hit) != 1:
            return None
        fam[who] = hit[0]
    out = {who: {float(k): c for k, c in v["rates"].items() if c.get("ran")} for who, v in fam.items()}
    return out, rel(f), {who: len(v["groups"]) for who, v in fam.items()}


def claim(D, g, col):
    r = D["claims"].get((g, col))
    if r is None:
        raise SystemExit(f"figures: results/claims.json has no row {g} {col}")
    return r


def prows(D, **eq):
    out = []
    for r in D["prow"]:
        if r.get("kept") is False:
            continue
        if all((abs((r.get(k) or 0) - v) < 1e-9) if isinstance(v, float) else r.get(k) == v for k, v in eq.items()):
            out.append(r)
    return out


def p95(r):
    v = r["ttft_p95_s"]
    if v is None or (isinstance(v, float) and math.isinf(v)):
        return r.get("ttft_p95_lower_s")
    return v


def profile(label):
    fs = sorted(glob.glob(str(ROOT / f"results/{label}/*/prof_summary.txt")))
    if not fs:
        raise SystemExit(f"figures: no prof_summary.txt for {label}")
    f = fs[-1]
    txt = pathlib.Path(f).read_text()
    load_part = txt.split("== load profile:", 1)[1]
    rate = float(re.match(r"\s*([\d.]+) req/s", load_part).group(1))
    tot = float(re.search(r"-- whole window\s*\ntotal GPU kernel time ([\d.]+) ms", load_part).group(1))
    steps_line = {m.group(1): m for m in re.finditer(
        r"^(mixed|decode-only)\s+steps\s+(\d+)\s+total\s+([\d.]+) ms\s+avg\s+([\d.]+) ms\s+avg ctx tok\s+(\d+)\s+avg gen reqs\s+([\d.]+)"
        r"\s+avg gen tok\s+([\d.]+)", load_part, re.M)}
    sec = {}
    for name, head in (("mixed", "-- mixed steps"), ("decode", "-- decode-only steps")):
        body = load_part.split(head, 1)[1]
        m = re.match(r"\s*\w[\w-]*: (\d+) steps, window ([\d.]+) ms, kernels ([\d.]+) ms", body)
        n, kern = int(m.group(1)), float(m.group(3))
        cats = dict(gemm=0.0, attn=0.0, other=0.0)
        listed = 0.0
        for line in body.splitlines()[1:]:
            if line.startswith("--") or line.startswith("=="):
                break
            km = re.match(r"\s+([\d.]+) ms/step x\s*([\d.]+)\s+(\S+)", line)
            if not km:
                continue
            ms = float(km.group(1)) * n
            k = km.group(3)
            c = "gemm" if "f4gemm" in k else "attn" if ("unified_attention" in k or "reduce_segments" in k) else "other"
            cats[c] += ms
            listed += ms
        cats["other"] += max(0.0, kern - listed)        # unlisted residual counts as everything else
        sec[name] = dict(n=n, kernels=kern, cats=cats, coverage=listed / kern if kern else 1.0)
    om = re.search(r"OUTSIDE ([\d.]+) ms", load_part)
    inside = sec["mixed"]["kernels"] + sec["decode"]["kernels"]
    outside = float(om.group(1)) if om else tot - inside
    mixed = steps_line["mixed"]
    first_pass = None                                   # the drafter's prefill-shaped first pass: one call per mixed step
    out_part = load_part.split("-- outside the target forward", 1)[1] if "-- outside the target forward" in load_part else ""
    for line in out_part.splitlines():
        km = re.match(r"\s+([\d.]+) ms\s+[\d.]+% x(\d+)\s+(\S+)", line)
        if km and "unified_attention_2d" in km.group(3) and "per_kv_8_BLOCK_SIZE_128" in km.group(3) \
                and int(km.group(2)) == int(mixed.group(2)):
            first_pass = dict(ms=float(km.group(1)), calls=int(km.group(2)))
    return dict(file=rel(f), rate=rate, total=tot, inside=inside, outside=outside, sec=sec,
                mixed=dict(steps=int(mixed.group(2)), ctx_tok=float(mixed.group(5)), gen_reqs=float(mixed.group(6)),
                           gen_tok=float(mixed.group(7))),
                first_pass=first_pass,
                shares=dict(gemm=(sec["mixed"]["cats"]["gemm"] + sec["decode"]["cats"]["gemm"]) / tot,
                            attn=(sec["mixed"]["cats"]["attn"] + sec["decode"]["cats"]["attn"]) / tot,
                            other=(sec["mixed"]["cats"]["other"] + sec["decode"]["cats"]["other"]) / tot,
                            draft=outside / tot))


# ----------------------------------------------------------------------------------------------------------- svg

def esc(s):
    return html.escape(str(s), quote=True)


FS = 1.2   # type scale over OpenRelay's sizes: an SVG in <img> shrinks with the page, so phones need the headroom (t037)


def tw(s, size, weight=400):
    """Conservative width of text at nominal `size` (DejaVu Sans is the widest common fallback); the guard measures."""
    return len(str(s)) * size * FS * (0.64 if weight >= 600 else 0.6)


class Svg:
    def __init__(self, h, T, alt):
        self.h, self.T, self.alt, self.parts = h, T, alt, []

    def c(self, k):
        return self.T.get(k, k)

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=12, fill="fg", anchor="start", weight=400, num=False, inside=False, halo=False, cls=None,
             plate=False):
        if plate:                                   # a background plate: a reference line behind the label stops at it
            w = tw(s, size, weight)
            x0 = x - (w / 2 if anchor == "middle" else w if anchor == "end" else 0)
            self.add(f'<rect x="{x0 - 3:.1f}" y="{y - size * FS:.1f}" width="{w + 6:.1f}" height="{size * FS + 5:.1f}" '
                     f'fill="{self.c("bg")}"/>')
        st = ["font-variant-numeric:tabular-nums"] if num else []
        if halo:
            st.append(f"paint-order:stroke;stroke:{self.c('bg')};stroke-width:4px;stroke-linejoin:round")
        a = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'fill="{self.c(fill)}"', f'font-size="{size * FS:.4g}"']
        if anchor != "start":
            a.append(f'text-anchor="{anchor}"')
        if weight != 400:
            a.append(f'font-weight="{weight}"')
        if st:
            a.append(f'style="{";".join(st)}"')
        if inside:
            a.append('data-in=""')
        if cls:
            a.append(f'class="{cls}"')
        self.add(f"<text {' '.join(a)}>{esc(s)}</text>")

    def line(self, x1, y1, x2, y2, stroke="grid", width=1, dash=None, opacity=None, mark=False):
        a = f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{self.c(stroke)}" stroke-width="{width}"'
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if opacity is not None:
            a += f' opacity="{opacity}"'
        if mark:
            a += ' data-mark="line"'
        self.add(f"<line {a}/>")

    def path(self, d, fill="none", stroke=None, width=None, mark=None, extra=""):
        a = f'd="{d}" fill="{self.c(fill)}"'
        if stroke:
            a += f' stroke="{self.c(stroke)}" stroke-width="{width or 1}" stroke-linejoin="round" stroke-linecap="round"'
        if mark:
            a += f' data-mark="{mark}"'
        self.add(f"<path {a}{extra}/>")

    def rect(self, x, y, w, h, fill, rx=0, mark=False, opacity=None, stroke=None, dash=None):
        a = f'x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" fill="{self.c(fill)}"'
        if rx:
            a += f' rx="{rx}"'
        if opacity is not None:
            a += f' fill-opacity="{opacity}"'
        if stroke:
            a += f' stroke="{self.c(stroke)}" stroke-width="1"'
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if mark:
            a += ' data-mark="bar"'
        self.add(f"<rect {a}/>")

    def bar(self, x0, y, x1, h, fill, r=4, tip=None):
        """Horizontal bar anchored at x0 with a rounded data end, as in OpenRelay's figures."""
        r = min(r, (x1 - x0) / 2, h / 2)
        d = (f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f} V{y + h - r:.1f} "
             f"A{r},{r} 0 0 1 {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z")
        t = f"<title>{esc(tip)}</title>" if tip else ""
        self.add(f'<path d="{d}" fill="{self.c(fill)}" data-mark="bar">{t}</path>')

    def point(self, x, y, fill, tip=None, r=4):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 2}" fill="{self.c("bg")}"/>')
        t = f"<title>{esc(tip)}</title>" if tip else ""
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{self.c(fill)}" data-mark="point">{t}</circle>')

    def dot(self, x, y, fill, tip=None):
        t = f"<title>{esc(tip)}</title>" if tip else ""
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{self.c(fill)}" fill-opacity="0.55" data-mark="point">{t}</circle>')

    def swatch(self, x, y, fill, label):
        self.rect(x, y - 10, 12, 12, fill, rx=3)
        self.text(x + 18, y, label, 12, "muted")

    def render(self):
        T = self.T
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {self.h}" width="{W}" height="{self.h}" '
                f'role="img" aria-label="{esc(self.alt)}" font-family="{esc(FONT)}">'
                f"<title>{esc(self.alt)}</title>"
                f'<rect x="0.5" y="0.5" width="{W - 1}" height="{self.h - 1}" rx="10" fill="{T["bg"]}" stroke="{T["border"]}"/>'
                + "".join(self.parts) + "</svg>\n")


def fmt_s(v):
    return f"{v:.2f} s" if v < 10 else f"{v:.1f} s"


def usd(v):
    return f"${v:.6f}"


def pct(v, nd=1):
    return f"{v * 100:.{nd}f}%"


# ----------------------------------------------------------------------------------------------------------- figures

def fig_knee(D, T, ours="mx_recipe_C1", theirs="or_recipe_paired_C1", test="the first paired test",
             later=("mx_recipe_C2", "mx_stack_C2"), ours_name="Our recipe", pooled=False):
    """`later`: (this recipe's group, the final recipe's group) in a later same-box test, for one honest sentence."""
    G = dict(ours=ours, theirs=theirs)
    data = {}
    for who, g in G.items():
        by = {}
        for r in prows(D, claim_group=g, column="A"):
            by.setdefault(r["offered_rps"], []).append(r)
        data[who] = dict(sorted(by.items()))
    cl = {who: claim(D, g, "A") for who, g in G.items()}
    SLO = slo(D, [r for d in data.values() for v in d.values() for r in v])
    win = int(min(r["scored_window_s"] for d in data.values() for v in d.values() for r in v)) // 60 * 60
    nseed = len({r["seed"] for d in data.values() for v in d.values() for r in v})
    for who in G:                                               # the figure must agree with the claims file
        c = cl[who]
        if not c["claim_grade"]:
            raise SystemExit(f"figures: {G[who]} A is not claim-grade")
        worst = max(p95(r) for r in data[who][c["max_rate_all_seeds_pass"]])
        if worst >= SLO or not all(r["slo_pass"] for r in data[who][c["max_rate_all_seeds_pass"]]):
            raise SystemExit(f"figures: {G[who]} claim rate has a failing box in protocol_rows")
    gain = cl["ours"]["worst_seed_achieved_rps"] / cl["theirs"]["worst_seed_achieved_rps"] - 1
    nbox = len({r["run_id"] for v in data["ours"].values() for r in v})
    rates = sorted({x for d in data.values() for x in d})
    vals = [p95(r) for d in data.values() for v in d.values() for r in v]
    ymin, ymax = 1.0, 2 ** math.ceil(math.log2(max(vals)))
    L, R, Tp, B = 64, 700, 100, 350
    lo, hi = rates[0] - 0.07 * (rates[-1] - rates[0]) * 2, rates[-1] + 0.07 * (rates[-1] - rates[0]) * 2

    def X(v):
        return L + (v - lo) / (hi - lo) * (R - L)

    def Y(v):
        return Tp + (math.log2(ymax) - math.log2(v)) / (math.log2(ymax) - math.log2(ymin)) * (B - Tp)

    r_o, r_t = cl["ours"]["max_rate_all_seeds_pass"], cl["theirs"]["max_rate_all_seeds_pass"]
    at35 = {who: max(p95(r) for r in data[who][r_o]) for who in G}
    alt = (f"p95 time to first token against offered load on the same {words(nbox)} MI355X boxes, OpenRelay's client. "
           f"OpenRelay's recipe stays under the {SLO:g} second target up to {r_t:.2f} requests per second on "
           f"{all_boxes(nbox)} of this test; "
           f"ours up to {r_o:.2f}, {pct(gain)} more completed requests per GPU.")
    s = Svg(412, T, alt)
    s.text(24, 26, "p95 time to first token (log scale)", 12.5, "fg", weight=600)
    t = 1
    while t <= ymax:
        y = Y(t)
        s.line(L, y, R, y)
        s.text(L - 10, y + 4, f"{t:g} s", 11.5, "faint", "end", num=True)
        t *= 2
    s.line(L, B, R, B)
    for v in rates:
        s.text(X(v), B + 20, f"{v:.2f}", 11.5, "faint", "middle", num=True)
    s.text((L + R) / 2, B + 42, "Requests per second, per GPU (offered)", 12, "muted", "middle")
    s.line(L, Y(SLO), R, Y(SLO), "fg", 1.2, "5 4", 0.7)
    s.text(L + 10, Y(SLO) - 8, f"target: {SLO:g} s", 12, "fg", weight=500, halo=True)
    dodge = dict(theirs=-6, ours=6)
    col = dict(theirs="theirs", ours="accent")
    name = dict(theirs="OpenRelay's recipe", ours=ours_name)
    # bracket: theirs' claim -> ours' claim, below the lowest dot
    ya = min(B - 10, max(Y(min(vals)) + 40, Y(1.6)))
    s.line(X(r_t) + dodge["theirs"], Y(max(p95(r) for r in data["theirs"][r_t])) + 6, X(r_t) + dodge["theirs"], ya + 6, "theirs")
    s.line(X(r_o) + dodge["ours"], Y(at35["ours"]) + 6, X(r_o) + dodge["ours"], ya + 6, "accent")
    s.line(X(r_t) + dodge["theirs"] + 2, ya, X(r_o) + dodge["ours"] - 6, ya, "fg", 1.4)
    ax = X(r_o) + dodge["ours"] - 4
    s.path(f"M{ax - 8:.1f},{ya - 4:.1f} L{ax:.1f},{ya:.1f} L{ax - 8:.1f},{ya + 4:.1f}", stroke="fg", width=1.4)
    s.text((X(r_t) + X(r_o)) / 2, ya - 9, f"+{gain * 100:.1f}% requests, same target", 13, "fg", "middle", 600, halo=True)
    for who in ("theirs", "ours"):
        d = data[who]
        pts = [(X(v) + dodge[who], Y(max(p95(r) for r in rs)), v, rs) for v, rs in d.items()]
        for v, rs in d.items():
            for r in rs:
                s.dot(X(v) + dodge[who], Y(p95(r)), col[who], f"{name[who]}, one box, {v:.2f} req/s: p95 TTFT {fmt_s(p95(r))}")
        s.path("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in pts), stroke=col[who], width=2, mark="line")
        for x, y, v, rs in pts:
            w = max(p95(r) for r in rs)
            s.point(x, y, col[who], f"{name[who]}, {v:.2f} req/s, worst of {len(rs)} boxes: p95 TTFT {fmt_s(w)}")
        x, y, _, _ = pts[-1]
        s.text(x + 12, y + 4, "Theirs" if who == "theirs" else "Ours", 12, "fg", weight=500, halo=True)
    s.swatch(24, 52, "theirs", f"{name['theirs']}: {r_t:.2f} req/s under target on {all_boxes(nbox, True)} of this test")
    s.swatch(24, 76, "accent", f"{name['ours']}: {r_o:.2f} req/s under target on {all_boxes(nbox, True)} of this test")
    ours_phrase = "ours" if ours_name == "Our recipe" else ours_name[0].lower() + ours_name[1:]
    cap = (f"In {test}, the same {words(nbox)} MI355X GPUs each served OpenRelay's recipe and {ours_phrase} back to back with "
           f"OpenRelay's own load generator ({win} s per rate, {words(nseed)} seeds, log scale). Lines follow the slowest box "
           f"at each rate, dots are single boxes. Ours holds the {SLO:g} s target at {r_o:.2f} req/s on {all_boxes(nbox)}, "
           f"theirs at {r_t:.2f}: {pct(gain)} more completed requests per GPU-hour. At {r_o:.2f} req/s the slowest box "
           f"answers in {fmt_s(at35['ours'])} with ours and {fmt_s(at35['theirs'])} with theirs.")
    src_later = []
    if later:                                   # what the same recipe did in the later test, so the claim is not overstated
        lr = prows(D, claim_group=later[0], column="A", offered_rps=r_o)
        fr = prows(D, claim_group=later[1], column="A", offered_rps=r_o)
        if lr and fr:
            nmiss = sum(not r["slo_pass"] for r in lr)
            fpass = sum(r["slo_pass"] for r in fr)
            if nmiss and fpass == len(fr):
                cap += (f" In a later test on {words(len(lr))} other boxes this recipe missed {r_o:.2f} req/s on "
                        f"{words(nmiss)} of them; the final recipe, with the persistent GEMM kernel, held it on all "
                        f"{words(len(fr))}.")
                src_later = sorted({r["file"] for r in lr + fr})
            elif nmiss:
                cap += f" In a later test this recipe missed {r_o:.2f} req/s on {words(nmiss)} of {words(len(lr))} boxes."
                src_later = sorted({r["file"] for r in lr})
    if pooled:                                  # a small test's gain must sit next to every same-box test's per-box counts
        pc = pooled_counts(D, ours, theirs, (r_o, r_t))
        if pc is None:
            raise SystemExit("figures: results/box_counts.json has no single column-A family for the knee's groups")
        cnt, f, ntests = pc
        o = cnt["ours"][r_o]
        th = [(r, c) for r, c in sorted(cnt["theirs"].items()) if r_t <= r <= r_o]
        cap += (f" Across every same-box test on their client, the final recipe held {r_o:.2f} req/s on {o['held']} of "
                f"{o['ran']} boxes; theirs held " + ", ".join(f"{r:.2f} on {c['held']} of {c['ran']}" for r, c in th[:-1])
                + (" and " if len(th) > 1 else "") + f"{th[-1][0]:.2f} on {th[-1][1]['held']} of {th[-1][1]['ran']}.")
        src_later.append(f)
    src = [D["files"]["prow"], D["files"]["claims"]] + src_later
    return s.render(), alt, cap, src


ABLATION = [(100, "OpenRelay's recipe"), (106, "+ 3 draft tokens instead of 2"), (110, "+ 96 sequences, graphs to 384"),
            (119, "+ drafter's first pass compacted"), (139, "+ persistent MXFP4 GEMM kernel")]


def fig_ablation(D, T):
    rate = claim(D, "mx_stack_C2", "A")["max_rate_all_seeds_pass"]      # the final recipe's claim rate
    rows, used = [], []
    for step, label in ABLATION:
        e = D["exp"].get(step)
        if e is None:
            raise SystemExit(f"figures: results/experiments.jsonl has no step {step}")
        server = e["config_id"].split(":", 1)[1]
        rs = [r for r in prows(D, column="A", offered_rps=rate, server=server) if r["run_id"] in e["run_ids"]]
        if step != 139:
            rs = [r for r in rs if r["seed"] == 8]
            if len(rs) != 1:
                raise SystemExit(f"figures: step {step} has {len(rs)} seed-8 rows at {rate} req/s (want 1)")
        if not rs:
            raise SystemExit(f"figures: step {step} has no rows at {rate} req/s")
        used += rs
        w = max(rs, key=p95)
        rows.append(dict(step=step, label=label, v=p95(w), n=len(rs), files=sorted({r["file"] for r in rs})))
    nconf = rows[-1]["n"]
    SLO = slo(D, used)
    L, X0, X1 = 258, 270, 652
    vmax = max(10.0, math.ceil(max(r["v"] for r in rows) / 2) * 2)

    def X(v):
        return X0 + v / vmax * (X1 - X0)

    top, rh, bh = 64, 32, 16
    h = top + rh * len(rows) + 44
    first_pass = next(i for i, r in enumerate(rows) if r["v"] < SLO)
    alt = (f"Cumulative ablation from OpenRelay's recipe, one change per row: p95 time to first token at {rate:g} requests "
           "per second on OpenRelay's client: " + "; ".join(f"{r['label'].lstrip('+ ')} {fmt_s(r['v'])}" for r in rows)
           + f". The target is {SLO:g} seconds; the first row under it is '{rows[first_pass]['label'].lstrip('+ ')}'.")
    s = Svg(h, T, alt)
    s.text(X0, 22, f"p95 time to first token at {rate:g} req/s", 12.5, "fg", weight=600)
    s.text(X0, 39, f"OpenRelay's load generator, {IN_TOK / 1000:g}K tokens in, {OUT_TOK} out", 11.5, "faint")
    yb = top + rh * len(rows) - 8
    for v in range(0, int(vmax) + 1, 2):
        s.text(X(v), yb + 20, f"{v} s", 11.5, "faint", "middle", num=True)
    s.line(X0, top - 10, X0, yb, "grid")
    s.line(X(SLO), top - 12, X(SLO), yb + 4, "fg", 1.2, "5 4", 0.7)
    s.text(X(SLO) + 6, top - 4, f"target {SLO:g} s", 11.5, "fg", weight=500, halo=True)
    prev = None
    for i, r in enumerate(rows):
        y = top + i * rh + 4
        s.text(L, y + 12, r["label"], 12, "fg", "end")
        fill = "theirs_bar" if i == 0 else "accent"
        tip = f"{r['label'].lstrip('+ ')}: p95 TTFT {fmt_s(r['v'])}" + (f", worst of {r['n']} boxes" if r["n"] > 1 else ", one box, seed 8")
        s.bar(X0, y, X(r["v"]), bh, fill, tip=tip)
        lab = fmt_s(r["v"]) + (f"  {(r['v'] / prev - 1) * 100:+.0f}%".replace("-", "−") if prev else "")
        under = r["v"] < SLO
        s.text(X(r["v"]) + 8, y + 12, lab, 11.5, "fg" if under else "muted", weight=600 if i == first_pass else 400,
               num=True, plate=True)
        prev = r["v"]
    cap = (f"Each row adds one of our changes to the row above, starting from OpenRelay's published recipe, measured at "
           f"{rate:g} req/s with their load generator. The first {words(len(rows) - 1)} rows are screens, one box each (seed 8), "
           f"on different boxes; the last row is the slowest of {words(nconf)} boxes in the paired confirm. The drafter fix "
           f"is the change that crosses the {SLO:g} s target.")
    src = sorted({f for r in rows for f in r["files"]}) + [D["files"]["exp"]]
    return s.render(), alt, cap, src


PROFILES = [("mx1_or_profile", "OpenRelay's recipe"), ("mx7_prof_mtp3_seqs96", "+ 3 drafts, 96 sequences"),
            ("mxP_ours_profile", "+ drafter fix (our recipe)")]
SEGS = [("gemm", "MXFP4 matrix multiplies", "accent", "on_dark"), ("attn", "Attention", "seg_attn", "on_dark"),
        ("other", "Everything else", "seg_else", "on_light"), ("draft", "Drafter and samplers", "seg_draft", None)]


def fig_gpu_time(D, T):
    P = [(profile(lab), name) for lab, name in PROFILES]
    for p, _ in P:
        for k, sec in p["sec"].items():
            if sec["coverage"] < 0.95:
                raise SystemExit(f"figures: {p['file']} {k} kernel list covers only {pct(sec['coverage'])}")
        if abs(p["inside"] + p["outside"] - p["total"]) > 0.02 * p["total"]:
            raise SystemExit(f"figures: {p['file']} inside + outside != total kernel time")
    L, X0, X1 = 222, 232, 572
    top, rh, bh = 116, 46, 22
    h = top + rh * len(P) + 14
    sh = [p["shares"] for p, _ in P]
    alt = ("Share of GPU kernel time in a 12-second load window at each recipe's knee: " + "; ".join(
        f"{name} at {p['rate']:g} req/s: matrix multiplies {pct(p['shares']['gemm'], 0)}, attention "
        f"{pct(p['shares']['attn'], 0)}, everything else {pct(p['shares']['other'], 0)}, drafter and samplers "
        f"{pct(p['shares']['draft'])}" for p, name in P) + ".")
    s = Svg(h, T, alt)
    s.text(24, 22, "Share of GPU time in a 12-second window at each recipe's knee", 12.5, "fg", weight=600)
    s.text(24, 39, "torch profile; the profiler slows the engine, so these are shares, not speeds", 11.5, "faint")
    x = 24
    for k, lab, fill, _ in SEGS:
        s.swatch(x, 66, fill, lab)
        x += 18 + len(lab) * 12 * FS * 0.56 + 20
    s.text(W - 16, top - 14, "drafter + samplers", 11.5, "faint", "end")
    for i, (p, name) in enumerate(P):
        y = top + i * rh
        s.text(L, y + 10, name, 12.5, "fg", "end", weight=500)
        s.text(L, y + 26, f"{p['rate']:g} req/s", 11.5, "faint", "end", num=True)
        x = X0
        tot = sum(p["shares"].values())
        for j, (k, lab, fill, ink) in enumerate(SEGS):
            w = p["shares"][k] / tot * (X1 - X0)
            last = j == len(SEGS) - 1
            gap = 0 if last else 2
            tip = f"{name}: {lab.lower()} {pct(p['shares'][k])} of GPU time"
            if last:
                s.bar(x, y, x + w, bh, fill, tip=tip)
            else:
                s.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w - gap, 0):.1f}" height="{bh}" fill="{s.c(fill)}" '
                      f'data-mark="bar"><title>{esc(tip)}</title></rect>')
            label = pct(p["shares"][k], 0)
            if ink and tw(label, 11.5) + 10 <= w - gap:
                s.text(x + (w - gap) / 2, y + 15.5, label, 11.5, ink, "middle", num=True, inside=True)
            x += w
        s.text(W - 16, y + 15.5, pct(p["shares"]["draft"]), 12.5, "fg", "end", weight=600 if i == len(P) - 1 else 400, num=True)
    d0, d1, d2 = (q["draft"] for q in sh)
    cap = (f"Where the GPU's time goes in a 12-second window at each recipe's knee, from a torch profile (the profiler slows "
           f"the engine, so these are shares, not speeds). The drafter and samplers run outside the target model's forward "
           f"pass; skipping the drafter's wasted first pass cut their share from {pct(d1)} to {pct(d2)} at the same load, "
           f"and the matrix multiplies (the MXFP4 GEMM kernels themselves), where a GPU should spend its time, rose to "
           f"{pct(sh[-1]['gemm'], 0)}.")
    return s.render(), alt, cap, [p["file"] for p, _ in P]


def fig_drafter(D, T):
    p = profile("mx7_prof_mtp3_seqs96")
    mx = p["mixed"]
    rows_in = mx["ctx_tok"] + mx["gen_tok"]
    per_req = round(mx["gen_tok"] / mx["gen_reqs"])
    fp = p["first_pass"]
    if not fp:
        raise SystemExit("figures: mx7 profile has no once-per-mixed-step first-pass attention call outside the forward")
    fp_ms = fp["ms"] / fp["calls"]
    ours = profile("mxP_ours_profile")
    if ours["first_pass"]:
        raise SystemExit("figures: our recipe's profile still shows the drafter's prefill-shaped first pass")
    alt = (f"Illustration, not a trace. Before: in a step that carries a prefill chunk, the drafter's first pass runs over "
           f"every token in the step, about {rows_in / 1000:.1f} thousand rows, and keeps one row per request. After: it runs "
           f"on one token per request.")
    s = Svg(272, T, alt)
    s.text(24, 26, "ILLUSTRATION, NOT A TRACE", 11, "faint")
    cw, cg = 9, 2
    x0 = 190
    n_chunk, n_req = 12, 4

    def strip(y, after):
        xs = []
        x = x0
        groups = [("chunk", n_chunk)] + [("req", per_req)] * n_req
        spans = []
        for gi, (kind, n) in enumerate(groups):
            gx = x
            for ci in range(n):
                kept = ci == n - 1
                if kind == "chunk" and ci == n // 2:
                    s.text(x + 6, y + 10, "···", 11, "faint", "middle", inside=True)
                    x += 14
                if kept:
                    s.rect(x, y, cw, 14, "accent", rx=2, mark=True)
                elif after:
                    s.rect(x + 0.5, y + 0.5, cw - 1, 13, "bg", rx=2, stroke="empty", dash="2 2")
                else:
                    s.rect(x + 0.5, y + 0.5, cw - 1, 13, "accent", rx=2, opacity=0.22, stroke="accent")
                x += cw + cg
            spans.append((gx, x - cg, kind))
            x += 10 if kind == "chunk" else 6
            xs.append(x)
        return spans, x
    # before
    yb = 76
    s.text(24, yb + 4, "Before", 13, "fg", weight=600)
    s.text(24, yb + 21, "first pass on every token", 11.5, "muted")
    spans, xend = strip(yb - 6, after=False)
    c0, c1, _ = spans[0]
    r0, r1 = spans[1][0], spans[-1][1]
    s.text((c0 + c1) / 2, yb - 16, "prefill chunk", 11.5, "muted", "middle")
    s.text((r0 + r1) / 2, yb - 16, f"decoding requests, {per_req} tokens each", 11.5, "muted", "middle")
    s.text(xend + 8, yb + 6, f"≈{rows_in / 1000:.1f}K rows", 13, "fg", weight=600, num=True)
    s.text(xend + 8, yb + 22, "per mixed step", 11, "faint")
    # after
    ya = 166
    s.text(24, ya + 4, "After", 13, "fg", weight=600)
    s.text(24, ya + 21, "one token per request", 11.5, "muted")
    spans, xend = strip(ya - 6, after=True)
    s.text(xend + 8, ya + 6, "1 row per request", 13, "fg", weight=600)
    s.text(xend + 8, ya + 22, "decode steps unchanged", 11, "faint")
    # legend
    ly = 226
    s.rect(24, ly - 10, 9, 13, "accent", rx=2)
    s.text(39, ly + 1, "kept: the last token of each request", 11.5, "muted")
    s.rect(290, ly - 10, 9, 13, "accent", rx=2, opacity=0.22, stroke="accent")
    s.text(305, ly + 1, "computed, then thrown away", 11.5, "muted")
    s.rect(500, ly - 10, 9, 13, "bg", rx=2, stroke="empty", dash="2 2")
    s.text(515, ly + 1, "not computed", 11.5, "muted")
    s.text(24, ly + 24, f"Measured before the fix: {fp_ms:.1f} ms of attention per mixed step, outside the target's forward pass.",
           11.5, "faint", num=True)
    cap = (f"Illustration, not a trace. vLLM runs the Gemma 4 MTP drafter's first pass over every token the target "
           f"scheduled, so a step carrying a prefill chunk sends about {rows_in / 1000:.1f}K rows through it at the knee, "
           f"and only the last row of each request is used. Gemma 4's drafter reads the target's KV cache and writes none, "
           f"so the fix runs that pass on one token per request; the outputs stay token-identical.")
    return s.render(), alt, cap, [p["file"], ours["file"]]


PRICE_ROWS = [  # (group heading, [(row label, claim group, column)])
    ("Gemma 4 31B on MI355X", [("their benchmark", "mx_recipe_c4m24", "A"), ("real text", "mx_recipe_C1", "B")]),
    ("Inkling 2 Mini", [("B300, their benchmark", "ikC_b300_mixed_dspark", "A"), ("B300, real text", "ikC_b300_mixed_dspark", "B"),
                        ("RTX PRO 6000, their benchmark", "ikC_pro6000_mixed_dspark", "A"),
                        ("RTX PRO 6000, real text", "ikC_pro6000_mixed_dspark", "B")]),
    ("GLM-5.3-Flash on MI355X", [("their benchmark", "glm_best_C", "A"), ("real text", "glm_best_C", "B")]),
]


def bar_label(r, long=False):
    n = r["price_bar_name"]
    if r.get("price_bar_is_openrelay_gemma"):
        return "OpenRelay Sync" if long else "Sync"
    if n.startswith("OpenRelay"):
        return "OpenRelay's price"
    if n.startswith("OpenRouter cheapest"):
        return "cheapest on OpenRouter"
    return n


def fig_price(D, T):
    batch = PB.usd_per_req(dict(in_usd_per_m=PB.bar_tiers({})["batch"]["in_usd_per_m"],
                                out_usd_per_m=PB.bar_tiers({})["batch"]["out_usd_per_m"]))
    groups = []
    for head, items in PRICE_ROWS:
        rs = []
        for lab, g, col in items:
            r = claim(D, g, col)
            if not r["claim_grade"] or r["usd_per_req"] is None:
                raise SystemExit(f"figures: {g} {col} is not a claim-grade priced row")
            rs.append(dict(label=lab, r=r))
        bars = {}
        for x in rs:
            bars[bar_label(x["r"])] = x["r"]["price_bar_usd_per_req"]
        if len(set(bars.values())) != len(bars) or len(bars) != 1:
            raise SystemExit(f"figures: {head} rows disagree on their price bar: {bars}")
        if rs[0]["r"].get("price_bar_is_openrelay_gemma"):
            bars["Batch"] = batch
        groups.append(dict(head=head, rows=rs, bars=bars))
    vmax = max([x["r"]["usd_per_req"] for g in groups for x in g["rows"]] + [v for g in groups for v in g["bars"].values()])
    vmax = math.ceil(vmax / 0.0002) * 0.0002
    L, X0, X1 = 234, 244, 554
    VX = 704

    def X(v):
        return X0 + v / vmax * (X1 - X0)

    gh, rh, bh = 34, 26, 14
    h = 40 + sum(gh + rh * len(g["rows"]) for g in groups) + 34
    alt = ("Our cost per request against the price we compare with, per model and load: " + "; ".join(
        f"{g['head']} {x['label']}: {usd(x['r']['usd_per_req'])}, {x['r']['ratio_vs_bar']:.2f} times "
        f"{bar_label(x['r'], long=True)}" for g in groups for x in g["rows"]) + ".")
    s = Svg(h, T, alt)
    s.text(X0, 22, f"Our cost per request ({IN_TOK / 1000:g}K tokens in, {OUT_TOK} out)", 12.5, "fg", weight=600)
    s.text(VX, 22, "vs the price", 11.5, "faint", "end")
    y = 40
    for g in groups:
        yh = y + 20
        s.text(24, yh, g["head"], 13, "fg", weight=600)
        y0 = yh + 8
        y1 = y0 + rh * len(g["rows"])
        for name, v in sorted(g["bars"].items(), key=lambda kv: kv[1]):
            s.line(X(v), y0, X(v), y1 - 6, "fg", 1.2, "3 3", 0.75)
            s.text(X(v), yh, f"{name} {usd(v)}", 11, "muted", "middle", num=True, halo=True)
        for i, x in enumerate(g["rows"]):
            r = x["r"]
            yy = y0 + i * rh + 4
            s.text(L, yy + 11, x["label"], 12, "fg", "end")
            s.bar(X0, yy, X(r["usd_per_req"]), bh, "accent",
                  tip=f"{g['head']}, {x['label']}: {usd(r['usd_per_req'])} per request at {r['max_rate_all_seeds_pass']:g} req/s")
            s.text(VX - 52, yy + 11, usd(r["usd_per_req"]), 11.5, "faint", "end", num=True)
            s.text(VX, yy + 11, f"{r['ratio_vs_bar']:.2f}×", 12, "fg", "end", 600, num=True)
        y = y1 + 8
    s.line(X0, y, X1, y, "grid")
    step = vmax / 5
    for k in range(6):
        v = k * step
        s.text(X(v), y + 18, "$0" if k == 0 else f"${v:.4f}", 11.5, "faint", "middle", num=True)
    ratios = {(g["head"], x["label"]): x["r"]["ratio_vs_bar"] for g in groups for x in g["rows"]}
    gem = groups[0]["rows"]
    cap = (f"Our cost per request at each model's claim rate, on OpenRelay's random-token benchmark and on real text, against "
           f"the price we compare with: OpenRelay's Sync and Batch tiers for Gemma, the cheapest OpenRouter provider for "
           f"Inkling 2 Mini, and OpenRelay's own price for GLM-5.3-Flash. Gemma comes in at "
           f"{gem[0]['r']['ratio_vs_bar']:.2f}× their Sync price ({gem[0]['r']['usd_per_req'] / batch:.2f}× Batch) and "
           f"Inkling at {min(x['r']['ratio_vs_bar'] for x in groups[1]['rows']):.2f}× at best; GLM costs "
           f"{min(x['r']['ratio_vs_bar'] for x in groups[2]['rows']):.2f}× or more, so there we lose.")
    del ratios
    return s.render(), alt, cap, [D["files"]["claims"], "bench/pricebar.py"]


# The final recipe (+ persistent MXFP4 GEMM) against theirs on the fill-in confirm C2b (journal: complete ladders for
# theirs, ours-today and the stack). Drawn only once both column-A rows are claim-grade; until then it is not written.
KNEE_FINAL = dict(ours="mx_stack_C2b", theirs="or_recipe_paired_C2b", test="the fill-in paired test", later=None,
                  ours_name="Our final recipe", pooled=True)


def knee_final_ready(D):
    return all((D["claims"].get((g, "A")) or {}).get("claim_grade") for g in (KNEE_FINAL["ours"], KNEE_FINAL["theirs"]))


FIGS = [("knee", fig_knee), ("ablation", fig_ablation), ("price", fig_price), ("gpu-time", fig_gpu_time),
        ("drafter-fix", fig_drafter)]


def main():
    D = load()
    verify = "--verify" in sys.argv
    if not verify:
        OUT.mkdir(parents=True, exist_ok=True)
    meta, drift = {}, []
    figs = list(FIGS)
    if knee_final_ready(D):
        figs.insert(1, ("knee-final", lambda D, T: fig_knee(D, T, **KNEE_FINAL)))
    for name, fn in figs:
        for theme, T in THEMES.items():
            svg, alt, cap, src = fn(D, T)
            for pat in LEAKS:
                for where, txt in (("svg", re.sub(r'<[^>]+>', " ", svg)), ("alt", alt), ("caption", cap)):
                    m = re.search(pat, txt)
                    if m:
                        raise SystemExit(f"figures: {name}-{theme} {where} carries an id or hash: {m.group(0)!r}")
            f = OUT / f"{name}-{theme}.svg"
            if verify:
                if not f.exists() or f.read_text() != svg:
                    drift.append(rel(f) if f.exists() else f"docs/img/{f.name} (missing)")
            else:
                f.write_text(svg)
        meta[name] = dict(alt=alt, caption=cap, light=f"docs/img/{name}-light.svg", dark=f"docs/img/{name}-dark.svg",
                          sources=src)
        if not verify:
            print(f"wrote docs/img/{name}-light.svg, docs/img/{name}-dark.svg")
    js = json.dumps(meta, indent=1, ensure_ascii=False) + "\n"
    if verify:
        f = OUT / "figures.json"
        if not f.exists() or f.read_text() != js:
            drift.append("docs/img/figures.json")
        if drift:
            print("FAIL: figures differ from a fresh render of the result files; run python3 -I analysis/figures.py --check:",
                  ", ".join(drift))
            sys.exit(1)
        print(f"OK: docs/img matches the result files ({len(figs)} figures x {len(THEMES)} themes + figures.json)")
        return
    (OUT / "figures.json").write_text(js)
    print("wrote docs/img/figures.json")
    if "--check" in sys.argv:
        files = [str(OUT / f"{n}-{t}.svg") for n, _ in figs for t in THEMES]
        rc = subprocess.run(["node", str(ROOT / "analysis/figures_check.mjs"), *files]).returncode
        sys.exit(rc)


if __name__ == "__main__":
    main()
