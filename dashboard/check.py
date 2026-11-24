#!/usr/bin/env python3
"""Verify the built dashboard against the source files, then render every page headless and check the DOM.

  python3 -I dashboard/check.py            # checks dashboard/site (real data)
  python3 -I dashboard/check.py --demo     # checks dashboard/site-demo
Exit 0 = all checks pass. Guards (each added after a failure it would have caught):
  * every cell latency/throughput number is re-derived from its raw traffic.py file (TTFT/TPOT/ITL/E2E percentiles, req/s);
  * every cell price equals its row in results/cost_rows.json (contract section 6), is internally consistent
    (usd_h / 3600 / req_s), and a cell with no row shows no price (the dashboard never fills one in);
  * the price bar is re-derived from the price JSON; the spend total is re-derived from the ledger by a separate loop;
  * every cell and run cites a run id and a file that exists;
  * protocol ladders: the dashboard's best result never comes from a non-kept seed;
  * protocol ladders: kept-seed fields equal results[kept_seed], the kept seed is the worse one, protocol_rows prices are
    $/h / 3600 / offered rate and / Sync; a ladder headline price comes from a kept, SLO-passing row; no evicted run feeds the best;
  * rendered app (every hash route, both themes, file://): no console errors, app mounted, theme applied, no NaN/undefined in
    visible text, cards carry run-id citations, key numbers (bar, spend, run ids, cell prices, journey steps) are in the DOM;
  * layout (dashboard/layout_check.mjs, real Chrome over CDP at 1366, 446 and 390 px, every route): no horizontal overflow,
    every chart drawn, no two SVG labels colliding or cut off at the chart edge, every plot area >= 40% of its card, every bar >= 2 px, no clipped KPI
    value (added after a price chart collapsed to a strip and labels collided in the lead's live view, while every
    screenshot-by-eye and DOM check passed).
Heavy work runs under taskset -c 8-15 nice -n 19 (chromium).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import subprocess
import sys
import time

DASH = pathlib.Path(__file__).resolve().parent
ROOT = DASH.parent
FAILS, OKS, SKIPS = [], [], []


def check(cond, msg):
    (OKS if cond else FAILS).append(msg)
    return cond


def close(a, b, rel=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)


def usd_fmt(x):
    """same rule as app.js usd()"""
    a = abs(x)
    if a >= 100:
        return f"${x:,.0f}"
    if a >= 1:
        return f"${x:.2f}"
    if a >= 0.01:
        return f"${x:.3f}"
    p = float(f"{x:.3g}")
    return f"${p:.{max(0, 2 - math.floor(math.log10(a)))}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    site = DASH / ("site-demo" if a.demo else "site")
    d = json.loads((site / "data.json").read_text())

    # 1. price bar
    orl = json.loads((ROOT / d["bars"]["openrelay"]["file"]).read_text())
    sync = orl["models"]["openrelay/gemma-4-31b"]["tiers"]["Sync"]
    bar = (9500 * sync["in_usd_per_m"] + 380 * sync["out_usd_per_m"]) / 1e6
    check(close(bar, d["bars"]["per_request"]["sync"]), f"bar: Sync $/req {bar:.7f} == data {d['bars']['per_request']['sync']:.7f}")
    check(abs(bar - 0.000779) < 5e-10, f"bar: Sync $/req at 9.5K/380 is $0.000779 as in docs/PLAN.md (got {bar:.9f})")

    # 2. cells re-derived from raw files; prices from cost_rows.json
    res_dir = (DASH / "demo" / "results") if a.demo else (ROOT / "results")
    cr_p = res_dir / "cost_rows.json"
    cost_rows = {r["file"]: r for r in json.loads(cr_p.read_text())} if cr_p.exists() else {}
    n_cells = 0
    for c in d["cells"]:
        n_cells += 1
        f = ROOT / c["file"]
        if not check(f.exists() and c.get("run_id"), f"cell {c['cell_id']}: cites run {c.get('run_id')} and existing file {c['file']}"):
            continue
        raw = json.loads(f.read_text())
        s = raw["summary"]
        for k in ("ttft_s", "tpot_s", "itl_s", "e2e_s"):
            for p in ("p50", "p90", "p95", "p99"):
                got = (c.get(k[:-2]) or {}).get(p)
                want = (s.get(k) or {}).get(p)
                if not close(got, want):
                    check(False, f"cell {c['cell_id']} ({c['run_id']}): {k} {p} {got} != raw {want}")
        params = raw.get("params") or {}
        fixed = raw["args"].get("workload") in ("synthetic", "longdoc_qa") and params.get("ignore_eos", True)
        rps = s["output_tok_per_s"] / params.get("output_tokens", 256) if fixed else s["req_per_s"]
        check(close(c["req_s"], rps), f"cell {c['cell_id']}: req/s {c['req_s']} == raw {rps}")
        if raw.get("schema") is not None:
            check(raw["schema"] == "traffic/1", f"cell {c['cell_id']}: schema traffic/1")
        cr = cost_rows.get(c["file"])
        if cr is None:
            check(c.get("our_usd_req") is None and c.get("ratio_sync") is None, f"cell {c['cell_id']}: no cost_rows.json row -> no price shown")
            continue
        for k_dash, k_cost in (("our_usd_req", "our_usd_req"), ("ratio_sync", "ratio_sync"), ("bar_sync", "openrelay_sync"), ("ratio_batch", "ratio_batch"), ("usd_h", "usd_h")):
            check(close(c.get(k_dash), cr.get(k_cost)), f"cell {c['cell_id']}: {k_dash} {c.get(k_dash)} == cost_rows {k_cost} {cr.get(k_cost)}")
        if cr.get("req_s") and cr.get("usd_h"):
            check(close(cr["our_usd_req"], cr["usd_h"] / 3600 / cr["req_s"]), f"cell {c['cell_id']}: cost_rows $/req == usd_h/3600/req_s")
            check(close(cr["req_s"], rps, rel=1e-6), f"cell {c['cell_id']}: cost_rows req/s {cr['req_s']} == raw-derived {rps}")
    check(n_cells == len(d["cells"]), f"{n_cells} cells checked")
    # requests completed etc. spot: count of cells per run equals files on disk
    for r in d["runs"]:
        rd = (ROOT / "results" / r["run_id"]) if not a.demo else (DASH / "demo" / "results" / r["run_id"])
        def traffic(f):   # same rule as build.py: protocol seed files written by vllm bench are not traffic cells
            try:
                h = json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                return False
            return h.get("schema") == "traffic/1" or "summary" in h
        files = list(rd.glob("m_*/cells/*.json")) + list(rd.glob("out/m_*/cells/*.json")) + [f for f in rd.glob("protocol_*/cells/*.json") if traffic(f)]
        on_disk = len(files)
        # a live run can land a cell between the build and this check: those (newer than the build) are not the build's to show
        t_b = d["build"].get("t_epoch") or 0
        older = sum(1 for f in files if f.stat().st_mtime <= t_b)
        check(r["n_cells"] in (on_disk, older), f"run {r['run_id']}: {r['n_cells']} cells in the dashboard == {on_disk} cell files on disk" + (f" ({older} older than the build)" if older != on_disk else ""))

    # 3. best result never from a non-kept seed; ineligible never eligible
    for b in d.get("best", []):
        check(b.get("protocol") != "other_seed", f"best {b['config_id']}: not from a non-kept seed")
        if b.get("eligible"):
            check(bool((b.get("gate") or {}).get("passed")) and not b.get("stop"), f"best {b['config_id']}: eligible implies gate passed and no stop rule")

    # 4. spend re-derived from the ledger, separately
    led = ROOT / d["spend"]["ledger_file"]
    rows = [json.loads(x) for x in led.read_text().splitlines() if x.strip()] if led.exists() else []
    built = d["build"].get("t_epoch") or (time.mktime(time.strptime(d["build"]["t"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone)
    ts = lambda s: time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone  # noqa: E731
    tot = 0.0
    dels = {}
    for r in rows:
        if r.get("event") == "delete":
            dels.setdefault(r["id"], r["t"])
    for r in rows:
        if r.get("event") == "create":
            end = ts(dels[r["id"]]) if r["id"] in dels else min(built, float(r.get("deadline") or built))
            tot += max(0.0, end - ts(r["t"])) / 3600 * r["usd_per_h"]["total"]
    import hashlib
    fp = next((f for f in d.get("inputs", []) if f["path"] == d["spend"]["ledger_file"]), None)
    same = fp is not None and led.exists() and hashlib.sha256(led.read_bytes()).hexdigest() == fp["sha256"]
    if same:
        check(abs(tot - d["spend"]["total_est"]) < 0.01, f"spend: ledger re-derivation ${tot:.4f} == dashboard ${d['spend']['total_est']:.4f}")
    else:
        SKIPS.append(f"spend: {d['spend']['ledger_file']} changed since the build (live runs); rebuild and re-run to re-derive the total")

    # 4a. evictions: re-derived from the ledger's `evicted` events; never part of a best/$-per-request
    evs = [r for r in rows if r.get("event") == "evicted"]
    if d.get("evictions") is not None and same:
        check(d["evictions"]["total_count"] == len(evs), f"evictions: {d['evictions']['total_count']} == {len(evs)} ledger events")
        check(close(round(d["evictions"]["total_usd"], 6), round(sum(e.get("usd_lost") or 0 for e in evs), 6)), "evictions: $ lost equals the ledger sum")
    for b in d.get("best", []):
        check(b["run_id"] not in {e.get("run_id") for e in evs}, f"best {b['config_id']}: not from an evicted run")

    def gemma_or(name):   # name rule: OpenRelay's own Gemma 4 31B bar (no name = the global Sync bar); GLM / Inkling bars are other models
        return name is None or ("openrelay" in str(name).lower() and all(t.split("/")[-1] == "gemma-4-31b" for t in str(name).split() if "/" in t))

    def gemma_bar(r):     # the build session's flag (price_bar_is_openrelay_gemma, cf9e936) decides; the name rule only when it is absent
        return bool(r["price_bar_is_openrelay_gemma"]) if "price_bar_is_openrelay_gemma" in r else gemma_or(r.get("price_bar_name"))

    for r in ((d.get("claims") or {}).get("claims") or []) + list(d.get("protocol_rows") or []):   # the flag and the bar name must tell the same story
        if "price_bar_is_openrelay_gemma" in r and r.get("price_bar_name") is not None:
            check(bool(r["price_bar_is_openrelay_gemma"]) == gemma_or(r["price_bar_name"]), f"price bar flag of {r.get('group') or r.get('run_id')}/{r.get('column')}: price_bar_is_openrelay_gemma agrees with '{r['price_bar_name']}'")

    # 4c. claims (results/claims.json): price = $/h / 3600 / worst seed's ACHIEVED req/s; ratio = that / the Sync bar
    cl = d.get("claims") or {}
    for row in cl.get("claims", []):
        tag = f"claim {row.get('group')}/{row.get('column')}"
        if row.get("usd_per_req") is not None:
            check(close(row["usd_per_req"], row["usd_per_h"] / 3600 / row["worst_seed_achieved_rps"], 1e-6), f"{tag}: $/req = $/h / 3600 / worst-seed achieved req/s")
            if row.get("ratio_vs_openrelay_sync") is not None or gemma_bar(row):
                check(close(row.get("ratio_vs_openrelay_sync"), row["usd_per_req"] / cl["openrelay_sync_usd_per_req"], 1e-6), f"{tag}: ratio = $/req / Sync bar")
        if row.get("usd_per_req_metered") is not None:
            check(close(row["usd_per_req_metered"], row["usd_per_h_metered"] / 3600 / row["worst_seed_achieved_rps"], 1e-6), f"{tag}: metered $/req arithmetic")
        if row.get("claim_grade"):
            check(len(row.get("seeds") or []) >= 2, f"{tag}: claim-grade rows have >= 2 seeds")
        if row.get("usd_per_req") is not None and row.get("price_bar_usd_per_req"):
            check(close(row["ratio_vs_bar"], row["usd_per_req"] / row["price_bar_usd_per_req"], 1e-6), f"{tag}: ratio_vs_bar = $/req / its bar ({row.get('price_bar_name')})")
            if not gemma_bar(row):
                check(row.get("ratio_vs_openrelay_sync") is None, f"{tag}: no OpenRelay Gemma ratio on another bar ({row.get('price_bar_name')})")
        o = row.get("origin")
        check(o in ("openrelay_recipe", "ours") or str(o).startswith("mixed:"), f"{tag}: origin {o!r} is openrelay_recipe / ours / mixed:a+b (contract v1.6)")
    # the overview headline (column B, with A beside it) is the claim-grade / most-seeds / cheapest row, as in headlineFor()
    def headline(col):
        rs = [r for r in cl.get("claims", []) if r.get("column") == col and r.get("usd_per_req") is not None and gemma_bar(r)]
        rs.sort(key=lambda r: (-int(bool(r.get("claim_grade"))), -len(r.get("seeds") or []), r["usd_per_req"]))
        return rs[0] if rs else None
    HEAD = {c: headline(c) for c in ("A", "B")}

    # 4d. ladder points (contract v1.5 merge): pass only if every seed passes; achieved = the worst seed's
    grp = {}
    for r in d.get("protocol_rows", []):
        if r.get("offered_rps") is not None:
            grp.setdefault((r.get("claim_group") or r.get("config_id"), r.get("column"), round(r["offered_rps"], 4)), []).append(r)
    for p in [x for x in d.get("ladder_points", []) if x.get("source") == "protocol_rows"]:
        rs = grp.get((p["group"], p["column"], p["rate"]), [])
        check(len(rs) == len(p["seeds"]), f"ladder point {p['group']}/{p['column']} @{p['rate']}: {len(p['seeds'])} seeds == {len(rs)} rows")
        check(p["pass_"] == all(bool(r.get("slo_pass")) for r in rs), f"ladder point {p['group']}/{p['column']} @{p['rate']}: passes only if every seed passes")
        if rs and p.get("achieved") is not None:
            check(close(p["achieved"], min(r["achieved_rps"] for r in rs if r.get("achieved_rps") is not None)), f"ladder point {p['group']}/{p['column']} @{p['rate']}: achieved = worst seed")

    # 4e. paired boxes (same GPU, ours vs theirs): every number re-read from its protocol_summary / gate file
    for bx in d.get("paired", []):
        for side, rec in bx["recipes"].items():
            tag = f"paired {bx['run_id']} {side}"
            for col, rungs in rec["ladders"].items():
                for x in rungs:
                    if x.get("source") == "protocol_rows":   # contract v1.8: the 300 s re-score wins over protocol_summary.json
                        rs = [r for r in d.get("protocol_rows", []) if r.get("run_id") == bx["run_id"] and r.get("server") == rec["server"] and r.get("column") == col
                              and r.get("offered_rps") is not None and abs(r["offered_rps"] - x["rate"]) < 1e-4]
                        r0 = next((r for r in rs if r.get("kept")), rs[0] if rs else None)
                        check(r0 is not None and close(r0.get("ttft_p95_s"), x["ttft95"]) and bool(r0.get("slo_pass")) == x["pass_"], f"{tag} col {col} @{x['rate']}: p95 TTFT and pass from protocol_rows (v1.8)")
                passing = [x["rate"] for x in rungs if x["pass_"] and not x.get("skipped")]
                want = max(passing) if passing else 0.0
                check(close(rec["max_rate"].get(col) or 0.0, want), f"{tag} col {col}: max rate at SLO {rec['max_rate'].get(col)} = highest passing rung {want}")
            for f in rec["files"]:
                check((ROOT / f).exists(), f"{tag}: cites existing {f}")
            if rec.get("gate"):
                g = json.loads((ROOT / rec["gate"]["file"]).read_text())
                task = next((t for t in (g.get("summary") or {}) if t.startswith("gsm8k")), None)
                check(task and close(g["summary"][task]["acc"], rec["gate"]["acc"]), f"{tag}: GSM8K {rec['gate']['acc']} matches {rec['gate']['file']}")
            if rec.get("prefill_runs"):
                check(close(rec["prefill_tok_s"], rec["prefill_runs"][-1]) or rec["prefill_tok_s"] in rec["prefill_runs"], f"{tag}: prefill tok/s is the counted run")


    # 4g. H1 same-box pair (docs/WRITEUP.md order): re-derived from claims.json + protocol_rows + each run's config.json
    crow = {(r.get("group"), r.get("column")): r for r in cl.get("claims", [])}
    for sb in d.get("same_box", []):
        o, t_ = crow.get((sb["ours"], "A")) or {}, crow.get((sb["theirs"], "A")) or {}
        tag = f"same_box {sb['ours']} vs {sb['theirs']}"
        check(o.get("origin") == "ours" and t_.get("origin") == "openrelay_recipe", f"{tag}: ours is origin 'ours', theirs 'openrelay_recipe'")
        check(bool(o.get("runs")) and set(o.get("runs") or []) == set(t_.get("runs") or []) == set(sb["runs"]), f"{tag}: both groups ran on the same {len(sb['runs'])} boxes")
        check(sb["ours_rate"] == o.get("max_rate_all_seeds_pass") and sb["theirs_rate"] == t_.get("max_rate_all_seeds_pass"), f"{tag}: rates {sb['ours_rate']} vs {sb['theirs_rate']} from claims.json")
        check(close(sb["gain"], o["worst_seed_achieved_rps"] / t_["worst_seed_achieved_rps"] - 1), f"{tag}: gain = worst-box achieved ratio - 1 ({sb['gain']:+.4f})")
        for b in sb["boxes"]:
            for side, g in (("ours", sb["ours"]), ("theirs", sb["theirs"])):
                rs = [r for r in d.get("protocol_rows", []) if r.get("claim_group") == g and r.get("column") == "A" and r.get("run_id") == b["run_id"]
                      and r.get("offered_rps") is not None and abs(r["offered_rps"] - sb["rate"]) < 1e-6]
                check(len(rs) == 1 and close(rs[0].get("ttft_p95_s"), b[side]["ttft95"]) and bool(rs[0].get("slo_pass")) == b[side]["pass_"],
                      f"{tag} {b['run_id']} {side}: p95 TTFT {b[side]['ttft95']} and pass at {sb['rate']} req/s from protocol_rows")
        cfgs = [json.loads((ROOT / "results" / run / "config.json").read_text()) for run in sb["runs"] if (ROOT / "results" / run / "config.json").exists()]
        digests = {c.get("snapshot", {}).get("image_digest_amd64") for c in cfgs}
        check(sb["same_image"] == (len(digests) == 1 and None not in digests), f"{tag}: same image digest = {sb['same_image']} (config.json)")
        mmls = {int(m) for c in cfgs for sv in c.get("servers", []) for m in re.findall(r"--max-model-len[ =](\d+)", sv.get("launch") or "")}
        check(set(x for x in sb["mml"] if x) <= mmls, f"{tag}: max-model-len {sb['mml']} from the launch lines")
        oB, tB = crow.get((sb["ours"], "B")) or {}, crow.get((sb["theirs"], "B")) or {}
        check(sb["b"]["ours_rate"] == oB.get("max_rate_all_seeds_pass") and sb["b"]["theirs_rate"] == tB.get("max_rate_all_seeds_pass"), f"{tag}: real-text rates from claims.json")
    # 4h. H3 host levers: same server and image, different host; numbers copied from claims.json
    for lv in d.get("host_levers", []):
        r = crow.get((lv["group"], "A")) or {}
        check(r.get("origin") == "ours" and close(lv["usd_per_h"], r.get("usd_per_h")), f"host lever {lv['group']}: ours, ${lv['usd_per_h']}/h from claims.json")
        check(lv["host"] != lv["base_host"], f"host lever {lv['group']}: host {lv['host']} differs from {lv['base_host']}")
        for c, v in lv["cols"].items():
            rc = crow.get((lv["group"], c)) or {}
            check(close(v["ratio"], rc.get("ratio_vs_bar") if rc.get("ratio_vs_bar") is not None else rc.get("ratio_vs_openrelay_sync")), f"host lever {lv['group']} {c}: ratio from claims.json")
    # 4i. H4 scope: the recipe listing is the saved GitHub contents file
    scp = d.get("scope") or {}
    if scp.get("recipes_file"):
        items = json.loads((ROOT / scp["recipes_file"]).read_text())
        check(scp["recipes"] == [x["name"] for x in items if isinstance(x, dict) and x.get("type") == "dir"], f"scope: published recipes {scp['recipes']} = the dirs in {scp['recipes_file']}")
    # 4j. quality_rows.json (schema quality/1): delta = acc - reference acc, every estimate inside its 95% interval
    q = d.get("quality") or {}
    if q:
        qref = next((c for c in q["configs"] if c.get("is_reference")), None)
        check(qref is not None, "quality: one reference config (is_reference)")
        for c in q["configs"]:
            for task in ("gsm8kor250", "mmlupro200"):
                x = c.get(task)
                if not x or not qref or not qref.get(task):
                    continue
                tag = f"quality {c['config']} {task}"
                check(close(x["acc"], x["correct"] / x["n"]), f"{tag}: acc = correct / n")
                check(x["ci95"][0] <= x["acc"] <= x["ci95"][1], f"{tag}: acc inside its 95% interval")
                check(close(x["delta_vs_ref"], x["acc"] - qref[task]["acc"], 1e-9) or abs(x["delta_vs_ref"] - (x["acc"] - qref[task]["acc"])) < 1e-9, f"{tag}: delta = acc - BF16 acc")
                if x.get("delta_ci95"):
                    check(x["delta_ci95"][0] - 1e-12 <= x["delta_vs_ref"] <= x["delta_ci95"][1] + 1e-12, f"{tag}: delta inside its 95% interval")
    for e in d.get("evals", []):   # the gate table's Wilson interval
        if e.get("ci95") and isinstance(e.get("correct"), int) and e.get("n"):
            z = 1.959963984540054; n_ = e["n"]; ph = e["correct"] / n_; den = 1 + z * z / n_
            mid, half = (ph + z * z / (2 * n_)) / den, z * math.sqrt(ph * (1 - ph) / n_ + z * z / (4 * n_ * n_)) / den
            check(close(e["ci95"][0], mid - half) and close(e["ci95"][1], mid + half), f"eval {e['run_id']} {e['task']}: Wilson 95% interval")

    # 4k. journey steps are priced against their own bar (experiments.jsonl price_bar; absent = OpenRelay's Gemma Sync bar)
    for st in d.get("journey", {}).get("steps", []):
        pb = st.get("price_bar") if isinstance(st.get("price_bar"), dict) else {}
        want = pb.get("usd_per_req") or d["bars"]["per_request"]["sync"]
        check(close(st.get("bar_usd"), want), f"journey step {st.get('step')}: its bar ${want} ({pb.get('name') or 'OpenRelay Gemma Sync'})")
        if isinstance(st.get("usd_per_req"), (int, float)):
            check(close(st.get("ratio_bar"), st["usd_per_req"] / want), f"journey step {st.get('step')}: ratio = $/req / its own bar")
        mods = [t.split("/")[-1] for t in str(pb.get("name") or "").split() if "/" in t]
        check(st.get("model") == (mods[0] if mods else d["build"].get("bench_model", "gemma-4-31b")), f"journey step {st.get('step')}: model {st.get('model')} from its bar")

    # 4f. quality view: every GSM8K row re-read from its eval/gate file
    for e in d.get("evals", []):
        if (e.get("task") or "").startswith("gsm8k") and e.get("file"):
            g = json.loads((ROOT / e["file"]).read_text())
            check(close((g.get("summary") or {}).get(e["task"], {}).get("acc"), e.get("acc")), f"quality {e['run_id']} {e['task']}: acc matches {e['file']}")

    # 4b. protocol ladders (contract 7b): kept-seed fields copied up match results[kept_seed]; protocol_rows prices are consistent
    for pr in d.get("protocols", []):
        for l in pr.get("ladder") or []:
            if l.get("skipped"):
                continue
            ks = str(l.get("kept_seed"))
            res = (l.get("results") or {}).get(ks)
            if res:
                for k in ("achieved_rps", "goodput_rps", "ttft_p95_s", "tpot_p50_s", "slo_pass"):
                    if k in res and k in l:
                        check(res[k] == l[k] or close(res.get(k), l.get(k)), f"ladder {pr['file']} x{l.get('multiple')}: {k} copied up from seed {ks}")
                if len(l.get("results") or {}) > 1:   # the kept seed is the worse one (higher p95 TTFT)
                    worst = max((l["results"].values()), key=lambda r: r.get("ttft_p95_s") or 0)
                    check(close(worst.get("ttft_p95_s"), res.get("ttft_p95_s")), f"ladder {pr['file']} x{l.get('multiple')}: kept seed is the worse seed")
    sync = d["bars"]["per_request"]["sync"]
    for r in d.get("protocol_rows", []):
        if r.get("usd_per_h") and r.get("offered_rps") and r.get("usd_per_req_at_offered") is not None:
            check(close(r["usd_per_req_at_offered"], r["usd_per_h"] / 3600 / r["offered_rps"], 1e-6), f"protocol_rows {r['run_id']} x{r.get('multiple')} s{r.get('seed')}: $/req at offered = $/h/3600/rate")
            if r.get("ratio_sync_at_offered") is not None:
                # the row's own bar (contract v1.7 per-config price bar, else OpenRelay Sync at its measured token counts)
                own = r.get("price_bar_usd_per_req")
                on_or = gemma_bar(r) if "price_bar_is_openrelay_gemma" in r else (own is None or gemma_or(r.get("price_bar_name")))
                bar = own or r.get("openrelay_sync_usd_per_req") or sync
                check(close(r["ratio_sync_at_offered"], r["usd_per_req_at_offered"] / bar, 1e-6), f"protocol_rows {r['run_id']} x{r.get('multiple')}: ratio vs its bar ({r.get('price_bar_name') or 'OpenRelay sync'})")
                if on_or:
                    check(abs(bar - sync) / sync < 0.02, f"protocol_rows {r['run_id']} x{r.get('multiple')}: its Sync bar {bar:.7f} within 2% of the 9.5K/380 bar")
    for b in d.get("best", []):
        if b.get("protocol") == "ladder":
            ok = any(r.get("run_id") == b["run_id"] and r.get("kept") and r.get("slo_pass") and close(r.get("usd_per_req_at_achieved"), b["our_usd_req"]) for r in d.get("protocol_rows", []))
            check(ok, f"best {b['config_id']}: ladder price is a kept, SLO-passing protocol_rows row priced on the ACHIEVED rate")
    for b in d.get("best", []):
        check(b["run_id"] not in {r["run_id"] for r in d["runs"] if r.get("excluded")}, f"best {b['config_id']}: not from an evicted/excluded run")

    # 5. render every route of the React app (hash routes, file://) and read the DOM
    if not a.no_render:
        import html as H
        first_run = next((r["run_id"] for r in d["runs"] if r["phases"]), d["runs"][0]["run_id"] if d["runs"] else None)
        routes = ["", "journey", "paired", "quality", "cost", "goodput", "latency", "runs", "spend", "startup", "mtp", "compare", "sources"] + ([f"run/{first_run.replace('/', '~')}"] if first_run else [])
        for rt in routes:
            for theme in ("light", "dark"):
                name = rt or "overview"
                cmd = ["taskset", "-c", "8-15", "nice", "-n", "19", "chromium", "--headless=new", "--disable-gpu", "--no-sandbox", "--window-size=1366,2400",
                       "--virtual-time-budget=6000", "--enable-logging=stderr", "--v=0", "--dump-dom", f"file://{site}/index.html?theme={theme}#/{rt}"]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                dom, log = r.stdout, r.stderr
                errs = [ln for ln in log.splitlines() if ("CONSOLE" in ln or "Uncaught" in ln) and ("rror" in ln or "Uncaught" in ln)]
                check(not errs, f"render {name} [{theme}]: no console errors" + (f" -> {errs[:2]}" if errs else ""))
                check("<h1" in dom and 'id="root"' in dom, f"render {name} [{theme}]: app mounted (h1 present)")
                check(bool(re.search(r'<html[^>]*class="[^"]*\bdark\b', dom)) == (theme == "dark"), f"render {name} [{theme}]: theme class applied")
                text = re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style)[^>]*>.*?</\\1>", " ", dom, flags=re.S))
                bad = [w for w in ("NaN", "undefined", "[object Object]", "Infinity") if re.search(r"(?<![A-Za-z])" + re.escape(w) + r"(?![A-Za-z])", text)]
                check(not bad, f"render {name} [{theme}]: no {bad or 'NaN/undefined/[object Object]'} in visible text")
                for m in re.finditer(r'data-bars="(\d+)"', dom):   # every claims bar chart draws one non-zero-width bar per row
                    seg = dom[m.end(): dom.find("</section>", m.end())]
                    widths = [float(w) for w in re.findall(r'<path[^>]*class="recharts-rectangle"[^>]*\swidth="([\d.]+)"', seg)]
                    widths += [float(w) for w in re.findall(r'<path[^>]*\swidth="([\d.]+)"[^>]*class="recharts-rectangle"', seg)]
                    drawn = sum(1 for w in widths if w > 0)
                    check(drawn >= int(m.group(1)), f"render {name} [{theme}]: claims chart draws {drawn} of {m.group(1)} bars with non-zero width")
                srcs = re.findall(r'<span class="sr-only" data-sources="">(.*?)</span>', dom, flags=re.S)
                if rt in ("", "cost", "goodput", "runs", "spend", "journey") and (d["cells"] or d["runs"]):
                    check(any(x.strip() for x in srcs), f"render {name} [{theme}]: cards cite run ids/files ({sum(1 for x in srcs if x.strip())} non-empty)")
                if theme == "light":
                    if rt == "spend":
                        check(f"${d['spend']['total_est']:.2f}" in dom, f"render spend: shows the ledger estimate ${d['spend']['total_est']:.2f}")
                    if rt == "":
                        check(usd_fmt(d["bars"]["per_request"]["sync"]) in dom, f"render overview: shows OpenRelay's bar {usd_fmt(d['bars']['per_request']['sync'])}")
                        for col, hr in HEAD.items():
                            if hr:
                                v = hr["ratio_vs_bar"] if hr.get("ratio_vs_bar") is not None else hr["ratio_vs_openrelay_sync"]
                                txt = f"{v:.2f}×" if v < 10 else f"{v:.1f}×"
                                check(txt in dom, f"render overview: column {col} headline {hr['group']} shows {txt} (claims.json)")
                        for col, hr in HEAD.items():   # the bar is named next to the headline ratio (contract v1.7)
                            if hr and hr.get("price_bar_name"):
                                short = " ".join(t for t in hr["price_bar_name"].split() if "/" not in t)
                                check(H.escape(short, quote=False) in dom, f"render overview: column {col} ratio names its bar '{short}'")
                        if HEAD.get("B"):
                            want = "B claim-grade" if HEAD["B"].get("claim_grade") else "B not claim-grade"
                            check(want in dom, f"render overview: shows '{want}' from claims.json")
                        for col, hr in HEAD.items():   # never a bare ratio: a non-claim-grade headline shows its reasons in the page
                            if hr and not hr.get("claim_grade"):
                                for why in hr.get("not_claim_grade_because") or []:
                                    check(H.escape(why, quote=False) in dom, f"render overview: column {col} reason shown: {why[:60]}")
                    if rt == "" and d.get("same_box"):
                        sb = d["same_box"][0]
                        for txt in (f"{sb['ours_rate']:.2f}", f"{sb['theirs_rate']:.2f}", f"{sb['ours_achieved']:,.3f}", f"{sb['theirs_achieved']:,.3f}", f"{'+' if sb['gain'] >= 0 else ''}{sb['gain'] * 100:.1f}%",
                                    f"same {len(sb['boxes'])} {sb['gpu']} boxes"):
                            check(txt in dom, f"render overview H1: shows {txt}")
                        for b in sb["boxes"]:
                            for side in ("ours", "theirs"):
                                v = b[side]["ttft95"]
                                check(f"{v:.2f} s" in dom, f"render overview H1: {b['run_id']} {side} p95 TTFT {v:.2f} s")
                        check("Our cost vs their API price" in dom and "a different question" in dom, "render overview H2: cost card labelled a different question")
                    if rt == "":
                        for lv in d.get("host_levers", []):
                            check(lv["group"] in dom and "Daytona lever" in dom and "not a recipe comparison" in dom, f"render overview H3: host lever {lv['group']} labelled, never a recipe win")
                        if scp.get("recipes"):
                            check(f"OpenRelay published {len(scp['recipes'])} recipe" in dom and "context: " + ("PENDING" if scp.get("long_context") != "done" else "measured") in dom, "render overview H4: scope chips")
                    if rt == "quality" and q:
                        for c in q["configs"]:
                            for task in ("gsm8kor250", "mmlupro200"):
                                x = c.get(task)
                                if x and not c.get("is_reference"):
                                    dv = x["delta_vs_ref"]
                                    txt = ("+" if dv > 0 else "−" if dv < 0 else "±") + f"{abs(dv * 100):.1f} pp"
                                    check(txt in dom, f"render quality: {c['config']} {task} delta {txt}")
                        for m in q.get("mtp_lossless", []):
                            if not m.get("same_box"):
                                check("different boxes" in dom, "render quality: the MTP pair is labelled 'different boxes'")
                    if rt == "runs":
                        for r_ in d["runs"]:
                            check(r_["run_id"] in dom, f"render runs: lists {r_['run_id']}")
                    if rt == "cost":
                        for c in d["cells"]:
                            if c.get("our_usd_req") and c.get("protocol") != "other_seed":
                                check(usd_fmt(c["our_usd_req"]) in dom, f"render cost: cell {c['cell_id']} $/req {usd_fmt(c['our_usd_req'])} in the table")
                    if rt == "quality":
                        for e in d.get("evals", []):
                            if (e.get("task") or "").startswith("gsm8k"):
                                check(e["run_id"].split("/")[0] in dom, f"render quality: shows {e['run_id']}")
                    if rt == "paired":
                        for bx in d.get("paired", []):
                            check(bx["run_id"] in dom, f"render paired: shows box {bx['run_id']}")
                    if rt == "journey":   # the default view is the benchmark model's journey; other models sit behind the switch
                        bm = d["build"].get("bench_model", "gemma-4-31b")
                        js = d.get("journey", {}).get("steps", [])
                        for st in js:
                            if (st.get("model") or bm) == bm:
                                check(H.escape(st.get("change") or "", quote=False) in dom, f"render journey: step {st.get('step')} is shown")
                        if d.get("same_box"):   # the journey ends at the headline recipe (H1 group), never at a host-sizing lever
                            check(f"ends at our recipe · {d['same_box'][0]['ours']}" in dom, f"render journey: ends at the H1 group {d['same_box'][0]['ours']}")
                            for lv in d.get("host_levers", []):
                                check(f"ends at our recipe · {lv['group']}" not in dom, f"render journey: never ends at the host lever {lv['group']}")
                        for mdl in sorted({st.get("model") for st in js if st.get("model") and st.get("model") != bm}):
                            check(mdl in dom, f"render journey: the model switch offers {mdl}")
    # 6. layout in real Chrome over CDP at 1366, 446 and 390 px (dashboard/layout_check.mjs): no colliding SVG labels, plot area >= 40%
    #    of its card, bars >= 2 px, no clipped KPI value, no horizontal overflow, every chart drawn
    if not a.no_render:
        import shutil
        import socket
        import tempfile
        if not shutil.which("node"):
            check(False, "layout check: node not found (needed for dashboard/layout_check.mjs)")
        else:
            sk = socket.socket(); sk.bind(("127.0.0.1", 0)); port = sk.getsockname()[1]; sk.close()
            prof = tempfile.mkdtemp(prefix="dash_layout_")
            ch = subprocess.Popen(["taskset", "-c", "8-15", "nice", "-n", "19", "chromium", "--headless=new", f"--remote-debugging-port={port}", "--remote-debugging-address=127.0.0.1",
                                   "--no-sandbox", "--disable-gpu", f"--user-data-dir={prof}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                for _ in range(50):
                    try:
                        socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                        break
                    except OSError:
                        time.sleep(0.2)
                r = subprocess.run(["node", str(DASH / "layout_check.mjs"), str(site), str(port), *routes], capture_output=True, text=True, timeout=1200)
                rows = [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")]
                ok0, fail0 = len(OKS), len(FAILS)
                check(len(rows) == 4 * len(routes), f"layout check: {len(rows)} of {4 * len(routes)} pages measured" + (f" ({r.stderr[-200:]})" if r.returncode else ""))
                for x in rows:
                    where = f"layout {x['route']} @{x['width']} [{x['theme']}]"
                    check(x["scrollWidth"] <= x["width"] + 1, f"{where}: no horizontal overflow (page {x['scrollWidth']} px)")
                    if x.get("charts"):
                        check((x.get("minChartWidth") or 0) >= 150, f"{where}: all {x['charts']} charts drawn at width (narrowest {x.get('minChartWidth') or 0:.0f} px)")
                    check(not x["overlaps"], f"{where}: no colliding chart labels" + (f" -> {x['overlaps'][:3]}" if x["overlaps"] else ""))
                    check(not x.get("textOut", ["probe missing"]), f"{where}: no chart label past its chart's edge" + (f" -> {x['textOut'][:3]}" if x.get("textOut") else ""))
                    thin = [p for p in x["plots"] if p["ratio"] < 0.40]
                    check(not thin, f"{where}: every plot area >= 40% of its card ({len(x['plots'])} plots)" + (f" -> {thin[:3]}" if thin else ""))
                    check(not x["smallBars"], f"{where}: every bar >= 2 px" + (f" -> {x['smallBars'][:3]}" if x["smallBars"] else ""))
                    check(not x["kpiClipped"], f"{where}: no clipped KPI value" + (f" -> {x['kpiClipped'][:3]}" if x["kpiClipped"] else ""))
                    check(not x.get("kpiBroken", ["probe missing"]), f"{where}: no KPI number split across lines" + (f" -> {x['kpiBroken'][:3]}" if x.get("kpiBroken") else ""))
                # the guard's own summary: what it measured, so a pass is visibly not a pass-by-absence
                pl = [p["ratio"] for x in rows for p in x["plots"]]
                mb = [x["minBar"] for x in rows if x.get("minBar") is not None]
                widths = sorted({f"{x['width']}{'' if x['theme'] == 'light' else ' ' + x['theme']}" for x in rows}, key=lambda s: (-int(s.split()[0]), s))
                print(f"layout guard: {len(rows)} pages ({len(routes)} routes x {', '.join(widths)}) · {sum(x.get('nLabels', 0) for x in rows)} chart labels, "
                      f"{sum(len(x['overlaps']) for x in rows)} colliding, {sum(len(x.get('textOut', [])) for x in rows)} cut off · {len(pl)} plots, narrowest {min(pl, default=0):.0%} of its card · "
                      f"{sum(x.get('nBars', 0) for x in rows)} bars, thinnest {min(mb, default=0):.1f} px · {sum(x.get('nKpi', 0) for x in rows)} KPI values, "
                      f"{sum(len(x['kpiClipped']) + len(x.get('kpiBroken', [])) for x in rows)} clipped or split · {len(OKS) - ok0} checks passed, {len(FAILS) - fail0} failed")
            finally:
                ch.terminate()
                try:
                    ch.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    ch.kill()
                shutil.rmtree(prof, ignore_errors=True)
    print(f"{len(OKS)} checks passed, {len(FAILS)} failed, {len(SKIPS)} skipped ({site.name})")
    for k in SKIPS:
        print("  SKIP", k)
    for f in FAILS[:40]:
        print("  FAIL", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
