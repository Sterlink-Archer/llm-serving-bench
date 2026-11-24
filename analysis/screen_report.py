#!/usr/bin/env python3
"""Protocol report across runs: one row per (run, server). It reads only result files (docs/DATA_CONTRACT.md) and
prints a markdown table, also written to results/PROTOCOL_TABLE.md. The columns are:
- the gate;
- startup facts;
- the max rate at the SLO with OpenRelay's client;
- $/request at that rate vs OpenRelay's sync price;
- the ladder detail, prefill and what the run cost.

  python3 -I analysis/screen_report.py [--runs 's1_*'] [--md results/PROTOCOL_TABLE.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import pricebar as PB  # noqa: E402  (the price-bar rule lives in bench/pricebar.py)
import protocol as PR  # noqa: E402  (score_cell: the one scoring rule for traffic cells)
sys.path.insert(0, str(ROOT / "analysis"))
import runfiles as RF  # noqa: E402  (which protocol dirs belong to which server: one rule, shared with journal.py)


def fmt(x, n=2):
    return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="*")
    ap.add_argument("--md", default=str(ROOT / "results" / "PROTOCOL_TABLE.md"))
    a = ap.parse_args()
    rows, ladders, prow = [], [], []
    # Daytona's metered $/h per run (results/metered_usd.json, written by infra/spend.py). Published-rate $/h stays the
    # headline (conservative); the metered rate is what the sandbox was actually billed.
    mfile = ROOT / "results" / "metered_usd.json"
    metered = {}
    orphans = []          # protocol dirs with a summary that no server of their run claims (RF.unattributed): listed, never dropped
    if mfile.exists():
        m = json.loads(mfile.read_text())
        for sid, r in m.get("sandboxes", {}).items():
            if r.get("run_id") and r.get("metered_usd_per_h"):
                metered[r["run_id"]] = dict(usd_h=r["metered_usd_per_h"], read_at=m.get("read_at"), sandbox=sid)
    for rd in sorted(glob.glob(str(ROOT / "results" / a.runs / "*"))):
        rd = pathlib.Path(rd)
        if not (rd / "run.json").exists():
            continue
        run = json.loads((rd / "run.json").read_text())
        cfg = json.loads((rd / "config.json").read_text())
        usd_h = run["usd_per_h"]["total"]
        met = metered.get(f"{rd.parent.name}/{rd.name}") or {}
        usd_h_m = met.get("usd_h")
        for p in RF.unattributed(rd, [s["name"] for s in cfg["servers"]], run):
            orphans.append(str(p.relative_to(ROOT)))
        for srv in cfg["servers"]:
            n = srv["name"]
            rec = run.get("servers", {}).get(n, {})
            sf = json.loads((rd / f"startup_facts_{n}.json").read_text()) if (rd / f"startup_facts_{n}.json").exists() else {}
            gate = None
            for st, v in (rec.get("steps") or {}).items():
                if v.get("gate"):
                    gate = v["gate"]
            for pdir in RF.protocol_dirs(rd, n, [s["name"] for s in cfg["servers"]], run):
                ps = pdir / "protocol_summary.json"
                if not ps.exists():
                    continue
                p = json.loads(ps.read_text())
                be = p["breakeven_rps"]
                mr = p.get("max_rate_at_slo_rps") or 0.0
                usd_req = usd_h / 3600 / mr if mr else None
                bar = PB.price_bar(cfg)
                or_req = PB.usd_per_req(bar)
                pre = p.get("prefill") or {}
                rows.append(dict(run=f"{rd.parent.name}/{rd.name}", server=n, protocol=pdir.name, client=p.get("client"), gpu=cfg["snapshot"]["gpu"],
                                 usd_h=usd_h, gate=gate, backend=(sf.get("attention_backend") or {}), kv=sf.get("kv_cache_tokens"),
                                 rejected=sf.get("rejected"), be=be, max_rate=mr, usd_req=usd_req, or_req=or_req,
                                 ratio=(usd_req / or_req) if usd_req else None, prefill=pre.get("prefill_tok_s"),
                                 usd_h_metered=usd_h_m, ratio_metered=(usd_h_m / 3600 / mr / or_req) if (usd_h_m and mr) else None,
                                 end=rec.get("end"), run_usd=run.get("est_cost_usd"), evicted=bool(run.get("evicted")), error=run.get("error")))
                for rung in p["ladder"]:
                    for sd, res in (rung.get("results") or {}).items():
                        rate = rung.get("rate")
                        res = dict(res)
                        cf = (rung.get("seeds") or {}).get(sd)
                        if "traffic" in str(res.get("client")) and cf and (pdir / cf).exists():
                            # re-score the saved cell with the one rule (last 300 s, censored first tokens = +inf): older
                            # cells were judged on a 240 s window that dropped warmup and end-of-run requests
                            cell = json.loads((pdir / cf).read_text())
                            sc = PR.score_cell(cell["requests"], cell["args"]["duration"], (p.get("slo") or {}).get("ttft_p95_s", 4.0))
                            res.update({f"window_{k}": res.get(f"window_{k}", res.get(k)) for k in ("achieved_rps", "ttft_p95_s", "slo_pass")})
                            res.update(sc, goodput_rps=sc["achieved_rps"] if sc["slo_pass"] else 0.0)
                        elif res.get("duration_s") is not None:
                            res["scored_window_s"] = res["duration_s"]   # vllm bench: the whole run counts
                        ach = res.get("achieved_rps")
                        prow.append(dict(run_id=f"{rd.parent.name}/{rd.name}", config_id=f"{rd.parent.name}:{n}", protocol=pdir.name, gpu=cfg["snapshot"]["gpu"],
                                         client=res.get("client") or p.get("client"), multiple=rung.get("multiple"), offered_rps=rate, seed=int(sd),
                                         kept=str(rung.get("kept_seed")) == sd, slo_pass=res.get("slo_pass"), achieved_rps=ach,
                                         goodput_rps=res.get("goodput_rps"), ttft_p95_s=res.get("ttft_p95_s"), tpot_p50_s=res.get("tpot_p50_s"),
                                         tpot_p95_s=res.get("tpot_p95_s"), errors=res.get("errors"), usd_per_h=usd_h,
                                         usd_per_req_at_offered=(usd_h / 3600 / rate) if rate else None,
                                         usd_per_req_at_achieved=(usd_h / 3600 / ach) if ach else None,
                                         openrelay_sync_usd_per_req=or_req if bar["is_openrelay_gemma"] else None,
                                         price_bar_name=bar["name"], price_bar_usd_per_req=or_req, price_bar_source=bar.get("source"),
                                         price_model=bar.get("model"), price_bar_is_openrelay_gemma=bar["is_openrelay_gemma"],
                                         ratio_sync_at_offered=((usd_h / 3600 / rate) / or_req) if rate else None,
                                         usd_per_h_metered=usd_h_m, metered_read_at=met.get("read_at"),
                                         duration_s=res.get("scored_window_s"), scored_window_s=res.get("scored_window_s"),
                                         ttft_p95_lower_s=res.get("ttft_p95_lower_s"), ttft_censored=res.get("ttft_censored"),
                                         window_ttft_p95_s=res.get("window_ttft_p95_s"), window_achieved_rps=res.get("window_achieved_rps"),
                                         prompt_tok_xcheck_ratio=res.get("prompt_tok_xcheck_ratio"), prompt_tok_xcheck_pass=res.get("prompt_tok_xcheck_pass"),
                                         spec_accept_len=res.get("spec_accept_len"), spec_acceptance_rate=res.get("spec_acceptance_rate"),
                                         prompt_tokens_mean=res.get("prompt_tokens_mean"), stratified_lengths=res.get("stratified_lengths"),
                                         server=n, claim_group=claim_group(f"{rd.parent.name}/{rd.name}", n) + (f"_{p['group_tag']}" if p.get("group_tag") else ""),
                                         column=p.get("column") or ("A" if "vllm bench" in str(res.get("client") or p.get("client")) or p.get("client") == "vllmbench" else "B"),
                                         workload=p.get("workload"), cached_tokens_mean=res.get("cached_tokens_mean"),
                                         completion_tokens_mean=res.get("completion_tokens_mean"),
                                         cached_share_cell=res.get("cached_share_cell"), cached_share_scored=res.get("cached_share_scored"),
                                         server_prefix_cache_hit_ratio=res.get("server_prefix_cache_hit_ratio"),
                                         price_bar_usd_per_req_cached_realized=cached_price(bar, res),
                                         exact_prompt_tokens=bool((p.get("calibration") or {}).get("exact_prose_tokens")),
                                         ratio_sync_at_offered_metered=((usd_h_m / 3600 / rate) / or_req) if (rate and usd_h_m) else None,
                                         file=str((pdir / (rung.get("seeds") or {}).get(sd, "")).relative_to(ROOT)) if (rung.get("seeds") or {}).get(sd) else None,
                                         summary_file=str(ps.relative_to(ROOT))))
                    ladders.append(dict(run=rd.parent.name, server=n, protocol=pdir.name, mult=rung.get("multiple"), rate=rung.get("rate"),
                                        p95=rung.get("ttft_p95_s"), tpot50=rung.get("tpot_p50_s"), tpot95=rung.get("tpot_p95_s"),
                                        achieved=rung.get("achieved_rps"), ok=rung.get("slo_pass"), errors=rung.get("errors"),
                                        skipped=rung.get("skipped"), accept=rung.get("spec_accept_len")))
    L = ["Bar: OpenRelay sync Gemma 4 31B, $0.000779 per 9.5K-in/380-out request, unless the config names another (`price_model`, e.g. "
         "GLM 5.3 Flash, or `price_bar`, e.g. Inkling's OpenRouter price; see claims.json price_bar_name). "
         "ratio = our $/req at the max SLO rate / the bar's (< 1 = cheaper; "
         "it is also the break-even utilisation). SLO: p95 TTFT < 4 s, errors < 1%. $/h = Daytona's PUBLISHED rates (GPU + vCPU + RAM + disk); "
         "'metered' = what Daytona actually billed that sandbox (results/metered_usd.json), shown alongside.", "",
         "| run | gpu | server / protocol (client) | $/h | gate | attn sliding / global | KV tokens | break-even req/s | max req/s at SLO | our $/req | ratio vs OpenRelay | metered $/h | ratio (metered) | prefill tok/s | run $ (est) | end |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = r["gate"] or {}
        L.append(f"| {r['run']} | {r['gpu']} | {r['server']} / {r['protocol']} ({r['client']}) | {r['usd_h']:.2f} | "
                 f"{fmt(g.get('acc'), 3)} {'PASS' if g.get('pass') else 'FAIL' if g else '-'} | {r['backend'].get('sliding')} / {r['backend'].get('global_512')} | "
                 f"{fmt(r['kv'])} | {fmt(r['be'])} | {fmt(r['max_rate'])} | {fmt(r['usd_req'], 6)} | {fmt(r['ratio'])} | "
                 f"{fmt(r['usd_h_metered'])} | {fmt(r['ratio_metered'])} | {fmt(r['prefill'], 0)} | "
                 f"{fmt(r['run_usd'])} | {r['end'] or ''}{' EVICTED' if r['evicted'] else ''}{(' ERROR ' + r['error'][:80]) if r['error'] else ''} |")
    L += ["", "| run / protocol | x break-even | offered req/s | achieved | p95 TTFT s | TPOT p50 / p95 ms | errors | SLO | MTP accept len |", "|---|---|---|---|---|---|---|---|---|"]
    for x in ladders:
        L.append(f"| {x['run']} / {x['protocol']} | {fmt(x['mult'])} | {fmt(x['rate'])} | {fmt(x['achieved'])} | {fmt(x['p95'])} | "
                 f"{fmt((x['tpot50'] or 0) * 1000, 1)} / {fmt((x['tpot95'] or 0) * 1000, 1)} | {fmt(x['errors'])} | "
                 f"{'PASS' if x['ok'] else ('skipped: ' + x['skipped']) if x['skipped'] else 'fail'} | {fmt(x['accept'])} |")
    if orphans:
        L += ["", "**Protocol dirs not attributed to any server (not in the tables above; fix the run's step names or "
              "run.json):** " + ", ".join(f"`{o}`" for o in orphans)]
        print(f"[screen_report] WARNING: {len(orphans)} protocol dir(s) belong to no server: {orphans}", file=sys.stderr)
    txt = "\n".join(L) + "\n"
    pathlib.Path(a.md).write_text(txt)
    (ROOT / "results" / "protocol_rows.json").write_text(json.dumps(prow, indent=1))
    claims(prow)
    print(txt)


CLAIM_MIN_SEEDS, CLAIM_MIN_DURATION_S = 2, 300


def current_config(label):
    """The current config file of a label (configs/ or recipes/*/configs/), {} if none."""
    for f in [ROOT / "configs" / f"{label}.json", *(ROOT / "recipes").glob(f"*/configs/{label}.json")]:
        if f.exists():
            return json.loads(f.read_text())
    return {}


def _cfgs(run_id):
    """(the run's own config.json copy, the label's current config file); either may be {}."""
    own_f = ROOT / "results" / run_id / "config.json"
    own = json.loads(own_f.read_text()) if own_f.exists() else {}
    return own, current_config(run_id.split("/")[0])


def cfg_key(run_id, server, key):
    """One lookup rule for claim_group / origin (MI355X recipe session's paired boxes run two recipes on one GPU):
    server-level key in the run's copy, else server-level in the current config, else config-level in the copy, else in
    the current config, else None."""
    own, cur = _cfgs(run_id)
    for cfg in (own, cur):
        for srv in cfg.get("servers", []):
            if srv.get("name") == server and srv.get(key):
                return srv[key]
    for cfg in (own, cur):
        if cfg.get(key):
            return cfg[key]
    return None


def run_meta(run_id, server=None):
    """origin: "openrelay_recipe" only for an exact reproduction of OpenRelay's recipe (the config or that server says so);
    everything else, including variants of their recipe, is "ours". Plus gpu and the engine of that server."""
    own, cur = _cfgs(run_id)
    cfg = own or cur
    srvs = [x for x in cfg.get("servers", []) if server is None or x.get("name") == server] or cfg.get("servers", [])
    launch = " ".join(x.get("launch", "") for x in srvs)
    engine = "sglang" if "sglang" in launch else "llama.cpp" if "llama-server" in launch else "vllm"
    return dict(origin=cfg_key(run_id, server, "origin") or "ours", gpu=(cfg.get("snapshot") or {}).get("gpu"), engine=engine)


def claim_group(run_id, server=None):
    """Runs that measure the same recipe share a claim group: an explicit `claim_group` (server-level, then config-level;
    see cfg_key), else the label without its _seedN / _text suffixes (twin runs of one config)."""
    g = cfg_key(run_id, server, "claim_group")
    if g:
        return g
    import re
    return re.sub(r"(_text)?(_seed\d+)?$", "", run_id.split("/")[0])


def cached_price(bar, res):
    """The bar's price for this cell's REALIZED mean request, cached input billed at the bar's cached rate (column C:
    OpenRelay bills cached input at 1/10 of input). None without realized prompt, completion and cached token means."""
    pm, cm, om = res.get("prompt_tokens_mean"), res.get("cached_tokens_mean"), res.get("completion_tokens_mean")
    if pm is None or cm is None or om is None:
        return None
    cr = bar.get("cached_usd_per_m")
    cr = bar["in_usd_per_m"] if cr is None else cr
    return ((pm - cm) * bar["in_usd_per_m"] + cm * cr + om * bar["out_usd_per_m"]) / 1e6


def cell_flaws(x, col):
    """Why one protocol cell cannot back a claim (empty list = claim-eligible)."""
    f = []
    sw = x.get("scored_window_s")
    if sw is None or sw < CLAIM_MIN_DURATION_S * 0.95:   # the SCORED window, not the configured duration (Grok review caveat 1)
        f.append(f"scored window {sw if sw is None else round(sw)} s < 300 s")
    if x.get("prompt_tok_xcheck_pass") is False:
        f.append("server-vs-client prompt tokens differ by > 1%")
    if x.get("prompt_tok_xcheck_pass") is None:
        f.append("no server-vs-client prompt-token check recorded")
    if col == "B":
        if not x.get("exact_prompt_tokens"):
            f.append("prompts sized by chars/token, not the tokenizer (ran ~5-9% short of 9.5K)")
        if not x.get("stratified_lengths"):
            f.append("lengths drawn per seed, not stratified (seed 7's draws averaged ~4% under 9.5K)")
        pmv = x.get("prompt_tokens_mean")
        if pmv and abs(pmv / 9500 - 1) > 0.02:
            f.append(f"realized prompt mean {pmv:.0f} tokens, outside 9.5K +-2%")
    return f


def monotonic_claim(by_rate):
    """THE claim-rate rule (lead, 2026-11-22): the claim rate is the highest rate R such that every cell of EVERY tested
    rung <= R passes (the monotonic prefix of the ladder). A rung where any cell fails ends the prefix, even when a higher
    rung passes on every seed (ikC_pro6000_mixed_dspark column B: seed 7 fails 1.35x, reproduced on a fresh box, then passes
    1.5x; the old highest-all-pass rule claimed 1.5x). Returns (best (rate, cells) or None, the first failing rate or None,
    the highest all-pass rate ABOVE the first failure or None), so the higher pass can be shown beside the claim."""
    best = first_fail = above = None
    for rate in sorted(by_rate):
        rs = by_rate[rate]
        ok = all(x.get("slo_pass") for x in rs)
        if first_fail is None:
            if ok:
                best = (rate, rs)
            else:
                first_fail = rate
        elif ok:
            above = rate
    return best, first_fail, above


def claims(prow):
    """CLAIMS.md / claims.json (brainstorm round 2): per (claim group, column) the highest offered rate at which EVERY seed
    meets the SLO, priced on the WORST seed's ACHIEVED req/s, at published and metered $/h. Column A = OpenRelay's
    random-token client (vllm bench serve); column B = our real-text client. A cell is claim-eligible with a 300 s window,
    the server-vs-client prompt-token check within 1%, and (column B) tokenizer-exact, stratified prompts whose realized mean
    is within 9.5K +-2%. When a group has eligible cells, the claim is judged on them alone; older ineligible cells are listed
    as excluded, and any excluded cell that FAILS at the claimed rate blocks the claim (no silent cherry-picking). A claim
    needs >= 2 distinct seeds at its rate; anything less is labelled with the reasons. The claim rate is the top of the
    MONOTONIC prefix: every cell of every tested rung up to it passes; a failing rung below a higher all-pass rung ends the
    prefix, and the higher pass is reported beside the claim (non_monotonic_pass_rps, fails_below_at_rps)."""
    groups = {}
    for r in prow:
        groups.setdefault((r["claim_group"], r["column"]), []).append(r)
    out = []
    for (g, col), rows in sorted(groups.items()):
        eligible = [x for x in rows if not cell_flaws(x, col)]
        pool = eligible or rows
        by_rate = {}
        for r in pool:
            by_rate.setdefault(r["offered_rps"], []).append(r)
        best, first_fail, above = monotonic_claim(by_rate)
        metas = [run_meta(rid, srv) for rid, srv in sorted({(x["run_id"], x.get("server")) for x in rows})]
        origins = sorted({m["origin"] for m in metas})
        excluded = [x for x in rows if x not in eligible] if eligible else []
        rec = dict(group=g, column=col, column_name={"A": "OpenRelay client, random tokens", "B": "our client, real text"}.get(
                       col, "cached traffic, supplementary: NOT OpenRelay's method"),
                   origin=origins[0] if len(origins) == 1 else "mixed:" + "+".join(origins),
                   gpu="+".join(sorted({m["gpu"] or "?" for m in metas})), engine="+".join(sorted({m["engine"] for m in metas})),
                   rates_tested=sorted({x["offered_rps"] for x in rows}), runs=sorted({x["run_id"] for x in rows}),
                   price_bar_is_openrelay_gemma=all(x.get("price_bar_is_openrelay_gemma") for x in rows),
                   price_model=rows[0].get("price_model") if len({x.get("price_model") for x in rows}) == 1 else "MIXED",
                   judged_on="claim-eligible cells only" if eligible else "all cells (none claim-eligible)",
                   first_failing_rate_rps=first_fail, non_monotonic_pass_rps=above,
                   non_monotonic_note=(f"passes {above} req/s on every seed, but fails the {first_fail} req/s rung below it"
                                       if above is not None else None),
                   excluded_cells=[dict(cell=f"{x['run_id']}:s{x['seed']}@{x['offered_rps']}", slo_pass=x.get("slo_pass"),
                                        why=cell_flaws(x, col)) for x in excluded])
        if best:
            rate, rs = best
            seeds = sorted({(x["run_id"], x["seed"]) for x in rs})
            worst = min(rs, key=lambda x: x.get("achieved_rps") or 0)
            ach = worst.get("achieved_rps")
            usd_h = max(x["usd_per_h"] for x in rs)
            bars = sorted({(x.get("price_bar_name"), x.get("price_bar_usd_per_req")) for x in rs})
            bar_name, bar_req = bars[0] if len(bars) == 1 else ("MIXED: " + " | ".join(str(b[0]) for b in bars), max(b[1] for b in bars))
            gemma_bar = len(bars) == 1 and all(x.get("price_bar_is_openrelay_gemma") for x in rs)   # vs-OpenRelay-Gemma fields only then
            std_req = bar_req            # the bar's price for the standard 9.5K/380 request, nothing cached
            if col == "C" and worst.get("price_bar_usd_per_req_cached_realized"):
                # column C is priced as the provider would bill THIS traffic: cached input at its cached rate, on the cached
                # tokens the worst seed's cell actually got (usage.prompt_tokens_details.cached_tokens)
                bar_name, bar_req = bar_name + ", cached input at its cached rate on the realized cached tokens", \
                    worst["price_bar_usd_per_req_cached_realized"]
            mets = [x.get("usd_per_h_metered") for x in rs]
            usd_h_m = max(mets) if mets and all(mets) else None
            why_not = []
            nseeds = len({sd for _, sd in seeds})
            if nseeds < CLAIM_MIN_SEEDS:
                why_not.append(f"{nseeds} seed(s) at this rate (need {CLAIM_MIN_SEEDS})")
            if not eligible:
                for x in rs:
                    for f in cell_flaws(x, col):
                        if f not in why_not:
                            why_not.append(f)
            fails = [f"{x['run_id']}:s{x['seed']}" for x in excluded if x["offered_rps"] == rate and not x.get("slo_pass")]
            if fails:
                why_not.append("an excluded (ineligible) cell FAILS at this rate: " + ", ".join(fails))
            pm = [x.get("prompt_tokens_mean") for x in rs if x.get("prompt_tokens_mean")]
            rec.update(max_rate_all_seeds_pass=rate, seeds=[f"{rid}:s{sd}" for rid, sd in seeds],
                       worst_seed_achieved_rps=ach, worst_seed_ttft_p95_s=max(x.get("ttft_p95_s") or 0 for x in rs),
                       prompt_tokens_mean_range=[min(pm), max(pm)] if pm else None,
                       spec_accept_len=[x.get("spec_accept_len") for x in rs],
                       xcheck_ratio_range=[min(v for v in [x.get("prompt_tok_xcheck_ratio") for x in rs] if v is not None),
                                           max(v for v in [x.get("prompt_tok_xcheck_ratio") for x in rs] if v is not None)]
                       if any(x.get("prompt_tok_xcheck_ratio") is not None for x in rs) else None,
                       usd_per_h=usd_h, usd_per_req=(usd_h / 3600 / ach) if ach else None,
                       price_bar_name=bar_name, price_bar_usd_per_req=bar_req, price_bar_is_openrelay_gemma=gemma_bar,
                       price_model=rs[0].get("price_model") if len({x.get("price_model") for x in rs}) == 1 else "MIXED",
                       ratio_vs_bar=(usd_h / 3600 / ach / bar_req) if ach else None,
                       ratio_vs_bar_metered=(usd_h_m / 3600 / ach / bar_req) if (ach and usd_h_m) else None,
                       ratio_vs_openrelay_sync=(usd_h / 3600 / ach / std_req) if (ach and gemma_bar) else None,
                       usd_per_h_metered=usd_h_m, usd_per_req_metered=(usd_h_m / 3600 / ach) if (ach and usd_h_m) else None,
                       ratio_vs_openrelay_sync_metered=(usd_h_m / 3600 / ach / std_req) if (ach and usd_h_m and gemma_bar) else None,
                       cached_share_scored=[x.get("cached_share_scored") for x in rs] if col == "C" else None,
                       server_prefix_cache_hit_ratio=[x.get("server_prefix_cache_hit_ratio") for x in rs] if col == "C" else None,
                       claim_grade=not why_not, not_claim_grade_because=why_not)
        else:
            why = ["no rate passed on every seed"] if first_fail is None else \
                [f"the lowest tested rung ({first_fail} req/s) fails on at least one cell"]
            rec.update(max_rate_all_seeds_pass=None, claim_grade=False, not_claim_grade_because=why)
        out.append(rec)
    (ROOT / "results" / "claims.json").write_text(json.dumps(dict(schema="claims/1", openrelay_sync_usd_per_req=PB.usd_per_req(PB.price_bar()),
                                                               rules=claims.__doc__, claims=out), indent=1))
    L = ["# Claims (generated by analysis/screen_report.py; do not edit)", "", claims.__doc__.strip(), "",
         "| group | origin | gpu / engine | column | claim rate (monotonic: every rung up to it passes on every seed) | seeds | worst-seed achieved req/s | worst p95 TTFT s | $/req (published $/h) | price bar | x bar | $/req (metered) | x bar (metered) | claim-grade? | higher pass above a failing rung |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in out:
        L.append(f"| {c['group']} | {c['origin']} | {c['gpu']} / {c['engine']} | {c['column']} ({c['column_name']}) | {fmt(c.get('max_rate_all_seeds_pass'))} | {len(c.get('seeds') or [])} | "
                 f"{fmt(c.get('worst_seed_achieved_rps'), 3)} | {fmt(c.get('worst_seed_ttft_p95_s'))} | {fmt(c.get('usd_per_req'), 6)} | "
                 f"{c.get('price_bar_name') or '-'} | {fmt(c.get('ratio_vs_bar'), 3)} | {fmt(c.get('usd_per_req_metered'), 6)} | {fmt(c.get('ratio_vs_bar_metered'), 3)} | "
                 f"{'YES' if c['claim_grade'] else 'no: ' + '; '.join(c['not_claim_grade_because'])} | {c.get('non_monotonic_note') or '-'} |")
    (ROOT / "results" / "CLAIMS.md").write_text("\n".join(L) + "\n")
    # README.md renders from claims.json (bin/build_readme.py; its --check is in the gate): rebuild it wherever claims.json
    # is written, so a claims regeneration never leaves the README stale (2026-11-22: a session ran screen_report.py
    # alone and every gate went red on the drift check)
    import subprocess
    if (ROOT / "analysis" / "figures.py").exists():   # figures and their captions also render from the result files
        subprocess.run([sys.executable, "-I", str(ROOT / "analysis" / "figures.py")], check=True, capture_output=True)
    subprocess.run([sys.executable, "-I", str(ROOT / "bin" / "build_readme.py")], check=True, capture_output=True)


if __name__ == "__main__":
    main()
