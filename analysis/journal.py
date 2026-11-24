#!/usr/bin/env python3
"""The experiment journey (Maks 11-22: "track the numbers you get as you try things... demonstrate how it started, what
was done and what worked and where that landed us"). One record per decision step in results/experiments.jsonl
(append-only; a later line with the same `step` supersedes an earlier one), rendered to docs/JOURNAL.md.

Metrics are computed from the run's own files (never typed): protocol_<server>/protocol_summary.json (OpenRelay client
preferred), startup_facts_<server>.json, gate_*.json / eval_*.json, run.json, cost_measured.json. The config diff vs the
parent step is computed from the two runs' config.json (GPU, host, image, weights, launch flags).

  python3 -I analysis/journal.py add --step 3 --parent 1 --kind change --run s1_x/STAMP --server nvfp4 --basis screen \
      --change "RTX PRO 6000 NVFP4" --hypothesis "..." --decision kept --why "..."
  python3 -I analysis/journal.py render          # -> docs/JOURNAL.md
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import glob
import json
import pathlib
import shlex
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import pricebar as PB  # noqa: E402  (the one rule for which bar a config is priced against)
sys.path.insert(0, str(ROOT / "analysis"))
import runfiles as RF  # noqa: E402  (which protocol dirs belong to which server)
EXP = ROOT / "results" / "experiments.jsonl"
DOC = ROOT / "docs" / "JOURNAL.md"
KEYS = ["slo_goodput_rps", "max_rate_at_slo_rps", "breakeven_rps", "achieved_rps_at_breakeven", "usd_per_req", "usd_per_req_at_achieved",
        "ratio_sync", "ttft_p95_s", "tpot_p50_s", "prefill_tok_s", "kv_tokens", "mtp_accept_len", "gsm8k_acc", "download_mb_s",
        "engine_ready_s", "usd_per_h", "run_usd_est"]


# Two writers append (the build session: steps 0-99; the MI355X recipe session: steps 100-199). Every append and every
# render holds an exclusive flock on results/experiments.jsonl.lock, and each record is written in ONE write() call.
STEP_RANGES = {"build": range(0, 100), "mi355x-recipe": range(100, 200), "inkling": range(200, 300), "glm": range(300, 400),
               "gemm-headroom": range(400, 500)}
COLUMN_CLIENT = {"A": "vllmbench", "B": "traffic"}   # A = OpenRelay's vllm bench serve, B = our real-text client


def owned_files(rd, prefix, server, servers):
    """<prefix>_*.json files that belong to `server` in a run with several servers: a file is a server's when its name starts
    with <prefix>_<server>_ (longest server name wins, so 'mx' never takes 'mx_recipe's files). A server with no files of its
    own gets the un-prefixed ones (single-server runs write gate_gsm8kor100.json)."""
    names = sorted(set(servers) | {server}, key=len, reverse=True)
    mine, unowned = [], []
    for g in sorted(glob.glob(str(rd / f"{prefix}_*.json"))):
        n = pathlib.Path(g).name
        owner = next((s for s in names if n.startswith(f"{prefix}_{s}_")), None)
        if owner == server:
            mine.append(g)
        elif owner is None:
            unowned.append(g)
    return mine or unowned


@contextlib.contextmanager
def locked():
    EXP.parent.mkdir(parents=True, exist_ok=True)
    with open(str(EXP) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def load():
    recs = {}
    if EXP.exists():
        for line in EXP.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["step"]] = r
    return [recs[k] for k in sorted(recs)]


def flags(launch):
    toks = shlex.split(launch or "")
    out, i = {}, 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                out[t] = toks[i + 1]
                i += 2
                continue
            out[t] = True
        i += 1
    return out


def run_dir(run_id):
    return ROOT / "results" / run_id


def server_cfg(run_id, server):
    cfg = json.loads((run_dir(run_id) / "config.json").read_text())
    return cfg, next(s for s in cfg["servers"] if s["name"] == server)


def metrics(run_id, server, column=None):
    """column: "A" or "B" picks that column's protocol summary and nothing else; None = A, falling back to B (with a
    warning) when the run has no column-A summary for this server."""
    rd = run_dir(run_id)
    run = json.loads((rd / "run.json").read_text())
    servers = list((run.get("servers") or {}).keys())
    m = {k: None for k in KEYS}
    m["usd_per_h"] = run["usd_per_h"]["total"]
    m["run_usd_est"] = run.get("est_cost_usd")
    srv = run.get("servers", {}).get(server, {})
    dl = srv.get("download") or {}
    if dl.get("bytes") and dl.get("secs"):
        m["download_mb_s"] = round(dl["bytes"] / 1e6 / dl["secs"], 1)
    m["engine_ready_s"] = (srv.get("serve") or {}).get("ready_s")
    sf = rd / f"startup_facts_{server}.json"
    if sf.exists():
        m["kv_tokens"] = json.loads(sf.read_text()).get("kv_cache_tokens")
    for g in owned_files(rd, "gate", server, servers) + owned_files(rd, "eval", server, servers):
        d = json.loads(pathlib.Path(g).read_text())
        acc = (d.get("gate") or {}).get("acc") or ((d.get("summary") or {}).get("gsm8kor") or (d.get("summary") or {}).get("gsm8k") or {}).get("acc")
        if acc is not None:
            m["gsm8k_acc"] = acc
    found = {}
    for pd in RF.protocol_dirs(rd, server, servers, run):          # one rule with screen_report (analysis/runfiles.py)
        p = pd / "protocol_summary.json"
        d = json.loads(p.read_text())
        col = next((c for c, cl in COLUMN_CLIENT.items() if d.get("client") == cl), None)
        if col == "A":
            found["A"] = (p, d)          # as before: the last vllm-bench summary, the first traffic one
        elif col:
            found.setdefault(col, (p, d))
    m["_column"] = column or "A"
    ps = found.get(m["_column"])
    if ps is None and column is None and "B" in found:
        ps, m["_column"] = found["B"], "B"
        print(f"[journal] {run_id}:{server} has no column-A (vllm bench) summary: recording column B; pass --column to choose")
    if ps:
        p, d = ps
        m["breakeven_rps"] = d.get("breakeven_rps")
        m["max_rate_at_slo_rps"] = d.get("max_rate_at_slo_rps")
        m["slo_goodput_rps"] = d.get("max_goodput_rps")
        m["prefill_tok_s"] = (d.get("prefill") or {}).get("prefill_tok_s")
        be = next((r for r in d["ladder"] if abs((r.get("multiple") or 0) - 1.0) < 0.05 and "kept_seed" in r), None)
        best = max((r for r in d["ladder"] if r.get("slo_pass")), key=lambda r: r["rate"], default=None)
        ref = best or be
        if ref:
            m["ttft_p95_s"], m["tpot_p50_s"], m["mtp_accept_len"] = ref.get("ttft_p95_s"), ref.get("tpot_p50_s"), ref.get("spec_accept_len")
        if be:
            m["achieved_rps_at_breakeven"] = be.get("achieved_rps")
        if m["max_rate_at_slo_rps"]:
            m["usd_per_req"] = m["usd_per_h"] / 3600 / m["max_rate_at_slo_rps"]
            # vs the run's OWN bar (config price_model / price_bar), not the ladder's break-even: GLM runs launched before
            # cf9e936 built their ladder on the Gemma price. For a Gemma run both give the same number.
            cfgp = rd / "config.json"
            bar = PB.price_bar(json.loads(cfgp.read_text()) if cfgp.exists() else {})
            m["ratio_sync"] = m["usd_per_req"] / PB.usd_per_req(bar)
        if m["achieved_rps_at_breakeven"]:
            m["usd_per_req_at_achieved"] = m["usd_per_h"] / 3600 / m["achieved_rps_at_breakeven"]
        m["_protocol_file"] = str(p.relative_to(ROOT))
    return m, run


def diff(parent, child):
    (pc, ps), (cc, cs) = parent, child
    d = {}
    for k in ("gpu", "cpu", "mem", "image"):
        if pc["snapshot"].get(k) != cc["snapshot"].get(k):
            d[k] = [pc["snapshot"].get(k), cc["snapshot"].get(k)]
    pw = (ps.get("weights") or [{}])[0].get("repo")
    cw = (cs.get("weights") or [{}])[0].get("repo")
    if pw != cw:
        d["weights"] = [pw, cw]
    pf, cf = flags(ps.get("launch")), flags(cs.get("launch"))
    add = {k: v for k, v in cf.items() if k not in pf or pf[k] != v}
    rem = {k: v for k, v in pf.items() if k not in cf}
    if add:
        d["flags_added_or_changed"] = add
    if rem:
        d["flags_removed"] = rem
    return d


def cmd_add(a):
    if a.step not in STEP_RANGES[a.owner]:
        raise SystemExit(f"step {a.step} is outside {a.owner}'s range {STEP_RANGES[a.owner]}")
    m, run = metrics(a.run[0], a.server, a.column) if a.run else ({k: None for k in KEYS}, {})
    recs = {r["step"]: r for r in load()}
    rec = dict(schema="experiment/1", step=a.step, t=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), kind=a.kind,
               config_id=f"{a.run[0].split('/')[0]}:{a.server}" if a.run else None, parent_step=a.parent,
               parent_config_id=(recs.get(a.parent) or {}).get("config_id"), change=a.change[:80], hypothesis=a.hypothesis, why=a.why,
               run_ids=a.run, basis=a.basis, decision=a.decision, **{k: v for k, v in m.items() if not k.startswith("_")})
    if a.run:
        rec["column"] = m.get("_column")
        try:
            bar = PB.price_bar(server_cfg(a.run[0], a.server)[0])
            rec["price_bar"] = dict(name=bar["name"], usd_per_req=round(PB.usd_per_req(bar), 7), source=bar.get("source"))
        except Exception as e:  # noqa: BLE001
            rec["price_bar"] = {"error": repr(e)[:200]}
    rec["evidence_files"] = [f"results/{r}/run.json" for r in a.run] + ([m["_protocol_file"]] if m.get("_protocol_file") else []) + \
        [str(p.relative_to(ROOT)) for r in a.run for p in run_dir(r).glob(f"startup_facts_{a.server}.json")] + (a.evidence or [])
    par = recs.get(a.parent)
    if par and par.get("run_ids") and a.run:
        try:
            rec["diff"] = diff(server_cfg(par["run_ids"][0], par["config_id"].split(":")[1]), server_cfg(a.run[0], a.server))
        except Exception as e:  # noqa: BLE001
            rec["diff"] = {"error": repr(e)[:200]}
        rec["delta_vs_parent"] = {k: [par.get(k), rec.get(k)] for k in ("slo_goodput_rps", "usd_per_req", "ttft_p95_s", "tpot_p50_s",
                                                                        "prefill_tok_s", "kv_tokens", "download_mb_s")
                                  if par.get(k) is not None or rec.get(k) is not None}
    if a.note:
        rec["note"] = a.note
    with locked():
        with open(EXP, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1)[:2500])


def fmt(x, n=2):
    return "-" if x is None else (f"{x:,.{n}f}" if isinstance(x, float) else f"{x:,}" if isinstance(x, int) else str(x))


def cmd_render(a):
    with locked():
        recs = load()
    L = ["# Journal: from the first box to the current best",
         "",
         "Each row is one decision step, read from `results/experiments.jsonl`, and is regenerated by",
         "`python3 -I analysis/journal.py render`.",
         "- Every metric comes from the files named in that step's `evidence_files`.",
         "- Price bar: OpenRelay sync Gemma 4 31B, $0.000779 per 9.5K-in / 380-out request, unless the step's `price_bar` names "
         "another (GLM steps 300+: OpenRelay sync GLM 5.3 Flash; Inkling steps 200+: OpenRouter PrimeIntellect). ratio_sync is vs that bar.",
         "- SLO: p95 TTFT under 4 s on OpenRelay's own client (`vllm bench serve`, random tokens, open-loop Poisson).",
         "- `ratio`: our $/request at the highest rate that meets the SLO, divided by OpenRelay's. Below 1 means we are cheaper.",
         "",
         "| step (column) | change | decision | GPU $/h | max req/s at SLO (break-even) | achieved at break-even | $/req | ratio | p95 TTFT s | TPOT p50 ms | prefill tok/s | KV tokens | GSM8K | download MB/s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        L.append(f"| {r['step']}{' (' + r['column'] + ')' if r.get('column') else ''} | {r['change']} | **{r['decision']}** | {fmt(r.get('usd_per_h'))} | {fmt(r.get('max_rate_at_slo_rps'))} "
                 f"({fmt(r.get('breakeven_rps'))}) | {fmt(r.get('achieved_rps_at_breakeven'))} | {fmt(r.get('usd_per_req'), 6)} | {fmt(r.get('ratio_sync'))} | "
                 f"{fmt(r.get('ttft_p95_s'))} | {fmt((r.get('tpot_p50_s') or 0) * 1000 if r.get('tpot_p50_s') is not None else None, 1)} | "
                 f"{fmt(r.get('prefill_tok_s'), 0)} | {fmt(r.get('kv_tokens'))} | {fmt(r.get('gsm8k_acc'), 3)} | {fmt(r.get('download_mb_s'), 0)} |")
    L.append("")
    for r in recs:
        L += [f"## Step {r['step']}: {r['change']}  ({r['decision']})", "",
              f"- **Hypothesis:** {r.get('hypothesis') or '-'}",
              f"- **What happened / why {r['decision']}:** {r.get('why') or '-'}",
              f"- **Parent:** step {r.get('parent_step')}  ·  **runs:** {', '.join('`' + x + '`' for x in r.get('run_ids') or []) or '-'}"]
        if r.get("diff"):
            L.append(f"- **Diff vs parent:** `{json.dumps(r['diff'])}`")
        if r.get("delta_vs_parent"):
            L.append("- **Delta vs parent:** " + "; ".join(f"{k} {fmt(v[0])} → {fmt(v[1])}" for k, v in r["delta_vs_parent"].items()))
        if r.get("note"):
            L.append(f"- **Note:** {r['note']}")
        L.append(f"- **Evidence:** " + ", ".join(f"`{e}`" for e in r.get("evidence_files") or []))
        L.append("")
    DOC.write_text("\n".join(L) + "\n")
    print(f"wrote {DOC} ({len(recs)} steps)")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("add")
    s.add_argument("--step", type=int, required=True)
    s.add_argument("--parent", type=int)
    s.add_argument("--kind", choices=["baseline", "change", "rejection", "infra", "calibration"], required=True)
    s.add_argument("--run", action="append", default=[])
    s.add_argument("--server", default="")
    s.add_argument("--basis", default="screen")
    s.add_argument("--change", required=True)
    s.add_argument("--hypothesis", default="")
    s.add_argument("--why", default="")
    s.add_argument("--decision", choices=["baseline", "kept", "reverted", "rejected", "info"], required=True)
    s.add_argument("--note", default="")
    s.add_argument("--evidence", action="append")
    s.add_argument("--owner", choices=sorted(STEP_RANGES), default="build", help="which session writes: fixes the allowed step range")
    s.add_argument("--column", choices=sorted(COLUMN_CLIENT), help="A = vllm bench serve summary, B = real-text traffic summary "
                   "(default A, falling back to B with a warning when the run has no A)")
    sp.add_parser("render")
    a = ap.parse_args()
    cmd_add(a) if a.cmd == "add" else cmd_render(a)


if __name__ == "__main__":
    main()
