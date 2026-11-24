#!/usr/bin/env python3
"""Build the local benchmark dashboard from the repo's own data files. Stdlib only.

  python3 -I dashboard/build.py                  # real data  -> dashboard/site/
  python3 -I dashboard/build.py --demo           # contract-shaped MOCK data (dashboard/demo/) -> dashboard/site-demo/
  python3 -I dashboard/serve.py                  # view it at http://127.0.0.1:8765

Reads ONLY the files named in dashboard/CONTRACT.md (the data contract agreed with the build session):
  infra/ledger.jsonl, results/<label>/<stamp>/{run.json, config.json, [out/]m_*/cells/*.json, [out/]m_*/matrix_summary.json,
  eval_*.json, gpumon.csv, startup_facts_*.json, protocol_*/protocol_summary.json, cost_measured.json},
  configs/*.json, research/openrelay_prices_*.json, research/openrouter_bar_google_gemma-4-31b-it_*.json,
  infra/spend_*.json, dashboard/references.json.
Every input file is fingerprinted (sha256, bytes, mtime) into the site's Sources page, and every derived number carries
the run id and file it came from. Per-cell prices come ONLY from results/cost_rows.json (written by analysis/report.py;
contract section 6: the JSON is the contract, the module is not imported). The 9.5K/380 bar is plain arithmetic on the
price JSON fields of contract section 8. Nothing outside dashboard/ is written.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dtm
import glob
import hashlib
import json
import math
import os
import pathlib
import re
import shlex
import shutil
import sys
import time

DASH = pathlib.Path(__file__).resolve().parent
ROOT = DASH.parent
sys.path.insert(0, str(ROOT / "infra"))

CONTRACT_VERSION = "v1-frozen"     # docs/DATA_CONTRACT.md, frozen by the build session 2026-11-22 ~04:05 UTC
CONTRACT_FILE = "docs/DATA_CONTRACT.md"
SHAPE_IN, SHAPE_OUT = 9500, 380     # OpenRelay's agentic mix (docs/PLAN.md section 1)
SHAPE_TOL = 0.12                    # a cell counts as "agentic shape" within +-12% on both lengths
SLO_TTFT_P95_S = 4.0                # OpenRelay's SLO (docs/PLAN.md section 1)
MAX_ERR_FRAC = 0.01                 # stop rule: errors over 1% (docs/PLAN.md section 4)
BENCH_MODEL = "gemma-4-31b"         # the benchmark's model slug in price bar names (openrelay/gemma-4-31b)
GSM8K_GATE = 0.965                  # quality gate (docs/PLAN.md section 1, fairness rule 1)
GATE_MIN_N = 30                     # "30 items to screen and 200 on the winner" (docs/PLAN.md section 1, rule 1)
MILESTONES = [50, 100, 200, 300, 400]
REPLAN_USD = 500

# fixed identity order (never cycled): GPU -> categorical slot. Configs on one GPU share its hue and differ by symbol.
GPU_ORDER = ["H100", "RTX-PRO-6000", "MI355X", "L40S", "RTX-5090", "H200", "B300", "2xH100"]
SYMBOLS = ["circle", "rect", "triangle", "diamond", "roundRect", "pin", "arrow"]
DASHES = ["solid", "dashed", "dotted", "solid", "dashed", "dotted", "solid"]

PHASE_CATS = [  # (category, phases) in display order; category colours are fixed slots on the Runs page
    ("provision", ["snapshot_build", "create", "failed_create"]),
    ("image pull", ["image_pull"]),
    ("weights", ["weight_download"]),
    ("engine start", ["engine_start", "torch_compile", "graph_capture"]),
    ("warmup", ["warmup", "smoke"]),
    ("measure", ["steady", "step"]),
    ("eval", ["eval"]),
    ("teardown", ["teardown", "reap_confirm", "eviction", "idle"]),
]
PHASE_TO_CAT = {p: c for c, ps in PHASE_CATS for p in ps}


class Inputs:
    """Every file read, fingerprinted. The Sources page lists them; nothing is read without being registered."""

    def __init__(self, root):
        self.root = root
        self.files = {}
        self.warnings = []

    def rel(self, p):
        p = pathlib.Path(p).resolve()
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def reg(self, p):
        p = pathlib.Path(p)
        r = self.rel(p)
        if r not in self.files and p.exists():
            b = p.read_bytes()
            st = p.stat()
            self.files[r] = dict(path=r, bytes=len(b), sha256=hashlib.sha256(b).hexdigest(),
                                 mtime=dtm.datetime.fromtimestamp(st.st_mtime, dtm.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        return r

    def json(self, p):
        self.reg(p)
        return json.loads(pathlib.Path(p).read_text())

    def jsonl(self, p):
        self.reg(p)
        out = []
        for i, line in enumerate(pathlib.Path(p).read_text().splitlines()):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    self.warn(f"{self.rel(p)} line {i + 1}: bad JSON ({e})")
        return out

    def text(self, p):
        self.reg(p)
        return pathlib.Path(p).read_text(errors="replace")

    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)


# ----------------------------------------------------------------------------- helpers
def iso_to_epoch(s):
    if not s:
        return None
    s = s.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dtm.datetime.strptime(s, fmt).replace(tzinfo=dtm.timezone.utc).timestamp()
        except ValueError:
            pass
    return None


def epoch_to_iso(t):
    return None if t is None else dtm.datetime.fromtimestamp(t, dtm.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def latest(pattern):
    xs = sorted(glob.glob(str(ROOT / pattern)))
    return pathlib.Path(xs[-1]) if xs else None


def gpu_key(gpu, gpus):
    return gpu if (gpus or 1) == 1 else f"{gpus}x{gpu}"


def pick(d, *ks):
    for k in ks:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def dist5(d):
    """keep the percentile block of a traffic.py dist()"""
    if not d:
        return None
    return {k: d.get(k) for k in ("n", "mean", "min", "max", "p50", "p90", "p95", "p99")}


def parse_launch(cmd):
    """Declared engine flags from a config's launch line (what we ASKED for; startup_facts says what we GOT)."""
    try:
        toks = shlex.split(cmd)
    except ValueError:
        toks = cmd.split()
    flags = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            k = t[2:]
            if "=" in k:
                k, v = k.split("=", 1)
                flags[k] = v
            elif i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                flags[k] = toks[i + 1]
                i += 1
            else:
                flags[k] = True
        elif t.startswith("-") and len(t) == 2 and i + 1 < len(toks):
            flags[t[1:]] = toks[i + 1]
            i += 1
        i += 1
    engine = "vllm" if "vllm" in cmd else "sglang" if "sglang" in cmd else "llama.cpp" if "llama-server" in cmd else "other"
    spec = flags.get("speculative-config") or flags.get("speculative_config")
    if spec and isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError:
            pass
    return dict(engine=engine, flags=flags, max_model_len=flags.get("max-model-len") or flags.get("context-length") or flags.get("c"),
                kv_cache_dtype=flags.get("kv-cache-dtype"), quantization=flags.get("quantization"),
                max_num_seqs=flags.get("max-num-seqs") or flags.get("max-running-requests"),
                tp=flags.get("tensor-parallel-size") or flags.get("tp"), gpu_mem_util=flags.get("gpu-memory-utilization") or flags.get("mem-fraction-static"),
                spec=spec, cuda_graph_sizes=flags.get("cuda-graph-sizes"))


# ----------------------------------------------------------------------------- loaders
MODEL_OPENRELAY = "openrelay/gemma-4-31b"
OPENROUTER_EXCLUDE = ("Modal",)     # as in analysis/cost.py (Modal's endpoint is not a public per-token price)


def tier_usd(t, i, o, cached=0.0):
    """$ per request at a price tier (contract section 8 fields): uncached in + cached in + out."""
    return ((i - cached) * t["in_usd_per_m"] + cached * (t.get("cached_in_usd_per_m") or t["in_usd_per_m"]) + o * t["out_usd_per_m"]) / 1e6


def latest_json(I, pattern):
    """Newest file matching pattern that parses; a snapshot being written right now (empty / partial) falls back to the
    previous one with a warning instead of failing the build."""
    for f in sorted(glob.glob(str(ROOT / pattern)), reverse=True):
        try:
            return pathlib.Path(f), I.json(f)
        except (json.JSONDecodeError, OSError) as e:
            I.warn(f"{I.rel(f)}: unreadable ({e}); using the previous snapshot")
    raise SystemExit(f"no readable {pattern}")


def load_bars(I):
    orl_p, orl = latest_json(I, "research/openrelay_prices_*.json")
    orb_p, orb = latest_json(I, "research/openrouter_bar_google_gemma-4-31b-it_*.json")
    tiers = orl["models"][MODEL_OPENRELAY]["tiers"]
    sync, batch = tiers["Sync"], tiers.get("Batch")
    live = [e for e in orb.get("endpoints", []) if (e.get("status") or 0) >= 0 and e.get("provider") not in OPENROUTER_EXCLUDE]
    best = min(live, key=lambda e: SHAPE_IN * e["in_usd_per_m"] + SHAPE_OUT * e["out_usd_per_m"]) if live else None
    orc = ((SHAPE_IN * best["in_usd_per_m"] + SHAPE_OUT * best["out_usd_per_m"]) / 1e6, best) if best else None
    per_req = dict(sync=tier_usd(sync, SHAPE_IN, SHAPE_OUT), sync_all_cached=tier_usd(sync, SHAPE_IN, SHAPE_OUT, SHAPE_IN),
                   batch=tier_usd(batch, SHAPE_IN, SHAPE_OUT) if batch else None, openrouter=orc[0] if orc else None)
    return dict(
        shape=dict(input_tokens=SHAPE_IN, output_tokens=SHAPE_OUT, tolerance=SHAPE_TOL),
        openrelay=dict(model=MODEL_OPENRELAY, sync=sync, batch=batch, source=orl.get("source"), saved_file=orl.get("saved_file"),
                       fetched_utc=orl.get("fetched_utc"), file=I.rel(orl_p)),
        openrouter=dict(provider=orc[1]["provider"] if orc else None, quant=orc[1]["quant"] if orc else None,
                        in_usd_per_m=orc[1]["in_usd_per_m"] if orc else None, out_usd_per_m=orc[1]["out_usd_per_m"] if orc else None,
                        snapshot=orb.get("snapshot"), file=I.rel(orb_p), excluded=["Modal"]),
        per_request=per_req,
    )


def load_protocol_rows(I, results):
    """results/protocol_rows.json (contract section 7b): one row per (rung, seed), priced; prices vllm-bench rungs."""
    p = results / "protocol_rows.json"
    if not p.exists():
        return [], None
    return I.json(p), I.rel(p)


def load_metered(I, results):
    """results/metered_usd.json (contract v1.4, infra/spend.py): Daytona-metered $ per sandbox. Optional."""
    p = results / "metered_usd.json"
    if not p.exists():
        return None
    x = I.json(p)
    x["file"] = I.rel(p)
    return x


def load_claims(I, results, runs, configs):
    """results/claims.json (analysis/screen_report.py): per (claim group, column) the highest rate at which every seed meets
    the SLO, priced on the worst seed's achieved req/s. Column A = OpenRelay's random-token client, B = our real-text client."""
    p = results / "claims.json"
    if not p.exists():
        I.warn(f"{I.rel(p)} not found: the headline shows 'awaiting claims'")
        return None
    c = I.json(p)
    rmap = {r["run_id"]: r for r in runs}
    for row in c.get("claims", []):
        rs = [rmap[x] for x in row.get("runs") or [] if x in rmap]
        # contract v1.6: origin ("openrelay_recipe" | "ours" | "mixed:a+b"), gpu and engine come with the row; no guessing
        row["gpu_key"] = row.get("gpu") or next((r.get("gpu_key") for r in rs if r.get("gpu_key")), None)
        row["config_ids"] = sorted({cid for r in rs for cid in (r.get("config_ids") or [])})
        if not row.get("origin"):
            I.warn(f"claims.json {row.get('group')}/{row.get('column')}: no origin field (contract v1.6)")
        # contract v1.7: each row is priced against its own bar (OpenRelay for Gemma 4 31B, OpenRouter PrimeIntellect for phase 2)
        row["bar_name"] = row.get("price_bar_name") or (f"OpenRelay sync" if row.get("ratio_vs_openrelay_sync") is not None else None)
        # the bar's short name keeps its model unless it is this benchmark's (Gemma 4 31B): "OpenRelay sync" for Gemma,
        # "OpenRelay sync glm-5.3-flash" for GLM (two OpenRelay bars must never share a name, or their rows would share a chart)
        gemma = row["price_bar_is_openrelay_gemma"] if "price_bar_is_openrelay_gemma" in row else None   # the build session's flag (cf9e936)
        row["bar_short"] = bar_short_of(row["bar_name"], gemma)
        row["bar_usd"] = row.get("price_bar_usd_per_req") or (c.get("openrelay_sync_usd_per_req") if row.get("ratio_vs_openrelay_sync") is not None else None)
        row["ratio"] = row.get("ratio_vs_bar") if row.get("ratio_vs_bar") is not None else row.get("ratio_vs_openrelay_sync")
        row["ratio_metered"] = row.get("ratio_vs_bar_metered") if row.get("ratio_vs_bar_metered") is not None else row.get("ratio_vs_openrelay_sync_metered")
        missing = [x for x in row.get("runs") or [] if x not in rmap]
        if missing:
            I.warn(f"claims.json {row.get('group')}/{row.get('column')}: runs not in results/: {', '.join(missing)}")
    c["file"] = I.rel(p)
    md = results / "CLAIMS.md"
    if md.exists():
        I.reg(md)
        c["md_file"] = I.rel(md)
    return c


def claim_hosts(I, results, claims, prow):
    """Host and launch facts per claim group, from each run's own config.json: the snapshot (image digest, vCPU, GiB) and the
    server's launch line (--max-model-len, --max-num-seqs). The server per claim group comes from protocol_rows (contract v1.5)."""
    if not claims:
        return
    srv = {}
    for r in prow:
        if r.get("claim_group") and r.get("server"):
            srv.setdefault(r["claim_group"], r["server"])
    for row in claims.get("claims", []):
        hosts = []
        for run in row.get("runs") or []:
            p = results / run / "config.json"
            if not p.exists():
                continue
            cfg = I.json(p)
            sn = cfg.get("snapshot") or {}
            s = next((x for x in cfg.get("servers", []) if x.get("name") == srv.get(row.get("group"))), None)
            launch = (s or {}).get("launch") or ""
            mm, ns = re.search(r"--max-model-len[ =](\d+)", launch), re.search(r"--max-num-seqs[ =](\d+)", launch)
            hosts.append(dict(cpu=sn.get("cpu"), mem=sn.get("mem"), disk=sn.get("disk"), snapshot=sn.get("name"), image=sn.get("image"),
                              image_digest=sn.get("image_digest_amd64"), server=srv.get(row.get("group")), max_model_len=int(mm.group(1)) if mm else None,
                              max_num_seqs=int(ns.group(1)) if ns else None, file=I.rel(p)))
        if hosts:
            key = lambda h: (h["cpu"], h["mem"], h["image_digest"], h["max_model_len"])  # noqa: E731
            row["host"] = dict(hosts[0], uniform=len({key(h) for h in hosts}) == 1, files=[h["file"] for h in hosts])


def same_box_pairs(claims, prow):
    """H1 of the write-up (docs/WRITEUP.md, 97ad059): our recipe vs theirs on the same boxes. A pair is an 'ours' and an
    'openrelay_recipe' claim group in column A (their client) whose runs are the same set; each run is one box serving both
    recipes back to back. Per box: both recipes' protocol_rows at ours' highest passing rate (one rate, both recipes)."""
    rows = (claims or {}).get("claims", [])
    get = {(r.get("group"), r.get("column")): r for r in rows}
    out = []
    for o in [r for r in rows if r.get("column") == "A" and r.get("origin") == "ours" and r.get("claim_grade") and r.get("max_rate_all_seeds_pass") is not None]:
        for t in [r for r in rows if r.get("column") == "A" and r.get("origin") == "openrelay_recipe" and r.get("runs") and set(r["runs"]) == set(o.get("runs") or [])]:
            rate = o["max_rate_all_seeds_pass"]
            boxes = []
            for run in o["runs"]:
                pick = lambda g: next((x for x in prow if x.get("claim_group") == g and x.get("column") == "A" and x.get("run_id") == run  # noqa: E731
                                       and x.get("offered_rps") is not None and abs(x["offered_rps"] - rate) < 1e-6), None)
                mo, mt = pick(o["group"]), pick(t["group"])
                if mo and mt:
                    view = lambda x: dict(ttft95=x.get("ttft_p95_s"), censored=bool(x.get("ttft_censored")), lower=x.get("ttft_p95_lower_s"), pass_=bool(x.get("slo_pass")),  # noqa: E731
                                          achieved=x.get("achieved_rps"), seed=x.get("seed"), file=x.get("summary_file") or x.get("file"))
                    boxes.append(dict(run_id=run, ours=view(mo), theirs=view(mt)))
            oB, tB = get.get((o["group"], "B")), get.get((t["group"], "B"))
            ho, ht = o.get("host") or {}, t.get("host") or {}
            gain = o["worst_seed_achieved_rps"] / t["worst_seed_achieved_rps"] - 1 if o.get("worst_seed_achieved_rps") and t.get("worst_seed_achieved_rps") else None
            out.append(dict(ours=o["group"], theirs=t["group"], gpu=o.get("gpu_key"), rate=rate, ours_rate=rate, theirs_rate=t.get("max_rate_all_seeds_pass"),
                            ours_achieved=o.get("worst_seed_achieved_rps"), theirs_achieved=t.get("worst_seed_achieved_rps"), gain=gain, boxes=boxes,
                            b=dict(ours_rate=(oB or {}).get("max_rate_all_seeds_pass"), ours_claim_grade=bool((oB or {}).get("claim_grade")),
                                   theirs_rate=(tB or {}).get("max_rate_all_seeds_pass"), theirs_because=(tB or {}).get("not_claim_grade_because") or []),
                            same_image=bool(ho.get("image_digest")) and ho.get("image_digest") == ht.get("image_digest"), image=ho.get("image"),
                            mml=[ho.get("max_model_len"), ht.get("max_model_len")], host=[ho.get("cpu"), ho.get("mem")],
                            runs=o["runs"], seeds=o.get("seeds") or [], files=sorted({b[k]["file"] for b in boxes for k in ("ours", "theirs") if b[k]["file"]})))
    # the headline pair: the same ranking as the dashboard's headline (claim-grade, more seeds, cheaper)
    rank = lambda p: (-len(get[(p["ours"], "A")].get("seeds") or []), get[(p["ours"], "A")].get("usd_per_req") or 9)  # noqa: E731
    return sorted(out, key=rank)


def host_levers(claims, pairs):
    """H3: the same recipe (same server, image digest and GPU as the H1 'ours' group) on a different Daytona host size. A
    lever of the rental, never a comparison with their recipe (their recipe would get the same saving)."""
    if not claims or not pairs:
        return []
    rows = claims.get("claims", [])
    get = {(r.get("group"), r.get("column")): r for r in rows}
    base = get.get((pairs[0]["ours"], "A")) or {}
    bh = base.get("host") or {}
    out = []
    for r in rows:
        h = r.get("host") or {}
        if (r.get("column") != "A" or r.get("origin") != "ours" or r.get("group") == base.get("group") or not h or h.get("server") != bh.get("server")
                or h.get("image_digest") != bh.get("image_digest") or r.get("gpu_key") != base.get("gpu_key") or (h.get("cpu"), h.get("mem")) == (bh.get("cpu"), bh.get("mem"))):
            continue
        cols = {c: dict(rate=x.get("max_rate_all_seeds_pass"), usd=x.get("usd_per_req"), ratio=x.get("ratio"), ratio_metered=x.get("ratio_metered"), claim_grade=bool(x.get("claim_grade")),
                        because=x.get("not_claim_grade_because") or [], bar_short=x.get("bar_short"))
                for c in ("A", "B") for x in [get.get((r["group"], c))] if x}
        out.append(dict(group=r["group"], host=[h.get("cpu"), h.get("mem")], base_group=base.get("group"), base_host=[bh.get("cpu"), bh.get("mem")],
                        usd_per_h=r.get("usd_per_h"), base_usd_per_h=base.get("usd_per_h"), cols=cols, base_cols={c: dict(ratio=(get.get((base["group"], c)) or {}).get("ratio")) for c in ("A", "B")},
                        runs=r.get("runs") or [], files=h.get("files") or []))
    return out


def load_scope(I, claims, pairs):
    """H4: what the comparison covers. OpenRelay's published recipes (the GitHub contents listing saved by the build session)
    and the context length their API advertises (their pricing page snapshot) vs what the recipes served."""
    out = {}
    fs = sorted(glob.glob(str(ROOT / "research/raw/openrelay_inference_recipes_contents_*.json")))
    if fs:
        items = I.json(fs[-1])
        out.update(recipes=[x.get("name") for x in items if isinstance(x, dict) and x.get("type") == "dir"], recipes_file=I.rel(fs[-1]))
    fs = sorted(glob.glob(str(ROOT / "research/raw/openrelay/inference-pricing_*.html")))
    if fs:
        t = I.text(fs[-1])
        out.update(api_context="256K" if "256K" in t else None, pricing_file=I.rel(fs[-1]))
    mml = [m for p in pairs[:1] for m in p.get("mml") or [] if m]
    out["served_context"] = max(mml) if mml else None
    long_rows = [r for r in (claims or {}).get("claims", []) if ((r.get("host") or {}).get("max_model_len") or 0) >= 262144 and r.get("claim_grade")]
    out["long_context"] = "done" if long_rows else "pending"
    out["long_context_groups"] = sorted({r["group"] for r in long_rows})
    return out


def load_quality(I, results):
    """results/quality_rows.json (schema quality/1, analysis/quality.py): every served config vs the BF16 original on GSM8K
    test[:250] and MMLU-Pro 200 (Wilson 95% intervals, paired deltas with their CI and McNemar p), greedy agreement vs the
    reference with the reference's own repeat agreement as the noise floor, and MTP on/off pairs."""
    p = results / "quality_rows.json"
    if not p.exists():
        return None
    q = I.json(p)
    if q.get("schema") != "quality/1":
        I.warn(f"{I.rel(p)}: schema {q.get('schema')!r}, expected quality/1")
    q["file"] = I.rel(p)
    md = results / "QUALITY_TABLE.md"
    if md.exists():
        q["md_file"] = I.rel(md)
    return q


def wilson(c, n, z=1.959963984540054):
    """Wilson 95% interval for c successes of n (the same interval quality_rows.json uses)."""
    if not n:
        return None
    ph = c / n
    d = 1 + z * z / n
    m = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [m - h, m + h]


def ladder_points(prow, protocols, configs):
    """Chart points for the ladder. From results/protocol_rows.json (contract v1.5): per (claim_group, column, offered rate) one
    point that passes only if every seed passes, at the worst seed's achieved req/s, worst p95 TTFT and worst TPOT: the same
    merge results/claims.json applies (the claim itself is read from claims.json, never from these points). Ladders not yet
    in protocol_rows fall back to their protocol_summary.json rungs (kept seed), grouped by config."""
    pts, covered = [], set()
    groups = {}
    for r in prow:
        if r.get("offered_rps") is None:
            continue
        covered.add((r.get("run_id"), r.get("protocol")))
        groups.setdefault((r.get("claim_group") or r.get("config_id"), r.get("column") or ("A" if "vllm" in str(r.get("client")) else "B"), round(r["offered_rps"], 4)), []).append(r)
    for (g, col, rate), rs in groups.items():
        worst = min(rs, key=lambda r: r.get("achieved_rps") if r.get("achieved_rps") is not None else 1e9)
        mx = lambda k: max((r.get(k) for r in rs if r.get(k) is not None), default=None)  # noqa: E731
        cid = worst.get("config_id")
        pts.append(dict(key=f"{g}|{col}", group=g, column=col, config_id=cid, gpu_key=(configs.get(cid) or {}).get("gpu_key") or worst.get("gpu"),
                        run_ids=sorted({r.get("run_id") for r in rs}), files=sorted({r.get("summary_file") or r.get("file") for r in rs if r.get("summary_file") or r.get("file")}),
                        rate=rate, multiple=worst.get("multiple"), achieved=worst.get("achieved_rps"), goodput=worst.get("goodput_rps") if all(r.get("slo_pass") for r in rs) else 0.0,
                        ttft95=mx("ttft_p95_s"), tpot50=mx("tpot_p50_s"), tpot95=mx("tpot_p95_s"), pass_=all(bool(r.get("slo_pass")) for r in rs),
                        censored=any(bool(r.get("ttft_censored")) for r in rs), scored_window_s=min((r.get("scored_window_s") for r in rs if r.get("scored_window_s")), default=None),
                        seeds=[f"{r.get('run_id')}:s{r.get('seed')}" for r in rs], client=worst.get("client"),
                        usd_achieved=worst.get("usd_per_req_at_achieved"), ratio_metered=worst.get("ratio_sync_at_offered_metered"), source="protocol_rows"))
    for p in protocols:
        if (p.get("run_id"), p.get("protocol_dir")) in covered:
            continue
        col = "A" if p.get("client") == "vllmbench" else "B"
        cid = p.get("config_id")
        for l in p.get("ladder") or []:
            if l.get("skipped") or l.get("rate") is None:
                continue
            pts.append(dict(key=f"{cid}|{col}", group=(configs.get(cid) or {}).get("label") or cid, column=col, config_id=cid, gpu_key=(configs.get(cid) or {}).get("gpu_key"),
                            run_ids=[p.get("run_id")], files=[p.get("file")], rate=l["rate"], multiple=l.get("multiple"), achieved=l.get("achieved_rps"), goodput=l.get("goodput_rps"),
                            ttft95=l.get("ttft_p95_s"), tpot50=l.get("tpot_p50_s"), tpot95=l.get("tpot_p95_s"), pass_=bool(l.get("slo_pass")),
                            seeds=[f"{p.get('run_id')}:s{l.get('kept_seed')}"], client=p.get("client"), usd_achieved=None, ratio_metered=None, source="protocol_summary (not yet in protocol_rows)"))
    pts.sort(key=lambda x: (x["key"], x["rate"]))
    return pts


def server_origins(I, rd, cfg):
    """origin per server ("openrelay_recipe" | "ours"), declared in the config. The run's own config.json copy may predate the
    field, so fall back to the recipe session's config (recipes/*/configs/<label>.json), then configs/<label>.json."""
    out = {s["name"]: s.get("origin") for s in cfg.get("servers", []) if s.get("origin")}
    if len(out) < len(cfg.get("servers", [])):
        for cand in sorted(ROOT.glob(f"recipes/*/configs/{cfg.get('label')}.json")) + [ROOT / "configs" / f"{cfg.get('label')}.json"]:
            if cand.exists():
                c2 = I.json(cand)
                for s in c2.get("servers", []):
                    if s.get("origin") and s["name"] not in out:
                        out[s["name"]] = s["origin"]
    return out


def rescore_paired(paired, prow):
    """Contract v1.8: protocol_rows (traffic cells re-scored over a 300 s window) are authoritative over the run's own
    protocol_summary.json (240 s judge). Each paired rung takes its row's p95 TTFT, pass flag, achieved rate and TPOT when the
    row exists (same run, server, column, offered rate; the kept seed first); the max rate at the SLO follows the new flags."""
    idx = {}
    for r in prow:
        if r.get("offered_rps") is not None and r.get("server"):
            idx.setdefault((r.get("run_id"), r["server"], r.get("column"), round(r["offered_rps"], 4)), []).append(r)
    for bx in paired:
        for rec in bx["recipes"].values():
            for col, rungs in rec["ladders"].items():
                hit = False
                for g in rungs:
                    rs = idx.get((bx["run_id"], rec["server"], col, round(g["rate"], 4))) if g.get("rate") is not None else None
                    if not rs:
                        continue
                    x = next((r for r in rs if r.get("kept")), rs[0])
                    g.update(ttft95=x.get("ttft_p95_s"), pass_=bool(x.get("slo_pass")), achieved=x.get("achieved_rps"), tpot50=x.get("tpot_p50_s"),
                             censored=bool(x.get("ttft_censored")), scored_window_s=x.get("scored_window_s"), source="protocol_rows")
                    hit = True
                if hit:
                    passing = [g["rate"] for g in rungs if g.get("pass_") and not g.get("skipped")]
                    rec["max_rate"][col] = max(passing) if passing else 0.0


def rung_view(l):
    return dict(multiple=l.get("multiple"), rate=l.get("rate"), skipped=l.get("skipped"), pass_=bool(l.get("slo_pass")), ttft95=l.get("ttft_p95_s"),
                tpot50=l.get("tpot_p50_s"), accept=l.get("spec_accept_len"), achieved=l.get("achieved_rps"), seed=l.get("kept_seed"))


def load_paired(I, results, runs, protocols, evals, startup):
    """Paired confirm boxes: one run (one sandbox, one GPU) serving OpenRelay's recipe and ours back to back, so box-to-box
    clock spread cancels. Per recipe: ladder per column (protocol_<server> = A, protocol_<server>_text = B), max rate at
    the SLO, prefill tok/s (the counted run), GSM8K gate, startup facts. All fields from contract sections 4, 5, 7, 7b."""
    boxes = []
    for r in runs:
        rd = results / r["run_id"]
        cj = rd / "config.json"
        if not cj.exists():
            continue
        cfg = json.loads(cj.read_text())
        org = server_origins(I, rd, cfg)
        theirs = [n for n, o in org.items() if o == "openrelay_recipe"]
        ours = [n for n, o in org.items() if o == "ours"]
        if not (theirs and ours):
            continue
        box = dict(run_id=r["run_id"], label=r["label"], status=r["status"], excluded=r.get("excluded"), gpu_key=r.get("gpu_key"), seed=None, recipes={}, files=[I.rel(cj)])
        for side, srv in (("theirs", theirs[0]), ("ours", ours[0])):
            cid = f"{cfg['label']}:{srv}"
            rec = dict(server=srv, config_id=cid, ladders={}, max_rate={}, prefill_tok_s=None, prefill_runs=None, gate=None, startup=None, files=[])
            for p in protocols:
                if p.get("run_id") != r["run_id"]:
                    continue
                pdir = p.get("protocol_dir") or ""
                if pdir == f"protocol_{srv}":
                    col = "A" if p.get("client") == "vllmbench" else "B"
                elif pdir == f"protocol_{srv}_text":
                    col = "B"
                else:
                    continue
                rec["ladders"][col] = [rung_view(l) for l in p.get("ladder") or [] if l.get("rate") is not None]
                rec["max_rate"][col] = p.get("max_rate_at_slo_rps")
                rec["files"].append(p.get("file"))
                pf = p.get("prefill") or {}
                if pf.get("prefill_tok_s") is not None and rec["prefill_tok_s"] is None:
                    rec["prefill_tok_s"] = pf["prefill_tok_s"]
                    rec["prefill_runs"] = [x.get("input_tok_s") for x in pf.get("runs") or []]
                if box["seed"] is None and p.get("seeds"):
                    box["seed"] = p["seeds"][0]
            gs = [e for e in evals if e["run_id"] == r["run_id"] and (e.get("label") or "").startswith(f"gate_{srv}") and (e.get("task") or "").startswith("gsm8k")]
            if gs:
                g = max(gs, key=lambda e: e.get("n") or 0)
                rec["gate"] = dict(acc=g.get("acc"), n=g.get("n"), split=g.get("split"), passed=(g.get("gate_file") or {}).get("pass"), file=g.get("file"))
                rec["files"].append(g.get("file"))
            sf = next((f for f in startup if f.get("run_id") == r["run_id"] and f.get("config_id") == cid), None)
            if sf:
                rec["startup"] = dict(spec=(sf.get("spec_decode") or {}).get("num_speculative_tokens"), graphs=len(sf.get("cuda_graph_sizes") or []),
                                      max_graph=max(sf.get("cuda_graph_sizes") or [0]), kv_tokens=sf.get("kv_cache_tokens"), rejected=sf.get("rejected"), file=sf.get("file"))
                rec["files"].append(sf.get("file"))
            box["recipes"][side] = rec
            box["files"] += [f for f in rec["files"] if f]
        boxes.append(box)
    boxes.sort(key=lambda b: (b.get("seed") or 0, b["run_id"]))
    return boxes


def load_cost_rows(I, results):
    """results/cost_rows.json (contract section 6) indexed by cell file path."""
    p = results / "cost_rows.json"
    if not p.exists():
        I.warn(f"{I.rel(p)} not found (contract section 6; written by analysis/report.py): cells show no $ until it exists")
        return {}, None
    rows = I.json(p)
    return {r.get("file"): r for r in rows if r.get("file")}, I.rel(p)


def load_configs(I, dt):
    configs, files = {}, sorted((ROOT / "configs").glob("*.json"))
    per_gpu_count = {}
    for f in files:
        try:
            cfg = I.json(f)
        except Exception as e:  # noqa: BLE001
            I.warn(f"{I.rel(f)}: unreadable ({e})")
            continue
        snap = cfg.get("snapshot", {})
        g = gpu_key(snap.get("gpu"), snap.get("gpus", 1))
        try:
            pph = dt.hourly_usd(snap["gpu"], snap.get("gpus", 1), snap["cpu"], snap["mem"], snap["disk"])
        except Exception:  # noqa: BLE001
            pph = None
        for srv in cfg.get("servers", []):
            cid = f"{cfg['label']}:{srv['name']}"
            n = per_gpu_count.get(g, 0)
            per_gpu_count[g] = n + 1
            w = (srv.get("weights") or [{}])[0]
            configs[cid] = dict(
                config_id=cid, label=cfg["label"], server=srv["name"], config_file=I.rel(f), note=cfg.get("note"),
                gpu=snap.get("gpu"), gpus=snap.get("gpus", 1), gpu_key=g, cpu=snap.get("cpu"), mem=snap.get("mem"), disk=snap.get("disk"),
                image=snap.get("image"), image_digest=snap.get("image_digest_amd64") or snap.get("image_digest"),
                weights_repo=w.get("repo"), weights_rev=w.get("rev"), launch=srv.get("launch"), declared=parse_launch(srv.get("launch") or ""),
                usd_per_h=pph, symbol=SYMBOLS[n % len(SYMBOLS)], dash=DASHES[n % len(DASHES)], symbol_index=n,
                steps=[s.get("preset") or s.get("name") for s in srv.get("steps", [])])
    return configs


def run_dirs(results):
    for rj in sorted(results.glob("*/*/run.json")):
        yield rj.parent
    # runs that crashed before run.json: config.json alone
    for cj in sorted(results.glob("*/*/config.json")):
        if not (cj.parent / "run.json").exists():
            yield cj.parent


def matrix_dirs(rd):
    seen = set()
    for pat in ("m_*", "out/m_*"):
        for m in sorted(rd.glob(pat)):
            if m.is_dir() and m.name not in seen:
                seen.add(m.name)
                yield m


def ensure_config(configs, cfg, rd, I, dt):
    """A run whose config file has since changed/vanished still gets a config entry from its own results/config.json."""
    snap = cfg.get("snapshot", {})
    g = gpu_key(snap.get("gpu"), snap.get("gpus", 1))
    out = []
    for srv in cfg.get("servers", []):
        cid = f"{cfg['label']}:{srv['name']}"
        if cid not in configs:
            n = sum(1 for c in configs.values() if c["gpu_key"] == g)
            try:
                pph = dt.hourly_usd(snap["gpu"], snap.get("gpus", 1), snap["cpu"], snap["mem"], snap["disk"])
            except Exception:  # noqa: BLE001
                pph = None
            w = (srv.get("weights") or [{}])[0]
            configs[cid] = dict(config_id=cid, label=cfg["label"], server=srv["name"], config_file=I.rel(rd / "config.json"),
                                note=cfg.get("note"), gpu=snap.get("gpu"), gpus=snap.get("gpus", 1), gpu_key=g, cpu=snap.get("cpu"),
                                mem=snap.get("mem"), disk=snap.get("disk"), image=snap.get("image"),
                                image_digest=snap.get("image_digest_amd64") or snap.get("image_digest"), weights_repo=w.get("repo"),
                                weights_rev=w.get("rev"), launch=srv.get("launch"), declared=parse_launch(srv.get("launch") or ""),
                                usd_per_h=pph, symbol=SYMBOLS[n % len(SYMBOLS)], dash=DASHES[n % len(DASHES)], symbol_index=n,
                                steps=[s.get("preset") or s.get("name") for s in srv.get("steps", [])])
        out.append(cid)
    return out


def cell_row(I, cf, run_id, cid, cfg_entry, mdir_name, costs, protocol=None):
    d = I.json(cf)
    s, p = d.get("summary") or {}, d.get("params") or {}
    args = d.get("args") or {}
    workload = args.get("workload")
    reqs = [r for r in d.get("requests", []) if r.get("prompt_tokens") and r.get("status") == 200]
    mean_in = sum(r["prompt_tokens"] for r in reqs) / len(reqs) if reqs else (pick(s, "prompt_tokens", "mean") or 0)
    mean_cached = sum(r.get("cached_tokens") or 0 for r in reqs) / len(reqs) if reqs else 0
    fixed_out = workload in ("synthetic", "longdoc_qa") and p.get("ignore_eos", True)
    otps = s.get("output_tok_per_s") or 0
    if fixed_out:
        o = p.get("output_tokens", 256)
        rps = otps / o if o else 0
        basis = "steady out tok/s / output_tokens (fixed length)"
    else:
        o = pick(s, "completion_tokens", "mean") or 0
        rps = s.get("req_per_s") or 0
        basis = "requests completed in steady window / window"
    cr = costs.get(I.rel(cf)) or {}          # results/cost_rows.json row for this exact file, or nothing
    tot = cr.get("usd_h")
    our = cr.get("our_usd_req")
    b_sync, b_cached, b_batch, b_or = cr.get("openrelay_sync"), cr.get("openrelay_sync_cached"), cr.get("openrelay_batch"), cr.get("openrouter")
    sm = s.get("server_metrics") or {}
    sent = s.get("requests_sent") or 0
    errors = s.get("errors") or 0
    err_frac = (errors / sent) if sent else (1.0 if errors else 0.0)
    ttft95 = pick(s, "ttft_s", "p95")
    agentic = bool(mean_in and o and abs(mean_in - SHAPE_IN) / SHAPE_IN <= SHAPE_TOL and abs(o - SHAPE_OUT) / SHAPE_OUT <= SHAPE_TOL)
    lim = 6.0 if mean_in > 4000 else 2.0
    tpot50 = pick(s, "tpot_s", "p50")
    row = dict(
        run_id=run_id, config_id=cid, gpu_key=(cfg_entry or {}).get("gpu_key"), matrix=mdir_name, cell=d.get("label") or cf.stem,
        cell_id=f"{mdir_name}/{d.get('label') or cf.stem}", file=I.rel(cf), t_start=d.get("t_start"), workload=workload,
        mode=s.get("mode"), C=s.get("concurrency"), rate=s.get("offered_rate"), params={k: v for k, v in p.items() if not isinstance(v, (dict, list))},
        window_s=s.get("window_s"), requests_sent=sent, completed=s.get("completed"), completed_in_window=s.get("completed_in_window"),
        errors=errors, err_frac=err_frac, dropped=s.get("dropped"), peak_inflight=s.get("peak_inflight"),
        in_mean=mean_in, cached_mean=mean_cached, out_per_req=o, req_s=rps, req_s_basis=basis, req_s_completed=s.get("req_per_s"),
        out_tok_s=otps, in_tok_s=s.get("input_tok_per_s"), cached_in_tok_s=s.get("cached_input_tok_per_s"), cache_hit=s.get("cache_hit_ratio"),
        srv_out_tok_s=sm.get("gen_tok_per_s"), srv_prompt_tok_s=sm.get("prompt_tok_per_s"), srv_pc_hit=sm.get("prefix_cache_hit_ratio"),
        preemptions=sm.get("preemptions"), spec_accept_rate=sm.get("spec_acceptance_rate"), spec_accept_len=sm.get("spec_mean_accepted_per_step"),
        per_user_tok_s=s.get("per_user_tok_per_s_p50"),
        ttft=dist5(s.get("ttft_s")), tpot=dist5(s.get("tpot_s")), itl=dist5(s.get("itl_s")), e2e=dist5(s.get("e2e_s")),
        usd_h=tot, our_usd_req=our, bar_sync=b_sync, bar_sync_cached=b_cached, bar_batch=b_batch, bar_openrouter=b_or,
        openrouter_who=cr.get("openrouter_who"), ratio_sync=cr.get("ratio_sync"), ratio_sync_cached=cr.get("ratio_sync_cached"),
        # cost_rows since cf9e936: openrelay_sync / ratio_sync are vs the row's OWN bar (GLM, Inkling); name it wherever it is shown
        bar_name=cr.get("price_bar_name"), bar_is_gemma=cr.get("price_bar_is_openrelay_gemma"),
        ratio_batch=cr.get("ratio_batch"), ratio_openrouter=cr.get("ratio_openrouter"), usd_per_m_out_all=cr.get("usd_per_m_out_all"),
        cost_req_s=cr.get("req_s"), priced=bool(cr), schema=d.get("schema"),
        agentic_shape=agentic,
        # contract section 5 SLO: p95 TTFT < 4 s, errors < 1%, no client-side drops
        slo_openrelay=bool(ttft95 is not None and ttft95 < SLO_TTFT_P95_S and err_frac < MAX_ERR_FRAC and not (s.get("dropped") or 0) and (s.get("completed") or 0) > 0),
        slo_interactive=bool(ttft95 is not None and tpot50 is not None and ttft95 <= lim and tpot50 <= 0.050 and not errors),
        protocol=protocol)
    if s.get("server_metrics") and otps and sm.get("gen_tok_per_s"):
        row["client_vs_server_out"] = otps / sm["gen_tok_per_s"] - 1
    return row


def ledger_phases(ledger, run_id):
    return [r for r in ledger if r.get("event") == "phase" and r.get("run_id") == run_id]


def coarse_phases(rep, rd, I):
    """Fallback phase profile from run.json durations (sequential, in run.py's order) until ledger `phase` events exist."""
    t0 = iso_to_epoch(rep.get("t_start"))
    if t0 is None:
        return []
    usd_h = pick(rep, "usd_per_h", "total") or 0
    out, t = [], t0
    ph = rep.get("phases") or {}

    def add(phase, secs, label, billed=True, detail=None):
        nonlocal t
        if secs is None:
            return
        out.append(dict(phase=phase, cat=PHASE_TO_CAT.get(phase, "measure"), label=label, t_start=epoch_to_iso(t), t_end=epoch_to_iso(t + secs),
                        dur_s=secs, billed=billed, usd=(secs / 3600 * usd_h) if billed else 0.0, detail=detail or {},
                        source=f"{I.rel(rd / 'run.json')} (sequential durations)"))
        t += secs
    add("snapshot_build", ph.get("snapshot_build_s"), "snapshot build (no sandbox billed)", billed=False)
    add("create", ph.get("create_s"), "sandbox create")
    for name, srv in (rep.get("servers") or {}).items():
        dl = srv.get("download") or {}
        w = srv.get("weights") or []
        nbytes = sum(x.get("bytes") or 0 for x in w)
        if dl.get("secs") is not None:
            add("weight_download", dl["secs"], f"{name}: weights", detail=dict(bytes=nbytes, mb_s=(nbytes / 1e6 / dl["secs"]) if dl["secs"] else None,
                                                                              rc=dl.get("rc")))
        sv = srv.get("serve") or {}
        if sv.get("ready_s") is not None:
            add("engine_start", sv["ready_s"], f"{name}: engine start to /health", detail=dict(ready=sv.get("ready")))
        for sname, st in (srv.get("steps") or {}).items():
            if st.get("secs") is not None:
                add("eval" if "eval" in sname else "step", st["secs"], f"{name}: {sname}", detail=dict(rc=st.get("rc"), step=sname))
    return out


def load_gpumon_norm(I, p, max_points=600):
    """gpumon_norm.csv (contract section 9): t_utc,gpu_index,util_pct,mem_used_mib,power_w,sm_mhz,temp_c"""
    rows = []
    I.reg(p)
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            t = iso_to_epoch((r.get("t_utc") or "").strip())
            if t is None:
                continue
            fl = lambda k: (float(r[k]) if r.get(k) not in (None, "", "NA", "nan") else None)  # noqa: E731
            try:
                rows.append([t, fl("util_pct"), fl("mem_used_mib"), fl("power_w"), fl("sm_mhz"), fl("temp_c")])
            except ValueError:
                continue
    if not rows:
        return None
    step = max(1, len(rows) // max_points)
    return dict(columns=["t", "util_pct", "mem_mib", "power_w", "sm_mhz", "temp_c"], rows=rows[::step], note="gpumon_norm.csv (normalised, both vendors)")


def load_gpumon(I, p, max_points=600):
    rows = []
    try:
        I.reg(p)
        with open(p, newline="") as f:
            for r in csv.reader(f):
                if len(r) < 7:
                    continue
                num = lambda x: float(re.sub(r"[^0-9.\-]", "", x) or "nan")  # noqa: E731
                try:
                    ts = r[0].strip().replace("/", "-")
                    t = dtm.datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dtm.timezone.utc).timestamp()
                except ValueError:
                    continue
                rows.append([t, num(r[2]), num(r[3]), num(r[4]), num(r[5]), num(r[6])])
    except OSError:
        return None
    if not rows:
        return None
    step = max(1, len(rows) // max_points)
    rows = rows[::step]
    return dict(columns=["t", "util_pct", "mem_mib", "power_w", "sm_mhz", "temp_c"], rows=[[round(x, 2) if not math.isnan(x) else None for x in r] for r in rows],
                note="nvidia-smi timestamps are the sandbox clock (assumed UTC)")


def load_runs(I, results, configs, ledger, costs, dt):
    runs, cells, evals, startup, protocols, measured = [], [], [], [], [], []
    xchecks = []
    for rd in run_dirs(results):
        label, stamp = rd.parent.name, rd.name
        run_id = f"{label}/{stamp}"
        if not (rd / "run.json").exists() and not any(True for _ in rd.rglob("cells/*.json")):
            I.warn(f"{I.rel(rd)}: no run.json yet (starting, or never started): not shown until run.py writes it")
            continue
        rep = I.json(rd / "run.json") if (rd / "run.json").exists() else {}
        cfg = I.json(rd / "config.json") if (rd / "config.json").exists() else {}
        cids = ensure_config(configs, cfg, rd, I, dt) if cfg else []
        lp = ledger_phases(ledger, run_id)
        phases = []
        if lp:
            for r in lp:
                phases.append(dict(phase=r.get("phase"), cat=PHASE_TO_CAT.get(r.get("phase"), "measure"), label=r.get("phase"),
                                   t_start=r.get("t_start"), t_end=r.get("t_end"), dur_s=r.get("dur_s"), billed=True, usd=r.get("usd"),
                                   nested_in=r.get("nested_in"), detail=r.get("detail") or {}, cell_id=r.get("cell_id"), config_id=r.get("config_id"),
                                   ok=r.get("ok"), source="infra/ledger.jsonl (phase event)"))
            phases.sort(key=lambda p: (p.get("t_start") or "", p.get("nested_in") is not None))
            phase_source = "ledger"
        else:
            phases = coarse_phases(rep, rd, I)
            phase_source = "run.json (coarse)"
        sid = rep.get("sandbox_id")
        lc = next((r for r in ledger if r.get("event") == "create" and r.get("id") == sid), None) if sid else None
        ld = next((r for r in ledger if r.get("event") == "delete" and r.get("id") == sid), None) if sid else None
        ev = next((r for r in ledger if r.get("event") == "evicted" and r.get("run_id") == run_id), None)
        status = "error" if rep.get("error") else "evicted" if (rep.get("evicted") or ev or rep.get("excluded") == "evicted") else "done" if (rep.get("deleted") or ld) else "running"
        if status == "running" and (iso_to_epoch(rep.get("t_saved")) or 0) < time.time() - 20 * 60:
            status = "stale"   # no delete recorded and run.json not rewritten for 20 min: the driver died; the reaper owns the sandbox
        files = sorted(I.rel(x) for x in rd.rglob("*") if x.is_file() and x.suffix in (".json", ".log", ".csv", ".prom", ".md"))
        driver = I.text(rd / "driver.log").splitlines()[-60:] if (rd / "driver.log").exists() else []
        n_cells = 0
        for cid in cids or [None]:
            srv = cid.split(":", 1)[1] if cid else None
            ce = configs.get(cid)
            for m in matrix_dirs(rd):
                if srv and not m.name.startswith(f"m_{srv}_"):
                    continue
                if (m / "matrix_summary.json").exists():
                    I.reg(m / "matrix_summary.json")
                for cf in sorted((m / "cells").glob("*.json")):
                    try:
                        cells.append(cell_row(I, cf, run_id, cid, ce, m.name, costs))
                        n_cells += 1
                    except Exception as e:  # noqa: BLE001
                        I.warn(f"{I.rel(cf)}: skipped ({e!r})")
            if srv:
                for sf in [rd / f"startup_facts_{srv}.json"]:
                    if sf.exists():
                        x = I.json(sf)
                        x.update(run_id=run_id, config_id=cid, file=I.rel(sf))
                        startup.append(x)
                for pdir in sorted(rd.glob("protocol_*")):
                    if not (pdir / "protocol_summary.json").exists() or not pdir.is_dir():
                        continue
                    # which server owns this ladder: the summary's config_id, else the dir prefix (contract: protocol_<server>),
                    # else the run's only server (some runs name the dir after the step, e.g. protocol_sgl_b300_A_random)
                    head_cid = json.loads((pdir / "protocol_summary.json").read_text()).get("config_id")
                    if head_cid:
                        if head_cid != cid:
                            continue
                    elif not (pdir.name.startswith(f"protocol_{srv}") or len(cids) == 1):
                        continue
                    x = I.json(pdir / "protocol_summary.json")
                    x.update(run_id=run_id, config_id=cid, file=I.rel(pdir / "protocol_summary.json"))
                    protocols.append(x)
                    kept = set()
                    for l in x.get("ladder") or []:
                        sp = (l.get("seeds") or {}).get(str(l.get("kept_seed")))
                        if sp:
                            cand = [pathlib.Path(sp)] if os.path.isabs(sp) else [pdir / sp, ROOT / sp]
                            kept.update(str(c.resolve()) for c in cand)
                    x["protocol_dir"] = pdir.name
                    for cf in sorted(pdir.rglob("cells/*.json")):
                        try:
                            head = json.loads(cf.read_text())
                        except Exception as e:  # noqa: BLE001
                            I.warn(f"{I.rel(cf)}: unreadable ({e})")
                            continue
                        if head.get("schema") != "traffic/1" and "summary" not in head:
                            continue   # vllm bench serve output (client "vllmbench"): priced via results/protocol_rows.json (contract section 7b)
                        try:
                            cells.append(cell_row(I, cf, run_id, cid, ce, pdir.name, costs, protocol="kept" if str(cf.resolve()) in kept else "other_seed"))
                            n_cells += 1
                        except Exception as e:  # noqa: BLE001
                            I.warn(f"{I.rel(cf)}: skipped ({e!r})")
        for ef in sorted(rd.glob("eval_*.json")) + sorted(rd.glob("out/eval_*.json")) + sorted(rd.glob("gate_*.json")):
            try:
                x = I.json(ef)
            except Exception as e:  # noqa: BLE001
                I.warn(f"{I.rel(ef)}: unreadable ({e})")
                continue
            for task, sm in (x.get("summary") or {}).items():
                gt = x.get("gate") if isinstance(x.get("gate"), dict) and (x["gate"].get("task") == task) else None
                evals.append(dict(run_id=run_id, config_ids=cids, label=x.get("label"), task=task, n=sm.get("n"), correct=sm.get("correct"),
                                  acc=sm.get("acc"), ci95=wilson(sm["correct"], sm["n"]) if isinstance(sm.get("correct"), int) and sm.get("n") else None,
                                  no_answer=sm.get("no_answer"), errors=sm.get("errors"), truncated=sm.get("truncated"),
                                  split=x.get("split"), sets=pick(x, "args", "sets"), limit=pick(x, "args", "limit"), gate_file=gt, file=I.rel(ef)))
        for vf in sorted(rd.glob("vllm_bench_*.json")):
            try:
                v = I.json(vf)
            except Exception as e:  # noqa: BLE001
                I.warn(f"{I.rel(vf)}: unreadable ({e})")
                continue
            ours = next((c for c in cells if c["run_id"] == run_id and "xcheck" in c["matrix"]), None)
            xchecks.append(dict(run_id=run_id, file=I.rel(vf), ours_file=ours["file"] if ours else None, ours_cell=ours["cell_id"] if ours else None,
                                theirs={k: v.get(k) for k in ("completed", "request_throughput", "output_throughput", "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
                                                              "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "median_itl_ms", "p99_itl_ms", "max_concurrency")},
                                ours=None if not ours else dict(completed=ours["completed"], request_throughput=ours["req_s_completed"], output_throughput=ours["out_tok_s"],
                                                                mean_ttft_ms=_ms(ours, "ttft", "mean"), median_ttft_ms=_ms(ours, "ttft", "p50"), p99_ttft_ms=_ms(ours, "ttft", "p99"),
                                                                mean_tpot_ms=_ms(ours, "tpot", "mean"), median_tpot_ms=_ms(ours, "tpot", "p50"), p99_tpot_ms=_ms(ours, "tpot", "p99"),
                                                                mean_itl_ms=_ms(ours, "itl", "mean"), median_itl_ms=_ms(ours, "itl", "p50"), p99_itl_ms=_ms(ours, "itl", "p99"),
                                                                max_concurrency=ours["C"])))
        if (rd / "cost_measured.json").exists():
            x = I.json(rd / "cost_measured.json")
            x.update(run_id=run_id, file=I.rel(rd / "cost_measured.json"))
            measured.append(x)
        gm = None
        if (rd / "gpumon_norm.csv").exists():
            gm = load_gpumon_norm(I, rd / "gpumon_norm.csv")
            if gm:
                gm["file"] = I.rel(rd / "gpumon_norm.csv")
        for g in ("gpumon.csv", "failed_gpumon.log") if not gm else ():
            if (rd / g).exists():
                gm = load_gpumon(I, rd / g)   # raw nvidia-smi fallback (frozen section 9) for runs before gpumon_norm.csv
                if gm:
                    gm["file"] = I.rel(rd / g)
                    gm["note"] = "raw nvidia-smi csv (no gpumon_norm.csv in this run); timestamps are the sandbox clock, assumed UTC"
                    break
        t_start = rep.get("t_start") or (lc or {}).get("t")
        t_end = (ld or {}).get("t") or rep.get("deleted") or rep.get("t_saved")
        runs.append(dict(
            run_id=run_id, label=label, stamp=stamp, config_ids=cids, gpu_key=configs[cids[0]]["gpu_key"] if cids else None,
            sandbox_id=sid, status=status, t_start=t_start, t_end=t_end, elapsed_s=rep.get("elapsed_s"), est_cost_usd=rep.get("est_cost_usd"),
            usd_per_h=rep.get("usd_per_h"), spot_capacity_at_start=rep.get("spot_capacity_at_start"), error=rep.get("error"), evicted=rep.get("evicted"),
            deleted=rep.get("deleted") or (ld or {}).get("t"), system=rep.get("system"), config_file=rep.get("config_file"),
            phases=phases, phase_source=phase_source, n_cells=n_cells, files=files, driver_log=driver, gpumon=gm,
            excluded=rep.get("excluded") or ("evicted" if ev else None), rerun_of_evicted=rep.get("rerun_of_evicted"), evicted_event=ev,
            servers={k: dict(weights=v.get("weights"), download={kk: vv for kk, vv in (v.get("download") or {}).items() if kk != "tail"},
                             serve={kk: vv for kk, vv in (v.get("serve") or {}).items() if kk in ("ready", "ready_s", "gpu_mem_after_load", "launch")},
                             steps={sk: {kk: vv for kk, vv in sv.items() if kk in ("rc", "secs", "skipped")} for sk, sv in (v.get("steps") or {}).items()},
                             end=v.get("end"), skipped=v.get("skipped")) for k, v in (rep.get("servers") or {}).items()},
            run_json=I.rel(rd / "run.json") if (rd / "run.json").exists() else None))
    return runs, cells, evals, startup, protocols, measured, xchecks


def _ms(c, k, p):
    v = (c.get(k) or {}).get(p)
    return None if v is None else v * 1000


def bar_short_of(name, gemma=None):
    """A price bar's short name: drop this benchmark's model token (openrelay/gemma-4-31b -> "OpenRelay sync"), keep any other
    model's ("OpenRelay sync glm-5.3-flash"), so two bars never share a name."""
    return " ".join(t if "/" not in t else t.split("/")[-1] for t in str(name or "").split()
                    if "/" not in t or t.split("/")[-1] != BENCH_MODEL or gemma is False) or None


def load_journey(I, results, runs, cells, sync_usd=None):
    """results/experiments.jsonl (contract addendum, schema experiment/1): one line per decision step.
    Checks: run ids exist in this build, evidence files exist, steps are unique; problems are listed, rows are kept."""
    p = results / "experiments.jsonl"
    if not p.exists():
        I.warn(f"{I.rel(p)} not found (contract addendum): the Journey page shows 'awaiting'")
        return dict(steps=[], file=None, problems=[])
    steps, problems = [], []
    run_ids = {r["run_id"] for r in runs}
    for i, r in enumerate(I.jsonl(p)):
        r["_line"] = i + 1
        for rid in r.get("run_ids") or []:
            if rid not in run_ids:
                problems.append(f"step {r.get('step')}: run {rid} is not in results/")
        for ef in r.get("evidence_files") or []:
            if not (ROOT / ef).exists():
                problems.append(f"step {r.get('step')}: evidence file {ef} does not exist")
            else:
                I.reg(ROOT / ef)
        if r.get("decision") not in ("baseline", "kept", "reverted", "rejected", "info"):
            problems.append(f"step {r.get('step')}: decision {r.get('decision')!r} is not one of baseline/kept/reverted/rejected/info")
        steps.append(r)
    latest = {}
    for r in steps:   # a later line with the same step supersedes an earlier one
        latest[r.get("step")] = r
    superseded = len(steps) - len(latest)
    steps = list(latest.values())
    # each step is priced against its own bar (experiments.jsonl price_bar {name, usd_per_req, source}; absent = OpenRelay's
    # Gemma 4 31B Sync bar). The model is the bar's model: one journey per model, never one path across models.
    for r in steps:
        pb = r.get("price_bar") if isinstance(r.get("price_bar"), dict) else {}
        r["bar_name"] = pb.get("name") or f"OpenRelay sync openrelay/{BENCH_MODEL}"
        r["bar_usd"] = pb.get("usd_per_req") or sync_usd
        r["bar_source"] = pb.get("source")
        models = [t.split("/")[-1] for t in r["bar_name"].split() if "/" in t]
        r["model"] = models[0] if models else BENCH_MODEL
        r["bar_short"] = bar_short_of(r["bar_name"])
        u = r.get("usd_per_req")
        r["ratio_bar"] = u / r["bar_usd"] if isinstance(u, (int, float)) and r["bar_usd"] else None
        if isinstance(r.get("ratio_sync"), (int, float)) and r["ratio_bar"] and abs(r["ratio_sync"] / r["ratio_bar"] - 1) > 0.005:
            problems.append(f"step {r.get('step')}: ratio_sync {r['ratio_sync']:.4f} is not $/req ÷ its own bar ({r['bar_short']}, {r['ratio_bar']:.4f}); the dashboard shows the latter")
    for pr in problems:
        I.warn(f"experiments.jsonl: {pr}")
    steps.sort(key=lambda r: (r.get("step") is None, r.get("step")))
    return dict(steps=steps, file=I.rel(p), problems=problems, superseded=superseded)


def spend_curve(ledger, now, I):
    """Estimated cumulative spend: each sandbox bills usd_per_h.total from create to delete (or to min(now, deadline) if open)."""
    creates = {r["id"]: r for r in ledger if r.get("event") == "create" and r.get("id")}
    deletes = {}
    for r in ledger:
        if r.get("event") == "delete" and r.get("id") and r["id"] not in deletes:
            deletes[r["id"]] = r
    sbx = []
    for sid, c in creates.items():
        t0 = iso_to_epoch(c.get("t"))
        d = deletes.get(sid)
        open_ = d is None
        t1 = iso_to_epoch(d["t"]) if d else min(now, float(c.get("deadline") or now))
        rate = pick(c, "usd_per_h", "total") or 0.0
        sbx.append(dict(id=sid, run=c.get("run"), run_id=c.get("run_id"), gpu=c.get("gpu"), gpus=c.get("gpus", 1), created=c.get("t"),
                        deleted=d["t"] if d else None, open=open_, why=(d or {}).get("why"), usd_per_h=rate,
                        hours=max(0.0, (t1 - t0) / 3600) if t0 else 0.0, est_usd=max(0.0, (t1 - t0) / 3600 * rate) if t0 else 0.0,
                        _t0=t0, _t1=t1))
    if not sbx:
        return dict(points=[], sandboxes=[], milestones=[], total_est=0.0)
    ts = sorted({x["_t0"] for x in sbx if x["_t0"]} | {x["_t1"] for x in sbx if x["_t1"]})
    pts = []
    for t in ts:
        tot = sum(max(0.0, min(t, x["_t1"]) - x["_t0"]) / 3600 * x["usd_per_h"] for x in sbx if x["_t0"])
        live = sum(1 for x in sbx if x["_t0"] and x["_t0"] <= t < x["_t1"])
        pts.append(dict(t=epoch_to_iso(t), usd=round(tot, 4), live=live))
    ms = []
    for m in MILESTONES + [REPLAN_USD]:
        cross = None
        for a, b in zip(pts, pts[1:]):
            if a["usd"] < m <= b["usd"]:
                ta, tb = iso_to_epoch(a["t"]), iso_to_epoch(b["t"])
                cross = epoch_to_iso(ta + (m - a["usd"]) / (b["usd"] - a["usd"]) * (tb - ta))
                break
        ms.append(dict(usd=m, crossed=cross, kind="replan" if m == REPLAN_USD else "milestone"))
    for x in sbx:
        x.pop("_t0")
        x.pop("_t1")
    return dict(points=pts, sandboxes=sbx, milestones=ms, total_est=pts[-1]["usd"] if pts else 0.0)


def metered_points(ledger, I):
    out = [dict(t=r.get("t"), est_total_usd=r.get("est_total_usd"), metered_total_usd=r.get("metered_total_usd"),
                file=r.get("metered_file"), source="infra/ledger.jsonl (spend event)") for r in ledger if r.get("event") == "spend"]
    return out   # contract section 1: only `spend` events; the infra/spend_*.json they point to are cited, not parsed


def stop_rules(startup):
    """PLAN.md section 3/4: a config is rejected if its startup facts are missing/incomplete or show TRITON attention on
    the global layers. Returns {config_id: [reasons]} for configs WITH facts; configs without facts are 'facts missing'."""
    out = {}
    for f in startup:
        why = []
        if f.get("missing"):
            why.append("startup facts incomplete: " + ", ".join(f["missing"]))
        g5 = (f.get("attention_backend") or {}).get("global_512")
        if g5 and "triton" in str(g5).lower():
            why.append(f"TRITON attention on global_512 ({g5}): PLAN section 4 stop rule")
        if f.get("rejected") and not why:
            why.append("startup_facts.rejected = true")
        if why and not f.get("rejected"):
            why.append("(note: file says rejected=false)")
        out[f.get("config_id")] = why
    return out


def best_results(cells, evals, configs, startup=(), prow=(), excluded_runs=()):
    """Per config: cheapest agentic-shape cell meeting OpenRelay's SLO (protocol ladders: kept seed only);
    gate = GSM8K >= 96.5% on that config (any of its runs); stop = PLAN section 3/4 rejection reasons."""
    sr = stop_rules(startup)
    gate = {}
    for e in evals:
        if (e.get("task") or "").lower().startswith("gsm8k") and e.get("acc") is not None:
            for cid in e.get("config_ids") or []:
                g = gate.get(cid)
                if g is None or (e["n"] or 0) > (g["n"] or 0):
                    fg = e.get("gate_file")      # the eval file's own gate {task, threshold, acc, pass} when present (contract section 7)
                    acc_ok = bool(fg["pass"]) if fg and "pass" in fg else e["acc"] >= GSM8K_GATE
                    gate[cid] = dict(acc=e["acc"], n=e["n"], passed=acc_ok and (e["n"] or 0) >= GATE_MIN_N, too_small=(e["n"] or 0) < GATE_MIN_N,
                                     split=e.get("split"), file_gate=fg, run_id=e["run_id"], file=e["file"])
    best = {}
    for c in cells:
        if not (c["agentic_shape"] and c["slo_openrelay"] and c["our_usd_req"]) or c.get("protocol") == "other_seed" or c["run_id"] in excluded_runs:
            continue
        b = best.get(c["config_id"])
        if b is None or c["our_usd_req"] < b["our_usd_req"]:
            best[c["config_id"]] = c
    # ladder rungs (protocol_rows.json): kept seed, SLO met; the headline price is at the offered rate (contract 7c: $/h / 3600 / max_rate_at_slo)
    for r in prow:
        if not (r.get("kept") and r.get("slo_pass") and r.get("usd_per_req_at_achieved")) or r.get("run_id") in excluded_runs:
            continue   # priced on the ACHIEVED rate (lead ruling; claims.json does the same)
        cid = r.get("config_id")
        c = dict(config_id=cid, gpu_key=(configs.get(cid) or {}).get("gpu_key"), run_id=r.get("run_id"), cell_id=f"{r.get('protocol')}/x{r.get('multiple')}_s{r.get('seed')}",
                 file=r.get("summary_file") or r.get("file"), our_usd_req=r["usd_per_req_at_achieved"], req_s=r.get("achieved_rps"),
                 ttft=dict(p95=r.get("ttft_p95_s")), tpot=dict(p50=r.get("tpot_p50_s")),
                 ratio_sync=(r["usd_per_req_at_achieved"] / r["openrelay_sync_usd_per_req"]) if r.get("openrelay_sync_usd_per_req") else None, protocol="ladder", client=r.get("client"),
                 ratio_sync_metered=r.get("ratio_sync_at_offered_metered"), usd_per_h_metered=r.get("usd_per_h_metered"), metered_read_at=r.get("metered_read_at"),
                 usd_per_req_metered=(r["usd_per_h_metered"] / 3600 / r["achieved_rps"]) if (r.get("usd_per_h_metered") and r.get("achieved_rps")) else None)
        b = best.get(cid)
        if b is None or c["our_usd_req"] < b["our_usd_req"]:
            best[cid] = c
    out = []
    for cid, c in best.items():
        out.append(dict(config_id=cid, gpu_key=c["gpu_key"], run_id=c["run_id"], cell_id=c["cell_id"], file=c["file"], our_usd_req=c["our_usd_req"],
                        req_s=c["req_s"], ttft_p95=pick(c, "ttft", "p95"), tpot_p50=pick(c, "tpot", "p50"), ratio_sync=c["ratio_sync"],
                        gate=gate.get(cid), protocol=c.get("protocol"), client=c.get("client"), ratio_sync_metered=c.get("ratio_sync_metered"),
                        usd_per_req_metered=c.get("usd_per_req_metered"), metered_read_at=c.get("metered_read_at"),
                        stop=sr.get(cid, ["no startup facts (contract section 4): PLAN section 3 rejects the cell"]),
                        eligible=bool(gate.get(cid, {}).get("passed")) and not sr.get(cid, ["missing"])))
    out.sort(key=lambda r: r["our_usd_req"])
    return out, gate


def assign_symbols(configs, runs):
    """Marker shape per config, stable as data lands: configs with runs first (by first run start), then the rest by id;
    counted within each GPU so (GPU colour, shape) identifies a config."""
    first = {}
    for r in sorted(runs, key=lambda r: r.get("t_start") or ""):
        for cid in r.get("config_ids") or []:
            first.setdefault(cid, r.get("t_start") or "")
    order = sorted(configs, key=lambda cid: (cid not in first, first.get(cid, ""), cid))
    n = {}
    for cid in order:
        c = configs[cid]
        i = n.get(c["gpu_key"], 0)
        n[c["gpu_key"]] = i + 1
        c.update(symbol=SYMBOLS[i % len(SYMBOLS)], dash=DASHES[i % len(DASHES)], symbol_index=i)


def build(a):
    I = Inputs(ROOT)
    import daytona as dt   # infra/daytona.py, read-only: only hourly_usd() and the price tables are used (no API call)
    results = pathlib.Path(a.results).resolve()
    ledger_p = pathlib.Path(a.ledger).resolve()
    now = time.time()
    bars = load_bars(I)
    costs, cost_file = load_cost_rows(I, results)
    configs = load_configs(I, dt)
    ledger = I.jsonl(ledger_p) if ledger_p.exists() else []
    runs, cells, evals, startup, protocols, measured, xchecks = load_runs(I, results, configs, ledger, costs, dt)
    unpriced = [c for c in cells if not c["priced"]]
    if costs and unpriced:
        I.warn(f"{len(unpriced)} of {len(cells)} cells have no row in {cost_file} (e.g. {unpriced[0]['file']}): shown without $")
    refs_p = DASH / "references.json"
    refs = I.json(refs_p)
    journey = load_journey(I, results, runs, cells, sync_usd=(bars.get("per_request") or {}).get("sync"))
    claims = load_claims(I, results, runs, configs)
    paired = load_paired(I, results, runs, protocols, evals, startup)
    qt = results / "QUALITY_TABLE.md"
    quality_table = I.rel(qt) if qt.exists() else None
    if qt.exists():
        I.reg(qt)
    spend = spend_curve(ledger, now, I)
    spend["metered"] = metered_points(ledger, I)
    met = load_metered(I, results)
    if met:
        spend["metered_snapshot"] = dict(read_at=met.get("read_at"), metered_total_usd=met.get("metered_total_usd"), file=met["file"], source=met.get("source"))
        for sb in spend["sandboxes"]:
            m = (met.get("sandboxes") or {}).get(sb["id"])
            if m:
                sb.update(metered_usd=m.get("metered_usd"), metered_usd_per_h=m.get("metered_usd_per_h"), published_usd_per_h=m.get("published_usd_per_h"))
        if isinstance(met.get("metered_total_usd"), (int, float)):
            spend["metered"].append(dict(t=met.get("read_at"), est_total_usd=None, metered_total_usd=met["metered_total_usd"], file=met["file"], source="results/metered_usd.json"))
            spend["metered"].sort(key=lambda m: m.get("t") or "")
    ev = [dict(t=r.get("t"), run_id=r.get("run_id"), sandbox_id=r.get("sandbox_id"), gpu=gpu_key(r.get("gpu"), r.get("gpus", 1)), evicted_at=r.get("evicted_at"),
               last_good_t=r.get("last_good_t"), interrupted=r.get("interrupted"), minutes_lost=r.get("minutes_lost"), usd_lost=r.get("usd_lost"),
               usd_per_h=r.get("usd_per_h"), label=r.get("label")) for r in ledger if r.get("event") == "evicted"]
    by = {}
    for e in ev:
        g = by.setdefault(e["gpu"], dict(gpu=e["gpu"], count=0, minutes_lost=0.0, usd_lost=0.0))
        g["count"] += 1
        g["minutes_lost"] += e.get("minutes_lost") or 0
        g["usd_lost"] += e.get("usd_lost") or 0
    reruns = {r["run_id"]: r.get("rerun_of_evicted") for r in runs if r.get("rerun_of_evicted")}
    for e in ev:
        e["rerun_run_ids"] = [k for k, v in reruns.items() if v and (v == e["run_id"] or (isinstance(v, dict) and v.get("run_id") == e["run_id"]))]
    evictions = dict(events=ev, by_gpu=sorted(by.values(), key=lambda g: -g["usd_lost"]), total_count=len(ev),
                     total_minutes=sum(e.get("minutes_lost") or 0 for e in ev), total_usd=sum(e.get("usd_lost") or 0 for e in ev),
                     note="spot preemption, outside our control: shown on its own, never folded into a cell or a $/request")
    spend["reported"] = [dict(t=r.get("t"), threshold_usd=r.get("threshold_usd"), total_usd=r.get("total_usd"), basis=r.get("basis"))
                         for r in ledger if r.get("event") == "milestone"]
    spend["ledger_file"] = I.rel(ledger_p)
    prow, prow_file = load_protocol_rows(I, results)
    rescore_paired(paired, prow)
    claim_hosts(I, results, claims, prow)
    same_box = same_box_pairs(claims, prow)
    levers = host_levers(claims, same_box)
    scope = load_scope(I, claims, same_box)
    quality = load_quality(I, results)
    run_groups = {}
    for r in prow:
        if r.get("claim_group"):
            run_groups.setdefault(r.get("run_id"), set()).add(r["claim_group"])
    for st in journey.get("steps", []):
        st["claim_groups"] = sorted({g for rid in st.get("run_ids") or [] for g in run_groups.get(rid, ())})
    excluded_runs = {r["run_id"] for r in runs if r.get("excluded")}
    best, gate = best_results(cells, evals, configs, startup, prow, excluded_runs)
    assign_symbols(configs, runs)
    # integrity checks -> warnings shown on the Sources page and as a count in the header
    for c in cells:
        if c.get("schema") not in (None, "traffic/1"):
            I.warn(f"{c['file']}: schema {c['schema']!r}, expected 'traffic/1' (contract section 2)")
        if c.get("client_vs_server_out") is not None and abs(c["client_vs_server_out"]) > 0.01 and c["mode"]:
            I.warn(f"{c['cell_id']} ({c['run_id']}): client out tok/s differs from the server counter by {c['client_vs_server_out'] * 100:+.1f}% (PLAN.md section 3 wants 1%)")
        if c["errors"]:
            I.warn(f"{c['cell_id']} ({c['run_id']}): {c['errors']} errors of {c['requests_sent']} sent")
    if not any(r.get("event") == "phase" for r in ledger):
        I.warn("infra/ledger.jsonl has no `phase` events yet (contract section 1): phase profiles are coarse, from run.json durations")
    if not startup:
        I.warn("no startup_facts_*.json yet (contract section 4): the Startup page shows declared flags only")
    for r in runs:
        if r["status"] == "done" and r["n_cells"] == 0 and r["servers"] and not any(pr["run_id"] == r["run_id"] for pr in protocols):
            I.warn(f"{r['run_id']}: finished but no matrix cells were found under m_*/ or out/m_*/")
    used_gpus = sorted({c["gpu_key"] for c in configs.values()}, key=lambda g: GPU_ORDER.index(g) if g in GPU_ORDER else 99)
    gpus = {g: dict(slot=(GPU_ORDER.index(g) + 1) if g in GPU_ORDER else None, spot_usd_h=dt.GPU_SPOT_USD_H.get(g.split("x")[-1]),
                    ondemand_usd_h=dt.GPU_ONDEMAND_USD_H.get(g.split("x")[-1])) for g in used_gpus}
    for g, v in gpus.items():
        if v["slot"] is None:
            I.warn(f"GPU {g} has no fixed colour slot in dashboard/build.py GPU_ORDER (shown in neutral)")
    data = dict(
        build=dict(t=epoch_to_iso(now), t_epoch=now, demo=a.demo, bench_model=BENCH_MODEL, contract_version=CONTRACT_VERSION, results_dir=I.rel(results), python=sys.version.split()[0],
                   builder="dashboard/build.py"),
        constants=dict(shape_in=SHAPE_IN, shape_out=SHAPE_OUT, shape_tol=SHAPE_TOL, slo_ttft_p95_s=SLO_TTFT_P95_S, max_err_frac=MAX_ERR_FRAC,
                       gsm8k_gate=GSM8K_GATE, gate_min_n=GATE_MIN_N, milestones=MILESTONES, replan_usd=REPLAN_USD, gpu_order=GPU_ORDER, phase_cats=[c for c, _ in PHASE_CATS]),
        bars=bars, refs=refs, gpus=gpus, configs=configs, runs=runs, cells=cells, evals=evals, startup=startup, protocols=protocols,
        measured=measured, xchecks=xchecks, spend=spend, best=best, gate=gate, journey=journey, protocol_rows=prow, evictions=evictions, claims=claims, ladder_points=ladder_points(prow, protocols, configs), paired=paired, quality_table=quality_table,
        same_box=same_box, host_levers=levers, scope=scope, quality=quality,
        ledger_counts={k: sum(1 for r in ledger if r.get("event") == k) for k in sorted({r.get("event") for r in ledger if r.get("event")})},
    )
    cp = ROOT / CONTRACT_FILE
    data["contract"] = I.text(cp) if cp.exists() else None
    data["build"]["contract_file"] = CONTRACT_FILE
    data["build"]["cost_rows_file"] = cost_file
    data["build"]["protocol_rows_file"] = prow_file
    data["inputs"] = sorted(I.files.values(), key=lambda x: x["path"])
    data["warnings"] = I.warnings
    return data


# ----------------------------------------------------------------------------- site
UI = DASH / "ui"   # compiled React bundle (committed). Rebuild only when dashboard/app/src changes: cd dashboard/app && npm ci && npm run build


def write_site(data, out, demo):
    """dashboard/ui/* (the app) + data.js (window.BENCH) + data.json. No node needed: the bundle is prebuilt."""
    out = pathlib.Path(out)
    if not (UI / "index.html").exists():
        sys.exit("dashboard/ui/ missing: cd dashboard/app && npm ci && npm run build")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(UI, out)
    (out / "data.js").write_text("window.BENCH = " + json.dumps(data, default=str, separators=(",", ":")).replace("</", "<\\/") + ";\n")
    (out / "data.json").write_text(json.dumps(data, default=str, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--demo", action="store_true", help="build from dashboard/demo/ (mock, contract-shaped) into dashboard/site-demo/")
    a = ap.parse_args()
    if a.demo:
        a.results = a.results or str(DASH / "demo" / "results")
        a.ledger = a.ledger or str(DASH / "demo" / "ledger.jsonl")
        a.out = a.out or str(DASH / "site-demo")
    a.results = a.results or str(ROOT / "results")
    a.ledger = a.ledger or str(ROOT / "infra" / "ledger.jsonl")
    a.out = a.out or str(DASH / "site")
    if not str(pathlib.Path(a.out).resolve()).startswith(str(DASH)):
        sys.exit("refusing to write outside dashboard/")
    t = time.time()
    data = build(a)
    write_site(data, a.out, a.demo)
    print(f"built {a.out}: {len(data['runs'])} runs, {len(data['cells'])} cells, {len(data['configs'])} configs, "
          f"{len(data['inputs'])} input files, {len(data['warnings'])} warnings, spend est ${data['spend']['total_est']:.2f} "
          f"({time.time() - t:.1f}s)")
    for w in data["warnings"][:12]:
        print("  warn:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
