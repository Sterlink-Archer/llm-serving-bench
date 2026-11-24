#!/usr/bin/env python3
"""MOCK data generator for developing the dashboard. NOT MEASUREMENTS.

Writes contract-shaped fake results (dashboard/CONTRACT.md) under dashboard/demo/results/ and dashboard/demo/ledger.jsonl,
so every page can be exercised before the real ladder, startup facts and phase events exist. Every file it writes carries
"_demo": true (JSON) or sits under dashboard/demo/. `python3 -I dashboard/build.py --demo` builds it into
dashboard/site-demo/ with a banner on every page. Nothing here is ever read by the real build.

  python3 -I dashboard/demo/make_demo.py
"""
from __future__ import annotations

import calendar
import json
import math
import pathlib
import random
import shutil
import time

DEMO = pathlib.Path(__file__).resolve().parent
ROOT = DEMO.parent.parent
OUT = DEMO / "results"
LEDGER = DEMO / "ledger.jsonl"
R = random.Random(20261122)
T0 = calendar.timegm(time.strptime("2026-11-22T04:30:00Z", "%Y-%m-%dT%H:%M:%SZ"))
VCPU, MEM, DISK = 0.0504, 0.0162, 0.000108
GPU_H = {"H100": 2.47, "RTX-PRO-6000": 1.95, "MI355X": 3.56, "L40S": 1.14, "RTX-5090": 0.74}


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def stamp(t):
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(t))


def pph(gpu, cpu=8, mem=48, disk=150):
    p = dict(gpu=GPU_H[gpu], cpu=cpu * VCPU, mem=mem * MEM, disk=(disk - 5) * DISK)
    p["total"] = sum(p.values())
    return {k: round(v, 4) for k, v in p.items()}


def dist(center, spread, n):
    xs = sorted(max(1e-4, R.lognormvariate(math.log(center), spread)) for _ in range(max(5, n)))
    pct = lambda p: xs[min(len(xs) - 1, int(p / 100 * (len(xs) - 1)))]  # noqa: E731
    return dict(n=len(xs), mean=sum(xs) / len(xs), min=xs[0], max=xs[-1], p50=pct(50), p90=pct(90), p95=pct(95), p99=pct(99))


def cell(label, mode, load, in_tok, out_tok, cap_rps, ttft0, tpot0, workload="synthetic", spec=None, window=120):
    """A traffic.py-shaped cell from a toy queueing model (fake)."""
    if mode == "open":
        rho = load / cap_rps
        achieved = min(load, cap_rps) * (0.985 if rho < 1 else 1.0)
        qf = 1 / max(0.04, 1 - min(rho, 0.97)) if rho < 1 else 25 + 40 * (rho - 1)
        ttft_c, tpot_c = ttft0 * qf ** 0.85, tpot0 * (1 + 0.6 * min(rho, 1.2))
    else:
        achieved = cap_rps * load / (load + 6)
        ttft_c, tpot_c = ttft0 * (1 + load / 24), tpot0 * (1 + load / 40)
    accept = None
    if spec:
        conc = load if mode == "closed" else load * 4
        accept = 1 + spec * (0.82 - 0.012 * conc) if spec else None
        accept = max(1.05, accept)
        tpot_c = tpot_c / (accept * (1 - 0.006 * conc)) if accept else tpot_c
        achieved *= max(0.55, accept * (1 - 0.0125 * conc))   # toy: speculation helps at low load, hurts past the knee
    n = int(achieved * window)
    out_tok_s = achieved * out_tok
    sm = dict(window_s=window, gen_tok_per_s=out_tok_s * (1 + R.uniform(-0.004, 0.004)), prompt_tok_per_s=achieved * in_tok, preemptions=0.0,
              prefix_cache_hit_ratio=0.0, counters=["vllm:generation_tokens_total"])
    if accept:
        sm["spec_mean_accepted_per_step"] = accept
        sm["spec_acceptance_rate"] = (accept - 1) / spec
    s = dict(window_s=window, mode=mode, offered_rate=load if mode == "open" else None, concurrency=load if mode == "closed" else None, peak_inflight=int(load if mode == "closed" else load * 6),
             requests_sent=n + 8, dropped=0, completed=n, completed_in_window=n, aborted_at_end=4, errors=0, req_per_s=achieved, output_tok_per_s=out_tok_s,
             input_tok_per_s=achieved * in_tok, cached_input_tok_per_s=0.0, total_tok_per_s=achieved * (in_tok + out_tok), cache_hit_ratio=0.0, cached_tokens_reported=True,
             prompt_tokens=dict(n=n, mean=in_tok, min=in_tok, max=in_tok, p50=in_tok, p90=in_tok, p95=in_tok, p99=in_tok),
             completion_tokens=dict(n=n, mean=out_tok, min=out_tok, max=out_tok, p50=out_tok, p90=out_tok, p95=out_tok, p99=out_tok),
             ttft_s=dist(ttft_c, 0.45, n), tpot_s=dist(tpot_c, 0.12, n), itl_s=dist(tpot_c * 0.97, 0.35, n * 50), e2e_s=dist(ttft_c + tpot_c * out_tok, 0.2, n),
             per_user_tok_per_s_p50=1 / tpot_c, server_metrics=sm)
    reqs = [dict(i=i, cls=workload, status=200, prompt_tokens=in_tok, cached_tokens=0, completion_tokens=out_tok) for i in range(min(n, 40))]
    return dict(_demo=True, schema="traffic/1", label=label, args=dict(workload=workload), params=dict(input_tokens=in_tok, output_tokens=out_tok, ignore_eos=True),
                t_start=None, summary=s, requests=reqs)


class Run:
    def __init__(self, cfg_file, t, ledger, evict=False):
        self.cfg = json.loads((ROOT / "configs" / cfg_file).read_text())
        self.label = self.cfg["label"]
        self.t = t
        self.dir = OUT / self.label / stamp(t)
        self.dir.mkdir(parents=True)
        snap = self.cfg["snapshot"]
        self.gpu = snap["gpu"]
        self.pph = pph(self.gpu, snap["cpu"], snap["mem"], snap["disk"])
        self.sid = f"demo-{R.randrange(16**8):08x}"
        self.run_id = f"{self.label}/{stamp(t)}"
        self.ledger = ledger
        self.phases = []
        self.cur = t
        self.rep = dict(_demo=True, label=self.label, config_file=f"configs/{cfg_file}", t_start=iso(t), usd_per_h=self.pph, phases={}, servers={}, sandbox_id=self.sid,
                        spot_capacity_at_start=R.randrange(20, 200))
        (self.dir / "config.json").write_text(json.dumps(dict(self.cfg, _demo=True), indent=1))
        ledger.append(dict(t=iso(t), event="create", id=self.sid, run=self.label, run_id=self.run_id, gpu=self.gpu, gpus=1, cpu=snap["cpu"], mem=snap["mem"], disk=snap["disk"],
                           spot=True, usd_per_h=self.pph, ttl_min=self.cfg["max_minutes"] + 5, deadline=int(t + (self.cfg["max_minutes"] + 5) * 60), _demo=True))
        self.evict = evict

    def phase(self, name, secs, cid=None, nested=None, detail=None, cell_id=None):
        t0 = self.cur if not nested else self.nest_t
        rec = dict(t=iso(t0 + secs), event="phase", run_id=self.run_id, config_id=cid, cell_id=cell_id, sandbox_id=self.sid, gpu=self.gpu, gpus=1,
                   image_digest=self.cfg["snapshot"].get("image_digest_amd64"), model_repo=None, model_rev=None, phase=name, t_start=iso(t0), t_end=iso(t0 + secs),
                   dur_s=secs, usd_per_h=self.pph["total"], usd=round(secs / 3600 * self.pph["total"], 5), nested_in=nested, ok=True, detail=detail or {}, _demo=True)
        self.ledger.append(rec)
        if nested:
            self.nest_t += secs
        else:
            self.nest_t = t0
            self.cur += secs

    def server(self, srv, cells_spec, protocol=None, eval_acc=None, kv=None, graphs=None, spec=None, attn=None):
        cid = f"{self.label}:{srv}"
        w = next(s for s in self.cfg["servers"] if s["name"] == srv)
        repo = (w.get("weights") or [{}])[0].get("repo")
        nbytes = R.choice([18e9, 33e9, 21e9])
        dl = nbytes / R.uniform(60e6, 400e6)
        self.phase("weight_download", dl, cid, detail=dict(bytes=int(nbytes), source="huggingface.co", mb_s=nbytes / 1e6 / dl))
        es = R.uniform(90, 200)
        self.phase("engine_start", es, cid)
        self.phase("torch_compile", es * 0.45, cid, nested="engine_start")
        self.phase("graph_capture", es * 0.2, cid, nested="engine_start", detail=dict(sizes=graphs))
        self.phase("warmup", R.uniform(8, 20), cid)
        mdir = self.dir / "out" / f"m_{srv}_screen" / "cells"
        mdir.mkdir(parents=True)
        for c in cells_spec:
            d = cell(**c)
            d["t_start"] = iso(self.cur)
            (mdir / f"{c['label']}.json").write_text(json.dumps(d, indent=1))
            self.phase("steady", c.get("window", 120) + 10, cid, cell_id=f"m_{srv}_screen/{c['label']}", detail=dict(requests_completed=d["summary"]["completed"]))
        if protocol:
            pdir = self.dir / f"protocol_{srv}"
            (pdir / "cells").mkdir(parents=True)
            be, cap, ttft0, tpot0 = protocol
            ladder = []
            for mult in (0.6, 0.9, 1.2, 1.6, 2.0):
                rate = round(be * mult, 3)
                seeds = {}
                worst = None
                for seed in (7, 8):
                    d = cell(label=f"ladder_r{rate}_s{seed}", mode="open", load=rate, in_tok=9500, out_tok=380, cap_rps=cap * R.uniform(0.97, 1.0), ttft0=ttft0, tpot0=tpot0, window=300, spec=spec)
                    p = pdir / "cells" / f"ladder_r{rate}_s{seed}.json"
                    p.write_text(json.dumps(d, indent=1))
                    seeds[str(seed)] = str(p.relative_to(ROOT))
                    if worst is None or d["summary"]["ttft_s"]["p95"] > worst[1]["summary"]["ttft_s"]["p95"]:
                        worst = (seed, d)
                s = worst[1]["summary"]
                ok = s["ttft_s"]["p95"] < 4.0
                ladder.append(dict(multiple=mult, rate=rate, seeds=seeds, kept_seed=worst[0], achieved_rps=s["req_per_s"], goodput_rps=s["req_per_s"] if ok else 0.0,
                                   ttft_p95_s=s["ttft_s"]["p95"], tpot_p50_s=s["tpot_s"]["p50"], tpot_p95_s=s["tpot_s"]["p95"], errors=0, slo_pass=ok))
                self.phase("steady", 310 * 2, cid, cell_id=f"protocol_{srv}/r{rate}")
            (pdir / "protocol_summary.json").write_text(json.dumps(dict(_demo=True, schema="protocol/1", run_id=self.run_id, config_id=cid, gpu=self.gpu, slo=dict(ttft_p95_s=4.0),
                breakeven_rps=be, duration_s=300, definitions="DEMO: goodput_rps = achieved req/s of the worse seed if its p95 TTFT < 4 s else 0; max_rate_at_slo_rps = highest offered rate with slo_pass",
                ladder=ladder, max_rate_at_slo_rps=max((l["rate"] for l in ladder if l["slo_pass"]), default=None),
                prefill=dict(concurrency=16, prompt_tokens=10000, prefill_tok_s=R.uniform(9000, 30000), file=None)), indent=1))
        if eval_acc is not None:
            n = 30
            c = round(eval_acc * n)
            (self.dir / f"eval_{srv}.json").write_text(json.dumps(dict(_demo=True, label=f"eval_{srv}", split="gsm8k_200[:30]", args=dict(limit=n, sets="data/eval/gsm8k_200.jsonl"),
                summary=dict(gsm8k=dict(n=n, correct=c, acc=c / n, no_answer=0, errors=0, truncated=0)),
                gate=dict(task="gsm8k", threshold=0.965, acc=c / n, **{"pass": c / n >= 0.965})), indent=1))
        (self.dir / f"startup_facts_{srv}.json").write_text(json.dumps(dict(_demo=True, schema="startup_facts/1", run_id=self.run_id, config_id=cid, engine="vllm", engine_version="0.33.0",
            image=self.cfg["snapshot"].get("image"), image_digest=self.cfg["snapshot"].get("image_digest_amd64"), model_repo=repo, model_rev=None,
            attention_backend=(attn := attn or {"sliding": "FLASH_ATTN", "global_512": "FLASHINFER"}),
            attention_backend_lines=[f"DEMO log line: Using {v} backend for {k} layers" for k, v in attn.items()],
            kv_cache_tokens=kv, kv_cache_gib=(kv * 0.000061 if kv else None), kv_cache_dtype="fp8", max_concurrency_at_max_len=(kv / 40960 if kv else None),
            cuda_graph_sizes=graphs, max_model_len=40960, quantization=dict(weights=(repo or "").split("-")[-1].lower(), detail=None),
            spec_decode=(dict(method="mtp", draft_model="google/gemma-4-31B-it-assistant", num_speculative_tokens=spec) if spec else None), gpu_mem_after_load_mib=R.randrange(60000, 79000),
            source_log=f"results/{self.run_id}/server_{srv}.log", missing=[] if kv else ["kv_cache_tokens"],
            rejected=(not kv) or "TRITON" in str(attn.get("global_512"))), indent=1))
        self.rep["servers"][srv] = dict(weights=[dict(repo=repo, bytes=int(nbytes))], download=dict(rc=0, secs=dl), serve=dict(ready=True, ready_s=es))

    def finish(self):
        if self.evict:
            self.phase("eviction", 5)
            self.rep["evicted"] = dict(state="destroyed", spotEvictedAt=iso(self.cur))
        else:
            self.phase("teardown", R.uniform(20, 40))
        self.ledger.append(dict(t=iso(self.cur), event="delete", id=self.sid, run_id=self.run_id, why="demo end", _demo=True))
        self.rep.update(deleted=iso(self.cur), elapsed_s=self.cur - self.t, est_cost_usd=round((self.cur - self.t) / 3600 * self.pph["total"], 3))
        (self.dir / "run.json").write_text(json.dumps(self.rep, indent=1))
        # gpu monitor csv, nvidia-smi style
        rows = []
        norm = ["t_utc,gpu_index,util_pct,mem_used_mib,power_w,sm_mhz,temp_c"]
        t = self.t
        while t < self.cur:
            busy = any(p["phase"] == "steady" and p["t_start"] <= iso(t) <= p["t_end"] for p in self.ledger if p.get("run_id") == self.run_id and p.get("event") == "phase")
            u, m, w, tc = (R.randint(88, 99) if busy else R.randint(0, 6)), (R.randint(70000, 78000) if t > self.t + 300 else R.randint(0, 2000)), (R.uniform(550, 690) if busy else R.uniform(70, 110)), R.randint(55, 72)
            rows.append(f"{time.strftime('%Y/%m/%d %H:%M:%S', time.gmtime(t))}.000, 0, {u} %, {m} MiB, {w:.2f} W, 1980 MHz, {tc}")
            norm.append(f"{iso(t)},0,{u},{m},{w:.2f},1980,{tc}")
            t += 10
        (self.dir / "gpumon.csv").write_text("\n".join(rows) + "\n")
        (self.dir / "gpumon_norm.csv").write_text("\n".join(norm) + "\n")
        return self.cur


def write_claims():
    """Mock results/claims.json (schema claims/1 as written by analysis/screen_report.py). Fake numbers."""
    runs = sorted(str(p.parent.relative_to(OUT)) for p in OUT.glob("*/*/run.json"))
    rtx = next(r for r in runs if r.startswith("e4b_")); mi = next(r for r in runs if r.startswith("e4d_"))
    def row(group, col, run, rate, ach, ttft, usd_h, grade, why):
        u = usd_h / 3600 / ach
        return dict(_demo=True, group=group, column=col, column_name="OpenRelay client, random tokens" if col == "A" else "our client, real text", rates_tested=[rate, rate + 0.25],
                    runs=[run], max_rate_all_seeds_pass=rate, seeds=[f"{run}:s7", f"{run}:s8"] if grade else [f"{run}:s7"], worst_seed_achieved_rps=ach, worst_seed_ttft_p95_s=ttft,
                    usd_per_h=usd_h, usd_per_req=u, ratio_vs_openrelay_sync=u / 0.000779, usd_per_h_metered=usd_h * 0.9, usd_per_req_metered=usd_h * 0.9 / 3600 / ach,
                    ratio_vs_openrelay_sync_metered=usd_h * 0.9 / 3600 / ach / 0.000779, claim_grade=grade, not_claim_grade_because=[] if grade else ["1 seed(s) at this rate (need 2)"],
                    origin="openrelay_recipe" if "or_recipe" in group else "ours", gpu="MI355X" if "mi355x" in group else "RTX-PRO-6000", engine="vllm")
    rows = [row("e4b_rtxpro6000_vllm", "B", rtx, 1.25, 1.21, 3.1, 3.147, True, None), row("e4b_rtxpro6000_vllm", "A", rtx, 1.5, 1.44, 3.4, 3.147, True, None),
            row("e4d_mi355x_or_recipe", "A", mi, 3.25, 3.18, 3.3, 4.7565, False, None), row("e4d_mi355x_or_recipe", "B", mi, 3.0, 2.76, 1.8, 4.7565, False, None)]
    (OUT / "claims.json").write_text(json.dumps(dict(schema="claims/1", openrelay_sync_usd_per_req=0.000779, rules="DEMO", claims=rows), indent=1))


def write_experiments():
    """Mock results/experiments.jsonl (schema experiment/1 as proposed for the contract addendum). Fake numbers."""
    runs = sorted(str(p.parent.relative_to(OUT)) for p in OUT.glob("*/*/run.json"))
    rtx = next(r for r in runs if r.startswith("e4b_")); h100 = next(r for r in runs if r.startswith("e2b_")); mi = next(r for r in runs if r.startswith("e4d_"))
    steps = [
        (0, "baseline", "e2b_h100_vllm_fp8_specdec:nospec", "H100, vLLM FP8 weights, BF16 KV, defaults", "the pilot's config", [h100], 0.92, 0.00111),
        (1, "kept", "e2b_h100_vllm_fp8_specdec:nospec", "FP8 KV cache", "doubles KV room; prefill-bound at 9.5K", [h100], 1.08, 0.00094),
        (2, "reverted", "e2b_h100_vllm_fp8_specdec:mtp4", "MTP, 4 draft tokens", "acceptance collapses past C=32", [h100], 0.96, 0.00106),
        (3, "kept", "e2b_h100_vllm_fp8_specdec:mtp2", "MTP, 2 draft tokens", "OpenRelay's width; pays at the SLO load", [h100], 1.21, 0.00084),
        (4, "kept", "e4b_rtxpro6000_vllm:fp8dyn", "move to RTX PRO 6000 (FP8)", "box $/h 2.76 vs 3.67", [rtx], 0.88, 0.00087),
        (5, "kept", "e4b_rtxpro6000_vllm:nvfp4", "NVFP4 weights", "Blackwell FP4 tensor cores; GSM8K held", [rtx], 1.19, 0.00065),
        (6, "reverted", "e4d_mi355x_vllm:mxfp4", "MI355X with OpenRelay's recipe", "TRITON on global layers: stop rule", [mi], 1.50, 0.00088),
        (7, "kept", "e4b_rtxpro6000_vllm:nvfp4", "max-num-seqs 128, graphs to 256", "fewer preemptions at the knee", [rtx], 1.31, 0.00059),
    ]
    with open(OUT / "experiments.jsonl", "w") as f:
        for st, dec, cid, change, why, rids, g, u in steps:
            f.write(json.dumps(dict(_demo=True, schema="experiment/1", step=st, t=iso(T0 + st * 3600), kind="baseline" if st == 0 else "change", config_id=cid,
                                    parent_config_id=None, change=change, why=why, run_ids=rids, basis="screen", slo_goodput_rps=g, usd_per_req=u,
                                    ratio_sync=u / 0.000779, ttft_p95_s=round(1.2 + st * 0.15, 2), tpot_p50_s=0.03, gsm8k_acc=0.975, decision=dec, evidence_files=[])) + "\n")


def write_cost_rows():
    """Mock results/cost_rows.json with analysis/cost.py's row field names (contract section 6). Fake numbers."""
    orl = json.loads(sorted((ROOT / "research").glob("openrelay_prices_*.json"))[-1].read_text())
    tiers = orl["models"]["openrelay/gemma-4-31b"]["tiers"]
    sync, batch = tiers["Sync"], tiers["Batch"]
    cost = lambda t, i, o, c=0.0: ((i - c) * t["in_usd_per_m"] + c * t["cached_in_usd_per_m"] + o * t["out_usd_per_m"]) / 1e6  # noqa: E731
    rows = []
    for cfgf in sorted(OUT.glob("*/*/config.json")):
        rd = cfgf.parent
        snap = json.loads(cfgf.read_text())["snapshot"]
        usd_h = pph(snap["gpu"], snap["cpu"], snap["mem"], snap["disk"])["total"]
        for cf in sorted(rd.glob("out/m_*/cells/*.json")) + sorted(rd.glob("protocol_*/cells/*.json")):
            d = json.loads(cf.read_text())
            s, p = d["summary"], d["params"]
            i, o = p["input_tokens"], p["output_tokens"]
            rps = s["output_tok_per_s"] / o
            our = usd_h / 3600 / rps if rps else None
            b = cost(sync, i, o)
            rows.append(dict(_demo=True, run=rd.parent.name, stamp=rd.name, matrix=cf.parent.parent.name, cell=d["label"], gpu=f"1x{snap['gpu']}", workload=d["args"]["workload"],
                             mode=s["mode"], C=s["concurrency"], rate=s["offered_rate"], in_mean=i, cached_mean=0, out_per_req=o, req_s=rps, out_tok_s=s["output_tok_per_s"],
                             usd_h=usd_h, our_usd_req=our, openrelay_sync=b, openrelay_sync_cached=b, openrelay_batch=cost(batch, i, o),
                             openrouter=(i * 0.10 + o * 0.34) / 1e6, openrouter_who="DEMO (unknown)", usd_per_m_out_all=usd_h / 3600 / s["output_tok_per_s"] * 1e6,
                             file=str(cf.relative_to(ROOT)), ratio_sync=our / b if our else None, ratio_sync_cached=our / b if our else None,
                             ratio_batch=our / cost(batch, i, o) if our else None, ratio_openrouter=our / ((i * 0.10 + o * 0.34) / 1e6) if our else None))
    (OUT / "cost_rows.json").write_text(json.dumps(rows, indent=1))


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    L = []
    t = T0
    graphs = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    # RTX PRO 6000: the cheap winner candidate
    r = Run("e4b_rtxpro6000_vllm.json", t, L)
    r.phase("create", 21)
    be = pph("RTX-PRO-6000")["total"] / 3600 / 0.000779
    r.server("nvfp4", [dict(label=f"agentic_r{x}", mode="open", load=round(be * x, 2), in_tok=9500, out_tok=380, cap_rps=be * 1.45, ttft0=0.55, tpot0=0.019) for x in (0.75, 1.0, 1.25)]
             + [dict(label=f"s1k1k__concurrency={c}", mode="closed", load=c, in_tok=1024, out_tok=1024, cap_rps=8, ttft0=0.05, tpot0=0.012, window=60) for c in (1, 16, 64, 128)],
             protocol=(be, be * 1.45, 0.55, 0.019), eval_acc=0.967, kv=310000, graphs=graphs, spec=None)
    r.server("fp8dyn", [dict(label=f"agentic_r{x}", mode="open", load=round(be * x, 2), in_tok=9500, out_tok=380, cap_rps=be * 1.1, ttft0=0.7, tpot0=0.024) for x in (0.75, 1.0, 1.25)]
             + [dict(label=f"s1k1k__concurrency={c}", mode="closed", load=c, in_tok=1024, out_tok=1024, cap_rps=6.5, ttft0=0.06, tpot0=0.015, window=60) for c in (1, 16, 64, 128)],
             eval_acc=0.983, kv=190000, graphs=graphs[:-2])
    t = r.finish() + 300
    # H100 spec decode
    r = Run("e2b_h100_vllm_fp8_specdec.json", t, L)
    r.phase("create", 14)
    for srv, k in (("nospec", None), ("mtp2", 2), ("mtp4", 4)):
        r.server(srv, [dict(label=f"chat__concurrency={c}", mode="closed", load=c, in_tok=180, out_tok=420, cap_rps=14, ttft0=0.04, tpot0=0.016, spec=k, workload="chat_natural", window=60) for c in (1, 8, 32, 64)],
                 eval_acc=0.975 if k != 4 else 0.97, kv=420000, graphs=graphs, spec=k)
    t = r.finish() + 200
    # MI355X calibration vs OpenRelay
    r = Run("e4d_mi355x_vllm.json", t, L)
    r.phase("create", 30)
    be = pph("MI355X")["total"] / 3600 / 0.000779
    r.server("mxfp4", [dict(label="agentic_r1.0", mode="open", load=round(be, 2), in_tok=9500, out_tok=380, cap_rps=2.6, ttft0=0.5, tpot0=0.026)], protocol=(be, 2.6, 0.5, 0.026),
             eval_acc=0.987, kv=1400000, graphs=graphs, spec=2, attn={"sliding": "TRITON_ATTN", "global_512": "TRITON_ATTN"})
    t = r.finish() + 600
    # RTX 5090: evicted mid-run
    r = Run("e4e_rtx5090_vllm_awq.json", t, L, evict=True)
    r.phase("failed_create", 40)
    r.phase("create", 25)
    be = pph("RTX-5090")["total"] / 3600 / 0.000779
    r.server("awq", [dict(label="agentic_r1.0", mode="open", load=round(be, 2), in_tok=9500, out_tok=380, cap_rps=be * 0.8, ttft0=1.1, tpot0=0.04)], kv=None, graphs=graphs[:8])
    t = r.finish() + 100
    # spend reconciliation events + a long tail of extra sandboxes so the spend curve crosses milestones
    for i in range(14):
        sid = f"demo-sweep-{i:02d}"
        gpu = R.choice(list(GPU_H))
        p = pph(gpu)
        L.append(dict(t=iso(t), event="create", id=sid, run=f"demo_sweep_{i}", gpu=gpu, gpus=1, cpu=8, mem=48, disk=150, spot=True, usd_per_h=p, ttl_min=180, deadline=int(t + 10800), _demo=True))
        t += R.uniform(1.2, 3.0) * 3600
        L.append(dict(t=iso(t), event="delete", id=sid, why="demo", _demo=True))
        if i % 4 == 3:
            L.append(dict(t=iso(t + 60), event="spend", est_total_usd=None, metered_total_usd=None, metered_file=None, _demo=True))
    L.sort(key=lambda r: r["t"])
    # running estimate -> spend events (fake meter ~3% under) and milestone events at first crossings
    live, tot, last, crossed = {}, 0.0, None, set()
    for r in list(L):
        tt = calendar.timegm(time.strptime(r["t"], "%Y-%m-%dT%H:%M:%SZ"))
        if last is not None:
            tot += sum(rate for rate in live.values()) * (tt - last) / 3600
        last = tt
        if r["event"] == "create":
            live[r["id"]] = r["usd_per_h"]["total"]
        elif r["event"] == "delete":
            live.pop(r["id"], None)
        elif r["event"] == "spend":
            r.update(est_total_usd=round(tot, 3), metered_total_usd=round(tot * 0.97, 3), metered_file="infra/spend_DEMO.json")
        for th in (50, 100, 200, 300, 400):
            if tot >= th and th not in crossed:
                crossed.add(th)
                L.append(dict(t=r["t"], event="milestone", threshold_usd=th, total_usd=round(tot, 3), basis="estimate", _demo=True))
    L.sort(key=lambda r: r["t"])
    write_cost_rows()
    write_experiments()
    write_claims()
    LEDGER.write_text("\n".join(json.dumps(x) for x in L) + "\n")
    print(f"wrote {OUT} and {LEDGER} ({len(L)} ledger events). MOCK DATA.")


if __name__ == "__main__":
    main()
