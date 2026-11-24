#!/usr/bin/env python3
"""OpenRelay's benchmark protocol, copied from their post "Gemma 4 Optimization with Agentic Kernel Engineering"
(https://openrelay.inc/blog/gemma-4-agentic-kernel-engineering, 2026-09-29) and recipe repo
(https://github.com/OpenRelayInc/inference-recipes/tree/main/gemma-4-31b-mi355x):
  open-loop Poisson arrivals, 300 s per offered rate, seeds 7 and 8, the WORSE seed is kept; SLO p95 TTFT < 4 s with
  median TPOT reported; requests 9.5K input / 380 output tokens (cold: unique prompts); prefill measured as 16
  concurrent 10K-token prompts.
Rates are a ladder of multiples of the box's break-even rate against OpenRelay's sync list price.

Definitions (docs/DATA_CONTRACT.md §5):
  slo_pass       p95 TTFT < slo AND errors < 1% of requests AND no client-side drops (steady window); achieved and
                 actually-arrived rates are reported beside it
  goodput_rps    completed requests per second in the steady window, if slo_pass, else 0
  kept seed      the seed with the lower goodput (tie: the higher p95 TTFT)
  max_rate_at_slo_rps   the highest offered rate whose kept seed passes: the headline req/s per GPU
Early stop: once a rate's kept seed fails with p95 TTFT > 2 x SLO, higher rates are skipped (they only queue harder).

  python3 bench/protocol.py --model gemma --breakeven-rps 1.31 --out out/protocol_fp8 [--multiples 0.6,0.9,1.2,1.6,2.0]
      [--seeds 7,8] [--duration 300] [--prefill-shot] [--metrics URL] [--max-minutes M]
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traffic as T  # noqa: E402
import workloads as Wl  # noqa: E402


def targs(a, extra):
    argv = ["--model", a.model, "--base", a.base, "--workload", a.workload, "--params", a.params,
            "--cpt", str(a.cpt), "--cpt-table", str(a.cpt_table)] + (["--metrics", a.metrics] if a.metrics else []) + \
        (["--tokenizer", a.tokenizer] if a.tokenizer else []) + ["--ttft-basis", a.ttft_basis] + extra
    return T.finish_args(T.parser().parse_args(argv))


SCORED_S = 300.0    # every claim cell is scored over a 300 s window (Grok review of the C1 claim, 2026-11-22)


def _pct(xs, q):
    """linear-interpolated percentile; an infinite sample anywhere it is used makes the result infinite."""
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    if xs[f] == float("inf") or (c != f and xs[c] == float("inf") and k > f):
        return float("inf")
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def score_cell(reqs, duration, slo, scored_s=SCORED_S):
    """THE rule for a traffic.py cell (run time and re-scoring of saved cells): score the LAST `scored_s` seconds of the cell
    (a 360 s cell with 60 s warmup -> 60..360; an older 300 s cell -> 0..300, warmup included). Every request sent in the window
    counts. A request with no first token by the window end is censored: for pass/fail its TTFT is +inf (conservative); its lower
    bound (end - send) is reported as ttft_p95_lower_s. achieved = requests finishing inside the window / window length."""
    hi = float(duration)
    lo = max(0.0, hi - scored_s)
    rs = [r for r in reqs if not r.get("dropped") and r.get("t_send") is not None and lo <= r["t_send"] <= hi]
    t_hi, t_lo, cens = [], [], 0
    for r in rs:
        if r.get("ttft") is not None:
            t_hi.append(r["ttft"])
            t_lo.append(r["ttft"])
        else:
            cens += 1
            t_hi.append(float("inf"))
            t_lo.append(max(0.0, hi - r["t_send"]))
    done = [r for r in reqs if r.get("e2e") is not None and not r.get("aborted") and r.get("t_send") is not None
            and lo <= r["t_send"] + r["e2e"] <= hi]
    errors = sum(1 for r in rs if r.get("status") not in (200, None))
    dropped = sum(1 for r in reqs if r.get("dropped") and r.get("t_sched") is not None and lo <= r["t_sched"] <= hi)
    p95 = _pct(t_hi, 0.95)
    ok = p95 is not None and p95 < slo and rs and errors / len(rs) < 0.01 and not dropped
    return dict(scored_window_s=hi - lo, scored_from_s=lo, requests_in_window=len(rs), ttft_censored=cens,
                ttft_p95_s=p95, ttft_p95_lower_s=_pct(t_lo, 0.95), achieved_rps=len(done) / (hi - lo) if hi > lo else None,
                errors=errors, dropped=dropped, slo_pass=bool(ok))


XCHECK_TOL = 0.01   # brainstorm round 2: server Prometheus prompt tok/s must be within 1% of what the client counted


def server_deltas(m0, m1):
    """vLLM Prometheus counter deltas between two traffic.scrape() results: prompt / generation tok/s, preemptions and MTP
    (spec_decode_*) acceptance. None when either scrape failed."""
    if not m0 or not m1 or "error" in m0 or "error" in m1:
        return None
    dtm = max(1e-9, m1["t"] - m0["t"])
    d = dict(window_s=dtm, prompt_tokens=m1["prompt"] - m0["prompt"], gen_tokens=m1["gen"] - m0["gen"],
             prompt_tok_per_s=(m1["prompt"] - m0["prompt"]) / dtm, gen_tok_per_s=(m1["gen"] - m0["gen"]) / dtm,
             preemptions=m1["preempt"] - m0["preempt"], counters=m1["found"])
    if m1["spec_draft"] - m0["spec_draft"] > 0:
        d["spec_acceptance_rate"] = (m1["spec_acc"] - m0["spec_acc"]) / (m1["spec_draft"] - m0["spec_draft"])
        if m1["spec_drafts"] - m0["spec_drafts"] > 0:
            d["spec_mean_accepted_per_step"] = 1 + (m1["spec_acc"] - m0["spec_acc"]) / (m1["spec_drafts"] - m0["spec_drafts"])
    return d


def xcheck(server_val, client_val):
    """ratio server/client and pass flag (|ratio - 1| <= 1%); None when either side is missing."""
    if not server_val or not client_val:
        return None, None
    r = server_val / client_val
    return r, abs(r - 1) <= XCHECK_TOL


def judge(s, offered, slo, reqs=None, win=None):
    """SLO = OpenRelay's: p95 TTFT < slo; plus errors < 1% and no client-side drops. A server that cannot keep up shows
    it as queueing in TTFT, so achieved-vs-offered is reported (with the real Poisson arrival rate) but not a criterion:
    in a short window the arrival count alone varies by ~1/sqrt(N)."""
    sent = max(1, s["requests_sent"])
    ttft95 = (s.get("ttft_s") or {}).get("p95")
    achieved = s["completed_in_window"] / s["window_s"] if s["window_s"] else 0.0
    arrived = None
    if reqs is not None and win:
        arrived = sum(1 for r in reqs if r.get("t_sched") is not None and win[0] <= r["t_sched"] <= win[1]) / max(1e-9, win[1] - win[0])
    ok = (ttft95 is not None and ttft95 < slo and s["errors"] / sent < 0.01 and not s["dropped"])
    return dict(achieved_rps=achieved, arrived_rps=arrived, goodput_rps=achieved if ok else 0.0, ttft_p50_s=(s.get("ttft_s") or {}).get("p50"),
                ttft_p95_s=ttft95, ttft_p99_s=(s.get("ttft_s") or {}).get("p99"), tpot_p50_s=(s.get("tpot_s") or {}).get("p50"),
                tpot_p95_s=(s.get("tpot_s") or {}).get("p95"), e2e_p50_s=(s.get("e2e_s") or {}).get("p50"),
                errors=s["errors"], requests_sent=s["requests_sent"], dropped=s["dropped"], slo_pass=ok,
                out_tok_s=s["output_tok_per_s"], in_tok_s=s["input_tok_per_s"], cached_in_tok_s=s["cached_input_tok_per_s"],
                server_gen_tok_s=(s.get("server_metrics") or {}).get("gen_tok_per_s"),
                spec_accept_len=(s.get("server_metrics") or {}).get("spec_mean_accepted_per_step"),
                preemptions=(s.get("server_metrics") or {}).get("preemptions"), peak_inflight=s["peak_inflight"],
                spec_acceptance_rate=(s.get("server_metrics") or {}).get("spec_acceptance_rate"),
                server_prompt_tok_s=(s.get("server_metrics") or {}).get("prompt_tok_per_s"),
                prompt_tok_window_ratio=(((s.get("server_metrics") or {}).get("prompt_tok_per_s") or 0) / s["input_tok_per_s"])
                if s.get("input_tok_per_s") else None,
                **(_xfields((s.get("server_totals") or {}).get("server_prompt_tokens"), (s.get("server_totals") or {}).get("client_prompt_tokens"),
                            "whole cell: server counter before the first request vs after all finished and settled, vs the sum of "
                            "prompt tokens in the completed responses (the window ratio is informational: engines count at "
                            "different moments)")
                   if s.get("server_totals") else
                   _xfields((s.get("server_metrics") or {}).get("prompt_tok_per_s"), s["input_tok_per_s"],
                            "steady window (no whole-cell scrape recorded)")))


def _xfields(server_val, client_val, basis):
    r, ok = xcheck(server_val, client_val)
    return dict(prompt_tok_xcheck_ratio=r, prompt_tok_xcheck_pass=ok, prompt_tok_xcheck_basis=basis)


def run_cell(a, extra, path, label):
    ns = targs(a, extra + ["--label", label, "--out", path])
    d = asyncio.run(T.run(ns))
    d["schema"] = "traffic/1"
    json.dump(d, open(path, "w"), indent=1)
    print(T.line(d), flush=True)
    return d


def vllm_bench_cell(a, rate, seed, path):
    """OpenRelay's exact client (their bench/bench.sh @6767b4d): `vllm bench serve` on /v1/completions, random tokens,
    input 9500 x U(0.2..1.8) (range ratio 0.8), output 380 x U(0.2..1.8), num_prompts = rate x DUR, Poisson at `rate`,
    the given seed; the whole run counts (no steady window). Returns a summary in judge()'s field names."""
    n = int(rate * a.duration)
    d, f = os.path.dirname(path), os.path.basename(path)
    cmd = [a.vllm_bin, "bench", "serve", "--backend", "openai", "--base-url", a.base, "--endpoint", "/v1/completions", "--model", a.model,
           "--tokenizer", a.tokenizer, "--dataset-name", "random", "--random-input-len", "9500", "--random-range-ratio", "0.8",
           "--random-output-len", "380", "--num-prompts", str(n), "--request-rate", str(rate), "--seed", str(seed),
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99", "--save-result", "--result-dir", d,
           "--result-filename", f]
    m0 = T.scrape(a.metrics) if a.metrics else None
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.duration * 4 + 600)
    m1 = T.scrape(a.metrics) if a.metrics else None
    sd = server_deltas(m0, m1)
    open(path + ".log", "w").write(r.stdout[-20000:] + r.stderr[-5000:])
    j = json.load(open(path)) if os.path.exists(path) else {}
    side = {k: j[k] for k in ("latencies", "ttfts", "itls", "input_lens", "output_lens", "start_times", "errors")
            if k in j and isinstance(j[k], list)}
    if side:   # the full per-request arrays, so p95s can be rebuilt from samples (Grok review nit 5); itls can be large
        json.dump(side, open(path.replace(".json", ".per_request.json"), "w"))
    for k in ("latencies", "queue_times", "ttfts", "itls", "generated_texts", "errors", "input_lens", "output_lens"):
        if k in j and isinstance(j[k], list) and len(j[k]) > 50:
            j[k] = j[k][:0]   # keep the summary file small; samples are in the .per_request.json sidecar
    j["openrelay_cmd"] = " ".join(cmd)
    j["wall_s"] = time.time() - t0
    j["server_metrics"] = sd if sd else dict(error="no scrape" if not a.metrics else ((m0 or {}).get("error") or (m1 or {}).get("error")))
    json.dump(j, open(path, "w"), indent=1)
    comp, fail = j.get("completed", 0), j.get("failed", 0)
    p95 = (j.get("p95_ttft_ms") or 0) / 1000 if j.get("p95_ttft_ms") is not None else None
    ok = p95 is not None and p95 < a.slo_ttft_p95 and comp > 0 and fail / max(1, comp + fail) < 0.01
    print(f"[vllmbench r={rate} s={seed}] done={comp} failed={fail} req/s={j.get('request_throughput')} p95 TTFT={p95} "
          f"median TPOT={j.get('median_tpot_ms')} ms in tok/s={j.get('total_input_tokens', 0) / max(1e-9, j.get('duration') or 1):.0f}", flush=True)
    return dict(client="vllm bench serve (OpenRelay's bench.sh)", achieved_rps=j.get("request_throughput"), arrived_rps=n / a.duration,
                goodput_rps=(j.get("request_throughput") or 0) if ok else 0.0, ttft_p50_s=(j.get("median_ttft_ms") or 0) / 1000,
                ttft_p95_s=p95, ttft_p99_s=(j.get("p99_ttft_ms") or 0) / 1000, tpot_p50_s=(j.get("median_tpot_ms") or 0) / 1000,
                tpot_p95_s=(j.get("p95_tpot_ms") or 0) / 1000, e2e_p50_s=(j.get("median_e2el_ms") or 0) / 1000, errors=fail,
                requests_sent=comp + fail, dropped=0, slo_pass=ok, out_tok_s=j.get("output_throughput"),
                in_tok_s=(j.get("total_input_tokens") or 0) / max(1e-9, j.get("duration") or 1), cached_in_tok_s=None,
                server_gen_tok_s=(sd or {}).get("gen_tok_per_s"), spec_accept_len=(sd or {}).get("spec_mean_accepted_per_step"),
                spec_acceptance_rate=(sd or {}).get("spec_acceptance_rate"), preemptions=(sd or {}).get("preemptions"),
                peak_inflight=j.get("max_concurrent_requests"), duration_s=j.get("duration"), rc=r.returncode,
                server_prompt_tokens=(sd or {}).get("prompt_tokens"), client_input_tokens=j.get("total_input_tokens"),
                prompt_tokens_mean=(j.get("total_input_tokens") or 0) / comp if comp else None,
                completion_tokens_mean=(j.get("total_output_tokens") or 0) / comp if comp else None,
                **_xfields((sd or {}).get("prompt_tokens"), j.get("total_input_tokens"),
                           "whole bench run: server prompt-token counter delta (scraped before/after) vs vllm bench total_input_tokens"))


def vllm_bench_prefill(a, path):
    """OpenRelay's prefill probe (their probe.sh): closed loop, 16 concurrent, 64 prompts of 10,000 random tokens,
    1 output token; run twice, the second counts. Prefill tok/s = vLLM bench's input token throughput."""
    out = []
    for k in (1, 2):
        pth = path.replace(".json", f"_run{k}.json")
        cmd = [a.vllm_bin, "bench", "serve", "--backend", "openai", "--base-url", a.base, "--endpoint", "/v1/completions", "--model", a.model,
               "--tokenizer", a.tokenizer, "--dataset-name", "random", "--random-input-len", str(a.prefill_tokens), "--random-range-ratio", "0",
               "--random-output-len", "1", "--num-prompts", str(4 * a.prefill_c), "--max-concurrency", str(a.prefill_c), "--seed", str(100 + k),
               "--percentile-metrics", "ttft,e2el", "--metric-percentiles", "50,95", "--save-result", "--result-dir", os.path.dirname(pth),
               "--result-filename", os.path.basename(pth)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        j = json.load(open(pth)) if os.path.exists(pth) else {}
        for key in ("latencies", "queue_times", "ttfts", "itls", "generated_texts", "errors", "input_lens", "output_lens"):
            j.pop(key, None)
        json.dump(j, open(pth, "w"), indent=1)
        out.append(dict(run=k, file=os.path.basename(pth), input_tok_s=(j.get("total_input_tokens") or 0) / max(1e-9, j.get("duration") or 1),
                        completed=j.get("completed"), p95_ttft_s=(j.get("p95_ttft_ms") or 0) / 1000, rc=r.returncode))
    return dict(client="vllm bench serve (OpenRelay's probe.sh)", concurrency=a.prefill_c, prompt_tokens_target=a.prefill_tokens,
                runs=out, prefill_tok_s=out[-1]["input_tok_s"], counted="second run")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--metrics", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--breakeven-rps", type=float, required=True)
    ap.add_argument("--multiples", default="0.6,0.9,1.2,1.6,2.0")
    ap.add_argument("--rates", default="", help="explicit offered rates (req/s); overrides --multiples (multiple = rate / break-even)")
    ap.add_argument("--seeds", default="7,8")
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--warmup", type=float, default=None, help="excluded from the steady window (default min(60, 20%%))")
    ap.add_argument("--slo-ttft-p95", type=float, default=4.0)
    ap.add_argument("--workload", default="synthetic")
    ap.add_argument("--params", default=json.dumps({"input_dist": {"uniform": [1900, 17100]}, "output_dist": {"uniform": [76, 684]}}),
                    help="OpenRelay's shape: input uniform 1.9K..17.1K (mean 9.5K), output uniform 76..684 (mean 380)")
    ap.add_argument("--max-inflight", type=int, default=1024)
    ap.add_argument("--prefill-shot", action="store_true")
    ap.add_argument("--prefill-c", type=int, default=16)
    ap.add_argument("--prefill-tokens", type=int, default=10000)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--client", choices=["traffic", "vllmbench"], default="traffic",
                    help="traffic = our steady-window client (any engine); vllmbench = OpenRelay's exact `vllm bench serve` command")
    ap.add_argument("--tokenizer", default="", help="model dir (vllmbench needs the tokenizer to make random-token prompts)")
    ap.add_argument("--ttft-basis", choices=["text", "token"], default="text",
                    help="traffic client only: what counts as the first token (bench/traffic.py --ttft-basis)")
    ap.add_argument("--vllm-bin", default=os.environ.get("OR_VLLM_BIN", "vllm"),
                    help="the `vllm` executable for OpenRelay's client; on SGLang images a client-only venv (bench/setup_vllm_client.sh)")
    ap.add_argument("--poisson-count", action="store_true",
                    help="traffic client: let the Poisson arrival COUNT vary (default: exactly rate x duration, like vllm bench serve)")
    ap.add_argument("--stratify", action="store_true",
                    help="traffic client: stratified uniform lengths (realized means = design means; needs exact arrivals)")
    ap.add_argument("--column", choices=["A", "B", "C"], default="",
                    help="C = supplementary cached traffic (shared prefixes / agent sessions), NOT OpenRelay's method; "
                         "default: A for vllmbench, B for traffic")
    ap.add_argument("--group-tag", default="", help="appended to the claim group (one server, several column-C workloads)")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--config-id", default="")
    a = ap.parse_args()
    a.cpt, a.cpt_table = 4.0, 3.0
    os.makedirs(os.path.join(a.out, "cells"), exist_ok=True)
    t0 = time.time()
    if a.client == "vllmbench":
        a.no_calibrate = True
        assert a.tokenizer, "--client vllmbench needs --tokenizer <model dir>"
        if os.path.isabs(a.vllm_bin) and not os.path.exists(a.vllm_bin):
            # the client-only venv is missing (e.g. the server that ran the setup step was skipped or failed): build it now
            here = os.path.dirname(os.path.abspath(__file__))
            r = subprocess.run(["bash", os.path.join(here, "setup_vllm_client.sh")], capture_output=True, text=True, timeout=1200)
            print((r.stdout + r.stderr)[-600:], flush=True)
            assert os.path.exists(a.vllm_bin), f"OpenRelay's client not available at {a.vllm_bin}"
    if a.tokenizer and a.client == "traffic":
        cal = dict(tokenizer=a.tokenizer, exact_prose_tokens=True)
    elif not a.no_calibrate:
        cal = T.calibrate(targs(a, ["--out", "/dev/null"]), Wl.Text())
        a.cpt, a.cpt_table = cal["cpt_prose"], cal["cpt_table"]
    else:
        cal = dict(cpt_prose=a.cpt, cpt_table=a.cpt_table, given=True)
    warm = a.warmup if a.warmup is not None else min(60.0, a.duration * 0.2)
    summ = dict(schema="protocol/1", run_id=a.run_id or None, config_id=a.config_id or None,
                source="https://openrelay.inc/blog/gemma-4-agentic-kernel-engineering (2026-09-29); "
                       "https://github.com/OpenRelayInc/inference-recipes/tree/main/gemma-4-31b-mi355x",
                slo=dict(ttft_p95_s=a.slo_ttft_p95), breakeven_rps=a.breakeven_rps, duration_s=a.duration, warmup_s=warm,
                seeds=[int(x) for x in a.seeds.split(",")], kept="worse", workload=a.workload, params=json.loads(a.params),
                calibration=cal, definitions=(
                    "goodput_rps: completed req/s in the steady window of one (rate, seed) cell when that cell meets the SLO "
                    "(p95 TTFT < slo, errors < 1%, no drops), else 0. max_rate_at_slo_rps: the highest offered "
                    "rate whose worse seed meets the SLO."),
                ladder=[], prefill=None, t_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path_s = os.path.join(a.out, "protocol_summary.json")

    def save():
        json.dump(summ, open(path_s, "w"), indent=1)

    summ["client"] = a.client
    summ["ttft_basis"] = a.ttft_basis if a.client == "traffic" else None
    summ["column"] = a.column or ("A" if a.client == "vllmbench" else "B")
    summ["workload"] = a.workload if a.client == "traffic" else "vllm bench random"
    summ["group_tag"] = a.group_tag or None
    summ["params"] = json.loads(a.params) if a.client == "traffic" else None
    if a.prefill_shot and a.client == "vllmbench":
        summ["prefill"] = vllm_bench_prefill(a, os.path.join(a.out, "cells", f"prefill_c{a.prefill_c}_{a.prefill_tokens}.json"))
        save()
    elif a.prefill_shot:
        p = os.path.join(a.out, "cells", f"prefill_c{a.prefill_c}_{a.prefill_tokens}.json")
        d = run_cell(a, ["--workload", "synthetic", "--params", json.dumps({"input_tokens": a.prefill_tokens, "output_tokens": 1}),
                         "--concurrency", str(a.prefill_c), "--max-requests", str(a.prefill_c), "--no-stagger"], p, "prefill_shot")
        rq = [r for r in d["requests"] if r["status"] == 200 and r["ttft"] is not None]
        if rq:
            span = max(r["t_send"] + r["ttft"] for r in rq) - min(r["t_send"] for r in rq)
            summ["prefill"] = dict(concurrency=a.prefill_c, prompt_tokens_target=a.prefill_tokens, requests=len(rq),
                                   prompt_tokens_mean=sum(r["prompt_tokens"] or 0 for r in rq) / len(rq),
                                   prefill_tok_s=sum(r["prompt_tokens"] or 0 for r in rq) / span if span > 0 else None,
                                   span_s=span, ttft_max_s=max(r["ttft"] for r in rq), file=os.path.relpath(p, a.out))
        save()
    stop = False
    ladder = ([(float(r) / a.breakeven_rps, float(r)) for r in a.rates.split(",")] if a.rates
              else [(float(m), round(float(m) * a.breakeven_rps, 4)) for m in a.multiples.split(",")])
    for m, rate in ladder:
        m = round(m, 3)
        rung = dict(multiple=m, rate=rate, seeds={}, results={})
        if stop:
            rung["skipped"] = "past SLO (an earlier rate failed with p95 TTFT > 2x SLO)"
            summ["ladder"].append(rung)
            continue
        for sd in summ["seeds"]:
            if a.max_minutes and (time.time() - t0 + a.duration + 20) / 60 > a.max_minutes:
                rung["skipped"] = "time budget"
                break
            p = os.path.join(a.out, "cells", f"r{m:g}x_{rate:g}rps_s{sd}.json")
            if a.client == "vllmbench":
                res = vllm_bench_cell(a, rate, sd, p)
                if res.get("rc") == 0 and res.get("requests_sent"):
                    rung["seeds"][str(sd)] = os.path.relpath(p, a.out)
                    rung["results"][str(sd)] = res
                    continue
                print(f"[protocol] vllm bench failed (rc={res.get('rc')}); falling back to our client for this cell", flush=True)
                p = p.replace(".json", "_traffic.json")
            cell_s = a.duration + warm          # e.g. 300 + 60: the scored window (score_cell) is the last a.duration seconds
            d = run_cell(a, ["--rate", str(rate), "--seed", str(sd), "--duration", str(cell_s), "--warmup", str(warm),
                             "--max-inflight", str(a.max_inflight)] + ([] if a.poisson_count else ["--exact-arrivals"]) +
                         (["--params", json.dumps(dict(json.loads(a.params), stratify=max(1, round(rate * cell_s))))]
                          if a.stratify else []), p, f"r{m:g}x_s{sd}")
            rung["seeds"][str(sd)] = os.path.relpath(p, a.out)
            w0 = d["args"]["warmup"]
            res = judge(d["summary"], rate, a.slo_ttft_p95, d["requests"], (w0, w0 + d["summary"]["window_s"]))
            sc = score_cell(d["requests"], d["args"]["duration"], a.slo_ttft_p95, scored_s=a.duration)
            res.update({f"window_{k}": res.get(k) for k in ("achieved_rps", "ttft_p95_s", "slo_pass", "goodput_rps")})
            res.update(sc, goodput_rps=sc["achieved_rps"] if sc["slo_pass"] else 0.0)
            rung["results"][str(sd)] = res
            rung["results"][str(sd)]["client"] = "traffic.py (steady window)" + (" FALLBACK" if a.client == "vllmbench" else "")
            okr = [x for x in d["requests"] if x.get("prompt_tokens")]
            rung["results"][str(sd)]["prompt_tokens_mean"] = (sum(x["prompt_tokens"] for x in okr) / len(okr)) if okr else None
            rung["results"][str(sd)]["completion_tokens_mean"] = (sum(x["completion_tokens"] or 0 for x in okr) / len(okr)) if okr else None
            # prompt-cache reuse, two independent reads: what the responses said (usage.prompt_tokens_details.cached_tokens)
            # over the whole cell and over the scored window, and the server's own prefix-cache counters (window)
            okc = [x for x in okr if x.get("cached_tokens") is not None]
            lo_s = d["args"]["duration"] - a.duration
            okw = [x for x in okc if x.get("t_send") is not None and x["t_send"] >= lo_s]
            rung["results"][str(sd)].update(
                cached_tokens_mean=(sum(x["cached_tokens"] for x in okc) / len(okc)) if okc else None,
                cached_share_cell=(sum(x["cached_tokens"] for x in okc) / max(1, sum(x["prompt_tokens"] for x in okc))) if okc else None,
                cached_share_scored=(sum(x["cached_tokens"] for x in okw) / max(1, sum(x["prompt_tokens"] for x in okw))) if okw else None,
                server_prefix_cache_hit_ratio=(d["summary"].get("server_metrics") or {}).get("prefix_cache_hit_ratio"),
                sessions=len({x.get("session") for x in d["requests"] if x.get("session") is not None}) or None)
            rung["results"][str(sd)]["exact_arrivals"] = not a.poisson_count
            rung["results"][str(sd)]["stratified_lengths"] = bool(a.stratify)
        if rung["results"]:
            kept = min(rung["results"].items(), key=lambda kv: (kv[1]["goodput_rps"], -(kv[1]["ttft_p95_s"] or 1e9)))
            rung["kept_seed"] = int(kept[0])
            rung.update({k: v for k, v in kept[1].items()})
            if not kept[1]["slo_pass"] and (kept[1]["ttft_p95_s"] or 1e9) > 2 * a.slo_ttft_p95:
                stop = True
        summ["ladder"].append(rung)
        passed = [r for r in summ["ladder"] if r.get("slo_pass")]
        summ["max_rate_at_slo_rps"] = max((r["rate"] for r in passed), default=0.0)
        summ["max_goodput_rps"] = max((r["goodput_rps"] for r in passed), default=0.0)
        save()
    summ["t_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summ["wall_minutes"] = (time.time() - t0) / 60
    save()
    print(f"[protocol] max_rate_at_slo={summ.get('max_rate_at_slo_rps')} req/s (break-even {a.breakeven_rps}); "
          f"prefill={(summ.get('prefill') or {}).get('prefill_tok_s')} tok/s -> {path_s}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
