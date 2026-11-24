#!/usr/bin/env python3
"""Sustained-traffic generator for any OpenAI-compatible chat endpoint (vLLM, SGLang, llama.cpp, TensorRT-LLM).
Stdlib only, so the SAME client runs inside every engine's container (next to the server: latencies are the server's).

Traffic modes
  --concurrency C           closed loop: C requests always in flight (a new one starts when one ends)
  --rate R                  open loop: Poisson arrivals at R req/s (seeded), independent of how fast the server is;
                            --max-inflight caps client-side concurrency (requests over the cap are counted as dropped)
Workloads (bench/workloads.py): synthetic (input_tokens, output_tokens, prefix_ratio, prefix_pool), ciu_screen,
fin_extract, console_chat (multi-turn, shared 10k system prompt), chat_natural, longdoc_qa, mixed. Params via --params JSON.

Measurement: steady window [t0 + warmup, t0 + duration]. New requests stop at the window end and in-flight ones are cut,
so a slow tail does not dilute throughput; tokens are counted from per-chunk timestamps inside the window. Every
request streams with usage, so TTFT, TPOT (decode time per output token), ITL (gap between chunks), E2E, prompt and
cached tokens (usage.prompt_tokens_details.cached_tokens, when the server reports it) are exact per request.
Requests whose shared prefix was used for the first time in this run are "cold", repeats are "warm": TTFT is reported
for both. With --metrics, the server's own Prometheus counters are read at both window edges as a cross-check.

  python3 bench/traffic.py --model gemma --workload synthetic --params '{"input_tokens":1024,"output_tokens":1024}' \
      --concurrency 64 --duration 90 --out out/x.json [--calibrate] [--metrics http://127.0.0.1:8000/metrics]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
import urllib.parse
import uuid
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workloads as W  # noqa: E402

PCTS = (50, 90, 95, 99)


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def dist(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    d = dict(n=len(xs), mean=statistics.fmean(xs), min=min(xs), max=max(xs))
    d.update({f"p{p}": pct(xs, p) for p in PCTS})
    return d


# ----------------------------------------------------------------------------- server metrics
COUNTERS = {"gen": ("generation_tokens_total", "tokens_predicted_total"), "prompt": ("prompt_tokens_total",),
            "pc_queries": ("prefix_cache_queries_total",), "pc_hits": ("prefix_cache_hits_total",),
            "preempt": ("num_preemptions_total",), "req_ok": ("request_success_total",),
            "spec_acc": ("spec_decode_num_accepted_tokens_total",), "spec_draft": ("spec_decode_num_draft_tokens_total",),
            "spec_drafts": ("spec_decode_num_drafts_total",)}


def scrape(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            txt = r.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        return dict(error=repr(e)[:200])
    out = {k: 0.0 for k in COUNTERS} | dict(t=time.time(), found=set())
    for line in txt.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name = line.split("{")[0].split(" ")[0]
        try:
            v = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
        for k, suffixes in COUNTERS.items():
            if name.endswith(suffixes):
                out[k] += v
                out["found"].add(name)
    out["found"] = sorted(out["found"])
    return out


# ----------------------------------------------------------------------------- one streaming request
class Req:
    __slots__ = ("i", "spec", "t_sched", "t_send", "t_first", "t_last", "t_tok", "chunks", "prompt_tokens", "cached_tokens",
                 "completion_tokens", "status", "error", "finish", "aborted", "cold", "dropped", "text")

    def __init__(self, i, spec, cold):
        self.i, self.spec, self.cold = i, spec, cold
        self.t_sched = self.t_send = self.t_first = self.t_last = None
        self.t_tok = None        # first chunk that carries a token (text, or usage.completion_tokens >= 1): every basis records it
        self.chunks = []
        self.prompt_tokens = self.cached_tokens = self.completion_tokens = None
        self.status = self.error = self.finish = None
        self.aborted = self.dropped = False
        self.text = []           # reply text, kept only when spec["keep_text"] (agent sessions re-send the real answer)


def body_for(a, spec):
    b = dict(model=a.model, messages=spec["messages"], stream=True, max_tokens=spec["max_tokens"], temperature=a.temperature,
             stream_options=dict(include_usage=True, continuous_usage_stats=True))
    if spec.get("ignore_eos"):
        b["ignore_eos"] = True
        b["min_tokens"] = spec["max_tokens"]
    if a.no_think:
        b["chat_template_kwargs"] = dict(enable_thinking=False)
    if a.extra:
        b.update(json.loads(a.extra))
    return b


async def send(host, port, path, body, r: Req, timeout_s, token_basis=False):
    """token_basis: a choice chunk whose continuous usage says >= 1 completion token counts as a token arrival even with no
    text (Inkling's first token is a special token the server swallows). Default: text chunks only, as every Gemma run."""
    data = json.dumps(body).encode()
    head = (f"POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\n"
            f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n").encode()
    reader, writer = await asyncio.open_connection(host, port, limit=2 ** 24)
    try:
        r.t_send = time.perf_counter()
        writer.write(head + data)
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout_s)
        r.status = int(line.split()[1]) if line else -1
        hdr = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            k, _, v = line.decode(errors="replace").partition(":")
            hdr[k.strip().lower()] = v.strip()
        if r.status != 200:
            r.error = f"HTTP {r.status}: {(await asyncio.wait_for(reader.read(3000), 30))[:300]!r}"
            return
        chunked = "chunked" in hdr.get("transfer-encoding", "")

        async def pieces():
            if chunked:
                while True:
                    sl = await reader.readline()
                    if not sl:
                        return
                    n = int(sl.strip().split(b";")[0] or b"0", 16)
                    if n == 0:
                        return
                    pc = await reader.readexactly(n)
                    await reader.readexactly(2)
                    yield pc
            else:
                while True:
                    pc = await reader.read(65536)
                    if not pc:
                        return
                    yield pc
        buf = b""
        async for pc in pieces():
            buf += pc
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                for ln in ev.split(b"\n"):
                    ln = ln.strip()
                    if not ln.startswith(b"data:"):
                        continue
                    payload = ln[5:].strip()
                    if payload == b"[DONE]":
                        continue
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    now = time.perf_counter()
                    u = obj.get("usage") or None
                    if u:
                        if u.get("prompt_tokens") is not None:
                            r.prompt_tokens = u["prompt_tokens"]
                        if u.get("completion_tokens") is not None:
                            r.completion_tokens = u["completion_tokens"]
                        ptd = u.get("prompt_tokens_details") or {}
                        ct = ptd.get("cached_tokens", u.get("cached_tokens"))
                        if ct is not None:
                            r.cached_tokens = ct
                    chs = obj.get("choices") or []
                    ch = chs[0] if chs else {}
                    if ch.get("finish_reason"):
                        r.finish = ch["finish_reason"]
                    dlt = ch.get("delta") or {}
                    txt = (dlt.get("content") or "") + (dlt.get("reasoning_content") or dlt.get("reasoning") or "") + (ch.get("text") or "")
                    tok = bool(chs) and ((u or {}).get("completion_tokens") or 0) >= 1
                    if (txt or tok) and r.t_tok is None:
                        r.t_tok = now
                    if txt and r.spec.get("keep_text"):
                        r.text.append(txt)
                    if txt or (token_basis and tok):
                        if r.t_first is None:
                            r.t_first = now
                        r.t_last = now
                        r.chunks.append((now, (u or {}).get("completion_tokens")))
    finally:
        writer.close()


# ----------------------------------------------------------------------------- calibration
def calibrate(a, text: W.Text):
    """Fit chars-per-token for prose and for table text from the server's own prompt_tokens (max_tokens=1 probes)."""
    def pt(content):
        body = dict(model=a.model, messages=[dict(role="user", content=content)], max_tokens=1, temperature=0)
        if a.no_think:
            body["chat_template_kwargs"] = dict(enable_thinking=False)
        req = urllib.request.Request(a.base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["usage"]["prompt_tokens"]
    base = pt("x")
    out = {}
    for kind, n in (("prose", 24000), ("table", 24000)):
        s = text.prose[10_000:10_000 + n] if kind == "prose" else text.table_tokens(n / text.cpt_table, "calib")[:n]
        out[kind] = len(s) / max(1, pt("x" + s) - base)
    text.cpt, text.cpt_table = out["prose"], out["table"]
    return dict(template_overhead_tokens=base, cpt_prose=out["prose"], cpt_table=out["table"])


# ----------------------------------------------------------------------------- the run
async def run(a):
    u = urllib.parse.urlparse(a.base)
    host, port = u.hostname, u.port or 80
    path = u.path.rstrip("/") + "/v1/chat/completions"
    params = json.loads(a.params) if a.params else {}
    text = W.Text(cpt=a.cpt, cpt_table=a.cpt_table)
    if getattr(a, "tokenizer", ""):
        from transformers import AutoTokenizer   # the served model's tokenizer: exact prompt token counts
        text.tok = await asyncio.to_thread(AutoTokenizer.from_pretrained, a.tokenizer)
        cal = dict(tokenizer=a.tokenizer, exact_prose_tokens=True, cpt_prose=a.cpt, cpt_table=a.cpt_table)
    else:
        cal = await asyncio.to_thread(calibrate, a, text) if a.calibrate else dict(cpt_prose=a.cpt, cpt_table=a.cpt_table, given=True)
    nonce = uuid.uuid4().hex[:10]
    ctx = dict(text=text, shapes=W.load_shapes(), seed=a.seed, nonce=nonce)
    make = W.CLASSES[a.workload]
    rng = random.Random(a.seed)
    seen = set()
    reqs: list[Req] = []
    counter = 0
    t0 = time.perf_counter()
    t_end = t0 + a.duration
    tasks = set()
    inflight = 0
    peak_inflight = 0

    def new_req(spec=None):
        nonlocal counter
        i = counter
        counter += 1
        spec = spec if spec is not None else make(params, i, rng, ctx)
        pid = spec.get("prefix_id")
        share = (spec.get("prefix_tokens") or 0) / (spec.get("target_in") or W.est_tokens(spec, text.cpt) or 1)
        cold = None if (pid is None or share < 0.25) else (pid not in seen)   # None: prefix too small to matter
        if pid is not None:
            seen.add(pid)
        r = Req(i, spec, cold)
        reqs.append(r)
        return r

    async def do(r: Req):
        nonlocal inflight, peak_inflight
        inflight += 1
        peak_inflight = max(peak_inflight, inflight)
        try:
            task = asyncio.ensure_future(send(host, port, path, body_for(a, r.spec), r, a.request_timeout,
                                                  token_basis=(a.ttft_basis == "token")))
            remaining = t_end - time.perf_counter()
            done, _ = await asyncio.wait({task}, timeout=max(0.0, remaining) if not a.max_requests else a.request_timeout)
            if task not in done:
                r.aborted = True
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            elif task.exception() is not None:
                r.error = repr(task.exception())[:300]
        finally:
            inflight -= 1

    async def closed_worker(w):
        """Stagger (default): worker w starts after a random 0-2 s delay, and its FIRST fixed-length request asks for a
        random 1..max_tokens tokens, so completions (and the prefills that replace them) are spread out from the start
        instead of arriving as one synchronized wave; first requests are sent before the window and never enter the
        latency statistics."""
        srng = random.Random(f"{a.seed}-stagger-{w}")
        if a.stagger and not a.max_requests:
            await asyncio.sleep(srng.uniform(0, min(2.0, a.warmup / 2)))
        first = True
        while time.perf_counter() < t_end:
            if a.max_requests and counter >= a.max_requests:
                return
            r = new_req()
            if first and a.stagger and r.spec.get("ignore_eos") and not a.max_requests:
                r.spec["max_tokens"] = srng.randint(1, r.spec["max_tokens"])
                r.spec["staggered"] = True
            first = False
            r.t_sched = time.perf_counter()
            await do(r)

    turns = int(params.get("turns", 4)) if a.workload == "agent_session" else 1
    sessions = 0

    async def session(spec, t_ready):
        """One agent session: its turns run one after another (each waits for the previous reply plus the planned think
        time); the session ends at the cell end, or at the first turn that errors, is aborted or is dropped."""
        while spec is not None and t_ready < t_end:
            await asyncio.sleep(max(0.0, t_ready - time.perf_counter()))
            r = new_req(spec)
            r.t_sched = t_ready
            if a.max_inflight and inflight >= a.max_inflight:
                r.dropped = True
                return
            await do(r)
            if r.error or r.aborted or r.status != 200:
                return
            nxt = W.agent_next(spec, "".join(r.text), ctx)
            if nxt is not None:
                t_ready = time.perf_counter() + spec["session"]["think"][spec["turn"] - 1]
            spec = nxt

    async def open_arrivals():
        nonlocal sessions
        arng = random.Random(a.seed + 1)
        sched = None
        lam = a.rate / turns          # agent sessions: SESSIONS arrive at rate/turns, so requests arrive at ~rate in steady state
        if a.exact_arrivals:
            # Exactly round(rate x duration) Poisson arrivals: exponential gaps rescaled to span the run. Same burstiness,
            # no Poisson count noise; matches `vllm bench serve --num-prompts rate*DUR --request-rate rate` (OpenRelay's client).
            n = max(1, round(lam * a.duration))
            gaps = [arng.expovariate(lam) for _ in range(n + 1)]
            scale = a.duration / sum(gaps)
            acc, sched = 0.0, []
            for g in gaps[:n]:
                acc += g * scale
                sched.append(t0 + acc)
        t_next = t0
        k = 0
        while True:
            if sched is not None:
                if k >= len(sched):
                    return
                t_next = sched[k]
                k += 1
            else:
                t_next += arng.expovariate(lam)
            if t_next >= t_end or (a.max_requests and counter >= a.max_requests):
                return
            await asyncio.sleep(max(0.0, t_next - time.perf_counter()))
            if a.workload == "agent_session":
                t = asyncio.ensure_future(session(make(params, sessions, rng, ctx), t_next))
                sessions += 1
                tasks.add(t)
                t.add_done_callback(tasks.discard)
                continue
            r = new_req()
            r.t_sched = t_next
            if a.max_inflight and inflight >= a.max_inflight:
                r.dropped = True
                continue
            t = asyncio.ensure_future(do(r))
            tasks.add(t)
            t.add_done_callback(tasks.discard)

    m0 = m1 = None
    finished = asyncio.Event()

    async def scrapes():
        """Server counters at the window edges: after warmup, and at the window end OR when the traffic is done
        (whichever is first; --max-requests runs have no meaningful window end)."""
        nonlocal m0, m1
        await asyncio.sleep(a.warmup)
        m0 = await asyncio.to_thread(scrape, a.metrics)
        try:
            await asyncio.wait_for(finished.wait(), timeout=max(0.0, t_end - time.perf_counter()))
        except asyncio.TimeoutError:
            pass
        m1 = await asyncio.to_thread(scrape, a.metrics)
    # whole-cell counter totals: just before the first request, and after the last finished and the counter settled. Engines
    # differ in WHEN they count prompt tokens (vLLM at prefill, SGLang at request completion with a variable lag), so only
    # whole-cell totals compare cleanly with what the client sent (2026-11-22, bench/counter_probe.py + d3_sglang_counter_probe).
    m_start = await asyncio.to_thread(scrape, a.metrics) if a.metrics else None
    scr = asyncio.ensure_future(scrapes()) if a.metrics else None
    if a.rate:
        await open_arrivals()
        if tasks:
            await asyncio.gather(*list(tasks), return_exceptions=True)
    else:
        await asyncio.gather(*(closed_worker(w) for w in range(a.concurrency)))
    finished.set()
    if scr:
        await scr
    t_stop = time.perf_counter()
    m_end = None
    if a.metrics and m_start and "error" not in m_start:
        prev = None
        for _ in range(10):                      # settle: two equal readings 3 s apart, at most ~30 s
            await asyncio.sleep(3)
            m_end = await asyncio.to_thread(scrape, a.metrics)
            if "error" in m_end or (prev and prev["prompt"] == m_end["prompt"]):
                break
            prev = m_end
    s = summarize(a, params, cal, reqs, t0, t_end, t_stop, m0, m1, peak_inflight, nonce)
    if m_end and "error" not in m_end:
        client_total = sum(r.prompt_tokens or 0 for r in reqs if r.status == 200 and r.prompt_tokens)
        srv_total = m_end["prompt"] - m_start["prompt"]
        s["summary"]["server_totals"] = dict(server_prompt_tokens=srv_total, client_prompt_tokens=client_total,
                                             ratio=(srv_total / client_total) if client_total else None,
                                             basis="whole cell: counter before the first request vs after all finished and settled")
    return s


def summarize(a, params, cal, reqs, t0, t_end, t_stop, m0, m1, peak_inflight, nonce=None):
    w_lo, w_hi = t0 + a.warmup, min(t_end, t_stop) if not a.max_requests else t_stop
    win = max(1e-9, w_hi - w_lo)
    ok = [r for r in reqs if r.status == 200 and not r.error and not r.dropped]
    done = [r for r in ok if not r.aborted and r.finish]
    with_u = [r for r in ok if r.completion_tokens and not r.aborted and r.chunks]
    tpc = (sum(r.completion_tokens for r in with_u) / sum(len(r.chunks) for r in with_u)) if with_u else 1.0

    def toks_in(r):
        if not r.chunks:
            return 0.0
        if all(c is not None for _, c in r.chunks):
            prev, n = 0, 0
            for t, c in r.chunks:
                if w_lo <= t <= w_hi:
                    n += c - prev
                prev = c
            return float(n)
        sc = (r.completion_tokens / len(r.chunks)) if (r.completion_tokens and not r.aborted) else tpc
        return sc * sum(1 for t, _ in r.chunks if w_lo <= t <= w_hi)

    def ntok(r):
        if r.completion_tokens and not r.aborted:
            return r.completion_tokens
        if r.chunks and r.chunks[-1][1] is not None:
            return r.chunks[-1][1]
        return len(r.chunks) * tpc

    out_win = sum(toks_in(r) for r in ok)
    first_in = [r for r in ok if r.t_first and w_lo <= r.t_first <= w_hi]
    in_win = sum(r.prompt_tokens or 0 for r in first_in)
    cached_win = sum(r.cached_tokens or 0 for r in first_in)
    done_win = [r for r in done if r.t_last is not None and w_lo <= r.t_last <= w_hi]   # a reply with no token chunk has no t_last

    def lat(rs):
        ttft = [r.t_first - r.t_send for r in rs if r.t_first]
        tpot, itl = [], []
        for r in rs:
            if r.t_first and r.t_last and len(r.chunks) > 1:
                n = ntok(r)
                if n and n > 1:
                    tpot.append((r.t_last - r.t_first) / (n - 1))
                ts = [t for t, _ in r.chunks]
                itl += [ts[k] - ts[k - 1] for k in range(1, len(ts))]
        e2e = [r.t_last - r.t_send for r in rs if r.t_last and not r.aborted and r.finish]
        qd = [r.t_send - r.t_sched for r in rs if r.t_send and r.t_sched]
        return dict(ttft_s=dist(ttft), tpot_s=dist(tpot), itl_s=dist(itl), e2e_s=dist(e2e), client_queue_s=dist(qd))
    steady = [r for r in ok if r.t_send and r.t_send >= w_lo]
    s = dict(
        window_s=win, mode=("open" if a.rate else "closed"), offered_rate=a.rate or None, concurrency=None if a.rate else a.concurrency,
        peak_inflight=peak_inflight, requests_sent=len([r for r in reqs if not r.dropped]), dropped=sum(r.dropped for r in reqs),
        completed=len(done), completed_in_window=len(done_win), aborted_at_end=sum(r.aborted for r in reqs),
        errors=sum(1 for r in reqs if r.error or (r.status not in (200, None))),
        req_per_s=len(done_win) / win, output_tok_per_s=out_win / win, input_tok_per_s=in_win / win,
        cached_input_tok_per_s=cached_win / win, total_tok_per_s=(out_win + in_win) / win,
        cache_hit_ratio=(cached_win / in_win) if in_win and any(r.cached_tokens is not None for r in first_in) else None,
        cached_tokens_reported=any(r.cached_tokens is not None for r in ok),
        prompt_tokens=dist([r.prompt_tokens for r in ok if r.prompt_tokens]),
        completion_tokens=dist([r.completion_tokens for r in done if r.completion_tokens]),
        target_in_tokens=dist([r.spec.get("target_in") for r in ok if r.spec.get("target_in")]),
        completed_no_first_token=sum(1 for r in done if r.t_first is None), ttft_basis=a.ttft_basis,
        tokens_per_chunk=tpc, finish_reasons={k: sum(1 for r in done if r.finish == k) for k in {r.finish for r in done}},
        **lat(steady or ok),
    )
    s["per_user_tok_per_s_p50"] = (1 / s["tpot_s"]["p50"]) if s.get("tpot_s") and s["tpot_s"]["p50"] else None
    cold = [r for r in ok if r.cold is True and r.t_first]
    warm = [r for r in ok if r.cold is False and r.t_first]
    s["ttft_cold_prefix_s"] = dist([r.t_first - r.t_send for r in cold])
    s["ttft_warm_prefix_s"] = dist([r.t_first - r.t_send for r in warm])
    s["first_requests_ttft_s"] = [round(r.t_first - r.t_send, 4) for r in sorted(ok, key=lambda r: r.i)[:5] if r.t_first]
    classes = sorted({r.spec["cls"] for r in ok})
    if len(classes) > 1:
        s["by_class"] = {c: dict(n=sum(1 for r in ok if r.spec["cls"] == c), **lat([r for r in ok if r.spec["cls"] == c])) for c in classes}
    if m0 and m1 and "error" not in m0 and "error" not in m1:
        dtm = m1["t"] - m0["t"]
        sm = dict(window_s=dtm, gen_tok_per_s=(m1["gen"] - m0["gen"]) / dtm, prompt_tok_per_s=(m1["prompt"] - m0["prompt"]) / dtm,
                  counters=m1["found"], preemptions=m1["preempt"] - m0["preempt"])
        if m1["spec_draft"] - m0["spec_draft"] > 0:
            sm["spec_acceptance_rate"] = (m1["spec_acc"] - m0["spec_acc"]) / (m1["spec_draft"] - m0["spec_draft"])
            if m1["spec_drafts"] - m0["spec_drafts"] > 0:
                sm["spec_mean_accepted_per_step"] = 1 + (m1["spec_acc"] - m0["spec_acc"]) / (m1["spec_drafts"] - m0["spec_drafts"])
        if m1["pc_queries"] - m0["pc_queries"] > 0:
            sm["prefix_cache_hit_ratio"] = (m1["pc_hits"] - m0["pc_hits"]) / (m1["pc_queries"] - m0["pc_queries"])
        s["server_metrics"] = sm
    elif a.metrics:
        s["server_metrics"] = dict(error=(m0 or {}).get("error") or (m1 or {}).get("error") or "no scrape")
    rows = [dict(i=r.i, cls=r.spec["cls"], staggered=bool(r.spec.get("staggered")), prefix_id=r.spec.get("prefix_id"), cold=r.cold, dropped=r.dropped, status=r.status,
                 aborted=r.aborted, finish=r.finish, target_in=r.spec.get("target_in"), max_tokens=r.spec["max_tokens"],
                 prompt_tokens=r.prompt_tokens, cached_tokens=r.cached_tokens, completion_tokens=r.completion_tokens,
                 t_sched=None if r.t_sched is None else round(r.t_sched - t0, 4), t_send=None if r.t_send is None else round(r.t_send - t0, 4),
                 ttft=None if not r.t_first else round(r.t_first - r.t_send, 4), e2e=None if not r.t_last else round(r.t_last - r.t_send, 4),
                 ttft_token=None if not r.t_tok else round(r.t_tok - r.t_send, 4), chunks=len(r.chunks),
                 session=(r.spec.get("session") or {}).get("sid"), turn=r.spec.get("turn")) for r in reqs]
    return dict(schema="traffic/1", label=a.label, args={k: v for k, v in vars(a).items()}, params=params, calibration=cal, nonce=nonce,
                t_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - (time.perf_counter() - t0))),
                summary=s, sample_errors=[r.error for r in reqs if r.error][:5], requests=rows)


def line(d):
    s = d["summary"]
    g = lambda k, f=1: ((s.get(k) or {}).get("p50") or 0) * f  # noqa: E731
    mode = f"R={s['offered_rate']}" if s["mode"] == "open" else f"C={s['concurrency']}"
    chr_ratio = "-" if s["cache_hit_ratio"] is None else f"{s['cache_hit_ratio']:.2f}"
    return (f"[{d['label']}] {mode} win={s['window_s']:.0f}s done={s['completed']} err={s['errors']} drop={s['dropped']} "
            f"req/s={s['req_per_s']:.2f} out={s['output_tok_per_s']:.0f} in={s['input_tok_per_s']:.0f} tok/s "
            f"cache={chr_ratio} "
            f"TTFT p50={g('ttft_s'):.3f}s p95={((s.get('ttft_s') or {}).get('p95') or 0):.3f}s TPOT p50={g('tpot_s', 1000):.1f}ms "
            f"srv={(s.get('server_metrics') or {}).get('gen_tok_per_s')}")


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", default="synthetic", choices=sorted(W.CLASSES))
    ap.add_argument("--params", default="", help="JSON params for the workload class")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--rate", type=float, default=0.0, help="open loop: Poisson arrivals per second (overrides --concurrency)")
    ap.add_argument("--max-inflight", type=int, default=0, help="open loop: drop arrivals beyond this many in flight (0 = no cap)")
    ap.add_argument("--duration", type=float, default=90)
    ap.add_argument("--warmup", type=float, default=None)
    ap.add_argument("--max-requests", type=int, default=0, help="send exactly N, wait for all (no cut); for small probes")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-think", action="store_true", default=True)
    ap.add_argument("--ttft-basis", choices=["text", "token"], default="text",
                    help="text: first token = first chunk with text (default, every Gemma run); token: also a chunk whose usage "
                         "says >= 1 completion token (models whose first token is a swallowed special token, e.g. Inkling)")
    ap.add_argument("--think", dest="no_think", action="store_false")
    ap.add_argument("--extra", default="")
    ap.add_argument("--metrics", default="")
    ap.add_argument("--cpt", type=float, default=4.0, help="chars per token for prose (use --calibrate to fit it)")
    ap.add_argument("--cpt-table", type=float, default=3.0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--tokenizer", default="", help="served model dir/id: size prose prompts to an exact token count")
    ap.add_argument("--exact-arrivals", action="store_true", help="open loop: exactly rate x duration Poisson arrivals")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-stagger", dest="stagger", action="store_false", default=True)
    ap.add_argument("--request-timeout", type=float, default=1800)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", required=True)
    return ap


def finish_args(a):
    if a.warmup is None:
        a.warmup = min(15.0, a.duration * 0.2)
    if a.max_requests:
        a.duration, a.warmup = 10 ** 6, 0.0
    return a


def main():
    a = finish_args(parser().parse_args())
    d = asyncio.run(run(a))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(d, open(a.out, "w"), indent=1)
    print(line(d), flush=True)
    return 0 if not d["summary"]["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
