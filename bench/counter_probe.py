#!/usr/bin/env python3
"""Exact semantics of an engine's Prometheus prompt-token counter, one request at a time.

Why: on e4_b300_sglang column B the server-vs-client cross-check read 1.0757 (SGLang's sglang:prompt_tokens_total grew
7.6% more than the prompt tokens its own responses reported), with no retractions; against arrival-time counting it read
~1.10 at both rates. This probe settles it with no concurrency: for each request it scrapes the counter before, after the
request is SENT but before it finishes (mid-stream, to see WHEN it counts), and after it finishes, and compares the
deltas with the response's usage.prompt_tokens. Streaming and non-streaming, several lengths, unique prompts (no cache).

  python3 bench/counter_probe.py --model gemma --metrics http://127.0.0.1:8000/metrics --out /root/or/out/counter_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traffic as T  # noqa: E402
import workloads as W  # noqa: E402


def chat(base, model, content, max_tokens, stream):
    body = dict(model=model, messages=[dict(role="user", content=content)], max_tokens=max_tokens, temperature=0,
                chat_template_kwargs=dict(enable_thinking=False), stream=stream)
    if stream:
        body["stream_options"] = dict(include_usage=True)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            return json.loads(r.read()).get("usage")
        for line in r:
            line = line.decode(errors="replace").strip()
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                d = json.loads(line[5:])
                usage = d.get("usage") or usage
    return usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--lengths", default="500,2000,9500")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    text = W.Text(cpt=4.07)
    rows = []
    for stream in (True, False):
        for L in [int(x) for x in a.lengths.split(",")]:
            for k in range(a.reps):
                content = f"[probe {time.time()} {stream} {L} {k}]\n" + text.prose_tokens(L, f"cp-{stream}-{L}-{k}") + "\n\nSummarize."
                m0 = T.scrape(a.metrics)
                res = {}
                th = threading.Thread(target=lambda: res.update(usage=chat(a.base, a.model, content, a.max_tokens, stream)))
                th.start()
                time.sleep(0.15)                       # sent, prefill under way or done, decode not finished
                m_mid = T.scrape(a.metrics)
                th.join()
                time.sleep(0.5)
                m1 = T.scrape(a.metrics)
                u = res.get("usage") or {}
                rows.append(dict(stream=stream, target=L, usage_prompt_tokens=u.get("prompt_tokens"),
                                 usage_completion_tokens=u.get("completion_tokens"),
                                 cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                                 server_prompt_delta_mid=m_mid["prompt"] - m0["prompt"], server_prompt_delta=m1["prompt"] - m0["prompt"],
                                 server_gen_delta=m1["gen"] - m0["gen"], counters=m1["found"]))
                r = rows[-1]
                print(f"[counter_probe] stream={stream} L={L}: usage.prompt={r['usage_prompt_tokens']} server delta={r['server_prompt_delta']} "
                      f"(mid-request {r['server_prompt_delta_mid']}) gen usage={r['usage_completion_tokens']} server={r['server_gen_delta']}", flush=True)
    ok = [r for r in rows if r["usage_prompt_tokens"]]
    summ = dict(ratio_server_over_usage=sum(r["server_prompt_delta"] for r in ok) / max(1, sum(r["usage_prompt_tokens"] for r in ok)),
                per_request_offsets=sorted({r["server_prompt_delta"] - r["usage_prompt_tokens"] for r in ok}),
                counted_before_finish=sum(1 for r in ok if r["server_prompt_delta_mid"] > 0), n=len(ok))
    json.dump(dict(schema="counter_probe/1", t=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), summary=summ, rows=rows),
              open(a.out, "w"), indent=1)
    print(f"[counter_probe] server/usage = {summ['ratio_server_over_usage']:.4f}; per-request offsets {summ['per_request_offsets']}; "
          f"counted before finish in {summ['counted_before_finish']}/{summ['n']}", flush=True)


if __name__ == "__main__":
    main()
