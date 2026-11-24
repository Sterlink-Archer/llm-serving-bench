#!/usr/bin/env python3
"""Single-prompt prefill speed vs prompt length (brainstorm round 2, item 3).

For each length L it sends `--reps` unique prompts of ~L tokens, ONE AT A TIME, with max_tokens=1, and records each
request's wall latency and server-counted prompt tokens. tok/s = prompt_tokens / latency (per request, median kept).
Reading it:
  - flat tok/s across lengths -> prefill is GEMM-bound: the chip's ceiling for this model;
  - tok/s falling with length -> the attention kernel (the 512-dim global layers attend over the whole prompt) is
    what costs, and it grows with context.
Prompts are unique public-domain-style prose (bench/workloads.py Text) with a per-request nonce, so no prefix-cache hits.
A first short request warms the path and is not counted.

  python3 bench/prefill_sweep.py --model gemma --lengths 2000,4000,10000,16000 --reps 3 --out /root/or/out/prefill_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workloads as W  # noqa: E402


def post(base, body, timeout=900):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="2000,4000,10000,16000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cpt", type=float, default=4.07, help="chars per token for sizing the prose")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = W.Text(cpt=a.cpt)

    def one(tokens, tag):
        p = f"[sweep {time.time()} {tag}]\n" + T.prose_tokens(tokens, f"sweep-{tag}") + "\n\nSummarize."
        body = dict(model=a.model, messages=[dict(role="user", content=p)], max_tokens=1, temperature=0,
                    chat_template_kwargs=dict(enable_thinking=False))
        t0 = time.perf_counter()
        r = post(a.base, body)
        dt = time.perf_counter() - t0
        pt = (r.get("usage") or {}).get("prompt_tokens")
        return dict(target=tokens, prompt_tokens=pt, latency_s=round(dt, 4), tok_s=(pt / dt) if pt else None)

    one(500, "warm")
    rows = []
    for L in [int(x) for x in a.lengths.split(",")]:
        reps = [one(L, f"{L}-{k}") for k in range(a.reps)]
        ts = [x["tok_s"] for x in reps if x["tok_s"]]
        rows.append(dict(length_target=L, prompt_tokens_median=statistics.median([x["prompt_tokens"] for x in reps if x["prompt_tokens"]]),
                         tok_s_median=statistics.median(ts) if ts else None, latency_s_median=statistics.median([x["latency_s"] for x in reps]),
                         reps=reps))
        print(f"[sweep] {L:>6} tokens: {rows[-1]['tok_s_median']:.0f} tok/s single-prompt prefill "
              f"({rows[-1]['latency_s_median']:.3f} s)", flush=True)
    base = rows[0]["tok_s_median"] or 0
    out = dict(schema="prefill_sweep/1", t=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), model=a.model, reps=a.reps,
               rows=rows, ratio_longest_to_shortest=(rows[-1]["tok_s_median"] / base) if base else None,
               reading="ratio ~1.0 = GEMM-bound (flat); ratio well below 1 = attention cost grows with prompt length")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[sweep] tok/s longest/shortest = {out['ratio_longest_to_shortest']:.2f}", flush=True)


if __name__ == "__main__":
    main()
