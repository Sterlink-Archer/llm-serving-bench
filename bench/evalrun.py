#!/usr/bin/env python3
"""Run the fixed eval set (bench/scoring.py prompts) against an OpenAI-compatible server. Stdlib only.
Greedy decoding (temperature 0), thinking off, so a quantized model and the BF16 reference see identical requests.
Saves every raw output, so analysis/quality.py can re-score and run paired tests vs the BF16 run on the PC.

  python3 bench/evalrun.py --model gemma --sets data/eval/gsm8k_200.jsonl,data/eval/mmlupro_200.jsonl --out eval.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring  # noqa: E402


def ask(base, model, prompt, max_tokens, no_think, timeout, ctk=None, api_key=None):
    body = dict(model=model, messages=[dict(role="user", content=prompt)], max_tokens=max_tokens, temperature=0.0)
    kw = dict(enable_thinking=False) if no_think else {}
    kw.update(ctk or {})            # e.g. Inkling 2 Mini reads reasoning_effort, not enable_thinking
    if kw:
        body["chat_template_kwargs"] = kw
    hdr = {"Content-Type": "application/json"}
    if api_key:
        hdr["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers=hdr)
    t0 = time.perf_counter()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            ch = d["choices"][0]
            msg = ch.get("message") or {}
            return dict(text=msg.get("content") or "", reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
                        finish=ch.get("finish_reason"), usage=d.get("usage"), secs=time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:300]
            time.sleep(2 * (attempt + 1))
    return dict(text=None, error=err, secs=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--label", default="eval")
    ap.add_argument("--gate-task", default="", help="e.g. gsm8k: write gate {task, threshold, acc, pass}")
    ap.add_argument("--gate-threshold", type=float, default=0.965)
    ap.add_argument("--chat-template-kwargs", default="", help="JSON merged into the request's chat_template_kwargs")
    ap.add_argument("--api-key-env", default="", help="name of an env var holding a Bearer token (hosted endpoints); never a literal")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = []
    for path in a.sets.split(","):
        task = os.path.basename(path).split("_")[0]
        items = [json.loads(l) for l in open(path) if l.strip()]
        if a.limit:
            items = items[:a.limit]
        for it in items:
            jobs.append((task, it))
    ctk = json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else None
    api_key = os.environ.get(a.api_key_env) if a.api_key_env else None
    if a.api_key_env and not api_key:
        raise SystemExit(f"--api-key-env {a.api_key_env}: variable not set")
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(ask, a.base, a.model, scoring.PROMPTS[t](it), a.max_tokens, not a.think, a.timeout, ctk, api_key): (t, it)
                for t, it in jobs}
        rows = []
        for f in cf.as_completed(futs):
            t, it = futs[f]
            r = f.result()
            pred = scoring.extract(t, r.get("text"))
            rows.append(dict(task=t, id=it["id"], gold=it.get("answer"), pred=pred, correct=scoring.is_correct(t, pred, it.get("answer")) if t != "agree" else None,
                             **r))
    rows.sort(key=lambda r: (r["task"], r["id"]))
    summ = {}
    for t in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == t]
        summ[t] = dict(n=len(rs), correct=sum(bool(r["correct"]) for r in rs), acc=sum(bool(r["correct"]) for r in rs) / len(rs),
                       no_answer=sum(r["pred"] is None for r in rs), errors=sum(1 for r in rs if r.get("error")),
                       truncated=sum(1 for r in rs if r.get("finish") == "length"),
                       mean_completion_tokens=sum(((r.get("usage") or {}).get("completion_tokens") or 0) for r in rs) / len(rs))
    split = ",".join(os.path.basename(p).rsplit(".", 1)[0] + (f"[:{a.limit}]" if a.limit else "") for p in a.sets.split(","))
    gate = None
    if a.gate_task:
        g = summ.get(a.gate_task) or {}
        gate = dict(task=a.gate_task, threshold=a.gate_threshold, acc=g.get("acc"), n=g.get("n"),
                    passed=g.get("correct"), **{"pass": bool(g) and g["acc"] >= a.gate_threshold and not g.get("errors")})
    out = dict(label=a.label, args=vars(a), split=split, gate=gate, wall_s=time.time() - t0,
               t_end=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), summary=summ, rows=rows)
    json.dump(out, open(a.out, "w"), indent=1)
    def line(t, s):
        if t == "agree":
            # nothing to score here: the continuations are compared with BF16's by analysis/quality.py. Printing the accuracy
            # template ("0.000 (0/400, noans 400)") was read as a parse failure twice on 2026-11-22.
            empty = sum(1 for r in rows if r["task"] == t and not (r.get("text") or "").strip())
            return (f"agree: {s['n'] - empty}/{s['n']} continuations saved (empty {empty}, err {s['errors']}, trunc {s['truncated']}; "
                    f"agreement vs BF16 = analysis/quality.py)")
        return f"{t}: {s['acc']:.3f} ({s['correct']}/{s['n']}, noans {s['no_answer']}, err {s['errors']}, trunc {s['truncated']})"
    print(f"[{a.label}] " + " ".join(line(t, s) for t, s in summ.items()) + f" wall={time.time() - t0:.0f}s", flush=True)
    return 0 if not any(s["errors"] for s in summ.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
