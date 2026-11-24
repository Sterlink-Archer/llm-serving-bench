#!/usr/bin/env python3
"""OpenRelay's prefill probe on its own (their probe.sh: 16 concurrent, 64 prompts of 10,000 random tokens, 1 output token,
run twice, the second counts) -> one JSON. Reuses bench/protocol.vllm_bench_prefill so there is one implementation.
  python3 bench/prefill_probe.py --model gemma --tokenizer /root/models/x --out /root/or/out/prefill_x.json"""
import argparse
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8000")
ap.add_argument("--model", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--c", type=int, default=16)
ap.add_argument("--tokens", type=int, default=10000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
ns = types.SimpleNamespace(base=a.base, model=a.model, tokenizer=a.tokenizer, prefill_c=a.c, prefill_tokens=a.tokens)
r = P.vllm_bench_prefill(ns, a.out.replace(".json", "_cell.json"))
json.dump(r, open(a.out, "w"), indent=1)
print(f"[prefill_probe] {r['prefill_tok_s']:.0f} tok/s (16 x {a.tokens}, 2nd run)", flush=True)
