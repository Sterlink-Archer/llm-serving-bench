"""identity_probe.py --out FILE [--conc N] [--short 50] [--long 10] [--long-tokens 10000] [--max-tokens 256]

Token-level output identity probe for the drafter compaction (patch_draft_compact.py), asked by the lead's review of
confirm C1. Sends a FIXED prompt set to the local engine (127.0.0.1:8000, model "gemma") with temperature 0 and
ignore_eos, and records the generated TOKEN IDS of every request:
  - `--short` GSM8K questions from data/eval/gsm8kor_test.jsonl as raw completion text;
  - `--long` prompts of `--long-tokens` random token ids (fixed seeds, like OpenRelay's profreq.py), which span two
    8192-token prefill chunks and so exercise the drafter's first pass on prefill steps.
At --conc 1 every step holds one request (least batch-shape noise); at higher --conc, prefill and decode share steps
(the mixed steps the patch changes). Compare two output files with --compare A B: identical sequences, and for each
differing pair the first divergent token position.
"""
import argparse
import concurrent.futures as cf
import json
import random
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def post(body, timeout=900):
    req = urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def token_ids(resp):
    ch = resp["choices"][0]
    if ch.get("token_ids"):
        return list(ch["token_ids"])
    toks = (ch.get("logprobs") or {}).get("tokens") or []
    return [int(t.split(":", 1)[1]) for t in toks]


def prompts(a):
    out = []
    with open("data/eval/gsm8kor_test.jsonl") as f:
        for i, line in enumerate(f):
            if i >= a.short:
                break
            q = json.loads(line)
            out.append(("short", i, (q.get("question") or q.get("prompt") or json.dumps(q))[:2000] + "\nAnswer:"))
    for j in range(a.long):
        rnd = random.Random(1000 + j)
        out.append(("long", j, [rnd.randint(1000, 200000) for _ in range(a.long_tokens)]))
    return out


def run(a):
    items = prompts(a)

    def one(it):
        kind, i, p = it
        body = dict(model="gemma", prompt=p, max_tokens=a.max_tokens, temperature=0, ignore_eos=True,
                    logprobs=1, return_tokens_as_token_ids=True, return_token_ids=True)
        t = time.time()
        r = post(body)
        return dict(kind=kind, i=i, ids=token_ids(r), secs=round(time.time() - t, 3))

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(one, items))
    json.dump(dict(conc=a.conc, max_tokens=a.max_tokens, n=len(res), wall_s=round(time.time() - t0, 1), results=res), open(a.out, "w"))
    print(f"identity_probe: {len(res)} requests, conc {a.conc}, {time.time() - t0:.0f}s -> {a.out}; "
          f"lengths {sorted({len(r['ids']) for r in res})}")


def compare(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    ka = {(r["kind"], r["i"]): r["ids"] for r in A["results"]}
    kb = {(r["kind"], r["i"]): r["ids"] for r in B["results"]}
    same, firsts, n = 0, [], 0
    for k in sorted(set(ka) & set(kb)):
        n += 1
        x, y = ka[k], kb[k]
        if x == y:
            same += 1
            continue
        d = next((p for p, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
        firsts.append((k[0], k[1], d))
    return dict(a=fa, b=fb, pairs=n, identical=same, differing=len(firsts),
                first_divergence=sorted(firsts, key=lambda t: t[2])[:20])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--short", type=int, default=50)
    ap.add_argument("--long", type=int, default=10)
    ap.add_argument("--long-tokens", type=int, default=10000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        print(json.dumps(compare(*a.compare), indent=1))
        sys.exit(0)
    run(a)
