#!/usr/bin/env python3
"""Fetch the fixed inputs once, on the PC, and freeze them in data/ (stdlib only; HF datasets-server rows API).

  eval/gsm8k_200.jsonl      200 GSM8K test items      (openai/gsm8k, MIT)            seeded sample, seed 20261122
  eval/mmlupro_200.jsonl    200 MMLU-Pro test items   (TIGER-Lab/MMLU-Pro, MIT)      seeded 10-row windows, seed 20261122
  corpus/wiki_articles.jsonl  long Wikipedia articles (wikimedia/wikipedia 20231101.en, CC BY-SA 4.0) -> fixed-length prompts
  chat/ultrachat_400.jsonl  400 first-turn chat prompts (HuggingFaceH4/ultrachat_200k test_sft, MIT) -> natural-length load
Each file gets a sidecar <name>.meta.json: source dataset, config/split, row indices, fetch time (UTC), sha256 of the file.
Re-running never overwrites an existing frozen file (use --force).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import ssl
import time
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
CTX = ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")
UA = "openrelay-bench/0.1 (dataset fetch)"
SEED = 20261122


def get(url):
    """Polite: 1.5 s between calls, long back-off on 429 (the rows API rate-limits bursts)."""
    for attempt in range(8):
        time.sleep(1.5)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), context=CTX, timeout=60) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            if attempt == 7:
                raise
            time.sleep(15 * (attempt + 1))


def size(ds, config, split):
    d = get("https://datasets-server.huggingface.co/size?" + urllib.parse.urlencode(dict(dataset=ds)))
    for s in d["size"]["splits"]:
        if s["config"] == config and s["split"] == split:
            return s["num_rows"]
    raise KeyError((ds, config, split))


def rows(ds, config, split, offset, length=1):
    q = urllib.parse.urlencode(dict(dataset=ds, config=config, split=split, offset=offset, length=length))
    return [r["row"] | {"_row_idx": r["row_idx"]} for r in get("https://datasets-server.huggingface.co/rows?" + q)["rows"]]


def freeze(path: pathlib.Path, recs: list, meta: dict, force: bool):
    if path.exists() and not force:
        print(f"keep {path} (exists)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = dict(meta, fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), n=len(recs),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {path} n={len(recs)}")


def sample_rows(ds, config, split, n, seed=SEED, window=10):
    """n rows as n/window random contiguous windows of `window` rows (one API call per window)."""
    total = size(ds, config, split)
    starts = sorted(random.Random(seed).sample(range(0, total - window, window), n // window))
    out = []
    for s0 in starts:
        out += rows(ds, config, split, s0, window)
    idx = [r["_row_idx"] for r in out]
    return total, idx, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    total, idx, rs = sample_rows("openai/gsm8k", "main", "test", 200)
    recs = [dict(id=f"gsm8k-{r['_row_idx']}", question=r["question"],
                 answer=r["answer"].split("####")[-1].strip().replace(",", "")) for r in rs]
    freeze(HERE / "eval/gsm8k_200.jsonl", recs, dict(source="https://huggingface.co/datasets/openai/gsm8k", config="main",
           split="test", total_rows=total, row_idx=idx, seed=SEED, license="MIT"), a.force)

    total, idx, rs = sample_rows("TIGER-Lab/MMLU-Pro", "default", "test", 200)
    recs = [dict(id=f"mmlupro-{r['_row_idx']}", question=r["question"], options=r["options"], answer=r["answer"],
                 category=r["category"]) for r in rs]
    freeze(HERE / "eval/mmlupro_200.jsonl", recs, dict(source="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
           config="default", split="test", total_rows=total, row_idx=idx, seed=SEED, license="MIT"), a.force)

    # long articles: walk fixed offsets, keep the first 120 articles with >= 12,000 characters (~2.5k+ tokens)
    ds, cfg = "wikimedia/wikipedia", "20231101.en"
    total = size(ds, cfg, "train")
    rng = random.Random(SEED)
    arts, seen = [], 0
    while len(arts) < 120 and seen < 3000:
        off = rng.randrange(0, total - 100)
        for r in rows(ds, cfg, "train", off, 100):
            seen += 1
            if len(r.get("text", "")) >= 12000:
                arts.append(dict(id=f"wiki-{r['_row_idx']}", title=r["title"], url=r.get("url"), text=r["text"]))
        time.sleep(0.2)
    freeze(HERE / "corpus/wiki_articles.jsonl", arts[:120], dict(source=f"https://huggingface.co/datasets/{ds}", config=cfg,
           split="train", total_rows=total, rule=">=12000 chars, random 100-row windows, seed %d" % SEED,
           license="CC BY-SA 4.0"), a.force)

    ds, cfg, sp = "HuggingFaceH4/ultrachat_200k", "default", "test_sft"
    total, idx, rs = sample_rows(ds, cfg, sp, 400)
    recs = [dict(id=f"ultrachat-{r['_row_idx']}", prompt=r["prompt"]) for r in rs]
    freeze(HERE / "chat/ultrachat_400.jsonl", recs, dict(source=f"https://huggingface.co/datasets/{ds}", config=cfg,
           split=sp, total_rows=total, row_idx=idx, seed=SEED, license="MIT"), a.force)


if __name__ == "__main__":
    main()
