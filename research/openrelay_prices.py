#!/usr/bin/env python3
"""Extract OpenRelay's published per-token prices from a saved https://openrelay.inc/inference-pricing page
(research/raw/openrelay/inference-pricing_<UTC stamp>.html) into JSON, so the bar is parsed, not hand-typed.

The page's table reads, per model: "<Name> openrelay/<id> <ctx> context <Tier> $in $cached $out [<Tier> $in $cached $out ...]".
Tiers seen 2026-11-22: Sync, Batch, "Async, within 1 hour", "Async, within 24 hours". Prices are USD per 1M tokens.

  python3 -I research/openrelay_prices.py research/raw/openrelay/inference-pricing_20261122T032303Z.html
"""
from __future__ import annotations

import html
import json
import pathlib
import re
import sys

TIER = r"(Sync|Batch|Async, within 1 hour|Async, within 24 hours)"
PRICE = r"\$([\d.]+)|n/a|-"


def parse(path):
    t = pathlib.Path(path).read_text(errors="ignore")
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", t, flags=re.S)
    txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", body)))
    start = txt.index("Model Tier Input Cached input Output")
    end = txt.index("All prices in USD per 1M tokens")
    table = txt[start:end]
    models = {}
    for m in re.finditer(r"([A-Z][\w.\- ]+?) (openrelay/[\w.\-]+) (\d+K) context ((?:" + TIER + r" (?:\S+) (?:\S+) (?:\S+) ?)+)", table):
        name, mid, ctx, tiers = m.group(1).strip(), m.group(2), m.group(3), m.group(4)
        rec = dict(name=name, id=mid, context=ctx, tiers={})
        for tm in re.finditer(TIER + r" (\S+) (\S+) (\S+)", tiers):
            def num(s):
                return float(s[1:]) if s.startswith("$") else None
            rec["tiers"][tm.group(1)] = dict(in_usd_per_m=num(tm.group(2)), cached_in_usd_per_m=num(tm.group(3)),
                                             out_usd_per_m=num(tm.group(4)))
        models[mid] = rec
    return dict(source="https://openrelay.inc/inference-pricing", saved_file=str(path),
                fetched_utc=pathlib.Path(path).stem.rsplit("_", 1)[-1], models=models)


if __name__ == "__main__":
    d = parse(sys.argv[1])
    out = pathlib.Path(__file__).resolve().parent / f"openrelay_prices_{d['fetched_utc']}.json"
    out.write_text(json.dumps(d, indent=1))
    for mid, r in d["models"].items():
        print(mid, json.dumps(r["tiers"]))
    print("wrote", out)
