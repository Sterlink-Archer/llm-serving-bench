#!/usr/bin/env python3
"""OpenRouter price bar for a model, from a saved /api/v1/models/<id>/endpoints snapshot (research/raw/or_endpoints_*.json).

For each workload shape (input tokens, output tokens per request) it computes what every listed provider would charge,
then the bar we must beat: (a) the cheapest provider of any precision, (b) the cheapest provider per precision class
(bf16 / fp8 / 4-bit), (c) the median provider. A provider with a negative status (OpenRouter marks degraded endpoints
that way) is listed but excluded from the bar. Prices are $ per token in the snapshot; we print $ per 1M tokens.

  python3 -I research/or_prices.py research/raw/or_endpoints_google_gemma-4-31b-it_<stamp>.json [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

SHAPES = {  # name: (input tokens, output tokens) per request
    "1k_in_1k_out": (1024, 1024),
    "1k_in_256_out": (1024, 256),
    "8k_in_1k_out": (8192, 1024),
    "1k_in_8k_out": (1024, 8192),
    "input_only_1M": (1_000_000, 0),
    "output_only_1M": (0, 1_000_000),
}
CLASS = {"bf16": "bf16", "fp8": "fp8", "fp4": "4bit", "nvfp4": "4bit", "int4": "4bit", "mxfp4": "4bit"}


def load(path):
    d = json.loads(pathlib.Path(path).read_text())["data"]
    eps = []
    for e in d["endpoints"]:
        p = e.get("pricing") or {}
        eps.append(dict(provider=e.get("provider_name"), tag=e.get("tag"), quant=e.get("quantization") or "unknown",
                        cls=CLASS.get(e.get("quantization") or "", "unknown"), status=e.get("status"),
                        uptime_30m=e.get("uptime_last_30m"), ctx=e.get("context_length"),
                        max_out=e.get("max_completion_tokens"),
                        in_usd_per_m=float(p.get("prompt") or 0) * 1e6, out_usd_per_m=float(p.get("completion") or 0) * 1e6,
                        cache_read_usd_per_m=(float(p["input_cache_read"]) * 1e6) if p.get("input_cache_read") else None))
    return d, eps


def bar(eps, shape):
    i, o = SHAPES[shape] if isinstance(shape, str) else shape
    live = [e for e in eps if (e["status"] or 0) >= 0]
    def cost(e):
        return (i * e["in_usd_per_m"] + o * e["out_usd_per_m"]) / 1e6
    rows = sorted(((cost(e), e) for e in live), key=lambda x: x[0])
    out = dict(shape=shape, in_tokens=i, out_tokens=o,
               cheapest=dict(usd_per_request=rows[0][0], provider=rows[0][1]["provider"], quant=rows[0][1]["quant"],
                             in_usd_per_m=rows[0][1]["in_usd_per_m"], out_usd_per_m=rows[0][1]["out_usd_per_m"]),
               median_usd_per_request=statistics.median(c for c, _ in rows), by_class={})
    for cls in ("bf16", "fp8", "4bit", "unknown"):
        rc = [(c, e) for c, e in rows if e["cls"] == cls]
        if rc:
            out["by_class"][cls] = dict(usd_per_request=rc[0][0], provider=rc[0][1]["provider"], quant=rc[0][1]["quant"],
                                        in_usd_per_m=rc[0][1]["in_usd_per_m"], out_usd_per_m=rc[0][1]["out_usd_per_m"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--json")
    a = ap.parse_args()
    d, eps = load(a.snapshot)
    stamp = pathlib.Path(a.snapshot).stem.rsplit("_", 1)[-1]
    print(f"# OpenRouter endpoints for {d.get('id')} — snapshot {stamp} ({len(eps)} endpoints)\n")
    print("| provider | quant | status | in $/1M | out $/1M | cache-read $/1M | ctx | max out |")
    print("|---|---|---|---|---|---|---|---|")
    for e in sorted(eps, key=lambda e: (e["in_usd_per_m"] + e["out_usd_per_m"])):
        cr = "" if e["cache_read_usd_per_m"] is None else f"{e['cache_read_usd_per_m']:.3f}"
        print(f"| {e['provider']} | {e['quant']} | {e['status']} | {e['in_usd_per_m']:.3f} | {e['out_usd_per_m']:.3f} | "
              f"{cr} | {e['ctx']} | {e['max_out']} |")
    bars = {s: bar(eps, s) for s in SHAPES}
    print("\n| workload (in/out tokens) | cheapest $/request | who (quant) | median $/request | cheapest bf16 | cheapest fp8 | cheapest 4-bit |")
    print("|---|---|---|---|---|---|---|")
    for s, b in bars.items():
        bc = b["by_class"]
        f = lambda k: (f"${bc[k]['usd_per_request']:.6f} ({bc[k]['provider']})" if k in bc else "-")
        print(f"| {s} ({b['in_tokens']}/{b['out_tokens']}) | ${b['cheapest']['usd_per_request']:.6f} | {b['cheapest']['provider']} "
              f"({b['cheapest']['quant']}) | ${b['median_usd_per_request']:.6f} | {f('bf16')} | {f('fp8')} | {f('4bit')} |")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(dict(model=d.get("id"), snapshot=a.snapshot, endpoints=eps, bars=bars), indent=1))


if __name__ == "__main__":
    main()
