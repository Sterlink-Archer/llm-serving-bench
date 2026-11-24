#!/usr/bin/env python3
"""Greedy agreement of one served Inkling config against a reference config, on the same GSM8K items.

Both inputs are bench/evalrun.py outputs (gate_*.json / eval_*.json: temperature 0, OpenRelay's prompt, every raw output
saved). The checkpoint is the release itself (NVFP4 + MXFP8), so the reference is the vendor's Balanced recipe on a
verified platform, and the question is whether our serving changes (engine flags, KV dtype, patches) move outputs.

Reported per pair (re-scored with bench/scoring.py, so a scoring fix re-scores everything):
  acc_ref / acc_cfg, b = ref right & cfg wrong, c = ref wrong & cfg right, exact two-sided McNemar p on (b, c),
  same_answer = identical extracted answer, same_text = identical full output (reasoning + content),
  prefix_chars_median = median length of the common prefix of the full outputs (where greedy decoding first diverges).

Claim gate (recipes/inkling/README.md): GSM8K accuracy >= 0.90 on test[:250] AND not worse than the reference on the pooled
GSM8K + MMLU-Pro items (McNemar p >= 0.05, or c >= b). Run it against BOTH reference runs: the vendor recipe against itself
(eval_ref_r1 vs r2) already differs at p = 0.02 through reasoning-loop truncations at 16K tokens, so one reference run is a
noisy anchor. Greedy outputs of non-batch-invariant engines differ between runs at near-ties, so same_text is reported,
not gated.

  python3 -I recipes/inkling/tools/agree.py REF.json CFG.json [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench"))
import scoring  # noqa: E402

FLOOR = 0.90


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def rows(path):
    d = json.loads(pathlib.Path(path).read_text())
    out = {}
    for r in d["rows"]:
        pred = scoring.extract(r["task"], r.get("text"))
        full = (r.get("reasoning") or "") + "\x00" + (r.get("text") or "")
        out[(r["task"], r["id"])] = dict(pred=pred, ok=scoring.is_correct(r["task"], pred, r["gold"]), full=full,
                                         err=bool(r.get("error")), trunc=r.get("finish") == "length",
                                         toks=((r.get("usage") or {}).get("completion_tokens")))
    return d, out


def prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _pair(R, C, keys):
    b = sum(1 for k in keys if R[k]["ok"] and not C[k]["ok"])
    c = sum(1 for k in keys if not R[k]["ok"] and C[k]["ok"])
    n = max(1, len(keys))
    pre = [prefix(R[k]["full"], C[k]["full"]) for k in keys]
    return dict(n=len(keys), acc_ref=round(sum(R[k]["ok"] for k in keys) / n, 4), acc_cfg=round(sum(C[k]["ok"] for k in keys) / n, 4),
                b_lost=b, c_gained=c, mcnemar_p=round(mcnemar_exact(b, c), 4),
                same_answer=round(sum(R[k]["pred"] == C[k]["pred"] for k in keys) / n, 4),
                same_text=round(sum(R[k]["full"] == C[k]["full"] for k in keys) / n, 4),
                prefix_chars_median=statistics.median(pre) if pre else None,
                truncated_ref=sum(R[k]["trunc"] for k in keys), truncated_cfg=sum(C[k]["trunc"] for k in keys),
                errors_cfg=sum(C[k]["err"] for k in keys))


def compare(ref_path, cfg_path):
    """Per task (GSM8K = OpenRelay's gsm8kor, MMLU-Pro) and pooled. Gate: GSM8K accuracy >= FLOOR, and not worse than the
    reference on the pooled items (exact McNemar p >= 0.05, or c >= b), and no request errors."""
    dr, R = rows(ref_path)
    dc, C = rows(cfg_path)
    keys = sorted(k for k in R if k in C)
    res = dict(ref=os.path.relpath(ref_path, ROOT), cfg=os.path.relpath(cfg_path, ROOT), split_ref=dr.get("split"), split_cfg=dc.get("split"),
               floor=FLOOR, pooled=_pair(R, C, keys), tasks={})
    for t in sorted({k[0] for k in keys}):
        res["tasks"][t] = _pair(R, C, [k for k in keys if k[0] == t])
    g = res["tasks"].get("gsm8kor") or res["tasks"].get("gsm8k") or {}
    po = res["pooled"]
    res["gate_pass"] = bool(g and g["acc_cfg"] >= FLOOR and (po["mcnemar_p"] >= 0.05 or po["c_gained"] >= po["b_lost"])
                            and po["errors_cfg"] == 0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cfg")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = compare(a.ref, a.cfg)
    print(json.dumps(r, indent=1))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(r, indent=1) + "\n")
    return 0 if r["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
