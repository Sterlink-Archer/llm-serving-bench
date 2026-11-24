#!/usr/bin/env python3
"""Quality of every served config vs the BF16 original (Maks 2026-11-22: "if we run the quantized version then lets compare it
to the original regular one"). Reads the quality-reference eval files written by run.py eval steps named
  q_gsm8kor250_<tag>.json   GSM8K test[:250], OpenRelay's prompt and scoring
  q_mmlupro200_<tag>.json   our 200-item MMLU-Pro set
  q_agree400_<tag>[_rep2].json   400 public-text prompts (data/eval/agree_400.jsonl), greedy, max_tokens 192, concurrency 16
under results/<label>/<stamp>/. <tag> is the server name; the reference is the server tagged `bf16_ref` (BF16 weights, BF16 KV).
Every answer is re-scored with bench/scoring.py, so a scoring fix re-scores every run.

Per config vs the reference, on the same items:
- accuracy with a Wilson 95% interval;
- paired delta vs BF16 with a 95% interval for paired proportions, the discordant counts b (BF16 right, config wrong) and
  c (BF16 wrong, config right), and an exact McNemar p-value;
- greedy agreement on the agreement set: share of prompts whose whole 192-token continuation is textually identical to BF16's,
  and the mean common-prefix fraction (characters), overall and per prompt shape;
- reference points: BF16 vs its own repeat (the run-to-run noise floor of greedy decoding at concurrency 16), and OpenRelay's
  recipe with MTP vs without (speculative decoding should be lossless; disagreement there is numerics, not quality).
Writes results/quality_rows.json (schema quality/1) and results/QUALITY_TABLE.md.
  python3 -I analysis/quality.py [--md results/QUALITY_TABLE.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import scoring  # noqa: E402

SETS = ("gsm8kor250", "mmlupro200", "agree400")
REF_TAG = "bf16_ref"


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def paired_delta_ci(b, c, n, z=1.96):
    """config - reference accuracy on paired items; Wald interval for a difference of paired proportions."""
    if not n:
        return None, None, None
    d = (c - b) / n
    var = ((b + c) - (c - b) ** 2 / n) / (n * n)
    h = z * math.sqrt(max(var, 0.0))
    return d, d - h, d + h


def precision_of(cfg, srv):
    w = " ".join(x["repo"] for x in srv.get("weights", []))
    launch = srv.get("launch", "")
    if "FP8-dynamic" in w:
        wq = "FP8 W8A8 (RedHat FP8-dynamic)"
    elif "FP8-block" in w:
        wq = "FP8 W8A8 block (RedHat)"
    elif "MXFP4" in w:
        wq = "MXFP4 W4A4 (amd/...-MXFP4)"
    elif "nvidia/" in w and "NVFP4" in w:
        wq = "NVFP4 (NVIDIA ModelOpt; attention projections BF16)"
    elif "NVFP4" in w:
        wq = "NVFP4 W4A4 (RedHat compressed-tensors)"
    elif "google/gemma-4-31B-it" in w:
        wq = "BF16 (google original)"
    else:
        wq = "unknown"
    m = re.search(r"--kv-cache-dtype (\S+)", launch)
    kv = m.group(1) if m else "auto"
    kv = "BF16" if kv in ("auto", "bfloat16") else "FP8" if "fp8" in kv else kv
    m = re.search(r'"num_speculative_tokens":\s*(\d+)', launch)
    mtp = int(m.group(1)) if m else 0
    engine = "sglang" if "sglang" in launch else "vllm"
    return dict(weights=wq, kv_cache=kv, mtp=mtp, engine=engine, gpu=(cfg.get("snapshot") or {}).get("gpu"))


def load():
    """{config_key: {meta, sets: {set: rows_by_id}, rep2: rows_by_id, files}} keeping the newest run per (label, tag)."""
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "*" / "*" / "q_*.json"))):
        p = pathlib.Path(f)
        m = re.match(r"q_(gsm8kor250|mmlupro200|agree400)_(.+?)(_rep2)?\.json$", p.name)
        if not m:
            continue
        st, tag, rep2 = m.group(1), m.group(2), bool(m.group(3))
        rd = p.parent
        cfg = json.loads((rd / "config.json").read_text()) if (rd / "config.json").exists() else {}
        srv = next((s for s in cfg.get("servers", []) if s.get("name") == tag), {})
        key = f"{rd.parent.name}:{tag}"
        d = json.loads(p.read_text())
        rows = {}
        for r in d.get("rows", []):
            pred = scoring.extract(r["task"], r.get("text"))
            rows[r["id"]] = dict(r, pred=pred, correct=scoring.is_correct(r["task"], pred, r.get("gold")) if r["task"] != "agree" else None)
        # one config = (label, server tag). Several runs of a label may each fill some sets (a follow-up run re-measures only
        # what a failed step lost); per set, the newest file wins. A rep2 pass is only used from the same run as its first pass.
        e = out.setdefault(key, dict(run=rd.parent.name, stamp=rd.name, tag=tag, meta=precision_of(cfg, srv), sets={}, rep2=None,
                                     files={}, stamps={}))
        e["stamp"] = max(e["stamp"], rd.name)
        k = "agree400_rep2" if rep2 else st
        if rd.name >= e["stamps"].get(k, ""):
            e["stamps"][k] = rd.name
            e["files"][k] = str(p.relative_to(ROOT))
            if rep2:
                e["rep2"] = rows
            else:
                e["sets"][st] = rows
    return out


def acc_block(rows_c, rows_r):
    ids = sorted(set(rows_c) & set(rows_r)) if rows_r is not None else sorted(rows_c)
    if not ids:
        return None
    n = len(ids)
    k = sum(1 for i in ids if rows_c[i]["correct"])
    lo, hi = wilson(k, n)
    res = dict(n=n, correct=k, acc=k / n, ci95=[lo, hi], errors=sum(1 for i in ids if rows_c[i].get("error")),
               truncated=sum(1 for i in ids if rows_c[i].get("finish") == "length"))
    if rows_r is not None:
        b = sum(1 for i in ids if rows_r[i]["correct"] and not rows_c[i]["correct"])
        c = sum(1 for i in ids if not rows_r[i]["correct"] and rows_c[i]["correct"])
        d, dlo, dhi = paired_delta_ci(b, c, n)
        res.update(delta_vs_ref=d, delta_ci95=[dlo, dhi], b_lost=b, c_gained=c, mcnemar_p=mcnemar_exact(b, c),
                   same_answer=sum(1 for i in ids if rows_r[i]["pred"] == rows_c[i]["pred"]) / n)
    return res


def prefix_frac(a, b):
    a, b = a or "", b or ""
    if not a and not b:
        return 1.0
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k / max(len(a), len(b))


def agree_block(rows_c, rows_r, shapes):
    if rows_c is None or rows_r is None:
        return None
    ids = sorted(set(rows_c) & set(rows_r))
    ok = [i for i in ids if rows_c[i].get("text") is not None and rows_r[i].get("text") is not None]
    if not ok:
        return None
    ex = [i for i in ok if rows_c[i]["text"] == rows_r[i]["text"]]
    pf = [prefix_frac(rows_c[i]["text"], rows_r[i]["text"]) for i in ok]
    by = {}
    for s in sorted(set(shapes.get(i, "?") for i in ok)):
        si = [i for i in ok if shapes.get(i) == s]
        by[s] = dict(n=len(si), exact=sum(1 for i in si if rows_c[i]["text"] == rows_r[i]["text"]) / len(si),
                     prefix=sum(prefix_frac(rows_c[i]["text"], rows_r[i]["text"]) for i in si) / len(si))
    lo, hi = wilson(len(ex), len(ok))
    return dict(n=len(ok), exact_match=len(ex) / len(ok), exact_ci95=[lo, hi], mean_prefix_frac=sum(pf) / len(pf), by_shape=by)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(ROOT / "results" / "QUALITY_TABLE.md"))
    a = ap.parse_args()
    shapes = {json.loads(l)["id"]: json.loads(l)["shape"] for l in open(ROOT / "data" / "eval" / "agree_400.jsonl") if l.strip()}
    cfgs = load()
    refs = sorted((c for c in cfgs.values() if c["tag"] == REF_TAG), key=lambda c: c["stamp"])
    ref = refs[-1] if refs else None
    rows = []
    for key, c in sorted(cfgs.items()):
        rr = (lambda st: ref["sets"].get(st)) if ref else (lambda st: None)
        row = dict(config=key, run_id=", ".join(f"{c['run']}/{st}" for st in sorted(set(c["stamps"].values()))), tag=c["tag"], is_reference=bool(ref and c is ref), **c["meta"],
                   gsm8kor250=acc_block(c["sets"]["gsm8kor250"], rr("gsm8kor250")) if "gsm8kor250" in c["sets"] else None,
                   mmlupro200=acc_block(c["sets"]["mmlupro200"], rr("mmlupro200")) if "mmlupro200" in c["sets"] else None,
                   agree_vs_ref=agree_block(c["sets"].get("agree400"), rr("agree400"), shapes) if ref and c is not ref else None,
                   agree_self_repeat=(agree_block(c["sets"].get("agree400"), c["rep2"], shapes)
                                      if c["rep2"] and c["stamps"].get("agree400_rep2") == c["stamps"].get("agree400") else None),
                   files=c["files"])
        rows.append(row)
    # speculative decoding losslessness: the same recipe with MTP on vs off on one box
    pairs = []
    for key, c in cfgs.items():
        if c["tag"].endswith("_nomtp"):
            on = cfgs.get(f"{c['run']}:{c['tag'][:-len('_nomtp')]}")
            if on:
                same_box = on["stamps"].get("agree400") == c["stamps"].get("agree400")
                pairs.append(dict(mtp_on=f"{on['run']}:{on['tag']}", mtp_off=key, same_box=same_box,
                                  agree_files=[on["files"].get("agree400"), c["files"].get("agree400")],
                                  agree=agree_block(on["sets"].get("agree400"), c["sets"].get("agree400"), shapes),
                                  gsm_same_answer=(acc_block(on["sets"]["gsm8kor250"], c["sets"]["gsm8kor250"]) or {}).get("same_answer")
                                  if "gsm8kor250" in on["sets"] and "gsm8kor250" in c["sets"] else None))
    out = dict(schema="quality/1", reference=ref and f"{ref['run']}/{ref['stamp']}:{ref['tag']}", rules=__doc__, configs=rows, mtp_lossless=pairs)
    (ROOT / "results" / "quality_rows.json").write_text(json.dumps(out, indent=1))

    def pct(x):
        return "-" if x is None else f"{100 * x:.1f}%"

    def acc_cell(b):
        if not b:
            return "-"
        s = f"{100 * b['acc']:.1f}% ({b['correct']}/{b['n']}, CI {100 * b['ci95'][0]:.1f}-{100 * b['ci95'][1]:.1f})"
        if b.get("delta_vs_ref") is not None:
            s += f"; Δ {100 * b['delta_vs_ref']:+.1f} pts [{100 * b['delta_ci95'][0]:+.1f}, {100 * b['delta_ci95'][1]:+.1f}], b/c {b['b_lost']}/{b['c_gained']}, p {b['mcnemar_p']:.2f}"
        return s

    L = ["# Quality vs the BF16 original (generated by analysis/quality.py; do not edit)", "",
         f"Reference: **{out['reference'] or 'NONE YET'}** (BF16 weights, BF16 KV). Greedy decoding everywhere. GSM8K = test[:250] "
         "with OpenRelay's prompt and scoring; MMLU-Pro = our 200-item set; agreement = 400 public-text prompts, 192-token greedy "
         "continuations, identical text vs BF16. Δ = config minus BF16 on the same items, paired 95% interval; b/c = items lost / gained "
         "vs BF16; p = exact McNemar. 250 items give roughly ±2-3 points of sampling noise on accuracy.", "",
         "| config | GPU / engine | weights | KV | MTP | GSM8K 250 | MMLU-Pro 200 | greedy agreement vs BF16 (exact / mean prefix) | run |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (not r["is_reference"], r["config"])):
        ag = r["agree_vs_ref"]
        agt = (f"{pct(ag['exact_match'])} / {pct(ag['mean_prefix_frac'])} (n {ag['n']})" if ag else
               ("reference" if r["is_reference"] else "-"))
        if r["agree_self_repeat"]:
            agt += f"; self-repeat {pct(r['agree_self_repeat']['exact_match'])}"
        L.append(f"| {r['config']} | {r['gpu']} / {r['engine']} | {r['weights']} | {r['kv_cache']} | {r['mtp'] or 'off'} | "
                 f"{acc_cell(r['gsm8kor250'])} | {acc_cell(r['mmlupro200'])} | {agt} | {r['run_id']} |")
    if pairs:
        L += ["", "Speculative decoding losslessness (same recipe, MTP on vs off). Read the agreement against the same engine's own "
              "self-repeat: greedy decoding at concurrency 16 is not run-to-run deterministic, and a different box adds more drift.", "",
              "| MTP on | MTP off | box | agreement exact / prefix | same GSM8K answer |", "|---|---|---|---|---|"]
        for p in pairs:
            ag = p["agree"] or {}
            L.append(f"| {p['mtp_on']} | {p['mtp_off']} | {'same' if p['same_box'] else 'different boxes (agreement re-run)'} | "
                     f"{pct(ag.get('exact_match'))} / {pct(ag.get('mean_prefix_frac'))} | {pct(p['gsm_same_answer'])} |")
    L += ["", "Not run: Cloudflare Workers AI does not serve Gemma 4 31B (catalog search 2026-11-22, "
              "research/raw/cloudflare_workers_ai_gemma_search_*.json); its closest model, gemma-4-26b-a4b-it, is a different model "
              "and its precision is not disclosed."]
    pathlib.Path(a.md).write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
