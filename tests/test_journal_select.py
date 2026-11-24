#!/usr/bin/env python3
"""analysis/journal.py picks the right files in a multi-server run (2026-11-22: step 122 recorded another server's gate) and
the column the writer asks for; the GLM range exists.  python3 -I tests/test_journal_select.py"""
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import journal as J  # noqa: E402

bad = []
with tempfile.TemporaryDirectory() as td:
    J.ROOT = pathlib.Path(td)
    rd = J.ROOT / "results" / "glmX" / "20261122T000000Z"
    rd.mkdir(parents=True)
    (rd / "run.json").write_text(json.dumps(dict(usd_per_h=dict(total=6.0), servers={"mx": {}, "mx_recipe": {}})))
    (rd / "config.json").write_text(json.dumps(dict(label="glmX", price_model="openrelay/glm-5.3-flash",
                                                    servers=[dict(name="mx"), dict(name="mx_recipe")])))
    for n, acc in (("gate_mx_gsm8kor250.json", 0.91), ("gate_mx_recipe_gsm8kor250.json", 0.80)):
        (rd / n).write_text(json.dumps(dict(gate=dict(acc=acc))))
    for d, client, rate in (("protocol_mx_random", "vllmbench", 3.5), ("protocol_mx_text", "traffic", 3.25),
                            ("protocol_mx_recipe_random", "vllmbench", 3.0)):
        (rd / d).mkdir()
        (rd / d / "protocol_summary.json").write_text(json.dumps(dict(client=client, breakeven_rps=1.0, max_rate_at_slo_rps=rate,
                                                                      max_goodput_rps=rate, ladder=[])))
    rid = "glmX/20261122T000000Z"
    m, _ = J.metrics(rid, "mx")
    if m["gsm8k_acc"] != 0.91:
        bad.append(f"server mx took gate acc {m['gsm8k_acc']} (mx_recipe's is 0.80)")
    if m["max_rate_at_slo_rps"] != 3.5 or m["_column"] != "A":
        bad.append(f"default column must be A (3.5): got {m['max_rate_at_slo_rps']} column {m['_column']}")
    want = (6.0 / 3600 / 3.5) / 0.0004959          # vs the config's price_model (GLM 5.3 Flash), not the ladder's break-even
    if abs((m["ratio_sync"] or 0) - want) > 1e-6:
        bad.append(f"ratio_sync {m['ratio_sync']} must use the config's price_model bar ({want:.4f}), not the break-even rate")
    m, _ = J.metrics(rid, "mx", "B")
    if m["max_rate_at_slo_rps"] != 3.25 or m["_column"] != "B":
        bad.append(f"--column B must take the traffic summary (3.25): got {m['max_rate_at_slo_rps']}")
    m, _ = J.metrics(rid, "mx_recipe")
    if m["gsm8k_acc"] != 0.80 or m["max_rate_at_slo_rps"] != 3.0:
        bad.append(f"server mx_recipe: acc {m['gsm8k_acc']} rate {m['max_rate_at_slo_rps']} (want 0.80, 3.0)")
    m, _ = J.metrics(rid, "mx_recipe", "B")
    if m["max_rate_at_slo_rps"] is not None:
        bad.append(f"mx_recipe has no column B: must record none, got {m['max_rate_at_slo_rps']}")
    (rd / "gate_mx_gsm8kor250.json").unlink()
    (rd / "gate_gsm8kor100.json").write_text(json.dumps(dict(gate=dict(acc=0.5))))
    m, _ = J.metrics(rid, "mx")
    if m["gsm8k_acc"] != 0.5:
        bad.append(f"a server with no gate of its own takes the un-prefixed gate: got {m['gsm8k_acc']}")
if list(J.STEP_RANGES.get("glm", [])) != list(range(300, 400)):
    bad.append(f"STEP_RANGES['glm'] = {J.STEP_RANGES.get('glm')}")
if list(J.STEP_RANGES.get("gemm-headroom", [])) != list(range(400, 500)):
    bad.append(f"STEP_RANGES['gemm-headroom'] = {J.STEP_RANGES.get('gemm-headroom')}")
_seen = sorted(x for r in J.STEP_RANGES.values() for x in r)
if len(_seen) != len(set(_seen)):
    bad.append("two owners share a step number")
print("FAIL\n  " + "\n  ".join(bad) if bad else "OK: journal takes each server's own gate, the asked column; owner ranges disjoint (GLM 300-399, GEMM headroom 400-499)")
sys.exit(1 if bad else 0)
