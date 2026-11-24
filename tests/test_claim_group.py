#!/usr/bin/env python3
"""claim_group / origin lookup (analysis/screen_report.py cfg_key): server-level key in the run's copy, else server-level in the
current config, else config-level, else the label rule. Paired boxes (MI355X recipe session, mxC1_*) run OpenRelay's recipe and
ours back to back on one GPU, so the two servers must land in different claim groups with different origins.
  python3 -I tests/test_claim_group.py"""
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import screen_report as SR  # noqa: E402

tmp = pathlib.Path(tempfile.mkdtemp(prefix="or_cg_"))
SR.ROOT = tmp
(tmp / "configs").mkdir()
(tmp / "recipes" / "mi355x" / "configs").mkdir(parents=True)
paired = {"label": "mxC1_s7_r1", "snapshot": {"gpu": "MI355X"}, "servers": [
    {"name": "mx_recipe", "launch": "vllm serve x", "claim_group": "mx_recipe_C1", "origin": "ours"},
    {"name": "or_recipe", "launch": "vllm serve y", "claim_group": "or_recipe_paired_C1", "origin": "openrelay_recipe"}]}
(tmp / "recipes" / "mi355x" / "configs" / "mxC1_s7_r1.json").write_text(json.dumps(paired))
# the running box's copy predates the keys
old_copy = {"label": "mxC1_s7_r1", "snapshot": {"gpu": "MI355X"}, "servers": [{"name": "mx_recipe", "launch": "vllm serve x"},
                                                                              {"name": "or_recipe", "launch": "vllm serve y"}]}
(tmp / "results" / "mxC1_s7_r1" / "20261122T062000Z").mkdir(parents=True)
(tmp / "results" / "mxC1_s7_r1" / "20261122T062000Z" / "config.json").write_text(json.dumps(old_copy))
twin = {"label": "s2_x_seed7", "origin": "openrelay_recipe", "snapshot": {"gpu": "MI355X"}, "servers": [{"name": "s", "launch": "vllm serve"}]}
(tmp / "results" / "s2_x_seed7" / "t").mkdir(parents=True)
(tmp / "results" / "s2_x_seed7" / "t" / "config.json").write_text(json.dumps(twin))
cases = [
    (("mxC1_s7_r1/20261122T062000Z", "mx_recipe"), ("mx_recipe_C1", "ours")),
    (("mxC1_s7_r1/20261122T062000Z", "or_recipe"), ("or_recipe_paired_C1", "openrelay_recipe")),
    (("s2_x_seed7/t", "s"), ("s2_x", "openrelay_recipe")),          # label rule + config-level origin
    (("nolabel_seed8/t", "s"), ("nolabel", "ours")),                 # nothing on disk: label rule, default origin
]
bad = []
for (rid, srv), (g, o) in cases:
    got = (SR.claim_group(rid, srv), SR.run_meta(rid, srv)["origin"])
    if got != (g, o):
        bad.append(f"{rid}:{srv} -> {got}, want {(g, o)}")
print("FAIL\n  " + "\n  ".join(bad) if bad else f"OK: {len(cases)} claim_group/origin lookups")
sys.exit(1 if bad else 0)
