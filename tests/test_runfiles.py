#!/usr/bin/env python3
"""analysis/runfiles.py: a protocol dir belongs to the server whose run.json steps record it, else to the longest server
name it starts with; nothing with a summary is silently dropped (2026-11-22: Inkling's protocol_b300_* and
protocol_rep_B_text dirs were missing from claims.json).  python3 -I tests/test_runfiles.py"""
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import runfiles as RF  # noqa: E402

bad = []
with tempfile.TemporaryDirectory() as td:
    rd = pathlib.Path(td)
    for d in ("protocol_b300_B_text", "protocol_mx_recipe_A", "protocol_mx_A", "protocol_stray", "protocol_nosummary"):
        (rd / d).mkdir()
        if d != "protocol_nosummary":
            (rd / d / "protocol_summary.json").write_text("{}")
    run = dict(servers=dict(mx=dict(steps={"protocol_b300_B_text": dict(rc=0, cmd="python3 bench/protocol.py --model m")}),
                            mx_recipe=dict(steps={})))
    servers = ["mx", "mx_recipe"]
    got = {s: [p.name for p in RF.protocol_dirs(rd, s, servers, run)] for s in servers}
    if got != {"mx": ["protocol_b300_B_text", "protocol_mx_A"], "mx_recipe": ["protocol_mx_recipe_A"]}:
        bad.append(f"attribution {got}")
    orph = [p.name for p in RF.unattributed(rd, servers, run)]
    if orph != ["protocol_stray"]:
        bad.append(f"unattributed {orph} (want ['protocol_stray']; a dir with no summary is not a protocol result)")
# the real case that went missing
real = sorted(ROOT.glob("results/ikC_pro6000_mixed_dspark_rep_s7/*/run.json"))
if real:
    rd = real[-1].parent
    run = json.loads(real[-1].read_text())
    if [p.name for p in RF.protocol_dirs(rd, "mixed_dspark", ["mixed_dspark"], run)] != ["protocol_rep_B_text"]:
        bad.append(f"{rd}: protocol_rep_B_text not attributed to mixed_dspark")
print("FAIL\n  " + "\n  ".join(bad) if bad else "OK: protocol dirs attributed by run.json steps, then longest server prefix; strays listed")
sys.exit(1 if bad else 0)
