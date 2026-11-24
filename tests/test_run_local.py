#!/usr/bin/env python3
"""End-to-end test of run.py with NO Daytona calls (~3 min): the sandbox is a local shell in a temp dir, the "engine" is
tests/mock_server.py on 127.0.0.1:8000, the ledger is a temp file. Exercises: bundle upload + extract, server start and
health, startup-facts capture (the mock is not vLLM, so the config sets allow_rejected), the eval gate, the protocol
ladder (traffic client) with the prefill shot, a smoke matrix, directory pulls, phase events, teardown with delete and
reap confirmation, cost_measured.json and run.json.  python3 -I tests/test_run_local.py
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The fake box serves on the box's fixed 127.0.0.1:8000, and the first launch is meant to FAIL (port 1). If anything else
# answers on 8000 (2026-11-22 09:52Z: another session's concurrent test_all), that launch looks healthy and the test fails
# with a misleading "server died". bin/test_all.sh holds a lock across sessions; this check names the cause if it recurs.
import socket  # noqa: E402
import time  # noqa: E402
for _ in range(30):
    with socket.socket() as _s:
        if _s.connect_ex(("127.0.0.1", 8000)) != 0:
            break
    time.sleep(2)
else:
    print("FAIL\n  127.0.0.1:8000 is in use by another process (another session's test or tunnel?); this test needs it free")
    sys.exit(1)
sys.path.insert(0, str(ROOT / "infra"))
sys.path.insert(0, str(ROOT))
os.environ["OR_LOCAL_TEST"] = "1"
import daytona as dt  # noqa: E402

tmp = pathlib.Path(tempfile.mkdtemp(prefix="or_runtest_"))
os.environ["OR_RESULTS_DIR"] = str(tmp / "results")
work = tmp / "box"
work.mkdir()
dt.WORK = str(work)
dt.LEDGER = tmp / "ledger.jsonl"
state = dict(deleted=False)


def fake_req(method, url, body=None, timeout=60):
    assert url.endswith("/process/execute"), url
    p = subprocess.run(["bash", "-c", body["command"]], capture_output=True, text=True, timeout=body.get("timeout") or 60)
    return dict(exitCode=p.returncode, result=p.stdout + p.stderr)


EVICT = os.environ.get("OR_TEST_EVICT") == "1"   # scenario: the spot box is preempted right after the eval gate


def fake_get_sandbox(sid):
    if state["deleted"]:
        raise dt.ApiError(404, "gone", "/sandbox")
    if EVICT and (work / "out" / "eval_gate.json").exists():
        return dict(id=sid, state="stopped", spot=True, spotEvictedAt=dt.now_iso())
    return dict(id=sid, state="started", spot=True)


def fake_delete(sid, why="", run_id=None):
    state["deleted"] = True
    subprocess.run(["bash", "-c", "pkill -f '^python3 -I " + str(ROOT) + "/tests/mock_server.py' || true"])
    dt.ledger(dict(event="delete", id=sid, run_id=run_id, why=why))


def fake_create(snapshot, label, max_minutes, wait_s=900, run_id=None, volumes=None):
    dt.ledger(dict(event="create", id="local-test", run=label, run_id=run_id, snapshot=snapshot["name"], gpu="H100", gpus=1, cpu=8, mem=48,
                   disk=150, spot=True, usd_per_h=dt.hourly_usd("H100", 1, 8, 48, 150), ttl_min=max_minutes, deadline=0))
    return dict(id="local-test", toolboxProxyUrl="http://fake", spot=True, _create_s=0.1)


dt._req = fake_req
dt.get_sandbox = fake_get_sandbox
dt.delete_sandbox = fake_delete
dt.create_gpu_sandbox = fake_create
dt.get_snapshot = lambda name: dict(name=name, state="active", gpu=1, gpuType=["H100"], cpu=8, mem=48, disk=150)   # mirrors the API: an active snapshot
dt.spot_available = lambda g, n=1: 9

import run as R  # noqa: E402
R.W = str(work)
R.ROOT = ROOT

cfg = {"label": "localtest", "snapshot": {"name": "local-snap", "image": "mock", "gpu": "H100", "gpus": 1, "cpu": 8, "mem": 48, "disk": 150},
       "max_minutes": 20, "budget_usd": 5.0,
       "servers": [{"name": "mock", "weights": [], "served_name": "m", "allow_rejected": True,
                    "launch": f"python3 -I {ROOT}/tests/mock_server.py --port $MOCK_PORT --knee 24 --itl 0.003 --prefill-tps 80000",
                    "env": {"MOCK_PORT": "1"},   # first launch fails (cannot bind port 1): exercises the same-box fallback
                    "fallbacks": [{"flags": "", "env": {"MOCK_PORT": "8000"}}],
                    "ready_timeout_s": 60,
                    "steps": [{"type": "eval", "name": "eval_gate", "sets": "data/eval/gsm8kor_test.jsonl", "limit": 20, "conc": 20,
                               "gate": {"task": "gsm8kor", "threshold": 0.0}, "est_s": 60},
                              {"type": "protocol", "multiples": [1.0, 2.0], "seeds": [7, 8], "duration": 12, "breakeven_rps": 2.0,
                               "prefill_shot": True, "client": "traffic"},
                              {"type": "matrix", "preset": "smoke", "only": "s1k_256|prefix", "scale_duration": 0.4}]}]}
R.Run.save_orig = R.Run.save
r = R.Run(cfg, "inline")
r.out = tmp / "results" / "localtest" / "x"
r.out.mkdir(parents=True)
r.go()
out = r.out
fails = []
if EVICT:
    rep = json.loads((out / "run.json").read_text())
    led = [json.loads(l) for l in dt.LEDGER.read_text().splitlines()]
    ev = [x for x in led if x["event"] == "evicted"]
    if len(ev) != 1 or not ev[0].get("interrupted") or ev[0]["interrupted"]["step"] != "protocol_mock" or ev[0]["label"] != "spot preemption, outside our control":
        fails.append(f"evicted record wrong: {ev}")
    if rep["servers"]["mock"]["steps"].get("protocol_mock", {}).get("excluded") != "evicted":
        fails.append(f"interrupted step not excluded: {rep['servers']['mock']['steps'].get('protocol_mock')}")
    if rep["servers"]["mock"]["steps"].get("eval_gate", {}).get("rc") != 0:
        fails.append("the step before the eviction should be intact")
    print(json.dumps(ev, indent=1)[:900])
    print("FAIL\n  " + "\n  ".join(fails) if fails else "OK: eviction recorded, interrupted step excluded, earlier step intact")
    sys.exit(1 if fails else 0)
rep = json.loads((out / "run.json").read_text())
if rep.get("error"):
    fails.append(f"run error: {rep['error']}")
srv = rep["servers"]["mock"]
steps = srv.get("steps", {})
for k in ("eval_gate", "protocol_mock", "m_mock_smoke"):
    if k not in steps or steps[k].get("rc") != 0:
        fails.append(f"step {k}: {steps.get(k)}")
ps = out / "protocol_mock" / "protocol_summary.json"
if not ps.exists():
    fails.append("no protocol_summary.json")
else:
    p = json.loads(ps.read_text())
    if not p.get("prefill") or not p["prefill"].get("prefill_tok_s"):
        fails.append("no prefill result")
    if len(p["ladder"]) != 2 or "kept_seed" not in p["ladder"][0]:
        fails.append(f"ladder incomplete: {p['ladder']}")
if not (out / "m_mock_smoke" / "matrix_summary.json").exists():
    fails.append("matrix dir not pulled to results/<run>/m_mock_smoke/")
if not (out / "startup_facts_mock.json").exists():
    fails.append("no startup facts")
if not (out / "cost_measured.json").exists():
    fails.append("no cost_measured.json")
led = [json.loads(l) for l in dt.LEDGER.read_text().splitlines()]
phases = {x["phase"] for x in led if x["event"] == "phase"}
for ph in ("image_pull", "create", "weight_download", "engine_start", "warmup", "eval", "steady", "teardown", "reap_confirm"):
    if ph not in phases:
        fails.append(f"missing phase {ph}")
if not any(x["event"] == "create" and x.get("run_id") for x in led):
    fails.append("create without run_id")
if not rep.get("delete_confirmed"):
    fails.append("delete not confirmed")
la = srv.get("launch_attempts") or []
if len(la) != 2 or la[0]["ok"] or not la[1]["ok"]:
    fails.append(f"same-box fallback not exercised: {la}")
print(json.dumps(dict(phases=sorted(phases), max_rate=json.loads(ps.read_text()).get("max_rate_at_slo_rps") if ps.exists() else None,
                      cost=json.loads((out / "cost_measured.json").read_text())["configs"][0] if (out / "cost_measured.json").exists() else None),
                 indent=1)[:2500])
if fails:
    print("FAIL\n  " + "\n  ".join(fails))
    sys.exit(1)
print(f"OK: run.py end to end, results in {out}")
