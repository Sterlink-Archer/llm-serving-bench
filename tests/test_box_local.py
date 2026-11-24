#!/usr/bin/env python3
"""Exercise infra/daytona.Box (exec, put/get, script files, background jobs) against a LOCAL shell instead of the
Daytona toolbox, to catch quoting/transfer bugs without creating a sandbox. Usage: python3 -I tests/test_box_local.py DIR"""
import os, pathlib, subprocess, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "infra"))
import daytona as dt

work = pathlib.Path(sys.argv[1]).resolve(); work.mkdir(parents=True, exist_ok=True)
dt.WORK = str(work)

def fake_req(method, url, body=None, timeout=60):
    assert url.endswith("/process/execute"), url
    p = subprocess.run(["sh", "-c", body["command"]], capture_output=True, text=True, timeout=body.get("timeout") or 60)
    return dict(exitCode=p.returncode, result=p.stdout + p.stderr)
dt._req = fake_req

b = dt.Box(dict(id="local", toolboxProxyUrl="http://fake"))
r = b.exec("echo hi && false"); assert r["rc"] == 1 and "hi" in r["out"], r
data = os.urandom(200_000) + b"'\"$`\\ end"
b.put_bytes(data, f"{work}/sub/blob.bin"); assert (work / "sub/blob.bin").read_bytes() == data
assert b.get(f"{work}/sub/blob.bin", work / "back.bin") and (work / "back.bin").read_bytes() == data
assert not b.get(f"{work}/nope.bin", work / "nope_local.bin")
r = b.sh("quote", "x='a b'; echo \"$x\" \"it's\" $((2+3))"); assert r["out"].strip() == "a b it's 5", r
b.bg("job1", "echo start; sleep 2; echo done >&2; exit 7")
assert b.job_rc("job1") is None
time.sleep(3.5)
assert b.job_rc("job1") == 7, b.job_rc("job1")
assert "done" in b.tail("job1"), b.tail("job1")
print("Box local test: OK", sorted(p.name for p in (work / "jobs").iterdir()))
