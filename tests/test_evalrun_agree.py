#!/usr/bin/env python3
"""bench/evalrun.py on the agreement set prints a saved-continuations line, not the accuracy template: "agree: 0.000
(0/400, noans 400)" was read as a parse failure twice on 2026-11-22 (the task has nothing to score; analysis/quality.py
compares the continuations with BF16's).  python3 -I tests/test_evalrun_agree.py"""
import pathlib
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
PORT = 18179
mock = subprocess.Popen([sys.executable, "-I", str(ROOT / "tests/mock_server.py"), "--port", str(PORT)])
bad = []
try:
    time.sleep(1.0)
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, "-I", str(ROOT / "bench/evalrun.py"), "--base", f"http://127.0.0.1:{PORT}", "--model", "m",
                            "--sets", str(ROOT / "data/eval/agree_400.jsonl"), "--limit", "6", "--label", "t_agree",
                            "--out", f"{td}/a.json"], capture_output=True, text=True, timeout=120)
        out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        if r.returncode != 0:
            bad.append(f"rc={r.returncode}: {r.stderr[-300:]}")
        elif "continuations saved" not in out or "0.000 (0/" in out or "6/6" not in out:
            bad.append(f"summary line: {out!r}")
finally:
    mock.terminate()
print("FAIL\n  " + "\n  ".join(bad) if bad else "OK: the agreement set prints 'N/N continuations saved', not an accuracy")
sys.exit(1 if bad else 0)
