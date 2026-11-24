#!/usr/bin/env python3
"""bin/secret_scan.sh on staged files in a throwaway repo: code decorators that look like addresses (@pytest.mark...,
@triton.jit inside test files embedded in configs) and the reserved @example.invalid domain pass; a real-looking personal
address and an absolute /home/<user>/ path still fail (2026-11-22: the up1/up2 configs carried both kinds; the paths were
made repo-relative, the decorators allow-listed).  python3 -I tests/test_secret_scan.py"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def scan(content):
    with tempfile.TemporaryDirectory() as td:
        t = pathlib.Path(td)
        (t / "bin").mkdir()
        shutil.copy(ROOT / "bin/secret_scan.sh", t / "bin/secret_scan.sh")
        subprocess.run(["git", "init", "-q"], cwd=t, check=True)
        (t / "f.json").write_text(content)
        subprocess.run(["git", "add", "f.json"], cwd=t, check=True)
        return subprocess.run(["bash", "bin/secret_scan.sh"], cwd=t, capture_output=True, text=True).returncode


bad = []
ok_text = '{"cmd": "cat > t.py <<EOF\\n@pytest.mark.parametrize(\\"x\\", [1])\\n@pytest.mark.skipif(True, reason=\\"\\")\\n@triton.jit\\ndef k(): pass\\nEOF", "who": "t@example.invalid"}\n'
if scan(ok_text) != 0:
    bad.append("decorators / example.invalid were flagged")
# the two leak fixtures are assembled at run time, so this file itself stays clean under the scanner it tests
for leak in ('{"owner": "jane.doe' + chr(64) + 'corp-mail.com"}\n', '{"run": "/' + 'home/someone/projects/x/results/r"}\n'):
    if scan(leak) == 0:
        bad.append(f"a real leak passed: {leak.strip()}")
print("FAIL\n  " + "\n  ".join(bad) if bad else "OK: decorators and @example.invalid pass; a personal address and a /home path still fail")
sys.exit(1 if bad else 0)
