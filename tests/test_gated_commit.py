#!/usr/bin/env python3
"""bin/gated_commit.sh never commits through a red gate, even when its output is piped (2026-11-22, twice: a piped
`commit_gate.sh | tail && git commit` committed through RED). Runs the real script in a throwaway repo with a stub gate.
python3 -I tests/test_gated_commit.py"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
bad = []
for gate_rc, want_commit in ((1, False), (0, True)):
    with tempfile.TemporaryDirectory() as td:
        t = pathlib.Path(td)
        (t / "bin").mkdir()
        shutil.copy(ROOT / "bin/gated_commit.sh", t / "bin/gated_commit.sh")
        (t / "bin/commit_gate.sh").write_text(f"#!/bin/bash\necho stub gate\nexit {gate_rc}\n")
        (t / "bin/commit_gate.sh").chmod(0o755)
        env = dict(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@example.invalid", PATH="/usr/bin:/bin:/run/current-system/sw/bin")
        subprocess.run(["git", "init", "-q"], cwd=t, check=True)
        (t / "old.txt").write_text("o\n")
        subprocess.run(["git", "add", "old.txt"], cwd=t, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "root"], cwd=t, check=True, env=env)
        subprocess.run(["git", "rm", "-q", "old.txt"], cwd=t, check=True)       # a deletion already staged, as in practice
        (t / "f.txt").write_text("x\n")
        r = subprocess.run("bash bin/gated_commit.sh -m change -- f.txt old.txt | tail -n 1", shell=True, cwd=t, env=env,
                           capture_output=True, text=True)
        n = int(subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=t, capture_output=True, text=True).stdout)
        tree = subprocess.run(["git", "ls-tree", "--name-only", "HEAD"], cwd=t, capture_output=True, text=True).stdout.split()
        if want_commit and tree != ["f.txt"]:
            bad.append(f"green gate: HEAD tree {tree} (want the new file added and the deleted file gone)")
        if (n == 2) != want_commit:
            bad.append(f"gate rc {gate_rc}: commits {n - 1} (want {'1' if want_commit else '0'}); output {r.stdout.strip()!r}")
print("FAIL\n  " + "\n  ".join(bad) if bad else "OK: gated_commit commits only after a green gate, even when piped")
sys.exit(1 if bad else 0)
