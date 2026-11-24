#!/usr/bin/env python3
"""The installed commit-msg hook (.git/hooks/commit-msg, a copy of bin/hooks/commit-msg) strips Claude co-author and
Claude-Session trailers (Maks 2026-11-13, for repos under his account: "remove my email and the claude things") and
keeps everything else. Install: cp bin/hooks/commit-msg .git/hooks/ && chmod +x .git/hooks/commit-msg
python3 -I tests/test_commit_msg_hook.py"""
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
common = pathlib.Path(subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
hook = (ROOT / common / "hooks" / "commit-msg") if not common.is_absolute() else common / "hooks" / "commit-msg"
bad = []
if not hook.exists():
    bad.append(f"{hook} is not installed (cp bin/hooks/commit-msg .git/hooks/)")
elif hook.read_bytes() != (ROOT / "bin/hooks/commit-msg").read_bytes():
    bad.append("installed hook differs from bin/hooks/commit-msg")
else:
    with tempfile.NamedTemporaryFile("w", suffix=".msg", delete=False) as f:
        f.write("Subject\n\nBody stays.\nCo-Authored-By: Someone Else <s@example.invalid>\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nco-authored-by: claude <x@y>\nClaude-Session: abc\n\n")
    subprocess.run(["sh", str(hook), f.name], check=True)
    out = pathlib.Path(f.name).read_text()
    if "Claude" in out or "claude" in out or "Body stays." not in out or "Someone Else" not in out or out.endswith("\n\n"):
        bad.append(f"hook output wrong: {out!r}")
print("FAIL\n  " + "\n  ".join(bad) if bad else "OK: commit-msg hook installed and strips Claude trailers only")
sys.exit(1 if bad else 0)
