#!/usr/bin/env python3
"""Run one shell command on a live recipe box (found by run label from the ledger's latest create event, as peek.py does).
For reading box state and starting side samplers; never for changing a run's steps.

  python3 -I recipes/mi355x/tools/boxsh.py LABEL 'command'
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "infra"))
import daytona as dt  # noqa: E402


def main(lab, cmd):
    c, deleted = None, set()
    for line in (ROOT / "infra" / "ledger.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("event") == "create" and r.get("run") == lab:
            c = r
        elif r.get("event") == "delete":
            deleted.add(r.get("id"))
    if not c or c["id"] in deleted:
        sys.exit(f"{lab}: no live sandbox")
    box = dt.Box(dt.get_sandbox(c["id"]))
    r = box.exec("bash -lc '" + cmd.replace("'", "'\\''") + "'", timeout=120)
    print(r["out"].rstrip())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
