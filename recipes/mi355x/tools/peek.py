#!/usr/bin/env python3
"""Read-only look at a live recipe box: the per-cell lines its protocol steps have printed so far.
Finds the sandbox from the ledger's create event for the run label (latest one), then tails /root/or/protocol_*.log.

  python3 -I recipes/mi355x/tools/peek.py mx5_or_mtp3_seqs96 [more labels...]
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "infra"))
import daytona as dt  # noqa: E402


def main(labels):
    creates, deleted = {}, set()
    for line in (ROOT / "infra" / "ledger.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("event") == "create" and r.get("run") in labels:
            creates[r["run"]] = r
        elif r.get("event") == "delete":
            deleted.add(r.get("id"))
    for lab in labels:
        c = creates.get(lab)
        if not c or c["id"] in deleted:
            print(f"== {lab}: no live sandbox")
            continue
        try:
            box = dt.Box(dt.get_sandbox(c["id"]))
            out = box.exec("bash -c 'grep -h \"\\[vllmbench\\|^\\[r\\|prefill=\\|max_rate\" /root/or/protocol_*.log 2>/dev/null | tail -n 8'", timeout=30)["out"]
            print(f"== {lab} ({c['id'][:8]})\n{out.strip()}")
        except Exception as e:  # noqa: BLE001
            print(f"== {lab}: {e!r}"[:200])


if __name__ == "__main__":
    main(sys.argv[1:])
