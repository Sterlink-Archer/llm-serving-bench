#!/usr/bin/env python3
"""This recipe's own Daytona spend: every run whose label starts with ik (Inkling-2-Mini, phase 2), from the shared ledger (infra/ledger.jsonl).
A sandbox costs $/h x (delete time, or now while it is still alive), the same rule as run.py's ledger_spend_estimate.

  python3 -I recipes/inkling/tools/spend.py          # total, then one line per sandbox
"""
from __future__ import annotations

import json
import pathlib
import time
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[3]


def ts(s):
    return datetime.strptime(s[:19] + "+0000", "%Y-%m-%dT%H:%M:%S%z").timestamp()


def main():
    creates, deletes = {}, {}
    for line in (ROOT / "infra" / "ledger.jsonl").read_text().splitlines():
        r = json.loads(line)
        lab = str(r.get("run") or r.get("run_id") or "")
        if r.get("event") == "create" and lab.startswith("ik"):
            creates[r["id"]] = r
        elif r.get("event") == "delete" and r.get("id"):
            deletes[r["id"]] = r
    now = time.time()
    tot, rows = 0.0, []
    for sid, c in creates.items():
        usd_h = c["usd_per_h"]["total"]
        t0 = ts(c["t"])
        d = deletes.get(sid)
        t1 = ts(d["t"]) if d else now
        usd = (usd_h or 0) * (t1 - t0) / 3600
        tot += usd
        rows.append(f"  {c['run_id']}: {sid[:8]} ${usd:.2f} ({(t1 - t0) / 60:.0f} min{'' if d else ', ALIVE'})")
    print(f"ik* spend: ${tot:.2f} over {len(creates)} sandboxes")
    print("\n".join(sorted(rows)))


if __name__ == "__main__":
    main()
