#!/usr/bin/env python3
"""GLM phase-3 Daytona spend: every run whose label starts with glm, from the shared ledger (infra/ledger.jsonl).
A sandbox costs $/h x (delete time, or now while it is still alive), the same rule as run.py's ledger_spend_estimate.

  python3 -I recipes/mi355x/tools/spend.py          # total, then one line per sandbox
"""
from __future__ import annotations

import json
import pathlib
import time
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[3]
# only runs whose label has a config in this recipe's configs/ (other sessions may share the label prefix)
import subprocess  # noqa: E402
# only this recipe's own runs: configs committed under recipes/<this>/configs (other sessions may drop untracked
# configs with the same label prefix into the folder, e.g. mxG1_* GEMM-headroom boxes on 2026-11-22)
OWN = {pathlib.Path(x).stem for x in subprocess.run(['git', '-C', str(ROOT), 'ls-files', 'recipes/glm/configs'],
        capture_output=True, text=True).stdout.split()}


def ts(s):
    return datetime.strptime(s[:19] + "+0000", "%Y-%m-%dT%H:%M:%S%z").timestamp()


def main():
    creates, deletes = {}, {}
    for line in (ROOT / "infra" / "ledger.jsonl").read_text().splitlines():
        r = json.loads(line)
        lab = str(r.get("run") or r.get("run_id") or "")
        if r.get("event") == "create" and lab.split("/")[0] in OWN:
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
    print(f"glm* spend: ${tot:.2f} over {len(creates)} sandboxes")
    print("\n".join(sorted(rows)))


if __name__ == "__main__":
    main()
