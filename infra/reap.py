#!/usr/bin/env python3
"""OpenRelay reaper: nothing labelled project=openrelay may outlive its deadline.

  python3 -I infra/reap.py            list project sandboxes, delete the ones past deadline (+2 min grace) and orphans
  python3 -I infra/reap.py --dry-run  list and say what WOULD be deleted
  python3 -I infra/reap.py --all      delete every project sandbox now (panic button)

Delete rules (any one is enough):
  1. label deadline (unix ts, set at create = create time + ttl) is more than 2 minutes in the past;
  2. the sandbox is older than --max-age-min (default 240) whatever its labels say;
  3. it carries project=openrelay but its id is not in infra/ledger.jsonl (created outside our launcher);
  4. it was evicted / errored (state error|build_failed or spotEvictedAt set) and still exists.
Other projects' sandboxes are listed (GPU ones flagged) but never touched.
Runs every 5 minutes from the user timer openrelay-reap.timer (infra/systemd/). Exit 3 = a project GPU sandbox is alive.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import daytona as dt  # noqa: E402


def age_min(s) -> float:
    c = s.get("createdAt") or ""
    try:
        return (time.time() - datetime.datetime.fromisoformat(c.replace("Z", "+00:00")).timestamp()) / 60
    except ValueError:
        return -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-age-min", type=float, default=240)
    a = ap.parse_args()
    known = set()
    if dt.LEDGER.exists():
        for line in dt.LEDGER.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "create":
                known.add(r["id"])
    now = time.time()
    alive_gpu = 0
    for s in dt.list_sandboxes():
        lab = s.get("labels") or {}
        mine = lab.get("project") == dt.PROJECT_LABEL
        m = age_min(s)
        line = (f"{dt.now_iso()} {s['id']} project={lab.get('project')} run={lab.get('run')} state={s.get('state')} "
                f"gpu={s.get('gpu')} {s.get('gpuType') or ''} spot={s.get('spot')} age_min={m:.0f}")
        if not mine:
            print(line + ("  (other project, GPU)" if s.get("gpu") else "  (other project)"))
            continue
        why = None
        dl = lab.get("deadline", "")
        if a.all:
            why = "--all"
        elif dl.isdigit() and now > int(dl) + 120:
            why = f"past deadline by {(now - int(dl)) / 60:.1f} min"
        elif m > a.max_age_min:
            why = f"older than {a.max_age_min} min"
        elif s["id"] not in known:
            why = "not in ledger (orphan)"
        elif str(s.get("state", "")).lower() in ("error", "build_failed") or s.get("spotEvictedAt"):
            why = f"dead: state={s.get('state')} evicted={s.get('spotEvictedAt')}"
        if s.get("gpu") and not why:
            alive_gpu += 1
        print(line + (f"  REAP: {why}" if why else f"  ok, deadline in {(int(dl) - now) / 60:.1f} min" if dl.isdigit() else ""))
        if why and not a.dry_run:
            try:
                dt.delete_sandbox(s["id"], why=f"reaper: {why}")
                print(f"{dt.now_iso()} deleted {s['id']}")
            except Exception as e:  # noqa: BLE001
                print(f"{dt.now_iso()} DELETE FAILED {s['id']}: {e!r}"[:300])
    print(f"{dt.now_iso()} project GPU sandboxes alive: {alive_gpu}")
    return 3 if alive_gpu else 0


if __name__ == "__main__":
    sys.exit(main())
