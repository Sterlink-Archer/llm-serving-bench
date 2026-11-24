#!/usr/bin/env python3
"""What Daytona itself metered for this project's sandboxes (ids from infra/ledger.jsonl), next to our own estimate.
Reads the analytics service named in GET /api/config (same method as the earlier t3990 dt_spend.py). Usage can lag.

  python3 -I infra/spend.py [--from 2026-11-22T00:00:00Z]      -> prints, and writes infra/spend_<stamp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import daytona as dt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", default="2026-11-22T00:00:00Z")
    ap.add_argument("--to", default=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 60)))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    creates, deletes = {}, {}
    for line in dt.LEDGER.read_text().splitlines() if dt.LEDGER.exists() else []:
        r = json.loads(line)
        if r.get("event") in ("create", "delete") and r.get("id"):   # phase/evicted/spend/milestone rows carry no sandbox id
            (creates if r["event"] == "create" else deletes)[r["id"]] = r
    cfg = dt._req("GET", dt.API + "/config")
    an = (cfg or {}).get("analyticsApiUrl") or (cfg or {}).get("analytics_api_url")
    out = dict(read_at=dt.now_iso(), frm=a.frm, to=a.to, analytics=bool(an), sandboxes={})
    q = urllib.parse.urlencode({"from": a.frm, "to": a.to})
    total_est = 0.0
    for sid, c in creates.items():
        d = deletes.get(sid)
        est = None
        if d:
            t0 = time.mktime(time.strptime(c["t"], "%Y-%m-%dT%H:%M:%SZ"))
            t1 = time.mktime(time.strptime(d["t"], "%Y-%m-%dT%H:%M:%SZ"))
            est = round((t1 - t0) / 3600 * c["usd_per_h"]["total"], 4)
            total_est += est
        rec = dict(run=c["run"], gpu=c["gpu"], created=c["t"], deleted=(d or {}).get("t"), our_estimate_usd=est)
        if an:
            try:
                rec["daytona_usage"] = dt._req("GET", an.rstrip("/") + f"/organization/{dt.ORG}/sandbox/{sid}/usage?{q}")
            except dt.ApiError as e:
                rec["daytona_usage"] = dict(status=e.status, body=str(e.body)[:200])
        out["sandboxes"][sid] = rec
    if an:
        try:
            out["org_aggregated"] = dt._req("GET", an.rstrip("/") + f"/organization/{dt.ORG}/usage/aggregated?{q}")
        except dt.ApiError as e:
            out["org_aggregated"] = dict(status=e.status, body=str(e.body)[:200])
    out["our_estimate_total_usd"] = round(total_est, 3)
    # Daytona's own metered dollars: the sum of `price` over each ledger sandbox's usage intervals (the org-wide
    # aggregate would include other projects' sandboxes). None until at least one interval is reported.
    met = [sum(float(x.get("price") or 0) for x in v["daytona_usage"]) for v in out["sandboxes"].values()
           if isinstance(v.get("daytona_usage"), list) and v["daytona_usage"]]
    out["metered_total_usd"] = round(sum(met), 4) if met else None
    out["metered_sandboxes"] = len(met)
    # Committed per-sandbox summary of Daytona's own metered charge (the raw spend_*.json snapshots are gitignored).
    # metered_usd_per_h = sum(price) / sum(hours) over the sandbox's closed usage intervals: the rate actually billed.
    def _ts(x):
        return time.mktime(time.strptime(str(x)[:19], "%Y-%m-%dT%H:%M:%S"))
    rows = {}
    for sid, v in out["sandboxes"].items():
        u = v.get("daytona_usage")
        if not isinstance(u, list) or not u:
            continue
        closed = [x for x in u if x.get("startAt") and x.get("endAt")]
        hrs = sum((_ts(x["endAt"]) - _ts(x["startAt"])) / 3600 for x in closed)
        usd = sum(float(x.get("price") or 0) for x in closed)
        c = creates.get(sid, {})
        rows[sid] = dict(run=v.get("run"), run_id=c.get("run_id"), gpu=v.get("gpu"), cpu=c.get("cpu"), mem=c.get("mem"), disk=c.get("disk"),
                         metered_usd=round(sum(float(x.get("price") or 0) for x in u), 6), closed_intervals=len(closed), open_intervals=len(u) - len(closed),
                         metered_hours=round(hrs, 5), metered_usd_per_h=round(usd / hrs, 4) if hrs > 0.005 else None,
                         published_usd_per_h=(c.get("usd_per_h") or {}).get("total"), our_estimate_usd=v.get("our_estimate_usd"))
    (dt.ROOT / "results" / "metered_usd.json").write_text(json.dumps(dict(
        schema="metered/1", read_at=out["read_at"], window=dict(frm=a.frm, to=a.to),
        source="Daytona analytics API GET /organization/{org}/sandbox/{id}/usage (sum of interval price), via infra/spend.py",
        metered_total_usd=out["metered_total_usd"], sandboxes=rows), indent=1))
    p = dt.ROOT / "infra" / f"spend_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    if not a.quiet:
        print(json.dumps(out, indent=1, default=str)[:5000])
    print(f"metered_total_usd={out['metered_total_usd']} est_total_usd={out['our_estimate_total_usd']} sandboxes={len(out['sandboxes'])}")
    print("wrote", p)


if __name__ == "__main__":
    main()
