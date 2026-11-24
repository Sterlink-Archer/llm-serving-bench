#!/usr/bin/env python3
"""Assemble the evidence tables for WRITEUP.md from results/ (nothing is typed by hand):
  results/COST_TABLE.md      every matrix cell priced vs OpenRelay Sync / cached / Batch / OpenRouter (analysis/cost.py)
  results/QUALITY_TABLE.md   every eval vs the BF16 reference (analysis/quality.py)
  results/HEADLINE.md        per config: best 1k/1k cost ratio (any latency, and interactive SLO), 8k/1k, prefix 90%, CIU,
                             cold start (download + server ready), run cost; plus the overall winner per bar.
  results/RUNS.md            one line per run: sandbox id, GPU, $/h, elapsed, estimated cost, error/eviction.
  results/COLUMN_C.md        cached traffic (supplementary, not OpenRelay's method): every cell, both cache reads (analysis/column_c.py)

  python3 -I analysis/report.py
"""
from __future__ import annotations

import glob
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def fmt(x, n=2):
    return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def main():
    py = sys.executable
    subprocess.run([py, "-I", str(ROOT / "analysis/cost.py"), "--md", str(RES / "COST_TABLE.md"), "--json", str(RES / "cost_rows.json")],
                   check=True, capture_output=True)
    subprocess.run([py, "-I", str(ROOT / "analysis/quality.py"), "--md", str(RES / "QUALITY_TABLE.md")], check=True, capture_output=True)
    subprocess.run([py, "-I", str(ROOT / "analysis/screen_report.py")], check=True, capture_output=True)   # PROTOCOL_TABLE.md, protocol_rows.json
    subprocess.run([py, "-I", str(ROOT / "analysis/journal.py"), "render"], check=True, capture_output=True)  # docs/JOURNAL.md
    subprocess.run([py, "-I", str(ROOT / "analysis/column_c.py")], check=True, capture_output=True)   # COLUMN_C.md, column_c.json
    subprocess.run([py, "-I", str(ROOT / "analysis/box_counts.py")], check=True, capture_output=True)  # BOX_COUNTS.md, box_counts.json
    subprocess.run([py, "-I", str(ROOT / "bin/build_readme.py")], check=True, capture_output=True)   # README.md from claims.json
    rows = json.loads((RES / "cost_rows.json").read_text())

    runs = []
    for rf in sorted(glob.glob(str(RES / "*/*/run.json"))):
        r = json.loads(pathlib.Path(rf).read_text())
        srv = r.get("servers", {})
        runs.append(dict(label=r["label"], stamp=pathlib.Path(rf).parent.name, sandbox=r.get("sandbox_id"), usd_h=r["usd_per_h"]["total"],
                         elapsed_min=r.get("elapsed_s", 0) / 60, est_usd=r.get("est_cost_usd"), error=r.get("error"), evicted=r.get("evicted"),
                         servers={k: dict(dl_s=(v.get("download") or {}).get("secs"), ready_s=(v.get("serve") or {}).get("ready_s"),
                                          ready=(v.get("serve") or {}).get("ready"), end=v.get("end")) for k, v in srv.items()}))
    L = ["| run | stamp | sandbox | $/h | minutes | est $ | servers (download s / ready s / end) | error |", "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        sv = "; ".join(f"{k}: {fmt(v['dl_s'], 0)} / {fmt(v['ready_s'], 0)} / {v['end'] or ('NOT READY' if v['ready'] is False else '-')}"
                       for k, v in r["servers"].items())
        L.append(f"| {r['label']} | {r['stamp']} | {r['sandbox']} | {r['usd_h']:.2f} | {r['elapsed_min']:.1f} | {fmt(r['est_usd'])} | {sv} | "
                 f"{(r['error'] or '')[:120]}{' EVICTED' if r['evicted'] else ''} |")
    L.append(f"\nTotal estimated: ${sum(r['est_usd'] or 0 for r in runs):.2f} over {len(runs)} runs (Daytona-metered: infra/spend.py).")
    (RES / "RUNS.md").write_text("\n".join(L) + "\n")

    def best(pred, key="ratio_sync", slo=False):
        xs = [r for r in rows if pred(r) and r.get(key) is not None and (r["slo_interactive"] or not slo)]
        return min(xs, key=lambda r: r[key]) if xs else None
    for r in rows:
        r["server"] = r["server"] or "?"     # an unattributed cell is shown under "?", never crashes the sort or goes missing
    cfgs = sorted({(r["run"], r["server"], r["gpu"]) for r in rows})
    H = ["| config (run / server) | gpu | $/h | 1k/1k best ratio vs OR-Sync (C) | 1k/1k interactive (C) | 8k/1k best (C) | prefix 90% (hit) | "
         "CIU 16 req/s | vs OR-Batch best | vs OpenRouter best | peak out tok/s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, srv, gpu in cfgs:
        mine = lambda r: r["run"] == run and r["server"] == srv  # noqa: E731
        a = best(lambda r: mine(r) and r["cell"].startswith("s1k_1k"))
        ai = best(lambda r: mine(r) and r["cell"].startswith("s1k_1k"), slo=True)
        b = best(lambda r: mine(r) and r["cell"].startswith("s8k_1k"))
        p = best(lambda r: mine(r) and r["cell"].startswith("prefix_8k") and "0.9" in r["cell"] and "0.99" not in r["cell"], key="ratio_sync_cached")
        c = best(lambda r: mine(r) and r["cell"].startswith("ciu_open"))
        bb = best(lambda r: mine(r), key="ratio_batch")
        orr = best(lambda r: mine(r), key="ratio_openrouter")
        pk = max((r["out_tok_s"] for r in rows if mine(r)), default=None)
        usd = next((r["usd_h"] for r in rows if mine(r)), None)
        H.append(f"| {run} / {srv} | {gpu} | {fmt(usd)} | {fmt(a and a['ratio_sync'])} ({a and a['C']}) | {fmt(ai and ai['ratio_sync'])} ({ai and ai['C']}) | "
                 f"{fmt(b and b['ratio_sync'])} ({b and b['C']}) | {fmt(p and p['ratio_sync_cached'])} | {fmt(c and c['ratio_sync'])} | "
                 f"{fmt(bb and bb['ratio_batch'])} ({bb and bb['cell']}) | {fmt(orr and orr['ratio_openrouter'])} | {fmt(pk, 0)} |")
    (RES / "HEADLINE.md").write_text("ratio = our cost / bar at 100% utilisation (= break-even utilisation); < 1 means cheaper. "
                                     "Source rows: results/cost_rows.json\n\n" + "\n".join(H) + "\n")
    print((RES / "HEADLINE.md").read_text())
    print((RES / "RUNS.md").read_text())


if __name__ == "__main__":
    main()
