#!/usr/bin/env python3
"""Per-cell sandbox CPU and GPU clock for a run: joins the run's cpumon.csv (per second) and raw AMD gpumon.csv (amd-smi,
every ~2 s) to every protocol cell's time window, so a pass or fail can be read as GPU-bound or CPU-starved (lead's condition
for the small-host claim).

Two cpumon.csv formats:
- box/cgmon.py (from 2026-11-22 09:22Z on): the sandbox's cgroup, i.e. cores used against the cpu.max quota and the 100 ms
  periods in which the quota ran out (throttling = starvation). This is the one to read.
- box/cpumon.py (earlier): /proc/stat, which in a Daytona sandbox is the 192-CPU HOST, so its busy % says nothing about the box;
  only its per-process role columns are valid.

Cell windows:
- their client (vllm bench serve result JSON): [date - duration, date], where `date` is the save time (UTC on the box);
- our client (traffic/1): [t_start + warmup, t_start + warmup + window_s].

  python3 -I recipes/mi355x/tools/cpu_cells.py results/<label>/<stamp> [...]
"""
from __future__ import annotations

import csv
import glob
import json
import pathlib
import statistics as st
import sys
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]


def ts(s, fmt):
    return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()


def load_cg(rd):
    f = rd / "cpumon.csv"
    if not f.exists() or "quota_cores" not in f.open().readline():
        return []
    rows = []
    for r in csv.DictReader(open(f)):
        try:
            rows.append((ts(r["t_utc"], "%Y-%m-%dT%H:%M:%S"), float(r["quota_cores"]), float(r["cg_used_cores"]), int(r["throttled_periods"]),
                         int(r["periods"]), float(r["throttled_ms"]), float(r["engine_pct"]), float(r["api_pct"]), float(r["client_pct"])))
        except (KeyError, ValueError):
            continue
    return rows


def load_cpu(rd):
    f = rd / "cpumon.csv"
    if not f.exists() or "quota_cores" in f.open().readline():
        return []
    rows = []
    for r in csv.DictReader(open(f)):
        try:
            rows.append((ts(r["t_utc"], "%Y-%m-%dT%H:%M:%S"), float(r["host_busy_pct"]), float(r["host_busy_cores"]), int(r["ncpu"]),
                         float(r["engine_pct"]), float(r["api_pct"]), float(r["client_pct"]), int(r["runq"])))
        except (KeyError, ValueError):
            continue
    return rows


def load_clk(rd):
    f = rd / "gpumon.csv"
    out, t = [], None
    if not f.exists():
        return out
    for row in csv.reader(open(f, errors="replace")):
        if row and row[0].startswith("T "):
            try:
                t = ts(row[0][2:].strip()[:19], "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                t = None
        elif t and len(row) >= 55 and row[0] == "0":
            try:
                if float(row[1]) >= 50:
                    out.append((t, st.mean(float(row[i]) for i in range(15, 51, 5)), float(row[9])))
            except ValueError:
                pass
    return out


def windows(rd):
    for ps in sorted(rd.glob("protocol_*/protocol_summary.json")):
        for cf in sorted(ps.parent.glob("cells/*.json")):
            j = json.loads(cf.read_text())
            if "date" in j and "duration" in j:
                end = ts(j["date"], "%Y%m%d-%H%M%S")
                yield ps.parent.name, cf.stem, end - j["duration"], end
            elif j.get("schema") == "traffic/1" and j.get("t_start"):
                t0 = j["t_start"]
                t0 = ts(t0.rstrip("Z")[:19], "%Y-%m-%dT%H:%M:%S") if isinstance(t0, str) else float(t0)
                w0 = t0 + j["args"].get("warmup", 0)
                yield ps.parent.name, cf.stem, w0, w0 + j["summary"]["window_s"]


def main_cg(rd, cg, clk):
    for proto, cell, a, b in windows(rd):
        c = [r for r in cg if a <= r[0] <= b]
        g = [r for r in clk if a <= r[0] <= b]
        if c:
            u = sorted(r[2] for r in c)
            thr, per = sum(r[3] for r in c), sum(r[4] for r in c)
            cs = (f"{st.mean(u):.2f} / {u[int(0.95 * (len(u) - 1))]:.2f} / {u[-1]:.2f} of {c[0][1]:.0f}",
                  f"{100 * thr / max(1, per):.1f}% ({sum(1 for r in c if r[3])} s)", f"{st.mean(r[6] for r in c):.0f}",
                  f"{st.mean(r[7] for r in c):.0f}", f"{st.mean(r[8] for r in c):.0f}")
        else:
            cs = ("no samples", "-", "-", "-", "-")
        gs = (f"{st.median(r[1] for r in g):.0f}", f"{st.median(r[2] for r in g):.0f}") if g else ("-", "-")
        print(f"| {rd.parent.name}/{rd.name} | {proto} | {cell} ({len(c)} s) | {' | '.join(cs)} | {gs[0]} | {gs[1]} |")


def main(dirs):
    cg_dirs = [d for d in dirs if load_cg(pathlib.Path(d))]
    if cg_dirs:
        print("| run | protocol | cell | sandbox cores used mean / p95 / max of quota | quota-throttled periods (seconds with any) "
              "| engine proc % of 1 core | API server % | client % | GFX MHz med | power W med |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for d in cg_dirs:
            rd = pathlib.Path(d)
            main_cg(rd, load_cg(rd), load_clk(rd))
    dirs = [d for d in dirs if d not in cg_dirs]
    if not dirs:
        return
    print("| run | protocol | cell | host CPU busy % mean / p95 / max (cores) | engine proc % of 1 core, mean / p95 | client % | runq mean | GFX MHz med | power W med |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in dirs:
        rd = pathlib.Path(d)
        cpu, clk = load_cpu(rd), load_clk(rd)
        for proto, cell, a, b in windows(rd):
            c = [r for r in cpu if a <= r[0] <= b]
            g = [r for r in clk if a <= r[0] <= b]
            if c:
                hb = sorted(r[1] for r in c)
                eg = sorted(r[4] for r in c)
                cs = (f"{st.mean(hb):.0f} / {hb[int(0.95 * (len(hb) - 1))]:.0f} / {hb[-1]:.0f} ({st.mean(r[2] for r in c):.2f} of {c[0][3]})",
                      f"{st.mean(eg):.0f} / {eg[int(0.95 * (len(eg) - 1))]:.0f}", f"{st.mean(r[6] for r in c):.0f}", f"{st.mean(r[7] for r in c):.1f}")
            else:
                cs = ("no samples", "-", "-", "-")
            gs = (f"{st.median(r[1] for r in g):.0f}", f"{st.median(r[2] for r in g):.0f}") if g else ("-", "-")
            print(f"| {rd.parent.name}/{rd.name} | {proto} | {cell} ({len(c)} s) | {cs[0]} | {cs[1]} | {cs[2]} | {cs[3]} | {gs[0]} | {gs[1]} |")


if __name__ == "__main__":
    main(sys.argv[1:])
