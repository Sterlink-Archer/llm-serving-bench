#!/usr/bin/env python3
"""The claim rate is the top of the MONOTONIC prefix (analysis/screen_report.py monotonic_claim): every cell of every tested
rung up to it passes. Lead, 2026-11-22: ikC_pro6000_mixed_dspark column B passed 1.25x, failed 1.35x on seed 7 (reproduced),
passed 1.5x; the old highest-all-pass rule claimed 1.5x.  python3 -I tests/test_claim_monotonic.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import screen_report as SR  # noqa: E402


def ladder(spec):
    return {rate: [dict(slo_pass=p, seed=sd) for sd, p in cells] for rate, cells in spec.items()}


bad = []
best, ff, above = SR.monotonic_claim(ladder({1.25: [(7, True), (8, True)], 1.35: [(7, False), (8, True)], 1.5: [(7, True), (8, True)]}))
if not (best and best[0] == 1.25 and ff == 1.35 and above == 1.5):
    bad.append(f"pass 1.25 / fail 1.35 / pass 1.5 -> {best and best[0]}, first fail {ff}, above {above} (want 1.25, 1.35, 1.5)")
best, ff, above = SR.monotonic_claim(ladder({3.25: [(7, True), (8, True)], 3.5: [(7, True), (8, True)], 3.75: [(7, False), (8, False)]}))
if not (best and best[0] == 3.5 and ff == 3.75 and above is None):
    bad.append(f"monotonic ladder -> {best and best[0]}, {ff}, {above} (want 3.5, 3.75, None)")
best, ff, above = SR.monotonic_claim(ladder({3.25: [(7, False), (8, True)], 3.5: [(7, True), (8, True)]}))
if not (best is None and ff == 3.25 and above == 3.5):
    bad.append(f"lowest rung fails -> {best and best[0]}, {ff}, {above} (want None, 3.25, 3.5)")
print("FAIL\n  " + "\n  ".join(bad) if bad else "OK: claim = top of the monotonic prefix; a higher pass above a failing rung is reported, not claimed")
sys.exit(1 if bad else 0)
