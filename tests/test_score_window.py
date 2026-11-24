#!/usr/bin/env python3
"""The claim scoring window (Grok review of the C1 claim, caveat 1): bench/protocol.py score_cell scores the LAST 300 s of a
cell, censors requests without a first token as +inf, and analysis/screen_report.py cell_flaws refuses a scored window < 300 s.
  python3 -I tests/test_score_window.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "analysis"))
import protocol as PR  # noqa: E402
import screen_report as SR  # noqa: E402

bad = []
# old 300 s cell: window 0..300, warmup requests count
reqs = [dict(t_send=float(t), ttft=1.0, e2e=5.0, status=200) for t in range(0, 300, 1)]
sc = PR.score_cell(reqs, 300.0, 4.0)
if (sc["scored_from_s"], sc["scored_window_s"], sc["requests_in_window"]) != (0.0, 300.0, 300) or not sc["slo_pass"]:
    bad.append(f"old cell window {sc}")
# new 360 s cell: window 60..360
sc = PR.score_cell([dict(t_send=float(t), ttft=1.0, e2e=5.0, status=200) for t in range(0, 360)], 360.0, 4.0)
if (sc["scored_from_s"], sc["scored_window_s"]) != (60.0, 300.0):
    bad.append(f"new cell window {sc}")
# censoring: 10% of requests never get a first token -> p95 is +inf (fail); lower bound stays finite
cens = [dict(t_send=float(t), ttft=(None if t % 10 == 0 else 1.0), e2e=(None if t % 10 == 0 else 5.0), status=200, aborted=(t % 10 == 0))
        for t in range(0, 300)]
sc = PR.score_cell(cens, 300.0, 4.0)
if sc["slo_pass"] or sc["ttft_p95_s"] != float("inf") or sc["ttft_censored"] != 30 or not (sc["ttft_p95_lower_s"] < float("inf")):
    bad.append(f"censoring {sc}")
# cell_flaws: a 240 s scored window is refused, 300 s accepted
ok_cell = dict(scored_window_s=300.0, prompt_tok_xcheck_pass=True)
if SR.cell_flaws(dict(ok_cell, scored_window_s=240.0), "A") == [] or SR.cell_flaws(ok_cell, "A") != []:
    bad.append(f"cell_flaws window: {SR.cell_flaws(dict(ok_cell, scored_window_s=240.0), 'A')} / {SR.cell_flaws(ok_cell, 'A')}")
print("FAIL\n  " + "\n  ".join(bad) if bad else "OK: 300 s scored window (old and new cells), censoring, 240 s refused")
sys.exit(1 if bad else 0)
