#!/usr/bin/env python3
"""Print every rung of every protocol_summary.json under the given run dirs (screen triage; claims come from
results/claims.json).  python3 -I recipes/inkling/tools/ladder.py results/ik1_*/2026*"""
import glob
import json
import sys

for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/protocol_*/protocol_summary.json")):
        s = json.load(open(f))
        print(f"== {f}  prefill={round((s.get('prefill') or {}).get('prefill_tok_s') or 0)} max_rate_at_slo={s.get('max_rate_at_slo_rps')}")
        for r in s["ladder"]:
            if r.get("skipped"):
                continue
            g = lambda k, n=2: (round(r[k], n) if isinstance(r.get(k), (int, float)) else r.get(k))  # noqa: E731
            print(f"  {r['multiple']:.2f}x {r['rate']:.3f} {'PASS' if r.get('slo_pass') else 'fail'} ach={g('achieved_rps')} "
                  f"ttft50={g('ttft_p50_s')} ttft95={g('ttft_p95_s')} tpot50={g('tpot_p50_s', 3)} tpot95={g('tpot_p95_s', 3)} "
                  f"err={r.get('errors')} in={g('prompt_tokens_mean', 0)} out={g('completion_tokens_mean', 0)} in_tok_s={g('in_tok_s', 0)}")
