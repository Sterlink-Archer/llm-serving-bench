#!/usr/bin/env python3
"""One line per (run, server, rung) for the MI355X recipe screens, read from the run's own files (never typed):
protocol_<server>*/protocol_summary.json, gate_<server>_*.json, startup_facts_<server>.json, metrics_<server>.prom.

  python3 -I recipes/mi355x/tools/screen.py [label-glob ...]     # default: mx* and the s2 reproduction
MTP acceptance per run = 1 + accepted / drafts from the engine's cumulative /metrics at engine stop (whole run: gate,
prefill probe and every rung together), so it is a run-level figure, not per rung.
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]


def prom(path):
    v = {}
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            m = re.match(r"^(vllm:spec_decode_num_(?:accepted|draft)_tokens_total|vllm:spec_decode_num_drafts_total)(?:\{[^}]*\})? ([0-9.e+]+)$", line)
            if m:
                v[m.group(1)] = v.get(m.group(1), 0.0) + float(m.group(2))
    d = v.get("vllm:spec_decode_num_drafts_total")
    a = v.get("vllm:spec_decode_num_accepted_tokens_total")
    dt = v.get("vllm:spec_decode_num_draft_tokens_total")
    return (1 + a / d if d else None), (a / dt if dt else None)


def main(globs):
    rows = []
    for g in globs:
        for rd in sorted(glob.glob(str(ROOT / "results" / g / "2*"))):
            rd = pathlib.Path(rd)
            for ps in sorted(rd.glob("protocol_*/protocol_summary.json")):
                s = json.loads(ps.read_text())
                srv = ps.parent.name.replace("protocol_", "")
                base = srv[:-5] if srv.endswith("_text") else srv
                gates = [json.loads(p.read_text()).get("gate") or {} for p in rd.glob(f"gate_*{base}*.json")] or \
                        [json.loads(p.read_text()).get("gate") or {} for p in rd.glob("gate_*.json")]
                sf = rd / f"startup_facts_{base}.json"
                kv = json.loads(sf.read_text()).get("kv_cache_tokens") if sf.exists() else None
                al, ar = prom(rd / f"metrics_{base}.prom")
                pre = (s.get("prefill") or {}).get("prefill_tok_s")
                head = (f"{rd.parent.name}/{rd.name}:{srv} client={s.get('client')} gate={gates[0].get('acc') if gates else None} "
                        f"kv={kv} prefill={pre and round(pre)} accept_len={al and round(al, 3)} accept_rate={ar and round(ar, 3)} "
                        f"max_rate_at_slo={s.get('max_rate_at_slo_rps')}")
                rows.append(head)
                for r in s.get("ladder", []):
                    if r.get("skipped"):
                        rows.append(f"   rate {r['rate']}: skipped ({r['skipped'][:40]})")
                        continue
                    for sd, x in (r.get("results") or {}).items():
                        rows.append(f"   rate {r['rate']} seed {sd}: pass={x.get('slo_pass')} p95TTFT={x.get('ttft_p95_s') and round(x['ttft_p95_s'], 2)}"
                                    f" p50TTFT={x.get('ttft_p50_s') and round(x['ttft_p50_s'], 2)} TPOTp50={x.get('tpot_p50_s') and round(x['tpot_p50_s'] * 1000, 1)}ms"
                                    f" achieved={x.get('achieved_rps') and round(x['achieved_rps'], 3)} in_tok_s={x.get('in_tok_s') and round(x['in_tok_s'])}"
                                    f" err={x.get('errors')} accept={x.get('spec_accept_len')}")
    print("\n".join(rows))


if __name__ == "__main__":
    main(sys.argv[1:] or ["s2_mi355x_or_recipe_seed*", "mx*"])
