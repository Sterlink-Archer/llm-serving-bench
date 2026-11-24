#!/usr/bin/env python3
"""Regression test for bench/matrix.py + bench/traffic.py against tests/mock_server.py (no GPU, ~2 min).
Guards the failures the first dry run caught (2026-11-22):
  1. cells reused the same request text, so a later "unique" cell hit the prefix cache of an earlier one (0.98 hits);
  2. closed-loop workers started in lockstep, so long-output cells measured a synchronized wave;
  3. mock listen backlog of 5 reset connections under a burst (harness counted them, mock fixed).
Asserts: unique cell repeated twice -> cache hit < 0.05 both times; 90% shared prefix -> 0.85..0.95; calibrated
prompt length within 3% of target; client output tok/s within 2% of the server's counter; no errors; staggered first
requests excluded from TTFT; knees present.
  python3 -I tests/test_matrix_mock.py
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
PORT = 18123


def main():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="or_mtest_"))
    spec = dict(name="test", defaults=dict(duration=8, warmup=2), slo=dict(ttft_p95_s=2.0, tpot_p50_ms=50), series=[
        dict(id="uniq_a", workload="synthetic", params=dict(input_tokens=2048, output_tokens=200), concurrency=8),
        dict(id="uniq_b", workload="synthetic", params=dict(input_tokens=2048, output_tokens=200), concurrency=8),
        dict(id="pfx90", workload="synthetic", params=dict(input_tokens=4096, output_tokens=64, prefix_ratio=0.9, prefix_pool=2), concurrency=8),
        dict(id="knee", workload="synthetic", params=dict(input_tokens=512, output_tokens=300), sweep=dict(concurrency=[4, 16, 64])),
        dict(id="ciu", workload="ciu_screen", rate=20, max_inflight=256),
        dict(id="chat", workload="console_chat", params=dict(system_tokens=4000, conversations=16, turns=3), concurrency=8),
    ])
    (tmp / "spec.json").write_text(json.dumps(spec))
    mock = subprocess.Popen([sys.executable, "-I", str(ROOT / "tests/mock_server.py"), "--port", str(PORT), "--knee", "16", "--itl", "0.004"])
    try:
        time.sleep(1.0)
        r = subprocess.run([sys.executable, "-I", str(ROOT / "bench/matrix.py"), "--spec", str(tmp / "spec.json"), "--model", "m",
                            "--base", f"http://127.0.0.1:{PORT}", "--metrics", f"http://127.0.0.1:{PORT}/metrics", "--out", str(tmp / "out")],
                           capture_output=True, text=True, timeout=600)
        print(r.stdout[-3000:], r.stderr[-2000:])
        assert r.returncode == 0, r.returncode
        summ = json.loads((tmp / "out/matrix_summary.json").read_text())
        rows = {x["cell"]: x for x in summ["rows"]}
        cell = lambda n: json.loads((tmp / "out/cells" / f"{n}.json").read_text())  # noqa: E731
        fails = []
        for n in ("uniq_a", "uniq_b"):
            if (rows[n]["srv_pc_hit"] or 0) > 0.05:
                fails.append(f"{n}: server prefix-cache hit {rows[n]['srv_pc_hit']:.2f} on unique prompts (cross-cell reuse)")
        h = rows["pfx90"]["srv_pc_hit"] or 0
        if not 0.85 <= h <= 0.95:
            fails.append(f"pfx90: hit {h:.2f} not in 0.85..0.95")
        d = cell("uniq_a")
        tgt, got = 2048, d["summary"]["prompt_tokens"]["p50"]
        if abs(got - tgt) / tgt > 0.03:
            fails.append(f"calibration: prompt p50 {got} vs target {tgt}")
        for n, x in rows.items():
            if x.get("skipped"):
                fails.append(f"{n}: skipped {x['skipped']}")
                continue
            if x["errors"]:
                fails.append(f"{n}: {x['errors']} errors")
            if x["srv_out_tok_s"] and abs(x["out_tok_s"] - x["srv_out_tok_s"]) / x["srv_out_tok_s"] > 0.02:
                fails.append(f"{n}: client {x['out_tok_s']:.0f} vs server {x['srv_out_tok_s']:.0f} tok/s")
        st = [q for q in cell("knee__concurrency=64")["requests"] if q["staggered"]]
        if not st:
            fails.append("knee c64: no staggered first requests")
        if "knee" not in summ.get("knees", {}):
            fails.append("knee series has no knee report")
        if fails:
            print("FAIL\n  " + "\n  ".join(fails))
            return 1
        print(f"OK: {len(rows)} cells; uniq hits {rows['uniq_a']['srv_pc_hit']}/{rows['uniq_b']['srv_pc_hit']}, pfx90 {h:.2f}, "
              f"prompt p50 {got}/{tgt}, knee {summ['knees']['knee']}")
        return 0
    finally:
        mock.terminate()
        mock.wait(5)


if __name__ == "__main__":
    sys.exit(main())
