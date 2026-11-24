#!/usr/bin/env python3
"""Column C (cached traffic, NOT OpenRelay's method) against tests/mock_server.py, which hashes 64-char prefix blocks like
vLLM and reports cached_tokens + vllm:prefix_cache_* counters (~1 min, no GPU).  python3 -I tests/test_cached_traffic.py
  1. shared_prefix: same pool entry = byte-identical prefix; realized cached share near the design share after warmup.
  2. agent_session: turns of a session are sequential, each re-sends the previous turn's prompt + the REAL reply, and
     later turns hit the cache; sessions arrive at rate / turns.
  3. protocol.py --column C records the column, the group tag and both cache reads; screen_report prices the realized
     mean request with cached input at the cached rate."""
import json
import pathlib
import random
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "analysis"))
import workloads as W  # noqa: E402

PORT = 18177
bad = []

# 1a. structure (no server): pool entries are identical text, different entries differ from their first block
ctx = dict(text=W.Text(cpt=4.0), seed=7, nonce="n0", shapes={})
p = dict(prefix_tokens=2000, prefix_pool=4, suffix_dist={"uniform": [100, 300]}, output_dist={"uniform": [5, 20]}, stratify=40)
specs = [W.shared_prefix(p, i, random.Random(i), ctx) for i in range(40)]
by = {}
for sp in specs:
    pre = sp["messages"][0]["content"].split("\n\n[request ")[0]
    by.setdefault(sp["prefix_id"], set()).add(pre)
if any(len(v) != 1 for v in by.values()) or len(by) < 3:
    bad.append(f"shared_prefix: pool entries not byte-identical or too few used: { {k: len(v) for k, v in by.items()} }")
if len({next(iter(v))[:40] for v in by.values()}) != len(by):
    bad.append("shared_prefix: two pool entries share their first block")
us = sorted(sp["target_in"] - 2000 for sp in specs)
if abs(sum(us) / len(us) - 200) > 3:
    bad.append(f"shared_prefix: stratified suffix mean {sum(us) / len(us)} (want 200)")

mock = subprocess.Popen([sys.executable, "-I", str(ROOT / "tests/mock_server.py"), "--port", str(PORT), "--knee", "32", "--itl", "0.003"])
try:
    time.sleep(1.0)
    with tempfile.TemporaryDirectory() as td:
        def traffic(workload, params, rate, dur, warm, out):
            cmd = [sys.executable, "-I", str(ROOT / "bench/traffic.py"), "--model", "m", "--base", f"http://127.0.0.1:{PORT}",
                   "--metrics", f"http://127.0.0.1:{PORT}/metrics", "--workload", workload, "--params", json.dumps(params),
                   "--rate", str(rate), "--duration", str(dur), "--warmup", str(warm), "--exact-arrivals", "--seed", "7",
                   "--out", out, "--label", workload]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if r.returncode != 0:
                bad.append(f"traffic {workload} rc={r.returncode}: {r.stderr[-400:]}")
                return None
            return json.load(open(out))

        # 1b. live: after warmup every pool entry is resident, so cached share ~ 2000 / (2000 + 200 + instruction/tags)
        d = traffic("shared_prefix", dict(p, stratify=None), 6, 20, 6, f"{td}/sp.json")
        if d:
            late = [x for x in d["requests"] if x["t_send"] and x["t_send"] > 6 and x["cached_tokens"] is not None and x["prompt_tokens"]]
            share = sum(x["cached_tokens"] for x in late) / max(1, sum(x["prompt_tokens"] for x in late))
            design = 2000 / (2000 + 200 + 30)
            if not (design - 0.08 < share <= design + 0.02):
                bad.append(f"shared_prefix live cached share {share:.3f} (design {design:.3f})")
            if (d["summary"].get("server_metrics") or {}).get("prefix_cache_hit_ratio") is None:
                bad.append("shared_prefix: no server prefix-cache counter read")

        # 2. agent sessions
        ap_ = dict(start_dist={"uniform": [800, 1200]}, tool_dist={"uniform": [100, 200]}, output_dist={"uniform": [5, 10]},
                   turns=3, think_s=[0.05, 0.1])
        d = traffic("agent_session", ap_, 6, 20, 4, f"{td}/ag.json")
        if d:
            rows = [x for x in d["requests"] if x["status"] == 200]
            sess = {}
            for x in rows:
                sess.setdefault(x["session"], []).append(x)
            n_s = len(sess)
            if not (0.6 * 6 / 3 * 20 <= n_s <= 6 / 3 * 20 + 1):
                bad.append(f"agent_session: {n_s} sessions (want ~{6 / 3 * 20:.0f}: rate / turns x duration)")
            seq_bad = hit_bad = grow_bad = 0
            for sid, xs in sess.items():
                xs.sort(key=lambda x: x["turn"])
                if [x["turn"] for x in xs] != list(range(1, len(xs) + 1)):
                    seq_bad += 1
                for a_, b_ in zip(xs, xs[1:]):
                    if b_["t_send"] < a_["t_send"] + a_["e2e"] - 1e-3:
                        seq_bad += 1
                    if not b_["cached_tokens"] or b_["cached_tokens"] < 0.8 * a_["prompt_tokens"]:
                        hit_bad += 1
                    if b_["prompt_tokens"] <= a_["prompt_tokens"] + a_["completion_tokens"]:
                        grow_bad += 1
            if seq_bad:
                bad.append(f"agent_session: {seq_bad} out-of-order or overlapping turns")
            if hit_bad:
                bad.append(f"agent_session: {hit_bad} later turns did not hit the previous turn's prompt in the cache")
            if grow_bad:
                bad.append(f"agent_session: {grow_bad} turns did not re-send the previous prompt + reply + a tool result")
            if not any(len(xs) == 3 for xs in sess.values()):
                bad.append("agent_session: no session reached its last turn")

        # 3. protocol --column C and the report's cached-rate price
        out = f"{td}/proto"
        r = subprocess.run([sys.executable, "-I", str(ROOT / "bench/protocol.py"), "--model", "m", "--base", f"http://127.0.0.1:{PORT}",
                            "--metrics", f"http://127.0.0.1:{PORT}/metrics", "--out", out, "--breakeven-rps", "1", "--rates", "4",
                            "--seeds", "7", "--duration", "12", "--warmup", "4", "--client", "traffic", "--column", "C",
                            "--group-tag", "c1fit", "--workload", "shared_prefix", "--stratify",
                            "--params", json.dumps(dict(p, stratify=None))], capture_output=True, text=True, timeout=240)
        if r.returncode != 0:
            bad.append(f"protocol --column C rc={r.returncode}: {r.stderr[-400:]}")
        else:
            s = json.load(open(f"{out}/protocol_summary.json"))
            res = s["ladder"][0]["results"]["7"]
            if s.get("column") != "C" or s.get("group_tag") != "c1fit" or s.get("workload") != "shared_prefix":
                bad.append(f"protocol summary column/group/workload: {s.get('column')}, {s.get('group_tag')}, {s.get('workload')}")
            for k in ("cached_tokens_mean", "cached_share_cell", "cached_share_scored", "server_prefix_cache_hit_ratio"):
                if res.get(k) is None:
                    bad.append(f"protocol cell lacks {k}")
            import screen_report as SR
            bar = dict(in_usd_per_m=0.07, out_usd_per_m=0.30, cached_usd_per_m=0.007)
            got = SR.cached_price(bar, dict(prompt_tokens_mean=9500, cached_tokens_mean=7600, completion_tokens_mean=380))
            want = (1900 * 0.07 + 7600 * 0.007 + 380 * 0.30) / 1e6
            if abs(got - want) > 1e-12:
                bad.append(f"cached_price {got} != {want}")
finally:
    mock.terminate()
print("FAIL\n  " + "\n  ".join(bad) if bad else "OK: column C shared-prefix pool, sequential agent sessions with real replies, protocol fields, cached-rate price")
sys.exit(1 if bad else 0)
