# outside.py <trace.json.gz> [topn]: GPU kernels that run OUTSIDE vLLM's execute_context_X(T)_generation_Y(U) step
# windows (the target model's forward), grouped by kernel name. On an MTP engine that is the drafter's forward passes,
# the top-k/top-p and rejection samplers and input preparation. Also prints every distinct gpu_user_annotation name
# with its total time, so drafter annotations (if any) show up. Companion to OpenRelay's steps.py / stepkern.py.
import bisect
import collections
import gzip
import json
import re
import sys

d = json.load(gzip.open(sys.argv[1]))
ev = d["traceEvents"]
win, ann = [], collections.Counter()
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation":
        ann[re.sub(r"\d+", "#", e.get("name", ""))[:80]] += e["dur"]
        if re.match(r"execute_context_\d+\(\d+\)_generation_\d+\(\d+\)", e.get("name", "")):
            win.append((e["ts"], e["ts"] + e["dur"]))
win.sort()
starts = [w[0] for w in win]
k, c = collections.Counter(), collections.Counter()
tot = inside = 0
t0, t1 = None, None
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "kernel":
        tot += e["dur"]
        t0 = e["ts"] if t0 is None else min(t0, e["ts"])
        t1 = e["ts"] + e["dur"] if t1 is None else max(t1, e["ts"] + e["dur"])
        i = bisect.bisect_right(starts, e["ts"]) - 1
        if i >= 0 and e["ts"] < win[i][1]:
            inside += e["dur"]
        else:
            k[e["name"][:100]] += e["dur"]
            c[e["name"][:100]] += 1
out = tot - inside
print(f"kernels {tot/1000:.1f} ms over a {((t1 or 0) - (t0 or 0))/1000:.1f} ms span; inside step windows {inside/1000:.1f} ms; "
      f"OUTSIDE {out/1000:.1f} ms ({out/max(tot,1)*100:.1f}%) across {len(win)} steps = {out/max(len(win),1)/1000:.2f} ms/step")
for n, v in k.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 25):
    print(f"  {v/1000:8.1f} ms {v/max(out,1)*100:5.1f}% x{c[n]:<6} {n}")
print("gpu_user_annotation totals:")
for n, v in ann.most_common(12):
    print(f"  {v/1000:9.1f} ms  {n}")
