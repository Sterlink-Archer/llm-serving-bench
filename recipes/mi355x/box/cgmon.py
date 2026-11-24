"""cgmon.py OUT.csv [interval_s=1]: per-second CPU of the SANDBOX (its cgroup), plus per-role process CPU.

Why not /proc/stat: a Daytona sandbox sees the host's 192 CPUs in /proc/stat and os.cpu_count(); its real limit is the cgroup
quota (/sys/fs/cgroup/cpu.max, e.g. "400000 100000" = 4 cores). So host-wide busy % says nothing about the box. This reads
/sys/fs/cgroup/cpu.stat instead: usage_usec (cores used), nr_throttled (100 ms periods in which the quota ran out) and
throttled_usec. Throttling is the direct evidence of CPU starvation the small-host claim needs.

Columns: t_utc, quota_cores, cg_used_cores, cg_used_pct_of_quota, throttled_periods, periods, throttled_ms,
engine_pct, api_pct, client_pct, other_pct (process CPU in % of ONE core, summed over the role's processes; roles as cpumon.py)."""
import os
import sys
import time

OUT = sys.argv[1]
DT = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
HZ = os.sysconf("SC_CLK_TCK")
CG = "/sys/fs/cgroup"


def quota():
    try:
        q, p = open(f"{CG}/cpu.max").read().split()
        return float("nan") if q == "max" else int(q) / int(p)
    except (OSError, ValueError):
        return float("nan")


def cgstat():
    d = {}
    try:
        for line in open(f"{CG}/cpu.stat"):
            k, v = line.split()
            d[k] = int(v)
    except (OSError, ValueError):
        pass
    return d


def role(cmd):
    if "EngineCore" in cmd or cmd.startswith("VLLM::"):
        return "engine"
    if "bench serve" in cmd or "bench/protocol.py" in cmd or "bench/traffic.py" in cmd:
        return "client"
    if "vllm serve" in cmd or "vllm.entrypoints" in cmd or "sglang.launch_server" in cmd:
        return "api"
    return "other"


def procs():
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                s = f.read()
            fields = s[s.rindex(")") + 2:].split()
            ticks = int(fields[11]) + int(fields[12])
            with open(f"/proc/{d}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
            if not cmd:
                with open(f"/proc/{d}/comm") as f:
                    cmd = f.read().strip()
            out[d] = (ticks, role(cmd))
        except (OSError, ValueError, IndexError):
            continue
    return out


with open(OUT, "a", buffering=1) as fo:
    if fo.tell() == 0:
        fo.write("t_utc,quota_cores,cg_used_cores,cg_used_pct_of_quota,throttled_periods,periods,throttled_ms,"
                 "engine_pct,api_pct,client_pct,other_pct\n")
    Q = quota()
    c0, w0, p0 = cgstat(), time.monotonic(), procs()
    while True:
        time.sleep(DT)
        c1, w1, p1 = cgstat(), time.monotonic(), procs()
        wall = w1 - w0
        used = (c1.get("usage_usec", 0) - c0.get("usage_usec", 0)) / 1e6 / wall
        thr = c1.get("nr_throttled", 0) - c0.get("nr_throttled", 0)
        per = c1.get("nr_periods", 0) - c0.get("nr_periods", 0)
        tms = (c1.get("throttled_usec", 0) - c0.get("throttled_usec", 0)) / 1e3
        acc = {"engine": 0, "api": 0, "client": 0, "other": 0}
        for pid, (tk, r) in p1.items():
            if pid in p0:
                acc[r] += max(0, tk - p0[pid][0])
        pct = {k: 100.0 * v / HZ / wall for k, v in acc.items()}
        fo.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())},{Q:.2f},{used:.2f},{100 * used / Q:.1f},{thr},{per},{tms:.0f},"
                 f"{pct['engine']:.1f},{pct['api']:.1f},{pct['client']:.1f},{pct['other']:.1f}\n")
        c0, w0, p0 = c1, w1, p1
