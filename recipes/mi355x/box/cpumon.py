"""cpumon.py OUT.csv [interval_s=1]: per-second host CPU and per-role process CPU, for attributing a benchmark cell's result to
CPU starvation (lead's ask for the small-host claim). Started detached (setsid nohup) from an `sh` step, so it outlives the step.

Columns: t_utc, ncpu, host_busy_pct (all cores, 100 = every core busy), host_busy_cores, engine_pct, api_pct, client_pct,
other_pct (process CPU in % of ONE core, summed over the role's processes), runq (procs_running from /proc/stat).
Roles by command line: engine = vLLM's EngineCore process ("EngineCore" / "VLLM::"), api = the `vllm serve` front end,
client = the load generator (`vllm bench serve`, bench/protocol.py, bench/traffic.py), other = everything else."""
import os
import sys
import time

OUT = sys.argv[1]
DT = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
HZ = os.sysconf("SC_CLK_TCK")
NCPU = os.cpu_count() or 1


def stat_cpu():
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
        rest = f.read()
    v = list(map(int, parts))
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    rq = next((int(l.split()[1]) for l in rest.splitlines() if l.startswith("procs_running")), -1)
    return sum(v), idle, rq


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
        fo.write("t_utc,ncpu,host_busy_pct,host_busy_cores,engine_pct,api_pct,client_pct,other_pct,runq\n")
    t0, i0, _ = stat_cpu()
    p0 = procs()
    while True:
        time.sleep(DT)
        t1, i1, rq = stat_cpu()
        p1 = procs()
        tot, idl = t1 - t0, i1 - i0
        busy = 100.0 * (tot - idl) / tot if tot > 0 else 0.0
        acc = {"engine": 0, "api": 0, "client": 0, "other": 0}
        for pid, (tk, r) in p1.items():
            if pid in p0:
                acc[r] += max(0, tk - p0[pid][0])
        pct = {k: 100.0 * v / HZ / DT for k, v in acc.items()}
        fo.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())},{NCPU},{busy:.1f},{busy * NCPU / 100:.2f},"
                 f"{pct['engine']:.1f},{pct['api']:.1f},{pct['client']:.1f},{pct['other']:.1f},{rq}\n")
        t0, i0, p0 = t1, i1, p1
