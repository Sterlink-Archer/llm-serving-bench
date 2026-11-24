#!/usr/bin/env python3
"""extend_live.py: add rungs to a live run.py box without releasing a scarce GPU (lead 11-22: "extend on the SAME box, do
not release and re-acquire").

run.py's plan is fixed and it deletes the sandbox right after its last step. This tool:
  1. pauses the local run.py process (SIGSTOP); the box keeps running its current background job;
  2. keeps the sandbox alive while paused: Daytona auto-stops a sandbox after 15 idle minutes (autoStopInterval=15,
     infra/daytona.py) and auto-deletes it on stop, so it polls the box with an exec every --poll seconds;
  3. waits for --wait-job (run.py's current step) to finish, then runs one extra protocol step whose command is the
     --from-step command copied from run.json with only --out, --rates, --seeds and --max-minutes replaced;
  4. pulls the new protocol dir and its log into the run's results dir and writes LIVE_EXTEND_<name>.json (what ran,
     why, times, rc);
  5. ALWAYS resumes run.py (SIGCONT, in `finally`), so its own teardown deletes the box as usual.
The new step's dir name must start with protocol_<server> so analysis/screen_report.py picks it up.

  python3 -I recipes/inkling/tools/extend_live.py --label ikC_b300_mixed_dspark --pid 3147433 \
      --wait-job protocol_b300_A_random --from-step protocol_b300_B_text --name protocol_mixed_dspark_b300ext_B_text \
      --rates 3.592 --seeds 7,8 --why "..."
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shlex
import signal
import sys
import tarfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "infra"))
import daytona as dt  # noqa: E402


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--wait-job", required=True)
    ap.add_argument("--from-step", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--rates", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--max-minutes", type=float, default=45)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--why", required=True)
    a = ap.parse_args()
    rd = sorted((ROOT / "results" / a.label).glob("2026*"))[-1]
    run = json.loads((rd / "run.json").read_text())
    src = next(v for s in run["servers"].values() for k, v in (s.get("steps") or {}).items() if k == a.from_step)
    cmd = src["cmd"]
    cmd = re.sub(r"--out \S+", f"--out {dt.WORK}/out/{a.name}", cmd)
    cmd = re.sub(r"--rates \S+", f"--rates {a.rates}", cmd)
    cmd = re.sub(r"--seeds \S+", f"--seeds {a.seeds}", cmd)
    cmd = re.sub(r"--max-minutes \S+", f"--max-minutes {a.max_minutes}", cmd)
    sid = json.loads([l for l in (ROOT / "infra" / "ledger.jsonl").read_text().splitlines()
                      if '"event": "create"' in l and f'"run": "{a.label}"' in l][-1])["id"]
    note = dict(t_start=now(), label=a.label, run_dir=str(rd.relative_to(ROOT)), sandbox_id=sid, runpy_pid=a.pid, why=a.why,
                wait_job=a.wait_job, from_step=a.from_step, name=a.name, cmd=cmd, events=[])
    out = rd / f"LIVE_EXTEND_{a.name}.json"

    def log(msg):
        note["events"].append(f"{now()} {msg}")
        out.write_text(json.dumps(note, indent=1) + "\n")
        print(f"{now()} [extend_live] {msg}", flush=True)

    os.kill(a.pid, signal.SIGSTOP)
    log(f"run.py {a.pid} paused (SIGSTOP)")
    try:
        box = dt.Box(dt.get_sandbox(sid))
        t0 = time.time()
        while box.job_rc(a.wait_job) is None:          # each poll is an exec: keeps the sandbox from auto-stopping
            if time.time() - t0 > 7200:
                raise TimeoutError(f"{a.wait_job} still running after 2 h")
            time.sleep(a.poll)
        log(f"{a.wait_job} finished rc={box.job_rc(a.wait_job)}; starting {a.name}")
        box.bg(a.name, cmd)
        t1 = time.time()
        while (rc := box.job_rc(a.name)) is None:
            if time.time() - t1 > a.max_minutes * 60 + 600:
                raise TimeoutError(f"{a.name} still running after {a.max_minutes} min + 10")
            time.sleep(a.poll)
        note["rc"], note["secs"] = rc, round(time.time() - t1, 1)
        log(f"{a.name} rc={rc} after {time.time() - t1:.0f}s; pulling")
        tgz = f"{dt.WORK}/{a.name}.tgz"
        box.exec(f"bash -c {shlex.quote(f'cd {dt.WORK}/out && tar czf {tgz} {a.name}')}", timeout=180)
        if box.get(tgz, rd / f"{a.name}.tgz"):
            with tarfile.open(rd / f"{a.name}.tgz") as tf:
                tf.extractall(rd, filter="data")
            (rd / f"{a.name}.tgz").unlink(missing_ok=True)
        box.get(f"{dt.WORK}/{a.name}.log", rd / f"{a.name}.log")
        log("pulled")
    except Exception as e:  # noqa: BLE001
        log(f"ERROR {e!r}"[:400])
    finally:
        os.kill(a.pid, signal.SIGCONT)
        note["t_end"] = now()
        log(f"run.py {a.pid} resumed (SIGCONT)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
