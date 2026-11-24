#!/usr/bin/env python3
"""Run one experiment config end to end on ONE preemptible Daytona GPU sandbox, then delete it.

  python3 -I run.py configs/<x>.json --dry-run     # expand the plan, print the cost ceiling, create nothing
  python3 -I run.py configs/<x>.json               # do it   (bin/launch.sh runs several detached)

Flow: snapshot (create if missing) -> spend guard -> spot sandbox (ttl) -> upload bench/ + data/ -> for each server
(one sandbox, many configs; the engine restarts between them): pinned weights (bench/hfdl.py) -> launch -> /health ->
startup facts from the engine log (a config whose global 512-dim layers run on TRITON_ATTN, or whose KV capacity /
graph sizes / backend cannot be read, is REJECTED and gets no measurement) -> warmup request -> its steps (eval gate,
protocol ladder, matrix presets, shell) -> engine stop. The sandbox is deleted in `finally`, on SIGTERM/SIGINT, on
spot eviction, and before max_minutes or budget_usd is crossed; deletion is then confirmed (404) and logged.

Observability (docs/DATA_CONTRACT.md): every phase (create, image_pull, weight_download, engine_start with nested
weight_load / torch_compile / graph_capture, warmup, eval, steady, teardown, reap_confirm, eviction) is appended to
infra/ledger.jsonl with run_id, config_id, sandbox, GPU, image, model revision, UTC start/end and estimated dollars.
Per run: results/<label>/<stamp>/{run.json, startup_facts_<server>.json, cost_measured.json, gpumon.csv,
gpumon_norm.csv, server logs, m_*/ matrices, protocol_*/ ladders, eval_*.json}. After the delete the project's
cumulative spend is reconciled (infra/spend.py) and milestone crossings are written to the ledger.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import pathlib
import re
import shlex
import signal
import sys
import tarfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "infra"))
sys.path.insert(0, str(ROOT / "bench"))
import daytona as dt  # noqa: E402
import startup_facts as SF  # noqa: E402

W = dt.WORK
HF = "https://huggingface.co"
PROJECT_REPLAN_USD = 500.0
MILESTONES = (50, 100, 200, 300, 400)
OPENRELAY_MODEL = "openrelay/gemma-4-31b"


# ----------------------------------------------------------------------------- helpers
from pricebar import openrelay_sync, price_bar, usd_per_req as bar_usd_per_req  # noqa: E402,F401  (one implementation)


def breakeven_rps(usd_h, in_tok=9500, out_tok=380, bar=None):
    usd_req = bar_usd_per_req(bar or price_bar(), in_tok, out_tok)
    return usd_h / 3600 / usd_req, usd_req


def hf_files(repo, rev, patterns=None):
    import fnmatch
    url = f"{HF}/api/models/{repo}/revision/{rev}?blobs=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "openrelay-bench/0.1"}), context=dt.CTX, timeout=60) as r:
        d = json.loads(r.read())
    out = []
    for s in d.get("siblings", []):
        f = s["rfilename"]
        if f.startswith(".") or f.endswith((".md", ".png", ".jpg", ".gitattributes")):
            continue
        if patterns and not any(fnmatch.fnmatch(f, p) for p in patterns):
            continue
        out.append((f, s.get("size") or 0))
    return d.get("sha"), out


def engine_of(srv):
    l = srv["launch"]
    return "sglang" if "sglang" in l else "llama.cpp" if "llama-server" in l else "vllm"


def expand_steps(server, usd_h=0.0, run_id="", config_id="", bar=None):
    import matrix as MX
    steps = []
    metrics = server.get("metrics_url") or "http://127.0.0.1:8000/metrics"
    served = server.get("served_name", "gemma")
    for st in server.get("steps", []):
        t = st["type"]
        if t == "matrix":
            preset = st["preset"]
            name = st.get("name", f"m_{server['name']}_{preset}")
            spec = json.loads((ROOT / "bench" / "presets" / f"{preset}.json").read_text())
            cells = [c for c in MX.cells_of(spec) if not st.get("only") or re.search(st["only"], c["id"])]
            scale = st.get("scale_duration", 1.0)
            est = sum(MX.est_s(c, scale) for c in cells) + 30
            cmd = (f"python3 bench/matrix.py --spec bench/presets/{preset}.json --model {served} --metrics {metrics} "
                   f"--out {W}/out/{name} --usd-per-hour {usd_h:.4f} --scale-duration {scale}"
                   + (f" --only {shlex.quote(st['only'])}" if st.get("only") else ""))
            steps.append(dict(name=name, kind="matrix", cmd=cmd, outputs=[f"out/{name}/"], est_s=est))
        elif t == "protocol":
            name = st.get("name", f"protocol_{server['name']}")
            be = st.get("breakeven_rps", "auto")
            be = breakeven_rps(usd_h, *st.get("shape", (9500, 380)), bar=bar)[0] if be == "auto" else float(be)
            mults = st.get("multiples", [0.6, 0.9, 1.2, 1.6, 2.0]) if not st.get("rates") else [r / be for r in st["rates"]]
            seeds = st.get("seeds", [7, 8])
            dur = st.get("duration", 300)
            params = st.get("params", {"input_dist": {"uniform": [1900, 17100]}, "output_dist": {"uniform": [76, 684]}})
            client = st.get("client", "traffic")
            tok = (server.get("weights") or [{}])[0].get("dir", "")
            parts = [f"python3 bench/protocol.py --model {served} --metrics {metrics} --out {W}/out/{name}",
                     f"--breakeven-rps {be:.4f} --client {client}"]
            if client == "traffic" and st.get("stratify", True):
                parts.append("--stratify")
            if st.get("column"):
                parts.append(f"--column {st['column']}")   # C = cached traffic, supplementary, NOT OpenRelay's method
            if st.get("group_tag"):
                parts.append(f"--group-tag {shlex.quote(st['group_tag'])}")
            if client == "traffic" and st.get("ttft_basis", "text") != "text":
                parts.append(f"--ttft-basis {st['ttft_basis']}")   # Inkling: first token is a swallowed special token
            if st.get("vllm_bin"):
                parts.append(f"--vllm-bin {shlex.quote(st['vllm_bin'])}")
            if client == "vllmbench" or (tok and st.get("exact_tokens", True)):
                parts.append(f"--tokenizer {tok}")   # vllmbench: random-token prompts; traffic: exact real-text prompt lengths
            if st.get("rates"):
                parts.append("--rates " + ",".join(str(r) for r in st["rates"]))
            else:
                parts.append("--multiples " + ",".join(str(m) for m in mults))
            parts += [f"--seeds {','.join(str(x) for x in seeds)} --duration {dur}",
                      f"--workload {st.get('workload', 'synthetic')} --params {shlex.quote(json.dumps(params))}",
                      f"--run-id {shlex.quote(run_id)} --config-id {shlex.quote(config_id)}"]
            if st.get("prefill_shot", True):
                parts.append("--prefill-shot")
            if st.get("slo_ttft_p95"):
                parts.append(f"--slo-ttft-p95 {st['slo_ttft_p95']}")
            cmd = " ".join(parts)
            est = len(mults) * len(seeds) * (dur + 20) + (90 if st.get("prefill_shot", True) else 0) + 30
            steps.append(dict(name=name, kind="protocol", cmd=cmd, outputs=[f"out/{name}/"], est_s=est, breakeven_rps=be))
        elif t == "eval":
            name = st.get("name", f"eval_{server['name']}")
            sets = st.get("sets", "data/eval/gsm8k_200.jsonl,data/eval/mmlupro_200.jsonl")
            gate = st.get("gate")
            ctk = st.get("chat_template_kwargs")
            cmd = (f"python3 bench/evalrun.py --model {served} --sets {sets} --concurrency {st.get('conc', 64)} "
                   + (f"--chat-template-kwargs {shlex.quote(json.dumps(ctk))} " if ctk else "")
                   + f"--max-tokens {st.get('max_tokens', 2048)} --label {name} --out {W}/out/{name}.json"
                   + (f" --limit {st['limit']}" if st.get("limit") else "")
                   + (f" --gate-task {gate['task']} --gate-threshold {gate['threshold']}" if gate else ""))
            steps.append(dict(name=name, kind="eval", cmd=cmd, outputs=[f"out/{name}.json"], est_s=st.get("est_s", 300), gate=gate))
        elif t == "sh":
            steps.append(dict(name=st["name"], kind="sh", cmd=st["cmd"], outputs=st.get("outputs", []), est_s=st.get("est_s", 120)))
        else:
            raise ValueError(t)
    return steps


def bundle() -> bytes:
    """bench/ + data/ (eval, chat, corpus, shapes) as a tar.gz, uploaded once per sandbox. No private text is in data/."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in ["bench", "data/eval", "data/chat", "data/corpus", "data/shapes"]:
            p = ROOT / rel
            if p.exists():
                tf.add(p, arcname=rel, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return buf.getvalue()


def ledger_spend_estimate():
    """Project spend estimated from the ledger: every created sandbox x $/h x (delete time, or now if not deleted)."""
    creates, deletes = {}, {}
    for line in dt.LEDGER.read_text().splitlines() if dt.LEDGER.exists() else []:
        r = json.loads(line)
        if r.get("event") == "create":
            creates[r["id"]] = r
        elif r.get("event") == "delete":
            deletes.setdefault(r["id"], r)
    tot = 0.0
    now = time.time()
    for sid, c in creates.items():
        t0 = time.mktime(time.strptime(c["t"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
        t1 = (time.mktime(time.strptime(deletes[sid]["t"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone) if sid in deletes else now
        tot += max(0.0, t1 - t0) / 3600 * c["usd_per_h"]["total"]
    return tot


def reconcile_spend():
    """Write a `spend` event (estimate + Daytona-metered when available) and any new `milestone` events."""
    import os
    if os.environ.get("OR_LOCAL_TEST"):
        return ledger_spend_estimate(), None, []
    est = ledger_spend_estimate()
    metered, mfile = None, None
    try:
        import subprocess
        r = subprocess.run([sys.executable, "-I", str(ROOT / "infra/spend.py"), "--quiet"], capture_output=True, text=True, timeout=300)
        m = re.search(r"wrote (\S+)", r.stdout)
        if m:
            mfile = str(pathlib.Path(m.group(1)).relative_to(ROOT))
            d = json.loads(pathlib.Path(m.group(1)).read_text())
            metered = d.get("metered_total_usd")
    except Exception:  # noqa: BLE001
        pass
    dt.ledger(dict(event="spend", est_total_usd=round(est, 3), metered_total_usd=metered, metered_file=mfile))
    total, basis = (metered, "metered") if metered else (est, "estimate")
    done = {r.get("threshold_usd") for r in (json.loads(l) for l in dt.LEDGER.read_text().splitlines()) if r.get("event") == "milestone"}
    crossed = []
    for th in MILESTONES:
        if total >= th and th not in done:
            dt.ledger(dict(event="milestone", threshold_usd=th, total_usd=round(total, 2), basis=basis))
            crossed.append(th)
    return est, metered, crossed


# ----------------------------------------------------------------------------- the run
class Run:
    def __init__(self, cfg, cfg_path):
        self.cfg = cfg
        snap = cfg["snapshot"]
        self.snap = snap
        self.pph = dt.hourly_usd(snap["gpu"], snap.get("gpus", 1), snap["cpu"], snap["mem"], snap["disk"])
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.run_id = f"{cfg['label']}/{stamp}"
        import os
        self.out = pathlib.Path(os.environ.get("OR_RESULTS_DIR") or (ROOT / "results")) / cfg["label"] / stamp
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(json.dumps(cfg, indent=1))
        self.rep = dict(label=cfg["label"], run_id=self.run_id, config_file=str(cfg_path), t_start=dt.now_iso(), usd_per_h=self.pph,
                        phases={}, servers={})
        self.t0 = None
        self.sb = None
        self.box = None
        self.dead = False
        self.phases = []
        self.last_good_t = None      # end of the last step whose results were pulled intact
        self.current = None          # (server, step) running now: what an eviction interrupts

    # bookkeeping
    def elapsed(self):
        return 0.0 if self.t0 is None else time.time() - self.t0

    def cost(self):
        return self.pph["total"] * self.elapsed() / 3600

    def left_s(self):
        by_t = self.cfg["max_minutes"] * 60 - self.elapsed()
        by_usd = (self.cfg["budget_usd"] - self.cost()) / self.pph["total"] * 3600
        return min(by_t, by_usd) - 90   # keep 90 s for pulling results and deleting

    def save(self):
        self.rep.update(elapsed_s=round(self.elapsed(), 1), est_cost_usd=round(self.cost(), 3), t_saved=dt.now_iso())
        (self.out / "run.json").write_text(json.dumps(self.rep, indent=1, default=str))

    def log(self, msg):
        line = f"{dt.now_iso()} [{self.cfg['label']}] {msg}"
        print(line, flush=True)
        with open(self.out / "driver.log", "a") as f:
            f.write(line + "\n")

    def phase(self, name, t_start, t_end, srv=None, cell_ids=None, nested_in=None, ok=True, detail=None, usd_per_h=None):
        w = (srv or {}).get("weights") or [{}]
        rate = self.pph["total"] if usd_per_h is None else usd_per_h
        rec = dict(event="phase", run_id=self.run_id, config_id=f"{self.cfg['label']}:{srv['name']}" if srv else None,
                   cell_id=None if not cell_ids else (cell_ids[0] if len(cell_ids) == 1 else None),
                   sandbox_id=(self.sb or {}).get("id"), gpu=self.snap["gpu"], gpus=self.snap.get("gpus", 1), image=self.snap["image"],
                   image_digest=self.snap.get("image_digest_amd64") or self.snap.get("image_digest_index"),
                   model_repo=w[0].get("repo"), model_rev=w[0].get("rev"), phase=name,
                   t_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)), t_end=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end)),
                   dur_s=round(t_end - t_start, 2), usd_per_h=rate, usd=round((t_end - t_start) / 3600 * rate, 5), nested_in=nested_in, ok=ok,
                   detail=dict(detail or {}, **({"cell_ids": cell_ids} if cell_ids else {})))
        dt.ledger(rec)
        self.phases.append(rec)
        return rec

    def alive(self):
        try:
            s = dt.get_sandbox(self.sb["id"])
        except dt.ApiError as e:
            if e.status == 404:
                self.dead = True
                return False
            return True
        st = str(s.get("state", "")).lower()
        if s.get("spotEvictedAt") or st in ("destroyed", "destroying", "stopped", "error"):
            self.record_eviction(s)
            self.dead = True
            return False
        return True

    def record_eviction(self, s):
        """Maks 11-22: a spot preemption is outside our control: record it, never let it into a benchmark number.
        One `evicted` ledger event (and an `eviction` phase spanning last-good -> eviction), the interrupted step marked
        excluded; main() re-runs the config on a fresh box."""
        if self.rep.get("evicted"):
            return
        import datetime
        ev = s.get("spotEvictedAt")
        try:
            ev_t = datetime.datetime.fromisoformat(ev.replace("Z", "+00:00")).timestamp() if ev else time.time()
        except ValueError:
            ev_t = time.time()
        last = self.last_good_t or self.t0 or ev_t
        mins = max(0.0, ev_t - last) / 60
        rec = dict(event="evicted", run_id=self.run_id, sandbox_id=(self.sb or {}).get("id"), gpu=self.snap["gpu"], gpus=self.snap.get("gpus", 1),
                   state=str(s.get("state")), spotEvictedAt=ev, evicted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ev_t)),
                   last_good_t=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last)),
                   interrupted=dict(server=self.current[0], step=self.current[1]) if self.current else None,
                   minutes_lost=round(mins, 2), usd_lost=round(mins / 60 * self.pph["total"], 4), usd_per_h=self.pph["total"],
                   label="spot preemption, outside our control")
        dt.ledger(rec)
        self.phase("eviction", last, ev_t, ok=False, detail=dict(spotEvictedAt=ev, interrupted=rec["interrupted"]))
        self.rep["evicted"] = rec
        if self.current:
            st = self.rep["servers"].get(self.current[0], {}).setdefault("steps", {}).setdefault(self.current[1], {})
            st["excluded"] = "evicted"
        self.log(f"EVICTED (spot preemption, outside our control): {rec['interrupted']} lost {mins:.1f} min / ${rec['usd_lost']:.2f}")
        self.save()

    def pull(self, rel, local_name=None):
        try:
            ok = self.box.get(f"{W}/{rel}", self.out / (local_name or rel.replace("/", "__")))
        except Exception as e:  # noqa: BLE001
            ok = False
            self.rep.setdefault("pull_errors", []).append(f"{rel}: {e!r}"[:300])
        return ok

    def pull_dir(self, rel):
        """tar a remote directory (archive root = its base name) and unpack it under the results dir."""
        rel = rel.rstrip("/")
        parent, base = str(pathlib.PurePosixPath(rel).parent), pathlib.PurePosixPath(rel).name
        self.box.exec(f"bash -c {shlex.quote(f'cd {W}/{parent} && tar czf {W}/{base}.tgz {base}')}", timeout=180)
        if self.pull(f"{base}.tgz", f"{base}.tgz"):
            with tarfile.open(self.out / f"{base}.tgz") as tf:
                tf.extractall(self.out, filter="data")
            (self.out / f"{base}.tgz").unlink(missing_ok=True)
            return True
        return False

    def wait_job(self, name, max_s, every=10, server_job=None):
        t0 = time.time()
        while time.time() - t0 < max_s:
            if self.left_s() <= 0:
                return "budget"
            rc = self.box.job_rc(name)
            if rc is not None:
                return rc
            if server_job and self.box.job_rc(server_job) is not None:
                return "server died"
            if int(time.time() - t0) % 60 < every and not self.alive():
                return "sandbox gone"
            time.sleep(every)
        return "timeout"

    def kill_engines(self):
        self.box.sh("kill_engines", r"""
for p in /proc/[0-9]*; do
  c=$(tr '\0' ' ' < $p/cmdline 2>/dev/null)
  case "$c" in *"vllm serve"*|*"sglang.launch_server"*|*llama-server*|*trtllm-serve*|*"VLLM::"*|*"sglang::"*|*"from multiprocessing"*) kill -9 ${p#/proc/} 2>/dev/null;; esac
done
sleep 10
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true""", timeout=90)

    # phases
    def setup(self):
        c = self.cfg
        s = self.snap
        t = time.time()
        snap = dt.get_snapshot(s["name"])
        built = False
        if not snap or str(snap.get("state", "")).lower() != "active":
            self.log(f"{'building' if not snap else 'waiting for ' + str(snap.get('state')) + ' snapshot'} {s['name']} from {s['image']}")
            snap = dt.create_snapshot(s["name"], s["image"], s["gpu"], s.get("gpus", 1), s["cpu"], s["mem"], s["disk"])
            built = True
            self.rep["phases"]["snapshot_build_s"] = round(time.time() - t, 1)
        self.phase("image_pull", t, time.time(), usd_per_h=0.0,
                   detail=dict(cached=not built, snapshot=s["name"], note="Daytona snapshot build (registry pull) before any sandbox exists; "
                               "the runner-side pull is inside `create`"))
        # spend guard: refuse a create that could take the project past the re-plan line
        spent = ledger_spend_estimate()
        ceiling = min(c["budget_usd"], self.pph["total"] * (c["max_minutes"] + 5) / 60)
        self.rep["spend_guard"] = dict(project_est_usd_before=round(spent, 2), this_run_ceiling_usd=round(ceiling, 2), replan_usd=PROJECT_REPLAN_USD)
        if spent + ceiling > PROJECT_REPLAN_USD:
            raise RuntimeError(f"spend guard: project ${spent:.2f} + this run up to ${ceiling:.2f} would pass ${PROJECT_REPLAN_USD:.0f}; re-plan with Maks")
        avail = dt.spot_available(s["gpu"], s.get("gpus", 1))
        self.rep["spot_capacity_at_start"] = avail
        if avail < 1:
            raise RuntimeError(f"no spot {s['gpu']} x{s.get('gpus', 1)} free right now")
        self.log(f"creating spot sandbox ({s['gpu']} x{s.get('gpus', 1)}, ${self.pph['total']}/h, ttl {c['max_minutes']} min, "
                 f"budget ${c['budget_usd']}; project so far ~${spent:.2f})")
        self.t0 = time.time()       # billing is counted from the create call
        try:
            vols = None
            if c.get("volume"):   # e.g. {"name": "or-weights", "mount": "/vol", "mode": "stage" | "read"}
                dt.ensure_volume(c["volume"]["name"])
                vols = [{"volumeId": c["volume"]["name"], "mountPath": c["volume"].get("mount", "/vol")}]
            self.sb = dt.create_gpu_sandbox(snap, c["label"], c["max_minutes"] + 5, run_id=self.run_id, volumes=vols)
        except Exception as e:
            self.phase("failed_create", self.t0, time.time(), ok=False, detail=dict(error=repr(e)[:300]))
            raise
        self.rep["sandbox_id"] = self.sb["id"]
        self.rep["phases"]["create_s"] = self.sb.get("_create_s")
        self.phase("create", self.t0, time.time(), detail=dict(includes_runner_image_pull=True, spot=self.sb.get("spot")))
        self.box = dt.Box(self.sb)
        self.save()
        self.box.put_bytes(bundle(), f"{W}/bundle.tgz")
        r = self.box.sh("setup", f"""cd {W} && tar xzf bundle.tgz && mkdir -p out models && echo ok
(nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv,noheader || amd-smi static --asic --vram 2>/dev/null | head -20) 2>&1
nproc; free -g | sed -n 2p; df -h /root | tail -1; python3 --version 2>&1; which curl 2>&1
python3 -c 'import vllm; print("vllm", vllm.__version__)' 2>/dev/null; python3 -c 'import sglang; print("sglang", sglang.__version__)' 2>/dev/null
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null; true""", timeout=180)
        self.rep["system"] = r["out"][-3000:]
        if c.get("setup_cmd"):     # e.g. images without python3 (llama.cpp): the harness needs it
            r2 = self.box.sh("setup_extra", c["setup_cmd"], timeout=900)
            self.rep["setup_extra"] = dict(rc=r2["rc"], tail=r2["out"][-800:])
        self.log("system: " + " | ".join(r["out"].strip().splitlines()[:8]))
        self.box.bg("gpumon", "if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=timestamp,index,utilization.gpu,"
                    "memory.used,power.draw,clocks.sm,temperature.gpu --format=csv,noheader -lms 2000; else while true; do "
                    "echo \"T $(date -u +%FT%T)\"; amd-smi metric --usage --power --mem-usage --clock --temperature --csv 2>/dev/null | tail -n +2; sleep 2; done; fi")
        self.save()

    def download(self, srv):
        cmds = []
        rec = self.rep["servers"][srv["name"]]
        rec["weights"] = []
        for w in srv.get("weights", []):
            sha, files = hf_files(w["repo"], w["rev"], w.get("patterns"))
            rec["weights"].append(dict(repo=w["repo"], rev=w["rev"], resolved_sha=sha, files=len(files), bytes=sum(sz for _, sz in files)))
            vol = self.cfg.get("volume") or {}
            stage = (f" --stage-dir {vol.get('mount', '/vol')}/hf" if vol.get("mode") == "stage" else
                     f" --from-stage {vol.get('mount', '/vol')}/hf" if vol.get("mode") == "read" else "")
            cmds.append(f"python3 bench/hfdl.py --repo {w['repo']} --rev {w['rev']} --dir {w['dir']} "
                        f"--files {shlex.quote(','.join(f for f, _ in files))} --sizes {','.join(str(sz) for _, sz in files)}{stage} || exit 1")
        name = f"download_{srv['name']}"
        t = time.time()
        self.box.bg(name, "\n".join(cmds))
        rc = self.wait_job(name, max_s=2400, every=10)
        tail = self.box.tail(name, 6)
        dl = [dict(re.findall(r"(\w+)=(\S+)", l)) for l in tail.splitlines() if l.startswith("DL ")]
        stage_w = [x for x in dl if x.get("method") == "stage-write"]
        dl = [x for x in dl if x.get("method") != "stage-write"]
        b = sum(int(x.get("bytes", 0)) for x in dl)
        if stage_w:
            sw_b, sw_s = sum(int(x.get("bytes", 0)) for x in stage_w), sum(float(x.get("secs", 0)) for x in stage_w)
            self.rep["servers"][srv["name"]]["stage_write"] = dict(bytes=sw_b, secs=sw_s, gb_s=round(sw_b / 1e9 / max(sw_s, 1e-9), 3))
        info = dict(rc=rc, secs=round(time.time() - t, 1), tail=tail[-800:], bytes=b, methods=sorted({x.get("method") for x in dl if x.get("method")}))
        rec["download"] = info
        self.phase("weight_download", t, time.time(), srv=srv, ok=rc == 0,
                   detail=dict(bytes=b, source=",".join(info["methods"]), mb_s=round(b / 1e6 / max(1e-9, time.time() - t), 1)))
        self.log(f"download {srv['name']}: rc={rc} {round(time.time() - t)}s {b / 1e9:.1f} GB via {info['methods']}")
        self.save()
        if rc != 0:
            raise RuntimeError(f"download failed: {rc}")

    def serve(self, srv):
        env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (srv.get("env") or {}).items())
        name = f"server_{srv['name']}"
        t = time.time()
        self.box.bg(name, f"{('export ' + env) if env else ''}\nexec {srv['launch']}")
        ok = False
        while time.time() - t < srv.get("ready_timeout_s", 1200) and self.left_s() > 0:
            r = self.box.exec("curl -s -o /dev/null -w '%{http_code}' " + srv.get("health_url", "http://127.0.0.1:8000/health"), timeout=30)
            if r["out"].strip() == "200":
                ok = True
                break
            if self.box.job_rc(name) is not None:
                break
            time.sleep(8)
        t_ready = time.time()
        info = dict(ready=ok, ready_s=round(t_ready - t, 1), launch=srv["launch"], env=srv.get("env") or {})
        info["gpu_mem_after_load"] = self.box.exec("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null"
                                                   " || amd-smi metric --mem-usage 2>/dev/null | head -8", timeout=30)["out"].strip()
        self.rep["servers"][srv["name"]]["serve"] = info
        self.pull(f"{name}.log", f"{name}.log")
        self.phase("engine_start", t, t_ready, srv=srv, ok=ok)
        facts = self.startup_facts(srv, info)
        self.log(f"server {srv['name']} ready={ok} after {info['ready_s']}s; gpu mem {info['gpu_mem_after_load']}; "
                 f"attn {facts.get('attention_backend')} kv_tokens {facts.get('kv_cache_tokens')} graphs {len(facts.get('cuda_graph_sizes') or [])}"
                 + (f" REJECTED: {facts.get('reject_reason')}" if facts.get("rejected") else ""))
        self.save()
        if ok:   # one tiny request so compile/warmup is not inside the first measurement
            t = time.time()
            body = json.dumps(dict(model=srv.get("served_name", "gemma"), messages=[dict(role="user", content="Say hi.")],
                                   max_tokens=8, temperature=0, chat_template_kwargs=dict(enable_thinking=False)))
            r = self.box.sh("smoke", f"curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' "
                            f"-d {shlex.quote(body)}", timeout=650)
            info["smoke"] = r["out"][-400:]
            self.phase("warmup", t, time.time(), srv=srv, ok='"choices"' in r["out"])
            self.save()
        return ok, name, facts

    def startup_facts(self, srv, info):
        p = self.out / f"server_{srv['name']}.log"
        txt = p.read_text(errors="replace") if p.exists() else ""
        f, ph = SF.parse(txt, engine_of(srv), year=int(time.strftime("%Y", time.gmtime())),
                         rules=srv.get("startup_rules") or self.cfg.get("startup_rules"))
        w = (srv.get("weights") or [{}])[0]
        f.update(schema="startup_facts/1", run_id=self.run_id, config_id=f"{self.cfg['label']}:{srv['name']}", image=self.snap["image"],
                 image_digest=self.snap.get("image_digest_amd64") or self.snap.get("image_digest_index"), model_repo=w.get("repo"),
                 model_rev=w.get("rev"), gpu_mem_after_load=info.get("gpu_mem_after_load"),
                 source_log=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p))
        for sp in SF.phases_from(ph):
            self.phase(sp["phase"], sp["t_start"], sp["t_end"], srv=srv, nested_in="engine_start", detail=sp.get("detail"))
        f["sub_phases"] = [dict(p, t_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p["t_start"])),
                                t_end=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p["t_end"]))) for p in SF.phases_from(ph)]
        (self.out / f"startup_facts_{srv['name']}.json").write_text(json.dumps(f, indent=1))
        self.rep["servers"][srv["name"]]["startup_facts"] = {k: f.get(k) for k in ("attention_backend", "kv_cache_tokens", "kv_cache_gib",
                                                                                    "rejected", "reject_reason", "max_model_len")}
        return f

    def run_steps(self, srv, server_job):
        cid = f"{self.cfg['label']}:{srv['name']}"
        for st in expand_steps(srv, self.pph["total"], self.run_id, cid, bar=price_bar(self.cfg)):
            rec = self.rep["servers"][srv["name"]].setdefault("steps", {})
            if self.left_s() < (st["est_s"] if st["kind"] not in ("matrix",) else 240):
                rec[st["name"]] = dict(skipped=f"needs ~{st['est_s']}s, {self.left_s():.0f}s left")
                self.log(f"skip {st['name']}: not enough time/budget left")
                continue
            t = time.time()
            self.current = (srv["name"], st["name"])
            cmd = st["cmd"]
            if st["kind"] == "matrix" and "--max-minutes" not in cmd:
                cmd += f" --max-minutes {max(2.0, (self.left_s() - 60) / 60):.1f}"
            if st["kind"] == "protocol":
                cmd += f" --max-minutes {max(2.0, (self.left_s() - 60) / 60):.1f}"
            self.box.bg(st["name"], cmd)
            rc = self.wait_job(st["name"], max_s=st["est_s"] * 3 + 300, every=10, server_job=server_job)
            tail = self.box.tail(st["name"], 3)
            rec[st["name"]] = dict(rc=rc, secs=round(time.time() - t, 1), cmd=cmd, tail=tail[-600:])
            if self.rep.get("evicted") and (self.rep["evicted"].get("interrupted") or {}).get("step") == st["name"]:
                rec[st["name"]]["excluded"] = "evicted"   # never part of a benchmark number (Maks 11-22)
            if "breakeven_rps" in st:
                rec[st["name"]]["breakeven_rps"] = st["breakeven_rps"]
            for o in st["outputs"]:
                if o.endswith("/"):
                    self.pull_dir(o)
                elif "*" in o:
                    for f in self.box.exec(f"bash -c {shlex.quote(f'cd {W} && ls -1 {o}')}", timeout=30)["out"].split():
                        self.pull(f, f.replace("/", "__"))
                else:
                    self.pull(o, pathlib.Path(o).name)
            self.pull(f"{st['name']}.log", f"{st['name']}.log")
            cells = sorted(str(p.relative_to(self.out)) for p in (self.out / st["name"]).glob("cells/*.json")) if st["kind"] in ("matrix", "protocol") else None
            self.phase("eval" if st["kind"] == "eval" else "steady", t, time.time(), srv=srv, ok=rc == 0, cell_ids=cells,
                       detail=dict(step=st["name"], kind=st["kind"]))
            self.log(f"{st['name']} rc={rc} {round(time.time() - t)}s :: {tail.strip().splitlines()[-1][:220] if tail.strip() else ''}")
            if rc == 0 and not self.dead:
                self.last_good_t = time.time()
                self.current = None
            self.save()
            if st["kind"] == "eval" and st.get("gate"):
                ef = self.out / f"{st['name']}.json"
                g = json.loads(ef.read_text()).get("gate") if ef.exists() else None
                rec[st["name"]]["gate"] = g
                if not g or not g.get("pass"):
                    self.log(f"GATE FAILED {st['name']}: {g} -> no further measurement on {srv['name']}")
                    return "gate failed"
            if rc in ("sandbox gone", "server died", "budget"):
                return rc
        return "ok"

    def go(self):
        c = self.cfg
        try:
            self.setup()
            for srv in c["servers"]:
                self.rep["servers"][srv["name"]] = {}
                deps = srv.get("skip_if_ok") or []   # a backup server: runs only if none of the named servers came up and ran ok
                deps = [deps] if isinstance(deps, str) else deps
                ok_dep = next((d for d in deps if str((self.rep["servers"].get(d) or {}).get("end", "")).startswith("ok")), None)
                if ok_dep:
                    self.rep["servers"][srv["name"]]["skipped"] = f"{ok_dep} ran ok"
                    self.log(f"server {srv['name']} skipped: {ok_dep} ran ok")
                    continue
                if self.left_s() < 300:
                    self.rep["servers"][srv["name"]]["skipped"] = "no time/budget left"
                    continue
                self.current = (srv["name"], "download+serve")
                self.download(srv)
                ok, server_job, facts = self.serve(srv)
                # Same-box fallbacks: a failed or rejected engine start retries with extra flags/env on the SAME sandbox
                # (weights already on disk) instead of giving a scarce spot GPU back. Item: "<flags>" or {"flags", "env"}.
                attempts = [dict(launch=srv["launch"], env=srv.get("env") or {}, ok=ok, rejected=bool(facts.get("rejected")),
                                 reason=facts.get("reject_reason") if ok else "engine did not become healthy")]
                for fb in srv.get("fallbacks") or []:
                    if (ok and (not facts.get("rejected") or srv.get("allow_rejected"))) or self.left_s() < 600 or self.dead:
                        break
                    fb = {"flags": fb} if isinstance(fb, str) else fb
                    self.log(f"engine {srv['name']} {'rejected' if ok else 'failed'}; same-box fallback: +{fb.get('flags', '')} env {fb.get('env', {})}")
                    self.pull(f"{server_job}.log", f"{server_job}_attempt{len(attempts)}.log")
                    self.kill_engines()
                    # a fallback may set allow_rejected: a last-resort config that is measured and labelled, not dropped
                    srv = dict(srv, launch=(srv["launch"] + " " + fb.get("flags", "")).strip(), env={**(srv.get("env") or {}), **fb.get("env", {})},
                               allow_rejected=fb.get("allow_rejected", srv.get("allow_rejected")))
                    ok, server_job, facts = self.serve(srv)
                    attempts.append(dict(launch=srv["launch"], env=srv["env"], ok=ok, rejected=bool(facts.get("rejected")),
                                         allow_rejected=bool(srv.get("allow_rejected")),
                                         reason=facts.get("reject_reason") if ok else "engine did not become healthy"))
                self.rep["servers"][srv["name"]]["launch_attempts"] = attempts
                if ok:
                    self.last_good_t, self.current = time.time(), None
                if ok and facts.get("rejected") and not srv.get("allow_rejected"):
                    self.rep["servers"][srv["name"]]["end"] = f"rejected: {facts.get('reject_reason')}"
                elif ok:
                    why = self.run_steps(srv, server_job)
                    self.rep["servers"][srv["name"]]["end"] = why
                    m = self.box.exec("curl -s http://127.0.0.1:8000/metrics", timeout=30)["out"]
                    (self.out / f"metrics_{srv['name']}.prom").write_text(m)
                self.pull(f"{server_job}.log", f"{server_job}.log")
                if self.dead:
                    break
                self.kill_engines()
                if srv.get("delete_weights_after", True):
                    self.box.exec(f"rm -rf {' '.join(sorted({w['dir'] for w in srv.get('weights', [])}))}", timeout=120)
            self.pull("gpumon.log", "gpumon.csv")
        except Exception as e:  # noqa: BLE001
            self.rep["error"] = repr(e)[:1500]
            self.log(f"ERROR {e!r}"[:600])
            if self.sb is not None and not self.dead:
                self.alive()          # an exception from a vanished sandbox is classified as an eviction here
            if self.box is not None and not self.dead:
                for f in ["gpumon.log"] + [f"server_{s['name']}.log" for s in c["servers"]]:
                    self.pull(f, f"failed_{f}" if f != "gpumon.log" else "gpumon.csv")
        finally:
            self.teardown()

    def teardown(self):
        if self.sb is not None and not self.rep.get("deleted"):
            t = time.time()
            try:
                dt.delete_sandbox(self.sb["id"], why=f"run.py end ({self.cfg['label']})", run_id=self.run_id)
                self.rep["deleted"] = dt.now_iso()
                self.phase("teardown", t, time.time())
            except Exception as e:  # noqa: BLE001
                self.rep["delete_error"] = repr(e)[:300]
                self.log(f"DELETE FAILED {e!r} -- the reaper (openrelay-reap.timer, 5 min) will catch it")
            t2 = time.time()
            gone = False
            while time.time() - t2 < 120:
                try:
                    st = str(dt.get_sandbox(self.sb["id"]).get("state", "")).lower()
                    gone = st in ("destroyed",)
                except dt.ApiError as e:
                    gone = e.status == 404
                if gone:
                    break
                time.sleep(5)
            self.phase("reap_confirm", t2, time.time(), ok=gone, usd_per_h=0.0, detail=dict(gone=gone))
            self.rep["delete_confirmed"] = gone
        try:
            self.write_cost_measured()
            self.write_gpumon_norm()
        except Exception as e:  # noqa: BLE001
            self.rep["post_error"] = repr(e)[:300]
        self.save()
        try:
            est, metered, crossed = reconcile_spend()
            self.rep["project_spend"] = dict(est_usd=round(est, 2), metered_usd=metered)
            for th in crossed:
                self.log(f"MILESTONE project spend crossed ${th} (est ${est:.2f}, metered {metered})")
        except Exception as e:  # noqa: BLE001
            self.rep["spend_error"] = repr(e)[:300]
        self.save()
        self.log(f"done: elapsed {self.elapsed() / 60:.1f} min, est ${self.cost():.2f}; project est ${self.rep.get('project_spend', {}).get('est_usd')}")

    def write_cost_measured(self):
        out = []
        tsum = lambda names, cid=None: sum(p["usd"] for p in self.phases if p["phase"] in names and not p["nested_in"]  # noqa: E731
                                           and (cid is None or p["config_id"] in (cid, None)))
        whole = dict(create=tsum({"create"}), failed_create=tsum({"failed_create"}), teardown=tsum({"teardown"}))
        life_usd = self.cost()
        for srv in self.cfg["servers"]:
            cid = f"{self.cfg['label']}:{srv['name']}"
            mine = [p for p in self.phases if p["config_id"] == cid and not p["nested_in"]]
            cold = {k: sum(p["usd"] for p in mine if p["phase"] == k) for k in ("weight_download", "engine_start", "warmup")}
            steady = sum(p["usd"] for p in mine if p["phase"] == "steady")
            ev = sum(p["usd"] for p in mine if p["phase"] == "eval")
            ps = self.out / f"protocol_{srv['name']}" / "protocol_summary.json"
            proto = json.loads(ps.read_text()) if ps.exists() else None
            rate = (proto or {}).get("max_rate_at_slo_rps") or 0.0
            steady_req = (self.pph["total"] / 3600 / rate) if rate else None
            _, or_req = breakeven_rps(self.pph["total"], bar=price_bar(self.cfg))
            cold_usd = sum(cold.values()) + whole["create"] / max(1, len(self.cfg["servers"]))
            payback = (cold_usd / ((or_req - steady_req) * rate * 60)) if steady_req and or_req > steady_req else None
            out.append(dict(run_id=self.run_id, config_id=cid, gpu=self.snap["gpu"], usd_per_h=self.pph,
                            steady_usd=round(steady, 4), steady_gpu_usd=round(steady * self.pph["gpu"] / self.pph["total"], 4),
                            steady_host_usd=round(steady * (1 - self.pph["gpu"] / self.pph["total"]), 4),
                            eval_usd=round(ev, 4), cold_start_usd=dict({k: round(v, 4) for k, v in cold.items()},
                                                                       create_share=round(whole["create"] / max(1, len(self.cfg["servers"])), 4)),
                            eviction_usd=round(sum(p["usd"] for p in self.phases if p["phase"] == "eviction"), 4),
                            failed_create_usd=round(whole["failed_create"], 4), run_total_est_usd=round(life_usd, 4),
                            max_rate_at_slo_rps=rate or None, steady_usd_per_req=steady_req, openrelay_sync_usd_per_req=or_req,
                            payback_min=payback, basis=("cold start = this config's share of create + its download, engine start and warmup; "
                                                        "payback = cold start / ((OpenRelay sync $/req - our steady $/req) x max_rate_at_slo x 60)")))
        (self.out / "cost_measured.json").write_text(json.dumps(dict(run_id=self.run_id, configs=out), indent=1))

    def write_gpumon_norm(self):
        normalize_gpumon(self.out / "gpumon.csv", self.out / "gpumon_norm.csv")

def normalize_gpumon(src, dst):
    """gpumon.csv (nvidia-smi rows, or amd-smi `metric --usage --power --mem-usage --clock --temperature --csv` rows each after
    a "T <utc>" line) -> gpumon_norm.csv with one schema. AMD field positions (MI355X recipe session, read off a live row
    2026-11-22): [1] gfx_activity %, [9] socket power W, [15,20,..,50] GFX clock MHz per XCD (sm_mhz = their mean);
    AMD memory and temperature columns are left empty (positions not pinned)."""
    import csv
    import io
    import statistics
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    if not src.exists():
        return 0
    rows = ["t_utc,gpu_index,util_pct,mem_used_mib,power_w,sm_mhz,temp_c"]
    num = lambda s: re.sub(r"[^\d.]", "", s) or ""  # noqa: E731
    t_amd = None
    for line in src.read_text(errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 7 and re.match(r"\d{4}/\d\d/\d\d", parts[0]):   # nvidia-smi
            tt = parts[0].replace("/", "-").replace(" ", "T").split(".")[0] + "Z"
            rows.append(",".join([tt, parts[1]] + [num(x) for x in parts[2:]]))
            continue
        m = re.match(r"^T (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", line)
        if m:
            t_amd = m.group(1) + "Z"
            continue
        if t_amd and line[:1].isdigit():                                      # amd-smi csv row
            r = next(csv.reader(io.StringIO(line)))
            if len(r) < 51:
                continue
            try:
                clk = statistics.mean(float(r[i]) for i in range(15, 51, 5))
                rows.append(",".join([t_amd, r[0], num(r[1]), "", num(r[9]), f"{clk:.0f}", ""]))
            except ValueError:
                continue
    dst.write_text("\n".join(rows) + "\n")
    return len(rows) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--evict-retries", type=int, default=2, help="re-run the whole config on a fresh box after a spot eviction")
    a = ap.parse_args()
    cfg = json.loads(pathlib.Path(a.config).read_text())
    assert cfg["max_minutes"] <= 240 and cfg["budget_usd"] <= 60, "per-run caps: 240 min, $60"
    snap = cfg["snapshot"]
    pph = dt.hourly_usd(snap["gpu"], snap.get("gpus", 1), snap["cpu"], snap["mem"], snap["disk"])
    plan = []
    for s in cfg["servers"]:
        steps = expand_steps(s, pph["total"], "dry-run", f"{cfg['label']}:{s['name']}", bar=price_bar(cfg))
        plan.append(dict(server=s["name"], steps=[x["name"] for x in steps], est_min=round(sum(x["est_s"] for x in steps) / 60, 1),
                         breakeven_rps=[round(x["breakeven_rps"], 3) for x in steps if "breakeven_rps" in x]))
    ceiling = min(cfg["budget_usd"], pph["total"] * (cfg["max_minutes"] + 5) / 60)
    be, or_req = breakeven_rps(pph["total"], bar=price_bar(cfg))
    print(json.dumps(dict(label=cfg["label"], usd_per_h=pph, max_minutes=cfg["max_minutes"], budget_usd=cfg["budget_usd"],
                          cost_ceiling_usd=round(ceiling, 2), breakeven_rps_9500_380=round(be, 3), openrelay_usd_per_req=round(or_req, 6),
                          project_spend_est_usd=round(ledger_spend_estimate(), 2), plan=plan), indent=1))
    if a.dry_run:
        return 0
    r = Run(cfg, a.config)

    def on_signal(sig, frm):
        r.log(f"signal {sig}: tearing down")
        r.teardown()
        sys.exit(130)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    r.go()
    tries = 0
    while r.rep.get("evicted") and tries < a.evict_retries:
        tries += 1
        r.log(f"re-running {cfg['label']} on a fresh box after spot eviction (retry {tries}/{a.evict_retries}); "
              f"evicted run kept as {r.run_id} with the interrupted step excluded")
        r = Run(cfg, a.config)
        r.rep["rerun_of_evicted"] = tries
        r.go()
    return 0 if not r.rep.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
