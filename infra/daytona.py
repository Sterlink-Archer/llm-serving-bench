#!/usr/bin/env python3
"""Minimal Daytona REST client for the OpenRelay serving-cost demo. Python stdlib only (no SDK, nothing to vet).

Safety rules enforced here, not left to callers:
  * every GPU sandbox is created with spot=True (preemptible) and is deleted at once if Daytona returns it non-spot;
  * every sandbox gets ttlMinutes (hard wall-clock limit), autoStopInterval and autoDeleteInterval=0 (deleted on stop);
  * every sandbox carries labels project=openrelay, run=<label>, deadline=<unix ts>, which infra/reap.py enforces;
  * every create/delete is appended to infra/ledger.jsonl (id, time, gpu, $/h) as soon as Daytona returns an id.
The API key is read from the handoff file for each call and never printed or logged.

API shapes were taken from the daytona 0.232.0 Python SDK source (daytona_api_client / daytona_toolbox_api_client).
Remote commands are shipped as script FILES and run by path, so every command that ran can be read afterwards.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import shlex
import ssl
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Where the Daytona API key FILE lives and which org to use: env DAYTONA_KEY_FILE / DAYTONA_ORG_ID, else the gitignored
# infra/local_settings.json {"daytona_key_file": "...", "daytona_org_id": "..."}. Neither the key nor its path is in git.
_LOCAL = ROOT / "infra" / "local_settings.json"
_SET = json.loads(_LOCAL.read_text()) if _LOCAL.exists() else {}
KEY_FILE = pathlib.Path(os.environ.get("DAYTONA_KEY_FILE") or _SET.get("daytona_key_file") or "DAYTONA_KEY_FILE-not-set")
ORG = os.environ.get("DAYTONA_ORG_ID") or _SET.get("daytona_org_id") or ""
API = "https://app.daytona.io/api"
LEDGER = ROOT / "infra" / "ledger.jsonl"
CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt"))
PROJECT_LABEL = "openrelay"
WORK = "/root/or"

# Daytona preemptible (spot) GPU prices, $/GPU-hour, read from https://www.daytona.io/pricing at 2026-11-22T03:05:55Z
# (raw page: research/raw/daytona_pricing_20261122T030555Z.html). Host resources are billed on top (same page).
GPU_SPOT_USD_H = {"B300": 5.17, "B200": 4.30, "MI355X": 3.56, "H200": 2.87, "H100": 2.47,
                  "RTX-PRO-6000": 1.95, "L40S": 1.14, "RTX-5090": 0.74, "RTX-4090": 0.57}
GPU_ONDEMAND_USD_H = {"B300": 8.99, "B200": 7.49, "MI355X": 6.20, "H200": 4.99, "H100": 4.29,
                      "RTX-PRO-6000": 3.39, "L40S": 1.99, "RTX-5090": 1.29, "RTX-4090": 0.99}
VCPU_USD_H, MEM_GIB_USD_H, DISK_GIB_USD_H, DISK_FREE_GIB = 0.0504, 0.0162, 0.000108, 5


def hourly_usd(gpu_type: str, gpus: int, cpu: int, mem_gib: int, disk_gib: int, spot: bool = True) -> dict:
    """Full $/h of one sandbox: GPUs + vCPU + RAM + disk. Returned as parts so the write-up can show the split."""
    g = (GPU_SPOT_USD_H if spot else GPU_ONDEMAND_USD_H)[gpu_type] * gpus
    parts = dict(gpu=g, cpu=cpu * VCPU_USD_H, mem=mem_gib * MEM_GIB_USD_H, disk=max(0, disk_gib - DISK_FREE_GIB) * DISK_GIB_USD_H)
    parts["total"] = sum(parts.values())
    return {k: round(v, 4) for k, v in parts.items()}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ApiError(RuntimeError):
    def __init__(self, status, body, path):
        super().__init__(f"HTTP {status} on {path}: {str(body)[:500]}")
        self.status, self.body, self.path = status, body, path


def _req(method: str, url: str, body=None, timeout=60):
    key = KEY_FILE.read_text().strip()
    h = {"Authorization": "Bearer " + key, "X-Daytona-Organization-ID": ORG, "Accept": "application/json",
         "X-Daytona-Source": "openrelay-bench"}
    del key
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, context=CTX, timeout=timeout) as resp:
            b = resp.read()
            return json.loads(b) if b else None
    except urllib.error.HTTPError as e:
        raise ApiError(e.code, e.read().decode(errors="replace")[:2000], url.split("?")[0].replace(API, "")) from None


def api(method: str, path: str, body=None, timeout=60):
    return _req(method, API + path, body, timeout)


# ---------------------------------------------------------------- capacity, snapshots
def gpu_capacity() -> dict:
    return api("GET", f"/organizations/{ORG}/gpu-capacity")


def spot_available(gpu_type: str, count: int = 1) -> int:
    for g in gpu_capacity().get("capacity", []):
        if g["gpuType"] == gpu_type:
            for s in g.get("availableSandboxes", []):
                if s["gpuCount"] == count:
                    return int(s.get("spot") or 0)
    return 0


def list_snapshots() -> list:
    d = api("GET", "/snapshots?limit=200")
    return d.get("items", d) if isinstance(d, dict) else d


def get_snapshot(name: str):
    for s in list_snapshots():
        if s.get("name") == name:
            return s
    return None


def create_snapshot(name: str, image: str, gpu_type: str, gpus: int, cpu: int, mem_gib: int, disk_gib: int,
                    region: str = "earth", wait_s: int = 3600) -> dict:
    """GPU snapshots live in region 'earth'. Host resources are baked into the snapshot (a sandbox inherits them)."""
    s = get_snapshot(name)
    if s and str(s.get("state", "")).lower() == "active":
        return s
    if not s:   # an existing snapshot still pulling/building (e.g. a restarted run) is waited for below, not returned early
        body = dict(name=name, imageName=image, entrypoint=["sleep", "infinity"], cpu=cpu, gpu=gpus, gpuType=[gpu_type],
                    memory=mem_gib, disk=disk_gib, regionId=region)
        s = api("POST", "/snapshots", body)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        s = get_snapshot(name) or s
        st = str(s.get("state", "")).lower()
        if st == "active":
            return s
        if st in ("error", "build_failed"):
            raise RuntimeError(f"snapshot {name} failed: {s.get('errorReason')}")
        time.sleep(15)
    raise TimeoutError(f"snapshot {name} not active after {wait_s}s")


# ---------------------------------------------------------------- sandboxes
def list_sandboxes(labels: dict | None = None) -> list:
    items, cursor = [], None
    while True:
        d = api("GET", "/sandbox" + (f"?cursor={cursor}" if cursor else ""))
        items += d.get("items", []) if isinstance(d, dict) else d
        cursor = d.get("nextCursor") if isinstance(d, dict) else None
        if not cursor:
            break
    if labels:
        items = [s for s in items if all((s.get("labels") or {}).get(k) == v for k, v in labels.items())]
    return items


def get_sandbox(sid: str) -> dict:
    return api("GET", f"/sandbox/{sid}")


def ledger(rec: dict):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(t=now_iso(), **rec)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def delete_sandbox(sid: str, why: str = "", run_id: str | None = None):
    try:
        r = api("DELETE", f"/sandbox/{sid}")
        ledger(dict(event="delete", id=sid, run_id=run_id, why=why))
        return r
    except ApiError as e:
        if e.status == 404:
            ledger(dict(event="delete", id=sid, run_id=run_id, why=why + " (already gone: 404)"))
            return None
        raise


def ensure_volume(name: str, wait_s: int = 300) -> dict:
    """A Daytona volume (S3-backed, mounted into sandboxes): created once, reused by name."""
    try:
        v = api("GET", f"/volumes/by-name/{name}")
    except ApiError as e:
        if e.status != 404:
            raise
        v = api("POST", "/volumes", dict(name=name))
    t0 = time.time()
    while str(v.get("state", "")).lower() not in ("ready", "active") and time.time() - t0 < wait_s:
        time.sleep(5)
        v = api("GET", f"/volumes/by-name/{name}")
    return v


def create_gpu_sandbox(snapshot: dict, run_label: str, max_minutes: int, wait_s: int = 900, run_id: str | None = None,
                       volumes: list | None = None) -> dict:
    """Create ONE preemptible GPU sandbox from a GPU snapshot dict (from get_snapshot).
    Returns the sandbox DTO once state == started. Records it in the ledger as soon as Daytona returns an id."""
    gpu_type = snapshot.get("gpuType")
    gpu_type = gpu_type[0] if isinstance(gpu_type, list) else gpu_type
    gpus = int(snapshot.get("gpu") or 0)
    assert gpus >= 1 and gpu_type, "only GPU snapshots (spot needs a GPU)"
    assert 5 <= max_minutes <= 240, "ttl must be set and bounded"
    deadline = int(time.time() + max_minutes * 60)
    body = dict(snapshot=snapshot["name"], spot=True, autoStopInterval=15, autoDeleteInterval=0, ttlMinutes=max_minutes,
                labels={"project": PROJECT_LABEL, "run": run_label[:60], "deadline": str(deadline)})
    if volumes:
        body["volumes"] = volumes
    t0 = time.time()
    sb = api("POST", "/sandbox", body, timeout=180)
    price = hourly_usd(gpu_type, gpus, int(snapshot.get("cpu") or 0), int(snapshot.get("mem") or 0), int(snapshot.get("disk") or 0))
    ledger(dict(event="create", id=sb["id"], run=run_label, run_id=run_id, snapshot=snapshot["name"], gpu=gpu_type, gpus=gpus,
                cpu=snapshot.get("cpu"), mem=snapshot.get("mem"), disk=snapshot.get("disk"), spot=sb.get("spot"),
                usd_per_h=price, ttl_min=max_minutes, deadline=deadline))
    if sb.get("spot") is not True:   # never run work on an on-demand GPU
        delete_sandbox(sb["id"], why=f"returned spot={sb.get('spot')!r}, refusing non-preemptible", run_id=run_id)
        raise RuntimeError("Daytona did not return a spot sandbox; deleted it")
    while time.time() - t0 < wait_s:
        st = str(sb.get("state", "")).lower()
        if st == "started":
            sb["_create_s"] = round(time.time() - t0, 1)
            return sb
        if st in ("error", "build_failed", "destroyed", "stopped"):
            delete_sandbox(sb["id"], why=f"create ended in {st}", run_id=run_id)
            raise RuntimeError(f"sandbox {sb['id']} -> {st}: {sb.get('errorReason')} evicted={sb.get('spotEvictedAt')}")
        time.sleep(3)
        sb = get_sandbox(sb["id"])
    delete_sandbox(sb["id"], why="not started in time", run_id=run_id)
    raise TimeoutError("sandbox did not start")


class Box:
    """Toolbox access to one sandbox: exec, small-file transfer, background jobs.
    Files move as base64 text decoded INTO A FILE (never into a shell); jobs are script files run by path."""

    def __init__(self, sb: dict):
        self.id = sb["id"]
        self.proxy = sb.get("toolboxProxyUrl") or api("GET", f"/sandbox/{self.id}/toolbox-proxy-url")["url"]
        self.base = self.proxy.rstrip("/") + "/" + self.id
        self.n = 0

    def exec(self, cmd: str, timeout: int = 120, tries: int = 4) -> dict:
        for attempt in range(tries):
            t0 = time.time()
            try:
                r = _req("POST", self.base + "/process/execute", dict(command=cmd, timeout=timeout), timeout=timeout + 30)
                rc = r.get("exitCode") if r.get("exitCode") is not None else r.get("code")   # SDK falls back to "code" too
                return dict(rc=rc, out=r.get("result") or "", wall_s=round(time.time() - t0, 1))
            except ApiError as e:
                if e.status not in (502, 503, 504) or attempt == tries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == tries - 1:
                    raise
            time.sleep(5 * (attempt + 1))

    def put_bytes(self, data: bytes, remote: str, chunk: int = 48_000):
        b = base64.b64encode(data).decode()
        q = shlex.quote(remote)
        self.exec(f"bash -c {shlex.quote(f'mkdir -p $(dirname {q}) && rm -f {q}.b64')}", timeout=30)
        for i in range(0, len(b), chunk):
            r = self.exec(f"bash -c {shlex.quote(f'printf %s {b[i:i + chunk]} >> {q}.b64')}", timeout=60)
            if r["rc"] not in (0, None):
                raise RuntimeError(f"upload chunk failed: {r['out'][:200]}")
        r = self.exec(f"bash -c {shlex.quote(f'base64 -d {q}.b64 > {q} && rm {q}.b64 && wc -c < {q}')}", timeout=60)
        if r["out"].strip() != str(len(data)):
            raise RuntimeError(f"upload size mismatch for {remote}: {r['out'][:200]} vs {len(data)}")

    def put(self, local, remote: str):
        self.put_bytes(pathlib.Path(local).read_bytes(), remote)

    def get(self, remote: str, local, chunk: int = 3_000_000) -> bool:
        q = shlex.quote(remote)
        r = self.exec(f"bash -c {shlex.quote(f'test -f {q} && wc -c < {q}')}", timeout=30)
        if r["rc"] not in (0, None) or not r["out"].strip().isdigit():
            return False
        size = int(r["out"].strip())
        buf = bytearray()
        for off in range(0, max(size, 1), chunk):
            r = self.exec(f"bash -c {shlex.quote(f'tail -c +{off + 1} {q} | head -c {chunk} | base64 -w0')}", timeout=120)
            buf += base64.b64decode(r["out"].strip())
        pathlib.Path(local).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(local).write_bytes(bytes(buf[:size]))
        return len(buf) >= size

    def script(self, name: str, body: str) -> str:
        """Write a bash script file into the sandbox and return its path (kept for the record)."""
        self.n += 1
        path = f"{WORK}/jobs/{self.n:03d}_{name}.sh"
        self.put_bytes(("#!/bin/bash\n" + body + "\n").encode(), path)
        return path

    def sh(self, name: str, body: str, timeout: int = 120) -> dict:
        return self.exec(f"bash {self.script(name, body)}", timeout=timeout)

    def bg(self, name: str, body: str) -> str:
        """Start a background job. Output -> {WORK}/{name}.log, exit code -> {WORK}/{name}.rc."""
        inner = self.script(name, f"cd {WORK}\n{body}")
        wrapper = self.script(name + "_bg", f"rm -f {WORK}/{name}.rc\n( bash {inner} ) > {WORK}/{name}.log 2>&1\necho $? > {WORK}/{name}.rc")
        self.exec(f"bash -c {shlex.quote(f'nohup bash {wrapper} > /dev/null 2>&1 & echo $!')}", timeout=30)
        return wrapper

    def job_rc(self, name: str):
        r = self.exec(f"bash -c {shlex.quote(f'cat {WORK}/{name}.rc 2>/dev/null || echo RUNNING')}", timeout=30)
        s = r["out"].strip()
        return None if s == "RUNNING" else (int(s) if s.lstrip("-").isdigit() else s)

    def tail(self, name: str, n: int = 20) -> str:
        return self.exec(f"tail -n {n} {WORK}/{name}.log", timeout=30)["out"]
