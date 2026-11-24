#!/usr/bin/env python3
"""Fast, pinned Hugging Face download inside a sandbox.
Uses huggingface_hub.snapshot_download (Xet-aware, parallel chunks) when the image has it; otherwise its own parallel
HTTP range downloader (stdlib: 32 threads x 16 MiB ranges per file), because one curl stream per file ran at ~107 MB/s on
the 2026-11-22 pilot while hf_hub reached ~1.4 GB/s on Daytona earlier. Files that already exist at the expected size are
skipped. Prints one line: DL repo rev files bytes secs GB/s method.

  python3 bench/hfdl.py --repo RedHatAI/gemma-4-31B-it-FP8-dynamic --rev <sha> --dir /root/models/fp8dyn [--files a,b] [--sizes 1,2]
      [--from-stage /vol/hf]   read a complete staged copy (Daytona volume) instead of Hugging Face, if one exists
      [--stage-dir /vol/hf]    after downloading, copy the checkpoint once to <stage-dir>/<repo>@<rev>/ for later boxes
Each path prints its own DL line (method=huggingface_hub | range | staged | stage-write) with bytes, seconds, GB/s.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time
import urllib.request

UA = {"User-Agent": "openrelay-bench/0.1"}


def ranged(url, dst, size, chunk=16 << 20, workers=32):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with open(tmp, "wb") as f:
        f.truncate(size)

    def get(off):
        end = min(size, off + chunk) - 1
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={off}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) != end - off + 1:
                    raise IOError(f"short read {len(data)}")
                with open(tmp, "r+b") as f:
                    f.seek(off)
                    f.write(data)
                return len(data)
            except Exception:  # noqa: BLE001
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))
    with cf.ThreadPoolExecutor(workers) as ex:
        sum(ex.map(get, range(0, size, chunk)))
    os.replace(tmp, dst)


def par_copy(src_dir, dst_dir, files, workers=16, chunk=64 << 20):
    """Copy files with many concurrent readers (one FUSE/S3 stream is slow; many ranged reads are not)."""
    jobs = []
    for f in files:
        s, d = os.path.join(src_dir, f), os.path.join(dst_dir, f)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        size = os.path.getsize(s)
        with open(d + ".part", "wb") as o:
            o.truncate(size)
        jobs += [(s, d + ".part", off, min(chunk, size - off)) for off in range(0, size, chunk)] or [(s, d + ".part", 0, 0)]

    def cp(j):
        s, d, off, n = j
        with open(s, "rb") as i, open(d, "r+b") as o:
            i.seek(off)
            o.seek(off)
            o.write(i.read(n))
        return n
    with cf.ThreadPoolExecutor(workers) as ex:
        total = sum(ex.map(cp, jobs))
    for f in files:
        os.replace(os.path.join(dst_dir, f) + ".part", os.path.join(dst_dir, f))
    return total


def staged_path(stage_root, repo, rev):
    return os.path.join(stage_root, repo.replace("/", "__") + "@" + rev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--files", default="", help="comma list (default: all, via hf_hub) ")
    ap.add_argument("--sizes", default="", help="comma list matching --files (needed for the range fallback)")
    ap.add_argument("--force-range", action="store_true")
    ap.add_argument("--from-stage", default="")
    ap.add_argument("--stage-dir", default="")
    a = ap.parse_args()
    files = [f for f in a.files.split(",") if f]
    sizes = [int(s) for s in a.sizes.split(",") if s]
    t0 = time.time()
    if a.from_stage:
        sp = staged_path(a.from_stage, a.repo, a.rev)
        if files and all(os.path.exists(os.path.join(sp, f)) and os.path.getsize(os.path.join(sp, f)) == sz for f, sz in zip(files, sizes)):
            b = par_copy(sp, a.dir, files)
            secs = time.time() - t0
            print(f"DL repo={a.repo} rev={a.rev[:10]} files={len(files)} bytes={b} secs={secs:.1f} GBps={b / 1e9 / max(secs, 1e-9):.3f} method=staged", flush=True)
            return 0
        print(f"DL staged copy missing or incomplete at {sp}; falling back to Hugging Face", flush=True)
    method = "range"
    try:
        if a.force_range:
            raise ImportError
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=a.repo, revision=a.rev, local_dir=a.dir, max_workers=16, allow_patterns=files or None)
        method = "huggingface_hub"
    except ImportError:
        todo = []
        for f, sz in zip(files, sizes):
            dst = os.path.join(a.dir, f)
            if os.path.exists(dst) and os.path.getsize(dst) == sz:
                continue
            todo.append((f, sz))
        for f, sz in sorted(todo, key=lambda x: -x[1]):
            url = f"https://huggingface.co/{a.repo}/resolve/{a.rev}/{f}"
            if sz < (32 << 20):
                os.makedirs(os.path.dirname(os.path.join(a.dir, f)), exist_ok=True)
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(os.path.join(a.dir, f), "wb") as o:
                    o.write(r.read())
            else:
                ranged(url, os.path.join(a.dir, f), sz)
    secs = time.time() - t0
    b = sum(os.path.getsize(os.path.join(dp, x)) for dp, _, fs in os.walk(a.dir) for x in fs)
    print(f"DL repo={a.repo} rev={a.rev[:10]} files={len(files) or 'all'} bytes={b} secs={secs:.1f} GBps={b / 1e9 / max(secs, 1e-9):.3f} method={method}", flush=True)
    if a.stage_dir and files:
        sp = staged_path(a.stage_dir, a.repo, a.rev)
        todo = [f for f, sz in zip(files, sizes) if not (os.path.exists(os.path.join(sp, f)) and os.path.getsize(os.path.join(sp, f)) == sz)]
        t1 = time.time()
        bw = par_copy(a.dir, sp, todo) if todo else 0
        s2 = time.time() - t1
        print(f"DL repo={a.repo} rev={a.rev[:10]} files={len(todo)} bytes={bw} secs={s2:.1f} GBps={bw / 1e9 / max(s2, 1e-9):.3f} method=stage-write", flush=True)


if __name__ == "__main__":
    sys.exit(main())
