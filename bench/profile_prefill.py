#!/usr/bin/env python3
"""Where does prefill time go? Run against a vLLM server started with
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"<DIR>","torch_profiler_with_stack":false}'
it POSTs /start_profile, sends N concurrent unique prompts of T tokens with max_tokens=1 (pure prefill), POSTs
/stop_profile, waits for the trace, and sums GPU kernel time by category:
  gemm       GEMM / matmul kernels (cutlass, cublas, nvjet, fp4/nvfp4, xmma, marlin)
  attention  attention kernels (flashinfer, fmha, flash_attn, paged attention, triton attention)
  other      everything else (norms, activations, rope, sampling, copies, ...)
Writes a summary JSON with the top kernels; the raw trace stays in the sandbox (large).

  python3 bench/profile_prefill.py --model gemma --dir /root/or/out/prof --n 4 --tokens 10000 --out /root/or/out/prof_summary.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import gzip
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workloads as W  # noqa: E402

# Order matters: attention is tested FIRST. FA3/FA4 kernels are cutlass templates ("cutlass::device_kernel<flash::
# FlashAttnFwdSm90<...>>"), so a gemm-first test filed all attention time under gemm (H100 d1 run, 2026-11-22: gemm=95.7%,
# attention=0%). tests/test_profile_classify.py pins this.
# Inductor-generated pointwise/reduction/persistent kernels ("triton_red_fused_..._amax_clamp_cutlass_scaled_mm_...") are
# activation quant, norms and residual adds fused together; the op names inside them would otherwise match "gemm".
CATS = [("fused_elementwise", r"^triton_(red|per|poi)_|cvt_fp16_to_fp4|scaled_fp[48]_quant|per_token.*quant|dynamic.*quant|"
                              r"quantize_with_block_size|rmsnorm|rms_norm|layernorm|normkernel"),
        # explicit GEMM words first ("gemm" but not "gemma": SGLang's _gemma_qkv_rmsnorm_kernel is a norm). FlashInfer's FP4
        # GEMM is "...flashinfergemmkernelsdense_blockscaled_gemm_sm100...", which a bare "flashinfer" attention rule misfiled.
        ("gemm", r"gemm(?!a)|nvjet|cublas|xmma|marlin|matmul|wgmma"),
        # SGLang's Triton attention kernels are bare "_fwd_kernel" (extend/prefill) and "_fwd_kernel_stage1/2" (decode)
        ("attention", r"fmha|flash_?attn|flashattn|flash::|flash_fwd|attention|attn|prefill_kernel|batch_?prefill|batch_?decode|"
                      r"single_?prefill|paged|unified_attention|trtllm_.*(context|gen)|^_fwd_kernel"),
        ("gemm", r"cutlass|fp4|nvfp4|mxfp|sm\d+_.*_tn|_nt_|_tt_")]


def classify(name, cat="kernel"):
    if str(cat).lower() != "kernel":
        return "memcpy/memset"
    return next((c for c, rx in CATS if re.search(rx, name, flags=re.I)), "other")


def summarize_trace(trace):
    """GPU kernel time by category + top kernels from a torch profiler chrome trace (.json or .json.gz)."""
    op = gzip.open if trace.endswith(".gz") else open
    with op(trace, "rt") as f:
        d = json.load(f)
    ev = [e for e in d.get("traceEvents", []) if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("kernel", "gpu_memcpy", "gpu_memset")]
    tot, by, names, ncat = 0.0, {}, {}, {}
    for e in ev:
        dur = float(e.get("dur", 0))
        name = e.get("name", "")
        cat = classify(name, e["cat"])
        by[cat] = by.get(cat, 0.0) + dur
        names[name] = names.get(name, 0.0) + dur
        ncat[name] = cat
        tot += dur
    return dict(trace=trace, gpu_kernel_us_total=tot, kernel_events=len(ev),
                categories={k: dict(us=v, share=v / tot if tot else None) for k, v in sorted(by.items(), key=lambda kv: -kv[1])},
                top_kernels=[dict(name=n[:160], cat=ncat[n], us=v, share=v / tot if tot else None)
                             for n, v in sorted(names.items(), key=lambda kv: -kv[1])[:25]])


ADMIN_KEY = os.environ.get("OR_ADMIN_KEY", "")


def post(base, path, body=None, timeout=600):
    hdr = {"Content-Type": "application/json"}
    if ADMIN_KEY and path in ("/start_profile", "/stop_profile"):
        # SGLang (2026-11 main/nightly): the profiler endpoints are AuthLevel.ADMIN_FORCE -- HTTP 403 unless the server was
        # started with --admin-api-key, 401 without this Bearer header. A throwaway local token, never a credential
        # (Inkling session's recipes/inkling/box/sgl_profile.py found it). Inference endpoints stay keyless.
        hdr["Authorization"] = "Bearer " + ADMIN_KEY
    req = urllib.request.Request(base + path, data=json.dumps(body or {}).encode(), headers=hdr, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=10000)
    ap.add_argument("--cpt", type=float, default=4.07)
    ap.add_argument("--out", required=True)
    ap.add_argument("--admin-key", default="", help="SGLang --admin-api-key value (sent as a Bearer token to the profiler "
                    "endpoints only); default $OR_ADMIN_KEY")
    ap.add_argument("--reclassify", help="only re-summarize this existing trace file (no server calls)")
    a = ap.parse_args()
    global ADMIN_KEY
    ADMIN_KEY = a.admin_key or ADMIN_KEY
    if a.reclassify:
        out = summarize_trace(a.reclassify)
        prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
        prev.update(out, reclassified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        json.dump(prev, open(a.out, "w"), indent=1)
        print("[profile] reclassified: " + " ".join(f"{k}={v['share']:.1%}" for k, v in out["categories"].items() if v.get("share") is not None), flush=True)
        return
    T = W.Text(cpt=a.cpt)
    prompts = [f"[profile {time.time()}-{i}]\n" + T.prose_tokens(a.tokens, f"prof-{i}") + "\n\nSummarize." for i in range(a.n)]

    def one(p):
        body = dict(model=a.model, messages=[dict(role="user", content=p)], max_tokens=1, temperature=0,
                    chat_template_kwargs=dict(enable_thinking=False))
        t0 = time.time()
        _, b = post(a.base, "/v1/chat/completions", body)
        return time.time() - t0, json.loads(b)["usage"]["prompt_tokens"]
    one(prompts[0][:2000])   # warm path outside the profile
    before = set(glob.glob(os.path.join(a.dir, "**", "*"), recursive=True))
    post(a.base, "/start_profile")
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.n) as ex:
        res = list(ex.map(one, prompts))
    wall = time.time() - t0
    post(a.base, "/stop_profile", timeout=1800)
    trace = None
    for _ in range(120):
        new = [f for f in glob.glob(os.path.join(a.dir, "**", "*.json*"), recursive=True) if f not in before and "pt.trace" in f or f.endswith(".json.gz")]
        new = [f for f in new if f not in before]
        if new:
            trace = max(new, key=os.path.getsize)
            time.sleep(5)
            break
        time.sleep(5)
    out = dict(n=a.n, tokens_target=a.tokens, prompt_tokens=[r[1] for r in res], latencies_s=[round(r[0], 3) for r in res],
               wall_s=wall, prefill_tok_s=sum(r[1] for r in res) / wall, trace=trace, categories={}, top_kernels=[])
    if trace:
        out.update(summarize_trace(trace))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[profile] prefill {out['prefill_tok_s']:.0f} tok/s over {a.n}x{a.tokens}; " +
          " ".join(f"{k}={v['share']:.1%}" for k, v in out["categories"].items() if v.get("share") is not None), flush=True)


if __name__ == "__main__":
    main()
