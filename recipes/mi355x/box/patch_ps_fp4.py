#!/usr/bin/env python3
"""patch_ps_fp4.py: route large-M MXFP4 GEMMs to aiter v0.1.30's persistent 256x256 f4gemm kernel (MX_PS_FP4=1).

Why: on one MI355X the image's aiter 0.1.21.post2 runs every prefill-chunk GEMM of Gemma 4 31B through the non-persistent
asm kernels; v0.1.30 added f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ps (grid = min(tiles, CUs), each workgroup loops
over its tiles). Microbenchmarked on the same box (notes/gemm_headroom.md, run mxG1_asm), it matches the engine path's
output and runs 1.08-1.13x faster at M=8192 on all six (N, K), at the same power cap and clock.

What it changes (runtime-gated, so one patched image serves both arms of a paired test):
  * writes aiter/ops/mx_ps_fp4.py: a ctypes launcher (hipModuleLoad + hipModuleLaunchKernel with v0.1.30's 376-byte
    PsKernelArgs: KernelArgs + num_cu at offset 372), and the routing test copied from v0.1.30's C++ ps_supported();
  * inserts a 5-line route at the top of aiter.gemm_a4w4's gfx950 path: with MX_PS_FP4=1 and M >= MX_PS_MIN_M
    (default 2048), the call launches the ps kernel; anything the kernel does not take falls back to the unchanged path.
    Decode steps (M <= 384, inside CUDA graphs) never route.
The .co is fetched by setup_cmd from the v0.1.30 tag and checked by sha256 before this script runs.
Status: /root/or/mx/patch_ps_fp4.status ("applied" or "FAILED: ..."); the ps server's launch refuses to start without it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys

CO = "/root/or/mx/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ps.co"
CO_SHA256 = "8b541b78bff2de71f3cb48db89572f1eb4adc403e09a675be9a8a546a60ff938"
STATUS = "/root/or/mx/patch_ps_fp4.status"

MODULE = r'''"""mx_ps_fp4 (written by /root/or/mx/patch_ps_fp4.py): aiter v0.1.30's persistent 256x256 f4gemm via ctypes."""
import ctypes
import os
import struct
import sys

import torch

ENABLED = os.environ.get("MX_PS_FP4", "0") == "1"
MIN_M = int(os.environ.get("MX_PS_MIN_M", "2048"))
CO = os.environ.get("MX_PS_CO", "__CO__")
KNAME = b"_ZN5aiter45f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_psE"
_st = dict(lib=None, fn=None, num_cu=None, n=0, seen=set(), fb=set(), dead=None)


def _log(msg):
    print(f"[mx_ps_fp4 pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


def _init():
    path = next((l.split()[-1] for l in open("/proc/self/maps") if "libamdhip64.so" in l), "libamdhip64.so")
    lib = ctypes.CDLL(path)
    lib.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    r = lib.hipModuleLoad(ctypes.byref(mod), CO.encode())
    if r:
        raise RuntimeError(f"hipModuleLoad({CO}) = {r}")
    r = lib.hipModuleGetFunction(ctypes.byref(fn), mod, KNAME)
    if r:
        raise RuntimeError(f"hipModuleGetFunction = {r}")
    _st.update(lib=lib, fn=fn, num_cu=torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
    _log(f"enabled: min_m={MIN_M} co={CO} hip={path} num_cu={_st['num_cu']}")


def route(m, n, k, A, B, A_scale, B_scale, bias, beta, bpreshuffle, dtype):
    """v0.1.30 ps_supported(): B preshuffled and contiguous, K % 64 == 0, N % 16 == 0, scales padded to whole 256-row
    tiles, A and B under 4 GiB, no bias term (bias with beta 0 adds nothing); plus bf16 output and M >= MIN_M."""
    if not ENABLED or m < MIN_M or _st["dead"]:
        return False
    a_row, b_row = A.stride(0), B.stride(0)        # bytes: fp4x2 elements
    why = ("b not preshuffled" if not bpreshuffle else
           "bias" if (bias is not None and beta not in (None, 0.0)) else
           "dtype" if dtype != torch.bfloat16 else
           "K%64 or N%16" if (k % 64 or n % 16) else
           "A row stride" if (a_row % 16 or A.stride(-1) != 1) else
           "B layout" if (b_row * 2 != k or not B.is_contiguous()) else
           "A_scale rows" if A_scale.shape[0] < (m + 255) // 256 * 256 else
           "B_scale rows" if B_scale.shape[0] < (n + 255) // 256 * 256 else
           "4 GiB" if (m * a_row >= 1 << 32 or n * b_row >= 1 << 32) else None)
    if why:
        if (n, k, why) not in _st["fb"]:
            _st["fb"].add((n, k, why))
            _log(f"fallback m={m} n={n} k={k}: {why} (A_scale {tuple(A_scale.shape)}, B_scale {tuple(B_scale.shape)})")
        return False
    if _st["fn"] is None:
        try:
            _init()
        except Exception as e:  # noqa: BLE001
            _st["dead"] = repr(e)
            _log(f"DISABLED: {e!r}")
            return False
    return True


def launch(A, B, A_scale, B_scale, out, m, n, k):
    b = bytearray(376)
    struct.pack_into("<Q", b, 0, out.data_ptr())            # ptr_D
    struct.pack_into("<Q", b, 32, A.data_ptr())             # ptr_A
    struct.pack_into("<Q", b, 48, B.data_ptr())             # ptr_B
    struct.pack_into("<f", b, 64, 1.0)                      # alpha
    struct.pack_into("<f", b, 80, 0.0)                      # beta
    struct.pack_into("<I", b, 128, out.stride(0))           # stride_C0
    struct.pack_into("<I", b, 160, A.stride(0) * 2)         # stride_A0 (fp4 elements)
    struct.pack_into("<I", b, 192, B.stride(0) * 2)         # stride_B0
    struct.pack_into("<I", b, 224, m)
    struct.pack_into("<I", b, 240, n)
    struct.pack_into("<I", b, 256, k)
    struct.pack_into("<Q", b, 272, A_scale.data_ptr())
    struct.pack_into("<Q", b, 288, B_scale.data_ptr())
    struct.pack_into("<I", b, 304, A_scale.stride(0))
    struct.pack_into("<I", b, 336, B_scale.stride(0))
    struct.pack_into("<i", b, 368, 0)                       # log2_k_split
    struct.pack_into("<I", b, 372, _st["num_cu"])
    tiles = ((m + 255) // 256) * ((n + 255) // 256)
    buf = ctypes.create_string_buffer(bytes(b), 376)
    size = ctypes.c_size_t(376)
    extra = (ctypes.c_void_p * 5)(1, ctypes.addressof(buf), 2, ctypes.addressof(size), 3)
    r = _st["lib"].hipModuleLaunchKernel(_st["fn"], min(tiles, _st["num_cu"]), 1, 1, 256, 1, 1, 0,
                                         ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), None, extra)
    if r:
        raise RuntimeError(f"hipModuleLaunchKernel(ps) = {r}")
    _st["n"] += 1
    if (n, k) not in _st["seen"]:
        _st["seen"].add((n, k))
        _log(f"routed first m={m} n={n} k={k} (A_scale {tuple(A_scale.shape)})")
    if _st["n"] % 24000 == 0:
        _log(f"routed {_st['n']} GEMMs so far")
'''

HOOK = '''    # MX_PS_FP4 (patch_ps_fp4.py): large-M calls -> aiter v0.1.30's persistent 256x256 kernel; others unchanged.
    if gfx_arch == "gfx950" and _mx_ps.route(m, n, k, A, B, A_scale, B_scale, bias, beta, bpreshuffle, dtype):
        out = torch.empty(((m + 255) // 256 * 256, n), dtype=dtype, device=A.device)
        _mx_ps.launch(A.view(m, k // 2), B, A_scale, B_scale, out, m, n, k)
        return out[:m].view(*A.shape[:-1], n)
'''
ANCHOR = "    out = torch.empty(((m + 31) // 32 * 32, n), dtype=dtype, device=A.device)\n"
IMPORT_ANCHOR = "from ..utility import dtypes\n"


def status(msg, rc):
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    open(STATUS, "w").write(msg + "\n")
    print(f"PATCH ps_fp4: {msg}")
    sys.exit(rc)


def main():
    if not os.path.exists(CO):
        status(f"FAILED: {CO} missing", 1)
    sha = hashlib.sha256(open(CO, "rb").read()).hexdigest()
    if sha != CO_SHA256:
        status(f"FAILED: {CO} sha256 {sha} != {CO_SHA256}", 1)
    ops = os.path.join(os.path.dirname(importlib.util.find_spec("aiter").origin), "ops")
    p = os.path.join(ops, "gemm_op_a4w4.py")
    s = open(p).read()
    open(os.path.join(ops, "mx_ps_fp4.py"), "w").write(MODULE.replace("__CO__", CO))
    if "_mx_ps.route(" in s:
        status("applied (already)", 0)
    body = s[s.index("def gemm_a4w4("):]
    if s.count(ANCHOR) != 1 or body.count(ANCHOR) != 1 or IMPORT_ANCHOR not in s:
        status(f"FAILED: anchor count {s.count(ANCHOR)} in {p}", 1)
    s = s.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + "from . import mx_ps_fp4 as _mx_ps\n", 1)
    s = s.replace(ANCHOR, HOOK + ANCHOR, 1)
    open(p, "w").write(s)
    compile(s, p, "exec")
    status(f"applied (aiter {p}; co sha256 {sha[:12]}; active only with MX_PS_FP4=1)", 0)


if __name__ == "__main__":
    main()
