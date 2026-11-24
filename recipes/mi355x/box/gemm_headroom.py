#!/usr/bin/env python3
"""gemm_headroom.py: how much faster can the MXFP4 GEMMs of Gemma 4 31B run on one MI355X (gfx950)?

Runs inside OpenRelay's vLLM 0.30 image (aiter 0.1.21.post2), with no engine on the GPU. Every row is appended to
<out>/<phase>.jsonl with the GFX clock (mean of the 8 XCDs, the definition tools/gpumon_amd.py uses) and socket power
sampled DURING that measurement, so each rate can be put against the clock-adjusted MXFP4 peak
(10.066 PFLOP/s dense at 2400 MHz = 256 CU x 16384 FLOP/clk, scaled by clock/2400).

Timing: each variant is captured into a HIP graph of k back-to-back calls that rotate over >= 768 MB of weight copies
(3x the 256 MB Infinity Cache, so every call streams its weights from HBM, as in the engine where each layer has its
own weights) and 2 activation copies; the graph is replayed until >= --min-s seconds and timed with events. Decode-size
GEMMs run inside CUDA graphs in the engine too, so launch overhead is excluded on both sides.

Phases (python3 gemm_headroom.py <phase> --out DIR):
  inventory  versions, kernel tables, prebuilt JIT modules, the tuned rows the engine resolves for our shapes
  mfma       a hipcc-built loop of v_mfma_scale_f32_32x32x64_f8f6f4 (FP4 x FP4): the silicon ceiling at the clock it holds
  sweep      every aiter asm f4gemm kernel (+ log2 split-K where the kernel supports it), the C++ heuristic default and
             the engine path (aiter.gemm_a4w4 = tuned CSV lookup, as vLLM calls it) at every (shape, M); also the
             synthetic squares 8192^3 and 16384^3. One subprocess per shape.
  ps         aiter v0.1.30's persistent 256x256 kernel (.co fetched by sha256), launched through a ctypes launcher with
             v0.1.30's 376-byte kernarg layout; the launcher is first validated against aiter's own launch of the
             image's 256x256 kernel (bitwise output).
  confirm    best valid variant vs engine path per (shape, M), interleaved A/B rounds, longer windows
  hipblaslt  torch._scaled_mm MXFP4 (hipBLASLt heuristic) at large M; hipblaslt-bench if the image has it
  triton     aiter Triton gemm_afp4wfp4_preshuffle (same weight layout as the asm path) with a config grid
  ck         CK a4w4 blockscale: every kernelId x splitK (JIT module; build time recorded)
  selftest   local, no GPU: kernarg layout and variant selection
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import glob
import json
import math
import os
import statistics as st
import struct
import subprocess
import sys
import threading
import time
import traceback
from operator import itemgetter

PEAK_2400 = 256 * 16384 * 2.4e9  # dense MXFP4 FLOP/s at 2400 MHz
SHAPES = [  # name, N, K, layers per forward (50 sliding + 10 global)
    ("qkv_s", 16384, 5376, 50), ("qkv_g", 20480, 5376, 10), ("o_s", 5376, 8192, 50), ("o_g", 5376, 16384, 10),
    ("gate_up", 43008, 5376, 60), ("down", 5376, 21504, 60)]
SQUARES = [("sq8k", 8192, 8192, 0, 8192), ("sq16k", 16384, 16384, 0, 16384)]
DECODE_M = [16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384]           # CUDA-graph sizes our recipe captures
MIXED_M = [1024, 2048, 3072, 4096, 5120, 6144, 6912, 7680, 8192, 8576, 16384]
PS_CO_URL = ("https://raw.githubusercontent.com/ROCm/aiter/77ba72b7061a3c76c4f726a50e8edfd2fe2a80e7/hsa/gfx950/f4gemm/"
             "f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ps.co")   # tag v0.1.30
PS_KNAME = "_ZN5aiter45f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_psE"
BY_US = itemgetter("us")
OUT = "."


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def emit(phase, **row):
    row = dict(phase=phase, t=round(time.time(), 3), **row)
    with open(os.path.join(OUT, f"{phase}.jsonl"), "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


# ----------------------------------------------------------------------------- clock / power sampler
class ClockMon:
    """GFX clock (mean over XCDs) + socket power, sampled in a thread. Source: amdsmi Python API, else the amd-smi CLI
    with run.py's gpumon command and tools/gpumon_amd.py's field positions."""

    def __init__(self, hz=20):
        self.samples = []      # (perf_counter, clk_mhz, power_w, activity)
        self.src, self.err, self.keys = None, None, None
        self.dt = 1.0 / hz
        try:
            for p in ("/opt/rocm/share/amd_smi", "/opt/rocm/libexec/amdsmi_cli"):
                if p not in sys.path and os.path.isdir(p):
                    sys.path.append(p)
            import amdsmi   # noqa
            amdsmi.amdsmi_init()
            self.a = amdsmi
            self.h = amdsmi.amdsmi_get_processor_handles()[0]
            m = amdsmi.amdsmi_get_gpu_metrics_info(self.h)
            self.keys = sorted(m.keys())
            if self._amdsmi()[0]:
                self.src = "amdsmi.gpu_metrics"
        except Exception as e:  # noqa: BLE001
            self.err = repr(e)[:300]
        if not self.src:
            self.src = "amd-smi-cli"
            self.dt = 0.25
        self.halt = False
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th.start()

    @staticmethod
    def _num(v):
        try:
            v = float(v)
            return v if 0 < v < 60000 else None
        except (TypeError, ValueError):
            return None

    def _amdsmi(self):
        m = self.a.amdsmi_get_gpu_metrics_info(self.h)
        clks = m.get("current_gfxclks")
        vals = [x for x in (self._num(c) for c in clks) if x] if isinstance(clks, (list, tuple)) else []
        if not vals and self._num(m.get("current_gfxclk")):
            vals = [self._num(m.get("current_gfxclk"))]
        p = self._num(m.get("current_socket_power")) or self._num(m.get("average_socket_power"))
        a = self._num(m.get("average_gfx_activity"))
        return (st.mean(vals) if vals else None, p, a)

    def _cli(self):
        r = subprocess.run("amd-smi metric --usage --power --mem-usage --clock --temperature --csv | tail -n +2",
                           shell=True, capture_output=True, text=True, timeout=15)
        for row in csv.reader(r.stdout.splitlines()):
            if len(row) >= 55 and row[0] == "0":
                try:
                    return st.mean(float(row[i]) for i in range(15, 51, 5)), float(row[9]), float(row[1])
                except ValueError:
                    return None, None, None
        return None, None, None

    def run(self):
        while not self.halt:
            t = time.perf_counter()
            try:
                c, p, a = self._amdsmi() if self.src == "amdsmi.gpu_metrics" else self._cli()
                if c:
                    self.samples.append(((t + time.perf_counter()) / 2, c, p, a))
            except Exception as e:  # noqa: BLE001
                self.err = repr(e)[:300]
            time.sleep(max(0.0, self.dt - (time.perf_counter() - t)))

    def window(self, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:   # a short window: the nearest samples within 0.5 s

            def dist(x):
                return min(abs(x[0] - t0), abs(x[0] - t1))
            s = [x for x in sorted(self.samples, key=dist)[:2] if dist(x) < 0.5]
        if not s:
            return dict(clk_mhz=None, clk_min=None, power_w=None, n_clk=0)
        c = [x[1] for x in s]
        p = [x[2] for x in s if x[2]]
        return dict(clk_mhz=round(st.median(c), 1), clk_min=round(min(c), 1), power_w=round(st.median(p), 1) if p else None,
                    n_clk=len(s))


MON = None


def rate_fields(flops, us, win):
    pf = flops / (us * 1e-6) / 1e15
    clk = win.get("clk_mhz")
    peak = PEAK_2400 / 1e15 * (clk / 2400.0) if clk else None
    return dict(us=round(us, 3), pflops=round(pf, 4), peak_at_clk_pf=round(peak, 3) if peak else None,
                pct_peak_at_clk=round(100 * pf / peak, 2) if peak else None, pct_peak_2400=round(100 * pf / (PEAK_2400 / 1e15), 2), **win)


# ----------------------------------------------------------------------------- HIP ctypes launcher
class Hip:
    def __init__(self):
        path = None
        for line in open("/proc/self/maps"):
            if "libamdhip64.so" in line:
                path = line.split()[-1]
                break
        self.path = path or "libamdhip64.so"
        self.lib = ctypes.CDLL(self.path)
        L = self.lib
        L.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        L.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        L.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.mods = {}

    def fn(self, co, name):
        key = (co, name)
        if key not in self.mods:
            mod, f = ctypes.c_void_p(), ctypes.c_void_p()
            r = self.lib.hipModuleLoad(ctypes.byref(mod), co.encode())
            if r:
                raise RuntimeError(f"hipModuleLoad({co}) = {r}")
            r = self.lib.hipModuleGetFunction(ctypes.byref(f), mod, name.encode())
            if r:
                raise RuntimeError(f"hipModuleGetFunction({name}) = {r}")
            self.mods[key] = f
        return self.mods[key]

    def launch(self, f, grid, block, args: bytes, stream: int):
        buf = ctypes.create_string_buffer(args, len(args))
        size = ctypes.c_size_t(len(args))
        extra = (ctypes.c_void_p * 5)(1, ctypes.cast(buf, ctypes.c_void_p).value,
                                      2, ctypes.cast(ctypes.byref(size), ctypes.c_void_p).value, 3)
        r = self.lib.hipModuleLaunchKernel(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0,
                                           ctypes.c_void_p(stream), None, extra)
        if r:
            raise RuntimeError(f"hipModuleLaunchKernel = {r}")


def f4gemm_kernargs(ptr_D, ptr_A, ptr_B, ptr_sA, ptr_sB, stride_C0, stride_A0, stride_B0, M, N, K, stride_sA0, stride_sB0,
                    log2_k_split, num_cu=None):
    """aiter's f4gemm KernelArgs (each field padded to 16 bytes; log2_k_split at 368, size 372) and, for ps=1 kernels,
    v0.1.30's PsKernelArgs (+ num_cu u32 at 372, size 376). Zero-initialised like v0.1.30's `KernelArgs args{}`."""
    b = bytearray(372 if num_cu is None else 376)
    struct.pack_into("<Q", b, 0, ptr_D)        # ptr_D
    struct.pack_into("<Q", b, 16, 0)           # ptr_C (bias): none
    struct.pack_into("<Q", b, 32, ptr_A)
    struct.pack_into("<Q", b, 48, ptr_B)
    struct.pack_into("<f", b, 64, 1.0)         # alpha
    struct.pack_into("<f", b, 80, 0.0)         # beta
    struct.pack_into("<I", b, 128, stride_C0)  # stride_C0 (the C++ sets only C0)
    struct.pack_into("<I", b, 160, stride_A0)
    struct.pack_into("<I", b, 192, stride_B0)
    struct.pack_into("<I", b, 224, M)
    struct.pack_into("<I", b, 240, N)
    struct.pack_into("<I", b, 256, K)
    struct.pack_into("<Q", b, 272, ptr_sA)
    struct.pack_into("<Q", b, 288, ptr_sB)
    struct.pack_into("<I", b, 304, stride_sA0)
    struct.pack_into("<I", b, 336, stride_sB0)
    struct.pack_into("<i", b, 368, log2_k_split)
    if num_cu is not None:
        struct.pack_into("<I", b, 372, num_cu)
    return bytes(b)


def k_slices(K, k_num):
    k_per_tg = K // k_num
    k_per_tg = (k_per_tg + 255) // 256 * 256
    return (K + k_per_tg - 1) // k_per_tg


# ----------------------------------------------------------------------------- data + timing (GPU)
def gpu_setup():
    global torch, aiter, shuffle_weight, QF
    import torch  # noqa
    import aiter  # noqa
    from aiter.ops.shuffle import shuffle_weight  # noqa
    QF = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    torch.manual_seed(0)


def asm_table():
    from aiter.jit.core import get_asm_dir
    path = os.path.join(get_asm_dir(), "f4gemm", "f4gemm_bf16_per1x32Fp4.csv")
    rows = []
    for r in csv.DictReader(l for l in open(path) if not l.startswith("#")):
        rows.append(dict(tile_M=int(r["tile_M"]), tile_N=int(r["tile_N"]), splitK=int(r["splitK"]), bpreshuffle=int(r["bpreshuffle"]),
                         ps=int(r.get("ps", 0) or 0), knl_name=r["knl_name"], co_name=r["co_name"]))
    return path, rows


class Weights:
    def __init__(self, N, K, rot_bytes=768 << 20):
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        self.q_u, self.s_u = QF(w, shuffle=False)          # unshuffled: torch reference only
        q, s = QF(w, shuffle=True)
        del w
        b = shuffle_weight(q)
        nbytes = b.numel() * b.element_size() + s.numel() * s.element_size()
        self.R = max(2, min(64, math.ceil(rot_bytes / nbytes)))
        self.B = [b] + [b.clone() for _ in range(self.R - 1)]
        self.S = [s] + [s.clone() for _ in range(self.R - 1)]
        self.nbytes = nbytes
        self.N, self.K = N, K


class Acts:
    def __init__(self, M, K, R=2):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        self.q_u, self.s_u = QF(x, shuffle=False)
        q, s = QF(x, shuffle=True)
        del x
        self.A = [q] + [q.clone() for _ in range(R - 1)]
        self.S = [s] + [s.clone() for _ in range(R - 1)]
        self.R, self.M = R, M


def torch_ref(acts, w, M, N):
    from aiter.utility import fp4_utils
    xf = fp4_utils.mxfp4_to_f32(acts.q_u) * fp4_utils.e8m0_to_f32(acts.s_u[:M].repeat_interleave(32, dim=1))
    wf = fp4_utils.mxfp4_to_f32(w.q_u) * fp4_utils.e8m0_to_f32(w.s_u[:N].repeat_interleave(32, dim=1))
    return torch.mm(xf, wf.T)[:M, :N]


def rel_err(y, ref):
    y = y.float()
    ref = ref.float()
    d = (y - ref).norm().item()
    n = ref.norm().item()
    return d / n if n else float("inf")


def time_graph(call, k, min_s=0.12, use_graph=True):
    """call(i) issues one GEMM on the current stream. Returns (us_per_call, t0, t1, mode)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    mode = "graph"
    g = None
    if use_graph:
        try:
            with torch.cuda.stream(s):
                for i in range(min(k, 3)):
                    call(i)        # warm (allocator, lazy module loads) outside capture
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(k):
                    call(i)
            g.replay()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            mode = f"eager ({repr(e)[:80]})"
            g = None
            torch.cuda.synchronize()

    def one():
        if g is not None:
            g.replay()
        else:
            for i in range(k):
                call(i)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    one()
    e1.record()
    e1.synchronize()
    est = max(e0.elapsed_time(e1) / 1e3, 1e-6)
    reps = max(2, min(20000, int(math.ceil(min_s / est))))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    e0.record()
    for _ in range(reps):
        one()
    e1.record()
    e1.synchronize()
    t1 = time.perf_counter()
    us = e0.elapsed_time(e1) * 1e3 / (reps * k)
    return us, t0, t1, mode


def engine_config(M, N, K):
    try:
        from aiter.ops.gemm_op_a4w4 import get_GEMM_config
        c = get_GEMM_config(M, N, K)
        return None if c is None else dict(kernelName=c.get("kernelName"), splitK=c.get("splitK"), kernelId=c.get("kernelId"),
                                           tuned_us=c.get("us"))
    except Exception as e:  # noqa: BLE001
        return dict(error=repr(e)[:200])


# ----------------------------------------------------------------------------- variants
def variants_for(M, N, table):
    """aiter asm kernels worth timing at this M (the tuner's space, minus clearly hopeless tiles)."""
    out = [("asm_default", "", 0)]
    for r in table:
        if not r["bpreshuffle"] or r["ps"]:
            continue
        tm, tn = r["tile_M"], r["tile_N"]
        if tm == 128 and tn == 512 and N % 512:
            continue                                   # the C++ heuristic skips it too (N tail unsupported)
        if M >= 1024 and tm == 32:
            continue                                   # 32-row tiles at prefill M: >5x the tiles of 160/256-row ones
        if M <= 64 and tm > 128 and not r["splitK"]:
            continue
        splits = [0]
        if r["splitK"]:
            splits = [0, 1, 2, 3] if M <= 512 else ([0, 1] if M <= 2048 else [0])
        for s in splits:
            out.append((f"asm_{tm}x{tn}" + (f"_ks{s}" if s else ""), r["knl_name"], s))
    return out


def run_point(phase, shape, N, K, M, w, table, min_s, extra_variants=(), ref_torch=False, only=None, reverse=False, tag=None):
    acts = Acts(M, K, R=2)
    k = w.R
    flops = 2.0 * M * N * K
    out = torch.empty(((M + 255) // 256 * 256, N), dtype=torch.bfloat16, device="cuda")
    eng_cfg = engine_config(M, N, K)

    def engine_call(i):
        return aiter.gemm_a4w4(acts.A[i % acts.R], w.B[i % w.R].view(acts.A[0].dtype), acts.S[i % acts.R],
                               w.S[i % w.R].view(acts.S[0].dtype), dtype=torch.bfloat16, bpreshuffle=True)
    ref = engine_call(0)[:M].clone()
    torch.cuda.synchronize()
    tref = None
    if ref_torch:
        try:
            tref = torch_ref(acts, w, M, N)
        except Exception as e:  # noqa: BLE001
            log(f"torch ref failed: {e!r}"[:200])
    res = []
    vs = [("engine", None, None)] + list(variants_for(M, N, table)) + list(extra_variants)
    if only:
        vs = [v for v in vs if v[0] in only]
    if reverse:
        vs = vs[::-1]
    for name, kname, ks in vs:
        try:
            if name == "engine":
                call = engine_call
                y = ref
            elif callable(kname):
                call = kname(acts, w, out, M, N, K)
                call(0)
                torch.cuda.synchronize()
                y = out[:M]
            else:
                def call(i, kname=kname, ks=ks):
                    aiter.gemm_a4w4_asm(acts.A[i % acts.R], w.B[i % w.R], acts.S[i % acts.R], w.S[i % w.R], out,
                                        kname, None, 1.0, 0.0, True, log2_k_split=ks)
                call(0)
                torch.cuda.synchronize()
                y = out[:M]
            err = rel_err(y, ref)
            err_t = rel_err(y, tref) if tref is not None else None
            us, t0, t1, mode = time_graph(call, k, min_s=min_s)
            win = MON.window(t0, t1) if MON else {}
            row = emit(phase, shape=shape, M=M, N=N, K=K, variant=name, kernel=kname if isinstance(kname, str) else name, log2_k_split=ks,
                       err_vs_engine=round(err, 5), err_vs_fp32=round(err_t, 5) if err_t is not None else None, mode=mode,
                       rot_sets=w.R, engine_cfg=eng_cfg, round=tag, **rate_fields(flops, us, win))
            res.append(row)
        except Exception as e:  # noqa: BLE001
            emit(phase, shape=shape, M=M, N=N, K=K, variant=name, kernel=kname if isinstance(kname, str) else name, log2_k_split=ks,
                 error=repr(e)[:300])
            torch.cuda.synchronize()
    return res


# ----------------------------------------------------------------------------- phases
def sh(cmd, timeout=60):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:  # noqa: BLE001
        return repr(e)[:200]


def ph_inventory(a):
    info = dict(date=now(), host=os.uname().nodename, nproc=os.cpu_count())
    for mod in ("torch", "triton", "vllm", "aiter"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "?")
            info[f"{mod}_file"] = getattr(m, "__file__", "?")
        except Exception as e:  # noqa: BLE001
            info[f"{mod}_error"] = repr(e)[:200]
    try:
        from importlib.metadata import version
        info["amd_aiter_dist"] = version("amd-aiter")
    except Exception as e:  # noqa: BLE001
        info["amd_aiter_dist"] = repr(e)[:100]
    gpu_setup()
    p = torch.cuda.get_device_properties(0)
    info.update(gpu=torch.cuda.get_device_name(0), cu=p.multi_processor_count, gcn=getattr(p, "gcnArchName", "?"),
                mem_gib=round(p.total_memory / 2**30, 1))
    path, rows = asm_table()
    info["asm_table"] = path
    info["asm_kernels"] = [f"{r['tile_M']}x{r['tile_N']} sk{r['splitK']} pre{r['bpreshuffle']} ps{r['ps']}" for r in rows]
    from aiter.jit.core import get_asm_dir
    info["asm_dir"] = get_asm_dir()
    info["asm_co_sha256"] = sh(f"cd {get_asm_dir()}/f4gemm && sha256sum *.co | head -40")
    jd = os.path.join(os.path.dirname(aiter.__file__), "jit")
    info["jit_modules"] = sorted(os.path.basename(x) for x in glob.glob(os.path.join(jd, "*.so")))
    info["jit_build_dirs"] = sorted(os.path.basename(x) for x in glob.glob(os.path.join(jd, "build", "*")))[:80]
    for tool in ("hipcc", "hipblaslt-bench", "amd-smi", "rocprofv3", "hipconfig"):
        info[f"which_{tool}"] = sh(f"command -v {tool} || ls /opt/rocm/bin/{tool} 2>/dev/null").strip()
    info["hipcc_version"] = sh("hipcc --version 2>&1 | head -4")
    info["hipblaslt_bench_scale_help"] = sh("hipblaslt-bench --help 2>&1 | grep -i -E 'scale|f4|mx' | head -30")[:3000]
    info["amd_smi_static_limit"] = sh("amd-smi static --limit 2>&1 | head -30")[:2000]
    info["clockmon"] = dict(src=MON.src, err=MON.err, keys=MON.keys)
    cfgs = {}
    for name, N, K, _ in SHAPES:
        for M in DECODE_M + MIXED_M:
            cfgs[f"{name}:M{M}"] = engine_config(M, N, K)
    info["engine_resolved"] = cfgs
    try:
        from aiter.jit.core import AITER_CONFIGS
        info["a4w4_tuned_file"] = AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE
    except Exception as e:  # noqa: BLE001
        info["a4w4_tuned_file"] = repr(e)[:200]
    info["env"] = {k: v for k, v in os.environ.items() if k.startswith(("AITER", "VLLM", "HIP", "ROCM", "HSA", "TRITON"))}
    emit("inventory", **info)
    log(json.dumps({k: v for k, v in info.items() if k not in ("engine_resolved", "jit_build_dirs", "asm_co_sha256")}, default=str)[:4000])


MFMA_SRC = r"""
#include <hip/hip_runtime.h>
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v16f __attribute__((ext_vector_type(16)));
typedef float v4f __attribute__((ext_vector_type(4)));
extern "C" __global__ __launch_bounds__(256) void mfma32(float* out, int iters, int zero) {
  v8i a, b;
  unsigned t = threadIdx.x + 977u * blockIdx.x;
  for (int i = 0; i < 8; ++i) {
    a[i] = zero ? 0 : (int)((t * 0x9E3779B1u) ^ ((unsigned)i * 0x85EBCA6Bu));
    b[i] = zero ? 0 : (int)((t * 0xC2B2AE35u) + (unsigned)i * 0x27D4EB2Fu);
  }
  v16f c0 = {}, c1 = {}, c2 = {}, c3 = {};
  int sa = 127 + (int)(t & 1), sb = 127;
  for (int it = 0; it < iters; ++it) {
    c0 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c0, 4, 4, 0, sa, 0, sb);
    c1 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(b, a, c1, 4, 4, 0, sa, 0, sb);
    c2 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, a, c2, 4, 4, 0, sb, 0, sa);
    c3 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(b, b, c3, 4, 4, 0, sb, 0, sa);
  }
  float s = 0.f;
  for (int i = 0; i < 16; ++i) s += c0[i] + c1[i] + c2[i] + c3[i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
extern "C" __global__ __launch_bounds__(256) void mfma16(float* out, int iters, int zero) {
  v8i a, b;
  unsigned t = threadIdx.x + 977u * blockIdx.x;
  for (int i = 0; i < 8; ++i) {
    a[i] = zero ? 0 : (int)((t * 0x9E3779B1u) ^ ((unsigned)i * 0x85EBCA6Bu));
    b[i] = zero ? 0 : (int)((t * 0xC2B2AE35u) + (unsigned)i * 0x27D4EB2Fu);
  }
  v4f c0 = {}, c1 = {}, c2 = {}, c3 = {};
  int sa = 127 + (int)(t & 1), sb = 127;
  for (int it = 0; it < iters; ++it) {
    c0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c0, 4, 4, 0, sa, 0, sb);
    c1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b, a, c1, 4, 4, 0, sa, 0, sb);
    c2 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, a, c2, 4, 4, 0, sb, 0, sa);
    c3 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b, b, c3, 4, 4, 0, sb, 0, sa);
  }
  float s = 0.f;
  for (int i = 0; i < 4; ++i) s += c0[i] + c1[i] + c2[i] + c3[i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
"""


def ph_mfma(a):
    gpu_setup()
    d = os.path.join(OUT, "mfma")
    os.makedirs(d, exist_ok=True)
    src, co = os.path.join(d, "mfma.hip"), os.path.join(d, "mfma.co")
    open(src, "w").write(MFMA_SRC)
    r = subprocess.run(["hipcc", "--offload-arch=gfx950", "-O3", "--genco", src, "-o", co], capture_output=True, text=True)
    if r.returncode:
        emit("mfma", error="hipcc failed", stderr=r.stderr[-2000:])
        log("hipcc failed: " + r.stderr[-800:])
        return
    nmfma = sh(f"/opt/rocm/llvm/bin/llvm-objdump -d {co} 2>/dev/null | grep -c v_mfma_scale").strip()
    hip = Hip()
    blocks = 256 * 8
    outb = torch.empty(blocks * 256, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for kname, flop_per_mfma in (("mfma32", 2 * 32 * 32 * 64), ("mfma16", 2 * 16 * 16 * 128)):
        f = hip.fn(co, kname)
        for zero in (0, 1):
            iters = 2000
            args = struct.pack("<Qii", outb.data_ptr(), iters, zero)
            for _ in range(2):
                hip.launch(f, (blocks, 1, 1), (256, 1, 1), args, stream)
            torch.cuda.synchronize()
            e0.record()
            hip.launch(f, (blocks, 1, 1), (256, 1, 1), args, stream)
            e1.record()
            e1.synchronize()
            per = e0.elapsed_time(e1) / 1e3
            iters = max(100, int(iters * 0.05 / max(per, 1e-6)))          # ~50 ms per launch
            args = struct.pack("<Qii", outb.data_ptr(), iters, zero)
            for hold in (0.5, 3.0):
                n = 0
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                e0.record()
                while time.perf_counter() - t0 < hold:
                    hip.launch(f, (blocks, 1, 1), (256, 1, 1), args, stream)
                    n += 1
                    if n % 4 == 0:
                        torch.cuda.synchronize()
                e1.record()
                e1.synchronize()
                t1 = time.perf_counter()
                sec = e0.elapsed_time(e1) / 1e3
                flops = n * blocks * 4 * iters * 4 * flop_per_mfma
                row = emit("mfma", kernel=kname, operands="zero" if zero else "random", hold_s=hold, launches=n, iters=iters,
                           mfma_in_co=nmfma, **rate_fields(flops, sec * 1e6, MON.window(t0, t1)))
                log(f"mfma {kname} {row['operands']} hold {hold}s: {row['pflops']} PF at {row['clk_mhz']} MHz "
                    f"({row['pct_peak_at_clk']}% of clock-adjusted peak, {row['power_w']} W)")


def ph_sweep(a):
    if not a.shape:    # driver: one subprocess per shape so a crash costs one shape
        names = a.shapes.split(",") if a.shapes else [s[0] for s in SHAPES] + [s[0] for s in SQUARES]
        for n in names:
            t = time.time()
            try:
                r = subprocess.run([sys.executable, __file__, "sweep", "--out", OUT, "--shape", n, "--min-s", str(a.min_s)]
                                   + (["--ms", a.ms] if a.ms else []), timeout=a.shape_timeout)
                log(f"sweep {n}: rc {r.returncode} in {time.time() - t:.0f}s")
            except subprocess.TimeoutExpired:
                log(f"sweep {n}: TIMEOUT after {time.time() - t:.0f}s")
        return
    gpu_setup()
    _, table = asm_table()
    allshapes = {s[0]: (s[1], s[2], None) for s in SHAPES}
    allshapes.update({s[0]: (s[1], s[2], [s[4]]) for s in SQUARES})
    N, K, ms = allshapes[a.shape]
    ms = ms or ([int(x) for x in a.ms.split(",")] if a.ms else DECODE_M + MIXED_M)
    w = Weights(N, K)
    log(f"sweep {a.shape} N={N} K={K}: {w.R} weight copies of {w.nbytes / 2**20:.1f} MiB")
    for M in ms:
        t = time.time()
        res = run_point("sweep", a.shape, N, K, M, w, table, a.min_s, ref_torch=(M in (16, 64, 256)))
        ok = [r for r in res if r.get("err_vs_engine", 9) < 0.03]
        eng = next((r for r in res if r["variant"] == "engine"), None)
        best = min(ok, key=BY_US) if ok else None
        if eng and best:
            log(f"  {a.shape} M={M}: engine {eng['us']:.1f} us {eng['pflops']:.3f} PF | best {best['variant']} {best['us']:.1f} us "
                f"{best['pflops']:.3f} PF ({eng['us'] / best['us']:.3f}x) clk {best.get('clk_mhz')} | {len(res)} variants {time.time() - t:.0f}s")


def fetch_ps_co():
    d = os.path.join(OUT, "ps")
    os.makedirs(d, exist_ok=True)
    co = os.path.join(d, "f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ps.co")
    if not os.path.exists(co):
        subprocess.run(["curl", "-sfL", "--retry", "3", "-o", co, PS_CO_URL], check=True, timeout=120)
    sha = subprocess.run(["sha256sum", co], capture_output=True, text=True).stdout.split()[0]
    return co, sha


def gemm_maker(hip, co, kname, num_cu, ps, ks, tm=256, tn=256):
    """A run_point variant that launches an f4gemm .co through the ctypes launcher, following v0.1.30's dispatch."""
    def make(acts, w, out, M, N, K):
        f = hip.fn(co, kname)
        k_num, l2 = 1 << ks, ks
        tiles = ((M + tm - 1) // tm) * ((N + tn - 1) // tn)
        if ps:
            while k_num > 1 and tiles * k_slices(K, k_num) > num_cu:
                k_num //= 2
                l2 -= 1
        gdz = k_slices(K, k_num) if ks else 1
        grid = (min(tiles * gdz, num_cu), 1, 1) if ps else ((N + tn - 1) // tn, (M + tm - 1) // tm, gdz)

        def call(i):
            A, B, SA, SB = acts.A[i % acts.R], w.B[i % w.R], acts.S[i % acts.R], w.S[i % w.R]
            if k_num > 1:
                out.zero_()
            args = f4gemm_kernargs(out.data_ptr(), A.data_ptr(), B.data_ptr(), SA.data_ptr(), SB.data_ptr(), out.stride(0),
                                   A.stride(0) * 2, B.stride(0) * 2, M, N, K, SA.stride(0), SB.stride(0), l2,
                                   num_cu if ps else None)
            hip.launch(f, grid, (256, 1, 1), args, torch.cuda.current_stream().cuda_stream)
        return call
    return make


def ph_ps(a):
    gpu_setup()
    from aiter.jit.core import get_asm_dir
    co_ps, sha = fetch_ps_co()
    co_std = os.path.join(get_asm_dir(), "f4gemm", "f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co")
    std_name = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"
    hip = Hip()
    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    emit("ps", note="fetched", co=co_ps, sha256=sha, url=PS_CO_URL, hip_lib=hip.path, num_cu=num_cu)
    _, table = asm_table()
    extra = [("ctypes_256x256", gemm_maker(hip, co_std, std_name, num_cu, False, 0), 0),
             ("ps_256x256", gemm_maker(hip, co_ps, PS_KNAME, num_cu, True, 0), 0)]
    small = [(f"ps_256x256_ks{k}", gemm_maker(hip, co_ps, PS_KNAME, num_cu, True, k), k) for k in (1, 2, 3)]
    ms = [int(x) for x in a.ms.split(",")] if a.ms else [16, 64, 128, 256, 384, 2048, 4096, 6912, 8192, 8576, 16384]
    for name, N, K, _ in SHAPES + [s[:4] for s in SQUARES]:
        w = Weights(N, K)
        for M in (ms if not name.startswith("sq") else [N]):
            vs = extra + (small if M <= 512 else [])
            only = {"engine", "asm_256x256"} | {v[0] for v in vs}
            res = run_point("ps", name, N, K, M, w, table, a.min_s, extra_variants=vs, only=only)
            d = {r["variant"]: r for r in res if "us" in r}
            if "ps_256x256" in d and "asm_256x256" in d:
                log(f"ps {name} M={M}: asm256 {d['asm_256x256']['us']:.1f} us, ctypes256 {d.get('ctypes_256x256', {}).get('us')} "
                    f"(err {d.get('ctypes_256x256', {}).get('err_vs_engine')}), ps {d['ps_256x256']['us']:.1f} us "
                    f"(err {d['ps_256x256']['err_vs_engine']}), engine {d.get('engine', {}).get('us')}")
        del w
        torch.cuda.empty_cache()


def ph_confirm(a):
    gpu_setup()
    _, table = asm_table()
    rows = [json.loads(l) for l in open(os.path.join(OUT, "sweep.jsonl"))]
    psf = os.path.join(OUT, "ps.jsonl")
    ps_rows = [json.loads(l) for l in open(psf)] if os.path.exists(psf) else []
    by = {}
    for r in rows + ps_rows:
        if "us" in r and r.get("err_vs_engine", 9) < 0.03 and not r["variant"].startswith("ctypes"):
            by.setdefault((r["shape"], r["M"]), []).append(r)
    allshapes = {s[0]: (s[1], s[2]) for s in SHAPES}
    allshapes.update({s[0]: (s[1], s[2]) for s in SQUARES})
    hip = Hip()
    co_ps = None
    try:
        co_ps, _ = fetch_ps_co()
    except Exception:  # noqa: BLE001
        pass
    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    min_s = a.min_s if MON.src.startswith("amdsmi") else max(a.min_s, 1.5)   # the CLI samples ~1 Hz: longer windows
    emit("confirm", note="start", clock_source=MON.src, min_s=min_s)
    for shape, (N, K) in allshapes.items():
        pts = sorted(M for (s, M) in by if s == shape)
        if not pts:
            continue
        w = Weights(N, K)
        for M in pts:
            cand = sorted(by[(shape, M)], key=BY_US)
            pick = ["engine"] + [r["variant"] for r in cand if r["variant"] != "engine"][:2]
            extra = []
            for v in pick:
                if v.startswith("ps_") and co_ps:
                    ks = int(v[-1]) if "_ks" in v else 0
                    extra.append((v, gemm_maker(hip, co_ps, PS_KNAME, num_cu, True, ks), ks))
            for _ in range(a.rounds):
                run_point("confirm", shape, N, K, M, w, table, min_s, extra_variants=extra, only=set(pick))
        del w
        torch.cuda.empty_cache()


def ph_ab(a):
    """Interleaved A/B: the engine path, the v0.1.30 persistent kernel and named asm variants at each (shape, M), for
    --rounds rounds with the variant order reversed every other round (clock drift within a point cancels)."""
    names = [s[0] for s in SHAPES] + [s[0] for s in SQUARES]
    if not a.shape:
        for n in (a.shapes.split(",") if a.shapes else names):
            t = time.time()
            try:
                r = subprocess.run([sys.executable, __file__, "ab", "--out", OUT, "--shape", n, "--min-s", str(a.min_s),
                                    "--rounds", str(a.rounds), "--variants", a.variants] + (["--ms", a.ms] if a.ms else []),
                                   timeout=a.shape_timeout)
                log(f"ab {n}: rc {r.returncode} in {time.time() - t:.0f}s")
            except subprocess.TimeoutExpired:
                log(f"ab {n}: TIMEOUT after {time.time() - t:.0f}s")
        return
    gpu_setup()
    _, table = asm_table()
    co_ps, sha = fetch_ps_co()
    hip = Hip()
    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    allshapes = {s[0]: (s[1], s[2], None) for s in SHAPES}
    allshapes.update({s[0]: (s[1], s[2], [s[4]]) for s in SQUARES})
    N, K, ms = allshapes[a.shape]
    ms = ms or [int(x) for x in a.ms.split(",")]
    w = Weights(N, K)
    extra = [("ps_256x256", gemm_maker(hip, co_ps, PS_KNAME, num_cu, True, 0), 0)]
    only = {"engine", "ps_256x256"} | set(a.variants.split(","))
    emit("ab", note="start", shape=a.shape, ps_sha256=sha, only=sorted(only), rounds=a.rounds, min_s=a.min_s)
    for M in ms:
        for rnd in range(a.rounds):
            res = run_point("ab", a.shape, N, K, M, w, table, a.min_s, extra_variants=extra, only=only, reverse=bool(rnd % 2), tag=rnd)
            d = {r["variant"]: r for r in res if "us" in r}
            if "engine" in d and "ps_256x256" in d:
                log(f"ab {a.shape} M={M} r{rnd}: engine {d['engine']['us']:.1f} us @{d['engine']['clk_mhz']} | ps {d['ps_256x256']['us']:.1f} us "
                    f"@{d['ps_256x256']['clk_mhz']} ({d['engine']['us'] / d['ps_256x256']['us']:.3f}x, err {d['ps_256x256']['err_vs_engine']})")


def ph_hipblaslt(a):
    gpu_setup()
    ok = hasattr(torch, "float4_e2m1fn_x2")
    for name, N, K, _ in SHAPES + [s[:4] for s in SQUARES]:
        for M in ([8192, 16384] if not name.startswith("sq") else [N]):
            try:
                A = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
                Bs = [torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2) for _ in range(2)]
                sa = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
                sb = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)

                def call(i, A=A, Bs=Bs, sa=sa, sb=sb):
                    return torch._scaled_mm(A, Bs[i % 2].t(), sa, sb, out_dtype=torch.bfloat16)
                us, t0, t1, mode = time_graph(call, 2, min_s=a.min_s, use_graph=False)
                emit("hipblaslt", shape=name, M=M, N=N, K=K, variant="torch._scaled_mm", mode=mode,
                     **rate_fields(2.0 * M * N * K, us, MON.window(t0, t1)))
                log(f"hipblaslt {name} M={M}: {us:.1f} us {2.0 * M * N * K / us / 1e9:.2f} PF/s")
            except Exception as e:  # noqa: BLE001
                emit("hipblaslt", shape=name, M=M, N=N, K=K, variant="torch._scaled_mm", error=repr(e)[:300], has_fp4_dtype=ok)
            torch.cuda.empty_cache()
    hb = sh("command -v hipblaslt-bench || ls /opt/rocm/bin/hipblaslt-bench 2>/dev/null").strip().splitlines()
    if hb:
        for M, N, K in ((8192, 43008, 5376), (8192, 5376, 21504), (16384, 16384, 16384)):
            for extra in ("--scaleA 3 --scaleB 3", "--scaleA 2 --scaleB 2"):
                cmd = (f"{hb[0]} -m {N} -n {M} -k {K} --transA T --transB N --a_type f4_r --b_type f4_r --c_type bf16_r "
                       f"--d_type bf16_r --compute_type f32_r {extra} --algo_method all -i 20 -j 5 --rotating 512 2>&1 | tail -n 25")
                t = time.time()
                try:
                    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
                    emit("hipblaslt", variant="hipblaslt-bench", cmd=cmd, rc=r.returncode, secs=round(time.time() - t, 1), tail=r.stdout[-3000:])
                    if r.returncode == 0 and "Winner" in r.stdout:
                        break
                except Exception as e:  # noqa: BLE001
                    emit("hipblaslt", variant="hipblaslt-bench", cmd=cmd, error=repr(e)[:300])


TRITON_CFGS = [  # BM, BN, BK, GROUP_M, warps, stages, wpe, nonkdim, cache
    (128, 128, 256, 4, 4, 2, 2, 16, None), (256, 128, 256, 4, 4, 2, 1, 16, ".cg"), (256, 128, 256, 8, 8, 2, 1, 16, None),
    (128, 256, 256, 4, 8, 2, 1, 16, None), (256, 256, 256, 4, 8, 2, 1, 16, None), (128, 128, 512, 4, 4, 2, 2, 16, None),
    (256, 128, 512, 4, 8, 2, 1, 16, None), (128, 256, 512, 4, 8, 2, 1, 16, None), (128, 128, 256, 4, 4, 2, 2, 32, None),
    (256, 256, 256, 8, 8, 2, 0, 32, None)]


def ph_triton(a):
    gpu_setup()
    try:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
    except Exception:  # noqa: BLE001
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
    t_end = time.time() + a.budget_s
    _, table = asm_table()
    for name, N, K, _ in [s for s in SHAPES if s[0] in ("gate_up", "down", "qkv_s", "o_s")]:
        if time.time() > t_end:
            emit("triton", note="budget reached", shape=name)
            break
        w = Weights(N, K)
        M = 8192
        extra = []
        for c in TRITON_CFGS:
            bm, bn, bk, g, wp, sg, wpe, mi, cm = c
            cfg = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=g, num_warps=wp, num_stages=sg,
                       waves_per_eu=wpe, matrix_instr_nonkdim=mi, cache_modifier=cm, NUM_KSPLIT=1)
            vname = f"triton_bm{bm}_bn{bn}_bk{bk}_g{g}_w{wp}_s{sg}_wpe{wpe}_mi{mi}" + (f"_{cm[1:]}" if cm else "")

            def make(acts, w, out, M, N, K, cfg=cfg):
                def call(i):
                    A, B, SA, SB = acts.A[i % acts.R], w.B[i % w.R], acts.S[i % acts.R], w.S[i % w.R]
                    gemm_afp4wfp4_preshuffle(A.view(torch.uint8), B.view(torch.uint8).view(N // 16, -1),
                                             SA.view(torch.uint8).view(-1, SA.shape[-1] * 32),
                                             SB.view(torch.uint8).view(-1, SB.shape[-1] * 32),
                                             dtype=torch.bfloat16, y=out[:M], config=dict(cfg))
                return call
            extra.append((vname, make, 0))
        only = {"engine"} | {v[0] for v in extra}
        res = run_point("triton", name, N, K, M, w, table, a.min_s, extra_variants=extra, only=only)
        for r in res:
            if "us" in r:
                log(f"triton {name} M={M} {r['variant']}: {r['us']:.1f} us {r['pflops']:.3f} PF err {r['err_vs_engine']}")
        del w
        torch.cuda.empty_cache()


def ph_ck(a):
    gpu_setup()
    t = time.time()
    try:
        w = Weights(5376, 8192)
        acts = Acts(64, 8192)
        out = torch.empty((256, 5376), dtype=torch.bfloat16, device="cuda")
        aiter.gemm_a4w4_blockscale_tune(acts.A[0], w.B[0], acts.S[0], w.S[0], out, 0, 0)
        torch.cuda.synchronize()
        emit("ck", note="module ready", build_s=round(time.time() - t, 1))
        log(f"CK a4w4 tune module ready after {time.time() - t:.0f}s")
        del w, acts, out
    except Exception as e:  # noqa: BLE001
        emit("ck", error=f"module build/load failed: {e!r}"[:600], build_s=round(time.time() - t, 1))
        log(f"CK module failed after {time.time() - t:.0f}s: {e!r}"[:400])
        return
    t_end = time.time() + a.budget_s
    _, table = asm_table()
    ms = [int(x) for x in a.ms.split(",")] if a.ms else [16, 32, 64, 128, 256, 384, 2048, 8192]
    for name, N, K, _ in SHAPES:
        w = Weights(N, K)
        for M in ms:
            if time.time() > t_end:
                emit("ck", note="budget reached", shape=name, M=M)
                return
            extra = []
            for kid in range(a.ck_kernels):
                for sk in ([0, 1, 2, 3] if M <= 128 else [0]):
                    def make(acts, w, out, M, N, K, kid=kid, sk=sk):
                        def call(i):
                            aiter.gemm_a4w4_blockscale_tune(acts.A[i % acts.R], w.B[i % w.R], acts.S[i % acts.R], w.S[i % w.R],
                                                            out, kid, sk)
                        return call
                    extra.append((f"ck_{kid}_sk{sk}", make, sk))
            only = {"engine"} | {v[0] for v in extra}
            res = run_point("ck", name, N, K, M, w, table, a.min_s, extra_variants=extra, only=only)
            ok = [r for r in res if "us" in r and r.get("err_vs_engine", 9) < 0.03 and r["variant"] != "engine"]
            eng = next((r for r in res if r["variant"] == "engine" and "us" in r), None)
            if ok and eng:
                b = min(ok, key=BY_US)
                log(f"ck {name} M={M}: engine {eng['us']:.1f} us | best CK {b['variant']} {b['us']:.1f} us ({eng['us'] / b['us']:.3f}x)")
        del w
        torch.cuda.empty_cache()


def selftest():
    a = f4gemm_kernargs(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2)
    b = f4gemm_kernargs(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2, num_cu=256)
    assert len(a) == 372 and len(b) == 376, (len(a), len(b))
    assert struct.unpack_from("<i", a, 368)[0] == 2 and struct.unpack_from("<I", b, 372)[0] == 256
    assert struct.unpack_from("<I", a, 224)[0] == 9 and struct.unpack_from("<Q", a, 288)[0] == 5
    assert k_slices(21504, 8) == 8 and k_slices(5376, 8) == 7 and k_slices(8192, 1) == 1   # C++: k_per_tg rounded up to 256
    tbl = [dict(tile_M=tm, tile_N=tn, splitK=sk, bpreshuffle=1, ps=0, knl_name=f"k{tm}x{tn}", co_name="") for tm, tn, sk in
           ((256, 256, 1), (128, 512, 1), (32, 128, 0), (160, 384, 0), (64, 128, 0))]
    names = [x[0] for x in variants_for(8192, 5376, tbl)]
    assert "asm_128x512" not in names and "asm_32x128" not in names and "asm_160x384" in names, names
    names = [x[0] for x in variants_for(64, 43008, tbl)]
    assert "asm_256x256_ks3" in names and "asm_128x512_ks2" in names and "asm_160x384" not in names, names
    m = ClockMon.__new__(ClockMon)
    m.samples = [(1.0, 2000, 900, 100), (1.05, 2100, 950, 100), (5.0, 1500, 500, 10)]
    w = ClockMon.window(m, 0.9, 1.1)
    assert w["clk_mhz"] == 2050 and w["n_clk"] == 2, w
    w = ClockMon.window(m, 1.2, 1.3)   # empty window: nearest within 0.5 s
    assert w["n_clk"] == 2, w
    r = rate_fields(2.0 * 8192 * 5376 * 21504, 400.0, dict(clk_mhz=2000.0))
    assert abs(r["peak_at_clk_pf"] - 8.389) < 0.01 and 50 < r["pct_peak_at_clk"] < 60, r
    print("selftest ok")


def main():
    global OUT, MON
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["inventory", "mfma", "sweep", "ps", "confirm", "ab", "hipblaslt", "triton", "ck", "selftest"])
    ap.add_argument("--out", default="/root/or/out/gemm")
    ap.add_argument("--shape", default="")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--ms", default="")
    ap.add_argument("--min-s", type=float, default=0.12)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--budget-s", type=float, default=900)
    ap.add_argument("--ck-kernels", type=int, default=21)
    ap.add_argument("--shape-timeout", type=float, default=1500)
    ap.add_argument("--variants", default="asm_256x256,asm_160x384,asm_128x384")
    a = ap.parse_args()
    import faulthandler
    faulthandler.enable()
    if a.phase == "selftest":
        return selftest()
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    if not (a.phase in ("sweep", "ab") and not a.shape):
        MON = ClockMon()
        time.sleep(0.5)
    t = time.time()
    try:
        globals()[f"ph_{a.phase}"](a)
    except Exception:
        traceback.print_exc()
        emit("errors", phase_failed=a.phase, tb=traceback.format_exc()[-3000:])
        raise
    finally:
        if MON:
            MON.halt = True
        log(f"{a.phase} done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
