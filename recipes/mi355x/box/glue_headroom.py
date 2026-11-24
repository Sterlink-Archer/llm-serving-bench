#!/usr/bin/env python3
"""glue_headroom.py: the non-GEMM "glue" of Gemma 4 31B on one MI355X: does aiter have faster or fused kernels for it?

Runs inside OpenRelay's vLLM 0.30 image (aiter 0.1.21.post2), no engine on the GPU, next to box/gemm_headroom.py (whose
clock sampler, graph timing and JSONL rows it reuses). Each "engine" variant is a torch.compile replica (token dim dynamic, as vLLM marks it) of
what vLLM 0.30 runs for Gemma 4 (vllm/model_executor/models/gemma4.py: per-head q_norm / k_norm / weightless v_norm,
neox RoPE from cos_sin_cache[positions], vllm reshape_and_cache_flash into an fp8 NHD paged cache; sandwich norms
rms(x)*w + residual -> rms(.)*w, then aiter.per_1x32_f4_quant_hip(shuffle=True) as vLLM's gemm_with_dynamic_quant does),
so Inductor emits the same kind of fused kernels the knee profile shows. Every variant's outputs are checked against it.

Phases (python3 glue_headroom.py <phase> --out DIR):
  prep      attention prep per layer type (sliding: 32 q / 16 kv heads x 256, default RoPE 1e4; global: 32 q / 4 kv
            heads x 512, k_eq_v, proportional RoPE 1e6 with 64 of 256 angle pairs):
              engine  Inductor(split + q/k/v RMSNorm + RoPE) + vllm reshape_and_cache_flash (fp8)
              rope_kv Inductor(split + norms) + aiter Triton fused_qk_rope_reshape_and_cache   (vLLM fuse_rope_kvcache)
              qkv_fused  Inductor(v_norm in place) + aiter Triton fused_qkv_split_qk_norm_rope_cache
              hip_fused  Inductor(v_norm in place) + aiter HIP fused_qk_norm_rope_cache_pts_quant_shuffle (d256 only;
                         vLLM's fuse_qk_norm_rope_kvcache kernel)
  norm      the two sandwich-norm sites per layer (+ the MXFP4 quant that feeds the next GEMM):
              engine  Inductor(rms(x)*w1 + res [* s]; rms(.)*w2) + aiter per_1x32_f4_quant_hip
              sandwich_v130  aiter v0.1.30 fused_rmsnorm_add_rmsnorm kernel (inlined; Gemma "1 + w" weights passed as w - 1)
                             + per_1x32_f4_quant_hip   (site without the layer scalar only)
              rms_quant      Inductor(rms(x)*w1 + res [* s]) + aiter Triton fused_rms_mxfp4_quant(shuffle=True)
  inventory which candidate aiter functions import in this image
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemm_headroom as G  # noqa: E402

EPS = 1e-6
LAYER = {  # Hq, Hkv, D, rope theta, rotated angle pairs (of D/2), k_eq_v, layers per forward
    "sliding": (32, 16, 256, 1e4, 128, False, 50),
    "global": (32, 4, 512, 1e6, 64, True, 10),
}
MS = [256, 1024, 2048, 4096, 6912, 8192]   # 6912 = the k=3 profile's mean mixed-step M


def gpu():
    G.gpu_setup()
    global torch, aiter
    import torch  # noqa
    import aiter  # noqa
    torch._dynamo.config.cache_size_limit = 64
    import torch._inductor.config as ic
    ic.combo_kernels = True                 # vLLM 0.30 sets both for torch >= 2.9 (config/compilation.py)
    ic.benchmark_combo_kernel = True


def comp(fn):
    """vLLM compiles one graph with a symbolic token dim and static hidden dims (mark_dynamic on dim 0), not
    dynamic=True (that also makes head_dim / hidden symbolic; in run mxGL1_glue the replica then took ~2x the knee
    profile's per-call time for the same kernels)."""
    return torch.compile(fn, dynamic=False)


def dyn0(*ts):
    for t in ts:
        if t is not None and t.dim() > 0:
            torch._dynamo.mark_dynamic(t, 0)
    return ts


# ----------------------------------------------------------------------------- engine replicas (Inductor)
def rms(x, w):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    if w is not None:
        y = y * w.float()
    return y.to(x.dtype)


def rope_neox(x, cos, sin):           # x [T, H, D]; cos/sin [T, D/2]
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def make_prep_fns(Hq, Hkv, D):
    qs, ks = Hq * D, Hkv * D

    def engine(qkv, pos, qw, kw, cos_sin):
        T = qkv.shape[0]
        q, k, v = qkv.split([qs, ks, ks], dim=-1)
        q = rms(q.view(T, Hq, D), qw)
        k = rms(k.view(T, Hkv, D), kw)
        v = rms(v.view(T, Hkv, D), None)
        cs = cos_sin.index_select(0, pos)
        cos, sin = cs.chunk(2, dim=-1)
        return rope_neox(q, cos, sin), rope_neox(k, cos, sin), v

    def norms(qkv, qw, kw):
        T = qkv.shape[0]
        q, k, v = qkv.split([qs, ks, ks], dim=-1)
        return rms(q.view(T, Hq, D), qw), rms(k.view(T, Hkv, D), kw), rms(v.view(T, Hkv, D), None)

    def vnorm_inplace(qkv):
        T = qkv.shape[0]
        v = qkv[:, qs + ks:]
        v.copy_(rms(v.view(T, Hkv, D), None).view(T, ks))
        return qkv

    return comp(engine), comp(norms), comp(vnorm_inplace)


def cos_sin_cache(D, theta, pairs, max_pos=32768):
    """vLLM's cache: [max_pos, D] = cat(cos, sin) over D/2 frequencies; Gemma 4 proportional RoPE puts the exponent over
    head_size and zero-pads the non-rotated pairs (cos 1, sin 0); the default RoPE is the same with pairs = D/2."""
    exps = torch.arange(0, 2 * pairs, 2, dtype=torch.float) / D
    inv = 1.0 / (theta ** exps)
    if pairs < D // 2:
        inv = torch.cat([inv, torch.zeros(D // 2 - pairs)])
    t = torch.arange(max_pos, dtype=torch.float)
    f = torch.outer(t, inv)
    return torch.cat([f.cos(), f.sin()], dim=-1).to(torch.bfloat16).cuda()


def fp8(t):
    return t.view(torch.float8_e4m3fn)


def rel(a, b):
    a, b = a.float(), b.float()
    n = b.norm().item()
    return (a - b).norm().item() / n if n else float("inf")


def import_first(*paths):
    errs = []
    for mod, name in paths:
        try:
            m = __import__(mod, fromlist=[name])
            return getattr(m, name), f"{mod}.{name}"
        except Exception as e:  # noqa: BLE001
            errs.append(f"{mod}.{name}: {e!r}"[:160])
    raise ImportError("; ".join(errs))


# ----------------------------------------------------------------------------- phases
def ph_inventory(a):
    gpu()
    found = {}
    for label, cands in {
        "fused_qk_rope_reshape_and_cache": [("aiter.ops.triton.fused_kv_cache", "fused_qk_rope_reshape_and_cache"),
                                            ("aiter.ops.triton.fusions.fused_kv_cache", "fused_qk_rope_reshape_and_cache")],
        "fused_qkv_split_qk_norm_rope_cache": [("aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cache",
                                                "fused_qkv_split_qk_norm_rope_cache")],
        "fused_qk_norm_rope_cache_pts_quant_shuffle": [("aiter.ops.fused_qk_norm_rope_cache_quant",
                                                        "fused_qk_norm_rope_cache_pts_quant_shuffle")],
        "fused_rms_mxfp4_quant": [("aiter.ops.triton.quant.fused_mxfp4_quant", "fused_rms_mxfp4_quant"),
                                  ("aiter.ops.triton.quant", "fused_rms_mxfp4_quant")],
        "fused_rmsnorm_add_rmsnorm (v0.1.30)": [("aiter.ops.triton.normalization.fused_rmsnorm_add_rmsnorm",
                                                 "fused_rmsnorm_add_rmsnorm")],
        "act_mul_and_mxfp4_quant": [("aiter.ops.triton.activation", "act_mul_and_mxfp4_quant")],
        "per_1x32_f4_quant_hip": [("aiter", "per_1x32_f4_quant_hip")],
        "vllm reshape_and_cache_flash": [("vllm._custom_ops", "reshape_and_cache_flash")],
    }.items():
        try:
            _, where = import_first(*cands)
            found[label] = where
        except ImportError as e:
            found[label] = f"NOT IMPORTABLE: {e}"[:300]
    G.emit("glue_inventory", **found)
    for k, v in found.items():
        G.log(f"{k}: {v}")


def ph_prep(a):
    gpu()
    from vllm import _custom_ops as ops
    rope_kv, rope_kv_where = import_first(("aiter.ops.triton.fused_kv_cache", "fused_qk_rope_reshape_and_cache"),
                                          ("aiter.ops.triton.fusions.fused_kv_cache", "fused_qk_rope_reshape_and_cache"))
    qkv_fused = import_first(("aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cache", "fused_qkv_split_qk_norm_rope_cache"))[0]
    try:
        hip_fused = import_first(("aiter.ops.fused_qk_norm_rope_cache_quant", "fused_qk_norm_rope_cache_pts_quant_shuffle"))[0]
    except ImportError:
        hip_fused = None
    ms = [int(x) for x in a.ms.split(",")] if a.ms else MS
    for lt in (a.layers.split(",") if a.layers else LAYER):
        Hq, Hkv, D, theta, pairs, k_eq_v, nlayers = LAYER[lt]
        qs, ks = Hq * D, Hkv * D
        engine, norms, vnorm = make_prep_fns(Hq, Hkv, D)
        csc = cos_sin_cache(D, theta, pairs)
        cos, sin = csc.chunk(2, dim=-1)
        qw = (torch.randn(D, device="cuda") * 0.1 + 1).to(torch.bfloat16)
        kw = (torch.randn(D, device="cuda") * 0.1 + 1).to(torch.bfloat16)
        qw1, kw1 = (qw.float() - 1).to(torch.bfloat16), (kw.float() - 1).to(torch.bfloat16)   # aiter Triton op: 1 + w
        kscale = torch.ones(1, dtype=torch.float32, device="cuda")
        kscale_cpu = torch.ones(1, dtype=torch.float32)
        bs = 128 if D == 512 else 64
        for M in ms:
            g = torch.Generator(device="cuda").manual_seed(M)
            nb = math.ceil(M / bs) * 2 + 4
            sets = []
            for r in range(2):
                qkv = torch.randn(M, qs + 2 * ks, device="cuda", dtype=torch.bfloat16, generator=g)
                if k_eq_v:
                    qkv[:, qs + ks:] = qkv[:, qs:qs + ks]
                pos = (torch.arange(M, device="cuda") + 1000 * (r + 1)).to(torch.int64)
                perm = torch.randperm(nb, device="cuda", generator=g)[: math.ceil(M / bs)]
                slots = (perm.repeat_interleave(bs)[:M] * bs + torch.arange(M, device="cuda") % bs).to(torch.int64)
                dyn0(qkv, pos, slots)
                sets.append(dict(qkv=qkv, pos=pos, slots=slots))
            kc = torch.zeros(nb, bs, Hkv, D, dtype=torch.uint8, device="cuda")
            vc = torch.zeros_like(kc)
            s0 = sets[0]
            q_e, k_e, v_e = engine(s0["qkv"], s0["pos"], qw, kw, csc)
            ops.reshape_and_cache_flash(k_e, v_e, kc, vc, s0["slots"], "fp8", kscale, kscale)
            idx = (s0["slots"] // bs, s0["slots"] % bs)
            kref, vref, qref = fp8(kc)[idx].float().clone(), fp8(vc)[idx].float().clone(), q_e.reshape(M, Hq, D).clone()

            def check(q_out):
                return dict(err_q=round(rel(q_out.reshape(M, Hq, D), qref), 5), err_k=round(rel(fp8(kc)[idx].float(), kref), 5),
                            err_v=round(rel(fp8(vc)[idx].float(), vref), 5))
            # precomputed operands for the kernel-only timings
            pre = [engine(st["qkv"], st["pos"], qw, kw, csc) for st in sets]          # roped q, k, v (engine path)
            nrm = [norms(st["qkv"], qw, kw) for st in sets]                           # normed, un-roped
            work = []
            for st in sets:                                                          # GEMM output with V normed in place
                w_ = st["qkv"].clone()
                vnorm(w_)
                work.append(w_)
            q_out = torch.empty(M, Hq, D, dtype=torch.bfloat16, device="cuda")
            k_out = torch.empty(M, Hkv, D, dtype=torch.bfloat16, device="cuda")

            def e_full(i):
                st = sets[i % 2]
                q, k, v = engine(st["qkv"], st["pos"], qw, kw, csc)
                ops.reshape_and_cache_flash(k, v, kc, vc, st["slots"], "fp8", kscale, kscale)
                return q

            def e_compute(i):
                st = sets[i % 2]
                return engine(st["qkv"], st["pos"], qw, kw, csc)[0]

            def e_cache(i):
                q, k, v = pre[i % 2]
                ops.reshape_and_cache_flash(k, v, kc, vc, sets[i % 2]["slots"], "fp8", kscale, kscale)

            def rk_full(i):
                st = sets[i % 2]
                q, k, v = norms(st["qkv"], qw, kw)
                rope_kv(q, k, v, fp8(kc), fp8(vc), st["slots"], st["pos"], cos, sin, kscale, kscale, True, flash_layout=True,
                        apply_scale=True, q_out=q, k_out=k, output_zeros=False)
                return q

            def rk_norms(i):
                st = sets[i % 2]
                return norms(st["qkv"], qw, kw)[0]

            def rk_kernel(i):
                q, k, v = nrm[i % 2]
                rope_kv(q, k, v, fp8(kc), fp8(vc), sets[i % 2]["slots"], sets[i % 2]["pos"], cos, sin, kscale, kscale, True,
                        flash_layout=True, apply_scale=True, q_out=q_out, k_out=k_out, output_zeros=False)
                return q_out

            def qf_call(buf, st):
                out = qkv_fused(buf, qw1, kw1, cos, sin, st["pos"], fp8(kc), fp8(vc), st["slots"], Hq, Hkv, D, is_neox=True,
                                reuse_freqs_front_part=True, k_scale=kscale, v_scale=kscale, eps=EPS, kv_cache_layout="NHD")
                return out[0] if isinstance(out, (tuple, list)) else out

            def qf_full(i):
                vnorm(work[i % 2])
                return qf_call(work[i % 2], sets[i % 2])

            def qf_kernel(i):
                return qf_call(work[i % 2], sets[i % 2])

            def vn_only(i):
                vnorm(work[i % 2])

            def hip_call(buf, st):
                hip_fused(buf, qw, kw, csc, st["pos"], M, Hq, Hkv, Hkv, D, True, EPS, q_out, fp8(kc), fp8(vc), st["slots"],
                          kscale_cpu, kscale_cpu, k_out, None, True, False, bs, 16, 0)
                return q_out

            def hip_full(i):
                vnorm(work[i % 2])
                return hip_call(work[i % 2], sets[i % 2])

            def hip_kernel(i):
                return hip_call(work[i % 2], sets[i % 2])

            variants = [("engine", e_full, True), ("engine_compute", e_compute, False), ("engine_cache_write", e_cache, False),
                        ("rope_kv", rk_full, True), ("rope_kv_norms", rk_norms, False), ("rope_kv_kernel", rk_kernel, False),
                        ("qkv_fused", qf_full, True), ("qkv_fused_kernel", qf_kernel, False), ("vnorm", vn_only, False)]
            if hip_fused is not None and D in (64, 128, 256):
                variants += [("hip_fused", hip_full, True), ("hip_fused_kernel", hip_kernel, False)]
            for name, fn, checked in variants:
                try:
                    kc.zero_(), vc.zero_()
                    if checked and name != "engine":     # checked variants start from the pristine GEMM output
                        for w_, st in zip(work, sets, strict=True):
                            w_.copy_(st["qkv"])
                    q = fn(0)
                    torch.cuda.synchronize()
                    chk = check(q) if checked else {}
                    us, t0, t1, mode = G.time_graph(fn, 2, min_s=a.min_s)
                    win = G.MON.window(t0, t1)
                    G.emit("glue_prep", layer=lt, M=M, Hq=Hq, Hkv=Hkv, D=D, variant=name, us=round(us, 3), mode=mode,
                           layers_per_forward=nlayers, rope_kv_import=rope_kv_where, **chk, **win)
                    G.log(f"prep {lt} M={M} {name}: {us:.1f} us {chk}")
                except Exception as e:  # noqa: BLE001
                    G.emit("glue_prep", layer=lt, M=M, variant=name, error=repr(e)[:400])
                    G.log(f"prep {lt} M={M} {name}: ERROR {e!r}"[:300])
                    torch.cuda.synchronize()


def sandwich_v130_kernel():
    """aiter v0.1.30's _fused_rmsnorm_add_rmsnorm_kernel (aiter/ops/triton/_triton_kernels/normalization/
    fused_rmsnorm_add_rmsnorm.py, MIT), inlined because the image's aiter 0.1.21 does not have it; _rmsnorm_op written out
    (x * rsqrt(mean(x^2) + eps) * w, fp32)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _k(x_ptr, residual_ptr, post_weight_ptr, pre_weight_ptr, residual_out_ptr, pre_norm_ptr, width: tl.constexpr,
           post_eps, pre_eps, BLOCK_SIZE_N: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK_SIZE_N)
        valid = col < width
        offset = row * width + col
        x = tl.load(x_ptr + offset, valid, other=0).to(tl.float32)
        residual = tl.load(residual_ptr + offset, valid, other=0).to(tl.float32)
        post_weight = tl.load(post_weight_ptr + col, valid, other=0).to(tl.float32)
        pre_weight = tl.load(pre_weight_ptr + col, valid, other=0).to(tl.float32)
        r1 = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / width + post_eps)
        added = x * r1 * (post_weight + 1.0) + residual
        r2 = 1.0 / tl.sqrt(tl.sum(added * added, axis=0) / width + pre_eps)
        pre_norm = added * r2 * (pre_weight + 1.0)
        tl.store(residual_out_ptr + offset, added.to(tl.bfloat16), valid)
        tl.store(pre_norm_ptr + offset, pre_norm.to(tl.bfloat16), valid)

    def run(x, residual, post_w_minus1, pre_w_minus1, residual_out, out):
        M, N = x.shape
        _k[(M,)](x, residual, post_w_minus1, pre_w_minus1, residual_out, out, N, EPS, EPS,
                 BLOCK_SIZE_N=triton.next_power_of_2(N), num_warps=4 if M <= 128 else 8)
        return out
    return run


def ph_norm(a):
    gpu()
    quant = aiter.per_1x32_f4_quant_hip
    rms_q = import_first(("aiter.ops.triton.quant.fused_mxfp4_quant", "fused_rms_mxfp4_quant"),
                         ("aiter.ops.triton.quant", "fused_rms_mxfp4_quant"))[0]
    sandwich = sandwich_v130_kernel()
    N = 5376
    w1 = (torch.randn(N, device="cuda") * 0.2 + 1).to(torch.bfloat16)
    w2 = (torch.randn(N, device="cuda") * 0.2 + 1).to(torch.bfloat16)
    w1m, w2m = w1.float() - 1.0, w2.float() - 1.0
    ms = [int(x) for x in a.ms.split(",")] if a.ms else MS
    for site, scalar in (("post_attn->pre_ff", None), ("post_ff*s->next_input", 0.875)):
        def eng(x, res, s=scalar):
            y = rms(x, w1) + res
            if s is not None:
                y = y * s
            return y, rms(y, w2)

        def norm_add(x, res, s=scalar):
            y = rms(x, w1) + res
            return y * s if s is not None else y
        eng_c, na_c = comp(eng), comp(norm_add)
        for M in ms:
            g = torch.Generator(device="cuda").manual_seed(M + 7)
            sets = [dict(x=torch.randn(M, N, device="cuda", dtype=torch.bfloat16, generator=g),
                         res=torch.randn(M, N, device="cuda", dtype=torch.bfloat16, generator=g) * 4) for _ in range(2)]
            for st in sets:
                dyn0(st["x"], st["res"])
            y_ref, x_ref = eng_c(sets[0]["x"], sets[0]["res"])
            qref, sref = quant(x_ref, shuffle=True)
            y_ref, qref, sref = y_ref.clone(), qref.view(torch.uint8).clone(), sref.view(torch.uint8).clone()
            xs = [eng_c(st["x"], st["res"])[1] for st in sets]
            outs = [torch.empty(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
            res_out = [torch.empty(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(2)]

            def v_engine(i):
                y, xn = eng_c(sets[i % 2]["x"], sets[i % 2]["res"])
                return y, quant(xn, shuffle=True)

            def v_engine_norm(i):
                return eng_c(sets[i % 2]["x"], sets[i % 2]["res"])

            def v_quant(i):
                return quant(xs[i % 2], shuffle=True)

            def v_sandwich(i):
                st = sets[i % 2]
                pre = sandwich(st["x"], st["res"], w1m, w2m, res_out[i % 2], outs[i % 2])
                return res_out[i % 2], quant(pre, shuffle=True)

            def v_sandwich_kernel(i):
                st = sets[i % 2]
                return sandwich(st["x"], st["res"], w1m, w2m, res_out[i % 2], outs[i % 2])

            def v_rms_quant(i):
                y = na_c(sets[i % 2]["x"], sets[i % 2]["res"])
                r = rms_q(y, w2, EPS, shuffle=True)
                return y, (r[0] if isinstance(r[0], (tuple, list)) else (r[0], r[1]))

            variants = [("engine", v_engine, True), ("engine_norms", v_engine_norm, False), ("engine_quant", v_quant, False),
                        ("rms_quant", v_rms_quant, True)]
            if scalar is None:
                variants += [("sandwich_v130", v_sandwich, True), ("sandwich_v130_kernel", v_sandwich_kernel, False)]
            for name, fn, checked in variants:
                try:
                    out = fn(0)
                    torch.cuda.synchronize()
                    chk = {}
                    if checked:
                        y, (q, sc) = out
                        q8, s8 = q.view(torch.uint8), sc.view(torch.uint8)
                        chk = dict(err_resid=round(rel(y, y_ref), 6),
                                   fp4_bytes_equal=round((q8[:M] == qref[:M]).float().mean().item(), 5),
                                   scale_bytes_equal=round((s8 == sref).float().mean().item(), 5) if s8.shape == sref.shape
                                   else f"scale shape {tuple(s8.shape)} vs {tuple(sref.shape)}")
                    us, t0, t1, mode = G.time_graph(fn, 2, min_s=a.min_s)
                    G.emit("glue_norm", site=site, M=M, N=N, variant=name, us=round(us, 3), mode=mode, layers_per_forward=60,
                           **chk, **G.MON.window(t0, t1))
                    G.log(f"norm {site} M={M} {name}: {us:.1f} us {chk}")
                except Exception as e:  # noqa: BLE001
                    G.emit("glue_norm", site=site, M=M, variant=name, error=repr(e)[:400], tb=traceback.format_exc()[-800:])
                    G.log(f"norm {site} M={M} {name}: ERROR {e!r}"[:300])
                    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["inventory", "prep", "norm"])
    ap.add_argument("--out", default="/root/or/out/glue")
    ap.add_argument("--ms", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--min-s", type=float, default=0.15)
    a = ap.parse_args()
    import faulthandler
    faulthandler.enable()
    G.OUT = a.out
    os.makedirs(a.out, exist_ok=True)
    G.MON = G.ClockMon()
    time.sleep(0.5)
    t = time.time()
    try:
        {"inventory": ph_inventory, "prep": ph_prep, "norm": ph_norm}[a.phase](a)
    except Exception:
        traceback.print_exc()
        G.emit("glue_errors", phase_failed=a.phase, tb=traceback.format_exc()[-3000:])
        raise
    finally:
        G.MON.halt = True
        G.log(f"glue {a.phase} done in {time.time() - t:.0f}s")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)          # mxGL1_glue: both phases finished, then the interpreter segfaulted at exit (rc 139)


if __name__ == "__main__":
    main()
