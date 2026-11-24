#!/usr/bin/env python3
"""bf16_headroom.py: can Gemma 4 31B's BF16 GEMMs (target lm_head, MTP drafter linears) run in FP8 on one MI355X?

Runs inside OpenRelay's vLLM 0.30 image, no engine on the GPU, next to box/gemm_headroom.py (clock sampler, graph timing,
JSONL rows). For each shape: BF16 F.linear (hipBLASLt, the engine path for unquantized linears) against FP8 e4m3
weights with dynamic activation quant, timed INCLUDING that activation quant:
  fp8_tensor   per-tensor scales, torch._scaled_mm (hipBLASLt FP8)
  fp8_rowwise  per-token x per-channel scales, torch._scaled_mm
  aiter_a8w8   per-token x per-channel, aiter.gemm_a8w8 (CK)
Weights rotate over >= 768 MB of copies per variant (3x the Infinity Cache) so every call streams from HBM.
Output error vs BF16: relative L2; for lm_head also top-1 / top-5 agreement of the logits.

Shapes: target lm_head 262144 x 5376 at M = sampled positions per step (k+1 = 4 per decoding request, up to 96 x 4);
drafter (google/gemma-4-31B-it-assistant: 4 KV-shared layers, hidden 1024, 32 q heads x 256 / 512, intermediate 8192,
backbone 5376, 2048 centroids) at M = requests per draft pass.
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

SHAPES = [  # name, N, K, M list
    ("lm_head", 262144, 5376, [16, 64, 144, 256, 384]),
    ("d_q_sliding", 8192, 1024, [36, 144]), ("d_q_global", 16384, 1024, [36, 144]),
    ("d_o_sliding", 1024, 8192, [36, 144]), ("d_o_global", 1024, 16384, [36, 144]),
    ("d_gate_up", 16384, 1024, [36, 144]), ("d_down", 1024, 8192, [36, 144]),
    ("d_in_proj", 1024, 6400, [36, 144]), ("d_out_proj", 5376, 1024, [36, 144]), ("d_centroids", 2048, 1024, [36, 144]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/or/out/bf16")
    ap.add_argument("--min-s", type=float, default=0.15)
    ap.add_argument("--shapes", default="", help="comma list; default all")
    a = ap.parse_args()
    import faulthandler
    faulthandler.enable()
    G.OUT = a.out
    os.makedirs(a.out, exist_ok=True)
    G.MON = G.ClockMon()
    time.sleep(0.5)
    G.gpu_setup()
    import torch
    import aiter
    f8 = torch.float8_e4m3fn
    F8MAX = torch.finfo(f8).max
    t_all = time.time()

    def q_tensor(x):
        s = (x.abs().amax().float() / F8MAX).clamp(min=1e-12)
        return (x.float() / s).to(f8), s.reshape(1)

    def q_row(x):            # per-row (token for activations, channel for weights)
        s = (x.abs().amax(dim=1, keepdim=True).float() / F8MAX).clamp(min=1e-12)
        return (x.float() / s).to(f8), s

    try:
        for name, N, K, ms in [x for x in SHAPES if not a.shapes or x[0] in a.shapes.split(",")]:
            w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
            wq_t, ws_t = q_tensor(w)
            wq_r, ws_r = q_row(w)
            r16 = max(2, min(32, math.ceil((768 << 20) / (N * K * 2))))
            r8 = max(2, min(64, math.ceil((768 << 20) / (N * K))))
            W16 = [w] + [w.clone() for _ in range(r16 - 1)]
            W8t = [wq_t] + [wq_t.clone() for _ in range(r8 - 1)]
            W8r = [wq_r] + [wq_r.clone() for _ in range(r8 - 1)]
            for M in ms:
                xs = [(torch.randn(M, K, device="cuda")).to(torch.bfloat16) for _ in range(2)]
                ref = torch.nn.functional.linear(xs[0], w)

                def bf16(i):
                    return torch.nn.functional.linear(xs[i % 2], W16[i % r16])

                def fp8_tensor(i):
                    xq, xs_ = q_tensor(xs[i % 2])
                    return torch._scaled_mm(xq, W8t[i % r8].t(), scale_a=xs_, scale_b=ws_t.reshape(1), out_dtype=torch.bfloat16)

                def fp8_rowwise(i):
                    xq, xs_ = q_row(xs[i % 2])
                    return torch._scaled_mm(xq, W8r[i % r8].t(), scale_a=xs_, scale_b=ws_r.t().contiguous(), out_dtype=torch.bfloat16)

                def aiter_a8w8(i):
                    xq, xs_ = q_row(xs[i % 2])
                    return aiter.gemm_a8w8(xq, W8r[i % r8], xs_, ws_r.t().contiguous(), dtype=torch.bfloat16)
                # mxBF1_fp8: hipBLASLt FP8 (torch._scaled_mm) segfaulted in CUDA-graph replay at the first drafter shape
                # (8192 x 1024, M=36), so off lm_head the _scaled_mm variants are timed eagerly, next to an eager BF16 row
                small = name != "lm_head"
                plan = [("bf16", bf16, r16, True), ("aiter_a8w8", aiter_a8w8, r8, True)]
                if small:
                    plan += [("bf16_eager", bf16, r16, False), ("fp8_tensor_eager", fp8_tensor, r8, False),
                             ("fp8_rowwise_eager", fp8_rowwise, r8, False)]
                else:
                    plan += [("fp8_tensor", fp8_tensor, r8, True), ("fp8_rowwise", fp8_rowwise, r8, True)]
                for vname, fn, k, use_graph in plan:
                    try:
                        y = fn(0)
                        torch.cuda.synchronize()
                        err = G.rel_err(y[:M], ref)
                        chk = dict(err_vs_bf16=round(err, 5))
                        if name == "lm_head":
                            t1 = (y.float().argmax(-1) == ref.float().argmax(-1)).float().mean().item()
                            top5 = ref.float().topk(5, dim=-1).indices
                            t5 = (top5 == y.float().argmax(-1, keepdim=True)).any(-1).float().mean().item()
                            chk.update(top1_agree=round(t1, 4), argmax_in_bf16_top5=round(t5, 4))
                        us, t0, t1_, mode = G.time_graph(fn, k, min_s=a.min_s, use_graph=use_graph)
                        byt = N * K * (2 if vname.startswith("bf16") else 1)
                        G.emit("bf16", shape=name, M=M, N=N, K=K, variant=vname, us=round(us, 3), mode=mode,
                               weight_tb_s=round(byt / us / 1e6, 3), **chk, **G.MON.window(t0, t1_))
                        G.log(f"{name} M={M} {vname}: {us:.1f} us ({byt / us / 1e6:.2f} TB/s weights) {chk}")
                    except Exception as e:  # noqa: BLE001
                        G.emit("bf16", shape=name, M=M, N=N, K=K, variant=vname, error=repr(e)[:400])
                        G.log(f"{name} M={M} {vname}: ERROR {e!r}"[:300])
                        torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc()
        G.emit("bf16_errors", tb=traceback.format_exc()[-3000:])
    finally:
        G.MON.halt = True
        G.log(f"bf16 done in {time.time() - t_all:.0f}s")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)          # the glue runs segfaulted at interpreter exit after finishing


if __name__ == "__main__":
    main()
