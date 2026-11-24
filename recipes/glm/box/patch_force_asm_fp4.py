"""vLLM >= 0.31 (ROCm): let MX_FORCE_ASM_FP4=1 restore v0.30's behaviour of running every MXFP4 linear on AITER's
assembly GEMM (gemm_a4w4).

Why: v0.30 always used the assembly kernel on gfx950 (later builds: VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1). v0.34 drops
the variable; `AiterMxfp4LinearKernel.process_weights_after_loading` sets
  use_asm_gemm = is_asm_fp4_gemm_tuned(n, k)            # AITER a4w4 CSV has rows for this (N, K) at M=1024 AND M=8192
                 and is_triton_gemm_afp4wfp4_presh_ws_tuned(n, k)   # a gfx950 Triton preshuffle JSON exists for (N, K)
AITER v0.1.28.post1's Gemma 4 31B rows (ROCm/aiter#5928) keep only rows >= 3% better than the default, so several
shapes have no M=1024/8192 row, and there is no gfx950 preshuffle JSON for any K=5376 shape: every Gemma 4 linear would
fall back to AITER's Triton FP4 GEMM (OpenRelay, vllm#59153: 1.89 vs 2.77 prefill req/s).
With the override, M <= 64 still takes the Triton preshuffle kernel only where it is tuned; everything else runs
gemm_a4w4 on (16,16)-shuffled weights, exactly as v0.30 did.

Env-gated (read at weight load), so a stock server on the same box is untouched. Idempotent; writes a PATCH line."""
import importlib.util
import os
import sys

p = os.path.join(os.path.dirname(importlib.util.find_spec("vllm").origin), "model_executor/kernels/linear/mxfp4/aiter.py")
s = open(p).read()
status = "/root/or/mx/patch_force_asm_fp4.status"
if "MX_FORCE_ASM_FP4" in s:
    msg = "PATCH force_asm_fp4: already applied"
else:
    anchor = "        if self.use_asm_gemm and not _asm_fp4_scale_swizzle_supported(\n"
    if s.count(anchor) != 1 or "self.use_asm_gemm = rocm_aiter_ops.is_asm_fp4_gemm_tuned(" not in s:
        msg = "PATCH force_asm_fp4: ANCHOR NOT FOUND (not applied)"
    else:
        s = s.replace(anchor, "        import os as _mx_os\n"
                      "        if _mx_os.environ.get(\"MX_FORCE_ASM_FP4\") == \"1\" and self.asm_gemm_supported:\n"
                      "            self.use_asm_gemm = True\n" + anchor, 1)
        open(p, "w").write(s)
        msg = "PATCH force_asm_fp4: applied (kernels/linear/mxfp4/aiter.py)"
os.makedirs(os.path.dirname(status), exist_ok=True)
open(status, "w").write(msg + "\n")
print(msg, file=sys.stderr)
