#!/bin/bash
# inspect.sh: what the engine image actually contains, written before any engine starts (runs from setup_cmd).
# Versions, the AITER pieces OpenRelay upstreamed (ROCm/aiter#5926-#5929) and the anchors our patches need.
# Output: /root/or/out/inspect.txt (pulled by the `inspect` sh step).
OUT=/root/or/out/inspect.txt; mkdir -p /root/or/out
{
echo "== date $(date -u +%FT%TZ)"
python3 - <<'PY'
import importlib.util, importlib.metadata as md, os, json, re, glob
def ver(p):
    try: return md.version(p)
    except Exception as e: return f"? ({e.__class__.__name__})"
for p in ("vllm", "torch", "triton", "amd-aiter", "aiter"):
    print(f"version {p}: {ver(p)}")
a = os.path.dirname(importlib.util.find_spec("aiter").origin)
v = os.path.dirname(importlib.util.find_spec("vllm").origin)
print("aiter dir", a); print("vllm dir", v)
ua = os.path.join(a, "ops/triton/attention/unified_attention.py")
s = open(ua).read() if os.path.exists(ua) else ""
m = re.search(r"\n(\s*if params\.head_size >= 512[^\n]*)", s)
print("ua3d anchor (use_2d D>=512 clause):", repr(m.group(1)) if m else "NOT FOUND")
print("ua3d already applied:", "max_seqlen_q > 16" in s)
tbl = os.path.join(a, "ops/triton/configs/gfx950/triton/attention/unified_attention/DEFAULT.json")
if os.path.exists(tbl):
    d = json.load(open(tbl))
    for sec, ent in d.items():
        if isinstance(ent, dict):
            for k, val in ent.items():
                if "512" in k or "GEQ_256" in k:
                    print(f"table {sec} {k} {val}")
else:
    print("table missing:", tbl)
for f in sorted(glob.glob(os.path.join(a, "configs/model_configs/*gemma*"))):
    print("model_config", os.path.basename(f), sum(1 for _ in open(f)) - 1, "rows")
act = os.path.join(a, "ops/triton/_triton_kernels/fusions/activation.py")
if not os.path.exists(act): act = os.path.join(a, "ops/triton/_triton_kernels/activation.py")
t = open(act).read() if os.path.exists(act) else ""
i = t.find("def _act_mul_and_dynamic_mxfp4_quant_kernel(")
print("scaleM_pad constexpr in act_mul kernel:", "scaleM_pad: tl.constexpr" in t[i:t.find("):", i)] if i >= 0 else "kernel not found")
g = os.path.join(v, "model_executor/models/gemma4.py")
gs = open(g).read() if os.path.exists(g) else ""
print("gemma4.py Gemma4MLP anchor:", "class Gemma4MLP(nn.Module):" in gs, "| g4 fuse applied:", "g4_gelu_mxfp4_gemm" in gs)
ao = os.path.join(v, "_aiter_ops.py")
ts = open(ao).read() if os.path.exists(ao) else ""
ka = os.path.join(v, "model_executor/kernels/linear/mxfp4/aiter.py")
ks = open(ka).read() if os.path.exists(ka) else ""
print("vllm is_asm_fp4_gemm_tuned present:", "def is_asm_fp4_gemm_tuned" in ts,
      "| presh_ws gate in kernel:", "is_triton_gemm_afp4wfp4_presh_ws_tuned" in ks, "| MX force patch:", "MX_FORCE_ASM_FP4" in ks)
tj = glob.glob(os.path.join(a, "ops/triton/configs/gfx950/triton/gemm/gemm_afp4wfp4_preshuffled/*K=5376*"))
print("gfx950 AFP4WFP4_PRESHUFFLED JSONs for K=5376:", [os.path.basename(x) for x in tj])
envs = open(os.path.join(v, "envs.py")).read()
print("VLLM_ROCM_USE_AITER_FP4_ASM_GEMM in envs:", "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM" in envs)
PY
ls /root/or/mx 2>/dev/null | sed 's/^/mx file: /'
grep -h "^PATCH" /root/or/mx/*.status 2>/dev/null
} > "$OUT" 2>&1
tail -n 40 "$OUT"
