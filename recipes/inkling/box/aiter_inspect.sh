#!/bin/bash
# aiter_inspect.sh: what the SGLang MI35x image actually runs for Inkling-2-Mini, before and after our AITER tuning.
# Versions, the AITER BF16 GEMM tuner and its CSV formats, the Inkling2 quantization code paths on ROCm (which MoE and
# GEMM functions an NVFP4 / MXFP8 expert reaches on gfx950), and how many "not found tuned config" shapes the running
# server logged. Output: /root/or/out/aiter_inspect_<tag>.txt (pulled by the step).
TAG=${1:-base}
OUT=/root/or/out/aiter_inspect_${TAG}.txt; mkdir -p /root/or/out
{
echo "== date $(date -u +%FT%TZ) tag=$TAG"
python3 - <<'PY'
import importlib.util, importlib.metadata as md, os, glob, re
def ver(p):
    try: return md.version(p)
    except Exception as e: return f"? ({e.__class__.__name__})"
for p in ("sglang", "torch", "triton", "amd-aiter", "aiter", "flashinfer-python"):
    print(f"version {p}: {ver(p)}")
a = os.path.dirname(importlib.util.find_spec("aiter").origin)
s = os.path.dirname(importlib.util.find_spec("sglang").origin)
root = os.path.dirname(a)
print("aiter dir", a, "| sglang dir", s)
for pat in ("**/gemm_tuner.py", "**/gradlib/**/*.py", "**/*bf16*untuned*.csv", "**/bf16_tuned_gemm.csv"):
    hits = sorted(glob.glob(os.path.join(root, pat), recursive=True))[:8]
    print(f"find {pat}: {hits}")
for f in sorted(glob.glob(os.path.join(a, "configs", "*bf16*untuned*.csv")))[:2] + sorted(glob.glob(os.path.join(a, "configs", "bf16_tuned_gemm.csv")))[:1]:
    with open(f) as fh:
        print(f"head {f}:", fh.readline().strip(), "|", fh.readline().strip())
print("model_configs bf16:", sorted(os.path.basename(x) for x in glob.glob(os.path.join(a, "configs", "model_configs", "*bf16*"))))
envs = set()
for f in glob.glob(os.path.join(a, "**", "*.py"), recursive=True):
    try:
        t = open(f, errors="ignore").read()
    except Exception:
        continue
    envs.update(re.findall(r"AITER_(?:CONFIG|TUNE)[A-Z0-9_]*", t))
print("aiter env names:", sorted(envs)[:40])
# Inkling2 quantization on ROCm: which files mention it, and which MoE / GEMM entry points they call
hits = []
for f in glob.glob(os.path.join(s, "srt", "**", "*.py"), recursive=True):
    t = open(f, errors="ignore").read()
    if re.search(r"inkling2", t, re.I) and re.search(r"quant|moe|fp4|mxfp8", t, re.I):
        hits.append(f)
print("sglang inkling2 quant/moe files:", [os.path.relpath(h, s) for h in hits][:20])
for h in hits[:6]:
    t = open(h, errors="ignore").read()
    calls = sorted(set(re.findall(r"\b(fused_moe\w*|ck_moe\w*|asm_moe\w*|triton_kernel_moe\w*|fused_experts\w*|dequant\w*|"
                                  r"mxfp4\w*|nvfp4\w*|gemm_a\w*|cutlass\w*|flashinfer\w*|_is_hip|is_hip|is_gfx95\w*|aiter\.\w+)", t)))
    print(f"calls in {os.path.relpath(h, s)}:", calls[:40])
PY
for L in /root/or/server_*.log; do
  [ -f "$L" ] || continue
  echo "log $L: untuned-shape lines $(grep -c 'not found tuned config' "$L"), distinct $(grep -o 'shape is M:[0-9]*, N:[0-9]*, K:[0-9]*' "$L" | sort -u | wc -l)"
done
} > "$OUT" 2>&1
tail -n 30 "$OUT"
