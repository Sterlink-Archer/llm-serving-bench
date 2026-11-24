#!/usr/bin/env python3
"""Regression test for bench/startup_facts.py against REAL engine logs saved in results/ (no GPU needed).
The MI355X case guards the 2026-11-22 bug: '--attention-backend' selected backends ('Using X backend (selected via
--attention-backend)') were not recognised, so OpenRelay's own recipe was wrongly rejected before measurement."""
import glob
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import startup_facts as SF  # noqa: E402

CASES = [("results/pilot_h100_vllm_fp8/*/server_fp8dyn.log", "FLASH_ATTN/FA4", False, 166343),
         ("results/s1_mi355x_openrelay_recipe/*/server_or_recipe.log", "ROCM_AITER_UNIFIED_ATTN", False, 977076),
         ("results/s1_rtx5090_nvfp4/*/server_nvfp4.log", "TRITON_ATTN", True, 20611),
         ("results/s1_rtxpro6000_nvfp4/*/server_nvfp4.log", "TRITON_ATTN", True, 269346)]
bad = []
for pat, glob512, rejected, kv in CASES:
    fs = sorted(glob.glob(str(ROOT / pat)))
    if not fs:
        bad.append(f"missing fixture {pat}")
        continue
    f, _ = SF.parse(pathlib.Path(fs[0]).read_text(errors="replace"), "vllm")
    if f["attention_backend"]["global_512"] != glob512 or f["rejected"] != rejected or f["kv_cache_tokens"] != kv:
        bad.append(f"{pat}: got {f['attention_backend']} rejected={f['rejected']} kv={f['kv_cache_tokens']}")
# SGLang 0.5.23 prints server_args as a dict and logs decode graphs as "Capture target decode CUDA graph end. bs=[...]"
# (e4_rtxpro6000_sglang, 2026-11-22: the first parser missed both and called the start "missing attention_backend").
_SGL = """[2026-11-22 05:41:32] server_args={'model_path': '/root/models/nvfp4_modelopt', 'attention_backend': 'triton', 'decode_attention_backend': None, 'prefill_attention_backend': None, 'kv_cache_dtype': 'fp8_e4m3'}
[2026-11-22 05:41:51] Use sliding window memory pool. full_layer_tokens=148789, swa_layer_tokens=119031
[2026-11-22 05:42:12] Capture target decode CUDA graph end. bs=[1, 2, 4, 8], elapsed=7.34 s
[2026-11-22 05:42:12] max_total_num_tokens=148789, chunked_prefill_size=8192, context_len=32768
"""
_f, _ = SF.parse(_SGL, "sglang")
if not (_f["attention_backend"]["global_512"] == "triton" and _f["cuda_graph_sizes"] == [1, 2, 4, 8] and _f["kv_cache_tokens"] == 148789
        and _f["kv_cache_dtype"] == "fp8_e4m3" and _f["rejected"] and "Triton" in (_f["reject_reason"] or "")):
    bad.append(f"sglang 0.5.23 dict-format log: {_f}")
# Speculative decoding (Inkling DSpark, ik2_pro6000_spec 2026-11-22) captures no decode graph: the batch-size list is the TARGET
# verify graph's; the draft verify graph and a later decode list must not be confused with it.
_SPEC = """[2026-11-22 08:15:33] Capture target prefill CUDA graph end. elapsed=23.93 s, mem usage=7.73 GB, avail mem=11.04 GB.
[2026-11-22 08:15:36] Capture target verify CUDA graph end. bs=[1, 2, 3, 4, 8, 16, 32], elapsed=3.77 s, mem usage=0.11 GB, avail mem=10.93 GB.
[2026-11-22 08:15:38] Capture draft verify CUDA graph end. bs=[1, 2, 4], elapsed=1.20 s, mem usage=0.09 GB, avail mem=10.84 GB.
"""
_f, _ = SF.parse(_SPEC, "sglang", rules={"require_global_512": False, "triton_ok": True})
if _f["cuda_graph_sizes"] != [1, 2, 3, 4, 8, 16, 32] or _f.get("cuda_graph_basis") != "target_verify":
    bad.append(f"sglang spec-decode graph list: {_f['cuda_graph_sizes']} basis {_f.get('cuda_graph_basis')}")
_f, _ = SF.parse(_SGL + _SPEC, "sglang")
if _f["cuda_graph_sizes"] != [1, 2, 4, 8] or _f.get("cuda_graph_basis") != "decode":
    bad.append(f"a decode list must win over a verify list: {_f['cuda_graph_sizes']} basis {_f.get('cuda_graph_basis')}")
_real = sorted(glob.glob(str(ROOT / "results/ik2_pro6000_spec/*/server_swa15_dspark.log")))
if _real:
    _f, _ = SF.parse(pathlib.Path(_real[-1]).read_text(errors="replace"), "sglang", rules={"require_global_512": False, "triton_ok": True})
    if not _f["cuda_graph_sizes"] or _f.get("cuda_graph_basis") != "target_verify":
        bad.append(f"{_real[-1]}: graph list {_f['cuda_graph_sizes']} basis {_f.get('cuda_graph_basis')}")
print("FAIL " + "; ".join(bad) if bad else f"startup facts OK on {len(CASES)} real logs + the SGLang 0.5.23 dict format + spec-decode verify graphs")
sys.exit(1 if bad else 0)

