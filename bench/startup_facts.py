#!/usr/bin/env python3
"""Parse an engine startup log into startup_facts_<server>.json (docs/DATA_CONTRACT.md §4) and the phase timings
inside engine start (weight load, torch.compile, CUDA-graph capture). Patterns are taken from real logs:
vLLM 0.33.0 on H100 (results/pilot_h100_vllm_fp8/20261122T034304Z/server_fp8dyn.log). SGLang and llama.cpp patterns
are best-effort and report what they cannot find in `missing`.

A config is `rejected` when the global (512-dim) attention backend, the KV capacity or the CUDA-graph sizes are missing,
or when the global layers run on TRITON_ATTN (PLAN.md §4 stop rule).

  python3 bench/startup_facts.py LOG --engine vllm [--run-id R --config-id C --year 2026 --out F]
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import re
import sys

TS = re.compile(r"(?:INFO|WARNING|ERROR|DEBUG)\s+(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)")


def ts_of(line, year):
    m = TS.search(line)
    if not m:
        return None
    mo, d, h, mi, s = map(int, m.groups())
    return datetime.datetime(year, mo, d, h, mi, s, tzinfo=datetime.timezone.utc).timestamp()


def iso(t):
    return None if t is None else datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def num(s):
    return float(s.replace(",", ""))


def parse_vllm(lines, year):
    f = dict(engine="vllm", attention_backend={"sliding": None, "global_512": None}, attention_backend_lines=[], cuda_graph_sizes=None,
             kv_cache_tokens=None, kv_cache_gib=None, kv_cache_dtype=None, max_concurrency_at_max_len=None, max_model_len=None,
             quantization=None, spec_decode=None, engine_version=None)
    ph = {}
    backend_main = None
    explicit = False
    for ln in lines:
        if "version " in ln and f["engine_version"] is None:
            m = re.search(r"version (\d+\.\d+\.\d+\S*)", ln)
            if m:
                f["engine_version"] = m.group(1)
        if "Initializing a V1 LLM engine" in ln:
            m = re.search(r"\(v([\d.]+\S*)\) with config", ln)
            if m:
                f["engine_version"] = m.group(1)
            for key, rx in (("max_model_len", r"max_seq_len=(\d+)"), ("quant", r"quantization=([\w-]+)"),
                            ("kvd", r"kv_cache_dtype=([\w]+)"), ("spec", r"speculative_config=(None|\{.*?\})(?:, tokenizer=)")):
                m = re.search(rx, ln)
                if m:
                    if key == "max_model_len":
                        f["max_model_len"] = int(m.group(1))
                    elif key == "quant":
                        f["quantization"] = dict(weights=m.group(1), detail=None)
                    elif key == "kvd":
                        f["kv_cache_dtype"] = m.group(1)
                    elif key == "spec" and m.group(1) != "None":
                        f["spec_decode"] = dict(raw=m.group(1)[:400])
            m = re.search(r"'cudagraph_capture_sizes': (\[[\d, ]*\])", ln)
            if m:
                f["cuda_graph_sizes"] = ast.literal_eval(m.group(1))
            ph["engine_init_t"] = ts_of(ln, year)
        if "non-default args" in ln:
            m = re.search(r"'speculative_config': (\{.*?\})", ln)
            if m:
                try:
                    sc = ast.literal_eval(m.group(1))
                    f["spec_decode"] = dict(method=sc.get("method", "draft_model"), draft_model=sc.get("model"),
                                            num_speculative_tokens=sc.get("num_speculative_tokens"))
                except (ValueError, SyntaxError):
                    pass
            m = re.search(r"'max_model_len': (\d+)", ln)
            if m and f["max_model_len"] is None:
                f["max_model_len"] = int(m.group(1))
            m = re.search(r"'kv_cache_dtype': '(\w+)'", ln)
            if m:
                f["kv_cache_dtype"] = m.group(1)
            ph["api_start_t"] = ph.get("api_start_t") or ts_of(ln, year)
        if re.search(r"attention backend|heterogeneous head dim|FlashAttention version|head_size=512|TRITON_ATTN|FLASHINFER|AttentionBackendEnum|selected via --attention-backend", ln):
            f["attention_backend_lines"].append(ln.strip()[:300])
            m = re.search(r"Using (\w+) attention backend", ln) or re.search(r"Using (\w+) backend \(selected via --attention-backend\)", ln)
            if m:
                backend_main = m.group(1)
                if "selected via --attention-backend" in ln:
                    explicit = True
            m = re.search(r"Using AttentionBackendEnum\.(\w+)", ln)
            if m:
                backend_main = backend_main or m.group(1)
        if "Loading weights took" in ln:
            m = re.search(r"Loading weights took ([\d.]+) seconds", ln)
            ph["weights_load_s"] = float(m.group(1)) if m else None
            ph["weights_load_end_t"] = ts_of(ln, year)
        if "Starting torch.compile" in ln:
            ph["compile_start_t"] = ts_of(ln, year)
        if "torch.compile took" in ln:
            m = re.search(r"torch.compile took ([\d.]+) s", ln)
            ph["compile_s"] = float(m.group(1)) if m else None
            ph["compile_end_t"] = ts_of(ln, year)
        if "Graph capturing finished in" in ln:
            m = re.search(r"finished in ([\d.]+) secs?, took ([\d.]+) GiB", ln)
            if m:
                ph.setdefault("graph_captures", []).append(dict(s=float(m.group(1)), gib=float(m.group(2)), end_t=ts_of(ln, year)))
        if "Available KV cache memory" in ln:
            m = re.search(r"Available KV cache memory: ([\d.]+) GiB", ln)
            f["kv_cache_gib"] = float(m.group(1)) if m else None
        if "GPU KV cache size" in ln:
            m = re.search(r"GPU KV cache size: ([\d,]+) tokens", ln)
            f["kv_cache_tokens"] = int(num(m.group(1))) if m else None
            m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", ln)
            if m:
                f["max_concurrency_at_max_len"] = float(m.group(2))
        if "init engine" in ln and "took" in ln:
            ph["init_engine_end_t"] = ts_of(ln, year)
        if "Starting vLLM server" in ln or "Application startup complete" in ln:
            ph["server_ready_t"] = ph.get("server_ready_t") or ts_of(ln, year)
    # per-layer-type attention: vLLM 0.33 logs the main backend, then FA3 -> FA4 for head_size=512 on SM90.
    txt = "\n".join(f["attention_backend_lines"])
    if backend_main:
        f["attention_backend"]["sliding"] = backend_main + ("/FA3" if "Using FlashAttention version 3" in txt else "")
        if "head_size=512" in txt and "upgrading FlashAttention 3 -> 4" in txt:
            f["attention_backend"]["global_512"] = backend_main + "/FA4"
        elif "per-layer" in txt and "TRITON_ATTN" in txt and "global" in txt.lower():
            f["attention_backend"]["global_512"] = "TRITON_ATTN"
        elif explicit or backend_main in ("TRITON_ATTN", "FLASHINFER", "ROCM_AITER_UNIFIED_ATTN", "ROCM_ATTN", "ROCM_AITER_FA") or "FlashAttention version 4" in txt:
            f["attention_backend"]["global_512"] = backend_main + ("/FA4" if "FlashAttention version 4" in txt and backend_main == "FLASH_ATTN" else "")
    return f, ph


def parse_sglang(lines, year):
    f = dict(engine="sglang", attention_backend={"sliding": None, "global_512": None}, attention_backend_lines=[], cuda_graph_sizes=None,
             kv_cache_tokens=None, kv_cache_gib=None, kv_cache_dtype=None, max_concurrency_at_max_len=None, max_model_len=None,
             quantization=None, spec_decode=None, engine_version=None)
    for ln in lines:
        # ServerArgs prints attention_backend=, prefill_attention_backend= and decode_attention_backend=; the prefill one
        # (else the general one) is what serves the 512-dim global layers in prefill. A None means "auto": keep looking.
        for key in ("prefill_attention_backend", "attention_backend"):
            # ServerArgs repr (key='x') or, from 0.5.2x, a dict ('key': 'x')
            m = re.search(r"(?<![a-z_])" + key + r"""['"]?\s*[:=]\s*['"]?(\w+)""", ln)
            if m and m.group(1) != "None" and (key == "prefill_attention_backend" or not f.get("_prefill_set")):
                f["attention_backend"]["sliding"] = f["attention_backend"]["global_512"] = m.group(1)
                f["attention_backend_lines"].append(ln.strip()[:300])
                f["_prefill_set"] = f.get("_prefill_set") or key == "prefill_attention_backend"
        m = re.search(r"[Uu]s(?:e|ing) (\w+) (?:as the )?(?:attention )?backend", ln)
        if m and not f["attention_backend"]["global_512"]:
            f["attention_backend"]["sliding"] = f["attention_backend"]["global_512"] = m.group(1)
            f["attention_backend_lines"].append(ln.strip()[:300])
        m = re.search(r"max_total_num_tokens=(\d+)", ln)
        if m:
            f["kv_cache_tokens"] = int(m.group(1))
        m = re.search(r"cuda graph bs \[?([\d, ]+)\]?", ln, flags=re.I) or re.search(r"decode CUDA graph end\. bs=\[([\d, ]+)\]", ln)
        if m:
            f["cuda_graph_sizes"] = [int(x) for x in re.findall(r"\d+", m.group(1))]
            f["cuda_graph_basis"] = "decode"
        # speculative decoding captures no decode graph: the TARGET model's verify graph is the one decode steps replay
        # ("Capture target verify CUDA graph end. bs=[...]"); the draft's own verify graph is not the batch-size list
        m = re.search(r"target verify CUDA graph end\. bs=\[([\d, ]+)\]", ln)
        if m and f.get("cuda_graph_basis") != "decode":
            f["cuda_graph_sizes"] = [int(x) for x in re.findall(r"\d+", m.group(1))]
            f["cuda_graph_basis"] = "target_verify"
        if "Disable prefill CUDA graph" in ln or "disabling prefill CUDA graph" in ln:
            f["prefill_cuda_graph"] = False
        m = re.search(r"full_layer_tokens=(\d+), swa_layer_tokens=(\d+)", ln)
        if m:
            f["kv_pool_tokens"] = dict(full_layers=int(m.group(1)), sliding_layers=int(m.group(2)))
        m = re.search(r"quant=(\w+), quant_algo=(\w+)", ln)
        if m:
            f["quantization"] = f"{m.group(1)}/{m.group(2)}"
        m = re.search(r"context_len=(\d+)", ln)
        if m:
            f["max_model_len"] = int(m.group(1))
        m = re.search(r"""(?<![a-z_])kv_cache_dtype['"]?\s*[:=]\s*['"]?([\w]+)""", ln)
        if m and not f["kv_cache_dtype"]:
            f["kv_cache_dtype"] = m.group(1)
        m = re.search(r"sglang(?:==| version[: ]+|-)v?(\d+\.\d+\.\d+\w*)", ln)
        if m and not f["engine_version"]:
            f["engine_version"] = m.group(1)
    f.pop("_prefill_set", None)
    return f, {}


def parse_llamacpp(lines, year):
    f = dict(engine="llama.cpp", attention_backend={"sliding": None, "global_512": None}, attention_backend_lines=[], cuda_graph_sizes=None,
             kv_cache_tokens=None, kv_cache_gib=None, kv_cache_dtype=None, max_concurrency_at_max_len=None, max_model_len=None,
             quantization=None, spec_decode=None, engine_version=None)
    for ln in lines:
        if "flash_attn" in ln or "flash attention" in ln.lower():
            f["attention_backend_lines"].append(ln.strip()[:300])
            if "= 1" in ln or "enabled" in ln:
                f["attention_backend"]["sliding"] = f["attention_backend"]["global_512"] = "llama.cpp-FA"
        m = re.search(r"n_ctx\s*=\s*(\d+)", ln)
        if m:
            f["kv_cache_tokens"] = int(m.group(1))
        m = re.search(r"build: (\d+)", ln)
        if m:
            f["engine_version"] = "b" + m.group(1)
        if "CUDA graph" in ln or "cuda graphs" in ln.lower():
            f["cuda_graph_sizes"] = f["cuda_graph_sizes"] or ["llama.cpp-auto"]
    return f, {}


def parse(log_text, engine="vllm", year=2026, rules=None):
    """rules (config `startup_rules`, model-aware): require_global_512 (default True: Gemma 4's 512-dim global layers must
    report a backend) and triton_ok (default False: Triton on those layers is the Gemma trap). A model without 512-dim heads
    (Inkling 2 Mini: head_dim 128 everywhere, and SGLang's verified MI355X recipe is Triton) sets both off."""
    rules = dict(dict(require_global_512=True, triton_ok=False), **(rules or {}))
    lines = log_text.splitlines()
    f, ph = {"vllm": parse_vllm, "sglang": parse_sglang, "llama.cpp": parse_llamacpp}[engine](lines, year)
    need = ["kv_cache_tokens", "cuda_graph_sizes"]
    missing = [k for k in need if not f.get(k)]
    if rules["require_global_512"] and not f["attention_backend"].get("global_512"):
        missing.append("attention_backend.global_512")
    f["missing"] = missing
    f["startup_rules"] = rules
    tri = (not rules["triton_ok"]) and "TRITON" in str(f["attention_backend"].get("global_512") or "").upper()   # vLLM TRITON_ATTN, SGLang 'triton'
    f["rejected"] = bool(missing) or tri
    f["reject_reason"] = ("Triton attention on global 512-dim layers" if tri
                          else ("missing " + ", ".join(missing)) if missing else None)
    return f, ph


def phases_from(ph):
    """Sub-phases of engine_start with absolute times, for the ledger (nested_in = engine_start)."""
    out = []
    if ph.get("weights_load_end_t") and ph.get("weights_load_s"):
        out.append(dict(phase="weight_load", t_start=ph["weights_load_end_t"] - ph["weights_load_s"], t_end=ph["weights_load_end_t"]))
    if ph.get("compile_start_t") and ph.get("compile_end_t"):
        out.append(dict(phase="torch_compile", t_start=ph["compile_start_t"], t_end=ph["compile_end_t"], detail=dict(compile_s=ph.get("compile_s"))))
    for g in ph.get("graph_captures", []):
        if g["end_t"]:
            out.append(dict(phase="graph_capture", t_start=g["end_t"] - g["s"], t_end=g["end_t"], detail=dict(s=g["s"], gib=g["gib"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--engine", default="vllm", choices=["vllm", "sglang", "llama.cpp"])
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--out")
    a = ap.parse_args()
    f, ph = parse(open(a.log, errors="replace").read(), a.engine, a.year)
    f["sub_phases"] = [dict(p, t_start=iso(p["t_start"]), t_end=iso(p["t_end"])) for p in phases_from(ph)]
    s = json.dumps(f, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
