#!/usr/bin/env python3
"""Price bar (bench/pricebar.py) and model-aware startup rules (bench/startup_facts.py).
  python3 -I tests/test_pricebar.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import pricebar as PB  # noqa: E402
import startup_facts as SF  # noqa: E402

bad = []
g = PB.price_bar({})
if not (g["is_openrelay_gemma"] and abs(PB.usd_per_req(g) - 0.000779) < 1e-9):
    bad.append(f"default bar {g} -> {PB.usd_per_req(g)}")
ik = PB.price_bar({"price_bar": {"name": "OpenRouter PrimeIntellect thinkingmachines/inkling-2-mini", "in_usd_per_m": 0.075,
                                  "out_usd_per_m": 0.30, "cached_usd_per_m": 0.015, "source": "research/raw/x.json"}})
if ik["is_openrelay_gemma"] or abs(PB.usd_per_req(ik) - 0.0008265) > 1e-9:
    bad.append(f"inkling bar -> {PB.usd_per_req(ik)}")
# phase 3 (GLM): a config-level price_model picks that OpenRelay model's sync price from the same price file
glm = PB.price_bar({"price_model": "openrelay/glm-5.3-flash"})
if glm["is_openrelay_gemma"] or glm["model"] != "openrelay/glm-5.3-flash" or abs(PB.usd_per_req(glm) - 0.0004959) > 1e-10:
    bad.append(f"GLM bar {glm} -> {PB.usd_per_req(glm)} (want 0.045*9500/1e6 + 0.18*380/1e6 = 0.0004959)")
gt = PB.bar_tiers({"price_model": "openrelay/glm-5.3-flash"})
if gt["is_openrelay_gemma"] or gt["sync"]["in_usd_per_m"] != 0.045:
    bad.append(f"GLM tiers {gt}")
if not PB.bar_tiers({})["is_openrelay_gemma"] or PB.bar_tiers({"price_bar": ik})["is_openrelay_gemma"]:
    bad.append("bar_tiers: only the default bar is OpenRelay Gemma")
for cfg, exc in (({"price_model": "openrelay/glm-5.3-flash", "price_bar": {"name": "x", "in_usd_per_m": 1, "out_usd_per_m": 1}}, ValueError),
                 ({"price_model": "openrelay/no-such-model"}, KeyError)):
    try:
        PB.price_bar(cfg)
        bad.append(f"{cfg} must raise {exc.__name__}")
    except exc:
        pass
log = """[2026-11-22 07:30:00] server_args={'attention_backend': 'triton', 'kv_cache_dtype': 'fp8_e4m3'}
[2026-11-22 07:30:10] Capture target decode CUDA graph end. bs=[1, 2, 4], elapsed=3 s
[2026-11-22 07:30:11] max_total_num_tokens=900000, chunked_prefill_size=8192, context_len=32768
"""
f, _ = SF.parse(log, "sglang")
if not f["rejected"]:
    bad.append("default rules must reject Triton on the 512-dim layers (Gemma 4)")
f, _ = SF.parse(log, "sglang", rules={"require_global_512": False, "triton_ok": True})
if f["rejected"]:
    bad.append(f"Inkling rules must accept: {f['reject_reason']}")
print("FAIL\n  " + "\n  ".join(bad) if bad else "OK: price bars (Gemma 0.000779, Inkling 0.0008265, GLM 5.3 Flash 0.0004959, both-set refused) and startup rules")
sys.exit(1 if bad else 0)
