#!/bin/bash
# inspect.sh (GLM): engine/library versions and GLM-5.3-Flash support markers, before any engine starts (setup_cmd).
OUT=/root/or/out/inspect.txt; mkdir -p /root/or/out
{
echo "== date $(date -u +%FT%TZ)"
python3 - <<'PY'
import importlib.metadata as md, importlib.util, os
for p in ("vllm", "sglang", "torch", "triton", "amd-aiter", "transformers", "tilelang"):
    try: print(f"version {p}: {md.version(p)}")
    except Exception as e: print(f"version {p}: - ({e.__class__.__name__})")
for mod, rel in (("vllm", "model_executor/models/glm5next.py"), ("vllm", "models/glm5next.py"), ("sglang", "srt/models/glm5_next.py")):
    s = importlib.util.find_spec(mod)
    if s:
        f = os.path.join(os.path.dirname(s.origin), rel)
        print(f"{mod}:{rel} exists={os.path.exists(f)}")
PY
amd-smi static --asic 2>/dev/null | grep -E "MARKET_NAME|TARGET_GRAPHICS" | head -3
} > "$OUT" 2>&1
tail -n 20 "$OUT"
