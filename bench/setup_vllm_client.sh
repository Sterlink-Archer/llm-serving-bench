#!/bin/bash
# OpenRelay's client is `vllm bench serve` (their bench.sh). Engine images without vLLM (SGLang) get it in a separate
# venv at /root/vb, client side only (it never touches the GPU). Pinned to the version our vLLM runs use.
set -euo pipefail
V=/root/vb
VER=${VLLM_CLIENT_VERSION:-0.33.0}
if [ -x "$V/bin/vllm" ]; then "$V/bin/vllm" --version; exit 0; fi
t0=$(date +%s)
if ! command -v uv >/dev/null 2>&1; then python3 -m pip install -q uv >/dev/null 2>&1 || true; fi
if command -v uv >/dev/null 2>&1; then
  uv venv -q "$V" --python python3
  VIRTUAL_ENV="$V" uv pip install -q "vllm==$VER"
else
  python3 -m venv "$V"
  "$V/bin/pip" install -q "vllm==$VER"
fi
echo "[setup_vllm_client] vllm==$VER client in $V after $(( $(date +%s) - t0 )) s: $("$V/bin/vllm" --version 2>/dev/null | tail -1)"
