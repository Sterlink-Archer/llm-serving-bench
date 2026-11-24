#!/bin/bash
# serve_ps.sh <engine command...>: start the engine only if patch_ps_fp4.py reported "applied" (the MX_PS_FP4 arm of a
# paired test must never silently run the unpatched GEMM path). Used as the launch prefix of the mx_ps server.
if ! grep -q '^applied' /root/or/mx/patch_ps_fp4.status 2>/dev/null; then
  echo "serve_ps.sh: patch_ps_fp4 not applied: $(cat /root/or/mx/patch_ps_fp4.status 2>/dev/null)" >&2
  exit 1
fi
echo "serve_ps.sh: $(cat /root/or/mx/patch_ps_fp4.status); MX_PS_FP4=${MX_PS_FP4:-unset} MX_PS_MIN_M=${MX_PS_MIN_M:-2048}" >&2
exec "$@"
