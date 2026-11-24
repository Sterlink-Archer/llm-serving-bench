"""OpenRelay's split-KV attention for head_dim 512 on short-query steps (their patch_aiter_ua3d.py + the two table
entries of ROCm/aiter#5929, which is NOT merged upstream), ported to the AITER in vLLM 0.34 (v0.1.28.post1).

Stock AITER forces the 2D kernel for head_size >= 512 whenever a step has more than one query token, so an MTP
verify step (3 query tokens per sequence) launches num_seqs x 4 KV heads programs, each walking ~9.5K tokens of
context, on 256 CUs. The patch keeps 2D for prefill (max_seqlen_q > 16) and lets AITER pick split-KV 3D otherwise;
the table entries are the ones #5929 retuned for 3-query steps (with the old ones, 48 sequences regresses).
The 0.1.28 dispatch clause is byte-identical to the one OpenRelay patched. Idempotent; writes a PATCH status line."""
import importlib.util
import json
import os
import sys

a = os.path.dirname(importlib.util.find_spec("aiter").origin)
msgs = []
p = os.path.join(a, "ops/triton/attention/unified_attention.py")
s = open(p).read()
old = "    if params.head_size >= 512 and not get_arch().is_rdna and not params.all_decode:\n"
new = "    if params.head_size >= 512 and not get_arch().is_rdna and not params.all_decode and params.max_seqlen_q > 16:\n"
if new in s:
    msgs.append("dispatch already applied")
elif old in s:
    open(p, "w").write(s.replace(old, new, 1))
    msgs.append("dispatch applied")
else:
    msgs.append("dispatch ANCHOR NOT FOUND")
t = os.path.join(a, "ops/triton/configs/gfx950/triton/attention/unified_attention/DEFAULT.json")
d = json.load(open(t))
before = (d.get("attn_3d", {}).get("D_GEQ_512"), d.get("kv_split", {}).get("D_GEQ_512.DT_fp8_fp8"))
d.setdefault("attn_3d", {})["D_GEQ_512"] = {"BLOCK_M": 32, "num_warps": 2, "num_stages": 1, "waves_per_eu": 1}
kv = dict(d.setdefault("kv_split", {}).get("D_GEQ_512.DT_fp8_fp8") or {"MAX_SEGMENTS": 128, "SEGMENT_TILE_MAX": 64})
kv.update({"MIN_SEGMENTS": 4, "SEGMENTS_PER_CU": 4, "TILE_SIZE_MIN": 64, "TILE_SIZE_MAX": 64})
d["kv_split"]["D_GEQ_512.DT_fp8_fp8"] = kv
json.dump(d, open(t, "w"), indent=2)
msgs.append(f"table attn_3d/kv_split D_GEQ_512 set (was {before})")
os.makedirs("/root/or/mx", exist_ok=True)
msg = "PATCH ua3d: " + "; ".join(msgs)
open("/root/or/mx/patch_ua3d.status", "w").write(msg + "\n")
print(msg, file=sys.stderr)
