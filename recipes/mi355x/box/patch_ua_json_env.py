"""aiter (OpenRelay's image): with MX_UA_JSON=<file>, unified attention reads its config table from <file> instead of the image's
gfx950 DEFAULT.json. Lets two servers on ONE box differ only in the attention table (paired same-box A/B). Unset, nothing changes.

Also writes /root/or/mx/ua_s2.json: the image's DEFAULT.json with one change, the d512 prefill-shaped tile
(attn_2d D_GEQ_512.Q_GEQ_256.DT_fp8_fp8: BLOCK_M 128, TILE 64, w4, wpe1) at num_stages 2 instead of 1. Same tile, deeper
software pipeline: output-identical in box/attn_mix_lab2.py (err 0) and 1.11x on our mixed step's d512 launch.

Idempotent; asserts its anchors; writes a PATCH status line."""
import importlib.util
import json
import os
import sys

a = os.path.dirname(importlib.util.find_spec("aiter").origin)
p = os.path.join(a, "ops/triton/utils/unified_attention_utils.py")
tbl = os.path.join(a, "ops/triton/configs/gfx950/triton/attention/unified_attention/DEFAULT.json")
status = "/root/or/mx/patch_ua_json_env.status"
s = open(p).read()
old = '    config = load_config_json(f"{cfg_dir}/DEFAULT.json", required=False)\n'
new = '    config = load_config_json(__import__("os").environ.get("MX_UA_JSON") or f"{cfg_dir}/DEFAULT.json", required=False)\n'
if new in s:
    msg = "PATCH ua_json_env: already applied"
elif s.count(old) != 1:
    msg = f"PATCH ua_json_env: ANCHOR NOT FOUND ({s.count(old)}) (not applied)"
else:
    s = s.replace(old, new, 1)
    compile(s, p, "exec")
    open(p, "w").write(s)
    msg = "PATCH ua_json_env: applied (active with MX_UA_JSON=<file>)"
d = json.load(open(tbl))
key = "D_GEQ_512.Q_GEQ_256.DT_fp8_fp8"
e = d["attn_2d"][key]
assert (e["BLOCK_M"], e["num_warps"], e["waves_per_eu"], e.get("TILE_SIZE_MIN"), e.get("TILE_SIZE_MAX")) == (128, 4, 1, 64, 64), e
e["num_stages"] = 2
os.makedirs("/root/or/mx", exist_ok=True)
json.dump(d, open("/root/or/mx/ua_s2.json", "w"), indent=2)
msg += f" | ua_s2.json: attn_2d {key} num_stages 1 -> 2"
open(status, "w").write(msg + "\n")
print(msg, file=sys.stderr)
