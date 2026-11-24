# ktrace_moe.py <trace.json.gz> [topn]: GPU kernel time by category for GLM-5.3-Flash (MoE + KDA linear attention + sparse MLA),
# companion to OpenRelay's ktrace.py (whose categories are Gemma 4's). Categories are regexes over kernel names, first match wins.
import gzip, json, sys, re, collections
d = json.load(gzip.open(sys.argv[1]))
ev = [e for e in d["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
cats = [("moe_fused(mxfp4)", r"moe|fmoe|flydsl|topk_softmax|expert|grouped"), ("kda_linear_attn", r"kda|chunk_|gated_delta|fused_recurrent|conv1d|short_conv|cumsum|wy_"),
        ("sparse_mla/indexer", r"mla|sparse|indexer|topk|kpool|paged|attn|attention|fmha|flash"), ("gemm_fp8_block", r"a8w8|fp8|blockscale|gemm_a8"),
        ("gemm_fp4", r"f4gemm|a4w4|fp4"), ("gemm_bf16/other", r"Cijk|gemm|wvSplitK|mm_"), ("quant", r"quant"), ("norm", r"rmsnorm|norm"),
        ("act", r"silu|gelu|act_"), ("kvcache", r"cache")]
tot = collections.Counter(); byname = collections.Counter(); cnt = collections.Counter()
for e in ev:
    n = e["name"]; c = "other"
    for k, p in cats:
        if re.search(p, n, re.I): c = k; break
    tot[c] += e["dur"]; byname[n[:110]] += e["dur"]; cnt[n[:110]] += 1
T = sum(tot.values())
print(f"total GPU kernel time {T/1000:.1f} ms")
for k, v in tot.most_common(): print(f"  {k:22} {v/1000:9.1f} ms {v/T*100:5.1f}%")
print("top kernels:")
for n, v in byname.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 30):
    print(f"  {v/1000:8.1f} ms {v/T*100:5.1f}% x{cnt[n]:<5} {n}")
