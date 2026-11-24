# attn_s2_safety.py: is num_stages=2 for the d512 prefill-shaped attention tile (aiter attn_2d, D_GEQ_512.Q_GEQ_256 entry:
# BLOCK_M 128, TILE 64, w4, wpe1) safe to ship in the engine?
#
# attn_mix_lab2.py measured it 1.11x faster than the shipped num_stages=1 on a realistic mixed step (40 decodes + 8K chunk),
# output-identical (err 0). But the same config hit a Triton/LLVM "Bad machine code" abort on the 96-decode shape. An abort in
# the engine kills EngineCore, so a speedup only counts if no shape the engine can launch triggers it. Triton compiles one
# binary per specialization of the integer arguments (divisible by 16, equal to 1) and pointer alignment, so the trigger is a
# property of the launch's integers, not of the config alone.
#
# Part 1 (bisect): the crashing m96 shape vs the passing m40 shape, swapping one factor at a time: number of sequences, block-
#   table width, prefill length.
# Part 2 (engine-like): the engine's block table is a persistent buffer, max_model_len / 128 wide (256 at 32K, 2048 at 256K);
#   sweep the per-step integers (sequences 1..97, chunk tokens divisible by 16 or not, contexts divisible or not).
# Each shape runs in its own subprocess; an abort marks that shape and the orchestrator moves on.
# usage: python3 attn_s2_safety.py           (writes /root/or/out/s2_safety.jsonl, prints a table)
import itertools, json, os, subprocess, sys

S1 = dict(BLOCK_M=128, TILE_SIZE=64, num_warps=4, num_stages=1, waves_per_eu=1)
S2 = dict(BLOCK_M=128, TILE_SIZE=64, num_warps=4, num_stages=2, waves_per_eu=1)


def shapes():
    out = []
    # part 1: (ndec, bt_width, prefill_tokens, prefill_ctx, dec_ctx0); lab2 m40: 40 dec, ctx 9500+37i -> width 86; m96 -> 102
    for nd, w, p in itertools.product([40, 96], [86, 102], [8032, 7808]):
        out.append(dict(part="bisect", ndec=nd, width=w, pre=p, pctx=9000, dctx=9500))
    # part 2: engine-like widths, per-step integers
    for w, nd, p, pctx in itertools.product([256, 2048], [0, 1, 15, 16, 35, 63, 96], [8192, 7001], [9000, 9001]):
        out.append(dict(part="engine", ndec=nd, width=w, pre=p - 4 * nd if p == 8192 else p, pctx=pctx, dctx=9500))
    return out


def worker(i, out):
    import torch, triton
    import aiter.ops.triton.attention.unified_attention as UA
    torch.manual_seed(0)
    s = shapes()[i]
    fp8 = torch.float8_e4m3fn
    D, HQ, HKV, BS = 512, 32, 4, 128
    seqs = [(s["pre"], s["pctx"])] + [(4, s["dctx"] + 37 * j) for j in range(s["ndec"])]
    qlens = [q for q, _ in seqs]; klens = [k for _, k in seqs]
    nblk = [triton.cdiv(k, BS) for k in klens]
    W = max(s["width"], max(nblk)); NB = sum(nblk) + 8
    kv = (torch.randn(NB, HKV, BS, 2 * D, device="cuda") * 0.5).to(fp8)
    kc, vc = kv.transpose(1, 2).split(D, dim=-1)
    bt = torch.zeros(len(seqs), W, dtype=torch.int32, device="cuda")
    perm = torch.randperm(NB - 8, device="cuda").int(); o = 0
    for j, n in enumerate(nblk):
        bt[j, :n] = perm[o:o + n]; o += n
    T = sum(qlens)
    q = torch.randn(T, HQ, D, device="cuda").to(fp8)
    cu = torch.tensor([0] + list(itertools.accumulate(qlens)), dtype=torch.int32, device="cuda")
    sk = torch.tensor(klens, dtype=torch.int32, device="cuda")
    one = torch.ones((), device="cuda")
    orig = UA.get_unified_attention_config
    OV = {}

    def patched(op, params, backend="triton", arch=None):
        c = orig(op, params, backend, arch)
        return dict(OV) if op == "attn_2d" and OV else c
    UA.get_unified_attention_config = patched

    def run():
        out_t = torch.empty(T, HQ, D, device="cuda", dtype=torch.bfloat16)
        UA.unified_attention(q=q, k=kc, v=vc, out=out_t, cu_seqlens_q=cu, max_seqlen_q=max(qlens), seqused_k=sk,
                             max_seqlen_k=max(klens), softmax_scale=D ** -0.5, causal=True, window_size=(-1, -1),
                             block_table=bt, softcap=0, q_descale=one, k_descale=one, v_descale=one)
        torch.cuda.synchronize()
        return out_t
    OV.update(S1); r1 = run()
    with open(out, "a") as fo:
        fo.write(json.dumps({"start": i}) + "\n")
    OV.clear(); OV.update(S2); r2 = run()
    err = (r1.float() - r2.float()).abs().max().item()
    with open(out, "a") as fo:
        fo.write(json.dumps({"ok": i, "shape": s, "T": T, "nseqs": len(seqs), "maxq": max(qlens), "err_vs_s1": err}) + "\n")


def main():
    out = os.environ.get("LAB_OUT", "/root/or/out") + "/s2_safety.jsonl"
    sh = shapes()
    for i in range(len(sh)):
        r = subprocess.run([sys.executable, __file__, "--worker", str(i), out], capture_output=True, text=True, timeout=600)
        rows = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
        if not any(x.get("ok") == i for x in rows):
            started = any(x.get("start") == i for x in rows)
            why = next((l for l in reversed(r.stderr.splitlines()) if "ERROR" in l or "Error" in l), (r.stderr.strip().splitlines() or ["?"])[-1])
            with open(out, "a") as fo:
                fo.write(json.dumps({"fail": i, "shape": sh[i], "stage": "s2" if started else "s1/setup", "rc": r.returncode,
                                     "why": why[:200]}) + "\n")
        last = [json.loads(l) for l in open(out)][-1]
        tag = "OK   " if "ok" in last else "ABORT"
        print(f"{tag} {i:3d} {sh[i]} " + (f"err {last['err_vs_s1']:.4f}" if "ok" in last else f"({last['stage']}: {last['why'][:80]})"), flush=True)
    rows = [json.loads(l) for l in open(out)]
    nok = sum(1 for x in rows if "ok" in x); nf = [x for x in rows if "fail" in x]
    print(f"=== {nok} shapes compiled and ran with num_stages=2, {len(nf)} aborted")
    for x in nf:
        print("   abort:", x["shape"], x["stage"])


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(int(sys.argv[2]), sys.argv[3])
    else:
        main()
