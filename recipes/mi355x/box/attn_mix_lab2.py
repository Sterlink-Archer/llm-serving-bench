# attn_mix_lab2.py: second pass of the attention lab for OUR mix (k=3 -> 4 query tokens per verify, 8K chunks), derived from
# OpenRelay's bench/kernels/attn_tune.py and attn_dec3d.py (inference-recipes @6767b4d, Apache-2.0): same call into
# aiter.ops.triton.attention.unified_attention with vLLM's paged fp8 KV layout and fp8 q.
#
# Why a second pass: the first (attn_mix_lab.py) died twice in its attn_2d sweep on a Triton/LLVM "Bad machine code" abort,
# which kills the process; and its timings were single 10-iteration runs (~4% run-to-run noise). Here:
# - every config runs in a worker subprocess; an abort marks that config failed and a fresh worker resumes after it;
# - each config is timed INTERLEAVED with the table default (3 rounds of 20 calls each, medians), so clock drift cancels;
# - err is max-abs against the default's output (0 = output-identical).
#
# Cases (realistic mix from the k=3/seqs96 knee profile: mixed steps carry ~6.8K prefill tokens + ~35 decoding requests):
#   m40_512 / m96_512  mixed step, d512 global (2D kernel): 8192 - n*4 prefill tokens at context 9000 + n x (4, ~9500)
#   m40_256            mixed step, d256 sliding (2D, window 1024)
#   d40_512 / d96_512  decode-only verify step, d512: n x (4, 9500 + 37 i), split-KV 3D kernel (OpenRelay's ua3d patch)
#   split40            one launch vs prefill alone + decode alone, d512 and d256 (what a split would save), interleaved
# usage: python3 attn_mix_lab2.py [case ...]      (default: all)
import itertools, json, os, statistics as st, subprocess, sys, time

CASES = {
    "m40_512": dict(D=512, ndec=40, kind="mixed"), "m96_512": dict(D=512, ndec=96, kind="mixed"),
    "m40_256": dict(D=256, ndec=40, kind="mixed"),
    "d40_512": dict(D=512, ndec=40, kind="decode"), "d96_512": dict(D=512, ndec=96, kind="decode"),
}
QDEC = 4


def grid(case):
    c = CASES[case]
    if c["kind"] == "mixed" and c["D"] == 512:
        return [{"attn_2d": dict(BLOCK_M=bm, TILE_SIZE=t, num_warps=4, num_stages=s, waves_per_eu=w)}
                for bm, t, s, w in itertools.product([64, 128], [32, 64, 128], [1, 2], [1, 2])]
    if c["kind"] == "mixed":
        return [{"attn_2d": dict(BLOCK_M=128, TILE_SIZE=t, num_warps=4, num_stages=s, waves_per_eu=w)}
                for t, s, w in itertools.product([32, 64, 128], [1, 2], [1, 2])]
    return [{"kv_split": dict(NUM_SEGMENTS=n, TILE_SIZE=t), "attn_3d": dict(num_warps=nw, num_stages=1, waves_per_eu=w)}
            for n, t, nw, w in itertools.product([4, 8, 16, 32], [32, 64, 128], [2, 4], [1, 2])]


# ----------------------------------------------------------------------------------------------------------- worker side
def worker(case, idxs, out):
    import torch, triton
    import aiter.ops.triton.attention.unified_attention as UA
    torch.manual_seed(0)
    fp8 = torch.float8_e4m3fn
    c = CASES.get(case, {})
    orig = UA.get_unified_attention_config
    OV = {}

    def patched(op, params, backend="triton", arch=None):
        cfg = orig(op, params, backend, arch)
        if op in OV:
            cfg.update(OV[op])
        return cfg
    UA.get_unified_attention_config = patched

    def setup(D, seqs):
        HQ = 32; HKV = 4 if D == 512 else 16; BS = 128 if D == 512 else 64
        window = (-1, -1) if D == 512 else (1023, 0)
        qlens = [s[0] for s in seqs]; klens = [s[1] for s in seqs]
        T = sum(qlens); nblk = [triton.cdiv(k, BS) for k in klens]; NB = sum(nblk) + 8
        kv = (torch.randn(NB, HKV, BS, 2 * D, device="cuda") * 0.5).to(fp8)
        kc, vc = kv.transpose(1, 2).split(D, dim=-1)
        bt = torch.zeros(len(seqs), max(nblk), dtype=torch.int32, device="cuda")
        perm = torch.randperm(NB - 8, device="cuda").int(); o = 0
        for i, n in enumerate(nblk):
            bt[i, :n] = perm[o:o + n]; o += n
        q = torch.randn(T, HQ, D, device="cuda").to(fp8)
        cu = torch.tensor([0] + list(itertools.accumulate(qlens)), dtype=torch.int32, device="cuda")
        sk = torch.tensor(klens, dtype=torch.int32, device="cuda")
        one = torch.ones((), device="cuda")
        out = torch.empty(T, HQ, D, device="cuda", dtype=torch.bfloat16)

        def run():
            UA.unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=max(qlens), seqused_k=sk,
                                 max_seqlen_k=max(klens), softmax_scale=D ** -0.5, causal=True, window_size=window,
                                 block_table=bt, softcap=0, q_descale=one, k_descale=one, v_descale=one)
            return out
        return run

    def bench(run, n=20):
        run(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n):
            run()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1000

    def seqs_for(D, ndec, kind):
        dec = [(QDEC, 9500 + 37 * i) for i in range(ndec)]
        return dec if kind == "decode" else [(8192 - ndec * QDEC, 9000)] + dec

    fo = open(out, "a", buffering=1)
    if case == "split40":
        for D in (512, 256):
            dec = [(QDEC, 9500 + 37 * i) for i in range(40)]
            pre = [(8192 - 40 * QDEC, 9000)]
            ra, rp, rd = setup(D, pre + dec), setup(D, pre), setup(D, dec)
            a, p, d = [], [], []
            for _ in range(5):
                a.append(bench(ra)); p.append(bench(rp)); d.append(bench(rd))
            fo.write(json.dumps({"case": "split40", "D": D, "one_launch_ms": st.median(a), "prefill_ms": st.median(p),
                                 "decode_ms": st.median(d), "saves_ms_per_layer": st.median(a) - st.median(p) - st.median(d)}) + "\n")
        return
    run = setup(c["D"], seqs_for(c["D"], c["ndec"], c["kind"]))
    OV.clear(); ref = run().clone()
    g = grid(case)
    for i in idxs:
        fo.write(json.dumps({"start": i}) + "\n"); os.fsync(fo.fileno())
        OV.clear(); OV.update(g[i])
        o = run(); torch.cuda.synchronize()
        err = (o.float() - ref.float()).abs().max().item()
        td, tc = [], []
        for _ in range(3):
            OV.clear(); td.append(bench(run))
            OV.update(g[i]); tc.append(bench(run))
        OV.clear()
        fo.write(json.dumps({"done": i, "cfg": g[i], "default_ms": st.median(td), "cfg_ms": st.median(tc),
                             "speedup": st.median(td) / st.median(tc), "err": err}) + "\n")
        os.fsync(fo.fileno())
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------- orchestrator side
def orchestrate(case, outdir):
    out = os.path.join(outdir, f"lab2_{case}.jsonl")
    if case == "split40":
        subprocess.run([sys.executable, __file__, "--worker", case, "", out], timeout=900)
        return
    n = len(grid(case)); done, failed = set(), set()
    while True:
        todo = [i for i in range(n) if i not in done and i not in failed]
        if not todo:
            break
        t0 = time.time()
        r = subprocess.run([sys.executable, __file__, "--worker", case, ",".join(map(str, todo)), out], timeout=3600,
                           capture_output=True, text=True)
        started = None
        for line in open(out):
            j = json.loads(line)
            if "start" in j:
                started = j["start"]
            elif "done" in j:
                done.add(j["done"]); started = None
        if started is not None and started not in done:
            failed.add(started)
            with open(out, "a") as fo:
                fo.write(json.dumps({"failed": started, "cfg": grid(case)[started], "rc": r.returncode,
                                     "why": (r.stderr.strip().splitlines() or ["?"])[-1][:200]}) + "\n")
        elif r.returncode != 0:
            print(f"{case}: worker rc={r.returncode} before any config; stderr tail: {r.stderr[-400:]}", flush=True)
            break
        print(f"{case}: {len(done)} done, {len(failed)} failed of {n} ({time.time() - t0:.0f} s this worker)", flush=True)


def summary(outdir, cases):
    for case in cases:
        f = os.path.join(outdir, f"lab2_{case}.jsonl")
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f)]
        if case == "split40":
            for r in rows:
                layers = 10 if r["D"] == 512 else 50
                print(f"split40 D={r['D']}: one launch {r['one_launch_ms']:.3f} ms | prefill {r['prefill_ms']:.3f} + decode "
                      f"{r['decode_ms']:.3f} | split saves {r['saves_ms_per_layer']:+.3f} ms/layer = "
                      f"{layers * r['saves_ms_per_layer']:+.2f} ms per mixed step ({layers} layers)")
            continue
        ok = sorted((r for r in rows if "done" in r), key=lambda r: -r["speedup"])
        fl = [r for r in rows if "failed" in r]
        base = st.median(r["default_ms"] for r in ok) if ok else float("nan")
        print(f"{case}: default {base:.3f} ms (median over {len(ok)} interleaved baselines); {len(fl)} configs aborted")
        for r in ok[:6]:
            print(f"   {r['speedup']:5.3f}x  {r['cfg_ms']:.3f} ms vs {r['default_ms']:.3f}  err={r['err']:.4f}  {r['cfg']}")
        for r in fl:
            print(f"   aborted: {r['cfg']} ({r['why'][:90]})")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(sys.argv[2], [int(x) for x in sys.argv[3].split(",") if x], sys.argv[4])
    else:
        cases = sys.argv[1:] or ["split40", "m40_512", "m40_256", "d40_512", "d96_512", "m96_512"]
        outdir = os.environ.get("LAB_OUT", "/root/or/out")
        for case in cases:
            orchestrate(case, outdir)
            summary(outdir, [case])
        print("=== summary")
        summary(outdir, cases)
