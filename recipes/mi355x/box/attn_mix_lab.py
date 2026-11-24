# attn_mix_lab.py: attention cost of OUR mixed steps (k=3 MTP, up to 96 decoding requests, 8192-token chunks), derived from
# OpenRelay's bench/kernels/attn_tune.py (inference-recipes @6767b4d, Apache-2.0): same call into
# aiter.ops.triton.attention.unified_attention with vLLM's paged fp8 KV layout and fp8 q.
#
# Question 1 (split): in a mixed step all sequences share ONE unified_attention launch; with max_seqlen_q > 16 the decode
#   requests (4 query tokens each) ride the prefill-shaped 2D tile (BLOCK_M 128 = 16 tokens x 8 q-heads per KV head for d512).
#   Time: (a) one launch for prefill + decode, (b) prefill alone + decode alone (decode alone takes split-KV 3D via
#   OpenRelay's ua3d dispatch patch, which their image carries). (a) - (b) is what splitting would save per layer.
# Question 2 (retune): sweep attn_2d for the mixed launch (as their attn_tune.py does for the 10K-prefill scenario).
# usage: python3 attn_mix_lab.py [ndec=96] [qdec=4] [quick]
import itertools, sys, time, torch, triton
import aiter.ops.triton.attention.unified_attention as UA

torch.manual_seed(0)
NDEC = int(sys.argv[1]) if len(sys.argv) > 1 else 96
QDEC = int(sys.argv[2]) if len(sys.argv) > 2 else 4
QUICK = len(sys.argv) > 3 and sys.argv[3] == "quick"
fp8 = torch.float8_e4m3fn
PREFILL = (8192 - NDEC * QDEC, 9000)            # one chunk of a long prompt: q_len tokens appended at context k_len
DECODE = [(QDEC, 9500)] * NDEC                  # verify steps at ~9.5K context


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
                             max_seqlen_k=max(klens), softmax_scale=D ** -0.5, causal=True, window_size=window, block_table=bt,
                             softcap=0, q_descale=one, k_descale=one, v_descale=one)
        return out
    return run


def bench(run, n=10):
    run(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        run()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


orig = UA.get_unified_attention_config
override = {}


def patched(op, params, backend="triton", arch=None):
    c = orig(op, params, backend, arch)
    if op == "attn_2d" and override:
        c = dict(override)
    return c


UA.get_unified_attention_config = patched

for D in (512, 256):
    t_all = bench(setup(D, [PREFILL] + DECODE))
    t_pre = bench(setup(D, [PREFILL]))
    t_dec = bench(setup(D, DECODE))
    layers = 10 if D == 512 else 50
    print(f"D={D} mixed step ({PREFILL[0]} prefill tokens + {NDEC} x {QDEC} decode): one launch {t_all:.3f} ms | prefill alone "
          f"{t_pre:.3f} + decode alone {t_dec:.3f} = {t_pre + t_dec:.3f} ms | split saves {t_all - t_pre - t_dec:+.3f} ms/layer "
          f"= {layers * (t_all - t_pre - t_dec):+.2f} ms per mixed step ({layers} layers)", flush=True)

# Question 2: attn_2d sweep for the mixed launch, d512 (their grid; quick = the cells around their pick)
run = setup(512, [PREFILL] + DECODE)
ref = run().clone(); base = bench(run)
grid = dict(BLOCK_M=[64, 128], TILE_SIZE=[32, 64, 128], num_warps=[4, 8], num_stages=[1, 2], waves_per_eu=[1, 2])
if QUICK:
    grid = dict(BLOCK_M=[64, 128], TILE_SIZE=[64, 128], num_warps=[4], num_stages=[1, 2], waves_per_eu=[1, 2])
res = []
for vals in itertools.product(*grid.values()):
    cfg = dict(zip(grid, vals))   # same keys as OpenRelay's attn_tune.py sweep (it ran on this image's aiter)
    override.clear(); override.update(cfg)
    try:
        o = run(); torch.cuda.synchronize()
        err = (o.float() - ref.float()).abs().max().item()
        ms = bench(run); res.append((ms, cfg, err))
    except Exception as e:  # noqa: BLE001
        print(f"   fail {cfg}: {str(e)[:100]}", flush=True)
    torch.cuda.empty_cache()
override.clear()
res.sort(key=lambda r: r[0])
print(f"d512 mixed launch, table default {base:.3f} ms; best attn_2d configs:")
for ms, c, e in res[:6]:
    print(f"  {ms:7.3f} ms {base / ms:5.2f}x err={e:.4f} {c}")
