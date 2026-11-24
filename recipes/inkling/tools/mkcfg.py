#!/usr/bin/env python3
"""Generate the Inkling-2-Mini (phase 2) run configs for run.py. One function per wave; every config lands in
recipes/inkling/configs/<label>.json. Labels start with `ik` so phase-2 spend can be summed from infra/ledger.jsonl.

  python3 -I recipes/inkling/tools/mkcfg.py S1        # stage-1 screen, one box per (GPU, engine)
  python3 -I run.py recipes/inkling/configs/<label>.json --dry-run

The price bar is OpenRouter's cheapest provider for thinkingmachines/inkling-2-mini (PrimeIntellect, fp4):
$0.075 / 1M input, $0.30 / 1M output, $0.015 cached (research/raw/phase2_or_endpoints_thinkingmachines_inkling-2-mini_
20261122T070046Z.json). At OpenRelay's agentic mix (9.5K uncached in / 380 out) that is $0.0008265 per request. run.py's
own break-even uses OpenRelay's Gemma price, so every protocol step here passes `breakeven_rps` and `rates` explicitly.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "infra"))
import daytona as dt  # noqa: E402

OUT = ROOT / "recipes" / "inkling" / "configs"
REPO, REV = "thinkingmachines/Inkling-2-Mini", "b85c3cfa615fc413cc9af3379f491b963f3b586a"
WDIR = "/root/models/inkling"
SERVED = "inkling"
BAR = dict(name="OpenRouter cheapest: PrimeIntellect thinkingmachines/inkling-2-mini (fp4)", in_usd_per_m=0.075,
           out_usd_per_m=0.30, cached_usd_per_m=0.015,
           source="research/raw/phase2_or_endpoints_thinkingmachines_inkling-2-mini_20261122T070046Z.json")
BAR_USD_REQ = (9500 * BAR["in_usd_per_m"] + 380 * BAR["out_usd_per_m"]) / 1e6

IMAGES = {   # Docker Hub, read 2026-11-22T07:1xZ (tag pushed 2026-11-20); amd64 digests
    "sglang": ("lmsysorg/sglang:inkling-2-mini", "sha256:421aed3ac11741381eac350bf7b1db28a0cafb6ae49b57ae78fba86a16a32306"),
    "sglang_amd": ("lmsysorg/sglang:inkling-2-mini-mi35x", "sha256:f53f969460964a88a203b41c8ac52efcd1cb99954f7d49c57489a3b4d761228a"),
    "vllm": ("vllm/vllm-openai:inkling-2-mini", "sha256:a485440c62099c39659a7b723636073ed726dcb41f94cf6326e16af3a1d6f6bc"),
}
GPU_SHORT = {"RTX-5090": "5090", "RTX-PRO-6000": "pro6000", "H100": "h100", "H200": "h200", "MI355X": "mi355x", "B300": "b300"}

# Inkling has head_dim 128 everywhere (20 SWA-1024 layers, 4 full-attention layers) and SGLang's own MI355X recipe uses
# Triton attention, so the Gemma-specific "global 512-dim / Triton" startup rejection does not apply (build 48ed68a).
STARTUP_RULES = {"require_global_512": False, "triton_ok": True}
# Reasoning setting for every eval: stated, not left to the template. "high" = 0.9 = the chat template's default.
EFFORT = "high"


def snapshot(gpu, engine, cpu=8, mem=32, disk=150):
    img, dig = IMAGES[engine]
    tag = img.split(":")[1].replace("inkling-2-mini", "ik2m").replace(".", "")
    return dict(name=f"or-{engine.split('_')[0]}-{tag}-{GPU_SHORT[gpu]}-c{cpu}m{mem}d{disk}", image=img, image_digest_amd64=dig,
                gpu=gpu, gpus=1, cpu=cpu, mem=mem, disk=disk)


def breakeven(snap):
    pph = dt.hourly_usd(snap["gpu"], snap.get("gpus", 1), snap["cpu"], snap["mem"], snap["disk"])["total"]
    return pph / 3600 / BAR_USD_REQ, pph


def sglang_launch(gpu, extra=""):
    base = (f"python3 -m sglang.launch_server --model-path {WDIR} --served-model-name {SERVED} --host 127.0.0.1 --port 8000 "
            f"--tp 1 --ep 1 --dtype bfloat16 --page-size 128 --context-length 32768 --reasoning-parser inkling "
            f"--enable-metrics --enable-cache-report")
    if gpu == "MI355X":
        base += " --attention-backend triton"
    elif gpu in ("H100", "H200", "B300", "B200"):
        base += " --attention-backend fa4 --bf16-gemm-backend torch"   # SGLang's H200 / GB300 Balanced recipe flags
    else:
        base += " --attention-backend fa4"                             # RTX PRO 6000 / RTX 5090 Balanced recipe
    return (base + " " + extra).strip()


def vllm_launch(extra=""):
    return (f"vllm serve {WDIR} --served-model-name {SERVED} --host 127.0.0.1 --port 8000 --max-model-len 32768 "
            f"--reasoning-parser inkling --enable-prompt-tokens-details --limit-mm-per-prompt '{{\"image\":0}}' " + extra).strip()


def steps(srv, be, mults, seeds=(7,), dur=120, gate_n=100, cols=("A", "B"), prefill=True, client_setup=True, profile=False,
          b_prefill=False, a_mults=None):
    """Steps for one server. Column A = OpenRelay's client (random tokens, their prefill probe); column B = real text
    (the claim). a_mults overrides the A ladder (e.g. one rung, to pair A and B on the same server)."""
    rates = [round(m * be, 3) for m in mults]
    st = []
    if client_setup and "A" in cols:
        st.append(dict(type="sh", name="setup_vllm_client", cmd="bash bench/setup_vllm_client.sh", est_s=300))
    # Gate. Inkling's chat template ignores enable_thinking; reasoning effort is stated explicitly (EFFORT). GSM8K
    # (OpenRelay's prompt and '####' scoring, test[:N]). Published: 93.6-94.2% on all 1319 (SGLang cookbook, 6 recipes),
    # so the screen bar is 0.88 at N=100 (~2.5 sigma under 0.94); the claim gate is reference-relative on test[:250].
    st.append(dict(type="eval", name=f"gate_{srv}_gsm8kor{gate_n}", sets="data/eval/gsm8kor_test.jsonl", limit=gate_n, conc=64,
                   max_tokens=16384, chat_template_kwargs={"reasoning_effort": EFFORT}, gate=dict(task="gsm8kor", threshold=0.88), est_s=600))
    if profile:
        st.append(dict(type="sh", name=f"profile_prefill_{srv}", est_s=300, outputs=[f"out/profile_prefill_{srv}.json"],
                       cmd=f"python3 bench/profile_prefill.py --model {SERVED} --dir /root/or/out/prof_{srv} --n 4 --tokens 10000 "
                           f"--out /root/or/out/profile_prefill_{srv}.json"))
    if "B" in cols:   # the claim column first: a time cap then cuts A, never B
        st.append(dict(type="protocol", name=f"protocol_{srv}_B_text", client="traffic", breakeven_rps=round(be, 4), rates=rates,
                       seeds=list(seeds), duration=dur, prefill_shot=b_prefill))
    if "A" in cols:
        ar = [round(m * be, 3) for m in (a_mults or mults)]
        st.append(dict(type="protocol", name=f"protocol_{srv}_A_random", client="vllmbench", vllm_bin="/root/vb/bin/vllm",
                       breakeven_rps=round(be, 4), rates=ar, seeds=list(seeds), duration=dur, prefill_shot=prefill))
    return st


def server(name, launch, st, env=None, fallbacks=None, ready=2400):
    s = dict(name=name, weights=[dict(repo=REPO, rev=REV, dir=WDIR)], served_name=SERVED, launch=launch, env=env or {},
             ready_timeout_s=ready, steps=st, delete_weights_after=False, startup_rules=STARTUP_RULES)
    if fallbacks:
        s["fallbacks"] = fallbacks
    return s


def config(label, snap, servers, note, minutes, budget, stage="ik-screen", claim_group=None):
    be, pph = breakeven(snap)
    return dict(label=label, stage=stage, claim_group=claim_group or label, model=REPO, model_rev=REV, price_bar=BAR,
                price_bar_usd_per_req=round(BAR_USD_REQ, 7), breakeven_rps_vs_bar=round(be, 4), note=note, snapshot=snap,
                max_minutes=minutes, budget_usd=budget, servers=servers)


def write(cfg):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{cfg['label']}.json"
    p.write_text(json.dumps(cfg, indent=1) + "\n")
    be, pph = breakeven(cfg["snapshot"])
    print(f"{p.relative_to(ROOT)}  {cfg['snapshot']['gpu']:<13} be={be:.3f} rps  ${pph:.3f}/h  cap {cfg['max_minutes']} min / ${cfg['budget_usd']}")


# ----------------------------------------------------------------------------- waves
def S1():
    """Stage-1 screen: does each (GPU, engine) serve Inkling correctly, how fast is prefill, where is the knee?
    Vendor 'Balanced' recipe (no speculation), text-only, context 32K, seed 7, 120 s rungs at multiples of each card's
    break-even vs the PrimeIntellect bar, both columns, early stop at p95 TTFT > 8 s."""
    note_common = ("Phase-2 screen, Inkling-2-Mini @b85c3cf (NVFP4 experts L0-19, MXFP8 L20-23, rest BF16). Vendor Balanced "
                   "recipe at TP1, no speculation, text-only (no --enable-multimodal / image limit 0), context 32768. Gate: "
                   "GSM8K test[:100], OpenRelay prompt and '####' scoring, reasoning effort = chat-template default 0.9, "
                   "temperature 0, max_tokens 16384. Then OpenRelay's prefill probe and column A (vllm bench serve, random "
                   "tokens) and column B (traffic.py real text, tokenizer-exact, stratified) at multiples of break-even vs the "
                   "PrimeIntellect bar ($0.0008265/req), seed 7, 120 s rungs (screen only).")
    out = []
    for gpu, mults in (("RTX-5090", [1.0, 1.5, 2.0, 3.0, 4.0]), ("RTX-PRO-6000", [1.0, 1.5, 2.0, 3.0, 4.0]),
                       ("H100", [1.0, 1.5, 2.0, 3.0, 4.0]), ("MI355X", [1.0, 1.5, 2.0, 3.0, 4.0, 5.0])):
        eng = "sglang_amd" if gpu == "MI355X" else "sglang"
        snap = snapshot(gpu, eng)
        be, _ = breakeven(snap)
        srv = "sgl"
        env = {"ROCM_QUICK_REDUCE_QUANTIZATION": "NONE"} if gpu == "MI355X" else {}
        st = steps(srv, be, mults)
        fb = [dict(flags="--attention-backend triton", why="FA4 failed to start on this card")] if gpu != "MI355X" else None
        out.append(config(f"ik1_{GPU_SHORT[gpu]}_sgl", snap, [server(srv, sglang_launch(gpu), st, env=env, fallbacks=fb)],
                          note_common + f" Engine: SGLang {IMAGES[eng][0]} (PR sgl-project/sglang#51926 build).", 100, 9.0))
    for gpu in ("H100", "RTX-PRO-6000"):
        snap = snapshot(gpu, "vllm")
        be, _ = breakeven(snap)
        st = steps("vllm", be, [1.0, 1.5, 2.0, 3.0, 4.0], client_setup=False)
        verified = "verified on H100 by the vLLM recipe" if gpu == "H100" else "NOT in vLLM's verified list (sm_120): a probe"
        out.append(config(f"ik1_{GPU_SHORT[gpu]}_vllm", snap, [server("vllm", vllm_launch(), st)],
                          note_common + f" Engine: vLLM {IMAGES['vllm'][0]} (0.35.0 nightly), {verified}.", 100, 9.0))
    for c in out:
        write(c)


# SGLang flags under test (one change per journal step). The RTX DSpark envs are SGLang's verified RTX recipe's.
DSPARK_ENV = {"SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL": "1", "SGLANG_OPT_DSV41_DSPARK_MID_FOLD": "1",
              "SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE": "1", "SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS": "1"}
SWA = "--swa-full-tokens-ratio 0.15"


def S2():
    """Wave 2: the PRO 6000 levers, paired on the same box (control first), column B only (the claim), 180 s rungs at
    1.0/1.5/2.0/2.5x break-even, seed 7, real-text prefill shot. ik1 found the PRO 6000's SWA KV pool (10% of the full
    pool, 125K tokens) at 100% usage with requests queued while the full pool sat at 63%: step 202 rebalances it
    (ratio 0.15 balances both pools at ~107 concurrent 9.5K requests). The other boxes measure single changes on top."""
    gpu = "RTX-PRO-6000"
    snap = snapshot(gpu, "sglang")
    be, _ = breakeven(snap)
    mults = [1.0, 1.5, 2.0, 2.5]
    kw = dict(dur=180, cols=("B",), b_prefill=False)
    base = sglang_launch(gpu)
    note = ("Wave 2 (PRO 6000, SGLang inkling-2-mini image): paired same-box servers, control first; column B (real text, "
            "the claim) only, 180 s rungs at 1.0/1.5/2.0/2.5x break-even vs the PrimeIntellect bar, seed 7, real-text 16x10K "
            "prefill shot; GSM8K test[:100] gate at reasoning effort 'high' (0.9) on every server. ")
    boxes = [
        ("ik2_pro6000_swa", [
            server("base", base, steps("base", be, mults, cols=("B", "A"), a_mults=[1.0], **{k: v for k, v in kw.items() if k != "cols"})),
            server("swa15", sglang_launch(gpu, SWA), steps("swa15", be, mults, **kw))],
         "Control = ik1's vendor Balanced recipe (also one column-A rung + OpenRelay's random-token prefill probe, to pair "
         "the columns on one server: random vs real-text prefill). Variant: --swa-full-tokens-ratio 0.15 (was 0.1)."),
        ("ik2_pro6000_sched", [
            server("swa15", sglang_launch(gpu, SWA), steps("swa15", be, mults, **kw)),
            server("swa15_mixed", sglang_launch(gpu, SWA + " --enable-mixed-chunk"), steps("swa15_mixed", be, mults, **kw)),
            server("swa15_chunk16k", sglang_launch(gpu, SWA + " --chunked-prefill-size 16384"), steps("swa15_chunk16k", be, mults, **kw))],
         "Control swa15; variants: --enable-mixed-chunk (decode rides in prefill steps instead of waiting for them), "
         "--chunked-prefill-size 16384 (was 8192)."),
        ("ik2_pro6000_spec", [
            server("swa15", sglang_launch(gpu, SWA), steps("swa15", be, mults, **kw)),
            server("swa15_dspark", sglang_launch(gpu, SWA + " --speculative-algorithm DSPARK --speculative-num-draft-tokens 8 "
                                                 "--max-running-requests 128"), steps("swa15_dspark", be, mults, **kw), env=DSPARK_ENV,
                   fallbacks=[dict(flags="--max-running-requests 32", why="DSpark refused 128 running requests")])],
         "Control swa15; variants: DSpark speculation (the bundled 3-layer draft head, 8 draft tokens, SGLang's RTX DSpark "
         "envs) with --max-running-requests 128 (the RTX recipe caps it at 4)."),
    ]
    for label, servers, why in boxes:
        write(config(label, snap, servers, note + why, 100, 7.0, stage="ik-promote"))


def S3():
    """Wave 3: the RTX 5090 (best prefill per dollar in ik1: 17.3K tok/s at $1.68/h) is memory-bound. ik1: weights 17.9 GB,
    prefill CUDA graphs 5.5 GB, KV pool 5.4 GB = 118K full tokens + 11.8K SWA tokens (~8 concurrent 9.5K requests), so
    1.5x break-even queued. One change per server, chained on the same box (each server's parent is the one before it):
    BF16 box: swa 0.25 -> + no prefill CUDA graphs. FP8 box: FP8 KV -> + swa 0.25 -> + no prefill CUDA graphs.
    Column B only, 180 s rungs at 1.0-3.0x break-even, seed 7, real-text prefill shot."""
    gpu = "RTX-5090"
    snap = snapshot(gpu, "sglang")
    be, _ = breakeven(snap)
    mults = [1.0, 1.5, 2.0, 2.5, 3.0]
    kw = dict(dur=180, cols=("B",), b_prefill=False)
    S25, NOPG, FP8 = "--swa-full-tokens-ratio 0.25", "--disable-prefill-cuda-graph", "--kv-cache-dtype fp8_e4m3"
    note = ("Wave 3 (RTX 5090, SGLang inkling-2-mini image): chained single changes on one box, column B (real text) only, "
            "180 s rungs at 1.0-3.0x break-even vs the PrimeIntellect bar, seed 7, real-text 16x10K prefill shot, GSM8K "
            "test[:100] gate at reasoning effort 'high' on every server. ")
    boxes = [
        ("ik3_5090_bf16", [
            server("swa25", sglang_launch(gpu, S25), steps("swa25", be, mults, **kw)),
            server("swa25_nopg", sglang_launch(gpu, f"{S25} {NOPG}"), steps("swa25_nopg", be, mults, **kw))],
         "BF16 KV: parent ik1_5090_sgl (vendor Balanced) -> --swa-full-tokens-ratio 0.25 -> + --disable-prefill-cuda-graph."),
        ("ik3_5090_fp8", [
            server("tri_nopg", sglang_launch(gpu, f"--attention-backend triton {NOPG}"), steps("tri_nopg", be, mults, **kw)),
            server("tri_nopg_fp8kv", sglang_launch(gpu, f"--attention-backend triton {NOPG} {FP8}"), steps("tri_nopg_fp8kv", be, mults, **kw)),
            server("tri_nopg_fp8kv_swa25", sglang_launch(gpu, f"--attention-backend triton {NOPG} {FP8} {S25}"),
                   steps("tri_nopg_fp8kv_swa25", be, mults, **kw))],
         "FP8 KV path. FA4 on sm120 refuses an FP8 KV cache (q bf16 != k fp8) and Triton attention cannot replay prefill "
         "CUDA graphs (ik3 attempt 1, 07:55Z), so the chain is: Triton attention + no prefill graphs (BF16 KV) -> + FP8 E4M3 "
         "KV -> + swa 0.25. Parent of the first server: ik1_5090_sgl."),
    ]
    for label, servers, why in boxes:
        write(config(label, snap, servers, note + why, 110, 4.0, stage="ik-promote"))


BOX = ROOT / "recipes" / "inkling" / "box"


def box_setup(files):
    """setup_cmd that writes recipes/inkling/box/<files> to /root/or/ik/ on the sandbox (heredocs: every command that ran
    can be read back from the config)."""
    parts = ["mkdir -p /root/or/ik /root/or/out"]
    for f in files:
        body = (BOX / f).read_text()
        tag = "IKEOF_" + f.upper().replace(".", "_")
        assert tag not in body
        parts.append(f"cat > /root/or/ik/{f} <<'{tag}'\n{body}{tag}")
    return "\n".join(parts)


def S4():
    """Wave 4: MI355X, the only card passing column B in ik1, is compute-bound (27.6K tok/s prefill) and AITER ran its
    default (torch solution 0) BF16 GEMM for all 263 logged Inkling shapes. Paired on one box: base server (inspect, gate,
    prefill profile, column B) -> AITER gradlib tuning of exactly the logged shapes, installed as a model_configs table
    -> tuned server (fresh start, so AITER merges the table) with the same measurements."""
    gpu = "MI355X"
    snap = snapshot(gpu, "sglang_amd")
    be, _ = breakeven(snap)
    mults = [1.0, 1.25, 1.5, 2.0]
    rates = [round(m * be, 3) for m in mults]
    servers = []
    for srv in ("base", "tuned"):
        st = [dict(type="sh", name=f"inspect_{srv}", cmd=f"bash /root/or/ik/aiter_inspect.sh {srv}",
                   outputs=[f"out/aiter_inspect_{srv}.txt"], est_s=60),
              dict(type="eval", name=f"gate_{srv}_gsm8kor100", sets="data/eval/gsm8kor_test.jsonl", limit=100, conc=64,
                   max_tokens=16384, chat_template_kwargs={"reasoning_effort": EFFORT}, gate=dict(task="gsm8kor", threshold=0.88),
                   est_s=600),
              dict(type="sh", name=f"profile_prefill_{srv}", est_s=300, outputs=[f"out/profile_prefill_{srv}.json"],
                   cmd=f"python3 bench/profile_prefill.py --model {SERVED} --dir /root/or/out/prof_{srv} --n 4 --tokens 10000 "
                       f"--out /root/or/out/profile_prefill_{srv}.json"),
              dict(type="protocol", name=f"protocol_{srv}_B_text", client="traffic", breakeven_rps=round(be, 4), rates=rates,
                   seeds=[7], duration=180, prefill_shot=False)]
        if srv == "base":
            st.append(dict(type="sh", name="aiter_tune", est_s=2700,
                           cmd="python3 /root/or/ik/aiter_tune.py --log /root/or/server_base.log --out /root/or/out --timeout 2400",
                           outputs=["out/aiter_untuned_inkling.csv", "out/aiter_tuned_inkling.csv", "out/aiter_tune_summary.json",
                                    "out/aiter_tune.log"]))
        env = {"ROCM_QUICK_REDUCE_QUANTIZATION": "NONE", "SGLANG_TORCH_PROFILER_DIR": f"/root/or/out/prof_{srv}"}
        servers.append(server(srv, sglang_launch(gpu), st, env=env))
    cfg = config("ik4_mi355x_aiter", snap, servers,
                 "Wave 4 (MI355X, SGLang inkling-2-mini-mi35x image): paired same-box AITER BF16 GEMM tuning for Inkling's "
                 "logged shapes (N=8600 K=2304 qkv+rel projection, N=2304 K=3072 output projection; ik1 ran AITER's default "
                 "'torch solution:0' for all 263). Base -> tune (aiter gradlib gemm_tuner.py, box/aiter_tune.py) -> tuned. "
                 "Each server: aiter_inspect (versions, tuner, Inkling2 quant code paths, untuned-shape count), GSM8K "
                 "test[:100] gate at effort 'high', torch-profiler prefill summary (4x10K), column B 180 s rungs at "
                 "1.0/1.25/1.5/2.0x break-even with a real-text 16x10K prefill shot, seed 7.",
                 140, 12.0, stage="ik-kernel")
    cfg["setup_cmd"] = box_setup(["aiter_inspect.sh", "aiter_tune.py"])
    write(cfg)


def S5():
    """Wave 5: DSpark speculation on the PRO 6000, re-run. ik2_pro6000_spec's DSpark server came up healthy (target-verify
    and draft CUDA graphs captured for bs 1-128, 885K KV tokens) but bench/startup_facts.py only reads SGLang's plain
    decode-graph line, so it was rejected for "missing cuda_graph_sizes" (reported to the build session). Paired with the
    vendor recipe as control (ik2: swa 0.15 was neutral)."""
    gpu = "RTX-PRO-6000"
    snap = snapshot(gpu, "sglang")
    be, _ = breakeven(snap)
    mults = [1.0, 1.25, 1.5, 2.0]
    kw = dict(dur=180, cols=("B",), b_prefill=False)
    ds = server("dspark", sglang_launch(gpu, "--speculative-algorithm DSPARK --speculative-num-draft-tokens 8 --max-running-requests 128"),
                steps("dspark", be, mults, **kw), env=DSPARK_ENV)
    ds["allow_rejected"] = True
    ds["allow_rejected_why"] = ("startup_facts misses SGLang's speculative CUDA-graph lines ('Capture target verify CUDA graph "
                                "end. bs=[...]'); graphs ARE captured (ik2_pro6000_spec server_swa15_dspark.log)")
    write(config("ik5_pro6000_dspark", snap, [server("base", sglang_launch(gpu), steps("base", be, mults, **kw)), ds],
                 "Wave 5 (PRO 6000, SGLang inkling-2-mini image): DSpark speculation (bundled 3-layer draft head, 8 draft "
                 "tokens, SGLang's RTX DSpark envs, --max-running-requests 128 instead of the RTX recipe's 4) vs the vendor "
                 "Balanced recipe on the same box. Column B only (random tokens accept almost nothing, per the vLLM recipe), "
                 "180 s rungs at 1.0/1.25/1.5/2.0x break-even, seed 7; GSM8K test[:100] gate at effort 'high' on both.",
                 90, 6.0, stage="ik-promote"))


def S6():
    """Wave 6: the RTX 5090's KV budget. ik3_5090_bf16 (swa 0.25) showed full-pool usage 0.88-0.97 with only 6-10 requests
    running and the queue growing, while per-chunk real-text prefill held ~16.7K tok/s: memory-bound. Disabling prefill
    CUDA graphs alone frees ~5.5 GB but SGLang sizes the pool from --mem-fraction-static (auto 0.789), so the pool barely
    grew (78.8K -> 83.7K full tokens). One memory-budget move: prefill graphs off + static fraction 0.88 + decode graphs
    up to bs 24 (they took 3.5 GB up to bs 48), then + swa 0.2. Fallback if it OOMs at start: static 0.85."""
    gpu = "RTX-5090"
    snap = snapshot(gpu, "sglang")
    be, _ = breakeven(snap)
    mults = [1.0, 1.5, 2.0, 2.5]
    kw = dict(dur=180, cols=("B",), b_prefill=False)
    MEM = "--disable-prefill-cuda-graph --mem-fraction-static 0.88 --cuda-graph-max-bs-decode 24"
    fb = [dict(flags="--mem-fraction-static 0.85", why="0.88 did not leave room for activations")]
    write(config("ik6_5090_mem", snap, [
        server("mem88", sglang_launch(gpu, MEM), steps("mem88", be, mults, **kw), fallbacks=fb),
        server("mem88_swa20", sglang_launch(gpu, MEM + " --swa-full-tokens-ratio 0.2"), steps("mem88_swa20", be, mults, **kw),
               fallbacks=fb)],
        "Wave 6 (RTX 5090, SGLang inkling-2-mini image): KV memory budget. Parent ik1_5090_sgl (vendor Balanced, 118K full "
        "tokens). mem88 = prefill CUDA graphs off + --mem-fraction-static 0.88 (auto 0.789) + --cuda-graph-max-bs-decode 24; then "
        "+ --swa-full-tokens-ratio 0.2. Column B only, 180 s rungs at 1.0/1.5/2.0/2.5x break-even, seed 7, GSM8K test[:100] "
        "gate at effort 'high' on both.", 80, 3.0, stage="ik-promote"))


QUALITY_SETS = "data/eval/gsm8kor_test.jsonl,data/eval/mmlupro_200.jsonl"   # GSM8K test[:250] (limit) + MMLU-Pro 200


def quality_eval(name, gate=True):
    """The quality row (lead, 11-22): GSM8K test[:250] with OpenRelay's prompt and '####' scoring plus MMLU-Pro 200, greedy,
    reasoning effort 'high' (0.9 = template default), max_tokens 16384. Every raw output is saved, so
    tools/agree.py scores it against the reference (accuracy delta, exact McNemar, same answer / same text)."""
    st = dict(type="eval", name=name, sets=QUALITY_SETS, limit=250, conc=64, max_tokens=16384,
              chat_template_kwargs={"reasoning_effort": EFFORT}, est_s=900)
    if gate:
        st["gate"] = dict(task="gsm8kor", threshold=0.90)
    return st


def S7():
    """Wave 7: MI355X AITER BF16 GEMM tuning, second attempt (ik4's tuner JIT-compiled flydsl/triton variants on 8 vCPUs and
    wrote nothing in 26 min). Precompiled backends only (torch baseline, asm, hipBLASLt), the 117 prefill shapes shipped
    as box/inkling2mini_bf16_untuned_gemm.csv, merged lowest-us per shape. Paired on one box: base (profile + column B) ->
    tune -> tuned (fresh start; AITER merges model_configs/inkling2mini_bf16_tuned_gemm.csv). The prefill profile needs
    SGLang's admin key (box/sgl_profile.py adds it to bench/profile_prefill.py's two profiler calls)."""
    gpu = "MI355X"
    snap = snapshot(gpu, "sglang_amd")
    be, _ = breakeven(snap)
    rates = [round(m * be, 3) for m in (1.0, 1.25, 1.5)]
    servers = []
    for srv in ("base", "tuned"):
        st = [dict(type="sh", name=f"inspect_{srv}", cmd=f"bash /root/or/ik/aiter_inspect.sh {srv}",
                   outputs=[f"out/aiter_inspect_{srv}.txt"], est_s=60),
              dict(type="eval", name=f"gate_{srv}_gsm8kor100", sets="data/eval/gsm8kor_test.jsonl", limit=100, conc=64,
                   max_tokens=16384, chat_template_kwargs={"reasoning_effort": EFFORT}, gate=dict(task="gsm8kor", threshold=0.88),
                   est_s=600),
              dict(type="sh", name=f"profile_prefill_{srv}", est_s=300, outputs=[f"out/profile_prefill_{srv}.json"],
                   cmd=f"python3 /root/or/ik/sgl_profile.py --model {SERVED} --dir /root/or/out/prof_{srv} --n 4 --tokens 10000 "
                       f"--out /root/or/out/profile_prefill_{srv}.json"),
              dict(type="protocol", name=f"protocol_{srv}_B_text", client="traffic", breakeven_rps=round(be, 4), rates=rates,
                   seeds=[7], duration=180, prefill_shot=False)]
        if srv == "base":
            st.append(dict(type="sh", name="aiter_tune", est_s=3400,
                           cmd="python3 /root/or/ik/aiter_tune.py --untuned /root/or/ik/inkling2mini_bf16_untuned_gemm.csv --out /root/or/out",
                           outputs=["out/aiter_tuned_inkling.csv", "out/aiter_tune_summary.json", "out/aiter_tuned_*.csv", "out/aiter_tune_*.log"]))
        env = {"ROCM_QUICK_REDUCE_QUANTIZATION": "NONE", "SGLANG_TORCH_PROFILER_DIR": f"/root/or/out/prof_{srv}",
               "IK_ADMIN_KEY": "local-profiler-token"}
        servers.append(server(srv, sglang_launch(gpu, '--admin-api-key "$IK_ADMIN_KEY"'), st, env=env))
    cfg = config("ik7_mi355x_aiter", snap, servers,
                 "Wave 7 (MI355X, SGLang inkling-2-mini-mi35x image): paired same-box AITER BF16 GEMM tuning of the 117 "
                 "prefill shapes (N=8600 K=2304, N=2304 K=3072; ik1/ik4: AITER default 'torch solution:0') with precompiled "
                 "backends only (torch baseline, asm, hipBLASLt; merged lowest-us). Base -> tune -> tuned. Each server: "
                 "aiter_inspect, GSM8K test[:100] gate at effort 'high', torch-profiler prefill summary (4x10K, via the admin "
                 "key: a throwaway local token), column B 180 s rungs at 1.0/1.25/1.5x break-even, seed 7. "
                 "--admin-api-key only unlocks SGLang's admin endpoints (profiler); inference endpoints stay keyless.",
                 150, 13.0, stage="ik-kernel")
    cfg["setup_cmd"] = box_setup(["aiter_inspect.sh", "aiter_tune.py", "sgl_profile.py", "inkling2mini_bf16_untuned_gemm.csv"])
    write(cfg)


def S8():
    """Wave 8: right-sized hosts (4 vCPU / 24 GiB instead of 8 / 32: Daytona bills vCPU and RAM separately, so break-even
    drops 11% on the PRO 6000 and 20% on the 5090) + the partial wins combined.
    PRO 6000: mixed-chunk (ik2: p95 TTFT at 1.5x 15.5 -> 7.2 s) as control, + DSpark (ik5: TPOT 42 -> 19 ms at 1.0x).
    5090: its real memory ceiling (prefill graphs off + static fraction 0.84; ik6 showed 0.88 and 0.85 OOM, ~0.31 GB of KV
    per 0.01) as control, + mixed-chunk. Rates are multiples of the NEW break-even; the PRO 6000 ladder includes 0.97 req/s
    (the old 1.0x) so it compares directly with waves 2-5."""
    kw = dict(dur=180, cols=("B",), b_prefill=False)
    gpu = "RTX-PRO-6000"
    snap = snapshot(gpu, "sglang", cpu=4, mem=24)
    be, _ = breakeven(snap)
    MIX = "--enable-mixed-chunk"
    DS = "--speculative-algorithm DSPARK --speculative-num-draft-tokens 8 --max-running-requests 128"
    ds = server("mixed_dspark", sglang_launch(gpu, f"{MIX} {DS}"), steps("mixed_dspark", be, [1.0, 1.13, 1.25, 1.5], **kw), env=DSPARK_ENV)
    ds["allow_rejected"], ds["allow_rejected_why"] = True, ("startup_facts misses SGLang's speculative CUDA-graph lines; graphs are "
                                                             "captured (ik5_pro6000_dspark server_dspark.log)")
    write(config("ik8_pro6000_c4", snap, [server("mixed", sglang_launch(gpu, MIX), steps("mixed", be, [1.0, 1.13, 1.25, 1.5], **kw)), ds],
                 "Wave 8 (PRO 6000 on a 4 vCPU / 24 GiB host, $2.556/h, break-even 0.859 req/s): mixed-chunk (control) and "
                 "mixed-chunk + DSpark (8 draft tokens, RTX DSpark envs, 128 running). Column B only, 180 s rungs at "
                 "1.0/1.13/1.25/1.5x the new break-even (1.13x = 0.97 req/s, the old 1.0x), seed 7, GSM8K test[:100] gate.",
                 90, 5.0, stage="ik-promote"))
    gpu = "RTX-5090"
    snap = snapshot(gpu, "sglang", cpu=4, mem=24)
    be, _ = breakeven(snap)
    # ik8_5090_c4/20261122T084151Z: 0.84 starts (156.7K full tokens) and passes the gate, then OOMs on the first real-text
    # prefill (108 MiB short, 25 MiB free). 0.81 keeps ~1 GB more for activations (~0.31 GB of KV per 0.01).
    MEM = "--disable-prefill-cuda-graph --mem-fraction-static 0.81"
    fb = [dict(flags="--mem-fraction-static 0.79", why="0.81 did not start")]
    write(config("ik8_5090_c4", snap, [
        server("mem81", sglang_launch(gpu, MEM), steps("mem81", be, [1.0, 1.25, 1.5, 2.0], **kw), fallbacks=fb),
        server("mem81_mixed", sglang_launch(gpu, f"{MEM} {MIX}"), steps("mem81_mixed", be, [1.0, 1.25, 1.5, 2.0], **kw), fallbacks=fb)],
        "Wave 8 (RTX 5090 on a 4 vCPU / 24 GiB host, $1.346/h, break-even 0.452 req/s): prefill CUDA graphs off + "
        "--mem-fraction-static 0.81 (0.84 OOMed under real-text prefill; 0.88/0.85 at start), then + mixed-chunk. Column B only, 180 s "
        "rungs at 1.0/1.25/1.5/2.0x the new break-even (1.25x = 0.565 req/s, the old 1.0x), seed 7, GSM8K test[:100] gate.",
        80, 3.0, stage="ik-promote"))


def S9():
    """Wave 9: MI355X AITER tuning, third attempt. ik7 tuned with SGLang holding the GPU: 0 bytes free, every hipBLASLt / torch
    benchmark OOMed (34/117 torch rows, no winner). Here the FIRST server is a stub (box/stub_health.py) that only answers
    run.py's /health, so the tune step has the whole GPU: torch baseline, hipBLASLt, asm, then flydsl (JIT; AITER's own
    inkling table is all flydsl) with per-backend timeouts. Then base (untuned) -> install the merged table -> tuned, same
    box, each with gate, prefill profile (admin key) and column B at 1.0/1.25/1.5x break-even."""
    gpu = "MI355X"
    snap = snapshot(gpu, "sglang_amd")
    be, _ = breakeven(snap)
    rates = [round(m * be, 3) for m in (1.0, 1.25, 1.5)]
    stub = server("tune", "python3 /root/or/ik/stub_health.py", [
        dict(type="sh", name="aiter_tune", est_s=4500,
             cmd="python3 /root/or/ik/aiter_tune.py --untuned /root/or/ik/inkling2mini_bf16_untuned_gemm.csv --out /root/or/out; "
                 "rc=$?; touch /root/or/out/STUB_DONE; exit $rc",
             outputs=["out/aiter_tuned_inkling.csv", "out/aiter_tune_summary.json", "out/aiter_tuned_*.csv", "out/aiter_tune_*.log"])],
        ready=300)
    stub["allow_rejected"], stub["allow_rejected_why"] = True, "stub health server (no engine): the GPU is free for AITER tuning"
    servers = [stub]
    env0 = {"ROCM_QUICK_REDUCE_QUANTIZATION": "NONE", "IK_ADMIN_KEY": "local-profiler-token"}
    for srv in ("base", "tuned"):
        st = [dict(type="sh", name=f"inspect_{srv}", cmd=f"bash /root/or/ik/aiter_inspect.sh {srv}",
                   outputs=[f"out/aiter_inspect_{srv}.txt"], est_s=60),
              dict(type="eval", name=f"gate_{srv}_gsm8kor100", sets="data/eval/gsm8kor_test.jsonl", limit=100, conc=64,
                   max_tokens=16384, chat_template_kwargs={"reasoning_effort": EFFORT}, gate=dict(task="gsm8kor", threshold=0.88),
                   est_s=600),
              dict(type="sh", name=f"profile_prefill_{srv}", est_s=300, outputs=[f"out/profile_prefill_{srv}.json"],
                   cmd=f"python3 /root/or/ik/sgl_profile.py --model {SERVED} --dir /root/or/out/prof_{srv} --n 4 --tokens 10000 "
                       f"--out /root/or/out/profile_prefill_{srv}.json"),
              dict(type="protocol", name=f"protocol_{srv}_B_text", client="traffic", breakeven_rps=round(be, 4), rates=rates,
                   seeds=[7], duration=180, prefill_shot=False, ttft_basis="token")]
        if srv == "base":
            st.append(dict(type="sh", name="aiter_install", est_s=30,
                           cmd="python3 /root/or/ik/aiter_tune.py --untuned /root/or/ik/inkling2mini_bf16_untuned_gemm.csv "
                               "--out /root/or/out --install-only"))
        servers.append(server(srv, sglang_launch(gpu, '--admin-api-key "$IK_ADMIN_KEY"'), st,
                              env=dict(env0, SGLANG_TORCH_PROFILER_DIR=f"/root/or/out/prof_{srv}")))
    cfg = config("ik9_mi355x_aiter", snap, servers,
                 "Wave 9 (MI355X): AITER BF16 GEMM tuning with the GPU free (stub server first; ik7 OOMed with SGLang up), "
                 "backends torch / hipBLASLt / asm / flydsl, 117 prefill shapes; then base (untuned) -> install -> tuned on the "
                 "same box: GSM8K test[:100] gate, prefill profile, column B 180 s at 1.0/1.25/1.5x break-even, seed 7.",
                 200, 16.0, stage="ik-kernel")
    cfg["setup_cmd"] = box_setup(["aiter_inspect.sh", "aiter_tune.py", "sgl_profile.py", "stub_health.py",
                                  "inkling2mini_bf16_untuned_gemm.csv"])
    write(cfg)


def confirm(tag, gpu, eng, flags, env, mults, seed, with_ref, allow_rejected_why=None, minutes=110, budget=10.0, cpu=8, mem=32):
    """Claim-grade confirm box (lead's rule: a price only at the worst seed's ACHIEVED rate, 300 s cells, 2 seeds, column B
    holding). One box per seed. The seed-7 box first serves the vendor Balanced recipe as the QUALITY REFERENCE (no
    full-precision Inkling-2-Mini exists: recipes/inkling/raw/reference_check_*.json), evaluated twice so the greedy
    run-to-run floor is measured, then the claim config: quality row, column B (real text) at the given multiples, then
    column A (OpenRelay's client + prefill probe) at the same rates. 300 s cells (the harness adds its warmup)."""
    snap = snapshot(gpu, eng, cpu=cpu, mem=mem)
    be, _ = breakeven(snap)
    rates = [round(m * be, 3) for m in mults]
    group = f"ikC_{GPU_SHORT[gpu]}_{tag}"
    servers = []
    if with_ref:
        servers.append(server("ref", sglang_launch(gpu), [quality_eval("eval_ref_r1", gate=False), quality_eval("eval_ref_r2", gate=False)],
                              env={"ROCM_QUICK_REDUCE_QUANTIZATION": "NONE"} if gpu == "MI355X" else {}))
    st = [dict(type="sh", name="setup_vllm_client", cmd="bash bench/setup_vllm_client.sh", est_s=300), quality_eval("eval_claim"),
          dict(type="protocol", name=f"protocol_{tag}_B_text", client="traffic", breakeven_rps=round(be, 4), rates=rates,
               seeds=[seed], duration=300, prefill_shot=False, ttft_basis="token"),   # Inkling's first token is a swallowed special token
          dict(type="protocol", name=f"protocol_{tag}_A_random", client="vllmbench", vllm_bin="/root/vb/bin/vllm",
               breakeven_rps=round(be, 4), rates=rates, seeds=[seed], duration=300, prefill_shot=True)]
    srv = server(tag, sglang_launch(gpu, flags), st, env=env)
    srv["claim_group"] = group
    if allow_rejected_why:
        srv["allow_rejected"], srv["allow_rejected_why"] = True, allow_rejected_why
    servers.append(srv)
    cfg = config(f"ikC_{GPU_SHORT[gpu]}_{tag}_s{seed}", snap, servers,
                 f"CONFIRM (claim-grade), {gpu}, seed {seed}: SGLang inkling-2-mini image + '{flags}' (env {env}). "
                 + ("Vendor Balanced recipe first as the quality reference (GSM8K test[:250] + MMLU-Pro 200, twice). " if with_ref else "")
                 + f"Claim config: quality row, column B real text at {mults}x break-even (300 s, seed {seed}), then column A "
                   f"(OpenRelay's client, random tokens) at the same rates with their prefill probe.",
                 minutes, budget, stage="ik-confirm", claim_group=group)
    write(cfg)
    return cfg


def C1():
    """Confirm 1: PRO 6000 on a 4 vCPU / 24 GiB host ($2.556/h, break-even 0.859 req/s), SGLang vendor recipe +
    --enable-mixed-chunk (ik8: column B passes 1.13x = 0.971 req/s at p95 2.75 s, fails 1.25x at 4.84 s). Rungs 1.13 / 1.2 /
    1.25x, seeds 7 and 8 on two boxes; the seed-7 box also serves the vendor recipe as the quality reference."""
    for seed in (7, 8):
        confirm("mixed", "RTX-PRO-6000", "sglang", "--enable-mixed-chunk", {}, [1.13, 1.2, 1.25], seed, with_ref=(seed == 7),
                minutes=150 if seed == 7 else 110, budget=8.0, cpu=4, mem=24)


def C1D():
    """Confirm 1D: + DSpark (8 draft tokens, RTX DSpark envs, 128 running). ik8_pro6000_c4 (180 s screen, column B):
    mixed + DSpark passes 1.25x (1.074 req/s, p95 3.41 s, TPOT 22 ms, acceptance 3.08) where mixed alone fails (4.84 s);
    1.5x fails at 4.83 s. Rungs 1.13 / 1.25 / 1.35 / 1.5x."""
    for seed in (7, 8):
        confirm("mixed_dspark", "RTX-PRO-6000", "sglang", "--enable-mixed-chunk --speculative-algorithm DSPARK "
                "--speculative-num-draft-tokens 8 --max-running-requests 128", DSPARK_ENV, [1.13, 1.25, 1.35, 1.5], seed,
                with_ref=False, minutes=140, budget=8.0, cpu=4, mem=24)


def C1R():
    """Replicate of the claim's knee (lead 11-22 ~10:3xZ): the confirm passed 1.5x on both seeds but seed 7 FAILED 1.35x
    (p95 4.38 s), so 1.5x is noise until replicated. Two fresh PRO 6000 small-host boxes (one per seed), the identical
    mixed-chunk + DSpark config, column B only at 1.35x and 1.5x, 300 s cells, gate 100. Its own claim group, so the
    original cells stand as recorded: if both rates pass on both replicate boxes, 1.5x is the headline with 4 cells per
    rate; otherwise the monotonic 1.25x stays."""
    gpu = "RTX-PRO-6000"
    snap = snapshot(gpu, "sglang", cpu=4, mem=24)
    be, _ = breakeven(snap)
    rates = [round(m * be, 3) for m in (1.35, 1.5)]
    flags = "--enable-mixed-chunk --speculative-algorithm DSPARK --speculative-num-draft-tokens 8 --max-running-requests 128"
    for seed in (7, 8):
        st = [dict(type="eval", name="gate_rep_gsm8kor100", sets="data/eval/gsm8kor_test.jsonl", limit=100, conc=64, max_tokens=16384,
                   chat_template_kwargs={"reasoning_effort": EFFORT}, gate=dict(task="gsm8kor", threshold=0.88), est_s=600),
              dict(type="protocol", name="protocol_rep_B_text", client="traffic", breakeven_rps=round(be, 4), rates=rates,
                   seeds=[seed], duration=300, prefill_shot=False, ttft_basis="token")]
        srv = server("mixed_dspark", sglang_launch(gpu, flags), st, env=DSPARK_ENV)
        srv["claim_group"] = "ikC_pro6000_mixed_dspark_rep"
        write(config(f"ikC_pro6000_mixed_dspark_rep_s{seed}", snap, [srv],
                     f"REPLICATE of the claim's knee, seed {seed}: PRO 6000 (4 vCPU / 24 GiB) + mixed-chunk + DSpark, column B at "
                     f"1.35x and 1.5x break-even ({rates} req/s), 300 s, fresh box. Original: ikC_pro6000_mixed_dspark_s7/s8.",
                     60, 4.0, stage="ik-confirm", claim_group="ikC_pro6000_mixed_dspark_rep"))


def B1():
    """Claim-grade B300 box, queued on the build session's B300/B200 auto-launcher (B300 spot is scarce). Inkling is native
    NVFP4 + MXFP8 and Blackwell runs both natively (the MI355X profile, step 222, shows ~37% of prefill in NVFP4
    emulation). SGLang's GB300 Balanced flags at TP1 + the PRO 6000 winners: mixed-chunk and DSpark (SGLang's GB300 DSpark
    recipe: default draft width from the checkpoint, no RTX-only SGLANG_OPT envs) with 128 running. Both seeds on this one
    box (the card is scarce), 300 s cells, column B (the claim) then column A, quality row first. 8 vCPU / 32 GiB host: at
    several req/s the client and tokenizer need the cores."""
    gpu = "B300"
    snap = snapshot(gpu, "sglang")
    be, _ = breakeven(snap)
    mults = [1.0, 1.5, 2.0, 2.5, 3.0]
    rates = [round(m * be, 3) for m in mults]
    flags = "--enable-mixed-chunk --speculative-algorithm DSPARK --max-running-requests 128"
    st = [dict(type="sh", name="setup_vllm_client", cmd="bash bench/setup_vllm_client.sh", est_s=300), quality_eval("eval_claim"),
          dict(type="protocol", name="protocol_b300_B_text", client="traffic", breakeven_rps=round(be, 4), rates=rates,
               seeds=[7, 8], duration=300, prefill_shot=False, ttft_basis="token"),
          dict(type="protocol", name="protocol_b300_A_random", client="vllmbench", vllm_bin="/root/vb/bin/vllm",
               breakeven_rps=round(be, 4), rates=rates, seeds=[7, 8], duration=300, prefill_shot=True)]
    # same-box fallback: one retry as-is (a B300 first start JIT-compiles FA4 / CuTe kernels; a timeout there is not a verdict)
    srv = server("mixed_dspark", sglang_launch(gpu, flags), st,
                 fallbacks=[dict(flags="", why="retry once as-is: B300 first start compiles FA4 / CuTe kernels")])
    srv["claim_group"] = "ikC_b300_mixed_dspark"
    cfg = config("ikC_b300_mixed_dspark", snap, [srv],
                 f"CONFIRM (claim-grade), B300, seeds 7 and 8 on one box: SGLang inkling-2-mini image, GB300 Balanced flags at "
                 f"TP1 + '{flags}'. Quality row (GSM8K test[:250] + MMLU-Pro 200, effort 'high'), column B real text at "
                 f"{mults}x break-even ({rates} req/s), 300 s, then column A at the same rates with OpenRelay's prefill probe.",
                 235, 30.0, stage="ik-confirm", claim_group="ikC_b300_mixed_dspark")
    write(cfg)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["S1"]:
        globals()[w]()
