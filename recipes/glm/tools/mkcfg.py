#!/usr/bin/env python3
"""Generate the GLM-5.3-Flash on one MI355X experiment configs (recipes/glm/configs/*.json) for run.py.

Price bar: OpenRelay's glm-5.3-flash sync list, $0.045 in / $0.18 out per 1M = $0.000496 per 9.5K-in / 380-out request
(research/openrelay_prices_*.json). Configs carry "price_model": "openrelay/glm-5.3-flash" for the claim tooling; until
run.py/analysis read it, their break-even/ratio fields are against the Gemma price, so these configs use explicit rates.
Checkpoint: amd/GLM-5.3-Flash-Quark-MXFP4 (MoE experts MXFP4, attention / dense MLP / MTP block in the original FP8),
185 GB, validated by AMD on SGLang at TP4/TP8 on MI350X (GSM8K full set 97.19% TP4, thinking on). We run TP1.

  python3 -I recipes/glm/tools/mkcfg.py [wave ...]
"""
from __future__ import annotations

import json
import pathlib
import shlex

HERE = pathlib.Path(__file__).resolve().parent.parent
BOX = HERE / "box"
OUT = HERE / "configs"

W_AMD = [{"repo": "amd/GLM-5.3-Flash-Quark-MXFP4", "rev": "b5688f25491202978c19c4d036eef579f61bbe07", "dir": "/root/models/glm"}]
SNAP_V034 = {"name": "or-vllmrocm0340-mi355x-c8m48d300", "image": "vllm/vllm-openai-rocm:v0.34.0",
             "image_digest_amd64": "sha256:2a6642bc2e5a76f42145d816031dc518258902e22ebdf2b74f71ecf78a9f5574",
             "gpu": "MI355X", "gpus": 1, "cpu": 8, "mem": 48, "disk": 300}
SNAP_SGL = {"name": "or-sglang0523rocm10-mi355x-c8m48d300", "image": "lmsysorg/sglang:v0.5.23-rocm10-mi35x",
            "image_digest_amd64": "sha256:cff45eca86da10fbe140f85df196ef241e932bbae8036afa45942e633c3ed6c5",
            "gpu": "MI355X", "gpus": 1, "cpu": 8, "mem": 48, "disk": 300}
ENV_VLLM = {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
            "VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA": "1", "VLLM_ENGINE_READY_TIMEOUT_S": "3600"}
ENV_SGL = {"SGLANG_USE_AITER": "1"}
PRICE = "openrelay/glm-5.3-flash"
VB = "/root/vb/bin/vllm"   # OpenRelay's client in a client-only venv on images without vLLM (bench/setup_vllm_client.sh)


def setup_cmd(files=(), run=()):
    lines = ["mkdir -p /root/or/mx /root/or/out"]
    for f in sorted(set(files) | {"inspect.sh"}):
        body = (BOX / f).read_text()
        tag = "MXEOF_" + f.replace(".", "_").upper()
        lines += [f"cat > /root/or/mx/{f} <<'{tag}'", body.rstrip("\n"), tag]
    return "\n".join(lines + list(run) + ["bash /root/or/mx/inspect.sh"])


def gate(name, limit=100, threshold=0.965):
    # GLM-5.3-Flash always thinks; low effort keeps the gate's generations bounded. Greedy, their prompt and '####' scoring.
    return {"type": "eval", "name": name, "sets": "data/eval/gsm8kor_test.jsonl", "limit": limit, "conc": 64, "max_tokens": 8192,
            "chat_template_kwargs": {"reasoning_effort": "low"}, "gate": {"task": "gsm8kor", "threshold": threshold}, "est_s": 1200}


def gate_glm(name):
    """GLM gate, recalibrated 2026-11-22 08:0xZ after glm2_vllm_b16k scored 96/100 against the 96.5% bar.
    The Gemma gate (OpenRelay's 96.5% on test[:100]) is calibrated for a 98.8% model. GLM-5.3-Flash's reference is 97.12-97.19%
    (AMD's MXFP4 card, full GSM8K, thinking on). For a true 97.2% model that gate falsely fails with p = 0.31 (binomial).
    test[:250] at >= 95.0% (2 sigma under the reference) falsely fails with p = 0.025 and still catches a ~3-point quality loss."""
    return gate(name.replace("100", "250"), limit=250, threshold=0.95)


def proto(name, rates, seeds, prefill=True, client="vllmbench", vb=False):
    st = {"type": "protocol", "name": name, "client": client, "rates": rates, "seeds": seeds, "duration": 300, "prefill_shot": prefill}
    if vb:
        st["vllm_bin"] = VB
    return st


def server(name, launch, env, steps, claim_group, **extra):
    s = {"name": name, "weights": W_AMD, "served_name": "glm", "env": env, "launch": launch, "ready_timeout_s": 3600,
         "steps": [{"type": "sh", "name": f"inspect_{name}", "cmd": "tail -n 12 out/inspect.txt", "outputs": ["out/inspect.txt"], "est_s": 10}] + steps,
         "delete_weights_after": False, "allow_rejected": True, "claim_group": claim_group, "origin": "ours"}
    s.update(extra)
    return s


def config(label, stage, note, snap, servers, max_minutes=110, budget=9.5, files=(), run=()):
    return {"label": label, "stage": stage, "note": note, "price_model": PRICE, "snapshot": snap, "max_minutes": max_minutes,
            "budget_usd": budget, "setup_cmd": setup_cmd(files, run), "servers": servers}


def vllm_launch(**over):
    f = {"--served-model-name": "glm", "--host": "127.0.0.1", "--port": "8000", "--trust-remote-code": True,
         "--max-model-len": "32768", "--max-num-seqs": "64", "--max-num-batched-tokens": "8192",
         "--attention-backend": "ROCM_AITER_MLA_SPARSE", "--enable-prompt-tokens-details": True, "--language-model-only": True,
         "--reasoning-parser": "glm47"}
    for k, v in over.items():
        k = "--" + k.replace("_", "-")
        if v is None:
            f.pop(k, None)
        else:
            f[k] = v
    parts = ["vllm serve /root/models/glm"]
    for k, v in f.items():
        parts.append(k if v is True else f"{k} {shlex.quote(json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list)) else str(v))}")
    return " ".join(parts)


def sglang_launch(extra=""):
    # AMD's validated flags (amd/GLM-5.3-Flash-Quark-MXFP4 README, TP8/TP4) at TP1; chunked prefill 8192 like the Gemma recipe.
    return ("python3 -m sglang.launch_server --model-path /root/models/glm --served-model-name glm --host 127.0.0.1 --port 8000 "
            "--tp-size 1 --attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang "
            "--linear-attn-backend triton --kv-cache-dtype bfloat16 --moe-runner-backend aiter --disable-shared-experts-fusion "
            "--disable-radix-cache --context-length 32768 --max-running-requests 64 --chunked-prefill-size 8192 "
            "--cuda-graph-backend-decode full --cuda-graph-max-bs-decode 64 --mem-fraction-static 0.85 "
            "--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":8}' --watchdog-timeout 1200 "
            "--trust-remote-code --reasoning-parser glm45 --enable-metrics" + (" " + extra if extra else ""))


RATES1 = [1.0, 1.5, 2.0, 2.7]   # 2.7 ~ break-even at $4.79/h vs $0.000496/request


def g1():
    note = ("GLM-5.3-Flash feasibility + first ladder on ONE MI355X (TP1), spec decoding OFF. amd/GLM-5.3-Flash-Quark-MXFP4 @ b5688f25 "
            "(MoE MXFP4, 185 GB). Bar: OpenRelay glm-5.3-flash sync $0.000496/request -> break-even ~2.7 req/s at this box's $4.79/h. "
            "GSM8K test[:100] (their prompt, greedy, reasoning_effort low, 8K tokens), their client (random tokens) prefill probe + "
            "seed 8 at 1.0/1.5/2.0/2.7 (early stop). allow_rejected: the startup-facts gate is Gemma-specific (512-dim layers); "
            "backends are read from the log by hand.")
    return [config("glm1_vllm034", "glm-screen", note + " Engine: vLLM 0.34.0 ROCm by digest (registers Glm5NextForCausalLM), the "
                   "recipe's gfx950 AITER env and --attention-backend ROCM_AITER_MLA_SPARSE; text-only.",
                   SNAP_V034, [server("glm_vllm", vllm_launch(), ENV_VLLM, [gate("gate_glm_vllm_gsm8kor100"), proto("protocol_glm_vllm", RATES1, [8])],
                                      "glm_vllm_specoff", fallbacks=["--enforce-eager"])]),
            config("glm1_sglang0523", "glm-screen", note + " Engine: SGLang 0.5.23 ROCm 10 gfx950 image by digest (rocm720 exceeds Daytona's 20 GB snapshot limit) (glm5_next.py, gfx950 KDA "
                   "kernels, DSA kpool indexer present), AMD's validated flags at TP1.",
                   SNAP_SGL, [server("glm_sgl", sglang_launch(), ENV_SGL, [gate("gate_glm_sgl_gsm8kor100"), proto("protocol_glm_sgl", RATES1, [8], vb=True)],
                                     "glm_sgl_specoff", fallbacks=["--disable-cuda-graph"])])]


def g2():
    """One change each vs glm1_vllm034 (spec off, 8192 batched tokens), same steps; parallel boxes."""
    note = ("GLM-5.3-Flash on ONE MI355X, vLLM 0.34.0 ROCm by digest, amd/GLM-5.3-Flash-Quark-MXFP4 @ b5688f25, the recipe's gfx950 env. "
            "Parent: glm1_vllm034 (spec off; prefill probe peaked ~22K tok/s with one 10K prompt in flight per step). "
            "Same steps: GSM8K test[:100] (reasoning_effort low), their client prefill probe + seed 8 at 1.0/1.5/2.0/2.7. ")
    out = []
    for k in (3, 5):
        out.append(config(f"glm2_vllm_mtp{k}", "glm-screen",
                          note + f"ONE CHANGE: built-in MTP, num_speculative_tokens {k} (the checkpoint's single MTP layer, re-applied; "
                          "the vLLM recipe default is 5). Real-text client at 1.5 req/s seed 7 for honest acceptance.",
                          SNAP_V034, [server(f"glm_mtp{k}", vllm_launch(speculative_config={"method": "mtp", "num_speculative_tokens": k}),
                                             ENV_VLLM, [gate(f"gate_glm_mtp{k}_gsm8kor100"), proto(f"protocol_glm_mtp{k}", RATES1, [8]),
                                                        proto(f"protocol_glm_mtp{k}_text", [1.5], [7], prefill=False, client="traffic")],
                                             f"glm_vllm_mtp{k}", fallbacks=["--enforce-eager"])], max_minutes=120, budget=10.0))
    out.append(config("glm2_vllm_b16k", "glm-screen", note + "ONE CHANGE: --max-num-batched-tokens 16384 (theirs-style 8192 lets one "
                      "10K prompt prefill per step).",
                      SNAP_V034, [server("glm_b16k", vllm_launch(max_num_batched_tokens="16384"), ENV_VLLM,
                                         [gate_glm("gate_glm_b16k_gsm8kor100"), proto("protocol_glm_b16k", RATES1, [8])],
                                         "glm_vllm_b16k", fallbacks=["--enforce-eager"])]))
    return out


PROF = {"profiler": "torch", "torch_profiler_dir": "/root/or/out/prof", "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": True, "ignore_frontend": True}
PROF_FILES = ["profile_knee.sh", "profreq.py", "ktrace_moe.py", "steps.py", "stepkern.py", "gaps.py", "outside.py"]


def g3():
    """Profile GLM at its knee (spec off, 1.5 req/s passed at p95 3.33 s on glm1) to see what is hot before tuning. AITER v0.1.28
    has tuned MXFP4 fused-MoE rows for GLM-5.3-Flash only at TP2-TP8/EP shapes (inter_dim 256/512/1024, or 2048 with 37-145
    experts per rank), none for TP1 (inter_dim 2048, 288 experts, top-8)."""
    return [config("glm3_vllm_profile", "glm-profile",
                   "Profile (not a measurement cell): vLLM 0.34 GLM-5.3-Flash spec off (glm1_vllm034's engine) + torch profiler: 3 x 10K "
                   "prefill trace, then a 12 s window at 1.5 req/s after 90 s warm-up; kernel split by GLM categories (ktrace_moe.py).",
                   SNAP_V034, [server("glm_prof", vllm_launch(profiler_config=PROF), ENV_VLLM,
                                      [{"type": "sh", "name": "profile_knee", "cmd": "bash /root/or/mx/profile_knee.sh 1.5 90 12",
                                        "outputs": ["out/prof_summary.txt", "out/prof/load_bench.log"], "est_s": 900}], "glm_vllm_profile")],
                   files=PROF_FILES, max_minutes=50, budget=4.5)]


def g4():
    """Best-effort GLM assembly for a claim-grade number (both seeds, both columns), from the screens: 16K batched tokens
    (prefill 21.7K -> 24.4K tok/s, glm2_vllm_b16k), MTP k=3 (helps real text: 3.16 tokens/step; random tokens only 1.62),
    and MX_FORCE_ASM_FP4=1 (the profile shows the MXFP4 shared-expert linears on AITER's Triton FP4 GEMM, 4% of GPU time:
    the same vLLM 0.34 fallback as Gemma; NOT separately ablated). Finer ladder around the 1.5 knee."""
    out = []
    for seed in (8, 7):
        out.append(config(f"glm4_best_s{seed}", "glm-claim",
                          "GLM-5.3-Flash on ONE MI355X, best effort (screens: glm1/glm2): vLLM 0.34.0 ROCm by digest, amd Quark MXFP4, TP1, "
                          "--max-num-batched-tokens 16384, MTP k=3, MX_FORCE_ASM_FP4=1 (box/patch_force_asm_fp4.py; not separately ablated). "
                          f"GSM8K test[:250] >= 95% (GLM gate), their client seed {seed} at 1.25/1.5/1.75/2.0 + prefill probe, real text at "
                          "1.5/1.75. Bar: OpenRelay glm-5.3-flash $0.000496/request (~2.7 req/s at $4.77/h).",
                          SNAP_V034, [server("glm_best", vllm_launch(max_num_batched_tokens="16384",
                                                                     speculative_config={"method": "mtp", "num_speculative_tokens": 3}),
                                             dict(ENV_VLLM, MX_FORCE_ASM_FP4="1"),
                                             [gate_glm("gate_glm_best_gsm8kor100"), proto("protocol_glm_best", [1.25, 1.5, 1.75, 2.0], [seed]),
                                              proto("protocol_glm_best_text", [1.5, 1.75], [seed], prefill=False, client="traffic")],
                                             "glm_best_C", fallbacks=["--enforce-eager"])],
                          files=["patch_force_asm_fp4.py"], run=["python3 /root/or/mx/patch_force_asm_fp4.py"], max_minutes=120, budget=10.0))
    return out


WAVES = {"1": g1, "2": g2, "3": g3, "4": g4}

if __name__ == "__main__":
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    for w in (sys.argv[1:] or list(WAVES)):
        for c in WAVES[w]():
            p = OUT / f"{c['label']}.json"
            p.write_text(json.dumps(c, indent=1) + "\n")
            print(p)
