#!/usr/bin/env python3
"""Generate the MI355X recipe experiment configs (recipes/mi355x/configs/*.json) for run.py.

One base per engine image, one change per config (the journal's "one change per step"):
  OR    = OpenRelay's published recipe exactly: their tuned image by digest, their flags (inference-recipes @6767b4d).
  V034  = vllm/vllm-openai-rocm:v0.34.0 by digest with OpenRelay's flags (AITER v0.1.28.post1 already contains their
          upstreamed ROCm/aiter#5926-#5928; #5929 and vllm#59153 are not upstream).
Files the box needs (patches, the profiling driver, OpenRelay's trace scripts) live in recipes/mi355x/box/ and are
written onto the sandbox by `setup_cmd` (heredocs) before any engine starts; inspect.sh then records what the image
contains and which patches applied (results/<label>/<stamp>/inspect.txt).

  python3 -I recipes/mi355x/tools/mkcfg.py            # (re)write every config
  python3 -I run.py recipes/mi355x/configs/<x>.json --dry-run
"""
from __future__ import annotations

import json
import pathlib
import shlex

HERE = pathlib.Path(__file__).resolve().parent.parent
BOX = HERE / "box"
OUT = HERE / "configs"

WEIGHTS = [
    {"repo": "amd/gemma-4-31B-it-MXFP4", "rev": "0606edf12b2a926dc998b2f7684fe395ee6d2a69", "dir": "/root/models/mxfp4"},
    {"repo": "google/gemma-4-31B-it-assistant", "rev": "627c5ec1458b9086b841a91e0512fd31fd2fbbf1",
     "dir": "/root/models/mxfp4/assistant"},
]
SNAP_OR = {"name": "or-openrelayrecipe-mi355x-c8m48d150", "image": "public.ecr.aws/o7n5w9k8/vllm-rocm-gemma4:v0.30.0-g4mi355",
           "image_digest_amd64": "sha256:f22210e5fbd75824e6deb8c88aaae2c449c7180f6f4c6aaa49df7775716d4cfa",
           "gpu": "MI355X", "gpus": 1, "cpu": 8, "mem": 48, "disk": 150}
SNAP_V034 = {"name": "or-vllmrocm0340-mi355x-c8m48d150", "image": "vllm/vllm-openai-rocm:v0.34.0",
             "image_digest_amd64": "sha256:2a6642bc2e5a76f42145d816031dc518258902e22ebdf2b74f71ecf78a9f5574",
             "gpu": "MI355X", "gpus": 1, "cpu": 8, "mem": 48, "disk": 150}
ENV_OR = {"VLLM_ROCM_USE_AITER": "1", "G4_FUSE_GELU": "1", "TRITON_CACHE_DIR": "/root/.cache/triton"}
ENV_V034 = {"VLLM_ROCM_USE_AITER": "1", "TRITON_CACHE_DIR": "/root/.cache/triton"}

CAPTURE = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256]


def or_flags(**over):
    """OpenRelay's shipped TUNED_ARGS (+ the README's launch extras), as an ordered dict; None removes a flag."""
    f = {
        "--served-model-name": "gemma", "--host": "127.0.0.1", "--port": "8000",
        "--max-model-len": "32768", "--max-num-seqs": "64", "--enable-prompt-tokens-details": True,
        "--enable-prefix-caching": True,
        "--hf-overrides": json.dumps({"text_config": {"use_bidirectional_attention": None}}, separators=(",", ":")),
        "--limit-mm-per-prompt": json.dumps({"image": 0, "audio": 0, "video": 0}, separators=(",", ":")),
        "--speculative-config": {"method": "mtp", "model": "/root/models/mxfp4/assistant", "num_speculative_tokens": 2},
        "--attention-backend": "ROCM_AITER_UNIFIED_ATTN", "--kv-cache-dtype": "fp8",
        "--max-num-batched-tokens": "8192",
        "--compilation-config": {"cudagraph_capture_sizes": CAPTURE},
        "--kernel-config": {"ir_op_priority": {"rms_norm": ["native"]}},
    }
    for k, v in over.items():
        k = "--" + k.replace("_", "-") if not k.startswith("--") else k
        if v is None:
            f.pop(k, None)
        else:
            f[k] = v
    return f


def launch(flags):
    parts = ["vllm serve /root/models/mxfp4"]
    for k, v in flags.items():
        if v is True:
            parts.append(k)
        else:
            s = json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else str(v)
            parts.append(f"{k} {shlex.quote(s)}")
    return " ".join(parts)


def setup_cmd(files, run=()):
    """Write recipes/mi355x/box/<f> to /root/or/mx/<f> via quoted heredocs, run the patch commands, then inspect.sh."""
    lines = ["mkdir -p /root/or/mx /root/or/out"]
    for f in sorted(set(files) | {"inspect.sh"}):
        body = (BOX / f).read_text()
        tag = "MXEOF_" + f.replace(".", "_").upper()
        assert tag not in body
        lines += [f"cat > /root/or/mx/{f} <<'{tag}'", body.rstrip("\n"), tag]
    lines += list(run)
    lines += ["bash /root/or/mx/inspect.sh"]
    return "\n".join(lines)


def gate(limit=100):
    return {"type": "eval", "name": f"gate_gsm8kor{limit}", "sets": "data/eval/gsm8kor_test.jsonl", "limit": limit, "conc": 64,
            "gate": {"task": "gsm8kor", "threshold": 0.965}, "est_s": 240 if limit <= 100 else 420}


def proto(rates, seeds, name="protocol_mx", prefill=True, client="vllmbench", duration=300):
    return {"type": "protocol", "name": name, "client": client, "rates": rates, "seeds": seeds, "duration": duration,
            "prefill_shot": prefill}


INSPECT_STEP = {"type": "sh", "name": "inspect", "cmd": "tail -n 3 out/inspect.txt", "outputs": ["out/inspect.txt"], "est_s": 10}


def server(name, flags, env, steps, **extra):
    """Step names are made unique per server (several servers can share one box and one results dir):
    gate -> gate_<server>_gsm8kor<N>, protocol_mx -> protocol_<server>, protocol_mx_text -> protocol_<server>_text."""
    steps = [dict(st, name=st["name"].replace("protocol_mx", f"protocol_{name}").replace("gate_", f"gate_{name}_"))
             if st["type"] in ("protocol", "eval") else st for st in steps]
    s = {"name": name, "weights": WEIGHTS, "served_name": "gemma", "env": env, "launch": launch(flags),
         "ready_timeout_s": 2400, "steps": [dict(INSPECT_STEP, name=f"inspect_{name}")] + steps, "delete_weights_after": False}
    s.update(extra)
    return s


def config(label, stage, note, snap, servers, files=(), run=(), max_minutes=45, budget=4.0):
    c = {"label": label, "stage": stage, "note": note, "snapshot": snap, "max_minutes": max_minutes, "budget_usd": budget,
         "setup_cmd": setup_cmd(files, run), "servers": servers}
    return c


# Confirm waves that ran before the ladder guard existed (2026-11-22 11:5xZ). Regenerating them reproduces a past run, so the
# guard only warns for them; every other confirm config is refused.
LEGACY_CONFIRMS = ("mxC0_", "mxC1_", "mxC1B_", "mxC2_s")


def check_confirm_ladders(c):
    """DURABLE GUARD (lead, after C2): a confirm config (stage mx-confirm) must give every server, on every column it runs, at
    least one rung BELOW the rate we expect to claim for it (server["expect"] = {"A": rate, "B": rate}). If the expected rate
    misses on one box, a lower rung must exist, or the claim group has no claim-grade rate at all.
    C2 broke this twice and needed a re-run (C2b):
    - theirs, column A at 3.25/3.5 only: failed 3.25 on both seed-8 boxes;
    - the stack, column B at 3.5/3.75 only: no 3.25 cell on seed 7.
    Returns the list of violations (empty = ok)."""
    if c.get("stage") != "mx-confirm":
        return []
    bad = []
    for sv in c["servers"]:
        exp = sv.get("expect") or {}
        for st in sv["steps"]:
            if st.get("type") != "protocol":
                continue
            col = "A" if st.get("client", "vllmbench") == "vllmbench" else "B"
            if col not in exp:
                bad.append(f"{sv['name']} column {col}: no expected claim rate declared (server expect={{'{col}': rate}})")
            elif not any(r < exp[col] for r in st["rates"]):
                bad.append(f"{sv['name']} column {col}: rates {st['rates']} have no rung below the expected claim rate {exp[col]}")
    return bad


def write(c):
    bad = check_confirm_ladders(c)
    if bad and not c["label"].startswith(LEGACY_CONFIRMS):
        raise SystemExit(f"REFUSED {c['label']}: confirm ladder guard: " + "; ".join(bad))
    if bad:
        print(f"warning (legacy confirm, written as it ran): {c['label']}: " + "; ".join(bad))
    p = OUT / f"{c['label']}.json"
    p.write_text(json.dumps(c, indent=1) + "\n")
    return p


SCREEN_RATES = [3.25, 3.5, 3.75]
SCREEN_SEED = [8]   # seed 8 was the harder seed in OpenRelay's knee (3.25: 3.37 s seed 7, 4.49 s seed 8)


def wave1():
    out = []
    base_note = ("Screen vs OpenRelay's recipe (their image by digest, their flags). Their client (vllm bench serve, random tokens, "
                 "input U(1.9K,17.1K), output U(76,684)), 300 s per rate, seed 8, rates 3.25/3.5/3.75 (early stop), prefill probe, "
                 "GSM8K gate (their prompt, first 100). Control: s2_mi355x_or_recipe_seed8 (same seed, same box type). ")
    # chunked-prefill budget: OpenRelay found 16384 worse than 8192 at the knee (3.25: 3.95/5.79 vs 3.37/4.49 s)
    for b in (4096, 6144):
        out.append(config(f"mx1_or_b{b // 1024}k", "mx1-screen",
                          base_note + f"ONE CHANGE: --max-num-batched-tokens {b} (theirs 8192). Hypothesis: smaller prefill chunks cut "
                          "head-of-line blocking for newly arrived prompts and piggy-back decode more often; p95 TTFT at 3.25-3.5 drops.",
                          SNAP_OR, [server("or_b" + str(b), or_flags(max_num_batched_tokens=str(b)), ENV_OR,
                                           [gate(), proto(SCREEN_RATES, SCREEN_SEED)])]))
    # MTP draft length 3 (random tokens accept 97-98%/position; on text OpenRelay saw 0.87-0.90, 0.79-0.80)
    out.append(config("mx1_or_mtp3", "mx1-screen",
                      base_note + "ONE CHANGE: num_speculative_tokens 3 (theirs 2). Hypothesis: with ~97% per-position acceptance on "
                      "their random-token prompts, k=3 cuts decode steps ~24% and frees GPU time for prefill. Also our real-text client "
                      "(traffic.py) at 3.0 and 3.25 req/s, seed 7, to show the honest (text) acceptance side by side.",
                      SNAP_OR, [server("or_mtp3", or_flags(speculative_config={"method": "mtp", "model": "/root/models/mxfp4/assistant",
                                                                             "num_speculative_tokens": 3}), ENV_OR,
                                       [gate(), proto(SCREEN_RATES, SCREEN_SEED),
                                        proto([3.0, 3.25], [7], name="protocol_mx_text", prefill=False, client="traffic")])],
                      max_minutes=60, budget=5.5))
    # profile at the knee on our box (OpenRelay's method: profile first, then chase the biggest slice)
    prof = {"profiler": "torch", "torch_profiler_dir": "/root/or/out/prof", "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True, "ignore_frontend": True}
    out.append(config("mx1_or_profile", "mx1-profile",
                      "OpenRelay's recipe exactly + the torch profiler (their profiles.md method): 3 x 10K prefill trace, then a 12 s "
                      "window of open-loop load at 3.25 req/s (the knee) after 90 s warm-up; kernel split per step type with their "
                      "analysis scripts (ktrace/steps/stepkern/gaps). Not a measurement cell: the profiler slows the engine.",
                      SNAP_OR, [server("or_prof", or_flags(profiler_config=prof), ENV_OR,
                                       [gate(), {"type": "sh", "name": "profile_knee", "cmd": "bash /root/or/mx/profile_knee.sh 3.25 90 12",
                                                 "outputs": ["out/prof_summary.txt", "out/prof/load_bench.log"], "est_s": 600}])],
                      files=["profile_knee.sh", "profreq.py", "ktrace.py", "steps.py", "stepkern.py", "gaps.py"], max_minutes=40, budget=3.5))
    # vLLM 0.34: stock (auto ASM-GEMM detection) then forced ASM on the same box
    v_note = ("vLLM 0.34.0 ROCm (by digest; AITER v0.1.28.post1 includes OpenRelay's ROCm/aiter#5926 d512 prefill tile, #5927 "
              "scaleM_pad runtime arg, #5928 Gemma 4 GEMM rows) with OpenRelay's flags; no OpenRelay image patches (their "
              "split-KV #5929 and GELU+quant fusion are not upstream). Server A: stock. Server B: same + MX_FORCE_ASM_FP4=1 "
              "(box/patch_force_asm_fp4.py: v0.34 only uses the assembly MXFP4 GEMM where AITER has tuned rows at M=1024 and "
              "8192, which Gemma 4's qkv/gate_up shapes lack). Each: GSM8K gate, prefill probe, seed 8 at 3.25/3.5/3.75.")
    vf = or_flags()
    out.append(config("mx1_v034", "mx1-screen", v_note, SNAP_V034,
                      [server("v034_stock", vf, ENV_V034, [gate(), proto(SCREEN_RATES, SCREEN_SEED)]),
                       server("v034_asm", vf, dict(ENV_V034, MX_FORCE_ASM_FP4="1"), [gate(), proto(SCREEN_RATES, SCREEN_SEED)])],
                      files=["patch_force_asm_fp4.py"], run=["python3 /root/or/mx/patch_force_asm_fp4.py"],
                      max_minutes=80, budget=7.0))
    return out


def wave2():
    """vLLM 0.34 with the assembly MXFP4 GEMM forced back on (mx1_v034 showed stock 0.34 runs every Gemma 4 linear on
    AITER's Triton FP4 GEMM), then OpenRelay's two non-upstream pieces ported on top."""
    out = []
    env_asm = dict(ENV_V034, MX_FORCE_ASM_FP4="1")
    note = ("vLLM 0.34.0 ROCm (by digest) with OpenRelay's flags. mx1_v034 (stock) showed 0.34 probes AITER for tuned a4w4 rows "
            "at M=1024 and 8192 plus a gfx950 Triton preshuffle JSON before using the assembly MXFP4 GEMM; Gemma 4's six shapes "
            "fail both, so every linear ran AITER's Triton FP4 GEMM. ")
    out.append(config("mx2_v034_asm", "mx2-screen",
                      note + "ONE CHANGE vs mx1_v034 stock: MX_FORCE_ASM_FP4=1 (box/patch_force_asm_fp4.py: use_asm_gemm=True, i.e. "
                      "v0.30's behaviour). Same steps as wave 1: gate, prefill probe, seed 8 at 3.25/3.5/3.75.",
                      SNAP_V034, [server("v034_asm", or_flags(), env_asm, [gate(), proto(SCREEN_RATES, SCREEN_SEED)])],
                      files=["patch_force_asm_fp4.py"], run=["python3 /root/or/mx/patch_force_asm_fp4.py"], max_minutes=45, budget=4.0))
    out.append(config("mx2_v034_orport", "mx2-screen",
                      note + "OpenRelay's recipe ported to 0.34: forced ASM GEMM + their split-KV head_dim-512 decode/verify dispatch "
                      "and table entries (box/patch_ua3d.py = their patch_aiter_ua3d.py + ROCm/aiter#5929's two entries, not "
                      "upstream) + their GELU(tanh)*up+MXFP4-quant fusion into down_proj (box/patch_g4_fuse_v034.py, G4_FUSE_GELU=1; "
                      "vllm#59153 does not apply to 0.34). Their #5926-#5928 are already in AITER 0.1.28.post1. Compare with "
                      "mx2_v034_asm (ablation: +ua3d +fuse) and with their image (s2_mi355x_or_recipe_seed8).",
                      SNAP_V034, [server("v034_orport", or_flags(), dict(env_asm, G4_FUSE_GELU="1"),
                                         [gate(), proto(SCREEN_RATES, SCREEN_SEED)])],
                      files=["patch_force_asm_fp4.py", "patch_ua3d.py", "patch_g4_fuse_v034.py"],
                      run=["python3 /root/or/mx/patch_force_asm_fp4.py", "python3 /root/or/mx/patch_ua3d.py",
                           "python3 /root/or/mx/patch_g4_fuse_v034.py"], max_minutes=45, budget=4.0))
    return out


def wave3():
    """max-num-seqs: the knee profile (mx1_or_profile) and Little's law say the 64-sequence cap binds near 3.5 req/s
    (TPOT ~45 ms x 380 tokens x 3.4 req/s ~ 58 decoding + the prefilling ones), so new prompts wait for a slot while the
    GPU runs decode-only steps. KV capacity is ~977K tokens (~98 average requests), so 96 fits without preemption;
    128 may preempt. Graph capture extended so a full MTP verify batch (seqs x 3 tokens) stays on a graph."""
    out = []
    note = ("Screen vs OpenRelay's recipe (their image by digest, their flags), their client, 300 s, seed 8, rates 3.25/3.5/3.75, "
            "prefill probe, GSM8K gate. Control: s2_mi355x_or_recipe_seed8. ")
    for seqs, extra in ((96, [288]), (128, [288, 320, 352, 384])):
        out.append(config(f"mx3_or_seqs{seqs}", "mx3-screen",
                          note + f"ONE CHANGE: --max-num-seqs {seqs} (theirs 64) with cudagraph_capture_sizes extended to "
                          f"{extra[-1]} (= {seqs} x 3 MTP tokens). Hypothesis: at 3.5 req/s the 64 cap blocks admission while "
                          "prefill budget is free; lifting it lets new prompts start at once and decode rides in mixed steps. "
                          "OpenRelay measured 128 (graphs to 384) at saturation only: +1-3% throughput, 2x TPOT; never the knee.",
                          SNAP_OR, [server(f"or_seqs{seqs}", or_flags(max_num_seqs=str(seqs),
                                                                      compilation_config={"cudagraph_capture_sizes": CAPTURE + extra}),
                                           ENV_OR, [gate(), proto(SCREEN_RATES, SCREEN_SEED)])]))
    return out


def wave4():
    """Chunk size, the other direction: 4096 failed 3.25 and 6144 was worse than 8192 at 3.25 and 3.5 (seed 8); OpenRelay
    found 16384 worse than 8192 at 3.25 on their host. Locate the optimum between them."""
    note = ("Screen vs OpenRelay's recipe (their image by digest, their flags), their client, 300 s, seed 8, rates 3.25/3.5/3.75, "
            "prefill probe, GSM8K gate. Control: s2_mi355x_or_recipe_seed8. ")
    return [config("mx4_or_b12k", "mx4-screen",
                   note + "ONE CHANGE: --max-num-batched-tokens 12288 (theirs 8192). Hypothesis: fewer, larger steps cut per-step "
                   "decode/drafter/sampler overhead (mx1 b4k/b6k got worse as steps got smaller).",
                   SNAP_OR, [server("or_b12k", or_flags(max_num_batched_tokens="12288"), ENV_OR, [gate(), proto(SCREEN_RATES, SCREEN_SEED)])])]


def mtp(k):
    return {"method": "mtp", "model": "/root/models/mxfp4/assistant", "num_speculative_tokens": k}


def cap_to(n):
    """OpenRelay's capture sizes, extended in steps of 32 up to n tokens (a full verify batch = seqs x (k+1))."""
    return CAPTURE + list(range(288, n + 1, 32))


def wave5():
    """MTP k=3 was the first gain (mx1_or_mtp3, seed 8: 3.25 p95 2.26 s vs 2.55, 3.5 5.85 s vs 9.23). Push it: k=4 alone, and
    k=3 with a higher sequence cap (the cap can still bind at 3.5)."""
    note = ("Screen, OpenRelay's image by digest and their flags except as stated; their client, 300 s, seed 8, rates 3.25/3.5/3.75, "
            "prefill probe, GSM8K gate; real-text client (traffic.py) at 3.0/3.25 seed 7 for honest MTP acceptance. ")
    txt = proto([3.0, 3.25], [7], name="protocol_mx_text", prefill=False, client="traffic")
    out = [config("mx5_or_mtp4", "mx5-screen",
                  note + "Parent: mx1_or_mtp3 (k=3). ONE CHANGE: num_speculative_tokens 4, capture sizes to 320 (64 x 5).",
                  SNAP_OR, [server("or_mtp4", or_flags(speculative_config=mtp(4), compilation_config={"cudagraph_capture_sizes": cap_to(320)}),
                                   ENV_OR, [gate(), proto(SCREEN_RATES, SCREEN_SEED), txt])], max_minutes=60, budget=5.5),
           config("mx5_or_mtp3_seqs96", "mx5-screen",
                  note + "Parent: mx1_or_mtp3 (k=3). ONE CHANGE: --max-num-seqs 96 with capture sizes to 384 (96 x 4).",
                  SNAP_OR, [server("or_mtp3_seqs96", or_flags(speculative_config=mtp(3), max_num_seqs="96",
                                                              compilation_config={"cudagraph_capture_sizes": cap_to(384)}),
                                   ENV_OR, [gate(), proto(SCREEN_RATES, SCREEN_SEED)])])]
    return out


def wave6():
    """k=3 + seqs 96 reached p95 4.16 s at 3.5 (seed 8), k=4 alone 4.77 s: combine k=4 with seqs 96, bracket the cap at k=3
    (80, 112), and run the k=3 + 96 combination on seed 7 (the harder seed for the control: 10.7 vs 9.2 s at 3.5)."""
    note = ("Screen, OpenRelay's image by digest and their flags except as stated; their client, 300 s, rates 3.25/3.5/3.75, "
            "prefill probe, GSM8K gate. ")
    out = [config("mx6_or_mtp4_seqs96", "mx6-screen",
                  note + "Seed 8. Parent mx5_or_mtp3_seqs96. k=4 with --max-num-seqs 96, graphs to 480 (96 x 5).",
                  SNAP_OR, [server("or_mtp4_seqs96", or_flags(speculative_config=mtp(4), max_num_seqs="96",
                                                              compilation_config={"cudagraph_capture_sizes": cap_to(480)}),
                                   ENV_OR, [gate(), proto(SCREEN_RATES, SCREEN_SEED)])])]
    for seqs in (80, 112):
        out.append(config(f"mx6_or_mtp3_seqs{seqs}", "mx6-screen",
                          note + f"Seed 8. Parent mx5_or_mtp3_seqs96. k=3 with --max-num-seqs {seqs}, graphs to {seqs * 4}.",
                          SNAP_OR, [server(f"or_mtp3_seqs{seqs}", or_flags(speculative_config=mtp(3), max_num_seqs=str(seqs),
                                                                           compilation_config={"cudagraph_capture_sizes": cap_to(seqs * 4)}),
                                           ENV_OR, [gate(), proto(SCREEN_RATES, SCREEN_SEED)])]))
    out.append(config("mx6_or_mtp3_seqs96_s7", "mx6-screen",
                      note + "SEED 7 (the other of their two seeds). Same engine as mx5_or_mtp3_seqs96 (k=3, --max-num-seqs 96, graphs to 384).",
                      SNAP_OR, [server("or_mtp3_seqs96", or_flags(speculative_config=mtp(3), max_num_seqs="96",
                                                                  compilation_config={"cudagraph_capture_sizes": cap_to(384)}),
                                       ENV_OR, [gate(), proto(SCREEN_RATES, [7])])]))
    return out


PROF = {"profiler": "torch", "torch_profiler_dir": "/root/or/out/prof", "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": True, "ignore_frontend": True}
PROF_FILES = ["profile_knee.sh", "profreq.py", "ktrace.py", "steps.py", "stepkern.py", "gaps.py", "outside.py"]


def wave7():
    """Profile the current best (k=3, --max-num-seqs 96, graphs to 384) at 3.5 req/s, where it just misses (p95 4.16 s)."""
    return [config("mx7_prof_mtp3_seqs96", "mx7-profile",
                   "Profile (not a measurement cell): OpenRelay's image + k=3 + --max-num-seqs 96 + graphs to 384, torch profiler; "
                   "3 x 10K prefill trace, then a 12 s window at 3.5 req/s after 90 s warm-up. Adds outside.py: kernels outside the "
                   "target-forward step windows (drafter, samplers).",
                   SNAP_OR, [server("prof_mtp3_seqs96", or_flags(speculative_config=mtp(3), max_num_seqs="96", profiler_config=PROF,
                                                                 compilation_config={"cudagraph_capture_sizes": cap_to(384)}),
                                    ENV_OR, [gate(), {"type": "sh", "name": "profile_knee", "cmd": "bash /root/or/mx/profile_knee.sh 3.5 90 12",
                                                      "outputs": ["out/prof_summary.txt", "out/prof/load_bench.log"], "est_s": 600}])],
                   files=PROF_FILES, max_minutes=40, budget=3.5)]


def wave8():
    """Lossless speculative-decoding knobs in vLLM 0.30's SpeculativeConfig, on the current best (k=3, seqs 96, graphs to 384).
    The benchmark samples (vllm bench serve no longer sends temperature 0, so generation_config's temperature 1, top_k 64,
    top_p 0.95 apply), so the acceptance rule matters. Both keep the target distribution exactly."""
    note = ("Screen, seed 8, OpenRelay's image by digest, their flags + k=3 + --max-num-seqs 96 + graphs to 384 (mx5_or_mtp3_seqs96), "
            "their client, 300 s, rates 3.25/3.5/3.75, prefill probe, GSM8K gate; real-text client at 3.0/3.25 seed 7. ")
    txt = proto([3.0, 3.25], [7], name="protocol_mx_text", prefill=False, client="traffic")
    out = []
    for tag, extra, what in (("block", {"rejection_sample_method": "block"},
                              "rejection_sample_method 'block' (block verification, Sun et al.: verifies the draft jointly; lossless, "
                              "accepts at least as many tokens as token-wise verification)"),
                             ("prob", {"draft_sample_method": "probabilistic"},
                              "draft_sample_method 'probabilistic' (the drafter samples and its full distribution enters the ratio test; "
                              "with greedy drafts the acceptance is only p_target(draft token))")):
        out.append(config(f"mx8_or_mtp3_seqs96_{tag}", "mx8-screen", note + "ONE CHANGE: " + what + ".",
                          SNAP_OR, [server(f"or_mtp3_seqs96_{tag}", or_flags(speculative_config=dict(mtp(3), **extra), max_num_seqs="96",
                                                                             compilation_config={"cudagraph_capture_sizes": cap_to(384)}),
                                           ENV_OR, [gate(), proto(SCREEN_RATES, SCREEN_SEED), txt])], max_minutes=60, budget=5.5))
    return out


def best_flags(**over):
    """Current best (mx5_or_mtp3_seqs96): OpenRelay's flags + MTP k=3 + --max-num-seqs 96 + graphs to 384."""
    f = dict(speculative_config=mtp(3), max_num_seqs="96", compilation_config={"cudagraph_capture_sizes": cap_to(384)})
    f.update(over)
    return or_flags(**f)


def wave9():
    """Replicates of the current best: at the cliff a single 300 s run swings a lot box to box (seed 8 at 3.5: 4.16 s on
    mx5; seed 7 at 3.5: 2.68 s on mx6). Two more seed-8 boxes and one more seed-7 box, 3.5 and 3.75 only."""
    out = []
    for tag, seed in (("r2s8", 8), ("r3s8", 8), ("r2s7", 7)):
        out.append(config(f"mx9_best_{tag}", "mx9-replicate",
                          f"Replicate {tag} of mx5_or_mtp3_seqs96 (same engine: OpenRelay's image + k=3 + --max-num-seqs 96 + graphs "
                          f"to 384), their client, seed {seed}, 300 s at 3.5 and 3.75 req/s, no prefill probe. Purpose: box-to-box spread "
                          "at the cliff before the confirm.",
                          SNAP_OR, [server("best", best_flags(), ENV_OR, [gate(), proto([3.5, 3.75], [seed], prefill=False)])],
                          max_minutes=35, budget=3.0))
    return out


def wave10():
    """Drafter compaction (box/patch_draft_compact.py, MX_DRAFT_COMPACT=1) on the current best, both seeds."""
    out = []
    for seed in (8, 7):
        out.append(config(f"mx10_best_dc_s{seed}", "mx10-screen",
                          "Current best (OpenRelay's image + k=3 + --max-num-seqs 96 + graphs to 384) + ONE CHANGE: MX_DRAFT_COMPACT=1 "
                          "(the Gemma 4 MTP drafter's first pass runs on one token per request instead of the whole mixed batch; "
                          f"profile mx7 put the waste at ~4-5 ms per ~165 ms mixed step). Their client, seed {seed}, 300 s, rates "
                          "3.25/3.5/3.75" + (", prefill probe" if seed == 8 else "") + ", GSM8K gate. Acceptance from /metrics vs "
                          "mx5_or_mtp3_seqs96 (3.62 tokens/step) shows whether drafts stay as good.",
                          SNAP_OR, [server("best_dc", best_flags(), dict(ENV_OR, MX_DRAFT_COMPACT="1"),
                                           [gate(), proto(SCREEN_RATES, [seed], prefill=(seed == 8))])],
                          files=["patch_draft_compact.py"], run=["python3 /root/or/mx/patch_draft_compact.py"]))
    return out


def confirm(dc: bool, tag: str = "C1"):
    """The headline measurement, paired: on each box the SAME GPU serves OpenRelay's recipe and ours back to back (weights
    kept), so the 1940-2060 MHz box-to-box clock spread (gpumon_amd.py) cancels in the comparison. Seeds 7 and 8 on two
    boxes each; the order (ours first / theirs first) alternates. Ours: their client at 3.25/3.5/3.75 + prefill probe +
    GSM8K test[:250] (their split and scoring). Theirs: 3.25/3.5 + prefill probe + GSM8K test[:100] (re-gated). Seed-7 boxes
    add our real-text client (traffic.py) at 3.25/3.5 on both engines."""
    out = []
    env_ours = dict(ENV_OR, MX_DRAFT_COMPACT="1") if dc else ENV_OR
    for seed in (7, 8):
        for rep, ours_first in ((1, seed == 7), (2, seed != 7)):
            text = seed == 7
            ours_steps = [gate(250), proto(SCREEN_RATES, [seed])] + (
                [proto([3.25, 3.5], [seed], name="protocol_mx_text", prefill=False, client="traffic")] if text else [])
            theirs_steps = [gate(100), proto([3.25, 3.5], [seed])] + (
                [proto([3.25, 3.5], [seed], name="protocol_mx_text", prefill=False, client="traffic")] if text else [])
            # claim metadata per server (one paired box holds both recipes): ours merges across all confirm boxes, theirs too
            ours = server("mx_recipe", best_flags(), env_ours, ours_steps, claim_group=f"mx_recipe_{tag}", origin="ours")
            theirs = server("or_recipe", or_flags(), ENV_OR, theirs_steps, claim_group=f"or_recipe_paired_{tag}", origin="openrelay_recipe")
            order = [ours, theirs] if ours_first else [theirs, ours]
            out.append(config(f"mx{tag}_s{seed}_r{rep}", "mx-confirm",
                              f"CONFIRM (paired), seed {seed}, replicate {rep}: the same MI355X serves "
                              f"{'ours then theirs' if ours_first else 'theirs then ours'}. Ours = OpenRelay's image by digest + MTP k=3 + "
                              "--max-num-seqs 96 + graphs to 384" + (" + MX_DRAFT_COMPACT=1 (box/patch_draft_compact.py)" if dc else "") +
                              ". Theirs = their recipe exactly. Their client (vllm bench serve, random tokens, 300 s per rate); GSM8K "
                              "test[:250] on ours" + ("; real-text client at 3.25/3.5 on both" if text else "") + ".",
                              SNAP_OR, order, files=["patch_draft_compact.py"] if dc else [],
                              run=["python3 /root/or/mx/patch_draft_compact.py"] if dc else [],
                              max_minutes=95 if text else 70, budget=8.0 if text else 6.0))
    return out


WAVES = {"1": wave1, "2": wave2, "3": wave3, "4": wave4, "5": wave5, "6": wave6, "7": wave7, "8": wave8, "9": wave9, "10": wave10,
         "C1": lambda: confirm(dc=True, tag="C1"), "C0": lambda: confirm(dc=False, tag="C0"), "11": lambda: wave11(), "12": lambda: wave12(), "C1B": lambda: confirm_text8(), "H4": lambda: small_host(), "S": lambda: sglang_track(), "ID": lambda: identity(), "S2": lambda: sglang_mxfp4_fix(), "S3": lambda: vllm_fp8_ref(), "R": lambda: small_host_rep(), "F": lambda: fine_knee(),
         "K": lambda: ctx256k(), "P": lambda: profile_ours(), "L": lambda: long_prefill(), "A": lambda: attn_lab(), "A2": lambda: attn_lab2(), "A3": lambda: attn_s2_safety(), "C2": lambda: stack_confirm(), "R4": lambda: round4(), "C2b": lambda: stack_confirm_b()}


def stack_confirm_b(tag="C2b"):
    """C2b (lead, ~11:5xZ): closes C2's two claim gaps with a separate claim group (nothing pooled with C2):
    - theirs had no rung below 3.25 on column A;
    - the stack had no 3.25 real-text cell.
    2 fresh boxes, one per seed, 4 vCPU / 24 GiB host. Each serves theirs, ours-today and the stack back to back (order rotated).
    - Column A: 3.0 / 3.25 / 3.5 for all three, plus 3.75 for the stack.
    - Column B (real text): 3.0 / 3.25 / 3.5 for all three, plus 2.75 for theirs (its expected B claim is 3.0, and the guard
      wants a rung below it).
    - GSM8K test[:250] on the stack, [:100] on the others. No prefill probe (C2 measured it on the same configs).
    Expected claim rates (the ladder guard checks one rung below each): theirs A 3.25 / B 3.0, ours-today A 3.5 / B 3.25,
    stack A 3.5 / B 3.5."""
    snap = dict(SNAP_OR, name="or-openrelayrecipe-mi355x-c4m24d150", cpu=4, mem=24)
    env_cur = dict(ENV_OR, MX_DRAFT_COMPACT="1")
    env_stk = dict(env_cur, MX_PS_FP4="1", MX_PS_MIN_M="1536")
    out = []
    for seed, order in ((7, "TCS"), (8, "SCT")):
        theirs = server("or_recipe", or_flags(), ENV_OR,
                        [gate(100), proto([3.0, 3.25, 3.5], [seed], prefill=False), TXT([2.75, 3.0, 3.25, 3.5], seed)],
                        claim_group=f"or_recipe_paired_{tag}", origin="openrelay_recipe", expect={"A": 3.25, "B": 3.0})
        cur = server("mx_recipe", best_flags(), env_cur,
                     [gate(100), proto([3.0, 3.25, 3.5], [seed], prefill=False), TXT([3.0, 3.25, 3.5], seed)],
                     claim_group=f"mx_recipe_{tag}", origin="ours", expect={"A": 3.5, "B": 3.25})
        stk = server("mx_stack", best_flags(), env_stk,
                     [gate(250), proto([3.0, 3.25, 3.5, 3.75], [seed], prefill=False), TXT([3.0, 3.25, 3.5], seed)],
                     claim_group=f"mx_stack_{tag}", origin="ours", expect={"A": 3.5, "B": 3.5})
        stk["launch"] = "bash /root/or/mx/serve_ps.sh " + stk["launch"]
        by = {"T": theirs, "C": cur, "S": stk}
        names = {"T": "theirs", "C": "ours-today", "S": "ours-stacked"}
        out.append(config(f"mx{tag}_s{seed}", "mx-confirm",
                          f"THREE-WAY CONFIRM C2b (paired, own claim groups), seed {seed}, 4 vCPU / 24 GiB host: the same MI355X serves "
                          f"{', '.join(names[x] for x in order)} back to back. Same three engines as C2 (ours-stacked = ours-today + "
                          "MX_PS_FP4=1 MX_PS_MIN_M=1536). Their client at 3.0/3.25/3.5 (+3.75 stack), real text at 3.0/3.25/3.5 "
                          "(+2.75 theirs), 300 s + 60 s warm-up per cell, GSM8K test[:250] on the stack. Closes C2's missing lower rungs.",
                          snap, [by[x] for x in order],
                          files=["patch_draft_compact.py", "patch_ps_fp4.py", "serve_ps.sh", "cgmon.py"],
                          run=["python3 /root/or/mx/patch_draft_compact.py"] + PS_RUN, max_minutes=170, budget=12.0))
    return out


def round4():
    """Round 4 (lead, ~10:40Z): MX_PS_FP4 made prefill GEMMs ~4% faster and TPOT 10-30% lower, so knobs tuned before it may
    have moved. Re-sweep on top of the stack (C1 + MX_PS_FP4 MIN_M 1536): each box pairs the stack with the stack + ONE
    change, seed 8 (the binding seed), their client at 3.75 and 4.0 (each cell has the protocol's 60 s warm-up), order
    alternating between boxes. 4 vCPU / 24 GiB host as in C2. Changes: --max-num-batched-tokens 10240 / 12288 (12K was
    neutral before, journal 109); --max-num-seqs 80 / 112 with graphs to seqs x 4 (faster decode -> fewer concurrent
    requests at the same rate). Lever 3 (aiter fused_qkv_split_qk_norm_rope_cache on the sliding layers, ~0.5% per
    notes/glue_headroom.md) is not built: it would replace the attention backend's KV-cache write and the compiled norm path."""
    snap = dict(SNAP_OR, name="or-openrelayrecipe-mi355x-c4m24d150", cpu=4, mem=24)
    env_stk = dict(ENV_OR, MX_DRAFT_COMPACT="1", MX_PS_FP4="1", MX_PS_MIN_M="1536")
    rates = [3.75, 4.0]
    variants = [("b10k", "--max-num-batched-tokens 10240", dict(max_num_batched_tokens="10240"), True),
                ("b12k", "--max-num-batched-tokens 12288", dict(max_num_batched_tokens="12288"), False),
                ("s80", "--max-num-seqs 80, graphs to 320", dict(max_num_seqs="80", compilation_config={"cudagraph_capture_sizes": cap_to(320)}), True),
                ("s112", "--max-num-seqs 112, graphs to 448", dict(max_num_seqs="112", compilation_config={"cudagraph_capture_sizes": cap_to(448)}), False)]
    out = []
    for tag, what, over, base_first in variants:
        base = server("mx_stack", best_flags(), env_stk, [gate(100), proto(rates, [8], prefill=False)],
                      claim_group=f"mx_stack_R4{tag}_ctrl", origin="ours")
        var = server(f"mx_stack_{tag}", best_flags(**over), env_stk, [gate(100), proto(rates, [8], prefill=False)],
                     claim_group=f"mx_stack_R4{tag}", origin="ours")
        for sv in (base, var):
            sv["launch"] = "bash /root/or/mx/serve_ps.sh " + sv["launch"]
        out.append(config(f"mxR4_{tag}", "mx-lever", f"Round 4 (paired, one box, seed 8): the stack (C1 + MX_PS_FP4 MIN_M 1536) vs "
                          f"the stack + {what}; {'stack first' if base_first else 'variant first'}; their client at 3.75 / 4.0, "
                          "300 s + 60 s warm-up per cell; 4 vCPU / 24 GiB host.",
                          snap, [base, var] if base_first else [var, base],
                          files=["patch_draft_compact.py", "patch_ps_fp4.py", "serve_ps.sh", "cgmon.py"],
                          run=["python3 /root/or/mx/patch_draft_compact.py"] + PS_RUN, max_minutes=60, budget=4.5))
    return out


# Round-3 stacked confirm (lead, ~10:10Z): the levers that helped. From this session: none (sampler 0.46% < 2%, journal 128;
# attention split/decode tile too small, num_stages-2 tile aborts the compiler, journal 129-130; long-prefill-token-threshold
# 6144 no gain on the same box, journal 133). From the GEMM session (notes/gemm_headroom.md, mxG2_*): MX_PS_FP4=1, aiter
# v0.1.30's persistent 256x256 MXFP4 kernel for prefill-size GEMMs (box/patch_ps_fp4.py, owned by that session; used as-is,
# MX_PS_MIN_M=1536 per its threshold note).
PS_CO = "f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ps.co"
PS_RUN = [f"curl -sfL --retry 3 -o /root/or/mx/{PS_CO} https://raw.githubusercontent.com/ROCm/aiter/"
          f"77ba72b7061a3c76c4f726a50e8edfd2fe2a80e7/hsa/gfx950/f4gemm/{PS_CO}",
          f"echo '8b541b78bff2de71f3cb48db89572f1eb4adc403e09a675be9a8a546a60ff938  /root/or/mx/{PS_CO}' | sha256sum -c -",
          "python3 /root/or/mx/patch_ps_fp4.py"]


def stack_confirm(tag="C2"):
    """THREE-WAY claim-grade confirm (lead): on each box the SAME GPU serves OpenRelay's exact recipe, our current recipe (C1)
    and our stacked recipe (C1 + MX_PS_FP4) back to back, weights kept, so the headline is theirs vs ours-today vs ours-stacked
    on the same GPUs. 4 boxes, seeds 7 and 8 twice, server order rotated. Host: 4 vCPU / 24 GiB (the headline-price host;
    CPU is not binding, journal 131). Their client: theirs 3.25/3.5 (their knee; 3.75+ is known to fail), ours-today
    3.25/3.5/3.75, stacked 3.25/3.5/3.75/4.0, each with the prefill probe. Real text (column B) on one box per seed: theirs
    3.25, ours-today 3.25/3.5, stacked 3.5/3.75. GSM8K test[:250] on the stacked server, [:100] on the other two."""
    snap = dict(SNAP_OR, name="or-openrelayrecipe-mi355x-c4m24d150", cpu=4, mem=24)
    env_cur = dict(ENV_OR, MX_DRAFT_COMPACT="1")
    env_stk = dict(env_cur, MX_PS_FP4="1", MX_PS_MIN_M="1536")
    out = []
    orders = {(7, 1): "TCS", (7, 2): "SCT", (8, 1): "CST", (8, 2): "STC"}
    for (seed, rep), order in orders.items():
        text = rep == 1
        theirs = server("or_recipe", or_flags(), ENV_OR,
                        [gate(100), proto([3.25, 3.5], [seed])] + ([TXT([3.25], seed)] if text else []),
                        claim_group=f"or_recipe_paired_{tag}", origin="openrelay_recipe")
        cur = server("mx_recipe", best_flags(), env_cur,
                     [gate(100), proto([3.25, 3.5, 3.75], [seed])] + ([TXT([3.25, 3.5], seed)] if text else []),
                     claim_group=f"mx_recipe_{tag}", origin="ours")
        stk = server("mx_stack", best_flags(), env_stk,
                     [gate(250), proto([3.25, 3.5, 3.75, 4.0], [seed])] + ([TXT([3.5, 3.75], seed)] if text else []),
                     claim_group=f"mx_stack_{tag}", origin="ours")
        stk["launch"] = "bash /root/or/mx/serve_ps.sh " + stk["launch"]
        by = {"T": theirs, "C": cur, "S": stk}
        names = {"T": "theirs", "C": "ours-today", "S": "ours-stacked"}
        out.append(config(f"mx{tag}_s{seed}_r{rep}", "mx-confirm",
                          f"THREE-WAY CONFIRM (paired), seed {seed}, replicate {rep}, 4 vCPU / 24 GiB host: the same MI355X serves "
                          f"{', '.join(names[x] for x in order)} back to back. Theirs = their recipe exactly. Ours-today = C1 (their "
                          "image by digest + MTP k=3 + --max-num-seqs 96 + graphs to 384 + MX_DRAFT_COMPACT=1). Ours-stacked = "
                          "ours-today + MX_PS_FP4=1 MX_PS_MIN_M=1536 (box/patch_ps_fp4.py: aiter v0.1.30's persistent 256x256 MXFP4 "
                          "kernel for M >= 1536, sha256-pinned .co). Their client, 300 s per rate, prefill probe on each" +
                          ("; real text on all three" if text else "") + "; GSM8K test[:250] on the stacked server.",
                          snap, [by[x] for x in order],
                          files=["patch_draft_compact.py", "patch_ps_fp4.py", "serve_ps.sh", "cgmon.py"],
                          run=["python3 /root/or/mx/patch_draft_compact.py"] + PS_RUN,
                          max_minutes=125 if text else 95, budget=9.0 if text else 7.0))
    return out


def attn_s2_safety():
    """attn_mix_lab2 found num_stages=2 on the d512 prefill-shaped tile 1.11x faster on our mixed step and output-identical, but
    the same config aborted the Triton/LLVM compiler on the 96-decode shape. box/attn_s2_safety.py compiles it for every launch
    specialization an engine step can produce (crash-isolated) before it may go into a confirm."""
    nogpu = {"name": "nogpu", "weights": [], "served_name": "none", "env": {}, "allow_rejected": True, "ready_timeout_s": 120,
             "launch": "python3 -m http.server 8000 --bind 127.0.0.1 --directory /root/or/hc", "delete_weights_after": False,
             "steps": [{"type": "sh", "name": "s2_safety", "est_s": 2400, "outputs": ["out/s2_safety.txt", "out/s2_safety.jsonl"],
                        "cmd": "LAB_OUT=/root/or/out python3 /root/or/mx/attn_s2_safety.py > out/s2_safety.txt 2>&1; tail -n 20 out/s2_safety.txt"}]}
    return [config("mxA3_s2_safety", "mx-lever", "Compile safety of num_stages=2 on the d512 prefill-shaped attention tile across "
                   "engine-like launch shapes (bisect of the m96 abort + block-table widths 256/2048 x per-step integers).",
                   SNAP_OR, [nogpu], files=["attn_s2_safety.py"], run=["mkdir -p /root/or/hc && echo ok > /root/or/hc/health"],
                   max_minutes=55, budget=4.5)]


def attn_lab2():
    """Second attention-lab pass (box/attn_mix_lab2.py): every config in a worker subprocess (the first pass died on a Triton/LLVM
    abort), interleaved against the table default, realistic mix (~35-40 decoding requests per mixed step). Three boxes in parallel:
    the mixed-step 2D tiles plus the split measurement, and the decode-only split-KV 3D sweep at 40 and 96 requests."""
    out = []
    for tag, cases, cases_out in (("mix", "split40 m40_512 m40_256 m96_512", ["split40", "m40_512", "m40_256", "m96_512"]),
                                  ("dec40", "d40_512", ["d40_512"]), ("dec96", "d96_512", ["d96_512"])):
        nogpu = {"name": "nogpu", "weights": [], "served_name": "none", "env": {}, "allow_rejected": True, "ready_timeout_s": 120,
                 "launch": "python3 -m http.server 8000 --bind 127.0.0.1 --directory /root/or/hc", "delete_weights_after": False,
                 "steps": [{"type": "sh", "name": f"attn_lab2_{tag}", "est_s": 2400,
                            "outputs": [f"out/attn_lab2_{tag}.txt"] + [f"out/lab2_{c}.jsonl" for c in cases_out],
                            "cmd": f"LAB_OUT=/root/or/out python3 /root/or/mx/attn_mix_lab2.py {cases} > out/attn_lab2_{tag}.txt 2>&1; "
                                   f"tail -n 30 out/attn_lab2_{tag}.txt"}]}
        out.append(config(f"mxA2_{tag}", "mx-lever", f"Attention lab pass 2 ({cases}): crash-isolated sweep, interleaved timing.",
                          SNAP_OR, [nogpu], files=["attn_mix_lab2.py"], run=["mkdir -p /root/or/hc && echo ok > /root/or/hc/health"],
                          max_minutes=55, budget=4.5))
    return out


def attn_lab():
    """Lead lever 4 (d512 attention for OUR mix): a kernel lab on OpenRelay's image, no engine (an http.server stands in for
    /health, the GEMM session's trick). box/attn_mix_lab.py: does splitting the decode requests out of the mixed step's single
    prefill-shaped launch save time, and is their attn_2d tile still best for k=3 / 96 requests / 8K chunks?"""
    nogpu = {"name": "nogpu", "weights": [], "served_name": "none", "env": {}, "allow_rejected": True, "ready_timeout_s": 120,
             "launch": "python3 -m http.server 8000 --bind 127.0.0.1 --directory /root/or/hc", "delete_weights_after": False,
             "steps": [{"type": "sh", "name": f"attn_lab_{n}x{q}", "est_s": 900, "outputs": [f"out/attn_lab_{n}x{q}.txt"],
                        "cmd": f"python3 /root/or/mx/attn_mix_lab.py {n} {q} > out/attn_lab_{n}x{q}.txt 2>&1; tail -n 12 out/attn_lab_{n}x{q}.txt"}
                       for n, q in ((96, 4), (48, 4), (64, 3))]}
    return [config("mxA_attn_lab", "mx-lever", "Attention lab (no engine): split-vs-one-launch cost of our mixed steps and an attn_2d sweep "
                   "for the mixed launch, d512 and d256, at 96x4 (our recipe at the knee), 48x4, and 64x3 (their recipe).",
                   SNAP_OR, [nogpu], files=["attn_mix_lab.py"], run=["mkdir -p /root/or/hc && echo ok > /root/or/hc/health"],
                   max_minutes=50, budget=4.0)]


def long_prefill():
    """Lead lever 2 (short prompts alongside long ones): vLLM 0.30's V1 scheduler has no max_num_partial_prefills (V0 only); it
    honours --long-prefill-token-threshold, a per-request cap on new tokens per step (scheduler.py: running and waiting paths),
    so with 8192-token steps a new prompt can prefill next to a long one instead of waiting behind it. Policy 'priority' is
    inert here (the client sends no priority). Paired on one box, seed 8: our recipe, then + threshold 6144, then + 4096.
    p95 is over ALL requests, so the long prompts' extra steps can cost as much as the short ones gain."""
    rates = [3.25, 3.5, 3.75]
    env = dict(ENV_OR, MX_DRAFT_COMPACT="1")
    base = server("mx_recipe", best_flags(), env, [CPUMON_START, proto(rates, [8], prefill=False)], claim_group="mx_recipe_lpt_ctrl", origin="ours")
    s6 = server("mx_lpt6144", best_flags(long_prefill_token_threshold="6144"), env, [gate(), proto(rates, [8], prefill=False)],
                claim_group="mx_recipe_lpt6144", origin="ours")
    s4 = server("mx_lpt4096", best_flags(long_prefill_token_threshold="4096"), env, [gate(), proto(rates, [8], prefill=False), CPUMON_PULL],
                claim_group="mx_recipe_lpt4096", origin="ours")
    return [config("mxL_lpt_s8", "mx-lever", "Lever (paired, one box, seed 8): --long-prefill-token-threshold 6144 and 4096 vs our C1 recipe, "
                   "their client at 3.25 (warm-up) / 3.5 / 3.75, 300 s. One change per server; same box, same seed.",
                   SNAP_OR, [base, s6, s4], files=["patch_draft_compact.py", "cgmon.py"], run=["python3 /root/or/mx/patch_draft_compact.py"],
                   max_minutes=85, budget=7.0)]


# Per-second host/process CPU sampler, started DETACHED (setsid) so it outlives the step (a setup_cmd nohup died with the exec,
# mxH4_*). Pulled at the end as out/cpumon.csv; tools/cpu_cells.py joins it (and gpumon.csv's GFX clock) to every cell.
# box/cgmon.py since 2026-11-22 09:22Z: a Daytona sandbox's /proc/stat is the 192-CPU host, so cpumon.py's host busy % was
# meaningless; cgmon.py reads the sandbox cgroup (cores used vs the cpu.max quota, quota-throttled periods). The runs launched
# before then (mxH4b_*, mxF_*) were switched to it on the box by tools/boxsh.py before their first load cell.
CPUMON_START = {"type": "sh", "name": "cpumon_start", "est_s": 10, "outputs": [],
                "cmd": "setsid nohup python3 /root/or/mx/cgmon.py /root/or/out/cpumon.csv 1 > /dev/null 2>&1 < /dev/null & sleep 3; tail -n 2 out/cpumon.csv"}
CPUMON_PULL = {"type": "sh", "name": "cpumon_pull", "cmd": "wc -l out/cpumon.csv; tail -n 1 out/cpumon.csv", "outputs": ["out/cpumon.csv"], "est_s": 10}
TXT = lambda rates, seed: proto(rates, [seed], name="protocol_mx_text", prefill=False, client="traffic")  # noqa: E731


def small_host_rep():
    """Lead: replicate mx_recipe_c4m24 before it becomes the headline price: 2 more small-host boxes (4 vCPU / 24 GiB), one per
    seed, with a working per-second CPU sampler (host % and engine-process %) and the GFX clock; columns A and B."""
    snap = dict(SNAP_OR, name="or-openrelayrecipe-mi355x-c4m24d150", cpu=4, mem=24)
    out = []
    for seed in (8, 7):
        steps = [CPUMON_START, gate(), proto(SCREEN_RATES, [seed]), TXT([3.25, 3.5], seed), CPUMON_PULL]
        out.append(config(f"mxH4b_s{seed}", "mx-host", f"REPLICATE of mxH4_s{seed}: our recipe on 4 vCPU / 24 GiB ($4.17/h), their client seed {seed} "
                          "at 3.25/3.5/3.75 + prefill probe, real text at 3.25/3.5, GSM8K gate; box/cpumon.py per second (host CPU %, "
                          "EngineCore %, client %), GFX clock from gpumon. Same claim group as mxH4_* (mx_recipe_c4m24).",
                          snap, [server("mx_recipe", best_flags(), dict(ENV_OR, MX_DRAFT_COMPACT="1"), steps,
                                        claim_group="mx_recipe_c4m24", origin="ours")],
                          files=["patch_draft_compact.py", "cgmon.py"], run=["python3 /root/or/mx/patch_draft_compact.py"],
                          max_minutes=60, budget=4.5))
    return out


def fine_knee():
    """Lead (Maks: beat them by more): the 0.25 req/s ladder hides the real gap. Paired finer knee, same box, both recipes:
    ours at 3.6/3.7, theirs at 3.3/3.4 (each after a warm-up rung at their last passing rate), seeds 7 and 8 on two boxes each,
    order alternating. Supplementary knee search, not their method. Box s8_r1 also runs our real-text client at 3.25/3.5 seed 8
    (a second seed-8 real-text cell, review caveat 4). CPU sampler on all."""
    out = []
    for seed in (8, 7):
        for rep in (1, 2):
            ours_steps = [proto([3.25, 3.6, 3.7], [seed], prefill=False)] + ([TXT([3.25, 3.5], seed)] if (seed, rep) == (8, 1) else [])
            theirs_steps = [proto([3.0, 3.3, 3.4], [seed], prefill=False)]
            ours = server("mx_recipe", best_flags(), dict(ENV_OR, MX_DRAFT_COMPACT="1"), [gate()] + ours_steps,
                          claim_group="mx_recipe_fine", origin="ours")
            theirs = server("or_recipe", or_flags(), ENV_OR, [gate()] + theirs_steps, claim_group="or_recipe_fine", origin="openrelay_recipe")
            first, second = (ours, theirs) if rep == 1 else (theirs, ours)
            first = dict(first, steps=[first["steps"][0], CPUMON_START] + first["steps"][1:])
            second = dict(second, steps=second["steps"] + [CPUMON_PULL])
            out.append(config(f"mxF_s{seed}_r{rep}", "mx-fineknee",
                              f"Paired FINER knee (supplementary to their 0.25 req/s ladder), seed {seed}, replicate {rep}: the same MI355X "
                              f"serves {'ours then theirs' if rep == 1 else 'theirs then ours'}. Ours (C1 recipe) at 3.25 (warm-up) / 3.6 / 3.7; "
                              "theirs (exact) at 3.0 (warm-up) / 3.3 / 3.4. Their client, 300 s per rate." +
                              (" Plus our real-text client at 3.25/3.5 seed 8 on ours." if (seed, rep) == (8, 1) else ""),
                              SNAP_OR, [first, second], files=["patch_draft_compact.py", "cgmon.py"],
                              run=["python3 /root/or/mx/patch_draft_compact.py"], max_minutes=85, budget=7.0))
    return out


def ctx256k():
    """Maks: OpenRelay's API advertises 256K context but their recipe serves --max-model-len 32768 (as did both C1 recipes).
    Does the ranking hold at 262144? Paired, both recipes on the same box, one box per seed; KV pool recorded from startup facts;
    if a recipe will not start at 262144 the same-box fallback appends --max-model-len 131072 (argparse: last wins)."""
    out = []
    for seed in (8, 7):
        ours = server("mx_recipe", best_flags(max_model_len="262144"), dict(ENV_OR, MX_DRAFT_COMPACT="1"),
                      [gate(), proto([3.25, 3.5], [seed]), TXT([3.25], seed)], claim_group="mx_recipe_ctx256k", origin="ours",
                      fallbacks=["--max-model-len 131072"])
        theirs = server("or_recipe", or_flags(max_model_len="262144"), ENV_OR,
                        [gate(), proto([3.25, 3.5], [seed]), TXT([3.25], seed)], claim_group="or_recipe_ctx256k", origin="openrelay_recipe",
                        fallbacks=["--max-model-len 131072"])
        first, second = (ours, theirs) if seed == 8 else (theirs, ours)
        out.append(config(f"mxK_s{seed}", "mx-ctx256k",
                          f"Paired at 256K context (--max-model-len 262144; fallback 131072), seed {seed}: the same MI355X serves "
                          f"{'ours then theirs' if seed == 8 else 'theirs then ours'}. Column A 3.25/3.5 + prefill probe, column B 3.25, GSM8K gate. "
                          "Everything else as C1. KV pool tokens from startup facts vs C1's 32K runs.",
                          SNAP_OR, [first, second], files=["patch_draft_compact.py"], run=["python3 /root/or/mx/patch_draft_compact.py"],
                          max_minutes=95, budget=8.0))
    return out


def profile_ours():
    """Lead lever 3: after the drafter compaction, what do the samplers (top-k 64 / top-p 0.95 from generation_config, rejection
    sampler) and the drafter cost outside the target step windows (box/outside.py), on OUR recipe at 3.5 req/s?"""
    return [config("mxP_ours_profile", "mx-profile",
                   "Profile (not a measurement cell): our recipe (C1: k=3, seqs 96, graphs to 384, MX_DRAFT_COMPACT=1) + torch profiler; "
                   "3 x 10K prefill trace, then 12 s at 3.5 req/s after 90 s warm-up; outside.py attributes drafter/sampler kernels.",
                   SNAP_OR, [server("prof_ours", best_flags(profiler_config=PROF), dict(ENV_OR, MX_DRAFT_COMPACT="1"),
                                    [{"type": "sh", "name": "profile_knee", "cmd": "bash /root/or/mx/profile_knee.sh 3.5 90 12",
                                      "outputs": ["out/prof_summary.txt", "out/prof/load_bench.log"], "est_s": 600}])],
                   files=PROF_FILES + ["patch_draft_compact.py"], run=["python3 /root/or/mx/patch_draft_compact.py"], max_minutes=40, budget=3.5)]


def vllm_fp8_ref():
    """Attribution for the SGLang FP8 result (mxS1 sgl_fp8: prefill 17.2K, 2.5 req/s fails): the same RedHat FP8-dynamic
    checkpoint on vLLM (OpenRelay's v0.30 image, their flags, spec off), same steps. Separates engine from checkpoint."""
    f = or_flags(speculative_config=None)
    # G4_FUSE_GELU=0: the image's GELU+MXFP4 fusion keys on down_proj having a weight_scale, which FP8 layers have too
    srv = server("vllm_fp8", f, dict(ENV_OR, G4_FUSE_GELU="0"), [gate(), proto([2.5, 3.0, 3.25], [8])], claim_group="vllm_fp8dyn_specoff", origin="ours")
    srv["weights"] = W_FP8
    srv["launch"] = srv["launch"].replace("/root/models/mxfp4", "/root/models/fp8dyn")
    return [config("mxS3_vllm_fp8", "mx-sglang",
                   "SGLang-track attribution: RedHatAI/gemma-4-31B-it-FP8-dynamic on vLLM (OpenRelay's image by digest, their flags, "
                   "spec off), prefill probe + seed 8 at 2.5/3.0/3.25, GSM8K gate. Compare with mxS1_sglang's sgl_fp8 (same checkpoint) "
                   "and mxS0_vllm_specoff (MXFP4).", SNAP_OR, [srv], max_minutes=45, budget=4.0)]


def sglang_mxfp4_fix():
    """mxS1's MXFP4 server died at start: SGLang builds Gemma 4's multimodal processor and amd/gemma-4-31B-it-MXFP4 ships no
    processor config ("Can't load feature extractor ... preprocessor_config.json"; OpenRelay hit the same on 0.5.20). Add
    google/gemma-4-31B-it's processor_config.json (ungated, the base model's own) to the checkpoint dir and retry, spec off."""
    w = WEIGHTS[:1] + [{"repo": "google/gemma-4-31B-it", "rev": "842da3794eaa0b77d5f08bae87a17459d91ff475", "dir": "/root/models/mxfp4",
                        "patterns": ["processor_config.json"]}]
    s1 = [c for c in sglang_track() if c["label"] == "mxS1_sglang"][0]
    srv = dict(s1["servers"][0], weights=w)
    return [dict(s1, label="mxS2_sglang_mxfp4", servers=[srv], max_minutes=55, budget=4.5,
                 note="SGLang track, spec OFF, MXFP4 retry: same as mxS1_sglang's server 1 + google/gemma-4-31B-it processor_config.json "
                      "in the checkpoint dir (SGLang builds the multimodal processor at start). Prefill probe, seed 8 at 2.5/3.0/3.25.")]


def identity():
    """Lead's review of C1, caveat 3: show token-level output identity of the drafter compaction. One box, our recipe twice:
    unpatched (MX_DRAFT_COMPACT unset; the patch is applied but env-gated) then patched. Each server runs the probe twice at
    concurrency 1 (run-to-run noise control) and once at concurrency 16 (mixed prefill+decode steps). Temperature 0,
    ignore_eos, 256 tokens; 50 short GSM8K prompts + 10 random-token 10K prompts. Compare offline with
    box/identity_probe.py --compare."""
    def probes(tag):
        return [{"type": "sh", "name": f"idprobe_{tag}_c1_a", "cmd": f"python3 /root/or/mx/identity_probe.py --conc 1 --out out/idprobe_{tag}_c1_a.json",
                 "outputs": [f"out/idprobe_{tag}_c1_a.json"], "est_s": 420},
                {"type": "sh", "name": f"idprobe_{tag}_c1_b", "cmd": f"python3 /root/or/mx/identity_probe.py --conc 1 --out out/idprobe_{tag}_c1_b.json",
                 "outputs": [f"out/idprobe_{tag}_c1_b.json"], "est_s": 420},
                {"type": "sh", "name": f"idprobe_{tag}_c16", "cmd": f"python3 /root/or/mx/identity_probe.py --conc 16 --out out/idprobe_{tag}_c16.json",
                 "outputs": [f"out/idprobe_{tag}_c16.json"], "est_s": 240}]
    off = server("unpatched", best_flags(), ENV_OR, probes("off"))
    on = server("patched", best_flags(), dict(ENV_OR, MX_DRAFT_COMPACT="1"), probes("on"))
    return [config("mxID_identity", "mx-verify",
                   "Identity probe for box/patch_draft_compact.py (lead's C1 review, caveat 3): our recipe (k=3, --max-num-seqs 96, graphs to "
                   "384) on one MI355X, unpatched then patched, temperature 0, token ids compared (box/identity_probe.py).",
                   SNAP_OR, [off, on], files=["patch_draft_compact.py", "identity_probe.py"],
                   run=["python3 /root/or/mx/patch_draft_compact.py"], max_minutes=75, budget=6.0)]


SNAP_SGL = {"name": "or-sglang0523rocm10-mi355x-c8m48d150", "image": "lmsysorg/sglang:v0.5.23-rocm10-mi35x",
            "image_digest_amd64": "sha256:cff45eca86da10fbe140f85df196ef241e932bbae8036afa45942e633c3ed6c5",
            "gpu": "MI355X", "gpus": 1, "cpu": 8, "mem": 48, "disk": 150}
W_FP8 = [{"repo": "RedHatAI/gemma-4-31B-it-FP8-dynamic", "rev": "d4ab4f579dd3516f97d8a6a4c98d0653480bad15", "dir": "/root/models/fp8dyn"}]


def sglang_track():
    """Maks: "perhaps try sglang instead of vllm?" (relayed by the lead, 11-22). Spec-off first, both engines, same client:
    SGLang 0.5.23 ROCm (gfx950 image, by digest) on the MXFP4 checkpoint (OpenRelay's README: sglang-rocm 0.5.20 could not
    load it), and on RedHat's FP8-dynamic if MXFP4 does not start; vLLM spec-off on OpenRelay's image as the reference.
    Prefill probe + seed 8 at 2.5/3.0/3.25 (early stop)."""
    rates = [2.5, 3.0, 3.25]
    sgl_common = ("--served-model-name gemma --host 127.0.0.1 --port 8000 --context-length 32768 --mem-fraction-static 0.88 "
                  "--kv-cache-dtype fp8_e4m3 --chunked-prefill-size 8192 --max-running-requests 64 --enable-metrics "
                  "--attention-backend aiter")
    env_sgl = {"SGLANG_USE_AITER": "1"}
    vb = "/root/vb/bin/vllm"
    def sgl_steps():
        return [gate(), dict(proto(rates, [8]), vllm_bin=vb)]
    s_mx = {"name": "sgl_mxfp4", "weights": WEIGHTS[:1], "served_name": "gemma", "env": env_sgl, "ready_timeout_s": 2400,
            "launch": f"python3 -m sglang.launch_server --model-path /root/models/mxfp4 {sgl_common}",
            "steps": [dict(INSPECT_STEP, name="inspect_sgl_mxfp4")] + sgl_steps(), "delete_weights_after": False,
            "allow_rejected": True, "claim_group": "sgl_mi355x_mxfp4_specoff", "origin": "ours"}
    s_fp8 = {"name": "sgl_fp8", "weights": W_FP8, "served_name": "gemma", "env": env_sgl, "ready_timeout_s": 2400,
             "launch": f"python3 -m sglang.launch_server --model-path /root/models/fp8dyn {sgl_common}",
             "steps": [dict(INSPECT_STEP, name="inspect_sgl_fp8")] + sgl_steps(), "delete_weights_after": True,
             "allow_rejected": True, "claim_group": "sgl_mi355x_fp8_specoff", "origin": "ours"}
    for s_ in (s_mx, s_fp8):   # gate/protocol names unique per server
        s_["steps"] = [dict(st, name=st["name"].replace("protocol_mx", f"protocol_{s_['name']}").replace("gate_", f"gate_{s_['name']}_"))
                       if st["type"] in ("protocol", "eval") else st for st in s_["steps"]]
    sgl = config("mxS1_sglang", "mx-sglang",
                 "SGLang track, spec OFF. lmsysorg/sglang:v0.5.23-rocm10-mi35x by digest (newest gfx950 tag that fits Daytona's 20 GB snapshot limit; rocm720 is 22.2 GB). Server 1: "
                 "amd/gemma-4-31B-it-MXFP4 (OpenRelay's checkpoint); server 2: RedHatAI/gemma-4-31B-it-FP8-dynamic. aiter attention, "
                 "FP8 KV, chunked prefill 8192, 64 running requests (vLLM recipe's values). Their client (client-only vLLM venv), "
                 "prefill probe, seed 8 at 2.5/3.0/3.25. allow_rejected: SGLang's log lacks vLLM's startup facts; backend read by hand.",
                 SNAP_SGL, [s_mx, s_fp8], max_minutes=90, budget=7.5)
    vll = config("mxS0_vllm_specoff", "mx-sglang",
                 "Reference for the SGLang track: OpenRelay's recipe with speculative decoding OFF (no --speculative-config), their image "
                 "by digest, everything else theirs. Prefill probe, seed 8 at 2.5/3.0/3.25 (early stop), GSM8K gate.",
                 SNAP_OR, [server("or_specoff", or_flags(speculative_config=None), ENV_OR, [gate(), proto(rates, [8])],
                                  claim_group="or_recipe_specoff", origin="ours")], max_minutes=45, budget=4.0)
    return [sgl, vll]


CPUMON = ("nohup bash -c 'while true; do echo T $(date -u +%FT%T) nproc $(nproc) loadavg $(cat /proc/loadavg); "
          "vmstat 1 2 | tail -1; sleep 4; done' > /root/or/out/cpumon.log 2>&1 &")


def small_host():
    """The build session's cost lever: host vCPU/RAM are billed on top of the GPU (8 vCPU / 48 GiB = $1.20/h of $4.76/h). Our
    recipe on 4 vCPU / 24 GiB ($4.17/h), both seeds. The load client runs on the same box, so CPU starvation would show as
    TTFT; a vmstat/loadavg sampler (out/cpumon.log) records runnable threads vs cores through the whole run."""
    snap = dict(SNAP_OR, name="or-openrelayrecipe-mi355x-c4m24d150", cpu=4, mem=24)
    out = []
    for seed in (8, 7):
        steps = [gate(), proto(SCREEN_RATES, [seed]), proto([3.25], [seed], name="protocol_mx_text", prefill=False, client="traffic"),
                 {"type": "sh", "name": "cpumon", "cmd": "tail -n 4 out/cpumon.log", "outputs": ["out/cpumon.log"], "est_s": 10}]
        out.append(config(f"mxH4_s{seed}", "mx-host",
                          f"Our recipe (OpenRelay's image + k=3 + --max-num-seqs 96 + graphs to 384 + MX_DRAFT_COMPACT=1) on a SMALL host: "
                          f"4 vCPU / 24 GiB (snapshot {snap['name']}, $4.17/h vs $4.76/h at 8 vCPU / 48 GiB). Their client, seed {seed}, "
                          "300 s, 3.25/3.5/3.75 + prefill probe + GSM8K gate; real text at 3.25. CPU sampler: out/cpumon.log.",
                          snap, [server("mx_recipe", best_flags(), dict(ENV_OR, MX_DRAFT_COMPACT="1"), steps,
                                        claim_group="mx_recipe_c4m24", origin="ours")],
                          files=["patch_draft_compact.py"], run=["python3 /root/or/mx/patch_draft_compact.py", CPUMON],
                          max_minutes=60, budget=4.5))
    return out


def confirm_text8(tag="C1"):
    """Column B (our real-text client: tokenizer-exact, stratified 9.5K/380 prompts) needs >= 2 seeds; the confirm boxes run it
    on seed 7 only. One more paired box: both recipes, seed 8, real text at 3.25 and 3.5."""
    txt = proto([3.25, 3.5], [8], name="protocol_mx_text", prefill=False, client="traffic")
    ours = server("mx_recipe", best_flags(), dict(ENV_OR, MX_DRAFT_COMPACT="1"), [gate(), txt], claim_group=f"mx_recipe_{tag}", origin="ours")
    theirs = server("or_recipe", or_flags(), ENV_OR, [gate(), txt], claim_group=f"or_recipe_paired_{tag}", origin="openrelay_recipe")
    return [config(f"mx{tag}_text_s8", "mx-confirm",
                   "CONFIRM (paired), column B only: our real-text client (traffic.py, --stratify --tokenizer: tokenizer-exact, "
                   "stratified 9.5K/380 prompts) at 3.25 and 3.5 req/s, seed 8, 300 s; the same MI355X serves ours then theirs. "
                   "Ours = OpenRelay's image + k=3 + --max-num-seqs 96 + graphs to 384 + MX_DRAFT_COMPACT=1. GSM8K gate (100) on each.",
                   SNAP_OR, [ours, theirs], files=["patch_draft_compact.py"], run=["python3 /root/or/mx/patch_draft_compact.py"],
                   max_minutes=50, budget=4.5)]


def wave12():
    """k=4 on top of the drafter compaction: with the first draft pass now cheap, does a 4th draft position add margin at 3.5?
    Seed 8, real-text client at 3.25/3.5 seed 7 alongside (k=4's extra position pays mostly on random tokens)."""
    txt = proto([3.25, 3.5], [7], name="protocol_mx_text", prefill=False, client="traffic")
    return [config("mx12_k4_dc", "mx12-screen",
                   "Current best + ONE CHANGE: num_speculative_tokens 4, graphs to 480 (96 x 5). OpenRelay's image + --max-num-seqs 96 + "
                   "MX_DRAFT_COMPACT=1. Their client, seed 8, 300 s, rates 3.25/3.5/3.75, prefill probe, GSM8K gate; real-text client "
                   "(stratified, tokenizer-exact) at 3.25/3.5 seed 7.",
                   SNAP_OR, [server("k4_dc", best_flags(speculative_config=mtp(4), compilation_config={"cudagraph_capture_sizes": cap_to(480)}),
                                    dict(ENV_OR, MX_DRAFT_COMPACT="1"), [gate(), proto(SCREEN_RATES, SCREEN_SEED), txt])],
                   files=["patch_draft_compact.py"], run=["python3 /root/or/mx/patch_draft_compact.py"], max_minutes=60, budget=5.5)]


def wave11():
    """Ablation: the drafter compaction alone on OpenRelay's unchanged recipe (k=2, 64 seqs), seed 8."""
    return [config("mx11_or_dc", "mx11-ablation",
                   "Ablation: OpenRelay's recipe exactly + ONE CHANGE: MX_DRAFT_COMPACT=1 (box/patch_draft_compact.py). Their client, "
                   "seed 8, 300 s, rates 3.25/3.5/3.75, prefill probe, GSM8K gate. Control: s2_mi355x_or_recipe_seed8.",
                   SNAP_OR, [server("or_dc", or_flags(), dict(ENV_OR, MX_DRAFT_COMPACT="1"), [gate(), proto(SCREEN_RATES, SCREEN_SEED)])],
                   files=["patch_draft_compact.py"], run=["python3 /root/or/mx/patch_draft_compact.py"])]

if __name__ == "__main__":
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    for w in (sys.argv[1:] or list(WAVES)):
        for c in WAVES[w]():
            print(write(c))
