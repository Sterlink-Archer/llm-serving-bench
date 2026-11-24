# Brainstorm round 2: independent Grok review of the interim screen (grok-4.7, agent 94833cc3, 2026-11-22 ~04:5x-05:1x UTC, 809 s, $0.35)

Input: docs/PLAN.md, docs/JOURNAL.md, playbook sections 2-5 and 9, plus the interim screen table as relayed by the lead.
Some of its points were already answered when it finished. The PRO 6000 log names `FlashInferCutlassNvFp4LinearKernel` (real W4A4 FP4), and a profile plus chunk-size diagnostic was launched at 04:53Z. Lead decisions are in docs/PLAN.md "Changes".

---
The overload account fits the H100 and the H200. It is ahead of the evidence as a rule for every NVIDIA card.

Offered prefill at break-even is 12.4K tok/s on the H100 and 12.8K on the H200. Shots are 12,767 and 13,166. A 120 s open loop at that load, with inputs uniform from 1.9K to 17.1K, drives TTFT into tens of seconds. H200 prefill is +3% over H100 despite 4.8 TB/s and 2.6× the KV. This prefill is compute-bound FP8 GEMM.

The PRO 6000 has not shown that NVFP4 ran. Shot 12,082 tok/s against 9.8K offered (~19% mean slack); the rung still did 0.79 req/s, p95 TTFT 10 s, TPOT 171 ms. A peak tie with H100 FP8 is plausible on SM120. Marlin W4A16, or FP4 GEMM left off, prints the same number. Attention is a minority of FLOPs at 10K on a dense 31B, and sliding layers cut it further, but a slow head-dim-512 kernel can still own the step. Hopper global layers were FA4. This box was FlashInfer.

One prefill step on that box settles the path. The linear-method line must name NVFP4 CUTLASS or FlashInfer FP4. Marlin or W4A16 means the FP4 cores are idle. `Using AttentionBackendEnum.FLASHINFER` names attention only. Profile one step: kernel names (`nvfp4` or cutlass fp4, versus marlin, fp16, or triton GEMM) and the time in GEMM, attention, and dequant. Tok/s at ~2K, 10K, and 16K: flat means GEMM-bound, a drop with length means the h512 kernel. Marlin, or NVFP4 matching an FP8 restart on the same GPU, means the journal's FP4-equals-H100-FP8 line is a fallback. An FP4 name and a flat sweep mean the tie is real.

B300 attempt 3 is that test on a larger chip: FA4, BF16 KV, MTP 2, after TRTLLM-GEN lacked a head-dim-512 context kernel and FA4 refused FP8 KV on SM103. BF16 KV barely moves compute-bound prefill. Near 12–18K tok/s, FP4 is idle. Above 41.9K beats their prefill; price still hinges on this box's own break-even.

Done req/s is not the shot. Those three rungs are ~60–70% of shot input tokens if completions have the mean length. Decode at the reported TPOTs is ~40–76 s, so with TTFT a request spans most of the 120 s rung and short jobs finish first. Prometheus prompt tok/s, required within 1% of the client, confirms or kills the shot.

Five runs, one warm box, under 30 minutes.

1. PRO 6000. GEMM name, one profiled step, and the 2K/10K/16K sweep, before any new flag. Flat and FP4-named: the plateau is the chip. Marlin, or tok/s falling with length: it is the kernel.
2. MI355X, before calling its req/s their 3.0. Confirm `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` and the 512-dim prefill tile, then their probe: 16×64 of 10K, twice, keep the second. Playbook: ~41.9K full path, ~27.7K tile missing, ~18.9K Triton GEMM. Public 0.33/0.34 can lack their vLLM 0.30 patches.
3. B300, speculation off, same probe, FA4 and BF16 KV held, plus CUDA-graph sizes. Well above 12K with an FP4 kernel name keeps the chip. Captures ending near 64 idle decode once MTP returns. Read startup facts before comparing TPOT.
4. MI355X, one rate, two sources, after run 2 is on the fast path. Random-token `vllm bench serve` versus real text, seed 7, a few minutes. Expect ~97–98% acceptance on random tokens. Real text should cut acceptance and goodput if decode helped make 3.0. Log acceptance, p95 TTFT, TPOT p50, server prompt tok/s.
5. PRO 6000 at 0.6× and 0.8× of 1.03 req/s, ~3 minutes, seed 7. TTFT under 4 s with prompt tok/s still ~12K means the 10 s / 171 ms point is the queue cliff and the card stays in the price hunt. Prompt tok/s ~7K means serving never hit the shot.

Their post binds prefill 27.8K→41.9K to MTP 2 (TPOT ~40–43 ms→22 ms, goodput 2.0→3.0). Random tokens inflate MTP, and shorter KV residency can lift goodput with it. A speculation-off prefill probe does not.

Column A: their client, 300 s, seeds 7 and 8, worse seed, second-run prefill, acceptance beside TPOT. Matching 3.0 reproduces them. Column B: the same ladder on real text. Price uses B. Clearing 1.69 req/s at p95 TTFT under 4 s on real text beats $0.000779 without seeing 3.0.

Beating the stack is column-A prefill above 41.9K, or higher column-B goodput, one diff at a time. Playbook leftovers are small: fused GELU+MXFP4 (~+1.8% prefill) and 8192 vs 16384 batched tokens on the probe only. Graphs to 256 and a third draft token are already theirs. Past 41.9K the candidate is B300 FP4, after run 3 names the kernel, at that box's loaded $/h.

H100 and H200 SLO misses can stand. These TTFT seconds cannot rank boxes: one seed, no steady window, a rung shorter than the latency. H200 TPOT is ~2× the H100 at the same prefill, so "8K chunks starve decode" is incomplete. KV (380K vs 145K) or noise is in that gap. The PRO 6000 fail is softer: the shot had slack, and host dollars sit inside 1.03 req/s. Their 41.9K is a long second probe. Sixteen prompts once is ~12 s. L40S is outside the plateau: 0.74 req/s offers ~7.0K tok/s, gate 98/100, KV 69,963. The 5090 outage is real (20,611 tokens, one sequence, Triton on global layers). Flags never set KV dtype. A B300 row carries MTP 2 and BF16 KV. Only its prefill shot matches the other rows.
