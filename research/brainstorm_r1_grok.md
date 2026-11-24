# Brainstorm round 1: independent Grok critique (grok-4.7, agent 36e2d4f0, 2026-11-22 ~03:3x-03:4x UTC)

Brief: read the playbook, 20_baseten/BENCHMARKS.md and the project README, then critique the plan. The facts given were the OpenRelay prices and method (verified by the lead), Daytona spot prices and a $150 cap.
The cap was later raised by Maks (re-plan near $500), so section (d)'s dollar limits are superseded by docs/PLAN.md.
The agent ran 773 s, cost $0.28, and used 772K tokens.
At the time of the review, README.md had already been switched to the OpenRelay bar, and the generator already had an open-loop Poisson mode. The memo predates or missed both.

---
# Memo: beating OpenRelay list price with Gemma 4 31B on spot GPUs

List revenue at 9.5K uncached input and 380 output is $0.000779 per request. Fully loaded spot break-even, GPU plus $0.81/h CPU, is 0.55 req/s on an RTX 5090, 0.70 on an L40S, 0.98 on an RTX PRO 6000, 1.17 on an H100, and 1.56 on an MI355X. Holding p95 TTFT under 4 s at that rate beats sync list. Their 3.0 req/s on one MI355X is efficiency.

README.md prices OpenRouter, and the generator is closed-loop. Their point is open-loop, 300 s, two seeds, worse kept. BENCHMARKS.md fixes concurrency and reports a knee. This demo needs goodput at an offered rate, so change the generator before any kernel hour.

Leave GLM-5.3-Flash out. About 184 GB at NVFP4 or 306 GB at FP8, four large GPUs, and a cold pull can spend the $3 pilot before a scored token.

## (a) Five cells

vLLM at the pinned digest. Public images until a log forces a patch.

1. RTX PRO 6000 first. NVFP4, FP8 KV, max length 16K, text-only, MTP off then 2 drafts. 96 GB of KV room, an FA4 path for the 512-dim global heads, and a bar of 0.98 req/s (about 9.3K input tokens/s offered).
2. RTX 5090, same stack. Bar 0.55 req/s (about 5.2K input tokens/s, 210 output tokens/s). 32 GB may hold only a few 10K prompts. The KV-token log decides. If they fit, this is the cheapest honest win.
3. H100, FP8 weights and FP8 KV, FlashInfer on the global heads. W4A16 takes Marlin: fast alone, slow in a batch. Add 2-token MTP only if the plain cell misses the SLO. Published speeds disagree by about 2×, so this is a screen at 1.17 req/s.
4. L40S, FP8 or Google QAT W4A16, only if the log says FlashInfer. Ada has no FP4. Triton means delete within minutes. The 0.70 req/s bar needs that kernel.
5. One later MI355X check, ledger healthy: MXFP4, 2-token MTP, AITER FP4 GEMM on, max batched tokens 8192, one 300 s seed at 1.6 req/s. Calibration against the blog, not the search.

Skip the 4090 (leftover KV is a few GB and a 2.5× guess), plus TP2, routing, DFlash, and EAGLE-3.

Use 2 draft tokens. They saw no gain from a third; still log acceptance here. Size CUDA graphs to batch × (1 + drafts). A cap near 64 idled their step. Capture to 256. Capture to 16K added memory and startup only.

## (b) Keeping the number fair

Print a price only after greedy GSM8K on that checkpoint clears 96.5% on a split we name. Their 98.8% protocol is unpublished. Smoke 30 items per screen cell. Run 200 once, on the winner. Rate tests may force 380 tokens. The exam must not.

Under $0.000779 per request at the SLO beats sync list. A 5090 at 0.6 req/s can do that at about a fifth of their per-GPU goodput. Show our req/s and a 16-way 10K prefill figure beside 3.0 req/s and 41.9K tokens/s.

Name the tier. Sync uncached is the headline. Batch at $0.035 / $0.15 is about $0.000390 a request, so the PRO 6000 bar moves from 0.98 to about 2.0 req/s.

Unique prompts on the headline, cached tokens taken from usage. Multi-turn is a second result: $0.007 on hits, $0.07 on misses. Their 3.0 req/s is the cold mix. One blended average lets the cache stand in for a kernel.

Report median and p95 TPOT with goodput. A TTFT pass at 80 ms median TPOT loses to their 22 ms. Quote the worse of seeds 7 and 8 over 300 s. Shorter runs only rank.

Publish steady-state $/token and measured $/token. Measured includes failed creates, pulls, and evictions. Tokens/s are the steady window only. Also publish payback minutes to repay create, pull, download, compile, and graphs. Weights reload every start, so a short life can lose while steady state wins. The CPU line is half a 5090 bill; dropping it about doubles that margin.

## (c) Generator, harness, meter

Sketches exist. This contract does not.

Open-loop, in the sandbox, streaming, token-exact 9.5K prompts, a unique nonce, cache cold, 380 forced tokens on the rate test. Log arrival, TTFT, inter-token time, and usage prompt / cached / completion counts. From BENCHMARKS.md: lengths came back at 77–84% of target, non-streaming hid TTFT, client rates on one cluster were stalls, and the first requests at a new concurrency were warmup. Quote the steady window only.

One sandbox, many cells. A create per config re-downloads weights. Stamp create-to-ready, image pull or cached, weight bytes and revision, process start to healthy, compile, graph capture, warmup, steady window, teardown. Reject a cell missing backend, KV tokens, or graph sizes. Delete in finally. Refresh the reaper deadline so the four-hour rule does not kill the quote.

Before create, add (GPU + CPU) × TTL to cumulative spend and refuse past the cap. After delete, reconcile Daytona-metered dollars with the estimate. Each cell prints overhead dollars, steady dollars, and the running total.

## (d) Halving inside the cap (cap superseded; see docs/PLAN.md)

Hard-stop at $120. Hold $30 for one Sunday rerun of the quoted cell.

Pilot, cap $3, TTL 45 min, one box. PRO 6000, NVFP4, MTP off: logged backend, logged KV count, 30 GSM8K, 120 s at 1.0 req/s. No spot there means the same pilot on a 5090.

Screen 5090, L40S, and H100: one quant, MTP off, 120 s at that card's break-even. Reuse the pilot box. Kill on Triton, errors over 1%, or p95 TTFT over 4 s. Keep the two lowest $/request. One retry each stays near $15.

Promote on the winner, same box, eight × 180 s. MTP 0 vs 2 at 0.75× and 1.25× break-even. Then max-seqs 32 vs 128. Then graphs to 256 if MTP is on. Then the other quant if it loads here. Keep a gain above 10% SLO goodput, or a flip of the TTFT pass.

Confirm, ten runs: rates 0.6, 0.9, 1.2, 1.6, 2.0 × break-even, seeds 7 and 8, 300 s, worse kept. Skip a rate the KV log cannot hold. Quote the highest SLO-valid rate. Then 200 GSM8K and one 16 × 10K prefill shot.

MI355X last, TTL 30 min, delete at 15 min if unhealthy, no retry.

That is 23 intervals and 4–6 creates when a box survives. A create per interval spends the cap on downloads. Only the two-seed cell is the result.

## (e) Day 1, and claims to drop

Before a second GPU, read the pilot log. Triton on the 512-dim layers makes MTP a loss. KV tokens, with max length 16K: the ~65K Qwen ladder in BENCHMARKS.md collapsed past the cache. Graph size versus draft width. Cached tokens at zero. Real prompt length 9.5K. Download and compile stamped off the steady window. Image and audio limits at zero, or validation can reject FlashInfer.

The MTP "2–2.5×, then 1.09× at concurrency 100" joins a single-stream H100 figure, short AMD GSM8K, a concurrency-16 blog, and one BF16 vendor point on two PRO 6000s. Acceptance at 9.5K is unmeasured. Start at 2 drafts and publish what we see.

LilaRest's NVFP4 speeds and inferencebench's 855–1,260 H100 tokens/s are another workload. JarvisLabs at concurrency 16, 375 output tokens/s, sits under the ~445 an H100 needs to break even. The SGLang B300 gain used simulated acceptance.

The 80 KiB/token figure and the 4090's 40–100K tokens are a labeled estimate with a 2.5× span. Plan from the log. The version paragraph disagrees with itself (0.29 unreleased and dated 9 Sep 2026, 0.32 via Wikipedia, MTP still nightly-only, README digest 0.33.0). Trust the digest and the log.

Input near 85% of revenue is a cold-cache fact. A shared prefix cuts revenue and prefill together. Keep the headline cold. Price $0.07 / $0.30. OpenRouter's $0.10 / $0.34, in the playbook and the README, is the wrong bar.

## (f) What would be new

Their blog is one point: 3.0 req/s, MI355X, after a night of kernels. Publish the knee on one RTX PRO 6000, NVFP4, backend copied from the log. Cold 9.5K/380 mix, draft width 0 and 2, graphs sized to that width. Plot SLO goodput, median TPOT, acceptance length, and fully loaded $/request against offered rate, and mark where the extra draft stops paying. Stack steady-state GPU, the CPU charge, cold-start payback, and eviction on the same figure.
