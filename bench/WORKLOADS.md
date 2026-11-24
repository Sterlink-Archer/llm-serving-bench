# Workloads and traffic for the serving matrix

Every serving config (engine x quantization x GPU x flags) is measured by the same matrix. A config is one
`configs/*.json`; each step `{"type": "matrix", "preset": "<name>"}` runs `bench/matrix.py` inside the sandbox, next to
the server. The presets are in `bench/presets/`.

## Workload classes (`bench/workloads.py`)

All request text is public or synthetic. The production-shaped classes replay only the **length and prefix shape** of
a real extraction pipeline's traffic: token-count quantiles built into `bench/workloads.py` (counts only). No production
prompt or document is used.

| class | what it models | input (tokens) | output (tokens) | shared prefix (cacheable) | shape source |
|---|---|---|---|---|---|
| `synthetic` | the standard sweep cell | `input_tokens` (128-32k) | `output_tokens` (16-2k), exact via `ignore_eos` | `prefix_ratio` x input, from `prefix_pool` distinct prefixes | knobs |
| `ciu_screen` | page screening: one web page x one item, yes/no JSON | sampled from 7,065 real reads: p50 1,707, p95 2,706, max 5,392 | sampled: p50 13, p95 94, max 1,136 | ~350-token item instruction, pool of 33 items | real read logs (counts) |
| `fin_extract` | financial-statement table extraction to quote-gated JSON | sampled from the pilot: 1.3k-24.7k, p50 19.5k (digit-heavy synthetic tables) | sampled: 196-7,598, p50 5,754 (cap `max_out`) | ~450-token instruction, shared by all | pilot usage (counts) |
| `console_chat` | data console: tool-calling chat | 10k-token system prompt + growing history | 100-800 (uniform; **assumed**, the console logs no reply lengths) | system prompt shared by every user; each turn's prompt is a prefix of the next | system prompt length |
| `chat_natural` | public chat (UltraChat first turns, MIT) | as written | natural (EOS honoured), cap 1,024 | none (nonce-tagged) | data/chat |
| `longdoc_qa` | public long-document QA (Wikipedia, CC BY-SA) | `doc_tokens` (8k-32k) + question | `output_tokens` | the document, pool of `docs`, several questions each | knobs |
| `mixed` | any weighted mix of the above in one stream | | | | `mix` weights |

## Traffic knobs (`bench/traffic.py`)

- **Closed loop** `--concurrency C`: C requests always in flight, so the server sets the pace. This finds the
  throughput knee.
- **Open loop** `--rate R`: Poisson arrivals at R req/s, seeded, whatever the server's speed. It gives latency at a
  given offered load and the highest sustainable rate. `--max-inflight` caps the client, and arrivals over the cap
  are counted as dropped.
- **Lengths:** fixed, uniform, or sampled from measured quantiles.
- **Cache control:** each prompt is [prefix] + [nonce + unique part] + [instruction]. A prefix's first use is
  labelled cold and its repeats warm, and TTFT is reported for each group.
- **Measurement:**
  - Every request streams, so each one has its own TTFT, TPOT and ITL.
  - Latencies report p50/p90/p95/p99. Throughput is aggregate input, output and cached-input tok/s over a steady
    window; requests still running at the window end are cut, so a slow tail cannot dilute the number.
  - Errors and drops are counted, along with the server-reported `cached_tokens`.
  - The server's own Prometheus counters (generated and prompt tokens, prefix-cache hits/queries, preemptions) are
    read at both window edges as a cross-check.
- **Calibration:** before a matrix, two probes with `max_tokens=1` fit characters per token for prose and for tables
  against the server's own `prompt_tokens`. Targets then land within about 1%. The Baseten harness's filler landed
  at 77-84%.

## Presets

| preset | cells | ~minutes | for |
|---|---|---|---|
| `smoke` | 8 | 4 | plumbing check, one cell per class |
| `xcheck` | 1 | 1.2 | same shape as `vllm bench serve` (1024/256, C=32), to validate our client against vLLM's |
| `quick` | 10 | 11 | comparing optimization configs: 1k/1k knee, 8k/1k, prefix 0 vs 90%, CIU at 16 req/s |
| `core` | 35 | 40 | the full picture for a finalist config: knees at 1k/1k and 8k/1k, 1k/16, 2k/2k, context sweep at C=1 (128-32k), prefix ratio 0/0.5/0.9/0.99, CIU open loop at 4/16/64 req/s plus C=128, fin extraction, console chat, public chat and long-doc QA |

## Knee and "optimal concurrency" (as in 20_baseten/BENCHMARKS.md)

Per series that sweeps C (or rate), `matrix_summary.md` reports:
- **peak:** the highest output tok/s;
- **optimal:** the smallest C reaching at least 95% of peak;
- **SLO-best:** the highest throughput meeting TTFT p95 ≤ 2 s (≤ 6 s for inputs over 4k) and TPOT p50 ≤ 50 ms;
- for rate series, **max sustained rate:** achieved ≥ 95% of offered, no drops, SLO met.

`stop_past_knee` skips a series' higher levels once one adds less than 3% throughput while TTFT p95 doubles.

## Turning an optimization idea into a matrix cell

Most items in an optimization list change the server, not the traffic. They become one more server block, i.e. one
more `launch` line in a config: quantization, KV dtype, max-num-seqs, chunked prefill or batched-token budget,
speculative decoding, prefix caching on/off, engine, GPU type and count. Each block runs the same preset, so every
config is compared cell for cell. Items that change the traffic itself (priority, batch-vs-sync mix, a context cap)
become a new series in a preset.

## Arrival pattern matters for TTFT (measured in the pilot)

`vllm bench serve --max-concurrency 32` starts all 32 requests at once. With equal output lengths they keep finishing
together, so prefills arrive in waves. In the pilot that gave a median TTFT of 1.26 s at 1024/256, C=32.

`bench/traffic.py` staggers closed-loop starts (random 0–2 s delay, random first output length). Prefills then
interleave with decode: median TTFT was 0.16 s. Output throughput agrees within 2.6% (827 vs 806 tok/s).
Source: `results/pilot_h100_vllm_fp8/20261122T034304Z/` (`vllm_bench_c32.json` vs `out/m_fp8dyn_xcheck`).

OpenRelay's protocol is **open-loop Poisson**, so its arrivals are staggered by construction, like ours. The burst
effect belongs only to fixed-concurrency runs started at once. For their protocol we run their exact client
(`bench/protocol.py --client vllmbench`), so the comparison does not depend on this.
