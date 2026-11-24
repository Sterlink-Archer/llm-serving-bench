# Matrix `xcheck`

endpoint model `gemma`, started 2026-11-22T04:00:50Z, calibration {"template_overhead_tokens": 14, "cpt_prose": 4.067796610169491, "cpt_table": 1.7482517482517483}
SLO: {"ttft_p95_s": 2.0, "tpot_p50_ms": 50}

| cell | mode | in p50 | out p50 | req/s | out tok/s (server) | in tok/s | cache hit (server) | TTFT p50/p95/p99 s | TPOT p50/p95 ms | cold/warm TTFT p50 s | $/1M out | err | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1k_256_c32 | C=32 | 947 | 256 | 3.23 | 827 (827) | 3237 | 0.00 (0.00) | 0.16/0.28/0.29 | 38.1/40.3 | -/- | 1.232 | 0 | ok |

## Knees

| series | by | peak out tok/s @ | optimal (>=95% of peak) @ | SLO-best @ | max sustained rate |
|---|---|---|---|---|---|
