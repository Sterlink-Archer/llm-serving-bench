# OpenRouter endpoints for google/gemma-4-31b-it — snapshot 20261122T030326Z (18 endpoints)

| provider | quant | status | in $/1M | out $/1M | cache-read $/1M | ctx | max out |
|---|---|---|---|---|---|---|---|
| KT | unknown | 0 | 0.100 | 0.340 | 0.100 | 262144 | 235929 |
| CoreWeave | fp4 | 0 | 0.100 | 0.340 | 0.100 | 262144 | 235929 |
| Modal | unknown | 0 | 0.340 | 0.100 | 0.100 | 131072 | 117964 |
| NextBit | fp4 | 0 | 0.125 | 0.340 | 0.075 | 262144 | 235929 |
| Venice | fp4 | 0 | 0.120 | 0.360 | 0.090 | 256000 | 16384 |
| DeepInfra | fp8 | 0 | 0.160 | 0.320 |  | 262144 | 235929 |
| Chutes | fp4 | 0 | 0.120 | 0.370 | 0.012 | 131072 | 65536 |
| PrimeIntellect | fp8 | 0 | 0.200 | 0.335 | 0.100 | 262144 | 16384 |
| Crusoe | bf16 | 0 | 0.140 | 0.400 | 0.140 | 262144 | 262141 |
| Friendli | unknown | 0 | 0.140 | 0.400 |  | 262144 | 8192 |
| Novita | bf16 | 0 | 0.140 | 0.400 |  | 262144 | 131072 |
| Parasail | fp8 | 0 | 0.150 | 0.400 | 0.060 | 262144 | 235929 |
| DeepInfra | fp8 | 0 | 0.200 | 0.400 |  | 262144 | 16384 |
| DeepInfra | fp4 | -2 | 0.135 | 0.510 | 0.075 | 262144 | 16384 |
| DeepInfra | fp8 | 0 | 0.300 | 0.600 |  | 262144 | 235929 |
| SambaNova | unknown | 0 | 0.380 | 1.150 |  | 131072 | 117964 |
| ModelRun | fp4 | 0 | 0.750 | 1.000 | 0.200 | 262144 | 235929 |
| SiliconFlow | fp8 | 0 | 0.750 | 1.000 | 0.250 | 262144 | 235929 |

| workload (in/out tokens) | cheapest $/request | who (quant) | median $/request | cheapest bf16 | cheapest fp8 | cheapest 4-bit |
|---|---|---|---|---|---|---|
| 1k_in_1k_out (1024/1024) | $0.000451 | KT (unknown) | $0.000553 | $0.000553 (Crusoe) | $0.000492 (DeepInfra) | $0.000451 (CoreWeave) |
| 1k_in_256_out (1024/256) | $0.000189 | KT (unknown) | $0.000246 | $0.000246 (Crusoe) | $0.000246 (DeepInfra) | $0.000189 (CoreWeave) |
| 8k_in_1k_out (8192/1024) | $0.001167 | KT (unknown) | $0.001638 | $0.001556 (Crusoe) | $0.001638 (Parasail) | $0.001167 (CoreWeave) |
| 1k_in_8k_out (1024/8192) | $0.001167 | Modal (unknown) | $0.003420 | $0.003420 (Crusoe) | $0.002785 (DeepInfra) | $0.002888 (CoreWeave) |
| input_only_1M (1000000/0) | $0.100000 | KT (unknown) | $0.150000 | $0.140000 (Crusoe) | $0.150000 (Parasail) | $0.100000 (CoreWeave) |
| output_only_1M (0/1000000) | $0.100000 | Modal (unknown) | $0.400000 | $0.400000 (Crusoe) | $0.320000 (DeepInfra) | $0.340000 (CoreWeave) |
