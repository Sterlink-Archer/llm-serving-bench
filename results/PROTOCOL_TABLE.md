Bar: OpenRelay sync Gemma 4 31B, $0.000779 per 9.5K-in/380-out request, unless the config names another (`price_model`, e.g. GLM 5.3 Flash, or `price_bar`, e.g. Inkling's OpenRouter price; see claims.json price_bar_name). ratio = our $/req at the max SLO rate / the bar's (< 1 = cheaper; it is also the break-even utilisation). SLO: p95 TTFT < 4 s, errors < 1%. $/h = Daytona's PUBLISHED rates (GPU + vCPU + RAM + disk); 'metered' = what Daytona actually billed that sandbox (results/metered_usd.json), shown alongside.

| run | gpu | server / protocol (client) | $/h | gate | attn sliding / global | KV tokens | break-even req/s | max req/s at SLO | our $/req | ratio vs OpenRelay | metered $/h | ratio (metered) | prefill tok/s | run $ (est) | end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cc_c1_s7/20261122T094451Z | MI355X | mx_recipe / protocol_mx_recipe_c1evict (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | - | 6.38 | ok |
| cc_c1_s7/20261122T094451Z | MI355X | mx_recipe / protocol_mx_recipe_c1fit (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 7.50 | 0.000176 | 0.23 | 4.26 | 0.20 | - | 6.38 | ok |
| cc_c1_s7/20261122T094451Z | MI355X | or_recipe / protocol_or_recipe_c1evict (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | - | 6.38 | ok |
| cc_c1_s7/20261122T094451Z | MI355X | or_recipe / protocol_or_recipe_c1fit (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 7.50 | 0.000176 | 0.23 | 4.26 | 0.20 | - | 6.38 | ok |
| cc_c1_s8/20261122T094453Z | MI355X | or_recipe / protocol_or_recipe_c1evict (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | - | 6.39 | ok |
| cc_c1_s8/20261122T094453Z | MI355X | or_recipe / protocol_or_recipe_c1fit (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 6.00 | 0.000220 | 0.28 | 4.26 | 0.25 | - | 6.39 | ok |
| cc_c1_s8/20261122T094453Z | MI355X | mx_recipe / protocol_mx_recipe_c1evict (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | - | 6.39 | ok |
| cc_c1_s8/20261122T094453Z | MI355X | mx_recipe / protocol_mx_recipe_c1fit (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 7.50 | 0.000176 | 0.23 | 4.26 | 0.20 | - | 6.39 | ok |
| cc_c2_s7/20261122T094455Z | MI355X | mx_recipe / protocol_mx_recipe_c2agent (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 7.50 | 0.000176 | 0.23 | 4.26 | 0.20 | - | 4.38 | ok |
| cc_c2_s7/20261122T094455Z | MI355X | or_recipe / protocol_or_recipe_c2agent (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 6.00 | 0.000220 | 0.28 | 4.26 | 0.25 | - | 4.38 | ok |
| cc_c2_s8/20261122T094457Z | MI355X | or_recipe / protocol_or_recipe_c2agent (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 6.00 | 0.000220 | 0.28 | 4.26 | 0.25 | - | 4.39 | ok |
| cc_c2_s8/20261122T094457Z | MI355X | mx_recipe / protocol_mx_recipe_c2agent (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 7.50 | 0.000176 | 0.23 | 4.26 | 0.20 | - | 4.39 | ok |
| cc_x2_s7/20261122T105237Z | MI355X | mx_recipe / protocol_mx_recipe_c1fit (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 8.00 | 0.000165 | 0.21 | 4.26 | 0.19 | - | 4.91 | ok |
| cc_x2_s7/20261122T105237Z | MI355X | mx_recipe / protocol_mx_recipe_c2agent (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 7.00 | 0.000189 | 0.24 | 4.26 | 0.22 | - | 4.91 | ok |
| cc_x2_s7/20261122T105237Z | MI355X | or_recipe / protocol_or_recipe_c1fit (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 7.00 | 0.000189 | 0.24 | 4.26 | 0.22 | - | 4.91 | ok |
| cc_x2_s7/20261122T105237Z | MI355X | or_recipe / protocol_or_recipe_c2agent (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 7.00 | 0.000189 | 0.24 | 4.26 | 0.22 | - | 4.91 | ok |
| cc_x2_s8/20261122T105239Z | MI355X | or_recipe / protocol_or_recipe_c1fit (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 0.00 | - | - | 4.26 | - | - | 4.90 | ok |
| cc_x2_s8/20261122T105239Z | MI355X | or_recipe / protocol_or_recipe_c2agent (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 7.00 | 0.000189 | 0.24 | 4.26 | 0.22 | - | 4.90 | ok |
| cc_x2_s8/20261122T105239Z | MI355X | mx_recipe / protocol_mx_recipe_c1fit (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 7.00 | 0.000189 | 0.24 | 4.26 | 0.22 | - | 4.90 | ok |
| cc_x2_s8/20261122T105239Z | MI355X | mx_recipe / protocol_mx_recipe_c2agent (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 7.00 | 0.000189 | 0.24 | 4.26 | 0.22 | - | 4.90 | ok |
| cc_x_s7/20261122T102748Z | MI355X | mx_recipe / protocol_mx_recipe_c1fit (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 0.00 | - | - | 4.26 | - | - | 2.44 | ok |
| cc_x_s7/20261122T102748Z | MI355X | mx_recipe / protocol_mx_recipe_c2agent (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 0.00 | - | - | 4.26 | - | - | 2.44 | ok |
| cc_x_s7/20261122T102748Z | MI355X | or_recipe / protocol_or_recipe_c1fit (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 6.75 | 0.000196 | 0.25 | 4.26 | 0.22 | - | 2.44 | ok |
| cc_x_s7/20261122T102748Z | MI355X | or_recipe / protocol_or_recipe_c2agent (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 6.75 | 0.000196 | 0.25 | 4.26 | 0.22 | - | 2.44 | ok |
| cc_x_s8/20261122T102750Z | MI355X | or_recipe / protocol_or_recipe_c1fit (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 6.75 | 0.000196 | 0.25 | 4.25 | 0.22 | - | 2.48 | ok |
| cc_x_s8/20261122T102750Z | MI355X | or_recipe / protocol_or_recipe_c2agent (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 6.75 | 0.000196 | 0.25 | 4.25 | 0.22 | - | 2.48 | ok |
| cc_x_s8/20261122T102750Z | MI355X | mx_recipe / protocol_mx_recipe_c1fit (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 0.00 | - | - | 4.25 | - | - | 2.48 | ok |
| cc_x_s8/20261122T102750Z | MI355X | mx_recipe / protocol_mx_recipe_c2agent (traffic) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 0.00 | - | - | 4.25 | - | - | 2.48 | ok |
| d3_sglang_counter_probe/20261122T072959Z | RTX-PRO-6000 | sgl_nvfp4_nv / protocol_sgl_counter_B_text (traffic) | 2.89 | - - | triton / triton | 148789 | 1.03 | 0.00 | - | - | 2.49 | - | - | 0.57 | ok |
| e4_b300_sglang/20261122T055226Z | B300 | sgl_nvfp4_nv / protocol_sgl_b300_A_random (vllmbench) | 6.64 | 0.990 PASS | trtllm_mha / trtllm_mha | 591296 | 2.37 | 2.37 | 0.000779 | 1.00 | 6.02 | 0.91 | 34512 | 4.33 | ok |
| e4_b300_sglang/20261122T055226Z | B300 | sgl_nvfp4_nv / protocol_sgl_b300_B_text (traffic) | 6.64 | 0.990 PASS | trtllm_mha / trtllm_mha | 591296 | 2.37 | 2.37 | 0.000779 | 1.00 | 6.02 | 0.91 | - | 4.33 | ok |
| e4_rtxpro6000_sglang/20261122T052658Z | RTX-PRO-6000 | sgl_nvfp4_nv / protocol_sgl_nvfp4_nv_B_text (traffic) | 2.89 | 0.990 PASS | triton / triton | 148789 | 1.03 | 0.00 | - | - | 2.50 | - | - | 0.62 | ok |
| e4a_rtxpro6000_sglang_colA/20261122T054545Z | RTX-PRO-6000 | sgl_nvfp4_nv / protocol_sgl_nvfp4_nv_A_random (vllmbench) | 2.89 | 0.980 PASS | triton / triton | 148789 | 1.03 | 0.00 | - | - | 2.50 | - | 6193 | 1.16 | ok |
| e5_b300_sglang_textonly/20261122T090736Z | B300 | sgl_rh_nvfp4_textonly / protocol_sgl_rh_nvfp4_textonly_A_random (vllmbench) | 6.64 | 0.990 PASS | trtllm_mha / trtllm_mha | 628288 | 2.37 | 2.96 | 0.000623 | 0.80 | 6.03 | 0.73 | 49032 | 4.96 | ok |
| e5_b300_sglang_textonly/20261122T090736Z | B300 | sgl_rh_nvfp4_textonly / protocol_sgl_rh_nvfp4_textonly_B_text (traffic) | 6.64 | 0.990 PASS | trtllm_mha / trtllm_mha | 628288 | 2.37 | 2.96 | 0.000623 | 0.80 | 6.03 | 0.73 | - | 4.96 | ok |
| e5_b300_sglang_textonly_s8/20261122T115147Z | B300 | sgl_rh_nvfp4_textonly / protocol_sgl_rh_nvfp4_textonly_A_random (vllmbench) | 6.64 | 0.990 PASS | trtllm_mha / trtllm_mha | 628288 | 2.37 | 2.96 | 0.000623 | 0.80 | 6.02 | 0.73 | 50530 | 4.88 | ok |
| e5_b300_sglang_textonly_s8/20261122T115147Z | B300 | sgl_rh_nvfp4_textonly / protocol_sgl_rh_nvfp4_textonly_B_text (traffic) | 6.64 | 0.990 PASS | trtllm_mha / trtllm_mha | 628288 | 2.37 | 2.96 | 0.000623 | 0.80 | 6.02 | 0.73 | - | 4.88 | ok |
| glm1_sglang0523/20261122T074555Z | MI355X | glm_sgl / protocol_glm_sgl (vllmbench) | 4.77 | 0.980 PASS | dsa / dsa | 6273536 | 1.70 | 1.00 | 0.001326 | 2.67 | 4.26 | 2.39 | 21273 | 2.44 | ok |
| glm1_vllm034/20261122T073206Z | MI355X | glm_vllm / protocol_glm_vllm (vllmbench) | 4.77 | 0.990 PASS | ROCM_AITER_MLA_SPARSE / ROCM_AITER_MLA_SPARSE | 4383201 | 1.70 | 1.50 | 0.000884 | 1.78 | 4.26 | 1.59 | 21668 | 2.26 | ok |
| glm2_vllm_b16k/20261122T075350Z | MI355X | glm_b16k / protocol_glm_b16k (vllmbench) | 4.77 | 0.976 PASS | ROCM_AITER_MLA_SPARSE / ROCM_AITER_MLA_SPARSE | 4096000 | 1.70 | 1.50 | 0.000884 | 1.78 | 4.26 | 1.59 | 24424 | 2.21 | ok |
| glm2_vllm_mtp3/20261122T074424Z | MI355X | glm_mtp3 / protocol_glm_mtp3 (vllmbench) | 4.77 | 1.000 PASS | ROCM_AITER_MLA_SPARSE / ROCM_AITER_MLA_SPARSE | 2554538 | 1.70 | 1.50 | 0.000884 | 1.78 | 4.26 | 1.59 | 21224 | 2.69 | ok |
| glm2_vllm_mtp3/20261122T074424Z | MI355X | glm_mtp3 / protocol_glm_mtp3_text (traffic) | 4.77 | 1.000 PASS | ROCM_AITER_MLA_SPARSE / ROCM_AITER_MLA_SPARSE | 2554538 | 1.70 | 1.50 | 0.000884 | 1.78 | 4.26 | 1.59 | - | 2.69 | ok |
| glm2_vllm_mtp5/20261122T074426Z | MI355X | glm_mtp5 / protocol_glm_mtp5 (vllmbench) | 4.77 | 0.980 PASS | ROCM_AITER_MLA_SPARSE / ROCM_AITER_MLA_SPARSE | 2035985 | 1.70 | 1.00 | 0.001326 | 2.67 | 4.26 | 2.39 | 21298 | 2.77 | ok |
| glm2_vllm_mtp5/20261122T074426Z | MI355X | glm_mtp5 / protocol_glm_mtp5_text (traffic) | 4.77 | 0.980 PASS | ROCM_AITER_MLA_SPARSE / ROCM_AITER_MLA_SPARSE | 2035985 | 1.70 | 1.50 | 0.000884 | 1.78 | 4.26 | 1.59 | - | 2.77 | ok |
| glm4_best_s7/20261122T082933Z | MI355X | glm_best / protocol_glm_best (vllmbench) | 4.77 | 0.976 PASS | ROCM_AITER_MLA_SPARSE / ROCM_AITER_MLA_SPARSE | 2330624 | 2.67 | 1.50 | 0.000884 | 1.78 | 4.27 | 1.59 | 23676 | 3.25 | ok |
| glm4_best_s7/20261122T082933Z | MI355X | glm_best / protocol_glm_best_text (traffic) | 4.77 | 0.976 PASS | ROCM_AITER_MLA_SPARSE / ROCM_AITER_MLA_SPARSE | 2330624 | 2.67 | 1.50 | 0.000884 | 1.78 | 4.27 | 1.59 | - | 3.25 | ok |
| glm4_best_s8/20261122T082931Z | MI355X | glm_best / protocol_glm_best (vllmbench) | 4.77 | 0.984 PASS | ROCM_AITER_MLA_SPARSE / ROCM_AITER_MLA_SPARSE | 2330624 | 2.67 | 1.50 | 0.000884 | 1.78 | 4.26 | 1.59 | 23665 | 3.28 | ok |
| glm4_best_s8/20261122T082931Z | MI355X | glm_best / protocol_glm_best_text (traffic) | 4.77 | 0.984 PASS | ROCM_AITER_MLA_SPARSE / ROCM_AITER_MLA_SPARSE | 2330624 | 2.67 | 1.75 | 0.000758 | 1.53 | 4.26 | 1.37 | - | 3.28 | ok |
| ik1_5090_sgl/20261122T071045Z | RTX-5090 | sgl / protocol_sgl_A_random (vllmbench) | 1.68 | 0.970 PASS | fa4 / fa4 | 118272 | 0.56 | 0.56 | 0.000826 | 1.00 | 1.29 | 0.77 | 17264 | 0.62 | ok |
| ik1_5090_sgl/20261122T071045Z | RTX-5090 | sgl / protocol_sgl_B_text (traffic) | 1.68 | 0.970 PASS | fa4 / fa4 | 118272 | 0.56 | 0.00 | - | - | 1.29 | - | - | 0.62 | ok |
| ik1_h100_sgl/20261122T071049Z | H100 | sgl / protocol_sgl_A_random (vllmbench) | 3.41 | 0.970 PASS | fa4 / fa4 | 937472 | 1.15 | 1.15 | 0.000827 | 1.00 | 3.02 | 0.89 | 18314 | 1.26 | ok |
| ik1_h100_sgl/20261122T071049Z | H100 | sgl / protocol_sgl_B_text (traffic) | 3.41 | 0.970 PASS | fa4 / fa4 | 937472 | 1.15 | 0.00 | - | - | 3.02 | - | - | 1.26 | ok |
| ik1_mi355x_sgl/20261122T071051Z | MI355X | sgl / protocol_sgl_A_random (vllmbench) | 4.50 | 0.970 PASS | triton / triton | 4828288 | 1.51 | 2.27 | 0.000551 | 0.67 | 4.11 | 0.61 | 27599 | 1.58 | ok |
| ik1_mi355x_sgl/20261122T071051Z | MI355X | sgl / protocol_sgl_B_text (traffic) | 4.50 | 0.970 PASS | triton / triton | 4828288 | 1.51 | 1.51 | 0.000827 | 1.00 | 4.11 | 0.91 | - | 1.58 | ok |
| ik1_pro6000_sgl/20261122T071047Z | RTX-PRO-6000 | sgl / protocol_sgl_A_random (vllmbench) | 2.89 | 0.970 PASS | fa4 / fa4 | 1250560 | 0.97 | 1.46 | 0.000551 | 0.67 | 2.50 | 0.58 | 20827 | 1.05 | ok |
| ik1_pro6000_sgl/20261122T071047Z | RTX-PRO-6000 | sgl / protocol_sgl_B_text (traffic) | 2.89 | 0.970 PASS | fa4 / fa4 | 1250560 | 0.97 | 0.00 | - | - | 2.50 | - | - | 1.05 | ok |
| ik1_pro6000_vllm/20261122T071055Z | RTX-PRO-6000 | vllm / protocol_vllm_A_random (vllmbench) | 2.89 | 0.980 PASS | None / None | 142607 | 0.97 | 0.00 | - | - | 2.50 | - | 23740 | 0.82 | ok |
| ik1_pro6000_vllm/20261122T071055Z | RTX-PRO-6000 | vllm / protocol_vllm_B_text (traffic) | 2.89 | 0.980 PASS | None / None | 142607 | 0.97 | 0.00 | - | - | 2.50 | - | - | 0.82 | ok |
| ik2_pro6000_sched/20261122T075734Z | RTX-PRO-6000 | swa15 / protocol_swa15_B_text (traffic) | 2.89 | 0.970 PASS | fa4 / fa4 | 1071872 | 0.97 | 0.97 | 0.000827 | 1.00 | 2.50 | 0.86 | - | 1.91 | ok |
| ik2_pro6000_sched/20261122T075734Z | RTX-PRO-6000 | swa15_mixed / protocol_swa15_mixed_B_text (traffic) | 2.89 | 0.970 PASS | fa4 / fa4 | 1071872 | 0.97 | 0.97 | 0.000827 | 1.00 | 2.50 | 0.86 | - | 1.91 | ok |
| ik2_pro6000_sched/20261122T075734Z | RTX-PRO-6000 | swa15_chunk16k / protocol_swa15_chunk16k_B_text (traffic) | 2.89 | 0.970 PASS | fa4 / fa4 | 868096 | 0.97 | 0.97 | 0.000827 | 1.00 | 2.50 | 0.86 | - | 1.91 | ok |
| ik2_pro6000_spec/20261122T075736Z | RTX-PRO-6000 | swa15 / protocol_swa15_B_text (traffic) | 2.89 | 0.970 PASS | fa4 / fa4 | 1071872 | 0.97 | 0.97 | 0.000827 | 1.00 | 2.50 | 0.87 | - | 0.90 | ok |
| ik2_pro6000_swa/20261122T075732Z | RTX-PRO-6000 | base / protocol_base_A_random (vllmbench) | 2.89 | 0.960 PASS | fa4 / fa4 | 1250560 | 0.97 | 0.97 | 0.000827 | 1.00 | 2.50 | 0.87 | 20837 | 1.47 | ok |
| ik2_pro6000_swa/20261122T075732Z | RTX-PRO-6000 | base / protocol_base_B_text (traffic) | 2.89 | 0.960 PASS | fa4 / fa4 | 1250560 | 0.97 | 0.97 | 0.000827 | 1.00 | 2.50 | 0.87 | - | 1.47 | ok |
| ik2_pro6000_swa/20261122T075732Z | RTX-PRO-6000 | swa15 / protocol_swa15_B_text (traffic) | 2.89 | 0.980 PASS | fa4 / fa4 | 1071872 | 0.97 | 0.97 | 0.000827 | 1.00 | 2.50 | 0.87 | - | 1.47 | ok |
| ik3_5090_bf16/20261122T075738Z | RTX-5090 | swa25 / protocol_swa25_B_text (traffic) | 1.68 | 0.970 PASS | fa4 / fa4 | 78848 | 0.56 | 0.00 | - | - | 1.29 | - | - | 0.79 | ok |
| ik3_5090_bf16/20261122T075738Z | RTX-5090 | swa25_nopg / protocol_swa25_nopg_B_text (traffic) | 1.68 | 0.970 PASS | fa4 / fa4 | 83712 | 0.56 | 0.00 | - | - | 1.29 | - | - | 0.79 | ok |
| ik4_mi355x_aiter/20261122T075742Z | MI355X | base / protocol_base_B_text (traffic) | 4.50 | 0.970 PASS | triton / triton | 4828288 | 1.51 | 0.00 | - | - | 4.11 | - | - | 3.07 |  |
| ik5_pro6000_dspark/20261122T081506Z | RTX-PRO-6000 | base / protocol_base_B_text (traffic) | 2.89 | 0.970 PASS | fa4 / fa4 | 1250560 | 0.97 | 0.97 | 0.000827 | 1.00 | 2.50 | 0.87 | - | 1.20 | ok |
| ik5_pro6000_dspark/20261122T081506Z | RTX-PRO-6000 | dspark / protocol_dspark_B_text (traffic) | 2.89 | 0.980 PASS | fa4 / fa4 | 1003136 | 0.97 | 0.97 | 0.000827 | 1.00 | 2.50 | 0.87 | - | 1.20 | ok |
| ik7_mi355x_aiter/20261122T084023Z | MI355X | base / protocol_base_B_text (traffic) | 4.50 | 0.980 PASS | triton / triton | 4828288 | 1.51 | 0.00 | - | - | 4.11 | - | - | 1.16 | ok |
| ik8_5090_c4/20261122T085129Z | RTX-5090 | mem81 / protocol_mem81_B_text (traffic) | 1.35 | 0.980 PASS | fa4 / fa4 | 138368 | 0.45 | 0.45 | 0.000827 | 1.00 | 1.09 | 0.81 | - | 0.77 | ok |
| ik8_5090_c4/20261122T085129Z | RTX-5090 | mem81_mixed / protocol_mem81_mixed_B_text (traffic) | 1.35 | 0.970 PASS | fa4 / fa4 | 138368 | 0.45 | 0.45 | 0.000827 | 1.00 | 1.09 | 0.81 | - | 0.77 | ok |
| ik8_pro6000_c4/20261122T084149Z | RTX-PRO-6000 | mixed / protocol_mixed_B_text (traffic) | 2.56 | 0.960 PASS | fa4 / fa4 | 1250560 | 0.86 | 0.97 | 0.000731 | 0.88 | 2.30 | 0.80 | - | 1.83 | ok |
| ik8_pro6000_c4/20261122T084149Z | RTX-PRO-6000 | mixed_dspark / protocol_mixed_dspark_B_text (traffic) | 2.56 | 0.970 PASS | fa4 / fa4 | 1003136 | 0.86 | 1.07 | 0.000661 | 0.80 | 2.30 | 0.72 | - | 1.83 | ok |
| ikC_b300_mixed_dspark/20261122T100257Z | B300 | mixed_dspark / protocol_b300_A_random (vllmbench) | 6.11 | 0.968 PASS | fa4 / fa4 | 3710720 | 2.05 | 2.05 | 0.000826 | 1.00 | 5.72 | 0.94 | 44850 | 10.57 | ok |
| ikC_b300_mixed_dspark/20261122T100257Z | B300 | mixed_dspark / protocol_b300_B_text (traffic) | 6.11 | 0.968 PASS | fa4 / fa4 | 3710720 | 2.05 | 3.08 | 0.000551 | 0.67 | 5.72 | 0.62 | - | 10.57 | ok |
| ikC_b300_mixed_dspark/20261122T100257Z | B300 | mixed_dspark / protocol_mixed_dspark_b300ext_B_text (traffic) | 6.11 | 0.968 PASS | fa4 / fa4 | 3710720 | 2.05 | 0.00 | - | - | 5.72 | - | - | 10.57 | ok |
| ikC_pro6000_mixed_dspark_rep_s7/20261122T103726Z | RTX-PRO-6000 | mixed_dspark / protocol_rep_B_text (traffic) | 2.56 | 0.970 PASS | fa4 / fa4 | 1003136 | 0.86 | 1.29 | 0.000551 | 0.67 | 2.30 | 0.60 | - | 0.79 | ok |
| ikC_pro6000_mixed_dspark_rep_s8/20261122T103728Z | RTX-PRO-6000 | mixed_dspark / protocol_rep_B_text (traffic) | 2.56 | 0.970 PASS | fa4 / fa4 | 1003136 | 0.86 | 1.29 | 0.000551 | 0.67 | 2.30 | 0.60 | - | 0.86 | ok |
| ikC_pro6000_mixed_dspark_s7/20261122T092529Z | RTX-PRO-6000 | mixed_dspark / protocol_mixed_dspark_A_random (vllmbench) | 2.56 | 0.968 PASS | fa4 / fa4 | 1003136 | 0.86 | 1.29 | 0.000551 | 0.67 | 2.30 | 0.60 | 20367 | 2.56 | ok |
| ikC_pro6000_mixed_dspark_s7/20261122T092529Z | RTX-PRO-6000 | mixed_dspark / protocol_mixed_dspark_B_text (traffic) | 2.56 | 0.968 PASS | fa4 / fa4 | 1003136 | 0.86 | 1.29 | 0.000551 | 0.67 | 2.30 | 0.60 | - | 2.56 | ok |
| ikC_pro6000_mixed_dspark_s8/20261122T092531Z | RTX-PRO-6000 | mixed_dspark / protocol_mixed_dspark_A_random (vllmbench) | 2.56 | 0.972 PASS | fa4 / fa4 | 1003136 | 0.86 | 1.16 | 0.000612 | 0.74 | 2.30 | 0.67 | 20381 | 2.63 | ok |
| ikC_pro6000_mixed_dspark_s8/20261122T092531Z | RTX-PRO-6000 | mixed_dspark / protocol_mixed_dspark_B_text (traffic) | 2.56 | 0.972 PASS | fa4 / fa4 | 1003136 | 0.86 | 1.29 | 0.000551 | 0.67 | 2.30 | 0.60 | - | 2.63 | ok |
| ikC_pro6000_mixed_s7/20261122T091132Z | RTX-PRO-6000 | mixed / protocol_mixed_A_random (vllmbench) | 2.56 | 0.964 PASS | fa4 / fa4 | 1250560 | 0.86 | 1.07 | 0.000661 | 0.80 | 2.30 | 0.72 | 20640 | 3.31 | ok |
| ikC_pro6000_mixed_s7/20261122T091132Z | RTX-PRO-6000 | mixed / protocol_mixed_B_text (traffic) | 2.56 | 0.964 PASS | fa4 / fa4 | 1250560 | 0.86 | 1.07 | 0.000661 | 0.80 | 2.30 | 0.72 | - | 3.31 | ok |
| ikC_pro6000_mixed_s8/20261122T091134Z | RTX-PRO-6000 | mixed / protocol_mixed_A_random (vllmbench) | 2.56 | 0.968 PASS | fa4 / fa4 | 1250560 | 0.86 | 1.07 | 0.000661 | 0.80 | 2.30 | 0.72 | 20599 | 2.30 | ok |
| ikC_pro6000_mixed_s8/20261122T091134Z | RTX-PRO-6000 | mixed / protocol_mixed_B_text (traffic) | 2.56 | 0.968 PASS | fa4 / fa4 | 1250560 | 0.86 | 1.07 | 0.000661 | 0.80 | 2.30 | 0.72 | - | 2.30 | ok |
| mx10_best_dc_s7/20261122T060316Z | MI355X | best_dc / protocol_best_dc (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | - | 1.72 | ok |
| mx10_best_dc_s8/20261122T060314Z | MI355X | best_dc / protocol_best_dc (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | 43287 | 1.80 | ok |
| mx11_or_dc/20261122T062018Z | MI355X | or_dc / protocol_or_dc (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 44463 | 1.80 | ok |
| mx12_k4_dc/20261122T062613Z | MI355X | k4_dc / protocol_k4_dc (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 972265 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | 43256 | 2.61 | ok |
| mx12_k4_dc/20261122T062613Z | MI355X | k4_dc / protocol_k4_dc_text (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 972265 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | - | 2.61 | ok |
| mx1_or_b4k/20261122T050156Z | MI355X | or_b4096 / protocol_or_b4096 (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 1615920 | 1.70 | 0.00 | - | - | 4.25 | - | 40768 | 1.39 | ok |
| mx1_or_b6k/20261122T050158Z | MI355X | or_b6144 / protocol_or_b6144 (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 1217931 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.25 | 0.47 | 41530 | 1.37 | ok |
| mx1_or_mtp3/20261122T050200Z | MI355X | or_mtp3 / protocol_or_mtp3 (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 976955 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 42886 | 2.64 | ok |
| mx1_or_mtp3/20261122T050200Z | MI355X | or_mtp3 / protocol_or_mtp3_text (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 976955 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | - | 2.64 | ok |
| mx1_v034/20261122T050152Z | MI355X | v034_stock / protocol_v034_stock (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 1061998 | 1.70 | 0.00 | - | - | 4.26 | - | 24499 | 1.11 | ok |
| mx2_v034_asm/20261122T050728Z | MI355X | v034_asm / protocol_v034_asm (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 1062402 | 1.70 | 0.00 | - | - | 4.25 | - | 41910 | 1.33 | ok |
| mx2_v034_orport/20261122T050730Z | MI355X | v034_orport / protocol_v034_orport (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 1062160 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.25 | 0.47 | 42925 | 1.39 | ok |
| mx3_or_seqs128/20261122T051339Z | MI355X | or_seqs128 / protocol_or_seqs128 (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974125 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.25 | 0.47 | 41582 | 1.37 | ok |
| mx3_or_seqs96/20261122T051337Z | MI355X | or_seqs96 / protocol_or_seqs96 (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 975904 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 43320 | 1.82 | ok |
| mx4_or_b12k/20261122T051936Z | MI355X | or_b12k / protocol_or_b12k (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 699910 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 43002 | 1.35 | ok |
| mx5_or_mtp3_seqs96/20261122T052526Z | MI355X | or_mtp3_seqs96 / protocol_or_mtp3_seqs96 (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 42726 | 1.82 | ok |
| mx5_or_mtp4/20261122T052524Z | MI355X | or_mtp4 / protocol_or_mtp4 (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 975398 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 42567 | 2.66 | ok |
| mx5_or_mtp4/20261122T052524Z | MI355X | or_mtp4 / protocol_or_mtp4_text (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 975398 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | - | 2.66 | ok |
| mx6_or_mtp3_seqs112/20261122T054308Z | MI355X | or_mtp3_seqs112 / protocol_or_mtp3_seqs112 (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 972973 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 41303 | 1.85 | ok |
| mx6_or_mtp3_seqs80/20261122T054926Z | MI355X | or_mtp3_seqs80 / protocol_or_mtp3_seqs80 (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 975317 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 43154 | 1.80 | ok |
| mx6_or_mtp3_seqs96_s7/20261122T054311Z | MI355X | or_mtp3_seqs96 / protocol_or_mtp3_seqs96 (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.25 | 0.43 | 43521 | 1.84 | ok |
| mx6_or_mtp4_seqs96/20261122T054304Z | MI355X | or_mtp4_seqs96 / protocol_or_mtp4_seqs96 (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 972265 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 41806 | 1.85 | ok |
| mx8_or_mtp3_seqs96_block/20261122T054943Z | MI355X | or_mtp3_seqs96_block / protocol_or_mtp3_seqs96_block (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.25 | 0.47 | 42680 | 2.58 | ok |
| mx8_or_mtp3_seqs96_block/20261122T054943Z | MI355X | or_mtp3_seqs96_block / protocol_or_mtp3_seqs96_block_text (traffic) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.25 | 0.47 | - | 2.58 | ok |
| mx8_or_mtp3_seqs96_prob/20261122T054945Z | MI355X | or_mtp3_seqs96_prob / protocol_or_mtp3_seqs96_prob (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 973721 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 41782 | 2.61 | ok |
| mx8_or_mtp3_seqs96_prob/20261122T054945Z | MI355X | or_mtp3_seqs96_prob / protocol_or_mtp3_seqs96_prob_text (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 973721 | 1.70 | 3.00 | 0.000440 | 0.57 | 4.26 | 0.51 | - | 2.61 | ok |
| mx9_best_r2s7/20261122T060050Z | MI355X | best / protocol_best (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 0.00 | - | - | 4.26 | - | - | 1.43 | ok |
| mx9_best_r2s8/20261122T060046Z | MI355X | best / protocol_best (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 0.00 | - | - | 4.26 | - | - | 1.43 | ok |
| mx9_best_r3s8/20261122T060048Z | MI355X | best / protocol_best (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 0.00 | - | - | 4.26 | - | - | 1.41 | ok |
| mxC1_s7_r1/20261122T061956Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.25 | 0.43 | 44437 | 4.57 | ok |
| mxC1_s7_r1/20261122T061956Z | MI355X | mx_recipe / protocol_mx_recipe_text (traffic) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.25 | 0.47 | - | 4.57 | ok |
| mxC1_s7_r1/20261122T061956Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.25 | 0.47 | 42867 | 4.57 | ok |
| mxC1_s7_r1/20261122T061956Z | MI355X | or_recipe / protocol_or_recipe_text (traffic) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 0.00 | - | - | 4.25 | - | - | 4.57 | ok |
| mxC1_s7_r2/20261122T061958Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 42158 | 4.58 | ok |
| mxC1_s7_r2/20261122T061958Z | MI355X | or_recipe / protocol_or_recipe_text (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 0.00 | - | - | 4.26 | - | - | 4.58 | ok |
| mxC1_s7_r2/20261122T061958Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.76 | 0.984 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | 43635 | 4.58 | ok |
| mxC1_s7_r2/20261122T061958Z | MI355X | mx_recipe / protocol_mx_recipe_text (traffic) | 4.76 | 0.984 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | - | 4.58 | ok |
| mxC1_s8_r1/20261122T062000Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 42922 | 2.95 | ok |
| mxC1_s8_r1/20261122T062000Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.76 | 0.984 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | 44162 | 2.95 | ok |
| mxC1_s8_r2/20261122T062002Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.76 | 0.988 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | 44160 | 2.96 | ok |
| mxC1_s8_r2/20261122T062002Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 42662 | 2.96 | ok |
| mxC1_text_s8/20261122T062710Z | MI355X | mx_recipe / protocol_mx_recipe_text (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.25 | 0.43 | - | 2.12 | ok |
| mxC1_text_s8/20261122T062710Z | MI355X | or_recipe / protocol_or_recipe_text (traffic) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 0.00 | - | - | 4.25 | - | - | 2.12 | ok |
| mxC2_s7_r1/20261122T100636Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.49 | 3.25 | 0.000356 | 0.46 | 3.91 | 0.43 | 41536 | 6.30 | ok |
| mxC2_s7_r1/20261122T100636Z | MI355X | or_recipe / protocol_or_recipe_text (traffic) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.49 | 0.00 | - | - | 3.91 | - | - | 6.30 | ok |
| mxC2_s7_r1/20261122T100636Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.50 | 0.000331 | 0.42 | 3.91 | 0.40 | 43067 | 6.30 | ok |
| mxC2_s7_r1/20261122T100636Z | MI355X | mx_recipe / protocol_mx_recipe_text (traffic) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 0.00 | - | - | 3.91 | - | - | 6.30 | ok |
| mxC2_s7_r1/20261122T100636Z | MI355X | mx_stack / protocol_mx_stack (vllmbench) | 4.17 | 0.984 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 978673 | 1.49 | 3.50 | 0.000331 | 0.42 | 3.91 | 0.40 | 44776 | 6.30 | ok |
| mxC2_s7_r1/20261122T100636Z | MI355X | mx_stack / protocol_mx_stack_text (traffic) | 4.17 | 0.984 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 978673 | 1.49 | 0.00 | - | - | 3.91 | - | - | 6.30 | ok |
| mxC2_s7_r2/20261122T100638Z | MI355X | mx_stack / protocol_mx_stack (vllmbench) | 4.17 | 0.984 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.75 | 0.000309 | 0.40 | 3.91 | 0.37 | 46411 | 4.42 | ok |
| mxC2_s7_r2/20261122T100638Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 978673 | 1.49 | 3.75 | 0.000309 | 0.40 | 3.91 | 0.37 | 45159 | 4.42 | ok |
| mxC2_s7_r2/20261122T100638Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.17 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.49 | 3.25 | 0.000356 | 0.46 | 3.91 | 0.43 | 43595 | 4.42 | ok |
| mxC2_s8_r1/20261122T100640Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.50 | 0.000331 | 0.42 | 3.91 | 0.40 | 43311 | 6.63 | ok |
| mxC2_s8_r1/20261122T100640Z | MI355X | mx_recipe / protocol_mx_recipe_text (traffic) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.25 | 0.000356 | 0.46 | 3.91 | 0.43 | - | 6.63 | ok |
| mxC2_s8_r1/20261122T100640Z | MI355X | mx_stack / protocol_mx_stack (vllmbench) | 4.17 | 0.988 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 978673 | 1.49 | 3.50 | 0.000331 | 0.42 | 3.91 | 0.40 | 45263 | 6.63 | ok |
| mxC2_s8_r1/20261122T100640Z | MI355X | mx_stack / protocol_mx_stack_text (traffic) | 4.17 | 0.988 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 978673 | 1.49 | 3.50 | 0.000331 | 0.42 | 3.91 | 0.40 | - | 6.63 | ok |
| mxC2_s8_r1/20261122T100640Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.17 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.49 | 0.00 | - | - | 3.91 | - | 41880 | 6.63 | ok |
| mxC2_s8_r1/20261122T100640Z | MI355X | or_recipe / protocol_or_recipe_text (traffic) | 4.17 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.49 | 0.00 | - | - | 3.91 | - | - | 6.63 | ok |
| mxC2_s8_r2/20261122T100642Z | MI355X | mx_stack / protocol_mx_stack (vllmbench) | 4.17 | 0.988 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.50 | 0.000331 | 0.42 | 3.91 | 0.40 | 44811 | 4.51 | ok |
| mxC2_s8_r2/20261122T100642Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.17 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.49 | 0.00 | - | - | 3.91 | - | 41469 | 4.51 | ok |
| mxC2_s8_r2/20261122T100642Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 978673 | 1.49 | 3.25 | 0.000356 | 0.46 | 3.91 | 0.43 | 42931 | 4.51 | ok |
| mxC2b_s7/20261122T114751Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.49 | 3.25 | 0.000356 | 0.46 | 3.91 | 0.43 | - | 8.44 | ok |
| mxC2b_s7/20261122T114751Z | MI355X | or_recipe / protocol_or_recipe_text (traffic) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.49 | 3.00 | 0.000386 | 0.50 | 3.91 | 0.47 | - | 8.44 | ok |
| mxC2b_s7/20261122T114751Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.50 | 0.000331 | 0.42 | 3.91 | 0.40 | - | 8.44 | ok |
| mxC2b_s7/20261122T114751Z | MI355X | mx_recipe / protocol_mx_recipe_text (traffic) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.00 | 0.000386 | 0.50 | 3.91 | 0.47 | - | 8.44 | ok |
| mxC2b_s7/20261122T114751Z | MI355X | mx_stack / protocol_mx_stack (vllmbench) | 4.17 | 0.988 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 978673 | 1.49 | 3.75 | 0.000309 | 0.40 | 3.91 | 0.37 | - | 8.44 | ok |
| mxC2b_s7/20261122T114751Z | MI355X | mx_stack / protocol_mx_stack_text (traffic) | 4.17 | 0.988 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 978673 | 1.49 | 3.00 | 0.000386 | 0.50 | 3.91 | 0.47 | - | 8.44 | ok |
| mxC2b_s8/20261122T114753Z | MI355X | mx_stack / protocol_mx_stack (vllmbench) | 4.17 | 0.988 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.50 | 0.000331 | 0.42 | 3.91 | 0.40 | - | 8.45 | ok |
| mxC2b_s8/20261122T114753Z | MI355X | mx_stack / protocol_mx_stack_text (traffic) | 4.17 | 0.988 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.25 | 0.000356 | 0.46 | 3.91 | 0.43 | - | 8.45 | ok |
| mxC2b_s8/20261122T114753Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 978673 | 1.49 | 3.25 | 0.000356 | 0.46 | 3.91 | 0.43 | - | 8.45 | ok |
| mxC2b_s8/20261122T114753Z | MI355X | mx_recipe / protocol_mx_recipe_text (traffic) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 978673 | 1.49 | 3.25 | 0.000356 | 0.46 | 3.91 | 0.43 | - | 8.45 | ok |
| mxC2b_s8/20261122T114753Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.17 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.49 | 3.00 | 0.000386 | 0.50 | 3.91 | 0.47 | - | 8.45 | ok |
| mxC2b_s8/20261122T114753Z | MI355X | or_recipe / protocol_or_recipe_text (traffic) | 4.17 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.49 | 2.75 | 0.000421 | 0.54 | 3.91 | 0.51 | - | 8.45 | ok |
| mxF_s7_r1/20261122T091529Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.70 | 0.000357 | 0.46 | 4.25 | 0.41 | - | 3.31 | ok |
| mxF_s7_r1/20261122T091529Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.76 | 0.970 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.30 | 0.000400 | 0.51 | 4.25 | 0.46 | - | 3.31 | ok |
| mxF_s7_r2/20261122T091531Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.00 | 0.000440 | 0.57 | 4.25 | 0.51 | - | 3.39 | ok |
| mxF_s7_r2/20261122T091531Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.60 | 0.000367 | 0.47 | 4.25 | 0.42 | - | 3.39 | ok |
| mxF_s8_r1/20261122T091525Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.60 | 0.000367 | 0.47 | 4.26 | 0.42 | - | 4.28 | ok |
| mxF_s8_r1/20261122T091525Z | MI355X | mx_recipe / protocol_mx_recipe_text (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | - | 4.28 | ok |
| mxF_s8_r1/20261122T091525Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.30 | 0.000400 | 0.51 | 4.26 | 0.46 | - | 4.28 | ok |
| mxF_s8_r2/20261122T091527Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.00 | 0.000440 | 0.57 | 4.26 | 0.51 | - | 3.33 | ok |
| mxF_s8_r2/20261122T091527Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | - | 3.33 | ok |
| mxG2_s7/20261122T090631Z | MI355X | mx_ps / protocol_mx_ps (vllmbench) | 4.76 | 0.988 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.75 | 0.000352 | 0.45 | 4.26 | 0.40 | 46069 | 2.52 | ok |
| mxG2_s7/20261122T090631Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 978673 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | 44218 | 2.52 | ok |
| mxG2_s8/20261122T090633Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.25 | 0.43 | 43730 | 2.52 | ok |
| mxG2_s8/20261122T090633Z | MI355X | mx_ps / protocol_mx_ps (vllmbench) | 4.76 | 0.988 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 978673 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.25 | 0.43 | 45437 | 2.52 | ok |
| mxH4_s7/20261122T072841Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.50 | 0.000331 | 0.42 | 3.91 | 0.40 | 44249 | 1.95 | ok |
| mxH4_s7/20261122T072841Z | MI355X | mx_recipe / protocol_mx_recipe_text (traffic) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.25 | 0.000356 | 0.46 | 3.91 | 0.43 | - | 1.95 | ok |
| mxH4_s8/20261122T072720Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.50 | 0.000331 | 0.42 | 3.91 | 0.40 | 43330 | 1.96 | ok |
| mxH4_s8/20261122T072720Z | MI355X | mx_recipe / protocol_mx_recipe_text (traffic) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.25 | 0.000356 | 0.46 | 3.91 | 0.43 | - | 1.96 | ok |
| mxH4b_s7/20261122T091523Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.50 | 0.000331 | 0.42 | 3.91 | 0.40 | 43994 | 2.51 | ok |
| mxH4b_s7/20261122T091523Z | MI355X | mx_recipe / protocol_mx_recipe_text (traffic) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 0.00 | - | - | 3.91 | - | - | 2.51 | ok |
| mxH4b_s8/20261122T091521Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.50 | 0.000331 | 0.42 | 3.91 | 0.40 | 45231 | 2.48 | ok |
| mxH4b_s8/20261122T091521Z | MI355X | mx_recipe / protocol_mx_recipe_text (traffic) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.50 | 0.000331 | 0.42 | 3.91 | 0.40 | - | 2.48 | ok |
| mxK_s7/20261122T091535Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 3712490 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 41501 | 3.57 | ok |
| mxK_s7/20261122T091535Z | MI355X | or_recipe / protocol_or_recipe_text (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 3712490 | 1.70 | 0.00 | - | - | 4.26 | - | - | 3.57 | ok |
| mxK_s7/20261122T091535Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 3701967 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | 42931 | 3.57 | ok |
| mxK_s7/20261122T091535Z | MI355X | mx_recipe / protocol_mx_recipe_text (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 3701967 | 1.70 | 0.00 | - | - | 4.26 | - | - | 3.57 | ok |
| mxK_s8/20261122T091533Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 3701967 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 42957 | 3.56 | ok |
| mxK_s8/20261122T091533Z | MI355X | mx_recipe / protocol_mx_recipe_text (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 3701967 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | - | 3.56 | ok |
| mxK_s8/20261122T091533Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 3712490 | 1.70 | 0.00 | - | - | 4.26 | - | 41496 | 3.56 | ok |
| mxK_s8/20261122T091533Z | MI355X | or_recipe / protocol_or_recipe_text (traffic) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 3712490 | 1.70 | 0.00 | - | - | 4.26 | - | - | 3.56 | ok |
| mxL_lpt_s8/20261122T091841Z | MI355X | mx_recipe / protocol_mx_recipe (vllmbench) | 4.76 | - - | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | - | 4.80 | ok |
| mxL_lpt_s8/20261122T091841Z | MI355X | mx_lpt6144 / protocol_mx_lpt6144 (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 978673 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | - | 4.80 | ok |
| mxL_lpt_s8/20261122T091841Z | MI355X | mx_lpt4096 / protocol_mx_lpt4096 (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 978673 | 1.70 | 3.50 | 0.000377 | 0.48 | 4.26 | 0.43 | - | 4.80 | ok |
| mxR4_b10k/20261122T105424Z | MI355X | mx_stack / protocol_mx_stack (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 0.00 | - | - | 3.91 | - | - | 2.13 | ok |
| mxR4_b10k/20261122T105424Z | MI355X | mx_stack_b10k / protocol_mx_stack_b10k (vllmbench) | 4.17 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 813325 | 1.49 | 0.00 | - | - | 3.91 | - | - | 2.13 | ok |
| mxR4_b12k/20261122T105426Z | MI355X | mx_stack_b12k / protocol_mx_stack_b12k (vllmbench) | 4.17 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 697910 | 1.49 | 3.75 | 0.000309 | 0.40 | 3.91 | 0.37 | - | 2.13 | ok |
| mxR4_b12k/20261122T105426Z | MI355X | mx_stack / protocol_mx_stack (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 3.75 | 0.000309 | 0.40 | 3.91 | 0.37 | - | 2.13 | ok |
| mxR4_s112/20261122T105430Z | MI355X | mx_stack_s112 / protocol_mx_stack_s112 (vllmbench) | 4.17 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 972973 | 1.49 | 0.00 | - | - | 3.91 | - | - | 2.14 | ok |
| mxR4_s112/20261122T105430Z | MI355X | mx_stack / protocol_mx_stack (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 0.00 | - | - | 3.91 | - | - | 2.14 | ok |
| mxR4_s80/20261122T105428Z | MI355X | mx_stack / protocol_mx_stack (vllmbench) | 4.17 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 974307 | 1.49 | 0.00 | - | - | 3.91 | - | - | 2.13 | ok |
| mxR4_s80/20261122T105428Z | MI355X | mx_stack_s80 / protocol_mx_stack_s80 (vllmbench) | 4.17 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 975317 | 1.49 | 0.00 | - | - | 3.91 | - | - | 2.13 | ok |
| mxS0_vllm_specoff/20261122T072832Z | MI355X | or_specoff / protocol_or_specoff (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 987729 | 1.70 | 2.50 | 0.000528 | 0.68 | 4.25 | 0.61 | 43820 | 1.30 | ok |
| mxS1_sglang/20261122T073259Z | MI355X | sgl_fp8 / protocol_sgl_fp8 (vllmbench) | 4.76 | 0.990 PASS | aiter / aiter | 530791 | 1.70 | 0.00 | - | - | 4.26 | - | 17217 | 1.37 | ok |
| mxS3_vllm_fp8/20261122T080325Z | MI355X | vllm_fp8 / protocol_vllm_fp8 (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 941357 | 1.70 | 0.00 | - | - | 4.26 | - | 29254 | 0.96 | ok |
| s1_b300_nvfp4_mtp2_r4/20261122T045943Z | B300 | nvfp4_mtp2_r4 / protocol_nvfp4_mtp2_r4 (vllmbench) | 8.09 | 0.980 PASS | FLASHINFER / FLASHINFER | 909418 | 2.89 | 0.00 | - | - | 6.86 | - | 22531 | 1.82 | ok |
| s1_b300_nvfp4_mtp2_r4/20261122T045943Z | B300 | nvfp4_mtp2_r4 / protocol_nvfp4_mtp2_r4_textclient (traffic) | 8.09 | 0.980 PASS | FLASHINFER / FLASHINFER | 909418 | 2.89 | 0.00 | - | - | 6.86 | - | - | 1.82 | ok |
| s1_h100_fp8/20261122T042445Z | H100 | fp8dyn / protocol_fp8dyn (vllmbench) | 3.67 | 0.980 PASS | FLASH_ATTN/FA3 / FLASH_ATTN/FA4 | 144878 | 1.31 | 0.00 | - | - | 3.17 | - | 12767 | 0.75 | ok |
| s1_h100_fp8/20261122T042445Z | H100 | fp8dyn / protocol_fp8dyn_textclient (traffic) | 3.67 | 0.980 PASS | FLASH_ATTN/FA3 / FLASH_ATTN/FA4 | 144878 | 1.31 | 0.00 | - | - | 3.17 | - | - | 0.75 | ok |
| s1_h200_fp8/20261122T042702Z | H200 | fp8dyn / protocol_fp8dyn (vllmbench) | 3.80 | 0.990 PASS | FLASH_ATTN/FA3 / FLASH_ATTN/FA4 | 380783 | 1.35 | 0.00 | - | - | 3.41 | - | 13166 | 0.78 | ok |
| s1_h200_fp8/20261122T042702Z | H200 | fp8dyn / protocol_fp8dyn_textclient (traffic) | 3.80 | 0.990 PASS | FLASH_ATTN/FA3 / FLASH_ATTN/FA4 | 380783 | 1.35 | 0.00 | - | - | 3.41 | - | - | 0.78 | ok |
| s1_l40s_w4a16_fi/20261122T043255Z | L40S | w4a16qat_fi / protocol_w4a16qat_fi (vllmbench) | 2.07 | 0.980 PASS | FLASHINFER / FLASHINFER | 69963 | 0.74 | 0.00 | - | - | 1.68 | - | 2570 | 0.95 | ok |
| s1_l40s_w4a16_fi/20261122T043255Z | L40S | w4a16qat_fi / protocol_w4a16qat_fi_textclient (traffic) | 2.07 | 0.980 PASS | FLASHINFER / FLASHINFER | 69963 | 0.74 | 0.00 | - | - | 1.68 | - | - | 0.95 | ok |
| s1_rtxpro6000_nvfp4_fi/20261122T043253Z | RTX-PRO-6000 | nvfp4_fi / protocol_nvfp4_fi (vllmbench) | 2.88 | 0.990 PASS | FLASHINFER / FLASHINFER | 265537 | 1.03 | 0.00 | - | - | 2.49 | - | 12082 | 0.59 | ok |
| s1_rtxpro6000_nvfp4_fi/20261122T043253Z | RTX-PRO-6000 | nvfp4_fi / protocol_nvfp4_fi_textclient (traffic) | 2.88 | 0.990 PASS | FLASHINFER / FLASHINFER | 265537 | 1.03 | 0.00 | - | - | 2.49 | - | - | 0.59 | ok |
| s2_mi355x_or_recipe_seed7/20261122T043703Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 41560 | 2.70 | ok |
| s2_mi355x_or_recipe_seed7/20261122T043703Z | MI355X | or_recipe / protocol_or_recipe_textclient (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.00 | 0.000440 | 0.57 | 4.26 | 0.51 | - | 2.70 | ok |
| s2_mi355x_or_recipe_seed8/20261122T043705Z | MI355X | or_recipe / protocol_or_recipe (vllmbench) | 4.76 | 0.980 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | 42758 | 2.23 | ok |
| s2a_mi355x_or_recipe_seed8/20261122T054934Z | MI355X | or_recipe / protocol_or_recipe_A_random (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.25 | 0.47 | 43096 | 1.75 | ok |
| s2b_mi355x_or_recipe_seed7/20261122T053337Z | MI355X | or_recipe / protocol_or_recipe_B_text_strat (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.00 | 0.000440 | 0.57 | 4.26 | 0.51 | - | 1.95 | ok |
| s2b_mi355x_or_recipe_seed8/20261122T054928Z | MI355X | or_recipe / protocol_or_recipe_B_text_strat (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.00 | 0.000440 | 0.57 | 4.25 | 0.51 | - | 1.96 | ok |
| s2c_mi355x_or_recipe_seed7/20261122T051851Z | MI355X | or_recipe / protocol_or_recipe_A_random (vllmbench) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.25 | 0.47 | 42797 | 3.00 | ok |
| s2c_mi355x_or_recipe_seed7/20261122T051851Z | MI355X | or_recipe / protocol_or_recipe_B_text (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.25 | 0.47 | - | 3.00 | ok |
| s2c_mi355x_or_recipe_seed8/20261122T051853Z | MI355X | or_recipe / protocol_or_recipe_B_text (traffic) | 4.76 | 0.990 PASS | ROCM_AITER_UNIFIED_ATTN / ROCM_AITER_UNIFIED_ATTN | 977076 | 1.70 | 3.25 | 0.000407 | 0.52 | 4.26 | 0.47 | - | 2.42 |  EVICTED ERROR ApiError('HTTP 404 on https://proxy.app.daytona.io/toolbox/1e73f7e8-bc0b-48b7-b6 |
| s3_rtxpro6000_small_mtp/20261122T050407Z | RTX-PRO-6000 | p6_small_base / protocol_p6_small_base (vllmbench) | 2.42 | 0.990 PASS | FLASHINFER / FLASHINFER | 265537 | 0.86 | 0.70 | 0.000960 | 1.23 | 2.22 | 1.13 | 12092 | 1.62 | ok |
| s3_rtxpro6000_small_mtp/20261122T050407Z | RTX-PRO-6000 | p6_small_mtp2 / protocol_p6_small_mtp2 (vllmbench) | 2.42 | 0.990 PASS | FLASHINFER / FLASHINFER | 260877 | 0.86 | 0.00 | - | - | 2.22 | - | 11509 | 1.62 | ok |
| s3_rtxpro6000_small_mtp/20261122T050407Z | RTX-PRO-6000 | p6_small_mtp2 / protocol_p6_small_mtp2_textclient (traffic) | 2.42 | 0.990 PASS | FLASHINFER / FLASHINFER | 260877 | 0.86 | 0.00 | - | - | 2.22 | - | - | 1.62 | ok |
| s3_rtxpro6000_small_mtp/20261122T050407Z | RTX-PRO-6000 | p6_small_mtp3 / protocol_p6_small_mtp3 (vllmbench) | 2.42 | 0.990 PASS | FLASHINFER / FLASHINFER | 261536 | 0.86 | 0.00 | - | - | 2.22 | - | - | 1.62 | ok |
| s3_rtxpro6000_small_mtp/20261122T050407Z | RTX-PRO-6000 | p6_small_mtp3 / protocol_p6_small_mtp3_textclient (traffic) | 2.42 | 0.990 PASS | FLASHINFER / FLASHINFER | 261536 | 0.86 | 0.00 | - | - | 2.22 | - | - | 1.62 | ok |

| run / protocol | x break-even | offered req/s | achieved | p95 TTFT s | TPOT p50 / p95 ms | errors | SLO | MTP accept len |
|---|---|---|---|---|---|---|---|---|
| cc_c1_s7 / protocol_mx_recipe_c1evict | 2.06 | 3.50 | 3.51 | 1.33 | 20.1 / 43.2 | 0 | PASS | 2.78 |
| cc_c1_s7 / protocol_mx_recipe_c1evict | 2.65 | 4.50 | 4.12 | inf | 59.2 / 67.6 | 0 | fail | 2.77 |
| cc_c1_s7 / protocol_mx_recipe_c1fit | 2.36 | 4.00 | 4.02 | 0.24 | 7.2 / 9.8 | 0 | PASS | 2.76 |
| cc_c1_s7 / protocol_mx_recipe_c1fit | 2.95 | 5.00 | 5.03 | 0.27 | 8.5 / 11.6 | 0 | PASS | 2.77 |
| cc_c1_s7 / protocol_mx_recipe_c1fit | 3.54 | 6.00 | 6.08 | 0.33 | 10.4 / 15.7 | 0 | PASS | 2.77 |
| cc_c1_s7 / protocol_mx_recipe_c1fit | 4.42 | 7.50 | 7.50 | 0.46 | 17.6 / 26.9 | 0 | PASS | 2.77 |
| cc_c1_s7 / protocol_or_recipe_c1evict | 2.06 | 3.50 | 3.50 | 2.28 | 26.1 / 48.0 | 0 | PASS | 2.37 |
| cc_c1_s7 / protocol_or_recipe_c1evict | 2.65 | 4.50 | 3.87 | inf | 42.3 / 49.3 | 0 | fail | 2.37 |
| cc_c1_s7 / protocol_or_recipe_c1fit | 2.36 | 4.00 | 4.02 | 0.25 | 7.7 / 10.3 | 0 | PASS | 2.36 |
| cc_c1_s7 / protocol_or_recipe_c1fit | 2.95 | 5.00 | 5.03 | 0.28 | 9.1 / 12.3 | 0 | PASS | 2.36 |
| cc_c1_s7 / protocol_or_recipe_c1fit | 3.54 | 6.00 | 6.07 | 0.34 | 11.3 / 16.9 | 0 | PASS | 2.36 |
| cc_c1_s7 / protocol_or_recipe_c1fit | 4.42 | 7.50 | 7.65 | 3.53 | 20.2 / 23.0 | 0 | PASS | 2.36 |
| cc_c1_s8 / protocol_or_recipe_c1evict | 2.06 | 3.50 | 3.60 | 1.74 | 22.9 / 46.8 | 0 | PASS | 2.36 |
| cc_c1_s8 / protocol_or_recipe_c1evict | 2.65 | 4.50 | 3.86 | inf | 42.7 / 49.0 | 0 | fail | 2.36 |
| cc_c1_s8 / protocol_or_recipe_c1fit | 2.36 | 4.00 | 3.96 | 0.27 | 7.8 / 9.9 | 0 | PASS | 2.36 |
| cc_c1_s8 / protocol_or_recipe_c1fit | 2.95 | 5.00 | 4.93 | 0.28 | 9.0 / 11.9 | 0 | PASS | 2.36 |
| cc_c1_s8 / protocol_or_recipe_c1fit | 3.54 | 6.00 | 5.95 | 0.36 | 11.0 / 15.3 | 0 | PASS | 2.36 |
| cc_c1_s8 / protocol_or_recipe_c1fit | 4.42 | 7.50 | 7.63 | 7.54 | 20.0 / 23.7 | 0 | fail | 2.36 |
| cc_c1_s8 / protocol_mx_recipe_c1evict | 2.06 | 3.50 | 3.54 | 1.41 | 19.2 / 43.4 | 0 | PASS | 2.77 |
| cc_c1_s8 / protocol_mx_recipe_c1evict | 2.65 | 4.50 | 4.10 | inf | 60.4 / 70.4 | 0 | fail | 2.76 |
| cc_c1_s8 / protocol_mx_recipe_c1fit | 2.36 | 4.00 | 3.96 | 0.27 | 7.1 / 9.3 | 0 | PASS | 2.77 |
| cc_c1_s8 / protocol_mx_recipe_c1fit | 2.95 | 5.00 | 4.94 | 0.29 | 8.4 / 11.1 | 0 | PASS | 2.77 |
| cc_c1_s8 / protocol_mx_recipe_c1fit | 3.54 | 6.00 | 5.93 | 0.34 | 10.2 / 14.1 | 0 | PASS | 2.77 |
| cc_c1_s8 / protocol_mx_recipe_c1fit | 4.42 | 7.50 | 7.58 | 0.64 | 17.1 / 29.4 | 0 | PASS | 2.76 |
| cc_c2_s7 / protocol_mx_recipe_c2agent | 2.36 | 4.00 | 3.94 | 0.36 | 7.8 / 11.8 | 0 | PASS | 3.05 |
| cc_c2_s7 / protocol_mx_recipe_c2agent | 2.95 | 5.00 | 4.95 | 0.43 | 9.8 / 14.2 | 0 | PASS | 3.06 |
| cc_c2_s7 / protocol_mx_recipe_c2agent | 3.54 | 6.00 | 6.05 | 0.51 | 13.2 / 19.2 | 0 | PASS | 3.05 |
| cc_c2_s7 / protocol_mx_recipe_c2agent | 4.42 | 7.50 | 7.04 | 2.82 | 30.3 / 39.3 | 0 | PASS | 3.03 |
| cc_c2_s7 / protocol_or_recipe_c2agent | 2.36 | 4.00 | 3.94 | 0.36 | 8.6 / 12.4 | 0 | PASS | 2.52 |
| cc_c2_s7 / protocol_or_recipe_c2agent | 2.95 | 5.00 | 4.96 | 0.41 | 11.4 / 16.3 | 0 | PASS | 2.51 |
| cc_c2_s7 / protocol_or_recipe_c2agent | 3.54 | 6.00 | 6.05 | 0.51 | 15.5 / 21.6 | 0 | PASS | 2.52 |
| cc_c2_s7 / protocol_or_recipe_c2agent | 4.42 | 7.50 | 6.63 | 14.68 | 24.4 / 28.9 | 0 | fail | 2.52 |
| cc_c2_s8 / protocol_or_recipe_c2agent | 2.36 | 4.00 | 3.95 | 0.34 | 8.4 / 11.5 | 0 | PASS | 2.52 |
| cc_c2_s8 / protocol_or_recipe_c2agent | 2.95 | 5.00 | 4.80 | 0.41 | 10.5 / 15.2 | 0 | PASS | 2.52 |
| cc_c2_s8 / protocol_or_recipe_c2agent | 3.54 | 6.00 | 5.92 | 0.56 | 14.8 / 20.7 | 0 | PASS | 2.52 |
| cc_c2_s8 / protocol_or_recipe_c2agent | 4.42 | 7.50 | 6.69 | 12.22 | 24.4 / 29.3 | 0 | fail | 2.51 |
| cc_c2_s8 / protocol_mx_recipe_c2agent | 2.36 | 4.00 | 3.94 | 0.34 | 7.6 / 10.5 | 0 | PASS | 3.05 |
| cc_c2_s8 / protocol_mx_recipe_c2agent | 2.95 | 5.00 | 4.80 | 0.43 | 9.4 / 14.2 | 0 | PASS | 3.05 |
| cc_c2_s8 / protocol_mx_recipe_c2agent | 3.54 | 6.00 | 5.90 | 0.53 | 12.8 / 18.1 | 0 | PASS | 3.04 |
| cc_c2_s8 / protocol_mx_recipe_c2agent | 4.42 | 7.50 | 7.10 | 0.90 | 28.8 / 37.7 | 0 | PASS | 3.05 |
| cc_x2_s7 / protocol_mx_recipe_c1fit | 4.13 | 7.00 | 7.17 | 0.39 | 13.6 / 21.9 | 0 | PASS | 2.77 |
| cc_x2_s7 / protocol_mx_recipe_c1fit | 4.72 | 8.00 | 8.19 | 2.01 | 25.0 / 30.9 | 0 | PASS | 2.77 |
| cc_x2_s7 / protocol_mx_recipe_c1fit | 5.01 | 8.50 | 8.52 | 10.83 | 28.9 / 32.5 | 0 | fail | 2.76 |
| cc_x2_s7 / protocol_mx_recipe_c2agent | 4.13 | 7.00 | 6.75 | 0.62 | 20.2 / 29.2 | 0 | PASS | 3.04 |
| cc_x2_s7 / protocol_mx_recipe_c2agent | 4.72 | 8.00 | 6.69 | inf | 34.3 / 62.9 | 0 | fail | 3.04 |
| cc_x2_s7 / protocol_or_recipe_c1fit | 4.13 | 7.00 | 7.18 | 0.88 | 15.3 / 22.0 | 0 | PASS | 2.36 |
| cc_x2_s7 / protocol_or_recipe_c1fit | 4.72 | 8.00 | 8.00 | 15.41 | 20.7 / 23.4 | 0 | fail | 2.36 |
| cc_x2_s7 / protocol_or_recipe_c1fit | 5.01 | 8.50 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| cc_x2_s7 / protocol_or_recipe_c2agent | 4.13 | 7.00 | 6.67 | 3.79 | 23.5 / 29.3 | 0 | PASS | 2.53 |
| cc_x2_s7 / protocol_or_recipe_c2agent | 4.72 | 8.00 | 5.73 | inf | 25.5 / 46.2 | 0 | fail | 2.52 |
| cc_x2_s8 / protocol_or_recipe_c1fit | 4.13 | 7.00 | 7.17 | 4.12 | 15.3 / 22.5 | 0 | fail | 2.36 |
| cc_x2_s8 / protocol_or_recipe_c1fit | 4.72 | 8.00 | 7.98 | 14.24 | 20.7 / 23.4 | 0 | fail | 2.36 |
| cc_x2_s8 / protocol_or_recipe_c1fit | 5.01 | 8.50 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| cc_x2_s8 / protocol_or_recipe_c2agent | 4.13 | 7.00 | 6.68 | 3.12 | 24.0 / 28.6 | 0 | PASS | 2.52 |
| cc_x2_s8 / protocol_or_recipe_c2agent | 4.72 | 8.00 | 6.10 | inf | 25.5 / 45.5 | 0 | fail | 2.52 |
| cc_x2_s8 / protocol_mx_recipe_c1fit | 4.13 | 7.00 | 6.97 | 0.39 | 13.2 / 19.5 | 0 | PASS | 2.76 |
| cc_x2_s8 / protocol_mx_recipe_c1fit | 4.72 | 8.00 | 8.12 | 4.25 | 21.2 / 31.8 | 0 | fail | 2.77 |
| cc_x2_s8 / protocol_mx_recipe_c1fit | 5.01 | 8.50 | 8.55 | 10.69 | 29.2 / 33.0 | 0 | fail | 2.76 |
| cc_x2_s8 / protocol_mx_recipe_c2agent | 4.13 | 7.00 | 6.83 | 0.68 | 20.4 / 28.7 | 0 | PASS | 3.03 |
| cc_x2_s8 / protocol_mx_recipe_c2agent | 4.72 | 8.00 | 7.09 | 9.08 | 33.8 / 42.0 | 0 | fail | 3.04 |
| cc_x_s7 / protocol_mx_recipe_c1fit | 5.31 | 9.00 | 8.57 | inf | 28.9 / 32.8 | 0 | fail | 2.77 |
| cc_x_s7 / protocol_mx_recipe_c1fit | 6.19 | 10.50 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| cc_x_s7 / protocol_mx_recipe_c1fit | 7.08 | 12.00 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| cc_x_s7 / protocol_mx_recipe_c2agent | 5.01 | 8.50 | 5.60 | inf | 39.2 / 68.0 | 0 | fail | 3.04 |
| cc_x_s7 / protocol_mx_recipe_c2agent | 5.60 | 9.50 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| cc_x_s7 / protocol_or_recipe_c1fit | 3.98 | 6.75 | 6.84 | 0.62 | 14.0 / 21.3 | 0 | PASS | 2.37 |
| cc_x_s7 / protocol_or_recipe_c2agent | 3.98 | 6.75 | 6.48 | 0.98 | 20.3 / 27.5 | 0 | PASS | 2.52 |
| cc_x_s8 / protocol_or_recipe_c1fit | 3.98 | 6.75 | 6.91 | 3.70 | 14.8 / 23.2 | 0 | PASS | 2.36 |
| cc_x_s8 / protocol_or_recipe_c2agent | 3.98 | 6.75 | 6.55 | 1.89 | 23.7 / 29.4 | 0 | PASS | 2.52 |
| cc_x_s8 / protocol_mx_recipe_c1fit | 5.31 | 9.00 | 8.27 | inf | 30.1 / 33.7 | 0 | fail | 2.77 |
| cc_x_s8 / protocol_mx_recipe_c1fit | 6.19 | 10.50 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| cc_x_s8 / protocol_mx_recipe_c1fit | 7.08 | 12.00 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| cc_x_s8 / protocol_mx_recipe_c2agent | 5.01 | 8.50 | 5.23 | inf | 44.5 / 72.5 | 0 | fail | 3.01 |
| cc_x_s8 / protocol_mx_recipe_c2agent | 5.60 | 9.50 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| d3_sglang_counter_probe / protocol_sgl_counter_B_text | 0.34 | 0.35 | 0.34 | 17.94 | 73.2 / 109.6 | 0 | fail | - |
| e4_b300_sglang / protocol_sgl_b300_A_random | 1.00 | 2.37 | 2.31 | 2.14 | 41.0 / 63.6 | 0 | PASS | - |
| e4_b300_sglang / protocol_sgl_b300_A_random | 1.50 | 3.55 | 2.55 | 105.29 | 57.5 / 68.1 | 0 | fail | - |
| e4_b300_sglang / protocol_sgl_b300_B_text | 1.00 | 2.37 | 2.50 | 3.07 | 36.5 / 63.6 | 0 | PASS | - |
| e4_b300_sglang / protocol_sgl_b300_B_text | 1.50 | 3.55 | 2.61 | 65.91 | 57.3 / 66.8 | 0 | fail | - |
| e4_rtxpro6000_sglang / protocol_sgl_nvfp4_nv_B_text | 1.00 | 1.03 | 0.44 | 175.36 | 84.9 / 106.0 | 0 | fail | - |
| e4_rtxpro6000_sglang / protocol_sgl_nvfp4_nv_B_text | 1.50 | 1.54 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| e4a_rtxpro6000_sglang_colA / protocol_sgl_nvfp4_nv_A_random | 1.00 | 1.03 | 0.38 | 485.87 | 87.0 / 111.7 | 0 | fail | - |
| e4a_rtxpro6000_sglang_colA / protocol_sgl_nvfp4_nv_A_random | 1.50 | 1.54 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| e5_b300_sglang_textonly / protocol_sgl_rh_nvfp4_textonly_A_random | 1.00 | 2.37 | 2.34 | 0.89 | 14.8 / 24.1 | 0 | PASS | - |
| e5_b300_sglang_textonly / protocol_sgl_rh_nvfp4_textonly_A_random | 1.25 | 2.96 | 2.91 | 1.05 | 25.1 / 43.4 | 0 | PASS | - |
| e5_b300_sglang_textonly / protocol_sgl_rh_nvfp4_textonly_A_random | 1.50 | 3.55 | 3.39 | 8.63 | 44.8 / 52.4 | 0 | fail | - |
| e5_b300_sglang_textonly / protocol_sgl_rh_nvfp4_textonly_B_text | 1.00 | 2.37 | 2.34 | 0.71 | 14.5 / 33.2 | 0 | PASS | - |
| e5_b300_sglang_textonly / protocol_sgl_rh_nvfp4_textonly_B_text | 1.25 | 2.96 | 2.93 | 2.45 | 23.7 / 48.2 | 0 | PASS | - |
| e5_b300_sglang_textonly_s8 / protocol_sgl_rh_nvfp4_textonly_A_random | 1.00 | 2.37 | 2.33 | 0.72 | 14.4 / 23.3 | 0 | PASS | - |
| e5_b300_sglang_textonly_s8 / protocol_sgl_rh_nvfp4_textonly_A_random | 1.25 | 2.96 | 2.93 | 0.92 | 23.3 / 42.1 | 0 | PASS | - |
| e5_b300_sglang_textonly_s8 / protocol_sgl_rh_nvfp4_textonly_A_random | 1.50 | 3.55 | 3.49 | 8.83 | 44.1 / 51.1 | 0 | fail | - |
| e5_b300_sglang_textonly_s8 / protocol_sgl_rh_nvfp4_textonly_B_text | 1.00 | 2.37 | 2.35 | 0.81 | 15.2 / 24.0 | 0 | PASS | - |
| e5_b300_sglang_textonly_s8 / protocol_sgl_rh_nvfp4_textonly_B_text | 1.25 | 2.96 | 2.96 | 0.86 | 22.3 / 39.2 | 0 | PASS | - |
| glm1_sglang0523 / protocol_glm_sgl | 0.59 | 1.00 | 0.97 | 1.89 | 38.0 / 58.9 | 0 | PASS | - |
| glm1_sglang0523 / protocol_glm_sgl | 0.88 | 1.50 | 1.40 | 7.33 | 104.2 / 126.3 | 0 | fail | - |
| glm1_sglang0523 / protocol_glm_sgl | 1.18 | 2.00 | 1.48 | 89.50 | 110.4 / 125.5 | 0 | fail | - |
| glm1_sglang0523 / protocol_glm_sgl | 1.59 | 2.70 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| glm1_vllm034 / protocol_glm_vllm | 0.59 | 1.00 | 0.97 | 1.82 | 35.2 / 54.8 | 0 | PASS | - |
| glm1_vllm034 / protocol_glm_vllm | 0.88 | 1.50 | 1.43 | 3.33 | 77.8 / 106.1 | 0 | PASS | - |
| glm1_vllm034 / protocol_glm_vllm | 1.18 | 2.00 | 1.56 | 69.40 | 103.1 / 115.3 | 0 | fail | - |
| glm1_vllm034 / protocol_glm_vllm | 1.59 | 2.70 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| glm2_vllm_b16k / protocol_glm_b16k | 0.59 | 1.00 | 0.97 | 1.82 | 33.3 / 50.8 | 0 | PASS | - |
| glm2_vllm_b16k / protocol_glm_b16k | 0.88 | 1.50 | 1.44 | 2.99 | 68.9 / 97.6 | 0 | PASS | - |
| glm2_vllm_b16k / protocol_glm_b16k | 1.18 | 2.00 | 1.61 | 58.39 | 99.9 / 112.4 | 0 | fail | - |
| glm2_vllm_b16k / protocol_glm_b16k | 1.59 | 2.70 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| glm2_vllm_mtp3 / protocol_glm_mtp3 | 0.59 | 1.00 | 0.97 | 1.98 | 33.8 / 64.6 | 0 | PASS | 1.62 |
| glm2_vllm_mtp3 / protocol_glm_mtp3 | 0.88 | 1.50 | 1.42 | 3.64 | 69.8 / 140.5 | 0 | PASS | 1.63 |
| glm2_vllm_mtp3 / protocol_glm_mtp3 | 1.18 | 2.00 | 1.58 | 61.48 | 93.3 / 168.7 | 0 | fail | 1.62 |
| glm2_vllm_mtp3 / protocol_glm_mtp3 | 1.59 | 2.70 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| glm2_vllm_mtp3 / protocol_glm_mtp3_text | 0.88 | 1.50 | 1.50 | 3.39 | 28.9 / 72.9 | 0 | PASS | 3.16 |
| glm2_vllm_mtp5 / protocol_glm_mtp5 | 0.59 | 1.00 | 0.96 | 2.13 | 44.3 / 85.4 | 0 | PASS | 1.62 |
| glm2_vllm_mtp5 / protocol_glm_mtp5 | 0.88 | 1.50 | 1.39 | 4.48 | 82.9 / 175.8 | 0 | fail | 1.74 |
| glm2_vllm_mtp5 / protocol_glm_mtp5 | 1.18 | 2.00 | 1.50 | 79.37 | 102.9 / 189.1 | 0 | fail | 1.70 |
| glm2_vllm_mtp5 / protocol_glm_mtp5 | 1.59 | 2.70 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| glm2_vllm_mtp5 / protocol_glm_mtp5_text | 0.88 | 1.50 | 1.49 | 3.63 | 29.2 / 73.3 | 0 | PASS | 3.76 |
| glm4_best_s7 / protocol_glm_best | 0.47 | 1.25 | 1.21 | 2.55 | 50.1 / 94.9 | 0 | PASS | 1.62 |
| glm4_best_s7 / protocol_glm_best | 0.56 | 1.50 | 1.44 | 3.15 | 72.6 / 146.6 | 0 | PASS | 1.68 |
| glm4_best_s7 / protocol_glm_best | 0.66 | 1.75 | 1.59 | 24.64 | 94.6 / 171.4 | 0 | fail | 1.67 |
| glm4_best_s7 / protocol_glm_best | 0.75 | 2.00 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| glm4_best_s7 / protocol_glm_best_text | 0.56 | 1.50 | 1.53 | 2.96 | 30.8 / 72.6 | 0 | PASS | 3.15 |
| glm4_best_s7 / protocol_glm_best_text | 0.66 | 1.75 | 1.68 | 5.78 | 43.7 / 103.8 | 0 | fail | 3.16 |
| glm4_best_s8 / protocol_glm_best | 0.47 | 1.25 | 1.20 | 2.51 | 48.2 / 96.2 | 0 | PASS | 1.61 |
| glm4_best_s8 / protocol_glm_best | 0.56 | 1.50 | 1.41 | 3.41 | 67.2 / 145.9 | 0 | PASS | 1.65 |
| glm4_best_s8 / protocol_glm_best | 0.66 | 1.75 | 1.54 | 26.09 | 91.5 / 165.4 | 0 | fail | 1.61 |
| glm4_best_s8 / protocol_glm_best | 0.75 | 2.00 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| glm4_best_s8 / protocol_glm_best_text | 0.56 | 1.50 | 1.47 | 3.04 | 28.6 / 58.4 | 0 | PASS | 3.16 |
| glm4_best_s8 / protocol_glm_best_text | 0.66 | 1.75 | 1.69 | 3.77 | 48.7 / 90.6 | 0 | PASS | 3.16 |
| ik1_5090_sgl / protocol_sgl_A_random | 1.00 | 0.56 | 0.55 | 3.12 | 13.3 / 21.9 | 0 | PASS | - |
| ik1_5090_sgl / protocol_sgl_A_random | 1.50 | 0.85 | 0.70 | 20.05 | 18.7 / 24.8 | 0 | fail | - |
| ik1_5090_sgl / protocol_sgl_A_random | 2.00 | 1.13 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_5090_sgl / protocol_sgl_A_random | 3.00 | 1.69 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_5090_sgl / protocol_sgl_A_random | 4.00 | 2.25 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_5090_sgl / protocol_sgl_B_text | 1.00 | 0.56 | 0.55 | 4.21 | 14.0 / 24.7 | 0 | fail | - |
| ik1_5090_sgl / protocol_sgl_B_text | 1.50 | 0.85 | 0.72 | 14.12 | 19.7 / 27.5 | 0 | fail | - |
| ik1_5090_sgl / protocol_sgl_B_text | 2.00 | 1.13 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_5090_sgl / protocol_sgl_B_text | 3.00 | 1.69 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_5090_sgl / protocol_sgl_B_text | 4.00 | 2.25 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_h100_sgl / protocol_sgl_A_random | 1.00 | 1.15 | 1.08 | 1.99 | 59.2 / 90.8 | 0 | PASS | - |
| ik1_h100_sgl / protocol_sgl_A_random | 1.50 | 1.72 | 1.28 | 29.03 | 106.0 / 144.4 | 0 | fail | - |
| ik1_h100_sgl / protocol_sgl_A_random | 2.00 | 2.29 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_h100_sgl / protocol_sgl_A_random | 3.00 | 3.44 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_h100_sgl / protocol_sgl_A_random | 4.00 | 4.58 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_h100_sgl / protocol_sgl_B_text | 1.00 | 1.15 | 1.00 | 8.12 | 50.7 / 105.1 | 0 | fail | - |
| ik1_h100_sgl / protocol_sgl_B_text | 1.50 | 1.72 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_h100_sgl / protocol_sgl_B_text | 2.00 | 2.29 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_h100_sgl / protocol_sgl_B_text | 3.00 | 3.44 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_h100_sgl / protocol_sgl_B_text | 4.00 | 4.58 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_mi355x_sgl / protocol_sgl_A_random | 1.00 | 1.51 | 1.34 | 1.73 | 76.2 / 89.5 | 0 | PASS | - |
| ik1_mi355x_sgl / protocol_sgl_A_random | 1.50 | 2.27 | 1.66 | 2.82 | 219.0 / 428.3 | 0 | PASS | - |
| ik1_mi355x_sgl / protocol_sgl_A_random | 2.00 | 3.02 | 1.77 | 10.43 | 310.9 / 878.4 | 0 | fail | - |
| ik1_mi355x_sgl / protocol_sgl_A_random | 3.00 | 4.53 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_mi355x_sgl / protocol_sgl_A_random | 4.00 | 6.05 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_mi355x_sgl / protocol_sgl_A_random | 5.00 | 7.56 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_mi355x_sgl / protocol_sgl_B_text | 1.00 | 1.51 | 1.26 | 3.07 | 86.7 / 110.8 | 0 | PASS | - |
| ik1_mi355x_sgl / protocol_sgl_B_text | 1.50 | 2.27 | 0.34 | 19.95 | 288.2 / 353.4 | 0 | fail | - |
| ik1_mi355x_sgl / protocol_sgl_B_text | 2.00 | 3.02 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_mi355x_sgl / protocol_sgl_B_text | 3.00 | 4.53 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_mi355x_sgl / protocol_sgl_B_text | 4.00 | 6.05 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_mi355x_sgl / protocol_sgl_B_text | 5.00 | 7.56 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_pro6000_sgl / protocol_sgl_A_random | 1.00 | 0.97 | 0.92 | 1.29 | 32.5 / 53.3 | 0 | PASS | - |
| ik1_pro6000_sgl / protocol_sgl_A_random | 1.50 | 1.46 | 1.27 | 2.49 | 111.8 / 146.9 | 0 | PASS | - |
| ik1_pro6000_sgl / protocol_sgl_A_random | 2.00 | 1.94 | 1.35 | 31.88 | 135.0 / 202.8 | 0 | fail | - |
| ik1_pro6000_sgl / protocol_sgl_A_random | 3.00 | 2.91 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_pro6000_sgl / protocol_sgl_A_random | 4.00 | 3.88 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_pro6000_sgl / protocol_sgl_B_text | 1.00 | 0.97 | 1.02 | 6.79 | 41.1 / 68.1 | 0 | fail | - |
| ik1_pro6000_sgl / protocol_sgl_B_text | 1.50 | 1.46 | 0.91 | 12.49 | 124.7 / 166.5 | 0 | fail | - |
| ik1_pro6000_sgl / protocol_sgl_B_text | 2.00 | 1.94 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_pro6000_sgl / protocol_sgl_B_text | 3.00 | 2.91 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_pro6000_sgl / protocol_sgl_B_text | 4.00 | 3.88 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_pro6000_vllm / protocol_vllm_A_random | 1.00 | 0.97 | 0.71 | 18.43 | 232.3 / 718.3 | 0 | fail | - |
| ik1_pro6000_vllm / protocol_vllm_A_random | 1.50 | 1.46 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_pro6000_vllm / protocol_vllm_A_random | 2.00 | 1.94 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_pro6000_vllm / protocol_vllm_A_random | 3.00 | 2.91 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_pro6000_vllm / protocol_vllm_A_random | 4.00 | 3.88 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_pro6000_vllm / protocol_vllm_B_text | 1.00 | 0.97 | 0.19 | 8.00 | 321.3 / 408.9 | 0 | fail | - |
| ik1_pro6000_vllm / protocol_vllm_B_text | 1.50 | 1.46 | 0.10 | 29.56 | 638.0 / 777.3 | 0 | fail | - |
| ik1_pro6000_vllm / protocol_vllm_B_text | 2.00 | 1.94 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_pro6000_vllm / protocol_vllm_B_text | 3.00 | 2.91 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik1_pro6000_vllm / protocol_vllm_B_text | 4.00 | 3.88 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik2_pro6000_sched / protocol_swa15_B_text | 1.00 | 0.97 | 1.03 | 3.30 | 41.8 / 79.8 | 0 | PASS | - |
| ik2_pro6000_sched / protocol_swa15_B_text | 1.50 | 1.46 | 1.21 | 15.51 | 147.2 / 200.4 | 0 | fail | - |
| ik2_pro6000_sched / protocol_swa15_B_text | 2.00 | 1.94 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik2_pro6000_sched / protocol_swa15_B_text | 2.50 | 2.43 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik2_pro6000_sched / protocol_swa15_mixed_B_text | 1.00 | 0.97 | 1.04 | 2.75 | 41.2 / 77.0 | 0 | PASS | - |
| ik2_pro6000_sched / protocol_swa15_mixed_B_text | 1.50 | 1.46 | 1.23 | 7.17 | 145.3 / 204.7 | 0 | fail | - |
| ik2_pro6000_sched / protocol_swa15_mixed_B_text | 2.00 | 1.94 | 1.25 | inf | 177.1 / 198.0 | 0 | fail | - |
| ik2_pro6000_sched / protocol_swa15_mixed_B_text | 2.50 | 2.43 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik2_pro6000_sched / protocol_swa15_chunk16k_B_text | 1.00 | 0.97 | 1.03 | 3.05 | 41.7 / 79.3 | 0 | PASS | - |
| ik2_pro6000_sched / protocol_swa15_chunk16k_B_text | 1.50 | 1.46 | 1.18 | 13.94 | 143.2 / 165.8 | 0 | fail | - |
| ik2_pro6000_sched / protocol_swa15_chunk16k_B_text | 2.00 | 1.94 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik2_pro6000_sched / protocol_swa15_chunk16k_B_text | 2.50 | 2.43 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik2_pro6000_spec / protocol_swa15_B_text | 1.00 | 0.97 | 1.02 | 3.29 | 41.7 / 79.3 | 0 | PASS | - |
| ik2_pro6000_spec / protocol_swa15_B_text | 1.50 | 1.46 | 1.21 | 15.90 | 143.0 / 200.0 | 0 | fail | - |
| ik2_pro6000_spec / protocol_swa15_B_text | 2.00 | 1.94 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik2_pro6000_spec / protocol_swa15_B_text | 2.50 | 2.43 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik2_pro6000_swa / protocol_base_A_random | 1.00 | 0.97 | 0.94 | 1.92 | 33.8 / 55.8 | 0 | PASS | - |
| ik2_pro6000_swa / protocol_base_B_text | 1.00 | 0.97 | 1.02 | 3.36 | 41.8 / 79.6 | 0 | PASS | - |
| ik2_pro6000_swa / protocol_base_B_text | 1.50 | 1.46 | 1.18 | 14.24 | 141.7 / 168.0 | 0 | fail | - |
| ik2_pro6000_swa / protocol_base_B_text | 2.00 | 1.94 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik2_pro6000_swa / protocol_base_B_text | 2.50 | 2.43 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik2_pro6000_swa / protocol_swa15_B_text | 1.00 | 0.97 | 1.02 | 3.34 | 41.8 / 79.5 | 0 | PASS | - |
| ik2_pro6000_swa / protocol_swa15_B_text | 1.50 | 1.46 | 1.21 | 15.50 | 146.2 / 204.0 | 0 | fail | - |
| ik2_pro6000_swa / protocol_swa15_B_text | 2.00 | 1.94 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik2_pro6000_swa / protocol_swa15_B_text | 2.50 | 2.43 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik3_5090_bf16 / protocol_swa25_B_text | 1.00 | 0.56 | 0.58 | 5.21 | 15.3 / 24.6 | 0 | fail | - |
| ik3_5090_bf16 / protocol_swa25_B_text | 1.50 | 0.85 | 0.76 | inf | 22.5 / 29.1 | 0 | fail | - |
| ik3_5090_bf16 / protocol_swa25_B_text | 2.00 | 1.13 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik3_5090_bf16 / protocol_swa25_B_text | 2.50 | 1.41 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik3_5090_bf16 / protocol_swa25_B_text | 3.00 | 1.69 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik3_5090_bf16 / protocol_swa25_nopg_B_text | 1.00 | 0.56 | 0.58 | 4.98 | 14.5 / 25.1 | 0 | fail | - |
| ik3_5090_bf16 / protocol_swa25_nopg_B_text | 1.50 | 0.85 | 0.77 | inf | 23.3 / 29.0 | 0 | fail | - |
| ik3_5090_bf16 / protocol_swa25_nopg_B_text | 2.00 | 1.13 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik3_5090_bf16 / protocol_swa25_nopg_B_text | 2.50 | 1.41 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik3_5090_bf16 / protocol_swa25_nopg_B_text | 3.00 | 1.69 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik4_mi355x_aiter / protocol_base_B_text | 1.00 | 1.51 | 1.51 | 6.11 | 86.3 / 118.0 | 0 | fail | - |
| ik4_mi355x_aiter / protocol_base_B_text | 1.25 | 1.89 | 1.18 | 9.24 | 246.8 / 391.0 | 0 | fail | - |
| ik4_mi355x_aiter / protocol_base_B_text | 1.50 | 2.27 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik4_mi355x_aiter / protocol_base_B_text | 2.00 | 3.02 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik5_pro6000_dspark / protocol_base_B_text | 1.00 | 0.97 | 1.02 | 3.31 | 42.0 / 79.2 | 0 | PASS | - |
| ik5_pro6000_dspark / protocol_base_B_text | 1.25 | 1.21 | 1.19 | inf | 63.1 / 117.5 | 0 | fail | - |
| ik5_pro6000_dspark / protocol_base_B_text | 1.50 | 1.46 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik5_pro6000_dspark / protocol_base_B_text | 2.00 | 1.94 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik5_pro6000_dspark / protocol_dspark_B_text | 1.00 | 0.97 | 1.01 | 2.91 | 19.0 / 42.5 | 0 | PASS | - |
| ik5_pro6000_dspark / protocol_dspark_B_text | 1.25 | 1.21 | 1.17 | inf | 25.6 / 67.6 | 0 | fail | - |
| ik5_pro6000_dspark / protocol_dspark_B_text | 1.50 | 1.46 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik5_pro6000_dspark / protocol_dspark_B_text | 2.00 | 1.94 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik7_mi355x_aiter / protocol_base_B_text | 1.00 | 1.51 | 1.49 | 6.19 | 85.8 / 116.5 | 0 | fail | - |
| ik7_mi355x_aiter / protocol_base_B_text | 1.25 | 1.89 | 1.19 | 9.93 | 244.5 / 389.0 | 0 | fail | - |
| ik7_mi355x_aiter / protocol_base_B_text | 1.50 | 2.27 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik8_5090_c4 / protocol_mem81_B_text | 1.00 | 0.45 | 0.50 | 2.54 | 13.0 / 23.3 | 0 | PASS | - |
| ik8_5090_c4 / protocol_mem81_B_text | 1.25 | 0.57 | 0.58 | 5.89 | 15.7 / 23.4 | 0 | fail | - |
| ik8_5090_c4 / protocol_mem81_B_text | 1.50 | 0.68 | 0.69 | 10.23 | 19.3 / 28.1 | 0 | fail | - |
| ik8_5090_c4 / protocol_mem81_B_text | 2.00 | 0.91 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik8_5090_c4 / protocol_mem81_mixed_B_text | 1.00 | 0.45 | 0.50 | 2.87 | 12.9 / 22.5 | 0 | PASS | - |
| ik8_5090_c4 / protocol_mem81_mixed_B_text | 1.25 | 0.57 | 0.58 | 7.79 | 17.9 / 25.9 | 0 | fail | - |
| ik8_5090_c4 / protocol_mem81_mixed_B_text | 1.50 | 0.68 | 0.71 | 10.26 | 20.6 / 27.8 | 0 | fail | - |
| ik8_5090_c4 / protocol_mem81_mixed_B_text | 2.00 | 0.91 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ik8_pro6000_c4 / protocol_mixed_B_text | 1.00 | 0.86 | 0.95 | 2.65 | 30.8 / 65.6 | 0 | PASS | - |
| ik8_pro6000_c4 / protocol_mixed_B_text | 1.13 | 0.97 | 1.01 | 2.75 | 40.6 / 77.6 | 0 | PASS | - |
| ik8_pro6000_c4 / protocol_mixed_B_text | 1.25 | 1.07 | 1.09 | 4.84 | 56.1 / 95.5 | 0 | fail | - |
| ik8_pro6000_c4 / protocol_mixed_B_text | 1.50 | 1.29 | 1.24 | 5.31 | 82.7 / 129.9 | 0 | fail | - |
| ik8_pro6000_c4 / protocol_mixed_dspark_B_text | 1.00 | 0.86 | 0.88 | 2.44 | 12.7 / 43.1 | 0 | PASS | - |
| ik8_pro6000_c4 / protocol_mixed_dspark_B_text | 1.13 | 0.97 | 1.01 | 2.75 | 19.2 / 43.0 | 0 | PASS | - |
| ik8_pro6000_c4 / protocol_mixed_dspark_B_text | 1.25 | 1.07 | 1.12 | 3.41 | 22.1 / 52.5 | 0 | PASS | - |
| ik8_pro6000_c4 / protocol_mixed_dspark_B_text | 1.50 | 1.29 | 1.27 | 4.83 | 33.8 / 78.5 | 0 | fail | - |
| ikC_b300_mixed_dspark / protocol_b300_A_random | 1.00 | 2.05 | 2.02 | 1.07 | 15.9 / 28.8 | 0 | PASS | - |
| ikC_b300_mixed_dspark / protocol_b300_A_random | 1.50 | 3.08 | 2.98 | 7.70 | 101.2 / 124.2 | 0 | fail | - |
| ikC_b300_mixed_dspark / protocol_b300_A_random | 2.00 | 4.11 | 3.04 | 92.07 | 106.1 / 128.0 | 0 | fail | - |
| ikC_b300_mixed_dspark / protocol_b300_A_random | 2.50 | 5.13 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ikC_b300_mixed_dspark / protocol_b300_A_random | 3.00 | 6.16 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ikC_b300_mixed_dspark / protocol_b300_B_text | 1.00 | 2.05 | 2.05 | 1.30 | 6.6 / 14.0 | 0 | PASS | - |
| ikC_b300_mixed_dspark / protocol_b300_B_text | 1.50 | 3.08 | 3.13 | 2.26 | 18.7 / 51.4 | 0 | PASS | - |
| ikC_b300_mixed_dspark / protocol_b300_B_text | 2.00 | 4.11 | 3.38 | inf | 93.1 / 115.9 | 0 | fail | - |
| ikC_b300_mixed_dspark / protocol_b300_B_text | 2.50 | 5.13 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ikC_b300_mixed_dspark / protocol_b300_B_text | 3.00 | 6.16 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ikC_b300_mixed_dspark / protocol_mixed_dspark_b300ext_B_text | 1.75 | 3.59 | 3.27 | 11.12 | 69.9 / 116.1 | 0 | fail | - |
| ikC_b300_mixed_dspark / protocol_mixed_dspark_b300ext_B_text | 1.90 | 3.90 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| ikC_pro6000_mixed_dspark_rep_s7 / protocol_rep_B_text | 1.35 | 1.16 | 1.23 | 4.07 | 26.5 / 60.8 | 0 | fail | - |
| ikC_pro6000_mixed_dspark_rep_s7 / protocol_rep_B_text | 1.50 | 1.29 | 1.35 | 3.55 | 33.4 / 89.9 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_rep_s8 / protocol_rep_B_text | 1.35 | 1.16 | 1.12 | 2.69 | 24.3 / 50.5 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_rep_s8 / protocol_rep_B_text | 1.50 | 1.29 | 1.25 | 3.28 | 33.4 / 69.5 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_s7 / protocol_mixed_dspark_A_random | 1.13 | 0.97 | 0.93 | 2.06 | 39.1 / 63.1 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_s7 / protocol_mixed_dspark_A_random | 1.25 | 1.07 | 1.05 | 2.47 | 51.9 / 80.8 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_s7 / protocol_mixed_dspark_A_random | 1.35 | 1.16 | 1.12 | 2.59 | 65.5 / 102.7 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_s7 / protocol_mixed_dspark_A_random | 1.50 | 1.29 | 1.21 | 3.63 | 104.6 / 139.0 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_s7 / protocol_mixed_dspark_B_text | 1.13 | 0.97 | 0.94 | 3.10 | 17.5 / 51.3 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_s7 / protocol_mixed_dspark_B_text | 1.25 | 1.07 | 1.07 | 3.63 | 19.2 / 56.7 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_s7 / protocol_mixed_dspark_B_text | 1.35 | 1.16 | 1.16 | 4.38 | 27.3 / 67.0 | 0 | fail | - |
| ikC_pro6000_mixed_dspark_s7 / protocol_mixed_dspark_B_text | 1.50 | 1.29 | 1.35 | 3.57 | 33.7 / 95.1 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_s8 / protocol_mixed_dspark_A_random | 1.13 | 0.97 | 0.95 | 2.09 | 40.7 / 71.1 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_s8 / protocol_mixed_dspark_A_random | 1.25 | 1.07 | 1.04 | 2.49 | 52.1 / 90.9 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_s8 / protocol_mixed_dspark_A_random | 1.35 | 1.16 | 1.11 | 2.61 | 65.2 / 102.6 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_s8 / protocol_mixed_dspark_A_random | 1.50 | 1.29 | 1.20 | 5.40 | 107.2 / 150.7 | 0 | fail | - |
| ikC_pro6000_mixed_dspark_s8 / protocol_mixed_dspark_B_text | 1.13 | 0.97 | 0.96 | 2.11 | 16.7 / 33.3 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_s8 / protocol_mixed_dspark_B_text | 1.25 | 1.07 | 1.03 | 2.74 | 21.0 / 44.3 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_s8 / protocol_mixed_dspark_B_text | 1.35 | 1.16 | 1.12 | 2.70 | 23.8 / 49.4 | 0 | PASS | - |
| ikC_pro6000_mixed_dspark_s8 / protocol_mixed_dspark_B_text | 1.50 | 1.29 | 1.25 | 3.26 | 33.0 / 70.8 | 0 | PASS | - |
| ikC_pro6000_mixed_s7 / protocol_mixed_A_random | 1.13 | 0.97 | 0.94 | 2.00 | 31.9 / 51.7 | 0 | PASS | - |
| ikC_pro6000_mixed_s7 / protocol_mixed_A_random | 1.20 | 1.03 | 1.02 | 2.35 | 38.7 / 60.2 | 0 | PASS | - |
| ikC_pro6000_mixed_s7 / protocol_mixed_A_random | 1.25 | 1.07 | 1.05 | 2.38 | 41.8 / 68.5 | 0 | PASS | - |
| ikC_pro6000_mixed_s7 / protocol_mixed_B_text | 1.13 | 0.97 | 0.96 | 2.78 | 39.5 / 71.8 | 0 | PASS | - |
| ikC_pro6000_mixed_s7 / protocol_mixed_B_text | 1.20 | 1.03 | 1.01 | 3.71 | 43.0 / 77.0 | 0 | PASS | - |
| ikC_pro6000_mixed_s7 / protocol_mixed_B_text | 1.25 | 1.07 | 1.06 | 3.45 | 46.1 / 88.3 | 0 | PASS | - |
| ikC_pro6000_mixed_s8 / protocol_mixed_A_random | 1.13 | 0.97 | 0.95 | 1.94 | 33.4 / 56.7 | 0 | PASS | - |
| ikC_pro6000_mixed_s8 / protocol_mixed_A_random | 1.20 | 1.03 | 1.00 | 2.18 | 37.0 / 65.8 | 0 | PASS | - |
| ikC_pro6000_mixed_s8 / protocol_mixed_A_random | 1.25 | 1.07 | 1.05 | 2.25 | 40.7 / 72.5 | 0 | PASS | - |
| ikC_pro6000_mixed_s8 / protocol_mixed_B_text | 1.13 | 0.97 | 0.96 | 1.97 | 34.6 / 55.1 | 0 | PASS | - |
| ikC_pro6000_mixed_s8 / protocol_mixed_B_text | 1.20 | 1.03 | 1.06 | 2.88 | 42.9 / 65.9 | 0 | PASS | - |
| ikC_pro6000_mixed_s8 / protocol_mixed_B_text | 1.25 | 1.07 | 1.05 | 2.85 | 48.9 / 77.5 | 0 | PASS | - |
| mx10_best_dc_s7 / protocol_best_dc | 1.92 | 3.25 | 3.21 | 3.18 | 23.3 / 55.6 | 0 | PASS | 3.59 |
| mx10_best_dc_s7 / protocol_best_dc | 2.06 | 3.50 | 3.46 | 2.30 | 30.6 / 59.3 | 0 | PASS | 3.57 |
| mx10_best_dc_s7 / protocol_best_dc | 2.21 | 3.75 | 3.63 | 4.67 | 54.8 / 71.1 | 0 | fail | 3.57 |
| mx10_best_dc_s8 / protocol_best_dc | 1.92 | 3.25 | 3.23 | 2.14 | 24.7 / 53.8 | 1 | PASS | 3.63 |
| mx10_best_dc_s8 / protocol_best_dc | 2.06 | 3.50 | 3.45 | 3.06 | 36.7 / 65.2 | 1 | PASS | 3.60 |
| mx10_best_dc_s8 / protocol_best_dc | 2.21 | 3.75 | 3.66 | 8.12 | 55.7 / 73.1 | 0 | fail | 3.59 |
| mx11_or_dc / protocol_or_dc | 1.92 | 3.25 | 3.23 | 1.99 | 26.2 / 54.9 | 0 | PASS | 2.83 |
| mx11_or_dc / protocol_or_dc | 2.06 | 3.50 | 3.45 | 5.58 | 41.3 / 56.6 | 0 | fail | 2.78 |
| mx11_or_dc / protocol_or_dc | 2.21 | 3.75 | 3.61 | 12.48 | 44.3 / 60.8 | 0 | fail | 2.80 |
| mx12_k4_dc / protocol_k4_dc | 1.92 | 3.25 | 3.23 | 2.44 | 22.2 / 50.5 | 0 | PASS | 4.37 |
| mx12_k4_dc / protocol_k4_dc | 2.06 | 3.50 | 3.46 | 3.36 | 33.9 / 58.4 | 0 | PASS | 4.28 |
| mx12_k4_dc / protocol_k4_dc | 2.21 | 3.75 | 3.60 | 8.45 | 46.0 / 68.7 | 0 | fail | 4.28 |
| mx12_k4_dc / protocol_k4_dc_text | 1.92 | 3.25 | 2.94 | 3.56 | 29.2 / 71.4 | 0 | PASS | 3.09 |
| mx12_k4_dc / protocol_k4_dc_text | 2.06 | 3.50 | 3.12 | 7.14 | 50.2 / 73.6 | 0 | fail | 3.10 |
| mx1_or_b4k / protocol_or_b4096 | 1.92 | 3.25 | 3.21 | 5.74 | 40.4 / 49.1 | 0 | fail | - |
| mx1_or_b4k / protocol_or_b4096 | 2.06 | 3.50 | 3.37 | 13.89 | 41.0 / 50.4 | 0 | fail | - |
| mx1_or_b4k / protocol_or_b4096 | 2.21 | 3.75 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mx1_or_b6k / protocol_or_b6144 | 1.92 | 3.25 | 3.23 | 3.79 | 41.3 / 58.8 | 0 | PASS | - |
| mx1_or_b6k / protocol_or_b6144 | 2.06 | 3.50 | 3.40 | 11.96 | 46.0 / 59.1 | 0 | fail | - |
| mx1_or_b6k / protocol_or_b6144 | 2.21 | 3.75 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mx1_or_mtp3 / protocol_or_mtp3 | 1.92 | 3.25 | 3.23 | 2.26 | 27.6 / 55.5 | 0 | PASS | - |
| mx1_or_mtp3 / protocol_or_mtp3 | 2.06 | 3.50 | 3.45 | 5.85 | 40.6 / 59.8 | 0 | fail | - |
| mx1_or_mtp3 / protocol_or_mtp3 | 2.21 | 3.75 | 3.59 | 13.17 | 42.7 / 62.5 | 0 | fail | - |
| mx1_or_mtp3 / protocol_or_mtp3_text | 1.77 | 3.00 | 2.77 | 1.66 | 14.1 / 35.0 | 0 | PASS | 2.79 |
| mx1_or_mtp3 / protocol_or_mtp3_text | 1.92 | 3.25 | 2.95 | 1.98 | 19.3 / 56.4 | 0 | PASS | 2.79 |
| mx1_v034 / protocol_v034_stock | 1.92 | 3.25 | 2.21 | 132.67 | 72.8 / 99.5 | 0 | fail | - |
| mx1_v034 / protocol_v034_stock | 2.06 | 3.50 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mx1_v034 / protocol_v034_stock | 2.21 | 3.75 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mx2_v034_asm / protocol_v034_asm | 1.92 | 3.25 | 3.22 | 4.93 | 44.1 / 60.8 | 0 | fail | - |
| mx2_v034_asm / protocol_v034_asm | 2.06 | 3.50 | 3.35 | 14.57 | 47.1 / 64.4 | 0 | fail | - |
| mx2_v034_asm / protocol_v034_asm | 2.21 | 3.75 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mx2_v034_orport / protocol_v034_orport | 1.92 | 3.25 | 3.23 | 3.00 | 35.9 / 61.3 | 1 | PASS | - |
| mx2_v034_orport / protocol_v034_orport | 2.06 | 3.50 | 3.41 | 10.50 | 45.7 / 60.7 | 0 | fail | - |
| mx2_v034_orport / protocol_v034_orport | 2.21 | 3.75 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mx3_or_seqs128 / protocol_or_seqs128 | 1.92 | 3.25 | 3.22 | 3.18 | 45.9 / 82.9 | 2 | PASS | 2.81 |
| mx3_or_seqs128 / protocol_or_seqs128 | 2.06 | 3.50 | 3.43 | 8.84 | 79.8 / 97.3 | 0 | fail | 2.77 |
| mx3_or_seqs128 / protocol_or_seqs128 | 2.21 | 3.75 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mx3_or_seqs96 / protocol_or_seqs96 | 1.92 | 3.25 | 3.23 | 2.46 | 32.0 / 65.0 | 0 | PASS | 2.82 |
| mx3_or_seqs96 / protocol_or_seqs96 | 2.06 | 3.50 | 3.45 | 4.93 | 59.4 / 77.7 | 0 | fail | 2.80 |
| mx3_or_seqs96 / protocol_or_seqs96 | 2.21 | 3.75 | 3.59 | 12.48 | 66.3 / 83.8 | 1 | fail | 2.80 |
| mx4_or_b12k / protocol_or_b12k | 1.92 | 3.25 | 3.23 | 2.66 | 36.1 / 62.0 | 0 | PASS | 2.81 |
| mx4_or_b12k / protocol_or_b12k | 2.06 | 3.50 | 3.43 | 9.64 | 45.2 / 61.1 | 0 | fail | 2.79 |
| mx4_or_b12k / protocol_or_b12k | 2.21 | 3.75 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mx5_or_mtp3_seqs96 / protocol_or_mtp3_seqs96 | 1.92 | 3.25 | 3.23 | 2.27 | 27.9 / 56.3 | 1 | PASS | 3.64 |
| mx5_or_mtp3_seqs96 / protocol_or_mtp3_seqs96 | 2.06 | 3.50 | 3.45 | 4.16 | 43.3 / 69.3 | 0 | fail | 3.62 |
| mx5_or_mtp3_seqs96 / protocol_or_mtp3_seqs96 | 2.21 | 3.75 | 3.65 | 10.53 | 57.9 / 77.2 | 0 | fail | 3.61 |
| mx5_or_mtp4 / protocol_or_mtp4 | 1.92 | 3.25 | 3.23 | 2.20 | 23.3 / 48.5 | 0 | PASS | 4.35 |
| mx5_or_mtp4 / protocol_or_mtp4 | 2.06 | 3.50 | 3.45 | 4.77 | 37.2 / 55.7 | 0 | fail | 4.33 |
| mx5_or_mtp4 / protocol_or_mtp4 | 2.21 | 3.75 | 3.61 | 11.62 | 41.0 / 64.8 | 0 | fail | 4.28 |
| mx5_or_mtp4 / protocol_or_mtp4_text | 1.77 | 3.00 | 2.82 | 1.93 | 17.7 / 60.1 | 0 | PASS | 3.09 |
| mx5_or_mtp4 / protocol_or_mtp4_text | 1.92 | 3.25 | 3.01 | 3.18 | 24.3 / 60.4 | 0 | PASS | 3.10 |
| mx6_or_mtp3_seqs112 / protocol_or_mtp3_seqs112 | 1.92 | 3.25 | 3.23 | 2.94 | 37.7 / 64.8 | 0 | PASS | 3.62 |
| mx6_or_mtp3_seqs112 / protocol_or_mtp3_seqs112 | 2.06 | 3.50 | 3.43 | 7.72 | 59.7 / 78.5 | 0 | fail | 3.55 |
| mx6_or_mtp3_seqs112 / protocol_or_mtp3_seqs112 | 2.21 | 3.75 | 3.49 | 16.24 | 60.6 / 83.7 | 0 | fail | 3.54 |
| mx6_or_mtp3_seqs80 / protocol_or_mtp3_seqs80 | 1.92 | 3.25 | 3.23 | 2.22 | 27.4 / 55.4 | 2 | PASS | 3.61 |
| mx6_or_mtp3_seqs80 / protocol_or_mtp3_seqs80 | 2.06 | 3.50 | 3.45 | 4.16 | 41.9 / 63.8 | 0 | fail | 3.60 |
| mx6_or_mtp3_seqs80 / protocol_or_mtp3_seqs80 | 2.21 | 3.75 | 3.63 | 10.75 | 52.2 / 72.6 | 0 | fail | 3.59 |
| mx6_or_mtp3_seqs96_s7 / protocol_or_mtp3_seqs96 | 1.92 | 3.25 | 3.18 | 2.98 | 24.7 / 57.2 | 0 | PASS | 3.57 |
| mx6_or_mtp3_seqs96_s7 / protocol_or_mtp3_seqs96 | 2.06 | 3.50 | 3.45 | 2.68 | 36.2 / 65.7 | 0 | PASS | 3.57 |
| mx6_or_mtp3_seqs96_s7 / protocol_or_mtp3_seqs96 | 2.21 | 3.75 | 3.57 | 6.75 | 56.5 / 71.9 | 0 | fail | 3.62 |
| mx6_or_mtp4_seqs96 / protocol_or_mtp4_seqs96 | 1.92 | 3.25 | 3.23 | 2.81 | 28.4 / 56.0 | 0 | PASS | 4.34 |
| mx6_or_mtp4_seqs96 / protocol_or_mtp4_seqs96 | 2.06 | 3.50 | 3.45 | 6.20 | 47.4 / 68.6 | 0 | fail | 4.28 |
| mx6_or_mtp4_seqs96 / protocol_or_mtp4_seqs96 | 2.21 | 3.75 | 3.55 | 12.90 | 48.5 / 71.5 | 0 | fail | 4.32 |
| mx8_or_mtp3_seqs96_block / protocol_or_mtp3_seqs96_block | 1.92 | 3.25 | 3.23 | 2.39 | 29.0 / 57.8 | 2 | PASS | 3.62 |
| mx8_or_mtp3_seqs96_block / protocol_or_mtp3_seqs96_block | 2.06 | 3.50 | 3.45 | 4.90 | 49.3 / 72.8 | 0 | fail | 3.58 |
| mx8_or_mtp3_seqs96_block / protocol_or_mtp3_seqs96_block | 2.21 | 3.75 | 3.61 | 12.01 | 58.5 / 78.7 | 0 | fail | 3.62 |
| mx8_or_mtp3_seqs96_block / protocol_or_mtp3_seqs96_block_text | 1.77 | 3.00 | 2.81 | 2.30 | 27.1 / 72.2 | 0 | PASS | 2.79 |
| mx8_or_mtp3_seqs96_block / protocol_or_mtp3_seqs96_block_text | 1.92 | 3.25 | 2.95 | 3.80 | 33.5 / 81.5 | 0 | PASS | 2.80 |
| mx8_or_mtp3_seqs96_prob / protocol_or_mtp3_seqs96_prob | 1.92 | 3.25 | 3.23 | 2.73 | 35.0 / 67.1 | 0 | PASS | 3.52 |
| mx8_or_mtp3_seqs96_prob / protocol_or_mtp3_seqs96_prob | 2.06 | 3.50 | 3.44 | 6.52 | 58.4 / 78.3 | 0 | fail | 3.53 |
| mx8_or_mtp3_seqs96_prob / protocol_or_mtp3_seqs96_prob | 2.21 | 3.75 | 3.54 | 14.34 | 59.7 / 84.7 | 1 | fail | 3.54 |
| mx8_or_mtp3_seqs96_prob / protocol_or_mtp3_seqs96_prob_text | 1.77 | 3.00 | 2.80 | 2.40 | 28.0 / 73.2 | 0 | PASS | 2.79 |
| mx8_or_mtp3_seqs96_prob / protocol_or_mtp3_seqs96_prob_text | 1.92 | 3.25 | 2.93 | 4.05 | 36.2 / 82.3 | 0 | fail | 2.79 |
| mx9_best_r2s7 / protocol_best | 2.06 | 3.50 | 3.43 | 4.53 | 56.7 / 75.1 | 0 | fail | 3.56 |
| mx9_best_r2s7 / protocol_best | 2.21 | 3.75 | 3.58 | 9.94 | 58.0 / 81.3 | 0 | fail | 3.57 |
| mx9_best_r2s8 / protocol_best | 2.06 | 3.50 | 3.45 | 4.62 | 51.8 / 70.5 | 0 | fail | 3.56 |
| mx9_best_r2s8 / protocol_best | 2.21 | 3.75 | 3.66 | 9.83 | 57.4 / 75.8 | 0 | fail | 3.59 |
| mx9_best_r3s8 / protocol_best | 2.06 | 3.50 | 3.45 | 5.20 | 54.1 / 71.8 | 1 | fail | 3.61 |
| mx9_best_r3s8 / protocol_best | 2.21 | 3.75 | 3.61 | 11.25 | 58.3 / 81.1 | 0 | fail | 3.55 |
| mxC1_s7_r1 / protocol_mx_recipe | 1.92 | 3.25 | 3.22 | 2.77 | 21.7 / 52.0 | 0 | PASS | 3.61 |
| mxC1_s7_r1 / protocol_mx_recipe | 2.06 | 3.50 | 3.41 | 2.23 | 28.9 / 57.2 | 0 | PASS | 3.59 |
| mxC1_s7_r1 / protocol_mx_recipe | 2.21 | 3.75 | 3.67 | 4.19 | 54.4 / 68.9 | 0 | fail | 3.63 |
| mxC1_s7_r1 / protocol_mx_recipe_text | 1.92 | 3.25 | 2.92 | 2.95 | 25.6 / 76.3 | 0 | PASS | 2.79 |
| mxC1_s7_r1 / protocol_mx_recipe_text | 2.06 | 3.50 | 3.14 | 6.05 | 42.8 / 74.1 | 0 | fail | 2.79 |
| mxC1_s7_r1 / protocol_or_recipe | 1.92 | 3.25 | 3.21 | 2.41 | 28.1 / 54.1 | 0 | PASS | 2.80 |
| mxC1_s7_r1 / protocol_or_recipe | 2.06 | 3.50 | 3.38 | 6.66 | 43.7 / 57.2 | 0 | fail | 2.79 |
| mxC1_s7_r1 / protocol_or_recipe_text | 1.92 | 3.25 | 3.02 | 4.20 | 31.9 / 60.9 | 0 | fail | 2.37 |
| mxC1_s7_r1 / protocol_or_recipe_text | 2.06 | 3.50 | 3.19 | 10.42 | 46.5 / 63.1 | 0 | fail | 2.38 |
| mxC1_s7_r2 / protocol_or_recipe | 1.92 | 3.25 | 3.20 | 2.97 | 37.8 / 61.7 | 0 | PASS | 2.81 |
| mxC1_s7_r2 / protocol_or_recipe | 2.06 | 3.50 | 3.34 | 10.01 | 45.4 / 61.9 | 0 | fail | 2.79 |
| mxC1_s7_r2 / protocol_or_recipe_text | 1.92 | 3.25 | 3.03 | 4.91 | 37.0 / 65.9 | 0 | fail | 2.37 |
| mxC1_s7_r2 / protocol_or_recipe_text | 2.06 | 3.50 | 3.15 | 13.83 | 48.2 / 60.4 | 0 | fail | 2.38 |
| mxC1_s7_r2 / protocol_mx_recipe | 1.92 | 3.25 | 3.21 | 1.80 | 22.3 / 52.3 | 0 | PASS | 3.58 |
| mxC1_s7_r2 / protocol_mx_recipe | 2.06 | 3.50 | 3.45 | 2.69 | 36.2 / 63.6 | 0 | PASS | 3.55 |
| mxC1_s7_r2 / protocol_mx_recipe | 2.21 | 3.75 | 3.62 | 6.84 | 56.3 / 70.2 | 0 | fail | 3.58 |
| mxC1_s7_r2 / protocol_mx_recipe_text | 1.92 | 3.25 | 2.92 | 3.30 | 29.4 / 79.2 | 0 | PASS | 2.79 |
| mxC1_s7_r2 / protocol_mx_recipe_text | 2.06 | 3.50 | 3.13 | 7.01 | 49.6 / 75.6 | 0 | fail | 2.79 |
| mxC1_s8_r1 / protocol_or_recipe | 1.92 | 3.25 | 3.22 | 2.57 | 35.2 / 61.8 | 0 | PASS | 2.82 |
| mxC1_s8_r1 / protocol_or_recipe | 2.06 | 3.50 | 3.44 | 9.32 | 44.2 / 61.9 | 0 | fail | 2.79 |
| mxC1_s8_r1 / protocol_mx_recipe | 1.92 | 3.25 | 3.23 | 1.72 | 20.3 / 51.2 | 0 | PASS | 3.60 |
| mxC1_s8_r1 / protocol_mx_recipe | 2.06 | 3.50 | 3.45 | 2.67 | 32.3 / 60.8 | 1 | PASS | 3.57 |
| mxC1_s8_r1 / protocol_mx_recipe | 2.21 | 3.75 | 3.66 | 6.37 | 54.6 / 70.7 | 0 | fail | 3.59 |
| mxC1_s8_r2 / protocol_mx_recipe | 1.92 | 3.25 | 3.23 | 2.05 | 22.3 / 52.1 | 0 | PASS | 3.61 |
| mxC1_s8_r2 / protocol_mx_recipe | 2.06 | 3.50 | 3.44 | 2.63 | 32.4 / 59.7 | 0 | PASS | 3.58 |
| mxC1_s8_r2 / protocol_mx_recipe | 2.21 | 3.75 | 3.67 | 6.52 | 54.5 / 76.3 | 0 | fail | 3.52 |
| mxC1_s8_r2 / protocol_or_recipe | 1.92 | 3.25 | 3.21 | 2.62 | 33.8 / 61.6 | 1 | PASS | 2.81 |
| mxC1_s8_r2 / protocol_or_recipe | 2.06 | 3.50 | 3.44 | 9.40 | 45.0 / 61.0 | 0 | fail | 2.81 |
| mxC1_text_s8 / protocol_mx_recipe_text | 1.92 | 3.25 | 3.20 | 2.04 | 30.7 / 60.9 | 0 | PASS | 2.80 |
| mxC1_text_s8 / protocol_mx_recipe_text | 2.06 | 3.50 | 3.53 | 2.95 | 45.6 / 67.9 | 0 | PASS | 2.79 |
| mxC1_text_s8 / protocol_or_recipe_text | 1.92 | 3.25 | 3.08 | 4.05 | 43.4 / 59.0 | 0 | fail | 2.38 |
| mxC1_text_s8 / protocol_or_recipe_text | 2.06 | 3.50 | 3.41 | 9.34 | 46.2 / 57.7 | 0 | fail | 2.38 |
| mxC2_s7_r1 / protocol_or_recipe | 2.19 | 3.25 | 3.19 | 3.50 | 41.4 / 61.8 | 0 | PASS | 2.81 |
| mxC2_s7_r1 / protocol_or_recipe | 2.36 | 3.50 | 3.33 | 11.39 | 46.3 / 63.8 | 0 | fail | 2.80 |
| mxC2_s7_r1 / protocol_or_recipe_text | 2.19 | 3.25 | 2.96 | 13.70 | 41.5 / 62.3 | 0 | fail | 2.38 |
| mxC2_s7_r1 / protocol_mx_recipe | 2.19 | 3.25 | 3.19 | 1.90 | 23.5 / 55.4 | 0 | PASS | 3.60 |
| mxC2_s7_r1 / protocol_mx_recipe | 2.36 | 3.50 | 3.42 | 2.99 | 39.6 / 65.7 | 1 | PASS | 3.58 |
| mxC2_s7_r1 / protocol_mx_recipe | 2.52 | 3.75 | 3.56 | 9.06 | 57.3 / 75.7 | 0 | fail | 3.57 |
| mxC2_s7_r1 / protocol_mx_recipe_text | 2.19 | 3.25 | 2.95 | 7.26 | 40.0 / 76.5 | 0 | fail | 2.79 |
| mxC2_s7_r1 / protocol_mx_recipe_text | 2.36 | 3.50 | 3.12 | inf | 46.4 / 77.0 | 0 | fail | 2.79 |
| mxC2_s7_r1 / protocol_mx_stack | 2.19 | 3.25 | 3.22 | 1.62 | 19.4 / 44.2 | 0 | PASS | 3.61 |
| mxC2_s7_r1 / protocol_mx_stack | 2.36 | 3.50 | 3.46 | 2.17 | 28.7 / 57.9 | 1 | PASS | 3.61 |
| mxC2_s7_r1 / protocol_mx_stack | 2.52 | 3.75 | 3.64 | 4.25 | 53.2 / 68.6 | 0 | fail | 3.57 |
| mxC2_s7_r1 / protocol_mx_stack | 2.69 | 4.00 | 3.75 | 14.73 | 56.3 / 75.1 | 0 | fail | 3.57 |
| mxC2_s7_r1 / protocol_mx_stack_text | 2.36 | 3.50 | 3.15 | 12.20 | 37.8 / 74.0 | 0 | fail | 2.79 |
| mxC2_s7_r1 / protocol_mx_stack_text | 2.52 | 3.75 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mxC2_s7_r2 / protocol_mx_stack | 2.19 | 3.25 | 3.22 | 2.56 | 18.4 / 46.5 | 0 | PASS | 3.64 |
| mxC2_s7_r2 / protocol_mx_stack | 2.36 | 3.50 | 3.42 | 1.79 | 22.7 / 51.9 | 0 | PASS | 3.59 |
| mxC2_s7_r2 / protocol_mx_stack | 2.52 | 3.75 | 3.69 | 2.83 | 39.7 / 63.7 | 1 | PASS | 3.61 |
| mxC2_s7_r2 / protocol_mx_stack | 2.69 | 4.00 | 3.83 | 8.58 | 53.7 / 71.5 | 0 | fail | 3.58 |
| mxC2_s7_r2 / protocol_mx_recipe | 2.19 | 3.25 | 3.19 | 1.55 | 18.6 / 42.5 | 0 | PASS | 3.60 |
| mxC2_s7_r2 / protocol_mx_recipe | 2.36 | 3.50 | 3.46 | 1.99 | 25.7 / 55.4 | 0 | PASS | 3.60 |
| mxC2_s7_r2 / protocol_mx_recipe | 2.52 | 3.75 | 3.68 | 3.29 | 48.6 / 67.6 | 0 | PASS | 3.56 |
| mxC2_s7_r2 / protocol_or_recipe | 2.19 | 3.25 | 3.20 | 2.30 | 26.9 / 53.2 | 0 | PASS | 2.79 |
| mxC2_s7_r2 / protocol_or_recipe | 2.36 | 3.50 | 3.41 | 5.23 | 42.4 / 56.3 | 0 | fail | 2.79 |
| mxC2_s8_r1 / protocol_mx_recipe | 2.19 | 3.25 | 3.23 | 2.24 | 26.1 / 56.3 | 1 | PASS | 3.62 |
| mxC2_s8_r1 / protocol_mx_recipe | 2.36 | 3.50 | 3.44 | 3.44 | 39.9 / 68.2 | 0 | PASS | 3.57 |
| mxC2_s8_r1 / protocol_mx_recipe | 2.52 | 3.75 | 3.67 | 8.87 | 56.6 / 79.7 | 0 | fail | 3.60 |
| mxC2_s8_r1 / protocol_mx_recipe_text | 2.19 | 3.25 | 3.30 | 3.03 | 40.7 / 72.7 | 0 | PASS | 2.78 |
| mxC2_s8_r1 / protocol_mx_recipe_text | 2.36 | 3.50 | 3.60 | 7.60 | 64.3 / 77.2 | 0 | fail | 2.80 |
| mxC2_s8_r1 / protocol_mx_stack | 2.19 | 3.25 | 3.23 | 1.60 | 17.7 / 48.6 | 0 | PASS | 3.62 |
| mxC2_s8_r1 / protocol_mx_stack | 2.36 | 3.50 | 3.44 | 2.20 | 27.6 / 54.7 | 0 | PASS | 3.57 |
| mxC2_s8_r1 / protocol_mx_stack | 2.52 | 3.75 | 3.65 | 4.75 | 49.1 / 69.2 | 0 | fail | 3.51 |
| mxC2_s8_r1 / protocol_mx_stack | 2.69 | 4.00 | 3.85 | 11.14 | 55.2 / 74.0 | 0 | fail | 3.63 |
| mxC2_s8_r1 / protocol_mx_stack_text | 2.36 | 3.50 | 3.55 | 3.54 | 43.9 / 72.4 | 0 | PASS | 2.79 |
| mxC2_s8_r1 / protocol_mx_stack_text | 2.52 | 3.75 | 3.53 | 13.44 | 66.4 / 77.5 | 0 | fail | 2.79 |
| mxC2_s8_r1 / protocol_or_recipe | 2.19 | 3.25 | 3.23 | 4.32 | 41.7 / 62.2 | 0 | fail | 2.83 |
| mxC2_s8_r1 / protocol_or_recipe | 2.36 | 3.50 | 3.38 | 13.23 | 46.6 / 61.2 | 0 | fail | 2.78 |
| mxC2_s8_r1 / protocol_or_recipe_text | 2.19 | 3.25 | 3.28 | 8.40 | 44.9 / 60.7 | 0 | fail | 2.37 |
| mxC2_s8_r2 / protocol_mx_stack | 2.19 | 3.25 | 3.23 | 2.03 | 21.4 / 51.5 | 0 | PASS | 3.64 |
| mxC2_s8_r2 / protocol_mx_stack | 2.36 | 3.50 | 3.45 | 2.41 | 30.8 / 58.7 | 1 | PASS | 3.57 |
| mxC2_s8_r2 / protocol_mx_stack | 2.52 | 3.75 | 3.66 | 5.69 | 53.9 / 72.0 | 0 | fail | 3.61 |
| mxC2_s8_r2 / protocol_mx_stack | 2.69 | 4.00 | 3.80 | 12.76 | 56.2 / 75.2 | 0 | fail | 3.61 |
| mxC2_s8_r2 / protocol_or_recipe | 2.19 | 3.25 | 3.23 | 5.39 | 44.1 / 59.6 | 0 | fail | 2.81 |
| mxC2_s8_r2 / protocol_or_recipe | 2.36 | 3.50 | 3.36 | 14.51 | 47.2 / 64.4 | 0 | fail | 2.80 |
| mxC2_s8_r2 / protocol_mx_recipe | 2.19 | 3.25 | 3.23 | 2.05 | 25.1 / 55.3 | 0 | PASS | 3.61 |
| mxC2_s8_r2 / protocol_mx_recipe | 2.36 | 3.50 | 3.45 | 4.17 | 43.6 / 68.5 | 0 | fail | 3.60 |
| mxC2_s8_r2 / protocol_mx_recipe | 2.52 | 3.75 | 3.62 | 10.08 | 57.4 / 75.2 | 1 | fail | 3.60 |
| mxC2b_s7 / protocol_or_recipe | 2.02 | 3.00 | 2.96 | 2.27 | 22.8 / 54.9 | 1 | PASS | 2.79 |
| mxC2b_s7 / protocol_or_recipe | 2.19 | 3.25 | 3.18 | 3.09 | 34.6 / 57.6 | 0 | PASS | 2.80 |
| mxC2b_s7 / protocol_or_recipe | 2.36 | 3.50 | 3.34 | 10.19 | 45.8 / 61.6 | 0 | fail | 2.79 |
| mxC2b_s7 / protocol_or_recipe_text | 1.85 | 2.75 | 2.53 | 1.91 | 17.7 / 57.6 | 0 | PASS | 2.37 |
| mxC2b_s7 / protocol_or_recipe_text | 2.02 | 3.00 | 2.80 | 3.82 | 31.0 / 56.7 | 0 | PASS | 2.37 |
| mxC2b_s7 / protocol_or_recipe_text | 2.19 | 3.25 | 2.97 | 13.08 | 40.4 / 61.3 | 0 | fail | 2.37 |
| mxC2b_s7 / protocol_or_recipe_text | 2.36 | 3.50 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mxC2b_s7 / protocol_mx_recipe | 2.02 | 3.00 | 2.96 | 1.53 | 15.5 / 38.3 | 0 | PASS | 3.57 |
| mxC2b_s7 / protocol_mx_recipe | 2.19 | 3.25 | 3.18 | 1.83 | 22.8 / 52.0 | 0 | PASS | 3.59 |
| mxC2b_s7 / protocol_mx_recipe | 2.36 | 3.50 | 3.44 | 2.85 | 37.8 / 65.2 | 0 | PASS | 3.52 |
| mxC2b_s7 / protocol_mx_recipe_text | 2.02 | 3.00 | 2.90 | 2.41 | 24.8 / 53.5 | 0 | PASS | 2.79 |
| mxC2b_s7 / protocol_mx_recipe_text | 2.19 | 3.25 | 2.96 | 6.87 | 39.0 / 75.7 | 0 | fail | 2.79 |
| mxC2b_s7 / protocol_mx_recipe_text | 2.36 | 3.50 | 3.13 | inf | 44.6 / 75.7 | 0 | fail | 2.79 |
| mxC2b_s7 / protocol_mx_stack | 2.02 | 3.00 | 2.96 | 1.39 | 13.5 / 31.0 | 0 | PASS | 3.56 |
| mxC2b_s7 / protocol_mx_stack | 2.19 | 3.25 | 3.22 | 1.55 | 18.3 / 41.5 | 0 | PASS | 3.63 |
| mxC2b_s7 / protocol_mx_stack | 2.36 | 3.50 | 3.46 | 2.01 | 26.0 / 55.4 | 0 | PASS | 3.59 |
| mxC2b_s7 / protocol_mx_stack | 2.52 | 3.75 | 3.69 | 3.40 | 47.9 / 66.5 | 0 | PASS | 3.61 |
| mxC2b_s7 / protocol_mx_stack_text | 2.02 | 3.00 | 2.94 | 2.03 | 20.5 / 43.6 | 0 | PASS | 2.78 |
| mxC2b_s7 / protocol_mx_stack_text | 2.19 | 3.25 | 3.06 | 4.23 | 32.4 / 70.5 | 0 | fail | 2.78 |
| mxC2b_s7 / protocol_mx_stack_text | 2.36 | 3.50 | 3.17 | 10.93 | 35.9 / 72.6 | 0 | fail | 2.79 |
| mxC2b_s8 / protocol_mx_stack | 2.02 | 3.00 | 2.98 | 1.93 | 16.0 / 40.1 | 0 | PASS | 3.63 |
| mxC2b_s8 / protocol_mx_stack | 2.19 | 3.25 | 3.23 | 1.67 | 18.6 / 51.0 | 0 | PASS | 3.61 |
| mxC2b_s8 / protocol_mx_stack | 2.36 | 3.50 | 3.46 | 2.42 | 30.2 / 58.8 | 0 | PASS | 3.61 |
| mxC2b_s8 / protocol_mx_stack | 2.52 | 3.75 | 3.68 | 5.81 | 53.5 / 72.8 | 0 | fail | 3.54 |
| mxC2b_s8 / protocol_mx_stack_text | 2.02 | 3.00 | 3.04 | 1.61 | 19.4 / 52.9 | 0 | PASS | 2.79 |
| mxC2b_s8 / protocol_mx_stack_text | 2.19 | 3.25 | 3.28 | 2.16 | 31.2 / 62.8 | 0 | PASS | 2.78 |
| mxC2b_s8 / protocol_mx_stack_text | 2.36 | 3.50 | 3.59 | 4.40 | 50.1 / 73.9 | 0 | fail | 2.80 |
| mxC2b_s8 / protocol_mx_recipe | 2.02 | 3.00 | 2.97 | 1.59 | 16.6 / 43.2 | 0 | PASS | 3.62 |
| mxC2b_s8 / protocol_mx_recipe | 2.19 | 3.25 | 3.23 | 2.04 | 25.2 / 56.1 | 0 | PASS | 3.61 |
| mxC2b_s8 / protocol_mx_recipe | 2.36 | 3.50 | 3.43 | 4.13 | 44.0 / 69.8 | 0 | fail | 3.59 |
| mxC2b_s8 / protocol_mx_recipe_text | 2.02 | 3.00 | 3.05 | 1.93 | 23.1 / 64.1 | 0 | PASS | 2.79 |
| mxC2b_s8 / protocol_mx_recipe_text | 2.19 | 3.25 | 3.30 | 3.27 | 42.8 / 74.2 | 0 | PASS | 2.78 |
| mxC2b_s8 / protocol_mx_recipe_text | 2.36 | 3.50 | 3.59 | 8.52 | 65.7 / 77.4 | 0 | fail | 2.80 |
| mxC2b_s8 / protocol_or_recipe | 2.02 | 3.00 | 2.97 | 1.78 | 23.1 / 55.6 | 1 | PASS | 2.82 |
| mxC2b_s8 / protocol_or_recipe | 2.19 | 3.25 | 3.23 | 5.24 | 44.1 / 60.6 | 0 | fail | 2.80 |
| mxC2b_s8 / protocol_or_recipe | 2.36 | 3.50 | 3.36 | 14.36 | 46.9 / 63.5 | 0 | fail | 2.79 |
| mxC2b_s8 / protocol_or_recipe_text | 1.85 | 2.75 | 2.78 | 1.60 | 19.1 / 39.6 | 0 | PASS | 2.38 |
| mxC2b_s8 / protocol_or_recipe_text | 2.02 | 3.00 | 3.07 | 4.20 | 32.0 / 59.1 | 0 | fail | 2.38 |
| mxC2b_s8 / protocol_or_recipe_text | 2.19 | 3.25 | 3.25 | 9.43 | 46.8 / 61.0 | 0 | fail | 2.37 |
| mxC2b_s8 / protocol_or_recipe_text | 2.36 | 3.50 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mxF_s7_r1 / protocol_mx_recipe | 1.92 | 3.25 | 3.19 | 3.21 | 23.3 / 55.0 | 1 | PASS | 3.60 |
| mxF_s7_r1 / protocol_mx_recipe | 2.12 | 3.60 | 3.55 | 2.84 | 38.3 / 64.7 | 0 | PASS | 3.60 |
| mxF_s7_r1 / protocol_mx_recipe | 2.18 | 3.70 | 3.62 | 3.87 | 52.8 / 70.3 | 0 | PASS | 3.58 |
| mxF_s7_r1 / protocol_or_recipe | 1.77 | 3.00 | 2.96 | 1.57 | 18.9 / 42.8 | 1 | PASS | 2.81 |
| mxF_s7_r1 / protocol_or_recipe | 1.95 | 3.30 | 3.24 | 3.10 | 33.3 / 56.3 | 2 | PASS | 2.81 |
| mxF_s7_r1 / protocol_or_recipe | 2.00 | 3.40 | 3.28 | 4.34 | 41.2 / 58.3 | 0 | fail | 2.77 |
| mxF_s7_r2 / protocol_or_recipe | 1.77 | 3.00 | 2.96 | 2.56 | 23.5 / 55.8 | 0 | PASS | 2.76 |
| mxF_s7_r2 / protocol_or_recipe | 1.95 | 3.30 | 3.23 | 4.42 | 40.4 / 59.5 | 0 | fail | 2.80 |
| mxF_s7_r2 / protocol_or_recipe | 2.00 | 3.40 | 3.29 | 6.35 | 44.9 / 61.6 | 0 | fail | 2.83 |
| mxF_s7_r2 / protocol_mx_recipe | 1.92 | 3.25 | 3.20 | 1.79 | 22.2 / 51.3 | 0 | PASS | 3.61 |
| mxF_s7_r2 / protocol_mx_recipe | 2.12 | 3.60 | 3.53 | 3.53 | 47.4 / 69.3 | 0 | PASS | 3.61 |
| mxF_s7_r2 / protocol_mx_recipe | 2.18 | 3.70 | 3.59 | 5.70 | 56.4 / 73.8 | 0 | fail | 3.63 |
| mxF_s8_r1 / protocol_mx_recipe | 1.92 | 3.25 | 3.23 | 2.10 | 21.9 / 52.2 | 0 | PASS | 3.63 |
| mxF_s8_r1 / protocol_mx_recipe | 2.12 | 3.60 | 3.52 | 3.17 | 37.3 / 63.8 | 0 | PASS | 3.57 |
| mxF_s8_r1 / protocol_mx_recipe | 2.18 | 3.70 | 3.63 | 4.36 | 46.3 / 66.8 | 0 | fail | 3.59 |
| mxF_s8_r1 / protocol_mx_recipe_text | 1.92 | 3.25 | 3.29 | 2.26 | 32.2 / 64.1 | 0 | PASS | 2.79 |
| mxF_s8_r1 / protocol_mx_recipe_text | 2.06 | 3.50 | 3.59 | 4.47 | 50.4 / 75.2 | 0 | fail | 2.79 |
| mxF_s8_r1 / protocol_or_recipe | 1.77 | 3.00 | 2.97 | 1.58 | 19.3 / 50.1 | 0 | PASS | 2.83 |
| mxF_s8_r1 / protocol_or_recipe | 1.95 | 3.30 | 3.25 | 3.40 | 36.8 / 58.8 | 0 | PASS | 2.78 |
| mxF_s8_r1 / protocol_or_recipe | 2.00 | 3.40 | 3.37 | 5.65 | 43.2 / 62.0 | 0 | fail | 2.81 |
| mxF_s8_r2 / protocol_or_recipe | 1.77 | 3.00 | 2.98 | 1.99 | 24.0 / 57.5 | 0 | PASS | 2.80 |
| mxF_s8_r2 / protocol_or_recipe | 1.95 | 3.30 | 3.26 | 6.02 | 44.8 / 60.7 | 0 | fail | 2.79 |
| mxF_s8_r2 / protocol_or_recipe | 2.00 | 3.40 | 3.36 | 9.12 | 45.0 / 63.2 | 0 | fail | 2.78 |
| mxF_s8_r2 / protocol_mx_recipe | 1.92 | 3.25 | 3.23 | 1.93 | 23.4 / 53.6 | 0 | PASS | 3.64 |
| mxF_s8_r2 / protocol_mx_recipe | 2.12 | 3.60 | 3.51 | 5.31 | 54.4 / 70.0 | 0 | fail | 3.56 |
| mxF_s8_r2 / protocol_mx_recipe | 2.18 | 3.70 | 3.60 | 7.39 | 56.4 / 72.8 | 0 | fail | 3.61 |
| mxG2_s7 / protocol_mx_ps | 2.06 | 3.50 | 3.46 | 3.22 | 26.9 / 56.6 | 0 | PASS | 3.59 |
| mxG2_s7 / protocol_mx_ps | 2.21 | 3.75 | 3.68 | 2.74 | 39.2 / 61.5 | 0 | PASS | 3.58 |
| mxG2_s7 / protocol_mx_recipe | 2.06 | 3.50 | 3.46 | 2.27 | 29.8 / 61.7 | 0 | PASS | 3.55 |
| mxG2_s7 / protocol_mx_recipe | 2.21 | 3.75 | 3.65 | 4.45 | 54.5 / 70.2 | 0 | fail | 3.60 |
| mxG2_s8 / protocol_mx_recipe | 2.06 | 3.50 | 3.45 | 3.06 | 37.5 / 65.3 | 0 | PASS | 3.56 |
| mxG2_s8 / protocol_mx_recipe | 2.21 | 3.75 | 3.68 | 7.06 | 55.2 / 70.5 | 0 | fail | 3.62 |
| mxG2_s8 / protocol_mx_ps | 2.06 | 3.50 | 3.45 | 2.13 | 26.3 / 52.8 | 1 | PASS | 3.58 |
| mxG2_s8 / protocol_mx_ps | 2.21 | 3.75 | 3.67 | 4.17 | 44.4 / 64.7 | 0 | fail | 3.61 |
| mxH4_s7 / protocol_mx_recipe | 2.19 | 3.25 | 3.20 | 3.13 | 22.7 / 55.8 | 0 | PASS | 3.55 |
| mxH4_s7 / protocol_mx_recipe | 2.36 | 3.50 | 3.45 | 2.29 | 29.9 / 59.6 | 0 | PASS | 3.59 |
| mxH4_s7 / protocol_mx_recipe | 2.52 | 3.75 | 3.60 | 4.74 | 54.9 / 71.4 | 0 | fail | 3.60 |
| mxH4_s7 / protocol_mx_recipe_text | 2.19 | 3.25 | 2.92 | 3.11 | 26.5 / 77.5 | 0 | PASS | 2.78 |
| mxH4_s8 / protocol_mx_recipe | 2.19 | 3.25 | 3.23 | 2.17 | 26.1 / 55.9 | 0 | PASS | 3.63 |
| mxH4_s8 / protocol_mx_recipe | 2.36 | 3.50 | 3.45 | 3.67 | 40.9 / 69.1 | 0 | PASS | 3.55 |
| mxH4_s8 / protocol_mx_recipe | 2.52 | 3.75 | 3.62 | 8.77 | 56.4 / 77.4 | 0 | fail | 3.57 |
| mxH4_s8 / protocol_mx_recipe_text | 2.19 | 3.25 | 3.13 | 2.47 | 38.0 / 66.6 | 0 | PASS | 2.79 |
| mxH4b_s7 / protocol_mx_recipe | 2.19 | 3.25 | 3.20 | 2.99 | 23.3 / 55.9 | 0 | PASS | 3.59 |
| mxH4b_s7 / protocol_mx_recipe | 2.36 | 3.50 | 3.44 | 2.34 | 31.0 / 60.1 | 0 | PASS | 3.61 |
| mxH4b_s7 / protocol_mx_recipe | 2.52 | 3.75 | 3.61 | 5.07 | 54.8 / 69.8 | 0 | fail | 3.60 |
| mxH4b_s7 / protocol_mx_recipe_text | 2.19 | 3.25 | 2.98 | 5.55 | 36.1 / 73.2 | 0 | fail | 2.79 |
| mxH4b_s7 / protocol_mx_recipe_text | 2.36 | 3.50 | 3.16 | 13.31 | 41.1 / 74.2 | 0 | fail | 2.79 |
| mxH4b_s8 / protocol_mx_recipe | 2.19 | 3.25 | 3.23 | 1.96 | 20.1 / 48.7 | 0 | PASS | 3.64 |
| mxH4b_s8 / protocol_mx_recipe | 2.36 | 3.50 | 3.46 | 2.17 | 27.7 / 53.0 | 0 | PASS | 3.61 |
| mxH4b_s8 / protocol_mx_recipe | 2.52 | 3.75 | 3.68 | 4.58 | 48.7 / 69.5 | 0 | fail | 3.57 |
| mxH4b_s8 / protocol_mx_recipe_text | 2.19 | 3.25 | 3.28 | 1.95 | 28.6 / 60.7 | 0 | PASS | 2.79 |
| mxH4b_s8 / protocol_mx_recipe_text | 2.36 | 3.50 | 3.55 | 3.52 | 43.4 / 72.5 | 0 | PASS | 2.79 |
| mxK_s7 / protocol_or_recipe | 1.92 | 3.25 | 3.19 | 3.57 | 41.8 / 62.4 | 0 | PASS | 2.81 |
| mxK_s7 / protocol_or_recipe | 2.06 | 3.50 | 3.33 | 11.51 | 46.2 / 62.7 | 0 | fail | 2.79 |
| mxK_s7 / protocol_or_recipe_text | 1.92 | 3.25 | 2.96 | 13.75 | 41.6 / 62.2 | 0 | fail | 2.38 |
| mxK_s7 / protocol_mx_recipe | 1.92 | 3.25 | 3.19 | 1.91 | 23.8 / 55.2 | 0 | PASS | 3.59 |
| mxK_s7 / protocol_mx_recipe | 2.06 | 3.50 | 3.43 | 3.16 | 42.0 / 67.3 | 1 | PASS | 3.59 |
| mxK_s7 / protocol_mx_recipe_text | 1.92 | 3.25 | 2.95 | 7.37 | 40.6 / 77.5 | 0 | fail | 2.79 |
| mxK_s8 / protocol_mx_recipe | 1.92 | 3.25 | 3.23 | 2.27 | 27.9 / 56.6 | 0 | PASS | 3.62 |
| mxK_s8 / protocol_mx_recipe | 2.06 | 3.50 | 3.43 | 4.19 | 43.5 / 68.4 | 1 | fail | 3.60 |
| mxK_s8 / protocol_mx_recipe_text | 1.92 | 3.25 | 3.29 | 3.24 | 43.2 / 73.6 | 0 | PASS | 2.78 |
| mxK_s8 / protocol_or_recipe | 1.92 | 3.25 | 3.23 | 5.23 | 44.1 / 59.6 | 0 | fail | 2.81 |
| mxK_s8 / protocol_or_recipe | 2.06 | 3.50 | 3.36 | 13.97 | 47.5 / 62.2 | 1 | fail | 2.81 |
| mxK_s8 / protocol_or_recipe_text | 1.92 | 3.25 | 3.24 | 9.48 | 47.0 / 61.0 | 0 | fail | 2.37 |
| mxL_lpt_s8 / protocol_mx_recipe | 1.92 | 3.25 | 3.21 | 2.46 | 26.5 / 58.0 | 0 | PASS | 3.57 |
| mxL_lpt_s8 / protocol_mx_recipe | 2.06 | 3.50 | 3.45 | 3.52 | 39.3 / 68.5 | 0 | PASS | 3.62 |
| mxL_lpt_s8 / protocol_mx_recipe | 2.21 | 3.75 | 3.67 | 8.73 | 56.5 / 77.0 | 1 | fail | 3.59 |
| mxL_lpt_s8 / protocol_mx_lpt6144 | 1.92 | 3.25 | 3.23 | 1.98 | 23.6 / 53.7 | 0 | PASS | 3.64 |
| mxL_lpt_s8 / protocol_mx_lpt6144 | 2.06 | 3.50 | 3.45 | 3.64 | 40.9 / 68.2 | 0 | PASS | 3.62 |
| mxL_lpt_s8 / protocol_mx_lpt6144 | 2.21 | 3.75 | 3.65 | 9.03 | 56.9 / 79.0 | 0 | fail | 3.56 |
| mxL_lpt_s8 / protocol_mx_lpt4096 | 1.92 | 3.25 | 3.22 | 2.11 | 24.5 / 54.9 | 0 | PASS | 3.61 |
| mxL_lpt_s8 / protocol_mx_lpt4096 | 2.06 | 3.50 | 3.45 | 3.52 | 40.0 / 67.5 | 0 | PASS | 3.58 |
| mxL_lpt_s8 / protocol_mx_lpt4096 | 2.21 | 3.75 | 3.67 | 8.90 | 56.3 / 73.8 | 0 | fail | 3.60 |
| mxR4_b10k / protocol_mx_stack | 2.52 | 3.75 | 3.68 | 4.23 | 46.5 / 69.7 | 1 | fail | 3.55 |
| mxR4_b10k / protocol_mx_stack | 2.69 | 4.00 | 3.86 | 10.10 | 54.9 / 74.6 | 0 | fail | 3.60 |
| mxR4_b10k / protocol_mx_stack_b10k | 2.52 | 3.75 | 3.68 | 4.33 | 50.8 / 78.1 | 0 | fail | 3.58 |
| mxR4_b10k / protocol_mx_stack_b10k | 2.69 | 4.00 | 3.83 | 10.83 | 60.5 / 84.2 | 0 | fail | 3.58 |
| mxR4_b12k / protocol_mx_stack_b12k | 2.52 | 3.75 | 3.68 | 3.36 | 44.6 / 80.2 | 0 | PASS | 3.56 |
| mxR4_b12k / protocol_mx_stack_b12k | 2.69 | 4.00 | 3.88 | 8.26 | 59.3 / 83.6 | 1 | fail | 3.60 |
| mxR4_b12k / protocol_mx_stack | 2.52 | 3.75 | 3.68 | 3.48 | 37.9 / 64.6 | 0 | PASS | 3.61 |
| mxR4_b12k / protocol_mx_stack | 2.69 | 4.00 | 3.89 | 9.32 | 54.4 / 70.8 | 0 | fail | 3.62 |
| mxR4_s112 / protocol_mx_stack_s112 | 2.52 | 3.75 | 3.66 | 5.38 | 54.1 / 70.6 | 0 | fail | 3.62 |
| mxR4_s112 / protocol_mx_stack_s112 | 2.69 | 4.00 | 3.81 | 12.49 | 56.2 / 76.9 | 0 | fail | 3.58 |
| mxR4_s112 / protocol_mx_stack | 2.52 | 3.75 | 3.68 | 5.04 | 51.1 / 68.8 | 0 | fail | 3.63 |
| mxR4_s112 / protocol_mx_stack | 2.69 | 4.00 | 3.77 | 12.21 | 55.7 / 74.1 | 0 | fail | 3.57 |
| mxR4_s80 / protocol_mx_stack | 2.52 | 3.75 | 3.68 | 4.09 | 43.2 / 65.7 | 0 | fail | 3.61 |
| mxR4_s80 / protocol_mx_stack | 2.69 | 4.00 | 3.88 | 9.61 | 54.4 / 71.4 | 0 | fail | 3.64 |
| mxR4_s80 / protocol_mx_stack_s80 | 2.52 | 3.75 | 3.69 | 4.26 | 42.3 / 64.2 | 0 | fail | 3.60 |
| mxR4_s80 / protocol_mx_stack_s80 | 2.69 | 4.00 | 3.84 | 11.09 | 49.8 / 68.1 | 0 | fail | 3.57 |
| mxS0_vllm_specoff / protocol_or_specoff | 1.47 | 2.50 | 2.44 | 1.31 | 35.8 / 54.4 | 1 | PASS | - |
| mxS0_vllm_specoff / protocol_or_specoff | 1.77 | 3.00 | 2.81 | 18.31 | 57.3 / 65.5 | 0 | fail | - |
| mxS0_vllm_specoff / protocol_or_specoff | 1.92 | 3.25 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mxS1_sglang / protocol_sgl_fp8 | 1.47 | 2.50 | 1.29 | 261.95 | 106.0 / 126.9 | 0 | fail | - |
| mxS1_sglang / protocol_sgl_fp8 | 1.77 | 3.00 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mxS1_sglang / protocol_sgl_fp8 | 1.92 | 3.25 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mxS3_vllm_fp8 / protocol_vllm_fp8 | 1.47 | 2.50 | 2.11 | 47.24 | 77.9 / 88.2 | 0 | fail | - |
| mxS3_vllm_fp8 / protocol_vllm_fp8 | 1.77 | 3.00 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| mxS3_vllm_fp8 / protocol_vllm_fp8 | 1.92 | 3.25 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| s1_b300_nvfp4_mtp2_r4 / protocol_nvfp4_mtp2_r4 | 1.00 | 2.89 | 1.70 | 69.50 | 154.0 / 214.4 | 0 | fail | - |
| s1_b300_nvfp4_mtp2_r4 / protocol_nvfp4_mtp2_r4 | 1.50 | 4.33 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| s1_b300_nvfp4_mtp2_r4 / protocol_nvfp4_mtp2_r4 | 2.00 | 5.77 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| s1_b300_nvfp4_mtp2_r4 / protocol_nvfp4_mtp2_r4_textclient | 1.00 | 2.89 | 1.32 | 31.33 | 163.3 / 181.5 | 0 | fail | 2.43 |
| s1_h100_fp8 / protocol_fp8dyn | 1.00 | 1.31 | 0.81 | 47.43 | 104.3 / 141.7 | 0 | fail | - |
| s1_h100_fp8 / protocol_fp8dyn | 1.50 | 1.96 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| s1_h100_fp8 / protocol_fp8dyn | 2.00 | 2.61 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| s1_h100_fp8 / protocol_fp8dyn_textclient | 1.00 | 1.31 | 0.83 | 27.45 | 94.9 / 108.4 | 0 | fail | - |
| s1_h200_fp8 / protocol_fp8dyn | 1.00 | 1.35 | 0.94 | 30.15 | 201.1 / 485.1 | 0 | fail | - |
| s1_h200_fp8 / protocol_fp8dyn | 1.50 | 2.03 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| s1_h200_fp8 / protocol_fp8dyn | 2.00 | 2.71 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| s1_h200_fp8 / protocol_fp8dyn_textclient | 1.00 | 1.35 | 0.31 | 14.43 | 287.8 / 342.9 | 0 | fail | - |
| s1_l40s_w4a16_fi / protocol_w4a16qat_fi | 1.00 | 0.74 | 0.19 | 301.12 | 179.9 / 265.4 | 0 | fail | - |
| s1_l40s_w4a16_fi / protocol_w4a16qat_fi | 1.50 | 1.11 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| s1_l40s_w4a16_fi / protocol_w4a16qat_fi | 2.00 | 1.48 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| s1_l40s_w4a16_fi / protocol_w4a16qat_fi_textclient | 1.00 | 0.74 | 0.16 | 74.08 | 138.4 / 179.6 | 0 | fail | - |
| s1_rtxpro6000_nvfp4_fi / protocol_nvfp4_fi | 1.00 | 1.03 | 0.79 | 10.00 | 171.2 / 295.6 | 0 | fail | - |
| s1_rtxpro6000_nvfp4_fi / protocol_nvfp4_fi | 1.50 | 1.54 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| s1_rtxpro6000_nvfp4_fi / protocol_nvfp4_fi | 2.00 | 2.05 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| s1_rtxpro6000_nvfp4_fi / protocol_nvfp4_fi_textclient | 1.00 | 1.03 | 0.53 | 4.84 | 153.4 / 271.0 | 0 | fail | - |
| s2_mi355x_or_recipe_seed7 / protocol_or_recipe | 1.47 | 2.50 | 2.47 | 1.40 | 13.1 / 30.0 | 0 | PASS | - |
| s2_mi355x_or_recipe_seed7 / protocol_or_recipe | 1.77 | 3.00 | 2.96 | 1.74 | 21.5 / 48.5 | 0 | PASS | - |
| s2_mi355x_or_recipe_seed7 / protocol_or_recipe | 1.92 | 3.25 | 3.17 | 3.28 | 35.7 / 58.6 | 0 | PASS | - |
| s2_mi355x_or_recipe_seed7 / protocol_or_recipe | 2.06 | 3.50 | 3.34 | 10.71 | 45.8 / 62.4 | 0 | fail | - |
| s2_mi355x_or_recipe_seed7 / protocol_or_recipe_textclient | 1.77 | 3.00 | 2.76 | 1.78 | 15.8 / 38.1 | 0 | PASS | 2.37 |
| s2_mi355x_or_recipe_seed8 / protocol_or_recipe | 1.47 | 2.50 | 2.48 | 1.45 | 12.5 / 25.5 | 0 | PASS | - |
| s2_mi355x_or_recipe_seed8 / protocol_or_recipe | 1.77 | 3.00 | 2.98 | 1.59 | 19.9 / 51.7 | 0 | PASS | - |
| s2_mi355x_or_recipe_seed8 / protocol_or_recipe | 1.92 | 3.25 | 3.23 | 2.55 | 33.5 / 62.1 | 0 | PASS | - |
| s2_mi355x_or_recipe_seed8 / protocol_or_recipe | 2.06 | 3.50 | 3.45 | 9.23 | 44.3 / 61.3 | 0 | fail | - |
| s2a_mi355x_or_recipe_seed8 / protocol_or_recipe_A_random | 1.77 | 3.00 | 2.98 | 1.80 | 20.7 / 48.3 | 0 | PASS | 2.81 |
| s2a_mi355x_or_recipe_seed8 / protocol_or_recipe_A_random | 1.92 | 3.25 | 3.23 | 2.32 | 31.2 / 61.3 | 0 | PASS | 2.81 |
| s2a_mi355x_or_recipe_seed8 / protocol_or_recipe_A_random | 2.06 | 3.50 | 3.44 | 8.85 | 44.2 / 59.1 | 0 | fail | 2.81 |
| s2b_mi355x_or_recipe_seed7 / protocol_or_recipe_B_text_strat | 1.47 | 2.50 | 2.38 | 1.50 | 13.6 / 37.5 | 0 | PASS | 2.38 |
| s2b_mi355x_or_recipe_seed7 / protocol_or_recipe_B_text_strat | 1.77 | 3.00 | 2.79 | 2.33 | 28.1 / 82.1 | 0 | PASS | 2.37 |
| s2b_mi355x_or_recipe_seed7 / protocol_or_recipe_B_text_strat | 1.92 | 3.25 | 3.02 | 4.70 | 35.5 / 66.0 | 0 | fail | 2.37 |
| s2b_mi355x_or_recipe_seed7 / protocol_or_recipe_B_text_strat | 2.06 | 3.50 | 3.15 | 13.48 | 48.1 / 62.7 | 0 | fail | 2.38 |
| s2b_mi355x_or_recipe_seed8 / protocol_or_recipe_B_text_strat | 1.47 | 2.50 | 2.48 | 1.11 | 14.2 / 28.3 | 0 | PASS | 2.38 |
| s2b_mi355x_or_recipe_seed8 / protocol_or_recipe_B_text_strat | 1.77 | 3.00 | 2.99 | 1.83 | 21.2 / 43.7 | 0 | PASS | 2.38 |
| s2b_mi355x_or_recipe_seed8 / protocol_or_recipe_B_text_strat | 1.92 | 3.25 | 3.07 | 4.62 | 44.9 / 58.7 | 0 | fail | 2.38 |
| s2b_mi355x_or_recipe_seed8 / protocol_or_recipe_B_text_strat | 2.06 | 3.50 | 3.40 | 10.23 | 46.4 / 57.1 | 0 | fail | 2.37 |
| s2c_mi355x_or_recipe_seed7 / protocol_or_recipe_A_random | 1.77 | 3.00 | 2.94 | 1.74 | 19.6 / 43.7 | 0 | PASS | 2.81 |
| s2c_mi355x_or_recipe_seed7 / protocol_or_recipe_A_random | 1.92 | 3.25 | 3.20 | 2.41 | 29.1 / 54.5 | 0 | PASS | 2.76 |
| s2c_mi355x_or_recipe_seed7 / protocol_or_recipe_A_random | 2.06 | 3.50 | 3.38 | 6.63 | 44.0 / 58.3 | 0 | fail | 2.80 |
| s2c_mi355x_or_recipe_seed7 / protocol_or_recipe_B_text | 1.47 | 2.50 | 2.45 | 1.49 | 12.3 / 34.4 | 0 | PASS | 2.38 |
| s2c_mi355x_or_recipe_seed7 / protocol_or_recipe_B_text | 1.77 | 3.00 | 2.80 | 1.88 | 19.3 / 75.8 | 0 | PASS | 2.38 |
| s2c_mi355x_or_recipe_seed7 / protocol_or_recipe_B_text | 1.92 | 3.25 | 3.00 | 3.20 | 25.9 / 60.1 | 0 | PASS | 2.38 |
| s2c_mi355x_or_recipe_seed8 / protocol_or_recipe_B_text | 1.47 | 2.50 | 2.53 | 1.19 | 14.2 / 25.7 | 0 | PASS | 2.37 |
| s2c_mi355x_or_recipe_seed8 / protocol_or_recipe_B_text | 1.77 | 3.00 | 3.01 | 1.59 | 22.0 / 39.7 | 0 | PASS | 2.37 |
| s2c_mi355x_or_recipe_seed8 / protocol_or_recipe_B_text | 1.92 | 3.25 | 3.28 | 2.89 | 33.0 / 60.1 | 0 | PASS | 2.37 |
| s3_rtxpro6000_small_mtp / protocol_p6_small_base | 0.81 | 0.70 | 0.62 | 3.10 | 50.5 / 97.2 | 0 | PASS | - |
| s3_rtxpro6000_small_mtp / protocol_p6_small_base | 1.00 | 0.86 | 0.74 | 4.25 | 116.7 / 190.1 | 0 | fail | - |
| s3_rtxpro6000_small_mtp / protocol_p6_small_base | 1.19 | 1.03 | 0.79 | 10.37 | 171.5 / 293.6 | 0 | fail | - |
| s3_rtxpro6000_small_mtp / protocol_p6_small_mtp2 | 1.00 | 0.86 | 0.77 | 5.64 | 41.0 / 128.3 | 0 | fail | - |
| s3_rtxpro6000_small_mtp / protocol_p6_small_mtp2 | 1.19 | 1.03 | 0.91 | 8.18 | 108.3 / 222.7 | 0 | fail | - |
| s3_rtxpro6000_small_mtp / protocol_p6_small_mtp2 | 1.39 | 1.20 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| s3_rtxpro6000_small_mtp / protocol_p6_small_mtp2_textclient | 1.00 | 0.86 | 0.88 | 4.92 | 40.6 / 75.1 | 0 | fail | 2.43 |
| s3_rtxpro6000_small_mtp / protocol_p6_small_mtp3 | 1.00 | 0.86 | 0.79 | 5.63 | 35.1 / 126.3 | 0 | fail | - |
| s3_rtxpro6000_small_mtp / protocol_p6_small_mtp3 | 1.19 | 1.03 | 0.93 | 8.20 | 99.0 / 204.2 | 0 | fail | - |
| s3_rtxpro6000_small_mtp / protocol_p6_small_mtp3 | 1.39 | 1.20 | - | - | 0 / 0 | - | skipped: past SLO (an earlier rate failed with p95 TTFT > 2x SLO) | - |
| s3_rtxpro6000_small_mtp / protocol_p6_small_mtp3_textclient | 1.00 | 0.86 | 0.95 | 5.18 | 31.8 / 68.3 | 0 | fail | 2.86 |
