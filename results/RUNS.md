| run | stamp | sandbox | $/h | minutes | est $ | servers (download s / ready s / end) | error |
|---|---|---|---|---|---|---|---|
| cc_c1_s7 | 20261122T094451Z | 3be8d688-b58b-4a6f-afbc-d40d037e6b4a | 4.76 | 80.4 | 6.38 | mx_recipe: 22 / 160 / ok; or_recipe: 12 / 85 / ok |  |
| cc_c1_s8 | 20261122T094453Z | ca2089b1-d0e4-4b21-86a2-eaa1158cff81 | 4.76 | 80.6 | 6.39 | or_recipe: 22 / 168 / ok; mx_recipe: 12 / 85 / ok |  |
| cc_c2_s7 | 20261122T094455Z | cee7f83d-3b24-4fda-b55b-c09cafe88b71 | 4.76 | 55.3 | 4.38 | mx_recipe: 22 / 160 / ok; or_recipe: 12 / 76 / ok |  |
| cc_c2_s8 | 20261122T094457Z | 92aa3762-d5d7-4e58-a322-88e41f3ce07e | 4.76 | 55.4 | 4.39 | or_recipe: 22 / 160 / ok; mx_recipe: 12 / 76 / ok |  |
| cc_x2_s7 | 20261122T105237Z | 5ec075d5-0966-451c-a1d6-32cfd4fc177c | 4.76 | 61.9 | 4.91 | mx_recipe: 22 / 160 / ok; or_recipe: 12 / 77 / ok |  |
| cc_x2_s8 | 20261122T105239Z | 165dc46c-5ec4-4f2a-a289-97b817a14212 | 4.76 | 61.8 | 4.90 | or_recipe: 22 / 160 / ok; mx_recipe: 12 / 76 / ok |  |
| cc_x_s7 | 20261122T102748Z | 70f1bf3a-4f2b-4c8e-b0f1-05c6bc34e231 | 4.76 | 30.8 | 2.44 | mx_recipe: 22 / 160 / ok; or_recipe: 12 / 77 / ok |  |
| cc_x_s8 | 20261122T102750Z | 0634be68-c23c-4fce-a62b-51ac1b51a22a | 4.76 | 31.3 | 2.48 | or_recipe: 22 / 177 / ok; mx_recipe: 12 / 85 / ok |  |
| d1_h100_prefill_diag | 20261122T045257Z | e5df49f7-7be4-4552-9620-d12f687d0ef0 | 3.67 | 15.1 | 0.92 | fp8dyn_mbt8192: 23 / 141 / ok; fp8dyn_mbt16384: 3 / 132 / ok; fp8dyn_mbt32768: 3 / 132 / - | ApiError('HTTP 404 on https://proxy.app.daytona.io/toolbox/e5df49f7-7be4-4552-9620-d12f687d0ef0/process/execute: {"statu EVICTED |
| d1_h100_prefill_diag | 20261122T050803Z | 7bbe1b8b-f705-46d3-a87e-9fdca8b9770b | 3.67 | 16.5 | 1.01 | fp8dyn_mbt8192: 24 / 148 / ok; fp8dyn_mbt16384: 3 / 134 / ok; fp8dyn_mbt32768: 3 / 132 / ok |  |
| d1_rtxpro6000_prefill_diag | 20261122T045255Z | a0b18f21-bfd1-4326-b65c-efa5fb97a673 | 2.88 | 14.4 | 0.69 | nvfp4_fi_mbt8192: 72 / 165 / ok; nvfp4_fi_mbt16384: 11 / 108 / ok; nvfp4_fi_mbt32768: 11 / 116 / server died |  |
| d2_b300_prefill_diag | 20261122T051320Z | 1f17d1f7-b8ed-4726-90ad-71bfaf747a43 | 8.09 | 7.2 | 0.97 | nvfp4_nospec: 73 / 198 / ok |  |
| d2_h100_prefill_sweep | 20261122T051203Z | af7348c2-b6d1-490a-bc6c-3caea6a7989e | 3.67 | 4.8 | 0.30 | fp8dyn: 62 / 166 / ok |  |
| d2_rtxpro6000_prefill_sweep | 20261122T051201Z | fa6261c9-0f33-4e39-9007-af669c1e81cc | 2.42 | 7.0 | 0.28 | nvfp4_fi: 92 / 230 / ok |  |
| d3_sglang_counter_probe | 20261122T072959Z | 7e1d411d-99a3-45c0-8e73-cf91cc5e1b23 | 2.89 | 11.8 | 0.57 | sgl_nvfp4_nv: 53 / 97 / ok |  |
| e4_b300_sglang | 20261122T053902Z | None | 6.64 | 0.0 | 0.00 |  |  |
| e4_b300_sglang | 20261122T053936Z | None | 6.64 | 0.0 | 0.00 |  |  |
| e4_b300_sglang | 20261122T054136Z | None | 6.64 | 0.0 | 0.00 |  | RuntimeError('no spot B300 x1 free right now') |
| e4_b300_sglang | 20261122T055226Z | 60b37c75-d0c9-47a8-b294-d2750bc2d912 | 6.64 | 39.1 | 4.33 | sgl_nvfp4_nv: 73 / 425 / ok |  |
| e4_rtxpro6000_sglang | 20261122T052658Z | e25b2048-be8e-4624-a078-0202d92abfcb | 2.89 | 12.9 | 0.62 | sgl_nvfp4_rh: 42 / 77 / NOT READY; sgl_nvfp4_nv: 62 / 71 / ok |  |
| e4a_rtxpro6000_sglang_colA | 20261122T054545Z | 9bbdacbf-332f-4480-91e5-d71f04924327 | 2.89 | 24.1 | 1.16 | sgl_nvfp4_nv: 109 / 103 / ok |  |
| e5_b300_sglang_textonly | 20261122T090736Z | c504e37e-0ea3-438a-918d-b5414a3686e9 | 6.64 | 44.8 | 4.96 | sgl_rh_nvfp4_textonly: 32 / 449 / ok; sgl_rh_fp8_textonly: - / - / -; sgl_nv_nvfp4_textonly: - / - / - |  |
| e5_b300_sglang_textonly_s8 | 20261122T115147Z | b616c508-9dcf-4c93-8b09-d148dd095882 | 6.64 | 44.0 | 4.88 | sgl_rh_nvfp4_textonly: 32 / 449 / ok |  |
| glm1_sglang0523 | 20261122T073208Z | None | 4.77 | 0.0 | 0.00 |  | RuntimeError('snapshot or-sglang0523rocm720-mi355x-c8m48d300 failed: Snapshot size (22.20GB) exceeds maximum allowed siz |
| glm1_sglang0523 | 20261122T073308Z | None | 4.77 | 0.0 | 0.00 |  | RuntimeError('snapshot or-sglang0523rocm10-mi355x-c8m48d300 failed: An operation is already in progress for this resourc |
| glm1_sglang0523 | 20261122T074555Z | f6dc5c96-942d-4c5d-82a4-b7ee7dc2983f | 4.77 | 30.7 | 2.44 | glm_sgl: 103 / 414 / ok |  |
| glm1_vllm034 | 20261122T073206Z | f56c5774-8299-4374-93e9-ac7d5f7c00b2 | 4.77 | 28.4 | 2.26 | glm_vllm: 113 / 354 / ok |  |
| glm2_vllm_b16k | 20261122T074428Z | d1979a7a-127a-46c7-93e3-26f99f188c8b | 4.77 | 8.9 | 0.71 | glm_b16k: 113 / 337 / gate failed |  |
| glm2_vllm_b16k | 20261122T075350Z | 2a9e830b-84b6-4635-8fb0-5a77ac8a02fb | 4.77 | 27.8 | 2.21 | glm_b16k: 103 / 343 / ok |  |
| glm2_vllm_mtp3 | 20261122T074424Z | c0caece1-1342-412a-8de6-7125541be8b0 | 4.77 | 33.8 | 2.69 | glm_mtp3: 103 / 391 / ok |  |
| glm2_vllm_mtp5 | 20261122T074426Z | 5e89b592-2477-4230-a319-ff2d84649b5b | 4.77 | 34.8 | 2.77 | glm_mtp5: 113 / 399 / ok |  |
| glm3_vllm_profile | 20261122T075541Z | f796c3d2-683e-4d0e-8830-8c81d0202d77 | 4.77 | 49.1 | 3.91 | glm_prof: 113 / 305 / budget |  |
| glm4_best_s7 | 20261122T082933Z | d7b1137e-e9da-4004-b0d3-9c55f9e0ed99 | 4.77 | 40.8 | 3.25 | glm_best: 113 / 357 / ok |  |
| glm4_best_s8 | 20261122T082931Z | fb6b142c-09eb-4529-b622-2f6c611a9e4f | 4.77 | 41.2 | 3.28 | glm_best: 114 / 351 / ok |  |
| ik1_5090_sgl | 20261122T071045Z | 29298b79-2ab0-4222-a012-247f7537ee92 | 1.68 | 22.3 | 0.62 | sgl: 67 / 282 / ok |  |
| ik1_h100_sgl | 20261122T071049Z | e1addf02-65f9-4e0a-adb2-60043a4de06f | 3.41 | 22.3 | 1.26 | sgl: 11 / 603 / ok |  |
| ik1_h100_vllm | 20261122T071053Z | None | 3.41 | 0.0 | 0.00 |  |  |
| ik1_mi355x_sgl | 20261122T071051Z | 38a12a44-8af9-495d-811a-2bdffe65153b | 4.50 | 21.1 | 1.58 | sgl: 12 / 236 / ok |  |
| ik1_pro6000_sgl | 20261122T071047Z | b1d4339e-4bfb-426b-be00-833d8cbf1d64 | 2.89 | 21.8 | 1.05 | sgl: 77 / 198 / ok |  |
| ik1_pro6000_vllm | 20261122T071055Z | 1b15bbef-734d-418e-a926-d759f04b2684 | 2.89 | 17.0 | 0.82 | vllm: 88 / 198 / ok |  |
| ik2_pro6000_sched | 20261122T074943Z | 9c3d29c8-e79a-407c-ac0a-fad9b6db58b6 | 2.89 | 6.8 | 0.33 | swa15: 41 / 207 / ok; swa15_mixed: 11 / - / - |  |
| ik2_pro6000_sched | 20261122T075734Z | 38a287e1-57fc-476c-a5e5-5f1e193be306 | 2.89 | 39.6 | 1.91 | swa15: 81 / 211 / ok; swa15_mixed: 4 / 68 / ok; swa15_chunk16k: 4 / 94 / ok |  |
| ik2_pro6000_spec | 20261122T074945Z | bb23d068-ecd8-494e-ab3c-0e7113a62c6b | 2.89 | 6.7 | 0.32 | swa15: 43 / 196 / ok; swa15_dspark: 2 / - / - |  |
| ik2_pro6000_spec | 20261122T075736Z | 9220575a-8df4-430d-a9f4-f28a8f54f103 | 2.89 | 18.7 | 0.90 | swa15: 90 / 208 / ok; swa15_dspark: 4 / 66 / rejected: missing cuda_graph_sizes |  |
| ik2_pro6000_swa | 20261122T074941Z | cd17b772-5ec2-42fd-b6f6-8380f0fdf9d6 | 2.89 | 6.9 | 0.33 | base: 76 / 198 / - |  |
| ik2_pro6000_swa | 20261122T075732Z | 1d29edfb-ffd8-4243-817a-93fba74eedad | 2.89 | 30.6 | 1.47 | base: 76 / 190 / ok; swa15: 4 / 66 / ok |  |
| ik3_5090_bf16 | 20261122T075120Z | 97b0a5d8-7142-411b-9799-967c3f0e9ee2 | 1.68 | 5.1 | 0.14 | swa25: 66 / - / - |  |
| ik3_5090_bf16 | 20261122T075738Z | f34c59d6-7ae6-41da-9c14-10eeca9caeef | 1.68 | 28.3 | 0.79 | swa25: 134 / 282 / ok; swa25_nopg: 4 / 68 / ok |  |
| ik3_5090_fp8 | 20261122T075122Z | f0337b0c-9e76-4b3a-9c54-ab9575073ab7 | 1.68 | 5.2 | 0.14 | fp8kv: 67 / 49 / NOT READY |  |
| ik3_5090_fp8 | 20261122T075740Z | 22065265-5026-4e98-aef2-cc38462db578 | 1.68 | 5.1 | 0.14 | tri_nopg: 122 / 58 / NOT READY; tri_nopg_fp8kv: 4 / - / - |  |
| ik4_mi355x_aiter | 20261122T075430Z | e30679bf-7a3b-44e8-9cfb-e965b4bcea3f | 4.50 | 2.0 | 0.15 | base: 12 / - / - |  |
| ik4_mi355x_aiter | 20261122T075742Z | 906a93ad-1c67-405a-9c8a-e56bf3504829 | 4.50 | 41.0 | 3.07 | base: 12 / 244 / - |  |
| ik5_pro6000_dspark | 20261122T081506Z | a6c74f63-b933-49d0-ae92-d8e2802ac9e6 | 2.89 | 25.0 | 1.20 | base: 53 / 213 / ok; dspark: 2 / 137 / ok |  |
| ik6_5090_mem | 20261122T081801Z | db24aed6-bc5b-4753-b2cf-53b98d497639 | 1.68 | 3.0 | 0.09 | mem88: 67 / 13 / NOT READY |  |
| ik6_5090_mem | 20261122T082104Z | 78cf5fb1-e238-4d19-b2e0-7fe3aadc41b2 | 1.68 | 5.9 | 0.17 | mem88: 57 / 178 / NOT READY |  |
| ik7_mi355x_aiter | 20261122T084023Z | f401d27d-d601-42d0-bdfb-9d568e8f04b7 | 4.50 | 15.5 | 1.16 | base: 12 / 236 / ok; tuned: 12 / - / - |  |
| ik8_5090_c4 | 20261122T084151Z | ec99a57f-d3db-419c-88e8-948eecf1a4f1 | 1.35 | 9.5 | 0.21 | mem84: 69 / 219 / server died; mem84_mixed: 4 / - / - |  |
| ik8_5090_c4 | 20261122T085129Z | 4fdcbaaf-04e2-4f9f-995a-b52df6ebbd15 | 1.35 | 34.1 | 0.77 | mem81: 58 / 220 / ok; mem81_mixed: 9 / 67 / ok |  |
| ik8_pro6000_c4 | 20261122T084149Z | 82497495-7307-4a18-838a-afff199ee8db | 2.56 | 43.1 | 1.83 | mixed: 81 / 204 / ok; mixed_dspark: 7 / 145 / ok |  |
| ik9_mi355x_aiter | 20261122T085622Z | b962e682-c44c-4cec-aaf5-a6cc73c6b4a7 | 4.50 | 35.7 | 2.68 | tune: 12 / 1 / ok; base: 12 / 236 / - |  |
| ikC_b300_mixed_dspark | 20261122T100257Z | 651353be-bf72-4069-8bfa-0eafb8d2ee72 | 6.11 | 103.8 | 10.57 | mixed_dspark: 32 / 464 / ok |  |
| ikC_pro6000_mixed_dspark_rep_s7 | 20261122T103726Z | 78194433-e08f-420f-92e1-3273d960647e | 2.56 | 18.5 | 0.79 | mixed_dspark: 42 / 228 / ok |  |
| ikC_pro6000_mixed_dspark_rep_s8 | 20261122T103728Z | f89f03b6-28a6-4e93-a568-999d18e37400 | 2.56 | 20.2 | 0.86 | mixed_dspark: 79 / 238 / ok |  |
| ikC_pro6000_mixed_dspark_s7 | 20261122T092529Z | ab32d9fa-3e6f-4e5b-862f-f0c159e7e969 | 2.56 | 60.1 | 2.56 | mixed_dspark: 63 / 238 / ok |  |
| ikC_pro6000_mixed_dspark_s8 | 20261122T092531Z | b714f008-2422-4e35-8ead-846271d4b2de | 2.56 | 61.8 | 2.63 | mixed_dspark: 78 / 240 / ok |  |
| ikC_pro6000_mixed_s7 | 20261122T091132Z | 5743fd42-7853-462b-b927-54c5845c6450 | 2.56 | 77.7 | 3.31 | ref: 48 / 209 / ok; mixed: 11 / 68 / ok |  |
| ikC_pro6000_mixed_s8 | 20261122T091134Z | 6c0ffb5e-6983-4d95-85f9-eda70298d799 | 2.56 | 53.9 | 2.30 | mixed: 74 / 201 / ok |  |
| mx10_best_dc_s7 | 20261122T060316Z | 289a1a9b-1a33-47df-a42e-f2798e376b6b | 4.76 | 21.7 | 1.72 | best_dc: 42 / 151 / ok |  |
| mx10_best_dc_s8 | 20261122T060314Z | ac4cd753-a64a-44d2-9d5c-d16c76fd5b03 | 4.76 | 22.7 | 1.80 | best_dc: 32 / 168 / ok |  |
| mx11_or_dc | 20261122T062018Z | 583e3d8a-266b-4f9c-ad93-130e08eea4b4 | 4.76 | 22.7 | 1.80 | or_dc: 22 / 168 / ok |  |
| mx12_k4_dc | 20261122T062613Z | b3e3f700-0dfc-408c-aa9f-0e88745fc5a2 | 4.76 | 32.9 | 2.61 | k4_dc: 22 / 168 / ok |  |
| mx1_or_b4k | 20261122T050156Z | e4410132-ecd1-4d2e-a724-6553fc95dccb | 4.76 | 17.5 | 1.39 | or_b4096: 52 / 176 / ok |  |
| mx1_or_b6k | 20261122T050158Z | 93067345-2d8f-4251-817c-c9aef9186116 | 4.76 | 17.3 | 1.37 | or_b6144: 42 / 177 / ok |  |
| mx1_or_mtp3 | 20261122T050200Z | 75cf76a8-a6e7-4ec4-8133-4d065c01b6ed | 4.76 | 33.3 | 2.64 | or_mtp3: 32 / 176 / ok |  |
| mx1_or_profile | 20261122T050154Z | 2c40c6a5-a0e6-4b0f-9353-4590d5d06945 | 4.76 | 10.1 | 0.80 | or_prof: 32 / 177 / ok |  |
| mx1_v034 | 20261122T050152Z | 6e17a4e9-1e01-4234-b1f3-35a03120c823 | 4.76 | 14.1 | 1.11 | v034_stock: 22 / 193 / ok; v034_asm: - / - / - |  |
| mx2_v034_asm | 20261122T050728Z | 79694502-b580-49a5-9bba-394324be30e9 | 4.76 | 16.8 | 1.33 | v034_asm: 32 / 160 / ok |  |
| mx2_v034_orport | 20261122T050730Z | 0bde8596-383a-4d58-9d1a-a911f7803dca | 4.76 | 17.6 | 1.39 | v034_orport: 52 / 185 / ok |  |
| mx3_or_seqs128 | 20261122T051339Z | 3f275fb7-f2eb-4465-aece-cee4c9eeb61b | 4.76 | 17.3 | 1.37 | or_seqs128: 42 / 170 / ok |  |
| mx3_or_seqs96 | 20261122T051337Z | 3e7539a7-7749-4f7d-9093-e728cda3231b | 4.76 | 22.9 | 1.82 | or_seqs96: 32 / 170 / ok |  |
| mx4_or_b12k | 20261122T051936Z | f86417d5-43f6-47e1-8506-cc1c47918c5c | 4.76 | 17.1 | 1.35 | or_b12k: 52 / 168 / ok |  |
| mx5_or_mtp3_seqs96 | 20261122T052526Z | 9efe3bcf-a3dc-489e-b32b-d4d16d470c44 | 4.76 | 23.0 | 1.82 | or_mtp3_seqs96: 52 / 160 / ok |  |
| mx5_or_mtp4 | 20261122T052524Z | 3c3f5ffc-820c-4aa2-ae75-f433d2358ccc | 4.76 | 33.6 | 2.66 | or_mtp4: 42 / 168 / ok |  |
| mx6_or_mtp3_seqs112 | 20261122T054308Z | b085a3b4-de15-4575-8821-2aec641bc4a1 | 4.76 | 23.4 | 1.85 | or_mtp3_seqs112: 52 / 168 / ok |  |
| mx6_or_mtp3_seqs80 | 20261122T054306Z | 06925433-68f5-45f1-bebb-d2a3acb36e66 | 4.76 | 6.3 | 0.50 | or_mtp3_seqs80: 42 / 168 / - | ApiError('HTTP 404 on https://proxy.app.daytona.io/toolbox/06925433-68f5-45f1-bebb-d2a3acb36e66/process/execute: {"statu EVICTED |
| mx6_or_mtp3_seqs80 | 20261122T054926Z | 10cacfa6-3076-405d-9a01-2db19ec05d81 | 4.76 | 22.7 | 1.80 | or_mtp3_seqs80: 32 / 168 / ok |  |
| mx6_or_mtp3_seqs96_s7 | 20261122T054311Z | afc651bc-e927-45a8-b14d-7cbee9f0c382 | 4.76 | 23.3 | 1.84 | or_mtp3_seqs96: 52 / 168 / ok |  |
| mx6_or_mtp4_seqs96 | 20261122T054304Z | 70ac2c28-a459-4a3c-b0f0-37468889d774 | 4.76 | 23.3 | 1.85 | or_mtp4_seqs96: 52 / 168 / ok |  |
| mx7_prof_mtp3_seqs96 | 20261122T054335Z | feba19ec-64d6-4b60-9a6d-d06a5827f57e | 4.76 | 10.3 | 0.81 | prof_mtp3_seqs96: 52 / 152 / ok |  |
| mx8_or_mtp3_seqs96_block | 20261122T054943Z | abca1134-a78f-462c-b672-3940a569dd8e | 4.76 | 32.6 | 2.58 | or_mtp3_seqs96_block: 22 / 160 / ok |  |
| mx8_or_mtp3_seqs96_prob | 20261122T054945Z | 875eadfa-246a-486c-9299-d0f474caf8b5 | 4.76 | 32.9 | 2.61 | or_mtp3_seqs96_prob: 32 / 160 / ok |  |
| mx9_best_r2s7 | 20261122T060050Z | 43c7c4a4-7f77-4d5e-af1d-0990381f04da | 4.76 | 18.0 | 1.43 | best: 134 / 168 / ok |  |
| mx9_best_r2s8 | 20261122T060046Z | 95c0945a-2dae-4393-8578-7967857bd753 | 4.76 | 18.0 | 1.43 | best: 154 / 160 / ok |  |
| mx9_best_r3s8 | 20261122T060048Z | 175fe31e-e207-4a7d-8c58-72e3d441d866 | 4.76 | 17.7 | 1.41 | best: 134 / 160 / ok |  |
| mxA2_dec40 | 20261122T092628Z | 3d28ed54-44c1-41a2-88ff-c0816dad28a6 | 4.76 | 1.6 | 0.13 | nogpu: 1 / 1 / ok |  |
| mxA2_dec96 | 20261122T092630Z | 2d7a9e21-20fe-475e-b4a5-9662a66017ce | 4.76 | 1.5 | 0.12 | nogpu: 2 / 1 / ok |  |
| mxA2_mix | 20261122T092626Z | 0f0edafc-341d-469c-ad02-9de2bc9dee68 | 4.76 | 3.5 | 0.28 | nogpu: 1 / 1 / ok |  |
| mxA3_s2_safety | 20261122T093130Z | 5a0f96f2-47e0-4e19-859e-978fbe1ec22f | 4.76 | 7.5 | 0.59 | nogpu: 1 / 1 / ok |  |
| mxA_attn_lab | 20261122T091936Z | bd9bf886-78be-4dea-a134-9cd9a1f42b63 | 4.76 | 2.9 | 0.23 | nogpu: 1 / 1 / ok |  |
| mxBF1_fp8 | 20261122T105550Z | fb9e60eb-d3b5-446b-84ba-96465d4d43b5 | 4.76 | 1.7 | 0.14 | nogpu: 1 / 1 / ok |  |
| mxBF1b_drafter | 20261122T105819Z | 299675d1-46c9-4a37-be0c-fbeb4ebd4be9 | 4.76 | 3.2 | 0.25 | nogpu: 1 / 1 / ok |  |
| mxC1_s7_r1 | 20261122T061956Z | 152241f2-2bd1-4752-9306-e3ad897343b2 | 4.76 | 57.7 | 4.57 | mx_recipe: 22 / 168 / ok; or_recipe: 12 / 85 / ok |  |
| mxC1_s7_r2 | 20261122T061958Z | 3b8d2933-e23f-423c-b15f-7059c6b66752 | 4.76 | 57.8 | 4.58 | or_recipe: 22 / 160 / ok; mx_recipe: 12 / 85 / ok |  |
| mxC1_s8_r1 | 20261122T062000Z | 5232133e-4964-4421-9c93-64acb1f02e30 | 4.76 | 37.2 | 2.95 | or_recipe: 22 / 168 / ok; mx_recipe: 12 / 85 / ok |  |
| mxC1_s8_r2 | 20261122T062002Z | bf96f277-3ad8-4a89-89d8-ec2b835e00ea | 4.76 | 37.4 | 2.96 | mx_recipe: 22 / 168 / ok; or_recipe: 12 / 85 / ok |  |
| mxC1_text_s8 | 20261122T062710Z | 822173ae-ddbf-4d88-9e78-5077b63b1e68 | 4.76 | 26.7 | 2.12 | mx_recipe: 22 / 160 / ok; or_recipe: 12 / 93 / ok |  |
| mxC2_s7_r1 | 20261122T100636Z | 72ca63b8-3987-431d-8864-c6d71398b3fd | 4.17 | 90.7 | 6.30 | or_recipe: 22 / 178 / ok; mx_recipe: 12 / 102 / ok; mx_stack: 12 / 77 / ok |  |
| mxC2_s7_r2 | 20261122T100638Z | 5b335ec4-ca82-4994-8c37-94c4c2ca7ac0 | 4.17 | 63.7 | 4.42 | mx_stack: 22 / 168 / ok; mx_recipe: 12 / 60 / ok; or_recipe: 12 / 94 / ok |  |
| mxC2_s8_r1 | 20261122T100640Z | 9e0d2dd0-b3cd-48b2-ab6e-1268cbfe9a6d | 4.17 | 95.4 | 6.63 | mx_recipe: 22 / 176 / ok; mx_stack: 12 / 69 / ok; or_recipe: 12 / 86 / ok |  |
| mxC2_s8_r2 | 20261122T100642Z | fec020f9-f2f7-4ba6-bc96-a82e12ae8a2f | 4.17 | 65.0 | 4.51 | mx_stack: 22 / 177 / ok; or_recipe: 12 / 101 / ok; mx_recipe: 12 / 77 / ok |  |
| mxC2b_s7 | 20261122T114751Z | 69a2eee6-cae2-4b9f-b9b2-74bca70ff376 | 4.17 | 121.5 | 8.44 | or_recipe: 22 / 160 / ok; mx_recipe: 12 / 94 / ok; mx_stack: 12 / 61 / ok |  |
| mxC2b_s8 | 20261122T114753Z | b60b8897-0563-4053-b810-78967f985f98 | 4.17 | 121.8 | 8.45 | mx_stack: 22 / 168 / ok; mx_recipe: 12 / 69 / ok; or_recipe: 12 / 94 / ok |  |
| mxF_s7_r1 | 20261122T091529Z | 6b0075c9-e248-4a7b-9c97-c24151ee0c59 | 4.76 | 41.7 | 3.31 | mx_recipe: 22 / 160 / ok; or_recipe: 12 / 76 / ok |  |
| mxF_s7_r2 | 20261122T091531Z | 8bbd8e64-0623-4cfe-889e-fc6fd694d79d | 4.76 | 42.8 | 3.39 | or_recipe: 22 / 168 / ok; mx_recipe: 12 / 93 / ok |  |
| mxF_s8_r1 | 20261122T091525Z | c9c37569-77cf-482a-af31-81938198d2ee | 4.76 | 54.0 | 4.28 | mx_recipe: 22 / 160 / ok; or_recipe: 12 / 85 / ok |  |
| mxF_s8_r2 | 20261122T091527Z | 917739ce-adbb-4b06-8f8e-e0210f2a7fbb | 4.76 | 42.0 | 3.33 | or_recipe: 22 / 168 / ok; mx_recipe: 12 / 93 / ok |  |
| mxG1_alt | 20261122T085009Z | cb64d6e8-05cd-4b10-bc96-ba1bd6a3f5f2 | 4.76 | 8.8 | 0.69 | nogpu: 1 / 1 / ok |  |
| mxG1_asm | 20261122T085007Z | d555cd1a-c879-4576-bdfa-822b4f0d9c13 | 4.76 | 11.7 | 0.93 | nogpu: 2 / 1 / ok |  |
| mxG1b_ab | 20261122T090635Z | e5299128-1ee2-4395-842a-88f33acabf4c | 4.76 | 6.8 | 0.54 | nogpu: 6 / 4 / ok |  |
| mxG2_s7 | 20261122T090631Z | e7a45bff-31f1-4ef4-85e6-c4a4c7edbe59 | 4.76 | 31.8 | 2.52 | mx_ps: 15 / 159 / ok; mx_recipe: 12 / 52 / ok |  |
| mxG2_s8 | 20261122T090633Z | ae9e305a-87e9-4a53-a3cd-2ac0f3574441 | 4.76 | 31.8 | 2.52 | mx_recipe: 16 / 158 / ok; mx_ps: 12 / 60 / ok |  |
| mxGL1_glue | 20261122T101921Z | e95f1bce-4032-4116-b03b-b47e21e6a6b4 | 4.76 | 2.8 | 0.22 | nogpu: 1 / 1 / ok |  |
| mxGL1b_glue | 20261122T102332Z | 1989f2df-57cf-48eb-902d-c6912f41913b | 4.76 | 2.9 | 0.23 | nogpu: 2 / 1 / ok |  |
| mxH4_s7 | 20261122T072841Z | cbf152e5-9815-4be2-ba5c-2eeeb46b082f | 4.17 | 28.1 | 1.95 | mx_recipe: 22 / 177 / ok |  |
| mxH4_s8 | 20261122T072720Z | 1ea0e9fb-c5d8-40c5-a7c8-1f0a1563fae7 | 4.17 | 28.1 | 1.96 | mx_recipe: 22 / 176 / ok |  |
| mxH4b_s7 | 20261122T091523Z | 6c2a6bdc-7a0d-4407-9f3c-7134c1b7c750 | 4.17 | 36.2 | 2.51 | mx_recipe: 22 / 177 / ok |  |
| mxH4b_s8 | 20261122T091521Z | 3056d209-29bf-4a1c-a661-c3608f3377aa | 4.17 | 35.7 | 2.48 | mx_recipe: 22 / 168 / ok |  |
| mxID_identity | 20261122T075235Z | f780a59c-f08b-40eb-8054-a5e5fd4998ce | 4.76 | 9.9 | 0.79 | unpatched: 22 / 168 / ok; patched: 12 / 60 / ok |  |
| mxK_s7 | 20261122T091535Z | 57462812-fd87-445e-96eb-7597bff419de | 4.76 | 45.1 | 3.57 | or_recipe: 22 / 176 / ok; mx_recipe: 12 / 85 / ok |  |
| mxK_s8 | 20261122T091533Z | 5ffe7a6a-0e71-4e8b-be7a-4e047174ae9d | 4.76 | 44.9 | 3.56 | mx_recipe: 23 / 168 / ok; or_recipe: 12 / 85 / ok |  |
| mxL_lpt_s8 | 20261122T091841Z | 245837ac-004f-4ed2-96ea-7c57429236f6 | 4.76 | 60.6 | 4.80 | mx_recipe: 22 / 168 / ok; mx_lpt6144: 12 / 60 / ok; mx_lpt4096: 12 / 60 / ok |  |
| mxP_ours_profile | 20261122T091537Z | ff05cc47-155c-4985-9150-9c7ee2cc25f7 | 4.76 | 10.9 | 0.86 | prof_ours: 22 / 168 / ok |  |
| mxR4_b10k | 20261122T105424Z | ad29c708-4ef5-4c31-b754-d81967030cf5 | 4.17 | 30.7 | 2.13 | mx_stack: 22 / 168 / ok; mx_stack_b10k: 12 / 93 / ok |  |
| mxR4_b12k | 20261122T105426Z | eaead5e6-4b63-4263-9d2f-d277e9cc6256 | 4.17 | 30.6 | 2.13 | mx_stack_b12k: 22 / 169 / ok; mx_stack: 12 / 94 / ok |  |
| mxR4_s112 | 20261122T105430Z | 59de923b-a13b-4ae8-9063-e28f84e29aa8 | 4.17 | 30.8 | 2.14 | mx_stack_s112: 22 / 168 / ok; mx_stack: 12 / 94 / ok |  |
| mxR4_s80 | 20261122T105428Z | 53944d2c-919a-4ba8-abeb-35ea47fade5c | 4.17 | 30.7 | 2.13 | mx_stack: 22 / 168 / ok; mx_stack_s80: 12 / 85 / ok |  |
| mxS0_vllm_specoff | 20261122T072832Z | 87521648-6f6a-41fb-8eb6-b7215bbf08e4 | 4.76 | 16.4 | 1.30 | or_specoff: 22 / 135 / ok |  |
| mxS1_sglang | 20261122T072834Z | None | 4.76 | 0.0 | 0.00 |  | RuntimeError('snapshot or-sglang0523rocm720-mi355x-c8m48d150 failed: Snapshot size (22.20GB) exceeds maximum allowed siz |
| mxS1_sglang | 20261122T073259Z | 0a20abf7-1345-4e5d-bdea-7a9d053f5079 | 4.76 | 17.3 | 1.37 | sgl_mxfp4: 12 / 26 / NOT READY; sgl_fp8: 22 / 119 / ok |  |
| mxS2_sglang_mxfp4 | 20261122T075418Z | 2536c932-936c-4bba-a40d-9e6b221ad49c | 4.76 | 1.9 | 0.15 | sgl_mxfp4: 22 / 35 / NOT READY |  |
| mxS3_vllm_fp8 | 20261122T080305Z | 0714fc55-39f5-4926-8563-659ad4742df6 | 4.76 | 0.7 | 0.06 |  |  |
| mxS3_vllm_fp8 | 20261122T080325Z | 10c1d4e5-b4a5-49f1-a51f-8b1bc006b515 | 4.76 | 12.2 | 0.96 | vllm_fp8: 22 / 143 / ok |  |
| pilot_h100_vllm_fp8 | 20261122T034304Z | d71b6546-0f11-4e36-be30-fd5877e23300 | 3.67 | 30.4 | 1.86 | fp8dyn: 697 / 152 / ok |  |
| q1_h200_bf16_fp8 | 20261122T073634Z | 7327edfb-afc8-4901-97e7-69d2bbc5ac32 | 3.80 | 17.3 | 1.09 | bf16_ref: 31 / 141 / ok; fp8dyn: 31 / 132 / ok |  |
| q1_h200_bf16_fp8 | 20261122T074352Z | 53aa8d0f-cdb3-4ad8-9475-08022d303377 | 3.80 | 8.9 | 0.56 | bf16_ref: 32 / 150 / ok |  |
| q2_mi355x_recipes | 20261122T073636Z | 4c398fdb-74ea-4be6-a0cd-2235ce0990fc | 4.76 | 13.1 | 1.04 | or_recipe: 22 / 168 / ok; or_recipe_nomtp: 12 / 76 / ok; mx_recipe: 12 / 85 / ok |  |
| q2_mi355x_recipes | 20261122T074354Z | 09fb5ecf-007a-4ac1-8b98-2c977915cec0 | 4.76 | 5.5 | 0.43 | or_recipe: 22 / 168 / ok |  |
| q3_pro6000_nvfp4 | 20261122T073638Z | 5d579ead-0255-411e-986b-acdbbc350e5f | 2.42 | 9.2 | 0.37 | nvfp4_fi: 42 / 206 / ok |  |
| q3_pro6000_nvfp4 | 20261122T074356Z | eea89ebe-e46e-4da2-9201-2321abe58e6c | 2.42 | 9.0 | 0.36 | nvfp4_fi: 98 / 215 / ok |  |
| s1_b300_nvfp4_mtp2_r3 | 20261122T045011Z | da9ec62b-0275-4a0d-a903-a839692896a1 | 8.09 | 8.0 | 1.08 | nvfp4_mtp2_r3: 63 / 52 / NOT READY |  |
| s1_b300_nvfp4_mtp2_r4 | 20261122T045943Z | 2b75650b-24b5-426f-9bc8-371e4cd4f504 | 8.09 | 13.5 | 1.82 | nvfp4_mtp2_r4: 53 / 221 / ok |  |
| s1_b300_nvfp4_mtp2_v034snap | 20261122T044101Z | None | 8.09 | 0.1 | 0.01 |  | ApiError('HTTP 503 on /sandbox: {"path":"/api/sandbox","timestamp":"2026-11-22T04:41:07.143Z","statusCode":503,"error":" |
| s1_b300_nvfp4_mtp2_v034snap | 20261122T044102Z | 0e4123bc-48d1-45e0-80b8-7c2c1295466f | 8.09 | 4.5 | 0.61 | nvfp4_mtp2: 63 / 170 / NOT READY |  |
| s1_b300_nvfp4_mtp2_v034snap_fa4 | 20261122T044603Z | 9aac419f-1cd9-4104-a26d-9cf3d4378c45 | 8.09 | 3.0 | 0.41 | nvfp4_mtp2_fa4: 63 / 78 / NOT READY |  |
| s1_h100_fp8 | 20261122T042445Z | 830fb81b-4616-4b70-9cf1-1ee1e9d6e3c3 | 3.67 | 12.2 | 0.75 | fp8dyn: 24 / 142 / ok |  |
| s1_h200_fp8 | 20261122T042702Z | 4eb042c0-9078-4431-a2c2-933729e504d2 | 3.80 | 12.4 | 0.78 | fp8dyn: 82 / 150 / ok |  |
| s1_l40s_w4a16 | 20261122T042449Z | 558b3239-d6a5-47b3-a1c5-e3859e4e0a23 | 2.07 | 6.2 | 0.21 | w4a16qat: 176 / 144 / rejected: TRITON_ATTN on global 512-dim layers |  |
| s1_l40s_w4a16_fi | 20261122T043255Z | d170b64c-2a96-4a2f-ba2b-12965790c772 | 2.07 | 27.5 | 0.95 | w4a16qat_fi: 166 / 253 / ok |  |
| s1_mi355x_openrelay_recipe | 20261122T042451Z | c1fd041e-a2c0-4e16-a83e-93068ccf60eb | 4.76 | 4.0 | 0.32 | or_recipe: 42 / 160 / rejected: missing attention_backend.global_512 |  |
| s1_rtx5090_nvfp4 | 20261122T042447Z | 27d42cba-95c7-4b49-976d-190075706547 | 1.67 | 5.9 | 0.17 | nvfp4: 78 / 201 / rejected: TRITON_ATTN on global 512-dim layers |  |
| s1_rtxpro6000_nvfp4 | 20261122T042443Z | a729020e-916f-44f6-961e-93e2c83b8b71 | 2.88 | 5.4 | 0.26 | nvfp4: 114 / 153 / rejected: TRITON_ATTN on global 512-dim layers |  |
| s1_rtxpro6000_nvfp4_fi | 20261122T043253Z | 7aadf5d6-023d-4409-b2db-ce1b76b831a7 | 2.88 | 12.3 | 0.59 | nvfp4_fi: 82 / 165 / ok |  |
| s2_mi355x_or_recipe_seed7 | 20261122T043703Z | 40eb1917-477e-4019-8084-3098f38caa74 | 4.76 | 34.0 | 2.70 | or_recipe: 52 / 168 / ok |  |
| s2_mi355x_or_recipe_seed8 | 20261122T043705Z | 1135d94e-11e9-4875-854a-bc2a6a8917f3 | 4.76 | 28.1 | 2.23 | or_recipe: 32 / 168 / ok |  |
| s2_mi355x_or_recipe_text_seed7 | 20261122T051007Z | abc73505-9e40-4671-b441-b164deaf792a | 4.76 | 5.6 | 0.45 | or_recipe: 32 / 160 / - |  |
| s2a_mi355x_or_recipe_seed8 | 20261122T054934Z | 29ad5614-9f47-459c-b3a4-ed68dee03211 | 4.76 | 22.1 | 1.75 | or_recipe: 22 / 160 / ok |  |
| s2b_mi355x_or_recipe_seed7 | 20261122T053337Z | 00d8bf80-06cb-4799-8f3e-3f3395bf6f1d | 4.76 | 24.6 | 1.95 | or_recipe: 32 / 168 / ok |  |
| s2b_mi355x_or_recipe_seed8 | 20261122T053339Z | 15ba3949-e8bd-4d62-9d8f-a35dc5339066 | 4.76 | 15.8 | 1.25 | or_recipe: 32 / 160 / - | ApiError('HTTP 404 on https://proxy.app.daytona.io/toolbox/15ba3949-e8bd-4d62-9d8f-a35dc5339066/process/execute: {"statu EVICTED |
| s2b_mi355x_or_recipe_seed8 | 20261122T054928Z | 1b0f53ee-d5b3-4c32-a406-a297fa83154c | 4.76 | 24.8 | 1.96 | or_recipe: 52 / 160 / ok |  |
| s2c_mi355x_or_recipe_seed7 | 20261122T051851Z | f3677465-3a16-48e3-8690-924d6f8e1b34 | 4.76 | 37.8 | 3.00 | or_recipe: 42 / 168 / ok |  |
| s2c_mi355x_or_recipe_seed8 | 20261122T051853Z | 1e73f7e8-bc0b-48b7-b63d-61aa9c98bb14 | 4.76 | 30.5 | 2.42 | or_recipe: 52 / 160 / - | ApiError('HTTP 404 on https://proxy.app.daytona.io/toolbox/1e73f7e8-bc0b-48b7-b63d-61aa9c98bb14/process/execute: {"statu EVICTED |
| s2c_mi355x_or_recipe_seed8 | 20261122T054922Z | b155e289-f543-4eb4-b833-ab3324afa0c8 | 4.76 | 0.5 | 0.04 |  |  |
| s3_rtxpro6000_small_mtp | 20261122T050407Z | 58d787d0-9a26-42c3-9b68-32e0cc8631da | 2.42 | 40.2 | 1.62 | p6_small_base: 88 / 215 / ok; p6_small_mtp2: 4 / 171 / ok; p6_small_mtp3: 4 / 144 / ok |  |
| up1_h100_draft_compact | 20261122T153343Z | None | 3.67 | 0.0 | 0.00 |  |  |
| up1_h100_draft_compact | 20261122T153455Z | cf1234e9-5732-4970-bb4b-c3dbfbb9bba2 | 3.67 | 46.0 | 2.81 | tests: 1 / 1 / ok; unpatched_s7: 32 / 157 / ok; patched_s7: 11 / 75 / - | ApiError('HTTP 404 on https://proxy.app.daytona.io/toolbox/cf1234e9-5732-4970-bb4b-c3dbfbb9bba2/process/execute: {"statu EVICTED |
| up1_h100_draft_compact | 20261122T163243Z | 97214921-0a22-409b-90b5-8029cbfcb32a | 3.67 | 99.2 | 6.06 | tests: 1 / 1 / ok; unpatched_s7: 21 / 174 / ok; patched_s7: 11 / 100 / ok; patched_s8: 11 / 92 / ok; unpatched_s8: 11 / 92 / ok; tests_existing: 1 / 1 / ok |  |
| up2_h100_e2e_tests | 20261122T161311Z | 89aa8399-e9c2-4676-84b4-0d38d5a91b3b | 3.67 | 5.6 | 0.34 | tests: 1 / 1 / ok |  |

Total estimated: $328.46 over 177 runs (Daytona-metered: infra/spend.py).
