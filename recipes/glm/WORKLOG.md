# WORKLOG: GLM-5.3-Flash phase 3 (2026-11-22 07:2x-09:1x UTC)

- 07:32Z: feasibility. vLLM 0.34 + amd Quark MXFP4 at TP1 started first try. The SGLang ROCm 7.2 image is over Daytona's 20 GB
  snapshot limit, so I moved to ROCm 10.
- 07:4x-08:2xZ: screens. Spec off 1.5 req/s; MTP k=3 1.5; k=5 1.0; 16K batched tokens 1.5 (prefill +13%); SGLang 1.0.
  - The GLM gate was recalibrated: the 100-item gate falsely fails a 97.2% model with p = 0.31.
  - The knee profile shows no dominant fixable slice.
- 08:29-09:1xZ: claim-grade best effort on both seeds. 1.5 req/s on both columns, 1.89x / 1.81x OpenRelay's GLM price.
- Spend $23.08 over 9 sandboxes (`tools/spend.py`).

DoD GLM phase 3 (t016): BASELINE: OpenRelay's glm-5.3-flash sync price ($0.000496/request) and AMD's published GSM8K for the
checkpoint (97.12-97.19%) | GAP-HUNT: claims.json re-derived (glm_best_C claim-grade A+B); the MoE tuning-gap hypothesis
checked and refuted (TP1 rows exist via the GLM-5.3 table); the profile box ran into its budget cap and was analyzed by hand |
GAPS-FILED: none (the answer is a measured "no")
