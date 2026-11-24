#!/usr/bin/env python3
"""Pins the prefill-profile kernel classifier: attention kernels that are cutlass templates must count as attention.
  python3 -I tests/test_profile_classify.py"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "bench"))
import profile_prefill as P  # noqa: E402

CASES = [
    ("void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>>, true>>>>(...)", "attention"),
    ("kernel_cutlass___call___vllmvllm_flash_attncutesm100_hd256_2cta_fmha_forwardBlackwellFusedMultiHeadAttentionForward", "attention"),
    ("void flashinfer::BatchPrefillWithPagedKVCacheKernel<...>", "attention"),
    ("_fwd_kernel_unified_attention_2d", "attention"),
    ("sm90_xmma_gemm_e4m3e4m3_e4m3f32_f32_tn_n_tilesize128x128x128_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas", "gemm"),
    ("void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<...Sm120BlockScaledNvf4...>>", "gemm"),
    ("nvjet_tst_192x192_64x3_2x1_v_bz_coopA_TNT", "gemm"),
    ("kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDenseGemmKernel_object_at__TiledMMA", "gemm"),
    ("void flashinfer::BatchPrefillWithPagedKVCacheKernel<true, flashinfer::KernelTraits<(flashinfer::MaskMode)1, 32u, 2u", "attention"),
    ("_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEENS1_10collective13CollectiveMma", "gemm"),
    ("triton_red_fused_add_mul_rsqrt_0", "fused_elementwise"),
    ("_fwd_kernel", "attention"),
    ("_gemma_qkv_rmsnorm_kernel", "fused_elementwise"),
    ("kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16", "fused_elementwise"),
    ("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>", "gemm"),
    ("triton_red_fused__to_copy_abs_amax_clamp_cutlass_scaled_mm_div_gelu_mul_reciprocal_slice_unsqueeze_2", "fused_elementwise"),
    ("void vllm::cvt_fp16_to_fp4<__nv_bfloat16, false, false>(int, int, int, int, __nv_bfloat16 const*, float const*", "fused_elementwise"),
    ("void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, unsigned char>", "other"),
    ("Memcpy HtoD (Pageable -> Device)", "memcpy/memset"),
]
fails = []
for name, want in CASES:
    got = P.classify(name, "gpu_memcpy" if name.startswith("Memcpy") else "kernel")
    if got != want:
        fails.append(f"{name[:70]}: got {got}, want {want}")
print("FAIL\n  " + "\n  ".join(fails) if fails else f"OK: {len(CASES)} kernel names classified")
sys.exit(1 if fails else 0)
