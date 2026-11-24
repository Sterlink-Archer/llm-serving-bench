
#include <hip/hip_runtime.h>
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v16f __attribute__((ext_vector_type(16)));
typedef float v4f __attribute__((ext_vector_type(4)));
extern "C" __global__ __launch_bounds__(256) void mfma32(float* out, int iters, int zero) {
  v8i a, b;
  unsigned t = threadIdx.x + 977u * blockIdx.x;
  for (int i = 0; i < 8; ++i) {
    a[i] = zero ? 0 : (int)((t * 0x9E3779B1u) ^ ((unsigned)i * 0x85EBCA6Bu));
    b[i] = zero ? 0 : (int)((t * 0xC2B2AE35u) + (unsigned)i * 0x27D4EB2Fu);
  }
  v16f c0 = {}, c1 = {}, c2 = {}, c3 = {};
  int sa = 127 + (int)(t & 1), sb = 127;
  for (int it = 0; it < iters; ++it) {
    c0 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c0, 4, 4, 0, sa, 0, sb);
    c1 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(b, a, c1, 4, 4, 0, sa, 0, sb);
    c2 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, a, c2, 4, 4, 0, sb, 0, sa);
    c3 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(b, b, c3, 4, 4, 0, sb, 0, sa);
  }
  float s = 0.f;
  for (int i = 0; i < 16; ++i) s += c0[i] + c1[i] + c2[i] + c3[i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
extern "C" __global__ __launch_bounds__(256) void mfma16(float* out, int iters, int zero) {
  v8i a, b;
  unsigned t = threadIdx.x + 977u * blockIdx.x;
  for (int i = 0; i < 8; ++i) {
    a[i] = zero ? 0 : (int)((t * 0x9E3779B1u) ^ ((unsigned)i * 0x85EBCA6Bu));
    b[i] = zero ? 0 : (int)((t * 0xC2B2AE35u) + (unsigned)i * 0x27D4EB2Fu);
  }
  v4f c0 = {}, c1 = {}, c2 = {}, c3 = {};
  int sa = 127 + (int)(t & 1), sb = 127;
  for (int it = 0; it < iters; ++it) {
    c0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c0, 4, 4, 0, sa, 0, sb);
    c1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b, a, c1, 4, 4, 0, sa, 0, sb);
    c2 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, a, c2, 4, 4, 0, sb, 0, sa);
    c3 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b, b, c3, 4, 4, 0, sb, 0, sa);
  }
  float s = 0.f;
  for (int i = 0; i < 4; ++i) s += c0[i] + c1[i] + c2[i] + c3[i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
