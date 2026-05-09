// G1 Gate forward kernel for SM100 (B200).
//
// N-adaptive launch geometry: small N (`n*hidden_size <= 1.5M elements`) uses
// `BLOCK=128, GRIDX=8` to maximize concurrent blocks per SM; large N falls
// back to `BLOCK=256, GRIDX=4` to amortize per-block scheduling cost.
// Inline-PTX `ex2.approx.ftz.f32` + `rcp.approx.ftz.f32` replace the standard
// `expf`/reciprocal paths, eliminating denormal-handling branches and the
// Newton-Raphson refine step.

#pragma once

#include <cuda_bf16.h>
#include <cstdint>
#include <algorithm>

namespace g1_cute {

struct alignas(16) bf16x8_t {
  __nv_bfloat162 v[4];
};

__device__ __forceinline__ bf16x8_t ld_bf16x8(const __nv_bfloat16* __restrict__ p) {
  bf16x8_t r;
  *reinterpret_cast<float4*>(&r) = __ldg(reinterpret_cast<const float4*>(p));
  return r;
}

__device__ __forceinline__ void st_bf16x8(__nv_bfloat16* __restrict__ p,
                                          const bf16x8_t& x) {
  *reinterpret_cast<float4*>(p) = *reinterpret_cast<const float4*>(&x);
}

__device__ __forceinline__ float ex2_ftz(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float rcp_ftz(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
  return rcp_ftz(1.f + ex2_ftz(-x * 1.4426950408889634f));
}

__device__ __forceinline__ void g1_pair(__nv_bfloat162 lin,
                                        __nv_bfloat162 att,
                                        __nv_bfloat162& out,
                                        __nv_bfloat162& gate) {
  float2 fl = __bfloat1622float2(lin);
  float2 fa = __bfloat1622float2(att);
  float gx = fast_sigmoid(fl.x);
  float gy = fast_sigmoid(fl.y);
  gate = __float22bfloat162_rn(make_float2(gx, gy));
  out  = __float22bfloat162_rn(make_float2(fa.x * gx, fa.y * gy));
}

__device__ __forceinline__ void g1_pair_out_only(__nv_bfloat162 lin,
                                                 __nv_bfloat162 att,
                                                 __nv_bfloat162& out) {
  float2 fl = __bfloat1622float2(lin);
  float2 fa = __bfloat1622float2(att);
  float gx = fast_sigmoid(fl.x);
  float gy = fast_sigmoid(fl.y);
  out = __float22bfloat162_rn(make_float2(fa.x * gx, fa.y * gy));
}

// 8-element vec full kernel.
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
g1_gate_cute_kernel_v2(
    const __nv_bfloat16* __restrict__ linear_out,
    const __nv_bfloat16* __restrict__ attn_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ gate,
    int64_t n_total) {
  const int64_t n_vec8 = n_total >> 3;
  const int64_t tid    = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * BLOCK;

  for (int64_t i = tid; i < n_vec8; i += stride) {
    const int64_t off = i << 3;
    bf16x8_t lin_v = ld_bf16x8(linear_out + off);
    bf16x8_t att_v = ld_bf16x8(attn_out + off);
    bf16x8_t out_v, gate_v;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      g1_pair(lin_v.v[j], att_v.v[j], out_v.v[j], gate_v.v[j]);
    }
    st_bf16x8(output + off, out_v);
    st_bf16x8(gate   + off, gate_v);
  }

  const int64_t tail_start = n_vec8 << 3;
  for (int64_t i = tail_start + tid; i < n_total; i += stride) {
    float fl = __bfloat162float(linear_out[i]);
    float fa = __bfloat162float(attn_out[i]);
    float fg = fast_sigmoid(fl);
    gate[i]   = __float2bfloat16(fg);
    output[i] = __float2bfloat16(fa * fg);
  }
}

// 16-element vec fused kernel.
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
g1_gate_cute_fused_kernel_v2(
    const __nv_bfloat16* __restrict__ linear_out,
    const __nv_bfloat16* __restrict__ attn_out,
    __nv_bfloat16* __restrict__ output,
    int64_t n_total) {
  const int64_t n_vec16 = n_total >> 4;
  const int64_t tid    = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * BLOCK;

  for (int64_t i = tid; i < n_vec16; i += stride) {
    const int64_t off = i << 4;
    bf16x8_t lin0 = ld_bf16x8(linear_out + off);
    bf16x8_t lin1 = ld_bf16x8(linear_out + off + 8);
    bf16x8_t att0 = ld_bf16x8(attn_out + off);
    bf16x8_t att1 = ld_bf16x8(attn_out + off + 8);
    bf16x8_t out0, out1;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      g1_pair_out_only(lin0.v[j], att0.v[j], out0.v[j]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      g1_pair_out_only(lin1.v[j], att1.v[j], out1.v[j]);
    }
    st_bf16x8(output + off,     out0);
    st_bf16x8(output + off + 8, out1);
  }

  const int64_t rem16 = n_vec16 << 4;
  const int64_t n_rem8 = (n_total - rem16) >> 3;
  for (int64_t i = tid; i < n_rem8; i += stride) {
    const int64_t off = rem16 + (i << 3);
    bf16x8_t lin_v = ld_bf16x8(linear_out + off);
    bf16x8_t att_v = ld_bf16x8(attn_out + off);
    bf16x8_t out_v;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      g1_pair_out_only(lin_v.v[j], att_v.v[j], out_v.v[j]);
    }
    st_bf16x8(output + off, out_v);
  }
  const int64_t scal = rem16 + (n_rem8 << 3);
  for (int64_t i = scal + tid; i < n_total; i += stride) {
    float fl = __bfloat162float(linear_out[i]);
    float fa = __bfloat162float(attn_out[i]);
    float fg = fast_sigmoid(fl);
    output[i] = __float2bfloat16(fa * fg);
  }
}

}  // namespace g1_cute

// N-adaptive launch policy. Threshold tuned empirically on B200.
// For D=7168 production: switch from BLOCK=128 to BLOCK=256 at N==210
// (n == 1.5M elements). Tunable via -DG1_BLOCK128_N_THRESHOLD=<elem-count>.
#ifndef G1_BLOCK128_N_THRESHOLD
#define G1_BLOCK128_N_THRESHOLD 1500000  // ~N=210 for D=7168
#endif

inline void launch_g1_gate_cute(
    const __nv_bfloat16* linear_out,
    const __nv_bfloat16* attn_out,
    __nv_bfloat16* output,
    __nv_bfloat16* gate,
    int64_t n,
    cudaStream_t stream,
    int device) {
  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

  const bool use_block128 = (n < G1_BLOCK128_N_THRESHOLD);

  if (gate != nullptr) {
    if (use_block128) {
      constexpr int BLOCK = 128;
      const int64_t need = (n + BLOCK * 8 - 1) / (BLOCK * 8);
      int blocks = (int)std::min<int64_t>(need, (int64_t)sm_count * 8);
      if (blocks < 1) blocks = 1;
      g1_cute::g1_gate_cute_kernel_v2<BLOCK><<<blocks, BLOCK, 0, stream>>>(
          linear_out, attn_out, output, gate, n);
    } else {
      constexpr int BLOCK = 256;
      const int64_t need = (n + BLOCK * 8 - 1) / (BLOCK * 8);
      int blocks = (int)std::min<int64_t>(need, (int64_t)sm_count * 4);
      if (blocks < 1) blocks = 1;
      g1_cute::g1_gate_cute_kernel_v2<BLOCK><<<blocks, BLOCK, 0, stream>>>(
          linear_out, attn_out, output, gate, n);
    }
  } else {
    if (use_block128) {
      constexpr int BLOCK = 128;
      const int64_t need = (n + BLOCK * 16 - 1) / (BLOCK * 16);
      int blocks = (int)std::min<int64_t>(need, (int64_t)sm_count * 8);
      if (blocks < 1) blocks = 1;
      g1_cute::g1_gate_cute_fused_kernel_v2<BLOCK><<<blocks, BLOCK, 0, stream>>>(
          linear_out, attn_out, output, n);
    } else {
      constexpr int BLOCK = 256;
      const int64_t need = (n + BLOCK * 16 - 1) / (BLOCK * 16);
      int blocks = (int)std::min<int64_t>(need, (int64_t)sm_count * 4);
      if (blocks < 1) blocks = 1;
      g1_cute::g1_gate_cute_fused_kernel_v2<BLOCK><<<blocks, BLOCK, 0, stream>>>(
          linear_out, attn_out, output, n);
    }
  }
}
