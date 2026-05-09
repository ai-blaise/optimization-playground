// G1 Gate forward kernel.
//
// Two entry points:
//   g1_gate_forward       -- writes both output and gate tensors
//   g1_gate_forward_fused -- output-only (inference fast-path)
//
// SM100+: CuTe warp-specialized persistent kernel (g1_attention_cute.cuh).
// SM90-:  vectorized kernel with __ldcs/__stcs cache-streaming hints.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cuda_bf16.h>

#include "utils.h"

#include "g1_attention_cute.cuh"

struct alignas(16) bf16x8 {
  __nv_bfloat162 v[4];
};

__device__ __forceinline__ bf16x8 load_bf16x8(const __nv_bfloat16* __restrict__ p) {
  bf16x8 result;
  *reinterpret_cast<float4*>(&result) = __ldg(reinterpret_cast<const float4*>(p));
  return result;
}

__device__ __forceinline__ bf16x8 load_bf16x8_cs(const __nv_bfloat16* __restrict__ p) {
  bf16x8 result;
  *reinterpret_cast<float4*>(&result) = __ldcs(reinterpret_cast<const float4*>(p));
  return result;
}

__device__ __forceinline__ void store_bf16x8(__nv_bfloat16* __restrict__ p, const bf16x8& x) {
  *reinterpret_cast<bf16x8*>(p) = x;
}

__device__ __forceinline__ void store_bf16x8_cs(__nv_bfloat16* __restrict__ p, const bf16x8& x) {
  __stcs(reinterpret_cast<float4*>(p), *reinterpret_cast<const float4*>(&x));
}

__device__ __forceinline__ float fast_sigmoid(float x) {
  return 1.f / (1.f + expf(-x));
}

__device__ __forceinline__ void compute_g1_gate(
    __nv_bfloat162 lin,
    __nv_bfloat162 attn,
    __nv_bfloat162& out,
    __nv_bfloat162& gate) {
  float2 fl = __bfloat1622float2(lin);
  float2 fa = __bfloat1622float2(attn);
  float2 fg = {fast_sigmoid(fl.x), fast_sigmoid(fl.y)};
  gate = __float22bfloat162_rn(fg);
  out = __float22bfloat162_rn({fa.x * fg.x, fa.y * fg.y});
}

__device__ __forceinline__ void compute_g1_gate_out_only(
    __nv_bfloat162 lin,
    __nv_bfloat162 attn,
    __nv_bfloat162& out) {
  float2 fl = __bfloat1622float2(lin);
  float2 fa = __bfloat1622float2(attn);
  float2 fg = {fast_sigmoid(fl.x), fast_sigmoid(fl.y)};
  out = __float22bfloat162_rn({fa.x * fg.x, fa.y * fg.y});
}

template <int BLOCK>
__global__ void __launch_bounds__(BLOCK) g1_gate_fwd_kernel(
    const __nv_bfloat16* __restrict__ linear_out,
    const __nv_bfloat16* __restrict__ attn_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ gate,
    int64_t n_total) {
  const int64_t tid = int64_t(blockIdx.x) * BLOCK + threadIdx.x;
  const int64_t stride = int64_t(gridDim.x) * BLOCK;
  const int64_t n_vec8 = n_total / 8;
  const int64_t rem_start = n_vec8 * 8;

  for (int64_t i = tid; i < n_vec8; i += stride) {
    const int64_t off = i * 8;
    bf16x8 lin = load_bf16x8(linear_out + off);
    bf16x8 attn = load_bf16x8(attn_out + off);
    bf16x8 o, g;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      compute_g1_gate(lin.v[j], attn.v[j], o.v[j], g.v[j]);
    }
    store_bf16x8(output + off, o);
    store_bf16x8(gate + off, g);
  }

  for (int64_t i = rem_start + tid; i < n_total; i += stride) {
    float fl = __bfloat162float(linear_out[i]);
    float fa = __bfloat162float(attn_out[i]);
    float fg = fast_sigmoid(fl);
    gate[i] = __float2bfloat16(fg);
    output[i] = __float2bfloat16(fa * fg);
  }
}

// Output-only fused kernel: 16-element vectors with cache-streaming hints.
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK) g1_gate_fused_fwd_kernel(
    const __nv_bfloat16* __restrict__ linear_out,
    const __nv_bfloat16* __restrict__ attn_out,
    __nv_bfloat16* __restrict__ output,
    int64_t n_total) {
  const int64_t tid = int64_t(blockIdx.x) * BLOCK + threadIdx.x;
  const int64_t stride = int64_t(gridDim.x) * BLOCK;

  const int64_t n_vec16 = n_total / 16;
  for (int64_t i = tid; i < n_vec16; i += stride) {
    const int64_t off = i * 16;
    bf16x8 lin0 = load_bf16x8_cs(linear_out + off);
    bf16x8 lin1 = load_bf16x8_cs(linear_out + off + 8);
    bf16x8 att0 = load_bf16x8_cs(attn_out + off);
    bf16x8 att1 = load_bf16x8_cs(attn_out + off + 8);
    bf16x8 o0, o1;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      compute_g1_gate_out_only(lin0.v[j], att0.v[j], o0.v[j]);
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      compute_g1_gate_out_only(lin1.v[j], att1.v[j], o1.v[j]);
    }
    store_bf16x8_cs(output + off, o0);
    store_bf16x8_cs(output + off + 8, o1);
  }

  const int64_t rem16_start = n_vec16 * 16;
  const int64_t n_rem_vec8 = (n_total - rem16_start) / 8;
  for (int64_t i = tid; i < n_rem_vec8; i += stride) {
    const int64_t off = rem16_start + i * 8;
    bf16x8 lin = load_bf16x8_cs(linear_out + off);
    bf16x8 attn = load_bf16x8_cs(attn_out + off);
    bf16x8 o;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      compute_g1_gate_out_only(lin.v[j], attn.v[j], o.v[j]);
    }
    store_bf16x8_cs(output + off, o);
  }

  const int64_t rem8_start = rem16_start + n_rem_vec8 * 8;
  for (int64_t i = rem8_start + tid; i < n_total; i += stride) {
    float fl = __bfloat162float(linear_out[i]);
    float fa = __bfloat162float(attn_out[i]);
    float fg = fast_sigmoid(fl);
    output[i] = __float2bfloat16(fa * fg);
  }
}

static int get_sm_version_cached(int device) {
  static thread_local int cached_device = -1;
  static thread_local int cached_version = 0;
  if (device != cached_device) {
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    cached_version = major * 10 + minor;
    cached_device = device;
  }
  return cached_version;
}

static void launch_g1_gate_common(
    const __nv_bfloat16* linear_out,
    const __nv_bfloat16* attn_out,
    __nv_bfloat16* output,
    __nv_bfloat16* gate,
    int64_t n,
    cudaStream_t stream,
    int device) {
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

  const int sm_version = get_sm_version_cached(device);
  if (sm_version >= 100) {
    launch_g1_gate_cute(linear_out, attn_out, output, gate, n, stream, device);
    return;
  }

  if (gate != nullptr) {
    constexpr int BLOCK = 256;
    int blocks = std::min(sm_count * 4, int((n + BLOCK * 8 - 1) / (BLOCK * 8)));
    if (blocks < 1) blocks = 1;
    g1_gate_fwd_kernel<BLOCK><<<blocks, BLOCK, 0, stream>>>(
        linear_out, attn_out, output, gate, n);
  } else {
    constexpr int BLOCK = 512;
    int blocks = std::min(sm_count * 2, int((n + BLOCK * 16 - 1) / (BLOCK * 16)));
    if (blocks < 1) blocks = 1;
    g1_gate_fused_fwd_kernel<BLOCK><<<blocks, BLOCK, 0, stream>>>(
        linear_out, attn_out, output, n);
  }
}

void g1_gate_forward(at::Tensor linear_out, at::Tensor attn_out, at::Tensor output, at::Tensor gate) {
  CHECK_INPUT(linear_out);
  CHECK_INPUT(attn_out);
  CHECK_INPUT(output);
  CHECK_INPUT(gate);
  CHECK_EQ(linear_out.dtype(), at::kBFloat16);
  CHECK_EQ(attn_out.dtype(), at::kBFloat16);
  CHECK_EQ(output.dtype(), at::kBFloat16);
  CHECK_EQ(gate.dtype(), at::kBFloat16);
  CHECK_EQ(linear_out.numel(), attn_out.numel());
  CHECK_EQ(output.numel(), linear_out.numel());
  CHECK_EQ(gate.numel(), linear_out.numel());

  int64_t n = linear_out.numel();
  if (n <= 0) return;

  const c10::cuda::OptionalCUDAGuard device_guard(linear_out.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  launch_g1_gate_common(
      reinterpret_cast<const __nv_bfloat16*>(linear_out.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(attn_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(gate.data_ptr()),
      n, stream, linear_out.get_device());
}

void g1_gate_forward_fused(at::Tensor linear_out, at::Tensor attn_out, at::Tensor output) {
  CHECK_INPUT(linear_out);
  CHECK_INPUT(attn_out);
  CHECK_INPUT(output);
  CHECK_EQ(linear_out.dtype(), at::kBFloat16);
  CHECK_EQ(attn_out.dtype(), at::kBFloat16);
  CHECK_EQ(output.dtype(), at::kBFloat16);
  CHECK_EQ(linear_out.numel(), attn_out.numel());
  CHECK_EQ(output.numel(), linear_out.numel());

  int64_t n = linear_out.numel();
  if (n <= 0) return;

  const c10::cuda::OptionalCUDAGuard device_guard(linear_out.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  launch_g1_gate_common(
      reinterpret_cast<const __nv_bfloat16*>(linear_out.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(attn_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      nullptr,  // no gate output
      n, stream, linear_out.get_device());
}
