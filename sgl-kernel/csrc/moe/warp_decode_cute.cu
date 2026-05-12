// Copyright 2024-2026 SGLang Team
// Licensed under the Apache License, Version 2.0.
// ==============================================================================
// Blog-strict warp_decode_cute dispatch (https://cursor.com/blog/warp-decode):
//   - Gate/Up: TILE_N=8 (one neuron per warp), TILE_K=1024, NUM_WARPS=8
//   - Down:    TILE_D=8 (one dim per warp),    TILE_N=2048, NUM_WARPS=8
// 3-stage cp.async pipeline; optional gate/up→down PDL chain via WD_PDL_ENABLED.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "warp_decode_cute.cuh"

namespace sglang {
namespace warp_decode {

constexpr int kGateUpTileN = 8;
constexpr int kGateUpTileK = 1024;
constexpr int kGateUpNumWarps = 8;
constexpr int kGateUpNumThreads = kGateUpNumWarps * 32;

constexpr int kDownTileD = 8;
constexpr int kDownTileN = 2048;
constexpr int kDownNumWarps = 8;
constexpr int kDownNumThreads = kDownNumWarps * 32;

template <int TILE_K>
constexpr int GateUpSmemBytes() {
  return 3 * (TILE_K + 2 * kGateUpTileN * TILE_K) *
      static_cast<int>(sizeof(__nv_bfloat16));
}

template <int TILE_N>
constexpr int DownSmemBytes() {
  return 3 * (TILE_N + kDownTileD * TILE_N) *
      static_cast<int>(sizeof(__nv_bfloat16));
}

template <typename KernelFunc>
void MaybeSetSmemAttribute(KernelFunc kernel, int smem_bytes) {
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }
}

void CheckWarpDecodeCuteShape(int hidden_size, int intermediate_size) {
  TORCH_CHECK(
      hidden_size % kGateUpTileK == 0,
      "warp_decode_cute requires hidden_size divisible by ", kGateUpTileK,
      ", got ", hidden_size);
  TORCH_CHECK(
      hidden_size % kDownTileD == 0,
      "warp_decode_cute requires hidden_size divisible by ", kDownTileD,
      ", got ", hidden_size);
  TORCH_CHECK(
      intermediate_size % kDownTileN == 0,
      "warp_decode_cute requires intermediate_size divisible by ", kDownTileN,
      ", got ", intermediate_size);
}

// Launches the kernel with cudaLaunchAttributeProgrammaticStreamSerialization
// when WD_PDL_ENABLED is set so the runtime can overlap the launch latency
// with the prior in-stream kernel's tail. Falls back to triple-chevron
// otherwise.
template <typename KernelPtr, typename... Args>
inline void WdLaunch(KernelPtr kernel, dim3 grid, dim3 block, size_t smem,
                     cudaStream_t stream, Args&&... args) {
#if defined(WD_PDL_ENABLED) && WD_PDL_ENABLED
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = 1;
  cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...);
#else
  kernel<<<grid, block, smem, stream>>>(std::forward<Args>(args)...);
#endif
}

template <int TILE_K>
void warp_decode_cute_gate_up_impl(
    const torch::Tensor& x, const torch::Tensor& w_gate, const torch::Tensor& w_up,
    torch::Tensor& out, const torch::Tensor& expert_ids,
    int hidden_size, int intermediate_size, int top_k, int num_tokens) {
  if (num_tokens == 0) return;
  TORCH_CHECK(x.dtype() == torch::kBFloat16);
  TORCH_CHECK(w_gate.dtype() == torch::kBFloat16);
  TORCH_CHECK(w_up.dtype() == torch::kBFloat16);
  TORCH_CHECK(out.dtype() == torch::kBFloat16);
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto kernel_fn =
      warp_decode_gate_up_cute_kernel<
          kGateUpTileN, TILE_K, kGateUpNumWarps>;
  constexpr int smem_bytes = GateUpSmemBytes<TILE_K>();
  MaybeSetSmemAttribute(kernel_fn, smem_bytes);
  dim3 grid((intermediate_size + kGateUpTileN - 1) / kGateUpTileN,
            num_tokens * top_k);
  dim3 block(kGateUpNumThreads);
  WdLaunch(kernel_fn, grid, block, smem_bytes, stream,
           reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
           reinterpret_cast<const __nv_bfloat16*>(w_gate.data_ptr()),
           reinterpret_cast<const __nv_bfloat16*>(w_up.data_ptr()),
           reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
           expert_ids.data_ptr<int>(), hidden_size, intermediate_size, top_k, num_tokens);
}

void warp_decode_cute_gate_up(
    const torch::Tensor& x, const torch::Tensor& w_gate, const torch::Tensor& w_up,
    torch::Tensor& out, const torch::Tensor& expert_ids,
    int hidden_size, int intermediate_size, int top_k, int num_tokens) {
  TORCH_CHECK(top_k == 8, "warp_decode_cute requires top_k=8, got ", top_k);
  warp_decode_cute_gate_up_impl<kGateUpTileK>(
      x, w_gate, w_up, out, expert_ids, hidden_size, intermediate_size,
      top_k, num_tokens);
}

template <int TILE_K>
void warp_decode_cute_gate_up_packed_impl(
    const torch::Tensor& x, const torch::Tensor& w13,
    torch::Tensor& out, const torch::Tensor& expert_ids,
    int hidden_size, int intermediate_size, int top_k, int num_tokens) {
  if (num_tokens == 0) return;
  TORCH_CHECK(x.dtype() == torch::kBFloat16);
  TORCH_CHECK(w13.dtype() == torch::kBFloat16);
  TORCH_CHECK(out.dtype() == torch::kBFloat16);
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto kernel_fn =
      warp_decode_gate_up_packed_cute_kernel<
          kGateUpTileN, TILE_K, kGateUpNumWarps>;
  constexpr int smem_bytes = GateUpSmemBytes<TILE_K>();
  MaybeSetSmemAttribute(kernel_fn, smem_bytes);
  dim3 grid((intermediate_size + kGateUpTileN - 1) / kGateUpTileN,
            num_tokens * top_k);
  dim3 block(kGateUpNumThreads);
  WdLaunch(kernel_fn, grid, block, smem_bytes, stream,
           reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
           reinterpret_cast<const __nv_bfloat16*>(w13.data_ptr()),
           reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
           expert_ids.data_ptr<int>(), hidden_size, intermediate_size, top_k, num_tokens);
}

void warp_decode_cute_gate_up_packed(
    const torch::Tensor& x, const torch::Tensor& w13,
    torch::Tensor& out, const torch::Tensor& expert_ids,
    int hidden_size, int intermediate_size, int top_k, int num_tokens) {
  TORCH_CHECK(top_k == 8, "warp_decode_cute requires top_k=8, got ", top_k);
  warp_decode_cute_gate_up_packed_impl<kGateUpTileK>(
      x, w13, out, expert_ids, hidden_size, intermediate_size, top_k,
      num_tokens);
}

template <int TILE_N>
void warp_decode_cute_down_impl(
    const torch::Tensor& intermediate, const torch::Tensor& w_down,
    const torch::Tensor& routing_weights, const torch::Tensor& expert_ids,
    torch::Tensor& out,
    int hidden_size, int intermediate_size, int top_k, int num_tokens) {
  if (num_tokens == 0) return;
  TORCH_CHECK(intermediate.dtype() == torch::kBFloat16);
  TORCH_CHECK(w_down.dtype() == torch::kBFloat16);
  TORCH_CHECK(routing_weights.dtype() == torch::kFloat32);
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto kernel_fn =
      warp_decode_down_cute_kernel<
          kDownTileD, TILE_N, kDownNumWarps>;
  constexpr int smem_bytes = DownSmemBytes<TILE_N>();
  MaybeSetSmemAttribute(kernel_fn, smem_bytes);
  dim3 grid((hidden_size + kDownTileD - 1) / kDownTileD, num_tokens);
  dim3 block(kDownNumThreads);
  WdLaunch(kernel_fn, grid, block, smem_bytes, stream,
           reinterpret_cast<const __nv_bfloat16*>(intermediate.data_ptr()),
           reinterpret_cast<const __nv_bfloat16*>(w_down.data_ptr()),
           routing_weights.data_ptr<float>(), expert_ids.data_ptr<int>(),
           reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
           hidden_size, intermediate_size, top_k, num_tokens);
}

void warp_decode_cute_down(
    const torch::Tensor& intermediate, const torch::Tensor& w_down,
    const torch::Tensor& routing_weights, const torch::Tensor& expert_ids,
    torch::Tensor& out,
    int hidden_size, int intermediate_size, int top_k, int num_tokens) {
  TORCH_CHECK(top_k == 8, "warp_decode_cute requires top_k=8, got ", top_k);
  warp_decode_cute_down_impl<kDownTileN>(
      intermediate, w_down, routing_weights, expert_ids, out, hidden_size,
      intermediate_size, top_k, num_tokens);
}

void warp_decode_cute_down_fp4(
    const torch::Tensor& intermediate, const torch::Tensor& w_down_packed,
    const torch::Tensor& w_down_scales, const torch::Tensor& w_down_alpha,
    const torch::Tensor& routing_weights, const torch::Tensor& expert_ids,
    torch::Tensor& out,
    int hidden_size, int intermediate_size, int top_k, int num_tokens, int group_size) {
  if (num_tokens == 0) return;
  constexpr int FP4_TILE_D = 32;
  constexpr int FP4_TILE_N = 128;
  constexpr int FP4_NUM_WARPS = 4;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int tile_packed_cols = FP4_TILE_N / 2;
  const int groups_per_tile = (FP4_TILE_N + group_size - 1) / group_size;
  const int smem_fp4 =
      FP4_TILE_N * static_cast<int>(sizeof(__nv_bfloat16)) +
      FP4_TILE_D * tile_packed_cols +
      FP4_TILE_D * groups_per_tile * static_cast<int>(sizeof(__nv_bfloat16));
  MaybeSetSmemAttribute(
      warp_decode_down_fp4_cute_kernel<FP4_TILE_D, FP4_TILE_N, FP4_NUM_WARPS>,
      smem_fp4);
  dim3 grid((hidden_size + FP4_TILE_D - 1) / FP4_TILE_D, num_tokens);
  dim3 block(FP4_NUM_WARPS * 32);
  warp_decode_down_fp4_cute_kernel<FP4_TILE_D, FP4_TILE_N, FP4_NUM_WARPS>
      <<<grid, block, smem_fp4, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(intermediate.data_ptr()),
          w_down_packed.data_ptr<uint8_t>(),
          reinterpret_cast<const __nv_bfloat16*>(w_down_scales.data_ptr()),
          w_down_alpha.data_ptr<float>(), routing_weights.data_ptr<float>(),
          expert_ids.data_ptr<int>(),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), hidden_size,
          intermediate_size, top_k, num_tokens, group_size);
}

torch::Tensor warp_decode_cute_moe(
    const torch::Tensor& hidden_states,
    const torch::Tensor& w_gate,
    const torch::Tensor& w_up,
    const torch::Tensor& w_down,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    bool inplace) {
  TORCH_CHECK(hidden_states.dim() == 2);
  TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16);
  TORCH_CHECK(w_gate.dim() == 3);
  TORCH_CHECK(w_up.dim() == 3);
  TORCH_CHECK(w_down.dim() == 3);
  TORCH_CHECK(w_gate.dtype() == torch::kBFloat16);
  TORCH_CHECK(w_up.dtype() == torch::kBFloat16);
  TORCH_CHECK(w_down.dtype() == torch::kBFloat16);
  const int num_tokens = hidden_states.size(0);
  const int hidden_size = hidden_states.size(1);
  const int intermediate_size = w_gate.size(1);
  const int top_k = topk_ids.size(1);
  if (num_tokens == 0) return inplace ? hidden_states : torch::empty_like(hidden_states);
  TORCH_CHECK(w_gate.size(2) == hidden_size);
  TORCH_CHECK(w_up.sizes() == w_gate.sizes());
  TORCH_CHECK(w_down.size(1) == hidden_size);
  TORCH_CHECK(w_down.size(2) == intermediate_size);
  CheckWarpDecodeCuteShape(hidden_size, intermediate_size);

  auto expert_ids_i32 = topk_ids.to(torch::kInt32);
  auto routing_f32 = topk_weights.to(torch::kFloat32);

  auto intermediate = torch::empty(
      {num_tokens * top_k, intermediate_size},
      torch::TensorOptions().dtype(torch::kBFloat16).device(hidden_states.device()));

  warp_decode_cute_gate_up(hidden_states, w_gate, w_up, intermediate,
                           expert_ids_i32, hidden_size, intermediate_size,
                           top_k, num_tokens);

  torch::Tensor output = inplace ? hidden_states : torch::empty_like(hidden_states);

  warp_decode_cute_down(intermediate, w_down, routing_f32, expert_ids_i32,
                        output, hidden_size, intermediate_size, top_k, num_tokens);

  return output;
}

torch::Tensor warp_decode_cute_moe_packed(
    const torch::Tensor& hidden_states, const torch::Tensor& w13, const torch::Tensor& w2,
    const torch::Tensor& topk_ids, const torch::Tensor& topk_weights,
    int64_t intermediate_size, bool inplace) {
  TORCH_CHECK(hidden_states.dim() == 2);
  TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16);
  TORCH_CHECK(w13.dim() == 3);
  TORCH_CHECK(w2.dim() == 3);
  TORCH_CHECK(w13.dtype() == torch::kBFloat16);
  TORCH_CHECK(w2.dtype() == torch::kBFloat16);
  const int num_tokens = hidden_states.size(0);
  const int hidden_size = hidden_states.size(1);
  if (intermediate_size <= 0) intermediate_size = w13.size(1) / 2;
  const int top_k = topk_ids.size(1);
  if (num_tokens == 0) return inplace ? hidden_states : torch::empty_like(hidden_states);
  TORCH_CHECK(w13.size(1) == 2 * intermediate_size);
  TORCH_CHECK(w13.size(2) == hidden_size);
  TORCH_CHECK(w2.size(1) == hidden_size);
  TORCH_CHECK(w2.size(2) == intermediate_size);
  CheckWarpDecodeCuteShape(hidden_size, static_cast<int>(intermediate_size));

  auto expert_ids_i32 = topk_ids.to(torch::kInt32);
  auto routing_f32 = topk_weights.to(torch::kFloat32);

  auto intermediate = torch::empty(
      {num_tokens * top_k, (int)intermediate_size},
      torch::TensorOptions().dtype(torch::kBFloat16).device(hidden_states.device()));

  warp_decode_cute_gate_up_packed(hidden_states, w13, intermediate,
                                  expert_ids_i32, hidden_size,
                                  static_cast<int>(intermediate_size), top_k, num_tokens);

  torch::Tensor output = inplace ? hidden_states : torch::empty_like(hidden_states);
  warp_decode_cute_down(intermediate, w2, routing_f32, expert_ids_i32, output,
                        hidden_size, static_cast<int>(intermediate_size),
                        top_k, num_tokens);
  return output;
}

}  // namespace warp_decode
}  // namespace sglang

torch::Tensor warp_decode_cute_moe_forward(
    const torch::Tensor& hidden_states,
    const torch::Tensor& w_gate,
    const torch::Tensor& w_up,
    const torch::Tensor& w_down,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    bool inplace) {
  return sglang::warp_decode::warp_decode_cute_moe(
      hidden_states, w_gate, w_up, w_down, topk_ids, topk_weights, inplace);
}

torch::Tensor warp_decode_cute_moe_packed_forward(
    const torch::Tensor& hidden_states,
    const torch::Tensor& w13,
    const torch::Tensor& w2,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    int64_t intermediate_size,
    bool inplace) {
  return sglang::warp_decode::warp_decode_cute_moe_packed(
      hidden_states, w13, w2, topk_ids, topk_weights, intermediate_size, inplace);
}
