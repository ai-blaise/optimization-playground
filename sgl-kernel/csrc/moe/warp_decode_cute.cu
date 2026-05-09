// Copyright 2024-2026 SGLang Team
// Licensed under the Apache License, Version 2.0.
// ==============================================================================
// BLOG-STRICT warp_decode_cute dispatch:
//   - Gate/Up: TILE_N=8 (one neuron per warp), TILE_K=128, NUM_WARPS=8
//   - Down:    TILE_D=8 (one dim per warp),    TILE_N=512, NUM_WARPS=8

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "warp_decode_cute.cuh"

namespace sglang {
namespace warp_decode {

// OPT3: opt2 large tiles + 3-stage cp.async pipeline.
constexpr int kGateUpTileN = 8;
constexpr int kGateUpTileK = 256;
constexpr int kGateUpNumWarps = 8;
constexpr int kGateUpNumThreads = kGateUpNumWarps * 32;

constexpr int kDownTileD = 8;
constexpr int kDownTileN = 1024;
constexpr int kDownNumWarps = 8;
constexpr int kDownNumThreads = kDownNumWarps * 32;

// 3 stages instead of 2 — deeper pipeline to better hide DRAM latency.
constexpr int kGateUpSmemBytes =
    3 * (kGateUpTileK + 2 * kGateUpTileN * kGateUpTileK) *
    static_cast<int>(sizeof(__nv_bfloat16));

constexpr int kDownSmemBytes =
    3 * (kDownTileN + kDownTileD * kDownTileN) *
    static_cast<int>(sizeof(__nv_bfloat16));

template <typename KernelFunc>
void MaybeSetSmemAttribute(KernelFunc kernel, int smem_bytes) {
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }
}

void warp_decode_cute_gate_up(
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
  MaybeSetSmemAttribute(
      warp_decode_gate_up_cute_kernel<kGateUpTileN, kGateUpTileK, kGateUpNumWarps>,
      kGateUpSmemBytes);
  dim3 grid((intermediate_size + kGateUpTileN - 1) / kGateUpTileN,
            num_tokens * top_k);
  dim3 block(kGateUpNumThreads);
  warp_decode_gate_up_cute_kernel<kGateUpTileN, kGateUpTileK, kGateUpNumWarps>
      <<<grid, block, kGateUpSmemBytes, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(w_gate.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(w_up.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
          expert_ids.data_ptr<int>(), hidden_size, intermediate_size, top_k, num_tokens);
}

void warp_decode_cute_gate_up_packed(
    const torch::Tensor& x, const torch::Tensor& w13,
    torch::Tensor& out, const torch::Tensor& expert_ids,
    int hidden_size, int intermediate_size, int top_k, int num_tokens) {
  if (num_tokens == 0) return;
  TORCH_CHECK(x.dtype() == torch::kBFloat16);
  TORCH_CHECK(w13.dtype() == torch::kBFloat16);
  TORCH_CHECK(out.dtype() == torch::kBFloat16);
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32);
  auto stream = at::cuda::getCurrentCUDAStream();
  MaybeSetSmemAttribute(
      warp_decode_gate_up_packed_cute_kernel<kGateUpTileN, kGateUpTileK, kGateUpNumWarps>,
      kGateUpSmemBytes);
  dim3 grid((intermediate_size + kGateUpTileN - 1) / kGateUpTileN,
            num_tokens * top_k);
  dim3 block(kGateUpNumThreads);
  warp_decode_gate_up_packed_cute_kernel<kGateUpTileN, kGateUpTileK, kGateUpNumWarps>
      <<<grid, block, kGateUpSmemBytes, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(w13.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
          expert_ids.data_ptr<int>(), hidden_size, intermediate_size, top_k, num_tokens);
}

void warp_decode_cute_down(
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
  MaybeSetSmemAttribute(
      warp_decode_down_cute_kernel<kDownTileD, kDownTileN, kDownNumWarps>,
      kDownSmemBytes);
  dim3 grid((hidden_size + kDownTileD - 1) / kDownTileD, num_tokens);
  dim3 block(kDownNumThreads);
  warp_decode_down_cute_kernel<kDownTileD, kDownTileN, kDownNumWarps>
      <<<grid, block, kDownSmemBytes, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(intermediate.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(w_down.data_ptr()),
          routing_weights.data_ptr<float>(), expert_ids.data_ptr<int>(),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
          hidden_size, intermediate_size, top_k, num_tokens);
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
    const torch::Tensor& hidden_states, const torch::Tensor& w_gate, const torch::Tensor& w_up, const torch::Tensor& w_down,
    const torch::Tensor& topk_ids, const torch::Tensor& topk_weights, bool inplace) {
  TORCH_CHECK(hidden_states.dim() == 2);
  TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16);
  const int num_tokens = hidden_states.size(0);
  const int hidden_size = hidden_states.size(1);
  const int intermediate_size = w_gate.size(1);
  const int top_k = topk_ids.size(1);
  if (num_tokens == 0) return inplace ? hidden_states : torch::empty_like(hidden_states);

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
  const int num_tokens = hidden_states.size(0);
  const int hidden_size = hidden_states.size(1);
  if (intermediate_size <= 0) intermediate_size = w13.size(1) / 2;
  const int top_k = topk_ids.size(1);
  if (num_tokens == 0) return inplace ? hidden_states : torch::empty_like(hidden_states);

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
    const torch::Tensor& hidden_states, const torch::Tensor& w_gate, const torch::Tensor& w_up,
    const torch::Tensor& w_down, const torch::Tensor& topk_ids, const torch::Tensor& topk_weights, bool inplace) {
  return sglang::warp_decode::warp_decode_cute_moe(
      hidden_states, w_gate, w_up, w_down, topk_ids, topk_weights, inplace);
}

torch::Tensor warp_decode_cute_moe_packed_forward(
    const torch::Tensor& hidden_states, const torch::Tensor& w13, const torch::Tensor& w2,
    const torch::Tensor& topk_ids, const torch::Tensor& topk_weights, int64_t intermediate_size, bool inplace) {
  return sglang::warp_decode::warp_decode_cute_moe_packed(
      hidden_states, w13, w2, topk_ids, topk_weights, intermediate_size, inplace);
}
