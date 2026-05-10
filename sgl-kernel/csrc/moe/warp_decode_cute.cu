// Copyright 2024-2026 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
/**
 * @file warp_decode_cute.cu
 * @brief CuTe-based warp decode MoE kernel launches and torch bindings.
 *
 * Provides torch-facing functions that select tile configurations, set
 * dynamic SMEM limits, and launch the appropriate kernel from
 * warp_decode_cute.cuh.
 *
 * Entry points:
 *   - warp_decode_cute_moe_forward:        Separate gate/up/down weights
 *   - warp_decode_cute_moe_packed_forward: Packed w13/w2 weights
 *
 * Individual kernel launches (for testing/benchmarking):
 *   - warp_decode_cute_gate_up
 *   - warp_decode_cute_gate_up_packed
 *   - warp_decode_cute_down
 *   - warp_decode_cute_down_fp4
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "warp_decode_cute.cuh"

namespace sglang {
namespace warp_decode {

// ---------------------------------------------------------------------------
// Tile configuration constants
// ---------------------------------------------------------------------------
// These match the template instantiations below. We hardcode TILE_N=32,
// TILE_K=128, NUM_WARPS=4 for the gate/up kernel, and TILE_D=32,
// TILE_N_DOWN=128, NUM_WARPS=4 for the down kernel. These sizes give good
// SM occupancy for decode batch sizes 1-64 on both SM80 and SM100.
// ---------------------------------------------------------------------------

constexpr int kGateUpTileN = 32;
constexpr int kGateUpTileK = 128;
constexpr int kGateUpNumWarps = 4;
constexpr int kGateUpNumThreads = kGateUpNumWarps * 32;

constexpr int kDownTileD = 32;
constexpr int kDownTileN = 128;
constexpr int kDownNumWarps = 4;
constexpr int kDownNumThreads = kDownNumWarps * 32;

// Shared memory sizes (must match GateUpTileConfig/DownTileConfig in .cuh).
constexpr int kGateUpSmemBytes =
    2 * (kGateUpTileK + 2 * kGateUpTileN * kGateUpTileK) *
    static_cast<int>(sizeof(__nv_bfloat16));

constexpr int kDownSmemBytes =
    2 * (kDownTileN + kDownTileD * kDownTileN) *
    static_cast<int>(sizeof(__nv_bfloat16));

// ---------------------------------------------------------------------------
// SMEM attribute helper
// ---------------------------------------------------------------------------
// When the kernel requires more than 48KB dynamic shared memory, we must
// call cudaFuncSetAttribute to raise the limit. This is a no-op when the
// requested size is within the default 48KB limit.
// ---------------------------------------------------------------------------

template <typename KernelFunc>
void MaybeSetSmemAttribute(KernelFunc kernel, int smem_bytes) {
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }
}

// ---------------------------------------------------------------------------
// Gate/Up kernel launch (separate gate/up weights)
// ---------------------------------------------------------------------------

void warp_decode_cute_gate_up(
    const torch::Tensor& x,
    const torch::Tensor& w_gate,
    const torch::Tensor& w_up,
    torch::Tensor& out,
    const torch::Tensor& expert_ids,
    int hidden_size,
    int intermediate_size,
    int top_k,
    int num_tokens) {
  if (num_tokens == 0) return;

  // Dtype checks.
  TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(w_gate.dtype() == torch::kBFloat16, "w_gate must be bf16");
  TORCH_CHECK(w_up.dtype() == torch::kBFloat16, "w_up must be bf16");
  TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be bf16");
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32, "expert_ids must be int32");

  // Device checks.
  TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
  TORCH_CHECK(w_gate.is_cuda(), "w_gate must be on CUDA");
  TORCH_CHECK(w_up.is_cuda(), "w_up must be on CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be on CUDA");

  // Contiguity checks.
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(w_gate.is_contiguous(), "w_gate must be contiguous");
  TORCH_CHECK(w_up.is_contiguous(), "w_up must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(expert_ids.is_contiguous(), "expert_ids must be contiguous");

  // Dimension checks.
  TORCH_CHECK(x.ndim() == 2, "x must be 2D [num_tokens, hidden_size]");
  TORCH_CHECK(w_gate.ndim() == 3, "w_gate must be 3D [E, N, K]");
  TORCH_CHECK(w_up.ndim() == 3, "w_up must be 3D [E, N, K]");
  TORCH_CHECK(expert_ids.ndim() == 2, "expert_ids must be 2D [T, top_k]");

  auto stream = at::cuda::getCurrentCUDAStream();

  MaybeSetSmemAttribute(
      warp_decode_gate_up_cute_kernel<kGateUpTileN, kGateUpTileK,
                                      kGateUpNumWarps>,
      kGateUpSmemBytes);

  dim3 grid((intermediate_size + kGateUpTileN - 1) / kGateUpTileN,
            num_tokens * top_k);
  dim3 block(kGateUpNumThreads);

  warp_decode_gate_up_cute_kernel<kGateUpTileN, kGateUpTileK,
                                  kGateUpNumWarps>
      <<<grid, block, kGateUpSmemBytes, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(w_gate.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(w_up.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
          expert_ids.data_ptr<int>(), hidden_size, intermediate_size, top_k,
          num_tokens);
}

// ---------------------------------------------------------------------------
// Gate/Up kernel launch (packed w13 weights)
// ---------------------------------------------------------------------------

void warp_decode_cute_gate_up_packed(
    const torch::Tensor& x,
    const torch::Tensor& w13,
    torch::Tensor& out,
    const torch::Tensor& expert_ids,
    int hidden_size,
    int intermediate_size,
    int top_k,
    int num_tokens) {
  if (num_tokens == 0) return;

  TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(w13.dtype() == torch::kBFloat16, "w13 must be bf16");
  TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be bf16");
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(x.is_cuda() && w13.is_cuda() && out.is_cuda() &&
                  expert_ids.is_cuda(),
              "all tensors must be on CUDA");
  TORCH_CHECK(x.is_contiguous() && w13.is_contiguous() &&
                  out.is_contiguous() && expert_ids.is_contiguous(),
              "all tensors must be contiguous");

  auto stream = at::cuda::getCurrentCUDAStream();

  MaybeSetSmemAttribute(
      warp_decode_gate_up_packed_cute_kernel<kGateUpTileN, kGateUpTileK,
                                             kGateUpNumWarps>,
      kGateUpSmemBytes);

  dim3 grid((intermediate_size + kGateUpTileN - 1) / kGateUpTileN,
            num_tokens * top_k);
  dim3 block(kGateUpNumThreads);

  warp_decode_gate_up_packed_cute_kernel<kGateUpTileN, kGateUpTileK,
                                         kGateUpNumWarps>
      <<<grid, block, kGateUpSmemBytes, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(w13.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
          expert_ids.data_ptr<int>(), hidden_size, intermediate_size, top_k,
          num_tokens);
}

// ---------------------------------------------------------------------------
// Down projection kernel launch (BF16)
// ---------------------------------------------------------------------------

void warp_decode_cute_down(
    const torch::Tensor& intermediate,
    const torch::Tensor& w_down,
    const torch::Tensor& routing_weights,
    const torch::Tensor& expert_ids,
    torch::Tensor& out,
    int hidden_size,
    int intermediate_size,
    int top_k,
    int num_tokens) {
  if (num_tokens == 0) return;

  TORCH_CHECK(intermediate.dtype() == torch::kBFloat16,
              "intermediate must be bf16");
  TORCH_CHECK(w_down.dtype() == torch::kBFloat16, "w_down must be bf16");
  TORCH_CHECK(routing_weights.dtype() == torch::kFloat32,
              "routing_weights must be float32");
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(
      intermediate.is_cuda() && w_down.is_cuda() &&
          routing_weights.is_cuda() && expert_ids.is_cuda() && out.is_cuda(),
      "all tensors must be on CUDA");
  TORCH_CHECK(intermediate.is_contiguous() && w_down.is_contiguous() &&
                  routing_weights.is_contiguous() &&
                  expert_ids.is_contiguous() && out.is_contiguous(),
              "all tensors must be contiguous");

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
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), hidden_size,
          intermediate_size, top_k, num_tokens);
}

// ---------------------------------------------------------------------------
// Down projection kernel launch (NVFP4)
// ---------------------------------------------------------------------------

void warp_decode_cute_down_fp4(
    const torch::Tensor& intermediate,
    const torch::Tensor& w_down_packed,
    const torch::Tensor& w_down_scales,
    const torch::Tensor& w_down_alpha,
    const torch::Tensor& routing_weights,
    const torch::Tensor& expert_ids,
    torch::Tensor& out,
    int hidden_size,
    int intermediate_size,
    int top_k,
    int num_tokens,
    int group_size) {
  if (num_tokens == 0) return;

  TORCH_CHECK(intermediate.dtype() == torch::kBFloat16,
              "intermediate must be bf16");
  TORCH_CHECK(w_down_packed.dtype() == torch::kUInt8,
              "w_down_packed must be uint8");
  TORCH_CHECK(w_down_scales.dtype() == torch::kBFloat16,
              "w_down_scales must be bf16");
  TORCH_CHECK(w_down_alpha.dtype() == torch::kFloat32,
              "w_down_alpha must be float32");
  TORCH_CHECK(routing_weights.dtype() == torch::kFloat32,
              "routing_weights must be float32");
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(group_size > 0 && (group_size & (group_size - 1)) == 0,
              "group_size must be a positive power of 2");
  TORCH_CHECK(intermediate_size % 2 == 0,
              "intermediate_size must be even for NVFP4 packing");

  auto stream = at::cuda::getCurrentCUDAStream();

  // Compute dynamic SMEM for FP4 variant.
  const int tile_packed_cols = kDownTileN / 2;
  const int groups_per_tile = (kDownTileN + group_size - 1) / group_size;
  const int smem_fp4 =
      kDownTileN * static_cast<int>(sizeof(__nv_bfloat16)) +  // intermediate
      kDownTileD * tile_packed_cols +                          // packed weights
      kDownTileD * groups_per_tile *
          static_cast<int>(sizeof(__nv_bfloat16));  // scales

  MaybeSetSmemAttribute(
      warp_decode_down_fp4_cute_kernel<kDownTileD, kDownTileN, kDownNumWarps>,
      smem_fp4);

  dim3 grid((hidden_size + kDownTileD - 1) / kDownTileD, num_tokens);
  dim3 block(kDownNumThreads);

  warp_decode_down_fp4_cute_kernel<kDownTileD, kDownTileN, kDownNumWarps>
      <<<grid, block, smem_fp4, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(intermediate.data_ptr()),
          w_down_packed.data_ptr<uint8_t>(),
          reinterpret_cast<const __nv_bfloat16*>(w_down_scales.data_ptr()),
          w_down_alpha.data_ptr<float>(), routing_weights.data_ptr<float>(),
          expert_ids.data_ptr<int>(),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), hidden_size,
          intermediate_size, top_k, num_tokens, group_size);
}

// ---------------------------------------------------------------------------
// Full MoE forward: gate_up + down (separate weights, BF16)
// ---------------------------------------------------------------------------

torch::Tensor warp_decode_cute_moe(
    const torch::Tensor& hidden_states,
    const torch::Tensor& w_gate,
    const torch::Tensor& w_up,
    const torch::Tensor& w_down,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    bool inplace) {
  TORCH_CHECK(hidden_states.ndim() == 2, "hidden_states must be 2D");
  TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16);

  const int num_tokens = hidden_states.size(0);
  const int hidden_size = hidden_states.size(1);
  const int intermediate_size = w_gate.size(1);
  const int top_k = topk_ids.size(1);

  if (num_tokens == 0) {
    return inplace ? hidden_states : torch::empty_like(hidden_states);
  }

  // Ensure correct dtypes for kernel expectations.
  auto expert_ids_i32 = topk_ids.to(torch::kInt32);
  auto routing_f32 = topk_weights.to(torch::kFloat32);

  // Allocate intermediate buffer.
  auto intermediate = torch::empty(
      {num_tokens * top_k, intermediate_size},
      torch::TensorOptions()
          .dtype(torch::kBFloat16)
          .device(hidden_states.device()));

  // Gate/Up kernel.
  warp_decode_cute_gate_up(hidden_states, w_gate, w_up, intermediate,
                           expert_ids_i32, hidden_size, intermediate_size,
                           top_k, num_tokens);

  // Output buffer.
  torch::Tensor output;
  if (inplace) {
    output = hidden_states;
  } else {
    output = torch::empty_like(hidden_states);
  }

  // Down kernel.
  warp_decode_cute_down(intermediate, w_down, routing_f32, expert_ids_i32,
                        output, hidden_size, intermediate_size, top_k,
                        num_tokens);

  return output;
}

// ---------------------------------------------------------------------------
// Full MoE forward: gate_up_packed + down (packed w13/w2, BF16)
// ---------------------------------------------------------------------------

torch::Tensor warp_decode_cute_moe_packed(
    const torch::Tensor& hidden_states,
    const torch::Tensor& w13,
    const torch::Tensor& w2,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    int64_t intermediate_size,
    bool inplace) {
  TORCH_CHECK(hidden_states.ndim() == 2, "hidden_states must be 2D");
  TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16);
  TORCH_CHECK(w13.dtype() == torch::kBFloat16, "w13 must be bf16");
  TORCH_CHECK(w2.dtype() == torch::kBFloat16, "w2 must be bf16");

  const int num_tokens = hidden_states.size(0);
  const int hidden_size = hidden_states.size(1);
  if (intermediate_size <= 0) {
    intermediate_size = w13.size(1) / 2;
  }
  const int top_k = topk_ids.size(1);

  if (num_tokens == 0) {
    return inplace ? hidden_states : torch::empty_like(hidden_states);
  }

  auto expert_ids_i32 = topk_ids.to(torch::kInt32);
  auto routing_f32 = topk_weights.to(torch::kFloat32);

  auto intermediate = torch::empty(
      {num_tokens * top_k, intermediate_size},
      torch::TensorOptions()
          .dtype(torch::kBFloat16)
          .device(hidden_states.device()));

  // Gate/Up with packed weights.
  warp_decode_cute_gate_up_packed(hidden_states, w13, intermediate,
                                  expert_ids_i32, hidden_size,
                                  static_cast<int>(intermediate_size), top_k,
                                  num_tokens);

  torch::Tensor output;
  if (inplace) {
    output = hidden_states;
  } else {
    output = torch::empty_like(hidden_states);
  }

  // Down + combine.
  warp_decode_cute_down(intermediate, w2, routing_f32, expert_ids_i32, output,
                        hidden_size, static_cast<int>(intermediate_size),
                        top_k, num_tokens);

  return output;
}

}  // namespace warp_decode
}  // namespace sglang

// ---------------------------------------------------------------------------
// Torch binding entry points (C-linkage for sgl_kernel registration)
// ---------------------------------------------------------------------------

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
      hidden_states, w13, w2, topk_ids, topk_weights, intermediate_size,
      inplace);
}
