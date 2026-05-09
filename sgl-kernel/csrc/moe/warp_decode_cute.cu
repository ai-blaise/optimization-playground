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
 * @brief CuTe-based warp decode MoE kernel launches.
 *
 * Provides torch-facing functions that select tile configurations based
 * on the problem dimensions and SM architecture, then launch the
 * appropriate kernel.
 *
 * Supported entry points:
 *   - warp_decode_cute_gate_up:     BF16, separate gate/up weights
 *   - warp_decode_cute_gate_up_packed: BF16, packed w13 weights
 *   - warp_decode_cute_down:        BF16 down projection + expert combine
 *   - warp_decode_cute_down_fp4:    NVFP4 down projection + expert combine
 *   - warp_decode_cute_moe:         Full MoE forward (gate_up + down)
 *   - warp_decode_cute_moe_packed:  Full MoE forward with packed weights
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include "warp_decode_cute.cuh"

namespace sglang {
namespace warp_decode {

// ---------------------------------------------------------------------------
// Tile configuration selection
// ---------------------------------------------------------------------------
// Select tile sizes based on problem dimensions and GPU capabilities.
// For SM100 (B200, 232KB SMEM), we can use larger tiles.
// For SM80-SM89, use conservative tile sizes.
// ---------------------------------------------------------------------------

struct TileSelection {
  int tile_n;
  int tile_k;
  int tile_d;
  int tile_n_down;
  int num_warps;
  int smem_gate_up;
  int smem_down;
};

static TileSelection select_tiles(int hidden_size, int intermediate_size) {
  TileSelection ts;

  // Default: TILE_N=32, TILE_K=128, NUM_WARPS=4
  // These fit within 32KB SMEM and give good occupancy on SM80+.
  // On SM100 with 232KB SMEM, we could go larger, but for decode
  // (small batch), occupancy matters more than tile size.
  ts.tile_n = 32;
  ts.tile_k = 128;
  ts.tile_d = 32;
  ts.tile_n_down = 128;
  ts.num_warps = 4;

  // Adjust for small dimensions
  if (intermediate_size < 32) {
    ts.tile_n = intermediate_size;
    ts.tile_d = intermediate_size;
  }
  if (hidden_size < 128) {
    ts.tile_k = hidden_size;
  }

  // Compute shared memory requirements
  // Gate/Up: 2 stages * (TILE_K + 2 * TILE_N * TILE_K) * 2 bytes
  ts.smem_gate_up = 2 * (ts.tile_k + 2 * ts.tile_n * ts.tile_k) * sizeof(__nv_bfloat16);

  // Down: 2 stages * (TILE_N_DOWN + TILE_D * TILE_N_DOWN) * 2 bytes
  ts.smem_down = 2 * (ts.tile_n_down + ts.tile_d * ts.tile_n_down) * sizeof(__nv_bfloat16);

  return ts;
}

// ---------------------------------------------------------------------------
// Gate/Up kernel launch (separate weights)
// ---------------------------------------------------------------------------

void warp_decode_cute_gate_up(
    const torch::Tensor& x,           // [num_tokens, hidden_size] bf16
    const torch::Tensor& w_gate,      // [num_experts, intermediate_size, hidden_size] bf16
    const torch::Tensor& w_up,        // [num_experts, intermediate_size, hidden_size] bf16
    torch::Tensor& out,                // [num_tokens * top_k, intermediate_size] bf16
    const torch::Tensor& expert_ids,   // [num_tokens, top_k] int32
    int hidden_size,
    int intermediate_size,
    int top_k,
    int num_tokens) {

  // Edge case: zero tokens is a no-op
  if (num_tokens == 0) return;

  // Dtype checks
  TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(w_gate.dtype() == torch::kBFloat16, "w_gate must be bf16");
  TORCH_CHECK(w_up.dtype() == torch::kBFloat16, "w_up must be bf16");
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32, "expert_ids must be int32");

  // Device checks
  TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
  TORCH_CHECK(w_gate.is_cuda(), "w_gate must be on CUDA");
  TORCH_CHECK(w_up.is_cuda(), "w_up must be on CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be on CUDA");

  // Contiguity checks
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(w_gate.is_contiguous(), "w_gate must be contiguous");
  TORCH_CHECK(w_up.is_contiguous(), "w_up must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(expert_ids.is_contiguous(), "expert_ids must be contiguous");

  // Dimension checks
  TORCH_CHECK(x.ndim() == 2, "x must be 2D [num_tokens, hidden_size]");
  TORCH_CHECK(w_gate.ndim() == 3, "w_gate must be 3D [E, N, K]");
  TORCH_CHECK(w_up.ndim() == 3, "w_up must be 3D [E, N, K]");
  TORCH_CHECK(expert_ids.ndim() == 2, "expert_ids must be 2D [num_tokens, top_k]");

  auto ts = select_tiles(hidden_size, intermediate_size);
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(
      (intermediate_size + ts.tile_n - 1) / ts.tile_n,
      num_tokens * top_k);
  dim3 block(ts.num_warps * 32);

  // Launch with selected tile config
  // We use the default TILE_N=32, TILE_K=128, NUM_WARPS=4 instantiation
  warp_decode_gate_up_cute_kernel<32, 128, 4><<<grid, block, ts.smem_gate_up, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w_gate.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w_up.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      expert_ids.data_ptr<int>(),
      hidden_size, intermediate_size, top_k, num_tokens);
}

// ---------------------------------------------------------------------------
// Gate/Up kernel launch (packed w13 weights)
// ---------------------------------------------------------------------------

void warp_decode_cute_gate_up_packed(
    const torch::Tensor& x,           // [num_tokens, hidden_size] bf16
    const torch::Tensor& w13,         // [num_experts, 2*intermediate_size, hidden_size] bf16
    torch::Tensor& out,                // [num_tokens * top_k, intermediate_size] bf16
    const torch::Tensor& expert_ids,   // [num_tokens, top_k] int32
    int hidden_size,
    int intermediate_size,
    int top_k,
    int num_tokens) {

  if (num_tokens == 0) return;

  TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(w13.dtype() == torch::kBFloat16, "w13 must be bf16");
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(x.is_cuda() && w13.is_cuda() && out.is_cuda() && expert_ids.is_cuda(),
      "all tensors must be on CUDA");
  TORCH_CHECK(x.is_contiguous() && w13.is_contiguous() && out.is_contiguous()
      && expert_ids.is_contiguous(), "all tensors must be contiguous");

  auto ts = select_tiles(hidden_size, intermediate_size);
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(
      (intermediate_size + ts.tile_n - 1) / ts.tile_n,
      num_tokens * top_k);
  dim3 block(ts.num_warps * 32);

  warp_decode_gate_up_packed_cute_kernel<32, 128, 4><<<grid, block, ts.smem_gate_up, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w13.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      expert_ids.data_ptr<int>(),
      hidden_size, intermediate_size, top_k, num_tokens);
}

// ---------------------------------------------------------------------------
// Down projection kernel launch (BF16)
// ---------------------------------------------------------------------------

void warp_decode_cute_down(
    const torch::Tensor& intermediate,      // [num_tokens * top_k, intermediate_size] bf16
    const torch::Tensor& w_down,             // [num_experts, hidden_size, intermediate_size] bf16
    const torch::Tensor& routing_weights,    // [num_tokens, top_k] float32
    const torch::Tensor& expert_ids,         // [num_tokens, top_k] int32
    torch::Tensor& out,                       // [num_tokens, hidden_size] bf16
    int hidden_size,
    int intermediate_size,
    int top_k,
    int num_tokens) {

  if (num_tokens == 0) return;

  TORCH_CHECK(intermediate.dtype() == torch::kBFloat16, "intermediate must be bf16");
  TORCH_CHECK(w_down.dtype() == torch::kBFloat16, "w_down must be bf16");
  TORCH_CHECK(routing_weights.dtype() == torch::kFloat32, "routing_weights must be float32");
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(intermediate.is_cuda() && w_down.is_cuda() && routing_weights.is_cuda()
      && expert_ids.is_cuda() && out.is_cuda(), "all tensors must be on CUDA");
  TORCH_CHECK(intermediate.is_contiguous() && w_down.is_contiguous()
      && routing_weights.is_contiguous() && expert_ids.is_contiguous()
      && out.is_contiguous(), "all tensors must be contiguous");

  auto ts = select_tiles(hidden_size, intermediate_size);
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(
      (hidden_size + ts.tile_d - 1) / ts.tile_d,
      num_tokens);
  dim3 block(ts.num_warps * 32);

  warp_decode_down_cute_kernel<32, 128, 4><<<grid, block, ts.smem_down, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(intermediate.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w_down.data_ptr()),
      routing_weights.data_ptr<float>(),
      expert_ids.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      hidden_size, intermediate_size, top_k, num_tokens);
}

// ---------------------------------------------------------------------------
// Down projection kernel launch (NVFP4)
// ---------------------------------------------------------------------------

void warp_decode_cute_down_fp4(
    const torch::Tensor& intermediate,
    const torch::Tensor& w_down_packed,     // [E, D, N/2] uint8
    const torch::Tensor& w_down_scales,     // [E, D, N/group_size] bf16
    const torch::Tensor& w_down_alpha,      // [1] float32
    const torch::Tensor& routing_weights,
    const torch::Tensor& expert_ids,
    torch::Tensor& out,
    int hidden_size,
    int intermediate_size,
    int top_k,
    int num_tokens,
    int group_size) {

  if (num_tokens == 0) return;

  TORCH_CHECK(intermediate.dtype() == torch::kBFloat16, "intermediate must be bf16");
  TORCH_CHECK(w_down_packed.dtype() == torch::kUInt8, "w_down_packed must be uint8");
  TORCH_CHECK(w_down_scales.dtype() == torch::kBFloat16, "w_down_scales must be bf16");
  TORCH_CHECK(w_down_alpha.dtype() == torch::kFloat32, "w_down_alpha must be float32");
  TORCH_CHECK(routing_weights.dtype() == torch::kFloat32, "routing_weights must be float32");
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(group_size > 0 && (group_size & (group_size - 1)) == 0,
              "group_size must be a positive power of 2");
  TORCH_CHECK(intermediate_size % 2 == 0,
              "intermediate_size must be even for NVFP4 packing");

  auto ts = select_tiles(hidden_size, intermediate_size);
  auto stream = at::cuda::getCurrentCUDAStream();

  // Compute shared memory for FP4 variant
  int groups_per_tile = (ts.tile_n_down + group_size - 1) / group_size;
  int smem_fp4 = ts.tile_n_down * sizeof(__nv_bfloat16)  // intermediate
      + ts.tile_d * (ts.tile_n_down / 2)                  // packed weights
      + ts.tile_d * groups_per_tile * sizeof(__nv_bfloat16); // scales

  dim3 grid(
      (hidden_size + ts.tile_d - 1) / ts.tile_d,
      num_tokens);
  dim3 block(ts.num_warps * 32);

  warp_decode_down_fp4_cute_kernel<32, 128, 4><<<grid, block, smem_fp4, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(intermediate.data_ptr()),
      w_down_packed.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(w_down_scales.data_ptr()),
      w_down_alpha.data_ptr<float>(),
      routing_weights.data_ptr<float>(),
      expert_ids.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      hidden_size, intermediate_size, top_k, num_tokens, group_size);
}

// ---------------------------------------------------------------------------
// Full MoE forward: gate_up + down (separate weights, BF16)
// ---------------------------------------------------------------------------

torch::Tensor warp_decode_cute_moe(
    const torch::Tensor& hidden_states,  // [num_tokens, hidden_size] bf16
    const torch::Tensor& w_gate,          // [E, N, D] bf16
    const torch::Tensor& w_up,            // [E, N, D] bf16
    const torch::Tensor& w_down,          // [E, D, N] bf16
    const torch::Tensor& topk_ids,        // [num_tokens, top_k] int32
    const torch::Tensor& topk_weights,    // [num_tokens, top_k] float32
    bool inplace) {

  TORCH_CHECK(hidden_states.ndim() == 2, "hidden_states must be 2D");
  TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16);

  const int num_tokens = hidden_states.size(0);
  const int hidden_size = hidden_states.size(1);
  const int intermediate_size = w_gate.size(1);
  const int top_k = topk_ids.size(1);

  // Ensure expert_ids are int32
  auto expert_ids_i32 = topk_ids.to(torch::kInt32);

  // Allocate intermediate buffer
  auto intermediate = torch::empty(
      {num_tokens * top_k, intermediate_size},
      torch::TensorOptions().dtype(torch::kBFloat16).device(hidden_states.device()));

  // Gate/Up kernel
  warp_decode_cute_gate_up(
      hidden_states, w_gate, w_up, intermediate, expert_ids_i32,
      hidden_size, intermediate_size, top_k, num_tokens);

  // Output buffer
  torch::Tensor output;
  if (inplace) {
    output = hidden_states;
  } else {
    output = torch::empty_like(hidden_states);
  }

  // Down kernel
  warp_decode_cute_down(
      intermediate, w_down, topk_weights, expert_ids_i32, output,
      hidden_size, intermediate_size, top_k, num_tokens);

  return output;
}

// ---------------------------------------------------------------------------
// Full MoE forward: gate_up_packed + down (packed w13/w2, BF16)
// ---------------------------------------------------------------------------

torch::Tensor warp_decode_cute_moe_packed(
    const torch::Tensor& hidden_states,
    const torch::Tensor& w13,             // [E, 2*N, D] bf16
    const torch::Tensor& w2,              // [E, D, N] bf16
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    int64_t intermediate_size,
    bool inplace) {

  TORCH_CHECK(hidden_states.ndim() == 2, "hidden_states must be 2D");
  TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16);
  TORCH_CHECK(w13.dtype() == torch::kBFloat16);
  TORCH_CHECK(w2.dtype() == torch::kBFloat16);

  const int num_tokens = hidden_states.size(0);
  const int hidden_size = hidden_states.size(1);
  if (intermediate_size <= 0) {
    intermediate_size = w13.size(1) / 2;
  }
  const int top_k = topk_ids.size(1);

  auto expert_ids_i32 = topk_ids.to(torch::kInt32);

  auto intermediate = torch::empty(
      {num_tokens * top_k, intermediate_size},
      torch::TensorOptions().dtype(torch::kBFloat16).device(hidden_states.device()));

  // Gate/Up with packed weights
  warp_decode_cute_gate_up_packed(
      hidden_states, w13, intermediate, expert_ids_i32,
      hidden_size, static_cast<int>(intermediate_size), top_k, num_tokens);

  torch::Tensor output;
  if (inplace) {
    output = hidden_states;
  } else {
    output = torch::empty_like(hidden_states);
  }

  // Down + combine
  warp_decode_cute_down(
      intermediate, w2, topk_weights, expert_ids_i32, output,
      hidden_size, static_cast<int>(intermediate_size), top_k, num_tokens);

  return output;
}

}  // namespace warp_decode
}  // namespace sglang

// ---------------------------------------------------------------------------
// Torch binding entry points (C-linkage for registration)
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
      hidden_states, w13, w2, topk_ids, topk_weights, intermediate_size, inplace);
}
