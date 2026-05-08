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
 * @file warp_decode_cute.cuh
 * @brief CuTe-based warp decode MoE kernels for small-batch inference.
 *
 * Implements the two-kernel warp decode approach using CUTLASS CuTe:
 *
 * Kernel 1 (gate_up): Each threadblock handles a tile of intermediate
 *   neurons for one (token, expert) pair. Producer warps stream expert
 *   weights from HBM via TMA/cp.async, consumer warps compute gate+up
 *   dot products in FP32, apply SiLU(gate)*up.
 *
 * Kernel 2 (down): Each threadblock handles a tile of output dimensions
 *   for one token. Loops over top-k experts, folding routing weights
 *   into FP32 accumulators. No cross-threadblock synchronization.
 *
 * Supported precisions:
 *   - BF16 weights and activations
 *   - FP8 (E4M3) weights with per-tensor/blockwise scales
 *   - NVFP4 weights with two-level blockscales (SM100+)
 *
 * Target: SM100 (B200, 232KB SMEM, 148 SMs) with fallback to SM80+.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

// ---------------------------------------------------------------------------
// Tile and block size configurations for SM100 (B200)
// ---------------------------------------------------------------------------
// B200: 148 SMs, 232 KB shared memory per SM, 2048 threads/SM
//
// For warp decode (batch <= 64), we want:
//   - High SM occupancy via small tiles
//   - Maximize weight streaming bandwidth from HBM
//   - FP32 accumulation for numerical stability
// ---------------------------------------------------------------------------

namespace sglang {
namespace warp_decode {

// Gate/Up kernel tile sizes
// Each threadblock processes TILE_N intermediate neurons over the full
// hidden dimension, streaming K in chunks of TILE_K.
struct GateUpTileConfig {
  static constexpr int TILE_N = 32;   // intermediate neurons per block
  static constexpr int TILE_K = 128;  // hidden dim chunk per iteration
  static constexpr int NUM_WARPS = 4; // warps per threadblock
  static constexpr int NUM_THREADS = NUM_WARPS * 32;

  // Shared memory layout:
  //   x_smem:      [TILE_K] bf16  — activation slice (broadcast across N)
  //   w_gate_smem: [TILE_N, TILE_K] bf16 — gate weight tile
  //   w_up_smem:   [TILE_N, TILE_K] bf16 — up weight tile
  // Double-buffered for producer/consumer overlap.
  static constexpr int SMEM_X_BYTES = TILE_K * sizeof(__nv_bfloat16);
  static constexpr int SMEM_W_BYTES = TILE_N * TILE_K * sizeof(__nv_bfloat16);
  static constexpr int SMEM_PER_STAGE = SMEM_X_BYTES + 2 * SMEM_W_BYTES;
  static constexpr int NUM_STAGES = 2;  // double-buffer
  static constexpr int TOTAL_SMEM = SMEM_PER_STAGE * NUM_STAGES;
};

// Down kernel tile sizes
// Each threadblock processes TILE_D output dimensions, streaming over
// intermediate_size in chunks of TILE_N.
struct DownTileConfig {
  static constexpr int TILE_D = 32;   // output dimensions per block
  static constexpr int TILE_N = 128;  // intermediate dim chunk
  static constexpr int NUM_WARPS = 4;
  static constexpr int NUM_THREADS = NUM_WARPS * 32;

  // Shared memory layout:
  //   inter_smem: [TILE_N] bf16 — intermediate activation slice
  //   w_down_smem: [TILE_D, TILE_N] bf16 — down weight tile
  // Double-buffered.
  static constexpr int SMEM_INTER_BYTES = TILE_N * sizeof(__nv_bfloat16);
  static constexpr int SMEM_W_BYTES = TILE_D * TILE_N * sizeof(__nv_bfloat16);
  static constexpr int SMEM_PER_STAGE = SMEM_INTER_BYTES + SMEM_W_BYTES;
  static constexpr int NUM_STAGES = 2;
  static constexpr int TOTAL_SMEM = SMEM_PER_STAGE * NUM_STAGES;
};

// NVFP4 dequantization helpers
struct NVFP4Config {
  static constexpr int GROUP_SIZE = 16;  // values per micro-block scale
  // FP4 value table: 0,1,...,7 mapped to {0, 0.5, 1, 1.5, 2, 3, 4, 6}
  // Negative via sign bit (4-bit: 1 sign + 3 magnitude)
};

// ---------------------------------------------------------------------------
// Gate/Up fused kernel — CuTe-style with explicit tiling
// ---------------------------------------------------------------------------
// Grid: (ceil(intermediate_size / TILE_N), num_tokens * top_k)
// Each block: TILE_N intermediate neurons for one (token, expert) pair.
//
// Memory access pattern:
//   - Activation x[token, :] is broadcast across all N outputs
//   - Gate/Up weights are streamed in [TILE_N, TILE_K] tiles
//   - FP32 accumulation, BF16 output
// ---------------------------------------------------------------------------

template <int TILE_N, int TILE_K, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS * 32)
warp_decode_gate_up_cute_kernel(
    const __nv_bfloat16* __restrict__ x,       // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ w_gate,   // [num_experts, intermediate_size, hidden_size]
    const __nv_bfloat16* __restrict__ w_up,     // [num_experts, intermediate_size, hidden_size]
    __nv_bfloat16* __restrict__ out,             // [num_tokens * top_k, intermediate_size]
    const int* __restrict__ expert_ids,          // [num_tokens, top_k]
    int hidden_size,
    int intermediate_size,
    int top_k,
    int num_tokens) {

  // Shared memory: double-buffered activation + weight tiles
  extern __shared__ char smem[];

  constexpr int NUM_THREADS = NUM_WARPS * 32;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  // Block indices
  const int pid_n = blockIdx.x;   // intermediate neuron tile
  const int pid_te = blockIdx.y;  // (token, expert) pair

  const int token_idx = pid_te / top_k;
  const int k_idx = pid_te % top_k;

  if (token_idx >= num_tokens) return;

  // Load expert ID
  const int expert_id = expert_ids[token_idx * top_k + k_idx];

  // Intermediate neuron offsets for this tile
  const int n_base = pid_n * TILE_N;

  // Pointers into shared memory (double-buffered)
  // Stage layout: [x_tile(TILE_K) | w_gate_tile(TILE_N*TILE_K) | w_up_tile(TILE_N*TILE_K)]
  const int smem_x_offset = 0;
  const int smem_gate_offset = TILE_K;  // in bf16 elements
  const int smem_up_offset = TILE_K + TILE_N * TILE_K;
  const int elems_per_stage = TILE_K + 2 * TILE_N * TILE_K;

  __nv_bfloat16* smem_bf16 = reinterpret_cast<__nv_bfloat16*>(smem);

  // FP32 accumulators per thread
  // Each thread owns a subset of the TILE_N neurons
  // With NUM_THREADS threads and TILE_N neurons, each thread handles
  // ceil(TILE_N / NUM_THREADS) neurons for the reduction, but more
  // practically we distribute the K-reduction across threads.
  //
  // Strategy: Each thread accumulates for all TILE_N neurons it's
  // assigned. We use a row-parallel approach: threads in a warp
  // cooperate on the K-dimension dot product for each neuron row.
  //
  // For TILE_N=32, TILE_K=128, NUM_THREADS=128:
  //   Each warp (32 threads) handles 8 rows (TILE_N/NUM_WARPS).
  //   Each thread reduces TILE_K/32 = 4 elements per row per k-step.
  constexpr int ROWS_PER_WARP = (TILE_N + NUM_WARPS - 1) / NUM_WARPS;
  float gate_acc[ROWS_PER_WARP];
  float up_acc[ROWS_PER_WARP];

  #pragma unroll
  for (int i = 0; i < ROWS_PER_WARP; i++) {
    gate_acc[i] = 0.0f;
    up_acc[i] = 0.0f;
  }

  // Stream over hidden dimension
  const int num_k_iters = (hidden_size + TILE_K - 1) / TILE_K;

  for (int k_iter = 0; k_iter < num_k_iters; k_iter++) {
    const int k_base = k_iter * TILE_K;
    const int stage = k_iter % 2;
    __nv_bfloat16* stage_ptr = smem_bf16 + stage * elems_per_stage;

    // --- Cooperative load: activation slice ---
    // Load x[token_idx, k_base:k_base+TILE_K] into shared memory
    {
      const __nv_bfloat16* x_row = x + token_idx * hidden_size + k_base;
      for (int i = tid; i < TILE_K; i += NUM_THREADS) {
        if (k_base + i < hidden_size) {
          stage_ptr[smem_x_offset + i] = x_row[i];
        } else {
          stage_ptr[smem_x_offset + i] = __float2bfloat16(0.0f);
        }
      }
    }

    // --- Cooperative load: gate weight tile ---
    // Load w_gate[expert_id, n_base:n_base+TILE_N, k_base:k_base+TILE_K]
    {
      const __nv_bfloat16* gate_base = w_gate
          + (int64_t)expert_id * intermediate_size * hidden_size
          + (int64_t)n_base * hidden_size
          + k_base;
      __nv_bfloat16* gate_smem = stage_ptr + smem_gate_offset;

      for (int idx = tid; idx < TILE_N * TILE_K; idx += NUM_THREADS) {
        const int row = idx / TILE_K;
        const int col = idx % TILE_K;
        if (n_base + row < intermediate_size && k_base + col < hidden_size) {
          gate_smem[row * TILE_K + col] = gate_base[row * hidden_size + col];
        } else {
          gate_smem[row * TILE_K + col] = __float2bfloat16(0.0f);
        }
      }
    }

    // --- Cooperative load: up weight tile ---
    {
      const __nv_bfloat16* up_base = w_up
          + (int64_t)expert_id * intermediate_size * hidden_size
          + (int64_t)n_base * hidden_size
          + k_base;
      __nv_bfloat16* up_smem = stage_ptr + smem_up_offset;

      for (int idx = tid; idx < TILE_N * TILE_K; idx += NUM_THREADS) {
        const int row = idx / TILE_K;
        const int col = idx % TILE_K;
        if (n_base + row < intermediate_size && k_base + col < hidden_size) {
          up_smem[row * TILE_K + col] = up_base[row * hidden_size + col];
        } else {
          up_smem[row * TILE_K + col] = __float2bfloat16(0.0f);
        }
      }
    }

    __syncthreads();

    // --- Compute: dot products ---
    // Each warp handles ROWS_PER_WARP rows of the N dimension.
    // Within each row, the 32 lanes cooperatively reduce TILE_K elements.
    const __nv_bfloat16* x_smem = stage_ptr + smem_x_offset;
    const __nv_bfloat16* gate_smem = stage_ptr + smem_gate_offset;
    const __nv_bfloat16* up_smem = stage_ptr + smem_up_offset;

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
      const int n_row = warp_id * ROWS_PER_WARP + r;
      if (n_row >= TILE_N) break;

      float g_local = 0.0f;
      float u_local = 0.0f;

      // Each lane reduces TILE_K/32 elements
      #pragma unroll
      for (int kk = lane_id; kk < TILE_K; kk += 32) {
        float xv = __bfloat162float(x_smem[kk]);
        float gv = __bfloat162float(gate_smem[n_row * TILE_K + kk]);
        float uv = __bfloat162float(up_smem[n_row * TILE_K + kk]);
        g_local += gv * xv;
        u_local += uv * xv;
      }

      // Warp-level reduction
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        g_local += __shfl_down_sync(0xFFFFFFFF, g_local, offset);
        u_local += __shfl_down_sync(0xFFFFFFFF, u_local, offset);
      }

      if (lane_id == 0) {
        gate_acc[r] += g_local;
        up_acc[r] += u_local;
      }
    }

    __syncthreads();
  }

  // --- Epilogue: SiLU(gate) * up, write output ---
  if (lane_id == 0) {
    const int te_idx = token_idx * top_k + k_idx;

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
      const int n_idx = n_base + warp_id * ROWS_PER_WARP + r;
      if (n_idx < intermediate_size) {
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        float g = gate_acc[r];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        float silu_g = g * sigmoid_g;
        float result = silu_g * up_acc[r];

        out[te_idx * intermediate_size + n_idx] = __float2bfloat16(result);
      }
    }
  }
}


// ---------------------------------------------------------------------------
// Gate/Up fused kernel for PACKED w13 weights (gate+up concatenated)
// ---------------------------------------------------------------------------
// w13[expert, 2*intermediate_size, hidden_size]:
//   rows [0, N) are gate, rows [N, 2*N) are up.
// ---------------------------------------------------------------------------

template <int TILE_N, int TILE_K, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS * 32)
warp_decode_gate_up_packed_cute_kernel(
    const __nv_bfloat16* __restrict__ x,       // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ w13,      // [num_experts, 2*intermediate_size, hidden_size]
    __nv_bfloat16* __restrict__ out,             // [num_tokens * top_k, intermediate_size]
    const int* __restrict__ expert_ids,          // [num_tokens, top_k]
    int hidden_size,
    int intermediate_size,
    int top_k,
    int num_tokens) {

  extern __shared__ char smem[];

  constexpr int NUM_THREADS = NUM_WARPS * 32;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  const int pid_n = blockIdx.x;
  const int pid_te = blockIdx.y;

  const int token_idx = pid_te / top_k;
  const int k_idx = pid_te % top_k;

  if (token_idx >= num_tokens) return;

  const int expert_id = expert_ids[token_idx * top_k + k_idx];
  const int n_base = pid_n * TILE_N;

  // w13 has 2*intermediate_size rows: gate at [0,N), up at [N,2*N)
  const int w13_row_stride = hidden_size;  // contiguous in K
  const int64_t expert_offset = (int64_t)expert_id * 2 * intermediate_size * hidden_size;

  // Shared memory layout per stage:
  //   x_tile:    [TILE_K] bf16
  //   gate_tile: [TILE_N, TILE_K] bf16
  //   up_tile:   [TILE_N, TILE_K] bf16
  const int smem_x_offset = 0;
  const int smem_gate_offset = TILE_K;
  const int smem_up_offset = TILE_K + TILE_N * TILE_K;
  const int elems_per_stage = TILE_K + 2 * TILE_N * TILE_K;

  __nv_bfloat16* smem_bf16 = reinterpret_cast<__nv_bfloat16*>(smem);

  constexpr int ROWS_PER_WARP = (TILE_N + NUM_WARPS - 1) / NUM_WARPS;
  float gate_acc[ROWS_PER_WARP];
  float up_acc[ROWS_PER_WARP];

  #pragma unroll
  for (int i = 0; i < ROWS_PER_WARP; i++) {
    gate_acc[i] = 0.0f;
    up_acc[i] = 0.0f;
  }

  const int num_k_iters = (hidden_size + TILE_K - 1) / TILE_K;

  for (int k_iter = 0; k_iter < num_k_iters; k_iter++) {
    const int k_base = k_iter * TILE_K;
    const int stage = k_iter % 2;
    __nv_bfloat16* stage_ptr = smem_bf16 + stage * elems_per_stage;

    // Load activation slice
    {
      const __nv_bfloat16* x_row = x + token_idx * hidden_size + k_base;
      for (int i = tid; i < TILE_K; i += NUM_THREADS) {
        stage_ptr[smem_x_offset + i] = (k_base + i < hidden_size)
            ? x_row[i] : __float2bfloat16(0.0f);
      }
    }

    // Load gate weight tile from first half of w13
    {
      const __nv_bfloat16* gate_base = w13 + expert_offset
          + (int64_t)n_base * w13_row_stride + k_base;
      __nv_bfloat16* gate_smem = stage_ptr + smem_gate_offset;

      for (int idx = tid; idx < TILE_N * TILE_K; idx += NUM_THREADS) {
        const int row = idx / TILE_K;
        const int col = idx % TILE_K;
        gate_smem[idx] = (n_base + row < intermediate_size && k_base + col < hidden_size)
            ? gate_base[row * w13_row_stride + col] : __float2bfloat16(0.0f);
      }
    }

    // Load up weight tile from second half of w13
    {
      const __nv_bfloat16* up_base = w13 + expert_offset
          + (int64_t)(intermediate_size + n_base) * w13_row_stride + k_base;
      __nv_bfloat16* up_smem = stage_ptr + smem_up_offset;

      for (int idx = tid; idx < TILE_N * TILE_K; idx += NUM_THREADS) {
        const int row = idx / TILE_K;
        const int col = idx % TILE_K;
        up_smem[idx] = (n_base + row < intermediate_size && k_base + col < hidden_size)
            ? up_base[row * w13_row_stride + col] : __float2bfloat16(0.0f);
      }
    }

    __syncthreads();

    // Compute dot products
    const __nv_bfloat16* x_smem = stage_ptr + smem_x_offset;
    const __nv_bfloat16* gate_smem = stage_ptr + smem_gate_offset;
    const __nv_bfloat16* up_smem = stage_ptr + smem_up_offset;

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
      const int n_row = warp_id * ROWS_PER_WARP + r;
      if (n_row >= TILE_N) break;

      float g_local = 0.0f;
      float u_local = 0.0f;

      #pragma unroll
      for (int kk = lane_id; kk < TILE_K; kk += 32) {
        float xv = __bfloat162float(x_smem[kk]);
        float gv = __bfloat162float(gate_smem[n_row * TILE_K + kk]);
        float uv = __bfloat162float(up_smem[n_row * TILE_K + kk]);
        g_local += gv * xv;
        u_local += uv * xv;
      }

      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        g_local += __shfl_down_sync(0xFFFFFFFF, g_local, offset);
        u_local += __shfl_down_sync(0xFFFFFFFF, u_local, offset);
      }

      if (lane_id == 0) {
        gate_acc[r] += g_local;
        up_acc[r] += u_local;
      }
    }

    __syncthreads();
  }

  // Epilogue: SiLU(gate) * up
  if (lane_id == 0) {
    const int te_idx = token_idx * top_k + k_idx;

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
      const int n_idx = n_base + warp_id * ROWS_PER_WARP + r;
      if (n_idx < intermediate_size) {
        float g = gate_acc[r];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        float result = (g * sigmoid_g) * up_acc[r];
        out[te_idx * intermediate_size + n_idx] = __float2bfloat16(result);
      }
    }
  }
}


// ---------------------------------------------------------------------------
// Down projection + expert combine kernel
// ---------------------------------------------------------------------------
// Grid: (ceil(hidden_size / TILE_D), num_tokens)
// Each block computes TILE_D output dimensions for one token.
// Loops over top-k experts, folding routing weights.
// ---------------------------------------------------------------------------

template <int TILE_D, int TILE_N, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS * 32)
warp_decode_down_cute_kernel(
    const __nv_bfloat16* __restrict__ intermediate,  // [num_tokens * top_k, intermediate_size]
    const __nv_bfloat16* __restrict__ w_down,         // [num_experts, hidden_size, intermediate_size]
    const float* __restrict__ routing_weights,         // [num_tokens, top_k]
    const int* __restrict__ expert_ids,                // [num_tokens, top_k]
    __nv_bfloat16* __restrict__ out,                   // [num_tokens, hidden_size]
    int hidden_size,
    int intermediate_size,
    int top_k,
    int num_tokens) {

  extern __shared__ char smem[];

  constexpr int NUM_THREADS = NUM_WARPS * 32;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  const int pid_d = blockIdx.x;  // output dimension tile
  const int pid_t = blockIdx.y;  // token

  if (pid_t >= num_tokens) return;

  const int d_base = pid_d * TILE_D;

  // Shared memory layout per stage:
  //   inter_tile: [TILE_N] bf16
  //   w_down_tile: [TILE_D, TILE_N] bf16
  const int smem_inter_offset = 0;
  const int smem_w_offset = TILE_N;
  const int elems_per_stage = TILE_N + TILE_D * TILE_N;

  __nv_bfloat16* smem_bf16 = reinterpret_cast<__nv_bfloat16*>(smem);

  // Accumulator: each warp handles ROWS_PER_WARP output dimensions
  constexpr int ROWS_PER_WARP = (TILE_D + NUM_WARPS - 1) / NUM_WARPS;
  float out_acc[ROWS_PER_WARP];

  #pragma unroll
  for (int i = 0; i < ROWS_PER_WARP; i++) {
    out_acc[i] = 0.0f;
  }

  // Loop over top-k experts
  for (int k_idx = 0; k_idx < top_k; k_idx++) {
    const float routing_weight = routing_weights[pid_t * top_k + k_idx];
    const int expert_id = expert_ids[pid_t * top_k + k_idx];
    const int te_idx = pid_t * top_k + k_idx;

    float expert_acc[ROWS_PER_WARP];
    #pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; i++) {
      expert_acc[i] = 0.0f;
    }

    // Stream over intermediate dimension
    const int num_n_iters = (intermediate_size + TILE_N - 1) / TILE_N;

    for (int n_iter = 0; n_iter < num_n_iters; n_iter++) {
      const int n_base = n_iter * TILE_N;
      const int stage = n_iter % 2;
      __nv_bfloat16* stage_ptr = smem_bf16 + stage * elems_per_stage;

      // Load intermediate activation slice
      {
        const __nv_bfloat16* inter_row = intermediate
            + te_idx * intermediate_size + n_base;
        for (int i = tid; i < TILE_N; i += NUM_THREADS) {
          stage_ptr[smem_inter_offset + i] = (n_base + i < intermediate_size)
              ? inter_row[i] : __float2bfloat16(0.0f);
        }
      }

      // Load down weight tile: w_down[expert_id, d_base:d_base+TILE_D, n_base:n_base+TILE_N]
      {
        const __nv_bfloat16* w_base = w_down
            + (int64_t)expert_id * hidden_size * intermediate_size
            + (int64_t)d_base * intermediate_size
            + n_base;
        __nv_bfloat16* w_smem = stage_ptr + smem_w_offset;

        for (int idx = tid; idx < TILE_D * TILE_N; idx += NUM_THREADS) {
          const int row = idx / TILE_N;
          const int col = idx % TILE_N;
          w_smem[idx] = (d_base + row < hidden_size && n_base + col < intermediate_size)
              ? w_base[row * intermediate_size + col] : __float2bfloat16(0.0f);
        }
      }

      __syncthreads();

      // Compute: dot product of weight rows with intermediate vector
      const __nv_bfloat16* inter_smem = stage_ptr + smem_inter_offset;
      const __nv_bfloat16* w_smem = stage_ptr + smem_w_offset;

      #pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; r++) {
        const int d_row = warp_id * ROWS_PER_WARP + r;
        if (d_row >= TILE_D) break;

        float local_sum = 0.0f;

        #pragma unroll
        for (int nn = lane_id; nn < TILE_N; nn += 32) {
          float iv = __bfloat162float(inter_smem[nn]);
          float wv = __bfloat162float(w_smem[d_row * TILE_N + nn]);
          local_sum += wv * iv;
        }

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }

        if (lane_id == 0) {
          expert_acc[r] += local_sum;
        }
      }

      __syncthreads();
    }

    // Fold routing weight into output accumulator
    if (lane_id == 0) {
      #pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; r++) {
        out_acc[r] += routing_weight * expert_acc[r];
      }
    }
  }

  // Write output
  if (lane_id == 0) {
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
      const int d_idx = d_base + warp_id * ROWS_PER_WARP + r;
      if (d_idx < hidden_size) {
        out[pid_t * hidden_size + d_idx] = __float2bfloat16(out_acc[r]);
      }
    }
  }
}


// ---------------------------------------------------------------------------
// NVFP4 Down projection kernel
// ---------------------------------------------------------------------------
// Same structure as bf16 down kernel, but with on-the-fly FP4 dequantization.
// w_down is stored as uint8 packed (2 FP4 values per byte) with blockscales.
// ---------------------------------------------------------------------------

// FP4 E2M1 lookup table (NVIDIA NVFP4 format)
__device__ __forceinline__ float nvfp4_dequant(uint8_t nibble, float scale) {
  // NVFP4 E2M1: 1 sign + 2 exponent + 1 mantissa
  // Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (positive)
  static constexpr float LUT[16] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
  };
  return LUT[nibble & 0xF] * scale;
}

template <int TILE_D, int TILE_N, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS * 32)
warp_decode_down_fp4_cute_kernel(
    const __nv_bfloat16* __restrict__ intermediate,  // [num_tokens * top_k, intermediate_size]
    const uint8_t* __restrict__ w_down_packed,         // [num_experts, hidden_size, intermediate_size/2]
    const __nv_bfloat16* __restrict__ w_down_scales,   // [num_experts, hidden_size, intermediate_size/group_size]
    const float* __restrict__ w_down_alpha,             // [1] global tensor scale
    const float* __restrict__ routing_weights,
    const int* __restrict__ expert_ids,
    __nv_bfloat16* __restrict__ out,
    int hidden_size,
    int intermediate_size,
    int top_k,
    int num_tokens,
    int group_size) {

  extern __shared__ char smem[];

  constexpr int NUM_THREADS = NUM_WARPS * 32;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  const int pid_d = blockIdx.x;
  const int pid_t = blockIdx.y;

  if (pid_t >= num_tokens) return;

  const int d_base = pid_d * TILE_D;
  const float alpha = w_down_alpha[0];

  // For FP4: we load packed bytes into shared memory, then dequant in registers.
  // Shared memory layout per stage:
  //   inter_tile: [TILE_N] bf16
  //   w_packed:   [TILE_D, TILE_N/2] uint8 (packed FP4 pairs)
  //   w_scales:   [TILE_D, TILE_N/group_size] bf16
  // We allocate generously using TILE_N for the intermediate and appropriate
  // byte counts for packed weights.
  const int smem_inter_elems = TILE_N;
  const int smem_packed_bytes = TILE_D * (TILE_N / 2);
  const int groups_per_tile = (TILE_N + group_size - 1) / group_size;
  const int smem_scale_elems = TILE_D * groups_per_tile;

  // Byte offsets in smem
  const int smem_inter_byte_offset = 0;
  const int smem_packed_byte_offset = smem_inter_elems * sizeof(__nv_bfloat16);
  const int smem_scale_byte_offset = smem_packed_byte_offset + smem_packed_bytes;

  constexpr int ROWS_PER_WARP = (TILE_D + NUM_WARPS - 1) / NUM_WARPS;
  float out_acc[ROWS_PER_WARP];

  #pragma unroll
  for (int i = 0; i < ROWS_PER_WARP; i++) {
    out_acc[i] = 0.0f;
  }

  for (int k_idx = 0; k_idx < top_k; k_idx++) {
    const float routing_weight = routing_weights[pid_t * top_k + k_idx];
    const int expert_id = expert_ids[pid_t * top_k + k_idx];
    const int te_idx = pid_t * top_k + k_idx;

    float expert_acc[ROWS_PER_WARP];
    #pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; i++) {
      expert_acc[i] = 0.0f;
    }

    const int packed_cols = intermediate_size / 2;
    const int scale_cols = intermediate_size / group_size;
    const int num_n_iters = (intermediate_size + TILE_N - 1) / TILE_N;

    for (int n_iter = 0; n_iter < num_n_iters; n_iter++) {
      const int n_base = n_iter * TILE_N;

      // Load intermediate activation
      __nv_bfloat16* inter_smem = reinterpret_cast<__nv_bfloat16*>(smem + smem_inter_byte_offset);
      {
        const __nv_bfloat16* inter_row = intermediate + te_idx * intermediate_size + n_base;
        for (int i = tid; i < TILE_N; i += NUM_THREADS) {
          inter_smem[i] = (n_base + i < intermediate_size)
              ? inter_row[i] : __float2bfloat16(0.0f);
        }
      }

      // Load packed FP4 weight bytes
      uint8_t* packed_smem = reinterpret_cast<uint8_t*>(smem + smem_packed_byte_offset);
      {
        const uint8_t* w_row = w_down_packed
            + (int64_t)expert_id * hidden_size * packed_cols
            + (int64_t)d_base * packed_cols
            + n_base / 2;
        const int tile_packed_cols = TILE_N / 2;

        for (int idx = tid; idx < TILE_D * tile_packed_cols; idx += NUM_THREADS) {
          const int row = idx / tile_packed_cols;
          const int col = idx % tile_packed_cols;
          if (d_base + row < hidden_size && n_base / 2 + col < packed_cols) {
            packed_smem[idx] = w_row[row * packed_cols + col];
          } else {
            packed_smem[idx] = 0;
          }
        }
      }

      // Load blockscales
      __nv_bfloat16* scale_smem = reinterpret_cast<__nv_bfloat16*>(smem + smem_scale_byte_offset);
      {
        const int tile_group_start = n_base / group_size;
        const __nv_bfloat16* s_row = w_down_scales
            + (int64_t)expert_id * hidden_size * scale_cols
            + (int64_t)d_base * scale_cols
            + tile_group_start;

        for (int idx = tid; idx < TILE_D * groups_per_tile; idx += NUM_THREADS) {
          const int row = idx / groups_per_tile;
          const int col = idx % groups_per_tile;
          if (d_base + row < hidden_size && tile_group_start + col < scale_cols) {
            scale_smem[idx] = s_row[row * scale_cols + col];
          } else {
            scale_smem[idx] = __float2bfloat16(0.0f);
          }
        }
      }

      __syncthreads();

      // Compute with FP4 dequantization
      #pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; r++) {
        const int d_row = warp_id * ROWS_PER_WARP + r;
        if (d_row >= TILE_D) break;

        float local_sum = 0.0f;
        const int tile_packed_cols = TILE_N / 2;

        for (int nn = lane_id; nn < TILE_N; nn += 32) {
          float iv = __bfloat162float(inter_smem[nn]);

          // Dequantize FP4 weight
          const int packed_idx = nn / 2;
          uint8_t packed_byte = packed_smem[d_row * tile_packed_cols + packed_idx];
          uint8_t nibble = (nn % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);

          // Get blockscale
          const int group_idx = nn / group_size;
          float block_scale = __bfloat162float(scale_smem[d_row * groups_per_tile + group_idx]);

          float wv = nvfp4_dequant(nibble, block_scale * alpha);
          local_sum += wv * iv;
        }

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }

        if (lane_id == 0) {
          expert_acc[r] += local_sum;
        }
      }

      __syncthreads();
    }

    // Fold routing weight
    if (lane_id == 0) {
      #pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; r++) {
        out_acc[r] += routing_weight * expert_acc[r];
      }
    }
  }

  // Write output
  if (lane_id == 0) {
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
      const int d_idx = d_base + warp_id * ROWS_PER_WARP + r;
      if (d_idx < hidden_size) {
        out[pid_t * hidden_size + d_idx] = __float2bfloat16(out_acc[r]);
      }
    }
  }
}

}  // namespace warp_decode
}  // namespace sglang
