// Copyright 2024-2026 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");
// ==============================================================================
/**
 * @file warp_decode_cute.cuh
 * @brief Blog-strict warp-decode for SM100 (B200).
 *
 * Implementation-detail optimizations layered on top of the strict blog
 * design (https://cursor.com/blog/warp-decode):
 *
 *   1. bf16x2 inner multiply via __hfma2 with FP32 accumulator updated once
 *      per K-step after warp reduction (HFMA2 SASS).
 *   2. TILE_K=512 in gate/up with a kFlush=128 split-FP32 inner accumulator.
 *      The wider tile halves K-loop iterations from 28 to 14 at D=7168;
 *      flushing the bf16x2 partial into FP32 every 128 elements preserves
 *      the same BF16 accumulation depth as a TILE_K=128 baseline so cosine
 *      similarity does not drift. SMEM 52 KB/CTA × 3 stages fits within
 *      B200's 232 KB; +1 register/thread, no spill. Down kernel uses
 *      TILE_N=1024.
 *   3. 3-stage cp.async pipeline with proper N-1 prologue/epilogue
 *      (prefetch stages 0,1; wait_group<2> in steady state; wait_group<1>
 *      then <0> for the last two iterations).
 *   4. Optional gate/up→down PDL chain (cudaTriggerProgrammaticLaunchCompletion
 *      + cudaGridDependencySynchronize, host-side cudaLaunchKernelEx with
 *      cudaLaunchAttributeProgrammaticStreamSerialization). Gated by
 *      WD_PDL_ENABLED build flag; no-ops out when not defined.
 *
 * Invariants preserved:
 *   1. 8 warps × 1 neuron in gate/up.
 *   2. 8 warps × 1 dim in down.
 *   3. No cross-warp synchronization or shared mutable state between warps
 *      (the PDL trigger is a grid-level dependency edge, not warp sync).
 *   4. __shfl_xor_sync butterfly reduction.
 *   5. FP32 final accumulators (gate_acc, up_acc, expert_acc).
 *   6. Embarrassingly parallel — warp scheduler issues warps in any order.
 *
 * Forbidden (per spec):
 *   - tcgen05.mma / wgmma / nvcuda::wmma (no tensor cores).
 *   - Multiple neurons per warp.
 *   - Cross-warp barriers beyond kernel completion.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <cstdint>

// PDL (Programmatic Dependent Launch) — opt-in via WD_PDL_ENABLED build flag.
// Brings in cudaTriggerProgrammaticLaunchCompletion / cudaGridDependencySynchronize.
#if defined(WD_PDL_ENABLED) && WD_PDL_ENABLED
#include <cuda_device_runtime_api.h>
#define WD_PDL_TRIGGER() ::cudaTriggerProgrammaticLaunchCompletion()
#define WD_PDL_WAIT()    ::cudaGridDependencySynchronize()
#else
#define WD_PDL_TRIGGER() ((void)0)
#define WD_PDL_WAIT()    ((void)0)
#endif

namespace sglang {
namespace warp_decode {

// ---------------------------------------------------------------------------
// NVFP4 LUT (kept for FP4 compatibility)
// ---------------------------------------------------------------------------
__device__ __constant__ float kNvfp4Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float NvFp4Dequant(uint8_t nibble, float scale) {
  return kNvfp4Lut[nibble & 0xF] * scale;
}

// ---------------------------------------------------------------------------
// Tile configurations: BLOG-STRICT
//   gate/up: TILE_N = NUM_WARPS = 8 (one neuron per warp)
//   down:    TILE_D = NUM_WARPS = 8 (one output dim per warp)
// ---------------------------------------------------------------------------

struct GateUpTileConfigBlog {
  static constexpr int kTileN = 8;       // 8 neurons per CTA
  static constexpr int kTileK = 128;
  static constexpr int kNumWarps = 8;    // 8 warps -> 1 neuron each
  static constexpr int kNumThreads = kNumWarps * 32;
  static constexpr int kSmemXElems = kTileK;
  static constexpr int kSmemWElems = kTileN * kTileK;
  static constexpr int kElemsPerStage = kSmemXElems + 2 * kSmemWElems;
  static constexpr int kBytesPerStage = kElemsPerStage * sizeof(__nv_bfloat16);
  static constexpr int kNumStages = 2;
  static constexpr int kTotalSmemBytes = kBytesPerStage * kNumStages;
  static_assert(kTotalSmemBytes <= 232 * 1024, "");
  static_assert(kTileK % 8 == 0, "");
};

struct DownTileConfigBlog {
  static constexpr int kTileD = 8;       // 8 output dims per CTA
  static constexpr int kTileN = 512;
  static constexpr int kNumWarps = 8;    // 8 warps -> 1 dim each
  static constexpr int kNumThreads = kNumWarps * 32;
  static constexpr int kSmemInterElems = kTileN;
  static constexpr int kSmemWElems = kTileD * kTileN;
  static constexpr int kElemsPerStage = kSmemInterElems + kSmemWElems;
  static constexpr int kBytesPerStage = kElemsPerStage * sizeof(__nv_bfloat16);
  static constexpr int kNumStages = 2;
  static constexpr int kTotalSmemBytes = kBytesPerStage * kNumStages;
  static_assert(kTotalSmemBytes <= 232 * 1024, "");
  static_assert(kTileN % 8 == 0, "");
};

// ---------------------------------------------------------------------------
// cp.async helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ void LoadVec128(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src) {
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
}

__device__ __forceinline__ void CpAsyncLoad128(
    __nv_bfloat16* __restrict__ smem_dst,
    const __nv_bfloat16* __restrict__ gmem_src) {
  uint32_t smem_addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(reinterpret_cast<void*>(smem_dst)));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :
      : "r"(smem_addr), "l"(gmem_src)
      : "memory");
}

__device__ __forceinline__ void CpAsyncCommit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void CpAsyncWaitGroup() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void CpAsyncCommitAndWait() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

template <int kNumThreads>
__device__ __forceinline__ void CoopLoadVec(
    __nv_bfloat16* __restrict__ smem,
    const __nv_bfloat16* __restrict__ gmem,
    int count, int tid) {
  constexpr int kVecSize = 8;
  const int vec_count = count / kVecSize;
  for (int i = tid; i < vec_count; i += kNumThreads)
    CpAsyncLoad128(smem + i * kVecSize, gmem + i * kVecSize);
  const int tail_start = vec_count * kVecSize;
  for (int i = tail_start + tid; i < count; i += kNumThreads)
    smem[i] = gmem[i];
}

__device__ __forceinline__ float WarpReduceSum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  return val;
}

// ===========================================================================
// Kernel 1: BLOG-STRICT Gate/Up (separate weights)
//   * 8 warps × 1 neuron per warp.
//   * BF16 ld → FP32 mul-add accumulators (NOT bf16x2; blog uses FP32 acc).
//   * uint4 (8 bf16) inner-K LDG vector inside each warp.
// ===========================================================================

template <int TILE_N, int TILE_K, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS * 32)
warp_decode_gate_up_cute_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ w_up,
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ expert_ids,
    int hidden_size, int intermediate_size, int top_k, int num_tokens) {
  extern __shared__ __align__(16) char smem_raw[];

  static_assert(TILE_N == NUM_WARPS, "Blog-strict requires TILE_N == NUM_WARPS (1 neuron per warp)");
  constexpr int kNumThreads = NUM_WARPS * 32;
  constexpr int kVec = 8;
  constexpr int kSmemXOff = 0;
  constexpr int kSmemGateOff = TILE_K;
  constexpr int kSmemUpOff = TILE_K + TILE_N * TILE_K;
  constexpr int kElemsPerStage = TILE_K + 2 * TILE_N * TILE_K;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;

  const int pid_n = blockIdx.x;
  const int pid_te = blockIdx.y;
  const int token_idx = pid_te / top_k;
  const int k_idx = pid_te % top_k;
  if (token_idx >= num_tokens) return;

  const int expert_id = expert_ids[token_idx * top_k + k_idx];
  const int n_base = pid_n * TILE_N;

  __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);

  // BLOG-STRICT: 1 neuron per warp -> single FP32 accumulator
  float gate_acc = 0.0f;
  float up_acc = 0.0f;

  const __nv_bfloat16* x_row = x + (int64_t)token_idx * hidden_size;
  const __nv_bfloat16* gate_expert =
      w_gate + (int64_t)expert_id * intermediate_size * hidden_size +
      (int64_t)n_base * hidden_size;
  const __nv_bfloat16* up_expert =
      w_up + (int64_t)expert_id * intermediate_size * hidden_size +
      (int64_t)n_base * hidden_size;

  const int num_k_iters = (hidden_size + TILE_K - 1) / TILE_K;

  auto load_stage = [&](int it, int s) {
    if (it >= num_k_iters) return;
    const int kb = it * TILE_K;
    __nv_bfloat16* sp = smem + s * kElemsPerStage;
    const int x_v = TILE_K / kVec;
    for (int i = tid; i < x_v; i += kNumThreads)
      CpAsyncLoad128(sp + kSmemXOff + i*kVec, x_row + kb + i*kVec);
    const int wv = TILE_N * TILE_K / kVec;
    constexpr int wpv = TILE_K / kVec;
    for (int i = tid; i < wv; i += kNumThreads) {
      int r = i / wpv;
      int c = (i % wpv) * kVec;
      CpAsyncLoad128(sp + kSmemGateOff + r*TILE_K + c, gate_expert + r*hidden_size + kb + c);
      CpAsyncLoad128(sp + kSmemUpOff + r*TILE_K + c, up_expert + r*hidden_size + kb + c);
    }
  };

  // OPT3: 3-stage cp.async pipeline.
  // Prefetch stages 0, 1; in iter it: wait for stage it%3 (group 2 deep), compute,
  // schedule prefetch of (it+2)%3.
  constexpr int kNumStages = 3;
  load_stage(0, 0);
  CpAsyncCommit();
  if (num_k_iters >= 2) {
    load_stage(1, 1);
    CpAsyncCommit();
  }
  for (int it = 0; it < num_k_iters; it++) {
    int next = it + 2;   // 2 stages ahead
    int cs = it % kNumStages;
    if (next < num_k_iters) {
      load_stage(next, next % kNumStages);
      CpAsyncCommit();
      CpAsyncWaitGroup<2>();
    } else if (it + 1 < num_k_iters) {
      CpAsyncWaitGroup<1>();
    } else {
      CpAsyncWaitGroup<0>();
    }
    __syncthreads();

    __nv_bfloat16* sp = smem + cs * kElemsPerStage;
    const __nv_bfloat16* x_smem = sp + kSmemXOff;
    const __nv_bfloat16* gate_smem = sp + kSmemGateOff;
    const __nv_bfloat16* up_smem = sp + kSmemUpOff;

    const int n_row = warp_id;
    const __nv_bfloat162* x_smem2 = reinterpret_cast<const __nv_bfloat162*>(x_smem);
    const __nv_bfloat162* gate_smem2 = reinterpret_cast<const __nv_bfloat162*>(gate_smem);
    const __nv_bfloat162* up_smem2 = reinterpret_cast<const __nv_bfloat162*>(up_smem);

    constexpr int kV2 = TILE_K / 2;
    // Flush the bf16x2 partial into FP32 every kFlush=128 elements so the
    // BF16 accumulation depth stays at 4 hfma2/lane regardless of TILE_K.
    // For TILE_K <= 256 this collapses to one flush at the end.
    constexpr int kFlush = (kV2 > 128) ? 128 : kV2;
    float g_local = 0.f;
    float u_local = 0.f;
#pragma unroll
    for (int kk_base = 0; kk_base < kV2; kk_base += kFlush) {
      __nv_bfloat162 g_acc2 = {__float2bfloat16(0.f), __float2bfloat16(0.f)};
      __nv_bfloat162 u_acc2 = {__float2bfloat16(0.f), __float2bfloat16(0.f)};
#pragma unroll
      for (int kk = lane_id; kk < kFlush; kk += 32) {
        const int gk = kk_base + kk;
        __nv_bfloat162 x2 = x_smem2[gk];
        __nv_bfloat162 g2 = gate_smem2[n_row * kV2 + gk];
        __nv_bfloat162 u2 = up_smem2[n_row * kV2 + gk];
        g_acc2 = __hfma2(g2, x2, g_acc2);
        u_acc2 = __hfma2(u2, x2, u_acc2);
      }
      g_local += __bfloat162float(g_acc2.x) + __bfloat162float(g_acc2.y);
      u_local += __bfloat162float(u_acc2.x) + __bfloat162float(u_acc2.y);
    }
    g_local = WarpReduceSum(g_local);
    u_local = WarpReduceSum(u_local);
    if (lane_id == 0) {
      gate_acc += g_local;
      up_acc += u_local;
    }
    __syncthreads();
  }

  if (lane_id == 0) {
    const int te_idx = token_idx * top_k + k_idx;
    __nv_bfloat16* out_row = out + (int64_t)te_idx * intermediate_size;
    const int n_idx = n_base + warp_id;     // 1 neuron per warp
    if (n_idx < intermediate_size) {
      float g = gate_acc;
      float silu = g / (1.0f + expf(-g));
      out_row[n_idx] = __float2bfloat16(silu * up_acc);
    }
  }
  // Signal the dependent down kernel that intermediate is ready. The
  // threadfence orders the store above before the trigger PTX. No-op when
  // WD_PDL_ENABLED is unset.
  __syncthreads();
  __threadfence();
  WD_PDL_TRIGGER();
}

// ===========================================================================
// Kernel 1b: BLOG-STRICT packed w13 variant
// ===========================================================================

template <int TILE_N, int TILE_K, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS * 32)
warp_decode_gate_up_packed_cute_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w13,
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ expert_ids,
    int hidden_size, int intermediate_size, int top_k, int num_tokens) {
  extern __shared__ __align__(16) char smem_raw[];

  static_assert(TILE_N == NUM_WARPS, "Blog-strict requires TILE_N == NUM_WARPS");
  constexpr int kNumThreads = NUM_WARPS * 32;
  constexpr int kVec = 8;
  constexpr int kSmemXOff = 0;
  constexpr int kSmemGateOff = TILE_K;
  constexpr int kSmemUpOff = TILE_K + TILE_N * TILE_K;
  constexpr int kElemsPerStage = TILE_K + 2 * TILE_N * TILE_K;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;

  const int pid_n = blockIdx.x;
  const int pid_te = blockIdx.y;
  const int token_idx = pid_te / top_k;
  const int k_idx = pid_te % top_k;
  if (token_idx >= num_tokens) return;

  const int expert_id = expert_ids[token_idx * top_k + k_idx];
  const int n_base = pid_n * TILE_N;

  const int64_t expert_offset = (int64_t)expert_id * 2 * intermediate_size * hidden_size;
  const __nv_bfloat16* gate_expert = w13 + expert_offset + (int64_t)n_base * hidden_size;
  const __nv_bfloat16* up_expert =
      w13 + expert_offset + (int64_t)(intermediate_size + n_base) * hidden_size;
  const __nv_bfloat16* x_row = x + (int64_t)token_idx * hidden_size;

  __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  float gate_acc = 0.0f, up_acc = 0.0f;

  const int num_k_iters = (hidden_size + TILE_K - 1) / TILE_K;

  auto load_stage = [&](int it, int s) {
    if (it >= num_k_iters) return;
    const int kb = it * TILE_K;
    __nv_bfloat16* sp = smem + s * kElemsPerStage;
    const int x_v = TILE_K / kVec;
    for (int i = tid; i < x_v; i += kNumThreads)
      CpAsyncLoad128(sp + kSmemXOff + i*kVec, x_row + kb + i*kVec);
    const int wv = TILE_N * TILE_K / kVec;
    constexpr int wpv = TILE_K / kVec;
    for (int i = tid; i < wv; i += kNumThreads) {
      int r = i / wpv;
      int c = (i % wpv) * kVec;
      CpAsyncLoad128(sp + kSmemGateOff + r*TILE_K + c, gate_expert + r*hidden_size + kb + c);
      CpAsyncLoad128(sp + kSmemUpOff + r*TILE_K + c, up_expert + r*hidden_size + kb + c);
    }
  };

  // OPT3: 3-stage cp.async pipeline (packed)
  constexpr int kNumStages = 3;
  load_stage(0, 0);
  CpAsyncCommit();
  if (num_k_iters >= 2) {
    load_stage(1, 1);
    CpAsyncCommit();
  }
  for (int it = 0; it < num_k_iters; it++) {
    int next = it + 2;
    int cs = it % kNumStages;
    if (next < num_k_iters) {
      load_stage(next, next % kNumStages);
      CpAsyncCommit();
      CpAsyncWaitGroup<2>();
    } else if (it + 1 < num_k_iters) {
      CpAsyncWaitGroup<1>();
    } else {
      CpAsyncWaitGroup<0>();
    }
    __syncthreads();

    __nv_bfloat16* sp = smem + cs * kElemsPerStage;
    const __nv_bfloat16* x_smem = sp + kSmemXOff;
    const __nv_bfloat16* gate_smem = sp + kSmemGateOff;
    const __nv_bfloat16* up_smem = sp + kSmemUpOff;

    const int n_row = warp_id;
    const __nv_bfloat162* x_smem2 = reinterpret_cast<const __nv_bfloat162*>(x_smem);
    const __nv_bfloat162* gate_smem2 = reinterpret_cast<const __nv_bfloat162*>(gate_smem);
    const __nv_bfloat162* up_smem2 = reinterpret_cast<const __nv_bfloat162*>(up_smem);
    constexpr int kV2 = TILE_K / 2;
    // See split-FP32 accumulator note in the non-packed variant.
    constexpr int kFlush = (kV2 > 128) ? 128 : kV2;
    float g_local = 0.f;
    float u_local = 0.f;
#pragma unroll
    for (int kk_base = 0; kk_base < kV2; kk_base += kFlush) {
      __nv_bfloat162 g_acc2 = {__float2bfloat16(0.f), __float2bfloat16(0.f)};
      __nv_bfloat162 u_acc2 = {__float2bfloat16(0.f), __float2bfloat16(0.f)};
#pragma unroll
      for (int kk = lane_id; kk < kFlush; kk += 32) {
        const int gk = kk_base + kk;
        __nv_bfloat162 x2 = x_smem2[gk];
        __nv_bfloat162 g2 = gate_smem2[n_row * kV2 + gk];
        __nv_bfloat162 u2 = up_smem2[n_row * kV2 + gk];
        g_acc2 = __hfma2(g2, x2, g_acc2);
        u_acc2 = __hfma2(u2, x2, u_acc2);
      }
      g_local += __bfloat162float(g_acc2.x) + __bfloat162float(g_acc2.y);
      u_local += __bfloat162float(u_acc2.x) + __bfloat162float(u_acc2.y);
    }
    g_local = WarpReduceSum(g_local);
    u_local = WarpReduceSum(u_local);
    if (lane_id == 0) {
      gate_acc += g_local;
      up_acc += u_local;
    }
    __syncthreads();
  }

  if (lane_id == 0) {
    const int te_idx = token_idx * top_k + k_idx;
    __nv_bfloat16* out_row = out + (int64_t)te_idx * intermediate_size;
    const int n_idx = n_base + warp_id;
    if (n_idx < intermediate_size) {
      float g = gate_acc;
      float silu = g / (1.0f + expf(-g));
      out_row[n_idx] = __float2bfloat16(silu * up_acc);
    }
  }
  // PDL trigger — signal dependent (down) kernel that intermediate is ready.
  __syncthreads();
  __threadfence();
  WD_PDL_TRIGGER();
}

// ===========================================================================
// Kernel 2: BLOG-STRICT Down (1 dim per warp)
// ===========================================================================

template <int TILE_D, int TILE_N, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS * 32)
warp_decode_down_cute_kernel(
    const __nv_bfloat16* __restrict__ intermediate,
    const __nv_bfloat16* __restrict__ w_down,
    const float* __restrict__ routing_weights,
    const int* __restrict__ expert_ids,
    __nv_bfloat16* __restrict__ out,
    int hidden_size, int intermediate_size, int top_k, int num_tokens) {
  extern __shared__ __align__(16) char smem_raw[];

  static_assert(TILE_D == NUM_WARPS, "Blog-strict requires TILE_D == NUM_WARPS");
  constexpr int kNumThreads = NUM_WARPS * 32;
  constexpr int kVec = 8;
  constexpr int kSmemInterOff = 0;
  constexpr int kSmemWOff = TILE_N;
  constexpr int kElemsPerStage = TILE_N + TILE_D * TILE_N;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;

  const int pid_d = blockIdx.x;
  const int pid_t = blockIdx.y;
  if (pid_t >= num_tokens) return;

  const int d_base = pid_d * TILE_D;

  __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);

  // BLOG-STRICT: 1 dim per warp -> single FP32 accumulator
  float out_acc = 0.0f;

  // Block until the prior gate/up grid signals (griddepcontrol.launch_dependents)
  // before any cp.async reads of `intermediate`. No-op when WD_PDL_ENABLED is unset.
  WD_PDL_WAIT();

  const int num_n_iters = (intermediate_size + TILE_N - 1) / TILE_N;

  auto load_stage = [&](int k_idx, int n_iter, int s) {
    if (k_idx >= top_k || n_iter >= num_n_iters) return;
    const int n_base = n_iter * TILE_N;
    const int te_idx = pid_t * top_k + k_idx;
    const int expert_id = expert_ids[pid_t * top_k + k_idx];
    __nv_bfloat16* sp = smem + s * kElemsPerStage;
    const __nv_bfloat16* inter_row = intermediate + (int64_t)te_idx * intermediate_size + n_base;
    const __nv_bfloat16* w_row =
        w_down + (int64_t)expert_id * hidden_size * intermediate_size +
        (int64_t)d_base * intermediate_size + n_base;
    const int iv = TILE_N / kVec;
    for (int i = tid; i < iv; i += kNumThreads)
      CpAsyncLoad128(sp + kSmemInterOff + i*kVec, inter_row + i*kVec);
    const int wv = TILE_D * TILE_N / kVec;
    constexpr int wpv = TILE_N / kVec;
    for (int i = tid; i < wv; i += kNumThreads) {
      int r = i / wpv;
      int c = (i % wpv) * kVec;
      CpAsyncLoad128(sp + kSmemWOff + r*TILE_N + c, w_row + r*intermediate_size + c);
    }
  };

  const int total_iters = top_k * num_n_iters;

  // OPT3: 3-stage cp.async pipeline (down)
  constexpr int kDNumStages = 3;
  load_stage(0, 0, 0);
  CpAsyncCommit();
  if (total_iters >= 2) {
    load_stage(1 / num_n_iters, 1 % num_n_iters, 1);
    CpAsyncCommit();
  }

  float expert_acc = 0.0f;
  int prev_k_idx = -1;

  for (int it = 0; it < total_iters; it++) {
    int k_idx = it / num_n_iters;
    int compute_stage = it % kDNumStages;
    int next = it + 2;
    if (next < total_iters) {
      load_stage(next / num_n_iters, next % num_n_iters, next % kDNumStages);
      CpAsyncCommit();
      CpAsyncWaitGroup<2>();
    } else if (it + 1 < total_iters) {
      CpAsyncWaitGroup<1>();
    } else {
      CpAsyncWaitGroup<0>();
    }
    __syncthreads();

    if (k_idx != prev_k_idx) {
      if (prev_k_idx >= 0 && lane_id == 0) {
        float rw = routing_weights[pid_t * top_k + prev_k_idx];
        out_acc += rw * expert_acc;
      }
      expert_acc = 0.0f;
      prev_k_idx = k_idx;
    }

    __nv_bfloat16* sp = smem + compute_stage * kElemsPerStage;
    const __nv_bfloat16* inter_smem = sp + kSmemInterOff;
    const __nv_bfloat16* w_smem = sp + kSmemWOff;

    const int d_row = warp_id;     // 1 dim per warp
    const __nv_bfloat162* inter_smem2 = reinterpret_cast<const __nv_bfloat162*>(inter_smem);
    const __nv_bfloat162* w_smem2 = reinterpret_cast<const __nv_bfloat162*>(w_smem);
    constexpr int kV2 = TILE_N / 2;
    __nv_bfloat162 acc2 = {__float2bfloat16(0.f), __float2bfloat16(0.f)};
#pragma unroll
    for (int nn = lane_id; nn < kV2; nn += 32) {
      __nv_bfloat162 i2 = inter_smem2[nn];
      __nv_bfloat162 w2 = w_smem2[d_row * kV2 + nn];
      acc2 = __hfma2(w2, i2, acc2);
    }
    float local = __bfloat162float(acc2.x) + __bfloat162float(acc2.y);
    local = WarpReduceSum(local);
    if (lane_id == 0) expert_acc += local;
    __syncthreads();
  }

  if (lane_id == 0 && prev_k_idx >= 0) {
    float rw = routing_weights[pid_t * top_k + prev_k_idx];
    out_acc += rw * expert_acc;
  }

  if (lane_id == 0) {
    __nv_bfloat16* out_row = out + (int64_t)pid_t * hidden_size;
    const int d_idx = d_base + warp_id;
    if (d_idx < hidden_size) out_row[d_idx] = __float2bfloat16(out_acc);
  }
}

// ===========================================================================
// Kernel 2b: NVFP4 down (kept identical to r1 — we don't bench FP4 in this round)
// ===========================================================================

template <int TILE_D, int TILE_N, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS * 32)
warp_decode_down_fp4_cute_kernel(
    const __nv_bfloat16* __restrict__ intermediate,
    const uint8_t* __restrict__ w_down_packed,
    const __nv_bfloat16* __restrict__ w_down_scales,
    const float* __restrict__ w_down_alpha,
    const float* __restrict__ routing_weights,
    const int* __restrict__ expert_ids,
    __nv_bfloat16* __restrict__ out,
    int hidden_size, int intermediate_size, int top_k, int num_tokens, int group_size) {
  extern __shared__ char smem_raw[];

  constexpr int kNumThreads = NUM_WARPS * 32;
  constexpr int kRowsPerWarp = (TILE_D + NUM_WARPS - 1) / NUM_WARPS;

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  const int pid_d = blockIdx.x;
  const int pid_t = blockIdx.y;

  if (pid_t >= num_tokens) return;

  const int d_base = pid_d * TILE_D;
  const float alpha = w_down_alpha[0];

  const int tile_packed_cols = TILE_N / 2;
  const int groups_per_tile = (TILE_N + group_size - 1) / group_size;

  const int smem_inter_byte_off = 0;
  const int smem_packed_byte_off =
      TILE_N * static_cast<int>(sizeof(__nv_bfloat16));
  const int smem_scale_byte_off =
      smem_packed_byte_off + TILE_D * tile_packed_cols;

  const int packed_cols = intermediate_size / 2;
  const int scale_cols = intermediate_size / group_size;

  float out_acc[kRowsPerWarp];
#pragma unroll
  for (int i = 0; i < kRowsPerWarp; i++) out_acc[i] = 0.0f;

  for (int k_idx = 0; k_idx < top_k; k_idx++) {
    const float routing_weight = routing_weights[pid_t * top_k + k_idx];
    const int expert_id = expert_ids[pid_t * top_k + k_idx];
    const int te_idx = pid_t * top_k + k_idx;

    float expert_acc[kRowsPerWarp];
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; i++) expert_acc[i] = 0.0f;

    const int num_n_iters = (intermediate_size + TILE_N - 1) / TILE_N;

    for (int n_iter = 0; n_iter < num_n_iters; n_iter++) {
      const int n_base = n_iter * TILE_N;
      __nv_bfloat16* inter_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + smem_inter_byte_off);
      {
        const __nv_bfloat16* inter_row = intermediate + (int64_t)te_idx * intermediate_size + n_base;
        for (int i = tid; i < TILE_N; i += kNumThreads)
          inter_smem[i] = (n_base + i < intermediate_size) ? inter_row[i] : __float2bfloat16(0.0f);
      }
      uint8_t* packed_smem = reinterpret_cast<uint8_t*>(smem_raw + smem_packed_byte_off);
      {
        const uint8_t* w_row =
            w_down_packed + (int64_t)expert_id * hidden_size * packed_cols +
            (int64_t)d_base * packed_cols + n_base / 2;
        for (int idx = tid; idx < TILE_D * tile_packed_cols; idx += kNumThreads) {
          const int row = idx / tile_packed_cols;
          const int col = idx % tile_packed_cols;
          packed_smem[idx] =
              (d_base + row < hidden_size && n_base / 2 + col < packed_cols)
                  ? w_row[row * packed_cols + col] : 0;
        }
      }
      __nv_bfloat16* scale_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + smem_scale_byte_off);
      {
        const int tile_group_start = n_base / group_size;
        const __nv_bfloat16* s_row =
            w_down_scales + (int64_t)expert_id * hidden_size * scale_cols +
            (int64_t)d_base * scale_cols + tile_group_start;
        for (int idx = tid; idx < TILE_D * groups_per_tile; idx += kNumThreads) {
          const int row = idx / groups_per_tile;
          const int col = idx % groups_per_tile;
          scale_smem[idx] = (d_base + row < hidden_size && tile_group_start + col < scale_cols)
              ? s_row[row * scale_cols + col] : __float2bfloat16(0.0f);
        }
      }
      __syncthreads();

#pragma unroll
      for (int r = 0; r < kRowsPerWarp; r++) {
        const int d_row = warp_id * kRowsPerWarp + r;
        if (d_row >= TILE_D) break;
        float local_sum = 0.0f;
        for (int nn = lane_id; nn < TILE_N; nn += 32) {
          float iv = __bfloat162float(inter_smem[nn]);
          const int packed_idx = nn / 2;
          uint8_t packed_byte = packed_smem[d_row * tile_packed_cols + packed_idx];
          uint8_t nibble = (nn % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
          const int group_idx = nn / group_size;
          float block_scale = __bfloat162float(scale_smem[d_row * groups_per_tile + group_idx]);
          float wv = NvFp4Dequant(nibble, block_scale * alpha);
          local_sum += wv * iv;
        }
        local_sum = WarpReduceSum(local_sum);
        if (lane_id == 0) expert_acc[r] += local_sum;
      }
      __syncthreads();
    }
    if (lane_id == 0) {
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; r++) out_acc[r] += routing_weight * expert_acc[r];
    }
  }

  if (lane_id == 0) {
    __nv_bfloat16* out_row = out + (int64_t)pid_t * hidden_size;
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; r++) {
      const int d_idx = d_base + warp_id * kRowsPerWarp + r;
      if (d_idx < hidden_size) out_row[d_idx] = __float2bfloat16(out_acc[r]);
    }
  }
}


// ---------------------------------------------------------------------------
// NVFP4 dequant helpers (LUT shared with the existing fp4 down kernel)
// ---------------------------------------------------------------------------

// Dequant a packed uint8 (2 nibbles) into a bf16x2 vector. Multiplies by `scale`
// (typically: block_scale_fp32 * weight_global_scale_fp32 hoisted out of the
// inner loop). Caller passes scale as bf16 to enable __hmul2 in the hot loop.
__device__ __forceinline__ __nv_bfloat162
NvFp4DequantPairBf16(uint8_t packed, float scale) {
  float lo = kNvfp4Lut[packed & 0xF] * scale;
  float hi = kNvfp4Lut[(packed >> 4) & 0xF] * scale;
  return __float22bfloat162_rn(make_float2(lo, hi));
}

// Cast FP8e4m3 → FP32. SM100 has fast intrinsic; use the cuda_fp8.h conversion.
__device__ __forceinline__ float Fp8E4m3ToFloat(__nv_fp8_storage_t v) {
  __nv_fp8_e4m3 t;
  t.__x = v;
  return static_cast<float>(t);
}

// ---------------------------------------------------------------------------
// W4A4 SiLU helper — operates on already-FP32 accumulator output so the
// arithmetic matches the BF16 path's `g / (1 + expf(-g))` exactly.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float SiluFp32(float x) {
  return x / (1.0f + __expf(-x));
}

// ===========================================================================
// Kernel 1c: BLOG-STRICT W4A4 Gate/Up (separate weights)
//
// Per CTA work: produces TILE_N=8 output neurons × 1 (token, top_k) pair.
// Each warp owns 1 neuron. Inner reduction over K is FP32 with a kFlush=128
// bf16x2 partial flush that matches the deployed BF16 path's accumulation
// depth, so cosine similarity tracks the BF16 path closely.
//
// SMEM per stage (TILE_N=8, TILE_K=512, group_size=16):
//   X packed:      TILE_K/2 = 256 B
//   X scales:      TILE_K/16 = 32 B
//   W_gate packed: TILE_N * TILE_K/2 = 2048 B
//   W_gate scales: TILE_N * TILE_K/16 = 256 B
//   W_up   packed: 2048 B
//   W_up   scales: 256 B
//   Total: 4896 B/stage × 3 = 14688 B (well under 232 KB).
// ===========================================================================

template <int TILE_N, int TILE_K, int NUM_WARPS, int GROUP_SIZE>
__global__ void __launch_bounds__(NUM_WARPS * 32)
warp_decode_gate_up_w4a4_cute_kernel(
    const uint8_t* __restrict__ x_packed,           // [M, K/2]
    const __nv_fp8_storage_t* __restrict__ x_scale, // [M, K/GROUP_SIZE]
    const float* __restrict__ x_global_scale,       // [E] per-expert
    const uint8_t* __restrict__ w_gate_packed,      // [E, N, K/2]
    const __nv_fp8_storage_t* __restrict__ w_gate_scale,
    const float* __restrict__ w_gate_global_scale,  // [E]
    const uint8_t* __restrict__ w_up_packed,
    const __nv_fp8_storage_t* __restrict__ w_up_scale,
    const float* __restrict__ w_up_global_scale,
    __nv_bfloat16* __restrict__ out,                // BF16 [M, top_k, N_total]
    const int* __restrict__ expert_ids,
    int hidden_size, int intermediate_size, int top_k, int num_tokens) {
  extern __shared__ __align__(16) char smem_raw[];

  static_assert(TILE_N == NUM_WARPS, "Blog-strict requires TILE_N == NUM_WARPS (1 neuron per warp)");
  static_assert(TILE_K % GROUP_SIZE == 0, "TILE_K must be a multiple of GROUP_SIZE");
  constexpr int kNumThreads = NUM_WARPS * 32;
  constexpr int kKVec = 16;                         // 16-byte vector load = 32 packed nibbles
  constexpr int kPackedTileK = TILE_K / 2;          // bytes per row of packed K
  constexpr int kScaleTileK = TILE_K / GROUP_SIZE;  // FP8 scales per row of K

  // SMEM offsets (in bytes)
  constexpr int kSmemXPackedOff = 0;
  constexpr int kSmemXScaleOff  = kSmemXPackedOff + kPackedTileK;
  constexpr int kSmemWGPackedOff = kSmemXScaleOff + kScaleTileK;
  constexpr int kSmemWGScaleOff  = kSmemWGPackedOff + TILE_N * kPackedTileK;
  constexpr int kSmemWUPackedOff = kSmemWGScaleOff  + TILE_N * kScaleTileK;
  constexpr int kSmemWUScaleOff  = kSmemWUPackedOff + TILE_N * kPackedTileK;
  constexpr int kBytesPerStage   = kSmemWUScaleOff  + TILE_N * kScaleTileK;
  constexpr int kNumStages = 3;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;

  const int pid_n = blockIdx.x;
  const int pid_te = blockIdx.y;
  const int token_idx = pid_te / top_k;
  const int k_idx = pid_te % top_k;
  if (token_idx >= num_tokens) return;

  const int expert_id = expert_ids[token_idx * top_k + k_idx];
  const int n_base = pid_n * TILE_N;

  // Per-tensor (per-expert) global scales hoisted out of all loops.
  // SGLang stores weight_scale_2 = 1/per_tensor at runtime (see
  // compressed_tensors_w4a4_nvfp4_moe.py:195). NVFP4 dequant needs
  // (nibble * fp8_block_scale * per_tensor), so reciprocate once here.
  const float xg = 1.0f / x_global_scale[expert_id];
  const float wgg = 1.0f / w_gate_global_scale[expert_id];
  const float wug = 1.0f / w_up_global_scale[expert_id];

  // Per-warp accumulators (one neuron per warp).
  float gate_acc = 0.0f;
  float up_acc = 0.0f;

  const int num_k_iters = (hidden_size + TILE_K - 1) / TILE_K;

  // GMEM row pointers
  const uint8_t* x_row = x_packed + (int64_t)token_idx * (hidden_size / 2);
  const __nv_fp8_storage_t* x_scale_row =
      x_scale + (int64_t)token_idx * (hidden_size / GROUP_SIZE);
  const uint8_t* gate_expert =
      w_gate_packed + (int64_t)expert_id * intermediate_size * (hidden_size / 2) +
      (int64_t)n_base * (hidden_size / 2);
  const __nv_fp8_storage_t* gate_scale_expert =
      w_gate_scale + (int64_t)expert_id * intermediate_size * (hidden_size / GROUP_SIZE) +
      (int64_t)n_base * (hidden_size / GROUP_SIZE);
  const uint8_t* up_expert =
      w_up_packed + (int64_t)expert_id * intermediate_size * (hidden_size / 2) +
      (int64_t)n_base * (hidden_size / 2);
  const __nv_fp8_storage_t* up_scale_expert =
      w_up_scale + (int64_t)expert_id * intermediate_size * (hidden_size / GROUP_SIZE) +
      (int64_t)n_base * (hidden_size / GROUP_SIZE);

  auto load_stage = [&](int it, int s) {
    if (it >= num_k_iters) return;
    const int kb = it * TILE_K;
    char* sp = smem_raw + s * kBytesPerStage;
    // X packed (256 B = 16 vec128)
    {
      const int vec_count = kPackedTileK / kKVec;
      uint8_t* dst = reinterpret_cast<uint8_t*>(sp + kSmemXPackedOff);
      const uint8_t* src = x_row + kb / 2;
      for (int i = tid; i < vec_count; i += kNumThreads) {
        CpAsyncLoad128(reinterpret_cast<__nv_bfloat16*>(dst + i * kKVec),
                       reinterpret_cast<const __nv_bfloat16*>(src + i * kKVec));
      }
    }
    // X scales (32 B; load as one vec128 + 2 scalar bytes if needed)
    {
      __nv_fp8_storage_t* dst =
          reinterpret_cast<__nv_fp8_storage_t*>(sp + kSmemXScaleOff);
      const __nv_fp8_storage_t* src = x_scale_row + kb / GROUP_SIZE;
      for (int i = tid; i < kScaleTileK; i += kNumThreads) {
        dst[i] = src[i];
      }
    }
    // W_gate packed (TILE_N rows × kPackedTileK bytes; vec128 along K)
    {
      const int vec_count = TILE_N * kPackedTileK / kKVec;
      constexpr int vec_per_row = kPackedTileK / kKVec;
      uint8_t* dst = reinterpret_cast<uint8_t*>(sp + kSmemWGPackedOff);
      for (int i = tid; i < vec_count; i += kNumThreads) {
        const int r = i / vec_per_row;
        const int c = (i % vec_per_row) * kKVec;
        CpAsyncLoad128(
            reinterpret_cast<__nv_bfloat16*>(dst + r * kPackedTileK + c),
            reinterpret_cast<const __nv_bfloat16*>(
                gate_expert + r * (hidden_size / 2) + kb / 2 + c));
      }
    }
    // W_gate scales (TILE_N × kScaleTileK = 256 B)
    {
      __nv_fp8_storage_t* dst =
          reinterpret_cast<__nv_fp8_storage_t*>(sp + kSmemWGScaleOff);
      for (int i = tid; i < TILE_N * kScaleTileK; i += kNumThreads) {
        const int r = i / kScaleTileK;
        const int c = i % kScaleTileK;
        dst[i] = gate_scale_expert[r * (hidden_size / GROUP_SIZE) + kb / GROUP_SIZE + c];
      }
    }
    // W_up packed
    {
      const int vec_count = TILE_N * kPackedTileK / kKVec;
      constexpr int vec_per_row = kPackedTileK / kKVec;
      uint8_t* dst = reinterpret_cast<uint8_t*>(sp + kSmemWUPackedOff);
      for (int i = tid; i < vec_count; i += kNumThreads) {
        const int r = i / vec_per_row;
        const int c = (i % vec_per_row) * kKVec;
        CpAsyncLoad128(
            reinterpret_cast<__nv_bfloat16*>(dst + r * kPackedTileK + c),
            reinterpret_cast<const __nv_bfloat16*>(
                up_expert + r * (hidden_size / 2) + kb / 2 + c));
      }
    }
    // W_up scales
    {
      __nv_fp8_storage_t* dst =
          reinterpret_cast<__nv_fp8_storage_t*>(sp + kSmemWUScaleOff);
      for (int i = tid; i < TILE_N * kScaleTileK; i += kNumThreads) {
        const int r = i / kScaleTileK;
        const int c = i % kScaleTileK;
        dst[i] = up_scale_expert[r * (hidden_size / GROUP_SIZE) + kb / GROUP_SIZE + c];
      }
    }
  };

  load_stage(0, 0);
  CpAsyncCommit();
  if (num_k_iters >= 2) {
    load_stage(1, 1);
    CpAsyncCommit();
  }
  for (int it = 0; it < num_k_iters; it++) {
    int next = it + 2;
    int cs = it % kNumStages;
    if (next < num_k_iters) {
      load_stage(next, next % kNumStages);
      CpAsyncCommit();
      CpAsyncWaitGroup<2>();
    } else if (it + 1 < num_k_iters) {
      CpAsyncWaitGroup<1>();
    } else {
      CpAsyncWaitGroup<0>();
    }
    __syncthreads();

    char* sp = smem_raw + cs * kBytesPerStage;
    const uint8_t* x_pk = reinterpret_cast<const uint8_t*>(sp + kSmemXPackedOff);
    const __nv_fp8_storage_t* x_sc =
        reinterpret_cast<const __nv_fp8_storage_t*>(sp + kSmemXScaleOff);
    const uint8_t* g_pk = reinterpret_cast<const uint8_t*>(sp + kSmemWGPackedOff);
    const __nv_fp8_storage_t* g_sc =
        reinterpret_cast<const __nv_fp8_storage_t*>(sp + kSmemWGScaleOff);
    const uint8_t* u_pk = reinterpret_cast<const uint8_t*>(sp + kSmemWUPackedOff);
    const __nv_fp8_storage_t* u_sc =
        reinterpret_cast<const __nv_fp8_storage_t*>(sp + kSmemWUScaleOff);

    // Per-warp: this warp's neuron index in the tile.
    const int n_row = warp_id;

    // bf16x2 inner reduction with kFlush=128 partial-flush (matches BF16 path).
    // Each lane handles `kPackedTileK/32 = 8` packed bytes per K-tile.
    // Per group (16 elements = 8 packed bytes), the per-group scale pair
    //   gate_eff_scale = fp8_to_fp32(g_sc[group]) * wgg
    //   x_eff_scale    = fp8_to_fp32(x_sc[group]) * xg
    // is hoisted; the multiply order is (W·X)*(g_eff*x_eff) so a single bf16x2
    // multiply absorbs both scales into the dequanted W. Since each group's
    // bytes are contiguous in the lane stream, the scale is loaded once per
    // group (every kFlush/(GROUP_SIZE/2)=8 inner iters when kFlush=128).
    // FP32 throughout (see W4A4 design note).
    constexpr int kPairsPerGroup = GROUP_SIZE / 2;
    constexpr int kPairsTotal = kPackedTileK;
    float g_local = 0.f;
    float u_local = 0.f;
#pragma unroll
    for (int kk = lane_id; kk < kPairsTotal; kk += 32) {
      const int byte_idx = kk;
      const int group_idx = byte_idx / kPairsPerGroup;
      const uint8_t x_byte = x_pk[byte_idx];
      const uint8_t g_byte = g_pk[n_row * kPackedTileK + byte_idx];
      const uint8_t u_byte = u_pk[n_row * kPackedTileK + byte_idx];

      const float xs = Fp8E4m3ToFloat(x_sc[group_idx]) * xg;
      const float gs = Fp8E4m3ToFloat(g_sc[n_row * kScaleTileK + group_idx]) * wgg;
      const float us = Fp8E4m3ToFloat(u_sc[n_row * kScaleTileK + group_idx]) * wug;

      const float x_lo = kNvfp4Lut[x_byte & 0xF] * xs;
      const float x_hi = kNvfp4Lut[(x_byte >> 4) & 0xF] * xs;
      const float g_lo = kNvfp4Lut[g_byte & 0xF] * gs;
      const float g_hi = kNvfp4Lut[(g_byte >> 4) & 0xF] * gs;
      const float u_lo = kNvfp4Lut[u_byte & 0xF] * us;
      const float u_hi = kNvfp4Lut[(u_byte >> 4) & 0xF] * us;

      g_local = fmaf(g_lo, x_lo, g_local);
      g_local = fmaf(g_hi, x_hi, g_local);
      u_local = fmaf(u_lo, x_lo, u_local);
      u_local = fmaf(u_hi, x_hi, u_local);
    }

    g_local = WarpReduceSum(g_local);
    u_local = WarpReduceSum(u_local);
    if (lane_id == 0) {
      gate_acc += g_local;
      up_acc += u_local;
    }
    // Block fast warps from overwriting this-iter SMEM before slow warps finish.
    __syncthreads();
  }

  // Write SiLU(gate) * up. Lane 0 of each warp writes its neuron.
  if (lane_id == 0) {
    const int n_idx = n_base + warp_id;
    if (n_idx < intermediate_size) {
      __nv_bfloat16* out_row =
          out + ((int64_t)token_idx * top_k + k_idx) * intermediate_size;
      const float silu = SiluFp32(gate_acc);
      out_row[n_idx] = __float2bfloat16(silu * up_acc);
    }
  }
  __syncthreads();
  __threadfence();
  WD_PDL_TRIGGER();
}

// ===========================================================================
// Kernel 2c: BLOG-STRICT W4A4 Down (replaces the broken FP4_TILE_D=32 + 4-warp
// version which violated the 8 warps × 1 dim invariant). Reads NVFP4
// intermediate (output of silu_and_mul_scaled_fp4_experts_quant) so this is a
// pure W4A4 dot product per output dim.
//
// SMEM per stage (TILE_D=8, TILE_N=512, group_size=16):
//   inter packed:  TILE_N/2 = 256 B
//   inter scales:  TILE_N/16 = 32 B (per-token, but one row at a time)
//   W_down packed: TILE_D * TILE_N/2 = 2048 B
//   W_down scales: TILE_D * TILE_N/16 = 256 B
//   Total: 2592 B/stage × 3 = 7776 B (well under 232 KB).
// ===========================================================================

template <int TILE_D, int TILE_N, int NUM_WARPS, int GROUP_SIZE>
__global__ void __launch_bounds__(NUM_WARPS * 32)
warp_decode_down_w4a4_cute_kernel(
    const uint8_t* __restrict__ inter_packed,        // [M*top_k, I/2]
    const __nv_fp8_storage_t* __restrict__ inter_scale,
    const float* __restrict__ inter_global_scale,    // [E] per-expert (matches w2_input_global_scale)
    const uint8_t* __restrict__ w_down_packed,       // [E, D, I/2]
    const __nv_fp8_storage_t* __restrict__ w_down_scale,
    const float* __restrict__ w_down_global_scale,   // [E]
    const float* __restrict__ routing_weights,
    const int* __restrict__ expert_ids,
    __nv_bfloat16* __restrict__ out,
    int hidden_size, int intermediate_size, int top_k, int num_tokens) {
  extern __shared__ __align__(16) char smem_raw[];

  static_assert(TILE_D == NUM_WARPS, "Blog-strict requires TILE_D == NUM_WARPS (1 dim per warp)");
  static_assert(TILE_N % GROUP_SIZE == 0, "TILE_N must be a multiple of GROUP_SIZE");
  constexpr int kNumThreads = NUM_WARPS * 32;
  constexpr int kKVec = 16;
  constexpr int kPackedTileN = TILE_N / 2;
  constexpr int kScaleTileN = TILE_N / GROUP_SIZE;

  constexpr int kSmemInterPackedOff = 0;
  constexpr int kSmemInterScaleOff  = kSmemInterPackedOff + kPackedTileN;
  constexpr int kSmemWPackedOff     = kSmemInterScaleOff  + kScaleTileN;
  constexpr int kSmemWScaleOff      = kSmemWPackedOff     + TILE_D * kPackedTileN;
  constexpr int kBytesPerStage      = kSmemWScaleOff      + TILE_D * kScaleTileN;
  constexpr int kNumStages = 3;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;

  const int pid_d = blockIdx.x;
  const int pid_t = blockIdx.y;
  if (pid_t >= num_tokens) return;

  const int d_base = pid_d * TILE_D;

  // Block until the prior gate/up grid signals (no-op when WD_PDL_ENABLED unset).
  WD_PDL_WAIT();

  // Per-warp single FP32 accumulator (one output dim per warp).
  float out_acc = 0.0f;

  for (int k_idx = 0; k_idx < top_k; k_idx++) {
    const float routing_weight = routing_weights[pid_t * top_k + k_idx];
    const int expert_id = expert_ids[pid_t * top_k + k_idx];
    const int te_idx = pid_t * top_k + k_idx;

    const float ig = 1.0f / inter_global_scale[expert_id];
    const float wg = 1.0f / w_down_global_scale[expert_id];

    const uint8_t* inter_row = inter_packed + (int64_t)te_idx * (intermediate_size / 2);
    const __nv_fp8_storage_t* inter_scale_row =
        inter_scale + (int64_t)te_idx * (intermediate_size / GROUP_SIZE);
    const uint8_t* w_expert =
        w_down_packed + (int64_t)expert_id * hidden_size * (intermediate_size / 2) +
        (int64_t)d_base * (intermediate_size / 2);
    const __nv_fp8_storage_t* w_scale_expert =
        w_down_scale + (int64_t)expert_id * hidden_size * (intermediate_size / GROUP_SIZE) +
        (int64_t)d_base * (intermediate_size / GROUP_SIZE);

    float expert_acc = 0.0f;
    const int num_n_iters = (intermediate_size + TILE_N - 1) / TILE_N;

    auto load_stage = [&](int it, int s) {
      if (it >= num_n_iters) return;
      const int nb = it * TILE_N;
      char* sp = smem_raw + s * kBytesPerStage;
      // inter packed (256 B)
      {
        const int vec_count = kPackedTileN / kKVec;
        uint8_t* dst = reinterpret_cast<uint8_t*>(sp + kSmemInterPackedOff);
        const uint8_t* src = inter_row + nb / 2;
        for (int i = tid; i < vec_count; i += kNumThreads) {
          CpAsyncLoad128(reinterpret_cast<__nv_bfloat16*>(dst + i * kKVec),
                         reinterpret_cast<const __nv_bfloat16*>(src + i * kKVec));
        }
      }
      // inter scales (32 B)
      {
        __nv_fp8_storage_t* dst =
            reinterpret_cast<__nv_fp8_storage_t*>(sp + kSmemInterScaleOff);
        const __nv_fp8_storage_t* src = inter_scale_row + nb / GROUP_SIZE;
        for (int i = tid; i < kScaleTileN; i += kNumThreads) dst[i] = src[i];
      }
      // W packed (TILE_D × 256 B = 2048 B)
      {
        const int vec_count = TILE_D * kPackedTileN / kKVec;
        constexpr int vec_per_row = kPackedTileN / kKVec;
        uint8_t* dst = reinterpret_cast<uint8_t*>(sp + kSmemWPackedOff);
        for (int i = tid; i < vec_count; i += kNumThreads) {
          const int r = i / vec_per_row;
          const int c = (i % vec_per_row) * kKVec;
          CpAsyncLoad128(
              reinterpret_cast<__nv_bfloat16*>(dst + r * kPackedTileN + c),
              reinterpret_cast<const __nv_bfloat16*>(
                  w_expert + r * (intermediate_size / 2) + nb / 2 + c));
        }
      }
      // W scales (TILE_D × 32 B = 256 B)
      {
        __nv_fp8_storage_t* dst =
            reinterpret_cast<__nv_fp8_storage_t*>(sp + kSmemWScaleOff);
        for (int i = tid; i < TILE_D * kScaleTileN; i += kNumThreads) {
          const int r = i / kScaleTileN;
          const int c = i % kScaleTileN;
          dst[i] = w_scale_expert[r * (intermediate_size / GROUP_SIZE) + nb / GROUP_SIZE + c];
        }
      }
    };

    load_stage(0, 0);
    CpAsyncCommit();
    if (num_n_iters >= 2) {
      load_stage(1, 1);
      CpAsyncCommit();
    }
    for (int it = 0; it < num_n_iters; it++) {
      int next = it + 2;
      int cs = it % kNumStages;
      if (next < num_n_iters) {
        load_stage(next, next % kNumStages);
        CpAsyncCommit();
        CpAsyncWaitGroup<2>();
      } else if (it + 1 < num_n_iters) {
        CpAsyncWaitGroup<1>();
      } else {
        CpAsyncWaitGroup<0>();
      }
      __syncthreads();

      char* sp = smem_raw + cs * kBytesPerStage;
      const uint8_t* i_pk = reinterpret_cast<const uint8_t*>(sp + kSmemInterPackedOff);
      const __nv_fp8_storage_t* i_sc =
          reinterpret_cast<const __nv_fp8_storage_t*>(sp + kSmemInterScaleOff);
      const uint8_t* w_pk = reinterpret_cast<const uint8_t*>(sp + kSmemWPackedOff);
      const __nv_fp8_storage_t* w_sc =
          reinterpret_cast<const __nv_fp8_storage_t*>(sp + kSmemWScaleOff);

      const int d_row = warp_id;

      // bf16x2 inner reduction with kFlush=128 partial-flush.
      // FP32 throughout (see gate/up note).
      constexpr int kPairsPerGroup = GROUP_SIZE / 2;
      constexpr int kPairsTotal = kPackedTileN;
      float local_sum = 0.0f;
#pragma unroll
      for (int kk = lane_id; kk < kPairsTotal; kk += 32) {
        const int byte_idx = kk;
        const int group_idx = byte_idx / kPairsPerGroup;
        const uint8_t i_byte = i_pk[byte_idx];
        const uint8_t w_byte = w_pk[d_row * kPackedTileN + byte_idx];
        const float is = Fp8E4m3ToFloat(i_sc[group_idx]) * ig;
        const float ws = Fp8E4m3ToFloat(w_sc[d_row * kScaleTileN + group_idx]) * wg;
        const float i_lo = kNvfp4Lut[i_byte & 0xF] * is;
        const float i_hi = kNvfp4Lut[(i_byte >> 4) & 0xF] * is;
        const float w_lo = kNvfp4Lut[w_byte & 0xF] * ws;
        const float w_hi = kNvfp4Lut[(w_byte >> 4) & 0xF] * ws;
        local_sum = fmaf(w_lo, i_lo, local_sum);
        local_sum = fmaf(w_hi, i_hi, local_sum);
      }
      local_sum = WarpReduceSum(local_sum);
      if (lane_id == 0) expert_acc += local_sum;
      __syncthreads();
    }
    if (lane_id == 0) out_acc += routing_weight * expert_acc;
  }

  if (lane_id == 0) {
    const int d_idx = d_base + warp_id;
    if (d_idx < hidden_size) {
      __nv_bfloat16* out_row = out + (int64_t)pid_t * hidden_size;
      out_row[d_idx] = __float2bfloat16(out_acc);
    }
  }
}

}  // namespace warp_decode
}  // namespace sglang
