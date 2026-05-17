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
 *   2. TILE_K=1024 in gate/up with a kFlush=128 split-FP32 inner accumulator.
 *      The wider tile cuts K-loop iterations from 14 to 7 at D=7168;
 *      flushing the bf16x2 partial into FP32 every 128 elements preserves
 *      the same BF16 accumulation depth as a smaller-tile baseline so cosine
 *      similarity does not drift. The larger 3-stage tiles fit within
 *      B200/B300 shared memory capacity. Down kernel uses TILE_N=2048 with
 *      the same split-FP32 partial accumulation.
 *   3. 3-stage cp.async pipeline with proper N-1 prologue/epilogue
 *      (prefetch stages 0,1; wait_group<2> in steady state; wait_group<1>
 *      then <0> for the last two iterations).
 *   4. Optional gate/up→down PDL chain (cudaTriggerProgrammaticLaunchCompletion
 *      + cudaGridDependencySynchronize, host-side cudaLaunchKernelEx with
 *      cudaLaunchAttributeProgrammaticStreamSerialization). Gated by
 *      WD_PDL_ENABLED build flag; no-ops out when not defined.
 *
 * Invariants preserved:
 *   1. 4 warps × 1 neuron in gate/up.
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
  static constexpr int kTileN = 4;       // 4 neurons per CTA
  static constexpr int kTileK = 128;
  static constexpr int kNumWarps = 4;    // 4 warps -> 1 neuron each
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

__device__ __forceinline__ float Ex2ApproxFtz(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float RcpApproxFtz(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float FastSigmoid(float x) {
  return RcpApproxFtz(1.0f + Ex2ApproxFtz(-x * 1.4426950408889634f));
}

__device__ __forceinline__ float FastSilu(float x) {
  return x * FastSigmoid(x);
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

  static_assert(
      TILE_N == NUM_WARPS,
      "Blog-strict requires TILE_N == NUM_WARPS (1 neuron per warp)");
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
  const int token_idx = pid_te >> 3;
  const int k_idx = pid_te & 7;
  if (token_idx >= num_tokens) return;

  const int expert_id = expert_ids[token_idx * 8 + k_idx];
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
    const int te_idx = token_idx * 8 + k_idx;
    __nv_bfloat16* out_row = out + (int64_t)te_idx * intermediate_size;
    const int n_idx = n_base + warp_id;     // 1 neuron per warp
    if (n_idx < intermediate_size) {
      float g = gate_acc;
      out_row[n_idx] = __float2bfloat16(FastSilu(g) * up_acc);
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
  const int token_idx = pid_te >> 3;
  const int k_idx = pid_te & 7;
  if (token_idx >= num_tokens) return;

  const int expert_id = expert_ids[token_idx * 8 + k_idx];
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
    const int te_idx = token_idx * 8 + k_idx;
    __nv_bfloat16* out_row = out + (int64_t)te_idx * intermediate_size;
    const int n_idx = n_base + warp_id;
    if (n_idx < intermediate_size) {
      float g = gate_acc;
      out_row[n_idx] = __float2bfloat16(FastSilu(g) * up_acc);
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
    if (k_idx >= 8 || n_iter >= num_n_iters) return;
    const int n_base = n_iter * TILE_N;
    const int te_idx = pid_t * 8 + k_idx;
    const int expert_id = expert_ids[pid_t * 8 + k_idx];
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

  if (num_n_iters == 1) {
    // Target-shape fast path: top_k=8 and TILE_N covers the full intermediate
    // row, so each expert contributes one staged dot product.
    constexpr int kDNumStages = 3;
    load_stage(0, 0, 0);
    CpAsyncCommit();
    load_stage(1, 0, 1);
    CpAsyncCommit();

#pragma unroll
    for (int k_idx = 0; k_idx < 8; ++k_idx) {
      constexpr int kNumTopK = 8;
      const int compute_stage = k_idx % kDNumStages;
      const int next = k_idx + 2;
      if (next < kNumTopK) {
        load_stage(next, 0, next % kDNumStages);
        CpAsyncCommit();
        CpAsyncWaitGroup<2>();
      } else if (k_idx + 1 < kNumTopK) {
        CpAsyncWaitGroup<1>();
      } else {
        CpAsyncWaitGroup<0>();
      }
      __syncthreads();

      __nv_bfloat16* sp = smem + compute_stage * kElemsPerStage;
      const __nv_bfloat16* inter_smem = sp + kSmemInterOff;
      const __nv_bfloat16* w_smem = sp + kSmemWOff;

      const int d_row = warp_id;
      const __nv_bfloat162* inter_smem2 = reinterpret_cast<const __nv_bfloat162*>(inter_smem);
      const __nv_bfloat162* w_smem2 = reinterpret_cast<const __nv_bfloat162*>(w_smem);
      constexpr int kV2 = TILE_N / 2;
      constexpr int kFlush = (kV2 > 128) ? 128 : kV2;
      float local = 0.0f;
#pragma unroll
      for (int nn_base = 0; nn_base < kV2; nn_base += kFlush) {
        __nv_bfloat162 acc2 = {__float2bfloat16(0.f), __float2bfloat16(0.f)};
#pragma unroll
        for (int nn = lane_id; nn < kFlush; nn += 32) {
          const int gnn = nn_base + nn;
          __nv_bfloat162 i2 = inter_smem2[gnn];
          __nv_bfloat162 w2 = w_smem2[d_row * kV2 + gnn];
          acc2 = __hfma2(w2, i2, acc2);
        }
        local += __bfloat162float(acc2.x) + __bfloat162float(acc2.y);
      }
      local = WarpReduceSum(local);
      if (lane_id == 0) {
        float rw = routing_weights[pid_t * 8 + k_idx];
        out_acc += rw * local;
      }
      __syncthreads();
    }

    if (lane_id == 0) {
      __nv_bfloat16* out_row = out + (int64_t)pid_t * hidden_size;
      const int d_idx = d_base + warp_id;
      if (d_idx < hidden_size) out_row[d_idx] = __float2bfloat16(out_acc);
    }
    return;
  }

  const int total_iters = 8 * num_n_iters;

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
        float rw = routing_weights[pid_t * 8 + prev_k_idx];
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
    constexpr int kFlush = (kV2 > 128) ? 128 : kV2;
    float local = 0.0f;
#pragma unroll
    for (int nn_base = 0; nn_base < kV2; nn_base += kFlush) {
      __nv_bfloat162 acc2 = {__float2bfloat16(0.f), __float2bfloat16(0.f)};
#pragma unroll
      for (int nn = lane_id; nn < kFlush; nn += 32) {
        const int gnn = nn_base + nn;
        __nv_bfloat162 i2 = inter_smem2[gnn];
        __nv_bfloat162 w2 = w_smem2[d_row * kV2 + gnn];
        acc2 = __hfma2(w2, i2, acc2);
      }
      local += __bfloat162float(acc2.x) + __bfloat162float(acc2.y);
    }
    local = WarpReduceSum(local);
    if (lane_id == 0) expert_acc += local;
    __syncthreads();
  }

  if (lane_id == 0 && prev_k_idx >= 0) {
    float rw = routing_weights[pid_t * 8 + prev_k_idx];
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
        const __nv_bfloat16* inter_row =
            intermediate + (int64_t)te_idx * intermediate_size + n_base;
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

}  // namespace warp_decode
}  // namespace sglang
