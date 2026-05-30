/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

==============================================================================
Fused (residual-add + RMSNorm + linear-layout NVFP4 quantize) kernel for the
DeepSeek-V3.2-REAP NVFP4 MoE deploy path on B200. Eliminates the BF16
hidden_states roundtrip between RMSNorm and fp4_quantize on the
non-allreduce-fusion path (small/medium batch decode, or when the closed
flashinfer allreduce-fusion is disabled).

Production target: hidden_size=7168, DP=TP=8, 58 MoE layers per decode step.

Math:
  1. r' = x + residual                   (fp32 fused-multiply-add on packed bf162)
  2. rms = rsqrt(eps + mean(r' * r'))    (CTA reduction of squared sums)
  3. y = (r' * rms) * weight             (BF16 PackedVec)
  4. fp4_y, sf_y = nvfp4_warp_quant(y)   (16 elts share one fp8_e4m3 scale)

Side-effects (mutate in place):
  * residual <- r'  (BF16, same layout as upstream fused_add_rmsnorm)

Outputs:
  * fp4_out:   [m, n/2]    uint8 (packed E2M1, 2 elts per byte)
  * sf_out:    [m, n/16]   fp8_e4m3 (row-major linear, no swizzling/padding)

Notes:
  * The cvt_warp_fp16_to_fp4 helper does `__shfl_xor_sync(..., 1)` to pair
    adjacent threads so 16 elements share one SF. Our iteration stride
    (colIdx = threadIdx.x + blockDim.x * iter) keeps thread (2k, 2k+1)
    pairs on adjacent PackedVecs in every iter, so the SF pairing matches.
  * MAX_VECS_PER_THREAD=2 statically bounds register usage. For H=7168 +
    blockDim=512: ceil(896/512)=2 vecs/thread. The launcher asserts.
  * cast_x_before_out_mul mirrors the existing `kCastXBeforeOutMul` flag
    on FusedAddRMSNormKernel — applied for HF-parity callers (Gemma,
    glm4 with that flag set).
==============================================================================*/

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include "nvfp4_quant.cuh"
// Pull in cvt_warp_fp16_to_fp4 from the standalone NVFP4 quant kernels
// header (it's defined as a SGL_DEVICE function template; including the
// header from here doesn't add any kernel symbols since the kernels in
// that header are __global__ template functions instantiated only by
// their own translation unit).
#include "nvfp4_quant_kernels.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace host;

// Per-thread cap on PackedVec count. Each PackedVec is 8 elements.
// Cap supports hidden_size up to 512 * 4 * 8 = 16384 with blockDim=512.
// Production target H=7168 falls comfortably inside (kVecsPerThreadCap=2 used).
constexpr int kFusedQuantMaxVecsPerThread = 4;

// Iter3 SMEM-staged variant: per-thread cap of 1 PackedVec — the cached
// (x+res) BF16 packed values live in shared memory between phase 1 and
// phase 2 instead of registers. Eliminates the cross-iteration register
// footprint that drove the m=256 regression in iter2. Active when
// blockDim >= numColVecs (one CTA covers the whole row in one pass).
constexpr int kFusedQuantWideMaxVecsPerThread = 1;

// Linear-layout SF address helper. Identical to the standalone
// nvfp4_quant_linear path so trtllm_fp4_block_scale_moe consumes the
// resulting [m, K/16] row-major fp8_e4m3 tensor unchanged.
template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
SGL_DEVICE uint8_t* cvt_quant_to_fp4_get_sf_out_offset_linear_fused(
    int rowIdx, int colIdx, int numCols, SFType* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;
    int32_t sf_cols = numCols / CVT_FP4_SF_VEC_SIZE;
    int64_t SFOffset = static_cast<int64_t>(mIdx) * sf_cols + kIdx;
    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
#endif
  return nullptr;
}

// Cast a 2-element bf16 packed value to float2 (and back) via the
// SGL trait. We use the trait's built-in cvt for both directions.
template <typename Type>
SGL_DEVICE float2 packed_to_f2(typename ::dtype_trait<Type>::packed_t v) {
  return device::cast<fp32x2_t>(v);
}
template <typename Type>
SGL_DEVICE typename ::dtype_trait<Type>::packed_t f2_to_packed(float2 v) {
  return device::cast<typename ::dtype_trait<Type>::packed_t>(v);
}

// Fused residual-add + RMSNorm + linear NVFP4 quantize.
//
// One CTA per row. Each CTA cooperatively normalizes `numCols` elements
// across `blockDim.x` threads. Each thread holds up to
// `kFusedQuantMaxVecsPerThread` PackedVecs (8 elts each) in registers
// across the two-phase load->normalize->quant pipeline.
template <class Type, bool kCastXBeforeOutMul = false, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fused_rmsnorm_to_fp4_linear(
#else
cvt_fused_rmsnorm_to_fp4_linear(
#endif
    int32_t numRows, int32_t numCols, Type* __restrict__ input,
    Type* __restrict__ residual, Type const* __restrict__ weight,
    float const* SFScale, uint32_t* fp4_out, uint32_t* SFout, float eps) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using packed_t = typename ::dtype_trait<Type>::packed_t;
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "PackedVec size mismatch");
  static_assert(CVT_FP4_ELTS_PER_THREAD == 8, "Expected 8 elts per FP4 vec");

  const float SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  const int numColVecs = numCols / CVT_FP4_ELTS_PER_THREAD;  // PackedVecs per row

  // Per-thread register buffer: fp32 normalized values (post add+rmsnorm,
  // pre-weight-multiply) cached across the two phases. We store the
  // packed bf16 representation of (x + residual) so the second phase
  // recovers them without re-reading global memory.
  //
  // 4 packed_t per PackedVec × kFusedQuantMaxVecsPerThread vecs/thread.
  packed_t cached_inp_res[kFusedQuantMaxVecsPerThread][4];
  PackedVec cached_weight[kFusedQuantMaxVecsPerThread];
  // Note on kCastXBeforeOutMul: when true, we also need the float2 sums
  // so the post-rms multiply rounds via fp32, mirroring the existing
  // fused_add_rmsnorm semantics. Stored in fp32 only when needed.
  float2 cached_f2[kFusedQuantMaxVecsPerThread][4];

  __shared__ float shared_inv_rms;  // single-value broadcast post-CTA-reduce

  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    PackedVec* p_in = reinterpret_cast<PackedVec*>(input) +
                      static_cast<int64_t>(rowIdx) * numColVecs;
    PackedVec* p_res = reinterpret_cast<PackedVec*>(residual) +
                       static_cast<int64_t>(rowIdx) * numColVecs;
    const PackedVec* p_weight = reinterpret_cast<const PackedVec*>(weight);

    // --- Phase 1: load x+res, accumulate squared sum, stash for phase 2 ---
    float thread_sq_sum = 0.0f;
    int local_count = 0;  // how many vecs this thread actually owns

    #pragma unroll 1
    for (int colIdx = threadIdx.x, iter = 0;
         colIdx < numColVecs;
         colIdx += blockDim.x, ++iter) {
      // Bounds: launcher must guarantee iter < kFusedQuantMaxVecsPerThread.
      PackedVec x_vec = p_in[colIdx];
      PackedVec r_vec = p_res[colIdx];
      cached_weight[iter] = p_weight[colIdx];

      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 xf = device::cast<fp32x2_t>(x_vec.elts[i]);
        float2 rf = device::cast<fp32x2_t>(r_vec.elts[i]);
        float2 sf = make_float2(xf.x + rf.x, xf.y + rf.y);
        thread_sq_sum += sf.x * sf.x + sf.y * sf.y;
        // Stash as bf16-packed (lossy) for phase 2 reload. The squared sum
        // above was computed in fp32 — that's the value RMS depends on.
        cached_inp_res[iter][i] = device::cast<packed_t>(sf);
        if constexpr (kCastXBeforeOutMul) {
          cached_f2[iter][i] = sf;
        }
      }
      local_count = iter + 1;
    }

    // Write residual <- (x + res) using the cached bf16 round. This is
    // the same lossy rounding the upstream fused_add_rmsnorm performs
    // before computing the rsqrt — keep it consistent so downstream
    // residual readers see identical bytes.
    {
      int iter = 0;
      for (int colIdx = threadIdx.x; colIdx < numColVecs;
           colIdx += blockDim.x, ++iter) {
        PackedVec out_v;
        #pragma unroll
        for (int i = 0; i < 4; ++i) out_v.elts[i] = cached_inp_res[iter][i];
        p_res[colIdx] = out_v;
      }
    }

    // --- CTA reduction: sum of squares -> rsqrt ---
    auto cg_warp = cooperative_groups::tiled_partition<32>(
        cooperative_groups::this_thread_block());
    float warp_sum = cooperative_groups::reduce(
        cg_warp, thread_sq_sum, cooperative_groups::plus<float>());

    __shared__ float warp_partials[32];  // 32 warps max @ blockDim=1024
    if (threadIdx.x % 32 == 0) {
      warp_partials[threadIdx.x / 32] = warp_sum;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      int nwarps = (blockDim.x + 31) / 32;
      float v = (threadIdx.x < nwarps) ? warp_partials[threadIdx.x] : 0.0f;
      float cta_sum = cooperative_groups::reduce(
          cg_warp, v, cooperative_groups::plus<float>());
      if (threadIdx.x == 0) {
        shared_inv_rms =
            rsqrtf(eps + cta_sum * (1.0f / static_cast<float>(numCols)));
      }
    }
    __syncthreads();
    const float inv_rms = shared_inv_rms;

    // --- Phase 2: normalize, weight-multiply, FP4 warp-quant, store ---
    int iter = 0;
    for (int colIdx = threadIdx.x; colIdx < numColVecs;
         colIdx += blockDim.x, ++iter) {
      // Rebuild the post-normalize PackedVec for cvt_warp_fp16_to_fp4.
      PackedVec y_vec;

      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 vf;
        if constexpr (kCastXBeforeOutMul) {
          vf = cached_f2[iter][i];
          // weight is BF16 — round through bf16 after rms multiply.
          float2 wf = device::cast<fp32x2_t>(cached_weight[iter].elts[i]);
          float2 normed = make_float2(vf.x * inv_rms, vf.y * inv_rms);
          packed_t rounded = device::cast<packed_t>(normed);
          float2 rf = device::cast<fp32x2_t>(rounded);
          y_vec.elts[i] = device::cast<packed_t>(
              make_float2(rf.x * wf.x, rf.y * wf.y));
        } else {
          vf = device::cast<fp32x2_t>(cached_inp_res[iter][i]);
          float2 wf = device::cast<fp32x2_t>(cached_weight[iter].elts[i]);
          y_vec.elts[i] = device::cast<packed_t>(make_float2(
              vf.x * wf.x * inv_rms, vf.y * wf.y * inv_rms));
        }
      }

      int64_t outOffset =
          static_cast<int64_t>(rowIdx) * numColVecs + colIdx;
      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset_linear_fused<
              uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(rowIdx, colIdx, numCols,
                                                   SFout);
      fp4_out[outOffset] =
          cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(y_vec, SFScaleVal, sf_out);
    }
    (void)local_count;  // silence unused warning when MAX==1
  }
#endif
}

// Iter3 wide+SMEM-staged variant. One PackedVec per thread (no inner
// loop), `cached_inp_res` (and `cached_f2` when kCastXBeforeOutMul) live
// in shared memory rather than registers so that occupancy at m>=256 is
// no longer pinned by per-thread reg footprint.
//
// blockDim is sized so that numColVecs <= blockDim — each active thread
// owns exactly one PackedVec. Threads with `threadIdx.x >= numColVecs`
// idle through both phases (they still participate in the CTA reduce
// with a zero contribution).
//
// SMEM usage at H=7168, blockDim=1024:
//   cached_inp_res_smem: 896 vecs * 4 * sizeof(packed_t) = 14 KB
//   cached_f2_smem    : 896 vecs * 4 * sizeof(float2)    = 28 KB (only
//                       when kCastXBeforeOutMul=true)
//   warp_partials     :  32 * 4 = 128 B
//   shared_inv_rms    :   4 B
//
// At kCastXBeforeOutMul=false (the production path) the per-CTA SMEM
// footprint is well under the B200 100 KB default, so 2 CTAs/SM are
// achievable and m=256 traffic-bound case is no longer dominated by reg
// spill.
template <class Type, bool kCastXBeforeOutMul = false, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(1024, 2) cvt_fused_rmsnorm_to_fp4_linear_wide(
#else
cvt_fused_rmsnorm_to_fp4_linear_wide(
#endif
    int32_t numRows, int32_t numCols, Type* __restrict__ input,
    Type* __restrict__ residual, Type const* __restrict__ weight,
    float const* SFScale, uint32_t* fp4_out, uint32_t* SFout, float eps) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using packed_t = typename ::dtype_trait<Type>::packed_t;
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "PackedVec size mismatch");
  static_assert(CVT_FP4_ELTS_PER_THREAD == 8, "Expected 8 elts per FP4 vec");

  const float SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  const int numColVecs = numCols / CVT_FP4_ELTS_PER_THREAD;
  const bool active = threadIdx.x < numColVecs;

  // Shared-memory staging of the per-thread cross-phase cache. One
  // packed_t[4] slot per active thread (16 B). When kCastXBeforeOutMul
  // is also true, a float2[4] (32 B) per thread for the fp32 pre-round
  // values needed for HF-parity weight multiply.
  //
  // Sized to blockDim.x (compile-time 1024) — `extern __shared__` not
  // needed.
  __shared__ packed_t cached_inp_res_smem[1024][4];
  __shared__ float2 cached_f2_smem[kCastXBeforeOutMul ? 1024 : 1][4];
  __shared__ float shared_inv_rms;

  // Weight is row-shared; cache locally in regs for the active threads.
  PackedVec cached_weight;
  if (active) {
    const PackedVec* p_weight = reinterpret_cast<const PackedVec*>(weight);
    cached_weight = p_weight[threadIdx.x];
  }

  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    PackedVec* p_in = reinterpret_cast<PackedVec*>(input) +
                      static_cast<int64_t>(rowIdx) * numColVecs;
    PackedVec* p_res = reinterpret_cast<PackedVec*>(residual) +
                       static_cast<int64_t>(rowIdx) * numColVecs;

    // --- Phase 1: load x+res, accumulate squared sum, stash in SMEM. ---
    float thread_sq_sum = 0.0f;

    if (active) {
      PackedVec x_vec = p_in[threadIdx.x];
      PackedVec r_vec = p_res[threadIdx.x];

      packed_t local_cache[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 xf = device::cast<fp32x2_t>(x_vec.elts[i]);
        float2 rf = device::cast<fp32x2_t>(r_vec.elts[i]);
        float2 sf = make_float2(xf.x + rf.x, xf.y + rf.y);
        thread_sq_sum += sf.x * sf.x + sf.y * sf.y;
        local_cache[i] = device::cast<packed_t>(sf);
        cached_inp_res_smem[threadIdx.x][i] = local_cache[i];
        if constexpr (kCastXBeforeOutMul) {
          cached_f2_smem[threadIdx.x][i] = sf;
        }
      }

      // Write residual <- (x + res) using the BF16-rounded cached value.
      PackedVec out_v;
      #pragma unroll
      for (int i = 0; i < 4; ++i) out_v.elts[i] = local_cache[i];
      p_res[threadIdx.x] = out_v;
    }

    // --- CTA reduction: sum of squares -> rsqrt. Idle threads add 0. ---
    auto cg_warp = cooperative_groups::tiled_partition<32>(
        cooperative_groups::this_thread_block());
    float warp_sum = cooperative_groups::reduce(
        cg_warp, thread_sq_sum, cooperative_groups::plus<float>());

    __shared__ float warp_partials[32];  // 32 warps max @ blockDim=1024
    if (threadIdx.x % 32 == 0) {
      warp_partials[threadIdx.x / 32] = warp_sum;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      int nwarps = (blockDim.x + 31) / 32;
      float v = (threadIdx.x < nwarps) ? warp_partials[threadIdx.x] : 0.0f;
      float cta_sum = cooperative_groups::reduce(
          cg_warp, v, cooperative_groups::plus<float>());
      if (threadIdx.x == 0) {
        shared_inv_rms =
            rsqrtf(eps + cta_sum * (1.0f / static_cast<float>(numCols)));
      }
    }
    __syncthreads();
    const float inv_rms = shared_inv_rms;

    // --- Phase 2: re-hydrate from SMEM, normalize, FP4 warp-quant. ---
    // Important: the `active` condition is uniform across each warp at
    // production H=7168 (numColVecs=896 is a multiple of 32, so warps
    // are either fully active or fully inactive). cvt_warp_fp16_to_fp4's
    // intra-warp `__shfl_xor_sync(uint32_t(-1), ..., 1)` only needs all
    // 32 lanes within the *same* warp to converge, which they do here.
    // Inactive warps simply skip both the re-hydrate and the cvt call.
    if (active) {
      PackedVec y_vec;
      packed_t local_cache[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        local_cache[i] = cached_inp_res_smem[threadIdx.x][i];
      }
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 vf;
        if constexpr (kCastXBeforeOutMul) {
          vf = cached_f2_smem[threadIdx.x][i];
          float2 wf = device::cast<fp32x2_t>(cached_weight.elts[i]);
          float2 normed = make_float2(vf.x * inv_rms, vf.y * inv_rms);
          packed_t rounded = device::cast<packed_t>(normed);
          float2 rf = device::cast<fp32x2_t>(rounded);
          y_vec.elts[i] = device::cast<packed_t>(
              make_float2(rf.x * wf.x, rf.y * wf.y));
        } else {
          vf = device::cast<fp32x2_t>(local_cache[i]);
          float2 wf = device::cast<fp32x2_t>(cached_weight.elts[i]);
          y_vec.elts[i] = device::cast<packed_t>(make_float2(
              vf.x * wf.x * inv_rms, vf.y * wf.y * inv_rms));
        }
      }

      int colIdx = threadIdx.x;
      int64_t outOffset =
          static_cast<int64_t>(rowIdx) * numColVecs + colIdx;
      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset_linear_fused<
              uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(rowIdx, colIdx, numCols,
                                                   SFout);
      fp4_out[outOffset] =
          cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(y_vec, SFScaleVal, sf_out);
    }
  }
#endif
}

template <typename T>
void invokeFusedRMSNormFP4QuantLinear(int m, int n, T* input, T* residual,
                                      T const* weight, float const* SFScale,
                                      int64_t* fp4_output, int32_t* SFOutput,
                                      float eps, bool cast_x_before_out_mul,
                                      bool useUE8M0, bool enable_pdl,
                                      int multiProcessorCount,
                                      DLDevice device) {
  // Iter3 dispatch: wide+SMEM variant for m >= kWideMinM, narrow iter2
  // variant otherwise. The wide variant uses blockDim=1024 so it covers
  // any hidden_size up to 1024 * 8 = 8192 in a single pass per row; at
  // production H=7168 there are 128 idle threads/CTA but the elimination
  // of the cross-phase register cache more than makes up for the loss.
  const int numColVecs = n / ELTS_PER_THREAD;
  constexpr int kWideBlockDim = 1024;
  // The wide variant relies on uniform-warp activity (each warp is
  // either fully active or fully inactive) so cvt_warp_fp16_to_fp4's
  // all-mask intra-warp shuffle stays convergent. Require numColVecs to
  // be a multiple of 32 (one full warp granularity).
  const bool can_use_wide =
      (numColVecs <= kWideBlockDim) && (numColVecs % 32 == 0);
  // Threshold derived from the iter2 bench: m=128 fused beats unfused by
  // ~0.32us/layer, m=256 fused regresses by 0.86us/layer. Wide variant's
  // sweet spot starts where reg-pressure dominates HBM amortization —
  // 256+ rows in the production decode mix. Override with
  // SGLANG_NVFP4_FUSED_WIDE_MIN_M=<N> for tuning sweeps.
  int kWideMinM = 192;
  if (const char* env = std::getenv("SGLANG_NVFP4_FUSED_WIDE_MIN_M")) {
    int v = std::atoi(env);
    if (v > 0) kWideMinM = v;
  }
  const bool use_wide = can_use_wide && (m >= kWideMinM);

  auto out_u32 = reinterpret_cast<uint32_t*>(fp4_output);
  auto sf_u32 = reinterpret_cast<uint32_t*>(SFOutput);

  if (use_wide) {
    dim3 block(kWideBlockDim);
    constexpr int numBlocksPerSM = 2;  // matches __launch_bounds__(1024, 2)
    dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));
    auto launcher = host::LaunchKernel(grid, block, device);
    launcher.enable_pdl(enable_pdl);

    if (useUE8M0) {
      if (cast_x_before_out_mul) {
        auto k = cvt_fused_rmsnorm_to_fp4_linear_wide<T, true, true>;
        launcher(k, m, n, input, residual, weight, SFScale, out_u32, sf_u32, eps);
      } else {
        auto k = cvt_fused_rmsnorm_to_fp4_linear_wide<T, false, true>;
        launcher(k, m, n, input, residual, weight, SFScale, out_u32, sf_u32, eps);
      }
    } else {
      if (cast_x_before_out_mul) {
        auto k = cvt_fused_rmsnorm_to_fp4_linear_wide<T, true, false>;
        launcher(k, m, n, input, residual, weight, SFScale, out_u32, sf_u32, eps);
      } else {
        auto k = cvt_fused_rmsnorm_to_fp4_linear_wide<T, false, false>;
        launcher(k, m, n, input, residual, weight, SFScale, out_u32, sf_u32, eps);
      }
    }
    return;
  }

  // Narrow path (iter2 kernel) — 8 elts/thread, block size capped at 512.
  int block_threads = std::min(int(n / ELTS_PER_THREAD), 512);
  dim3 block(block_threads);
  int const numBlocksPerSM = 2048 / block.x;
  dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

  int vecs_per_thread = (n / ELTS_PER_THREAD + block_threads - 1) /
                        block_threads;
  RuntimeCheck(vecs_per_thread <= kFusedQuantMaxVecsPerThread,
               "fused RMSNorm FP4: vecs_per_thread ", vecs_per_thread,
               " exceeds cap ", kFusedQuantMaxVecsPerThread);

  auto launcher = host::LaunchKernel(grid, block, device);
  launcher.enable_pdl(enable_pdl);

  if (useUE8M0) {
    if (cast_x_before_out_mul) {
      auto k = cvt_fused_rmsnorm_to_fp4_linear<T, true, true>;
      launcher(k, m, n, input, residual, weight, SFScale, out_u32, sf_u32, eps);
    } else {
      auto k = cvt_fused_rmsnorm_to_fp4_linear<T, false, true>;
      launcher(k, m, n, input, residual, weight, SFScale, out_u32, sf_u32, eps);
    }
  } else {
    if (cast_x_before_out_mul) {
      auto k = cvt_fused_rmsnorm_to_fp4_linear<T, true, false>;
      launcher(k, m, n, input, residual, weight, SFScale, out_u32, sf_u32, eps);
    } else {
      auto k = cvt_fused_rmsnorm_to_fp4_linear<T, false, false>;
      launcher(k, m, n, input, residual, weight, SFScale, out_u32, sf_u32, eps);
    }
  }
}

template void invokeFusedRMSNormFP4QuantLinear(int m, int n, half* input,
                                               half* residual,
                                               half const* weight,
                                               float const* SFScale,
                                               int64_t* fp4_output,
                                               int32_t* SFOutput, float eps,
                                               bool cast_x_before_out_mul,
                                               bool useUE8M0, bool enable_pdl,
                                               int multiProcessorCount,
                                               DLDevice device);

template void invokeFusedRMSNormFP4QuantLinear(int m, int n,
                                               __nv_bfloat16* input,
                                               __nv_bfloat16* residual,
                                               __nv_bfloat16 const* weight,
                                               float const* SFScale,
                                               int64_t* fp4_output,
                                               int32_t* SFOutput, float eps,
                                               bool cast_x_before_out_mul,
                                               bool useUE8M0, bool enable_pdl,
                                               int multiProcessorCount,
                                               DLDevice device);

inline int getSMVersionFusedQuant(int device_id) {
  int sm_major = 0, sm_minor = 0;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(
      &sm_major, cudaDevAttrComputeCapabilityMajor, device_id));
  RuntimeDeviceCheck(cudaDeviceGetAttribute(
      &sm_minor, cudaDevAttrComputeCapabilityMinor, device_id));
  return sm_major * 10 + sm_minor;
}

// Public entry. Mutates `residual` in place (writes x+res back). Writes
// FP4-packed activations and linear-layout E4M3 scales.
//
// Matching iter1 standalone linear: SF layout is row-major [m, n/16] fp8_e4m3
// (returned as int32-packed 4-wide so callers .view(torch.float8_e4m3fn)).
//
// Iter2 vector list:
//   - Primary: this fused kernel (saves BF16 hidden_states roundtrip)
//   - Secondary: enable_pdl=True lets this kernel slip under the shadow
//     of an immediately-preceding global store (allreduce store, or the
//     upstream attention output write). Host adapter chains
//     .enable_pdl(true) below.
void fused_rmsnorm_scaled_fp4_quant_linear_sm100a_sm120a(
    tvm::ffi::TensorView input, tvm::ffi::TensorView residual,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView input_sf,
    tvm::ffi::TensorView fp4_output, tvm::ffi::TensorView sf_output,
    double eps, bool cast_x_before_out_mul, bool enable_pdl) {
  RuntimeCheck(input.device().device_type == kDLCUDA,
               "input must be a CUDA tensor");
  RuntimeCheck(residual.device() == input.device(),
               "residual must be on the same device as input");
  RuntimeCheck(weight.device() == input.device(),
               "weight must be on the same device as input");
  RuntimeCheck(input_sf.device() == input.device(),
               "input_sf must be on the same device as input");
  RuntimeCheck(fp4_output.device() == input.device(),
               "fp4_output must be on the same device as input");
  RuntimeCheck(sf_output.device() == input.device(),
               "sf_output must be on the same device as input");
  RuntimeCheck(input.dim() == 2, "input must be 2D");
  RuntimeCheck(residual.dim() == 2, "residual must be 2D");
  RuntimeCheck(weight.dim() == 1, "weight must be 1D");
  RuntimeCheck(input_sf.numel() == 1,
               "input_sf must have exactly one element");
  RuntimeCheck(fp4_output.dim() == 2, "fp4_output must be 2D");
  RuntimeCheck(sf_output.dim() == 2, "sf_output must be 2D");
  RuntimeCheck(host::is_type<uint8_t>(fp4_output.dtype()),
               "fp4_output must be uint8");
  RuntimeCheck(host::is_type<int32_t>(sf_output.dtype()),
               "sf_output must be int32 (fp8 packed)");
  RuntimeCheck(host::is_type<float>(input_sf.dtype()),
               "input_sf must be float32");
  RuntimeCheck(host::is_type<fp16_t>(input.dtype()) ||
                   host::is_type<bf16_t>(input.dtype()),
               "input dtype must be fp16 or bf16");

  const int device_id = input.device().device_id;
  RuntimeCheck(getSMVersionFusedQuant(device_id) >= 100,
               "fused_rmsnorm_fp4_quant_linear requires sm100+");

  const int32_t m = static_cast<int32_t>(input.size(0));
  const int32_t n = static_cast<int32_t>(input.size(1));

  RuntimeCheck(residual.size(0) == m && residual.size(1) == n,
               "residual shape mismatch");
  RuntimeCheck(weight.size(0) == n, "weight shape mismatch");
  RuntimeCheck(fp4_output.size(0) == m && fp4_output.size(1) == n / 2,
               "fp4_output shape mismatch (expect [m, n/2])");
  RuntimeCheck(n % 16 == 0, "n must be multiple of 16");
  RuntimeCheck((n / 16) % 4 == 0,
               "n/16 must be multiple of 4 for int32-packed SF");
  RuntimeCheck(sf_output.size(0) == m && sf_output.size(1) == (n / 16) / 4,
               "sf_output shape mismatch (expect [m, n/64] int32)");

  const int multiProcessorCount =
      static_cast<int>(runtime::get_sm_count(device_id));
  const DLDevice device = input.device();

  auto sf_in_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out_ptr = static_cast<int32_t*>(sf_output.data_ptr());
  auto fp4_out_ptr = static_cast<int64_t*>(fp4_output.data_ptr());

  constexpr bool useUE8M0 = false;
  if (host::is_type<fp16_t>(input.dtype())) {
    auto in_ptr = reinterpret_cast<half*>(input.data_ptr());
    auto res_ptr = reinterpret_cast<half*>(residual.data_ptr());
    auto w_ptr = reinterpret_cast<half const*>(weight.data_ptr());
    invokeFusedRMSNormFP4QuantLinear(m, n, in_ptr, res_ptr, w_ptr, sf_in_ptr,
                                     fp4_out_ptr, sf_out_ptr,
                                     static_cast<float>(eps),
                                     cast_x_before_out_mul, useUE8M0,
                                     enable_pdl, multiProcessorCount, device);
  } else {
    auto in_ptr = reinterpret_cast<__nv_bfloat16*>(input.data_ptr());
    auto res_ptr = reinterpret_cast<__nv_bfloat16*>(residual.data_ptr());
    auto w_ptr = reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr());
    invokeFusedRMSNormFP4QuantLinear(m, n, in_ptr, res_ptr, w_ptr, sf_in_ptr,
                                     fp4_out_ptr, sf_out_ptr,
                                     static_cast<float>(eps),
                                     cast_x_before_out_mul, useUE8M0,
                                     enable_pdl, multiProcessorCount, device);
  }
}

// Iter2 Stretch goal (NOT implemented in this commit): vendor a copy of
// trtllm_fused_moe_kernel_launcher.cu inside SGLang's csrc tree, relax
// the MxE2m1 dtype check on the activation path, and rebuild via the
// SGLang JIT pipeline. That would unlock the in-CUBIN
// bmm_Bfloat16_E2m1E2m1_* variants (58 of them, already in
// fused_moe_trtllm_sm100.so) which consume BF16 activations directly
// and eliminate the entire fp4_quantize step — including this fused
// kernel. ETA multi-day; biggest possible win but separate workstream.
