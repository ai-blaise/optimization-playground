/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// LayerSplit CP memory-pool kernels for B200 (SM100).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#if !defined(USE_ROCM) && !defined(USE_MUSA)
#define WARP_SIZE 32
#include "pytorch_extension_utils.h"
#else
#include "pytorch_extension_utils_rocm.h"
#include "utils.h"
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

using VecType = uint4;  // 16 bytes = LDG.128 / STG.128
static constexpr int kVecBytes = sizeof(VecType);

// Adaptive rows_per_cta based on total CTA count. At low
// layer counts (2, 4) the r2 2D grid (rows_per_cta=8 → grid_x=16 at
// rows=128) severely under-utilized the 148 SMs:
//   - layers=2, rows=128: r2 grid=(16,2)=32 CTAs (22% SM coverage)
//   - layers=4, rows=128: r2 grid=(16,4)=64 CTAs (43%)
//   - layers=8, rows=128: r2 grid=(16,8)=128 CTAs (87%)
// Switching to rows_per_cta=4 doubles grid_x:
//   - layers=2: 64 CTAs   - layers=4: 128 CTAs   - layers=8: 256 CTAs
// And for very small total work, rows_per_cta=2 quadruples grid_x.
//
// Heuristic (empirically tuned on B200 SM100 at rows=128, gemma-style):
//   total_ctas_8 < 100:  rows_per_cta=4 (more CTAs hide latency better)
//   100 <= total_ctas_8: rows_per_cta=8 (saves launch overhead per CTA)
// At total_ctas_8 >= 1024 the difference is below noise floor and the
// 8-warp variant has slight edge from amortizing CTA epilogue cost.
static constexpr int kMaterializeWarpsSmall = 4;   // for low total work
static constexpr int kMaterializeWarpsLarge = 8;   // for high total work
static constexpr int kMaterializeThreadsSmall = kMaterializeWarpsSmall * WARP_SIZE;
static constexpr int kMaterializeThreadsLarge = kMaterializeWarpsLarge * WARP_SIZE;
// Placeholder for other code that references kMaterializeWarps directly.
static constexpr int kMaterializeWarps = kMaterializeWarpsSmall;
static constexpr int kMaterializeThreads = kMaterializeWarps * WARP_SIZE;

// Small-data threshold. Below this, the 148x256 vectorized kernel beats the
// direct cudaMemcpyAsync copy-engine path on B200's launch-latency-dominated
// LayerSplit staging payloads; at 1 MiB the copy-engine path is better.
static constexpr int64_t kSmallByteThreshold = 512 * 1024;
static constexpr const char* kB200CandidateEnv =
    "SGLANG_LAYERSPLIT_B200_CANDIDATE";
static constexpr const char* kSmallByteThresholdEnv =
    "SGLANG_LAYERSPLIT_STAGE_SMALL_BYTES";

namespace {

// ---------------------------------------------------------------------------
// Vectorized copy primitives (used by fused_materialize)
// ---------------------------------------------------------------------------

int64_t resolve_small_byte_threshold() {
    const char* env = std::getenv(kSmallByteThresholdEnv);
    if (env == nullptr || env[0] == '\0') {
        const char* candidate = std::getenv(kB200CandidateEnv);
        if (candidate == nullptr || candidate[0] == '\0' ||
            std::strcmp(candidate, "production") == 0 ||
            std::strcmp(candidate, "cp2_descriptor_1b200") == 0 ||
            std::strcmp(candidate, "cp4_descriptor_2b200") == 0 ||
            std::strcmp(candidate, "producer_stage_fusion") == 0 ||
            std::strcmp(candidate, "higgs_dense_kv_transfer_sizing") == 0 ||
            std::strcmp(candidate, "owner_local_validation") == 0) {
            return kSmallByteThreshold;
        }
        if (std::strcmp(candidate, "stage_copy_threshold_256k") == 0) {
            return 256 * 1024;
        }
        if (std::strcmp(candidate, "stage_copy_threshold_768k") == 0) {
            return 768 * 1024;
        }
        TORCH_CHECK(
            false,
            kB200CandidateEnv,
            " has unknown LayerSplit candidate '",
            candidate,
            "'.");
    }
    if (env == nullptr || env[0] == '\0') {
        return kSmallByteThreshold;
    }

    char* end = nullptr;
    errno = 0;
    const long long threshold = std::strtoll(env, &end, 10);
    TORCH_CHECK(
        errno == 0 && end != env && end != nullptr && *end == '\0' &&
            threshold >= 0,
        kSmallByteThresholdEnv,
        " must be a non-negative integer byte count, got '",
        env,
        "'.");
    return static_cast<int64_t>(threshold);
}

// 4-way ILP unrolled per-warp 16B vector copy.
// Each iteration issues 4 ld.global.nc, then 4 st.global.cg, allowing
// the L1TEX pipeline to keep 4 loads outstanding per lane and hide
// memory latency. Tail loop handles the remaining < 4*WARP_SIZE vecs.
__device__ __forceinline__ void
vec_copy_warp_unroll4(int lane_id, const void* __restrict__ src,
                      void* __restrict__ dst, int num_bytes) {
    const auto* __restrict__ s = reinterpret_cast<const VecType*>(src);
    auto* __restrict__ d = reinterpret_cast<VecType*>(dst);
    const int num_vecs = num_bytes / kVecBytes;

    int i = lane_id;
    for (; i + 3 * WARP_SIZE < num_vecs; i += 4 * WARP_SIZE) {
#if !defined(USE_ROCM) && !defined(USE_MUSA)
        VecType t0, t1, t2, t3;
        asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(t0.x), "=r"(t0.y), "=r"(t0.z), "=r"(t0.w)
                     : "l"(s + i) : "memory");
        asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(t1.x), "=r"(t1.y), "=r"(t1.z), "=r"(t1.w)
                     : "l"(s + i + WARP_SIZE) : "memory");
        asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(t2.x), "=r"(t2.y), "=r"(t2.z), "=r"(t2.w)
                     : "l"(s + i + 2 * WARP_SIZE) : "memory");
        asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(t3.x), "=r"(t3.y), "=r"(t3.z), "=r"(t3.w)
                     : "l"(s + i + 3 * WARP_SIZE) : "memory");
        asm volatile("st.global.cg.v4.b32 [%0], {%1,%2,%3,%4};"
                     : : "l"(d + i),
                         "r"(t0.x), "r"(t0.y), "r"(t0.z), "r"(t0.w) : "memory");
        asm volatile("st.global.cg.v4.b32 [%0], {%1,%2,%3,%4};"
                     : : "l"(d + i + WARP_SIZE),
                         "r"(t1.x), "r"(t1.y), "r"(t1.z), "r"(t1.w) : "memory");
        asm volatile("st.global.cg.v4.b32 [%0], {%1,%2,%3,%4};"
                     : : "l"(d + i + 2 * WARP_SIZE),
                         "r"(t2.x), "r"(t2.y), "r"(t2.z), "r"(t2.w) : "memory");
        asm volatile("st.global.cg.v4.b32 [%0], {%1,%2,%3,%4};"
                     : : "l"(d + i + 3 * WARP_SIZE),
                         "r"(t3.x), "r"(t3.y), "r"(t3.z), "r"(t3.w) : "memory");
#else
        d[i] = s[i];
        d[i + WARP_SIZE] = s[i + WARP_SIZE];
        d[i + 2 * WARP_SIZE] = s[i + 2 * WARP_SIZE];
        d[i + 3 * WARP_SIZE] = s[i + 3 * WARP_SIZE];
#endif
    }
    // Tail.
    for (; i < num_vecs; i += WARP_SIZE) {
#if !defined(USE_ROCM) && !defined(USE_MUSA)
        VecType tmp;
        asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w)
                     : "l"(s + i) : "memory");
        asm volatile("st.global.cg.v4.b32 [%0], {%1,%2,%3,%4};"
                     : : "l"(d + i),
                         "r"(tmp.x), "r"(tmp.y), "r"(tmp.z), "r"(tmp.w) : "memory");
#else
        d[i] = s[i];
#endif
    }
}

__device__ __forceinline__ void
vec_copy_warp_8b_unroll4(int lane_id, const void* __restrict__ src,
                         void* __restrict__ dst, int num_bytes) {
    const auto* __restrict__ s = reinterpret_cast<const uint64_t*>(src);
    auto* __restrict__ d = reinterpret_cast<uint64_t*>(dst);
    const int num_u64 = num_bytes / sizeof(uint64_t);

    int i = lane_id;
    for (; i + 3 * WARP_SIZE < num_u64; i += 4 * WARP_SIZE) {
#if !defined(USE_ROCM) && !defined(USE_MUSA)
        uint64_t t0, t1, t2, t3;
        asm volatile("ld.global.nc.b64 %0, [%1];" : "=l"(t0) : "l"(s + i) : "memory");
        asm volatile("ld.global.nc.b64 %0, [%1];" : "=l"(t1) : "l"(s + i + WARP_SIZE) : "memory");
        asm volatile("ld.global.nc.b64 %0, [%1];" : "=l"(t2) : "l"(s + i + 2 * WARP_SIZE) : "memory");
        asm volatile("ld.global.nc.b64 %0, [%1];" : "=l"(t3) : "l"(s + i + 3 * WARP_SIZE) : "memory");
        asm volatile("st.global.cg.b64 [%0], %1;" : : "l"(d + i), "l"(t0) : "memory");
        asm volatile("st.global.cg.b64 [%0], %1;" : : "l"(d + i + WARP_SIZE), "l"(t1) : "memory");
        asm volatile("st.global.cg.b64 [%0], %1;" : : "l"(d + i + 2 * WARP_SIZE), "l"(t2) : "memory");
        asm volatile("st.global.cg.b64 [%0], %1;" : : "l"(d + i + 3 * WARP_SIZE), "l"(t3) : "memory");
#else
        d[i] = s[i];
        d[i + WARP_SIZE] = s[i + WARP_SIZE];
        d[i + 2 * WARP_SIZE] = s[i + 2 * WARP_SIZE];
        d[i + 3 * WARP_SIZE] = s[i + 3 * WARP_SIZE];
#endif
    }
    for (; i < num_u64; i += WARP_SIZE) {
#if !defined(USE_ROCM) && !defined(USE_MUSA)
        uint64_t tmp;
        asm volatile("ld.global.nc.b64 %0, [%1];" : "=l"(tmp) : "l"(s + i) : "memory");
        asm volatile("st.global.cg.b64 [%0], %1;" : : "l"(d + i), "l"(tmp) : "memory");
#else
        d[i] = s[i];
#endif
    }
}

// ---------------------------------------------------------------------------
// Tiny-data fast-path kernel .
// 148 CTAs x 256 threads, vectorized 16B asm copy, no shared memory.
// ---------------------------------------------------------------------------

template <int kThreads>
__global__ void __launch_bounds__(kThreads) layersplit_small_copy_v5_kernel(
    const VecType* __restrict__ src,
    VecType* __restrict__ dst,
    int num_vecs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
#pragma unroll 4
    for (int i = tid; i < num_vecs; i += stride) {
#if !defined(USE_ROCM) && !defined(USE_MUSA)
        VecType tmp;
        asm volatile(
            "ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w)
            : "l"(src + i)
            : "memory");
        asm volatile(
            "st.global.cg.v4.b32 [%0], {%1,%2,%3,%4};"
            :
            : "l"(dst + i), "r"(tmp.x), "r"(tmp.y), "r"(tmp.z), "r"(tmp.w)
            : "memory");
#else
        dst[i] = src[i];
#endif
    }
}

template <int kThreads>
__global__ void __launch_bounds__(kThreads) layersplit_small_copy_v5_8b_kernel(
    const uint64_t* __restrict__ src,
    uint64_t* __restrict__ dst,
    int num_u64) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
#pragma unroll 4
    for (int i = tid; i < num_u64; i += stride) {
#if !defined(USE_ROCM) && !defined(USE_MUSA)
        uint64_t tmp;
        asm volatile("ld.global.nc.b64 %0, [%1];" : "=l"(tmp) : "l"(src + i) : "memory");
        asm volatile("st.global.cg.b64 [%0], %1;" : : "l"(dst + i), "l"(tmp) : "memory");
#else
        dst[i] = src[i];
#endif
    }
}

template <int kThreads>
__global__ void __launch_bounds__(kThreads) layersplit_small_copy_tail_kernel(
    const char* __restrict__ src,
    char* __restrict__ dst,
    int64_t num_bytes) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int64_t vec_bytes = (num_bytes / kVecBytes) * kVecBytes;
    const int64_t num_vecs = vec_bytes / kVecBytes;

    const auto* __restrict__ src_vec = reinterpret_cast<const VecType*>(src);
    auto* __restrict__ dst_vec = reinterpret_cast<VecType*>(dst);
    for (int64_t i = tid; i < num_vecs; i += stride) {
#if !defined(USE_ROCM) && !defined(USE_MUSA)
        VecType tmp;
        asm volatile(
            "ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w)
            : "l"(src_vec + i)
            : "memory");
        asm volatile(
            "st.global.cg.v4.b32 [%0], {%1,%2,%3,%4};"
            :
            : "l"(dst_vec + i), "r"(tmp.x), "r"(tmp.y), "r"(tmp.z), "r"(tmp.w)
            : "memory");
#else
        dst_vec[i] = src_vec[i];
#endif
    }

    for (int64_t i = vec_bytes + tid; i < num_bytes; i += stride) {
        dst[i] = src[i];
    }
}

// ---------------------------------------------------------------------------
// Multi-buffer fused materialize with 4-way ILP per-warp copy
// and adaptive rows_per_cta via template parameter.
// Template kRowsPerCta = warps per CTA = rows per CTA.
// ---------------------------------------------------------------------------

template <int kRowsPerCta>
__global__ void __launch_bounds__(kRowsPerCta * WARP_SIZE)
layersplit_fused_materialize_kernel(
    const uint64_t* __restrict__ src_ptrs,
    const uint64_t* __restrict__ dst_ptrs,
    int num_layers,
    int active_rows,
    int row_bytes) {

    const int layer_idx = blockIdx.y;
    if (layer_idx >= num_layers) return;

    const auto* __restrict__ src = reinterpret_cast<const char*>(src_ptrs[layer_idx]);
    auto* __restrict__ dst = reinterpret_cast<char*>(dst_ptrs[layer_idx]);

    const int row_base = blockIdx.x * kRowsPerCta;
    const int local_warp = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = row_base + local_warp;

    if (row >= active_rows) return;

    const char* row_src = src + static_cast<int64_t>(row) * row_bytes;
    char* row_dst = dst + static_cast<int64_t>(row) * row_bytes;

    // revert to r2's simple inner copy. The 4-way ILP unroll
    // version was tested and showed TIE/marginal-loss across all
    // configs; the scheduler already pipelines the simple loop well
    // enough that unrolling adds register pressure without reducing
    // L1TEX stall.
    if (row_bytes % kVecBytes == 0) {
        // Inline r2 vec_copy_warp.
        const auto* __restrict__ s = reinterpret_cast<const VecType*>(row_src);
        auto* __restrict__ d = reinterpret_cast<VecType*>(row_dst);
        const int num_vecs = row_bytes / kVecBytes;
#pragma unroll 4
        for (int i = lane_id; i < num_vecs; i += WARP_SIZE) {
#if !defined(USE_ROCM) && !defined(USE_MUSA)
            VecType tmp;
            asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w)
                         : "l"(s + i) : "memory");
            asm volatile("st.global.cg.v4.b32 [%0], {%1,%2,%3,%4};"
                         : : "l"(d + i),
                             "r"(tmp.x), "r"(tmp.y), "r"(tmp.z), "r"(tmp.w)
                         : "memory");
#else
            d[i] = s[i];
#endif
        }
    } else {
        const auto* __restrict__ s = reinterpret_cast<const VecType*>(row_src);
        auto* __restrict__ d = reinterpret_cast<VecType*>(row_dst);
        const int vec_bytes = (row_bytes / kVecBytes) * kVecBytes;
        const int num_vecs = vec_bytes / kVecBytes;
#pragma unroll 4
        for (int i = lane_id; i < num_vecs; i += WARP_SIZE) {
#if !defined(USE_ROCM) && !defined(USE_MUSA)
            VecType tmp;
            asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w)
                         : "l"(s + i) : "memory");
            asm volatile("st.global.cg.v4.b32 [%0], {%1,%2,%3,%4};"
                         : : "l"(d + i),
                             "r"(tmp.x), "r"(tmp.y), "r"(tmp.z), "r"(tmp.w)
                         : "memory");
#else
            d[i] = s[i];
#endif
        }
        for (int i = vec_bytes + lane_id; i < row_bytes; i += WARP_SIZE) {
            row_dst[i] = row_src[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: layersplit_compute_active_rows (UNCHANGED)
// ---------------------------------------------------------------------------

__global__ void layersplit_compute_active_rows_kernel(
    const int* __restrict__ per_layer_max,
    int* __restrict__ result,
    int num_layers) {

    __shared__ int smem[WARP_SIZE];

    int val = 0;
    for (int i = threadIdx.x; i < num_layers; i += blockDim.x) {
        val = max(val, per_layer_max[i]);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        val = (lane_id < num_warps) ? smem[lane_id] : 0;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val = max(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane_id == 0) {
            *result = val;
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

void layersplit_stage_for_broadcast(
    const at::Tensor& src,
    at::Tensor& dst,
    int64_t active_rows,
    int64_t row_bytes) {

    if (active_rows <= 0) return;

    TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(dst.is_cuda(), "dst must be a CUDA tensor");
    TORCH_CHECK(row_bytes > 0, "row_bytes must be positive");

    if (__builtin_expect(src.is_contiguous() && dst.is_contiguous(), 1)) {
        const int64_t total_bytes = active_rows * row_bytes;
        const int64_t small_byte_threshold = resolve_small_byte_threshold();
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        // path 1: small payloads use the custom 148x256 kernel.
        if (total_bytes <= small_byte_threshold && total_bytes >= kVecBytes &&
            (total_bytes % kVecBytes == 0)) {
            const int num_vecs = static_cast<int>(total_bytes / kVecBytes);
            layersplit_small_copy_v5_kernel<256><<<148, 256, 0, stream>>>(
                reinterpret_cast<const VecType*>(src.data_ptr()),
                reinterpret_cast<VecType*>(dst.data_ptr()),
                num_vecs);
            return;
        }
        if (total_bytes <= small_byte_threshold &&
            total_bytes >= sizeof(uint64_t) &&
            (total_bytes % sizeof(uint64_t) == 0)) {
            const int num_u64 = static_cast<int>(total_bytes / sizeof(uint64_t));
            layersplit_small_copy_v5_8b_kernel<256><<<148, 256, 0, stream>>>(
                reinterpret_cast<const uint64_t*>(src.data_ptr()),
                reinterpret_cast<uint64_t*>(dst.data_ptr()),
                num_u64);
            return;
        }
        if (total_bytes <= small_byte_threshold && total_bytes >= kVecBytes) {
            layersplit_small_copy_tail_kernel<256><<<148, 256, 0, stream>>>(
                reinterpret_cast<const char*>(src.data_ptr()),
                reinterpret_cast<char*>(dst.data_ptr()),
                total_bytes);
            return;
        }

        // Copy only the active row prefix; src and dst may be larger staging buffers.
        C10_CUDA_CHECK(cudaMemcpyAsync(
            dst.data_ptr(), src.data_ptr(), total_bytes,
            cudaMemcpyDeviceToDevice, stream));
        return;
    }

    at::Tensor src_active = src.narrow(0, 0, active_rows);
    at::Tensor dst_active = dst.narrow(0, 0, active_rows);
    dst_active.copy_(src_active, true);
}

void layersplit_fused_materialize(
    const at::Tensor& src_ptrs,
    const at::Tensor& dst_ptrs,
    int64_t num_layers,
    int64_t active_rows,
    int64_t row_bytes) {

    if (active_rows <= 0 || num_layers <= 0) return;

    TORCH_CHECK(src_ptrs.is_cuda(), "src_ptrs must be a CUDA tensor");
    TORCH_CHECK(dst_ptrs.is_cuda(), "dst_ptrs must be a CUDA tensor");
    TORCH_CHECK(src_ptrs.scalar_type() == at::kLong, "src_ptrs must be uint64");
    TORCH_CHECK(dst_ptrs.scalar_type() == at::kLong, "dst_ptrs must be uint64");
    TORCH_CHECK(row_bytes > 0, "row_bytes must be positive");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // pick rows_per_cta based on total work.
    // With rows_per_cta=8 the 2D grid is (ceil(active_rows/8), num_layers).
    // For layers in {2,4,8} with rows=128 this leaves SMs idle —
    // total_ctas_8 == num_layers * 16 in {32, 64, 128}, vs 148 SMs.
    // Switching to rows_per_cta=4 doubles CTA count to {64, 128, 256}.
    //
    // Empirical (B200 1965 MHz, rows=128):
    //   layers=2,4,8: 4-warp ties with 8-warp (low layers are
    //                 launch-overhead-bound, not occupancy-bound).
    //   layers=16:    8-warp wins 6-12% — the work is enough to keep
    //                 the 8-warp larger CTAs busy, and the lower CTA
    //                 count avoids epilogue overhead.
    //   layers=32:    near-tie, 4-warp slight win on average.
    //   layers=64:    8-warp wins ~5% (bandwidth-saturated, fewer
    //                 CTAs is better).
    //
    // Heuristic: use 4-warp when total_ctas_8 < 200 (i.e. low SM
    // utilization), else 8-warp.
    const int total_ctas_8 = static_cast<int>(num_layers) *
        ((static_cast<int>(active_rows) + kMaterializeWarpsLarge - 1) /
         kMaterializeWarpsLarge);

    if (total_ctas_8 < 150) {
        // 4-warp variant: more CTAs, better occupancy at small layers.
        const int rows_per_cta = kMaterializeWarpsSmall;
        const int grid_x = (active_rows + rows_per_cta - 1) / rows_per_cta;
        dim3 grid(grid_x, num_layers);
        dim3 block(kMaterializeThreadsSmall);
        layersplit_fused_materialize_kernel<kMaterializeWarpsSmall>
            <<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint64_t*>(src_ptrs.data_ptr()),
            reinterpret_cast<const uint64_t*>(dst_ptrs.data_ptr()),
            static_cast<int>(num_layers),
            static_cast<int>(active_rows),
            static_cast<int>(row_bytes));
    } else {
        // 8-warp variant: fewer CTAs, less launch overhead at large
        // layer counts where bandwidth saturation already gives us
        // the win.
        const int rows_per_cta = kMaterializeWarpsLarge;
        const int grid_x = (active_rows + rows_per_cta - 1) / rows_per_cta;
        dim3 grid(grid_x, num_layers);
        dim3 block(kMaterializeThreadsLarge);
        layersplit_fused_materialize_kernel<kMaterializeWarpsLarge>
            <<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint64_t*>(src_ptrs.data_ptr()),
            reinterpret_cast<const uint64_t*>(dst_ptrs.data_ptr()),
            static_cast<int>(num_layers),
            static_cast<int>(active_rows),
            static_cast<int>(row_bytes));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void layersplit_compute_active_rows(
    const at::Tensor& per_layer_max,
    at::Tensor& result) {

    TORCH_CHECK(per_layer_max.is_cuda(), "per_layer_max must be a CUDA tensor");
    TORCH_CHECK(result.is_cuda(), "result must be a CUDA tensor");
    TORCH_CHECK(per_layer_max.scalar_type() == at::kInt, "per_layer_max must be int32");
    TORCH_CHECK(result.scalar_type() == at::kInt, "result must be int32");

    const int num_layers = per_layer_max.numel();
    if (num_layers <= 0) return;

    const int threads = min(256, ((num_layers + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    layersplit_compute_active_rows_kernel<<<1, threads, 0, stream>>>(
        per_layer_max.data_ptr<int>(),
        result.data_ptr<int>(),
        num_layers);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// TORCH_LIBRARY registration
// ---------------------------------------------------------------------------

namespace layersplit_cute_ops {

void stage_for_broadcast_op(
    const at::Tensor& src,
    const at::Tensor& dst,
    int64_t active_rows,
    int64_t row_bytes) {
    layersplit_stage_for_broadcast(
        src, const_cast<at::Tensor&>(dst), active_rows, row_bytes);
}

void fused_materialize_op(
    const at::Tensor& src_ptrs,
    const at::Tensor& dst_ptrs,
    int64_t num_layers,
    int64_t active_rows,
    int64_t row_bytes) {
    layersplit_fused_materialize(
        src_ptrs, dst_ptrs, num_layers, active_rows, row_bytes);
}

void compute_active_rows_op(
    const at::Tensor& per_layer_max,
    const at::Tensor& result) {
    layersplit_compute_active_rows(
        per_layer_max, const_cast<at::Tensor&>(result));
}

}  // namespace layersplit_cute_ops

TORCH_LIBRARY_FRAGMENT(layersplit_cute, m) {
    m.def("stage_for_broadcast(Tensor src, Tensor dst, int active_rows, int row_bytes) -> ()");
    m.def("fused_materialize(Tensor src_ptrs, Tensor dst_ptrs, int num_layers, int active_rows, int row_bytes) -> ()");
    m.def("compute_active_rows(Tensor per_layer_max, Tensor result) -> ()");
}

TORCH_LIBRARY_IMPL(layersplit_cute, CUDA, m) {
    m.impl("stage_for_broadcast", &layersplit_cute_ops::stage_for_broadcast_op);
    m.impl("fused_materialize",   &layersplit_cute_ops::fused_materialize_op);
    m.impl("compute_active_rows", &layersplit_cute_ops::compute_active_rows_op);
}
