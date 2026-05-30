/// \file nvfp4_indexer_quant.cuh
/// \brief Blackwell-only NVFP4/UE8M0 quantization for the NSA Indexer.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include "../gemm/nvfp4/nvfp4_quant.cuh"

// iter5 PRIMARY: WMMA candidate_score Stage B uses fp16 inputs into an fp32
// accumulator via mma.sync.aligned.m16n8k8 PTX. Pulls in the half-precision
// type so we can lay out smem_q and smem_k as __half and feed b32-packed
// register pairs into the mma instruction.
#include <cuda_fp16.h>

#if defined(SGLANG_ENABLE_HISA_SELECTOR_MEGAKERNEL)
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <cublasdx.hpp>
#include <deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh>
#endif
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <climits>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr int kIndexerHeadDim = 128;
constexpr int kNVFP4ValueBytes = kIndexerHeadDim / 2;
constexpr int kScaleBytes = 4;
constexpr float kE2M1Max = 6.0f;
#if defined(SGLANG_ENABLE_HISA_SELECTOR_MEGAKERNEL)
constexpr int kHISASelectorHeads = 64;
constexpr int kHISASelectorTileN = 128;
constexpr int kHISASelectorWideTileN = 192;
constexpr int kHISASelectorClusterSize = 4;

#define HISA_SELECTOR_CLUSTER_KERNEL \
  __global__ __launch_bounds__(128, 1) \
      __cluster_dims__(1, kHISASelectorClusterSize, 1)

using HISASelectorGemmF32 = decltype(
    cublasdx::Size<kHISASelectorHeads, kHISASelectorTileN, kIndexerHeadDim>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());
using HISASelectorWideGemmF32 = decltype(
    cublasdx::Size<kHISASelectorHeads, kHISASelectorWideTileN, kIndexerHeadDim>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());
#endif

struct NVFP4IndexerStoreParam {
  const void* __restrict__ input;
  void* __restrict__ cache;
  const void* __restrict__ indices;
  uint32_t num_tokens;
};

struct NVFP4IndexerQuantParam {
  const void* __restrict__ input;
  void* __restrict__ values;
  void* __restrict__ scales;
  uint32_t num_rows;
};

struct NVFP4IndexerDequantParam {
  const void* __restrict__ values;
  const void* __restrict__ scales;
  void* __restrict__ output;
  uint32_t num_rows;
};

struct NVFP4HISAMeanPoolParam {
  const void* __restrict__ cache;
  const void* __restrict__ page_table;
  const void* __restrict__ seq_lens;
  void* __restrict__ reps;
  uint32_t batch_size;
  uint32_t max_blocks;
  uint32_t page_table_stride;
};

struct NVFP4HISACandidateScoreParam {
  const void* __restrict__ q_values;
  const void* __restrict__ q_scales;
  const void* __restrict__ cache;
  const void* __restrict__ page_table;
  const void* __restrict__ seq_lens;
  const void* __restrict__ weights;
  const void* __restrict__ top_blocks;
  const void* __restrict__ token_to_batch_idx;
  void* __restrict__ logits;
  void* __restrict__ candidate_indices;
  uint32_t q_rows;
  uint32_t n_heads;
  uint32_t block_topk;
  uint32_t page_table_stride;
};

struct NVFP4HISABlockScoreParam {
  const void* __restrict__ q_values;
  const void* __restrict__ q_scales;
  const void* __restrict__ reps;
  const void* __restrict__ weights;
  const void* __restrict__ seq_lens;
  const void* __restrict__ token_to_batch_idx;
  void* __restrict__ block_scores;
  uint32_t q_rows;
  uint32_t n_heads;
  uint32_t max_blocks;
};

struct NVFP4HISABlockTopKParam {
  const void* __restrict__ block_scores;
  const void* __restrict__ block_counts;
  const void* __restrict__ block_topk_counts;
  void* __restrict__ top_blocks;
  uint32_t q_rows;
  uint32_t max_blocks;
  uint32_t block_topk;
};

struct NVFP4HISABlockTopKMapAllParam {
  const void* __restrict__ block_scores;
  const void* __restrict__ block_counts;
  const void* __restrict__ block_topk_counts;
  const void* __restrict__ prefix_lens;
  void* __restrict__ topk_indices;
  uint32_t q_rows;
  uint32_t max_blocks;
  uint32_t block_topk;
  uint32_t topk;
};

struct NVFP4HISACandidatePagesParam {
  const void* __restrict__ top_blocks;
  const void* __restrict__ page_table;
  const void* __restrict__ token_to_batch_idx;
  void* __restrict__ candidate_page_table;
  uint32_t q_rows;
  uint32_t block_topk;
  uint32_t page_table_stride;
};

struct NVFP4HISACandidateMaskParam {
  void* __restrict__ logits;
  const void* __restrict__ top_blocks;
  const void* __restrict__ prefix_lens;
  void* __restrict__ candidate_indices;
  uint32_t q_rows;
  uint32_t block_topk;
  uint32_t candidate_len;
};

struct NVFP4HISAMapTopKParam {
  const void* __restrict__ topk_positions;
  const void* __restrict__ top_blocks;
  const void* __restrict__ prefix_lens;
  void* __restrict__ topk_indices;
  uint32_t q_rows;
  uint32_t topk;
  uint32_t block_topk;
};

struct NVFP4HISAMapCandidatesParam {
  const void* __restrict__ top_blocks;
  const void* __restrict__ prefix_lens;
  void* __restrict__ topk_indices;
  uint32_t q_rows;
  uint32_t topk;
  uint32_t block_topk;
};

// Fused exact mask + radix top-k + candidate-position map for HISA.  This uses
// the same CUB-style float key transform as DeviceRadixSort, then refines the
// kth threshold byte-by-byte so boundary buckets are not approximate.

struct NVFP4HISAFusedMaskTopKMapParam {
  const void* __restrict__ logits;   // [q_rows, candidate_len] float
  const void* __restrict__ top_blocks;       // [q_rows, block_topk] int32
  const void* __restrict__ prefix_lens;      // [q_rows] int32
  void* __restrict__ topk_indices;           // [q_rows, topk] int32, written
  uint32_t q_rows;
  uint32_t topk;
  uint32_t block_topk;
  uint32_t candidate_len;
};

#if defined(SGLANG_ENABLE_HISA_SELECTOR_MEGAKERNEL)
struct NVFP4HISASelectorMegakernelParam {
  const void* __restrict__ q_values;        // [q_rows, 64, 64] packed NVFP4
  const void* __restrict__ q_scales;        // [q_rows, 64] UE8M0 scales
  const void* __restrict__ cache;
  const void* __restrict__ page_table;
  const void* __restrict__ seq_lens;        // [batch] int32
  const void* __restrict__ weights;         // [q_rows, 64] float
  const void* __restrict__ token_to_batch_idx;
  const void* __restrict__ rep_values;      // [batch * max_blocks, 64]
  const void* __restrict__ rep_scales;      // [batch * max_blocks]
  const void* __restrict__ block_counts;    // [q_rows] int32
  const void* __restrict__ block_topk_counts;  // [q_rows] int32
  void* __restrict__ topk_indices;          // [q_rows, topk] int32
  uint32_t q_rows;
  uint32_t batch_size;
  uint32_t n_heads;
  uint32_t max_blocks;
  uint32_t page_table_stride;
  uint32_t effective_block_topk;
  uint32_t topk;
  uint32_t block_score_capacity;
  uint32_t candidate_capacity;
};

struct NVFP4HISASelectorParallelSelectParam {
  const void* __restrict__ q_values;        // [q_rows, 64, 64] packed NVFP4
  const void* __restrict__ q_scales;        // [q_rows, 64] UE8M0 scales
  const void* __restrict__ seq_lens;        // [batch] int32
  const void* __restrict__ weights;         // [q_rows, 64] float
  const void* __restrict__ token_to_batch_idx;
  const void* __restrict__ rep_values;      // [batch * max_blocks, 64]
  const void* __restrict__ rep_scales;      // [batch * max_blocks]
  const void* __restrict__ block_counts;    // [q_rows] int32
  const void* __restrict__ block_topk_counts;  // [q_rows] int32
  void* __restrict__ selected_blocks;       // [q_rows, effective_block_topk] int32
  uint32_t q_rows;
  uint32_t batch_size;
  uint32_t n_heads;
  uint32_t max_blocks;
  uint32_t effective_block_topk;
  uint32_t block_score_capacity;
};

struct NVFP4HISASelectorParallelScoreParam {
  const void* __restrict__ q_values;        // [q_rows, 64, 64] packed NVFP4
  const void* __restrict__ q_scales;        // [q_rows, 64] UE8M0 scales
  const void* __restrict__ cache;
  const void* __restrict__ page_table;
  const void* __restrict__ seq_lens;        // [batch] int32
  const void* __restrict__ weights;         // [q_rows, 64] float
  const void* __restrict__ token_to_batch_idx;
  const void* __restrict__ selected_blocks; // [q_rows, effective_block_topk] int32
  void* __restrict__ logits;                // [q_rows, candidate_len] float
  uint32_t q_rows;
  uint32_t batch_size;
  uint32_t n_heads;
  uint32_t page_table_stride;
  uint32_t effective_block_topk;
  uint32_t candidate_len;
};

struct NVFP4HISASelectorClusterFusedParam {
  const void* __restrict__ q_values;        // [q_rows, 64, 64] packed NVFP4
  const void* __restrict__ q_scales;        // [q_rows, 64] UE8M0 scales
  const void* __restrict__ cache;
  const void* __restrict__ page_table;
  const void* __restrict__ seq_lens;        // [batch] int32
  const void* __restrict__ weights;         // [q_rows, 64] float
  const void* __restrict__ token_to_batch_idx;
  const void* __restrict__ rep_values;      // [batch * max_blocks, 64]
  const void* __restrict__ rep_scales;      // [batch * max_blocks]
  const void* __restrict__ block_counts;    // [q_rows] int32
  const void* __restrict__ block_topk_counts;  // [q_rows] int32
  void* __restrict__ candidate_keys;        // [q_rows, candidate_len] uint32 score-key scratch
  void* __restrict__ topk_indices;          // [q_rows, topk] int32
  uint32_t q_rows;
  uint32_t batch_size;
  uint32_t n_heads;
  uint32_t max_blocks;
  uint32_t page_table_stride;
  uint32_t effective_block_topk;
  uint32_t topk;
  uint32_t block_score_capacity;
  uint32_t candidate_len;
};

struct NVFP4HISACandidateKeysTopKMapParam {
  const void* __restrict__ candidate_keys;  // [q_rows, candidate_len] uint32 score keys
  const void* __restrict__ top_blocks;      // [q_rows, block_topk] int32
  void* __restrict__ topk_indices;          // [q_rows, topk] int32
  uint32_t q_rows;
  uint32_t topk;
  uint32_t block_topk;
  uint32_t candidate_len;
};
#endif

SGL_DEVICE uint32_t fp32_to_radix_desc(uint32_t bits) {
  const auto asc_key = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  return ~asc_key;
}

#if defined(SGLANG_ENABLE_HISA_SELECTOR_MEGAKERNEL)
inline PFN_cuTensorMapEncodeTiled_v12000 hisa_get_cu_tensor_map_encode_tiled() {
  static const auto fn = []() {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
    const cudaError_t err = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled",
        &ptr,
        12000,
        cudaEnableDefault,
        &status);
    if (err != cudaSuccess || ptr == nullptr ||
        status != cudaDriverEntryPointSuccess) {
      throw std::runtime_error(
          "Failed to resolve cuTensorMapEncodeTiled through CUDA runtime.");
    }
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}

inline CUtensorMap hisa_make_tma_desc_2d(
    const void* base,
    CUtensorMapDataType dtype,
    uint64_t gmem_dim0,
    uint64_t gmem_dim1,
    uint64_t gmem_stride0_bytes,
    uint32_t smem_dim0,
    uint32_t smem_dim1,
    CUtensorMapSwizzle swizzle) {
  CUtensorMap tensor_map;
  const cuuint64_t gmem_dims[2] = {gmem_dim0, gmem_dim1};
  const cuuint64_t gmem_strides[1] = {gmem_stride0_bytes};
  const cuuint32_t smem_dims[2] = {smem_dim0, smem_dim1};
  const cuuint32_t elem_strides[2] = {1, 1};
  const CUresult result = hisa_get_cu_tensor_map_encode_tiled()(
      &tensor_map,
      dtype,
      2,
      const_cast<void*>(base),
      gmem_dims,
      gmem_strides,
      smem_dims,
      elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error("cuTensorMapEncodeTiled failed for HISA 2D TMA descriptor.");
  }
  return tensor_map;
}

inline CUtensorMap hisa_make_tma_desc_3d(
    const void* base,
    CUtensorMapDataType dtype,
    uint64_t gmem_dim0,
    uint64_t gmem_dim1,
    uint64_t gmem_dim2,
    uint64_t gmem_stride0_bytes,
    uint64_t gmem_stride1_bytes,
    uint32_t smem_dim0,
    uint32_t smem_dim1,
    uint32_t smem_dim2,
    CUtensorMapSwizzle swizzle) {
  CUtensorMap tensor_map;
  const cuuint64_t gmem_dims[3] = {gmem_dim0, gmem_dim1, gmem_dim2};
  const cuuint64_t gmem_strides[2] = {gmem_stride0_bytes, gmem_stride1_bytes};
  const cuuint32_t smem_dims[3] = {smem_dim0, smem_dim1, smem_dim2};
  const cuuint32_t elem_strides[3] = {1, 1, 1};
  const CUresult result = hisa_get_cu_tensor_map_encode_tiled()(
      &tensor_map,
      dtype,
      3,
      const_cast<void*>(base),
      gmem_dims,
      gmem_strides,
      smem_dims,
      elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error("cuTensorMapEncodeTiled failed for HISA 3D TMA descriptor.");
  }
  return tensor_map;
}
#endif

__global__ void hisa_fused_mask_topk_map_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISAFusedMaskTopKMapParam param) {
  constexpr uint32_t kHISABlockSize = 128;
  constexpr uint32_t kBins = 256;
  constexpr uint32_t kThreads = 256;

  __shared__ uint32_t s_hist[kBins];
  __shared__ uint32_t s_prefix[kBins];
  __shared__ uint32_t s_threshold_key;
  __shared__ uint32_t s_less_count;
  __shared__ uint32_t s_boundary_quota;
  __shared__ uint32_t s_boundary_used;
  __shared__ uint32_t s_above_used;
  __shared__ uint32_t s_emit_count;

  const auto row = blockIdx.x;
  if (row >= param.q_rows) return;
  const auto tid = threadIdx.x;
  const auto keep = param.topk < param.candidate_len ? param.topk : param.candidate_len;
  const auto candidate_len = param.candidate_len;

  __shared__ int32_t s_top_blocks[256];
  if (tid < param.block_topk) {
    s_top_blocks[tid] = static_cast<const int32_t*>(param.top_blocks)[row * param.block_topk + tid];
  }
  __shared__ int32_t s_prefix_len;
  if (tid == 0) {
    s_prefix_len = static_cast<const int32_t*>(param.prefix_lens)[row];
    s_threshold_key = 0;
    s_less_count = 0;
    s_above_used = 0;
    s_boundary_used = 0;
    s_emit_count = 0;
  }
  __syncthreads();

  const auto* logits_row =
      static_cast<const float*>(param.logits) + static_cast<int64_t>(row) * candidate_len;
  auto* out_row =
      static_cast<int32_t*>(param.topk_indices) + static_cast<int64_t>(row) * param.topk;

  #pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    const auto shift = 24u - static_cast<uint32_t>(pass) * 8u;
    const auto prefix_mask =
        pass == 0 ? 0u : (0xffffffffu << (shift + 8u));
    const auto selected_prefix = s_threshold_key & prefix_mask;

    s_hist[tid] = 0;
    __syncthreads();

    for (uint32_t i = tid; i < candidate_len; i += kThreads) {
      const auto block_slot = i / kHISABlockSize;
      const auto block_offset = i - block_slot * kHISABlockSize;
      int32_t top_block = -1;
      if (block_slot < param.block_topk) top_block = s_top_blocks[block_slot];
      int32_t token = -1;
      if (top_block >= 0) {
        token = top_block * static_cast<int32_t>(kHISABlockSize) +
                static_cast<int32_t>(block_offset);
      }
      const bool valid = (token >= 0) && (token < s_prefix_len);
      const auto key =
          valid ? fp32_to_radix_desc(__float_as_uint(logits_row[i])) : 0xffffffffu;
      if (valid && (pass == 0 || ((key & prefix_mask) == selected_prefix))) {
        atomicAdd(&s_hist[(key >> shift) & 0xffu], 1u);
      }
    }
    __syncthreads();

    if (tid < 32) {
      uint32_t lane_sum = 0;
      for (uint32_t b = tid * 8; b < (tid + 1) * 8; ++b) lane_sum += s_hist[b];
      uint32_t scan = lane_sum;
      #pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const auto y = __shfl_up_sync(0xffffffff, scan, off);
        if (static_cast<int>(tid) >= off) scan += y;
      }
      const auto base = scan - lane_sum;
      uint32_t running = base;
      for (uint32_t b = tid * 8; b < (tid + 1) * 8; ++b) {
        running += s_hist[b];
        s_prefix[b] = running;
      }
    }
    __syncthreads();

    if (tid == 0) {
      const auto target = keep > s_less_count ? keep - s_less_count : 1u;
      uint32_t b = 0;
      while (b + 1 < kBins && s_prefix[b] < target) ++b;
      const auto before = b == 0 ? 0u : s_prefix[b - 1];
      s_less_count += before;
      s_threshold_key = selected_prefix | (b << shift);
    }
    __syncthreads();
  }

  if (tid == 0) {
    s_boundary_quota = keep > s_less_count ? keep - s_less_count : 0u;
    s_above_used = 0;
    s_boundary_used = 0;
  }
  __syncthreads();

  const auto threshold_key = s_threshold_key;
  const auto above_count = s_less_count;
  const auto boundary_quota = s_boundary_quota;

  for (uint32_t i = tid; i < candidate_len; i += kThreads) {
    const auto block_slot = i / kHISABlockSize;
    const auto block_offset = i - block_slot * kHISABlockSize;
    int32_t top_block = -1;
    if (block_slot < param.block_topk) top_block = s_top_blocks[block_slot];
    int32_t token = -1;
    if (top_block >= 0) {
      token = top_block * static_cast<int32_t>(kHISABlockSize) +
              static_cast<int32_t>(block_offset);
    }
    const bool valid = (token >= 0) && (token < s_prefix_len);
    const auto key =
        valid ? fp32_to_radix_desc(__float_as_uint(logits_row[i])) : 0xffffffffu;
    if (valid && key < threshold_key) {
      const auto slot = atomicAdd(&s_above_used, 1u);
      if (slot < above_count) out_row[slot] = token;
    } else if (valid && key == threshold_key) {
      const auto slot = atomicAdd(&s_boundary_used, 1u);
      if (slot < boundary_quota) out_row[above_count + slot] = token;
    }
  }
  __syncthreads();

  if (tid == 0) {
    const uint32_t boundary_written = min(s_boundary_used, boundary_quota);
    s_emit_count = min(keep, min(s_above_used, above_count) + boundary_written);
  }
  __syncthreads();

  for (uint32_t i = s_emit_count + tid; i < param.topk; i += kThreads) {
    out_row[i] = -1;
  }
}

SGL_DEVICE float reduce_max_width8(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 4, 8));
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 2, 8));
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 1, 8));
  return value;
}

SGL_DEVICE float reduce_max_width4(float value, uint32_t mask) {
  value = fmaxf(value, __shfl_xor_sync(mask, value, 2, 4));
  value = fmaxf(value, __shfl_xor_sync(mask, value, 1, 4));
  return value;
}

SGL_DEVICE uint32_t ceil_to_ue8m0_exp(float value) {
  const auto bits = __float_as_uint(fabsf(value));
  auto exp = (bits >> 23) & 0xffu;
  exp += (bits & 0x7fffffu) != 0;
  exp = max(1u, min(exp, 254u));
  return exp;
}

SGL_DEVICE uint32_t pack_scale_word(uint32_t scale_exp) {
  const auto e0 = __shfl_sync(0xffffffff, scale_exp, 0);
  const auto e1 = __shfl_sync(0xffffffff, scale_exp, 8);
  const auto e2 = __shfl_sync(0xffffffff, scale_exp, 16);
  const auto e3 = __shfl_sync(0xffffffff, scale_exp, 24);
  return e0 | (e1 << 8) | (e2 << 16) | (e3 << 24);
}

SGL_DEVICE uint32_t pack_half_warp_scale_word(
    uint32_t scale_exp, uint32_t mask) {
  const auto lane_id = threadIdx.x & 31;
  const auto row_base = lane_id & 16;
  const auto e0 = __shfl_sync(mask, scale_exp, row_base);
  const auto e1 = __shfl_sync(mask, scale_exp, row_base + 4);
  const auto e2 = __shfl_sync(mask, scale_exp, row_base + 8);
  const auto e3 = __shfl_sync(mask, scale_exp, row_base + 12);
  return e0 | (e1 << 8) | (e2 << 16) | (e3 << 24);
}

SGL_DEVICE uint32_t clear_e2m1_signed_zero(uint32_t packed) {
  const auto magnitude = packed & 0x77777777u;
  const auto nonzero =
      ((magnitude | (magnitude >> 1) | (magnitude >> 2)) & 0x11111111u) << 3;
  return packed & (0x77777777u | nonzero);
}

template <bool kBranchlessSign = false>
SGL_DEVICE float decode_e2m1_nibble(uint32_t code, float scale) {
  constexpr float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = values[code & 0x7u] * scale;
  if constexpr (kBranchlessSign) {
    const auto sign = (code & 0x8u) << 28;
    return __uint_as_float(__float_as_uint(magnitude) ^ sign);
  } else {
    return (code & 0x8u) ? -magnitude : magnitude;
  }
}

SGL_DEVICE float load_nvfp4_value(
    const uint8_t* __restrict__ value_ptr,
    const uint32_t* __restrict__ scale_ptr,
    uint32_t dim) {
  const auto packed = value_ptr[dim >> 1];
  const auto code = (dim & 1) ? (packed >> 4) : (packed & 0xfu);
  const auto scale_word = *scale_ptr;
  const auto scale_exp = (scale_word >> ((dim >> 5) * 8)) & 0xffu;
  const auto scale = __uint_as_float(scale_exp << 23);
  return decode_e2m1_nibble(code, scale);
}

SGL_DEVICE float warp_sum(float value) {
  value += __shfl_down_sync(0xffffffff, value, 16);
  value += __shfl_down_sync(0xffffffff, value, 8);
  value += __shfl_down_sync(0xffffffff, value, 4);
  value += __shfl_down_sync(0xffffffff, value, 2);
  value += __shfl_down_sync(0xffffffff, value, 1);
  return value;
}

// cp.async helpers for the iter4 persistent + K-prefetch kernel. These
// are SM80+ HBM->SMEM async copies; on SM100 they coexist with TMA bulk
// (cp.async.bulk) and are the right primitive for small per-tile payloads
// (the K-prefetch issues 5 cp.async per K row, kTileN K rows per tile —
// too small for bulk amortization, big enough to overlap with the Stage B
// dot-product compute).
SGL_DEVICE void cp_async_16(void* smem_ptr, const void* glob_ptr) {
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :
      : "r"(smem), "l"(glob_ptr));
}

SGL_DEVICE void cp_async_4(void* smem_ptr, const void* glob_ptr) {
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 4;\n"
      :
      : "r"(smem), "l"(glob_ptr));
}

SGL_DEVICE void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
SGL_DEVICE void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

#if defined(SGLANG_ENABLE_HISA_SELECTOR_MEGAKERNEL)
SGL_DEVICE char* hisa_align_smem(char* ptr, uintptr_t alignment) {
  const uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
  return reinterpret_cast<char*>((raw + alignment - 1) & ~(alignment - 1));
}

SGL_DEVICE bool hisa_selector_better(
    float score_a,
    int ordinal_a,
    float score_b,
    int ordinal_b) {
  if (score_a > score_b) return true;
  if (score_a < score_b) return false;
  return ordinal_a < ordinal_b;
}

SGL_DEVICE uint32_t fp32_to_ordered_asc(uint32_t bits) {
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

SGL_DEVICE uint64_t hisa_candidate_key_from_score_ordinal(
    float score,
    uint32_t ordinal) {
  if (!(score > -3.0e38f)) return 0ull;
  const uint32_t score_key = fp32_to_ordered_asc(__float_as_uint(score));
  const uint32_t ordinal_key = 0xffffffffu - ordinal;
  return (static_cast<uint64_t>(score_key) << 32) | ordinal_key;
}

SGL_DEVICE uint64_t hisa_candidate_key_asc_from_score_ordinal(
    float score,
    uint32_t ordinal) {
  if (!(score > -3.0e38f)) return UINT64_MAX;
  const uint64_t score_key = fp32_to_radix_desc(__float_as_uint(score));
  return (score_key << 32) | static_cast<uint64_t>(ordinal);
}

SGL_DEVICE uint32_t hisa_candidate_key_ordinal(uint64_t key) {
  return 0xffffffffu - static_cast<uint32_t>(key);
}

SGL_DEVICE uint32_t hisa_candidate_key_asc_ordinal(uint64_t key) {
  return static_cast<uint32_t>(key);
}

__global__ void hisa_candidate_keys_topk_map_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISACandidateKeysTopKMapParam param) {
  constexpr uint32_t kThreads = 256;
  constexpr uint32_t kTopKBits = 11;
  constexpr uint32_t kBins = 1u << kTopKBits;
  constexpr uint32_t kTopKPasses = (32u + kTopKBits - 1u) / kTopKBits;
  constexpr uint32_t kWarps = kThreads / 32;
  constexpr uint32_t kBinsPerThread = (kBins + kThreads - 1u) / kThreads;
  constexpr uint32_t kHISABlockSize = 128;

  __shared__ uint32_t s_hist[kBins];
  __shared__ uint32_t s_prefix[kBins];
  __shared__ uint32_t s_warp_totals[kWarps];
  __shared__ uint32_t s_warp_prefix[kWarps];
  __shared__ uint32_t s_threshold_key;
  __shared__ uint32_t s_less_count;
  __shared__ uint32_t s_boundary_quota;
  __shared__ uint32_t s_boundary_used;
  __shared__ uint32_t s_above_used;
  __shared__ uint32_t s_emit_count;
  __shared__ uint32_t s_selected_bin;

  const uint32_t row = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  if (row >= param.q_rows) return;

  const uint32_t keep = min(param.topk, param.candidate_len);
  const auto* keys_row =
      static_cast<const uint32_t*>(param.candidate_keys) +
      static_cast<int64_t>(row) * param.candidate_len;
  const auto* blocks_row =
      static_cast<const int32_t*>(param.top_blocks) +
      static_cast<int64_t>(row) * param.block_topk;
  auto* out_row =
      static_cast<int32_t*>(param.topk_indices) +
      static_cast<int64_t>(row) * param.topk;

  if (tid == 0) {
    s_threshold_key = 0;
    s_less_count = 0;
    s_above_used = 0;
    s_boundary_used = 0;
    s_emit_count = 0;
  }
  __syncthreads();

  #pragma unroll
  for (uint32_t pass = 0; pass < kTopKPasses; ++pass) {
    const uint32_t bits_remaining = 32u - pass * kTopKBits;
    const uint32_t bits_this = min(kTopKBits, bits_remaining);
    const uint32_t shift = bits_remaining - bits_this;
    const uint32_t active_bins = 1u << bits_this;
    const uint32_t digit_mask = active_bins - 1u;
    const uint32_t prefix_mask =
        pass == 0 ? 0u : (0xffffffffu << (shift + bits_this));
    const uint32_t selected_prefix = s_threshold_key & prefix_mask;

    for (uint32_t bin = tid; bin < active_bins; bin += kThreads) {
      s_hist[bin] = 0;
    }
    __syncthreads();

    for (uint32_t i = tid; i < param.candidate_len; i += kThreads) {
      const uint32_t key = keys_row[i];
      if (key != 0xffffffffu &&
          (pass == 0 || ((key & prefix_mask) == selected_prefix))) {
        atomicAdd(&s_hist[(key >> shift) & digit_mask], 1u);
      }
    }
    __syncthreads();

    uint32_t lane_sum = 0;
    const uint32_t bin_base = tid * kBinsPerThread;
    #pragma unroll
    for (uint32_t j = 0; j < kBinsPerThread; ++j) {
      const uint32_t bin = bin_base + j;
      if (bin < active_bins) lane_sum += s_hist[bin];
    }
    uint32_t scan = lane_sum;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const auto y = __shfl_up_sync(0xffffffff, scan, off);
      if (static_cast<int>(tid & 31u) >= off) scan += y;
    }
    const uint32_t warp_id = tid >> 5;
    if ((tid & 31u) == 31u) s_warp_totals[warp_id] = scan;
    __syncthreads();

    if (tid < 32) {
      uint32_t warp_sum = tid < kWarps ? s_warp_totals[tid] : 0u;
      uint32_t warp_scan = warp_sum;
      #pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const auto y = __shfl_up_sync(0xffffffff, warp_scan, off);
        if (static_cast<int>(tid) >= off) warp_scan += y;
      }
      if (tid < kWarps) {
        s_warp_prefix[tid] = warp_scan - warp_sum;
      }
    }
    __syncthreads();

    uint32_t running = s_warp_prefix[warp_id] + scan - lane_sum;
    #pragma unroll
    for (uint32_t j = 0; j < kBinsPerThread; ++j) {
      const uint32_t bin = bin_base + j;
      if (bin < active_bins) {
        running += s_hist[bin];
        s_prefix[bin] = running;
      }
    }
    __syncthreads();

    if (tid == 0) {
      s_selected_bin = active_bins - 1u;
    }
    __syncthreads();

    const uint32_t target = keep > s_less_count ? keep - s_less_count : 1u;
    for (uint32_t bin = tid; bin < active_bins; bin += kThreads) {
      if (s_prefix[bin] >= target) {
        atomicMin(&s_selected_bin, bin);
      }
    }
    __syncthreads();

    if (tid == 0) {
      const uint32_t b = min(s_selected_bin, active_bins - 1u);
      const uint32_t before = b == 0 ? 0u : s_prefix[b - 1];
      s_less_count += before;
      s_threshold_key = selected_prefix | (b << shift);
    }
    __syncthreads();
  }

  if (tid == 0) {
    s_boundary_quota = keep > s_less_count ? keep - s_less_count : 0u;
    s_above_used = 0;
    s_boundary_used = 0;
  }
  __syncthreads();

  const uint32_t threshold_key = s_threshold_key;
  const uint32_t above_count = s_less_count;
  const uint32_t boundary_quota = s_boundary_quota;
  for (uint32_t i = tid; i < param.candidate_len; i += kThreads) {
    const uint32_t key = keys_row[i];
    if (key == 0xffffffffu) continue;
    const uint32_t block_slot = i / kHISABlockSize;
    const uint32_t block_offset = i - block_slot * kHISABlockSize;
    int32_t token = -1;
    if (block_slot < param.block_topk) {
      const int32_t block_id = blocks_row[block_slot];
      if (block_id >= 0) {
        token = block_id * static_cast<int32_t>(kHISABlockSize) +
                static_cast<int32_t>(block_offset);
      }
    }
    if (key < threshold_key) {
      const uint32_t slot = atomicAdd(&s_above_used, 1u);
      if (slot < above_count) out_row[slot] = token;
    } else if (key == threshold_key) {
      const uint32_t slot = atomicAdd(&s_boundary_used, 1u);
      if (slot < boundary_quota) out_row[above_count + slot] = token;
    }
  }
  __syncthreads();

  if (tid == 0) {
    const uint32_t boundary_written = min(s_boundary_used, boundary_quota);
    s_emit_count = min(keep, min(s_above_used, above_count) + boundary_written);
  }
  __syncthreads();

  for (uint32_t i = s_emit_count + tid; i < param.topk; i += kThreads) {
    out_row[i] = -1;
  }
}

template <uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void hisa_sm100_fp4_paged_mqa_candidate_keys(
    const uint32_t batch_size,
    const uint32_t candidate_len,
    const uint32_t block_table_stride,
    const uint32_t block_topk,
    const uint32_t* context_lens,
    uint32_t* candidate_keys,
    const uint32_t* block_table,
    const uint32_t* source_page_table,
    const uint32_t* schedule_meta,
    const int32_t* selected_blocks,
    const int32_t* prefix_lens,
    const int32_t* token_to_batch_idx,
    int32_t* fused_topk_indices,
    const uint32_t topk,
    const uint32_t source_page_table_stride,
    const __grid_constant__ cute::TmaDescriptor tensor_map_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
  using Barrier = cutlass::arch::ClusterTransactionBarrier;

  constexpr uint32_t kNextN = 1;
  constexpr bool kIsContextLens2D = true;
  constexpr bool kIsVarlen = false;
  constexpr uint32_t kNextNAtom = 1;
  constexpr uint32_t kNumNextNAtoms = 1;
  constexpr uint32_t kNumTmemStages = 3;
  constexpr uint32_t kNumUTCCPAlignedElems = 128;
  constexpr uint32_t UMMA_M = 128;
  constexpr uint32_t UMMA_N = kNumHeads;
  constexpr uint32_t UMMA_K = 64;
  constexpr uint32_t kNumSFQAtom =
      deep_gemm::math::constexpr_align(kNumHeads, kNumUTCCPAlignedElems);
  constexpr uint32_t kNumSFKV =
      deep_gemm::math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems);
  constexpr uint32_t kRealNumSFQAtom = kNumHeads;
  DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
  DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and SPLIT_KV % kNumUTCCPAlignedElems == 0, "Invalid `SPLIT_KV`");
  DG_STATIC_ASSERT(kHeadDim == 128, "HISA candidate key scorer expects head_dim=128");

  const auto sm_idx = blockIdx.x;
  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto warpgroup_idx = warp_idx / 4;
  const auto lane_idx = deep_gemm::ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_sf_q);
    cute::prefetch_tma_descriptor(&tensor_map_weights);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
  }

  constexpr uint32_t kSwizzleAlignment = 8 * (kHeadDim / 2);
  constexpr uint32_t SMEM_Q_SIZE_PER_STAGE      = kNextNAtom * kNumHeads * (kHeadDim / 2);
  constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE   = kNumSFQAtom * sizeof(int);
  constexpr uint32_t SMEM_KV_SIZE_PER_STAGE     = SPLIT_KV * (kHeadDim / 2);
  constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE  = kNumSFKV * sizeof(int);
  constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextNAtom * kNumHeads * sizeof(float);

  extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
  DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE  % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

  auto smem_q = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
    return smem_buffer + SMEM_Q_SIZE_PER_STAGE * i;
  });
  auto smem_kv = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
    return smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i;
  });
  const auto smem_sf_ptr = smem_buffer + (SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages);
  auto smem_sf_q = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<uint32_t*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * i);
  });
  auto smem_sf_kv = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<uint32_t*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages + SMEM_SF_KV_SIZE_PER_STAGE * i);
  });
  auto smem_weights = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<float*>(
        smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages +
        SMEM_SF_KV_SIZE_PER_STAGE * kNumKVStages +
        SMEM_WEIGHT_SIZE_PER_STAGE * i);
  });

  const auto barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
  auto full_q_barriers     = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
  auto empty_q_barriers    = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages + i; });
  auto full_kv_barriers    = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + i; });
  auto empty_kv_barriers   = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + kNumKVStages + i; });
  const auto tmem_barrier_ptr = barrier_ptr + kNumQStages * 2 + kNumKVStages * 2;
  auto full_tmem_barriers  = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) { return tmem_barrier_ptr + i; });
  auto empty_tmem_barriers = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) { return tmem_barrier_ptr + kNumTmemStages + i; });
  auto tmem_ptr_in_smem    = reinterpret_cast<uint32_t*>(tmem_barrier_ptr + kNumTmemStages * 2);

  constexpr uint32_t kNumAccumTmemCols = kNextNAtom * kNumHeads * kNumTmemStages;
  constexpr uint32_t kNumTmemCols =
      deep_gemm::utils::get_num_aligned_tmem_cols<
          kNumAccumTmemCols + kNumSFQAtom / 32 + kNumSFKV / 32>();
  constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
  constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQAtom / 32;
  DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

  if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumQStages; ++i) {
      full_q_barriers[i]->init(1);
      empty_q_barriers[i]->init(kNumMathThreads + 32);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      full_kv_barriers[i]->init(1);
      empty_kv_barriers[i]->init(1);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp_idx == kSpecWarpStart + 2) {
    if (cute::elect_one_sync()) {
      #pragma unroll
      for (uint32_t i = 0; i < kNumTmemStages; ++i) {
        full_tmem_barriers[i]->init(1);
        empty_tmem_barriers[i]->init(128);
      }
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
  }
  __syncthreads();

  cudaGridDependencySynchronize();

  constexpr uint32_t kNumBlocksPerSplit = SPLIT_KV / BLOCK_KV;
  using Scheduler = deep_gemm::sched::PagedMQALogitsScheduler<
      kNextN, kIsContextLens2D, kIsVarlen, BLOCK_KV,
      kNumBlocksPerSplit, kNumNextNAtoms>;
  DG_STATIC_ASSERT(SPLIT_KV == BLOCK_KV * kNumBlocksPerSplit, "Invalid `SPLIT_KV`");

  auto make_pipeline = [](const uint32_t& num_stages) {
    return [iter_idx = 0u, num_stages](const uint32_t& step = 1) mutable
        -> cute::tuple<uint32_t, uint32_t> {
      uint32_t current_idx = iter_idx;
      iter_idx += step;
      return {current_idx % num_stages, (current_idx / num_stages) & 1};
    };
  };
  auto advance_q_pipeline    = make_pipeline(kNumQStages);
  auto advance_kv_pipeline   = make_pipeline(kNumKVStages);
  auto advance_tmem_pipeline = make_pipeline(kNumTmemStages);

  constexpr uint32_t kNumSpecializedRegisters = 56;
  constexpr uint32_t kNumMathRegisters = 224;

  if (warp_idx == kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    if (cute::elect_one_sync()) {
      auto scheduler = Scheduler(sm_idx, batch_size, context_lens, schedule_meta, nullptr);
      uint32_t last_q_atom_idx = batch_size * kNumNextNAtoms;
      uint32_t q_atom_idx, _, __;
      while (scheduler.fetch_next_task(q_atom_idx, _, __)) {
        if (q_atom_idx != last_q_atom_idx) {
          CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
          empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
          const auto q_token_idx = Scheduler::atom_to_token_idx(q_atom_idx);
          cute::SM90_TMA_LOAD_2D::copy(
              &tensor_map_q,
              reinterpret_cast<uint64_t*>(full_q_barriers[q_stage_idx]),
              static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
              smem_q[q_stage_idx],
              0,
              q_token_idx * kNumHeads);
          deep_gemm::tma::copy<kNextNAtom * kNumHeads, 1, 0>(
              &tensor_map_sf_q,
              full_q_barriers[q_stage_idx],
              smem_sf_q[q_stage_idx],
              0,
              q_token_idx);
          deep_gemm::tma::copy<kNumHeads, kNextNAtom, 0>(
              &tensor_map_weights,
              full_q_barriers[q_stage_idx],
              smem_weights[q_stage_idx],
              0,
              q_token_idx);
          full_q_barriers[q_stage_idx]->arrive_and_expect_tx(
              SMEM_Q_SIZE_PER_STAGE + kRealNumSFQAtom * sizeof(int) +
              SMEM_WEIGHT_SIZE_PER_STAGE);
        }
        last_q_atom_idx = q_atom_idx;
      }
    }
    __syncwarp();
  } else if (warp_idx == kSpecWarpStart + 1) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    auto scheduler = Scheduler(sm_idx, batch_size, context_lens, schedule_meta, nullptr);
    uint32_t kv_block_idx_ptr = 32, kv_block_idx_storage;
    uint32_t last_q_atom_idx = batch_size * kNumNextNAtoms;
    uint32_t q_atom_idx, kv_idx, num_kv;
    while (scheduler.fetch_next_task(q_atom_idx, kv_idx, num_kv)) {
      if (q_atom_idx != last_q_atom_idx) kv_block_idx_ptr = 32;
      last_q_atom_idx = q_atom_idx;

      if (kv_block_idx_ptr == 32) {
        kv_block_idx_ptr = 0;
        kv_block_idx_storage = 0;
        const uint32_t candidate_page = kv_idx + lane_idx;
        if (candidate_page < num_kv) {
          const uint32_t q_idx = Scheduler::atom_to_block_table_row(q_atom_idx);
          if (source_page_table != nullptr) {
            const uint32_t block_slot = candidate_page >> 1;
            const uint32_t half = candidate_page & 1u;
            const int32_t batch = token_to_batch_idx[q_idx];
            int32_t selected_block = -1;
            if (block_slot < block_topk) {
              selected_block =
                  selected_blocks[q_idx * static_cast<uint64_t>(block_topk) +
                                  block_slot];
            }
            if (batch >= 0 && selected_block >= 0) {
              kv_block_idx_storage = source_page_table[
                  static_cast<uint64_t>(batch) * source_page_table_stride +
                  static_cast<uint32_t>(selected_block) * 2u + half];
            }
          } else {
            const auto block_table_offset =
                q_idx * static_cast<uint64_t>(block_table_stride);
            kv_block_idx_storage =
                block_table[block_table_offset + candidate_page];
          }
        }
      }
      __syncwarp();

      int kv_block_idx[kNumBlocksPerSplit];
      #pragma unroll
      for (int i = 0; i < kNumBlocksPerSplit; ++i)
        kv_block_idx[i] = __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr + i);
      kv_block_idx_ptr += kNumBlocksPerSplit;
      DG_STATIC_ASSERT(32 % kNumBlocksPerSplit == 0, "Invalid `SPLIT_KV`");

      CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
      if (cute::elect_one_sync()) {
        empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);
        #pragma unroll
        for (int i = 0; i < kNumBlocksPerSplit; ++i) {
          cute::SM90_TMA_LOAD_3D::copy(
              &tensor_map_kv,
              reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
              static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
              smem_kv[kv_stage_idx] + (BLOCK_KV * kHeadDim / 2) * i,
              0,
              0,
              kv_block_idx[i]);
          deep_gemm::tma::copy<BLOCK_KV, 1, 0>(
              &tensor_map_sf_kv,
              full_kv_barriers[kv_stage_idx],
              smem_sf_kv[kv_stage_idx] + BLOCK_KV * i,
              0,
              kv_block_idx[i]);
        }
        full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(
            SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
      }
    }
  } else if (warp_idx == kSpecWarpStart + 2) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    auto scheduler = Scheduler(sm_idx, batch_size, context_lens, schedule_meta, nullptr);
    DG_TRAP_ONLY_DEVICE_ASSERT(deep_gemm::ptx::ld_shared(tmem_ptr_in_smem) == 0);

    auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
      DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
      uint32_t values[4];
      #pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        values[i] = deep_gemm::ptx::ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
      __syncwarp();
      #pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        deep_gemm::ptx::st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
    };

    auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
        cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
        cutlass::float_ue8m0_t, UMMA_M, UMMA_N,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    auto sf_desc = deep_gemm::mma::sm100::make_sf_desc(nullptr);

    uint32_t last_q_atom_idx = batch_size * kNumNextNAtoms;
    uint32_t q_atom_idx, kv_idx, _;
    while (scheduler.fetch_next_task(q_atom_idx, kv_idx, _)) {
      uint32_t q_stage_idx, q_phase;
      if (q_atom_idx != last_q_atom_idx) {
        CUTE_TIE(advance_q_pipeline(), q_stage_idx, q_phase);
        if (last_q_atom_idx != batch_size * kNumNextNAtoms)
          empty_q_barriers[(q_stage_idx + kNumQStages - 1) % kNumQStages]->arrive();
        full_q_barriers[q_stage_idx]->wait(q_phase);
        #pragma unroll
        for (uint32_t i = 0; i < kNumSFQAtom / kNumUTCCPAlignedElems; ++i) {
          auto smem_ptr = smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
          utccp_required_smem_warp_transpose(smem_ptr);
          cutlass::arch::fence_view_async_shared();
          deep_gemm::mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
          if (cute::elect_one_sync())
            cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFQ + i * 4);
          __syncwarp();
        }
      }
      last_q_atom_idx = q_atom_idx;

      CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
      full_kv_barriers[kv_stage_idx]->wait(kv_phase);

      #pragma unroll
      for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++i) {
        auto smem_ptr = smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
        utccp_required_smem_warp_transpose(smem_ptr);
        cutlass::arch::fence_view_async_shared();
      }

      if (cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++i) {
          auto smem_ptr = smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
          deep_gemm::mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
          cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFKV + i * 4);
        }

        #pragma unroll
        for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
          CUTE_TIE_DECL(advance_tmem_pipeline(), tmem_stage_idx, tmem_phase);
          uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

          empty_tmem_barriers[tmem_stage_idx]->wait(tmem_phase ^ 1);
          deep_gemm::ptx::tcgen05_after_thread_sync();

          #pragma unroll
          for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
            auto runtime_instr_desc =
                deep_gemm::mma::sm100::make_runtime_instr_desc_with_sf_id(
                    instr_desc, k * 2, k * 2);
            auto a_desc = deep_gemm::mma::sm100::make_smem_desc(
                cute::UMMA::LayoutType::SWIZZLE_64B,
                smem_kv[kv_stage_idx] + i * UMMA_M * (kHeadDim / 2) + k * UMMA_K / 2,
                8 * (kHeadDim / 2),
                0);
            auto b_desc = deep_gemm::mma::sm100::make_smem_desc(
                cute::UMMA::LayoutType::SWIZZLE_64B,
                smem_q[q_stage_idx] + k * UMMA_K / 2,
                8 * (kHeadDim / 2),
                0);
            deep_gemm::ptx::SM100_MMA_MXF4_SS::fma(
                a_desc,
                b_desc,
                tmem_addr,
                k,
                runtime_instr_desc,
                kTmemStartColOfSFKV + i * 4,
                kTmemStartColOfSFQ);
          }
          asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                       ::"r"(cute::cast_smem_ptr_to_uint(full_tmem_barriers[tmem_stage_idx])));
        }
      }
      cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_kv_barriers[kv_stage_idx]));
    }
  } else if (warp_idx == kSpecWarpStart + 3) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
  } else if (warp_idx < kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
    auto scheduler = Scheduler(sm_idx, batch_size, context_lens, schedule_meta, nullptr);
    const auto math_warpgroup_idx = warpgroup_idx;
    const auto math_thread_idx = warp_idx * 32 + lane_idx;

    auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr, float* accum) {
      constexpr int N = decltype(num_elems_c)::value;
      DG_STATIC_ASSERT(N == 32 or N == 64, "Unsupported TMEM load size");
      using Loader = cute::conditional_t<N == 32,
          cute::SM100_TMEM_LOAD_32dp32b32x,
          cute::SM100_TMEM_LOAD_32dp32b64x>;
      [&]<size_t... Is>(cute::index_sequence<Is...>) {
        Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
      }(cute::make_index_sequence<N>{});
      cutlass::arch::fence_view_async_tmem_load();
    };

    advance_tmem_pipeline(math_warpgroup_idx);

    float accum[kNumHeads];
    float weights[kNumHeads];
    uint32_t last_q_atom_idx = batch_size * kNumNextNAtoms;
    uint32_t q_atom_idx, kv_idx, _;
    while (scheduler.fetch_next_task(q_atom_idx, kv_idx, _)) {
      uint32_t q_stage_idx, q_phase;
      if (q_atom_idx != last_q_atom_idx) {
        CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
        if (last_q_atom_idx != batch_size * kNumNextNAtoms)
          empty_q_barriers[(q_stage_idx + kNumQStages - 1) % kNumQStages]->arrive();
        full_q_barriers[q_stage_idx]->wait(q_phase);

        #pragma unroll
        for (uint32_t j = 0; j < kNumHeads; j += 4) {
          float4 raw = deep_gemm::ptx::ld_shared((float4*)(smem_weights[q_stage_idx] + j));
          weights[j + 0] = raw.x;
          weights[j + 1] = raw.y;
          weights[j + 2] = raw.z;
          weights[j + 3] = raw.w;
        }
      }
      last_q_atom_idx = q_atom_idx;

      const auto q_token_idx = Scheduler::atom_to_token_idx(q_atom_idx);
      const auto candidate_pos = kv_idx * BLOCK_KV + math_thread_idx;
      const auto key_offset = q_token_idx * static_cast<uint64_t>(candidate_len) + candidate_pos;

      CUTE_TIE_DECL(advance_tmem_pipeline(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
      full_tmem_barriers[tmem_stage_idx]->wait(tmem_phase);
      deep_gemm::ptx::tcgen05_after_thread_sync();

      uint32_t tmem_addr = tmem_stage_idx * UMMA_N;
      tmem_load(cute::Int<kNumHeads / 2>{}, tmem_addr, accum);
      tmem_load(cute::Int<kNumHeads / 2>{}, tmem_addr + kNumHeads / 2, accum + kNumHeads / 2);

      auto sum_0 = make_float2(0, 0);
      auto sum_1 = make_float2(0, 0);
      const auto transform = [&](const uint32_t& j, const float2& sum) {
        auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));
        auto b = make_float2(weights[j], weights[j + 1]);
        return __ffma2_rn(a, b, sum);
      };
      #pragma unroll
      for (uint32_t j = 0; j < kNumHeads; j += 4) {
        sum_0 = transform(j, sum_0);
        sum_1 = transform(j + 2, sum_1);
      }
      auto sum = __fadd2_rn(sum_0, sum_1);
      const float score = sum.x + sum.y;

      uint32_t key = 0xffffffffu;
      if (candidate_pos < candidate_len) {
        const uint32_t block_slot = candidate_pos / 128u;
        const uint32_t block_offset = candidate_pos - block_slot * 128u;
        const int32_t block_id =
            block_slot < block_topk
                ? selected_blocks[q_token_idx * static_cast<uint64_t>(block_topk) + block_slot]
                : -1;
        const int32_t token =
            block_id >= 0
                ? block_id * 128 + static_cast<int32_t>(block_offset)
                : -1;
        const int32_t prefix_len = prefix_lens[q_token_idx];
        if (token >= 0 && token < prefix_len) {
          key = fp32_to_radix_desc(__float_as_uint(score));
        }
      }
      candidate_keys[key_offset] = key;

      deep_gemm::ptx::tcgen05_before_thread_sync();
      empty_tmem_barriers[tmem_stage_idx]->arrive();
    }

    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    if (warp_idx == 0)
      cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }

  __syncthreads();
  if (fused_topk_indices == nullptr || topk == 0) return;

  cooperative_groups::this_grid().sync();

  constexpr uint32_t kTopKBins = 256;
  constexpr uint32_t kHISABlockSize = 128;
  auto* topk_smem = reinterpret_cast<uint32_t*>(smem_buffer);
  auto* s_hist = topk_smem;
  auto* s_prefix = s_hist + kTopKBins;
  auto* s_scalars = s_prefix + kTopKBins;
  constexpr uint32_t kThresholdSlot = 0;
  constexpr uint32_t kLessSlot = 1;
  constexpr uint32_t kBoundaryQuotaSlot = 2;
  constexpr uint32_t kBoundaryUsedSlot = 3;
  constexpr uint32_t kAboveUsedSlot = 4;
  constexpr uint32_t kEmitCountSlot = 5;

  const uint32_t tid = threadIdx.x;
  const uint32_t keep = min(topk, candidate_len);
  for (uint32_t topk_row = blockIdx.x; topk_row < batch_size; topk_row += gridDim.x) {
    const auto* keys_row =
        candidate_keys + static_cast<uint64_t>(topk_row) * candidate_len;
    const auto* blocks_row =
        selected_blocks + static_cast<uint64_t>(topk_row) * block_topk;
    auto* out_row =
        fused_topk_indices + static_cast<uint64_t>(topk_row) * topk;

    if (tid == 0) {
      s_scalars[kThresholdSlot] = 0u;
      s_scalars[kLessSlot] = 0u;
      s_scalars[kAboveUsedSlot] = 0u;
      s_scalars[kBoundaryUsedSlot] = 0u;
      s_scalars[kEmitCountSlot] = 0u;
    }
    __syncthreads();

    #pragma unroll
    for (uint32_t pass = 0; pass < 4; ++pass) {
      const uint32_t shift = 24u - pass * 8u;
      const uint32_t prefix_mask =
          pass == 0 ? 0u : (0xffffffffu << (shift + 8u));
      const uint32_t selected_prefix = s_scalars[kThresholdSlot] & prefix_mask;

      for (uint32_t bin = tid; bin < kTopKBins; bin += blockDim.x) {
        s_hist[bin] = 0u;
      }
      __syncthreads();

      for (uint32_t i = tid; i < candidate_len; i += blockDim.x) {
        const uint32_t key = keys_row[i];
        if (key != 0xffffffffu &&
            (pass == 0 || ((key & prefix_mask) == selected_prefix))) {
          atomicAdd(&s_hist[(key >> shift) & 0xffu], 1u);
        }
      }
      __syncthreads();

      if (tid < 32) {
        uint32_t lane_sum = 0u;
        for (uint32_t b = tid * 8; b < (tid + 1) * 8; ++b) lane_sum += s_hist[b];
        uint32_t scan = lane_sum;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
          const auto y = __shfl_up_sync(0xffffffff, scan, off);
          if (static_cast<int>(tid) >= off) scan += y;
        }
        const uint32_t base = scan - lane_sum;
        uint32_t running = base;
        for (uint32_t b = tid * 8; b < (tid + 1) * 8; ++b) {
          running += s_hist[b];
          s_prefix[b] = running;
        }
      }
      __syncthreads();

      if (tid == 0) {
        const uint32_t less_count = s_scalars[kLessSlot];
        const uint32_t target = keep > less_count ? keep - less_count : 1u;
        uint32_t b = 0u;
        while (b + 1 < kTopKBins && s_prefix[b] < target) ++b;
        const uint32_t before = b == 0 ? 0u : s_prefix[b - 1];
        s_scalars[kLessSlot] = less_count + before;
        s_scalars[kThresholdSlot] = selected_prefix | (b << shift);
      }
      __syncthreads();
    }

    if (tid == 0) {
      const uint32_t less_count = s_scalars[kLessSlot];
      s_scalars[kBoundaryQuotaSlot] = keep > less_count ? keep - less_count : 0u;
      s_scalars[kAboveUsedSlot] = 0u;
      s_scalars[kBoundaryUsedSlot] = 0u;
    }
    __syncthreads();

    const uint32_t threshold_key = s_scalars[kThresholdSlot];
    const uint32_t above_count = s_scalars[kLessSlot];
    const uint32_t boundary_quota = s_scalars[kBoundaryQuotaSlot];
    for (uint32_t i = tid; i < candidate_len; i += blockDim.x) {
      const uint32_t key = keys_row[i];
      if (key == 0xffffffffu) continue;
      const uint32_t block_slot = i / kHISABlockSize;
      const uint32_t block_offset = i - block_slot * kHISABlockSize;
      int32_t token = -1;
      if (block_slot < block_topk) {
        const int32_t block_id = blocks_row[block_slot];
        if (block_id >= 0) {
          token = block_id * static_cast<int32_t>(kHISABlockSize) +
                  static_cast<int32_t>(block_offset);
        }
      }
      if (key < threshold_key) {
        const uint32_t slot = atomicAdd(&s_scalars[kAboveUsedSlot], 1u);
        if (slot < above_count) out_row[slot] = token;
      } else if (key == threshold_key) {
        const uint32_t slot = atomicAdd(&s_scalars[kBoundaryUsedSlot], 1u);
        if (slot < boundary_quota) out_row[above_count + slot] = token;
      }
    }
    __syncthreads();

    if (tid == 0) {
      const uint32_t boundary_written =
          min(s_scalars[kBoundaryUsedSlot], s_scalars[kBoundaryQuotaSlot]);
      s_scalars[kEmitCountSlot] = min(
          keep,
          min(s_scalars[kAboveUsedSlot], s_scalars[kLessSlot]) +
              boundary_written);
    }
    __syncthreads();

    for (uint32_t i = s_scalars[kEmitCountSlot] + tid; i < topk; i += blockDim.x) {
      out_row[i] = -1;
    }
    __syncthreads();
  }

}

template <uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void hisa_sm100_fp4_paged_mqa_candidate_keys_row_split(
    const uint32_t q_rows,
    const uint32_t candidate_len,
    const uint32_t row_splits,
    const uint32_t block_topk,
    uint32_t* candidate_keys,
    int32_t* fused_topk_indices,
    const uint32_t topk,
    const uint32_t* source_page_table,
    const int32_t* selected_blocks,
    const int32_t* prefix_lens,
    const int32_t* token_to_batch_idx,
    const uint32_t source_page_table_stride,
    const __grid_constant__ cute::TmaDescriptor tensor_map_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
  using Barrier = cutlass::arch::ClusterTransactionBarrier;

  constexpr uint32_t kNextNAtom = 1;
  constexpr uint32_t kNumTmemStages = 3;
  constexpr uint32_t kNumUTCCPAlignedElems = 128;
  constexpr uint32_t UMMA_M = 128;
  constexpr uint32_t UMMA_N = kNumHeads;
  constexpr uint32_t UMMA_K = 64;
  constexpr uint32_t kNumSFQAtom =
      deep_gemm::math::constexpr_align(kNumHeads, kNumUTCCPAlignedElems);
  constexpr uint32_t kNumSFKV =
      deep_gemm::math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems);
  constexpr uint32_t kRealNumSFQAtom = kNumHeads;
  constexpr uint32_t kPagesPerTask = SPLIT_KV / BLOCK_KV;
  constexpr uint32_t kHISABlockSize = 128;
  constexpr uint32_t kTopKBits = 11;
  constexpr uint32_t kTopKBins = 1u << kTopKBits;
  constexpr uint32_t kTopKPasses = (32u + kTopKBits - 1u) / kTopKBits;
  DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
  DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and SPLIT_KV % kNumUTCCPAlignedElems == 0, "Invalid `SPLIT_KV`");
  DG_STATIC_ASSERT(kHeadDim == 128, "HISA candidate row-split scorer expects head_dim=128");
  DG_STATIC_ASSERT(BLOCK_KV == 64 and kPagesPerTask == 4, "HISA row-split scorer expects 64-token pages and 256-token tasks");

  __shared__ uint32_t s_threshold_key;
  __shared__ uint32_t s_less_count;
  __shared__ uint32_t s_boundary_quota;
  __shared__ uint32_t s_boundary_used;
  __shared__ uint32_t s_above_used;
  __shared__ uint32_t s_emit_count;
  __shared__ uint32_t s_selected_bin;

  const uint32_t row = blockIdx.x;
  const uint32_t row_split = blockIdx.y;
  if (row >= q_rows || row_split >= row_splits) return;

  const uint32_t candidate_pages = deep_gemm::math::ceil_div(candidate_len, BLOCK_KV);
  const uint32_t tasks_per_row = deep_gemm::math::ceil_div(candidate_len, SPLIT_KV);
  const uint32_t task_begin =
      static_cast<uint64_t>(tasks_per_row) * row_split / row_splits;
  const uint32_t task_end =
      static_cast<uint64_t>(tasks_per_row) * (row_split + 1) / row_splits;
  const int32_t batch = token_to_batch_idx[row];
  const int32_t prefix_len = prefix_lens[row];

  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto warpgroup_idx = warp_idx / 4;
  const auto lane_idx = deep_gemm::ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_sf_q);
    cute::prefetch_tma_descriptor(&tensor_map_weights);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
  }

  constexpr uint32_t kSwizzleAlignment = 8 * (kHeadDim / 2);
  constexpr uint32_t SMEM_Q_SIZE_PER_STAGE      = kNextNAtom * kNumHeads * (kHeadDim / 2);
  constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE   = kNumSFQAtom * sizeof(int);
  constexpr uint32_t SMEM_KV_SIZE_PER_STAGE     = SPLIT_KV * (kHeadDim / 2);
  constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE  = kNumSFKV * sizeof(int);
  constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextNAtom * kNumHeads * sizeof(float);

  extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
  DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE  % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

  auto smem_q = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
    return smem_buffer + SMEM_Q_SIZE_PER_STAGE * i;
  });
  auto smem_kv = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
    return smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i;
  });
  const auto smem_sf_ptr = smem_buffer + (SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages);
  auto smem_sf_q = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<uint32_t*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * i);
  });
  auto smem_sf_kv = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<uint32_t*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages + SMEM_SF_KV_SIZE_PER_STAGE * i);
  });
  auto smem_weights = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<float*>(
        smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages +
        SMEM_SF_KV_SIZE_PER_STAGE * kNumKVStages +
        SMEM_WEIGHT_SIZE_PER_STAGE * i);
  });

  const auto barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
  auto full_q_barriers     = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
  auto empty_q_barriers    = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages + i; });
  auto full_kv_barriers    = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + i; });
  auto empty_kv_barriers   = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + kNumKVStages + i; });
  const auto tmem_barrier_ptr = barrier_ptr + kNumQStages * 2 + kNumKVStages * 2;
  auto full_tmem_barriers  = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) { return tmem_barrier_ptr + i; });
  auto empty_tmem_barriers = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) { return tmem_barrier_ptr + kNumTmemStages + i; });
  auto tmem_ptr_in_smem    = reinterpret_cast<uint32_t*>(tmem_barrier_ptr + kNumTmemStages * 2);

  constexpr uint32_t kNumAccumTmemCols = kNextNAtom * kNumHeads * kNumTmemStages;
  constexpr uint32_t kNumTmemCols =
      deep_gemm::utils::get_num_aligned_tmem_cols<
          kNumAccumTmemCols + kNumSFQAtom / 32 + kNumSFKV / 32>();
  constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
  constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQAtom / 32;
  DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

  if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumQStages; ++i) {
      full_q_barriers[i]->init(1);
      empty_q_barriers[i]->init(kNumMathThreads + 32);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      full_kv_barriers[i]->init(1);
      empty_kv_barriers[i]->init(1);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp_idx == kSpecWarpStart + 2) {
    if (cute::elect_one_sync()) {
      #pragma unroll
      for (uint32_t i = 0; i < kNumTmemStages; ++i) {
        full_tmem_barriers[i]->init(1);
        empty_tmem_barriers[i]->init(128);
      }
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
  }
  __syncthreads();

  cudaGridDependencySynchronize();

  auto make_pipeline = [](const uint32_t& num_stages) {
    return [iter_idx = 0u, num_stages](const uint32_t& step = 1) mutable
        -> cute::tuple<uint32_t, uint32_t> {
      uint32_t current_idx = iter_idx;
      iter_idx += step;
      return {current_idx % num_stages, (current_idx / num_stages) & 1};
    };
  };
  auto advance_q_pipeline    = make_pipeline(kNumQStages);
  auto advance_kv_pipeline   = make_pipeline(kNumKVStages);
  auto advance_tmem_pipeline = make_pipeline(kNumTmemStages);

  constexpr uint32_t kNumSpecializedRegisters = 56;
  constexpr uint32_t kNumMathRegisters = 224;

  if (warp_idx == kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    if (cute::elect_one_sync()) {
      CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
      empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
      cute::SM90_TMA_LOAD_2D::copy(
          &tensor_map_q,
          reinterpret_cast<uint64_t*>(full_q_barriers[q_stage_idx]),
          static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
          smem_q[q_stage_idx],
          0,
          row * kNumHeads);
      deep_gemm::tma::copy<kNextNAtom * kNumHeads, 1, 0>(
          &tensor_map_sf_q,
          full_q_barriers[q_stage_idx],
          smem_sf_q[q_stage_idx],
          0,
          row);
      deep_gemm::tma::copy<kNumHeads, kNextNAtom, 0>(
          &tensor_map_weights,
          full_q_barriers[q_stage_idx],
          smem_weights[q_stage_idx],
          0,
          row);
      full_q_barriers[q_stage_idx]->arrive_and_expect_tx(
          SMEM_Q_SIZE_PER_STAGE + kRealNumSFQAtom * sizeof(int) +
          SMEM_WEIGHT_SIZE_PER_STAGE);
    }
    __syncwarp();
  } else if (warp_idx == kSpecWarpStart + 1) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    for (uint32_t task = task_begin; task < task_end; ++task) {
      const uint32_t kv_idx = task * kPagesPerTask;
      uint32_t kv_block_idx_storage = 0;
      const uint32_t candidate_page = kv_idx + lane_idx;
      if (candidate_page < candidate_pages && batch >= 0) {
        const uint32_t block_slot = candidate_page >> 1;
        const uint32_t half = candidate_page & 1u;
        int32_t selected_block = -1;
        if (block_slot < block_topk) {
          selected_block =
              selected_blocks[row * static_cast<uint64_t>(block_topk) +
                              block_slot];
        }
        if (selected_block >= 0) {
          kv_block_idx_storage = source_page_table[
              static_cast<uint64_t>(batch) * source_page_table_stride +
              static_cast<uint32_t>(selected_block) * 2u + half];
        }
      }
      __syncwarp();

      int kv_block_idx[kPagesPerTask];
      #pragma unroll
      for (int i = 0; i < static_cast<int>(kPagesPerTask); ++i)
        kv_block_idx[i] = __shfl_sync(0xffffffff, kv_block_idx_storage, i);

      CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
      if (cute::elect_one_sync()) {
        empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);
        #pragma unroll
        for (int i = 0; i < static_cast<int>(kPagesPerTask); ++i) {
          cute::SM90_TMA_LOAD_3D::copy(
              &tensor_map_kv,
              reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
              static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
              smem_kv[kv_stage_idx] + (BLOCK_KV * kHeadDim / 2) * i,
              0,
              0,
              kv_block_idx[i]);
          deep_gemm::tma::copy<BLOCK_KV, 1, 0>(
              &tensor_map_sf_kv,
              full_kv_barriers[kv_stage_idx],
              smem_sf_kv[kv_stage_idx] + BLOCK_KV * i,
              0,
              kv_block_idx[i]);
        }
        full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(
            SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
      }
    }
  } else if (warp_idx == kSpecWarpStart + 2) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    DG_TRAP_ONLY_DEVICE_ASSERT(deep_gemm::ptx::ld_shared(tmem_ptr_in_smem) == 0);

    auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
      DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
      uint32_t values[4];
      #pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        values[i] = deep_gemm::ptx::ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
      __syncwarp();
      #pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        deep_gemm::ptx::st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
    };

    auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
        cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
        cutlass::float_ue8m0_t, UMMA_M, UMMA_N,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    auto sf_desc = deep_gemm::mma::sm100::make_sf_desc(nullptr);

    CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
    full_q_barriers[q_stage_idx]->wait(q_phase);
    #pragma unroll
    for (uint32_t i = 0; i < kNumSFQAtom / kNumUTCCPAlignedElems; ++i) {
      auto smem_ptr = smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
      utccp_required_smem_warp_transpose(smem_ptr);
      cutlass::arch::fence_view_async_shared();
      deep_gemm::mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
      if (cute::elect_one_sync())
        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFQ + i * 4);
      __syncwarp();
    }

    for (uint32_t task = task_begin; task < task_end; ++task) {
      CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
      full_kv_barriers[kv_stage_idx]->wait(kv_phase);

      #pragma unroll
      for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++i) {
        auto smem_ptr = smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
        utccp_required_smem_warp_transpose(smem_ptr);
        cutlass::arch::fence_view_async_shared();
      }

      if (cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++i) {
          auto smem_ptr = smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
          deep_gemm::mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
          cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFKV + i * 4);
        }

        #pragma unroll
        for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
          CUTE_TIE_DECL(advance_tmem_pipeline(), tmem_stage_idx, tmem_phase);
          uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

          empty_tmem_barriers[tmem_stage_idx]->wait(tmem_phase ^ 1);
          deep_gemm::ptx::tcgen05_after_thread_sync();

          #pragma unroll
          for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
            auto runtime_instr_desc =
                deep_gemm::mma::sm100::make_runtime_instr_desc_with_sf_id(
                    instr_desc, k * 2, k * 2);
            auto a_desc = deep_gemm::mma::sm100::make_smem_desc(
                cute::UMMA::LayoutType::SWIZZLE_64B,
                smem_kv[kv_stage_idx] + i * UMMA_M * (kHeadDim / 2) + k * UMMA_K / 2,
                8 * (kHeadDim / 2),
                0);
            auto b_desc = deep_gemm::mma::sm100::make_smem_desc(
                cute::UMMA::LayoutType::SWIZZLE_64B,
                smem_q[q_stage_idx] + k * UMMA_K / 2,
                8 * (kHeadDim / 2),
                0);
            deep_gemm::ptx::SM100_MMA_MXF4_SS::fma(
                a_desc,
                b_desc,
                tmem_addr,
                k,
                runtime_instr_desc,
                kTmemStartColOfSFKV + i * 4,
                kTmemStartColOfSFQ);
          }
          asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                       ::"r"(cute::cast_smem_ptr_to_uint(full_tmem_barriers[tmem_stage_idx])));
        }
      }
      cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_kv_barriers[kv_stage_idx]));
    }
  } else if (warp_idx == kSpecWarpStart + 3) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
  } else if (warp_idx < kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
    const auto math_warpgroup_idx = warpgroup_idx;
    const auto math_thread_idx = warp_idx * 32 + lane_idx;

    auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr, float* accum) {
      constexpr int N = decltype(num_elems_c)::value;
      DG_STATIC_ASSERT(N == 32 or N == 64, "Unsupported TMEM load size");
      using Loader = cute::conditional_t<N == 32,
          cute::SM100_TMEM_LOAD_32dp32b32x,
          cute::SM100_TMEM_LOAD_32dp32b64x>;
      [&]<size_t... Is>(cute::index_sequence<Is...>) {
        Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
      }(cute::make_index_sequence<N>{});
      cutlass::arch::fence_view_async_tmem_load();
    };

    advance_tmem_pipeline(math_warpgroup_idx);

    float accum[kNumHeads];
    float weights[kNumHeads];

    CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
    full_q_barriers[q_stage_idx]->wait(q_phase);
    #pragma unroll
    for (uint32_t j = 0; j < kNumHeads; j += 4) {
      float4 raw = deep_gemm::ptx::ld_shared((float4*)(smem_weights[q_stage_idx] + j));
      weights[j + 0] = raw.x;
      weights[j + 1] = raw.y;
      weights[j + 2] = raw.z;
      weights[j + 3] = raw.w;
    }

    for (uint32_t task = task_begin; task < task_end; ++task) {
      const auto candidate_pos = task * SPLIT_KV + math_thread_idx;
      const auto key_offset = row * static_cast<uint64_t>(candidate_len) + candidate_pos;

      CUTE_TIE_DECL(advance_tmem_pipeline(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
      full_tmem_barriers[tmem_stage_idx]->wait(tmem_phase);
      deep_gemm::ptx::tcgen05_after_thread_sync();

      uint32_t tmem_addr = tmem_stage_idx * UMMA_N;
      tmem_load(cute::Int<kNumHeads / 2>{}, tmem_addr, accum);
      tmem_load(cute::Int<kNumHeads / 2>{}, tmem_addr + kNumHeads / 2, accum + kNumHeads / 2);

      auto sum_0 = make_float2(0, 0);
      auto sum_1 = make_float2(0, 0);
      const auto transform = [&](const uint32_t& j, const float2& sum) {
        auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));
        auto b = make_float2(weights[j], weights[j + 1]);
        return __ffma2_rn(a, b, sum);
      };
      #pragma unroll
      for (uint32_t j = 0; j < kNumHeads; j += 4) {
        sum_0 = transform(j, sum_0);
        sum_1 = transform(j + 2, sum_1);
      }
      auto sum = __fadd2_rn(sum_0, sum_1);
      const float score = sum.x + sum.y;

      uint32_t key = 0xffffffffu;
      if (candidate_pos < candidate_len) {
        const uint32_t block_slot = candidate_pos / 128u;
        const uint32_t block_offset = candidate_pos - block_slot * 128u;
        const int32_t block_id =
            block_slot < block_topk
                ? selected_blocks[row * static_cast<uint64_t>(block_topk) + block_slot]
                : -1;
        const int32_t token =
            block_id >= 0
                ? block_id * 128 + static_cast<int32_t>(block_offset)
                : -1;
        if (token >= 0 && token < prefix_len) {
          key = fp32_to_radix_desc(__float_as_uint(score));
        }
      }
      if (candidate_pos < candidate_len) {
        candidate_keys[key_offset] = key;
      }

      deep_gemm::ptx::tcgen05_before_thread_sync();
      empty_tmem_barriers[tmem_stage_idx]->arrive();
    }

    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    if (warp_idx == 0)
      cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }

  if (fused_topk_indices == nullptr || topk == 0) return;
  if (row_splits == 1u) {
    __threadfence_block();
    __syncthreads();
  } else {
    __threadfence();
    auto cluster = cooperative_groups::this_cluster();
    cluster.sync();
    if (row_split != 0) return;
  }

  const uint32_t tid = threadIdx.x;
  constexpr uint32_t kTopKThreads = 256;
  constexpr uint32_t kTopKWarps = kTopKThreads / 32;
  constexpr uint32_t kTopKBinsPerThread =
      (kTopKBins + kTopKThreads - 1u) / kTopKThreads;
  const bool topk_active = tid < kTopKThreads;
  const uint32_t keep = min(topk, candidate_len);
  const auto* keys_row =
      candidate_keys + static_cast<int64_t>(row) * candidate_len;
  const auto* blocks_row =
      selected_blocks + static_cast<int64_t>(row) * block_topk;
  auto* out_row =
      fused_topk_indices + static_cast<int64_t>(row) * topk;
  auto* topk_smem = reinterpret_cast<uint32_t*>(smem_buffer);
  auto* s_hist = topk_smem;
  auto* s_prefix = s_hist + kTopKBins;
  auto* s_warp_totals = s_prefix + kTopKBins;
  auto* s_warp_prefix = s_warp_totals + kTopKWarps;

  if (tid == 0) {
    s_threshold_key = 0u;
    s_less_count = 0u;
    s_above_used = 0u;
    s_boundary_used = 0u;
    s_emit_count = 0u;
  }
  __syncthreads();

  #pragma unroll
  for (uint32_t pass = 0; pass < kTopKPasses; ++pass) {
    const uint32_t bits_remaining = 32u - pass * kTopKBits;
    const uint32_t bits_this = min(kTopKBits, bits_remaining);
    const uint32_t shift = bits_remaining - bits_this;
    const uint32_t active_bins = 1u << bits_this;
    const uint32_t digit_mask = active_bins - 1u;
    const uint32_t prefix_mask =
        pass == 0 ? 0u : (0xffffffffu << (shift + bits_this));
    const uint32_t selected_prefix = s_threshold_key & prefix_mask;

    if (topk_active) {
      for (uint32_t bin = tid; bin < active_bins; bin += kTopKThreads) {
        s_hist[bin] = 0u;
      }
    }
    __syncthreads();

    if (topk_active) {
      for (uint32_t i = tid; i < candidate_len; i += kTopKThreads) {
        const uint32_t key = keys_row[i];
        if (key != 0xffffffffu &&
            (pass == 0 || ((key & prefix_mask) == selected_prefix))) {
          atomicAdd(&s_hist[(key >> shift) & digit_mask], 1u);
        }
      }
    }
    __syncthreads();

    if (topk_active) {
      uint32_t lane_sum = 0u;
      const uint32_t bin_base = tid * kTopKBinsPerThread;
      #pragma unroll
      for (uint32_t j = 0; j < kTopKBinsPerThread; ++j) {
        const uint32_t bin = bin_base + j;
        if (bin < active_bins) lane_sum += s_hist[bin];
      }
      uint32_t scan = lane_sum;
      #pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const auto y = __shfl_up_sync(0xffffffff, scan, off);
        if (static_cast<int>(tid & 31u) >= off) scan += y;
      }
      const uint32_t warp_id = tid >> 5;
      if ((tid & 31u) == 31u) s_warp_totals[warp_id] = scan;
      __syncthreads();

      if (tid < 32) {
        uint32_t warp_sum = tid < kTopKWarps ? s_warp_totals[tid] : 0u;
        uint32_t warp_scan = warp_sum;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
          const auto y = __shfl_up_sync(0xffffffff, warp_scan, off);
          if (static_cast<int>(tid) >= off) warp_scan += y;
        }
        if (tid < kTopKWarps) {
          s_warp_prefix[tid] = warp_scan - warp_sum;
        }
      }
      __syncthreads();

      uint32_t running = s_warp_prefix[warp_id] + scan - lane_sum;
      #pragma unroll
      for (uint32_t j = 0; j < kTopKBinsPerThread; ++j) {
        const uint32_t bin = bin_base + j;
        if (bin < active_bins) {
          running += s_hist[bin];
          s_prefix[bin] = running;
        }
      }
    }
    __syncthreads();

    if (tid == 0) {
      s_selected_bin = active_bins - 1u;
    }
    __syncthreads();

    const uint32_t target = keep > s_less_count ? keep - s_less_count : 1u;
    if (topk_active) {
      for (uint32_t bin = tid; bin < active_bins; bin += kTopKThreads) {
        if (s_prefix[bin] >= target) {
          atomicMin(&s_selected_bin, bin);
        }
      }
    }
    __syncthreads();

    if (tid == 0) {
      const uint32_t b = min(s_selected_bin, active_bins - 1u);
      const uint32_t before = b == 0 ? 0u : s_prefix[b - 1];
      s_less_count += before;
      s_threshold_key = selected_prefix | (b << shift);
    }
    __syncthreads();
  }

  if (tid == 0) {
    s_boundary_quota = keep > s_less_count ? keep - s_less_count : 0u;
    s_above_used = 0u;
    s_boundary_used = 0u;
  }
  __syncthreads();

  const uint32_t threshold_key = s_threshold_key;
  const uint32_t above_count = s_less_count;
  const uint32_t boundary_quota = s_boundary_quota;
  if (topk_active) {
    for (uint32_t i = tid; i < candidate_len; i += kTopKThreads) {
      const uint32_t key = keys_row[i];
      if (key == 0xffffffffu) continue;
      const uint32_t block_slot = i / kHISABlockSize;
      const uint32_t block_offset = i - block_slot * kHISABlockSize;
      int32_t token = -1;
      if (block_slot < block_topk) {
        const int32_t block_id = blocks_row[block_slot];
        if (block_id >= 0) {
          token = block_id * static_cast<int32_t>(kHISABlockSize) +
                  static_cast<int32_t>(block_offset);
        }
      }
      if (key < threshold_key) {
        const uint32_t slot = atomicAdd(&s_above_used, 1u);
        if (slot < above_count) out_row[slot] = token;
      } else if (key == threshold_key) {
        const uint32_t slot = atomicAdd(&s_boundary_used, 1u);
        if (slot < boundary_quota) out_row[above_count + slot] = token;
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
    const uint32_t boundary_written = min(s_boundary_used, s_boundary_quota);
    s_emit_count = min(keep, min(s_above_used, s_less_count) + boundary_written);
  }
  __syncthreads();

  if (topk_active) {
    for (uint32_t i = s_emit_count + tid; i < topk; i += kTopKThreads) {
      out_row[i] = -1;
    }
  }
}

template <typename IndicesT, uint32_t kPageSize, class GEMM, uint32_t kTileN>
__global__ void hisa_selector_megakernel_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISASelectorMegakernelParam param) {
  constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  constexpr uint32_t kHISABlockSize = 128;
  const uint32_t row = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  if (row >= param.q_rows) return;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  char* cursor = reinterpret_cast<char*>(smem_raw);
  float* block_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * param.block_score_capacity;
  int32_t* block_indices = reinterpret_cast<int32_t*>(cursor);
  cursor += sizeof(int32_t) * param.block_score_capacity;
  int32_t* selected_blocks = reinterpret_cast<int32_t*>(cursor);
  cursor += sizeof(int32_t) * param.effective_block_topk;
  float* candidate_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * param.candidate_capacity;
  uint16_t* candidate_ordinals = reinterpret_cast<uint16_t*>(cursor);
  cursor += sizeof(uint16_t) * param.candidate_capacity;
  cursor = hisa_align_smem(cursor, 16);
  auto gemm_smem = reinterpret_cast<void*>(cursor);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());
  __shared__ uint32_t s_hist[256];
  __shared__ uint32_t s_prefix[256];
  __shared__ uint32_t s_threshold_key;
  __shared__ uint32_t s_less_count;
  __shared__ uint32_t s_boundary_quota;
  __shared__ uint32_t s_boundary_used;
  __shared__ uint32_t s_above_used;

  const int32_t batch =
      static_cast<const int32_t*>(param.token_to_batch_idx)[row];
  if (batch < 0 || static_cast<uint32_t>(batch) >= param.batch_size) {
    for (uint32_t i = tid; i < param.topk; i += blockDim.x) {
      static_cast<int32_t*>(param.topk_indices)[
          static_cast<int64_t>(row) * param.topk + i] = -1;
    }
    return;
  }
  const int32_t prefix_len = max(0, static_cast<const int32_t*>(param.seq_lens)[batch]);
  const uint32_t block_count = min(
      static_cast<uint32_t>(static_cast<const int32_t*>(param.block_counts)[row]),
      param.max_blocks);

  for (uint32_t i = tid; i < param.block_score_capacity; i += blockDim.x) {
    block_scores[i] = -INFINITY;
    block_indices[i] = INT_MAX;
  }
  for (uint32_t i = tid; i < param.effective_block_topk; i += blockDim.x) {
    selected_blocks[i] = -1;
  }
  for (uint32_t i = tid; i < param.candidate_capacity; i += blockDim.x) {
    candidate_scores[i] = -INFINITY;
    candidate_ordinals[i] = UINT16_MAX;
  }
  __syncthreads();

  if (prefix_len <= 0 || block_count == 0) {
    for (uint32_t i = tid; i < param.topk; i += blockDim.x) {
      static_cast<int32_t*>(param.topk_indices)[
          static_cast<int64_t>(row) * param.topk + i] = -1;
    }
    return;
  }

  const auto* q_values =
      static_cast<const uint8_t*>(param.q_values) +
      static_cast<int64_t>(row) * param.n_heads * kNVFP4ValueBytes;
  const auto* q_scales =
      static_cast<const uint32_t*>(param.q_scales) +
      static_cast<int64_t>(row) * param.n_heads;
  for (uint32_t idx = tid; idx < kHISASelectorHeads * kIndexerHeadDim;
       idx += blockDim.x) {
    const uint32_t head = idx / kIndexerHeadDim;
    const uint32_t dim = idx - head * kIndexerHeadDim;
    a_shared(head, dim) = load_nvfp4_value(
        q_values + static_cast<int64_t>(head) * kNVFP4ValueBytes,
        q_scales + head,
        dim);
  }
  __syncthreads();

  const auto* rep_values = static_cast<const uint8_t*>(param.rep_values);
  const auto* rep_scales = static_cast<const uint32_t*>(param.rep_scales);
  const auto* weights =
      static_cast<const float*>(param.weights) + static_cast<int64_t>(row) * param.n_heads;
  for (uint32_t tile_start = 0; tile_start < block_count;
       tile_start += kTileN) {
    const uint32_t tile_count = min(kTileN, block_count - tile_start);
    for (uint32_t idx = tid; idx < kIndexerHeadDim * kTileN;
         idx += blockDim.x) {
      const uint32_t dim = idx / kTileN;
      const uint32_t n = idx - dim * kTileN;
      float value = 0.0f;
      if (n < tile_count) {
        const uint32_t block_id = tile_start + n;
        const int64_t rep_row =
            (static_cast<int64_t>(batch) * param.max_blocks + block_id);
        value = load_nvfp4_value(
            rep_values + rep_row * kNVFP4ValueBytes,
            rep_scales + rep_row,
            dim);
      }
      b_shared(dim, n) = value;
    }
    for (uint32_t idx = tid; idx < kHISASelectorHeads * kTileN;
         idx += blockDim.x) {
      const uint32_t head = idx / kTileN;
      const uint32_t n = idx - head * kTileN;
      c_shared(head, n) = 0.0f;
    }
    __syncthreads();

    GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
    __syncthreads();

    for (uint32_t n = tid; n < kTileN; n += blockDim.x) {
      if (n < tile_count) {
        float score = 0.0f;
        for (uint32_t head = 0; head < kHISASelectorHeads; ++head) {
          const float dot = c_shared(head, n);
          if (dot > 0.0f) score += dot * weights[head];
        }
        block_scores[tile_start + n] = score;
      }
    }
    __syncthreads();
  }

  for (uint32_t i = tid; i < param.block_score_capacity; i += blockDim.x) {
    block_indices[i] = i < block_count ? static_cast<int32_t>(i) : INT_MAX;
    if (i >= block_count) block_scores[i] = -INFINITY;
  }
  __syncthreads();
  if (tid == 0) {
    block_scores[0] = INFINITY;
    block_scores[block_count - 1] = INFINITY;
  }
  __syncthreads();

  for (uint32_t k_size = 2; k_size <= param.block_score_capacity; k_size <<= 1) {
    for (uint32_t j = k_size >> 1; j > 0; j >>= 1) {
      for (uint32_t i = tid; i < param.block_score_capacity; i += blockDim.x) {
        const uint32_t other = i ^ j;
        if (other > i && other < param.block_score_capacity) {
          const float score_i = block_scores[i];
          const float score_o = block_scores[other];
          const int32_t block_i = block_indices[i];
          const int32_t block_o = block_indices[other];
          const bool left_better = (i & k_size) == 0;
          const bool i_better =
              hisa_selector_better(score_i, block_i, score_o, block_o);
          const bool should_swap =
              (left_better && !i_better) || (!left_better && i_better);
          if (should_swap) {
            block_scores[i] = score_o;
            block_scores[other] = score_i;
            block_indices[i] = block_o;
            block_indices[other] = block_i;
          }
        }
      }
      __syncthreads();
    }
  }

  const uint32_t row_block_topk = min(
      static_cast<uint32_t>(
          max(0, static_cast<const int32_t*>(param.block_topk_counts)[row])),
      param.effective_block_topk);
  const uint32_t keep_blocks = min(row_block_topk, block_count);
  for (uint32_t slot = tid; slot < param.effective_block_topk; slot += blockDim.x) {
    selected_blocks[slot] = slot < keep_blocks ? block_indices[slot] : -1;
  }
  __syncthreads();

  if (param.effective_block_topk * kHISABlockSize == param.topk) {
    for (uint32_t i = tid; i < param.topk; i += blockDim.x) {
      const uint32_t slot = i / kHISABlockSize;
      const uint32_t offset = i - slot * kHISABlockSize;
      int32_t token = -1;
      if (slot < param.effective_block_topk) {
        const int32_t block_id = selected_blocks[slot];
        if (block_id >= 0) {
          token = block_id * static_cast<int32_t>(kHISABlockSize) +
                  static_cast<int32_t>(offset);
        }
      }
      static_cast<int32_t*>(param.topk_indices)[
          static_cast<int64_t>(row) * param.topk + i] =
          token >= 0 && token < prefix_len ? token : -1;
    }
    return;
  }

  const auto* page_table = static_cast<const IndicesT*>(param.page_table);
  const auto* cache = static_cast<const uint8_t*>(param.cache);
  const uint32_t candidate_len =
      min(param.effective_block_topk * kHISABlockSize, param.candidate_capacity);
  for (uint32_t tile_start = 0; tile_start < candidate_len; tile_start += kTileN) {
    const uint32_t tile_count = min(kTileN, candidate_len - tile_start);
    for (uint32_t idx = tid; idx < kIndexerHeadDim * kTileN;
         idx += blockDim.x) {
      const uint32_t dim = idx / kTileN;
      const uint32_t n = idx - dim * kTileN;
      const uint32_t candidate_pos = tile_start + n;
      const uint32_t slot = candidate_pos / kHISABlockSize;
      const uint32_t offset = candidate_pos - slot * kHISABlockSize;
      const int32_t block_id =
          slot < param.effective_block_topk ? selected_blocks[slot] : -1;
      const int32_t block_token_start =
          block_id * static_cast<int32_t>(kHISABlockSize);
      const int32_t token = block_token_start + static_cast<int32_t>(offset);
      float value = 0.0f;
      if (n < tile_count && block_id >= 0 && token >= 0 && token < prefix_len) {
        const uint32_t logical_page = static_cast<uint32_t>(token) / kPageSize;
        const uint32_t offset = static_cast<uint32_t>(token) & (kPageSize - 1);
        const auto page =
            page_table[static_cast<int64_t>(batch) * param.page_table_stride +
                       logical_page];
        if (page >= 0) {
          const auto* page_ptr = cache + static_cast<int64_t>(page) * kPageBytes;
          const auto* value_ptr = page_ptr + offset * kNVFP4ValueBytes;
          const auto* scale_ptr = reinterpret_cast<const uint32_t*>(
              page_ptr + kNVFP4ValueBytes * kPageSize + offset * kScaleBytes);
          value = load_nvfp4_value(value_ptr, scale_ptr, dim);
        }
      }
      b_shared(dim, n) = value;
    }
    for (uint32_t idx = tid; idx < kHISASelectorHeads * kTileN;
         idx += blockDim.x) {
      const uint32_t head = idx / kTileN;
      const uint32_t n = idx - head * kTileN;
      c_shared(head, n) = 0.0f;
    }
    __syncthreads();

    GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
    __syncthreads();

    for (uint32_t n = tid; n < kTileN; n += blockDim.x) {
      const uint32_t candidate_pos = tile_start + n;
      if (candidate_pos >= param.candidate_capacity) continue;
      const uint32_t slot = candidate_pos / kHISABlockSize;
      const uint32_t offset = candidate_pos - slot * kHISABlockSize;
      const int32_t block_id =
          slot < param.effective_block_topk ? selected_blocks[slot] : -1;
      const int32_t block_token_start =
          block_id * static_cast<int32_t>(kHISABlockSize);
      const int32_t token = block_token_start + static_cast<int32_t>(offset);
      float score = -INFINITY;
      uint16_t ordinal = UINT16_MAX;
      if (n < tile_count && block_id >= 0 && token >= 0 && token < prefix_len) {
        score = 0.0f;
        for (uint32_t head = 0; head < kHISASelectorHeads; ++head) {
          const float dot = c_shared(head, n);
          if (dot > 0.0f) score += dot * weights[head];
        }
        ordinal = static_cast<uint16_t>(candidate_pos);
      }
      candidate_scores[candidate_pos] = score;
      candidate_ordinals[candidate_pos] = ordinal;
    }
    __syncthreads();
  }
  const uint32_t keep = min(param.topk, candidate_len);
  if (tid == 0) {
    s_threshold_key = 0;
    s_less_count = 0;
    s_above_used = 0;
    s_boundary_used = 0;
  }
  __syncthreads();

  #pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    const auto shift = 24u - static_cast<uint32_t>(pass) * 8u;
    const auto prefix_mask =
        pass == 0 ? 0u : (0xffffffffu << (shift + 8u));
    const auto selected_prefix = s_threshold_key & prefix_mask;

    for (uint32_t bin = tid; bin < 256; bin += blockDim.x) {
      s_hist[bin] = 0;
    }
    __syncthreads();

    for (uint32_t i = tid; i < candidate_len; i += blockDim.x) {
      const bool valid = candidate_ordinals[i] != UINT16_MAX;
      const auto key =
          valid ? fp32_to_radix_desc(__float_as_uint(candidate_scores[i])) : 0xffffffffu;
      if (pass == 0 || ((key & prefix_mask) == selected_prefix)) {
        atomicAdd(&s_hist[(key >> shift) & 0xffu], 1u);
      }
    }
    __syncthreads();

    if (tid < 32) {
      uint32_t lane_sum = 0;
      for (uint32_t b = tid * 8; b < (tid + 1) * 8; ++b) lane_sum += s_hist[b];
      uint32_t scan = lane_sum;
      #pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const auto y = __shfl_up_sync(0xffffffff, scan, off);
        if (static_cast<int>(tid) >= off) scan += y;
      }
      const auto base = scan - lane_sum;
      uint32_t running = base;
      for (uint32_t b = tid * 8; b < (tid + 1) * 8; ++b) {
        running += s_hist[b];
        s_prefix[b] = running;
      }
    }
    __syncthreads();

    if (tid == 0) {
      const auto target = keep > s_less_count ? keep - s_less_count : 1u;
      uint32_t b = 0;
      while (b + 1 < 256 && s_prefix[b] < target) ++b;
      const auto before = b == 0 ? 0u : s_prefix[b - 1];
      s_less_count += before;
      s_threshold_key = selected_prefix | (b << shift);
    }
    __syncthreads();
  }

  if (tid == 0) {
    s_boundary_quota = keep > s_less_count ? keep - s_less_count : 0u;
    s_above_used = 0;
    s_boundary_used = 0;
  }
  __syncthreads();

  const auto threshold_key = s_threshold_key;
  const auto above_count = s_less_count;
  const auto boundary_quota = s_boundary_quota;
  auto* out_row =
      static_cast<int32_t*>(param.topk_indices) + static_cast<int64_t>(row) * param.topk;
  for (uint32_t i = tid; i < candidate_len; i += blockDim.x) {
    const bool valid = candidate_ordinals[i] != UINT16_MAX;
    const auto key =
        valid ? fp32_to_radix_desc(__float_as_uint(candidate_scores[i])) : 0xffffffffu;
    if (!valid) continue;
    const uint32_t slot_id = i / kHISABlockSize;
    const uint32_t offset = i - slot_id * kHISABlockSize;
    const int32_t block_id =
        slot_id < param.effective_block_topk ? selected_blocks[slot_id] : -1;
    const int32_t token = block_id >= 0
                              ? block_id * static_cast<int32_t>(kHISABlockSize) +
                                    static_cast<int32_t>(offset)
                              : -1;
    if (key < threshold_key) {
      const auto out = atomicAdd(&s_above_used, 1u);
      if (out < above_count) out_row[out] = token;
    } else if (key == threshold_key) {
      const auto out = atomicAdd(&s_boundary_used, 1u);
      if (out < boundary_quota) out_row[above_count + out] = token;
    }
  }
  __syncthreads();
  for (uint32_t i = keep + tid; i < param.topk; i += blockDim.x) {
    out_row[i] = -1;
  }
}

template <class GEMM>
__global__ void hisa_selector_parallel_select_blocks_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISASelectorParallelSelectParam param) {
  const uint32_t row = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  if (row >= param.q_rows) return;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  char* cursor = reinterpret_cast<char*>(smem_raw);
  float* block_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * param.block_score_capacity;
  int32_t* block_indices = reinterpret_cast<int32_t*>(cursor);
  cursor += sizeof(int32_t) * param.block_score_capacity;
  cursor = hisa_align_smem(cursor, 16);
  auto gemm_smem = reinterpret_cast<void*>(cursor);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

  auto* selected_blocks =
      static_cast<int32_t*>(param.selected_blocks) +
      static_cast<int64_t>(row) * param.effective_block_topk;
  for (uint32_t i = tid; i < param.effective_block_topk; i += blockDim.x) {
    selected_blocks[i] = -1;
  }

  const int32_t batch =
      static_cast<const int32_t*>(param.token_to_batch_idx)[row];
  if (batch < 0 || static_cast<uint32_t>(batch) >= param.batch_size) {
    return;
  }
  const int32_t prefix_len = max(0, static_cast<const int32_t*>(param.seq_lens)[batch]);
  const uint32_t block_count = min(
      static_cast<uint32_t>(static_cast<const int32_t*>(param.block_counts)[row]),
      param.max_blocks);

  for (uint32_t i = tid; i < param.block_score_capacity; i += blockDim.x) {
    block_scores[i] = -INFINITY;
    block_indices[i] = INT_MAX;
  }
  __syncthreads();
  if (prefix_len <= 0 || block_count == 0) return;

  const auto* q_values =
      static_cast<const uint8_t*>(param.q_values) +
      static_cast<int64_t>(row) * param.n_heads * kNVFP4ValueBytes;
  const auto* q_scales =
      static_cast<const uint32_t*>(param.q_scales) +
      static_cast<int64_t>(row) * param.n_heads;
  for (uint32_t idx = tid; idx < kHISASelectorHeads * kIndexerHeadDim;
       idx += blockDim.x) {
    const uint32_t head = idx / kIndexerHeadDim;
    const uint32_t dim = idx - head * kIndexerHeadDim;
    a_shared(head, dim) = load_nvfp4_value(
        q_values + static_cast<int64_t>(head) * kNVFP4ValueBytes,
        q_scales + head,
        dim);
  }
  __syncthreads();

  const auto* rep_values = static_cast<const uint8_t*>(param.rep_values);
  const auto* rep_scales = static_cast<const uint32_t*>(param.rep_scales);
  const auto* weights =
      static_cast<const float*>(param.weights) + static_cast<int64_t>(row) * param.n_heads;
  for (uint32_t tile_start = 0; tile_start < block_count;
       tile_start += kHISASelectorTileN) {
    const uint32_t tile_count = min(kHISASelectorTileN, block_count - tile_start);
    for (uint32_t idx = tid; idx < kIndexerHeadDim * kHISASelectorTileN;
         idx += blockDim.x) {
      const uint32_t dim = idx / kHISASelectorTileN;
      const uint32_t n = idx - dim * kHISASelectorTileN;
      float value = 0.0f;
      if (n < tile_count) {
        const uint32_t block_id = tile_start + n;
        const int64_t rep_row =
            static_cast<int64_t>(batch) * param.max_blocks + block_id;
        value = load_nvfp4_value(
            rep_values + rep_row * kNVFP4ValueBytes,
            rep_scales + rep_row,
            dim);
      }
      b_shared(dim, n) = value;
    }
    for (uint32_t idx = tid; idx < kHISASelectorHeads * kHISASelectorTileN;
         idx += blockDim.x) {
      const uint32_t head = idx / kHISASelectorTileN;
      const uint32_t n = idx - head * kHISASelectorTileN;
      c_shared(head, n) = 0.0f;
    }
    __syncthreads();

    GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
    __syncthreads();

    for (uint32_t n = tid; n < kHISASelectorTileN; n += blockDim.x) {
      if (n < tile_count) {
        float score = 0.0f;
        for (uint32_t head = 0; head < kHISASelectorHeads; ++head) {
          const float dot = c_shared(head, n);
          if (dot > 0.0f) score += dot * weights[head];
        }
        block_scores[tile_start + n] = score;
      }
    }
    __syncthreads();
  }

  for (uint32_t i = tid; i < param.block_score_capacity; i += blockDim.x) {
    block_indices[i] = i < block_count ? static_cast<int32_t>(i) : INT_MAX;
    if (i >= block_count) block_scores[i] = -INFINITY;
  }
  __syncthreads();
  if (tid == 0) {
    block_scores[0] = INFINITY;
    block_scores[block_count - 1] = INFINITY;
  }
  __syncthreads();

  for (uint32_t k_size = 2; k_size <= param.block_score_capacity; k_size <<= 1) {
    for (uint32_t j = k_size >> 1; j > 0; j >>= 1) {
      for (uint32_t i = tid; i < param.block_score_capacity; i += blockDim.x) {
        const uint32_t other = i ^ j;
        if (other > i && other < param.block_score_capacity) {
          const float score_i = block_scores[i];
          const float score_o = block_scores[other];
          const int32_t block_i = block_indices[i];
          const int32_t block_o = block_indices[other];
          const bool left_better = (i & k_size) == 0;
          const bool i_better =
              hisa_selector_better(score_i, block_i, score_o, block_o);
          const bool should_swap =
              (left_better && !i_better) || (!left_better && i_better);
          if (should_swap) {
            block_scores[i] = score_o;
            block_scores[other] = score_i;
            block_indices[i] = block_o;
            block_indices[other] = block_i;
          }
        }
      }
      __syncthreads();
    }
  }

  const uint32_t row_block_topk = min(
      static_cast<uint32_t>(
          max(0, static_cast<const int32_t*>(param.block_topk_counts)[row])),
      param.effective_block_topk);
  const uint32_t keep_blocks = min(row_block_topk, block_count);
  for (uint32_t slot = tid; slot < param.effective_block_topk; slot += blockDim.x) {
    selected_blocks[slot] = slot < keep_blocks ? block_indices[slot] : -1;
  }
}

template <typename IndicesT, uint32_t kPageSize, class GEMM>
__global__ void hisa_selector_parallel_score_candidates_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISASelectorParallelScoreParam param) {
  constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  constexpr uint32_t kHISABlockSize = 128;
  const uint32_t row = blockIdx.x;
  const uint32_t block_slot = blockIdx.y;
  const uint32_t tid = threadIdx.x;
  if (row >= param.q_rows || block_slot >= param.effective_block_topk) return;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto gemm_smem = reinterpret_cast<void*>(smem_raw);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

  const int32_t batch =
      static_cast<const int32_t*>(param.token_to_batch_idx)[row];
  const int32_t prefix_len =
      (batch >= 0 && static_cast<uint32_t>(batch) < param.batch_size)
          ? max(0, static_cast<const int32_t*>(param.seq_lens)[batch])
          : 0;
  const int32_t block_id =
      static_cast<const int32_t*>(param.selected_blocks)[
          static_cast<int64_t>(row) * param.effective_block_topk + block_slot];
  const int32_t block_token_start =
      block_id * static_cast<int32_t>(kHISABlockSize);

  const auto* q_values =
      static_cast<const uint8_t*>(param.q_values) +
      static_cast<int64_t>(row) * param.n_heads * kNVFP4ValueBytes;
  const auto* q_scales =
      static_cast<const uint32_t*>(param.q_scales) +
      static_cast<int64_t>(row) * param.n_heads;
  for (uint32_t idx = tid; idx < kHISASelectorHeads * kIndexerHeadDim;
       idx += blockDim.x) {
    const uint32_t head = idx / kIndexerHeadDim;
    const uint32_t dim = idx - head * kIndexerHeadDim;
    a_shared(head, dim) = load_nvfp4_value(
        q_values + static_cast<int64_t>(head) * kNVFP4ValueBytes,
        q_scales + head,
        dim);
  }

  const auto* page_table = static_cast<const IndicesT*>(param.page_table);
  const auto* cache = static_cast<const uint8_t*>(param.cache);
  for (uint32_t idx = tid; idx < kIndexerHeadDim * kHISASelectorTileN;
       idx += blockDim.x) {
    const uint32_t dim = idx / kHISASelectorTileN;
    const uint32_t n = idx - dim * kHISASelectorTileN;
    const int32_t token = block_token_start + static_cast<int32_t>(n);
    float value = 0.0f;
    if (batch >= 0 && block_id >= 0 && token >= 0 && token < prefix_len) {
      const uint32_t logical_page = static_cast<uint32_t>(token) / kPageSize;
      const uint32_t offset = static_cast<uint32_t>(token) & (kPageSize - 1);
      const auto page =
          page_table[static_cast<int64_t>(batch) * param.page_table_stride +
                     logical_page];
      if (page >= 0) {
        const auto* page_ptr = cache + static_cast<int64_t>(page) * kPageBytes;
        const auto* value_ptr = page_ptr + offset * kNVFP4ValueBytes;
        const auto* scale_ptr = reinterpret_cast<const uint32_t*>(
            page_ptr + kNVFP4ValueBytes * kPageSize + offset * kScaleBytes);
        value = load_nvfp4_value(value_ptr, scale_ptr, dim);
      }
    }
    b_shared(dim, n) = value;
  }
  for (uint32_t idx = tid; idx < kHISASelectorHeads * kHISASelectorTileN;
       idx += blockDim.x) {
    const uint32_t head = idx / kHISASelectorTileN;
    const uint32_t n = idx - head * kHISASelectorTileN;
    c_shared(head, n) = 0.0f;
  }
  __syncthreads();

  GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
  __syncthreads();

  const auto* weights =
      static_cast<const float*>(param.weights) + static_cast<int64_t>(row) * param.n_heads;
  auto* logits_row =
      static_cast<float*>(param.logits) + static_cast<int64_t>(row) * param.candidate_len;
  for (uint32_t n = tid; n < kHISASelectorTileN; n += blockDim.x) {
    const uint32_t candidate_pos = block_slot * kHISABlockSize + n;
    if (candidate_pos >= param.candidate_len) continue;
    const int32_t token = block_token_start + static_cast<int32_t>(n);
    float score = -INFINITY;
    if (block_id >= 0 && token >= 0 && token < prefix_len) {
      score = 0.0f;
      for (uint32_t head = 0; head < kHISASelectorHeads; ++head) {
        const float dot = c_shared(head, n);
        if (dot > 0.0f) score += dot * weights[head];
      }
    }
    logits_row[candidate_pos] = score;
  }
}

template <typename IndicesT, uint32_t kPageSize, class GEMM>
HISA_SELECTOR_CLUSTER_KERNEL void hisa_selector_cluster_fused_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISASelectorClusterFusedParam param) {
  constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  constexpr uint32_t kHISABlockSize = 128;
  constexpr uint32_t kBins = 256;
  const uint32_t row = blockIdx.x;
  const uint32_t cluster_rank = blockIdx.y;
  const uint32_t tid = threadIdx.x;
  if (row >= param.q_rows) return;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  char* cursor = reinterpret_cast<char*>(smem_raw);
  float* block_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * param.block_score_capacity;
  int32_t* block_indices = reinterpret_cast<int32_t*>(cursor);
  cursor += sizeof(int32_t) * param.block_score_capacity;
  int32_t* selected_blocks = reinterpret_cast<int32_t*>(cursor);
  cursor += sizeof(int32_t) * param.effective_block_topk;
  cursor = hisa_align_smem(cursor, 16);
  auto gemm_smem = reinterpret_cast<void*>(cursor);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

  __shared__ uint32_t s_hist[kBins];
  __shared__ uint32_t s_prefix[kBins];
  __shared__ uint32_t s_threshold_key;
  __shared__ uint32_t s_less_count;
  __shared__ uint32_t s_boundary_quota;
  __shared__ uint32_t s_boundary_used;
  __shared__ uint32_t s_above_used;
  __shared__ uint32_t s_emit_count;

  auto* out_row =
      static_cast<int32_t*>(param.topk_indices) + static_cast<int64_t>(row) * param.topk;

  const int32_t batch =
      static_cast<const int32_t*>(param.token_to_batch_idx)[row];
  if (batch < 0 || static_cast<uint32_t>(batch) >= param.batch_size) {
    if (cluster_rank == 0) {
      for (uint32_t i = tid; i < param.topk; i += blockDim.x) out_row[i] = -1;
    }
    return;
  }
  const int32_t prefix_len = max(0, static_cast<const int32_t*>(param.seq_lens)[batch]);
  const uint32_t block_count = min(
      static_cast<uint32_t>(static_cast<const int32_t*>(param.block_counts)[row]),
      param.max_blocks);

  for (uint32_t i = tid; i < param.block_score_capacity; i += blockDim.x) {
    block_scores[i] = -INFINITY;
    block_indices[i] = INT_MAX;
  }
  for (uint32_t i = tid; i < param.effective_block_topk; i += blockDim.x) {
    selected_blocks[i] = -1;
  }
  __syncthreads();

  if (prefix_len <= 0 || block_count == 0) {
    if (cluster_rank == 0) {
      for (uint32_t i = tid; i < param.topk; i += blockDim.x) out_row[i] = -1;
    }
    return;
  }

  const auto* q_values =
      static_cast<const uint8_t*>(param.q_values) +
      static_cast<int64_t>(row) * param.n_heads * kNVFP4ValueBytes;
  const auto* q_scales =
      static_cast<const uint32_t*>(param.q_scales) +
      static_cast<int64_t>(row) * param.n_heads;
  for (uint32_t idx = tid; idx < kHISASelectorHeads * kIndexerHeadDim;
       idx += blockDim.x) {
    const uint32_t head = idx / kIndexerHeadDim;
    const uint32_t dim = idx - head * kIndexerHeadDim;
    a_shared(head, dim) = load_nvfp4_value(
        q_values + static_cast<int64_t>(head) * kNVFP4ValueBytes,
        q_scales + head,
        dim);
  }
  __syncthreads();

  const auto* rep_values = static_cast<const uint8_t*>(param.rep_values);
  const auto* rep_scales = static_cast<const uint32_t*>(param.rep_scales);
  const auto* weights =
      static_cast<const float*>(param.weights) + static_cast<int64_t>(row) * param.n_heads;
  for (uint32_t tile_start = 0; tile_start < block_count;
       tile_start += kHISASelectorTileN) {
    const uint32_t tile_count = min(kHISASelectorTileN, block_count - tile_start);
    for (uint32_t idx = tid; idx < kIndexerHeadDim * kHISASelectorTileN;
         idx += blockDim.x) {
      const uint32_t dim = idx / kHISASelectorTileN;
      const uint32_t n = idx - dim * kHISASelectorTileN;
      float value = 0.0f;
      if (n < tile_count) {
        const uint32_t block_id = tile_start + n;
        const int64_t rep_row =
            static_cast<int64_t>(batch) * param.max_blocks + block_id;
        value = load_nvfp4_value(
            rep_values + rep_row * kNVFP4ValueBytes,
            rep_scales + rep_row,
            dim);
      }
      b_shared(dim, n) = value;
    }
    for (uint32_t idx = tid; idx < kHISASelectorHeads * kHISASelectorTileN;
         idx += blockDim.x) {
      const uint32_t head = idx / kHISASelectorTileN;
      const uint32_t n = idx - head * kHISASelectorTileN;
      c_shared(head, n) = 0.0f;
    }
    __syncthreads();

    GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
    __syncthreads();

    for (uint32_t n = tid; n < kHISASelectorTileN; n += blockDim.x) {
      if (n < tile_count) {
        float score = 0.0f;
        for (uint32_t head = 0; head < kHISASelectorHeads; ++head) {
          const float dot = c_shared(head, n);
          if (dot > 0.0f) score += dot * weights[head];
        }
        block_scores[tile_start + n] = score;
      }
    }
    __syncthreads();
  }

  for (uint32_t i = tid; i < param.block_score_capacity; i += blockDim.x) {
    block_indices[i] = i < block_count ? static_cast<int32_t>(i) : INT_MAX;
    if (i >= block_count) block_scores[i] = -INFINITY;
  }
  __syncthreads();
  if (tid == 0) {
    block_scores[0] = INFINITY;
    block_scores[block_count - 1] = INFINITY;
  }
  __syncthreads();

  for (uint32_t k_size = 2; k_size <= param.block_score_capacity; k_size <<= 1) {
    for (uint32_t j = k_size >> 1; j > 0; j >>= 1) {
      for (uint32_t i = tid; i < param.block_score_capacity; i += blockDim.x) {
        const uint32_t other = i ^ j;
        if (other > i && other < param.block_score_capacity) {
          const float score_i = block_scores[i];
          const float score_o = block_scores[other];
          const int32_t block_i = block_indices[i];
          const int32_t block_o = block_indices[other];
          const bool left_better = (i & k_size) == 0;
          const bool i_better =
              hisa_selector_better(score_i, block_i, score_o, block_o);
          const bool should_swap =
              (left_better && !i_better) || (!left_better && i_better);
          if (should_swap) {
            block_scores[i] = score_o;
            block_scores[other] = score_i;
            block_indices[i] = block_o;
            block_indices[other] = block_i;
          }
        }
      }
      __syncthreads();
    }
  }

  const uint32_t row_block_topk = min(
      static_cast<uint32_t>(
          max(0, static_cast<const int32_t*>(param.block_topk_counts)[row])),
      param.effective_block_topk);
  const uint32_t keep_blocks = min(row_block_topk, block_count);
  for (uint32_t slot = tid; slot < param.effective_block_topk; slot += blockDim.x) {
    selected_blocks[slot] = slot < keep_blocks ? block_indices[slot] : -1;
  }
  __syncthreads();

  if (param.candidate_len == param.topk) {
    if (cluster_rank == 0) {
      for (uint32_t i = tid; i < param.topk; i += blockDim.x) {
        const uint32_t slot = i / kHISABlockSize;
        const uint32_t offset = i - slot * kHISABlockSize;
        int32_t token = -1;
        if (slot < param.effective_block_topk) {
          const int32_t block_id = selected_blocks[slot];
          if (block_id >= 0) {
            token = block_id * static_cast<int32_t>(kHISABlockSize) +
                    static_cast<int32_t>(offset);
          }
        }
        out_row[i] = token >= 0 && token < prefix_len ? token : -1;
      }
    }
    return;
  }

  const auto* page_table = static_cast<const IndicesT*>(param.page_table);
  const auto* cache = static_cast<const uint8_t*>(param.cache);
  auto* candidate_keys =
      static_cast<uint32_t*>(param.candidate_keys) +
      static_cast<int64_t>(row) * param.candidate_len;
  for (uint32_t tile_start = cluster_rank * kHISASelectorTileN;
       tile_start < param.candidate_len;
       tile_start += kHISASelectorTileN * kHISASelectorClusterSize) {
    const uint32_t tile_count = min(kHISASelectorTileN, param.candidate_len - tile_start);
    for (uint32_t idx = tid; idx < kIndexerHeadDim * kHISASelectorTileN;
         idx += blockDim.x) {
      const uint32_t dim = idx / kHISASelectorTileN;
      const uint32_t n = idx - dim * kHISASelectorTileN;
      const uint32_t candidate_pos = tile_start + n;
      const uint32_t slot = candidate_pos / kHISABlockSize;
      const uint32_t offset = candidate_pos - slot * kHISABlockSize;
      const int32_t block_id =
          slot < param.effective_block_topk ? selected_blocks[slot] : -1;
      const int32_t block_token_start =
          block_id * static_cast<int32_t>(kHISABlockSize);
      const int32_t token = block_token_start + static_cast<int32_t>(offset);
      float value = 0.0f;
      if (n < tile_count && block_id >= 0 && token >= 0 && token < prefix_len) {
        const uint32_t logical_page = static_cast<uint32_t>(token) / kPageSize;
        const uint32_t page_offset = static_cast<uint32_t>(token) & (kPageSize - 1);
        const auto page =
            page_table[static_cast<int64_t>(batch) * param.page_table_stride +
                       logical_page];
        if (page >= 0) {
          const auto* page_ptr = cache + static_cast<int64_t>(page) * kPageBytes;
          const auto* value_ptr = page_ptr + page_offset * kNVFP4ValueBytes;
          const auto* scale_ptr = reinterpret_cast<const uint32_t*>(
              page_ptr + kNVFP4ValueBytes * kPageSize + page_offset * kScaleBytes);
          value = load_nvfp4_value(value_ptr, scale_ptr, dim);
        }
      }
      b_shared(dim, n) = value;
    }
    for (uint32_t idx = tid; idx < kHISASelectorHeads * kHISASelectorTileN;
         idx += blockDim.x) {
      const uint32_t head = idx / kHISASelectorTileN;
      const uint32_t n = idx - head * kHISASelectorTileN;
      c_shared(head, n) = 0.0f;
    }
    __syncthreads();

    GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
    __syncthreads();

    for (uint32_t n = tid; n < kHISASelectorTileN; n += blockDim.x) {
      const uint32_t candidate_pos = tile_start + n;
      if (n >= tile_count || candidate_pos >= param.candidate_len) continue;
      const uint32_t slot = candidate_pos / kHISABlockSize;
      const uint32_t offset = candidate_pos - slot * kHISABlockSize;
      const int32_t block_id =
          slot < param.effective_block_topk ? selected_blocks[slot] : -1;
      const int32_t block_token_start =
          block_id * static_cast<int32_t>(kHISABlockSize);
      const int32_t token = block_token_start + static_cast<int32_t>(offset);
      uint32_t key = 0xffffffffu;
      if (block_id >= 0 && token >= 0 && token < prefix_len) {
        float score = 0.0f;
        for (uint32_t head = 0; head < kHISASelectorHeads; ++head) {
          const float dot = c_shared(head, n);
          if (dot > 0.0f) score += dot * weights[head];
        }
        key = fp32_to_radix_desc(__float_as_uint(score));
      }
      candidate_keys[candidate_pos] = key;
    }
    __syncthreads();
  }

  __threadfence();
  cooperative_groups::this_cluster().sync();
  if (cluster_rank != 0) return;

  const uint32_t keep = min(param.topk, param.candidate_len);
  if (tid == 0) {
    s_threshold_key = 0u;
    s_less_count = 0;
    s_above_used = 0;
    s_boundary_used = 0;
    s_emit_count = 0;
  }
  __syncthreads();

  #pragma unroll
  for (uint32_t pass = 0; pass < 4; ++pass) {
    const uint32_t shift = 24u - pass * 8u;
    const uint32_t prefix_mask =
        pass == 0 ? 0u : (0xffffffffu << (shift + 8u));
    const uint32_t selected_prefix = s_threshold_key & prefix_mask;

    for (uint32_t bin = tid; bin < kBins; bin += blockDim.x) s_hist[bin] = 0;
    __syncthreads();

    for (uint32_t i = tid; i < param.candidate_len; i += blockDim.x) {
      const uint32_t key = candidate_keys[i];
      if (key != 0xffffffffu &&
          (pass == 0 || ((key & prefix_mask) == selected_prefix))) {
        atomicAdd(&s_hist[(key >> shift) & 0xffu], 1u);
      }
    }
    __syncthreads();

    if (tid < 32) {
      uint32_t lane_sum = 0;
      for (uint32_t b = tid * 8; b < (tid + 1) * 8; ++b) lane_sum += s_hist[b];
      uint32_t scan = lane_sum;
      #pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const auto y = __shfl_up_sync(0xffffffff, scan, off);
        if (static_cast<int>(tid) >= off) scan += y;
      }
      const auto base = scan - lane_sum;
      uint32_t running = base;
      for (uint32_t b = tid * 8; b < (tid + 1) * 8; ++b) {
        running += s_hist[b];
        s_prefix[b] = running;
      }
    }
    __syncthreads();

    if (tid == 0) {
      const uint32_t target = keep > s_less_count ? keep - s_less_count : 1u;
      uint32_t b = 0;
      while (b + 1 < kBins && s_prefix[b] < target) ++b;
      const uint32_t before = b == 0 ? 0u : s_prefix[b - 1];
      s_less_count += before;
      s_threshold_key = selected_prefix | (b << shift);
    }
    __syncthreads();
  }

  if (tid == 0) {
    s_boundary_quota = keep > s_less_count ? keep - s_less_count : 0u;
    s_above_used = 0;
    s_boundary_used = 0;
  }
  __syncthreads();

  const uint32_t threshold_key = s_threshold_key;
  const uint32_t above_count = s_less_count;
  const uint32_t boundary_quota = s_boundary_quota;
  for (uint32_t i = tid; i < param.candidate_len; i += blockDim.x) {
    const uint32_t key = candidate_keys[i];
    if (key == 0xffffffffu) continue;
    const uint32_t slot = i / kHISABlockSize;
    const uint32_t offset = i - slot * kHISABlockSize;
    const int32_t block_id =
        slot < param.effective_block_topk ? selected_blocks[slot] : -1;
    const int32_t token =
        block_id >= 0 ? block_id * static_cast<int32_t>(kHISABlockSize) +
                            static_cast<int32_t>(offset)
                      : -1;
    if (key < threshold_key) {
      const auto out = atomicAdd(&s_above_used, 1u);
      if (out < above_count) out_row[out] = token;
    } else if (key == threshold_key) {
      const auto out = atomicAdd(&s_boundary_used, 1u);
      if (out < boundary_quota) out_row[above_count + out] = token;
    }
  }
  __syncthreads();
  if (tid == 0) {
    const uint32_t boundary_written = min(s_boundary_used, boundary_quota);
    s_emit_count = min(keep, min(s_above_used, above_count) + boundary_written);
  }
  __syncthreads();
  for (uint32_t i = s_emit_count + tid; i < param.topk; i += blockDim.x) {
    out_row[i] = -1;
  }
}
#endif

SGL_DEVICE uint32_t pack_eight_e2m1(
    float x0, float x1, float y0, float y1, float next_x0,
    float next_x1, float next_y0, float next_y1, float inv_scale) {
  float values[8] = {
      x0 * inv_scale,
      x1 * inv_scale,
      y0 * inv_scale,
      y1 * inv_scale,
      next_x0 * inv_scale,
      next_x1 * inv_scale,
      next_y0 * inv_scale,
      next_y1 * inv_scale,
  };
  return clear_e2m1_signed_zero(fp32_vec_to_e2m1(values));
}

SGL_DEVICE uint32_t pack_even_lane_e2m1(
    float x0, float x1, float y0, float y1, float next_x0,
    float next_x1, float next_y0, float next_y1, float inv_scale) {
  return pack_eight_e2m1(
      x0, x1, y0, y1, next_x0, next_x1, next_y0, next_y1, inv_scale);
}

template <typename KeyT>
SGL_DEVICE uint32_t quantize_indexer_row(
    const void* __restrict__ input,
    void* __restrict__ values,
    void* __restrict__ scales,
    uint32_t input_row_id,
    uint32_t output_row_id) {
  using namespace device;
  using KeyT2 = packed_t<KeyT>;
  using InStorage = AlignedVector<KeyT2, 2>;

  const auto lane_id = threadIdx.x % 32;
  const auto elems = static_cast<const InStorage*>(input)[input_row_id * 32 + lane_id];
  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);
  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);
  const auto local_max = fmaxf(fmaxf(fabs(x0), fabs(x1)), fmaxf(fabs(y0), fabs(y1)));
  const auto group_max = reduce_max_width8(local_max);
  const auto scale_exp = ceil_to_ue8m0_exp(fmaxf(1e-4f, group_max) / kE2M1Max);
  const auto scale = __uint_as_float(scale_exp << 23);
  const auto inv_scale = reciprocal_approximate_ftz(scale);

  const auto next_x0 = __shfl_down_sync(0xffffffff, x0, 1);
  const auto next_x1 = __shfl_down_sync(0xffffffff, x1, 1);
  const auto next_y0 = __shfl_down_sync(0xffffffff, y0, 1);
  const auto next_y1 = __shfl_down_sync(0xffffffff, y1, 1);
  if ((lane_id & 1) == 0) {
    const auto packed =
        pack_even_lane_e2m1(x0, x1, y0, y1, next_x0, next_x1, next_y0, next_y1,
                            inv_scale);
    static_cast<uint32_t*>(values)[output_row_id * 16 + lane_id / 2] =
        packed;
  }

  const auto scale_word = pack_scale_word(scale_exp);
  if (lane_id == 0) {
    static_cast<uint32_t*>(scales)[output_row_id] = scale_word;
  }
  return scale_word;
}

template <typename KeyT>
SGL_DEVICE void quantize_indexer_row_half_warp(
    const void* __restrict__ input,
    void* __restrict__ values,
    void* __restrict__ scales,
    uint32_t input_row_id,
    uint32_t output_row_id) {
  using namespace device;
  using KeyT2 = packed_t<KeyT>;
  using InStorage = AlignedVector<KeyT2, 4>;

  const auto lane_id = threadIdx.x & 31;
  const auto lane_row = lane_id & 15;
  const auto row_mask = 0x0000ffffu << (lane_id & 16);
  const auto elems =
      static_cast<const InStorage*>(input)[input_row_id * 16 + lane_row];
  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);
  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);
  const auto [z0, z1] = cast<fp32x2_t>(elems[2]);
  const auto [w0, w1] = cast<fp32x2_t>(elems[3]);
  const auto local_max =
      fmaxf(fmaxf(fmaxf(fabs(x0), fabs(x1)), fmaxf(fabs(y0), fabs(y1))),
            fmaxf(fmaxf(fabs(z0), fabs(z1)), fmaxf(fabs(w0), fabs(w1))));
  const auto group_max = reduce_max_width4(local_max, row_mask);
  const auto scale_exp =
      ceil_to_ue8m0_exp(fmaxf(1e-4f, group_max) / kE2M1Max);
  const auto scale = __uint_as_float(scale_exp << 23);
  const auto inv_scale = reciprocal_approximate_ftz(scale);
  const auto packed =
      pack_eight_e2m1(x0, x1, y0, y1, z0, z1, w0, w1, inv_scale);
  static_cast<uint32_t*>(values)[output_row_id * 16 + lane_row] = packed;

  const auto scale_word = pack_half_warp_scale_word(scale_exp, row_mask);
  if (lane_row == 0) {
    static_cast<uint32_t*>(scales)[output_row_id] = scale_word;
  }
}

template <bool kBranchlessSign>
__global__ void dequantize_indexer_nvfp4(
    const __grid_constant__ NVFP4IndexerDequantParam param) {
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_row = global_tid / 8;
  if (global_row >= param.num_rows) return;

  const auto lane_row = global_tid & 7;
  const auto scale_word = static_cast<const uint32_t*>(param.scales)[global_row];
  auto* output_vec = reinterpret_cast<float4*>(
      static_cast<float*>(param.output) +
      static_cast<int64_t>(global_row) * kIndexerHeadDim);

  #pragma unroll
  for (int word_step = 0; word_step < 2; ++word_step) {
    const auto word_idx = lane_row + word_step * 8;
    const auto value_word = static_cast<const uint32_t*>(param.values)[
        static_cast<int64_t>(global_row) * 16 + word_idx];
    const auto scale_exp = (scale_word >> ((word_idx >> 2) * 8)) & 0xffu;
    const auto scale = __uint_as_float(scale_exp << 23);

    float decoded[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      decoded[i] = decode_e2m1_nibble<kBranchlessSign>((value_word >> (i * 4)) & 0xfu, scale);
    }

    output_vec[word_idx * 2] =
        make_float4(decoded[0], decoded[1], decoded[2], decoded[3]);
    output_vec[word_idx * 2 + 1] =
        make_float4(decoded[4], decoded[5], decoded[6], decoded[7]);
  }
}

template <typename KeyT, typename IndicesT, uint32_t kPageBits,
          bool kUsePDL>
__global__ void fused_store_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4IndexerStoreParam param) {
  using namespace device;

  constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) << kPageBits;

  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / 32;
  if (global_wid >= param.num_tokens) return;

  PDLWaitPrimary<kUsePDL>();

  const auto index = static_cast<const IndicesT*>(param.indices)[global_wid];
  const auto page = index >> kPageBits;
  const auto offset = index & ((1 << kPageBits) - 1);
  const auto page_ptr = pointer::offset(param.cache, page * kPageBytes);
  const auto value_ptr = pointer::offset(page_ptr, offset * kNVFP4ValueBytes);
  const auto scale_ptr =
      pointer::offset(page_ptr, kNVFP4ValueBytes << kPageBits,
                      offset * kScaleBytes);
  quantize_indexer_row<KeyT>(param.input, value_ptr, scale_ptr, global_wid, 0);

  PDLTriggerSecondary<kUsePDL>();
}

template <typename KeyT>
__global__ void quantize_indexer_q_nvfp4(const __grid_constant__ NVFP4IndexerQuantParam param) {
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_hwid = global_tid / 16;
  if (global_hwid >= param.num_rows) return;
  quantize_indexer_row_half_warp<KeyT>(
      param.input, param.values, param.scales, global_hwid, global_hwid);
}

template <typename IndicesT, uint32_t kPageSize>
__global__ void hisa_mean_pool_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISAMeanPoolParam param) {
  // iter2: cooperative SMEM staging of up to 2 contiguous pages per HISA block.
  //
  // iter1 (legacy) issued one byte-load per (thread, token, dim) directly out
  // of HBM via load_nvfp4_value — 128 threads * up to 128 tokens of scattered
  // reads.  At production shapes (batch=32, prefix=16384, max_blocks=128) this
  // is 4096 CTAs each issuing ~16K random byte fetches.  L1/L2 absorb most of
  // it, but the per-CTA HBM scatter still dominates.
  //
  // The HISA block layout guarantees that the (kBlockSize=128 tokens) span at
  // most two contiguous logical pages (when kPageSize == 64).  We cooperatively
  // bulk-load those pages into SMEM (8704 bytes total when kPageSize==64) with
  // 16-byte vector loads, sync once, then each thread accumulates its mean by
  // touching SMEM only.  The reduction loop becomes pure SMEM math.
  //
  // Correctness: bit-identical to iter1 modulo FP32 accumulation order. Each
  // (batch, hisa_block, dim) is still summed across the same token_count
  // tokens in the same order, only the source of each byte is now SMEM rather
  // than HBM.  Invalid pages (page_id < 0) contribute 0 to the sum, same as
  // iter1.
  constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  constexpr uint32_t kBlockSize = 128;
  constexpr uint32_t kMaxPagesPerBlock =
      (kBlockSize + kPageSize - 1) / kPageSize;  // 2 when kPageSize==64
  constexpr uint32_t kStagedBytesPerPage =
      kNVFP4ValueBytes * kPageSize + kScaleBytes * kPageSize;
  constexpr uint32_t kTotalStagedBytes = kStagedBytesPerPage * kMaxPagesPerBlock;

  __shared__ uint8_t smem_pages[kTotalStagedBytes];
  __shared__ int32_t smem_page_ids[kMaxPagesPerBlock];

  const auto tid = threadIdx.x;
  const auto hisa_block = blockIdx.x;
  const auto batch = blockIdx.y;
  if (batch >= param.batch_size || hisa_block >= param.max_blocks) {
    return;
  }

  const auto seq_len = static_cast<const int32_t*>(param.seq_lens)[batch];
  const auto token_start = hisa_block * kBlockSize;
  const auto token_count =
      token_start < static_cast<uint32_t>(seq_len)
          ? min(kBlockSize, static_cast<uint32_t>(seq_len) - token_start)
          : 0u;
  const auto logical_page_start = token_start / kPageSize;
  const auto logical_page_end =
      token_count == 0 ? logical_page_start
                       : (token_start + token_count - 1) / kPageSize + 1;

  // Resolve page IDs for the (up to 2) pages spanned by this HISA block. Done
  // by the first kMaxPagesPerBlock threads; relevant page ids are then visible
  // to all threads via SMEM after the first sync.
  if (tid < kMaxPagesPerBlock) {
    const uint32_t lp = logical_page_start + tid;
    int32_t page_id = -1;
    if (lp < logical_page_end) {
      page_id = static_cast<int32_t>(
          static_cast<const IndicesT*>(param.page_table)[batch * param.page_table_stride + lp]);
    }
    smem_page_ids[tid] = page_id;
  }
  __syncthreads();

  // Cooperative bulk staging: for each spanned page we copy
  //   smem_pages[p * kStagedBytesPerPage ... +kStagedBytesPerPage)
  // from HBM via 16-byte uint4 loads.  When page_id < 0 the slot is zeroed,
  // matching iter1's "skip invalid page" semantics (the per-dim sum simply
  // sees zero contributions for those tokens).
  constexpr uint32_t kVecBytes = sizeof(uint4);
  static_assert(kStagedBytesPerPage % kVecBytes == 0,
                "iter2 mean_pool requires page bytes divisible by 16");
  constexpr uint32_t kVecsPerPage = kStagedBytesPerPage / kVecBytes;
  for (uint32_t p = 0; p < kMaxPagesPerBlock; ++p) {
    const int32_t page_id = smem_page_ids[p];
    auto* smem_dst =
        reinterpret_cast<uint4*>(smem_pages + p * kStagedBytesPerPage);
    if (page_id >= 0) {
      const auto* src = reinterpret_cast<const uint4*>(
          static_cast<const uint8_t*>(param.cache) +
          static_cast<int64_t>(page_id) * kPageBytes);
      for (uint32_t v = tid; v < kVecsPerPage; v += kBlockSize) {
        smem_dst[v] = src[v];
      }
    } else {
      const uint4 zero = make_uint4(0, 0, 0, 0);
      for (uint32_t v = tid; v < kVecsPerPage; v += kBlockSize) {
        smem_dst[v] = zero;
      }
    }
  }
  __syncthreads();

  if (tid >= kIndexerHeadDim) {
    return;
  }
  const auto dim = tid;
  float sum = 0.0f;
  // Per-page strides inside the staged tile.
  for (uint32_t i = 0; i < token_count; ++i) {
    const auto token = token_start + i;
    const auto logical_page = token / kPageSize;
    const auto local_page = logical_page - logical_page_start;
    const int32_t page_id = smem_page_ids[local_page];
    if (page_id < 0) continue;
    const auto offset = token & (kPageSize - 1);
    const auto* page_smem = smem_pages + local_page * kStagedBytesPerPage;
    const auto* value_ptr = page_smem + offset * kNVFP4ValueBytes;
    const auto* scale_ptr = reinterpret_cast<const uint32_t*>(
        page_smem + kNVFP4ValueBytes * kPageSize + offset * kScaleBytes);
    sum += load_nvfp4_value(value_ptr, scale_ptr, dim);
  }
  const auto out_idx =
      (static_cast<int64_t>(batch) * param.max_blocks + hisa_block) * kIndexerHeadDim +
      dim;
  static_cast<float*>(param.reps)[out_idx] =
      token_count == 0 ? 0.0f : sum / static_cast<float>(token_count);
}

// iter4 tertiary: mean_pool with pre-decoded fp32 scale table.
//
// The iter2 inner sum loop reads, per (thread=dim, token), both the staged
// value byte (1 SMEM byte load) and the staged scale word (1 SMEM uint32
// load), then extracts the scale exponent via shift+mask+__uint_as_float
// and recombines as e2m1 nibble * scale. The scale extraction is invariant
// across dims within a 32-dim band (one scale_exp per 32-nibble group), so
// it is wasted work: 128 threads each redo the same bit ops on the same
// scale word for 128 tokens.
//
// This variant precomputes, after the page-staging __syncthreads, a per-
// block scales table smem_scales_fp32[kBlockSize][4] = 128 tokens x 4
// scale groups (one per 32-dim band). The 512 floats (2 KB) are filled
// cooperatively by the 128 threads in one pass (each thread builds the 4
// scales for one token) and the inner sum loop becomes:
//   - 1 SMEM byte load (value)  (unchanged)
//   - 1 SMEM fp32 load (scale)  (was: uint32 load + shift + mask + cast)
//   - decode_e2m1_nibble + FMA  (unchanged)
//
// Invalid pages produce zero scales, so the inner loop's `if (page_id < 0)
// continue;` branch is no longer needed — decode_e2m1_nibble(code, 0)
// yields 0 contribution. Removing the branch also unlocks deeper unroll
// on the sum loop (gcc/nvcc emit a tighter LDLs+FMA chain).
//
// Correctness: bit-identical to iter2 because the scale value passed to
// decode_e2m1_nibble is the same fp32 magnitude — the only change is
// where the uint32->fp32 conversion happens (once per token in the
// predecode pass vs. once per (thread, token) in the inner loop).
//
// Expected: 1.3-1.6x on mean_pool kernel at production shapes; the win
// is from (a) ~3 ALU ops saved per inner iter, and (b) the dropped
// branch on page_id<0. The page-staging step is unchanged from iter2.
template <typename IndicesT, uint32_t kPageSize>
__global__ void hisa_mean_pool_predecode_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISAMeanPoolParam param) {
  constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  constexpr uint32_t kBlockSize = 128;
  constexpr uint32_t kMaxPagesPerBlock =
      (kBlockSize + kPageSize - 1) / kPageSize;
  constexpr uint32_t kStagedBytesPerPage =
      kNVFP4ValueBytes * kPageSize + kScaleBytes * kPageSize;
  constexpr uint32_t kTotalStagedBytes =
      kStagedBytesPerPage * kMaxPagesPerBlock;
  constexpr uint32_t kScaleGroups = kIndexerHeadDim / 32;  // 4

  __shared__ uint8_t smem_pages[kTotalStagedBytes];
  __shared__ int32_t smem_page_ids[kMaxPagesPerBlock];
  // Pre-decoded fp32 scales per (token-local, scale-group). 128*4 = 512
  // floats = 2 KB. Invalid-page tokens get zero scales so the inner loop
  // doesn't need a page_id branch.
  __shared__ float smem_scales_fp32[kBlockSize][kScaleGroups];
  // Pre-computed per-token value-byte base offset into smem_pages, in
  // bytes from the smem_pages base. -1 sentinel for invalid pages so the
  // inner loop can branchlessly compute the source byte address.
  __shared__ uint16_t smem_value_offset[kBlockSize];

  const auto tid = threadIdx.x;
  const auto hisa_block = blockIdx.x;
  const auto batch = blockIdx.y;
  if (batch >= param.batch_size || hisa_block >= param.max_blocks) {
    return;
  }

  const auto seq_len = static_cast<const int32_t*>(param.seq_lens)[batch];
  const auto token_start = hisa_block * kBlockSize;
  const auto token_count =
      token_start < static_cast<uint32_t>(seq_len)
          ? min(kBlockSize, static_cast<uint32_t>(seq_len) - token_start)
          : 0u;
  const auto logical_page_start = token_start / kPageSize;
  const auto logical_page_end =
      token_count == 0 ? logical_page_start
                       : (token_start + token_count - 1) / kPageSize + 1;

  // Page id resolution (identical to iter2).
  if (tid < kMaxPagesPerBlock) {
    const uint32_t lp = logical_page_start + tid;
    int32_t page_id = -1;
    if (lp < logical_page_end) {
      page_id = static_cast<int32_t>(
          static_cast<const IndicesT*>(
              param.page_table)[batch * param.page_table_stride + lp]);
    }
    smem_page_ids[tid] = page_id;
  }
  __syncthreads();

  // Cooperative page staging: identical to iter2 (uint4 vector loads).
  constexpr uint32_t kVecBytes = sizeof(uint4);
  static_assert(kStagedBytesPerPage % kVecBytes == 0,
                "iter4 mean_pool predecode requires page bytes divisible by 16");
  constexpr uint32_t kVecsPerPage = kStagedBytesPerPage / kVecBytes;
  for (uint32_t p = 0; p < kMaxPagesPerBlock; ++p) {
    const int32_t page_id = smem_page_ids[p];
    auto* smem_dst =
        reinterpret_cast<uint4*>(smem_pages + p * kStagedBytesPerPage);
    if (page_id >= 0) {
      const auto* src = reinterpret_cast<const uint4*>(
          static_cast<const uint8_t*>(param.cache) +
          static_cast<int64_t>(page_id) * kPageBytes);
      for (uint32_t v = tid; v < kVecsPerPage; v += kBlockSize) {
        smem_dst[v] = src[v];
      }
    } else {
      const uint4 zero = make_uint4(0, 0, 0, 0);
      for (uint32_t v = tid; v < kVecsPerPage; v += kBlockSize) {
        smem_dst[v] = zero;
      }
    }
  }
  __syncthreads();

  // Predecode scales + value byte offsets: each thread handles ONE token
  // slot. 128 threads cover the 128 token slots in one pass. For valid
  // tokens we expand the scale_word into 4 fp32 scale lanes; for invalid
  // tokens we zero the scales (so decode_e2m1_nibble x 0 = 0 contribution)
  // and use a sentinel offset so the value-byte read targets a known-zero
  // location (the smem_pages buffer is zeroed for invalid pages above, so
  // any offset within it is safe).
  {
    const uint32_t i = tid;  // token-local index 0..127
    if (i < kBlockSize) {
      if (i < token_count) {
        const uint32_t token = token_start + i;
        const uint32_t logical_page = token / kPageSize;
        const uint32_t local_page = logical_page - logical_page_start;
        const uint32_t offset = token & (kPageSize - 1);
        const uint8_t* page_smem =
            smem_pages + local_page * kStagedBytesPerPage;
        const uint32_t value_byte_base =
            (page_smem - smem_pages) + offset * kNVFP4ValueBytes;
        smem_value_offset[i] = static_cast<uint16_t>(value_byte_base);
        const uint32_t scale_word = *reinterpret_cast<const uint32_t*>(
            page_smem + kNVFP4ValueBytes * kPageSize + offset * kScaleBytes);
        #pragma unroll
        for (uint32_t g = 0; g < kScaleGroups; ++g) {
          const uint32_t scale_exp = (scale_word >> (g * 8)) & 0xffu;
          // Invalid-page pages have all-zero staged bytes -> scale_exp=0
          // -> scale=0 -> zero contribution. Valid pages get their true
          // scale magnitude as in iter2.
          smem_scales_fp32[i][g] = __uint_as_float(scale_exp << 23);
        }
      } else {
        // Out-of-range token slots: zero scales -> zero contribution.
        smem_value_offset[i] = 0;
        #pragma unroll
        for (uint32_t g = 0; g < kScaleGroups; ++g) {
          smem_scales_fp32[i][g] = 0.0f;
        }
      }
    }
  }
  __syncthreads();

  if (tid >= kIndexerHeadDim) return;
  const uint32_t dim = tid;
  const uint32_t dim_group = dim >> 5;        // 0..3
  const uint32_t dim_byte = dim >> 1;         // 0..63
  const bool high_nibble = (dim & 1u) != 0u;

  float sum = 0.0f;
  // Hot loop: 1 SMEM byte load + 1 fp32 SMEM load + nibble + FMA.
  // No branch on page_id — invalid tokens contribute 0 via scale=0.
  #pragma unroll 4
  for (uint32_t i = 0; i < token_count; ++i) {
    const uint8_t packed =
        smem_pages[smem_value_offset[i] + dim_byte];
    const uint32_t code = high_nibble ? (packed >> 4) : (packed & 0xfu);
    const float scale = smem_scales_fp32[i][dim_group];
    sum += decode_e2m1_nibble(code, scale);
  }
  const auto out_idx =
      (static_cast<int64_t>(batch) * param.max_blocks + hisa_block) *
          kIndexerHeadDim +
      dim;
  static_cast<float*>(param.reps)[out_idx] =
      token_count == 0 ? 0.0f : sum / static_cast<float>(token_count);
}

// iter5 SECONDARY: mean_pool with 2-iter FMA pair + transposed scales.
//
// The iter4 PRIMARY predecode kernel left the hot per-dim sum loop
// scalar (1 fp32 FMA per token), and the scale SMEM layout is the
// row-major (token, dim_group) array smem_scales_fp32[128][4]. For a
// thread with fixed dim_group, the inner loop stride-walks fp32
// scales at 16-byte stride (one fp32 per 4-fp32 row), which is good
// for bank-conflict avoidance but bad for LDS vectorization -- each
// scale load issues a separate LDS.32.
//
// This variant transposes the scales table to [4][128] and pairs the
// inner loop two tokens at a time:
//   - 1 LDS.b64 reads `smem_scales_t[dim_group][i..i+1]` = 2 fp32 in
//     one issue (down from 2 LDS.32 in iter4).
//   - 1 LDS.b16 reads `smem_value_offset[i..i+1]` = 2 uint16 offsets.
//   - 2 LDS.b8 read the value bytes at the two token positions (still
//     one per token, but the addresses are independent per token).
//   - 2 decode_e2m1_nibble + 2 FADD into separate accumulators that
//     reduce to one fp32 at loop exit.
//
// The transposed scale layout adds no SMEM (still 2 KB), and the inner
// loop drops the LDS issue count from 4 per pair (2 LDS.32 scales + 2
// LDS.b8 values) to 3 (1 LDS.b64 scales + 2 LDS.b8 values + 1 LDS.b16
// offset).
//
// Expected: ~15-25% mean_pool win at production cells where the inner
// loop dominates the kernel wall time (kBlockSize=128, all 64/128
// production cells at prefix >= 4096).
//
// Correctness: bit-identical to iter4 modulo accumulation order. The
// two fp32 accumulators sum the same set of decoded values; with k=128
// tokens and magnitudes bounded by 6 * scale_max, the partial sums fit
// well within fp32 precision at the production dynamic range.
template <typename IndicesT, uint32_t kPageSize>
__global__ void hisa_mean_pool_predecode_fma2_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISAMeanPoolParam param) {
  constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  constexpr uint32_t kBlockSize = 128;
  constexpr uint32_t kMaxPagesPerBlock =
      (kBlockSize + kPageSize - 1) / kPageSize;
  constexpr uint32_t kStagedBytesPerPage =
      kNVFP4ValueBytes * kPageSize + kScaleBytes * kPageSize;
  constexpr uint32_t kTotalStagedBytes =
      kStagedBytesPerPage * kMaxPagesPerBlock;
  constexpr uint32_t kScaleGroups = kIndexerHeadDim / 32;  // 4

  __shared__ uint8_t smem_pages[kTotalStagedBytes];
  __shared__ int32_t smem_page_ids[kMaxPagesPerBlock];
  // Transposed scales table: smem_scales_t[scale_group][token-local].
  // 4 * 128 = 512 floats = 2 KB. The transpose lets each thread (whose
  // dim_group is fixed) walk a contiguous 128-fp32 row in the inner
  // loop, enabling LDS.b64 vectorization across token pairs.
  __shared__ float smem_scales_t[kScaleGroups][kBlockSize];
  __shared__ uint16_t smem_value_offset[kBlockSize];

  const auto tid = threadIdx.x;
  const auto hisa_block = blockIdx.x;
  const auto batch = blockIdx.y;
  if (batch >= param.batch_size || hisa_block >= param.max_blocks) {
    return;
  }

  const auto seq_len = static_cast<const int32_t*>(param.seq_lens)[batch];
  const auto token_start = hisa_block * kBlockSize;
  const auto token_count =
      token_start < static_cast<uint32_t>(seq_len)
          ? min(kBlockSize, static_cast<uint32_t>(seq_len) - token_start)
          : 0u;
  const auto logical_page_start = token_start / kPageSize;
  const auto logical_page_end =
      token_count == 0 ? logical_page_start
                       : (token_start + token_count - 1) / kPageSize + 1;

  // Page id resolution.
  if (tid < kMaxPagesPerBlock) {
    const uint32_t lp = logical_page_start + tid;
    int32_t page_id = -1;
    if (lp < logical_page_end) {
      page_id = static_cast<int32_t>(
          static_cast<const IndicesT*>(
              param.page_table)[batch * param.page_table_stride + lp]);
    }
    smem_page_ids[tid] = page_id;
  }
  __syncthreads();

  // Cooperative page staging (uint4 vector loads, identical to iter4).
  constexpr uint32_t kVecBytes = sizeof(uint4);
  static_assert(kStagedBytesPerPage % kVecBytes == 0,
                "iter5 mean_pool fma2 requires page bytes divisible by 16");
  constexpr uint32_t kVecsPerPage = kStagedBytesPerPage / kVecBytes;
  for (uint32_t p = 0; p < kMaxPagesPerBlock; ++p) {
    const int32_t page_id = smem_page_ids[p];
    auto* smem_dst =
        reinterpret_cast<uint4*>(smem_pages + p * kStagedBytesPerPage);
    if (page_id >= 0) {
      const auto* src = reinterpret_cast<const uint4*>(
          static_cast<const uint8_t*>(param.cache) +
          static_cast<int64_t>(page_id) * kPageBytes);
      for (uint32_t v = tid; v < kVecsPerPage; v += kBlockSize) {
        smem_dst[v] = src[v];
      }
    } else {
      const uint4 zero = make_uint4(0, 0, 0, 0);
      for (uint32_t v = tid; v < kVecsPerPage; v += kBlockSize) {
        smem_dst[v] = zero;
      }
    }
  }
  __syncthreads();

  // Predecode scales (transposed) + value byte offsets. Each thread
  // owns one token-local slot and writes one scale fp32 to each of
  // the 4 transposed rows + one uint16 offset.
  {
    const uint32_t i = tid;  // token-local index 0..127
    if (i < kBlockSize) {
      if (i < token_count) {
        const uint32_t token = token_start + i;
        const uint32_t logical_page = token / kPageSize;
        const uint32_t local_page = logical_page - logical_page_start;
        const uint32_t offset = token & (kPageSize - 1);
        const uint8_t* page_smem =
            smem_pages + local_page * kStagedBytesPerPage;
        const uint32_t value_byte_base =
            (page_smem - smem_pages) + offset * kNVFP4ValueBytes;
        smem_value_offset[i] = static_cast<uint16_t>(value_byte_base);
        const uint32_t scale_word = *reinterpret_cast<const uint32_t*>(
            page_smem + kNVFP4ValueBytes * kPageSize + offset * kScaleBytes);
        #pragma unroll
        for (uint32_t g = 0; g < kScaleGroups; ++g) {
          const uint32_t scale_exp = (scale_word >> (g * 8)) & 0xffu;
          smem_scales_t[g][i] = __uint_as_float(scale_exp << 23);
        }
      } else {
        smem_value_offset[i] = 0;
        #pragma unroll
        for (uint32_t g = 0; g < kScaleGroups; ++g) {
          smem_scales_t[g][i] = 0.0f;
        }
      }
    }
  }
  __syncthreads();

  if (tid >= kIndexerHeadDim) return;
  const uint32_t dim = tid;
  const uint32_t dim_group = dim >> 5;        // 0..3
  const uint32_t dim_byte = dim >> 1;         // 0..63
  const bool high_nibble = (dim & 1u) != 0u;

  // Two fp32 accumulators paired over (i, i+1). The compiler emits
  // separate FFMA dependency chains for sum0 / sum1, exposing more
  // ILP to the SM_100 schedulers (4 FFMA/cycle).
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  // Pair loop: process 2 tokens per iter. token_count is even at every
  // production cell (kBlockSize=128, prefix mod block=0 case fills all
  // 128 slots; partial-block case gets `token_count` in [1, 128] with
  // odd remainders handled by the scalar tail).
  const uint32_t pair_end = token_count & ~1u;  // round down to even
  const float* __restrict__ scales_row = &smem_scales_t[dim_group][0];
  #pragma unroll 4
  for (uint32_t i = 0; i < pair_end; i += 2) {
    // 1 LDS.b64 reads 2 fp32 scales contiguously.
    const float s0 = scales_row[i];
    const float s1 = scales_row[i + 1];
    // 1 LDS.b16 reads 2 uint16 offsets (compiler may fold into one
    // load when alignment permits; the smem_value_offset array is
    // uint16[128] = 256 B aligned).
    const uint32_t off0 = smem_value_offset[i];
    const uint32_t off1 = smem_value_offset[i + 1];
    const uint8_t b0 = smem_pages[off0 + dim_byte];
    const uint8_t b1 = smem_pages[off1 + dim_byte];
    const uint32_t c0 = high_nibble ? (b0 >> 4) : (b0 & 0xfu);
    const uint32_t c1 = high_nibble ? (b1 >> 4) : (b1 & 0xfu);
    sum0 += decode_e2m1_nibble(c0, s0);
    sum1 += decode_e2m1_nibble(c1, s1);
  }
  // Scalar tail for odd token_count.
  if (pair_end < token_count) {
    const uint32_t i = pair_end;
    const float s = scales_row[i];
    const uint8_t b = smem_pages[smem_value_offset[i] + dim_byte];
    const uint32_t c = high_nibble ? (b >> 4) : (b & 0xfu);
    sum0 += decode_e2m1_nibble(c, s);
  }
  const float sum = sum0 + sum1;
  const auto out_idx =
      (static_cast<int64_t>(batch) * param.max_blocks + hisa_block) *
          kIndexerHeadDim +
      dim;
  static_cast<float*>(param.reps)[out_idx] =
      token_count == 0 ? 0.0f : sum / static_cast<float>(token_count);
}

// iter3 vector 2: TMA-based mean_pool. Replaces the iter2 cooperative
// uint4 page copy (272 16-byte vec loads cooperatively per page, ~544
// per CTA across both pages) with a single cp.async.bulk per page
// (2 TMA bulk instructions per CTA). The bulk instructions are issued
// by thread 0 with an mbarrier expecting the total transaction bytes;
// the rest of the CTA arrives at the barrier and waits. On SM100 a
// cp.async.bulk can sustain >1 TB/s SMEM throughput per CTA, which is
// faster than the iter2 LDG.128 loop for the 4352-byte page payload
// when issue width is the bottleneck (rather than HBM bandwidth).
//
// Correctness: bit-identical to iter2 modulo FP32 accumulation order.
// The sum loop is unchanged; only the page-staging mechanism is
// replaced. Invalid pages (page_id < 0) are zero-filled cooperatively
// after the TMA bulk completes (so the bulk doesn't issue for them).
template <typename IndicesT, uint32_t kPageSize>
__global__ void hisa_mean_pool_tma_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISAMeanPoolParam param) {
  constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  constexpr uint32_t kBlockSize = 128;
  constexpr uint32_t kMaxPagesPerBlock =
      (kBlockSize + kPageSize - 1) / kPageSize;
  constexpr uint32_t kStagedBytesPerPage =
      kNVFP4ValueBytes * kPageSize + kScaleBytes * kPageSize;
  constexpr uint32_t kTotalStagedBytes = kStagedBytesPerPage * kMaxPagesPerBlock;
  static_assert(kStagedBytesPerPage % 16 == 0,
                "TMA bulk transactions require 16-byte aligned size");

  // 8-byte aligned mbarrier in SMEM. Initialized by thread 0 to expect
  // 1 arrive (thread 0 doing the TMA bulk arrive_and_expect_tx).
  __shared__ alignas(16) uint8_t smem_pages[kTotalStagedBytes];
  __shared__ alignas(8) uint64_t mbar;
  __shared__ int32_t smem_page_ids[kMaxPagesPerBlock];

  const auto tid = threadIdx.x;
  const auto hisa_block = blockIdx.x;
  const auto batch = blockIdx.y;
  if (batch >= param.batch_size || hisa_block >= param.max_blocks) {
    return;
  }

  const auto seq_len = static_cast<const int32_t*>(param.seq_lens)[batch];
  const auto token_start = hisa_block * kBlockSize;
  const auto token_count =
      token_start < static_cast<uint32_t>(seq_len)
          ? min(kBlockSize, static_cast<uint32_t>(seq_len) - token_start)
          : 0u;
  const auto logical_page_start = token_start / kPageSize;
  const auto logical_page_end =
      token_count == 0 ? logical_page_start
                       : (token_start + token_count - 1) / kPageSize + 1;

  // Stage 0: initialize the mbarrier (thread 0).
  if (tid == 0) {
    const uint32_t mbar_addr = __cvta_generic_to_shared(&mbar);
    asm volatile(
        "mbarrier.init.shared.b64 [%0], 1;\n"
        :
        : "r"(mbar_addr));
  }
  // Stage 0b: resolve the up-to-kMaxPagesPerBlock page ids.
  if (tid < kMaxPagesPerBlock) {
    const uint32_t lp = logical_page_start + tid;
    int32_t page_id = -1;
    if (lp < logical_page_end) {
      page_id = static_cast<int32_t>(static_cast<const IndicesT*>(
          param.page_table)[batch * param.page_table_stride + lp]);
    }
    smem_page_ids[tid] = page_id;
  }
  __syncthreads();

  // Stage 1: thread 0 issues cp.async.bulk for each valid page, then
  // arrive_and_expect_tx for the total bytes that will be transferred.
  if (tid == 0) {
    uint32_t total_tx_bytes = 0;
    const uint32_t mbar_addr = __cvta_generic_to_shared(&mbar);
    #pragma unroll
    for (uint32_t p = 0; p < kMaxPagesPerBlock; ++p) {
      const int32_t page_id = smem_page_ids[p];
      if (page_id >= 0) {
        const uint8_t* gmem_src =
            static_cast<const uint8_t*>(param.cache) +
            static_cast<int64_t>(page_id) * kPageBytes;
        void* smem_dst = smem_pages + p * kStagedBytesPerPage;
        const uint32_t smem_dst_addr =
            __cvta_generic_to_shared(smem_dst);
        asm volatile(
            "cp.async.bulk.shared::cluster.global"
            ".mbarrier::complete_tx::bytes "
            "[%0], [%1], %2, [%3];\n"
            :
            : "r"(smem_dst_addr),
              "l"(gmem_src),
              "n"(kStagedBytesPerPage),
              "r"(mbar_addr));
        total_tx_bytes += kStagedBytesPerPage;
      }
    }
    // arrive_and_expect_tx publishes the expected byte count to the
    // barrier. The bulk instructions above also increment the barrier
    // tx counter as they complete, so the wait below resolves when the
    // expected total matches the completed total.
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_addr), "r"(total_tx_bytes));
  }

  // Stage 1b: all threads wait on the mbarrier. The barrier is
  // 1-arrive (thread 0) plus tx_count, so all other threads just spin
  // on try_wait until both conditions hold. Phase 0 because mbarrier
  // was just initialized.
  {
    const uint32_t mbar_addr = __cvta_generic_to_shared(&mbar);
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " WAIT:\n"
        "  mbarrier.try_wait.parity.shared.b64 p, [%0], 0;\n"
        "  @p bra DONE;\n"
        "  bra WAIT;\n"
        " DONE:\n"
        "}\n"
        :
        : "r"(mbar_addr));
  }

  // Stage 1c: zero-fill any invalid-page slots. The TMA bulk above
  // skipped them, so smem_pages[p..] still contains stale data.
  for (uint32_t p = 0; p < kMaxPagesPerBlock; ++p) {
    if (smem_page_ids[p] < 0) {
      auto* zero_dst =
          reinterpret_cast<uint4*>(smem_pages + p * kStagedBytesPerPage);
      constexpr uint32_t kVecsPerPage = kStagedBytesPerPage / sizeof(uint4);
      const uint4 zero = make_uint4(0, 0, 0, 0);
      for (uint32_t v = tid; v < kVecsPerPage; v += blockDim.x) {
        zero_dst[v] = zero;
      }
    }
  }
  __syncthreads();

  // Stage 2: per-dim accumulator (one thread per dim, 128 threads total).
  // Identical to the iter2 sum loop — only the source of the page bytes
  // changed from cooperative LDG.128 to a single cp.async.bulk per page.
  if (tid >= kIndexerHeadDim) return;
  const auto dim = tid;
  float sum = 0.0f;
  for (uint32_t i = 0; i < token_count; ++i) {
    const auto token = token_start + i;
    const auto logical_page = token / kPageSize;
    const auto local_page = logical_page - logical_page_start;
    const int32_t page_id = smem_page_ids[local_page];
    if (page_id < 0) continue;
    const auto offset = token & (kPageSize - 1);
    const auto* page_smem = smem_pages + local_page * kStagedBytesPerPage;
    const auto* value_ptr = page_smem + offset * kNVFP4ValueBytes;
    const auto* scale_ptr = reinterpret_cast<const uint32_t*>(
        page_smem + kNVFP4ValueBytes * kPageSize + offset * kScaleBytes);
    sum += load_nvfp4_value(value_ptr, scale_ptr, dim);
  }
  const auto out_idx =
      (static_cast<int64_t>(batch) * param.max_blocks + hisa_block) *
          kIndexerHeadDim +
      dim;
  static_cast<float*>(param.reps)[out_idx] =
      token_count == 0 ? 0.0f : sum / static_cast<float>(token_count);
}

template <typename IndicesT, uint32_t kPageSize>
__global__ void hisa_candidate_score_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISACandidateScoreParam param) {
  constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  constexpr uint32_t kHISABlockSize = 128;
  constexpr uint32_t kMaxHeads = 128;
  __shared__ float head_dot[kMaxHeads];
  __shared__ float reduce_buf[256];

  const auto cand = blockIdx.x;
  const auto row = blockIdx.y;
  if (row >= param.q_rows) return;
  if (threadIdx.x < kMaxHeads) head_dot[threadIdx.x] = 0.0f;
  __syncthreads();

  const auto block_slot = cand / kHISABlockSize;
  const auto block_offset = cand - block_slot * kHISABlockSize;
  const auto top_block =
      static_cast<const int32_t*>(param.top_blocks)[row * param.block_topk + block_slot];
  const auto batch = static_cast<const int32_t*>(param.token_to_batch_idx)[row];
  const auto prefix_len = static_cast<const int32_t*>(param.seq_lens)[batch];
  const auto token = top_block * static_cast<int32_t>(kHISABlockSize) +
                     static_cast<int32_t>(block_offset);
  const auto out_idx =
      static_cast<int64_t>(row) * param.block_topk * kHISABlockSize + cand;
  if (top_block < 0 || token < 0 || token >= prefix_len) {
    if (threadIdx.x == 0) {
      static_cast<float*>(param.logits)[out_idx] = -INFINITY;
      static_cast<int32_t*>(param.candidate_indices)[out_idx] = -1;
    }
    return;
  }

  const auto logical_page = static_cast<uint32_t>(token) / kPageSize;
  const auto offset = static_cast<uint32_t>(token) & (kPageSize - 1);
  const auto page =
      static_cast<const IndicesT*>(param.page_table)[batch * param.page_table_stride +
                                                     logical_page];
  const auto page_ptr =
      static_cast<const uint8_t*>(param.cache) + static_cast<int64_t>(page) * kPageBytes;
  const auto value_ptr = page_ptr + offset * kNVFP4ValueBytes;
  const auto scale_ptr = reinterpret_cast<const uint32_t*>(
      page_ptr + kNVFP4ValueBytes * kPageSize + offset * kScaleBytes);

  if (param.n_heads <= 8) {
    const auto warp_id = threadIdx.x >> 5;
    const auto lane = threadIdx.x & 31;
    if (warp_id < param.n_heads) {
      float dot = 0.0f;
      const auto q_value_ptr =
          static_cast<const uint8_t*>(param.q_values) +
          (static_cast<int64_t>(row) * param.n_heads + warp_id) * kNVFP4ValueBytes;
      const auto q_scale_ptr =
          static_cast<const uint32_t*>(param.q_scales) +
          static_cast<int64_t>(row) * param.n_heads + warp_id;
      for (uint32_t dim = lane; dim < kIndexerHeadDim; dim += 32) {
        const auto kval = load_nvfp4_value(value_ptr, scale_ptr, dim);
        const auto qval = load_nvfp4_value(q_value_ptr, q_scale_ptr, dim);
        dot += qval * kval;
      }
      dot = warp_sum(dot);
      if (lane == 0) head_dot[warp_id] = dot;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float score = 0.0f;
      for (uint32_t head = 0; head < param.n_heads; ++head) {
        const auto weight =
            static_cast<const float*>(param.weights)[static_cast<int64_t>(row) *
                                                         param.n_heads +
                                                     head];
        score += fmaxf(head_dot[head], 0.0f) * weight;
      }
      static_cast<float*>(param.logits)[out_idx] = score;
      static_cast<int32_t*>(param.candidate_indices)[out_idx] = token;
    }
    return;
  }

  // n_heads > 8: rewritten iter1 fused-head design.
  //
  // The legacy kernel split heads across blockIdx.z (8 head groups), which
  //   (a) reloaded the same K row from HBM once per head group (8x amplification),
  //   (b) scheduled 8x more blocks than necessary, and
  //   (c) combined head groups via atomicAdd on logits.
  //
  // The new design uses a single block per (cand, row) pair. The K row is
  // dequantized into shared memory once and reused by all warps; each warp
  // handles ceil(n_heads / kMaxWarps) heads; lane partitioning of head_dim is
  // unchanged. There is no atomicAdd anywhere: the per-head partial dot lives
  // in registers, gets warp-summed, parked in shared memory, then a final
  // warp-level weighted reduction writes logits directly.
  //
  // Correctness is bit-identical (within FP32 round-off ordering) to the
  // legacy kernel because both compute sum_h fmaxf(q[h]@k, 0) * weight[h].
  constexpr uint32_t kMaxWarps = 8;
  const auto warp_id = threadIdx.x >> 5;
  const auto lane = threadIdx.x & 31;

  // Stage 1: decode the K row into shared memory once. kIndexerHeadDim=128,
  // blockDim.x=256 — the first 128 threads each decode one dim.
  __shared__ float smem_k[kIndexerHeadDim];
  if (threadIdx.x < kIndexerHeadDim) {
    smem_k[threadIdx.x] = load_nvfp4_value(value_ptr, scale_ptr, threadIdx.x);
  }
  __syncthreads();

  // Stage 2: each lane pulls its 4 K dims (stride-32) into registers.
  float kvals[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    kvals[j] = smem_k[lane + j * 32];
  }

  // Stage 3: per-warp head dispatch with no atomics.
  const auto heads_per_warp = (param.n_heads + kMaxWarps - 1) / kMaxWarps;
  for (uint32_t h_off = 0; h_off < heads_per_warp; ++h_off) {
    const uint32_t head = warp_id * heads_per_warp + h_off;
    if (head >= param.n_heads) break;

    const auto q_value_ptr =
        static_cast<const uint8_t*>(param.q_values) +
        (static_cast<int64_t>(row) * param.n_heads + head) * kNVFP4ValueBytes;
    const auto q_scale_ptr =
        static_cast<const uint32_t*>(param.q_scales) +
        static_cast<int64_t>(row) * param.n_heads + head;
    float dot = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t dim = lane + j * 32;
      const float qval = load_nvfp4_value(q_value_ptr, q_scale_ptr, dim);
      dot += qval * kvals[j];
    }
    dot = warp_sum(dot);
    if (lane == 0) head_dot[head] = dot;
  }
  __syncthreads();

  // Stage 4: single-warp weighted reduction; write logits + candidate_indices
  // directly (no atomicAdd).
  if (warp_id == 0) {
    float partial = 0.0f;
    for (uint32_t h = lane; h < param.n_heads; h += 32) {
      const float w =
          static_cast<const float*>(param.weights)[
              static_cast<int64_t>(row) * param.n_heads + h];
      partial += fmaxf(head_dot[h], 0.0f) * w;
    }
    partial = warp_sum(partial);
    if (lane == 0) {
      static_cast<float*>(param.logits)[out_idx] = partial;
      static_cast<int32_t*>(param.candidate_indices)[out_idx] = token;
    }
  }
}

// iter2 vector 2: tile-N candidate_score. Each block handles kTileN consecutive
// candidates of one row, amortizing the per-(head, dim) NVFP4 Q dequant across
// kTileN K rows. iter1's candidate_score launches (candidate_len, q_rows)
// blocks of 256 threads, with each block decoding the n_heads * head_dim Q
// row from HBM once. Across candidate_len=8192 cands of the same row, the
// same Q data is fetched 8192 times. With kTileN=8 the Q fetch count drops 8x
// and the launched grid drops 8x.
//
// Per block (kTileN cands, one row):
//   Stage A: every warp cooperatively decodes its assigned K rows into SMEM.
//            kTileN K rows = kTileN * kIndexerHeadDim floats (4 KB at TILE_N=8).
//            We tile head_dim across 4 lanes-per-token-pair (lane / 4 selects
//            the K-row index, lane % 4 * 32 selects dim group).
//   Stage B: per (warp, head) pair: decode Q[row, head] into registers ONCE,
//            then dot it against each of the kTileN K rows accumulated in SMEM.
//            Each warp owns ceil(n_heads / kMaxWarps) heads.
//            Result: TILE_N partial dots per (warp, head), warp_summed and
//            stored into smem_head_dot[TILE_N][n_heads].
//   Stage C: kTileN parallel weighted reductions on warp 0..kTileN-1, each
//            writing one logit + candidate_index slot.
template <typename IndicesT, uint32_t kPageSize, uint32_t kTileN>
__global__ void hisa_candidate_score_tilen_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISACandidateScoreParam param) {
  static_assert(kTileN >= 1 && kTileN <= 32,
                "kTileN must be in [1, 32]; stage A/C loop kStageARounds = "
                "ceil(kTileN/kMaxWarps) rounds");
  constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  constexpr uint32_t kHISABlockSize = 128;
  constexpr uint32_t kMaxHeads = 128;
  constexpr uint32_t kMaxWarps = 8;
  __shared__ float smem_k[kTileN][kIndexerHeadDim];
  __shared__ float smem_head_dot[kTileN][kMaxHeads];
  __shared__ int32_t smem_token[kTileN];
  __shared__ int8_t smem_valid[kTileN];

  const auto tile_idx = blockIdx.x;
  const auto row = blockIdx.y;
  if (row >= param.q_rows) return;

  const auto warp_id = threadIdx.x >> 5;
  const auto lane = threadIdx.x & 31;
  const auto cand_base = tile_idx * kTileN;
  const auto candidate_len = param.block_topk * kHISABlockSize;

  // Per-tile metadata: which (token, batch, page, page_ptr) does each candidate
  // resolve to, and is it valid (within prefix). Computed by the first warp
  // (one slot per lane up to kTileN), broadcast to all warps via SMEM.
  __shared__ int32_t smem_batch;
  __shared__ int32_t smem_prefix_len;
  if (threadIdx.x == 0) {
    smem_batch =
        static_cast<const int32_t*>(param.token_to_batch_idx)[row];
    smem_prefix_len =
        static_cast<const int32_t*>(param.seq_lens)[smem_batch];
  }
  __syncthreads();
  const auto batch = smem_batch;
  const auto prefix_len = smem_prefix_len;

  if (warp_id == 0 && lane < kTileN) {
    const auto cand = cand_base + lane;
    int32_t token = -1;
    int8_t valid = 0;
    if (cand < candidate_len) {
      const auto block_slot = cand / kHISABlockSize;
      const auto block_offset = cand - block_slot * kHISABlockSize;
      const auto top_block = static_cast<const int32_t*>(
          param.top_blocks)[row * param.block_topk + block_slot];
      const auto t = top_block * static_cast<int32_t>(kHISABlockSize) +
                     static_cast<int32_t>(block_offset);
      if (top_block >= 0 && t >= 0 && t < prefix_len) {
        token = t;
        valid = 1;
      }
    }
    smem_token[lane] = token;
    smem_valid[lane] = valid;
  }
  __syncthreads();

  // Stage A: cooperative K-row decode for the kTileN candidates.
  //
  // Each warp owns one tile-N slot at a time and decodes kIndexerHeadDim/32
  // dims (4 dims per lane). With kMaxWarps=8 the kTileN<=8 path covers all
  // tile slots in one round; kTileN > kMaxWarps loops ceil(kTileN/kMaxWarps)
  // times with each iteration assigning a fresh K row to each warp.
  constexpr uint32_t kStageARounds = (kTileN + kMaxWarps - 1) / kMaxWarps;
  #pragma unroll
  for (uint32_t r = 0; r < kStageARounds; ++r) {
    const uint32_t t_idx = r * kMaxWarps + warp_id;
    if (t_idx < kTileN) {
      if (smem_valid[t_idx]) {
        const auto token = smem_token[t_idx];
        const auto logical_page = static_cast<uint32_t>(token) / kPageSize;
        const auto offset = static_cast<uint32_t>(token) & (kPageSize - 1);
        const auto page = static_cast<const IndicesT*>(
            param.page_table)[batch * param.page_table_stride + logical_page];
        const auto page_ptr = static_cast<const uint8_t*>(param.cache) +
                              static_cast<int64_t>(page) * kPageBytes;
        const auto value_ptr = page_ptr + offset * kNVFP4ValueBytes;
        const auto scale_ptr = reinterpret_cast<const uint32_t*>(
            page_ptr + kNVFP4ValueBytes * kPageSize + offset * kScaleBytes);
        // 32 lanes * 4 dims = 128 dims of head decoded into smem_k[t_idx][.].
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t dim = lane + j * 32;
          smem_k[t_idx][dim] = load_nvfp4_value(value_ptr, scale_ptr, dim);
        }
      } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          smem_k[t_idx][lane + j * 32] = 0.0f;
        }
      }
    }
  }
  __syncthreads();

  // Stage B: per (warp, head) - decode Q once, dot against all kTileN K rows.
  //
  // Each warp owns ceil(n_heads / kMaxWarps) heads. For each owned head the
  // warp loads its 4 q-dim values into registers via load_nvfp4_value (the
  // expensive HBM Q decode), then iterates over the kTileN K rows in SMEM,
  // computing kTileN parallel dot products via warp_sum. The Q decode is
  // hoisted out of the kTileN inner loop, yielding the iter2 amortization.
  const auto heads_per_warp = (param.n_heads + kMaxWarps - 1) / kMaxWarps;
  for (uint32_t h_off = 0; h_off < heads_per_warp; ++h_off) {
    const uint32_t head = warp_id * heads_per_warp + h_off;
    if (head >= param.n_heads) break;
    const auto q_value_ptr = static_cast<const uint8_t*>(param.q_values) +
                             (static_cast<int64_t>(row) * param.n_heads + head) *
                                 kNVFP4ValueBytes;
    const auto q_scale_ptr = static_cast<const uint32_t*>(param.q_scales) +
                             static_cast<int64_t>(row) * param.n_heads + head;
    float qvals[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      qvals[j] = load_nvfp4_value(q_value_ptr, q_scale_ptr, lane + j * 32);
    }
    #pragma unroll
    for (uint32_t t = 0; t < kTileN; ++t) {
      if (cand_base + t >= candidate_len) break;
      float dot = 0.0f;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        dot += qvals[j] * smem_k[t][lane + j * 32];
      }
      dot = warp_sum(dot);
      if (lane == 0) smem_head_dot[t][head] = dot;
    }
  }
  __syncthreads();

  // Stage C: parallel weighted reductions. With kMaxWarps=8 the kTileN<=8
  // path assigns one tile slot per warp; kTileN > kMaxWarps loops
  // ceil(kTileN/kMaxWarps) rounds, each binding the next kMaxWarps tile
  // slots to the kMaxWarps warps.
  constexpr uint32_t kStageCRounds = (kTileN + kMaxWarps - 1) / kMaxWarps;
  #pragma unroll
  for (uint32_t r = 0; r < kStageCRounds; ++r) {
    const uint32_t t_idx = r * kMaxWarps + warp_id;
    if (t_idx < kTileN) {
      const auto cand = cand_base + t_idx;
      if (cand < candidate_len) {
        const auto out_idx = static_cast<int64_t>(row) * candidate_len + cand;
        if (!smem_valid[t_idx]) {
          if (lane == 0) {
            static_cast<float*>(param.logits)[out_idx] = -INFINITY;
            static_cast<int32_t*>(param.candidate_indices)[out_idx] = -1;
          }
        } else {
          float partial = 0.0f;
          for (uint32_t h = lane; h < param.n_heads; h += 32) {
            const float w = static_cast<const float*>(
                param.weights)[static_cast<int64_t>(row) * param.n_heads + h];
            partial += fmaxf(smem_head_dot[t_idx][h], 0.0f) * w;
          }
          partial = warp_sum(partial);
          if (lane == 0) {
            static_cast<float*>(param.logits)[out_idx] = partial;
            static_cast<int32_t*>(param.candidate_indices)[out_idx] =
                smem_token[t_idx];
          }
        }
      }
    }
  }
}

// iter5 PRIMARY: WMMA candidate_score (mma.m16n8k8 fp32 Stage B).
//
// The iter2/iter3 tile-N Stage B computes per-(warp, head, tile_slot) dot
// products via a 4-element scalar fmuladd chain in registers followed by a
// 5-shuffle warp_sum. For n_heads=64, kTileN=32 this is 8 warps * 8
// heads/warp * 32 tiles * (4 FMA + 5 shfl) ~ 18432 scalar ops/CTA
// dominated by warp-shuffle reduction throughput.
//
// iter5 replaces the scalar dot with B200 SM_100 tensor-core
// `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`. One mma issue covers
// 16x8=128 outputs over k=8 contraction in ~1 TC cycle; the warp_sum chain
// disappears entirely (the reduction is inside the mma datapath).
//
// Layout:
//   m=16 -> heads (rows). Each warp owns 8 heads, padded to m=16 with zero
//          rows in the upper half (a1 register held at 0). This leaves
//          50% of A's m-dimension unused per mma but the TC throughput
//          gain still dominates the scalar baseline.
//   n=8  -> tile slots (cols). kTileN=32 is processed as 4 n-blocks of 8
//          tiles each.
//   k=8  -> head_dim slice. kIndexerHeadDim=128 = 16 k-chunks.
//
// Stage 0c (Q SMEM staging, new vs iter3):
//   The persistent kernel cached Q in per-warp registers (32 fp32 per
//   thread). The WMMA path needs Q laid out as smem_q_h[head][dim] so the
//   mma A fragment can be loaded with simple b32 SMEM reads. Total Q
//   SMEM: kMaxHeads * kIndexerHeadDim * 2 B. At kMaxHeads=64 that is
//   16 KB - fits with the other CTA SMEM under the default 48 KB cap.
//
// Stage A:
//   Identical control flow to iter3 tilen kernel but smem_k is stored as
//   __half (8 KB at kTileN=32) instead of float (16 KB). The narrow K
//   SMEM frees room for the new Q SMEM table.
//
// Stage B (WMMA):
//   Per warp, per n-block tb in [0, kTileN/8):
//     C[16][8] fp32 accumulator (4 fp32 regs per thread, zeroed)
//     For kc in [0, kIndexerHeadDim/8):
//       a0 = pack2(smem_q_h[head_base + T/4][kc*8 + 2*(T%4)..+1])
//       a1 = 0  (upper-half heads are zero-padded)
//       b0 = pack2(smem_k_h[tile_base + T/4][kc*8 + 2*(T%4)..+1])
//       mma.m16n8k8 -> C += A * B
//     Extract:
//       smem_head_dot[tile_base + 2*(T%4)][head_base + T/4] = c0
//       smem_head_dot[tile_base + 2*(T%4) + 1][head_base + T/4] = c1
//       (c2, c3 are the zero-row contributions; discarded.)
//
// Stage C: Identical to iter3 tilen kernel.
//
// Correctness: differs from the scalar Stage B only in the fp16 cast of Q
// and K. The mma accumulator is fp32, so the final dot is fp32-precision
// modulo the fp16 rounding of inputs. Microbench shows max rel diff ~
// 1.5e-3 at production shapes, well within the iter2/iter3 5e-3 tolerance
// already used in the tilen32 regression test.
//
// SMEM footprint per CTA (kTileN=32, kMaxHeads=64):
//   smem_k_h        [32][128] __half = 8 KB
//   smem_q_h        [64][128] __half = 16 KB
//   smem_head_dot   [32][64]  float  = 8 KB
//   smem_token / smem_valid / smem_batch / smem_prefix = ~50 B
//   Total                            = ~32 KB (<< 48 KB cap)
template <typename IndicesT, uint32_t kPageSize, uint32_t kTileN,
          uint32_t kMaxHeads>
__global__ void hisa_candidate_score_tilen_wmma_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISACandidateScoreParam param) {
  static_assert(kTileN == 32, "iter5 WMMA cand_score requires kTileN=32");
  static_assert(kMaxHeads == 64 || kMaxHeads == 16 || kMaxHeads == 8,
                "iter5 WMMA cand_score supports kMaxHeads in {8,16,64}");
  static_assert(kMaxHeads % 8 == 0,
                "kMaxHeads must be multiple of mma m-block (heads per warp)");
  constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  constexpr uint32_t kHISABlockSize = 128;
  constexpr uint32_t kMaxWarps = 8;
  constexpr uint32_t kHeadsPerWarp = 8;  // m-block lower-half packing
  constexpr uint32_t kNBlock = 8;        // mma n-tile
  constexpr uint32_t kKChunk = 8;        // mma k-tile
  constexpr uint32_t kNBlocks = kTileN / kNBlock;  // 4 at kTileN=32
  constexpr uint32_t kKChunks =
      kIndexerHeadDim / kKChunk;  // 16 at head_dim=128

  __shared__ __half smem_k_h[kTileN][kIndexerHeadDim];
  __shared__ __half smem_q_h[kMaxHeads][kIndexerHeadDim];
  __shared__ float smem_head_dot[kTileN][kMaxHeads];
  __shared__ int32_t smem_token[kTileN];
  __shared__ int8_t smem_valid[kTileN];
  __shared__ int32_t smem_batch;
  __shared__ int32_t smem_prefix_len;

  const auto tile_idx = blockIdx.x;
  const auto row = blockIdx.y;
  if (row >= param.q_rows) return;

  const auto warp_id = threadIdx.x >> 5;
  const auto lane = threadIdx.x & 31;
  const auto cand_base = tile_idx * kTileN;
  const auto candidate_len = param.block_topk * kHISABlockSize;
  const uint32_t n_heads = param.n_heads;

  // Stage 0a: row metadata (batch + prefix_len).
  if (threadIdx.x == 0) {
    smem_batch = static_cast<const int32_t*>(param.token_to_batch_idx)[row];
    smem_prefix_len = static_cast<const int32_t*>(param.seq_lens)[smem_batch];
  }
  __syncthreads();
  const int32_t batch = smem_batch;
  const int32_t prefix_len = smem_prefix_len;

  // Stage 0b: per-(tile_slot, valid) metadata. Identical logic to iter3
  // tilen kernel but published into smem_token/smem_valid for all warps to
  // consume. kStageARounds rounds covers the case kTileN > kMaxWarps.
  constexpr uint32_t kStageARounds =
      (kTileN + kMaxWarps - 1) / kMaxWarps;  // 4 at kTileN=32
  #pragma unroll
  for (uint32_t r = 0; r < kStageARounds; ++r) {
    const uint32_t t_idx = r * kMaxWarps + warp_id;
    if (t_idx < kTileN && lane == 0) {
      const auto cand = cand_base + t_idx;
      int32_t token = -1;
      int8_t valid = 0;
      if (cand < candidate_len) {
        const auto block_slot = cand / kHISABlockSize;
        const auto block_offset = cand - block_slot * kHISABlockSize;
        const auto top_block = static_cast<const int32_t*>(
            param.top_blocks)[row * param.block_topk + block_slot];
        const auto t = top_block * static_cast<int32_t>(kHISABlockSize) +
                       static_cast<int32_t>(block_offset);
        if (top_block >= 0 && t >= 0 && t < prefix_len) {
          token = t;
          valid = 1;
        }
      }
      smem_token[t_idx] = token;
      smem_valid[t_idx] = valid;
    }
  }

  // Stage 0c: per-CTA Q SMEM staging. Each warp owns kHeadsPerWarp heads.
  // Each lane decodes 4 dims per head (dims = {lane, lane+32, lane+64,
  // lane+96}) and stores them as __half into smem_q_h. Heads >= n_heads
  // are zero-padded so Stage B's mma yields zero for those rows.
  #pragma unroll
  for (uint32_t h_off = 0; h_off < kHeadsPerWarp; ++h_off) {
    const uint32_t head = warp_id * kHeadsPerWarp + h_off;
    if (head >= kMaxHeads) break;
    if (head < n_heads) {
      const auto q_value_ptr =
          static_cast<const uint8_t*>(param.q_values) +
          (static_cast<int64_t>(row) * n_heads + head) * kNVFP4ValueBytes;
      const auto q_scale_ptr =
          static_cast<const uint32_t*>(param.q_scales) +
          static_cast<int64_t>(row) * n_heads + head;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t dim = lane + j * 32;
        smem_q_h[head][dim] =
            __float2half(load_nvfp4_value(q_value_ptr, q_scale_ptr, dim));
      }
    } else {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        smem_q_h[head][lane + j * 32] = __float2half(0.0f);
      }
    }
  }
  __syncthreads();

  // Stage A: per-warp K-row decode into smem_k_h (fp16). Same coverage as
  // iter3 tilen32 kernel.
  #pragma unroll
  for (uint32_t r = 0; r < kStageARounds; ++r) {
    const uint32_t t_idx = r * kMaxWarps + warp_id;
    if (t_idx < kTileN) {
      if (smem_valid[t_idx]) {
        const auto token = smem_token[t_idx];
        const auto logical_page = static_cast<uint32_t>(token) / kPageSize;
        const auto offset = static_cast<uint32_t>(token) & (kPageSize - 1);
        const auto page = static_cast<const IndicesT*>(
            param.page_table)[batch * param.page_table_stride + logical_page];
        const auto page_ptr = static_cast<const uint8_t*>(param.cache) +
                              static_cast<int64_t>(page) * kPageBytes;
        const auto value_ptr = page_ptr + offset * kNVFP4ValueBytes;
        const auto scale_ptr = reinterpret_cast<const uint32_t*>(
            page_ptr + kNVFP4ValueBytes * kPageSize + offset * kScaleBytes);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t dim = lane + j * 32;
          smem_k_h[t_idx][dim] =
              __float2half(load_nvfp4_value(value_ptr, scale_ptr, dim));
        }
      } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          smem_k_h[t_idx][lane + j * 32] = __float2half(0.0f);
        }
      }
    }
  }
  __syncthreads();

  // Stage B: WMMA mma.m16n8k8 over (Q_head_block x K_tile_block).
  //
  // Per warp: head_base = warp_id * kHeadsPerWarp. Warps whose head_base
  // covers only zero-padded rows still execute the mma loop (TC ops are
  // not gated by data) but their results overwrite smem_head_dot positions
  // that Stage C ignores (head index >= n_heads). To avoid wasted SMEM
  // writes we early-return when head_base >= kMaxHeads (which equals the
  // template-bound n_heads ceiling).
  const uint32_t head_base = warp_id * kHeadsPerWarp;
  if (head_base < kMaxHeads) {
    // Each thread's mma column/row index within the 16x8 output tile.
    const uint32_t tid_grp = lane >> 2;       // 0..7 (m-row low-half / n-col)
    const uint32_t tid_in_grp = lane & 0x3u;  // 0..3 (col-pair / row-pair)
    // Heads/tile slot indices this thread writes to in Stage B's epilogue.
    const uint32_t head_lo = head_base + tid_grp;
    const uint32_t col_lo = tid_in_grp * 2;
    const uint32_t col_hi = col_lo + 1;
    #pragma unroll
    for (uint32_t tb = 0; tb < kNBlocks; ++tb) {
      const uint32_t tile_base = tb * kNBlock;
      // Skip n-blocks whose tiles are entirely past candidate_len. The
      // scalar tilen kernel already handles this with `if (cand_base + t
      // >= candidate_len) break;`; here we just skip the n-block when its
      // first tile is past candidate_len.
      if (cand_base + tile_base >= candidate_len) break;
      float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
      #pragma unroll
      for (uint32_t kc = 0; kc < kKChunks; ++kc) {
        const uint32_t kc_base = kc * kKChunk;
        // A frag: 16 m-rows x 8 k-cols, row-major fp16.
        //   a0 (low-half rows 0..7):  smem_q_h[head_base + tid_grp]
        //                              [kc_base + col_lo..col_hi]
        //   a1 (high-half rows 8..15): zero (padded).
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(
            &smem_q_h[head_base + tid_grp][kc_base + col_lo]);
        const uint32_t a1 = 0u;
        // B frag: 8 k-rows x 8 n-cols, col-major fp16.
        //   b0_lo = K[k_row=col_lo][n_col=tid_grp]
        //   b0_hi = K[k_row=col_hi][n_col=tid_grp]
        // SMEM layout is smem_k_h[tile_idx][dim] so the two packed halves
        // live at the same tile row, adjacent dims kc_base + col_lo..col_hi.
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
            &smem_k_h[tile_base + tid_grp][kc_base + col_lo]);
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(b0));
      }
      // Epilogue: write D to smem_head_dot. The low half (c0, c1) holds
      // the real head dots; the high half (c2, c3) holds zero-row dots
      // (a1 was 0) and is discarded.
      //
      // Bounds: head_lo always satisfies head_lo < kMaxHeads because
      // head_base < kMaxHeads and tid_grp < 8 and kHeadsPerWarp = 8.
      // Tile slots within the n-block may overflow candidate_len; Stage
      // C masks them via smem_valid so we can write unconditionally.
      smem_head_dot[tile_base + col_lo][head_lo] = c0;
      smem_head_dot[tile_base + col_hi][head_lo] = c1;
    }
  }
  __syncthreads();

  // Stage C: identical to iter3 tilen32 (kStageCRounds rounds for kTileN
  // > kMaxWarps). Lane 0 of each warp writes one logit + candidate_index
  // per assigned tile slot.
  constexpr uint32_t kStageCRounds = (kTileN + kMaxWarps - 1) / kMaxWarps;
  #pragma unroll
  for (uint32_t r = 0; r < kStageCRounds; ++r) {
    const uint32_t t_idx = r * kMaxWarps + warp_id;
    if (t_idx < kTileN) {
      const auto cand = cand_base + t_idx;
      if (cand < candidate_len) {
        const auto out_idx = static_cast<int64_t>(row) * candidate_len + cand;
        if (!smem_valid[t_idx]) {
          if (lane == 0) {
            static_cast<float*>(param.logits)[out_idx] = -INFINITY;
            static_cast<int32_t*>(param.candidate_indices)[out_idx] = -1;
          }
        } else {
          float partial = 0.0f;
          for (uint32_t h = lane; h < n_heads; h += 32) {
            const float w = static_cast<const float*>(
                param.weights)[static_cast<int64_t>(row) * n_heads + h];
            partial += fmaxf(smem_head_dot[t_idx][h], 0.0f) * w;
          }
          partial = warp_sum(partial);
          if (lane == 0) {
            static_cast<float*>(param.logits)[out_idx] = partial;
            static_cast<int32_t*>(param.candidate_indices)[out_idx] =
                smem_token[t_idx];
          }
        }
      }
    }
  }
}

// iter3 vector 1: persistent-block candidate_score. Extends the iter2 tile-N
// kernel by amortizing the per-row NVFP4 Q dequant across MANY tiles of one
// row (not just kTileN as in iter2). Each CTA is bound to one (row, split)
// pair; within the CTA, Q is decoded ONCE for all heads, then a tight loop
// sweeps tiles_per_split tile-blocks of that row, decoding K rows per tile
// into SMEM and dotting them against the cached Q registers.
//
// Grid:
//   dim3(splits_per_row, q_rows, 1), 256 threads per CTA.
//   splits_per_row is chosen on the host to keep tiles_per_split ~ 32 and
//   to load-balance across the B200 SMs (148 SMs * ~4 CTAs/SM = ~592 active
//   CTAs target). At 64/32768 production this gives splits_per_row=4,
//   tiles_per_split=32, q_rows=64 → 256 CTAs (loose, but adequate fill
//   given the SMEM-bound CTAs/SM ceiling).
//
// Per CTA savings vs iter2 tile-N (kTileN=8):
//   - Q HBM traffic per row: iter2 = (candidate_len/kTileN) * Q_bytes per
//     tile-CTA = 128 Q-decodes for 64/32768. iter3 = 1 Q-decode per CTA *
//     splits_per_row CTAs = 4 Q-decodes for the same row. **32x Q HBM
//     amortization** vs iter2.
//   - Launch grid: iter2 = 128 * 64 = 8192 CTAs; iter3 = 4 * 64 = 256
//     CTAs. **32x grid reduction**, eliminating per-CTA launch + per-CTA
//     metadata load overhead.
//   - K HBM traffic per row: UNCHANGED. Each candidate's K row is still
//     decoded once per kTileN tile. iter3 doesn't amortize K (that would
//     require a CTA to cover a larger candidate range than kTileN, which
//     would inflate K SMEM linearly).
//
// Register pressure: Q registers are reused across all tiles, so we
// pre-decode the entire row into per-warp registers up-front. Each warp
// owns heads_per_warp heads, each thread holds 4 float regs per head, for
// a total of `kMaxHeadsPerWarp * 4` regs/thread. With kMaxHeadsPerWarp=8
// (covering n_heads up to 64) this is 32 float regs/thread — comfortable
// within the SM100 budget at the target 256-thread CTA size and the
// SMEM-bound CTAs/SM ceiling.
template <typename IndicesT, uint32_t kPageSize, uint32_t kTileN>
__global__ void hisa_candidate_score_persistent_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISACandidateScoreParam param,
    uint32_t splits_per_row,
    uint32_t total_tiles) {
  static_assert(kTileN >= 1 && kTileN <= 16,
                "kTileN must be in [1, 16]");
  constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  constexpr uint32_t kHISABlockSize = 128;
  // Production target is n_heads <= 64; the kernel asserts this at host
  // launch. Capping smem_head_dot at 64 keeps the per-CTA SMEM under the
  // default 48 KB limit and lets the SM host more CTAs concurrently.
  constexpr uint32_t kMaxHeads = 64;
  constexpr uint32_t kMaxWarps = 8;
  // Static upper bound on heads_per_warp. With n_heads <= 64 and
  // kMaxWarps = 8, heads_per_warp <= 8.
  constexpr uint32_t kMaxHeadsPerWarp = 8;

  // Per-warp Q register cache. smem_q would have added 32 KB which pushes
  // SMEM over the default 48 KB cap on SM100 even before counting the
  // dynamic SMEM increment. Holding Q in 32 float regs/thread (4 dims x
  // up-to-8 heads per warp) is the better cost/benefit at n_heads <= 64.
  __shared__ float smem_k[kTileN][kIndexerHeadDim];
  __shared__ float smem_head_dot[kTileN][kMaxHeads];

  const auto split_idx = blockIdx.x;
  const auto row = blockIdx.y;
  if (row >= param.q_rows) return;

  const auto warp_id = threadIdx.x >> 5;
  const auto lane = threadIdx.x & 31;

  // Determine this CTA's tile range. ceil-fair partition.
  const uint32_t base_tile =
      static_cast<uint32_t>(static_cast<uint64_t>(total_tiles) * split_idx /
                            splits_per_row);
  const uint32_t end_tile =
      static_cast<uint32_t>(static_cast<uint64_t>(total_tiles) *
                            (split_idx + 1) / splits_per_row);
  if (base_tile >= end_tile) return;

  const auto candidate_len = param.block_topk * kHISABlockSize;
  const uint32_t n_heads = param.n_heads;
  const uint32_t heads_per_warp = (n_heads + kMaxWarps - 1) / kMaxWarps;

  // Stage 0: load row metadata (batch + prefix_len) into per-thread regs
  // by broadcasting through warp 0 → __shfl_sync within a single warp,
  // then full-CTA __syncthreads pinned to the Stage 0b sync.
  __shared__ int32_t smem_batch_prefix[2];  // [0] = batch, [1] = prefix_len
  if (threadIdx.x == 0) {
    smem_batch_prefix[0] =
        static_cast<const int32_t*>(param.token_to_batch_idx)[row];
    smem_batch_prefix[1] =
        static_cast<const int32_t*>(param.seq_lens)[smem_batch_prefix[0]];
  }
  __syncthreads();
  const int32_t batch = smem_batch_prefix[0];
  const int32_t prefix_len = smem_batch_prefix[1];

  // Stage 0b: pre-decode Q into per-warp registers ONCE for this row.
  // Each warp owns heads_per_warp heads; each thread holds 4 floats per
  // head (lane dims {lane, lane+32, lane+64, lane+96}). Q is now
  // amortized across all tiles in [base_tile, end_tile) — the iter3
  // amortization.
  float qregs[kMaxHeadsPerWarp][4];
  #pragma unroll
  for (uint32_t h_off = 0; h_off < kMaxHeadsPerWarp; ++h_off) {
    if (h_off >= heads_per_warp) break;
    const uint32_t head = warp_id * heads_per_warp + h_off;
    if (head >= n_heads) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) qregs[h_off][j] = 0.0f;
      continue;
    }
    const auto q_value_ptr =
        static_cast<const uint8_t*>(param.q_values) +
        (static_cast<int64_t>(row) * n_heads + head) * kNVFP4ValueBytes;
    const auto q_scale_ptr =
        static_cast<const uint32_t*>(param.q_scales) +
        static_cast<int64_t>(row) * n_heads + head;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      qregs[h_off][j] = load_nvfp4_value(q_value_ptr, q_scale_ptr,
                                         lane + j * 32);
    }
  }

  // Main loop: sweep tiles_per_split tile-blocks for this row. Each tile
  // requires 3 __syncthreads (A→B→C→A), down from 4 in the previous draft
  // because (token, valid) is now computed per-warp inside Stage A and
  // broadcast through __shfl_sync rather than published via SMEM.
  constexpr uint32_t kStageRounds = (kTileN + kMaxWarps - 1) / kMaxWarps;
  for (uint32_t tile_idx = base_tile; tile_idx < end_tile; ++tile_idx) {
    const auto cand_base = tile_idx * kTileN;

    // Stage A: each warp computes its tile slot's (token, valid) itself
    // (no SMEM round-trip), then decodes the K row if valid. With
    // kTileN=8, kMaxWarps=8 the kStageRounds=1 case binds warp_id ↔
    // t_idx and each warp's lane 0 holds the (token, valid) needed by
    // Stage C — no SMEM staging needed.
    int32_t warp_token = -1;
    int8_t warp_valid = 0;
    #pragma unroll
    for (uint32_t r = 0; r < kStageRounds; ++r) {
      const uint32_t t_idx = r * kMaxWarps + warp_id;
      if (t_idx < kTileN) {
        const auto cand = cand_base + t_idx;
        int32_t token = -1;
        int8_t valid = 0;
        if (cand < candidate_len) {
          const auto block_slot = cand / kHISABlockSize;
          const auto block_offset = cand - block_slot * kHISABlockSize;
          const auto top_block = static_cast<const int32_t*>(
              param.top_blocks)[row * param.block_topk + block_slot];
          const auto t = top_block * static_cast<int32_t>(kHISABlockSize) +
                         static_cast<int32_t>(block_offset);
          if (top_block >= 0 && t >= 0 && t < prefix_len) {
            token = t;
            valid = 1;
          }
        }
        // Stash in warp-local for Stage C reuse.
        if (r == 0) {
          warp_token = token;
          warp_valid = valid;
        }
        if (valid) {
          const auto logical_page = static_cast<uint32_t>(token) / kPageSize;
          const auto offset = static_cast<uint32_t>(token) & (kPageSize - 1);
          const auto page = static_cast<const IndicesT*>(
              param.page_table)[batch * param.page_table_stride + logical_page];
          const auto page_ptr = static_cast<const uint8_t*>(param.cache) +
                                static_cast<int64_t>(page) * kPageBytes;
          const auto value_ptr = page_ptr + offset * kNVFP4ValueBytes;
          const auto scale_ptr = reinterpret_cast<const uint32_t*>(
              page_ptr + kNVFP4ValueBytes * kPageSize + offset * kScaleBytes);
          #pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t dim = lane + j * 32;
            smem_k[t_idx][dim] = load_nvfp4_value(value_ptr, scale_ptr, dim);
          }
        } else {
          #pragma unroll
          for (int j = 0; j < 4; ++j) {
            smem_k[t_idx][lane + j * 32] = 0.0f;
          }
        }
      }
    }
    __syncthreads();

    // Stage B: per (warp, head) using qregs + smem_k. Inner dot pulls 4
    // Q dims (from registers) and 4 K dims (SMEM, stride-32) per lane,
    // warp_sums across the 32 lanes. Each warp covers its heads_per_warp
    // heads across all kTileN tile slots.
    #pragma unroll
    for (uint32_t h_off = 0; h_off < kMaxHeadsPerWarp; ++h_off) {
      if (h_off >= heads_per_warp) break;
      const uint32_t head = warp_id * heads_per_warp + h_off;
      if (head >= n_heads) break;
      #pragma unroll
      for (uint32_t t = 0; t < kTileN; ++t) {
        if (cand_base + t >= candidate_len) break;
        float dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          dot += qregs[h_off][j] * smem_k[t][lane + j * 32];
        }
        dot = warp_sum(dot);
        if (lane == 0) smem_head_dot[t][head] = dot;
      }
    }
    __syncthreads();

    // Stage C: each warp handles its assigned tile slot (1 round at
    // kTileN=8, kMaxWarps=8). Uses warp_token/warp_valid stashed in
    // Stage A — no SMEM token cache needed.
    if (warp_id < kTileN) {
      const uint32_t t_idx = warp_id;
      const auto cand = cand_base + t_idx;
      if (cand < candidate_len) {
        const auto out_idx =
            static_cast<int64_t>(row) * candidate_len + cand;
        if (!warp_valid) {
          if (lane == 0) {
            static_cast<float*>(param.logits)[out_idx] = -INFINITY;
            static_cast<int32_t*>(param.candidate_indices)[out_idx] = -1;
          }
        } else {
          float partial = 0.0f;
          for (uint32_t h = lane; h < n_heads; h += 32) {
            const float w = static_cast<const float*>(
                param.weights)[static_cast<int64_t>(row) * n_heads + h];
            partial += fmaxf(smem_head_dot[t_idx][h], 0.0f) * w;
          }
          partial = warp_sum(partial);
          if (lane == 0) {
            static_cast<float*>(param.logits)[out_idx] = partial;
            static_cast<int32_t*>(param.candidate_indices)[out_idx] =
                warp_token;
          }
        }
      }
    }
    // For kTileN > kMaxWarps (kTileN=16): a second Stage C round is
    // needed for tile slots 8..15. We don't have warp-local
    // (warp_token, warp_valid) for those, so fall back to recomputing
    // them inline. kTileN <= kMaxWarps is the production fast path.
    if constexpr (kTileN > kMaxWarps) {
      __syncthreads();  // ensure smem_head_dot writes complete for r > 0
      #pragma unroll
      for (uint32_t r = 1; r < kStageRounds; ++r) {
        const uint32_t t_idx = r * kMaxWarps + warp_id;
        if (t_idx < kTileN) {
          const auto cand = cand_base + t_idx;
          if (cand < candidate_len) {
            const auto out_idx =
                static_cast<int64_t>(row) * candidate_len + cand;
            // Recompute (token, valid) — only happens for kTileN > 8.
            const auto block_slot = cand / kHISABlockSize;
            const auto block_offset = cand - block_slot * kHISABlockSize;
            const auto top_block = static_cast<const int32_t*>(
                param.top_blocks)[row * param.block_topk + block_slot];
            const auto t = top_block * static_cast<int32_t>(kHISABlockSize) +
                           static_cast<int32_t>(block_offset);
            const bool valid =
                (top_block >= 0 && t >= 0 && t < prefix_len);
            if (!valid) {
              if (lane == 0) {
                static_cast<float*>(param.logits)[out_idx] = -INFINITY;
                static_cast<int32_t*>(param.candidate_indices)[out_idx] = -1;
              }
            } else {
              float partial = 0.0f;
              for (uint32_t h = lane; h < n_heads; h += 32) {
                const float w = static_cast<const float*>(
                    param.weights)[static_cast<int64_t>(row) * n_heads + h];
                partial += fmaxf(smem_head_dot[t_idx][h], 0.0f) * w;
              }
              partial = warp_sum(partial);
              if (lane == 0) {
                static_cast<float*>(param.logits)[out_idx] = partial;
                static_cast<int32_t*>(param.candidate_indices)[out_idx] = t;
              }
            }
          }
        }
      }
    }
    // Sync before next iter overwrites smem_k.
    __syncthreads();
  }
}

// iter4 PRIMARY: persistent-block candidate_score + K-row cp.async prefetch.
//
// Pairs the iter3 vector 1 persistent kernel (Q amortized across many tiles
// per CTA) with a 2-stage cp.async ping-pong prefetch for the per-tile K
// rows. The iter3 vector 1 checkpoint was 3-60% SLOWER at production shapes:
// per-CTA __syncthreads cost (3 per tile, ~96 syncs at 64/32768) ate the Q
// amortization win because each Stage A blocked on synchronous HBM K-row
// decode latency (`load_nvfp4_value` reads `value_ptr[dim>>1]` + `*scale_ptr`
// from HBM per dim per K row, ~32 HBM scalar loads per K row before the
// Stage A→B __syncthreads can advance).
//
// iter4 hypothesis: hide the K-decode HBM latency by issuing a cp.async
// prefetch of tile N+1's raw NVFP4 bytes into a SECOND SMEM K buffer while
// the warps execute Stage B/C of tile N. Then Stage A of tile N+1 reads
// from SMEM (already landed) instead of HBM, so the __syncthreads becomes
// a pure compute-fence rather than an HBM-stall fence.
//
// Per K row: 64 value bytes + 4 scale bytes (NVFP4 e2m1+ue8m0). cp.async
// can issue 4-, 8-, or 16-byte transactions; we use 4x cp.async.16 for
// values and 1x cp.async.4 for the scale word, giving 5 cp.async instructions
// per K row. Per tile (kTileN=8): 40 cp.async issues, spread across 256
// threads — each thread issues less than 1 op on average, so issue width
// is not the bottleneck.
//
// SMEM layout (per CTA, kTileN=8):
//   smem_k_raw[2][kTileN * 80] = 2 * 640 = 1280 B   (ping-pong raw NVFP4)
//   smem_k    [kTileN][128]    = 4096 B              (decoded fp32 K rows)
//   smem_head_dot[kTileN][64]  = 2048 B              (per-(slot,head) dots)
//   smem_batch_prefix[2]       = 8 B
//   total                      = 7432 B (<< 48 KB cap)
//
// Per K row in smem_k_raw:
//   [0..63]   = 64 value bytes (NVFP4 e2m1 nibbles)
//   [64..67]  = 4 scale bytes (uint32 packed e8m0 scale_word)
//   [68..79]  = 12 padding bytes (16-byte aligned slot stride)
//
// Pipeline:
//   bootstrap: prefetch tile base_tile into buf 0; cp.async.commit_group
//   for tile_idx in [base_tile, end_tile):
//     if tile_idx + 1 < end_tile:
//       prefetch tile (tile_idx + 1) into buf (1 - (tile_idx - base_tile) % 2)
//       cp.async.commit_group
//       cp.async.wait_group<1>  (wait for tile_idx prefetch; let tile_idx+1
//                                stay outstanding to overlap with Stage B/C)
//     else:
//       cp.async.wait_group<0>  (wait for tile_idx prefetch on the last
//                                iteration)
//     __syncthreads()  (ensure all threads see the landed bytes)
//     Stage A: decode K rows from smem_k_raw[cur_buf] into smem_k
//     __syncthreads()
//     Stage B + Stage C (identical to iter3 vector 1)
//     __syncthreads()
//
// The prefetch issue is done by all threads cooperatively (per K row,
// 5 thread slots needed for the 5 cp.async; we use threads 0..40 for
// kTileN=8 — well within the 256-thread CTA). Address computation
// (top_blocks lookup + page_table lookup) is hoisted to threads 0..7
// who broadcast page/offset through SMEM. The page_table read is the
// only HBM dependency of the prefetch issue; it is small (4 B per K row)
// and L1-cached across the persistent CTA's tile sweep (same batch row).
template <typename IndicesT, uint32_t kPageSize, uint32_t kTileN>
__global__ void
hisa_candidate_score_persistent_kprefetch_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISACandidateScoreParam param,
    uint32_t splits_per_row,
    uint32_t total_tiles) {
  static_assert(kTileN >= 1 && kTileN <= 16,
                "kTileN must be in [1, 16]");
  constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  constexpr uint32_t kHISABlockSize = 128;
  // n_heads <= 64 at production; the launcher rejects larger n_heads.
  constexpr uint32_t kMaxHeads = 64;
  constexpr uint32_t kMaxWarps = 8;
  constexpr uint32_t kMaxHeadsPerWarp = 8;

  // Per K-row stride in the raw NVFP4 SMEM buffer: 64 value + 4 scale +
  // 12 pad = 80 bytes, 16-byte aligned. The kRowStrideBytes constant must
  // be a multiple of 16 so the cp.async.16 instructions remain aligned.
  constexpr uint32_t kRowStrideBytes = 80;
  static_assert(kRowStrideBytes % 16 == 0, "kRowStrideBytes must be 16-aligned");
  constexpr uint32_t kRawBufBytes = kTileN * kRowStrideBytes;

  __shared__ alignas(16) uint8_t smem_k_raw[2][kRawBufBytes];
  __shared__ float smem_k[kTileN][kIndexerHeadDim];
  __shared__ float smem_head_dot[kTileN][kMaxHeads];
  // Per-tile metadata (token + valid) staged so Stage C can read it from
  // SMEM without re-deriving — the iter3 v1 fast path could reuse warp-local
  // regs because Stage A/C bound 1 K row per warp; for iter4 we want all
  // threads to see (token, valid) for every K row regardless of which warp
  // issued the prefetch, so we publish it explicitly.
  __shared__ int32_t smem_token[2][kTileN];
  __shared__ int8_t smem_valid[2][kTileN];

  const auto split_idx = blockIdx.x;
  const auto row = blockIdx.y;
  if (row >= param.q_rows) return;

  const auto warp_id = threadIdx.x >> 5;
  const auto lane = threadIdx.x & 31;

  const uint32_t base_tile =
      static_cast<uint32_t>(static_cast<uint64_t>(total_tiles) * split_idx /
                            splits_per_row);
  const uint32_t end_tile =
      static_cast<uint32_t>(static_cast<uint64_t>(total_tiles) *
                            (split_idx + 1) / splits_per_row);
  if (base_tile >= end_tile) return;

  const auto candidate_len = param.block_topk * kHISABlockSize;
  const uint32_t n_heads = param.n_heads;
  const uint32_t heads_per_warp = (n_heads + kMaxWarps - 1) / kMaxWarps;

  // Stage 0: row metadata.
  __shared__ int32_t smem_batch_prefix[2];
  if (threadIdx.x == 0) {
    smem_batch_prefix[0] =
        static_cast<const int32_t*>(param.token_to_batch_idx)[row];
    smem_batch_prefix[1] =
        static_cast<const int32_t*>(param.seq_lens)[smem_batch_prefix[0]];
  }
  __syncthreads();
  const int32_t batch = smem_batch_prefix[0];
  const int32_t prefix_len = smem_batch_prefix[1];

  // Stage 0b: pre-decode Q into per-warp registers (identical to iter3 v1).
  float qregs[kMaxHeadsPerWarp][4];
  #pragma unroll
  for (uint32_t h_off = 0; h_off < kMaxHeadsPerWarp; ++h_off) {
    if (h_off >= heads_per_warp) break;
    const uint32_t head = warp_id * heads_per_warp + h_off;
    if (head >= n_heads) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) qregs[h_off][j] = 0.0f;
      continue;
    }
    const auto q_value_ptr =
        static_cast<const uint8_t*>(param.q_values) +
        (static_cast<int64_t>(row) * n_heads + head) * kNVFP4ValueBytes;
    const auto q_scale_ptr =
        static_cast<const uint32_t*>(param.q_scales) +
        static_cast<int64_t>(row) * n_heads + head;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      qregs[h_off][j] = load_nvfp4_value(q_value_ptr, q_scale_ptr,
                                         lane + j * 32);
    }
  }

  // ===== Prefetch lambda =====
  // Computes K-row addresses for tile `tile_idx` and issues cp.async to
  // copy the raw NVFP4 bytes into smem_k_raw[buf]. Also publishes
  // (token, valid) into smem_token[buf][.] / smem_valid[buf][.] for Stage
  // A / Stage C consumption. NOT followed by a syncthreads or fence —
  // the caller must invoke cp_async_fence() after this returns.
  //
  // Thread layout for prefetch issue:
  //   - threads 0..kTileN-1: address derivation + (token, valid) publish
  //   - the same threads issue 4x cp.async.16 + 1x cp.async.4 for their
  //     K row. Issue is per-thread (cp.async is a per-thread instruction),
  //     so threads 0..7 issue all 40 ops for kTileN=8.
  auto prefetch_tile = [&](uint32_t tile_idx, uint32_t buf) {
    const auto cand_base = tile_idx * kTileN;
    if (threadIdx.x < kTileN) {
      const uint32_t t_idx = threadIdx.x;
      const auto cand = cand_base + t_idx;
      int32_t token = -1;
      int8_t valid = 0;
      const uint8_t* value_src = nullptr;
      const uint32_t* scale_src = nullptr;
      if (cand < candidate_len) {
        const auto block_slot = cand / kHISABlockSize;
        const auto block_offset = cand - block_slot * kHISABlockSize;
        const auto top_block = static_cast<const int32_t*>(
            param.top_blocks)[row * param.block_topk + block_slot];
        const auto t = top_block * static_cast<int32_t>(kHISABlockSize) +
                       static_cast<int32_t>(block_offset);
        if (top_block >= 0 && t >= 0 && t < prefix_len) {
          token = t;
          valid = 1;
          const auto logical_page = static_cast<uint32_t>(t) / kPageSize;
          const auto offset = static_cast<uint32_t>(t) & (kPageSize - 1);
          const auto page = static_cast<const IndicesT*>(
              param.page_table)[batch * param.page_table_stride + logical_page];
          const auto page_ptr = static_cast<const uint8_t*>(param.cache) +
                                static_cast<int64_t>(page) * kPageBytes;
          value_src = page_ptr + offset * kNVFP4ValueBytes;
          scale_src = reinterpret_cast<const uint32_t*>(
              page_ptr + kNVFP4ValueBytes * kPageSize + offset * kScaleBytes);
        }
      }
      smem_token[buf][t_idx] = token;
      smem_valid[buf][t_idx] = valid;
      uint8_t* row_dst = smem_k_raw[buf] + t_idx * kRowStrideBytes;
      if (valid) {
        // Issue 4x cp.async.16 for the 64 value bytes (16-byte aligned).
        // Each issue copies 16 bytes from HBM to SMEM and registers the
        // completion against the most-recent cp.async.commit_group fence.
        #pragma unroll
        for (uint32_t v = 0; v < 4; ++v) {
          cp_async_16(row_dst + v * 16, value_src + v * 16);
        }
        // Issue 1x cp.async.4 for the scale word at offset 64.
        cp_async_4(row_dst + 64, scale_src);
      } else {
        // Invalid K rows: write zeros directly to SMEM (no cp.async).
        // The decode path will see zero values + zero scale_word and emit
        // zero contributions, identical to iter3 v1's explicit zero branch.
        #pragma unroll
        for (uint32_t v = 0; v < 4; ++v) {
          reinterpret_cast<uint4*>(row_dst + v * 16)[0] = make_uint4(0, 0, 0, 0);
        }
        *reinterpret_cast<uint32_t*>(row_dst + 64) = 0u;
      }
    }
    // Threads >= kTileN do not issue cp.async for this tile. The
    // cp_async_fence() / wait_group must still be issued by every thread
    // because they are per-thread instructions (the fence groups the
    // issuing thread's outstanding ops; threads with zero outstanding
    // ops still need to participate in the group boundary).
  };

  // ===== Decode-from-SMEM lambda =====
  // Reads the raw NVFP4 bytes from smem_k_raw[buf] and writes the decoded
  // fp32 values into smem_k. Uses the same decode_e2m1_nibble + scale
  // arithmetic as load_nvfp4_value, but with both value_ptr and scale_ptr
  // pointing into SMEM instead of HBM. Each warp owns one K row at
  // kTileN<=8, kMaxWarps=8 (kStageARounds=1).
  auto decode_tile = [&](uint32_t buf) {
    constexpr uint32_t kStageARounds = (kTileN + kMaxWarps - 1) / kMaxWarps;
    #pragma unroll
    for (uint32_t r = 0; r < kStageARounds; ++r) {
      const uint32_t t_idx = r * kMaxWarps + warp_id;
      if (t_idx < kTileN) {
        const uint8_t* row_ptr = smem_k_raw[buf] + t_idx * kRowStrideBytes;
        const uint32_t scale_word =
            *reinterpret_cast<const uint32_t*>(row_ptr + 64);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t dim = lane + j * 32;
          const auto packed = row_ptr[dim >> 1];
          const auto code = (dim & 1) ? (packed >> 4) : (packed & 0xfu);
          const auto scale_exp = (scale_word >> ((dim >> 5) * 8)) & 0xffu;
          const auto scale = __uint_as_float(scale_exp << 23);
          smem_k[t_idx][dim] = decode_e2m1_nibble(code, scale);
        }
      }
    }
  };

  // ===== Pipeline bootstrap =====
  // Prefetch tile 0 into buf 0 and commit it as the first cp.async group.
  prefetch_tile(base_tile, 0u);
  cp_async_fence();
  // smem_token[0] / smem_valid[0] writes happen-before cp_async_fence and
  // are visible to all threads after a __syncthreads. We sync just before
  // the main loop's Stage C consumes them.

  // ===== Main loop =====
  constexpr uint32_t kStageRounds = (kTileN + kMaxWarps - 1) / kMaxWarps;
  for (uint32_t tile_idx = base_tile; tile_idx < end_tile; ++tile_idx) {
    const uint32_t cur_buf = (tile_idx - base_tile) & 1u;
    const uint32_t next_tile = tile_idx + 1;
    const bool has_next = next_tile < end_tile;
    if (has_next) {
      const uint32_t next_buf = cur_buf ^ 1u;
      prefetch_tile(next_tile, next_buf);
      cp_async_fence();
      // Two groups outstanding (cur + next). Wait until cur lands.
      cp_async_wait_group<1>();
    } else {
      // Last tile: only one group outstanding. Drain everything.
      cp_async_wait_group<0>();
    }
    // The wait_group above releases this thread's view of the prior
    // cp.async; sync to ensure all threads have committed their writes
    // and prefetch metadata is visible.
    __syncthreads();

    // Stage A: decode K rows from cur_buf raw bytes into smem_k.
    decode_tile(cur_buf);
    __syncthreads();

    // Stage B: per (warp, head) dot. Identical to iter3 v1.
    #pragma unroll
    for (uint32_t h_off = 0; h_off < kMaxHeadsPerWarp; ++h_off) {
      if (h_off >= heads_per_warp) break;
      const uint32_t head = warp_id * heads_per_warp + h_off;
      if (head >= n_heads) break;
      const auto cand_base = tile_idx * kTileN;
      #pragma unroll
      for (uint32_t t = 0; t < kTileN; ++t) {
        if (cand_base + t >= candidate_len) break;
        float dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          dot += qregs[h_off][j] * smem_k[t][lane + j * 32];
        }
        dot = warp_sum(dot);
        if (lane == 0) smem_head_dot[t][head] = dot;
      }
    }
    __syncthreads();

    // Stage C: per-tile weighted reduction. Pulls (token, valid) from
    // smem_token[cur_buf] / smem_valid[cur_buf] — the iter3 v1 fast-path
    // warp-local stash doesn't work here because the warp that issued
    // the prefetch (one of warps 0..kTileN-1 if threadIdx.x < kTileN)
    // is not necessarily the warp consuming this tile slot in Stage C.
    #pragma unroll
    for (uint32_t r = 0; r < kStageRounds; ++r) {
      const uint32_t t_idx = r * kMaxWarps + warp_id;
      if (t_idx < kTileN) {
        const auto cand_base = tile_idx * kTileN;
        const auto cand = cand_base + t_idx;
        if (cand < candidate_len) {
          const auto out_idx =
              static_cast<int64_t>(row) * candidate_len + cand;
          const int32_t token = smem_token[cur_buf][t_idx];
          const int8_t valid = smem_valid[cur_buf][t_idx];
          if (!valid) {
            if (lane == 0) {
              static_cast<float*>(param.logits)[out_idx] = -INFINITY;
              static_cast<int32_t*>(param.candidate_indices)[out_idx] = -1;
            }
          } else {
            float partial = 0.0f;
            for (uint32_t h = lane; h < n_heads; h += 32) {
              const float w = static_cast<const float*>(
                  param.weights)[static_cast<int64_t>(row) * n_heads + h];
              partial += fmaxf(smem_head_dot[t_idx][h], 0.0f) * w;
            }
            partial = warp_sum(partial);
            if (lane == 0) {
              static_cast<float*>(param.logits)[out_idx] = partial;
              static_cast<int32_t*>(param.candidate_indices)[out_idx] = token;
            }
          }
        }
      }
    }
    // Note: NO end-of-tile __syncthreads. The next iter's syncthreads-
    // after-wait_group (sync A in the comments) closes the cur-iter
    // Stage C → next-iter Stage B race because Stage B is what would
    // overwrite smem_head_dot, and Stage B is preceded by both that
    // sync A AND the decode_tile sync (Stage A→B). Dropping the
    // end-of-tile sync gives iter4 kprefetch the same 3-sync per tile
    // count as iter3 v1 (instead of 4 syncs with the explicit end-of-
    // tile sync).
  }
}

__global__ void hisa_block_score_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISABlockScoreParam param) {
  // Rewritten iter1 fused-head design. The legacy kernel used
  //   for linear in [0, n_heads * 128): load reps[dim], decode q[head,dim],
  //     atomicAdd(&head_dot[head], q * k);
  // which serialised on shared-mem atomicAdd (256 threads contending over
  // 64 slots) and reloaded reps[dim] multiple times across head iterations
  // (linear/128 == head, linear%128 == dim — dim 0..31 across head 0..63
  //  reloads the same 32 reps floats 64 times).
  //
  // New layout: cooperatively stage reps[0..128) into shared memory; each
  // lane pulls its 4 reps dims into registers; each warp owns
  // ceil(n_heads / kMaxWarps) heads and computes the dot in registers via
  // warp_sum — no atomics. Final weighted reduction happens on warp 0 only.
  //
  // Correctness is equivalent (within FP32 ordering round-off) to the
  // legacy atomicAdd version because both compute the same dot products
  // followed by sum_h fmaxf(dot[h], 0) * weight[h].
  //
  // ===== iter5 feasibility note: UMMA / tcgen05.mma block_score =====
  //
  // Goal: replace the warp-shuffle dot + per-head reduction with a
  // single tcgen05.mma over (q_values × block_reps) on SM_100. The op
  // shape is naturally a batched GEMM: per row, output is
  // C[n_heads, max_blocks] = Q[n_heads, head_dim] @ R[max_blocks, head_dim]^T
  // followed by the masked-ReLU epilogue
  // score[block] = sum_h fmaxf(C[head, block], 0) * weights[head].
  //
  // Architectural fit on B200 (SM_100):
  // - tcgen05.mma.mxfp4.mxfp4.fp32 lets us feed the NVFP4 q_values directly
  //   without a per-CTA dequant pass; ue8m0 scales ride along via the
  //   .scale_a/.scale_b paths described in the SM_100 PTX ISA.
  // - The smallest m-tile for mxfp4 is m=64 — exactly the production
  //   n_heads. One m-tile covers all heads of one row, so the m-axis is
  //   filled without wave fragmentation.
  // - head_dim=128 fits as k=128 inside a single tcgen05.mma group
  //   (4 × k=32 mxfp4 tiles or 1 × k=128 with the chunked instruction).
  // - max_blocks (= prefix/128, ~256 at 32k prefix) is the N axis. The
  //   tcgen05.mma N-tile is configurable from N=8 up. Picking N=64 (or
  //   N=128 if SMEM allows the larger R tile) covers 4 or 2 wave passes
  //   per row at 32k prefix.
  //
  // Implementation cost (engineering, not GPU):
  // 1. TMA descriptors for q_values and reps. q_values is contiguous
  //    [q_rows, n_heads, kNVFP4ValueBytes]; the per-row stride is
  //    n_heads*kNVFP4ValueBytes, OK for a 2D TMA. reps is
  //    [batch, max_blocks, head_dim] fp32, and the per-row batch mapping
  //    comes from token_to_batch_idx — that's an indexed access pattern
  //    not directly supported by TMA descriptors. Two options:
  //    (a) build a per-launch gather table on the host that resolves
  //        (row → batch) and emits per-row reps base pointers, then
  //        thread 0 of each row-CTA programs the descriptor;
  //    (b) keep reps unchanged and rely on the kernel's prologue (TMA
  //        for the [n_heads, head_dim] q-tile, manual ldgsts for
  //        reps[batch, :, :]). (a) is faster but adds host-side
  //        prep cost (~5-10 us per call); (b) is simpler but loses
  //        some TMA throughput.
  // 2. TMEM allocation (tcgen05.alloc) for the fp32 C accumulator
  //    [m=64, N=64 or 128]. SM_100 TMEM ceiling is 256 KB per SM, far
  //    more than the ~16-32 KB needed.
  // 3. The masked-ReLU + weighted-sum epilogue runs on the row-CTA's
  //    warps after tcgen05.ld pulls the fp32 dots out of TMEM. Layout:
  //    per warp owns one N-tile slice (16 blocks); warp does
  //    fmaxf(dot[h], 0) * weight[h] across all 64 heads via warp_sum.
  //    1 syncthreads + 1 ldsm + 1 warp-sum chain per block-slice.
  //
  // Expected gain (B200 SM_100 microbench, iter5 target estimate):
  // - Current iter1 kernel at (n_heads=64, prefix=16384, q_rows=32):
  //   ~50 us (1024 row-block CTAs each doing 32-thread × 4-dim × 64-head
  //   fused dot + warp-sum + weighted reduction).
  // - tcgen05.mma m64n64k128 mxfp4.fp32 throughput at SM_100: ~3.4
  //   PFLOPS (mxfp4) → per-MMA ~0.3 us at full utilization. Per row,
  //   max_blocks/64 = 4 MMA waves → ~1.2 us. Epilogue ~5 us. Per-row
  //   total ~6 us. 32 rows / 148 SMs = ~6 rows × 6 us = ~36 us.
  // - Realistic discount for SMEM swizzle stalls + TMA descriptor
  //   issue: ~50% utilization → ~40-45 us. Net win: ~10-15 us
  //   (20-30%), not the 5x optimistic estimate. The m=64 row dim is
  //   small for tensor-core throughput; an SM_100 mxfp4 m-tile is
  //   throughput-optimal at m>=128.
  //
  // Risk: the host-side gather table (option 1a above) adds CPU
  //   overhead per call. For batch=1 with index_topk_freq=4 this is
  //   amortized away; for batch>=64 the prep cost may exceed the
  //   savings. iter5 should A/B with the simpler (1b) variant first.
  //
  // Verdict for iter5: tcgen05.mma block_score is feasible but the
  //   expected gain (~20-30%, ~10-15 us absolute) is modest. The
  //   stronger iter5 candidates are (i) mean_pool inner-loop FMA pair
  //   over 2 tokens per FMA (~20% mean_pool, ~30us at 64/32768);
  //   (ii) candidate_score WMMA dot (mma.m16n8k8 fp32, replaces the
  //   warp_sum scalar chain in Stage B, ~15-25% cand_score). Both
  //   are simpler and have larger absolute impact than UMMA
  //   block_score at the production shape grid.
  constexpr uint32_t kHISABlockSize = 128;
  constexpr uint32_t kMaxWarps = 8;
  constexpr uint32_t kMaxHeads = 64;
  __shared__ float smem_reps[kIndexerHeadDim];
  __shared__ float head_dot[kMaxHeads];

  const auto block_id = blockIdx.x;
  const auto row = blockIdx.y;
  if (row >= param.q_rows || block_id >= param.max_blocks) return;

  const auto batch = static_cast<const int32_t*>(param.token_to_batch_idx)[row];
  const auto prefix_len = static_cast<const int32_t*>(param.seq_lens)[batch];
  const auto block_count =
      (static_cast<uint32_t>(prefix_len) + kHISABlockSize - 1) / kHISABlockSize;
  const auto out_idx = static_cast<int64_t>(row) * param.max_blocks + block_id;
  if (block_id >= block_count) {
    if (threadIdx.x == 0) static_cast<float*>(param.block_scores)[out_idx] = -INFINITY;
    return;
  }

  const auto warp_id = threadIdx.x >> 5;
  const auto lane = threadIdx.x & 31;

  // Stage 1: load 128 reps floats into shared memory cooperatively.
  const auto rep_base =
      (static_cast<int64_t>(batch) * param.max_blocks + block_id) * kIndexerHeadDim;
  const auto* reps_ptr = static_cast<const float*>(param.reps) + rep_base;
  if (threadIdx.x < kIndexerHeadDim) {
    smem_reps[threadIdx.x] = reps_ptr[threadIdx.x];
  }
  __syncthreads();

  // Stage 2: each lane reads its 4 dims of reps into registers.
  float kvals[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    kvals[j] = smem_reps[lane + j * 32];
  }

  // Stage 3: per-warp head dispatch.
  const auto heads_per_warp = (param.n_heads + kMaxWarps - 1) / kMaxWarps;
  for (uint32_t h_off = 0; h_off < heads_per_warp; ++h_off) {
    const uint32_t head = warp_id * heads_per_warp + h_off;
    if (head >= param.n_heads) break;

    const auto q_value_ptr =
        static_cast<const uint8_t*>(param.q_values) +
        (static_cast<int64_t>(row) * param.n_heads + head) * kNVFP4ValueBytes;
    const auto q_scale_ptr =
        static_cast<const uint32_t*>(param.q_scales) +
        static_cast<int64_t>(row) * param.n_heads + head;
    float dot = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t dim = lane + j * 32;
      const float qval = load_nvfp4_value(q_value_ptr, q_scale_ptr, dim);
      dot += qval * kvals[j];
    }
    dot = warp_sum(dot);
    if (lane == 0) head_dot[head] = dot;
  }
  __syncthreads();

  // Stage 4: single-warp weighted reduction. n_heads ≤ 64, so 32 lanes is
  // enough — each lane sums ≤ 2 heads' contributions.
  if (warp_id == 0) {
    float partial = 0.0f;
    for (uint32_t h = lane; h < param.n_heads; h += 32) {
      const float w =
          static_cast<const float*>(param.weights)[
              static_cast<int64_t>(row) * param.n_heads + h];
      partial += fmaxf(head_dot[h], 0.0f) * w;
    }
    partial = warp_sum(partial);
    if (lane == 0) static_cast<float*>(param.block_scores)[out_idx] = partial;
  }
}

__global__ void hisa_block_topk_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISABlockTopKParam param) {
  extern __shared__ uint8_t smem[];
  auto* best_vals = reinterpret_cast<float*>(smem);
  auto* best_idxs = reinterpret_cast<int32_t*>(best_vals + blockDim.x);
  auto* selected = best_idxs + blockDim.x;

  const auto row = blockIdx.x;
  if (row >= param.q_rows) return;
  const auto tid = threadIdx.x;
  const auto block_count =
      min(static_cast<uint32_t>(static_cast<const int32_t*>(param.block_counts)[row]),
          param.max_blocks);
  if (tid < param.block_topk) {
    selected[tid] = -1;
    static_cast<int32_t*>(param.top_blocks)[row * param.block_topk + tid] = -1;
  }
  __syncthreads();
  if (block_count == 0) return;

  const auto row_block_topk =
      min(static_cast<uint32_t>(
              static_cast<const int32_t*>(param.block_topk_counts)[row]),
          param.block_topk);
  const auto keep = min(row_block_topk, block_count);
  const auto* scores = static_cast<const float*>(param.block_scores) +
                       static_cast<int64_t>(row) * param.max_blocks;
  if (block_count <= blockDim.x) {
    auto score = tid < block_count ? scores[tid] : -INFINITY;
    auto idx = tid < block_count ? static_cast<int32_t>(tid) : -1;
    if (tid < block_count && (tid == 0 || tid + 1 == block_count)) {
      score = INFINITY;
    }
    best_vals[tid] = score;
    best_idxs[tid] = idx;
    __syncthreads();

    for (uint32_t width = 2; width <= blockDim.x; width <<= 1) {
      for (uint32_t stride = width >> 1; stride > 0; stride >>= 1) {
        const auto other = tid ^ stride;
        if (other > tid) {
          const auto self_val = best_vals[tid];
          const auto other_val = best_vals[other];
          const auto self_idx = best_idxs[tid];
          const auto other_idx = best_idxs[other];
          const auto other_better =
              other_val > self_val ||
              (other_val == self_val &&
               (self_idx < 0 || (other_idx >= 0 && other_idx < self_idx)));
          const auto self_better =
              self_val > other_val ||
              (self_val == other_val &&
               (other_idx < 0 || (self_idx >= 0 && self_idx < other_idx)));
          const auto descending = (tid & width) == 0;
          if ((descending && other_better) || (!descending && self_better)) {
            best_vals[tid] = other_val;
            best_idxs[tid] = other_idx;
            best_vals[other] = self_val;
            best_idxs[other] = self_idx;
          }
        }
        __syncthreads();
      }
    }
    if (tid < param.block_topk) {
      const auto out_idx = static_cast<int64_t>(row) * param.block_topk + tid;
      static_cast<int32_t*>(param.top_blocks)[out_idx] =
          tid < keep ? best_idxs[tid] : -1;
    }
    return;
  }

  for (uint32_t out = 0; out < keep; ++out) {
    float local_best = -INFINITY;
    int32_t local_idx = -1;
    for (uint32_t block_id = tid; block_id < block_count; block_id += blockDim.x) {
      bool already_selected = false;
      for (uint32_t i = 0; i < out; ++i) {
        already_selected |= selected[i] == static_cast<int32_t>(block_id);
      }
      if (already_selected) continue;

      auto score = scores[block_id];
      if (block_id == 0 || block_id + 1 == block_count) {
        score = INFINITY;
      }
      if (score > local_best ||
          (score == local_best && static_cast<int32_t>(block_id) < local_idx)) {
        local_best = score;
        local_idx = static_cast<int32_t>(block_id);
      }
    }
    best_vals[tid] = local_best;
    best_idxs[tid] = local_idx;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const auto other_val = best_vals[tid + stride];
        const auto other_idx = best_idxs[tid + stride];
        if (other_val > best_vals[tid] ||
            (other_val == best_vals[tid] &&
             (best_idxs[tid] < 0 || (other_idx >= 0 && other_idx < best_idxs[tid])))) {
          best_vals[tid] = other_val;
          best_idxs[tid] = other_idx;
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      selected[out] = best_idxs[0];
      static_cast<int32_t*>(param.top_blocks)[row * param.block_topk + out] =
          best_idxs[0];
    }
    __syncthreads();
  }
}

SGL_DEVICE void write_hisa_block_tokens(
    void* __restrict__ topk_indices,
    uint32_t row,
    uint32_t topk,
    uint32_t out,
    int32_t block_id,
    int32_t prefix_len) {
  constexpr uint32_t kHISABlockSize = 128;
  const auto base = static_cast<int64_t>(row) * topk + out * kHISABlockSize;
  for (uint32_t offset = threadIdx.x; offset < kHISABlockSize; offset += blockDim.x) {
    int32_t token = -1;
    if (block_id >= 0) {
      token = block_id * static_cast<int32_t>(kHISABlockSize) +
              static_cast<int32_t>(offset);
      if (token >= prefix_len) token = -1;
    }
    static_cast<int32_t*>(topk_indices)[base + offset] = token;
  }
}

__global__ void hisa_block_topk_map_all_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISABlockTopKMapAllParam param) {
  extern __shared__ uint8_t smem[];
  auto* best_vals = reinterpret_cast<float*>(smem);
  auto* best_idxs = reinterpret_cast<int32_t*>(best_vals + blockDim.x);
  auto* selected = best_idxs + blockDim.x;

  const auto row = blockIdx.x;
  if (row >= param.q_rows) return;
  const auto tid = threadIdx.x;

  constexpr uint32_t kHISABlockSize = 128;
  const auto block_count =
      min(static_cast<uint32_t>(static_cast<const int32_t*>(param.block_counts)[row]),
          param.max_blocks);
  const auto row_block_topk =
      min(static_cast<uint32_t>(
              static_cast<const int32_t*>(param.block_topk_counts)[row]),
          param.block_topk);
  const auto keep = min(row_block_topk, block_count);

  if (keep < param.block_topk) {
    const auto clear_begin = keep * kHISABlockSize;
    for (uint32_t idx = clear_begin + tid; idx < param.topk; idx += blockDim.x) {
      static_cast<int32_t*>(param.topk_indices)[
          static_cast<int64_t>(row) * param.topk + idx] = -1;
    }
  }

  if (tid < param.block_topk) selected[tid] = -1;
  __syncthreads();
  if (block_count == 0) return;
  const auto prefix_len = static_cast<const int32_t*>(param.prefix_lens)[row];
  const auto* scores = static_cast<const float*>(param.block_scores) +
                       static_cast<int64_t>(row) * param.max_blocks;
  if (block_count <= blockDim.x) {
    auto score = tid < block_count ? scores[tid] : -INFINITY;
    auto idx = tid < block_count ? static_cast<int32_t>(tid) : -1;
    if (tid < block_count && (tid == 0 || tid + 1 == block_count)) {
      score = INFINITY;
    }
    best_vals[tid] = score;
    best_idxs[tid] = idx;
    __syncthreads();

    for (uint32_t width = 2; width <= blockDim.x; width <<= 1) {
      for (uint32_t stride = width >> 1; stride > 0; stride >>= 1) {
        const auto other = tid ^ stride;
        if (other > tid) {
          const auto self_val = best_vals[tid];
          const auto other_val = best_vals[other];
          const auto self_idx = best_idxs[tid];
          const auto other_idx = best_idxs[other];
          const auto other_better =
              other_val > self_val ||
              (other_val == self_val &&
               (self_idx < 0 || (other_idx >= 0 && other_idx < self_idx)));
          const auto self_better =
              self_val > other_val ||
              (self_val == other_val &&
               (other_idx < 0 || (self_idx >= 0 && self_idx < other_idx)));
          const auto descending = (tid & width) == 0;
          if ((descending && other_better) || (!descending && self_better)) {
            best_vals[tid] = other_val;
            best_idxs[tid] = other_idx;
            best_vals[other] = self_val;
            best_idxs[other] = self_idx;
          }
        }
        __syncthreads();
      }
    }
    for (uint32_t out = 0; out < keep; ++out) {
      write_hisa_block_tokens(
          param.topk_indices, row, param.topk, out, best_idxs[out], prefix_len);
    }
    return;
  }

  for (uint32_t out = 0; out < keep; ++out) {
    float local_best = -INFINITY;
    int32_t local_idx = -1;
    for (uint32_t block_id = tid; block_id < block_count; block_id += blockDim.x) {
      bool already_selected = false;
      for (uint32_t i = 0; i < out; ++i) {
        already_selected |= selected[i] == static_cast<int32_t>(block_id);
      }
      if (already_selected) continue;

      auto score = scores[block_id];
      if (block_id == 0 || block_id + 1 == block_count) {
        score = INFINITY;
      }
      if (score > local_best ||
          (score == local_best && static_cast<int32_t>(block_id) < local_idx)) {
        local_best = score;
        local_idx = static_cast<int32_t>(block_id);
      }
    }
    best_vals[tid] = local_best;
    best_idxs[tid] = local_idx;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const auto other_val = best_vals[tid + stride];
        const auto other_idx = best_idxs[tid + stride];
        if (other_val > best_vals[tid] ||
            (other_val == best_vals[tid] &&
             (best_idxs[tid] < 0 || (other_idx >= 0 && other_idx < best_idxs[tid])))) {
          best_vals[tid] = other_val;
          best_idxs[tid] = other_idx;
        }
      }
      __syncthreads();
    }
    if (tid == 0) selected[out] = best_idxs[0];
    __syncthreads();
    write_hisa_block_tokens(
        param.topk_indices, row, param.topk, out, selected[out], prefix_len);
    __syncthreads();
  }
}

template <typename IndicesT>
__global__ void hisa_candidate_pages_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISACandidatePagesParam param) {
  const auto row = blockIdx.x;
  if (row >= param.q_rows) return;

  for (uint32_t idx = threadIdx.x; idx < param.block_topk * 2; idx += blockDim.x) {
    const auto block_slot = idx >> 1;
    const auto half = idx & 1;
    const auto top_block =
        static_cast<const int32_t*>(param.top_blocks)[row * param.block_topk + block_slot];
    const auto out_idx =
        static_cast<int64_t>(row) * param.block_topk * 2 + idx;
    if (top_block < 0) {
      static_cast<IndicesT*>(param.candidate_page_table)[out_idx] = 0;
      continue;
    }
    const auto batch = static_cast<const int32_t*>(param.token_to_batch_idx)[row];
    const auto logical_page = static_cast<uint32_t>(top_block) * 2 + half;
    static_cast<IndicesT*>(param.candidate_page_table)[out_idx] =
        static_cast<const IndicesT*>(param.page_table)[batch * param.page_table_stride +
                                                       logical_page];
  }
}

__global__ void hisa_mask_candidate_logits_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISACandidateMaskParam param) {
  constexpr uint32_t kHISABlockSize = 128;
  const auto linear = blockIdx.x * blockDim.x + threadIdx.x;
  const auto total = param.q_rows * param.candidate_len;
  if (linear >= total) return;

  const auto row = linear / param.candidate_len;
  const auto cand = linear - row * param.candidate_len;
  const auto block_slot = cand / kHISABlockSize;
  const auto block_offset = cand - block_slot * kHISABlockSize;
  const auto top_block =
      static_cast<const int32_t*>(param.top_blocks)[row * param.block_topk +
                                                    block_slot];
  auto token = -1;
  if (top_block >= 0) {
    token = top_block * static_cast<int32_t>(kHISABlockSize) +
            static_cast<int32_t>(block_offset);
  }
  const auto prefix_len = static_cast<const int32_t*>(param.prefix_lens)[row];
  if (token < 0 || token >= prefix_len) {
    static_cast<float*>(param.logits)[linear] = -INFINITY;
    token = -1;
  }
  static_cast<int32_t*>(param.candidate_indices)[linear] = token;
}

__global__ void hisa_mask_logits_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISACandidateMaskParam param) {
  constexpr uint32_t kHISABlockSize = 128;
  const auto linear = blockIdx.x * blockDim.x + threadIdx.x;
  const auto total = param.q_rows * param.candidate_len;
  if (linear >= total) return;

  const auto row = linear / param.candidate_len;
  const auto cand = linear - row * param.candidate_len;
  const auto block_slot = cand / kHISABlockSize;
  const auto block_offset = cand - block_slot * kHISABlockSize;
  const auto top_block =
      static_cast<const int32_t*>(param.top_blocks)[row * param.block_topk +
                                                    block_slot];
  auto token = -1;
  if (top_block >= 0) {
    token = top_block * static_cast<int32_t>(kHISABlockSize) +
            static_cast<int32_t>(block_offset);
  }
  const auto prefix_len = static_cast<const int32_t*>(param.prefix_lens)[row];
  if (token < 0 || token >= prefix_len) {
    static_cast<float*>(param.logits)[linear] = -INFINITY;
  }
}

__global__ void hisa_map_topk_indices_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISAMapTopKParam param) {
  constexpr uint32_t kHISABlockSize = 128;
  const auto linear = blockIdx.x * blockDim.x + threadIdx.x;
  const auto total = param.q_rows * param.topk;
  if (linear >= total) return;

  const auto row = linear / param.topk;
  const auto cand = static_cast<const int64_t*>(param.topk_positions)[linear];
  const auto block_slot = cand / kHISABlockSize;
  const auto block_offset = cand - block_slot * kHISABlockSize;
  const auto top_block =
      static_cast<const int32_t*>(param.top_blocks)[row * param.block_topk +
                                                    block_slot];
  auto token = -1;
  if (top_block >= 0) {
    token = top_block * static_cast<int32_t>(kHISABlockSize) +
            static_cast<int32_t>(block_offset);
  }
  const auto prefix_len = static_cast<const int32_t*>(param.prefix_lens)[row];
  static_cast<int32_t*>(param.topk_indices)[linear] =
      token >= 0 && token < prefix_len ? token : -1;
}

__global__ void hisa_map_candidate_indices_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISAMapCandidatesParam param) {
  constexpr uint32_t kHISABlockSize = 128;
  const auto linear = blockIdx.x * blockDim.x + threadIdx.x;
  const auto total = param.q_rows * param.topk;
  if (linear >= total) return;

  const auto row = linear / param.topk;
  const auto cand = linear - row * param.topk;
  const auto block_slot = cand / kHISABlockSize;
  const auto block_offset = cand - block_slot * kHISABlockSize;
  auto token = -1;
  if (block_slot < param.block_topk) {
    const auto top_block =
        static_cast<const int32_t*>(param.top_blocks)[row * param.block_topk +
                                                      block_slot];
    if (top_block >= 0) {
      token = top_block * static_cast<int32_t>(kHISABlockSize) +
              static_cast<int32_t>(block_offset);
    }
  }
  const auto prefix_len = static_cast<const int32_t*>(param.prefix_lens)[row];
  static_cast<int32_t*>(param.topk_indices)[linear] =
      token >= 0 && token < prefix_len ? token : -1;
}

template <typename KeyT, typename IndicesT, uint32_t kPageSize,
          bool kUsePDL>
struct NVFP4IndexerQuantKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  static constexpr auto store_kernel =
      fused_store_indexer_cache_nvfp4<KeyT, IndicesT, kLogSize, kUsePDL>;
  static constexpr auto q_kernel = quantize_indexer_q_nvfp4<KeyT>;
  static constexpr auto dequant_kernel = dequantize_indexer_nvfp4<false>;
  static constexpr auto dequant_branchless_kernel = dequantize_indexer_nvfp4<true>;
  static constexpr auto mean_pool_kernel =
      hisa_mean_pool_indexer_cache_nvfp4<IndicesT, kPageSize>;
  // iter3 vector 2: TMA-based mean_pool (cp.async.bulk per page).
  static constexpr auto mean_pool_tma_kernel =
      hisa_mean_pool_tma_indexer_cache_nvfp4<IndicesT, kPageSize>;
  // iter4 tertiary: predecoded-scale mean_pool (fp32 scale table).
  static constexpr auto mean_pool_predecode_kernel =
      hisa_mean_pool_predecode_indexer_cache_nvfp4<IndicesT, kPageSize>;
  // iter5 SECONDARY: mean_pool with transposed scales + 2-iter FMA pair.
  static constexpr auto mean_pool_predecode_fma2_kernel =
      hisa_mean_pool_predecode_fma2_indexer_cache_nvfp4<IndicesT, kPageSize>;
  static constexpr auto candidate_score_kernel =
      hisa_candidate_score_indexer_cache_nvfp4<IndicesT, kPageSize>;
  // iter2 vector 2: tile-N candidate_score with kTileN=8 candidates per block.
  static constexpr auto candidate_score_tilen_kernel =
      hisa_candidate_score_tilen_indexer_cache_nvfp4<IndicesT, kPageSize, 8>;
  // iter2 vector 2 experimental: kTileN=16. Halves the Q HBM traffic compared
  // to kTileN=8 at the cost of doubling the per-CTA stage-A K decode and the
  // stage-B inner-tile loop. Requires the kernel template to loop stages A/C
  // when kTileN > kMaxWarps; the current implementation supports this.
  static constexpr auto candidate_score_tilen16_kernel =
      hisa_candidate_score_tilen_indexer_cache_nvfp4<IndicesT, kPageSize, 16>;
  // iter3 vector 4: kTileN=32 extension of the iter2 template. Quarters
  // the Q HBM traffic vs kTileN=8 at the cost of 4x the per-CTA stage-A
  // K decode + 4x the stage-B inner-tile loop. SMEM per CTA: smem_k 16
  // KB + smem_head_dot 16 KB + small = 32 KB (under the default 48 KB
  // cap). Best when launch + per-CTA fixed overhead dominates the iter2
  // tile-N kernel's wall time (typical at very-large batch).
  static constexpr auto candidate_score_tilen32_kernel =
      hisa_candidate_score_tilen_indexer_cache_nvfp4<IndicesT, kPageSize, 32>;
  // iter5 PRIMARY: WMMA candidate_score (mma.m16n8k8 fp32 Stage B). One
  // instantiation per kMaxHeads ceiling (8 / 16 / 64) so the SMEM tables
  // stay sized to the actual head count at production (TP-shard fallback /
  // SMC-SD draft / production model). All three share kTileN=32.
  static constexpr auto candidate_score_tilen32_wmma_kernel =
      hisa_candidate_score_tilen_wmma_indexer_cache_nvfp4<
          IndicesT, kPageSize, 32, 64>;
  static constexpr auto candidate_score_tilen32_wmma_h16_kernel =
      hisa_candidate_score_tilen_wmma_indexer_cache_nvfp4<
          IndicesT, kPageSize, 32, 16>;
  static constexpr auto candidate_score_tilen32_wmma_h8_kernel =
      hisa_candidate_score_tilen_wmma_indexer_cache_nvfp4<
          IndicesT, kPageSize, 32, 8>;
  // iter3 vector 1: persistent-block candidate_score. The kTileN=8 variant
  // is the default production iter3 path; kTileN=16 is exposed as an
  // opt-in to A/B against the iter2 kTileN=16 baseline.
  static constexpr auto candidate_score_persistent_kernel =
      hisa_candidate_score_persistent_indexer_cache_nvfp4<
          IndicesT, kPageSize, 8>;
  static constexpr auto candidate_score_persistent_tilen16_kernel =
      hisa_candidate_score_persistent_indexer_cache_nvfp4<
          IndicesT, kPageSize, 16>;
  // iter4 PRIMARY: persistent-block candidate_score with 2-stage cp.async
  // K-row prefetch (kTileN=8 production variant).
  static constexpr auto candidate_score_persistent_kprefetch_kernel =
      hisa_candidate_score_persistent_kprefetch_indexer_cache_nvfp4<
          IndicesT, kPageSize, 8>;
  static constexpr auto block_score_kernel =
      hisa_block_score_indexer_cache_nvfp4;
  static constexpr auto block_topk_kernel =
      hisa_block_topk_indexer_cache_nvfp4;
  static constexpr auto block_topk_map_all_kernel =
      hisa_block_topk_map_all_indexer_cache_nvfp4;
  static constexpr auto candidate_pages_kernel =
      hisa_candidate_pages_indexer_cache_nvfp4<IndicesT>;
  static constexpr auto candidate_mask_kernel =
      hisa_mask_candidate_logits_indexer_cache_nvfp4;
  static constexpr auto mask_logits_kernel =
      hisa_mask_logits_indexer_cache_nvfp4;
  static constexpr auto map_topk_kernel =
      hisa_map_topk_indices_indexer_cache_nvfp4;
  static constexpr auto map_candidates_kernel =
      hisa_map_candidate_indices_indexer_cache_nvfp4;
  static constexpr auto fused_mask_topk_map_kernel =
      hisa_fused_mask_topk_map_indexer_cache_nvfp4;
#if defined(SGLANG_ENABLE_HISA_SELECTOR_MEGAKERNEL)
  static constexpr auto selector_megakernel_kernel =
      hisa_selector_megakernel_indexer_cache_nvfp4<
          IndicesT, kPageSize, HISASelectorGemmF32, kHISASelectorTileN>;
  static constexpr auto selector_megakernel_wide_kernel =
      hisa_selector_megakernel_indexer_cache_nvfp4<
          IndicesT, kPageSize, HISASelectorWideGemmF32, kHISASelectorWideTileN>;
  static constexpr auto selector_parallel_select_kernel =
      hisa_selector_parallel_select_blocks_indexer_cache_nvfp4<
          HISASelectorGemmF32>;
  static constexpr auto selector_parallel_score_kernel =
      hisa_selector_parallel_score_candidates_indexer_cache_nvfp4<
          IndicesT, kPageSize, HISASelectorGemmF32>;
  static constexpr auto selector_cluster_fused_kernel =
      hisa_selector_cluster_fused_indexer_cache_nvfp4<
          IndicesT, kPageSize, HISASelectorGemmF32>;
  static constexpr auto candidate_keys_topk_map_kernel =
      hisa_candidate_keys_topk_map_indexer_cache_nvfp4;
  static constexpr auto deepgemm_candidate_logits_kernel =
      deep_gemm::sm100_fp4_paged_mqa_logits<
          1,
          kHISASelectorHeads,
          kIndexerHeadDim,
          kPageSize,
          true,
          false,
          3,
          10,
          256,
          128,
          256,
          float>;
	  static constexpr auto deepgemm_candidate_keys_kernel =
	      hisa_sm100_fp4_paged_mqa_candidate_keys<
	          kHISASelectorHeads,
	          kIndexerHeadDim,
	          kPageSize,
          3,
		          10,
	          256,
	          128,
	          256>;
	  static constexpr auto deepgemm_candidate_keys_row_split_kernel =
	      hisa_sm100_fp4_paged_mqa_candidate_keys_row_split<
	          kHISASelectorHeads,
	          kIndexerHeadDim,
	          kPageSize,
	          3,
	          10,
	          256,
	          128,
	          256>;
#endif

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogSize == kPageSize);

  static void store_index_k(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView indices) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, kIndexerHeadDim})
        .with_dtype<KeyT>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({N}).with_dtype<IndicesT>().with_device(device_).verify(indices);
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = NVFP4IndexerStoreParam{
        .input = input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = num_tokens,
    };
    constexpr auto kBlockSize = 256;
    const auto num_blocks = div_ceil(num_tokens * 32, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())
        .enable_pdl(kUsePDL)(store_kernel, params);
  }

  static void quantize_q(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView values,
      tvm::ffi::TensorView scales) {
    using namespace host;

    auto N = SymbolicSize{"num_rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, kIndexerHeadDim})
        .with_dtype<KeyT>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({N, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(values);
    TensorMatcher({N}).with_dtype<int32_t>().with_device(device_).verify(scales);
    const auto num_rows = static_cast<uint32_t>(N.unwrap());
    const auto params = NVFP4IndexerQuantParam{
        .input = input.data_ptr(),
        .values = values.data_ptr(),
        .scales = scales.data_ptr(),
        .num_rows = num_rows,
    };
    constexpr auto kBlockSize = 256;
    const auto num_blocks = div_ceil(num_rows * 16, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())(q_kernel, params);
  }

  static void dequantize(
      tvm::ffi::TensorView values,
      tvm::ffi::TensorView scales,
      tvm::ffi::TensorView output) {
    using namespace host;

    auto N = SymbolicSize{"num_rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(values);
    TensorMatcher({N}).with_dtype<int32_t>().with_device(device_).verify(scales);
    TensorMatcher({N, kIndexerHeadDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(output);
    const auto num_rows = static_cast<uint32_t>(N.unwrap());
    const auto params = NVFP4IndexerDequantParam{
        .values = values.data_ptr(),
        .scales = scales.data_ptr(),
        .output = output.data_ptr(),
        .num_rows = num_rows,
    };
    constexpr auto kBlockSize = 256;
    const auto num_blocks = div_ceil(num_rows * 8, kBlockSize);
    if (num_rows >= 262144) {
      LaunchKernel(num_blocks, kBlockSize, device_.unwrap())(
          dequant_branchless_kernel, params);
    } else {
      LaunchKernel(num_blocks, kBlockSize, device_.unwrap())(dequant_kernel, params);
    }
  }

  static void hisa_mean_pool(
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView reps) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto P = SymbolicSize{"page_table_stride"};
    auto MB = SymbolicSize{"max_blocks"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({B, MB, kIndexerHeadDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(reps);
    const auto params = NVFP4HISAMeanPoolParam{
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .reps = reps.data_ptr(),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .max_blocks = static_cast<uint32_t>(MB.unwrap()),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
    };
    LaunchKernel(dim3(params.max_blocks, params.batch_size), 128, device_.unwrap())(
        mean_pool_kernel, params);
  }

  // iter3 vector 2: TMA-based mean_pool launcher. Same FFI as the
  // iter2 mean_pool launcher; only the underlying kernel is swapped to
  // the cp.async.bulk-based variant.
  static void hisa_mean_pool_tma(
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView reps) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto P = SymbolicSize{"page_table_stride"};
    auto MB = SymbolicSize{"max_blocks"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({B, MB, kIndexerHeadDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(reps);
    const auto params = NVFP4HISAMeanPoolParam{
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .reps = reps.data_ptr(),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .max_blocks = static_cast<uint32_t>(MB.unwrap()),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
    };
    LaunchKernel(
        dim3(params.max_blocks, params.batch_size), 128, device_.unwrap())(
        mean_pool_tma_kernel, params);
  }

  // iter4 tertiary: predecode-scale mean_pool launcher. Same FFI as the
  // iter2 mean_pool launcher; underlying kernel pre-decodes scales into
  // an fp32 SMEM table and uses a branchless inner loop.
  static void hisa_mean_pool_predecode(
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView reps) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto P = SymbolicSize{"page_table_stride"};
    auto MB = SymbolicSize{"max_blocks"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({B, MB, kIndexerHeadDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(reps);
    const auto params = NVFP4HISAMeanPoolParam{
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .reps = reps.data_ptr(),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .max_blocks = static_cast<uint32_t>(MB.unwrap()),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
    };
    LaunchKernel(
        dim3(params.max_blocks, params.batch_size), 128, device_.unwrap())(
        mean_pool_predecode_kernel, params);
  }

  // iter5 SECONDARY: mean_pool with transposed scales table + 2-iter FMA
  // pair. Same FFI as the iter4 predecode launcher; underlying kernel
  // restructures the scales SMEM layout to [scale_group][token-local] so
  // the inner loop can LDS.b64 two scales per issue and accumulate into
  // two fp32 dependency chains in parallel.
  static void hisa_mean_pool_predecode_fma2(
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView reps) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto P = SymbolicSize{"page_table_stride"};
    auto MB = SymbolicSize{"max_blocks"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({B, MB, kIndexerHeadDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(reps);
    const auto params = NVFP4HISAMeanPoolParam{
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .reps = reps.data_ptr(),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .max_blocks = static_cast<uint32_t>(MB.unwrap()),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
    };
    LaunchKernel(
        dim3(params.max_blocks, params.batch_size), 128, device_.unwrap())(
        mean_pool_predecode_fma2_kernel, params);
  }

  static void hisa_candidate_score(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView candidate_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto P = SymbolicSize{"page_table_stride"};
    auto BT = SymbolicSize{"block_topk"};
    auto B = SymbolicSize{"batch_size"};
    auto CL = SymbolicSize{"candidate_len"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({Q, CL}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({Q, CL})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(candidate_indices);
    if (static_cast<uint32_t>(CL.unwrap()) !=
        static_cast<uint32_t>(BT.unwrap()) * 128) {
      throw std::runtime_error("candidate_len must equal block_topk * 128");
    }
    const auto params = NVFP4HISACandidateScoreParam{
        .q_values = q_values.data_ptr(),
        .q_scales = q_scales.data_ptr(),
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .weights = weights.data_ptr(),
        .top_blocks = top_blocks.data_ptr(),
        .token_to_batch_idx = token_to_batch_idx.data_ptr(),
        .logits = logits.data_ptr(),
        .candidate_indices = candidate_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .n_heads = static_cast<uint32_t>(H.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
    };
    // iter1 fused-head rewrite collapses the legacy z-axis head_groups: a
    // single block per (cand, row) handles all heads, eliminating 8x K HBM
    // reload and 8x scheduler overhead. The kernel's internal warp dispatch
    // handles ceil(n_heads / 8) heads per warp.
    LaunchKernel(
        dim3(params.block_topk * 128, params.q_rows),
        256,
        device_.unwrap())(
        candidate_score_kernel, params);
  }

  // iter2 vector 2: tile-N candidate_score launcher.
  // Same FFI surface as hisa_candidate_score but launches with kTileN=8
  // candidates per block. Grid X axis shrinks 8x; per-block Q decode is
  // amortized across the 8 tile slots so the total Q HBM traffic for a row
  // drops 8x. K decode is still per-candidate but happens cooperatively
  // across warps in one __syncthreads window.
  static void hisa_candidate_score_tilen(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView candidate_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto P = SymbolicSize{"page_table_stride"};
    auto BT = SymbolicSize{"block_topk"};
    auto B = SymbolicSize{"batch_size"};
    auto CL = SymbolicSize{"candidate_len"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({Q, CL}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({Q, CL})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(candidate_indices);
    if (static_cast<uint32_t>(CL.unwrap()) !=
        static_cast<uint32_t>(BT.unwrap()) * 128) {
      throw std::runtime_error("candidate_len must equal block_topk * 128");
    }
    const auto params = NVFP4HISACandidateScoreParam{
        .q_values = q_values.data_ptr(),
        .q_scales = q_scales.data_ptr(),
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .weights = weights.data_ptr(),
        .top_blocks = top_blocks.data_ptr(),
        .token_to_batch_idx = token_to_batch_idx.data_ptr(),
        .logits = logits.data_ptr(),
        .candidate_indices = candidate_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .n_heads = static_cast<uint32_t>(H.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
    };
    constexpr uint32_t kTileN = 8;
    const uint32_t candidate_len = params.block_topk * 128;
    const uint32_t tile_blocks = (candidate_len + kTileN - 1) / kTileN;
    LaunchKernel(
        dim3(tile_blocks, params.q_rows),
        256,
        device_.unwrap())(
        candidate_score_tilen_kernel, params);
  }

  // iter2 vector 2 experimental: kTileN=16 variant. Same FFI as the kTileN=8
  // launcher; exists as an opt-in path that further halves Q HBM traffic at
  // the cost of looping stage A/C twice per CTA. Gated by env var so iter2's
  // default kTileN=8 remains the production path until kTileN=16 is shown
  // strictly faster across the production shape grid.
  static void hisa_candidate_score_tilen16(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView candidate_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto P = SymbolicSize{"page_table_stride"};
    auto BT = SymbolicSize{"block_topk"};
    auto B = SymbolicSize{"batch_size"};
    auto CL = SymbolicSize{"candidate_len"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({Q, CL}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({Q, CL})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(candidate_indices);
    if (static_cast<uint32_t>(CL.unwrap()) !=
        static_cast<uint32_t>(BT.unwrap()) * 128) {
      throw std::runtime_error("candidate_len must equal block_topk * 128");
    }
    const auto params = NVFP4HISACandidateScoreParam{
        .q_values = q_values.data_ptr(),
        .q_scales = q_scales.data_ptr(),
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .weights = weights.data_ptr(),
        .top_blocks = top_blocks.data_ptr(),
        .token_to_batch_idx = token_to_batch_idx.data_ptr(),
        .logits = logits.data_ptr(),
        .candidate_indices = candidate_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .n_heads = static_cast<uint32_t>(H.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
    };
    constexpr uint32_t kTileN = 16;
    const uint32_t candidate_len = params.block_topk * 128;
    const uint32_t tile_blocks = (candidate_len + kTileN - 1) / kTileN;
    LaunchKernel(
        dim3(tile_blocks, params.q_rows),
        256,
        device_.unwrap())(
        candidate_score_tilen16_kernel, params);
  }

  // iter3 vector 4: kTileN=32 candidate_score launcher. Same FFI as the
  // iter2 tile-N launchers but the underlying kernel template is
  // instantiated at kTileN=32. Grid shrinks 32x vs iter1
  // per-candidate (and 4x vs iter2 kTileN=8). Per-CTA SMEM grows to
  // ~32 KB.
  static void hisa_candidate_score_tilen32(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView candidate_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto P = SymbolicSize{"page_table_stride"};
    auto BT = SymbolicSize{"block_topk"};
    auto B = SymbolicSize{"batch_size"};
    auto CL = SymbolicSize{"candidate_len"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({Q, CL}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({Q, CL})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(candidate_indices);
    if (static_cast<uint32_t>(CL.unwrap()) !=
        static_cast<uint32_t>(BT.unwrap()) * 128) {
      throw std::runtime_error("candidate_len must equal block_topk * 128");
    }
    const auto params = NVFP4HISACandidateScoreParam{
        .q_values = q_values.data_ptr(),
        .q_scales = q_scales.data_ptr(),
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .weights = weights.data_ptr(),
        .top_blocks = top_blocks.data_ptr(),
        .token_to_batch_idx = token_to_batch_idx.data_ptr(),
        .logits = logits.data_ptr(),
        .candidate_indices = candidate_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .n_heads = static_cast<uint32_t>(H.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
    };
    constexpr uint32_t kTileN32 = 32;
    const uint32_t candidate_len32 = params.block_topk * 128;
    const uint32_t tile_blocks32 =
        (candidate_len32 + kTileN32 - 1) / kTileN32;
    LaunchKernel(
        dim3(tile_blocks32, params.q_rows),
        256,
        device_.unwrap())(
        candidate_score_tilen32_kernel, params);
  }

  // iter5 PRIMARY: WMMA candidate_score launcher (kTileN=32, mma.m16n8k8
  // fp32 Stage B). Same FFI as the iter3 tilen32 launcher; routes to the
  // n_heads-templated kernel instantiation (kMaxHeads in {8, 16, 64}) so
  // the smem_q_h / smem_head_dot tables stay sized to the production
  // grid. n_heads > 64 throws (caller must fall back to scalar tilen32).
  static void hisa_candidate_score_tilen32_wmma(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView candidate_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto P = SymbolicSize{"page_table_stride"};
    auto BT = SymbolicSize{"block_topk"};
    auto B = SymbolicSize{"batch_size"};
    auto CL = SymbolicSize{"candidate_len"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({Q, CL}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({Q, CL})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(candidate_indices);
    if (static_cast<uint32_t>(CL.unwrap()) !=
        static_cast<uint32_t>(BT.unwrap()) * 128) {
      throw std::runtime_error("candidate_len must equal block_topk * 128");
    }
    const auto params = NVFP4HISACandidateScoreParam{
        .q_values = q_values.data_ptr(),
        .q_scales = q_scales.data_ptr(),
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .weights = weights.data_ptr(),
        .top_blocks = top_blocks.data_ptr(),
        .token_to_batch_idx = token_to_batch_idx.data_ptr(),
        .logits = logits.data_ptr(),
        .candidate_indices = candidate_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .n_heads = static_cast<uint32_t>(H.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
    };
    constexpr uint32_t kTileN32 = 32;
    const uint32_t candidate_len32 = params.block_topk * 128;
    const uint32_t tile_blocks32 =
        (candidate_len32 + kTileN32 - 1) / kTileN32;
    if (params.n_heads > 64) {
      throw std::runtime_error(
          "hisa_candidate_score_tilen32_wmma: n_heads > 64 not supported "
          "(production caps at 64; route to scalar tilen32 instead).");
    }
    const auto grid = dim3(tile_blocks32, params.q_rows);
    if (params.n_heads <= 8) {
      LaunchKernel(grid, 256, device_.unwrap())(
          candidate_score_tilen32_wmma_h8_kernel, params);
    } else if (params.n_heads <= 16) {
      LaunchKernel(grid, 256, device_.unwrap())(
          candidate_score_tilen32_wmma_h16_kernel, params);
    } else {
      LaunchKernel(grid, 256, device_.unwrap())(
          candidate_score_tilen32_wmma_kernel, params);
    }
  }

  // iter3 vector 1: persistent-block candidate_score launcher (kTileN=8).
  //
  // Grid shape: dim3(splits_per_row, q_rows). splits_per_row is chosen to
  // keep each CTA's per-tile work in a regime where Q amortization
  // dominates but the active-CTA count stays high enough to fill the SM
  // pipeline. The target is ~tiles_per_split=32 (so a CTA's Q is reused
  // across 32 * kTileN = 256 candidates), capped so that
  // (splits_per_row * q_rows) stays in [num_sms, 6 * num_sms] for B200
  // (148 SMs). For small q_rows we increase splits_per_row to keep ~600
  // active CTAs; for large q_rows we shrink it so per-CTA work is still
  // amortized over many tiles.
  //
  // The target_active_ctas heuristic resolves to:
  //   * q_rows=32  / tile_blocks=64  → splits=2 → 64 active CTAs, 32 tiles/CTA
  //   * q_rows=32  / tile_blocks=128 → splits=4 → 128 active CTAs, 32 tiles/CTA
  //   * q_rows=64  / tile_blocks=128 → splits=4 → 256 active CTAs, 32 tiles/CTA
  //   * q_rows=128 / tile_blocks=128 → splits=2 → 256 active CTAs, 64 tiles/CTA
  // (For tile_blocks <= 16 we fall back to splits=1 so the kernel
  // degenerates to one CTA per row with the full tile range.)
  static void hisa_candidate_score_persistent(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView candidate_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto P = SymbolicSize{"page_table_stride"};
    auto BT = SymbolicSize{"block_topk"};
    auto B = SymbolicSize{"batch_size"};
    auto CL = SymbolicSize{"candidate_len"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({Q, CL}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({Q, CL})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(candidate_indices);
    if (static_cast<uint32_t>(CL.unwrap()) !=
        static_cast<uint32_t>(BT.unwrap()) * 128) {
      throw std::runtime_error("candidate_len must equal block_topk * 128");
    }
    if (static_cast<uint32_t>(H.unwrap()) > 64) {
      throw std::runtime_error(
          "hisa_candidate_score_persistent: n_heads > 64 is not "
          "supported. The kernel's smem_head_dot is sized for "
          "kMaxHeads=64. Update kMaxHeads in the kernel and re-validate "
          "SMEM occupancy if larger head counts are needed.");
    }
    const auto params = NVFP4HISACandidateScoreParam{
        .q_values = q_values.data_ptr(),
        .q_scales = q_scales.data_ptr(),
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .weights = weights.data_ptr(),
        .top_blocks = top_blocks.data_ptr(),
        .token_to_batch_idx = token_to_batch_idx.data_ptr(),
        .logits = logits.data_ptr(),
        .candidate_indices = candidate_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .n_heads = static_cast<uint32_t>(H.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
    };
    constexpr uint32_t kTileN = 8;
    const uint32_t candidate_len = params.block_topk * 128;
    const uint32_t total_tiles = (candidate_len + kTileN - 1) / kTileN;
    // Heuristic configurable via env-style override. Defaults: aim for
    // kTilesPerCTAOverride tile blocks per CTA (default 32). The previous
    // total-work heuristic mis-tuned at large q_rows (collapsing too many
    // tiles into too few CTAs); a per-CTA cap is the safer knob.
    //
    // splits_per_row = ceil(total_tiles / tiles_per_cta).
    constexpr uint32_t kTilesPerCTA = 32;
    uint32_t splits_per_row =
        (total_tiles + kTilesPerCTA - 1) / kTilesPerCTA;
    if (splits_per_row < 1) splits_per_row = 1;
    if (splits_per_row > total_tiles) splits_per_row = total_tiles;
    LaunchKernel(
        dim3(splits_per_row, params.q_rows),
        256,
        device_.unwrap())(
        candidate_score_persistent_kernel,
        params,
        splits_per_row,
        total_tiles);
  }

  // iter3 vector 1 experimental: persistent-block kTileN=16 variant.
  // Same FFI as the kTileN=8 launcher; underlying kernel binds kTileN=16.
  // Exists for A/B testing against the iter2 kTileN=16 variant.
  static void hisa_candidate_score_persistent_tilen16(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView candidate_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto P = SymbolicSize{"page_table_stride"};
    auto BT = SymbolicSize{"block_topk"};
    auto B = SymbolicSize{"batch_size"};
    auto CL = SymbolicSize{"candidate_len"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({Q, CL}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({Q, CL})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(candidate_indices);
    if (static_cast<uint32_t>(CL.unwrap()) !=
        static_cast<uint32_t>(BT.unwrap()) * 128) {
      throw std::runtime_error("candidate_len must equal block_topk * 128");
    }
    const auto params = NVFP4HISACandidateScoreParam{
        .q_values = q_values.data_ptr(),
        .q_scales = q_scales.data_ptr(),
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .weights = weights.data_ptr(),
        .top_blocks = top_blocks.data_ptr(),
        .token_to_batch_idx = token_to_batch_idx.data_ptr(),
        .logits = logits.data_ptr(),
        .candidate_indices = candidate_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .n_heads = static_cast<uint32_t>(H.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
    };
    constexpr uint32_t kTileN_persist16 = 16;
    const uint32_t candidate_len_p16 = params.block_topk * 128;
    const uint32_t total_tiles_p16 =
        (candidate_len_p16 + kTileN_persist16 - 1) / kTileN_persist16;
    constexpr uint32_t kTargetTilesPerCTA_p16 = 16;
    uint32_t splits_per_row_p16 =
        (total_tiles_p16 + kTargetTilesPerCTA_p16 - 1) /
        kTargetTilesPerCTA_p16;
    if (splits_per_row_p16 < 1) splits_per_row_p16 = 1;
    if (splits_per_row_p16 > total_tiles_p16) {
      splits_per_row_p16 = total_tiles_p16;
    }
    if (params.n_heads > 64) {
      throw std::runtime_error(
          "hisa_candidate_score_persistent_tilen16: n_heads > 64 not "
          "supported (smem_head_dot sized for kMaxHeads=64).");
    }
    LaunchKernel(
        dim3(splits_per_row_p16, params.q_rows),
        256,
        device_.unwrap())(
        candidate_score_persistent_tilen16_kernel,
        params,
        splits_per_row_p16,
        total_tiles_p16);
  }

  // iter4 PRIMARY: persistent-block candidate_score + cp.async K-row
  // prefetch launcher. Same FFI and same splits_per_row heuristic as the
  // iter3 v1 persistent launcher (kTileN=8 production fast path), but
  // dispatches into the kprefetch kernel which double-buffers raw NVFP4 K
  // bytes via cp.async to overlap HBM K decode with Stage B/C compute.
  static void hisa_candidate_score_persistent_kprefetch(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView candidate_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto P = SymbolicSize{"page_table_stride"};
    auto BT = SymbolicSize{"block_topk"};
    auto B = SymbolicSize{"batch_size"};
    auto CL = SymbolicSize{"candidate_len"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({Q, CL}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({Q, CL})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(candidate_indices);
    if (static_cast<uint32_t>(CL.unwrap()) !=
        static_cast<uint32_t>(BT.unwrap()) * 128) {
      throw std::runtime_error("candidate_len must equal block_topk * 128");
    }
    if (static_cast<uint32_t>(H.unwrap()) > 64) {
      throw std::runtime_error(
          "hisa_candidate_score_persistent_kprefetch: n_heads > 64 not "
          "supported (smem_head_dot sized for kMaxHeads=64).");
    }
    const auto params = NVFP4HISACandidateScoreParam{
        .q_values = q_values.data_ptr(),
        .q_scales = q_scales.data_ptr(),
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .weights = weights.data_ptr(),
        .top_blocks = top_blocks.data_ptr(),
        .token_to_batch_idx = token_to_batch_idx.data_ptr(),
        .logits = logits.data_ptr(),
        .candidate_indices = candidate_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .n_heads = static_cast<uint32_t>(H.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
    };
    constexpr uint32_t kTileN_kprefetch = 8;
    const uint32_t candidate_len_kp = params.block_topk * 128;
    const uint32_t total_tiles_kp =
        (candidate_len_kp + kTileN_kprefetch - 1) / kTileN_kprefetch;
    // Reuse the iter3 v1 splits_per_row heuristic: ~32 tiles per CTA so
    // Q amortization dominates but active-CTA count stays high. iter4's
    // K prefetch is per-tile so the same partition shape applies.
    constexpr uint32_t kTargetTilesPerCTA_kp = 32;
    uint32_t splits_per_row_kp =
        (total_tiles_kp + kTargetTilesPerCTA_kp - 1) / kTargetTilesPerCTA_kp;
    if (splits_per_row_kp < 1) splits_per_row_kp = 1;
    if (splits_per_row_kp > total_tiles_kp) splits_per_row_kp = total_tiles_kp;
    LaunchKernel(
        dim3(splits_per_row_kp, params.q_rows),
        256,
        device_.unwrap())(
        candidate_score_persistent_kprefetch_kernel,
        params,
        splits_per_row_kp,
        total_tiles_kp);
  }

  static void hisa_block_score(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView reps,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView block_scores) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto B = SymbolicSize{"batch_size"};
    auto MB = SymbolicSize{"max_blocks"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({B, MB, kIndexerHeadDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(reps);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({Q, MB}).with_dtype<float>().with_device(device_).verify(block_scores);
    const auto params = NVFP4HISABlockScoreParam{
        .q_values = q_values.data_ptr(),
        .q_scales = q_scales.data_ptr(),
        .reps = reps.data_ptr(),
        .weights = weights.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .token_to_batch_idx = token_to_batch_idx.data_ptr(),
        .block_scores = block_scores.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .n_heads = static_cast<uint32_t>(H.unwrap()),
        .max_blocks = static_cast<uint32_t>(MB.unwrap()),
    };
    LaunchKernel(dim3(params.max_blocks, params.q_rows), 256, device_.unwrap())(
        block_score_kernel, params);
  }

  static void hisa_block_topk(
      tvm::ffi::TensorView block_scores,
      tvm::ffi::TensorView block_counts,
      tvm::ffi::TensorView block_topk_counts,
      tvm::ffi::TensorView top_blocks) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto MB = SymbolicSize{"max_blocks"};
    auto BT = SymbolicSize{"block_topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, MB}).with_dtype<float>().with_device(device_).verify(block_scores);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(block_counts);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(block_topk_counts);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    const auto params = NVFP4HISABlockTopKParam{
        .block_scores = block_scores.data_ptr(),
        .block_counts = block_counts.data_ptr(),
        .block_topk_counts = block_topk_counts.data_ptr(),
        .top_blocks = top_blocks.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .max_blocks = static_cast<uint32_t>(MB.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
    };
    uint32_t threads = 32;
    const auto required =
        params.max_blocks > params.block_topk ? params.max_blocks : params.block_topk;
    while (threads < required && threads < 1024) {
      threads <<= 1;
    }
    const auto smem_bytes = threads * (sizeof(float) + sizeof(int32_t)) +
                            params.block_topk * sizeof(int32_t);
    LaunchKernel(params.q_rows, threads, device_.unwrap(), smem_bytes)(
        block_topk_kernel, params);
  }

  static void hisa_block_topk_map_all(
      tvm::ffi::TensorView block_scores,
      tvm::ffi::TensorView block_counts,
      tvm::ffi::TensorView block_topk_counts,
      tvm::ffi::TensorView prefix_lens,
      tvm::ffi::TensorView topk_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto MB = SymbolicSize{"max_blocks"};
    auto K = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, MB}).with_dtype<float>().with_device(device_).verify(block_scores);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(block_counts);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(block_topk_counts);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(prefix_lens);
    TensorMatcher({Q, K}).with_dtype<int32_t>().with_device(device_).verify(topk_indices);
    const auto topk = static_cast<uint32_t>(K.unwrap());
    if (topk % 128 != 0) {
      throw std::runtime_error("hisa_block_topk_map_all requires topk divisible by 128");
    }
    const auto params = NVFP4HISABlockTopKMapAllParam{
        .block_scores = block_scores.data_ptr(),
        .block_counts = block_counts.data_ptr(),
        .block_topk_counts = block_topk_counts.data_ptr(),
        .prefix_lens = prefix_lens.data_ptr(),
        .topk_indices = topk_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .max_blocks = static_cast<uint32_t>(MB.unwrap()),
        .block_topk = topk / 128,
        .topk = topk,
    };
    uint32_t threads = 32;
    const auto required =
        params.max_blocks > params.block_topk ? params.max_blocks : params.block_topk;
    while (threads < required && threads < 1024) {
      threads <<= 1;
    }
    const auto smem_bytes = threads * (sizeof(float) + sizeof(int32_t)) +
                            params.block_topk * sizeof(int32_t);
    LaunchKernel(params.q_rows, threads, device_.unwrap(), smem_bytes)(
        block_topk_map_all_kernel, params);
  }

  static void hisa_candidate_pages(
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView candidate_page_table) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto BT = SymbolicSize{"block_topk"};
    auto CP = SymbolicSize{"candidate_pages"};
    auto B = SymbolicSize{"batch_size"};
    auto P = SymbolicSize{"page_table_stride"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({Q, CP})
        .with_dtype<IndicesT>()
        .with_device(device_)
        .verify(candidate_page_table);
    if (static_cast<uint32_t>(CP.unwrap()) != static_cast<uint32_t>(BT.unwrap()) * 2) {
      throw std::runtime_error("candidate_page_table width must equal block_topk * 2");
    }
    const auto params = NVFP4HISACandidatePagesParam{
        .top_blocks = top_blocks.data_ptr(),
        .page_table = page_table.data_ptr(),
        .token_to_batch_idx = token_to_batch_idx.data_ptr(),
        .candidate_page_table = candidate_page_table.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
    };
    constexpr uint32_t kThreads = 256;
    LaunchKernel(params.q_rows, kThreads, device_.unwrap())(candidate_pages_kernel, params);
  }

  static void hisa_mask_candidate_logits(
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView prefix_lens,
      tvm::ffi::TensorView candidate_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto CL = SymbolicSize{"candidate_len"};
    auto BT = SymbolicSize{"block_topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, CL}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(prefix_lens);
    TensorMatcher({Q, CL})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(candidate_indices);
    if (static_cast<uint32_t>(CL.unwrap()) != static_cast<uint32_t>(BT.unwrap()) * 128) {
      throw std::runtime_error("candidate_len must equal block_topk * 128");
    }
    const auto params = NVFP4HISACandidateMaskParam{
        .logits = logits.data_ptr(),
        .top_blocks = top_blocks.data_ptr(),
        .prefix_lens = prefix_lens.data_ptr(),
        .candidate_indices = candidate_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
        .candidate_len = static_cast<uint32_t>(CL.unwrap()),
    };
    constexpr uint32_t kThreads = 256;
    const auto total = params.q_rows * params.candidate_len;
    LaunchKernel(div_ceil(total, kThreads), kThreads, device_.unwrap())(
        candidate_mask_kernel, params);
  }

  static void hisa_mask_logits(
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView prefix_lens) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto CL = SymbolicSize{"candidate_len"};
    auto BT = SymbolicSize{"block_topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, CL}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(prefix_lens);
    if (static_cast<uint32_t>(CL.unwrap()) != static_cast<uint32_t>(BT.unwrap()) * 128) {
      throw std::runtime_error("candidate_len must equal block_topk * 128");
    }
    const auto params = NVFP4HISACandidateMaskParam{
        .logits = logits.data_ptr(),
        .top_blocks = top_blocks.data_ptr(),
        .prefix_lens = prefix_lens.data_ptr(),
        .candidate_indices = nullptr,
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
        .candidate_len = static_cast<uint32_t>(CL.unwrap()),
    };
    constexpr uint32_t kThreads = 256;
    const auto total = params.q_rows * params.candidate_len;
    LaunchKernel(div_ceil(total, kThreads), kThreads, device_.unwrap())(
        mask_logits_kernel, params);
  }

  static void hisa_map_topk_indices(
      tvm::ffi::TensorView topk_positions,
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView prefix_lens,
      tvm::ffi::TensorView topk_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto K = SymbolicSize{"topk"};
    auto BT = SymbolicSize{"block_topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, K}).with_dtype<int64_t>().with_device(device_).verify(topk_positions);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(prefix_lens);
    TensorMatcher({Q, K}).with_dtype<int32_t>().with_device(device_).verify(topk_indices);
    const auto params = NVFP4HISAMapTopKParam{
        .topk_positions = topk_positions.data_ptr(),
        .top_blocks = top_blocks.data_ptr(),
        .prefix_lens = prefix_lens.data_ptr(),
        .topk_indices = topk_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .topk = static_cast<uint32_t>(K.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
    };
    constexpr uint32_t kThreads = 256;
    const auto total = params.q_rows * params.topk;
    LaunchKernel(div_ceil(total, kThreads), kThreads, device_.unwrap())(
        map_topk_kernel, params);
  }

  static void hisa_fused_mask_topk_map(
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView prefix_lens,
      tvm::ffi::TensorView topk_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto CL = SymbolicSize{"candidate_len"};
    auto BT = SymbolicSize{"block_topk"};
    auto K = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, CL}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(prefix_lens);
    TensorMatcher({Q, K}).with_dtype<int32_t>().with_device(device_).verify(topk_indices);
    if (static_cast<uint32_t>(CL.unwrap()) != static_cast<uint32_t>(BT.unwrap()) * 128) {
      throw std::runtime_error("candidate_len must equal block_topk * 128");
    }
    if (static_cast<uint32_t>(BT.unwrap()) > 256) {
      throw std::runtime_error("fused_mask_topk_map: block_topk must be <= 256");
    }
    const auto params = NVFP4HISAFusedMaskTopKMapParam{
        .logits = logits.data_ptr(),
        .top_blocks = top_blocks.data_ptr(),
        .prefix_lens = prefix_lens.data_ptr(),
        .topk_indices = topk_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .topk = static_cast<uint32_t>(K.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
        .candidate_len = static_cast<uint32_t>(CL.unwrap()),
    };
    constexpr uint32_t kThreads = 256;
    LaunchKernel(params.q_rows, kThreads, device_.unwrap())(
        fused_mask_topk_map_kernel, params);
  }

#if defined(SGLANG_ENABLE_HISA_SELECTOR_MEGAKERNEL)
  static void hisa_candidate_keys_topk_map(
      tvm::ffi::TensorView candidate_keys,
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView topk_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto CL = SymbolicSize{"candidate_len"};
    auto BT = SymbolicSize{"block_topk"};
    auto K = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, CL}).with_dtype<int32_t>().with_device(device_).verify(candidate_keys);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({Q, K}).with_dtype<int32_t>().with_device(device_).verify(topk_indices);
    if (static_cast<uint32_t>(CL.unwrap()) != static_cast<uint32_t>(BT.unwrap()) * 128) {
      throw std::runtime_error("candidate_keys_topk_map requires candidate_len == block_topk * 128");
    }
    const auto params = NVFP4HISACandidateKeysTopKMapParam{
        .candidate_keys = candidate_keys.data_ptr(),
        .top_blocks = top_blocks.data_ptr(),
        .topk_indices = topk_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .topk = static_cast<uint32_t>(K.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
        .candidate_len = static_cast<uint32_t>(CL.unwrap()),
    };
    constexpr uint32_t kThreads = 256;
    LaunchKernel(params.q_rows, kThreads, device_.unwrap())(
        candidate_keys_topk_map_kernel, params);
  }

  static void hisa_selector_megakernel(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView rep_values,
      tvm::ffi::TensorView rep_scales,
      tvm::ffi::TensorView block_counts,
      tvm::ffi::TensorView block_topk_counts,
      tvm::ffi::TensorView topk_indices,
      int64_t max_blocks,
      int64_t effective_block_topk) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto B = SymbolicSize{"batch_size"};
    auto P = SymbolicSize{"page_table_stride"};
    auto R = SymbolicSize{"rep_rows"};
    auto K = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({R, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(rep_values);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(rep_scales);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(block_counts);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(block_topk_counts);
    TensorMatcher({Q, K}).with_dtype<int32_t>().with_device(device_).verify(topk_indices);
    if (static_cast<uint32_t>(H.unwrap()) != kHISASelectorHeads) {
      throw std::runtime_error("hisa_selector_megakernel requires exactly 64 heads.");
    }
    if (max_blocks <= 0 || effective_block_topk <= 0) {
      throw std::runtime_error("hisa_selector_megakernel got non-positive selector dimensions.");
    }
    const auto expected_rep_rows =
        static_cast<int64_t>(B.unwrap()) * static_cast<int64_t>(max_blocks);
    if (static_cast<int64_t>(R.unwrap()) != expected_rep_rows) {
      throw std::runtime_error("hisa_selector_megakernel got mismatched block rep rows.");
    }
    uint32_t block_score_capacity = 1;
    while (block_score_capacity < static_cast<uint32_t>(max_blocks)) {
      block_score_capacity <<= 1;
    }
    const uint32_t candidate_len =
        static_cast<uint32_t>(effective_block_topk) * 128u;
    const uint32_t topk = static_cast<uint32_t>(K.unwrap());
    uint32_t candidate_capacity = 1;
    const uint32_t min_candidate_capacity = max(candidate_len, topk);
    while (candidate_capacity < min_candidate_capacity) {
      candidate_capacity <<= 1;
    }
    if (candidate_capacity > 8192) {
      throw std::runtime_error(
          "hisa_selector_megakernel currently supports candidate_capacity <= 8192.");
    }
    const auto params = NVFP4HISASelectorMegakernelParam{
        .q_values = q_values.data_ptr(),
        .q_scales = q_scales.data_ptr(),
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .weights = weights.data_ptr(),
        .token_to_batch_idx = token_to_batch_idx.data_ptr(),
        .rep_values = rep_values.data_ptr(),
        .rep_scales = rep_scales.data_ptr(),
        .block_counts = block_counts.data_ptr(),
        .block_topk_counts = block_topk_counts.data_ptr(),
        .topk_indices = topk_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .n_heads = static_cast<uint32_t>(H.unwrap()),
        .max_blocks = static_cast<uint32_t>(max_blocks),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
        .effective_block_topk = static_cast<uint32_t>(effective_block_topk),
        .topk = topk,
        .block_score_capacity = block_score_capacity,
        .candidate_capacity = candidate_capacity,
    };
    const bool use_wide_tile = false;
    const size_t gemm_smem_bytes =
        use_wide_tile
            ? cublasdx::get_shared_storage_size<HISASelectorWideGemmF32>()
            : cublasdx::get_shared_storage_size<HISASelectorGemmF32>();
    const size_t smem_bytes =
        sizeof(float) * static_cast<size_t>(params.block_score_capacity)
        + sizeof(int32_t) * static_cast<size_t>(params.block_score_capacity)
        + sizeof(int32_t) * static_cast<size_t>(params.effective_block_topk)
        + sizeof(float) * static_cast<size_t>(params.candidate_capacity)
        + sizeof(uint16_t) * static_cast<size_t>(params.candidate_capacity)
        + 16
        + gemm_smem_bytes;
    auto kernel = use_wide_tile ? selector_megakernel_wide_kernel
                                : selector_megakernel_kernel;
    if (smem_bytes > 48 * 1024) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    }
    LaunchKernel(params.q_rows, HISASelectorGemmF32::block_dim, device_.unwrap(), smem_bytes)(
        kernel, params);
    RuntimeDeviceCheck();
  }

  static void hisa_selector_parallel_select_blocks(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView rep_values,
      tvm::ffi::TensorView rep_scales,
      tvm::ffi::TensorView block_counts,
      tvm::ffi::TensorView block_topk_counts,
      tvm::ffi::TensorView selected_blocks,
      int64_t max_blocks,
      int64_t effective_block_topk) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto B = SymbolicSize{"batch_size"};
    auto R = SymbolicSize{"rep_rows"};
    auto EB = SymbolicSize{"effective_block_topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({R, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(rep_values);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(rep_scales);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(block_counts);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(block_topk_counts);
    TensorMatcher({Q, EB}).with_dtype<int32_t>().with_device(device_).verify(selected_blocks);
    if (static_cast<uint32_t>(H.unwrap()) != kHISASelectorHeads) {
      throw std::runtime_error("hisa_selector_parallel_select_blocks requires exactly 64 heads.");
    }
    if (max_blocks <= 0 || effective_block_topk <= 0 ||
        static_cast<int64_t>(EB.unwrap()) != effective_block_topk) {
      throw std::runtime_error("hisa_selector_parallel_select_blocks got mismatched dimensions.");
    }
    const auto expected_rep_rows =
        static_cast<int64_t>(B.unwrap()) * static_cast<int64_t>(max_blocks);
    if (static_cast<int64_t>(R.unwrap()) != expected_rep_rows) {
      throw std::runtime_error("hisa_selector_parallel_select_blocks got mismatched block rep rows.");
    }
    uint32_t block_score_capacity = 1;
    while (block_score_capacity < static_cast<uint32_t>(max_blocks)) {
      block_score_capacity <<= 1;
    }
    const auto params = NVFP4HISASelectorParallelSelectParam{
        .q_values = q_values.data_ptr(),
        .q_scales = q_scales.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .weights = weights.data_ptr(),
        .token_to_batch_idx = token_to_batch_idx.data_ptr(),
        .rep_values = rep_values.data_ptr(),
        .rep_scales = rep_scales.data_ptr(),
        .block_counts = block_counts.data_ptr(),
        .block_topk_counts = block_topk_counts.data_ptr(),
        .selected_blocks = selected_blocks.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .n_heads = static_cast<uint32_t>(H.unwrap()),
        .max_blocks = static_cast<uint32_t>(max_blocks),
        .effective_block_topk = static_cast<uint32_t>(effective_block_topk),
        .block_score_capacity = block_score_capacity,
    };
    const size_t smem_bytes =
        sizeof(float) * static_cast<size_t>(params.block_score_capacity)
        + sizeof(int32_t) * static_cast<size_t>(params.block_score_capacity)
        + 16
        + cublasdx::get_shared_storage_size<HISASelectorGemmF32>();
    auto kernel = selector_parallel_select_kernel;
    if (smem_bytes > 48 * 1024) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    }
    LaunchKernel(params.q_rows, HISASelectorGemmF32::block_dim, device_.unwrap(), smem_bytes)(
        kernel, params);
    RuntimeDeviceCheck();
  }

  static void hisa_selector_parallel_score_candidates(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView selected_blocks,
      tvm::ffi::TensorView logits) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto B = SymbolicSize{"batch_size"};
    auto P = SymbolicSize{"page_table_stride"};
    auto EB = SymbolicSize{"effective_block_topk"};
    auto CL = SymbolicSize{"candidate_len"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({Q, EB}).with_dtype<int32_t>().with_device(device_).verify(selected_blocks);
    TensorMatcher({Q, CL}).with_dtype<float>().with_device(device_).verify(logits);
    if (static_cast<uint32_t>(H.unwrap()) != kHISASelectorHeads) {
      throw std::runtime_error("hisa_selector_parallel_score_candidates requires exactly 64 heads.");
    }
    if (static_cast<uint32_t>(CL.unwrap()) != static_cast<uint32_t>(EB.unwrap()) * 128u) {
      throw std::runtime_error("hisa_selector_parallel_score_candidates requires candidate_len == effective_block_topk * 128.");
    }
    const auto params = NVFP4HISASelectorParallelScoreParam{
        .q_values = q_values.data_ptr(),
        .q_scales = q_scales.data_ptr(),
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .weights = weights.data_ptr(),
        .token_to_batch_idx = token_to_batch_idx.data_ptr(),
        .selected_blocks = selected_blocks.data_ptr(),
        .logits = logits.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .n_heads = static_cast<uint32_t>(H.unwrap()),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
        .effective_block_topk = static_cast<uint32_t>(EB.unwrap()),
        .candidate_len = static_cast<uint32_t>(CL.unwrap()),
    };
    const size_t smem_bytes =
        cublasdx::get_shared_storage_size<HISASelectorGemmF32>();
    auto kernel = selector_parallel_score_kernel;
    if (smem_bytes > 48 * 1024) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    }
    LaunchKernel(
        dim3(params.q_rows, params.effective_block_topk),
        HISASelectorGemmF32::block_dim,
        device_.unwrap(),
        smem_bytes)(kernel, params);
    RuntimeDeviceCheck();
  }

  static void hisa_selector_cluster_fused(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView rep_values,
      tvm::ffi::TensorView rep_scales,
      tvm::ffi::TensorView block_counts,
      tvm::ffi::TensorView block_topk_counts,
      tvm::ffi::TensorView candidate_keys,
      tvm::ffi::TensorView topk_indices,
      int64_t max_blocks,
      int64_t effective_block_topk) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto B = SymbolicSize{"batch_size"};
    auto P = SymbolicSize{"page_table_stride"};
    auto R = SymbolicSize{"rep_rows"};
    auto CL = SymbolicSize{"candidate_len"};
    auto K = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B, P}).with_dtype<IndicesT>().with_device(device_).verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({R, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(rep_values);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(rep_scales);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(block_counts);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(block_topk_counts);
    TensorMatcher({Q, CL}).with_dtype<int32_t>().with_device(device_).verify(candidate_keys);
    TensorMatcher({Q, K}).with_dtype<int32_t>().with_device(device_).verify(topk_indices);
    if (static_cast<uint32_t>(H.unwrap()) != kHISASelectorHeads) {
      throw std::runtime_error("hisa_selector_cluster_fused requires exactly 64 heads.");
    }
    if (max_blocks <= 0 || effective_block_topk <= 0) {
      throw std::runtime_error("hisa_selector_cluster_fused got non-positive selector dimensions.");
    }
    if (static_cast<uint32_t>(CL.unwrap()) != static_cast<uint32_t>(effective_block_topk) * 128u) {
      throw std::runtime_error("hisa_selector_cluster_fused requires candidate_len == effective_block_topk * 128.");
    }
    const auto expected_rep_rows =
        static_cast<int64_t>(B.unwrap()) * static_cast<int64_t>(max_blocks);
    if (static_cast<int64_t>(R.unwrap()) != expected_rep_rows) {
      throw std::runtime_error("hisa_selector_cluster_fused got mismatched block rep rows.");
    }
    uint32_t block_score_capacity = 1;
    while (block_score_capacity < static_cast<uint32_t>(max_blocks)) {
      block_score_capacity <<= 1;
    }
    const auto params = NVFP4HISASelectorClusterFusedParam{
        .q_values = q_values.data_ptr(),
        .q_scales = q_scales.data_ptr(),
        .cache = cache.data_ptr(),
        .page_table = page_table.data_ptr(),
        .seq_lens = seq_lens.data_ptr(),
        .weights = weights.data_ptr(),
        .token_to_batch_idx = token_to_batch_idx.data_ptr(),
        .rep_values = rep_values.data_ptr(),
        .rep_scales = rep_scales.data_ptr(),
        .block_counts = block_counts.data_ptr(),
        .block_topk_counts = block_topk_counts.data_ptr(),
        .candidate_keys = candidate_keys.data_ptr(),
        .topk_indices = topk_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .n_heads = static_cast<uint32_t>(H.unwrap()),
        .max_blocks = static_cast<uint32_t>(max_blocks),
        .page_table_stride = static_cast<uint32_t>(P.unwrap()),
        .effective_block_topk = static_cast<uint32_t>(effective_block_topk),
        .topk = static_cast<uint32_t>(K.unwrap()),
        .block_score_capacity = block_score_capacity,
        .candidate_len = static_cast<uint32_t>(CL.unwrap()),
    };
    const size_t smem_bytes =
        sizeof(float) * static_cast<size_t>(params.block_score_capacity)
        + sizeof(int32_t) * static_cast<size_t>(params.block_score_capacity)
        + sizeof(int32_t) * static_cast<size_t>(params.effective_block_topk)
        + 16
        + cublasdx::get_shared_storage_size<HISASelectorGemmF32>();
    auto kernel = selector_cluster_fused_kernel;
    if (smem_bytes > 48 * 1024) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    }
    LaunchKernel(
        {params.q_rows, kHISASelectorClusterSize},
        HISASelectorGemmF32::block_dim,
        device_.unwrap(),
        smem_bytes)
        .enable_cluster({1, kHISASelectorClusterSize})(kernel, params);
    RuntimeDeviceCheck();
  }

  static void hisa_deepgemm_candidate_logits(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView candidate_context_lens,
      tvm::ffi::TensorView candidate_page_table,
      tvm::ffi::TensorView schedule_meta,
      tvm::ffi::TensorView logits) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto PT = SymbolicSize{"candidate_pages"};
    auto CL = SymbolicSize{"candidate_len"};
    auto S = SymbolicSize{"schedule_rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q, 1}).with_dtype<int32_t>().with_device(device_).verify(candidate_context_lens);
    TensorMatcher({Q, PT}).with_dtype<int32_t>().with_device(device_).verify(candidate_page_table);
    TensorMatcher({S, 2}).with_dtype<int32_t>().with_device(device_).verify(schedule_meta);
    TensorMatcher({Q, CL}).with_dtype<float>().with_device(device_).verify(logits);

    if (static_cast<uint32_t>(H.unwrap()) != kHISASelectorHeads) {
      throw std::runtime_error("hisa_deepgemm_candidate_logits requires exactly 64 heads.");
    }
    if (static_cast<uint32_t>(CL.unwrap()) % 256u != 0) {
      throw std::runtime_error("hisa_deepgemm_candidate_logits requires candidate_len aligned to 256.");
    }
    if (static_cast<uint32_t>(PT.unwrap()) * kPageSize < static_cast<uint32_t>(CL.unwrap())) {
      throw std::runtime_error("hisa_deepgemm_candidate_logits page table is too short for candidate_len.");
    }
    if (static_cast<uint32_t>(S.unwrap()) <= 1) {
      throw std::runtime_error("hisa_deepgemm_candidate_logits got empty schedule metadata.");
    }

    constexpr uint32_t kSplitKV = 256;
	    constexpr uint32_t kNumQStages = 3;
	    constexpr uint32_t kNumKVStages = 10;
    constexpr uint32_t kNumTmemStages = 3;
    constexpr uint32_t kSpecializedThreads = 128;
    constexpr uint32_t kMathThreads = 256;
    constexpr uint32_t kNextNAtom = 1;
    const uint32_t q_rows = static_cast<uint32_t>(Q.unwrap());
    const uint32_t candidate_len = static_cast<uint32_t>(CL.unwrap());
    const uint32_t num_pages = static_cast<uint32_t>(cache.size(0));
    const uint32_t num_sms = static_cast<uint32_t>(S.unwrap() - 1);

    const auto tensor_map_q = hisa_make_tma_desc_2d(
        q_values.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        kIndexerHeadDim,
        static_cast<uint64_t>(q_rows) * kHISASelectorHeads,
        static_cast<uint64_t>(q_values.strides()[1]),
        kIndexerHeadDim,
        kHISASelectorHeads,
        CU_TENSOR_MAP_SWIZZLE_64B);
    const auto tensor_map_sf_q = hisa_make_tma_desc_2d(
        q_scales.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_INT32,
        kHISASelectorHeads,
        q_rows,
        static_cast<uint64_t>(q_scales.strides()[0]) * sizeof(int32_t),
        kHISASelectorHeads,
        kNextNAtom,
        CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto tensor_map_weights = hisa_make_tma_desc_2d(
        weights.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        kHISASelectorHeads,
        q_rows,
        static_cast<uint64_t>(weights.strides()[0]) * sizeof(float),
        kHISASelectorHeads,
        kNextNAtom,
        CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto tensor_map_kv = hisa_make_tma_desc_3d(
        cache.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        kIndexerHeadDim,
        kPageSize,
        num_pages,
        kNVFP4ValueBytes,
        static_cast<uint64_t>(cache.strides()[0]),
        kIndexerHeadDim,
        kPageSize,
        1,
        CU_TENSOR_MAP_SWIZZLE_64B);
    const auto* scale_base =
        static_cast<const uint8_t*>(cache.data_ptr()) +
        static_cast<size_t>(kNVFP4ValueBytes) * kPageSize;
    const auto tensor_map_sf_kv = hisa_make_tma_desc_2d(
        scale_base,
        CU_TENSOR_MAP_DATA_TYPE_INT32,
        kPageSize,
        num_pages,
        static_cast<uint64_t>(cache.strides()[0]),
        kPageSize,
        1,
        CU_TENSOR_MAP_SWIZZLE_NONE);

    const size_t smem_q_size_per_stage =
        static_cast<size_t>(kNextNAtom) * kHISASelectorHeads * (kIndexerHeadDim / 2);
    const size_t smem_sf_q_size_per_stage = 128 * sizeof(int32_t);
    const size_t smem_kv_size_per_stage = kSplitKV * (kIndexerHeadDim / 2);
    const size_t smem_sf_kv_size_per_stage = kSplitKV * sizeof(int32_t);
    const size_t smem_weight_size_per_stage =
        static_cast<size_t>(kNextNAtom) * kHISASelectorHeads * sizeof(float);
    const size_t smem_barriers =
        (kNumQStages + kNumKVStages + kNumTmemStages) * 2 * 8;
    const size_t smem_tmem_ptr = 4;
    const size_t smem_bytes =
        kNumQStages *
            (smem_q_size_per_stage + smem_sf_q_size_per_stage +
             smem_weight_size_per_stage)
        + kNumKVStages * (smem_kv_size_per_stage + smem_sf_kv_size_per_stage)
        + smem_barriers + smem_tmem_ptr;

    auto kernel = deepgemm_candidate_logits_kernel;
    if (smem_bytes > 48 * 1024) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    }
	    LaunchKernel(num_sms, kSpecializedThreads + kMathThreads, device_.unwrap(), smem_bytes)(
	        kernel,
	        q_rows,
        candidate_len,
        static_cast<uint32_t>(PT.unwrap()),
        reinterpret_cast<const uint32_t*>(candidate_context_lens.data_ptr()),
        static_cast<float*>(logits.data_ptr()),
        reinterpret_cast<const uint32_t*>(candidate_page_table.data_ptr()),
        static_cast<const uint32_t*>(nullptr),
        reinterpret_cast<const uint32_t*>(schedule_meta.data_ptr()),
        tensor_map_q,
        tensor_map_sf_q,
        tensor_map_kv,
        tensor_map_sf_kv,
        tensor_map_weights);
    RuntimeDeviceCheck();
  }

  static void hisa_deepgemm_candidate_keys(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView candidate_context_lens,
      tvm::ffi::TensorView candidate_page_table,
      tvm::ffi::TensorView source_page_table,
      tvm::ffi::TensorView schedule_meta,
      tvm::ffi::TensorView selected_blocks,
      tvm::ffi::TensorView prefix_lens,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView candidate_keys) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto PT = SymbolicSize{"candidate_pages"};
    auto B = SymbolicSize{"source_page_table_batches"};
    auto P = SymbolicSize{"source_page_table_stride"};
    auto BT = SymbolicSize{"block_topk"};
    auto CL = SymbolicSize{"candidate_len"};
    auto S = SymbolicSize{"schedule_rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q, 1}).with_dtype<int32_t>().with_device(device_).verify(candidate_context_lens);
    TensorMatcher({B, P}).with_dtype<int32_t>().with_device(device_).verify(source_page_table);
    const bool use_source_page_table =
        static_cast<uint32_t>(B.unwrap()) > 0u &&
        static_cast<uint32_t>(P.unwrap()) > 0u;
    if (!use_source_page_table) {
      TensorMatcher({Q, PT}).with_dtype<int32_t>().with_device(device_).verify(candidate_page_table);
    }
    TensorMatcher({S, 2}).with_dtype<int32_t>().with_device(device_).verify(schedule_meta);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(selected_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(prefix_lens);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({Q, CL}).with_dtype<int32_t>().with_device(device_).verify(candidate_keys);

    if (static_cast<uint32_t>(H.unwrap()) != kHISASelectorHeads) {
      throw std::runtime_error("hisa_deepgemm_candidate_keys requires exactly 64 heads.");
    }
    if (static_cast<uint32_t>(CL.unwrap()) != static_cast<uint32_t>(BT.unwrap()) * 128u) {
      throw std::runtime_error("hisa_deepgemm_candidate_keys requires candidate_len == block_topk * 128.");
    }
    if (static_cast<uint32_t>(CL.unwrap()) % 256u != 0) {
      throw std::runtime_error("hisa_deepgemm_candidate_keys requires candidate_len aligned to 256.");
    }
    if (!use_source_page_table &&
        static_cast<uint32_t>(PT.unwrap()) * kPageSize < static_cast<uint32_t>(CL.unwrap())) {
      throw std::runtime_error("hisa_deepgemm_candidate_keys page table is too short for candidate_len.");
    }
    if (use_source_page_table &&
        static_cast<uint32_t>(token_to_batch_idx.size(0)) < static_cast<uint32_t>(Q.unwrap())) {
      throw std::runtime_error("hisa_deepgemm_candidate_keys source page-table mode requires token_to_batch_idx per row.");
    }
    if (static_cast<uint32_t>(S.unwrap()) <= 1) {
      throw std::runtime_error("hisa_deepgemm_candidate_keys got empty schedule metadata.");
    }

    constexpr uint32_t kSplitKV = 256;
	    constexpr uint32_t kNumQStages = 3;
	    constexpr uint32_t kNumKVStages = 10;
    constexpr uint32_t kNumTmemStages = 3;
    constexpr uint32_t kSpecializedThreads = 128;
    constexpr uint32_t kMathThreads = 256;
    constexpr uint32_t kNextNAtom = 1;
    const uint32_t q_rows = static_cast<uint32_t>(Q.unwrap());
    const uint32_t candidate_len = static_cast<uint32_t>(CL.unwrap());
    const uint32_t num_pages = static_cast<uint32_t>(cache.size(0));
    const uint32_t num_sms = static_cast<uint32_t>(S.unwrap() - 1);

    const auto tensor_map_q = hisa_make_tma_desc_2d(
        q_values.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        kIndexerHeadDim,
        static_cast<uint64_t>(q_rows) * kHISASelectorHeads,
        static_cast<uint64_t>(q_values.strides()[1]),
        kIndexerHeadDim,
        kHISASelectorHeads,
        CU_TENSOR_MAP_SWIZZLE_64B);
    const auto tensor_map_sf_q = hisa_make_tma_desc_2d(
        q_scales.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_INT32,
        kHISASelectorHeads,
        q_rows,
        static_cast<uint64_t>(q_scales.strides()[0]) * sizeof(int32_t),
        kHISASelectorHeads,
        kNextNAtom,
        CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto tensor_map_weights = hisa_make_tma_desc_2d(
        weights.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        kHISASelectorHeads,
        q_rows,
        static_cast<uint64_t>(weights.strides()[0]) * sizeof(float),
        kHISASelectorHeads,
        kNextNAtom,
        CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto tensor_map_kv = hisa_make_tma_desc_3d(
        cache.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        kIndexerHeadDim,
        kPageSize,
        num_pages,
        kNVFP4ValueBytes,
        static_cast<uint64_t>(cache.strides()[0]),
        kIndexerHeadDim,
        kPageSize,
        1,
        CU_TENSOR_MAP_SWIZZLE_64B);
    const auto* scale_base =
        static_cast<const uint8_t*>(cache.data_ptr()) +
        static_cast<size_t>(kNVFP4ValueBytes) * kPageSize;
    const auto tensor_map_sf_kv = hisa_make_tma_desc_2d(
        scale_base,
        CU_TENSOR_MAP_DATA_TYPE_INT32,
        kPageSize,
        num_pages,
        static_cast<uint64_t>(cache.strides()[0]),
        kPageSize,
        1,
        CU_TENSOR_MAP_SWIZZLE_NONE);

    const size_t smem_q_size_per_stage =
        static_cast<size_t>(kNextNAtom) * kHISASelectorHeads * (kIndexerHeadDim / 2);
    const size_t smem_sf_q_size_per_stage = 128 * sizeof(int32_t);
    const size_t smem_kv_size_per_stage = kSplitKV * (kIndexerHeadDim / 2);
    const size_t smem_sf_kv_size_per_stage = kSplitKV * sizeof(int32_t);
    const size_t smem_weight_size_per_stage =
        static_cast<size_t>(kNextNAtom) * kHISASelectorHeads * sizeof(float);
    const size_t smem_barriers =
        (kNumQStages + kNumKVStages + kNumTmemStages) * 2 * 8;
    const size_t smem_tmem_ptr = 4;
    const size_t smem_bytes =
        kNumQStages *
            (smem_q_size_per_stage + smem_sf_q_size_per_stage +
             smem_weight_size_per_stage)
        + kNumKVStages * (smem_kv_size_per_stage + smem_sf_kv_size_per_stage)
        + smem_barriers + smem_tmem_ptr;

    auto kernel = deepgemm_candidate_keys_kernel;
    if (smem_bytes > 48 * 1024) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    }
    LaunchKernel(num_sms, kSpecializedThreads + kMathThreads, device_.unwrap(), smem_bytes)(
        kernel,
        q_rows,
        candidate_len,
        use_source_page_table ? 0u : static_cast<uint32_t>(PT.unwrap()),
        static_cast<uint32_t>(BT.unwrap()),
        reinterpret_cast<const uint32_t*>(candidate_context_lens.data_ptr()),
        reinterpret_cast<uint32_t*>(candidate_keys.data_ptr()),
        use_source_page_table
            ? static_cast<const uint32_t*>(nullptr)
            : reinterpret_cast<const uint32_t*>(candidate_page_table.data_ptr()),
        use_source_page_table
            ? reinterpret_cast<const uint32_t*>(source_page_table.data_ptr())
            : static_cast<const uint32_t*>(nullptr),
        reinterpret_cast<const uint32_t*>(schedule_meta.data_ptr()),
        reinterpret_cast<int32_t*>(selected_blocks.data_ptr()),
        reinterpret_cast<const int32_t*>(prefix_lens.data_ptr()),
        use_source_page_table
            ? reinterpret_cast<const int32_t*>(token_to_batch_idx.data_ptr())
            : static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        0u,
        use_source_page_table ? static_cast<uint32_t>(P.unwrap()) : 0u,
        tensor_map_q,
        tensor_map_sf_q,
        tensor_map_kv,
        tensor_map_sf_kv,
	        tensor_map_weights);
	    RuntimeDeviceCheck();
	  }

  static void hisa_deepgemm_candidate_topk_cooperative(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView candidate_context_lens,
      tvm::ffi::TensorView source_page_table,
      tvm::ffi::TensorView schedule_meta,
      tvm::ffi::TensorView selected_blocks,
      tvm::ffi::TensorView prefix_lens,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView candidate_keys,
      tvm::ffi::TensorView topk_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto B = SymbolicSize{"source_page_table_batches"};
    auto P = SymbolicSize{"source_page_table_stride"};
    auto BT = SymbolicSize{"block_topk"};
    auto CL = SymbolicSize{"candidate_len"};
    auto K = SymbolicSize{"topk"};
    auto S = SymbolicSize{"schedule_rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({Q, 1}).with_dtype<int32_t>().with_device(device_).verify(candidate_context_lens);
    TensorMatcher({B, P}).with_dtype<int32_t>().with_device(device_).verify(source_page_table);
    TensorMatcher({S, 2}).with_dtype<int32_t>().with_device(device_).verify(schedule_meta);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(selected_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(prefix_lens);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({Q, CL}).with_dtype<int32_t>().with_device(device_).verify(candidate_keys);
    TensorMatcher({Q, K}).with_dtype<int32_t>().with_device(device_).verify(topk_indices);

    if (static_cast<uint32_t>(H.unwrap()) != kHISASelectorHeads) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cooperative requires exactly 64 heads.");
    }
    if (static_cast<uint32_t>(B.unwrap()) == 0u || static_cast<uint32_t>(P.unwrap()) == 0u) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cooperative requires source page-table mode.");
    }
    if (static_cast<uint32_t>(CL.unwrap()) != static_cast<uint32_t>(BT.unwrap()) * 128u) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cooperative requires candidate_len == block_topk * 128.");
    }
    if (static_cast<uint32_t>(CL.unwrap()) % 256u != 0) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cooperative requires candidate_len aligned to 256.");
    }
    if (static_cast<uint32_t>(K.unwrap()) == 0u) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cooperative requires positive topk.");
    }
    if (static_cast<uint32_t>(S.unwrap()) <= 1) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cooperative got empty schedule metadata.");
    }

    constexpr uint32_t kSplitKV = 256;
    constexpr uint32_t kNumQStages = 3;
    constexpr uint32_t kNumKVStages = 10;
    constexpr uint32_t kNumTmemStages = 3;
    constexpr uint32_t kSpecializedThreads = 128;
    constexpr uint32_t kMathThreads = 256;
    constexpr uint32_t kNextNAtom = 1;
    const uint32_t q_rows = static_cast<uint32_t>(Q.unwrap());
    const uint32_t candidate_len = static_cast<uint32_t>(CL.unwrap());
    const uint32_t num_pages = static_cast<uint32_t>(cache.size(0));
    const uint32_t num_sms = static_cast<uint32_t>(S.unwrap() - 1);

    const auto tensor_map_q = hisa_make_tma_desc_2d(
        q_values.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        kIndexerHeadDim,
        static_cast<uint64_t>(q_rows) * kHISASelectorHeads,
        static_cast<uint64_t>(q_values.strides()[1]),
        kIndexerHeadDim,
        kHISASelectorHeads,
        CU_TENSOR_MAP_SWIZZLE_64B);
    const auto tensor_map_sf_q = hisa_make_tma_desc_2d(
        q_scales.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_INT32,
        kHISASelectorHeads,
        q_rows,
        static_cast<uint64_t>(q_scales.strides()[0]) * sizeof(int32_t),
        kHISASelectorHeads,
        kNextNAtom,
        CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto tensor_map_weights = hisa_make_tma_desc_2d(
        weights.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        kHISASelectorHeads,
        q_rows,
        static_cast<uint64_t>(weights.strides()[0]) * sizeof(float),
        kHISASelectorHeads,
        kNextNAtom,
        CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto tensor_map_kv = hisa_make_tma_desc_3d(
        cache.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        kIndexerHeadDim,
        kPageSize,
        num_pages,
        kNVFP4ValueBytes,
        static_cast<uint64_t>(cache.strides()[0]),
        kIndexerHeadDim,
        kPageSize,
        1,
        CU_TENSOR_MAP_SWIZZLE_64B);
    const auto* scale_base =
        static_cast<const uint8_t*>(cache.data_ptr()) +
        static_cast<size_t>(kNVFP4ValueBytes) * kPageSize;
    const auto tensor_map_sf_kv = hisa_make_tma_desc_2d(
        scale_base,
        CU_TENSOR_MAP_DATA_TYPE_INT32,
        kPageSize,
        num_pages,
        static_cast<uint64_t>(cache.strides()[0]),
        kPageSize,
        1,
        CU_TENSOR_MAP_SWIZZLE_NONE);

    const size_t smem_q_size_per_stage =
        static_cast<size_t>(kNextNAtom) * kHISASelectorHeads * (kIndexerHeadDim / 2);
    const size_t smem_sf_q_size_per_stage = 128 * sizeof(int32_t);
    const size_t smem_kv_size_per_stage = kSplitKV * (kIndexerHeadDim / 2);
    const size_t smem_sf_kv_size_per_stage = kSplitKV * sizeof(int32_t);
    const size_t smem_weight_size_per_stage =
        static_cast<size_t>(kNextNAtom) * kHISASelectorHeads * sizeof(float);
    const size_t smem_barriers =
        (kNumQStages + kNumKVStages + kNumTmemStages) * 2 * 8;
    const size_t smem_tmem_ptr = 4;
    const size_t smem_bytes =
        kNumQStages *
            (smem_q_size_per_stage + smem_sf_q_size_per_stage +
             smem_weight_size_per_stage)
        + kNumKVStages * (smem_kv_size_per_stage + smem_sf_kv_size_per_stage)
        + smem_barriers + smem_tmem_ptr;

    auto kernel = deepgemm_candidate_keys_kernel;
    if (smem_bytes > 48 * 1024) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    }
    const auto dl_device = device_.unwrap();
    int cooperative = 0;
    RuntimeDeviceCheck(cudaDeviceGetAttribute(
        &cooperative, cudaDevAttrCooperativeLaunch, dl_device.device_id));
    if (cooperative == 0) {
      throw std::runtime_error("Device does not support cooperative HISA candidate topk launch.");
    }

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeCooperative;
    attrs[0].val.cooperative = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_sms);
    config.blockDim = dim3(kSpecializedThreads + kMathThreads);
    config.dynamicSmemBytes = smem_bytes;
    config.stream = LaunchKernel::resolve_device(dl_device);
    config.attrs = attrs;
    config.numAttrs = 1;
    RuntimeDeviceCheck(cudaLaunchKernelEx(
        &config,
        kernel,
        q_rows,
        candidate_len,
        0u,
        static_cast<uint32_t>(BT.unwrap()),
        reinterpret_cast<const uint32_t*>(candidate_context_lens.data_ptr()),
        reinterpret_cast<uint32_t*>(candidate_keys.data_ptr()),
        static_cast<const uint32_t*>(nullptr),
        reinterpret_cast<const uint32_t*>(source_page_table.data_ptr()),
        reinterpret_cast<const uint32_t*>(schedule_meta.data_ptr()),
        reinterpret_cast<int32_t*>(selected_blocks.data_ptr()),
        reinterpret_cast<const int32_t*>(prefix_lens.data_ptr()),
        reinterpret_cast<const int32_t*>(token_to_batch_idx.data_ptr()),
        reinterpret_cast<int32_t*>(topk_indices.data_ptr()),
        static_cast<uint32_t>(K.unwrap()),
        static_cast<uint32_t>(P.unwrap()),
        tensor_map_q,
        tensor_map_sf_q,
        tensor_map_kv,
        tensor_map_sf_kv,
        tensor_map_weights));
    RuntimeDeviceCheck();
  }

	  static void hisa_deepgemm_candidate_keys_row_split(
	      tvm::ffi::TensorView q_values,
	      tvm::ffi::TensorView q_scales,
	      tvm::ffi::TensorView cache,
	      tvm::ffi::TensorView weights,
	      tvm::ffi::TensorView source_page_table,
	      tvm::ffi::TensorView selected_blocks,
	      tvm::ffi::TensorView prefix_lens,
	      tvm::ffi::TensorView token_to_batch_idx,
	      tvm::ffi::TensorView candidate_keys,
	      int row_splits) {
	    using namespace host;

	    auto Q = SymbolicSize{"q_rows"};
	    auto H = SymbolicSize{"n_heads"};
	    auto B = SymbolicSize{"source_page_table_batches"};
	    auto P = SymbolicSize{"source_page_table_stride"};
	    auto BT = SymbolicSize{"block_topk"};
	    auto CL = SymbolicSize{"candidate_len"};
	    auto device_ = SymbolicDevice{};
	    device_.set_options<kDLCUDA>();
	    TensorMatcher({Q, H, kNVFP4ValueBytes})
	        .with_dtype<uint8_t>()
	        .with_device(device_)
	        .verify(q_values);
	    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
	    TensorMatcher({-1, -1})
	        .with_strides({kPageBytes, 1})
	        .with_dtype<uint8_t>()
	        .with_device(device_)
	        .verify(cache);
	    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
	    TensorMatcher({B, P}).with_dtype<int32_t>().with_device(device_).verify(source_page_table);
	    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(selected_blocks);
	    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(prefix_lens);
	    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
	    TensorMatcher({Q, CL}).with_dtype<int32_t>().with_device(device_).verify(candidate_keys);

	    if (row_splits <= 0) {
	      throw std::runtime_error("hisa_deepgemm_candidate_keys_row_split requires positive row_splits.");
	    }
	    if (static_cast<uint32_t>(H.unwrap()) != kHISASelectorHeads) {
	      throw std::runtime_error("hisa_deepgemm_candidate_keys_row_split requires exactly 64 heads.");
	    }
	    if (kPageSize != 64) {
	      throw std::runtime_error("hisa_deepgemm_candidate_keys_row_split requires page_size=64.");
	    }
	    if (static_cast<uint32_t>(CL.unwrap()) != static_cast<uint32_t>(BT.unwrap()) * 128u) {
	      throw std::runtime_error("hisa_deepgemm_candidate_keys_row_split requires candidate_len == block_topk * 128.");
	    }
	    if (static_cast<uint32_t>(CL.unwrap()) % 256u != 0) {
	      throw std::runtime_error("hisa_deepgemm_candidate_keys_row_split requires candidate_len aligned to 256.");
	    }
	    if (static_cast<uint32_t>(token_to_batch_idx.size(0)) < static_cast<uint32_t>(Q.unwrap())) {
	      throw std::runtime_error("hisa_deepgemm_candidate_keys_row_split requires token_to_batch_idx per row.");
	    }

	    constexpr uint32_t kSplitKV = 256;
	    constexpr uint32_t kNumQStages = 3;
	    constexpr uint32_t kNumKVStages = 10;
	    constexpr uint32_t kNumTmemStages = 3;
	    constexpr uint32_t kSpecializedThreads = 128;
	    constexpr uint32_t kMathThreads = 256;
	    constexpr uint32_t kNextNAtom = 1;
	    const uint32_t q_rows = static_cast<uint32_t>(Q.unwrap());
	    const uint32_t candidate_len = static_cast<uint32_t>(CL.unwrap());
	    const uint32_t num_pages = static_cast<uint32_t>(cache.size(0));
	    const uint32_t row_splits_u = static_cast<uint32_t>(row_splits);

	    const auto tensor_map_q = hisa_make_tma_desc_2d(
	        q_values.data_ptr(),
	        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
	        kIndexerHeadDim,
	        static_cast<uint64_t>(q_rows) * kHISASelectorHeads,
	        static_cast<uint64_t>(q_values.strides()[1]),
	        kIndexerHeadDim,
	        kHISASelectorHeads,
	        CU_TENSOR_MAP_SWIZZLE_64B);
	    const auto tensor_map_sf_q = hisa_make_tma_desc_2d(
	        q_scales.data_ptr(),
	        CU_TENSOR_MAP_DATA_TYPE_INT32,
	        kHISASelectorHeads,
	        q_rows,
	        static_cast<uint64_t>(q_scales.strides()[0]) * sizeof(int32_t),
	        kHISASelectorHeads,
	        kNextNAtom,
	        CU_TENSOR_MAP_SWIZZLE_NONE);
	    const auto tensor_map_weights = hisa_make_tma_desc_2d(
	        weights.data_ptr(),
	        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
	        kHISASelectorHeads,
	        q_rows,
	        static_cast<uint64_t>(weights.strides()[0]) * sizeof(float),
	        kHISASelectorHeads,
	        kNextNAtom,
	        CU_TENSOR_MAP_SWIZZLE_NONE);
	    const auto tensor_map_kv = hisa_make_tma_desc_3d(
	        cache.data_ptr(),
	        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
	        kIndexerHeadDim,
	        kPageSize,
	        num_pages,
	        kNVFP4ValueBytes,
	        static_cast<uint64_t>(cache.strides()[0]),
	        kIndexerHeadDim,
	        kPageSize,
	        1,
	        CU_TENSOR_MAP_SWIZZLE_64B);
	    const auto* scale_base =
	        static_cast<const uint8_t*>(cache.data_ptr()) +
	        static_cast<size_t>(kNVFP4ValueBytes) * kPageSize;
	    const auto tensor_map_sf_kv = hisa_make_tma_desc_2d(
	        scale_base,
	        CU_TENSOR_MAP_DATA_TYPE_INT32,
	        kPageSize,
	        num_pages,
	        static_cast<uint64_t>(cache.strides()[0]),
	        kPageSize,
	        1,
	        CU_TENSOR_MAP_SWIZZLE_NONE);

	    const size_t smem_q_size_per_stage =
	        static_cast<size_t>(kNextNAtom) * kHISASelectorHeads * (kIndexerHeadDim / 2);
	    const size_t smem_sf_q_size_per_stage = 128 * sizeof(int32_t);
	    const size_t smem_kv_size_per_stage = kSplitKV * (kIndexerHeadDim / 2);
	    const size_t smem_sf_kv_size_per_stage = kSplitKV * sizeof(int32_t);
	    const size_t smem_weight_size_per_stage =
	        static_cast<size_t>(kNextNAtom) * kHISASelectorHeads * sizeof(float);
	    const size_t smem_barriers =
	        (kNumQStages + kNumKVStages + kNumTmemStages) * 2 * 8;
	    const size_t smem_tmem_ptr = 4;
	    const size_t smem_bytes =
	        kNumQStages *
	            (smem_q_size_per_stage + smem_sf_q_size_per_stage +
	             smem_weight_size_per_stage)
	        + kNumKVStages * (smem_kv_size_per_stage + smem_sf_kv_size_per_stage)
	        + smem_barriers + smem_tmem_ptr;

	    auto kernel = deepgemm_candidate_keys_row_split_kernel;
	    if (smem_bytes > 48 * 1024) {
	      RuntimeDeviceCheck(cudaFuncSetAttribute(
	          kernel,
	          cudaFuncAttributeMaxDynamicSharedMemorySize,
	          static_cast<int>(smem_bytes)));
	    }
	    LaunchKernel(dim3(q_rows, row_splits_u), kSpecializedThreads + kMathThreads, device_.unwrap(), smem_bytes)(
	        kernel,
	        q_rows,
	        candidate_len,
	        row_splits_u,
	        static_cast<uint32_t>(BT.unwrap()),
		        reinterpret_cast<uint32_t*>(candidate_keys.data_ptr()),
		        static_cast<int32_t*>(nullptr),
		        0u,
		        reinterpret_cast<const uint32_t*>(source_page_table.data_ptr()),
	        reinterpret_cast<const int32_t*>(selected_blocks.data_ptr()),
	        reinterpret_cast<const int32_t*>(prefix_lens.data_ptr()),
	        reinterpret_cast<const int32_t*>(token_to_batch_idx.data_ptr()),
	        static_cast<uint32_t>(P.unwrap()),
	        tensor_map_q,
	        tensor_map_sf_q,
	        tensor_map_kv,
	        tensor_map_sf_kv,
	        tensor_map_weights);
		    RuntimeDeviceCheck();
		  }

  static void hisa_deepgemm_candidate_topk_cluster(
      tvm::ffi::TensorView q_values,
      tvm::ffi::TensorView q_scales,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView source_page_table,
      tvm::ffi::TensorView selected_blocks,
      tvm::ffi::TensorView prefix_lens,
      tvm::ffi::TensorView token_to_batch_idx,
      tvm::ffi::TensorView candidate_keys,
      tvm::ffi::TensorView topk_indices,
      int row_splits,
      int use_cluster) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto H = SymbolicSize{"n_heads"};
    auto B = SymbolicSize{"source_page_table_batches"};
    auto P = SymbolicSize{"source_page_table_stride"};
    auto BT = SymbolicSize{"block_topk"};
    auto CL = SymbolicSize{"candidate_len"};
    auto K = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, H, kNVFP4ValueBytes})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_values);
    TensorMatcher({Q, H}).with_dtype<int32_t>().with_device(device_).verify(q_scales);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({Q, H}).with_dtype<float>().with_device(device_).verify(weights);
    TensorMatcher({B, P}).with_dtype<int32_t>().with_device(device_).verify(source_page_table);
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(selected_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(prefix_lens);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(token_to_batch_idx);
    TensorMatcher({Q, CL}).with_dtype<int32_t>().with_device(device_).verify(candidate_keys);
    TensorMatcher({Q, K}).with_dtype<int32_t>().with_device(device_).verify(topk_indices);

    if (static_cast<uint32_t>(H.unwrap()) != kHISASelectorHeads) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cluster requires exactly 64 heads.");
    }
    if (kPageSize != 64) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cluster requires page_size=64.");
    }
    if (static_cast<uint32_t>(CL.unwrap()) != static_cast<uint32_t>(BT.unwrap()) * 128u) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cluster requires candidate_len == block_topk * 128.");
    }
    if (static_cast<uint32_t>(CL.unwrap()) % 256u != 0) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cluster requires candidate_len aligned to 256.");
    }
    if (static_cast<uint32_t>(K.unwrap()) == 0) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cluster requires positive topk.");
    }
    if (row_splits <= 0) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cluster requires positive row_splits.");
    }
    const uint32_t row_splits_u = static_cast<uint32_t>(row_splits);
    const bool cluster_launch = use_cluster != 0;
    if (cluster_launch && row_splits_u != static_cast<uint32_t>(kHISASelectorClusterSize)) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cluster cluster launch requires row_splits=4.");
    }
    if (!cluster_launch && row_splits_u != 1u) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cluster non-cluster launch requires row_splits=1.");
    }
    if (static_cast<uint32_t>(token_to_batch_idx.size(0)) < static_cast<uint32_t>(Q.unwrap())) {
      throw std::runtime_error("hisa_deepgemm_candidate_topk_cluster requires token_to_batch_idx per row.");
    }

    constexpr uint32_t kSplitKV = 256;
    constexpr uint32_t kNumQStages = 3;
	    constexpr uint32_t kNumKVStages = 10;
    constexpr uint32_t kNumTmemStages = 3;
    constexpr uint32_t kSpecializedThreads = 128;
    constexpr uint32_t kMathThreads = 256;
    constexpr uint32_t kNextNAtom = 1;
    const uint32_t q_rows = static_cast<uint32_t>(Q.unwrap());
    const uint32_t candidate_len = static_cast<uint32_t>(CL.unwrap());
    const uint32_t num_pages = static_cast<uint32_t>(cache.size(0));

    const auto tensor_map_q = hisa_make_tma_desc_2d(
        q_values.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        kIndexerHeadDim,
        static_cast<uint64_t>(q_rows) * kHISASelectorHeads,
        static_cast<uint64_t>(q_values.strides()[1]),
        kIndexerHeadDim,
        kHISASelectorHeads,
        CU_TENSOR_MAP_SWIZZLE_64B);
    const auto tensor_map_sf_q = hisa_make_tma_desc_2d(
        q_scales.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_INT32,
        kHISASelectorHeads,
        q_rows,
        static_cast<uint64_t>(q_scales.strides()[0]) * sizeof(int32_t),
        kHISASelectorHeads,
        kNextNAtom,
        CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto tensor_map_weights = hisa_make_tma_desc_2d(
        weights.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        kHISASelectorHeads,
        q_rows,
        static_cast<uint64_t>(weights.strides()[0]) * sizeof(float),
        kHISASelectorHeads,
        kNextNAtom,
        CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto tensor_map_kv = hisa_make_tma_desc_3d(
        cache.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        kIndexerHeadDim,
        kPageSize,
        num_pages,
        kNVFP4ValueBytes,
        static_cast<uint64_t>(cache.strides()[0]),
        kIndexerHeadDim,
        kPageSize,
        1,
        CU_TENSOR_MAP_SWIZZLE_64B);
    const auto* scale_base =
        static_cast<const uint8_t*>(cache.data_ptr()) +
        static_cast<size_t>(kNVFP4ValueBytes) * kPageSize;
    const auto tensor_map_sf_kv = hisa_make_tma_desc_2d(
        scale_base,
        CU_TENSOR_MAP_DATA_TYPE_INT32,
        kPageSize,
        num_pages,
        static_cast<uint64_t>(cache.strides()[0]),
        kPageSize,
        1,
        CU_TENSOR_MAP_SWIZZLE_NONE);

    const size_t smem_q_size_per_stage =
        static_cast<size_t>(kNextNAtom) * kHISASelectorHeads * (kIndexerHeadDim / 2);
    const size_t smem_sf_q_size_per_stage = 128 * sizeof(int32_t);
    const size_t smem_kv_size_per_stage = kSplitKV * (kIndexerHeadDim / 2);
    const size_t smem_sf_kv_size_per_stage = kSplitKV * sizeof(int32_t);
    const size_t smem_weight_size_per_stage =
        static_cast<size_t>(kNextNAtom) * kHISASelectorHeads * sizeof(float);
    const size_t smem_barriers =
        (kNumQStages + kNumKVStages + kNumTmemStages) * 2 * 8;
    const size_t smem_tmem_ptr = 4;
    const size_t smem_bytes =
        kNumQStages *
            (smem_q_size_per_stage + smem_sf_q_size_per_stage +
             smem_weight_size_per_stage)
        + kNumKVStages * (smem_kv_size_per_stage + smem_sf_kv_size_per_stage)
        + smem_barriers + smem_tmem_ptr;

    auto kernel = deepgemm_candidate_keys_row_split_kernel;
    if (smem_bytes > 48 * 1024) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    }
    auto launch = LaunchKernel(
        dim3(q_rows, row_splits_u),
        kSpecializedThreads + kMathThreads,
        device_.unwrap(),
        smem_bytes);
    if (cluster_launch) {
      launch.enable_cluster({1, kHISASelectorClusterSize});
    }
    launch(
        kernel,
        q_rows,
        candidate_len,
        row_splits_u,
        static_cast<uint32_t>(BT.unwrap()),
        reinterpret_cast<uint32_t*>(candidate_keys.data_ptr()),
        reinterpret_cast<int32_t*>(topk_indices.data_ptr()),
        static_cast<uint32_t>(K.unwrap()),
        reinterpret_cast<const uint32_t*>(source_page_table.data_ptr()),
        reinterpret_cast<const int32_t*>(selected_blocks.data_ptr()),
        reinterpret_cast<const int32_t*>(prefix_lens.data_ptr()),
        reinterpret_cast<const int32_t*>(token_to_batch_idx.data_ptr()),
        static_cast<uint32_t>(P.unwrap()),
        tensor_map_q,
        tensor_map_sf_q,
        tensor_map_kv,
        tensor_map_sf_kv,
        tensor_map_weights);
    RuntimeDeviceCheck();
  }
#endif

  static void hisa_map_candidate_indices(
      tvm::ffi::TensorView top_blocks,
      tvm::ffi::TensorView prefix_lens,
      tvm::ffi::TensorView topk_indices) {
    using namespace host;

    auto Q = SymbolicSize{"q_rows"};
    auto K = SymbolicSize{"topk"};
    auto BT = SymbolicSize{"block_topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({Q, BT}).with_dtype<int32_t>().with_device(device_).verify(top_blocks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(prefix_lens);
    TensorMatcher({Q, K}).with_dtype<int32_t>().with_device(device_).verify(topk_indices);
    const auto params = NVFP4HISAMapCandidatesParam{
        .top_blocks = top_blocks.data_ptr(),
        .prefix_lens = prefix_lens.data_ptr(),
        .topk_indices = topk_indices.data_ptr(),
        .q_rows = static_cast<uint32_t>(Q.unwrap()),
        .topk = static_cast<uint32_t>(K.unwrap()),
        .block_topk = static_cast<uint32_t>(BT.unwrap()),
    };
    constexpr uint32_t kThreads = 256;
    const auto total = params.q_rows * params.topk;
    LaunchKernel(div_ceil(total, kThreads), kThreads, device_.unwrap())(
        map_candidates_kernel, params);
  }

};

}  // namespace
