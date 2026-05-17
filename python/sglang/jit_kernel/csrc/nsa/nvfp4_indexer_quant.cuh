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

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr int kIndexerHeadDim = 128;
constexpr int kNVFP4ValueBytes = kIndexerHeadDim / 2;
constexpr int kScaleBytes = 4;
constexpr float kE2M1Max = 6.0f;

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

SGL_DEVICE uint32_t fp32_to_radix_desc(uint32_t bits) {
  const auto asc_key = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  return ~asc_key;
}

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

  // Pad remaining [keep, topk) slots with -1.
  for (uint32_t i = keep + tid; i < param.topk; i += kThreads) {
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

SGL_DEVICE float decode_e2m1_nibble(uint32_t code, float scale) {
  constexpr float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = values[code & 0x7u] * scale;
  return (code & 0x8u) ? -magnitude : magnitude;
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
  constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  constexpr uint32_t kBlockSize = 128;
  const auto dim = threadIdx.x;
  const auto hisa_block = blockIdx.x;
  const auto batch = blockIdx.y;
  if (batch >= param.batch_size || hisa_block >= param.max_blocks ||
      dim >= kIndexerHeadDim) {
    return;
  }

  const auto seq_len = static_cast<const int32_t*>(param.seq_lens)[batch];
  const auto token_start = hisa_block * kBlockSize;
  const auto token_count =
      token_start < static_cast<uint32_t>(seq_len)
          ? min(kBlockSize, static_cast<uint32_t>(seq_len) - token_start)
          : 0u;
  float sum = 0.0f;
  for (uint32_t i = 0; i < token_count; ++i) {
    const auto token = token_start + i;
    const auto logical_page = token / kPageSize;
    const auto offset = token & (kPageSize - 1);
    const auto page =
        static_cast<const IndicesT*>(param.page_table)[batch * param.page_table_stride +
                                                       logical_page];
    if (page < 0) continue;
    const auto page_ptr =
        static_cast<const uint8_t*>(param.cache) + static_cast<int64_t>(page) * kPageBytes;
    const auto value_ptr = page_ptr + offset * kNVFP4ValueBytes;
    const auto scale_ptr = reinterpret_cast<const uint32_t*>(
        page_ptr + kNVFP4ValueBytes * kPageSize + offset * kScaleBytes);
    sum += load_nvfp4_value(value_ptr, scale_ptr, dim);
  }
  const auto out_idx =
      (static_cast<int64_t>(batch) * param.max_blocks + hisa_block) * kIndexerHeadDim +
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

  constexpr uint32_t kHeadsPerGroup = 8;
  const auto head_group = blockIdx.z;
  const auto warp_id = threadIdx.x >> 5;
  const auto lane = threadIdx.x & 31;
  const auto head = head_group * kHeadsPerGroup + warp_id;
  if (warp_id < kHeadsPerGroup && head < param.n_heads) {
    float dot = 0.0f;
    const auto q_value_ptr =
        static_cast<const uint8_t*>(param.q_values) +
        (static_cast<int64_t>(row) * param.n_heads + head) * kNVFP4ValueBytes;
    const auto q_scale_ptr =
        static_cast<const uint32_t*>(param.q_scales) +
        static_cast<int64_t>(row) * param.n_heads + head;
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
    float partial = 0.0f;
    for (uint32_t i = 0; i < kHeadsPerGroup; ++i) {
      const auto global_head = head_group * kHeadsPerGroup + i;
      if (global_head >= param.n_heads) break;
      const auto weight =
          static_cast<const float*>(param.weights)[static_cast<int64_t>(row) *
                                                       param.n_heads +
                                                   global_head];
      partial += fmaxf(head_dot[i], 0.0f) * weight;
    }
    atomicAdd(static_cast<float*>(param.logits) + out_idx, partial);
    if (head_group == 0) {
      static_cast<int32_t*>(param.candidate_indices)[out_idx] = token;
    }
  }
}

__global__ void hisa_block_score_indexer_cache_nvfp4(
    const __grid_constant__ NVFP4HISABlockScoreParam param) {
  constexpr uint32_t kHISABlockSize = 128;
  constexpr uint32_t kMaxHeads = 128;
  __shared__ float head_dot[kMaxHeads];
  __shared__ float reduce_buf[256];

  const auto block_id = blockIdx.x;
  const auto row = blockIdx.y;
  if (row >= param.q_rows || block_id >= param.max_blocks) return;
  if (threadIdx.x < kMaxHeads) head_dot[threadIdx.x] = 0.0f;
  __syncthreads();

  const auto batch = static_cast<const int32_t*>(param.token_to_batch_idx)[row];
  const auto prefix_len = static_cast<const int32_t*>(param.seq_lens)[batch];
  const auto block_count =
      (static_cast<uint32_t>(prefix_len) + kHISABlockSize - 1) / kHISABlockSize;
  const auto out_idx = static_cast<int64_t>(row) * param.max_blocks + block_id;
  if (block_id >= block_count) {
    if (threadIdx.x == 0) static_cast<float*>(param.block_scores)[out_idx] = -INFINITY;
    return;
  }

  const auto rep_base =
      (static_cast<int64_t>(batch) * param.max_blocks + block_id) * kIndexerHeadDim;
  for (uint32_t linear = threadIdx.x; linear < param.n_heads * kIndexerHeadDim;
       linear += blockDim.x) {
    const auto head = linear / kIndexerHeadDim;
    const auto dim = linear - head * kIndexerHeadDim;
    const auto kval = static_cast<const float*>(param.reps)[rep_base + dim];
    const auto q_value_ptr =
        static_cast<const uint8_t*>(param.q_values) +
        (static_cast<int64_t>(row) * param.n_heads + head) * kNVFP4ValueBytes;
    const auto q_scale_ptr =
        static_cast<const uint32_t*>(param.q_scales) +
        static_cast<int64_t>(row) * param.n_heads + head;
    const auto qval = load_nvfp4_value(q_value_ptr, q_scale_ptr, dim);
    atomicAdd(&head_dot[head], qval * kval);
  }
  __syncthreads();

  float partial = 0.0f;
  for (uint32_t head = threadIdx.x; head < param.n_heads; head += blockDim.x) {
    const auto weight =
        static_cast<const float*>(param.weights)[static_cast<int64_t>(row) *
                                                     param.n_heads +
                                                 head];
    partial += fmaxf(head_dot[head], 0.0f) * weight;
  }
  reduce_buf[threadIdx.x] = partial;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) static_cast<float*>(param.block_scores)[out_idx] = reduce_buf[0];
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

  for (uint32_t idx = tid; idx < param.topk; idx += blockDim.x) {
    static_cast<int32_t*>(param.topk_indices)[
        static_cast<int64_t>(row) * param.topk + idx] = -1;
  }

  const auto block_count =
      min(static_cast<uint32_t>(static_cast<const int32_t*>(param.block_counts)[row]),
          param.max_blocks);
  if (tid < param.block_topk) selected[tid] = -1;
  __syncthreads();
  if (block_count == 0) return;

  const auto row_block_topk =
      min(static_cast<uint32_t>(
              static_cast<const int32_t*>(param.block_topk_counts)[row]),
          param.block_topk);
  const auto keep = min(row_block_topk, block_count);
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
  static constexpr auto mean_pool_kernel =
      hisa_mean_pool_indexer_cache_nvfp4<IndicesT, kPageSize>;
  static constexpr auto candidate_score_kernel =
      hisa_candidate_score_indexer_cache_nvfp4<IndicesT, kPageSize>;
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
    const auto head_groups = params.n_heads <= 8 ? 1 : div_ceil(params.n_heads, 8u);
    LaunchKernel(
        dim3(params.block_topk * 128, params.q_rows, head_groups),
        256,
        device_.unwrap())(
        candidate_score_kernel, params);
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
