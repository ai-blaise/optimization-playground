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

template <typename KeyT, typename IndicesT, uint32_t kPageSize,
          bool kUsePDL>
struct NVFP4IndexerQuantKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = (kNVFP4ValueBytes + kScaleBytes) * kPageSize;
  static constexpr auto store_kernel =
      fused_store_indexer_cache_nvfp4<KeyT, IndicesT, kLogSize, kUsePDL>;
  static constexpr auto q_kernel = quantize_indexer_q_nvfp4<KeyT>;

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
};

}  // namespace
