// HIGGS 2-bit MHA/GQA KV materialization kernel.
//
// Store-time packing uses the Triton packer so it shares the same verified
// FWHT layout as fused HIGGS draft decode. This CUDA file provides only the
// fallback materialization path for non-fused attention backends.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace higgs_mha_2bit_kv_detail {

constexpr int kHeadDim = 128;
constexpr int kPairDim = 2;
constexpr int kCodebookSize = 16;
constexpr int kNumPairs = kHeadDim / kPairDim;
constexpr int kPackedBytes = kNumPairs / 2;
constexpr int kNormBytes = 2;
constexpr int kSlotBytes = kPackedBytes + kNormBytes;
constexpr float kInvSqrtHeadDim = 0.08838834764831845f;
constexpr int kBlockThreads = kHeadDim;

__device__ __forceinline__ float fwht_128(float value, float* __restrict__ scratch) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int len = 1; len < 32; len <<= 1) {
    const float other = __shfl_xor_sync(0xffffffff, value, len);
    value = (tid & len) ? other - value : value + other;
  }
  scratch[tid] = value;
  __syncthreads();
#pragma unroll
  for (int len = 32; len < kHeadDim; len <<= 1) {
    const int wht_group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = wht_group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = scratch[a];
    const float y = scratch[b];
    __syncthreads();
    if (pos < len) {
      scratch[a] = x + y;
      scratch[b] = x - y;
    }
    __syncthreads();
  }
  return scratch[tid];
}

__global__ void __launch_bounds__(kBlockThreads, 8)
higgs_mha_2bit_dequant_kernel(
    const uint8_t* __restrict__ packed,
    bf16_t* __restrict__ out,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t num_heads,
    int64_t packed_stride_0,
    int64_t packed_stride_1,
    int64_t out_stride_0,
    int64_t out_stride_1) {
  const int64_t row = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= num_rows || head >= num_heads) return;

  const uint8_t* slot = packed + row * packed_stride_0 + head * packed_stride_1;
  const int pair_idx = tid >> 1;
  const int coord = tid & 1;
  const int byte_idx = pair_idx >> 1;
  const uint8_t byte = __ldg(slot + byte_idx);
  const uint32_t cb_idx = (pair_idx & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
  const half scale_h = *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = __half2float(scale_h);
  const float xrot = scale * __ldg(&codebook[cb_idx * kPairDim + coord]);

  __shared__ float fwht_scratch[kHeadDim];
  const float x = fwht_128(xrot, fwht_scratch) * kInvSqrtHeadDim;
  bf16_t* out_row = out + row * out_stride_0 + head * out_stride_1;
  out_row[tid] = __float2bfloat16(x);
}

struct HiggsMHA2BitDequantKernel {
  static void run(
      tvm::ffi::TensorView packed,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView codebook) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto H = SymbolicSize{"num_heads"};
    auto packed_stride_0 = SymbolicSize{"packed_stride_0"};
    auto packed_stride_1 = SymbolicSize{"packed_stride_1"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto out_stride_1 = SymbolicSize{"out_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, H, kSlotBytes})
        .with_strides({packed_stride_0, packed_stride_1, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(packed);
    TensorMatcher({S, H, kHeadDim})
        .with_strides({out_stride_0, out_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    if (S.unwrap() == 0 || H.unwrap() == 0) return;

    LaunchKernel(dim3(S.unwrap(), H.unwrap()), kBlockThreads, device.unwrap())(
        higgs_mha_2bit_dequant_kernel,
        static_cast<const uint8_t*>(packed.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        S.unwrap(),
        H.unwrap(),
        packed_stride_0.unwrap(),
        packed_stride_1.unwrap(),
        out_stride_0.unwrap(),
        out_stride_1.unwrap());
  }
};

}  // namespace higgs_mha_2bit_kv_detail
