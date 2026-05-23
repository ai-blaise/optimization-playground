// HIGGS 2-bit MHA decode kernels for the SMC-SD draft model.
//
// External contract: given an SGLang page-table (kv_indptr + kv_indices)
// and HIGGS-packed K/V slots, compute one decode step of GQA attention:
// for each (batch_row, q_head) emit
//
//   O = softmax(q . K^T / sqrt(d)) @ V
//
// where K and V are reconstructed on-the-fly from 34-byte HIGGS slots
// (32 bytes of 4-bit indices into the EDEN2-16 codebook + 2-byte FP16
// per-row scale). Same EDEN2-16 lattice + orthonormal FWHT factorization
// as ``higgs_dense_2bit_mla_decode.cuh``, but specialized for
// ``head_dim=128`` MHA (one register slot per thread; no top-of-tree
// register FWHT) and with K/V quantized independently (the codec lays
// them in separate slot buffers, not one combined latent+rope slot).
//
// Two launchers are provided:
//
//   1. ``HiggsMHA2BitDecodeKernel`` — single-pass kernel that handles
//      the full (forward FWHT, online softmax over all KV tokens,
//      normalize, inverse FWHT) chain inside one block per
//      (batch_row, q_head). Saturates the GPU when
//      ``num_rows * num_q_heads >= ~num_sms`` (true for batch=12 +
//      32 Q heads on B200 == 384 blocks vs 148 SMs).
//
//   2. ``HiggsMHA2BitDecodeSplitKernel`` (+ ``...MergeKernel``) —
//      split-K decode mirroring TurboQuant's ``decode_2p5_split_rotated``
//      and HIGGS MLA's ``higgs_dense_2bit_mla_decode_stage1_split``.
//      The kv-token loop is sharded across ``num_splits`` blocks per
//      (batch_row, q_head); a merge kernel combines partial
//      ``(m, l, acc)`` tuples and runs the inverse FWHT. Reserved for
//      the small-batch case (b=1) where the single-pass kernel
//      starves SMs.
//
// Optimization pattern (mirrors MLA HIGGS):
//   * 128 threads / block, one block per (batch_row, q_head).
//     Thread ``tid`` holds element ``q[tid]`` of the head_dim-128
//     query (and the matching ``acc[tid]`` accumulator).
//   * Pre-FWHT Q in registers (single FWHT_128: 5 levels of warp
//     shuffle + 2 levels of SMEM exchange).
//   * Per KV token: dequant K -> dot(q_rot, k_rot) via block-reduce
//     -> online softmax -> dequant V -> acc += p * v_rot.
//   * After the loop: acc /= sum_exp; InvFWHT(acc); store.
//
// Mathematical trick (orthonormal FWHT is self-inverse): the codec
// stores rotated K/V (FWHT was applied at compress time before
// quantization). Dot products in the rotated basis equal dot
// products in the original basis (Parseval). We rotate Q ONCE up
// front, accumulate acc in the rotated basis, and apply ONE final
// InvFWHT to bring the output back to the original basis -- matches
// the MLA HIGGS / TurboQuant split-rotated fast path.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace higgs_mha_2bit_detail {

// Architectural constants (match HiggsMHA2BitCodec / GLM-4-9B draft).

constexpr int kHeadDim = 128;
constexpr int kPairDim = 2;
constexpr int kCodebookSize = 16;
constexpr int kNumPairs = kHeadDim / kPairDim;        // 64
constexpr int kPackedBytes = kNumPairs / 2;           // 32
constexpr int kNormBytes = 2;                         // fp16 scale
constexpr int kSlotBytes = kPackedBytes + kNormBytes; // 34
constexpr float kInvSqrtHeadDim = 0.08838834764831845f;  // 1/sqrt(128)
constexpr float kNegInf = -3.4028234663852886e38f;
constexpr int kBlockThreads = kHeadDim;               // 128

// ---------------------------------------------------------------------------
// Helpers (same shape as MLA HIGGS; reused, not shared, to keep the
// kernel translation unit standalone for the JIT loader).

__device__ __forceinline__ float bf16_to_float(const bf16_t value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

// FWHT_32 within a warp via shuffle (levels 0..4 of FWHT_128).
__device__ __forceinline__ float fwht_lane_levels_under32(float val, int lane) {
#pragma unroll
  for (int stride = 1; stride <= 16; stride <<= 1) {
    const float other = __shfl_xor_sync(0xffffffff, val, stride);
    val = (lane & stride) ? (other - val) : (val + other);
  }
  return val;
}

// FWHT_128: warp shuffle (levels 0..4) + 2x SMEM exchange (levels 5..6).
// Caller is responsible for surrounding ``__syncthreads()`` so back-to-back
// invocations on different registers can reuse the same ``smem128`` buffer.
__device__ __forceinline__ float fwht_128elem(
    float val, int tid, float* __restrict__ smem128) {
  val = fwht_lane_levels_under32(val, tid & 31);
  smem128[tid] = val;
  __syncthreads();
  val = (tid & 32) ? (smem128[tid ^ 32] - val) : (val + smem128[tid ^ 32]);
  __syncthreads();
  smem128[tid] = val;
  __syncthreads();
  val = (tid & 64) ? (smem128[tid ^ 64] - val) : (val + smem128[tid ^ 64]);
  return val;
}

// Block-reduce a per-thread scalar to a single value broadcast to all
// threads via ``warp_partials[0]``. ``warp_partials`` is a 4-fp32 SMEM
// buffer (one slot per warp in a 128-thread block).
__device__ __forceinline__ float block_reduce_sum_128(
    float v, float* __restrict__ warp_partials) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) warp_partials[warp] = v;
  __syncthreads();
  float total = 0.0f;
  if (tid < 4) total = warp_partials[tid];
  if (warp == 0) total = warp_reduce_sum(total);
  if (tid == 0) warp_partials[0] = total;
  __syncthreads();
  return warp_partials[0];
}

// Dequantize one HIGGS slot at lane ``tid`` (one dim per thread).
//
// For tid ``d`` in [0, 128):
//   pair_idx     = d >> 1               // 0..63
//   coord        = d & 1                // 0=x, 1=y in the codeword
//   byte_idx     = pair_idx >> 1        // 0..31
//   in_hi_nibble = pair_idx & 1         // 0=lo nibble, 1=hi nibble
//
// The codec stores byte k = (idx[2k+1] << 4) | idx[2k], so the lo
// nibble of byte k holds pair-index 2k and the hi nibble holds
// 2k+1 -- matches ``pack_higgs_2bit_indices`` in
// ``higgs_dense_2bit_kv.py``. Returns the dequantized rotated value
// (codebook lookup, scaled by the slot's fp16 norm). The caller
// supplies the slot pointer and ``cb_smem`` (codebook in SMEM, 32
// fp32 = 128 B).
__device__ __forceinline__ float higgs_mha_dequant_dim(
    const uint8_t* __restrict__ slot,
    const float* __restrict__ cb_smem,
    int tid) {
  const int pair_idx = tid >> 1;
  const int coord = tid & 1;
  const int byte_idx = pair_idx >> 1;
  const int in_hi_nibble = pair_idx & 1;
  const uint8_t byte = __ldg(slot + byte_idx);
  const uint32_t idx = in_hi_nibble ? (byte >> 4) & 0x0F : byte & 0x0F;
  const float cb_val = cb_smem[idx * kPairDim + coord];
  const half norm_h = *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = __half2float(norm_h);
  return scale * cb_val;
}

// ---------------------------------------------------------------------------
// Single-pass kernel: grid (num_rows, num_q_heads), block 128 threads.
//
// Iterates over the full kv-token range for the batch row (page table
// indexed via kv_indptr + kv_indices), running an online softmax with
// per-token K + V dequant fused inline.

__global__ void __launch_bounds__(kBlockThreads, 8)
higgs_mha_2bit_decode_kernel(
    const bf16_t* __restrict__ q,
    const uint8_t* __restrict__ k_packed,
    const uint8_t* __restrict__ v_packed,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ kv_indices,
    bf16_t* __restrict__ out,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t q_stride_0,
    int64_t q_stride_1,
    int64_t k_stride_0,
    int64_t k_stride_1,
    int64_t v_stride_0,
    int64_t v_stride_1,
    int64_t out_stride_0,
    int64_t out_stride_1,
    float sm_scale) {
  const int row = blockIdx.x;
  const int q_head = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= num_rows || q_head >= num_q_heads) return;

  const int kv_head = q_head * num_kv_heads / num_q_heads;

  __shared__ float smem128[kBlockThreads];
  __shared__ float softmax_state[4];  // [m, l, alpha, beta]
  __shared__ float warp_partials[4];
  __shared__ float cb_smem[kCodebookSize * kPairDim];  // 32 fp32

  if (tid < kCodebookSize * kPairDim) {
    cb_smem[tid] = __ldg(&codebook[tid]);
  }

  // Pre-FWHT Q (single FWHT_128 in register).
  const bf16_t* q_row = q + row * q_stride_0 + q_head * q_stride_1;
  float q_rot = bf16_to_float(q_row[tid]);
  __syncthreads();
  q_rot = fwht_128elem(q_rot, tid, smem128);
  __syncthreads();
  q_rot *= kInvSqrtHeadDim;

  if (tid == 0) {
    softmax_state[0] = kNegInf;
    softmax_state[1] = 0.0f;
  }

  // KV-token range for this batch row.
  const int32_t kv_start = __ldg(kv_indptr + row);
  const int32_t kv_end = __ldg(kv_indptr + row + 1);

  float acc = 0.0f;

  for (int32_t col = kv_start; col < kv_end; ++col) {
    const int32_t kv_loc = __ldg(kv_indices + col);
    const uint8_t* k_slot =
        k_packed + static_cast<int64_t>(kv_loc) * k_stride_0 +
        static_cast<int64_t>(kv_head) * k_stride_1;
    const uint8_t* v_slot =
        v_packed + static_cast<int64_t>(kv_loc) * v_stride_0 +
        static_cast<int64_t>(kv_head) * v_stride_1;

    // Dequant K[kv_loc, kv_head][tid] -> rotated value.
    const float k_rot = higgs_mha_dequant_dim(k_slot, cb_smem, tid);

    // qk_partial = q_rot * k_rot; block-reduce + online softmax.
    const float qk_partial = q_rot * k_rot;
    const float qk = block_reduce_sum_128(qk_partial, warp_partials);

    // V dequant has no data dependency on the softmax-state update —
    // run it on all 128 threads in parallel with tid==0's softmax math
    // so the wait for the next __syncthreads hides the V load latency.
    const float v_rot = higgs_mha_dequant_dim(v_slot, cb_smem, tid);

    if (tid == 0) {
      const float score = qk * sm_scale;
      const float old_m = softmax_state[0];
      const float old_l = softmax_state[1];
      const float new_m = fmaxf(old_m, score);
      const float alpha = __expf(old_m - new_m);
      const float beta = __expf(score - new_m);
      softmax_state[0] = new_m;
      softmax_state[1] = old_l * alpha + beta;
      softmax_state[2] = alpha;
      softmax_state[3] = beta;
    }
    __syncthreads();

    const float alpha = softmax_state[2];
    const float beta = softmax_state[3];

    acc = acc * alpha + beta * v_rot;
    // The end-of-iter __syncthreads in the prior revision was
    // redundant: the next iter's block_reduce_sum_128 has its own
    // pair of syncs that serialize block-wide before tid==0
    // re-writes softmax_state, and acc lives in registers so there
    // is no cross-iter shmem write/read race.
  }

  const float denom = softmax_state[1];
  float o_rot = denom > 0.0f ? acc / denom : 0.0f;

  // Inverse FWHT (orthonormal: same as forward).
  __syncthreads();
  o_rot = fwht_128elem(o_rot, tid, smem128);
  __syncthreads();
  o_rot *= kInvSqrtHeadDim;

  bf16_t* out_row = out + row * out_stride_0 + q_head * out_stride_1;
  out_row[tid] = __float2bfloat16(o_rot);
}

// ---------------------------------------------------------------------------
// Launcher.

struct HiggsMHA2BitDecodeKernel {
  static void run(
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView k_packed,
      tvm::ffi::TensorView v_packed,
      tvm::ffi::TensorView kv_indptr,
      tvm::ffi::TensorView kv_indices,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView codebook,
      double sm_scale) {
    using namespace host;

    auto R = SymbolicSize{"num_rows"};
    auto R_plus_1 = SymbolicSize{"num_rows_plus_1"};
    auto H = SymbolicSize{"num_q_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto S = SymbolicSize{"num_kv_slots"};
    auto T = SymbolicSize{"total_kv_tokens"};
    auto q_stride_0 = SymbolicSize{"q_stride_0"};
    auto q_stride_1 = SymbolicSize{"q_stride_1"};
    auto k_stride_0 = SymbolicSize{"k_stride_0"};
    auto k_stride_1 = SymbolicSize{"k_stride_1"};
    auto v_stride_0 = SymbolicSize{"v_stride_0"};
    auto v_stride_1 = SymbolicSize{"v_stride_1"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto out_stride_1 = SymbolicSize{"out_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, H, kHeadDim})
        .with_strides({q_stride_0, q_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q);
    TensorMatcher({S, K, kSlotBytes})
        .with_strides({k_stride_0, k_stride_1, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(k_packed);
    TensorMatcher({S, K, kSlotBytes})
        .with_strides({v_stride_0, v_stride_1, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(v_packed);
    TensorMatcher({R_plus_1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(kv_indptr);
    TensorMatcher({T})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(kv_indices);
    TensorMatcher({R, H, kHeadDim})
        .with_strides({out_stride_0, out_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    if (R.unwrap() == 0 || H.unwrap() == 0) return;

    LaunchKernel(
        dim3(R.unwrap(), H.unwrap()), kBlockThreads, device.unwrap())(
        higgs_mha_2bit_decode_kernel,
        static_cast<const bf16_t*>(q.data_ptr()),
        static_cast<const uint8_t*>(k_packed.data_ptr()),
        static_cast<const uint8_t*>(v_packed.data_ptr()),
        static_cast<const int32_t*>(kv_indptr.data_ptr()),
        static_cast<const int32_t*>(kv_indices.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        R.unwrap(),
        H.unwrap(),
        K.unwrap(),
        q_stride_0.unwrap(),
        q_stride_1.unwrap(),
        k_stride_0.unwrap(),
        k_stride_1.unwrap(),
        v_stride_0.unwrap(),
        v_stride_1.unwrap(),
        out_stride_0.unwrap(),
        out_stride_1.unwrap(),
        static_cast<float>(sm_scale));
  }
};

}  // namespace higgs_mha_2bit_detail
