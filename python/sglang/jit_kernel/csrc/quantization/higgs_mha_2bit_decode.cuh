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
// ``head_dim=128`` MHA and with K/V quantized independently.
//
// v2 design (BLOCK_N=32 token tile):
// - 128 threads / block, one block per (batch_row, q_head).
// - Thread ``tid`` holds element ``q[tid]`` of the head_dim-128 query
//   and the matching ``acc[tid]`` accumulator.
// - Pre-FWHT Q in registers (single FWHT_128: 5 levels of warp shuffle
//   + 2 levels of SMEM exchange).
// - Per BLOCK_N=32-token tile:
//   * Stream-dequant K[0..tile_n) per token; accumulate per-token
//     qk_partial[n] = q_rot * k_rot[tid] in registers.
//   * One block-reduce_sum over qk_partial[0..32) -> qk_smem[0..32)
//     (3 syncs: 1 in the warp_partials write, 1 after warp-0 reduce,
//     and 1 trailing for the broadcast).
//   * Single-thread online softmax: update m, l; compute p[0..tile_n)
//     in qk_smem; stash alpha in a dedicated slot.
//   * Sync to broadcast alpha + p[].
//   * Stream-dequant V[0..tile_n) per token; acc += p[n] * v_rot[tid].
//   * No trailing sync; next tile's block_reduce serializes
//     softmax_state writes for us.
// - After last tile: acc /= l; InvFWHT_128(acc); store.
//
// Mathematical trick (orthonormal FWHT is self-inverse): the codec
// stores rotated K/V (FWHT was applied at compress time before
// quantization). Dot products in the rotated basis equal dot products
// in the original basis (Parseval). We rotate Q once up front,
// accumulate acc in the rotated basis, and apply one final InvFWHT
// to bring the output back to the original basis -- matches the
// MLA HIGGS dense fast path.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace higgs_mha_2bit_detail {

constexpr int kHeadDim = 128;
constexpr int kPairDim = 2;
constexpr int kCodebookSize = 16;
constexpr int kNumPairs = kHeadDim / kPairDim;        // 64
constexpr int kPackedBytes = kNumPairs / 2;           // 32
constexpr int kNormBytes = 2;
constexpr int kSlotBytes = kPackedBytes + kNormBytes; // 34
constexpr float kInvSqrtHeadDim = 0.08838834764831845f;  // 1/sqrt(128)
constexpr float kNegInf = -3.4028234663852886e38f;
constexpr int kBlockThreads = kHeadDim;               // 128
constexpr int kBlockN = 32;                           // KV tokens per tile
constexpr int kWarpSize = 32;

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

// Block-reduce kBlockN per-thread fp32 vectors into shmem.
// Each of the 4 warps does warp-reduce on each of the kBlockN slots,
// then a single warp folds the 4 partials.
__device__ __forceinline__ void block_reduce_sum_n_into(
    const float partial[kBlockN], int tid,
    float* __restrict__ warp_partials,  // [4 * kBlockN] fp32
    float* __restrict__ result          // [kBlockN] fp32
) {
  const int lane = tid & 31;
  const int warp = tid >> 5;
#pragma unroll
  for (int n = 0; n < kBlockN; ++n) {
    float v = warp_reduce_sum(partial[n]);
    if (lane == 0) warp_partials[warp * kBlockN + n] = v;
  }
  __syncthreads();
  if (warp == 0) {
    for (int n = lane; n < kBlockN; n += kWarpSize) {
      const float s = warp_partials[0 * kBlockN + n]
                    + warp_partials[1 * kBlockN + n]
                    + warp_partials[2 * kBlockN + n]
                    + warp_partials[3 * kBlockN + n];
      result[n] = s;
    }
  }
  __syncthreads();
}

// Dequant one HIGGS slot at lane ``tid`` -> rotated value for dim tid.
// Codec layout: byte k holds idx[2k] in low nibble, idx[2k+1] in high
// nibble. Flat layout: dim 2j+0 = pair[j].x, dim 2j+1 = pair[j].y.
__device__ __forceinline__ float dequant_slot_dim(
    const uint8_t* __restrict__ slot,
    const float* __restrict__ cb_smem,
    int tid) {
  const int pair_idx = tid >> 1;
  const int coord = tid & 1;
  const int byte_idx = pair_idx >> 1;
  const int in_hi = pair_idx & 1;
  const uint8_t byte = __ldg(slot + byte_idx);
  const uint32_t idx = in_hi ? (byte >> 4) & 0x0F : byte & 0x0F;
  const half norm_h = *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = __half2float(norm_h);
  return scale * cb_smem[idx * kPairDim + coord];
}

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

  __shared__ float smem128[kBlockThreads];                   // FWHT scratch
  __shared__ float qk_smem[kBlockN];                         // [n] -> qk then p
  __shared__ float warp_partials_n[4 * kBlockN];             // block-reduce
  __shared__ float cb_smem[kCodebookSize * kPairDim];        // codebook
  __shared__ float softmax_state[3];                         // [m, l, alpha]

  if (tid < kCodebookSize * kPairDim) {
    cb_smem[tid] = __ldg(&codebook[tid]);
  }

  // Pre-FWHT Q in registers.
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

  const int32_t kv_start = __ldg(kv_indptr + row);
  const int32_t kv_end = __ldg(kv_indptr + row + 1);
  const int32_t kv_len = kv_end - kv_start;

  float acc = 0.0f;

  for (int32_t tile_off = 0; tile_off < kv_len; tile_off += kBlockN) {
    const int tile_n = min(kBlockN, kv_len - tile_off);

    // Stream-dequant K[0..tile_n) and accumulate per-token qk_partial[n]
    // in registers. No cross-token sync; each thread sees its own dim.
    float qk_partial[kBlockN];
#pragma unroll
    for (int n = 0; n < kBlockN; ++n) qk_partial[n] = 0.0f;

    for (int n = 0; n < tile_n; ++n) {
      const int32_t loc = __ldg(kv_indices + kv_start + tile_off + n);
      const uint8_t* k_slot =
          k_packed + static_cast<int64_t>(loc) * k_stride_0
                   + static_cast<int64_t>(kv_head) * k_stride_1;
      const float k_rot = dequant_slot_dim(k_slot, cb_smem, tid);
      qk_partial[n] = q_rot * k_rot;
    }

    // Block-reduce qk_partial -> qk_smem[n] for each n in tile.
    block_reduce_sum_n_into(qk_partial, tid, warp_partials_n, qk_smem);

    // Single-thread online softmax. Updates m, l, computes p[0..tile_n)
    // in qk_smem; stashes alpha in softmax_state[2].
    if (tid == 0) {
      const float m_old = softmax_state[0];
      const float l_old = softmax_state[1];
      float m_new = m_old;
      for (int n = 0; n < tile_n; ++n) {
        const float s = qk_smem[n] * sm_scale;
        qk_smem[n] = s;
        if (s > m_new) m_new = s;
      }
      const float alpha = __expf(m_old - m_new);
      float l_new = l_old * alpha;
      for (int n = 0; n < tile_n; ++n) {
        const float p = __expf(qk_smem[n] - m_new);
        qk_smem[n] = p;
        l_new += p;
      }
      softmax_state[0] = m_new;
      softmax_state[1] = l_new;
      softmax_state[2] = alpha;
    }
    __syncthreads();

    const float alpha = softmax_state[2];
    acc *= alpha;

    // Stream-dequant V[0..tile_n) and accumulate acc += p[n] * v_rot.
    for (int n = 0; n < tile_n; ++n) {
      const int32_t loc = __ldg(kv_indices + kv_start + tile_off + n);
      const uint8_t* v_slot =
          v_packed + static_cast<int64_t>(loc) * v_stride_0
                   + static_cast<int64_t>(kv_head) * v_stride_1;
      const float v_rot = dequant_slot_dim(v_slot, cb_smem, tid);
      acc += qk_smem[n] * v_rot;
    }
    // No trailing sync: the next tile's block_reduce has its own syncs
    // that serialize before the next softmax_state update.
  }

  const float denom = softmax_state[1];
  float o_rot = denom > 0.0f ? acc / denom : 0.0f;

  __syncthreads();
  o_rot = fwht_128elem(o_rot, tid, smem128);
  __syncthreads();
  o_rot *= kInvSqrtHeadDim;

  bf16_t* out_row = out + row * out_stride_0 + q_head * out_stride_1;
  out_row[tid] = __float2bfloat16(o_rot);
}

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
        .with_dtype<bf16_t>().with_device(device).verify(q);
    TensorMatcher({S, K, kSlotBytes})
        .with_strides({k_stride_0, k_stride_1, 1})
        .with_dtype<uint8_t>().with_device(device).verify(k_packed);
    TensorMatcher({S, K, kSlotBytes})
        .with_strides({v_stride_0, v_stride_1, 1})
        .with_dtype<uint8_t>().with_device(device).verify(v_packed);
    TensorMatcher({R_plus_1}).with_dtype<int32_t>().with_device(device).verify(kv_indptr);
    TensorMatcher({T}).with_dtype<int32_t>().with_device(device).verify(kv_indices);
    TensorMatcher({R, H, kHeadDim})
        .with_strides({out_stride_0, out_stride_1, 1})
        .with_dtype<bf16_t>().with_device(device).verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>().with_device(device).verify(codebook);

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
        R.unwrap(), H.unwrap(), K.unwrap(),
        q_stride_0.unwrap(), q_stride_1.unwrap(),
        k_stride_0.unwrap(), k_stride_1.unwrap(),
        v_stride_0.unwrap(), v_stride_1.unwrap(),
        out_stride_0.unwrap(), out_stride_1.unwrap(),
        static_cast<float>(sm_scale));
  }
};

}  // namespace higgs_mha_2bit_detail
