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
// v4 design (BLOCK_H=16 Q heads, BLOCK_N=32 KV tokens, split-K):
// - Stage 1 ``stage1_split_kernel``: grid (num_rows, num_kv_heads,
//   num_splits). 128 threads. Each block processes kBlockH = 16 Q
//   heads sharing one kv_head over a 1/num_splits slice of kv_len.
//   Writes per-head (m, l, acc[head_dim]) for this split to ``mid``.
// - Stage 2 ``stage2_merge_kernel``: grid (num_rows, num_q_heads).
//   128 threads. Each block merges num_splits partials for one
//   (row, q_head), normalizes, applies InvFWHT_128, writes BF16 output.
//
// Why split-K: with BLOCK_H = kv_group = 16 we get 1 block per
// (row, kv_head). For B = 12 batch + num_kv_heads = 2 that's only
// 24 blocks on 148 SMs. Splitting kv_len into N pieces multiplies
// block count by N, restoring SM occupancy. Matches Triton's
// kv_splits = 8 default.
//
// Mathematical trick (orthonormal FWHT is self-inverse): the codec
// stores rotated K/V (FWHT was applied at compress time before
// quantization). Dot products in the rotated basis equal dot products
// in the original basis (Parseval). We rotate Q once up front per Q
// head, accumulate acc in the rotated basis, and apply one final
// InvFWHT_128 in stage 2 to bring the output back to the original basis.

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
constexpr int kBlockThreads = 128;
constexpr int kBlockH = 16;                           // Q heads per block
constexpr int kBlockN = 32;                           // KV tokens per tile
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kBlockThreads / kWarpSize;  // 4
constexpr int kHeadsPerWarp = kBlockH / kNumWarps;    // 4

// SMEM row stride for `[row][kHeadDim]` BF16 layouts. With raw 128-BF16
// stride the bank offset between rows is 64 four-byte words = 0 mod 32,
// so all 32 lanes reading row=lane hit the same bank (32-way conflict).
// Padding by 2 BF16 (4 bytes = 1 word) makes bank stride 65 mod 32 = 1
// (coprime to 32) -> conflict free.
constexpr int kSmemRowStrideBF16 = kHeadDim + 2;      // 130 BF16 = 260 B
// FP32 acc_smem: stride 128 fp32 = 128 words mod 32 = 0 (32-way conflict
// when lane reads acc[h][lane]). Pad +1 fp32 -> stride 129 mod 32 = 1
// (coprime) -> conflict free.
constexpr int kSmemRowStrideF32 = kHeadDim + 1;       // 129 fp32 = 516 B

// Per-(row, kv_head, split, Q-head) partial: 2 floats (m, l) + 128
// floats (acc). Layout in the mid tensor:
//   mid[row, kv_head, split, Q-head_in_block, 0]      = m
//   mid[row, kv_head, split, Q-head_in_block, 1]      = l
//   mid[row, kv_head, split, Q-head_in_block, 2..129] = acc
constexpr int kPartialPerHead = 2 + kHeadDim;

__device__ __forceinline__ float bf16_to_float(const bf16_t value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float fwht_lane_levels_under32(float val, int lane) {
#pragma unroll
  for (int stride = 1; stride <= 16; stride <<= 1) {
    const float other = __shfl_xor_sync(0xffffffff, val, stride);
    val = (lane & stride) ? (other - val) : (val + other);
  }
  return val;
}

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

// FWHT_128 on 4 vectors per thread. Same syncs as the scalar
// fwht_128elem (3 __syncthreads) but processes 4x the data per call
// via ILP. Used for prologue when we have multiple Q heads to rotate.
__device__ __forceinline__ void fwht_128elem_x4(
    float* __restrict__ v,  // 4 input vals, replaced in-place
    int tid,
    float* __restrict__ smem512  // [4][128] fp32 scratch
) {
  const int lane = tid & 31;
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    v[b] = fwht_lane_levels_under32(v[b], lane);
  }
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    smem512[b * kBlockThreads + tid] = v[b];
  }
  __syncthreads();
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const float other = smem512[b * kBlockThreads + (tid ^ 32)];
    v[b] = (tid & 32) ? (other - v[b]) : (v[b] + other);
  }
  __syncthreads();   // matches scalar fwht_128elem level-5 -> level-6 sync
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    smem512[b * kBlockThreads + tid] = v[b];
  }
  __syncthreads();
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const float other = smem512[b * kBlockThreads + (tid ^ 64)];
    v[b] = (tid & 64) ? (other - v[b]) : (v[b] + other);
  }
}

__device__ __forceinline__ float dequant_slot_dim(
    const uint8_t* __restrict__ slot,
    const float* __restrict__ cb_smem,
    int dim) {
  const int pair_idx = dim >> 1;
  const int coord = dim & 1;
  const int byte_idx = pair_idx >> 1;
  const int in_hi = pair_idx & 1;
  const uint8_t byte = __ldg(slot + byte_idx);
  const uint32_t idx = in_hi ? (byte >> 4) & 0x0F : byte & 0x0F;
  const half norm_h = *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = __half2float(norm_h);
  return scale * cb_smem[idx * kPairDim + coord];
}

// Stage 1: per (row, kv_head, split) compute online softmax over the
// split's slice of kv_len for kBlockH Q heads. Write (m, l, acc[D])
// in FP32 to ``mid`` for each Q head.
__global__ void __launch_bounds__(kBlockThreads, 4)
higgs_mha_2bit_decode_stage1_split_kernel(
    const bf16_t* __restrict__ q,
    const uint8_t* __restrict__ k_packed,
    const uint8_t* __restrict__ v_packed,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ kv_indices,
    float* __restrict__ mid,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t num_splits,
    int64_t q_stride_0,
    int64_t q_stride_1,
    int64_t k_stride_0,
    int64_t k_stride_1,
    int64_t v_stride_0,
    int64_t v_stride_1,
    int64_t mid_stride_0,
    int64_t mid_stride_1,
    int64_t mid_stride_2,
    int64_t mid_stride_3,
    float sm_scale) {
  const int row = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split = blockIdx.z;
  const int tid = threadIdx.x;
  if (row >= num_rows || kv_head >= num_kv_heads || split >= num_splits) return;

  const int kv_group = static_cast<int>(num_q_heads / num_kv_heads);
  const int q_head_base = kv_head * kv_group;
  const int active_h = kv_group < kBlockH ? kv_group : kBlockH;

  const int32_t kv_start_row = __ldg(kv_indptr + row);
  const int32_t kv_end_row = __ldg(kv_indptr + row + 1);
  const int32_t kv_len = kv_end_row - kv_start_row;
  const int32_t chunk = (kv_len + static_cast<int32_t>(num_splits) - 1)
                      / static_cast<int32_t>(num_splits);
  const int32_t split_begin = static_cast<int32_t>(split) * chunk;
  const int32_t split_end = min(split_begin + chunk, kv_len);

  float* mid_split = mid + row * mid_stride_0
                         + kv_head * mid_stride_1
                         + split * mid_stride_2;

  __shared__ float smem128[kBlockThreads];
  __shared__ float smem512[4 * kBlockThreads];                     // 4-way FWHT scratch (2 KB)
  __shared__ float cb_smem[kCodebookSize * kPairDim];
  __shared__ bf16_t q_smem[kBlockH][kSmemRowStrideBF16];           // 4.2 KB
  __shared__ bf16_t k_tile_smem[kBlockN][kSmemRowStrideBF16];      // 8.3 KB
  __shared__ bf16_t v_tile_smem[kBlockN][kSmemRowStrideBF16];      // 8.3 KB
  __shared__ float  qk_smem[kBlockH][kBlockN];                     // 2 KB
  __shared__ float  softmax_state[kBlockH][3];                     // m, l, alpha
  __shared__ float  acc_smem[kBlockH][kSmemRowStrideF32];          // 8.25 KB

  if (tid < kCodebookSize * kPairDim) {
    cb_smem[tid] = __ldg(&codebook[tid]);
  }

  // Initialize accumulator + softmax state.
#pragma unroll
  for (int h = 0; h < kBlockH; ++h) acc_smem[h][tid] = 0.0f;
  if (tid < kBlockH) {
    softmax_state[tid][0] = kNegInf;
    softmax_state[tid][1] = 0.0f;
  }

  // Empty split (split_begin >= split_end): emit neutral partials and exit.
  if (split_begin >= split_end) {
    if (tid < kBlockH) {
      float* p = mid_split + tid * mid_stride_3;
      p[0] = kNegInf;
      p[1] = 0.0f;
    }
    for (int h = 0; h < kBlockH; ++h) {
      float* p = mid_split + h * mid_stride_3;
      p[2 + tid] = 0.0f;
    }
    return;
  }

  // Pre-FWHT all kBlockH Q heads into q_smem (scalar one at a time).
  __syncthreads();
  for (int h = 0; h < kBlockH; ++h) {
    const int q_head = q_head_base + h;
    float qv = 0.0f;
    if (h < active_h) {
      const bf16_t* q_row = q + row * q_stride_0 + q_head * q_stride_1;
      qv = bf16_to_float(q_row[tid]);
    }
    qv = fwht_128elem(qv, tid, smem128);
    __syncthreads();
    qv *= kInvSqrtHeadDim;
    q_smem[h][tid] = __float2bfloat16(qv);
  }

  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  for (int32_t tile_off = split_begin; tile_off < split_end; tile_off += kBlockN) {
    const int tile_n = min(kBlockN, split_end - tile_off);

    // Cooperative dequant K_tile -> k_tile_smem[n][tid] BF16.
#pragma unroll 4
    for (int n = 0; n < kBlockN; ++n) {
      float kv = 0.0f;
      if (n < tile_n) {
        const int32_t loc = __ldg(kv_indices + kv_start_row + tile_off + n);
        const uint8_t* k_slot =
            k_packed + static_cast<int64_t>(loc) * k_stride_0
                     + static_cast<int64_t>(kv_head) * k_stride_1;
        kv = dequant_slot_dim(k_slot, cb_smem, tid);
      }
      k_tile_smem[n][tid] = __float2bfloat16(kv);
    }
    __syncthreads();

    // q @ K^T: warp w owns rows h_base..h_base+kHeadsPerWarp.
    // Lane l owns col n = lane.
    {
      const int h_base = warp_id * kHeadsPerWarp;
      const int n = lane;
      float qk[kHeadsPerWarp];
#pragma unroll
      for (int i = 0; i < kHeadsPerWarp; ++i) qk[i] = 0.0f;
#pragma unroll 8
      for (int k = 0; k < kHeadDim; ++k) {
        const float kv = bf16_to_float(k_tile_smem[n][k]);
#pragma unroll
        for (int i = 0; i < kHeadsPerWarp; ++i) {
          const float qv = bf16_to_float(q_smem[h_base + i][k]);
          qk[i] += qv * kv;
        }
      }
#pragma unroll
      for (int i = 0; i < kHeadsPerWarp; ++i) {
        qk_smem[h_base + i][n] = qk[i];
      }
    }
    __syncthreads();

    // Per-head online softmax + V_tile dequant interleaved. The softmax
    // only uses 16 threads (4 per warp); the other 112 spin on the V
    // dequant gmem loads so memory latency is masked.
    {
      const int h_base = warp_id * kHeadsPerWarp;
      if (lane < kHeadsPerWarp) {
        const int h = h_base + lane;
        if (h < active_h) {
          const float m_old = softmax_state[h][0];
          const float l_old = softmax_state[h][1];
          float m_new = m_old;
          for (int n = 0; n < tile_n; ++n) {
            const float s = qk_smem[h][n] * sm_scale;
            qk_smem[h][n] = s;
            if (s > m_new) m_new = s;
          }
          const float alpha = __expf(m_old - m_new);
          float l_new = l_old * alpha;
          for (int n = 0; n < tile_n; ++n) {
            const float p = __expf(qk_smem[h][n] - m_new);
            qk_smem[h][n] = p;
            l_new += p;
          }
          softmax_state[h][0] = m_new;
          softmax_state[h][1] = l_new;
          softmax_state[h][2] = alpha;
        } else {
          softmax_state[h][2] = 1.0f;
          for (int n = 0; n < tile_n; ++n) qk_smem[h][n] = 0.0f;
        }
      }
      // All 128 threads do V_tile dequant in parallel with the softmax.
      // Each thread owns dim=tid across BLOCK_N tokens.
#pragma unroll 4
      for (int n = 0; n < kBlockN; ++n) {
        float vv = 0.0f;
        if (n < tile_n) {
          const int32_t loc = __ldg(kv_indices + kv_start_row + tile_off + n);
          const uint8_t* v_slot =
              v_packed + static_cast<int64_t>(loc) * v_stride_0
                       + static_cast<int64_t>(kv_head) * v_stride_1;
          vv = dequant_slot_dim(v_slot, cb_smem, tid);
        }
        v_tile_smem[n][tid] = __float2bfloat16(vv);
      }
    }
    __syncthreads();

    // acc += p @ V. Thread d owns dim=d=tid across all 16 Q heads.
    {
      const int d = tid;
#pragma unroll
      for (int h = 0; h < kBlockH; ++h) {
        const float alpha = softmax_state[h][2];
        float acc_h = acc_smem[h][d] * alpha;
#pragma unroll 4
        for (int n = 0; n < tile_n; ++n) {
          acc_h += qk_smem[h][n] * bf16_to_float(v_tile_smem[n][d]);
        }
        acc_smem[h][d] = acc_h;
      }
    }
    __syncthreads();
  }

  // Emit partials.
  if (tid < kBlockH) {
    float* p = mid_split + tid * mid_stride_3;
    p[0] = softmax_state[tid][0];
    p[1] = softmax_state[tid][1];
  }
#pragma unroll
  for (int h = 0; h < kBlockH; ++h) {
    float* p = mid_split + h * mid_stride_3;
    p[2 + tid] = acc_smem[h][tid];
  }
}

// Stage 2: merge num_splits partials for one (row, q_head). Apply
// InvFWHT_128, normalize, write BF16 output.
__global__ void __launch_bounds__(kBlockThreads, 8)
higgs_mha_2bit_decode_stage2_kernel(
    const float* __restrict__ mid,
    bf16_t* __restrict__ out,
    int64_t num_rows,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t num_splits,
    int64_t mid_stride_0,
    int64_t mid_stride_1,
    int64_t mid_stride_2,
    int64_t mid_stride_3,
    int64_t out_stride_0,
    int64_t out_stride_1) {
  const int row = blockIdx.x;
  const int q_head = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= num_rows || q_head >= num_q_heads) return;

  const int kv_group = static_cast<int>(num_q_heads / num_kv_heads);
  const int kv_head = q_head / kv_group;
  const int h_in_block = q_head - kv_head * kv_group;
  if (h_in_block >= kBlockH) return;

  __shared__ float smem128[kBlockThreads];

  const float* base = mid + row * mid_stride_0 + kv_head * mid_stride_1
                          + h_in_block * mid_stride_3;

  // Pass 1: find global m_max across splits.
  float m_max = kNegInf;
  for (int s = 0; s < num_splits; ++s) {
    const float m_s = base[s * mid_stride_2 + 0];
    if (m_s > m_max) m_max = m_s;
  }

  // Pass 2: rescaled sum over splits.
  float l_tot = 0.0f;
  float acc = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const float* sb = base + s * mid_stride_2;
    const float m_s = sb[0];
    const float l_s = sb[1];
    const float scale = (l_s > 0.0f) ? __expf(m_s - m_max) : 0.0f;
    l_tot += l_s * scale;
    acc += sb[2 + tid] * scale;
  }

  const bool ok = l_tot > 0.0f;
  float o = ok ? acc / l_tot : 0.0f;

  __syncthreads();
  o = fwht_128elem(o, tid, smem128);
  __syncthreads();
  o *= kInvSqrtHeadDim;

  bf16_t* out_row = out + row * out_stride_0 + q_head * out_stride_1;
  out_row[tid] = __float2bfloat16(o);
}

// Single Python entry: takes a scratch ``mid`` tensor shape
// (num_rows, num_kv_heads, num_splits, kBlockH, 2 + kHeadDim) F32,
// runs stage 1 then stage 2.
struct HiggsMHA2BitDecodeKernel {
  static void run(
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView k_packed,
      tvm::ffi::TensorView v_packed,
      tvm::ffi::TensorView kv_indptr,
      tvm::ffi::TensorView kv_indices,
      tvm::ffi::TensorView mid,
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
    auto P = SymbolicSize{"num_splits"};
    auto q_stride_0 = SymbolicSize{"q_stride_0"};
    auto q_stride_1 = SymbolicSize{"q_stride_1"};
    auto k_stride_0 = SymbolicSize{"k_stride_0"};
    auto k_stride_1 = SymbolicSize{"k_stride_1"};
    auto v_stride_0 = SymbolicSize{"v_stride_0"};
    auto v_stride_1 = SymbolicSize{"v_stride_1"};
    auto mid_stride_0 = SymbolicSize{"mid_stride_0"};
    auto mid_stride_1 = SymbolicSize{"mid_stride_1"};
    auto mid_stride_2 = SymbolicSize{"mid_stride_2"};
    auto mid_stride_3 = SymbolicSize{"mid_stride_3"};
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
    TensorMatcher({R, K, P, kBlockH, 2 + kHeadDim})
        .with_strides({mid_stride_0, mid_stride_1, mid_stride_2, mid_stride_3, 1})
        .with_dtype<float>().with_device(device).verify(mid);
    TensorMatcher({R, H, kHeadDim})
        .with_strides({out_stride_0, out_stride_1, 1})
        .with_dtype<bf16_t>().with_device(device).verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>().with_device(device).verify(codebook);

    if (R.unwrap() == 0 || H.unwrap() == 0 || P.unwrap() == 0) return;

    LaunchKernel(
        dim3(R.unwrap(), K.unwrap(), P.unwrap()),
        kBlockThreads, device.unwrap())(
        higgs_mha_2bit_decode_stage1_split_kernel,
        static_cast<const bf16_t*>(q.data_ptr()),
        static_cast<const uint8_t*>(k_packed.data_ptr()),
        static_cast<const uint8_t*>(v_packed.data_ptr()),
        static_cast<const int32_t*>(kv_indptr.data_ptr()),
        static_cast<const int32_t*>(kv_indices.data_ptr()),
        static_cast<float*>(mid.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        R.unwrap(), H.unwrap(), K.unwrap(), P.unwrap(),
        q_stride_0.unwrap(), q_stride_1.unwrap(),
        k_stride_0.unwrap(), k_stride_1.unwrap(),
        v_stride_0.unwrap(), v_stride_1.unwrap(),
        mid_stride_0.unwrap(), mid_stride_1.unwrap(),
        mid_stride_2.unwrap(), mid_stride_3.unwrap(),
        static_cast<float>(sm_scale));

    LaunchKernel(
        dim3(R.unwrap(), H.unwrap()),
        kBlockThreads, device.unwrap())(
        higgs_mha_2bit_decode_stage2_kernel,
        static_cast<const float*>(mid.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        R.unwrap(), H.unwrap(), K.unwrap(), P.unwrap(),
        mid_stride_0.unwrap(), mid_stride_1.unwrap(),
        mid_stride_2.unwrap(), mid_stride_3.unwrap(),
        out_stride_0.unwrap(), out_stride_1.unwrap());
  }
};

}  // namespace higgs_mha_2bit_detail
