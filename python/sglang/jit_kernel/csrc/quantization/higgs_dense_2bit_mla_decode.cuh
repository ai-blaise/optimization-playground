// HIGGS 2-bit dense MLA decode kernels.
//
// External contract: given a page table of slot indices, dot the
// dequantized latent + rope KV against a (BF16) query, apply softmax
// with on-line max-rescale, and emit a BF16 result per (row, head).
//
// Two launchers are provided:
//
//   1. ``HiggsDense2BitMLADecodeKernel`` — single-pass kernel that
//      handles the full (forward FWHT, online softmax over topk slots,
//      normalize, inverse FWHT) chain inside one block per (row, head).
//      Optimal when ``num_rows * num_heads`` already saturates the GPU
//      (typically batch >= 8 on H200).
//
//   2. ``HiggsDense2BitMLADecodeSplitKernel`` — split-K decode that
//      mirrors TurboQuant's ``decode_2p5_split_rotated`` (see
//      ``turboquant_dense_mla_decode.cuh``). The topk dimension is
//      sharded across ``num_splits`` blocks per (row, head), each
//      producing an online-softmax partial ``(m, l, acc)``. A merge
//      kernel combines partials, normalizes, and runs the inverse
//      FWHT. Restores small-batch throughput on H200 (b=1 sees the
//      topk reduction limited by single-block occupancy in the
//      single-pass kernel; ``num_splits=16`` matches TurboQuant's
//      tuned default and recovers ~3-4x at b=1, topk>=2048).
//
// Optimization pattern adopted from the TurboQuant 2.5-bit MLA decode
// kernel that lives next to this file
// (``turboquant_dense_mla_decode.cuh``): 128 threads per block, each
// holding four latent dims as register state. The full FWHT_512
// factors as four parallel FWHT_128s (warp-shuffle for levels 0..4,
// single SMEM exchange for levels 5..6) stitched together by a 4-way
// register FWHT (``fwht_register_top2``). This cuts SMEM traffic and
// lets us keep multiple resident blocks per SM on H200.
//
// Mathematical trick (orthonormal FWHT is self-inverse): we rotate the
// QUERY once into ``q_rot``, keep KV reconstructions in rotated
// coordinates (just ``scale * G[idx]`` per token), accumulate
// ``softmax(q_rot . scale * G[idx]) * scale * G[idx]`` across the topk
// loop, and apply ONE final InvFWHT to the result to bring it back to
// the original basis. This matches TurboQuant's split-rotated fast
// path.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace higgs_dense_2bit_mla_detail {

// Architectural constants.

constexpr int kLatentDim = 512;
constexpr int kRopeDim = 64;
constexpr int kPairDim = 2;
constexpr int kCodebookSize = 16;
constexpr int kNumPairs = kLatentDim / kPairDim;
constexpr int kPackedBytes = kNumPairs / 2;       // 128
constexpr int kNormBytes = 2;
constexpr int kSlotBytes = kPackedBytes + kNormBytes + kRopeDim * 2;  // 258
constexpr float kInvSqrtLatentDim = 0.044194173824159216f;  // 1/sqrt(512)
constexpr float kNegInf = -3.4028234663852886e38f;
constexpr int kBlockThreads = 128;
constexpr int kDimsPerThread = kLatentDim / kBlockThreads;  // 4

// Helpers (adapted from turboquant_dense_mla_decode.cuh).

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

__device__ __forceinline__ float block_reduce_sum_128(
    float v, float* __restrict__ warp_sums) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  float sum = 0.0f;
  if (tid < 4) sum = warp_sums[tid];
  if (warp == 0) sum = warp_reduce_sum(sum);
  if (tid == 0) warp_sums[0] = sum;
  __syncthreads();
  return warp_sums[0];
}

// FWHT_4 on four register values (last two levels of the 512-pt
// transform).
__device__ __forceinline__ void fwht_register_top2(
    float& v0, float& v1, float& v2, float& v3) {
  const float a = v0 + v1;
  const float b = v0 - v1;
  const float c = v2 + v3;
  const float d = v2 - v3;
  v0 = a + c;
  v2 = a - c;
  v1 = b + d;
  v3 = b - d;
}

// FWHT_32 within a warp via shuffle (levels 0..4).
__device__ __forceinline__ float fwht_lane_levels_under32(
    float val, int lane) {
#pragma unroll
  for (int stride = 1; stride <= 16; stride <<= 1) {
    const float other = __shfl_xor_sync(0xffffffff, val, stride);
    val = (lane & stride) ? (other - val) : (val + other);
  }
  return val;
}

// FWHT_128 (levels 0..6) via warp shuffle (intra-warp) + SMEM exchange
// for levels 5..6.
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

// Unpack one HIGGS slot at lane ``tid`` (one quantized pair per
// thread x 4 groups; thread holds (idx0, idx1, idx2, idx3) for the 4
// 32-byte group blocks of the 128-byte payload).
__device__ __forceinline__ void higgs_unpack_indices(
    const uint8_t* __restrict__ slot, int tid,
    uint32_t& i0, uint32_t& i1, uint32_t& i2, uint32_t& i3) {
  const int pair_within_group = tid >> 1;
  const bool coord_lane = tid & 1;
  const int byte_in_group = pair_within_group >> 1;
  const int nibble = pair_within_group & 1;
  uint32_t b0 = 0;
  uint32_t b1 = 0;
  uint32_t b2 = 0;
  uint32_t b3 = 0;
  if (!coord_lane) {
    b0 = __ldg(slot + 0 * 32 + byte_in_group);
    b1 = __ldg(slot + 1 * 32 + byte_in_group);
    b2 = __ldg(slot + 2 * 32 + byte_in_group);
    b3 = __ldg(slot + 3 * 32 + byte_in_group);
  }
  const uint32_t peer_b0 = __shfl_xor_sync(0xffffffff, b0, 1);
  const uint32_t peer_b1 = __shfl_xor_sync(0xffffffff, b1, 1);
  const uint32_t peer_b2 = __shfl_xor_sync(0xffffffff, b2, 1);
  const uint32_t peer_b3 = __shfl_xor_sync(0xffffffff, b3, 1);
  b0 = coord_lane ? peer_b0 : b0;
  b1 = coord_lane ? peer_b1 : b1;
  b2 = coord_lane ? peer_b2 : b2;
  b3 = coord_lane ? peer_b3 : b3;
  i0 = nibble ? (b0 >> 4) : (b0 & 0x0F);
  i1 = nibble ? (b1 >> 4) : (b1 & 0x0F);
  i2 = nibble ? (b2 >> 4) : (b2 & 0x0F);
  i3 = nibble ? (b3 >> 4) : (b3 & 0x0F);
}

// Single-pass kernel: grid (num_rows, num_heads), block 128 threads.
// Handles forward FWHT, online softmax over the full topk, normalize,
// and inverse FWHT inside one block. Used when num_splits == 1.

__global__ void __launch_bounds__(kBlockThreads, 8)
higgs_dense_2bit_mla_decode_kernel(
    const bf16_t* __restrict__ q_nope,
    const bf16_t* __restrict__ q_rope,
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    bf16_t* __restrict__ out,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t num_heads,
    int64_t topk,
    int64_t q_nope_stride_0,
    int64_t q_nope_stride_1,
    int64_t q_rope_stride_0,
    int64_t q_rope_stride_1,
    int64_t compressed_stride_0,
    int64_t page_table_stride_0,
    int64_t out_stride_0,
    int64_t out_stride_1,
    float sm_scale) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= num_rows || head >= num_heads) return;

  __shared__ float smem128[kBlockThreads];
  __shared__ float softmax_state[4];  // [m, l, alpha, beta]
  __shared__ float warp_partials[4];
  __shared__ float cb_smem[kCodebookSize * kPairDim];  // 32 fp32 = 128 B

  if (tid < kCodebookSize * kPairDim) {
    cb_smem[tid] = __ldg(&codebook[tid]);
  }

  // Forward FWHT_512 on q_nope.
  const bf16_t* q_nope_row =
      q_nope + row * q_nope_stride_0 + head * q_nope_stride_1;
  float v0 = bf16_to_float(q_nope_row[0 * 128 + tid]);
  float v1 = bf16_to_float(q_nope_row[1 * 128 + tid]);
  float v2 = bf16_to_float(q_nope_row[2 * 128 + tid]);
  float v3 = bf16_to_float(q_nope_row[3 * 128 + tid]);

  v0 = fwht_128elem(v0, tid, smem128);
  __syncthreads();
  v1 = fwht_128elem(v1, tid, smem128);
  __syncthreads();
  v2 = fwht_128elem(v2, tid, smem128);
  __syncthreads();
  v3 = fwht_128elem(v3, tid, smem128);
  __syncthreads();
  fwht_register_top2(v0, v1, v2, v3);

  v0 *= kInvSqrtLatentDim;
  v1 *= kInvSqrtLatentDim;
  v2 *= kInvSqrtLatentDim;
  v3 *= kInvSqrtLatentDim;

  if (tid == 0) {
    softmax_state[0] = kNegInf;
    softmax_state[1] = 0.0f;
  }
  __syncthreads();

  float q_rope_val = 0.0f;
  if (tid < kRopeDim) {
    const bf16_t* q_rope_row =
        q_rope + row * q_rope_stride_0 + head * q_rope_stride_1;
    q_rope_val = bf16_to_float(q_rope_row[tid]);
  }
  const int32_t* pages = page_table + row * page_table_stride_0;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  for (int64_t col = 0; col < topk; ++col) {
    const int32_t page = __ldg(&pages[col]);
    const bool valid = page >= 0;
    const uint8_t* slot =
        compressed + (valid ? static_cast<int64_t>(page) : 0) *
        compressed_stride_0;

    uint32_t i0, i1, i2, i3;
    higgs_unpack_indices(slot, tid, i0, i1, i2, i3);

    const int coord = tid & 1;
    const float c0 = cb_smem[i0 * kPairDim + coord];
    const float c1 = cb_smem[i1 * kPairDim + coord];
    const float c2 = cb_smem[i2 * kPairDim + coord];
    const float c3 = cb_smem[i3 * kPairDim + coord];

    const half norm_h =
        *reinterpret_cast<const half*>(slot + kPackedBytes);
    const float scale = __half2float(norm_h);

    float val = valid
        ? scale * (v0 * c0 + v1 * c1 + v2 * c2 + v3 * c3)
        : 0.0f;
    if (tid < kRopeDim && valid) {
      const bf16_t* rope =
          reinterpret_cast<const bf16_t*>(slot + kPackedBytes + kNormBytes);
      val += q_rope_val * bf16_to_float(rope[tid]);
    }

    const float warp_sum = warp_reduce_sum(val);
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) warp_partials[warp_id] = warp_sum;
    __syncthreads();

    if (tid == 0) {
      const float total =
          warp_partials[0] + warp_partials[1] +
          warp_partials[2] + warp_partials[3];
      const float score = valid ? total * sm_scale : kNegInf;
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
    const float beta_scaled = softmax_state[3] * scale;
    acc0 = acc0 * alpha + beta_scaled * c0;
    acc1 = acc1 * alpha + beta_scaled * c1;
    acc2 = acc2 * alpha + beta_scaled * c2;
    acc3 = acc3 * alpha + beta_scaled * c3;
  }

  const float denom = softmax_state[1];
  const bool ok = denom > 0.0f;
  v0 = ok ? acc0 / denom : 0.0f;
  v1 = ok ? acc1 / denom : 0.0f;
  v2 = ok ? acc2 / denom : 0.0f;
  v3 = ok ? acc3 / denom : 0.0f;

  fwht_register_top2(v0, v1, v2, v3);
  __syncthreads();
  v0 = fwht_128elem(v0, tid, smem128);
  __syncthreads();
  v1 = fwht_128elem(v1, tid, smem128);
  __syncthreads();
  v2 = fwht_128elem(v2, tid, smem128);
  __syncthreads();
  v3 = fwht_128elem(v3, tid, smem128);
  __syncthreads();

  bf16_t* out_row = out + row * out_stride_0 + head * out_stride_1;
  out_row[0 * 128 + tid] = __float2bfloat16(v0 * kInvSqrtLatentDim);
  out_row[1 * 128 + tid] = __float2bfloat16(v1 * kInvSqrtLatentDim);
  out_row[2 * 128 + tid] = __float2bfloat16(v2 * kInvSqrtLatentDim);
  out_row[3 * 128 + tid] = __float2bfloat16(v3 * kInvSqrtLatentDim);
}

__device__ __forceinline__ float saw_scalar2_value(uint32_t code) {
  code &= 0x3u;
  const uint32_t is_small = ((code + 1u) & 0x2u) != 0u;
  const uint32_t magnitude = is_small ? 0x3ee80000u : 0x3fc10000u;
  const uint32_t sign = (code < 2u) ? 0x80000000u : 0u;
  return __uint_as_float(magnitude | sign);
}


__device__ __forceinline__ void saw_scalar2_unpack_pair_lanes(
    const uint8_t* __restrict__ slot, int tid,
    float& c0, float& c1, float& c2, float& c3) {
  uint32_t i0, i1, i2, i3;
  higgs_unpack_indices(slot, tid, i0, i1, i2, i3);
  const int coord_shift = (tid & 1) << 1;
  c0 = saw_scalar2_value(i0 >> coord_shift);
  c1 = saw_scalar2_value(i1 >> coord_shift);
  c2 = saw_scalar2_value(i2 >> coord_shift);
  c3 = saw_scalar2_value(i3 >> coord_shift);
}

__global__ void __launch_bounds__(kBlockThreads, 8)
higgs_dense_2bit_mla_decode_saw_scalar2_kernel(
    const bf16_t* __restrict__ q_nope,
    const bf16_t* __restrict__ q_rope,
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    bf16_t* __restrict__ out,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t num_heads,
    int64_t topk,
    int64_t q_nope_stride_0,
    int64_t q_nope_stride_1,
    int64_t q_rope_stride_0,
    int64_t q_rope_stride_1,
    int64_t compressed_stride_0,
    int64_t page_table_stride_0,
    int64_t out_stride_0,
    int64_t out_stride_1,
    float sm_scale) {
  (void)codebook;
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= num_rows || head >= num_heads) return;

  __shared__ float softmax_state[4];  // [m, l, alpha, beta]
  __shared__ float warp_partials[4];

  const bf16_t* q_nope_row =
      q_nope + row * q_nope_stride_0 + head * q_nope_stride_1;
  const float q0 = bf16_to_float(q_nope_row[0 * 128 + tid]);
  const float q1 = bf16_to_float(q_nope_row[1 * 128 + tid]);
  const float q2 = bf16_to_float(q_nope_row[2 * 128 + tid]);
  const float q3 = bf16_to_float(q_nope_row[3 * 128 + tid]);

  if (tid == 0) {
    softmax_state[0] = kNegInf;
    softmax_state[1] = 0.0f;
  }
  __syncthreads();

  float q_rope_val = 0.0f;
  if (tid < kRopeDim) {
    const bf16_t* q_rope_row =
        q_rope + row * q_rope_stride_0 + head * q_rope_stride_1;
    q_rope_val = bf16_to_float(q_rope_row[tid]);
  }
  const int32_t* pages = page_table + row * page_table_stride_0;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  for (int64_t col = 0; col < topk; ++col) {
    const int32_t page = __ldg(&pages[col]);
    const bool valid = page >= 0;
    const uint8_t* slot =
        compressed + (valid ? static_cast<int64_t>(page) : 0) *
        compressed_stride_0;

    float c0 = 0.0f;
    float c1 = 0.0f;
    float c2 = 0.0f;
    float c3 = 0.0f;
    if (valid) {
      saw_scalar2_unpack_pair_lanes(slot, tid, c0, c1, c2, c3);
    }

    float val = valid ? (q0 * c0 + q1 * c1 + q2 * c2 + q3 * c3) : 0.0f;
    if (tid < kRopeDim && valid) {
      const bf16_t* rope =
          reinterpret_cast<const bf16_t*>(slot + kPackedBytes + kNormBytes);
      val += q_rope_val * bf16_to_float(rope[tid]);
    }

    const float warp_sum = warp_reduce_sum(val);
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) warp_partials[warp_id] = warp_sum;
    __syncthreads();

    if (tid == 0) {
      const float total =
          warp_partials[0] + warp_partials[1] +
          warp_partials[2] + warp_partials[3];
      const float score = valid ? total * sm_scale : kNegInf;
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
    acc0 = acc0 * alpha + beta * c0;
    acc1 = acc1 * alpha + beta * c1;
    acc2 = acc2 * alpha + beta * c2;
    acc3 = acc3 * alpha + beta * c3;
  }

  const float denom = softmax_state[1];
  const bool ok = denom > 0.0f;
  bf16_t* out_row = out + row * out_stride_0 + head * out_stride_1;
  out_row[0 * 128 + tid] = __float2bfloat16(ok ? acc0 / denom : 0.0f);
  out_row[1 * 128 + tid] = __float2bfloat16(ok ? acc1 / denom : 0.0f);
  out_row[2 * 128 + tid] = __float2bfloat16(ok ? acc2 / denom : 0.0f);
  out_row[3 * 128 + tid] = __float2bfloat16(ok ? acc3 / denom : 0.0f);
}

// Pre-rotate q_nope into a float32 buffer holding FWHT_512(q_nope) *
// kInvSqrtLatentDim. Called once per (row, head) before stage1_split.
// This matches TurboQuant's rotate_query kernel; pre-rotating once and
// having every split read from gmem is cheaper than redoing FWHT_512
// inside each of ``num_splits`` blocks.

__global__ void higgs_dense_2bit_mla_rotate_query_kernel(
    const bf16_t* __restrict__ q_nope,
    float* __restrict__ q_rotated,
    int64_t num_rows,
    int64_t num_heads,
    int64_t q_nope_stride_0,
    int64_t q_nope_stride_1,
    int64_t q_rot_stride_0,
    int64_t q_rot_stride_1) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= num_rows || head >= num_heads) return;

  __shared__ float smem128[kBlockThreads];

  const bf16_t* q_nope_row =
      q_nope + row * q_nope_stride_0 + head * q_nope_stride_1;
  float v0 = bf16_to_float(q_nope_row[0 * 128 + tid]);
  float v1 = bf16_to_float(q_nope_row[1 * 128 + tid]);
  float v2 = bf16_to_float(q_nope_row[2 * 128 + tid]);
  float v3 = bf16_to_float(q_nope_row[3 * 128 + tid]);

  v0 = fwht_128elem(v0, tid, smem128);
  __syncthreads();
  v1 = fwht_128elem(v1, tid, smem128);
  __syncthreads();
  v2 = fwht_128elem(v2, tid, smem128);
  __syncthreads();
  v3 = fwht_128elem(v3, tid, smem128);
  __syncthreads();
  fwht_register_top2(v0, v1, v2, v3);

  float* out_row = q_rotated + row * q_rot_stride_0 + head * q_rot_stride_1;
  out_row[0 * 128 + tid] = v0 * kInvSqrtLatentDim;
  out_row[1 * 128 + tid] = v1 * kInvSqrtLatentDim;
  out_row[2 * 128 + tid] = v2 * kInvSqrtLatentDim;
  out_row[3 * 128 + tid] = v3 * kInvSqrtLatentDim;
}

// Stage 1 of split-K decode: grid (num_rows, num_heads, num_splits).
// Each block processes ``ceil(topk / num_splits)`` slots and writes a
// partial ``(m, l, acc0..acc511)`` into mid[row, head, split, :].
// Layout: mid[..., 0] = m, mid[..., 1] = l, mid[..., 2 + g*128 + tid]
// = acc for that thread's dim in group g.

__global__ void __launch_bounds__(kBlockThreads, 8)
higgs_dense_2bit_mla_decode_stage1_split_kernel(
    const float* __restrict__ q_rotated,
    const bf16_t* __restrict__ q_rope,
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    float* __restrict__ mid,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t num_heads,
    int64_t topk,
    int64_t num_splits,
    int64_t q_rot_stride_0,
    int64_t q_rot_stride_1,
    int64_t q_rope_stride_0,
    int64_t q_rope_stride_1,
    int64_t compressed_stride_0,
    int64_t page_table_stride_0,
    int64_t mid_stride_0,
    int64_t mid_stride_1,
    int64_t mid_stride_2,
    float sm_scale) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int split = blockIdx.z;
  const int tid = threadIdx.x;
  if (row >= num_rows || head >= num_heads || split >= num_splits) return;

  __shared__ float softmax_state[4];
  __shared__ float warp_partials[4];
  __shared__ float cb_smem[kCodebookSize * kPairDim];

  if (tid < kCodebookSize * kPairDim) {
    cb_smem[tid] = __ldg(&codebook[tid]);
  }

  const float* q_rot_row =
      q_rotated + row * q_rot_stride_0 + head * q_rot_stride_1;
  const float v0 = q_rot_row[0 * 128 + tid];
  const float v1 = q_rot_row[1 * 128 + tid];
  const float v2 = q_rot_row[2 * 128 + tid];
  const float v3 = q_rot_row[3 * 128 + tid];

  if (tid == 0) {
    softmax_state[0] = kNegInf;
    softmax_state[1] = 0.0f;
  }
  __syncthreads();

  float q_rope_val = 0.0f;
  if (tid < kRopeDim) {
    const bf16_t* q_rope_row =
        q_rope + row * q_rope_stride_0 + head * q_rope_stride_1;
    q_rope_val = bf16_to_float(q_rope_row[tid]);
  }

  const int64_t chunk = (topk + num_splits - 1) / num_splits;
  const int64_t begin = split * chunk;
  const int64_t end = min(begin + chunk, topk);
  const int32_t* pages = page_table + row * page_table_stride_0;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  for (int64_t col = begin; col < end; ++col) {
    const int32_t page = __ldg(&pages[col]);
    const bool valid = page >= 0;
    const uint8_t* slot =
        compressed + (valid ? static_cast<int64_t>(page) : 0) *
        compressed_stride_0;

    uint32_t i0, i1, i2, i3;
    higgs_unpack_indices(slot, tid, i0, i1, i2, i3);

    const int coord = tid & 1;
    const float c0 = cb_smem[i0 * kPairDim + coord];
    const float c1 = cb_smem[i1 * kPairDim + coord];
    const float c2 = cb_smem[i2 * kPairDim + coord];
    const float c3 = cb_smem[i3 * kPairDim + coord];

    const half norm_h =
        *reinterpret_cast<const half*>(slot + kPackedBytes);
    const float scale = __half2float(norm_h);

    float val = valid
        ? scale * (v0 * c0 + v1 * c1 + v2 * c2 + v3 * c3)
        : 0.0f;
    if (tid < kRopeDim && valid) {
      const bf16_t* rope =
          reinterpret_cast<const bf16_t*>(slot + kPackedBytes + kNormBytes);
      val += q_rope_val * bf16_to_float(rope[tid]);
    }

    const float warp_sum = warp_reduce_sum(val);
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) warp_partials[warp_id] = warp_sum;
    __syncthreads();

    if (tid == 0) {
      const float total =
          warp_partials[0] + warp_partials[1] +
          warp_partials[2] + warp_partials[3];
      const float score = valid ? total * sm_scale : kNegInf;
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
    const float beta_scaled = softmax_state[3] * scale;
    acc0 = acc0 * alpha + beta_scaled * c0;
    acc1 = acc1 * alpha + beta_scaled * c1;
    acc2 = acc2 * alpha + beta_scaled * c2;
    acc3 = acc3 * alpha + beta_scaled * c3;
  }

  // Write partials. Layout matches TurboQuant's stage1_rotated_fast:
  // mid[..., 0] = m, mid[..., 1] = l, mid[..., 2 + g*128 + tid] = acc_g.
  float* mid_row =
      mid + row * mid_stride_0 + head * mid_stride_1 +
      split * mid_stride_2;
  if (tid == 0) {
    mid_row[0] = softmax_state[0];
    mid_row[1] = softmax_state[1];
  }
  mid_row[2 + 0 * 128 + tid] = acc0;
  mid_row[2 + 1 * 128 + tid] = acc1;
  mid_row[2 + 2 * 128 + tid] = acc2;
  mid_row[2 + 3 * 128 + tid] = acc3;
}

// Stage 2: merge per-split partials, normalize, run inverse FWHT_512,
// write BF16 output. Grid (num_rows, num_heads). 128 threads, each
// owning four latent dims. Merge formula matches TurboQuant
// stage2_kernel (log-sum-exp merge across splits).

__global__ void __launch_bounds__(kBlockThreads, 8)
higgs_dense_2bit_mla_decode_stage2_kernel(
    const float* __restrict__ mid,
    bf16_t* __restrict__ out,
    int64_t num_rows,
    int64_t num_heads,
    int64_t num_splits,
    int64_t mid_stride_0,
    int64_t mid_stride_1,
    int64_t mid_stride_2,
    int64_t out_stride_0,
    int64_t out_stride_1) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= num_rows || head >= num_heads) return;

  __shared__ float smem128[kBlockThreads];
  __shared__ float denom_smem;

  const float* mid_base =
      mid + row * mid_stride_0 + head * mid_stride_1;

  // Pass 1: find global max m across splits.
  float m = kNegInf;
  for (int64_t s = 0; s < num_splits; ++s) {
    m = fmaxf(m, mid_base[s * mid_stride_2]);
  }

  // Pass 2: rescaled sum-of-l and acc.
  float l = 0.0f;
  float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
  for (int64_t s = 0; s < num_splits; ++s) {
    const float* sb = mid_base + s * mid_stride_2;
    const float split_m = sb[0];
    const float split_l = sb[1];
    const float scale = split_l > 0.0f ? __expf(split_m - m) : 0.0f;
    l += split_l * scale;
    v0 += sb[2 + 0 * 128 + tid] * scale;
    v1 += sb[2 + 1 * 128 + tid] * scale;
    v2 += sb[2 + 2 * 128 + tid] * scale;
    v3 += sb[2 + 3 * 128 + tid] * scale;
  }

  if (tid == 0) denom_smem = l;
  __syncthreads();
  const float denom = denom_smem;
  const bool ok = denom > 0.0f;
  v0 = ok ? v0 / denom : 0.0f;
  v1 = ok ? v1 / denom : 0.0f;
  v2 = ok ? v2 / denom : 0.0f;
  v3 = ok ? v3 / denom : 0.0f;

  // Inverse FWHT_512 (FWHT is self-inverse up to scale).
  fwht_register_top2(v0, v1, v2, v3);
  __syncthreads();
  v0 = fwht_128elem(v0, tid, smem128);
  __syncthreads();
  v1 = fwht_128elem(v1, tid, smem128);
  __syncthreads();
  v2 = fwht_128elem(v2, tid, smem128);
  __syncthreads();
  v3 = fwht_128elem(v3, tid, smem128);
  __syncthreads();

  bf16_t* out_row = out + row * out_stride_0 + head * out_stride_1;
  out_row[0 * 128 + tid] = __float2bfloat16(v0 * kInvSqrtLatentDim);
  out_row[1 * 128 + tid] = __float2bfloat16(v1 * kInvSqrtLatentDim);
  out_row[2 * 128 + tid] = __float2bfloat16(v2 * kInvSqrtLatentDim);
  out_row[3 * 128 + tid] = __float2bfloat16(v3 * kInvSqrtLatentDim);
}

__global__ void __launch_bounds__(kBlockThreads, 8)
higgs_dense_2bit_mla_decode_saw_scalar2_stage1_split_kernel(
    const bf16_t* __restrict__ q_nope,
    const bf16_t* __restrict__ q_rope,
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    float* __restrict__ mid,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t num_heads,
    int64_t topk,
    int64_t num_splits,
    int64_t q_nope_stride_0,
    int64_t q_nope_stride_1,
    int64_t q_rope_stride_0,
    int64_t q_rope_stride_1,
    int64_t compressed_stride_0,
    int64_t page_table_stride_0,
    int64_t mid_stride_0,
    int64_t mid_stride_1,
    int64_t mid_stride_2,
    float sm_scale) {
  (void)codebook;
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int split = blockIdx.z;
  const int tid = threadIdx.x;
  if (row >= num_rows || head >= num_heads || split >= num_splits) return;

  __shared__ float softmax_state[4];
  __shared__ float warp_partials[4];

  const bf16_t* q_nope_row =
      q_nope + row * q_nope_stride_0 + head * q_nope_stride_1;
  const float q0 = bf16_to_float(q_nope_row[0 * 128 + tid]);
  const float q1 = bf16_to_float(q_nope_row[1 * 128 + tid]);
  const float q2 = bf16_to_float(q_nope_row[2 * 128 + tid]);
  const float q3 = bf16_to_float(q_nope_row[3 * 128 + tid]);

  if (tid == 0) {
    softmax_state[0] = kNegInf;
    softmax_state[1] = 0.0f;
  }
  __syncthreads();

  float q_rope_val = 0.0f;
  if (tid < kRopeDim) {
    const bf16_t* q_rope_row =
        q_rope + row * q_rope_stride_0 + head * q_rope_stride_1;
    q_rope_val = bf16_to_float(q_rope_row[tid]);
  }

  const int64_t chunk = (topk + num_splits - 1) / num_splits;
  const int64_t begin = split * chunk;
  const int64_t end = min(begin + chunk, topk);
  const int32_t* pages = page_table + row * page_table_stride_0;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  for (int64_t col = begin; col < end; ++col) {
    const int32_t page = __ldg(&pages[col]);
    const bool valid = page >= 0;
    const uint8_t* slot =
        compressed + (valid ? static_cast<int64_t>(page) : 0) *
        compressed_stride_0;

    float c0 = 0.0f;
    float c1 = 0.0f;
    float c2 = 0.0f;
    float c3 = 0.0f;
    if (valid) {
      saw_scalar2_unpack_pair_lanes(slot, tid, c0, c1, c2, c3);
    }

    float val = valid ? (q0 * c0 + q1 * c1 + q2 * c2 + q3 * c3) : 0.0f;
    if (tid < kRopeDim && valid) {
      const bf16_t* rope =
          reinterpret_cast<const bf16_t*>(slot + kPackedBytes + kNormBytes);
      val += q_rope_val * bf16_to_float(rope[tid]);
    }

    const float warp_sum = warp_reduce_sum(val);
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) warp_partials[warp_id] = warp_sum;
    __syncthreads();

    if (tid == 0) {
      const float total =
          warp_partials[0] + warp_partials[1] +
          warp_partials[2] + warp_partials[3];
      const float score = valid ? total * sm_scale : kNegInf;
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
    acc0 = acc0 * alpha + beta * c0;
    acc1 = acc1 * alpha + beta * c1;
    acc2 = acc2 * alpha + beta * c2;
    acc3 = acc3 * alpha + beta * c3;
  }

  float* mid_row =
      mid + row * mid_stride_0 + head * mid_stride_1 +
      split * mid_stride_2;
  if (tid == 0) {
    mid_row[0] = softmax_state[0];
    mid_row[1] = softmax_state[1];
  }
  mid_row[2 + 0 * 128 + tid] = acc0;
  mid_row[2 + 1 * 128 + tid] = acc1;
  mid_row[2 + 2 * 128 + tid] = acc2;
  mid_row[2 + 3 * 128 + tid] = acc3;
}

__global__ void __launch_bounds__(kBlockThreads, 8)
higgs_dense_2bit_mla_decode_saw_scalar2_stage2_kernel(
    const float* __restrict__ mid,
    bf16_t* __restrict__ out,
    int64_t num_rows,
    int64_t num_heads,
    int64_t num_splits,
    int64_t mid_stride_0,
    int64_t mid_stride_1,
    int64_t mid_stride_2,
    int64_t out_stride_0,
    int64_t out_stride_1) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= num_rows || head >= num_heads) return;

  __shared__ float denom_smem;
  const float* mid_base =
      mid + row * mid_stride_0 + head * mid_stride_1;

  float m = kNegInf;
  for (int64_t s = 0; s < num_splits; ++s) {
    m = fmaxf(m, mid_base[s * mid_stride_2]);
  }

  float l = 0.0f;
  float v0 = 0.0f;
  float v1 = 0.0f;
  float v2 = 0.0f;
  float v3 = 0.0f;
  for (int64_t s = 0; s < num_splits; ++s) {
    const float* sb = mid_base + s * mid_stride_2;
    const float split_m = sb[0];
    const float split_l = sb[1];
    const float scale = split_l > 0.0f ? __expf(split_m - m) : 0.0f;
    l += split_l * scale;
    v0 += sb[2 + 0 * 128 + tid] * scale;
    v1 += sb[2 + 1 * 128 + tid] * scale;
    v2 += sb[2 + 2 * 128 + tid] * scale;
    v3 += sb[2 + 3 * 128 + tid] * scale;
  }

  if (tid == 0) denom_smem = l;
  __syncthreads();
  const float denom = denom_smem;
  const bool ok = denom > 0.0f;
  bf16_t* out_row = out + row * out_stride_0 + head * out_stride_1;
  out_row[0 * 128 + tid] = __float2bfloat16(ok ? v0 / denom : 0.0f);
  out_row[1 * 128 + tid] = __float2bfloat16(ok ? v1 / denom : 0.0f);
  out_row[2 * 128 + tid] = __float2bfloat16(ok ? v2 / denom : 0.0f);
  out_row[3 * 128 + tid] = __float2bfloat16(ok ? v3 / denom : 0.0f);
}

// Host-side launchers.

struct HiggsDense2BitMLADecodeKernel {
  static void run(
      tvm::ffi::TensorView q_nope,
      tvm::ffi::TensorView q_rope,
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView codebook,
      double sm_scale) {
    using namespace host;

    auto R = SymbolicSize{"num_rows"};
    auto H = SymbolicSize{"num_heads"};
    auto K = SymbolicSize{"topk"};
    auto S = SymbolicSize{"num_slots"};
    auto q_nope_stride_0 = SymbolicSize{"q_nope_stride_0"};
    auto q_nope_stride_1 = SymbolicSize{"q_nope_stride_1"};
    auto q_rope_stride_0 = SymbolicSize{"q_rope_stride_0"};
    auto q_rope_stride_1 = SymbolicSize{"q_rope_stride_1"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto page_table_stride_0 = SymbolicSize{"page_table_stride_0"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto out_stride_1 = SymbolicSize{"out_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, H, kLatentDim})
        .with_strides({q_nope_stride_0, q_nope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_nope);
    TensorMatcher({R, H, kRopeDim})
        .with_strides({q_rope_stride_0, q_rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_rope);
    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({R, K})
        .with_strides({page_table_stride_0, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({R, H, kLatentDim})
        .with_strides({out_stride_0, out_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    if (R.unwrap() == 0 || H.unwrap() == 0 || K.unwrap() == 0) return;

    dim3 grid(R.unwrap(), H.unwrap());
    LaunchKernel(grid, kBlockThreads, device.unwrap())(
        higgs_dense_2bit_mla_decode_kernel,
        static_cast<const bf16_t*>(q_nope.data_ptr()),
        static_cast<const bf16_t*>(q_rope.data_ptr()),
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        R.unwrap(),
        H.unwrap(),
        K.unwrap(),
        q_nope_stride_0.unwrap(),
        q_nope_stride_1.unwrap(),
        q_rope_stride_0.unwrap(),
        q_rope_stride_1.unwrap(),
        compressed_stride_0.unwrap(),
        page_table_stride_0.unwrap(),
        out_stride_0.unwrap(),
        out_stride_1.unwrap(),
        static_cast<float>(sm_scale));
  }
};

struct HiggsDense2BitMLADecodeSawScalar2Kernel {
  static void run(
      tvm::ffi::TensorView q_nope,
      tvm::ffi::TensorView q_rope,
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView codebook,
      double sm_scale) {
    using namespace host;

    auto R = SymbolicSize{"num_rows"};
    auto H = SymbolicSize{"num_heads"};
    auto K = SymbolicSize{"topk"};
    auto S = SymbolicSize{"num_slots"};
    auto q_nope_stride_0 = SymbolicSize{"q_nope_stride_0"};
    auto q_nope_stride_1 = SymbolicSize{"q_nope_stride_1"};
    auto q_rope_stride_0 = SymbolicSize{"q_rope_stride_0"};
    auto q_rope_stride_1 = SymbolicSize{"q_rope_stride_1"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto page_table_stride_0 = SymbolicSize{"page_table_stride_0"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto out_stride_1 = SymbolicSize{"out_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, H, kLatentDim})
        .with_strides({q_nope_stride_0, q_nope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_nope);
    TensorMatcher({R, H, kRopeDim})
        .with_strides({q_rope_stride_0, q_rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_rope);
    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({R, K})
        .with_strides({page_table_stride_0, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({R, H, kLatentDim})
        .with_strides({out_stride_0, out_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    if (R.unwrap() == 0 || H.unwrap() == 0 || K.unwrap() == 0) return;

    dim3 grid(R.unwrap(), H.unwrap());
    LaunchKernel(grid, kBlockThreads, device.unwrap())(
        higgs_dense_2bit_mla_decode_saw_scalar2_kernel,
        static_cast<const bf16_t*>(q_nope.data_ptr()),
        static_cast<const bf16_t*>(q_rope.data_ptr()),
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        R.unwrap(),
        H.unwrap(),
        K.unwrap(),
        q_nope_stride_0.unwrap(),
        q_nope_stride_1.unwrap(),
        q_rope_stride_0.unwrap(),
        q_rope_stride_1.unwrap(),
        compressed_stride_0.unwrap(),
        page_table_stride_0.unwrap(),
        out_stride_0.unwrap(),
        out_stride_1.unwrap(),
        static_cast<float>(sm_scale));
  }
};

struct HiggsDense2BitMLARotateQueryKernel {
  static void run(
      tvm::ffi::TensorView q_nope,
      tvm::ffi::TensorView q_rotated) {
    using namespace host;

    auto R = SymbolicSize{"num_rows"};
    auto H = SymbolicSize{"num_heads"};
    auto q_nope_stride_0 = SymbolicSize{"q_nope_stride_0"};
    auto q_nope_stride_1 = SymbolicSize{"q_nope_stride_1"};
    auto q_rot_stride_0 = SymbolicSize{"q_rot_stride_0"};
    auto q_rot_stride_1 = SymbolicSize{"q_rot_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, H, kLatentDim})
        .with_strides({q_nope_stride_0, q_nope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_nope);
    TensorMatcher({R, H, kLatentDim})
        .with_strides({q_rot_stride_0, q_rot_stride_1, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(q_rotated);

    if (R.unwrap() == 0 || H.unwrap() == 0) return;

    LaunchKernel(
        dim3(R.unwrap(), H.unwrap()), kBlockThreads, device.unwrap())(
        higgs_dense_2bit_mla_rotate_query_kernel,
        static_cast<const bf16_t*>(q_nope.data_ptr()),
        static_cast<float*>(q_rotated.data_ptr()),
        R.unwrap(),
        H.unwrap(),
        q_nope_stride_0.unwrap(),
        q_nope_stride_1.unwrap(),
        q_rot_stride_0.unwrap(),
        q_rot_stride_1.unwrap());
  }
};

struct HiggsDense2BitMLADecodeSplitKernel {
  static void run(
      tvm::ffi::TensorView q_rotated,
      tvm::ffi::TensorView q_rope,
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView mid,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView codebook,
      double sm_scale) {
    using namespace host;

    auto R = SymbolicSize{"num_rows"};
    auto H = SymbolicSize{"num_heads"};
    auto K = SymbolicSize{"topk"};
    auto S = SymbolicSize{"num_slots"};
    auto P = SymbolicSize{"num_splits"};
    auto q_rot_stride_0 = SymbolicSize{"q_rot_stride_0"};
    auto q_rot_stride_1 = SymbolicSize{"q_rot_stride_1"};
    auto q_rope_stride_0 = SymbolicSize{"q_rope_stride_0"};
    auto q_rope_stride_1 = SymbolicSize{"q_rope_stride_1"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto page_table_stride_0 = SymbolicSize{"page_table_stride_0"};
    auto mid_stride_0 = SymbolicSize{"mid_stride_0"};
    auto mid_stride_1 = SymbolicSize{"mid_stride_1"};
    auto mid_stride_2 = SymbolicSize{"mid_stride_2"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto out_stride_1 = SymbolicSize{"out_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, H, kLatentDim})
        .with_strides({q_rot_stride_0, q_rot_stride_1, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(q_rotated);
    TensorMatcher({R, H, kRopeDim})
        .with_strides({q_rope_stride_0, q_rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_rope);
    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({R, K})
        .with_strides({page_table_stride_0, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({R, H, P, kLatentDim + 2})
        .with_strides({mid_stride_0, mid_stride_1, mid_stride_2, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(mid);
    TensorMatcher({R, H, kLatentDim})
        .with_strides({out_stride_0, out_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    if (R.unwrap() == 0 || H.unwrap() == 0 || K.unwrap() == 0 ||
        P.unwrap() == 0) {
      return;
    }

    LaunchKernel(
        dim3(R.unwrap(), H.unwrap(), P.unwrap()), kBlockThreads,
        device.unwrap())(
        higgs_dense_2bit_mla_decode_stage1_split_kernel,
        static_cast<const float*>(q_rotated.data_ptr()),
        static_cast<const bf16_t*>(q_rope.data_ptr()),
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<float*>(mid.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        R.unwrap(),
        H.unwrap(),
        K.unwrap(),
        P.unwrap(),
        q_rot_stride_0.unwrap(),
        q_rot_stride_1.unwrap(),
        q_rope_stride_0.unwrap(),
        q_rope_stride_1.unwrap(),
        compressed_stride_0.unwrap(),
        page_table_stride_0.unwrap(),
        mid_stride_0.unwrap(),
        mid_stride_1.unwrap(),
        mid_stride_2.unwrap(),
        static_cast<float>(sm_scale));

    LaunchKernel(
        dim3(R.unwrap(), H.unwrap()), kBlockThreads, device.unwrap())(
        higgs_dense_2bit_mla_decode_stage2_kernel,
        static_cast<const float*>(mid.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        R.unwrap(),
        H.unwrap(),
        P.unwrap(),
        mid_stride_0.unwrap(),
        mid_stride_1.unwrap(),
        mid_stride_2.unwrap(),
        out_stride_0.unwrap(),
        out_stride_1.unwrap());
  }
};

struct HiggsDense2BitMLADecodeSawScalar2SplitKernel {
  static void run(
      tvm::ffi::TensorView q_nope,
      tvm::ffi::TensorView q_rope,
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView mid,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView codebook,
      double sm_scale) {
    using namespace host;

    auto R = SymbolicSize{"num_rows"};
    auto H = SymbolicSize{"num_heads"};
    auto K = SymbolicSize{"topk"};
    auto S = SymbolicSize{"num_slots"};
    auto P = SymbolicSize{"num_splits"};
    auto q_nope_stride_0 = SymbolicSize{"q_nope_stride_0"};
    auto q_nope_stride_1 = SymbolicSize{"q_nope_stride_1"};
    auto q_rope_stride_0 = SymbolicSize{"q_rope_stride_0"};
    auto q_rope_stride_1 = SymbolicSize{"q_rope_stride_1"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto page_table_stride_0 = SymbolicSize{"page_table_stride_0"};
    auto mid_stride_0 = SymbolicSize{"mid_stride_0"};
    auto mid_stride_1 = SymbolicSize{"mid_stride_1"};
    auto mid_stride_2 = SymbolicSize{"mid_stride_2"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto out_stride_1 = SymbolicSize{"out_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, H, kLatentDim})
        .with_strides({q_nope_stride_0, q_nope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_nope);
    TensorMatcher({R, H, kRopeDim})
        .with_strides({q_rope_stride_0, q_rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_rope);
    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({R, K})
        .with_strides({page_table_stride_0, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({R, H, P, kLatentDim + 2})
        .with_strides({mid_stride_0, mid_stride_1, mid_stride_2, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(mid);
    TensorMatcher({R, H, kLatentDim})
        .with_strides({out_stride_0, out_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    if (R.unwrap() == 0 || H.unwrap() == 0 || K.unwrap() == 0 ||
        P.unwrap() == 0) {
      return;
    }

    LaunchKernel(
        dim3(R.unwrap(), H.unwrap(), P.unwrap()), kBlockThreads,
        device.unwrap())(
        higgs_dense_2bit_mla_decode_saw_scalar2_stage1_split_kernel,
        static_cast<const bf16_t*>(q_nope.data_ptr()),
        static_cast<const bf16_t*>(q_rope.data_ptr()),
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<float*>(mid.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        R.unwrap(),
        H.unwrap(),
        K.unwrap(),
        P.unwrap(),
        q_nope_stride_0.unwrap(),
        q_nope_stride_1.unwrap(),
        q_rope_stride_0.unwrap(),
        q_rope_stride_1.unwrap(),
        compressed_stride_0.unwrap(),
        page_table_stride_0.unwrap(),
        mid_stride_0.unwrap(),
        mid_stride_1.unwrap(),
        mid_stride_2.unwrap(),
        static_cast<float>(sm_scale));

    LaunchKernel(
        dim3(R.unwrap(), H.unwrap()), kBlockThreads, device.unwrap())(
        higgs_dense_2bit_mla_decode_saw_scalar2_stage2_kernel,
        static_cast<const float*>(mid.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        R.unwrap(),
        H.unwrap(),
        P.unwrap(),
        mid_stride_0.unwrap(),
        mid_stride_1.unwrap(),
        mid_stride_2.unwrap(),
        out_stride_0.unwrap(),
        out_stride_1.unwrap());
  }
};

}  // namespace higgs_dense_2bit_mla_detail
