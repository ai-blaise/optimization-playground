// HIGGS 2-bit dense MLA decode kernel.
//
// Same external contract as ``turboquant_dense_mla_decode.cuh``: given a
// page table of slot indices, dot the dequantized latent + rope KV
// against a (BF16) query, apply softmax with on-line max-rescale, and
// emit a BF16 result per (row, head).
//
// Optimization pattern adopted verbatim from the TurboQuant
// 2.5-bit MLA decode kernel that lives next to this file
// (``turboquant_dense_mla_decode.cuh``, the
// ``turboquant_dense_mla_decode_2p5_kernel`` launcher): 128
// threads per block, each holding **four** latent dims as register
// state. The full FWHT_512 factors as four parallel FWHT_128s
// (warp-shuffle for levels 0..4, single SMEM exchange for levels 5..6)
// stitched together by a 4-way register FWHT
// (``fwht_register_top2``). This cuts SMEM traffic and lets us
// keep 8 resident blocks per SM on H200.
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

// ─── Architectural constants ─────────────────────────────────────────────────

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

// ─── Helpers (adapted from turboquant_dense_mla_decode.cuh:34-117) ──────────

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
// transform). Adapted from turboquant_dense_mla_decode.cuh:84-93.
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

// FWHT_32 within a warp via shuffle (levels 0..4). Adapted from
// turboquant_dense_mla_decode.cuh:96-103.
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
// for levels 5..6. Adapted from turboquant_dense_mla_decode.cuh:107-117.
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

// ─── HIGGS 2-bit MLA decode kernel ──────────────────────────────────────────
// Grid: (num_rows, num_heads). Block: 128 threads. Each thread owns four
// latent dims at indices ``g * 128 + tid`` for g in {0, 1, 2, 3}.
//
// Algorithmic flow per (row, head):
//
//   1. Load q_nope (BF16) for the 4 latent dims this thread owns.
//   2. Forward FWHT_512 = (FWHT_128 per group, cooperative) + (FWHT_4
//      across groups, register-resident). After step 2 each thread
//      holds q_rot for its 4 dims.
//   3. Multiply by kInvSqrtLatentDim (orthonormal scale).
//   4. Online softmax + value accumulator loop over topk slots:
//        - Unpack 4 codebook indices per thread (one per group).
//        - Look up codeword for this thread's coordinate.
//        - Compute per-thread dot ``Σ_g q_rot_g * scale * G[idx_g, coord]``
//          and rope contribution (only tid < kRopeDim).
//        - Block-reduce -> scalar score; update (m, l, α, β).
//        - Update per-thread accumulators acc{0..3}.
//   5. Normalize by l, run inverse FWHT_512, multiply by
//      kInvSqrtLatentDim, write BF16 output.

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
  __shared__ float softmax_state[4];    // [m, l, alpha, beta]
  __shared__ float warp_partials[4];
  __shared__ float cb_smem[kCodebookSize * kPairDim];  // 32 fp32 = 128 B

  // Cooperatively load the codebook into SMEM once per block. 128 threads
  // load 32 floats, so each thread loads at most one entry.
  if (tid < kCodebookSize * kPairDim) {
    cb_smem[tid] = __ldg(&codebook[tid]);
  }

  // ─── 1. Forward FWHT_512 on q_nope ───────────────────────────────────────

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

  // ─── 2. Codebook → SMEM (loaded above). The codebook is 16 × 2 fp32 =
  //       128 B; loading it once into SMEM cuts the register footprint
  //       from 16/thread to 0, which lets the compiler keep more state
  //       (FWHT v0..v3 + acc0..3 + softmax state) live without
  //       spilling. The per-thread coord projection happens at lookup
  //       time below.

  // ─── 3. Online softmax + value accumulator ─────────────────────────────
  if (tid == 0) {
    softmax_state[0] = kNegInf;  // m
    softmax_state[1] = 0.0f;     // l
  }
  // Sync also makes cb_smem visible to all threads.
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
        compressed + (valid ? static_cast<int64_t>(page) : 0) * compressed_stride_0;

    // Unpack 4 codebook indices (one per group). Each thread owns
    // pair_idx = g*64 + (tid >> 1) for g in 0..3, i.e., 4 distinct
    // 4-bit codebook indices per thread.
    const int pair_within_group = tid >> 1;
    // byte_in_group = pair_within_group >> 1; nibble = pair_within_group & 1.
    // The packed region is 128 B: 32 B per group × 4 groups.
    const int byte_in_group = pair_within_group >> 1;
    const int nibble = pair_within_group & 1;
    const uint8_t b0 = slot[0 * 32 + byte_in_group];
    const uint8_t b1 = slot[1 * 32 + byte_in_group];
    const uint8_t b2 = slot[2 * 32 + byte_in_group];
    const uint8_t b3 = slot[3 * 32 + byte_in_group];
    const uint32_t i0 = nibble ? (b0 >> 4) : (b0 & 0x0F);
    const uint32_t i1 = nibble ? (b1 >> 4) : (b1 & 0x0F);
    const uint32_t i2 = nibble ? (b2 >> 4) : (b2 & 0x0F);
    const uint32_t i3 = nibble ? (b3 >> 4) : (b3 & 0x0F);

    const int coord = tid & 1;
    const float c0 = cb_smem[i0 * kPairDim + coord];
    const float c1 = cb_smem[i1 * kPairDim + coord];
    const float c2 = cb_smem[i2 * kPairDim + coord];
    const float c3 = cb_smem[i3 * kPairDim + coord];

    // Per-token block scale.
    const half norm_h =
        *reinterpret_cast<const half*>(slot + kPackedBytes);
    const float scale = __half2float(norm_h);

    // Score contribution: latent + rope.
    float val = valid
        ? scale * (v0 * c0 + v1 * c1 + v2 * c2 + v3 * c3)
        : 0.0f;
    if (tid < kRopeDim && valid) {
      const bf16_t* rope =
          reinterpret_cast<const bf16_t*>(slot + kPackedBytes + kNormBytes);
      val += q_rope_val * bf16_to_float(rope[tid]);
    }

    // Block reduction -> one scalar score.
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

  // ─── 4. Normalize + inverse FWHT_512 (FWHT is self-inverse) ─────────────
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

// ─── Host-side launcher ─────────────────────────────────────────────────────

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

}  // namespace higgs_dense_2bit_mla_detail
