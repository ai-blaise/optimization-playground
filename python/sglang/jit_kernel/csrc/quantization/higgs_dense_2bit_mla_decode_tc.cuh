// HIGGS 2-bit dense MLA decode kernel — tensor-core variant.
//
// CuTe + SM80 m16n8k16 BF16 mma.sync (works on Blackwell B200). Same
// external contract as higgs_dense_2bit_mla_decode.cuh.
//
// Iter 2: full tensor-core path. Both score MMA (q.K^T) and V update
// (acc += p.V) use cute::gemm() over the SM80_16x8x16 BF16 atom.
// kBlockH=16 Q heads per CTA so per-thread FP32 register accumulator
// is 16*512/128 = 64 fp32 (fits the register budget). Online softmax
// in registers with alpha-rescale of acc each slot tile.
//
// SMEM:
//   q_smem[16][576]  BF16  = 18.0 KB   (rotated q, K-major)
//   k_smem[64][576]  BF16  = 73.7 KB   (dequant K slot tile, K-major)
//   p_smem[16][64]   BF16  =  2.0 KB   (softmax weights for V MMA)
//   softmax_state + codebook + scratch = ~2 KB
//   Total ~95 KB (well within 228 KB B200 cap).
//
// References:
//   * cutlass/examples/cute/tutorial/sgemm_sm80.cu (canonical CuTe MMA
//     pattern with make_tiled_copy + s2r ldmatrix).
//   * cutlass/include/cute/atom/mma_traits_sm80.hpp (atom traits).
//   * (Future) togethercomputer/saw-int4 BDR + INT4 KV patterns for
//     bank-conflict-free SMEM, packed-byte coalescing, FWHT
//     scheduling.
//
// Mathematical trick (orthonormal FWHT is self-inverse): rotate q
// through FWHT_512 + 1/sqrt(512), keep KV in rotated coordinates
// (scale * G[idx]), accumulate softmax(q_rot . k_concat) * v across
// the topk loop in the rotated basis, apply InvFWHT_512 at the end.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm80.hpp>

namespace higgs_dense_2bit_mla_tc_detail {

using namespace cute;

// Architectural constants.
constexpr int kLatentDim = 512;
constexpr int kRopeDim = 64;
constexpr int kFullDim = kLatentDim + kRopeDim;  // 576
constexpr int kPairDim = 2;
constexpr int kCodebookSize = 16;
constexpr int kNumPairs = kLatentDim / kPairDim;  // 256
constexpr int kPackedBytes = kNumPairs / 2;       // 128
constexpr int kNormBytes = 2;
// Payload that the kernel actually reads (offsets in
// ``[0, kPayloadBytes)``); matches HiggsDense2BitConfig.payload_bytes.
constexpr int kPayloadBytes = kPackedBytes + kNormBytes + kRopeDim * 2;  // 258
// Iter4 (#16) per-slot stride: 16-byte aligned. Matches
// ``kSlotBytes`` in ``higgs_dense_2bit_kv.cuh``.
constexpr int kSlotAlignmentBytes = 16;
constexpr int kSlotBytes =
    (kPayloadBytes + kSlotAlignmentBytes - 1) /
    kSlotAlignmentBytes * kSlotAlignmentBytes;
static_assert(kPayloadBytes == 258, "expected payload bytes = 258");
static_assert(kSlotBytes == 272, "expected padded slot bytes = 272");
constexpr float kInvSqrtLatentDim = 0.044194173824159216f;
constexpr float kNegInf = -3.4028234663852886e38f;

// MMA tile shape — uses SM80 m16n8k16 atom (BF16xBF16 -> FP32).
// Per CTA: M=16 Q heads, N=64 KV slots, K=16 latent-dim atom. With 4
// warps in (1, 4, 1) AtomLayout, each gemm() covers (M=16, N=64,
// K=16) — 1 M-atom × 8 N-atoms expanded by Tile<>.
constexpr int kBlockH = 16;
constexpr int kBlockN = 64;
constexpr int kBlockK = 16;
constexpr int kBlockThreads = 128;
constexpr int kNumWarps = kBlockThreads / 32;  // 4

using BFloat16 = bf16_t;

// SMEM storage.
struct SharedStorage {
  alignas(128) BFloat16 q_smem[kBlockH * kFullDim];        // [H, D] K-major BF16
  alignas(128) BFloat16 k_smem[kBlockN * kFullDim];        // [N, D] K-major BF16
  alignas(128) BFloat16 v_smem[kBlockN * kLatentDim];      // [D, N] K-major BF16
  alignas(128) BFloat16 p_smem[kBlockH * kBlockN];         // softmax weights BF16
  alignas(16) float softmax_state_m[kBlockH];              // online softmax max
  alignas(16) float softmax_state_l[kBlockH];              // online softmax sum
  alignas(16) float softmax_alpha[kBlockH];                // per-slot acc rescale
  alignas(16) float codebook_smem[kCodebookSize * kPairDim];
  alignas(16) float p_fp32[kBlockH * kBlockN];             // softmax weights FP32
};

// ---------------------------------------------------------------------------
// HIGGS unpack (port of baseline higgs_unpack_indices).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void higgs_unpack_indices(
    const uint8_t* __restrict__ slot, int tid,
    uint32_t& i0, uint32_t& i1, uint32_t& i2, uint32_t& i3) {
  const int pair_within_group = tid >> 1;
  const bool coord_lane = tid & 1;
  const int byte_in_group = pair_within_group >> 1;
  const int nibble = pair_within_group & 1;
  uint32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
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

// FWHT helpers (port of baseline).
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

// Rotate q_nope through FWHT_512 + scale, append q_rope, into q_smem.
__device__ __forceinline__ void rotate_q_into_smem(
    const BFloat16* __restrict__ q_nope_row_base,
    const BFloat16* __restrict__ q_rope_row_base,
    int64_t q_nope_stride_1,
    int64_t q_rope_stride_1,
    int head_count,
    int head_base,
    BFloat16* q_smem_flat,
    float* fwht_scratch) {
  const int tid = threadIdx.x;
  for (int h_local = 0; h_local < kBlockH; ++h_local) {
    const int h_global = head_base + h_local;
    const bool valid = h_global < head_count;
    const BFloat16* q_nope_row =
        valid ? q_nope_row_base + h_global * q_nope_stride_1 : q_nope_row_base;
    const BFloat16* q_rope_row =
        valid ? q_rope_row_base + h_global * q_rope_stride_1 : q_rope_row_base;
    float v0 = valid ? __bfloat162float(q_nope_row[0 * 128 + tid]) : 0.0f;
    float v1 = valid ? __bfloat162float(q_nope_row[1 * 128 + tid]) : 0.0f;
    float v2 = valid ? __bfloat162float(q_nope_row[2 * 128 + tid]) : 0.0f;
    float v3 = valid ? __bfloat162float(q_nope_row[3 * 128 + tid]) : 0.0f;

    v0 = fwht_128elem(v0, tid, fwht_scratch);
    __syncthreads();
    v1 = fwht_128elem(v1, tid, fwht_scratch);
    __syncthreads();
    v2 = fwht_128elem(v2, tid, fwht_scratch);
    __syncthreads();
    v3 = fwht_128elem(v3, tid, fwht_scratch);
    __syncthreads();
    fwht_register_top2(v0, v1, v2, v3);

    v0 *= kInvSqrtLatentDim;
    v1 *= kInvSqrtLatentDim;
    v2 *= kInvSqrtLatentDim;
    v3 *= kInvSqrtLatentDim;

    BFloat16* dst = q_smem_flat + h_local * kFullDim;
    dst[0 * 128 + tid] = __float2bfloat16(v0);
    dst[1 * 128 + tid] = __float2bfloat16(v1);
    dst[2 * 128 + tid] = __float2bfloat16(v2);
    dst[3 * 128 + tid] = __float2bfloat16(v3);
    if (tid < kRopeDim) {
      dst[kLatentDim + tid] = valid ? q_rope_row[tid] : BFloat16(0.0f);
    }
    __syncthreads();
  }
}

// Cooperative dequant of one slot. Writes the latent in BOTH k_smem
// (row-major (slot, dim) for the score MMA) and v_smem (transposed
// (dim, slot) K-major for the V MMA). The rope tail lives only in
// k_smem; the V MMA reads the latent-only B operand.
__device__ __forceinline__ void dequant_one_slot_dual(
    const uint8_t* __restrict__ slot, bool valid,
    int slot_idx,
    const float* __restrict__ cb_smem,
    BFloat16* k_row, BFloat16* v_smem_base) {
  const int tid = threadIdx.x;
  uint32_t i0, i1, i2, i3;
  higgs_unpack_indices(slot, tid, i0, i1, i2, i3);
  const int coord = tid & 1;
  const float c0 = cb_smem[i0 * kPairDim + coord];
  const float c1 = cb_smem[i1 * kPairDim + coord];
  const float c2 = cb_smem[i2 * kPairDim + coord];
  const float c3 = cb_smem[i3 * kPairDim + coord];

  const half norm_h = *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = valid ? __half2float(norm_h) : 0.0f;

  const BFloat16 b0 = __float2bfloat16(scale * c0);
  const BFloat16 b1 = __float2bfloat16(scale * c1);
  const BFloat16 b2 = __float2bfloat16(scale * c2);
  const BFloat16 b3 = __float2bfloat16(scale * c3);

  const int d0 = 0 * 128 + tid;
  const int d1 = 1 * 128 + tid;
  const int d2 = 2 * 128 + tid;
  const int d3 = 3 * 128 + tid;

  k_row[d0] = b0;
  k_row[d1] = b1;
  k_row[d2] = b2;
  k_row[d3] = b3;

  v_smem_base[d0 * kBlockN + slot_idx] = b0;
  v_smem_base[d1 * kBlockN + slot_idx] = b1;
  v_smem_base[d2 * kBlockN + slot_idx] = b2;
  v_smem_base[d3 * kBlockN + slot_idx] = b3;

  if (tid < kRopeDim) {
    const BFloat16* rope = reinterpret_cast<const BFloat16*>(
        slot + kPackedBytes + kNormBytes);
    k_row[kLatentDim + tid] = valid ? rope[tid] : BFloat16(0.0f);
  }
}

// ---------------------------------------------------------------------------
// CuTe TiledMMA: SM80 m16n8k16 BF16 atom. (M=16, N=64, K=16) per gemm()
// for score; (M=16, N=512, K=BLOCK_N=64) for V.
// ---------------------------------------------------------------------------

using AtomMMA = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;

// Score MMA: 1 M-atom × 8 N-atoms (expanded by Tile<>) × 1 K-atom.
// AtomLayout (1, 4, 1) = 4 warps cooperate along N.
using TiledMmaScore = decltype(make_tiled_mma(
    AtomMMA{},
    Layout<Shape<_1, _4, _1>>{},
    Tile<_16, _64, _16>{}));

// V-update MMA: same atom; 1 M × 4 N × 1 K AtomLayout, expand via
// Tile<_16, _512, _16> so one gemm() covers (M=16, N=512, K=16). The
// K dim of the V MMA corresponds to BLOCK_N (slot dim); we iterate
// BLOCK_N/16 = 4 K-tiles per slot tile.
using TiledMmaV = decltype(make_tiled_mma(
    AtomMMA{},
    Layout<Shape<_1, _4, _1>>{},
    Tile<_16, _512, _16>{}));

// SMEM layouts.
using SmemLayoutQ = Layout<Shape<Int<kBlockH>, Int<kFullDim>>,
                            Stride<Int<kFullDim>, _1>>;
using SmemLayoutK = Layout<Shape<Int<kBlockN>, Int<kFullDim>>,
                            Stride<Int<kFullDim>, _1>>;
// p is (kBlockH, kBlockN) row-major BF16.
using SmemLayoutP = Layout<Shape<Int<kBlockH>, Int<kBlockN>>,
                            Stride<Int<kBlockN>, _1>>;
// V is staged into v_smem with (D, N) K-major layout so the TN MMA's
// B operand (B is N-outer K-inner) can use ldmatrix directly. The
// dequant phase writes K-as-(slot, dim) into k_smem, and a separate
// transposed copy stages it into v_smem as (dim, slot).
using SmemLayoutV = Layout<Shape<Int<kLatentDim>, Int<kBlockN>>,
                            Stride<Int<kBlockN>, _1>>;

// ---------------------------------------------------------------------------
// Single-pass tensor-core kernel.
// Grid: (num_rows, ceil(num_heads / kBlockH)). Block: kBlockThreads.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(kBlockThreads, 1)
higgs_dense_2bit_mla_decode_tc_kernel(
    const BFloat16* __restrict__ q_nope,
    const BFloat16* __restrict__ q_rope,
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    BFloat16* __restrict__ out,
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
  const int head_group = blockIdx.y;
  const int head_base = head_group * kBlockH;
  if (row >= num_rows || head_base >= num_heads) return;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;

  extern __shared__ uint8_t raw_shared_memory[];
  SharedStorage& ss = *reinterpret_cast<SharedStorage*>(raw_shared_memory);

  // Init codebook + softmax state.
  if (tid < kCodebookSize * kPairDim) {
    ss.codebook_smem[tid] = __ldg(&codebook[tid]);
  }
  if (tid < kBlockH) {
    ss.softmax_state_m[tid] = kNegInf;
    ss.softmax_state_l[tid] = 0.0f;
  }
  __syncthreads();

  // Step 1: rotate q via FWHT_512 + append q_rope. Use ss.p_fp32 as a
  // 128-fp32 FWHT scratchpad (need 128 fp32 = 512 B; p_fp32 is much
  // larger, plenty of space).
  float* fwht_scratch = ss.p_fp32;
  rotate_q_into_smem(
      q_nope + row * q_nope_stride_0,
      q_rope + row * q_rope_stride_0,
      q_nope_stride_1,
      q_rope_stride_1,
      static_cast<int>(num_heads),
      head_base,
      ss.q_smem,
      fwht_scratch);
  __syncthreads();

  // CuTe SMEM tensors.
  Tensor sQ = make_tensor(make_smem_ptr(ss.q_smem), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(ss.k_smem), SmemLayoutK{});
  Tensor sP = make_tensor(make_smem_ptr(ss.p_smem), SmemLayoutP{});
  Tensor sV = make_tensor(make_smem_ptr(ss.v_smem), SmemLayoutV{});

  // TiledMMA for score (q @ K^T).
  TiledMmaScore tiled_mma_score;
  auto thr_mma_score = tiled_mma_score.get_thread_slice(tid);
  Tensor tCrA_score = thr_mma_score.partition_fragment_A(sQ);
  Tensor tCrB_score = thr_mma_score.partition_fragment_B(sK);
  Tensor cScore = make_identity_tensor(
      make_shape(Int<kBlockH>{}, Int<kBlockN>{}));
  Tensor tCcScore = thr_mma_score.partition_C(cScore);
  Tensor tCrScore = make_tensor<float>(shape(tCcScore));

  using S2RCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, BFloat16>;
  using S2RCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, BFloat16>;
  auto s2r_copy_a_score = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma_score);
  auto s2r_copy_b_score = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma_score);
  auto thr_s2r_a_score = s2r_copy_a_score.get_slice(tid);
  auto thr_s2r_b_score = s2r_copy_b_score.get_slice(tid);
  Tensor tXsQ = thr_s2r_a_score.partition_S(sQ);
  Tensor tXsK = thr_s2r_b_score.partition_S(sK);
  Tensor tXrA_score = thr_s2r_a_score.retile_D(tCrA_score);
  Tensor tXrB_score = thr_s2r_b_score.retile_D(tCrB_score);

  // TiledMMA for V update (P @ V).
  TiledMmaV tiled_mma_v;
  auto thr_mma_v = tiled_mma_v.get_thread_slice(tid);
  Tensor tVrA = thr_mma_v.partition_fragment_A(sP);
  Tensor tVrB = thr_mma_v.partition_fragment_B(sV);
  Tensor cAcc = make_identity_tensor(
      make_shape(Int<kBlockH>{}, Int<kLatentDim>{}));
  Tensor tVcAcc = thr_mma_v.partition_C(cAcc);
  // Persistent FP32 accumulator in registers across slot tiles.
  Tensor tVrAcc = make_tensor<float>(shape(tVcAcc));
  clear(tVrAcc);

  auto s2r_copy_a_v = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma_v);
  auto s2r_copy_b_v = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma_v);
  auto thr_s2r_a_v = s2r_copy_a_v.get_slice(tid);
  auto thr_s2r_b_v = s2r_copy_b_v.get_slice(tid);
  Tensor tXsP = thr_s2r_a_v.partition_S(sP);
  Tensor tXsV = thr_s2r_b_v.partition_S(sV);
  Tensor tVxA = thr_s2r_a_v.retile_D(tVrA);
  Tensor tVxB = thr_s2r_b_v.retile_D(tVrB);

  const int32_t* pages = page_table + row * page_table_stride_0;

  // ---------------------------------------------------------------
  // Slot loop. Each iteration processes kBlockN slots.
  // ---------------------------------------------------------------
  for (int64_t tile_begin = 0; tile_begin < topk; tile_begin += kBlockN) {
    const int tile_count = static_cast<int>(
        min<int64_t>(kBlockN, topk - tile_begin));

    // (A) Dequant: write each slot's latent into BOTH k_smem (for the
    // score MMA, (slot, dim) row-major) and v_smem (for the V MMA,
    // (dim, slot) K-major). No separate transpose pass.
    for (int n = 0; n < kBlockN; ++n) {
      const int64_t col = tile_begin + n;
      const bool valid = col < topk;
      int32_t page = -1;
      if (valid) page = __ldg(&pages[col]);
      const bool page_valid = page >= 0;
      const uint8_t* slot = compressed +
          (page_valid ? static_cast<int64_t>(page) : 0) * compressed_stride_0;
      BFloat16* k_row = ss.k_smem + n * kFullDim;
      dequant_one_slot_dual(slot, valid && page_valid, n,
                            ss.codebook_smem, k_row, ss.v_smem);
    }
    __syncthreads();

    // (B) Score MMA.
    constexpr int kNumKTilesScore = kFullDim / 16;
    clear(tCrScore);
    CUTE_UNROLL
    for (int kt = 0; kt < kNumKTilesScore; ++kt) {
      copy(s2r_copy_a_score, tXsQ(_, _, kt), tXrA_score(_, _, kt));
      copy(s2r_copy_b_score, tXsK(_, _, kt), tXrB_score(_, _, kt));
      gemm(tiled_mma_score, tCrA_score(_, _, kt), tCrB_score(_, _, kt),
           tCrScore);
    }

    // (C) Scatter scores into p_fp32 SMEM (sm_scale, mask OOB N).
    CUTE_UNROLL
    for (int i = 0; i < size(tCrScore); ++i) {
      auto coord = tCcScore(i);
      const int m_idx = get<0>(coord);
      const int n_idx = get<1>(coord);
      const float v = tCrScore(i) * sm_scale;
      const bool valid_n = n_idx < tile_count;
      ss.p_fp32[m_idx * kBlockN + n_idx] = valid_n ? v : kNegInf;
    }
    __syncthreads();

    // (D) Per-row online softmax in registers; commit alpha to SMEM
    // so the V MMA rescale sees a global view across warps.
    constexpr int kRowsPerWarp = kBlockH / kNumWarps;
    for (int m_local = 0; m_local < kRowsPerWarp; ++m_local) {
      const int m_idx = warp_id * kRowsPerWarp + m_local;
      const float s0 = ss.p_fp32[m_idx * kBlockN + lane_id];
      const float s1 = ss.p_fp32[m_idx * kBlockN + lane_id + 32];
      float local_max = fmaxf(s0, s1);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
      }
      const float old_m = ss.softmax_state_m[m_idx];
      const float new_m = fmaxf(old_m, local_max);
      const float alpha = (old_m == kNegInf) ? 0.0f : __expf(old_m - new_m);
      const float e0 = __expf(s0 - new_m);
      const float e1 = __expf(s1 - new_m);
      float local_sum = e0 + e1;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, o);
      }
      const float old_l = ss.softmax_state_l[m_idx];
      const float new_l = old_l * alpha + local_sum;
      if (lane_id == 0) {
        ss.softmax_state_m[m_idx] = new_m;
        ss.softmax_state_l[m_idx] = new_l;
        ss.softmax_alpha[m_idx] = alpha;
      }
      // Store BF16 p for V MMA.
      ss.p_smem[m_idx * kBlockN + lane_id] = __float2bfloat16(e0);
      ss.p_smem[m_idx * kBlockN + lane_id + 32] = __float2bfloat16(e1);
    }
    __syncthreads();

    // (E) Rescale per-thread acc fragment by alpha; alpha is now
    // visible to all threads via softmax_alpha SMEM.
    CUTE_UNROLL
    for (int i = 0; i < size(tVrAcc); ++i) {
      auto coord = tVcAcc(i);
      const int m_idx = get<0>(coord);
      tVrAcc(i) *= ss.softmax_alpha[m_idx];
    }

    // (F) V MMA: acc += p @ V. K of the V MMA = BLOCK_N=64; 4 K-tiles.
    constexpr int kNumKTilesV = kBlockN / 16;
    CUTE_UNROLL
    for (int kt = 0; kt < kNumKTilesV; ++kt) {
      copy(s2r_copy_a_v, tXsP(_, _, kt), tVxA(_, _, kt));
      copy(s2r_copy_b_v, tXsV(_, _, kt), tVxB(_, _, kt));
      gemm(tiled_mma_v, tVrA(_, _, kt), tVrB(_, _, kt), tVrAcc);
    }
    __syncthreads();
  }

  // (G) Normalize per-thread acc by per-row l, then write to SMEM for
  // the InvFWHT_512 epilogue.
  CUTE_UNROLL
  for (int i = 0; i < size(tVrAcc); ++i) {
    auto coord = tVcAcc(i);
    const int m_idx = get<0>(coord);
    const float l = ss.softmax_state_l[m_idx];
    tVrAcc(i) = (l > 0.0f) ? (tVrAcc(i) / l) : 0.0f;
  }

  // Stash acc into k_smem as FP32 (reuse SMEM; need (kBlockH,
  // kLatentDim) FP32 = 16*512*4 = 32 KB; k_smem is 73 KB BF16). We
  // alias via reinterpret_cast.
  float* acc_fp32_smem = reinterpret_cast<float*>(ss.k_smem);
  CUTE_UNROLL
  for (int i = 0; i < size(tVrAcc); ++i) {
    auto coord = tVcAcc(i);
    const int m_idx = get<0>(coord);
    const int n_idx = get<1>(coord);
    acc_fp32_smem[m_idx * kLatentDim + n_idx] = tVrAcc(i);
  }
  __syncthreads();

  // (H) InvFWHT_512 per head and write output. Reuse p_fp32 SMEM as
  // FWHT_128 scratchpad.
  float* fwht_scratch_post = ss.p_fp32;
  for (int m_local = 0; m_local < kBlockH; ++m_local) {
    const int m_global = head_base + m_local;
    const bool valid = m_global < num_heads;
    float* acc_row = acc_fp32_smem + m_local * kLatentDim;
    float v0 = acc_row[0 * 128 + tid];
    float v1 = acc_row[1 * 128 + tid];
    float v2 = acc_row[2 * 128 + tid];
    float v3 = acc_row[3 * 128 + tid];

    fwht_register_top2(v0, v1, v2, v3);
    __syncthreads();
    v0 = fwht_128elem(v0, tid, fwht_scratch_post);
    __syncthreads();
    v1 = fwht_128elem(v1, tid, fwht_scratch_post);
    __syncthreads();
    v2 = fwht_128elem(v2, tid, fwht_scratch_post);
    __syncthreads();
    v3 = fwht_128elem(v3, tid, fwht_scratch_post);
    __syncthreads();

    if (valid) {
      BFloat16* out_row = out + row * out_stride_0 + m_global * out_stride_1;
      out_row[0 * 128 + tid] = __float2bfloat16(v0 * kInvSqrtLatentDim);
      out_row[1 * 128 + tid] = __float2bfloat16(v1 * kInvSqrtLatentDim);
      out_row[2 * 128 + tid] = __float2bfloat16(v2 * kInvSqrtLatentDim);
      out_row[3 * 128 + tid] = __float2bfloat16(v3 * kInvSqrtLatentDim);
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Host-side launcher.
// ---------------------------------------------------------------------------

struct HiggsDense2BitMLADecodeTCKernel {
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
        .with_dtype<BFloat16>()
        .with_device(device)
        .verify(q_nope);
    TensorMatcher({R, H, kRopeDim})
        .with_strides({q_rope_stride_0, q_rope_stride_1, 1})
        .with_dtype<BFloat16>()
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
        .with_dtype<BFloat16>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    if (R.unwrap() == 0 || H.unwrap() == 0 || K.unwrap() == 0) return;

    dim3 grid(R.unwrap(), (H.unwrap() + kBlockH - 1) / kBlockH);
    const std::size_t smem_bytes = sizeof(SharedStorage);
    cudaFuncSetAttribute(
        higgs_dense_2bit_mla_decode_tc_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    LaunchKernel(grid, kBlockThreads, device.unwrap(), smem_bytes)(
        higgs_dense_2bit_mla_decode_tc_kernel,
        static_cast<const BFloat16*>(q_nope.data_ptr()),
        static_cast<const BFloat16*>(q_rope.data_ptr()),
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<BFloat16*>(out.data_ptr()),
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

}  // namespace higgs_dense_2bit_mla_tc_detail
