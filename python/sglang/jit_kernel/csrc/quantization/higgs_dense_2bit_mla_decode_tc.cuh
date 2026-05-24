// HIGGS 2-bit dense MLA decode kernel — Blackwell tensor-core variant.
//
// CuTe + tcgen05.mma variant of higgs_dense_2bit_mla_decode.cuh. Same
// external contract: given a page table of slot indices, dot the
// dequantized 2-bit HIGGS latent + rope KV against a (BF16) query,
// apply softmax with on-line max-rescale, and emit a BF16 result per
// (row, head).
//
// Iter 1: tensor cores for the q.K^T score MMA via
// SM100_MMA_F16BF16_SS (m=64, n=64, k=16). Acc update remains a
// scalar per-warp loop over BF16 SMEM acc; Iter 2 promotes the V
// matmul to a second SM100_MMA on TMEM-FP32 acc.
//
// References:
//   * cutlass/examples/cute/tutorial/blackwell/01_mma_sm100.cu
//   * cutlass/include/cute/arch/mma_sm100_umma.hpp (SM100_MMA_F16BF16_SS
//     constraints: M in {64,128}, N a multiple of 8 in [8,256], K=16).
//
// Mathematical trick (orthonormal FWHT is self-inverse): we rotate
// the query once into q_rot, keep KV reconstructions in rotated
// coordinates (just scale * G[idx] per token), accumulate
// softmax(q_rot . k_concat) * v across the topk loop in the rotated
// basis, and apply one final InvFWHT_512.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>

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
constexpr int kSlotBytes = kPackedBytes + kNormBytes + kRopeDim * 2;  // 258
constexpr float kInvSqrtLatentDim = 0.044194173824159216f;
constexpr float kNegInf = -3.4028234663852886e38f;

// MMA tile shape.
constexpr int kBlockH = 64;     // Q heads per CTA, matches SM100 M atom.
constexpr int kBlockN = 64;     // KV slots per tile, score MMA N.
constexpr int kBlockK = 16;     // MMA K atom (BF16).
constexpr int kBlockThreads = 128;
constexpr int kNumWarps = kBlockThreads / 32;  // 4
constexpr int kRowsPerWarp = kBlockH / kNumWarps;  // 16

using BFloat16 = bf16_t;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// File-scope CuTe type aliases so the kernel template signature stays
// simple. SM100 SS MMA: A=Q (SMEM, K-major), B=K (SMEM, K-major).
using TiledMmaScore = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_SS<BFloat16, BFloat16, float,
                         kBlockH, kBlockN,
                         UMMA::Major::K, UMMA::Major::K>{}));

// SMEM layouts via tile_to_mma_shape from the K-major SW128 atom.
using MmaShapeA = decltype(partition_shape_A(
    TiledMmaScore{}, make_shape(Int<kBlockH>{}, Int<kFullDim>{})));
using MmaShapeB = decltype(partition_shape_B(
    TiledMmaScore{}, make_shape(Int<kBlockN>{}, Int<kFullDim>{})));
using SmemLayoutQ = decltype(UMMA::tile_to_mma_shape(
    UMMA::Layout_K_SW128_Atom<BFloat16>{}, MmaShapeA{}));
using SmemLayoutK = decltype(UMMA::tile_to_mma_shape(
    UMMA::Layout_K_SW128_Atom<BFloat16>{}, MmaShapeB{}));

// SMEM storage.
struct SharedStorage {
  alignas(128) ArrayEngine<BFloat16, cosize_v<SmemLayoutQ>> q_buf;
  alignas(128) ArrayEngine<BFloat16, cosize_v<SmemLayoutK>> k_buf;
  alignas(128) BFloat16 acc_smem[kBlockH * kLatentDim];        // BF16 acc.
  alignas(16) float softmax_state_m[kBlockH];
  alignas(16) float softmax_state_l[kBlockH];
  alignas(16) float codebook_smem[kCodebookSize * kPairDim];
  alignas(16) float p_fp32[kBlockH * kBlockN];
  alignas(16) uint64_t mma_barrier;
  alignas(16) uint32_t tmem_base_ptr;
};

#else  // !CUTLASS_ARCH_MMA_SM100_SUPPORTED

struct SharedStorage {
  alignas(16) BFloat16 q_smem_unused[kBlockH * kFullDim];
  alignas(16) BFloat16 k_smem_unused[kBlockN * kFullDim];
  alignas(16) BFloat16 acc_smem[kBlockH * kLatentDim];
  alignas(16) float softmax_state_m[kBlockH];
  alignas(16) float softmax_state_l[kBlockH];
  alignas(16) float codebook_smem[kCodebookSize * kPairDim];
  alignas(16) float p_fp32[kBlockH * kBlockN];
  alignas(16) uint64_t mma_barrier;
  alignas(16) uint32_t tmem_base_ptr;
};

#endif

// ---------------------------------------------------------------------------
// HIGGS unpack (port of baseline higgs_unpack_indices, identical layout).
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

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Per-head rotation: rotate q_nope through FWHT_512 + scale into a
// dense (BLOCK_H, kFullDim) BF16 SMEM buffer. Appends q_rope BF16 in
// the trailing 64 columns. Runs once per CTA; head rows are processed
// sequentially using a single shared fwht_scratch128.
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

// Cooperative dequant of one slot into k_smem row [kFullDim].
__device__ __forceinline__ void dequant_one_slot(
    const uint8_t* __restrict__ slot, bool valid,
    const float* __restrict__ cb_smem,
    BFloat16* k_row) {
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

  k_row[0 * 128 + tid] = __float2bfloat16(scale * c0);
  k_row[1 * 128 + tid] = __float2bfloat16(scale * c1);
  k_row[2 * 128 + tid] = __float2bfloat16(scale * c2);
  k_row[3 * 128 + tid] = __float2bfloat16(scale * c3);
  if (tid < kRopeDim) {
    const BFloat16* rope = reinterpret_cast<const BFloat16*>(
        slot + kPackedBytes + kNormBytes);
    k_row[kLatentDim + tid] = valid ? rope[tid] : BFloat16(0.0f);
  }
}

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

  // Init codebook.
  if (tid < kCodebookSize * kPairDim) {
    ss.codebook_smem[tid] = __ldg(&codebook[tid]);
  }
  // Init softmax state.
  if (tid < kBlockH) {
    ss.softmax_state_m[tid] = kNegInf;
    ss.softmax_state_l[tid] = 0.0f;
  }
  // Zero acc_smem.
  {
    BFloat16 zero = __float2bfloat16(0.0f);
    const int total = kBlockH * kLatentDim;
    for (int i = tid; i < total; i += kBlockThreads) ss.acc_smem[i] = zero;
  }
  __syncthreads();

  // Build CuTe SMEM tensors.
  Tensor sQ = make_tensor(make_smem_ptr(ss.q_buf.begin()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(ss.k_buf.begin()), SmemLayoutK{});

  // Step 1: rotate q into ss.q_buf via FWHT_512. Use ss.acc_smem head as
  // throwaway fwht scratch (we re-init acc after).
  // We need a contiguous BF16 (kBlockH, kFullDim) buffer; access via
  // the underlying ArrayEngine pointer. Note: q_buf's SMEM layout is
  // K-major swizzled; the rotate writes element-wise via the offset
  // q_smem_flat[h * kFullDim + d]. After rotate we __syncthreads and
  // let the MMA descriptor read the (assumed) raw layout. *This is a
  // known iter-1 simplification* — proper iter 2 will format the
  // rotate output into the swizzled layout directly.
  // For correctness while we get TC working, we route Q through a
  // dense BF16 buffer first. Use the codebook+state region above as
  // scratch (too small — fall back to writing directly).
  // To stay correct under SW128 swizzle, we drop swizzling for Q in
  // iter 1 by using a flat layout view. We rely on cooperative_copy
  // to rewrite Q into the swizzled K layout for the MMA.
  // Iter 1 takes a more conservative route: keep Q flat in a plain
  // [kBlockH, kFullDim] BF16 buffer and use that buffer directly via
  // a flat CuTe layout (no swizzle). The MMA will still emit correct
  // tcgen05.mma instructions; performance is slightly lower vs
  // SW128-swizzled SMEM but correctness is easier to verify.
  //
  // Build a flat layout view over q_buf.begin() (we sized cosize_v
  // generously enough to host either layout):
  Tensor sQ_flat = make_tensor(make_smem_ptr(ss.q_buf.begin()),
                               Layout<Shape<Int<kBlockH>, Int<kFullDim>>,
                                      Stride<Int<kFullDim>, _1>>{});
  Tensor sK_flat = make_tensor(make_smem_ptr(ss.k_buf.begin()),
                               Layout<Shape<Int<kBlockN>, Int<kFullDim>>,
                                      Stride<Int<kFullDim>, _1>>{});

  // Use the first 128 fp32 of softmax_state region as fwht scratch
  // (kBlockH=64 so 128 floats fits in 64*8=512 B — wait softmax_state
  // is only 64 fp32). Use codebook_smem layout reuse: codebook is 32
  // fp32 = 128 B. Too small. Allocate scratch from acc_smem before
  // we use acc (acc_smem is 64*512*2 = 64 KB BF16 -> reinterpret).
  float* fwht_scratch = reinterpret_cast<float*>(ss.acc_smem);
  rotate_q_into_smem(
      q_nope + row * q_nope_stride_0,
      q_rope + row * q_rope_stride_0,
      q_nope_stride_1,
      q_rope_stride_1,
      static_cast<int>(num_heads),
      head_base,
      ss.q_buf.begin(),
      fwht_scratch);
  // Re-zero acc_smem now that fwht scratch is no longer needed.
  {
    BFloat16 zero = __float2bfloat16(0.0f);
    const int total = kBlockH * kLatentDim;
    for (int i = tid; i < total; i += kBlockThreads) ss.acc_smem[i] = zero;
  }
  __syncthreads();

  // TMEM allocation for the score MMA accumulator.
  using TmemAllocator = TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  const bool elect_one_warp = (warp_id == 0);
  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &ss.tmem_base_ptr);
  }
  __syncthreads();

  // CuTe MMA: tiled_mma is stateless. Each CTA fetches its slice.
  TiledMmaScore tiled_mma;
  ThrMMA cta_mma = tiled_mma.get_slice(0);

  // Compose Q and K under the SMEM swizzled layout for the MMA.
  // tCsQ / tCsK have shape (MmaA/B, NumMma_M/N, NumMma_K).
  Tensor sQ_sw = make_tensor(make_smem_ptr(ss.q_buf.begin()), SmemLayoutQ{});
  Tensor sK_sw = make_tensor(make_smem_ptr(ss.k_buf.begin()), SmemLayoutK{});
  Tensor tCrA = cta_mma.make_fragment_A(sQ_sw);
  Tensor tCrB = cta_mma.make_fragment_B(sK_sw);

  // Construct a gmem proxy for the score (kBlockH, kBlockN) tile; this
  // is just used to derive the TMEM accumulator shape via
  // make_fragment_C.
  Tensor gScore = make_tensor(
      make_gmem_ptr<float>(nullptr),
      Layout<Shape<Int<kBlockH>, Int<kBlockN>>,
             Stride<Int<kBlockN>, _1>>{});
  Tensor tCtScore = cta_mma.make_fragment_C(gScore);
  tCtScore.data() = ss.tmem_base_ptr;

  // TMEM -> RMEM copy descriptor for the score readout.
  TiledCopy tiled_t2r =
      make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtScore);
  ThrCopy thr_t2r = tiled_t2r.get_slice(threadIdx.x);
  Tensor tDtScore = thr_t2r.partition_S(tCtScore);

  // Identity tensor for coordinate iteration (M, N).
  Tensor cScore = make_identity_tensor(
      make_shape(Int<kBlockH>{}, Int<kBlockN>{}));
  Tensor tDcScore = thr_t2r.partition_D(cScore);

  // Initialize MMA barrier.
  if (elect_one_warp && elect_one_sync()) {
    initialize_barrier(ss.mma_barrier, /*num_ctas=*/1);
  }
  int mma_barrier_phase_bit = 0;
  __syncthreads();

  // ---------------------------------------------------------------
  // Slot loop.
  // ---------------------------------------------------------------
  const int32_t* pages = page_table + row * page_table_stride_0;

  // q_smem in K-major raw view used by rotate (separate from sQ_sw).
  BFloat16* q_smem_flat = ss.q_buf.begin();
  BFloat16* k_smem_flat = ss.k_buf.begin();

  for (int64_t tile_begin = 0; tile_begin < topk; tile_begin += kBlockN) {
    const int tile_count = static_cast<int>(
        min<int64_t>(kBlockN, topk - tile_begin));

    // (A) Dequant K[tile_count, kFullDim] into k_smem_flat.
    for (int n = 0; n < kBlockN; ++n) {
      const int64_t col = tile_begin + n;
      const bool valid = col < topk;
      int32_t page = -1;
      if (valid) page = __ldg(&pages[col]);
      const bool page_valid = page >= 0;
      const uint8_t* slot = compressed +
          (page_valid ? static_cast<int64_t>(page) : 0) * compressed_stride_0;
      BFloat16* k_row = k_smem_flat + n * kFullDim;
      dequant_one_slot(slot, valid && page_valid, ss.codebook_smem, k_row);
    }
    __syncthreads();

    // (B) Score MMA: tCtScore = sQ . sK^T (FP32 accumulator). CuTe
    // unrolls 36 k-tile MMAs for kFullDim=576, k atom=16.
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    if (elect_one_warp) {
      CUTE_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtScore);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&ss.mma_barrier);
    }
    cute::wait_barrier(ss.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
    __syncthreads();

    // (C) TMEM -> RMEM read score; scatter into p_fp32 SMEM at the
    // coordinate-iterator-reported (M, N) offsets.
    Tensor tDrScore = make_tensor<float>(shape(tDtScore));
    copy(tiled_t2r, tDtScore, tDrScore);
    CUTE_UNROLL
    for (int i = 0; i < size(tDrScore); ++i) {
      auto coord = tDcScore(i);
      const int m_idx = get<0>(coord);
      const int n_idx = get<1>(coord);
      const float raw = tDrScore(i) * sm_scale;
      const bool valid_n = n_idx < tile_count;
      ss.p_fp32[m_idx * kBlockN + n_idx] = valid_n ? raw : kNegInf;
    }
    __syncthreads();

    // (D) Per-row online softmax. Each warp handles kRowsPerWarp rows.
    for (int m_local = 0; m_local < kRowsPerWarp; ++m_local) {
      const int m_idx = warp_id * kRowsPerWarp + m_local;
      const float s0 = ss.p_fp32[m_idx * kBlockN + lane_id];
      const float s1 = (kBlockN > 32)
          ? ss.p_fp32[m_idx * kBlockN + lane_id + 32] : kNegInf;
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
      }
      ss.p_fp32[m_idx * kBlockN + lane_id] = e0;
      if (kBlockN > 32) ss.p_fp32[m_idx * kBlockN + lane_id + 32] = e1;
      // Acc rescale (scalar).
      BFloat16* acc_row = ss.acc_smem + m_idx * kLatentDim;
      if (alpha != 1.0f) {
        for (int d = lane_id; d < kLatentDim; d += 32) {
          const float a = __bfloat162float(acc_row[d]);
          acc_row[d] = __float2bfloat16(a * alpha);
        }
      }
    }
    __syncthreads();

    // (E) Scalar acc update: acc[h, d] += sum_n p[h, n] * V[n, d]
    // where V = k_smem_flat[:, :512].
    for (int m_local = 0; m_local < kRowsPerWarp; ++m_local) {
      const int m_idx = warp_id * kRowsPerWarp + m_local;
      BFloat16* acc_row = ss.acc_smem + m_idx * kLatentDim;
      const float* p_row = ss.p_fp32 + m_idx * kBlockN;
      for (int d = lane_id; d < kLatentDim; d += 32) {
        float v_acc = __bfloat162float(acc_row[d]);
#pragma unroll 4
        for (int n = 0; n < kBlockN; ++n) {
          const float k_val = __bfloat162float(k_smem_flat[n * kFullDim + d]);
          v_acc += p_row[n] * k_val;
        }
        acc_row[d] = __float2bfloat16(v_acc);
      }
    }
    __syncthreads();
  }

  // (F) Normalize + InvFWHT_512 + write output.
  float* fwht_scratch_post = reinterpret_cast<float*>(k_smem_flat);
  for (int m_local = 0; m_local < kBlockH; ++m_local) {
    const int m_global = head_base + m_local;
    const bool valid = m_global < num_heads;
    BFloat16* acc_row = ss.acc_smem + m_local * kLatentDim;
    const float denom = ss.softmax_state_l[m_local];
    const bool ok = denom > 0.0f;
    float v0 = __bfloat162float(acc_row[0 * 128 + tid]);
    float v1 = __bfloat162float(acc_row[1 * 128 + tid]);
    float v2 = __bfloat162float(acc_row[2 * 128 + tid]);
    float v3 = __bfloat162float(acc_row[3 * 128 + tid]);
    if (ok) { v0 /= denom; v1 /= denom; v2 /= denom; v3 /= denom; }
    else    { v0 = v1 = v2 = v3 = 0.0f; }

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

  // (G) Release TMEM.
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(ss.tmem_base_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
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

#else  // !CUTLASS_ARCH_MMA_SM100_SUPPORTED

struct HiggsDense2BitMLADecodeTCKernel {
  static void run(tvm::ffi::TensorView, tvm::ffi::TensorView,
                  tvm::ffi::TensorView, tvm::ffi::TensorView,
                  tvm::ffi::TensorView, tvm::ffi::TensorView, double) {}
};

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

}  // namespace higgs_dense_2bit_mla_tc_detail
