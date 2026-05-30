// HIGGS 2-bit dense MLA decode kernels.
//
// External contract: given a page table of slot indices, dot the
// dequantized latent + rope KV against a (BF16) query, apply softmax
// with on-line max-rescale, and emit a BF16 result per (row, head).
//
// Iter3 (#16): the single-pass decode kernel and its saw_scalar2
// sibling were dropped. The split-K path at ``num_splits = 16`` runs
// ~12.5× faster than the single-pass kernel at every production batch
// the REAP B200 deploy hits (split B=1: 93 us, single B=1: 1169 us;
// split B=16: 325 us, single B=16: 1235 us — measured 2026-05-29 on
// a4-us-002-rl9 sm_100 with topk=2048, num_heads=16 for DP=TP=8).
// The Python-side split-policy selector (``select_higgs_mla_decode_*
// num_splits``) returns >= 32 at every production shape, so the
// ``num_splits <= 1`` branch in ``memory_pool.forward_higgs_dense_
// 2bit_mla_decode`` was dead anyway; the iter3 patch removes that
// branch and folds the API surface down to one entry point per codec
// variant (``higgs_dense_2bit_mla_decode_split`` for the EDEN2-16
// codebook path, ``higgs_dense_2bit_mla_decode_saw_scalar2_split``
// for the SAW scalar2 codec).
//
// Optimization pattern adopted from the TurboQuant 2.5-bit MLA decode
// kernel that lives next to this file
// (``turboquant_dense_mla_decode.cuh``): 128 threads per block, each
// holding four latent dims as register state. The full FWHT_512
// factors as four parallel FWHT_128s (warp-shuffle for levels 0..4,
// single SMEM exchange for levels 5..6) stitched together by a 4-way
// register FWHT (``fwht_register_top2``). This cuts SMEM traffic and
// lets us keep multiple resident blocks per SM on B200.
//
// Mathematical trick (orthonormal FWHT is self-inverse): we rotate the
// QUERY once into ``q_rot``, keep KV reconstructions in rotated
// coordinates (just ``scale * G[idx]`` per token), accumulate
// ``softmax(q_rot . scale * G[idx]) * scale * G[idx]`` across the topk
// loop, and apply ONE final InvFWHT to the result to bring it back to
// the original basis. This matches TurboQuant's split-rotated fast
// path.
//
// Iter4 (#16) Stage B: cp.async.16 one-slot lookahead in the split
// kernel hot loop. Each iteration:
//
//   (1) issues a 17-lane cp.async.16 transfer of the *next* slot
//       into the inactive ping-pong smem buffer (kSlotBytes = 272 =
//       17 * 16, exactly one tile per lane),
//   (2) commit_group + wait_group<1> drains the *current* iter's
//       prefetch so the slot the loop body is about to consume is
//       readable from smem,
//   (3) the hot loop reads ``higgs_unpack_indices_smem`` + the norm
//       fp16 + the rope bf16 entirely from smem; no LDG.NC in the
//       slot data path inside the loop body.
//
// This unblocks the iter3 cp.async-prefetch vector (which #16 iter3
// closed out as blocked on the bare 258-byte slot stride being only
// 2-byte aligned). The iter4 Stage A pad to 272 B = pad_up(258, 16)
// makes the cp.async source pointer 16-byte aligned for every page
// index.
//
// Occupancy note: the iter4 ping-pong smem (2 × kSlotBytes = 544 B)
// is small enough that the existing ``__launch_bounds__(128, 8)``
// resident-CTA hint still fits within the B200 sm_100 228 KB SMEM
// budget at 8 CTAs/SM (each CTA also holds 16 B warp_partials + 128
// B cb_smem + 4 KB q_rot registers; 8 × ~1.5 KB SMEM is comfortably
// within budget).

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
// Payload that every kernel actually reads (offsets in
// ``[0, kPayloadBytes)``); matches HiggsDense2BitConfig.payload_bytes.
constexpr int kPayloadBytes = kPackedBytes + kNormBytes + kRopeDim * 2;  // 258
// Iter4 (#16) per-slot stride: 16-byte aligned so ``cp.async.16``
// from the slot base is legal. Matches ``kSlotBytes`` in
// ``higgs_dense_2bit_kv.cuh``.
constexpr int kSlotAlignmentBytes = 16;
constexpr int kSlotBytes =
    (kPayloadBytes + kSlotAlignmentBytes - 1) /
    kSlotAlignmentBytes * kSlotAlignmentBytes;
constexpr int kSlotPadBytes = kSlotBytes - kPayloadBytes;
static_assert(kPayloadBytes == 258, "expected payload bytes = 258");
static_assert(kSlotBytes == 272, "expected padded slot bytes = 272");
static_assert(kSlotBytes % kSlotAlignmentBytes == 0,
              "slot stride must be 16-byte aligned for cp.async.16");
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

// ─── Iter4 (#16) Stage B: cp.async.16 one-slot lookahead helpers ────

// SMEM-flavor of higgs_unpack_indices. Same arithmetic, but reads
// the 4 packed-byte slots from a smem pointer (no LDG.NC qualifier;
// the slot has been staged into smem via cp.async).
__device__ __forceinline__ void higgs_unpack_indices_smem(
    const uint8_t* __restrict__ slot_smem, int tid,
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
    b0 = slot_smem[0 * 32 + byte_in_group];
    b1 = slot_smem[1 * 32 + byte_in_group];
    b2 = slot_smem[2 * 32 + byte_in_group];
    b3 = slot_smem[3 * 32 + byte_in_group];
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

// SMEM-flavor of saw_scalar2_unpack_pair_lanes; mirror of the smem
// variant of higgs_unpack_indices above.
__device__ __forceinline__ void saw_scalar2_unpack_pair_lanes_smem(
    const uint8_t* __restrict__ slot_smem, int tid,
    float& c0, float& c1, float& c2, float& c3) {
  uint32_t i0, i1, i2, i3;
  higgs_unpack_indices_smem(slot_smem, tid, i0, i1, i2, i3);
  const int coord_shift = (tid & 1) << 1;
  c0 = saw_scalar2_value(i0 >> coord_shift);
  c1 = saw_scalar2_value(i1 >> coord_shift);
  c2 = saw_scalar2_value(i2 >> coord_shift);
  c3 = saw_scalar2_value(i3 >> coord_shift);
}

// cp.async.16 prefetch of one HIGGS slot (kSlotBytes = 272) from
// global into smem. ``kCpAsyncSlotLanes = 17`` lanes (lane 0..16)
// each issue one ``cp.async.ca.shared.global`` of 16 B; lanes
// 17..127 are inactive. ``kSlotBytes = 17 * 16 = 272`` exactly
// (iter4 #16 stride pad). The source pointer ``slot_gmem`` MUST be
// 16-byte aligned (guaranteed by the codec's slot allocation thanks
// to the iter4 stride pad — see ``HiggsDense2BitConfig.slot_bytes``);
// ``slot_smem`` MUST be 16-byte aligned (allocate with
// ``__align__(16)``).
constexpr int kCpAsyncSlotLanes = kSlotBytes / 16;  // 17
static_assert(kCpAsyncSlotLanes * 16 == kSlotBytes,
              "kSlotBytes must divide evenly into 16-byte cp.async tiles");
static_assert(kCpAsyncSlotLanes <= kBlockThreads,
              "cp.async slot prefetch must fit within one CTA");

__device__ __forceinline__ void higgs_cp_async_prefetch_slot(
    uint8_t* slot_smem, const uint8_t* slot_gmem, int tid) {
  if (tid < kCpAsyncSlotLanes) {
    uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(slot_smem + tid * 16));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(slot_gmem + tid * 16));
  }
}

__device__ __forceinline__ void higgs_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

// Wait until at most ``N`` cp.async commit groups remain in flight.
template <int N>
__device__ __forceinline__ void higgs_cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
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

  __shared__ float warp_partials[4];
  __shared__ float cb_smem[kCodebookSize * kPairDim];
  // Iter4 (#16) Stage B: ping-pong smem staging for cp.async.16
  // one-slot lookahead. ``slot_smem[col & 1]`` holds the slot the
  // current loop iteration consumes; ``slot_smem[(col + 1) & 1]``
  // is the destination of the prefetch issued at the top of the
  // iteration. 16-byte alignment is required by cp.async.16; the
  // declaration aligns the start of the buffer.
  __shared__ __align__(16) uint8_t slot_smem[2][kSlotBytes];

  // Online-softmax state lives in registers (iter2 vector 3 — see the
  // single-pass dense-codebook kernel for the sync-reduction rationale).
  float softmax_state_m = kNegInf;
  float softmax_state_l = 0.0f;

  if (tid < kCodebookSize * kPairDim) {
    cb_smem[tid] = __ldg(&codebook[tid]);
  }

  const float* q_rot_row =
      q_rotated + row * q_rot_stride_0 + head * q_rot_stride_1;
  const float v0 = q_rot_row[0 * 128 + tid];
  const float v1 = q_rot_row[1 * 128 + tid];
  const float v2 = q_rot_row[2 * 128 + tid];
  const float v3 = q_rot_row[3 * 128 + tid];

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

  if (begin >= end) {
    // Nothing to do for this split; write zero partials and exit.
    float* mid_row_empty =
        mid + row * mid_stride_0 + head * mid_stride_1 +
        split * mid_stride_2;
    if (tid == 0) {
      mid_row_empty[0] = kNegInf;
      mid_row_empty[1] = 0.0f;
    }
    mid_row_empty[2 + 0 * 128 + tid] = 0.0f;
    mid_row_empty[2 + 1 * 128 + tid] = 0.0f;
    mid_row_empty[2 + 2 * 128 + tid] = 0.0f;
    mid_row_empty[2 + 3 * 128 + tid] = 0.0f;
    return;
  }

  // Iter4 (#16) Stage B: prologue. Issue cp.async.16 for the first
  // slot so the in-loop ``wait_group<1>`` always has a current-iter
  // group to drain. ``prev_valid`` tracks the validity of the slot
  // the loop will consume on the next iteration (i.e. the slot
  // currently being prefetched).
  const int32_t first_page = __ldg(&pages[begin]);
  const bool first_valid = first_page >= 0;
  {
    const int64_t first_page_safe =
        first_valid ? static_cast<int64_t>(first_page) : 0;
    const uint8_t* first_slot_gmem =
        compressed + first_page_safe * compressed_stride_0;
    higgs_cp_async_prefetch_slot(slot_smem[0], first_slot_gmem, tid);
    higgs_cp_async_commit();
  }
  bool prev_valid = first_valid;

  for (int64_t col = begin; col < end; ++col) {
    const int buf = static_cast<int>(col & 1);
    const int next_buf = static_cast<int>((col + 1) & 1);

    // Prefetch the next slot (col + 1) into the ping-pong slot the
    // current iteration will not consume. If col + 1 == end, the
    // ``has_next`` guard skips the cp.async but still emits a commit
    // group, so the wait-group bookkeeping stays consistent (the
    // commit group is empty on the last iteration; ``wait_group<1>``
    // is still a no-op overhead).
    bool next_valid = false;
    if (col + 1 < end) {
      const int32_t next_page = __ldg(&pages[col + 1]);
      next_valid = next_page >= 0;
      const int64_t next_page_safe =
          next_valid ? static_cast<int64_t>(next_page) : 0;
      const uint8_t* next_slot_gmem =
          compressed + next_page_safe * compressed_stride_0;
      higgs_cp_async_prefetch_slot(slot_smem[next_buf], next_slot_gmem, tid);
    }
    higgs_cp_async_commit();
    // Drain the current-iter prefetch group: leave at most 1 group in
    // flight (the next-iter prefetch we just submitted).
    higgs_cp_async_wait_group<1>();
    __syncthreads();

    const uint8_t* slot = slot_smem[buf];
    const bool valid = prev_valid;

    uint32_t i0, i1, i2, i3;
    higgs_unpack_indices_smem(slot, tid, i0, i1, i2, i3);

    const int coord = tid & 1;
    const float c0 = cb_smem[i0 * kPairDim + coord];
    const float c1 = cb_smem[i1 * kPairDim + coord];
    const float c2 = cb_smem[i2 * kPairDim + coord];
    const float c3 = cb_smem[i3 * kPairDim + coord];

    // Norm / rope read from smem (the slot has been staged).
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

    const float total =
        warp_partials[0] + warp_partials[1] +
        warp_partials[2] + warp_partials[3];
    const float score = valid ? total * sm_scale : kNegInf;
    const float old_m = softmax_state_m;
    const float old_l = softmax_state_l;
    const float new_m = fmaxf(old_m, score);
    const float alpha = __expf(old_m - new_m);
    const float beta = __expf(score - new_m);
    softmax_state_m = new_m;
    softmax_state_l = old_l * alpha + beta;

    const float beta_scaled = beta * scale;
    acc0 = acc0 * alpha + beta_scaled * c0;
    acc1 = acc1 * alpha + beta_scaled * c1;
    acc2 = acc2 * alpha + beta_scaled * c2;
    acc3 = acc3 * alpha + beta_scaled * c3;

    prev_valid = next_valid;
  }
  // Drain any remaining (empty) cp.async groups so a downstream
  // kernel that reuses the same SMEM page sees a quiescent pipeline.
  higgs_cp_async_wait_group<0>();

  // Write partials. Layout matches TurboQuant's stage1_rotated_fast:
  // mid[..., 0] = m, mid[..., 1] = l, mid[..., 2 + g*128 + tid] = acc_g.
  float* mid_row =
      mid + row * mid_stride_0 + head * mid_stride_1 +
      split * mid_stride_2;
  if (tid == 0) {
    mid_row[0] = softmax_state_m;
    mid_row[1] = softmax_state_l;
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

  __shared__ float warp_partials[4];
  // Iter4 (#16) Stage B: ping-pong smem staging for cp.async.16
  // one-slot lookahead (mirrors the dense-codebook split kernel).
  __shared__ __align__(16) uint8_t slot_smem[2][kSlotBytes];

  // Online-softmax state lives in registers (iter2 vector 3 — see the
  // single-pass dense-codebook kernel for the sync-reduction rationale).
  float softmax_state_m = kNegInf;
  float softmax_state_l = 0.0f;

  const bf16_t* q_nope_row =
      q_nope + row * q_nope_stride_0 + head * q_nope_stride_1;
  const float q0 = bf16_to_float(q_nope_row[0 * 128 + tid]);
  const float q1 = bf16_to_float(q_nope_row[1 * 128 + tid]);
  const float q2 = bf16_to_float(q_nope_row[2 * 128 + tid]);
  const float q3 = bf16_to_float(q_nope_row[3 * 128 + tid]);

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

  if (begin >= end) {
    float* mid_row_empty =
        mid + row * mid_stride_0 + head * mid_stride_1 +
        split * mid_stride_2;
    if (tid == 0) {
      mid_row_empty[0] = kNegInf;
      mid_row_empty[1] = 0.0f;
    }
    mid_row_empty[2 + 0 * 128 + tid] = 0.0f;
    mid_row_empty[2 + 1 * 128 + tid] = 0.0f;
    mid_row_empty[2 + 2 * 128 + tid] = 0.0f;
    mid_row_empty[2 + 3 * 128 + tid] = 0.0f;
    return;
  }

  // Iter4 (#16) Stage B prologue: prefetch slot at ``begin``.
  const int32_t first_page = __ldg(&pages[begin]);
  const bool first_valid = first_page >= 0;
  {
    const int64_t first_page_safe =
        first_valid ? static_cast<int64_t>(first_page) : 0;
    const uint8_t* first_slot_gmem =
        compressed + first_page_safe * compressed_stride_0;
    higgs_cp_async_prefetch_slot(slot_smem[0], first_slot_gmem, tid);
    higgs_cp_async_commit();
  }
  bool prev_valid = first_valid;

  for (int64_t col = begin; col < end; ++col) {
    const int buf = static_cast<int>(col & 1);
    const int next_buf = static_cast<int>((col + 1) & 1);

    bool next_valid = false;
    if (col + 1 < end) {
      const int32_t next_page = __ldg(&pages[col + 1]);
      next_valid = next_page >= 0;
      const int64_t next_page_safe =
          next_valid ? static_cast<int64_t>(next_page) : 0;
      const uint8_t* next_slot_gmem =
          compressed + next_page_safe * compressed_stride_0;
      higgs_cp_async_prefetch_slot(slot_smem[next_buf], next_slot_gmem, tid);
    }
    higgs_cp_async_commit();
    higgs_cp_async_wait_group<1>();
    __syncthreads();

    const uint8_t* slot = slot_smem[buf];
    const bool valid = prev_valid;

    float c0 = 0.0f;
    float c1 = 0.0f;
    float c2 = 0.0f;
    float c3 = 0.0f;
    if (valid) {
      saw_scalar2_unpack_pair_lanes_smem(slot, tid, c0, c1, c2, c3);
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

    const float total =
        warp_partials[0] + warp_partials[1] +
        warp_partials[2] + warp_partials[3];
    const float score = valid ? total * sm_scale : kNegInf;
    const float old_m = softmax_state_m;
    const float old_l = softmax_state_l;
    const float new_m = fmaxf(old_m, score);
    const float alpha = __expf(old_m - new_m);
    const float beta = __expf(score - new_m);
    softmax_state_m = new_m;
    softmax_state_l = old_l * alpha + beta;

    acc0 = acc0 * alpha + beta * c0;
    acc1 = acc1 * alpha + beta * c1;
    acc2 = acc2 * alpha + beta * c2;
    acc3 = acc3 * alpha + beta * c3;

    prev_valid = next_valid;
  }
  higgs_cp_async_wait_group<0>();

  float* mid_row =
      mid + row * mid_stride_0 + head * mid_stride_1 +
      split * mid_stride_2;
  if (tid == 0) {
    mid_row[0] = softmax_state_m;
    mid_row[1] = softmax_state_l;
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
