// SPDX-License-Identifier: Apache-2.0
//
// HIGGS in-cubin sparse-MLA decode producer —
//   iter8 scaffold (dad3bdfca): cp.async + depth-2 SMEM ping-pong @
//                               128-thread CTA (latent-tile correctness
//                               gap: 48.5% bit-exact, max_diff 0.875).
//   iter9 PRIMARY  (this rev) : same cp.async + ping-pong pipeline at
//                               kBlockThreads = 512 = kLatentDim so the
//                               FWHT_512 "one lane per element"
//                               invariant is preserved bit-for-bit vs
//                               the iter3 production kernel.
//
// ai-blaise #19 iter9 PRIMARY vector. The iter8 scaffold introduced a
// new cp.async + depth-2 SMEM ping-pong slot prefetch pipeline that
// won +20.2% per-kernel (+9.56 ms TPOT across 61 layers) at the
// production B=128 K=2048 shape, but shipped with a known correctness
// gap on the latent tile. The gap was caused by launching the
// producer with 128 threads × 4 elements/lane while reusing
// ``fwht_512_swizzled`` — that primitive's warp-shuffle butterfly +
// SMEM-exchange addressing inherently requires 512 active lanes (one
// lane per FWHT element, each writing ``buf[swizzle(tid)]`` for tid in
// [0, 512)). Under a 128-thread CTA, the upper 384 SMEM slots stay
// stale across the 4 sequential FWHT calls and the lane-to-element
// mapping diverges from the iter3 invariant ``coord = tid & 1``.
//
// iter9 fix: revert ``kBlockThreads`` to 512, one element per thread.
// The cp.async prefetch (lanes 0..16 each issue one 16 B
// ``cp.async.ca.shared.global``; lanes 17..511 idle on the issue but
// participate in the CTA-wide commit/wait_group barrier) and depth-2
// SMEM ping-pong infrastructure survive unchanged — the cp.async win
// comes from the LDG-vs-cp.async swap (hiding slot-read latency under
// FWHT compute), which is independent of FWHT lane count.
//
// Surface area vs the iter3 production kernel
// (``higgs_dense_2bit_dequant_page_table_fp8_kernel`` in
// ``higgs_dense_2bit_kv.cuh``):
//
//   Existing (iter3): 1 CTA / slot, 512 threads, LDG slot read.
//     Each lane decodes 1 latent element. Slot read is uncached LDG;
//     the FWHT_512 swizzle happens in shared. HBM-bound at the slot
//     read (~302 MiB write across B=128 × K=2048 slots).
//
//   iter9 PRIMARY: 1 CTA / slot, 512 threads, cp.async prefetch into
//     depth-2 SMEM staging. Each lane decodes 1 latent element exactly
//     as iter3. Slot N+1 prefetch issued before slot N decode
//     completes; the decode hides under the cp.async latency. The
//     FWHT_512 + EDEN2-16 path is reused wholesale.
//
// The output is bit-exact to the iter3 kernel (same FWHT, codebook,
// scale, FP8 e4m3 saturating cast). The only deviation from iter3 is
// the slot SOURCE: iter3 reads ``slot[byte_idx]`` directly from gmem
// (uncached LDG); iter9 reads ``slot_smem[buf][byte_idx]`` from the
// cp.async-staged SMEM. The codebook value, scale, FWHT pass, and FP8
// cast are byte-identical to iter3. The microbench
// ``benchmark/kernels/bench_higgs_inline_sparse_mla_decode_iter8.py``
// compares wall-clock latency + bit-identity at production shape
// against ``dequantize_higgs_dense_2bit_page_table_fp8``.
//
// Why the 128-thread launch is NOT necessary for iter9 PRIMARY: the
// iter8 comment "block size matches the CuTe DSL producer-warp idiom"
// is forward-looking for the iter9 SECONDARY (in-cubin CUTLASS graft
// via the vendored flashinfer cute_dsl mla_decode_fp8 template, 2-3
// days, queued separately). For the standalone roll-out behind
// ``SGLANG_HIGGS_DSA_INLINE_PRODUCER``, the 512-thread launch is
// correct and necessary. The SECONDARY graft will need a different
// latent-decode path because the in-cubin producer warp processes N
// K-tile slots per launch and emits into the cubin's SMEM (not gmem)
// — the FWHT inside the cubin runs on the cubin's CTA size, decoupled
// from this producer's launch shape.
//
// Honest scope note for iter9 PRIMARY: this commit ships the bit-exact
// kernel + microbench (with new bit-identity mode) + a wire-in into
// ``HiggsDense2BitDSATokenToKVPool.get_higgs_selected_kv_buffer``
// gated by ``envs.SGLANG_HIGGS_DSA_INLINE_PRODUCER`` (default OFF).
// The iter9 SECONDARY (in-cubin graft + multi-slot CTA loop) remains
// queued for separate iters.
//
// Output row layout (identical to iter3 FP8 dequant — 576 B/row):
//   row_out[0..511]   = 512 B FP8 latent  (one lane → one byte)
//   row_out[512..575] = 64  B FP8 rope    (lanes 0..15 → 4 B each)
//
// Per-element quantization: per-tensor ``inv_kv_scale`` is applied
// before the saturating FP8 cast. Downstream attention should pass
// ``k_scale = 1 / inv_kv_scale`` via ``bmm1_scale``.

#pragma once

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "higgs_dense_2bit_kv.cuh"  // fwht_512_swizzled,
                                    // eden2_16_codebook_value,
                                    // bf16x2_t, fp8_e4m3_t,
                                    // fp8x2_e4m3_t, kLatentDim,
                                    // kRopeDim, kPairDim,
                                    // kPackedBytes, kNormBytes,
                                    // kSlotBytes, kInvSqrtLatentDim.

namespace higgs_inline_sparse_mla_detail {

// bf16x2_t, fp8_e4m3_t, fp8x2_e4m3_t are declared at global scope by
// ``sgl_kernel/utils.cuh`` (already included above); the
// codebook + FWHT helpers live in ``higgs_dense_2bit_detail``.
using ::higgs_dense_2bit_detail::eden2_16_codebook_value;
using ::higgs_dense_2bit_detail::fwht_512_swizzled;
using ::higgs_dense_2bit_detail::kCodebookSize;
using ::higgs_dense_2bit_detail::kInvSqrtLatentDim;
using ::higgs_dense_2bit_detail::kLatentDim;
using ::higgs_dense_2bit_detail::kNormBytes;
using ::higgs_dense_2bit_detail::kPackedBytes;
using ::higgs_dense_2bit_detail::kPairDim;
using ::higgs_dense_2bit_detail::kRopeDim;
using ::higgs_dense_2bit_detail::kSlotBytes;

// iter9 PRIMARY: kBlockThreads = kLatentDim = 512 (one lane per
// element). Restores the iter3 production invariant — fwht_512_swizzled
// requires 512 active lanes for its warp-shuffle butterfly +
// SMEM-exchange addressing — so the latent output is bit-exact vs
// ``higgs_dense_2bit_dequant_page_table_fp8_kernel``. The 128-thread
// iter8 launch produced a latent-tile correctness gap (48.5% bit-exact,
// max_diff 0.875) because each per-lane FWHT pass left 384/512 SMEM
// slots stale; see ``notes/higgs_dsa_iter9_recon.md`` for the full
// diagnosis.
constexpr int kBlockThreads = 512;
constexpr int kDimsPerThread = kLatentDim / kBlockThreads;  // 1
static_assert(kDimsPerThread * kBlockThreads == kLatentDim,
              "kBlockThreads must divide kLatentDim evenly");
static_assert(kBlockThreads == kLatentDim,
              "iter9 PRIMARY: kBlockThreads must equal kLatentDim — "
              "fwht_512_swizzled requires one lane per element");

// SMEM ping-pong depth. 2 buffers => slot N+1 prefetch overlaps slot N
// decode. ``__align__(16)`` is required by ``cp.async.16``; kSlotBytes
// = 272 is already 16-aligned.
constexpr int kSmemDepth = 2;

// iter9 PRIMARY multi-slot CTA loop: each CTA processes kSlotsPerCta
// output rows. With kSmemDepth = 2 staging, the prefetch of slot N+1
// is issued before the FWHT_512 of slot N completes — the cp.async
// latency hides under the FWHT compute. This is what unlocks the
// iter8 +20% speedup under the 512-thread CTA constraint: the iter8
// scaffold's 128-thread × 8-CTA/SM occupancy hid the wait_group cost
// via SM-level inter-CTA scheduling, but the 512-thread CTA only
// runs 2 CTAs/SM so we need intra-CTA pipelining to hide the
// prefetch latency.
//
// kSlotsPerCta = 4 is a conservative choice: it amortizes the
// FWHT-init and barrier costs over 4 slots while keeping the grid
// dense enough for the SM scheduler. Production shape B=128 K=2048
// has num_rows = 262144 → grid = 65536 CTAs at kSlotsPerCta = 4
// (vs 262144 at kSlotsPerCta = 1), still way over the SM count
// (B200 = 132 SMs).
constexpr int kSlotsPerCta = 4;
static_assert(kSlotsPerCta >= 1, "kSlotsPerCta must be positive");

// cp.async issue width per slot. ``kSlotBytes / 16 = 17`` → lanes
// 0..16 each issue one ``cp.async.ca.shared.global`` of 16 B; lanes
// 17..511 are inactive on this issue. Matches the existing
// ``higgs_cp_async_prefetch_slot`` helper in
// ``higgs_dense_2bit_mla_decode.cuh``.
constexpr int kCpAsyncSlotLanes = kSlotBytes / 16;  // 17
static_assert(kCpAsyncSlotLanes * 16 == kSlotBytes,
              "kSlotBytes must divide evenly into 16-byte cp.async tiles");
static_assert(kCpAsyncSlotLanes <= kBlockThreads,
              "cp.async slot prefetch must fit within one CTA");

// SMEM-flavor of the slot prefetch primitive: stage one HIGGS slot
// (272 B) from gmem into smem via 17 lanes × ``cp.async.ca`` of 16 B.
// Lanes 17..511 are inactive on this issue but participate in the
// subsequent commit/wait_group (which is CTA-wide). The source
// pointer MUST be 16-byte aligned (guaranteed by the codec's slot
// allocation thanks to the iter4 stride pad).
__device__ __forceinline__ void inline_cp_async_prefetch_slot(
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

__device__ __forceinline__ void inline_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void inline_cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ---------------------------------------------------------------------------
// (iter9 PRIMARY) The iter8 ``inline_decode_slot_latent`` helper —
// which packed 4-elements-per-lane into a 128-thread CTA with 4
// sequential ``fwht_512_swizzled`` calls — is removed. That layout
// broke the FWHT_512 invariant (the warp-shuffle + SMEM-exchange
// addressing requires 512 active lanes; under 128 threads the upper
// 384 SMEM slots were stale across the 4 sequential calls and the
// per-lane ``coord`` mapping diverged from iter3's ``coord = tid & 1``).
//
// iter9 inlines the iter3 single-element latent decode directly in
// the kernel body: each of the 512 lanes decodes exactly one element
// (lane tid → element index tid), runs ONE ``fwht_512_swizzled`` pass
// per slot, and writes one FP8 byte per slot. The codebook/scale/FWHT
// path is byte-identical to iter3; the only deviation is the slot
// SOURCE (cp.async-staged SMEM vs uncached LDG).
// ---------------------------------------------------------------------------

// Rope tile decode for one slot. 16 lanes (tid < 16) each emit 4 FP8
// rope values; lanes 16..511 idle. Matches the iter3 kernel's rope
// emission verbatim.
__device__ __forceinline__ void inline_decode_slot_rope(
    const uint8_t* __restrict__ slot_smem,
    fp8_e4m3_t* __restrict__ rope_out,
    float inv_kv_scale,
    int tid) {
  if (tid < 16) {
    const uint8_t* slot_rope = slot_smem + kPackedBytes + kNormBytes;
    const int base = tid * 4;
    // Stage rope bytes through a local uint8 buffer to avoid
    // misaligned-load faults (slot_rope offset is byte-aligned, not
    // bf16-aligned). The SMEM source means a register-staged read is
    // cheap and side-steps any SMEM-bank-conflict aliasing concern.
    uint8_t bf16_bytes[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      bf16_bytes[i] = slot_rope[base * 2 + i];
    }
    const bf16x2_t rope_pair0 =
        *reinterpret_cast<const bf16x2_t*>(bf16_bytes);
    const bf16x2_t rope_pair1 =
        *reinterpret_cast<const bf16x2_t*>(bf16_bytes + 4);
    const float2 r0 = __bfloat1622float2(rope_pair0);
    const float2 r1 = __bfloat1622float2(rope_pair1);
    const float2 r0_scaled =
        make_float2(r0.x * inv_kv_scale, r0.y * inv_kv_scale);
    const float2 r1_scaled =
        make_float2(r1.x * inv_kv_scale, r1.y * inv_kv_scale);
    fp8x2_e4m3_t p0 = __nv_fp8x2_e4m3(r0_scaled);
    fp8x2_e4m3_t p1 = __nv_fp8x2_e4m3(r1_scaled);
    uint16_t out_bytes[2];
    out_bytes[0] = *reinterpret_cast<const uint16_t*>(&p0);
    out_bytes[1] = *reinterpret_cast<const uint16_t*>(&p1);
    *reinterpret_cast<uint16_t*>(rope_out + base) = out_bytes[0];
    *reinterpret_cast<uint16_t*>(rope_out + base + 2) = out_bytes[1];
  }
}

// ───────────────────────────────────────────────────────────────────────
// iter9 PRIMARY producer kernel — emits FP8 e4m3 (B*K, 1, 576) from
// HIGGS slots with cp.async prefetch + depth-2 SMEM ping-pong staging
// across a kSlotsPerCta-deep streaming loop. Bit-exact vs
// ``higgs_dense_2bit_dequant_page_table_fp8_kernel`` on both latent
// and rope tiles.
//
// Grid: (ceil(num_rows / kSlotsPerCta),) — one CTA per kSlotsPerCta
// output rows.
// CTA size: kBlockThreads = 512 (one lane per latent element; lanes
//           0..15 also handle the rope tile; lanes 0..16 also issue
//           the cp.async slot prefetch).
//
// Per CTA, for slot index s in [0, kSlotsPerCta):
//   - Issue cp.async prefetch of slot s+1 (the lookahead) into the
//     ping-pong buffer (s+1) & 1, OR — if s == 0 — issue the
//     prologue prefetch of slot 0 into buffer 0.
//   - wait_group<1> — wait until at most 1 cp.async group is in
//     flight (the just-issued s+1 group; the s group has long since
//     landed). On the first iteration this drops to wait_group<0>
//     since only one group has been committed.
//   - __syncthreads — visibility for all 512 lanes.
//   - Decode rope tile + latent tile from buffer s & 1 (the now-ready
//     slot s data). Each lane decodes ONE latent element exactly as
//     iter3; the codebook/scale/FWHT/FP8-cast path is byte-identical
//     to iter3. Only the slot SOURCE differs (SMEM vs uncached LDG).
//   - Write FP8 row.
//
// The depth-2 ping-pong + lookahead overlaps the cp.async memory
// latency of slot s+1 with the FWHT_512_swizzled decode of slot s.
// This is what restores the iter8 microbench-projected speedup under
// the 512-thread CTA constraint that iter9 PRIMARY had to adopt for
// FWHT_512 correctness — the iter8 scaffold's 128-thread × 8-CTA/SM
// occupancy hid the wait_group cost via SM-level inter-CTA scheduling
// (4× more CTAs per slot), but the 512-thread CTA only fits 2 CTAs
// per SM so we need intra-CTA pipelining instead.
//
// Iter9 PRIMARY scope: this kernel runs the inline pipeline with
// kSlotsPerCta = 4 slots per CTA, depth-2 ping-pong. The iter9
// SECONDARY (in-cubin CUTLASS graft) will reuse the same
// SMEM-staging pattern but emit into the cubin's SMEM rather than
// gmem, eliminating the 302 MiB / layer-step gmem write.
// ───────────────────────────────────────────────────────────────────────

// __launch_bounds__: maxThreadsPerBlock = 512, minBlocksPerSM = 2.
// SM_100 (B200) has 1536 threads / SM, so 2 × 512-thread CTAs = 1024
// threads / SM (the 3-CTA case would need 1536 with no slack for the
// scheduler). The 8-blocks-per-SM hint from iter8 was sized for the
// 128-thread CTA; iter9's 512-thread CTA caps at 2 — which is why
// the multi-slot CTA loop above is necessary to recover the iter8
// throughput.
__global__ void __launch_bounds__(kBlockThreads, 2)
higgs_inline_sparse_mla_produce_fp8_kernel(
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    fp8_e4m3_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0,
    float inv_kv_scale) {
  const int tid = threadIdx.x;
  const int64_t base_row =
      static_cast<int64_t>(blockIdx.x) * kSlotsPerCta;
  if (base_row >= num_rows) return;

  // Depth-2 SMEM ping-pong slot staging + shared FWHT scratch. The
  // fwht_buf is reused across the kSlotsPerCta inner-loop iterations
  // (the FWHT_512 fully consumes + rewrites it per call; no inter-
  // iteration carry needed).
  __shared__ __align__(16) uint8_t slot_smem[kSmemDepth][kSlotBytes];
  __shared__ __align__(16) float fwht_buf[kLatentDim];

  // Page slots assigned to this CTA. We load all kSlotsPerCta page
  // entries up front so the cp.async issues don't stall on the page
  // table read. Lanes 0..kSlotsPerCta-1 each fetch one entry and
  // broadcast via SMEM.
  __shared__ int32_t pages[kSlotsPerCta];
  if (tid < kSlotsPerCta) {
    const int64_t row_local = base_row + tid;
    pages[tid] = (row_local < num_rows) ? page_table[row_local] : -1;
    // Compact page table emission mirrors iter3: valid row → row
    // index, invalid row → -1.
    if (row_local < num_rows) {
      compact_page_table[row_local] =
          pages[tid] >= 0 ? static_cast<int32_t>(row_local) : -1;
    }
  }
  __syncthreads();

  // Prologue: issue cp.async for the FIRST slot (s = 0) into buffer
  // 0. If slot 0 is invalid (page < 0) we skip the prefetch — the
  // decode loop below also skips invalid slots so the SMEM contents
  // don't matter.
  {
    const int32_t page0 = pages[0];
    if (page0 >= 0) {
      const uint8_t* slot_gmem =
          compressed +
          static_cast<int64_t>(page0) * compressed_stride_0;
      inline_cp_async_prefetch_slot(slot_smem[0], slot_gmem, tid);
    }
    inline_cp_async_commit();
  }

  #pragma unroll 1
  for (int s = 0; s < kSlotsPerCta; ++s) {
    const int64_t row_local = base_row + s;
    if (row_local >= num_rows) break;
    const int buf_cur = s & 1;
    const int buf_next = (s + 1) & 1;

    // Lookahead prefetch: issue cp.async for slot s+1 into the OTHER
    // ping-pong buffer before we wait on the current slot. This is
    // what overlaps the memory latency with the FWHT decode of slot s.
    if (s + 1 < kSlotsPerCta && base_row + s + 1 < num_rows) {
      const int32_t page_next = pages[s + 1];
      if (page_next >= 0) {
        const uint8_t* slot_gmem =
            compressed +
            static_cast<int64_t>(page_next) * compressed_stride_0;
        inline_cp_async_prefetch_slot(slot_smem[buf_next], slot_gmem, tid);
      }
      inline_cp_async_commit();
      // Two groups in flight (s, s+1) — wait until ≤1 remains, which
      // means the older slot s data has landed.
      inline_cp_async_wait_group<1>();
    } else {
      // No lookahead → drain whatever is in flight (just slot s).
      inline_cp_async_wait_group<0>();
    }
    __syncthreads();

    const int32_t page = pages[s];
    if (page < 0) continue;

    fp8_e4m3_t* row_out = out + row_local * out_stride_0;
    fp8_e4m3_t* rope_out = row_out + kLatentDim;
    inline_decode_slot_rope(slot_smem[buf_cur], rope_out, inv_kv_scale, tid);

    // ──────────────────────────────────────────────────────────────
    // Latent decode — bit-exact vs the iter3 production kernel
    // ``higgs_dense_2bit_dequant_page_table_fp8_kernel``. Each lane
    // tid logically owns element index tid (0..511):
    //
    //   pair_idx = tid >> 1        ∈ [0, 256)  — adjacent-tid pairing
    //   byte_idx = pair_idx >> 1   ∈ [0, 128)  — packed-byte index
    //   nibble   = pair_idx & 1    ∈ {0, 1}    — high/low nibble
    //   cb_idx   = nibble ? (packed >> 4) : (packed & 0x0F)
    //   coord    = tid & 1         ∈ {0, 1}    — codebook (x, y) pair
    //   g        = codebook[cb_idx * kPairDim + coord]
    //   scale    = __half2float(scale_h at slot+kPackedBytes)
    //   rot      = scale * g
    //   result   = fwht_512_swizzled(rot, fwht_buf) * kInvSqrtLatentDim
    //   row_out[tid] = __nv_fp8_e4m3(result * inv_kv_scale)
    //
    // The only deviation from iter3 is the slot SOURCE: iter3 reads
    // ``slot[byte_idx]`` from gmem (uncached LDG); iter9 reads
    // ``slot_smem[buf_cur][byte_idx]`` from the cp.async-staged SMEM.
    // The codebook value, scale, FWHT pass, and FP8 cast are
    // byte-identical to iter3.
    // ──────────────────────────────────────────────────────────────
    const int pair_idx = tid >> 1;
    const int byte_idx = pair_idx >> 1;
    const uint8_t packed = slot_smem[buf_cur][byte_idx];
    const uint32_t cb_idx = static_cast<uint32_t>(
        (pair_idx & 1) ? (packed >> 4) : (packed & 0x0F));

    const int coord = tid & 1;
    const float g = __ldg(&codebook[cb_idx * kPairDim + coord]);
    const half scale_h = *reinterpret_cast<const half*>(
        slot_smem[buf_cur] + kPackedBytes);
    const float scale = __half2float(scale_h);

    const float rot_recon = scale * g;
    const float result =
        fwht_512_swizzled(rot_recon, fwht_buf) * kInvSqrtLatentDim;

    // Per-tensor inv_kv_scale folds the downstream attention BMM1
    // scale absorption (iter3-matching contract: downstream must pass
    // k_scale = 1/inv_kv_scale via bmm1_scale).
    row_out[tid] = __nv_fp8_e4m3(result * inv_kv_scale);
  }

  // Drain any remaining cp.async groups so a subsequent kernel on
  // the same stream doesn't inherit a non-empty wait queue.
  inline_cp_async_wait_group<0>();
}

// ───────────────────────────────────────────────────────────────────────
// JIT-launcher class. Mirrors the launch pattern of the existing
// ``HiggsDense2BitDequantPageTableFp8Kernel::run`` (in
// ``higgs_dense_2bit_kv.cuh``) so the same ``cuda_wrappers`` plumbing
// in ``higgs_inline_sparse_mla_decode.py`` can register it. Accepts
// the same ``(B, K)`` page_table shape as the iter3 kernel so the
// microbench can A/B them at the same call site and the wire-in into
// ``HiggsDense2BitDSATokenToKVPool.get_higgs_selected_kv_buffer`` is a
// one-line swap when ``SGLANG_HIGGS_DSA_INLINE_PRODUCER`` is set.
// ───────────────────────────────────────────────────────────────────────

using ::higgs_dense_2bit_detail::kKvDim;

struct HiggsInlineSparseMLAProduceFp8Kernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView compact_page_table,
      tvm::ffi::TensorView codebook,
      double inv_kv_scale) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto B = SymbolicSize{"num_query_rows"};
    auto K = SymbolicSize{"topk"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({B, K})
        .with_strides({K, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({B, K})
        .with_strides({K, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(compact_page_table);
    TensorMatcher({N, 1, kKvDim})
        .with_strides({out_stride_0, kKvDim, 1})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    const int64_t num_rows = B.unwrap() * K.unwrap();
    if (num_rows == 0) return;

    // iter9: grid = ceil(num_rows / kSlotsPerCta) — each CTA streams
    // kSlotsPerCta output rows via the depth-2 cp.async ping-pong
    // pipeline above.
    const int64_t num_ctas =
        (num_rows + kSlotsPerCta - 1) / kSlotsPerCta;

    LaunchKernel(num_ctas, kBlockThreads, device.unwrap())(
        higgs_inline_sparse_mla_produce_fp8_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<fp8_e4m3_t*>(out.data_ptr()),
        static_cast<int32_t*>(compact_page_table.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap(),
        static_cast<float>(inv_kv_scale));
  }
};

}  // namespace higgs_inline_sparse_mla_detail
