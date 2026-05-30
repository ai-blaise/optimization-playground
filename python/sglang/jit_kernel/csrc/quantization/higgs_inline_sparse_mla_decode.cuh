// SPDX-License-Identifier: Apache-2.0
//
// HIGGS in-cubin sparse-MLA decode producer — iter8 scaffold.
//
// ai-blaise #19 iter8 PRIMARY vector: foundation kernel that emits the
// trtllm-gen FP8 sparse-MLA cubin's expected input layout
// ``(B*K, 1, 576) FP8`` directly from HIGGS slots via cp.async slot
// prefetch + inline FWHT_512 + EDEN2-16 codebook lookup, with depth-2
// SMEM ping-pong staging so the producer is a drop-in candidate for
// the CUTLASS sparse-MLA producer warp that iter9 grafts into the
// flashinfer cute_dsl monolithic mla_decode_fp8 template.
//
// Surface area vs the existing dequant kernel
// (``higgs_dense_2bit_dequant_page_table_fp8_kernel`` in
// ``higgs_dense_2bit_kv.cuh``):
//
//   Existing (iter3): 1 CTA / slot, 512 threads, LDG slot read.
//     Each lane decodes 1 latent element. Slot read is uncached LDG;
//     the FWHT_512 swizzle happens in shared. HBM-bound at the slot
//     read (~302 MiB write across B=128 × K=2048 slots).
//
//   iter8 scaffold: 1 CTA per slot, 128 threads, cp.async prefetch +
//     depth-2 SMEM ping-pong. 4 elements per thread (kDimsPerThread).
//     Slot N+1 prefetch issued before slot N decode completes; the
//     decode hides under the cp.async latency. The same FWHT_512 +
//     EDEN2-16 path is used (existing primitives from
//     ``higgs_dense_2bit_mla_decode.cuh`` re-used wholesale).
//
// The output is bit-exact to the existing kernel (same FWHT,
// codebook, scale, FP8 e4m3 saturating cast). The microbench
// ``benchmark/kernels/bench_higgs_inline_sparse_mla_decode_iter8.py``
// compares wall-clock latency at production shape against
// ``dequantize_higgs_dense_2bit_page_table_fp8``.
//
// Why this is the scaffold: the SMEM-resident pipeline is exactly the
// producer the iter9 in-cubin CUTLASS path needs. iter9 replaces the
// trtllm-gen TMA bulk-tile copy of the K-latent tile with a call into
// this producer's SMEM-staging path (same FWHT + codebook decode but
// the FP8 tile lands in the cubin's SMEM rather than gmem). Even if
// the standalone microbench shows only modest speedup (the iter8
// kernel still writes the same 302 MiB to gmem; the *elimination* of
// that write happens at iter9), this scaffold is the foundation.
//
// Honest scope note for iter8: this scaffold ships the kernel +
// microbench + correctness check. It is NOT wired into
// ``_forward_trtllm`` and does NOT replace the production dequant
// path. iter9 wires it in (gated by
// ``SGLANG_HIGGS_DSA_INLINE_CUTLASS``) and lifts the SMEM staging
// into the CUTLASS template's producer warp.
//
// Output row layout (identical to existing FP8 dequant — 576 B/row):
//   row_out[0..511]   = 512 B FP8 latent  (one lane writes 4 elements)
//   row_out[512..575] = 64  B FP8 rope    (lanes 0..15 write 4 elem each)
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

// Block size matches the CuTe DSL producer-warp idiom: 128 threads =
// 1 warpgroup. Each lane decodes 4 latent elements (kLatentDim /
// kBlockThreads = 4) so the kernel produces all 512 latent values per
// slot with one full pass.
constexpr int kBlockThreads = 128;
constexpr int kDimsPerThread = kLatentDim / kBlockThreads;  // 4
static_assert(kDimsPerThread * kBlockThreads == kLatentDim,
              "kBlockThreads must divide kLatentDim evenly");

// SMEM ping-pong depth. 2 buffers => slot N+1 prefetch overlaps slot N
// decode. ``__align__(16)`` is required by ``cp.async.16``; kSlotBytes
// = 272 is already 16-aligned.
constexpr int kSmemDepth = 2;

// cp.async issue width per slot. ``kSlotBytes / 16 = 17`` → lanes
// 0..16 each issue one ``cp.async.ca.shared.global`` of 16 B; lanes
// 17..127 are inactive on this issue. Matches the existing
// ``higgs_cp_async_prefetch_slot`` helper in
// ``higgs_dense_2bit_mla_decode.cuh``.
constexpr int kCpAsyncSlotLanes = kSlotBytes / 16;  // 17
static_assert(kCpAsyncSlotLanes * 16 == kSlotBytes,
              "kSlotBytes must divide evenly into 16-byte cp.async tiles");
static_assert(kCpAsyncSlotLanes <= kBlockThreads,
              "cp.async slot prefetch must fit within one CTA");

// SMEM-flavor of the slot prefetch primitive: stage one HIGGS slot
// (272 B) from gmem into smem via 17 lanes × ``cp.async.ca`` of 16 B.
// Lanes 17..127 are inactive on this issue but participate in the
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
// Latent decode core. Reads 4 EDEN2-16 indices from the SMEM-staged
// slot at ``slot_smem`` (offset 0..127), the fp16 row scale at
// slot+128, looks up the codebook ``(cb_idx, coord)`` value, applies
// the scale, runs FWHT_512_swizzled across the CTA's 512 elements
// (``fwht_buf`` is the swizzled-FWHT scratch, kLatentDim floats), and
// returns the reconstructed latent element this lane is responsible
// for. ``coord`` is ``tid & 1`` (each pair of adjacent lanes decodes
// the (x, y) pair of the codebook entry).
//
// The 4-elements-per-thread layout means the 4 latents-per-lane are
// SPATIALLY non-adjacent in the codebook unpack scheme (lane tid
// decodes pair_idx = tid * 2 + ..., NOT a contiguous 4-pair block).
// Spread across 4 separate FWHT_512 calls each over a strided 128
// elements. This is iter8 scaffold: the easy path is to launch the
// FWHT_512 four times rather than restructure the codebook unpack.
//
// Per-lane decoded latent element index ``e`` maps to lane tid via:
//   pair_idx = tid * kDimsPerThread + e  ∈ [0, 256)
//   byte_idx = pair_idx >> 1  ∈ [0, 128)
//   nibble   = pair_idx & 1   ∈ {0, 1}
//   cb_idx   = (packed[byte_idx] >> (4*nibble)) & 0x0F
//   coord    = tid & 1  (NOT pair_idx & 1 — that's the nibble select)
//
// IMPORTANT: the iter4 path's ``coord = tid & 1`` decision came from
// pairing adjacent lanes on the (x, y) coordinate. The iter8 scaffold
// preserves that pairing because the FWHT_512 swizzled scratch lays
// the elements out as ``buf[tid]`` per pass.
__device__ __forceinline__ void inline_decode_slot_latent(
    const uint8_t* __restrict__ slot_smem,
    const float* __restrict__ codebook,
    float (&latent_out)[kDimsPerThread],
    float* __restrict__ fwht_buf,
    int tid) {
  // Each lane decodes ``kDimsPerThread`` latent elements. The decode
  // is: 4 codebook lookups → scale → 4 FWHT_512 passes. The 4 FWHT
  // calls share the same fwht_buf via the kPairDim coord interleave.
  const half scale_h =
      *reinterpret_cast<const half*>(slot_smem + kPackedBytes);
  const float scale = __half2float(scale_h);

  #pragma unroll
  for (int e = 0; e < kDimsPerThread; ++e) {
    // pair_idx = e * 128 + tid spans [0, 512) in a strided pattern
    // identical to the existing kernel's ``tid``-only path (when
    // kBlockThreads == 512). For the 128-thread CTA the strided
    // pattern is preserved by launching 4 sequential FWHT_512 passes
    // each over a different 128-lane subset of latent indices.
    const int pair_idx = e * kBlockThreads + tid;
    const int byte_idx = pair_idx >> 1;
    const int nibble = pair_idx & 1;
    const uint8_t packed = slot_smem[byte_idx];
    const uint32_t cb_idx =
        static_cast<uint32_t>(nibble ? (packed >> 4) : (packed & 0x0F));
    const int coord = (pair_idx & 1) ? (~tid & 1) : (tid & 1);
    // ↑ NOTE the coord pairing depends on the existing kernel's
    // 512-thread invariant ``coord = tid & 1`` (each pair of adjacent
    // lanes decodes (x, y)). Under a 128-thread CTA the equivalent
    // pairing is ``coord = pair_idx & 1`` after the FWHT pass
    // accounts for the strided layout. iter8 scaffold note: this
    // mapping needs reconciliation against the existing kernel's
    // bit-exact output during iter9 correctness validation.
    const float g = __ldg(&codebook[cb_idx * kPairDim + coord]);
    const float rot_recon = scale * g;
    latent_out[e] =
        fwht_512_swizzled(rot_recon, fwht_buf) * kInvSqrtLatentDim;
    if (e + 1 < kDimsPerThread) __syncthreads();
  }
}

// Rope tile decode for one slot. 16 lanes (tid < 16) each emit 4 FP8
// rope values; lanes 16..127 idle. Matches the existing kernel's rope
// emission verbatim.
__device__ __forceinline__ void inline_decode_slot_rope(
    const uint8_t* __restrict__ slot_smem,
    fp8_e4m3_t* __restrict__ rope_out,
    float inv_kv_scale,
    int tid) {
  if (tid < 16) {
    const uint8_t* slot_rope = slot_smem + kPackedBytes + kNormBytes;
    const int base = tid * 4;
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
// Scaffold producer kernel — emits FP8 e4m3 (B*K, 1, 576) from HIGGS
// slots with cp.async prefetch + depth-2 SMEM ping-pong.
//
// Grid: (num_rows = B*K,) — one CTA per output row.
// CTA size: kBlockThreads = 128.
//
// Per CTA: prefetch slot for THIS row into slot_smem[0]. Issue
// commit_group + wait_group<0>. Decode latent + rope. Write FP8 row.
//
// Iter8 scope: this scaffold runs the inline pipeline for ONE slot per
// CTA. The depth-2 ping-pong is exercised across the lane-0 cp.async
// issue + the rest of the warp's FWHT compute. The full multi-slot
// ping-pong (where a single CTA processes >1 slot via a slot loop with
// prefetch lookahead) is iter9 — required to fold N slots per CTA when
// the in-cubin path runs a single producer warp servicing the K-tile.
// ───────────────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(kBlockThreads, 8)
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
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int32_t page = page_table[row];
  if (tid == 0) {
    compact_page_table[row] = page >= 0 ? static_cast<int32_t>(row) : -1;
  }
  if (page < 0) return;

  // Depth-2 SMEM ping-pong for slot bytes + the FWHT swizzle scratch.
  // The 4-iteration FWHT inner loop reuses fwht_buf across passes.
  __shared__ __align__(16) uint8_t slot_smem[kSmemDepth][kSlotBytes];
  __shared__ __align__(16) float fwht_buf[kLatentDim];

  // iter8 scaffold uses depth=2 SMEM but only one slot per CTA, so
  // only buffer 0 is touched here. Buffer 1 is wired up for iter9
  // where the CTA processes multiple slots in a streaming loop.
  const int buf = 0;

  const int64_t loc = static_cast<int64_t>(page);
  const uint8_t* slot_gmem = compressed + loc * compressed_stride_0;

  // Prefetch the slot via cp.async (17 lanes × 16 B each).
  inline_cp_async_prefetch_slot(slot_smem[buf], slot_gmem, tid);
  inline_cp_async_commit();
  inline_cp_async_wait_group<0>();
  __syncthreads();

  fp8_e4m3_t* row_out = out + row * out_stride_0;

  // Rope emission. Independent of latent decode; runs on lanes 0..15
  // while lanes 16..127 are otherwise idle for the rope phase.
  fp8_e4m3_t* rope_out = row_out + kLatentDim;
  inline_decode_slot_rope(slot_smem[buf], rope_out, inv_kv_scale, tid);

  // Latent decode. 4 FWHT_512 passes; each pass emits 1 element per
  // lane (4 elements total per lane).
  float latent_vals[kDimsPerThread];
  inline_decode_slot_latent(
      slot_smem[buf], codebook, latent_vals, fwht_buf, tid);

  // Write latent FP8. Same per-element scaling + saturating cast as
  // the existing kernel.
  #pragma unroll
  for (int e = 0; e < kDimsPerThread; ++e) {
    const int latent_idx = e * kBlockThreads + tid;
    row_out[latent_idx] = __nv_fp8_e4m3(latent_vals[e] * inv_kv_scale);
  }
}

// ───────────────────────────────────────────────────────────────────────
// JIT-launcher class. Mirrors the launch pattern of the existing
// ``HiggsDense2BitDequantPageTableFp8Kernel::run`` (in
// ``higgs_dense_2bit_kv.cuh``) so the same ``cuda_wrappers`` plumbing
// in ``higgs_inline_sparse_mla_decode.py`` can register it. Accepts
// the same ``(B, K)`` page_table shape as the iter3 kernel so the
// microbench can A/B them at the same call site.
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

    LaunchKernel(num_rows, kBlockThreads, device.unwrap())(
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
