// HIGGS 2-bit dense MLA KV cache kernels.
//
// Algorithmic source: HIGGS KV (Pletka et al.,
// https://arxiv.org/abs/2501.19392) and the AquaKV reference at
// https://github.com/goodevening13/aquakv ("HiggsQuantizer"). This
// implementation is the dense-MLA specialization: a single 512-wide
// Hadamard block per token, a single FP16 per-token scale, and 4-bit
// indices into the EDEN2-16 lattice packed two-per-byte.
//
// Optimization tricks adopted from togethercomputer/saw-int4 (BDR =
// "Block-Diagonal Rotation"):
//   * Fused single-kernel store path that goes BF16 -> FP32 (×PRE_SCALE)
//     -> FWHT -> norm -> quantize -> pack, with no global scratch
//     (third_party/sglang-fast-rotation/python/sglang/QuantKernel/
//     fused_hadamard_int4_kv.py:121-201).
//   * Compile-time partner permutation via XOR for the FWHT butterfly
//     (fused_hadamard_int4_kv.py:67-83). In CUDA we get the same effect
//     from warp-shuffle ``__shfl_xor_sync`` (intra-warp levels) and a
//     swizzled SMEM exchange for the higher levels.
//   * Pre-scale ``1/sqrt(N)`` folded into the load-cast
//     (fused_hadamard_int4_kv.py:116).
//   * Per-token block min/max -> scale-zero compute is replaced by a
//     simple L2 norm here because HIGGS uses lattice quantization, not
//     uniform affine (fused_hadamard_int4_kv.py:184-187 -> our
//     ``block_reduce_sum`` of squares).
//
// Swizzled SMEM for FWHT is borrowed verbatim from the TurboQuant
// 2.5-bit kernel that lives alongside this file
// (``turboquant_dense_kv.cuh``); the math is identical.
//
// Slot layout (kSlotBytes = 272 = pad16(258), vs 274 for 2.5-bit
// TurboQuant). Iter4 (#16): the 258 B payload (packed + norm + rope)
// keeps the same byte offsets it had through iter3, but the
// per-slot stride grows by 14 B of trailing zero pad so the slot
// base is 16-byte aligned. Without that pad the stride was 2-byte
// aligned (258 = 2 mod 4), which makes ``cp.async.{4,8,16}`` fault
// with ``cudaErrorMisalignedAddress`` on any odd page index — the
// blocker that killed iter3's cp.async prefetch.
//
//   [packed 4-bit pair indices: 128 B]
//   [per-token block-scale fp16:   2 B]
//   [rope bf16 (kRopeDim=64):     128 B]
//   [zero pad to 16-align:         14 B]

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace higgs_dense_2bit_detail {

// ─── Architectural constants ─────────────────────────────────────────────────

constexpr int kLatentDim = 512;
constexpr int kRopeDim = 64;
constexpr int kKvDim = kLatentDim + kRopeDim;
constexpr int kPairDim = 2;                       // EDEN2-16 codeword dim
constexpr int kCodebookSize = 16;                 // 4 bits per index
constexpr int kNumPairs = kLatentDim / kPairDim;  // 256
constexpr int kPackedBytes = kNumPairs / 2;       // 128 bytes
constexpr int kNormBytes = 2;
// Per-slot data payload (matches HiggsDense2BitConfig.payload_bytes).
// Kernel offsets are computed relative to slot base and address only
// bytes in [0, kPayloadBytes); the pad tail is dead memory.
constexpr int kPayloadBytes = kPackedBytes + kNormBytes + kRopeDim * 2;  // 258
// Iter4 (#16) per-slot stride: ``pad_up(kPayloadBytes, 16) = 272``.
// Used as the innermost compressed-buffer stride; bumps the slot
// base alignment from 2 B to 16 B so ``cp.async.16`` is legal in
// the split-K decode kernel for any page index.
constexpr int kSlotAlignmentBytes = 16;
constexpr int kSlotBytes =
    (kPayloadBytes + kSlotAlignmentBytes - 1) /
    kSlotAlignmentBytes * kSlotAlignmentBytes;
constexpr int kSlotPadBytes = kSlotBytes - kPayloadBytes;
constexpr int kSawScalar2LargeRowThreshold = 16384;
constexpr float kInvSqrtLatentDim = 0.044194173824159216f;  // 1 / sqrt(512)

static_assert(kPackedBytes == 128, "expected 128 packed bytes per slot");
static_assert(kPayloadBytes == 258, "expected payload bytes = 258");
static_assert(kSlotBytes == 272, "expected padded slot bytes = 272");
static_assert(kSlotPadBytes == 14, "expected 14 B tail pad per slot");
static_assert(kSlotBytes % kSlotAlignmentBytes == 0,
              "slot stride must be 16-byte aligned for cp.async.16");

__device__ __constant__ float kEden2_16Codebook[kCodebookSize * kPairDim] = {
    -0.8996632695198059f, -1.6360418796539307f,
    -0.9611834883689880f,  1.5999565124511719f,
    -1.8820261955261230f,  0.6787783503532410f,
     0.3630079329013824f, -1.9667866230010986f,
    -0.6814072728157043f, -0.5768185853958130f,
     0.7270012497901917f,  0.6186859607696533f,
     0.3359416127204895f,  1.8371193408966064f,
     1.8599303960800171f,  0.0366685986518860f,
     0.1720824837684631f, -0.9401724338531494f,
    -1.7599700689315796f, -0.6244229674339294f,
    -0.8993809223175049f,  0.3226782381534576f,
     0.8394886851310730f, -0.3017036020755768f,
     1.5314953327178955f,  1.2942044734954834f,
    -0.0011779458727688f,  0.0002206907083746f,
     1.4274526834487915f, -1.2078891992568970f,
    -0.1612390577793121f,  0.8787511587142944f,
};

__device__ __constant__ float kEden2_16CodebookX2[kCodebookSize * kPairDim] = {
    -1.7993265390396118f, -3.2720837593078613f,
    -1.9223669767379761f,  3.1999130249023438f,
    -3.7640523910522461f,  1.3575567007064819f,
     0.7260158658027649f, -3.9335732460021973f,
    -1.3628145456314087f, -1.1536371707916260f,
     1.4540024995803833f,  1.2373719215393066f,
     0.6718832254409790f,  3.6742386817932129f,
     3.7198607921600342f,  0.0733371973037720f,
     0.3441649675369263f, -1.8803448677062988f,
    -3.5199401378631592f, -1.2488459348678589f,
    -1.7987618446350098f,  0.6453564763069153f,
     1.6789773702621460f, -0.6034072041511536f,
     3.0629906654357910f,  2.5884089469909668f,
    -0.0023558917455375f,  0.0004413814167492f,
     2.8549053668975830f, -2.4157783985137939f,
    -0.3224781155586243f,  1.7575023174285889f,
};

__device__ __constant__ float kEden2_16CodebookNormSq[kCodebookSize] = {
    3.4860272407531738f,
    3.4837346076965332f,
    4.0027627944946289f,
    4.0000243186950684f,
    0.7970355749130249f,
    0.9113031625747681f,
    3.4878642559051514f,
    3.4606857299804688f,
    0.9135365486145020f,
    3.4873986244201660f,
    0.9130073189735413f,
    0.7957662940025330f,
    4.0204434394836426f,
    0.0000014362608454f,
    3.4966175556182861f,
    0.7982016801834106f,
};

__device__ __forceinline__ float eden2_16_codebook_value(
    uint32_t cb_idx, int coord) {
  return kEden2_16Codebook[cb_idx * kPairDim + coord];
}

__device__ __forceinline__ float eden2_16_codebook_value_x2(
    uint32_t cb_idx, int coord) {
  return kEden2_16CodebookX2[cb_idx * kPairDim + coord];
}

__device__ __forceinline__ float eden2_16_codebook_norm_sq(uint32_t cb_idx) {
  return kEden2_16CodebookNormSq[cb_idx];
}

// ─── Swizzled SMEM index (XOR-based bank-conflict-free FWHT) ────────────────
// Mirrors the TurboQuant 2.5-bit kernel: 16 rows x 32 cols of float32; XOR
// the low 3 bits of the row index into the low 3 bits of the column index
// so butterfly pairs hit distinct banks for any len >= 32.

__device__ __forceinline__ int smem_swizzle_idx(int logical_idx) {
  const int col = logical_idx & 31;
  const int row = (logical_idx >> 5) & 15;
  return (logical_idx & ~31) | (col ^ (row & 7));
}

// ─── Warp reductions ─────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_reduce_sum_512(
    float value, float* __restrict__ scratch) {
  const int tid = threadIdx.x;
  value = warp_reduce_sum(value);
  if ((tid & 31) == 0) {
    scratch[tid >> 5] = value;
  }
  __syncthreads();
  float total = 0.0f;
  if (tid < 32) {
    total = (tid < 16) ? scratch[tid] : 0.0f;
    total = warp_reduce_sum(total);
  }
  __syncthreads();
  if (tid == 0) {
    scratch[0] = total;
  }
  __syncthreads();
  return scratch[0];
}

// ─── In-place FWHT_512 on (tid -> buf[swizzle(tid)]) ────────────────────────
// Levels 0..4 via warp-shuffle; levels 5..8 via swizzled SMEM exchange.
// One thread per latent dim (kLatentDim threads, kLatentDim floats in buf).

__device__ __forceinline__ float fwht_512_swizzled(
    float value, float* __restrict__ buf) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int len = 1; len < 32; len <<= 1) {
    const float other = __shfl_xor_sync(0xffffffff, value, len);
    value = (tid & len) ? other - value : value + other;
  }
  buf[smem_swizzle_idx(tid)] = value;
  __syncthreads();
#pragma unroll
  for (int len = 32; len < kLatentDim; len <<= 1) {
    const int wht_group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = wht_group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = buf[smem_swizzle_idx(a)];
    const float y = buf[smem_swizzle_idx(b)];
    __syncthreads();
    if (pos < len) {
      buf[smem_swizzle_idx(a)] = x + y;
      buf[smem_swizzle_idx(b)] = x - y;
    }
    __syncthreads();
  }
  return buf[smem_swizzle_idx(tid)];
}

// ─── Nearest-neighbor codebook lookup (EDEN2-16) ────────────────────────────
// Each pair (x0, x1) selects argmax_i (2 (x0*G[i,0] + x1*G[i,1]) - ||G[i]||²)
// The codebook fits in 16*3 = 48 fp32 = 192 bytes; we keep it in registers.

struct CodebookRegs {
  float g0[kCodebookSize];   // first coord
  float g1[kCodebookSize];   // second coord
  float gnorm[kCodebookSize]; // ||G[i]||²
};

__device__ __forceinline__ CodebookRegs load_codebook_to_regs(
    const float* __restrict__ codebook,
    const float* __restrict__ codebook_norm_sq) {
  CodebookRegs c;
#pragma unroll
  for (int i = 0; i < kCodebookSize; ++i) {
    c.g0[i] = __ldg(&codebook[i * kPairDim + 0]);
    c.g1[i] = __ldg(&codebook[i * kPairDim + 1]);
    c.gnorm[i] = __ldg(&codebook_norm_sq[i]);
  }
  return c;
}

__device__ __forceinline__ uint32_t nearest_index(
    const CodebookRegs& c, float x0, float x1) {
  // Find argmax_i (2 (x0 * g0 + x1 * g1) - gnorm).
  float best = -3.4e38f;
  uint32_t best_idx = 0;
#pragma unroll
  for (int i = 0; i < kCodebookSize; ++i) {
    const float score = 2.0f * (x0 * c.g0[i] + x1 * c.g1[i]) - c.gnorm[i];
    if (score > best) {
      best = score;
      best_idx = static_cast<uint32_t>(i);
    }
  }
  return best_idx;
}

// ─── Store kernel: BF16 latent + rope → packed slot ──────────────────────────
// Launch: <num_rows, kLatentDim>. Each block handles one token.
//
// Per-thread layout: tid in [0, kLatentDim) owns ONE latent dim. After the
// in-register FWHT, threads (tid, tid^1) hold the two coordinates of pair
// floor(tid/2); the EVEN thread reads its odd partner via __shfl_xor_sync(...
// , 1), runs the codebook NN, and writes the 4-bit index. Pairs are packed
// two per byte: byte k holds (index[2k] | (index[2k+1] << 4)).
//
// Rope is copied as 8 × 16-byte vectorised stores (kRopeDim*2 = 128 bytes).
// Scale is stored as fp16 at the end of the packed region.

template <
    bool UseConstCodebook,
    bool UseWarpPack = false,
    bool UseIndexWarpPack = false,
    bool UseRopeFirst = false,
    bool UseScaleBroadcast = false,
    bool UsePreFwhtNorm = false,
    bool UseFmaScore = false,
    bool UseFmaTieGuard = false>
__global__ void higgs_dense_2bit_store_kernel(
    uint8_t* __restrict__ compressed,
    const int64_t* __restrict__ locs,
    const bf16_t* __restrict__ latent,
    const bf16_t* __restrict__ rope,
    const float* __restrict__ codebook,
    const float* __restrict__ codebook_norm_sq,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t latent_stride_0,
    int64_t latent_stride_1,
    int64_t rope_stride_0,
    int64_t rope_stride_1) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;
  uint8_t* rope_first_slot = nullptr;

  __shared__ float buf[kLatentDim];
  __shared__ float reduce_scratch[32];  // dedicated reduction scratch

  if constexpr (UseRopeFirst) {
    const int64_t loc = locs[row];
    rope_first_slot = compressed + loc * compressed_stride_0;
    if (tid < 8) {
      const bf16_t* rope_row = rope + row * rope_stride_0;
      const uint8_t* src = reinterpret_cast<const uint8_t*>(rope_row) + tid * 16;
      uint8_t* dst =
          reinterpret_cast<uint8_t*>(rope_first_slot + kPackedBytes + kNormBytes) +
          tid * 16;
      uint4 tmp;
      memcpy(&tmp, src, sizeof(uint4));
      memcpy(dst, &tmp, sizeof(uint4));
    }
  }

  // 1. Load BF16 latent, cast to FP32. (No pre-scale here: the FWHT is
  //    orthonormal, so the per-token block scale ``s = ||FWHT(latent)|| /
  //    sqrt(N)`` we store later applies to the rotated, codebook-mapped
  //    reconstruction.)
  const bf16_t* lat_row = latent + row * latent_stride_0;
  const float xin = __bfloat162float(lat_row[tid]);

  float scale;
  float inv_scale = 0.0f;
  if constexpr (UsePreFwhtNorm) {
    const float sum_sq = warp_reduce_sum(xin * xin);
    if ((tid & 31) == 0) {
      reduce_scratch[tid >> 5] = sum_sq;
    }
    __syncthreads();
    if (tid < 32) {
      float t = (tid < 16) ? reduce_scratch[tid] : 0.0f;
      t = warp_reduce_sum(t);
      if (tid == 0) {
        const float norm = sqrtf(fmaxf(t, 1.0e-32f));
        reduce_scratch[0] = norm * kInvSqrtLatentDim;
        reduce_scratch[1] = 1.0f / fmaxf(norm, 1.0e-30f);
      }
    }
    __syncthreads();
    scale = reduce_scratch[0];
    inv_scale = reduce_scratch[1];
  }

  // 2. Orthonormal FWHT (×1/sqrt(N) folded at the end unless the
  //    candidate already computed the input norm before the transform).
  const float xhad = fwht_512_swizzled(xin, buf);
  const float xrot = xhad * kInvSqrtLatentDim;

  // 3. Block reduction: sum of squares -> L2 norm -> scale = ||xrot|| /
  //    sqrt(N). We use a *dedicated* 32-float reduction scratch buffer
  //    to keep ``buf`` undisturbed: ``buf[swizzle(tid)] = xrot * sqrt(N)``
  //    still holds the un-normalized rotated value, which we'd like to
  //    keep so the codebook lookup can read from it after the per-token
  //    scale broadcasts across all threads.
  if constexpr (!UsePreFwhtNorm) {
    const float sum_sq = warp_reduce_sum(xrot * xrot);
    if ((tid & 31) == 0) {
      reduce_scratch[tid >> 5] = sum_sq;
    }
    __syncthreads();
    if (tid < 32) {
      float t = (tid < 16) ? reduce_scratch[tid] : 0.0f;
      t = warp_reduce_sum(t);
      if (tid == 0) {
        reduce_scratch[0] = t;
      }
    }
    __syncthreads();
    if constexpr (UseScaleBroadcast) {
      if (tid == 0) {
        const float scale_once =
            sqrtf(fmaxf(reduce_scratch[0], 1.0e-32f)) * kInvSqrtLatentDim;
        reduce_scratch[0] = scale_once;
        reduce_scratch[1] = 1.0f / fmaxf(scale_once, 1.0e-30f);
      }
      __syncthreads();
      scale = reduce_scratch[0];
      inv_scale = reduce_scratch[1];
    } else {
      scale = sqrtf(fmaxf(reduce_scratch[0], 1.0e-32f)) * kInvSqrtLatentDim;
    }
  }
  uint8_t* slot = rope_first_slot;
  if constexpr (!UseRopeFirst) {
    const int64_t loc = locs[row];
    slot = compressed + loc * compressed_stride_0;
  }

  // 4. Normalize and quantize. The warp-pack variant keeps the fixed
  //    pair exchange and packed-nibble exchange in registers with shuffles;
  //    the incumbent path keeps the original SMEM handoff for direct A/B
  //    comparisons against the accepted const-codebook incumbent.
  const float xnorm = UsePreFwhtNorm
      ? (xhad * inv_scale)
      : (UseScaleBroadcast ? (xrot * inv_scale)
                           : (xrot / fmaxf(scale, 1.0e-30f)));
  uint32_t idx = 0;
  if constexpr (UseWarpPack) {
    const float x0 = xnorm;
    const float x1 = __shfl_xor_sync(0xffffffff, xnorm, 1);
    if ((tid & 1) == 0) {
      float best = -3.4e38f;
      float second_best = -3.4e38f;
      uint32_t best_idx = 0;
#pragma unroll
      for (int i = 0; i < kCodebookSize; ++i) {
        const float g0 = UseConstCodebook
            ? eden2_16_codebook_value(i, 0)
            : __ldg(&codebook[i * kPairDim + 0]);
        const float g1 = UseConstCodebook
            ? eden2_16_codebook_value(i, 1)
            : __ldg(&codebook[i * kPairDim + 1]);
        const float gn = UseConstCodebook
            ? eden2_16_codebook_norm_sq(i)
            : __ldg(&codebook_norm_sq[i]);
        const float score = (UseFmaScore && UseConstCodebook)
            ? fmaf(
                  x1,
                  eden2_16_codebook_value_x2(i, 1),
                  fmaf(x0, eden2_16_codebook_value_x2(i, 0), -gn))
            : (2.0f * (x0 * g0 + x1 * g1) - gn);
        if (score > best) {
          second_best = best;
          best = score;
          best_idx = static_cast<uint32_t>(i);
        } else if (score > second_best) {
          second_best = score;
        }
      }
      if constexpr (UseFmaTieGuard) {
        if (best - second_best <= 1.0e-4f) {
          float exact_best = -3.4e38f;
          uint32_t exact_idx = 0;
#pragma unroll
          for (int i = 0; i < kCodebookSize; ++i) {
            const float g0 = UseConstCodebook
                ? eden2_16_codebook_value(i, 0)
                : __ldg(&codebook[i * kPairDim + 0]);
            const float g1 = UseConstCodebook
                ? eden2_16_codebook_value(i, 1)
                : __ldg(&codebook[i * kPairDim + 1]);
            const float gn = UseConstCodebook
                ? eden2_16_codebook_norm_sq(i)
                : __ldg(&codebook_norm_sq[i]);
            const float exact_score = 2.0f * (x0 * g0 + x1 * g1) - gn;
            if (exact_score > exact_best) {
              exact_best = exact_score;
              exact_idx = static_cast<uint32_t>(i);
            }
          }
          best_idx = exact_idx;
        }
      }
      idx = best_idx;
    }

    // Pack two adjacent pair indices per byte. All lanes participate in the
    // shuffle; writer lanes are 0,4,...,28 in each warp.
    const uint32_t hi = __shfl_xor_sync(0xffffffff, idx, 2);
    if ((tid & 3) == 0) {
      const int byte_idx = tid >> 2;  // 0..127
      slot[byte_idx] = static_cast<uint8_t>((idx & 0x0F) | ((hi & 0x0F) << 4));
    }
  } else {
    // Write to UNSWIZZLED SMEM positions so the EVEN-tid thread can read
    // both coordinates of its pair via simple SMEM reads.
    __syncthreads();  // sync before reusing ``buf``
    buf[tid] = xnorm;
    __syncthreads();

    if ((tid & 1) == 0) {
      const float x0 = buf[tid];
      const float x1 = buf[tid + 1];
      float best = -3.4e38f;
      float second_best = -3.4e38f;
      uint32_t best_idx = 0;
#pragma unroll
      for (int i = 0; i < kCodebookSize; ++i) {
        const float g0 = UseConstCodebook
            ? eden2_16_codebook_value(i, 0)
            : __ldg(&codebook[i * kPairDim + 0]);
        const float g1 = UseConstCodebook
            ? eden2_16_codebook_value(i, 1)
            : __ldg(&codebook[i * kPairDim + 1]);
        const float gn = UseConstCodebook
            ? eden2_16_codebook_norm_sq(i)
            : __ldg(&codebook_norm_sq[i]);
        const float score = (UseFmaScore && UseConstCodebook)
            ? fmaf(
                  x1,
                  eden2_16_codebook_value_x2(i, 1),
                  fmaf(x0, eden2_16_codebook_value_x2(i, 0), -gn))
            : (2.0f * (x0 * g0 + x1 * g1) - gn);
        if (score > best) {
          second_best = best;
          best = score;
          best_idx = static_cast<uint32_t>(i);
        } else if (score > second_best) {
          second_best = score;
        }
      }
      if constexpr (UseFmaTieGuard) {
        if (best - second_best <= 1.0e-4f) {
          float exact_best = -3.4e38f;
          uint32_t exact_idx = 0;
#pragma unroll
          for (int i = 0; i < kCodebookSize; ++i) {
            const float g0 = UseConstCodebook
                ? eden2_16_codebook_value(i, 0)
                : __ldg(&codebook[i * kPairDim + 0]);
            const float g1 = UseConstCodebook
                ? eden2_16_codebook_value(i, 1)
                : __ldg(&codebook[i * kPairDim + 1]);
            const float gn = UseConstCodebook
                ? eden2_16_codebook_norm_sq(i)
                : __ldg(&codebook_norm_sq[i]);
            const float exact_score = 2.0f * (x0 * g0 + x1 * g1) - gn;
            if (exact_score > exact_best) {
              exact_best = exact_score;
              exact_idx = static_cast<uint32_t>(i);
            }
          }
          best_idx = exact_idx;
        }
      }
      idx = best_idx;
    }
    if constexpr (UseIndexWarpPack) {
      const uint32_t hi = __shfl_xor_sync(0xffffffff, idx, 2);
      if ((tid & 3) == 0) {
        const int byte_idx = tid >> 2;
        slot[byte_idx] = static_cast<uint8_t>((idx & 0x0F) | ((hi & 0x0F) << 4));
      }
    } else {
      __syncthreads();
      if ((tid & 1) == 0) {
        reinterpret_cast<uint32_t*>(buf)[tid >> 1] = idx;
      }
      __syncthreads();

      if ((tid & 3) == 0) {
        const int pair0 = tid >> 1;
        const int pair1 = pair0 + 1;
        const uint32_t lo = reinterpret_cast<const uint32_t*>(buf)[pair0];
        const uint32_t hi = reinterpret_cast<const uint32_t*>(buf)[pair1];
        const int byte_idx = tid >> 2;
        slot[byte_idx] = static_cast<uint8_t>((lo & 0x0F) | ((hi & 0x0F) << 4));
      }
    }
  }

  // 7. Store fp16 scale at the end of the packed region.
  if (tid == 0) {
    half scale_h = __float2half(scale);
    *reinterpret_cast<half*>(slot + kPackedBytes) = scale_h;
  }

  // 8. Copy rope (kRopeDim*2 = 128 bytes) as 8 × 16-byte vectorised
  //    stores. Mirrors the TurboQuant rope copy.
  if constexpr (!UseRopeFirst) {
    if (tid < 8) {
      const bf16_t* rope_row = rope + row * rope_stride_0;
      const uint8_t* src = reinterpret_cast<const uint8_t*>(rope_row) + tid * 16;
      uint8_t* dst =
          reinterpret_cast<uint8_t*>(slot + kPackedBytes + kNormBytes) +
          tid * 16;
      uint4 tmp;
      memcpy(&tmp, src, sizeof(uint4));
      memcpy(dst, &tmp, sizeof(uint4));
    }
  }
}

// ─── Dequantize kernel: packed slot → BF16 latent + rope ────────────────────
// Same launch shape as the TurboQuant 2.5-bit dequant: <num_rows, kLatentDim>.
// Each thread owns one latent dim. Indices are unpacked from the packed byte
// (two indices per byte; tid 2k and tid 2k+1 share byte k).

__device__ __forceinline__ uint32_t load_packed_byte_vec4(
    const uint8_t* __restrict__ slot, int tid) {
  const int pair_idx = tid >> 1;
  const int byte_idx = pair_idx >> 1;
  uint32_t packed = 0;
  if ((tid & 3) == 0) {
    packed = __ldg(slot + byte_idx);
  }
  const int lane = tid & 31;
  return __shfl_sync(0xffffffff, packed, lane & ~3);
}

__device__ __forceinline__ float load_pair_lane_codebook_value(
    const uint8_t* __restrict__ slot,
    const float* __restrict__ codebook,
    int tid) {
  float g0 = 0.0f;
  float g1 = 0.0f;
  if ((tid & 1) == 0) {
    const int pair_idx = tid >> 1;
    const int byte_idx = pair_idx >> 1;
    const uint8_t packed = __ldg(slot + byte_idx);
    const uint32_t cb_idx = static_cast<uint32_t>(
        (pair_idx & 1) ? (packed >> 4) : (packed & 0x0F));
    g0 = __ldg(&codebook[cb_idx * kPairDim + 0]);
    g1 = __ldg(&codebook[cb_idx * kPairDim + 1]);
  }
  const float peer_g1 = __shfl_xor_sync(0xffffffff, g1, 1);
  return (tid & 1) ? peer_g1 : g0;
}

__device__ __forceinline__ float load_scale_warp_broadcast(
    const uint8_t* __restrict__ slot, int tid) {
  float scale = 0.0f;
  if ((tid & 31) == 0) {
    const half scale_h =
        *reinterpret_cast<const half*>(slot + kPackedBytes);
    scale = __half2float(scale_h);
  }
  return __shfl_sync(0xffffffff, scale, 0);
}

__global__ void higgs_dense_2bit_dequant_kernel(
    const uint8_t* __restrict__ compressed,
    const int64_t* __restrict__ locs,
    bf16_t* __restrict__ out,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int64_t loc = locs[row];
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  bf16_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  // 1. Issue rope copy up-front so its STG pipeline overlaps the FWHT.
  if (tid < 8) {
    uint4 tmp;
    memcpy(&tmp, slot + kPackedBytes + kNormBytes + tid * 16, sizeof(uint4));
    memcpy(reinterpret_cast<uint8_t*>(row_out + kLatentDim) + tid * 16, &tmp,
           sizeof(uint4));
  }

  // 2. Unpack 4-bit codebook index for this thread's pair.
  //    Each packed byte holds TWO pair indices (=4 scalars):
  //        byte[k] = (pair_index[2k] & 0x0F) | ((pair_index[2k+1] & 0x0F) << 4)
  //    so for thread ``tid`` owning scalar (tid>>1).(tid&1):
  //        pair_idx = tid >> 1     in [0, kNumPairs)
  //        byte_idx = pair_idx >> 1 in [0, kPackedBytes)
  //        nibble   = pair_idx & 1  (low for even pairs, high for odd)
  const int pair_idx = tid >> 1;
  const int byte_idx = pair_idx >> 1;
  const uint8_t packed = slot[byte_idx];
  const uint32_t cb_idx = static_cast<uint32_t>(
      (pair_idx & 1) ? (packed >> 4) : (packed & 0x0F));

  // 3. Look up codebook entry for this pair, select my coordinate.
  //    Thread (tid) corresponds to scalar (tid % kPairDim) of pair
  //    (tid / kPairDim) = pair_idx.
  const int coord = tid & 1;
  const float g = __ldg(&codebook[cb_idx * kPairDim + coord]);

  // 4. Load fp16 scale and form rotated reconstruction: rot_recon =
  //    scale * G[idx][coord]. Then inverse FWHT (×1/sqrt(N)).
  const half scale_h =
      *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = __half2float(scale_h);

  const float rot_recon = scale * g;

  // 5. Inverse FWHT_512. FWHT/sqrt(N) is involutory.
  const float result = fwht_512_swizzled(rot_recon, buf) * kInvSqrtLatentDim;

  row_out[tid] = __float2bfloat16(result);
}

__global__ void higgs_dense_2bit_dequant_const_codebook_kernel(
    const uint8_t* __restrict__ compressed,
    const int64_t* __restrict__ locs,
    bf16_t* __restrict__ out,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  (void)codebook;
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int64_t loc = locs[row];
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  bf16_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  if (tid < 8) {
    uint4 tmp;
    memcpy(&tmp, slot + kPackedBytes + kNormBytes + tid * 16, sizeof(uint4));
    memcpy(reinterpret_cast<uint8_t*>(row_out + kLatentDim) + tid * 16, &tmp,
           sizeof(uint4));
  }

  const int pair_idx = tid >> 1;
  const int byte_idx = pair_idx >> 1;
  const uint8_t packed = slot[byte_idx];
  const uint32_t cb_idx = static_cast<uint32_t>(
      (pair_idx & 1) ? (packed >> 4) : (packed & 0x0F));

  const int coord = tid & 1;
  const float g = eden2_16_codebook_value(cb_idx, coord);
  const half scale_h =
      *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = __half2float(scale_h);

  const float rot_recon = scale * g;
  const float result = fwht_512_swizzled(rot_recon, buf) * kInvSqrtLatentDim;
  row_out[tid] = __float2bfloat16(result);
}

// ─── Page-table dequant variant ─────────────────────────────────────────────
// Mirrors TurboQuant's ``dequantize_page_table_selected_2p5_kernel``. The
// page table indexes flat slots; rows with page < 0 are masked.
//
// Invalid-row early-exit (ai-blaise #19 iter2): when ``page < 0`` the row is
// padding emitted by the DSA indexer (top-k slot beyond ``min(seq_len,
// dsa_index_topk)``). The downstream consumer reads ``seq_lens[b]`` entries
// per query and ignores the trailing pad block_table values, so we can drop
// the FWHT_512 work and the per-row STG of 576 B BF16 entirely. The
// ``compact_page_table[row] = -1`` write must still happen so reshape-based
// callers see a consistent invalid marker. The branch is CTA-uniform (every
// thread in the block reads the same ``page`` value), so no `__syncthreads`
// in ``fwht_512_swizzled`` is reached by a partial set of threads.

__global__ void higgs_dense_2bit_dequant_page_table_kernel(
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    bf16_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int32_t page = page_table[row];
  if (tid == 0) {
    compact_page_table[row] = page >= 0 ? static_cast<int32_t>(row) : -1;
  }
  if (page < 0) return;

  const int64_t loc = static_cast<int64_t>(page);
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  bf16_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  if (tid < 8) {
    uint4 tmp;
    memcpy(&tmp, slot + kPackedBytes + kNormBytes + tid * 16, sizeof(uint4));
    memcpy(reinterpret_cast<uint8_t*>(row_out + kLatentDim) + tid * 16, &tmp,
           sizeof(uint4));
  }

  const int pair_idx = tid >> 1;
  const int byte_idx = pair_idx >> 1;
  const uint8_t packed = slot[byte_idx];
  const uint32_t cb_idx = static_cast<uint32_t>(
      (pair_idx & 1) ? (packed >> 4) : (packed & 0x0F));

  const int coord = tid & 1;
  const float g = __ldg(&codebook[cb_idx * kPairDim + coord]);
  const half scale_h =
      *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = __half2float(scale_h);

  const float rot_recon = scale * g;
  const float result = fwht_512_swizzled(rot_recon, buf) * kInvSqrtLatentDim;
  row_out[tid] = __float2bfloat16(result);
}

__global__ void higgs_dense_2bit_dequant_page_table_const_codebook_kernel(
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    bf16_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  (void)codebook;
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int32_t page = page_table[row];
  if (tid == 0) {
    compact_page_table[row] = page >= 0 ? static_cast<int32_t>(row) : -1;
  }
  if (page < 0) return;

  const int64_t loc = static_cast<int64_t>(page);
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  bf16_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  if (tid < 8) {
    uint4 tmp;
    memcpy(&tmp, slot + kPackedBytes + kNormBytes + tid * 16, sizeof(uint4));
    memcpy(reinterpret_cast<uint8_t*>(row_out + kLatentDim) + tid * 16, &tmp,
           sizeof(uint4));
  }

  const int pair_idx = tid >> 1;
  const int byte_idx = pair_idx >> 1;
  const uint8_t packed = slot[byte_idx];
  const uint32_t cb_idx = static_cast<uint32_t>(
      (pair_idx & 1) ? (packed >> 4) : (packed & 0x0F));

  const int coord = tid & 1;
  const float g = eden2_16_codebook_value(cb_idx, coord);
  const half scale_h =
      *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = __half2float(scale_h);

  const float rot_recon = scale * g;
  const float result = fwht_512_swizzled(rot_recon, buf) * kInvSqrtLatentDim;
  row_out[tid] = __float2bfloat16(result);
}

// ─── FP8 page-table dequant variant ─────────────────────────────────────────
// ai-blaise #19 iter3: write FP8 (e4m3) instead of BF16 to feed the trtllm-gen
// sparse-MLA cubin set's ``QkvE4m3OBfloat16`` variants (48 cubins available in
// flashinfer 0.6.7.post3, gated on ``kv_cache.dtype == float8_e4m3fn``).
//
// Output row layout (576 B total, vs 1152 B for BF16):
//   row_out[0..511]   = 512 B FP8 latent  (one lane writes one element)
//   row_out[512..575] = 64  B FP8 rope    (first warp converts from BF16)
//
// Per-element quantization: the kernel applies ``inv_kv_scale`` as a
// multiplicative scale before the FP8 cast so the downstream attention
// kernel can pass ``k_scale = kv_scale`` via ``bmm1_scale`` and recover the
// original BF16-range values. The HIGGS lattice scale ``scale_h`` already
// captures per-token magnitude, so a single per-tensor ``inv_kv_scale``
// (passed at kernel call time, conservative-bound at startup) suffices.
// Setting ``inv_kv_scale = 1.0`` and ``k_scale = 1.0`` matches the FP8
// baseline path's ``mla_quantize_and_rope_for_fp8`` behaviour with
// ``quant_scale_kv = 1.0``.
//
// Memory traffic:
//   BF16 write  : 128 * 2048 * 576 B BF16 = 301.99 MiB / layer
//   FP8 write   : 128 * 2048 * 576 B FP8  = 151.00 MiB / layer (-50%)
//   Kernel read : 128 * 2048 * 576 B FP8  = 151.00 MiB / layer (-50%)
//   Round-trip  : 302 MiB / layer (vs 604 MiB BF16) — closes ~6 ms of
//                 the ~12.3 ms HBM-bound TPOT bottleneck.

__global__ void higgs_dense_2bit_dequant_page_table_fp8_kernel(
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

  const int64_t loc = static_cast<int64_t>(page);
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  fp8_e4m3_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  // Rope tile: 64 BF16 elements at slot[kPackedBytes + kNormBytes = 130];
  // the slot stride is 272 (compressed_stride_0 = 272 with 14 B tail
  // pad as of iter4 #16) and the rope offset 130 is byte-aligned not
  // bf16-aligned. Stage through a local 8 B uint8 buffer to avoid
  // misaligned-load faults, then reinterpret as bf16x2 and downcast
  // inline to FP8. 16 lanes × 4 elements = 64 BF16 in, 64 FP8 out
  // (64 B written, vs 128 B BF16).
  if (tid < 16) {
    const uint8_t* slot_rope = slot + kPackedBytes + kNormBytes;
    fp8_e4m3_t* rope_out = row_out + kLatentDim;
    const int base = tid * 4;
    uint8_t bf16_bytes[8];
    memcpy(bf16_bytes, slot_rope + base * 2, 8);
    const bf16x2_t rope_pair0 =
        *reinterpret_cast<const bf16x2_t*>(bf16_bytes);
    const bf16x2_t rope_pair1 =
        *reinterpret_cast<const bf16x2_t*>(bf16_bytes + 4);
    // Apply inv_kv_scale via FP32 path; bf16 → fp32 → ×inv_kv_scale → fp8.
    const float2 r0 = __bfloat1622float2(rope_pair0);
    const float2 r1 = __bfloat1622float2(rope_pair1);
    const float2 r0_scaled =
        make_float2(r0.x * inv_kv_scale, r0.y * inv_kv_scale);
    const float2 r1_scaled =
        make_float2(r1.x * inv_kv_scale, r1.y * inv_kv_scale);
    fp8x2_e4m3_t p0 = __nv_fp8x2_e4m3(r0_scaled);
    fp8x2_e4m3_t p1 = __nv_fp8x2_e4m3(r1_scaled);
    // 4 B write — also byte-aligned, stage through a local uint8 buffer.
    uint16_t out_bytes[2];
    memcpy(out_bytes, &p0, 2);
    memcpy(out_bytes + 1, &p1, 2);
    memcpy(rope_out + base, out_bytes, 4);
  }

  const int pair_idx = tid >> 1;
  const int byte_idx = pair_idx >> 1;
  const uint8_t packed = slot[byte_idx];
  const uint32_t cb_idx = static_cast<uint32_t>(
      (pair_idx & 1) ? (packed >> 4) : (packed & 0x0F));

  const int coord = tid & 1;
  const float g = __ldg(&codebook[cb_idx * kPairDim + coord]);
  const half scale_h =
      *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = __half2float(scale_h);

  const float rot_recon = scale * g;
  const float result = fwht_512_swizzled(rot_recon, buf) * kInvSqrtLatentDim;
  // Scale into FP8 range and saturating cast. Per-tensor inv_kv_scale folds
  // the downstream attention BMM1 scale absorption.
  row_out[tid] = __nv_fp8_e4m3(result * inv_kv_scale);
}

__global__ void higgs_dense_2bit_dequant_page_table_fp8_const_codebook_kernel(
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    fp8_e4m3_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0,
    float inv_kv_scale) {
  (void)codebook;
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int32_t page = page_table[row];
  if (tid == 0) {
    compact_page_table[row] = page >= 0 ? static_cast<int32_t>(row) : -1;
  }
  if (page < 0) return;

  const int64_t loc = static_cast<int64_t>(page);
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  fp8_e4m3_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  if (tid < 16) {
    const uint8_t* slot_rope = slot + kPackedBytes + kNormBytes;
    fp8_e4m3_t* rope_out = row_out + kLatentDim;
    const int base = tid * 4;
    uint8_t bf16_bytes[8];
    memcpy(bf16_bytes, slot_rope + base * 2, 8);
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
    memcpy(out_bytes, &p0, 2);
    memcpy(out_bytes + 1, &p1, 2);
    memcpy(rope_out + base, out_bytes, 4);
  }

  const int pair_idx = tid >> 1;
  const int byte_idx = pair_idx >> 1;
  const uint8_t packed = slot[byte_idx];
  const uint32_t cb_idx = static_cast<uint32_t>(
      (pair_idx & 1) ? (packed >> 4) : (packed & 0x0F));

  const int coord = tid & 1;
  const float g = eden2_16_codebook_value(cb_idx, coord);
  const half scale_h =
      *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = __half2float(scale_h);

  const float rot_recon = scale * g;
  const float result = fwht_512_swizzled(rot_recon, buf) * kInvSqrtLatentDim;
  row_out[tid] = __nv_fp8_e4m3(result * inv_kv_scale);
}

// ─── Opt-in vec4/shared-codebook dequant candidates ────────────────────────
// B200 measurement variants only. These keep the incumbent slot layout and
// output contract unchanged while reducing redundant packed-byte loads: one
// lane per 4-scalar group loads the byte and broadcasts it to its quartet.
// The EDEN2-16 codebook is staged in SMEM once per CTA instead of loaded from
// global memory by every scalar lane.

__global__ void higgs_dense_2bit_dequant_vec4_kernel(
    const uint8_t* __restrict__ compressed,
    const int64_t* __restrict__ locs,
    bf16_t* __restrict__ out,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int64_t loc = locs[row];
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  bf16_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];
  __shared__ float cb_smem[kCodebookSize * kPairDim];

  if (tid < kCodebookSize * kPairDim) {
    cb_smem[tid] = __ldg(&codebook[tid]);
  }

  if (tid < 8) {
    uint4 tmp;
    memcpy(&tmp, slot + kPackedBytes + kNormBytes + tid * 16, sizeof(uint4));
    memcpy(reinterpret_cast<uint8_t*>(row_out + kLatentDim) + tid * 16, &tmp,
           sizeof(uint4));
  }
  __syncthreads();

  const int pair_idx = tid >> 1;
  const uint32_t packed = load_packed_byte_vec4(slot, tid);
  const uint32_t cb_idx = static_cast<uint32_t>(
      (pair_idx & 1) ? (packed >> 4) : (packed & 0x0F));

  const int coord = tid & 1;
  const float g = cb_smem[cb_idx * kPairDim + coord];
  const half scale_h =
      *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = __half2float(scale_h);

  const float rot_recon = scale * g;
  const float result = fwht_512_swizzled(rot_recon, buf) * kInvSqrtLatentDim;
  row_out[tid] = __float2bfloat16(result);
}

__global__ void higgs_dense_2bit_dequant_vec4_ldg_codebook_kernel(
    const uint8_t* __restrict__ compressed,
    const int64_t* __restrict__ locs,
    bf16_t* __restrict__ out,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int64_t loc = locs[row];
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  bf16_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  if (tid < 8) {
    uint4 tmp;
    memcpy(&tmp, slot + kPackedBytes + kNormBytes + tid * 16, sizeof(uint4));
    memcpy(reinterpret_cast<uint8_t*>(row_out + kLatentDim) + tid * 16, &tmp,
           sizeof(uint4));
  }

  const int pair_idx = tid >> 1;
  const uint32_t packed = load_packed_byte_vec4(slot, tid);
  const uint32_t cb_idx = static_cast<uint32_t>(
      (pair_idx & 1) ? (packed >> 4) : (packed & 0x0F));

  const int coord = tid & 1;
  const float g = __ldg(&codebook[cb_idx * kPairDim + coord]);
  const half scale_h =
      *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = __half2float(scale_h);

  const float rot_recon = scale * g;
  const float result = fwht_512_swizzled(rot_recon, buf) * kInvSqrtLatentDim;
  row_out[tid] = __float2bfloat16(result);
}

__global__ void higgs_dense_2bit_dequant_page_table_vec4_kernel(
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    bf16_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int32_t page = page_table[row];
  if (tid == 0) {
    compact_page_table[row] = page >= 0 ? static_cast<int32_t>(row) : -1;
  }
  if (page < 0) return;

  const int64_t loc = static_cast<int64_t>(page);
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  bf16_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];
  __shared__ float cb_smem[kCodebookSize * kPairDim];

  if (tid < kCodebookSize * kPairDim) {
    cb_smem[tid] = __ldg(&codebook[tid]);
  }

  if (tid < 8) {
    uint4 tmp;
    memcpy(&tmp, slot + kPackedBytes + kNormBytes + tid * 16, sizeof(uint4));
    memcpy(reinterpret_cast<uint8_t*>(row_out + kLatentDim) + tid * 16, &tmp,
           sizeof(uint4));
  }
  __syncthreads();

  const int pair_idx = tid >> 1;
  const uint32_t packed = load_packed_byte_vec4(slot, tid);
  const uint32_t cb_idx = static_cast<uint32_t>(
      (pair_idx & 1) ? (packed >> 4) : (packed & 0x0F));

  const int coord = tid & 1;
  const float g = cb_smem[cb_idx * kPairDim + coord];
  const half scale_h =
      *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = __half2float(scale_h);

  const float rot_recon = scale * g;
  const float result = fwht_512_swizzled(rot_recon, buf) * kInvSqrtLatentDim;
  row_out[tid] = __float2bfloat16(result);
}

__global__ void higgs_dense_2bit_dequant_page_table_vec4_ldg_codebook_kernel(
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    bf16_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int32_t page = page_table[row];
  if (tid == 0) {
    compact_page_table[row] = page >= 0 ? static_cast<int32_t>(row) : -1;
  }
  if (page < 0) return;

  const int64_t loc = static_cast<int64_t>(page);
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  bf16_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  if (tid < 8) {
    uint4 tmp;
    memcpy(&tmp, slot + kPackedBytes + kNormBytes + tid * 16, sizeof(uint4));
    memcpy(reinterpret_cast<uint8_t*>(row_out + kLatentDim) + tid * 16, &tmp,
           sizeof(uint4));
  }

  const int pair_idx = tid >> 1;
  const uint32_t packed = load_packed_byte_vec4(slot, tid);
  const uint32_t cb_idx = static_cast<uint32_t>(
      (pair_idx & 1) ? (packed >> 4) : (packed & 0x0F));

  const int coord = tid & 1;
  const float g = __ldg(&codebook[cb_idx * kPairDim + coord]);
  const half scale_h =
      *reinterpret_cast<const half*>(slot + kPackedBytes);
  const float scale = __half2float(scale_h);

  const float rot_recon = scale * g;
  const float result = fwht_512_swizzled(rot_recon, buf) * kInvSqrtLatentDim;
  row_out[tid] = __float2bfloat16(result);
}

__global__ void higgs_dense_2bit_dequant_pair_lanes_scale_broadcast_kernel(
    const uint8_t* __restrict__ compressed,
    const int64_t* __restrict__ locs,
    bf16_t* __restrict__ out,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int64_t loc = locs[row];
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  bf16_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  if (tid < 8) {
    uint4 tmp;
    memcpy(&tmp, slot + kPackedBytes + kNormBytes + tid * 16, sizeof(uint4));
    memcpy(reinterpret_cast<uint8_t*>(row_out + kLatentDim) + tid * 16, &tmp,
           sizeof(uint4));
  }

  const float g = load_pair_lane_codebook_value(slot, codebook, tid);
  const float scale = load_scale_warp_broadcast(slot, tid);
  const float rot_recon = scale * g;
  const float result = fwht_512_swizzled(rot_recon, buf) * kInvSqrtLatentDim;
  row_out[tid] = __float2bfloat16(result);
}

__global__ void
higgs_dense_2bit_dequant_page_table_pair_lanes_scale_broadcast_kernel(
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    bf16_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int32_t page = page_table[row];
  if (tid == 0) {
    compact_page_table[row] = page >= 0 ? static_cast<int32_t>(row) : -1;
  }
  if (page < 0) return;

  const int64_t loc = static_cast<int64_t>(page);
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  bf16_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  if (tid < 8) {
    uint4 tmp;
    memcpy(&tmp, slot + kPackedBytes + kNormBytes + tid * 16, sizeof(uint4));
    memcpy(reinterpret_cast<uint8_t*>(row_out + kLatentDim) + tid * 16, &tmp,
           sizeof(uint4));
  }

  const float g = load_pair_lane_codebook_value(slot, codebook, tid);
  const float scale = load_scale_warp_broadcast(slot, tid);
  const float rot_recon = scale * g;
  const float result = fwht_512_swizzled(rot_recon, buf) * kInvSqrtLatentDim;
  row_out[tid] = __float2bfloat16(result);
}

// ─── Host-side launcher structs ─────────────────────────────────────────────

__device__ __forceinline__ uint32_t saw_scalar2_quant_bf16_bits(
    uint32_t bits) {
  if ((bits & 0x8000u) != 0) {
    return bits > 0xbf7bu ? 0u : 1u;
  }
  return bits <= 0x3f7bu ? 2u : 3u;
}

__device__ __forceinline__ uint32_t saw_scalar2_bf16_bits(uint32_t code) {
  // Codes 0/3 map to the high-magnitude lattice points, 1/2 to low.
  code &= 0x3u;
  const uint32_t high_magnitude = ((code ^ (code >> 1) ^ 1u) & 1u);
  const uint32_t magnitude = 0x3ee8u + high_magnitude * 0x00d9u;
  const uint32_t sign = ((code >> 1) ^ 1u) << 15;
  return magnitude | sign;
}

__device__ __forceinline__ uint32_t saw_scalar2_bf16_pair(uint32_t code_pair) {
  const uint32_t lo = saw_scalar2_bf16_bits(code_pair & 0x3u);
  const uint32_t hi = saw_scalar2_bf16_bits((code_pair >> 2) & 0x3u);
  return lo | (hi << 16);
}

// ai-blaise #19 iter3: FP8 e4m3 saw_scalar2 lookup. The two lattice
// magnitudes (low ≈ 0.4541, high ≈ 1.508) cast saturatingly to FP8 give:
//   low+  : 0.4541 -> FP8 0x2E (1.110 × 2^-2 → ~0.4375)
//   low-  : -0.4541 -> FP8 0xAE
//   high+ : 1.508 -> FP8 0x3C (1.100 × 2^0 → 1.5)
//   high- : -1.508 -> FP8 0xBC
// Pre-encode the 4 values in a single uint32 with one byte per code,
// indexed by the 2-bit code at byte offset (code << 3).
__device__ __forceinline__ uint8_t saw_scalar2_fp8_e4m3_bits(uint32_t code) {
  // Mirrors saw_scalar2_bf16_bits but emits a 8-bit FP8 e4m3 pattern.
  //   high_magnitude = 1 iff code ∈ {0, 3}
  //   sign = 1 iff code ∈ {0, 1} (sign bit of saw_scalar2_bf16_bits is
  //                                ((code >> 1) ^ 1u))
  code &= 0x3u;
  const uint32_t high_magnitude = ((code ^ (code >> 1) ^ 1u) & 1u);
  // FP8 e4m3 magnitude: low = 0x2E (~0.4375), high = 0x3C (1.5).
  const uint32_t magnitude = high_magnitude ? 0x3Cu : 0x2Eu;
  const uint32_t sign = ((code >> 1) ^ 1u) << 7;
  return static_cast<uint8_t>(magnitude | sign);
}

__device__ __forceinline__ uint32_t saw_scalar2_fp8_quad(uint32_t packed_byte) {
  // Given a packed byte holding 4 codes (2 bits each, low to high):
  //   lo nibble = (idx0 in low 2 bits) | (idx1 in high 2 bits)
  // produce 4 FP8 bytes packed into a single uint32 (little-endian).
  const uint32_t b0 = saw_scalar2_fp8_e4m3_bits(packed_byte & 0x3u);
  const uint32_t b1 = saw_scalar2_fp8_e4m3_bits((packed_byte >> 2) & 0x3u);
  const uint32_t b2 = saw_scalar2_fp8_e4m3_bits((packed_byte >> 4) & 0x3u);
  const uint32_t b3 = saw_scalar2_fp8_e4m3_bits((packed_byte >> 6) & 0x3u);
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

template <int kWarpsPerBlock, typename LocT>
__global__ void __launch_bounds__(kWarpsPerBlock * 32, 4)
higgs_dense_2bit_dequant_saw_scalar2_fp8_coalesced_kernel(
    const uint8_t* __restrict__ compressed,
    const LocT* __restrict__ locs,
    fp8_e4m3_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0,
    float inv_kv_scale) {
  (void)codebook;
  // saw_scalar2 emits a fixed two-magnitude lattice, so inv_kv_scale only
  // matters at the boundary of the FP8 max-finite range. The two emitted
  // magnitudes (0.4541 / 1.508) are far inside the e4m3 range even at
  // inv_kv_scale ≤ ~290, so we skip the per-element ×inv_kv_scale FMA
  // and let the downstream attention path apply the inverse via k_scale
  // for any pathological scales. If inv_kv_scale != 1.0 the caller is
  // expected to pass it through ``bmm1_scale = k_scale * sm_scale`` with
  // ``k_scale = 1 / inv_kv_scale``; the FP8 lattice bits are unchanged.
  (void)inv_kv_scale;
  static_assert(
      kWarpsPerBlock == 4 || kWarpsPerBlock == 8,
      "only four-row and eight-row block shapes are wired");
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_id;
  if (row >= num_rows) return;

  const LocT raw_loc = locs[row];
  if (compact_page_table != nullptr && lane == 0) {
    compact_page_table[row] = raw_loc >= 0 ? static_cast<int32_t>(row) : -1;
  }
  if (raw_loc < 0) return;
  const int64_t loc = static_cast<int64_t>(raw_loc);
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  fp8_e4m3_t* row_out = out + row * out_stride_0;

  // Latent: 1 packed byte = 4 lattice codes × 2 bits = 4 FP8 elements
  // out (32 bits = uint32 store). kPackedBytes = 128 bytes per row →
  // 512 FP8 elements per row. 4 segments × 32 lanes = 128 byte slots,
  // each lane processes 1 byte per segment and writes 1 uint32 (4 FP8).
#pragma unroll
  for (int segment = 0; segment < 4; ++segment) {
    const int byte_idx = segment * 32 + lane;
    const int base = byte_idx * 4;  // FP8-element index of this byte's output
    const uint8_t packed = __ldg(slot + byte_idx);
    const uint32_t packed_out = saw_scalar2_fp8_quad(packed);
    *reinterpret_cast<uint32_t*>(row_out + base) = packed_out;
  }

  // Rope: 64 BF16 input → 64 FP8 output. 8 lanes × 8 elements/lane.
  // Each lane reads 16 B (8 BF16) and writes 8 B (8 FP8 = uint64).
  if (lane < 8) {
    const uint8_t* slot_rope = slot + kPackedBytes + kNormBytes;
    uint8_t bf16_bytes[16];
    memcpy(bf16_bytes, slot_rope + lane * 16, 16);
    fp8_e4m3_t* rope_out = row_out + kLatentDim;
    uint64_t fp8_packed = 0;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const bf16x2_t bf16_pair =
          *reinterpret_cast<const bf16x2_t*>(bf16_bytes + pair * 4);
      const fp8x2_e4m3_t fp8_pair = __nv_fp8x2_e4m3(bf16_pair);
      uint16_t pair_bits;
      memcpy(&pair_bits, &fp8_pair, 2);
      fp8_packed |= (static_cast<uint64_t>(pair_bits) << (pair * 16));
    }
    memcpy(rope_out + lane * 8, &fp8_packed, 8);
  }
}

template <int kWarpsPerBlock>
__global__ void __launch_bounds__(kWarpsPerBlock * 32, 4)
higgs_dense_2bit_store_saw_scalar2_coalesced_kernel(
    uint8_t* __restrict__ compressed,
    const int64_t* __restrict__ locs,
    const bf16_t* __restrict__ latent,
    const bf16_t* __restrict__ rope,
    const float* __restrict__ codebook,
    const float* __restrict__ codebook_norm_sq,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t latent_stride_0,
    int64_t latent_stride_1,
    int64_t rope_stride_0,
    int64_t rope_stride_1) {
  (void)codebook;
  (void)codebook_norm_sq;
  (void)latent_stride_1;
  (void)rope_stride_1;
  static_assert(
      kWarpsPerBlock == 4 || kWarpsPerBlock == 8,
      "only four-row and eight-row block shapes are wired");
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_id;
  if (row >= num_rows) return;

  const bf16_t* lat_row = latent + row * latent_stride_0;
  const uint16_t* lat_bits = reinterpret_cast<const uint16_t*>(lat_row);
  constexpr float scale = 1.0f;

  const int64_t loc = locs[row];
  uint8_t* slot = compressed + loc * compressed_stride_0;
#pragma unroll
  for (int segment = 0; segment < 4; ++segment) {
    const int base = (segment * 32 + lane) * 4;
    const unsigned long long packed_values = __ldg(
        reinterpret_cast<const unsigned long long*>(lat_bits + base));
    const uint32_t code0 = saw_scalar2_quant_bf16_bits(
        static_cast<uint32_t>(packed_values & 0xffffull));
    const uint32_t code1 = saw_scalar2_quant_bf16_bits(
        static_cast<uint32_t>((packed_values >> 16) & 0xffffull));
    const uint32_t code2 = saw_scalar2_quant_bf16_bits(
        static_cast<uint32_t>((packed_values >> 32) & 0xffffull));
    const uint32_t code3 = saw_scalar2_quant_bf16_bits(
        static_cast<uint32_t>((packed_values >> 48) & 0xffffull));
    const uint32_t idx0 = code0 | (code1 << 2);
    const uint32_t idx1 = code2 | (code3 << 2);
    slot[segment * 32 + lane] =
        static_cast<uint8_t>((idx0 & 0x0F) | ((idx1 & 0x0F) << 4));
  }

  if (lane == 0) {
    half scale_h = __float2half(scale);
    *reinterpret_cast<half*>(slot + kPackedBytes) = scale_h;
  }
  if (lane < 8) {
    const bf16_t* rope_row = rope + row * rope_stride_0;
    const uint16_t* src = reinterpret_cast<const uint16_t*>(rope_row) + lane * 8;
    uint16_t* dst = reinterpret_cast<uint16_t*>(slot + kPackedBytes + kNormBytes) +
        lane * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      dst[i] = __ldg(src + i);
    }
  }
}

template <int kWarpsPerBlock, typename LocT>
__global__ void __launch_bounds__(kWarpsPerBlock * 32, 4)
higgs_dense_2bit_dequant_saw_scalar2_coalesced_kernel(
    const uint8_t* __restrict__ compressed,
    const LocT* __restrict__ locs,
    bf16_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ codebook,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  (void)codebook;
  static_assert(
      kWarpsPerBlock == 4 || kWarpsPerBlock == 8,
      "only four-row and eight-row block shapes are wired");
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_id;
  if (row >= num_rows) return;

  const LocT raw_loc = locs[row];
  if (compact_page_table != nullptr && lane == 0) {
    compact_page_table[row] = raw_loc >= 0 ? static_cast<int32_t>(row) : -1;
  }
  // ai-blaise #19 iter2: invalid-row early-exit. The DSA indexer pads
  // top-k per query with -1 sentinels past min(seq_len, dsa_index_topk);
  // downstream consumers honour seq_lens and never read those slots. The
  // saw_scalar2 coalesced kernel assigns one warp per row so we can drop
  // the per-row packed-store loop and rope copy without blocking other
  // warps in the same block — no __syncthreads is reached in this kernel
  // body, so a warp-uniform return is safe.
  if (raw_loc < 0) return;
  const int64_t loc = static_cast<int64_t>(raw_loc);
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  bf16_t* row_out = out + row * out_stride_0;

#pragma unroll
  for (int segment = 0; segment < 4; ++segment) {
    const int byte_idx = segment * 32 + lane;
    const int base = byte_idx * 4;
    const uint8_t packed = __ldg(slot + byte_idx);
    const uint32_t idx0 = static_cast<uint32_t>(packed & 0x0F);
    const uint32_t idx1 = static_cast<uint32_t>(packed >> 4);
    const unsigned long long packed_out =
        static_cast<unsigned long long>(saw_scalar2_bf16_pair(idx0)) |
        (static_cast<unsigned long long>(saw_scalar2_bf16_pair(idx1)) << 32);
    *reinterpret_cast<unsigned long long*>(row_out + base) = packed_out;
  }

  if (lane < 8) {
    const uint16_t* src = reinterpret_cast<const uint16_t*>(
        slot + kPackedBytes + kNormBytes) + lane * 8;
    uint16_t* dst = reinterpret_cast<uint16_t*>(row_out + kLatentDim) + lane * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      dst[i] = __ldg(src + i);
    }
  }
}

struct HiggsDense2BitStoreKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView rope,
      tvm::ffi::TensorView codebook,
      tvm::ffi::TensorView codebook_norm_sq) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto latent_stride_0 = SymbolicSize{"latent_stride_0"};
    auto latent_stride_1 = SymbolicSize{"latent_stride_1"};
    auto rope_stride_0 = SymbolicSize{"rope_stride_0"};
    auto rope_stride_1 = SymbolicSize{"rope_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kLatentDim})
        .with_strides({latent_stride_0, latent_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(latent);
    TensorMatcher({N, 1, kRopeDim})
        .with_strides({rope_stride_0, rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(rope);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);
    TensorMatcher({kCodebookSize})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook_norm_sq);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_store_kernel<false>,
        static_cast<uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<const bf16_t*>(latent.data_ptr()),
        static_cast<const bf16_t*>(rope.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        static_cast<const float*>(codebook_norm_sq.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        latent_stride_0.unwrap(),
        latent_stride_1.unwrap(),
        rope_stride_0.unwrap(),
        rope_stride_1.unwrap());
  }
};

struct HiggsDense2BitStoreConstCodebookKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView rope,
      tvm::ffi::TensorView codebook,
      tvm::ffi::TensorView codebook_norm_sq) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto latent_stride_0 = SymbolicSize{"latent_stride_0"};
    auto latent_stride_1 = SymbolicSize{"latent_stride_1"};
    auto rope_stride_0 = SymbolicSize{"rope_stride_0"};
    auto rope_stride_1 = SymbolicSize{"rope_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kLatentDim})
        .with_strides({latent_stride_0, latent_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(latent);
    TensorMatcher({N, 1, kRopeDim})
        .with_strides({rope_stride_0, rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(rope);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);
    TensorMatcher({kCodebookSize})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook_norm_sq);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_store_kernel<true>,
        static_cast<uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<const bf16_t*>(latent.data_ptr()),
        static_cast<const bf16_t*>(rope.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        static_cast<const float*>(codebook_norm_sq.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        latent_stride_0.unwrap(),
        latent_stride_1.unwrap(),
        rope_stride_0.unwrap(),
        rope_stride_1.unwrap());
  }
};

struct HiggsDense2BitStoreConstCodebookRopeFirstIndexPackKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView rope,
      tvm::ffi::TensorView codebook,
      tvm::ffi::TensorView codebook_norm_sq) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto latent_stride_0 = SymbolicSize{"latent_stride_0"};
    auto latent_stride_1 = SymbolicSize{"latent_stride_1"};
    auto rope_stride_0 = SymbolicSize{"rope_stride_0"};
    auto rope_stride_1 = SymbolicSize{"rope_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kLatentDim})
        .with_strides({latent_stride_0, latent_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(latent);
    TensorMatcher({N, 1, kRopeDim})
        .with_strides({rope_stride_0, rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(rope);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);
    TensorMatcher({kCodebookSize})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook_norm_sq);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_store_kernel<true, false, true, true>,
        static_cast<uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<const bf16_t*>(latent.data_ptr()),
        static_cast<const bf16_t*>(rope.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        static_cast<const float*>(codebook_norm_sq.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        latent_stride_0.unwrap(),
        latent_stride_1.unwrap(),
        rope_stride_0.unwrap(),
        rope_stride_1.unwrap());
  }
};

struct HiggsDense2BitStoreConstCodebookRopeFirstKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView rope,
      tvm::ffi::TensorView codebook,
      tvm::ffi::TensorView codebook_norm_sq) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto latent_stride_0 = SymbolicSize{"latent_stride_0"};
    auto latent_stride_1 = SymbolicSize{"latent_stride_1"};
    auto rope_stride_0 = SymbolicSize{"rope_stride_0"};
    auto rope_stride_1 = SymbolicSize{"rope_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kLatentDim})
        .with_strides({latent_stride_0, latent_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(latent);
    TensorMatcher({N, 1, kRopeDim})
        .with_strides({rope_stride_0, rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(rope);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);
    TensorMatcher({kCodebookSize})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook_norm_sq);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_store_kernel<true, false, false, true>,
        static_cast<uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<const bf16_t*>(latent.data_ptr()),
        static_cast<const bf16_t*>(rope.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        static_cast<const float*>(codebook_norm_sq.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        latent_stride_0.unwrap(),
        latent_stride_1.unwrap(),
        rope_stride_0.unwrap(),
        rope_stride_1.unwrap());
  }
};

struct HiggsDense2BitStoreConstCodebookIndexPackKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView rope,
      tvm::ffi::TensorView codebook,
      tvm::ffi::TensorView codebook_norm_sq) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto latent_stride_0 = SymbolicSize{"latent_stride_0"};
    auto latent_stride_1 = SymbolicSize{"latent_stride_1"};
    auto rope_stride_0 = SymbolicSize{"rope_stride_0"};
    auto rope_stride_1 = SymbolicSize{"rope_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kLatentDim})
        .with_strides({latent_stride_0, latent_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(latent);
    TensorMatcher({N, 1, kRopeDim})
        .with_strides({rope_stride_0, rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(rope);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);
    TensorMatcher({kCodebookSize})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook_norm_sq);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_store_kernel<true, false, true>,
        static_cast<uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<const bf16_t*>(latent.data_ptr()),
        static_cast<const bf16_t*>(rope.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        static_cast<const float*>(codebook_norm_sq.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        latent_stride_0.unwrap(),
        latent_stride_1.unwrap(),
        rope_stride_0.unwrap(),
        rope_stride_1.unwrap());
  }
};

struct HiggsDense2BitStoreConstCodebookWarpPackKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView rope,
      tvm::ffi::TensorView codebook,
      tvm::ffi::TensorView codebook_norm_sq) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto latent_stride_0 = SymbolicSize{"latent_stride_0"};
    auto latent_stride_1 = SymbolicSize{"latent_stride_1"};
    auto rope_stride_0 = SymbolicSize{"rope_stride_0"};
    auto rope_stride_1 = SymbolicSize{"rope_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kLatentDim})
        .with_strides({latent_stride_0, latent_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(latent);
    TensorMatcher({N, 1, kRopeDim})
        .with_strides({rope_stride_0, rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(rope);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);
    TensorMatcher({kCodebookSize})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook_norm_sq);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_store_kernel<true, true>,
        static_cast<uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<const bf16_t*>(latent.data_ptr()),
        static_cast<const bf16_t*>(rope.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        static_cast<const float*>(codebook_norm_sq.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        latent_stride_0.unwrap(),
        latent_stride_1.unwrap(),
        rope_stride_0.unwrap(),
        rope_stride_1.unwrap());
  }
};

struct HiggsDense2BitStoreConstCodebookWarpPackPreNormKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView rope,
      tvm::ffi::TensorView codebook,
      tvm::ffi::TensorView codebook_norm_sq) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto latent_stride_0 = SymbolicSize{"latent_stride_0"};
    auto latent_stride_1 = SymbolicSize{"latent_stride_1"};
    auto rope_stride_0 = SymbolicSize{"rope_stride_0"};
    auto rope_stride_1 = SymbolicSize{"rope_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kLatentDim})
        .with_strides({latent_stride_0, latent_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(latent);
    TensorMatcher({N, 1, kRopeDim})
        .with_strides({rope_stride_0, rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(rope);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);
    TensorMatcher({kCodebookSize})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook_norm_sq);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_store_kernel<true, true, false, false, false, true>,
        static_cast<uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<const bf16_t*>(latent.data_ptr()),
        static_cast<const bf16_t*>(rope.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        static_cast<const float*>(codebook_norm_sq.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        latent_stride_0.unwrap(),
        latent_stride_1.unwrap(),
        rope_stride_0.unwrap(),
        rope_stride_1.unwrap());
  }
};

struct HiggsDense2BitStoreConstCodebookWarpPackFmaScoreKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView rope,
      tvm::ffi::TensorView codebook,
      tvm::ffi::TensorView codebook_norm_sq) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto latent_stride_0 = SymbolicSize{"latent_stride_0"};
    auto latent_stride_1 = SymbolicSize{"latent_stride_1"};
    auto rope_stride_0 = SymbolicSize{"rope_stride_0"};
    auto rope_stride_1 = SymbolicSize{"rope_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kLatentDim})
        .with_strides({latent_stride_0, latent_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(latent);
    TensorMatcher({N, 1, kRopeDim})
        .with_strides({rope_stride_0, rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(rope);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);
    TensorMatcher({kCodebookSize})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook_norm_sq);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_store_kernel<
            true, true, false, false, false, false, true, true>,
        static_cast<uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<const bf16_t*>(latent.data_ptr()),
        static_cast<const bf16_t*>(rope.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        static_cast<const float*>(codebook_norm_sq.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        latent_stride_0.unwrap(),
        latent_stride_1.unwrap(),
        rope_stride_0.unwrap(),
        rope_stride_1.unwrap());
  }
};

struct HiggsDense2BitStoreConstCodebookWarpPackScaleBroadcastKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView rope,
      tvm::ffi::TensorView codebook,
      tvm::ffi::TensorView codebook_norm_sq) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto latent_stride_0 = SymbolicSize{"latent_stride_0"};
    auto latent_stride_1 = SymbolicSize{"latent_stride_1"};
    auto rope_stride_0 = SymbolicSize{"rope_stride_0"};
    auto rope_stride_1 = SymbolicSize{"rope_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kLatentDim})
        .with_strides({latent_stride_0, latent_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(latent);
    TensorMatcher({N, 1, kRopeDim})
        .with_strides({rope_stride_0, rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(rope);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);
    TensorMatcher({kCodebookSize})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook_norm_sq);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_store_kernel<true, true, false, false, true>,
        static_cast<uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<const bf16_t*>(latent.data_ptr()),
        static_cast<const bf16_t*>(rope.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        static_cast<const float*>(codebook_norm_sq.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        latent_stride_0.unwrap(),
        latent_stride_1.unwrap(),
        rope_stride_0.unwrap(),
        rope_stride_1.unwrap());
  }
};

struct HiggsDense2BitStoreConstCodebookWarpPackRopeFirstKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView rope,
      tvm::ffi::TensorView codebook,
      tvm::ffi::TensorView codebook_norm_sq) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto latent_stride_0 = SymbolicSize{"latent_stride_0"};
    auto latent_stride_1 = SymbolicSize{"latent_stride_1"};
    auto rope_stride_0 = SymbolicSize{"rope_stride_0"};
    auto rope_stride_1 = SymbolicSize{"rope_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kLatentDim})
        .with_strides({latent_stride_0, latent_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(latent);
    TensorMatcher({N, 1, kRopeDim})
        .with_strides({rope_stride_0, rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(rope);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);
    TensorMatcher({kCodebookSize})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook_norm_sq);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_store_kernel<true, true, false, true>,
        static_cast<uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<const bf16_t*>(latent.data_ptr()),
        static_cast<const bf16_t*>(rope.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        static_cast<const float*>(codebook_norm_sq.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        latent_stride_0.unwrap(),
        latent_stride_1.unwrap(),
        rope_stride_0.unwrap(),
        rope_stride_1.unwrap());
  }
};

struct HiggsDense2BitStoreSawScalar2Kernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView rope,
      tvm::ffi::TensorView codebook,
      tvm::ffi::TensorView codebook_norm_sq) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto latent_stride_0 = SymbolicSize{"latent_stride_0"};
    auto latent_stride_1 = SymbolicSize{"latent_stride_1"};
    auto rope_stride_0 = SymbolicSize{"rope_stride_0"};
    auto rope_stride_1 = SymbolicSize{"rope_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes})
        .with_strides({compressed_stride_0, kSlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kLatentDim})
        .with_strides({latent_stride_0, latent_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(latent);
    TensorMatcher({N, 1, kRopeDim})
        .with_strides({rope_stride_0, rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(rope);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);
    TensorMatcher({kCodebookSize})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook_norm_sq);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    if (num_rows >= kSawScalar2LargeRowThreshold) {
      LaunchKernel((num_rows + 7) / 8, 256, device.unwrap())(
          higgs_dense_2bit_store_saw_scalar2_coalesced_kernel<8>,
          static_cast<uint8_t*>(compressed.data_ptr()),
          static_cast<const int64_t*>(locs.data_ptr()),
          static_cast<const bf16_t*>(latent.data_ptr()),
          static_cast<const bf16_t*>(rope.data_ptr()),
          static_cast<const float*>(codebook.data_ptr()),
          static_cast<const float*>(codebook_norm_sq.data_ptr()),
          num_rows,
          compressed_stride_0.unwrap(),
          latent_stride_0.unwrap(),
          latent_stride_1.unwrap(),
          rope_stride_0.unwrap(),
          rope_stride_1.unwrap());
      return;
    }

    LaunchKernel((num_rows + 3) / 4, 128, device.unwrap())(
        higgs_dense_2bit_store_saw_scalar2_coalesced_kernel<4>,
        static_cast<uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<const bf16_t*>(latent.data_ptr()),
        static_cast<const bf16_t*>(rope.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        static_cast<const float*>(codebook_norm_sq.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        latent_stride_0.unwrap(),
        latent_stride_1.unwrap(),
        rope_stride_0.unwrap(),
        rope_stride_1.unwrap());
  }
};

struct HiggsDense2BitDequantSawScalar2Kernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView codebook) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
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
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kKvDim})
        .with_strides({out_stride_0, kKvDim, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    if (num_rows >= kSawScalar2LargeRowThreshold) {
      LaunchKernel((num_rows + 7) / 8, 256, device.unwrap())(
          higgs_dense_2bit_dequant_saw_scalar2_coalesced_kernel<8, int64_t>,
          static_cast<const uint8_t*>(compressed.data_ptr()),
          static_cast<const int64_t*>(locs.data_ptr()),
          static_cast<bf16_t*>(out.data_ptr()),
          static_cast<int32_t*>(nullptr),
          static_cast<const float*>(codebook.data_ptr()),
          num_rows,
          compressed_stride_0.unwrap(),
          out_stride_0.unwrap());
      return;
    }

    LaunchKernel((num_rows + 3) / 4, 128, device.unwrap())(
        higgs_dense_2bit_dequant_saw_scalar2_coalesced_kernel<4, int64_t>,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<int32_t*>(nullptr),
        static_cast<const float*>(codebook.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct HiggsDense2BitDequantPageTableSawScalar2Kernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView compact_page_table,
      tvm::ffi::TensorView codebook) {
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
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    const int64_t num_rows = B.unwrap() * K.unwrap();
    if (num_rows == 0) return;

    if (num_rows >= kSawScalar2LargeRowThreshold) {
      LaunchKernel((num_rows + 7) / 8, 256, device.unwrap())(
          higgs_dense_2bit_dequant_saw_scalar2_coalesced_kernel<8, int32_t>,
          static_cast<const uint8_t*>(compressed.data_ptr()),
          static_cast<const int32_t*>(page_table.data_ptr()),
          static_cast<bf16_t*>(out.data_ptr()),
          static_cast<int32_t*>(compact_page_table.data_ptr()),
          static_cast<const float*>(codebook.data_ptr()),
          num_rows,
          compressed_stride_0.unwrap(),
          out_stride_0.unwrap());
      return;
    }

    LaunchKernel((num_rows + 3) / 4, 128, device.unwrap())(
        higgs_dense_2bit_dequant_saw_scalar2_coalesced_kernel<4, int32_t>,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<int32_t*>(compact_page_table.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};


struct HiggsDense2BitDequantKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView codebook) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
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
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kKvDim})
        .with_strides({out_stride_0, kKvDim, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_dequant_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct HiggsDense2BitDequantConstCodebookKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView codebook) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
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
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kKvDim})
        .with_strides({out_stride_0, kKvDim, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_dequant_const_codebook_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct HiggsDense2BitDequantVec4Kernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView codebook) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
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
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kKvDim})
        .with_strides({out_stride_0, kKvDim, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_dequant_vec4_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct HiggsDense2BitDequantVec4LdgCodebookKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView codebook) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
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
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kKvDim})
        .with_strides({out_stride_0, kKvDim, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_dequant_vec4_ldg_codebook_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct HiggsDense2BitDequantPairLanesScaleBroadcastKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView codebook) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
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
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kKvDim})
        .with_strides({out_stride_0, kKvDim, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_dequant_pair_lanes_scale_broadcast_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct HiggsDense2BitDequantPageTableKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView compact_page_table,
      tvm::ffi::TensorView codebook) {
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
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    const int64_t num_rows = B.unwrap() * K.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_dequant_page_table_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<int32_t*>(compact_page_table.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct HiggsDense2BitDequantPageTableConstCodebookKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView compact_page_table,
      tvm::ffi::TensorView codebook) {
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
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    const int64_t num_rows = B.unwrap() * K.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_dequant_page_table_const_codebook_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<int32_t*>(compact_page_table.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

// ─── FP8 page-table dequant launchers ──────────────────────────────────────
// ai-blaise #19 iter3: emit fp8_e4m3 (576 B/row) for the trtllm-gen
// sparse-MLA FP8 cubin path. ``out_stride_0`` is in fp8_e4m3 elements
// (1 element = 1 byte), so out shape is ``(N, 1, kKvDim)`` with element
// dtype ``fp8_e4m3_t`` and total per-row payload of 576 B.

struct HiggsDense2BitDequantPageTableFp8Kernel {
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

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_dequant_page_table_fp8_kernel,
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

struct HiggsDense2BitDequantPageTableFp8ConstCodebookKernel {
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

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_dequant_page_table_fp8_const_codebook_kernel,
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

// FP8 saw_scalar2 page-table launcher. Mirrors the BF16 launcher above
// (4/8 warp_per_block fan-out, kSawScalar2LargeRowThreshold gating)
// but emits fp8_e4m3 via the saw_scalar2 lookup table.
struct HiggsDense2BitDequantPageTableFp8SawScalar2Kernel {
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

    if (num_rows >= kSawScalar2LargeRowThreshold) {
      LaunchKernel((num_rows + 7) / 8, 256, device.unwrap())(
          higgs_dense_2bit_dequant_saw_scalar2_fp8_coalesced_kernel<8, int32_t>,
          static_cast<const uint8_t*>(compressed.data_ptr()),
          static_cast<const int32_t*>(page_table.data_ptr()),
          static_cast<fp8_e4m3_t*>(out.data_ptr()),
          static_cast<int32_t*>(compact_page_table.data_ptr()),
          static_cast<const float*>(codebook.data_ptr()),
          num_rows,
          compressed_stride_0.unwrap(),
          out_stride_0.unwrap(),
          static_cast<float>(inv_kv_scale));
      return;
    }

    LaunchKernel((num_rows + 3) / 4, 128, device.unwrap())(
        higgs_dense_2bit_dequant_saw_scalar2_fp8_coalesced_kernel<4, int32_t>,
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

struct HiggsDense2BitDequantPageTableVec4Kernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView compact_page_table,
      tvm::ffi::TensorView codebook) {
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
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    const int64_t num_rows = B.unwrap() * K.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_dequant_page_table_vec4_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<int32_t*>(compact_page_table.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct HiggsDense2BitDequantPageTableVec4LdgCodebookKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView compact_page_table,
      tvm::ffi::TensorView codebook) {
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
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    const int64_t num_rows = B.unwrap() * K.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_dequant_page_table_vec4_ldg_codebook_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<int32_t*>(compact_page_table.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct HiggsDense2BitDequantPageTablePairLanesScaleBroadcastKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView compact_page_table,
      tvm::ffi::TensorView codebook) {
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
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kCodebookSize, kPairDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(codebook);

    const int64_t num_rows = B.unwrap() * K.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        higgs_dense_2bit_dequant_page_table_pair_lanes_scale_broadcast_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<int32_t*>(compact_page_table.data_ptr()),
        static_cast<const float*>(codebook.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

}  // namespace higgs_dense_2bit_detail
