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
// Slot layout (kSlotBytes = 258, vs 274 for 2.5-bit TurboQuant):
//
//   [packed 4-bit pair indices: 128 B]
//   [per-token block-scale fp16:   2 B]
//   [rope bf16 (kRopeDim=64):     128 B]

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
constexpr int kSlotBytes = kPackedBytes + kNormBytes + kRopeDim * 2;  // 258
constexpr float kInvSqrtLatentDim = 0.044194173824159216f;  // 1 / sqrt(512)

static_assert(kPackedBytes == 128, "expected 128 packed bytes per slot");
static_assert(kSlotBytes == 258, "expected slot bytes = 258");

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

  __shared__ float buf[kLatentDim];
  __shared__ float reduce_scratch[32];  // dedicated reduction scratch

  // 1. Load BF16 latent, cast to FP32. (No pre-scale here: the FWHT is
  //    orthonormal, so the per-token block scale ``s = ||FWHT(latent)|| /
  //    sqrt(N)`` we store later applies to the rotated, codebook-mapped
  //    reconstruction.)
  const bf16_t* lat_row = latent + row * latent_stride_0;
  const float xin = __bfloat162float(lat_row[tid]);

  // 2. Orthonormal FWHT (×1/sqrt(N) folded at the end).
  const float xrot = fwht_512_swizzled(xin, buf) * kInvSqrtLatentDim;

  // 3. Block reduction: sum of squares -> L2 norm -> scale = ||xrot|| /
  //    sqrt(N). We use a *dedicated* 32-float reduction scratch buffer
  //    to keep ``buf`` undisturbed: ``buf[swizzle(tid)] = xrot * sqrt(N)``
  //    still holds the un-normalized rotated value, which we'd like to
  //    keep so the codebook lookup can read from it after the per-token
  //    scale broadcasts across all threads.
  const float sum_sq = warp_reduce_sum(xrot * xrot);
  if ((tid & 31) == 0) {
    reduce_scratch[tid >> 5] = sum_sq;
  }
  __syncthreads();
  float scale;
  if (tid < 32) {
    float t = (tid < 16) ? reduce_scratch[tid] : 0.0f;
    t = warp_reduce_sum(t);
    if (tid == 0) {
      reduce_scratch[0] = t;
    }
  }
  __syncthreads();
  scale = sqrtf(fmaxf(reduce_scratch[0], 1.0e-32f)) * kInvSqrtLatentDim;

  // 4. Normalize and write to UNSWIZZLED SMEM positions so the EVEN-tid
  //    thread can read both coordinates of its pair via simple SMEM
  //    reads (the partner read is cross-warp for tid >= 32, which
  //    __shfl_xor_sync cannot do).
  const float xnorm = xrot / fmaxf(scale, 1.0e-30f);
  __syncthreads();          // sync before reusing ``buf``
  buf[tid] = xnorm;
  __syncthreads();

  // 5. Quantize: each EVEN thread owns one pair, looks up its partner
  //    (buf[tid + 1]), and runs nearest-neighbor over the 16-entry
  //    codebook. We inline the lookup so the compiler can keep the
  //    codebook in registers without an explicit struct allocation
  //    (which historically led to local-mem spills on Ampere when the
  //    quant-side predicate was divergent across the warp).
  uint32_t idx = 0;
  if ((tid & 1) == 0) {
    const float x0 = buf[tid];
    const float x1 = buf[tid + 1];
    float best = -3.4e38f;
    uint32_t best_idx = 0;
#pragma unroll
    for (int i = 0; i < kCodebookSize; ++i) {
      const float g0 = __ldg(&codebook[i * kPairDim + 0]);
      const float g1 = __ldg(&codebook[i * kPairDim + 1]);
      const float gn = __ldg(&codebook_norm_sq[i]);
      const float score = 2.0f * (x0 * g0 + x1 * g1) - gn;
      if (score > best) { best = score; best_idx = static_cast<uint32_t>(i); }
    }
    idx = best_idx;
  }
  __syncthreads();
  if ((tid & 1) == 0) {
    // Stash the 4-bit index at SMEM slot (tid >> 1) = pair index.
    reinterpret_cast<uint32_t*>(buf)[tid >> 1] = idx;
  }
  __syncthreads();

  // 6. Pack two adjacent 4-bit indices into one byte. byte[k] holds
  //    pair indices (2k, 2k+1). One writer thread per byte (tid 4k).
  const int64_t loc = locs[row];
  uint8_t* slot = compressed + loc * compressed_stride_0;

  if ((tid & 3) == 0) {
    const int pair0 = tid >> 1;     // pair index for the low nibble
    const int pair1 = pair0 + 1;    // pair index for the high nibble
    const uint32_t lo = reinterpret_cast<const uint32_t*>(buf)[pair0];
    const uint32_t hi = reinterpret_cast<const uint32_t*>(buf)[pair1];
    const int byte_idx = tid >> 2;  // 0..127
    slot[byte_idx] = static_cast<uint8_t>((lo & 0x0F) | ((hi & 0x0F) << 4));
  }

  // 7. Store fp16 scale at the end of the packed region.
  if (tid == 0) {
    half scale_h = __float2half(scale);
    *reinterpret_cast<half*>(slot + kPackedBytes) = scale_h;
  }

  // 8. Copy rope (kRopeDim*2 = 128 bytes) as 8 × 16-byte vectorised
  //    stores. Mirrors the TurboQuant rope copy.
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

// ─── Dequantize kernel: packed slot → BF16 latent + rope ────────────────────
// Same launch shape as the TurboQuant 2.5-bit dequant: <num_rows, kLatentDim>.
// Each thread owns one latent dim. Indices are unpacked from the packed byte
// (two indices per byte; tid 2k and tid 2k+1 share byte k).

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

// ─── Page-table dequant variant ─────────────────────────────────────────────
// Mirrors TurboQuant's ``dequantize_page_table_selected_2p5_kernel``. The
// page table indexes flat slots; rows with page < 0 are masked.

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

  const int64_t loc = page >= 0 ? static_cast<int64_t>(page) : 0;
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

// ─── Host-side launcher structs ─────────────────────────────────────────────

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
        higgs_dense_2bit_store_kernel,
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

}  // namespace higgs_dense_2bit_detail
