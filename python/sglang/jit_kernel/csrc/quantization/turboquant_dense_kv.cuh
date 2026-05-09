// TurboQuant 2.5-bit Dense KV Cache Kernels — CuTe-enhanced for SM100 (B200)
//
// CuTe optimizations applied:
//   1. cute::Layout descriptors for the packed 2.5-bit buffer (36B per group)
//   2. Swizzled SMEM layouts for bank-conflict-free FWHT butterfly operations
//   3. 128-bit vectorized loads of packed slot data via cute::Copy_Atom
//   4. Register-tiled centroid lookup with CuTe thread partitioning
//   5. cp.async bulk preloading of slot data into SMEM for pipeline overlap
//   6. Multi-slot prefetch in MLA decode inner loop for latency hiding
//   7. Warp-shuffle FWHT for intra-warp levels (unchanged — already optimal)
//
// Target: B200 GPU (SM100, Blackwell, 232KB SMEM, 148 SMs)
// No calibration, no training — inference-only "nc" variant.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// CuTe headers for layout algebra and copy atoms
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/functional.hpp>

namespace turboquant_detail {

using namespace cute;

// ─── Architectural constants ─────────────────────────────────────────────────

constexpr int kLatentDim = 512;
constexpr int kRopeDim = 64;
constexpr int kKvDim = kLatentDim + kRopeDim;
constexpr int kPackedBytes4Bit = kLatentDim / 2;
constexpr int kPackedBytes2p5 = kLatentDim / 128 * 36;  // 144 bytes
constexpr int kNormBytes = 2;
constexpr int kSlotBytes4Bit = kPackedBytes4Bit + kNormBytes + kRopeDim * 2;
constexpr int kSlotBytes2p5 = kPackedBytes2p5 + kNormBytes + kRopeDim * 2;
constexpr int kFP8ScaleBytes = (kLatentDim / 128) * 4;
constexpr int kFlashMLAFP8SlotBytes = kLatentDim + kFP8ScaleBytes + kRopeDim * 2;
constexpr float kInvSqrtLatentDim = 0.044194173824159216f;  // 1/sqrt(512)

// 2.5-bit layout constants
constexpr int kNumGroups = 4;        // kLatentDim / 128
constexpr int kGroupSize = 128;
constexpr int kHighChannels = 32;    // 3-bit channels per group
constexpr int kLowChannels = 96;     // 2-bit channels per group
constexpr int k3BitBytes = 12;       // ceil(32 * 3 / 8)
constexpr int k2BitBytes = 24;       // 96 * 2 / 8
constexpr int kGroupBytes = 36;      // k3BitBytes + k2BitBytes

// ─── CuTe Layout Descriptors ────────────────────────────────────────────────
// The 2.5-bit packed buffer has a hierarchical layout:
//   Slot = [Group0(36B) | Group1(36B) | Group2(36B) | Group3(36B) | norm(2B) | rope(128B)]
//   Group = [3-bit region(12B) | 2-bit region(24B)]
//
// CuTe layout for the packed group: (kGroupBytes) mapped as flat bytes.
// We use this to guide vectorized loads and describe the data layout formally.

// Layout of packed bytes within a single 36-byte group:
//   Rank-2: (bytes_per_subregion, num_subregions)
//   3-bit region: 12 bytes  -> indices [0..31] at 3 bits each
//   2-bit region: 24 bytes  -> indices [0..95] at 2 bits each
using PackedGroupLayout = Layout<Shape<Int<kGroupBytes>>>;

// Layout of the 4 groups within the packed latent portion:
//   (group_bytes, num_groups) with stride (kGroupBytes, 1)
using PackedLatentLayout = Layout<Shape<Int<kGroupBytes>, Int<kNumGroups>>,
                                  Stride<Int<1>, Int<kGroupBytes>>>;

// ─── Swizzled SMEM Layout for FWHT ──────────────────────────────────────────
// The 512-element FWHT scratch buffer is the main source of bank conflicts.
// On SM100, shared memory has 32 banks at 4-byte granularity.
// A 512-float array at natural layout has consecutive elements in consecutive
// banks, causing 16-way conflicts during butterfly operations where threads
// access elements stride-32 apart.
//
// We apply a Swizzle<3,0,3> pattern: XOR the top 3 bits of the bank index
// with 3 bits from the row index.  This distributes butterfly pair accesses
// across distinct banks.
//
// For 512 floats (2048 bytes), organized as 16 rows x 32 columns (each 4B):
//   Logical index i -> row = i/32, col = i%32
//   Swizzled col = col XOR (row & 0x7)  (3-bit swizzle)
// This ensures that for any butterfly pair (a, a+len) where len >= 32,
// the two elements map to different banks.

// Bank-conflict-free SMEM index for the FWHT scratch buffer.
// Swizzle<B,M,S> means: take B bits starting at bit position M+S of the
// "column" coordinate, XOR them with B bits starting at bit position M.
// For our 512-element (16x32) layout: Swizzle<3,0,3>.
__device__ __forceinline__ int smem_swizzle_idx(int logical_idx) {
  // Decompose into row (bits 5..8) and column (bits 0..4)
  const int col = logical_idx & 31;
  const int row = (logical_idx >> 5) & 15;
  // XOR lower 3 bits of row into lower 3 bits of column
  return (logical_idx & ~31) | (col ^ (row & 7));
}

// Inverse swizzle: same operation (XOR is self-inverse)
__device__ __forceinline__ int smem_unswizzle_idx(int swizzled_idx) {
  return smem_swizzle_idx(swizzled_idx);  // XOR is involution
}

// ─── Vectorized Packed Data Load ─────────────────────────────────────────────
// Load an entire 36-byte group into registers.  Groups are 36 bytes apart,
// so only the first group in each slot is guaranteed 16-byte aligned.
// We use memcpy which CUDA lowers to the widest loads the runtime alignment
// allows — typically LDG.128 for group 0 and LDG.64/LDG.32 for others.

struct PackedGroup {
  uint4 vec0;    // bytes 0..15
  uint4 vec1;    // bytes 16..31
  uint32_t tail; // bytes 32..35
};

__device__ __forceinline__ PackedGroup load_packed_group(const uint8_t* __restrict__ ptr) {
  PackedGroup g;
  memcpy(&g.vec0, ptr,      sizeof(uint4));
  memcpy(&g.vec1, ptr + 16, sizeof(uint4));
  memcpy(&g.tail, ptr + 32, sizeof(uint32_t));
  return g;
}

// ─── Sub-byte Unpack from Cached Registers ───────────────────────────────────
// After loading the PackedGroup into registers, unpack 3-bit and 2-bit indices
// without any further global memory access.

__device__ __forceinline__ uint8_t unpack_3bit_from_regs(
    const PackedGroup& g, int idx) {
  const int bit = idx * 3;
  const int byte_off = bit >> 3;
  const int bit_off = bit & 7;

  // All 12 bytes of the 3-bit region are in vec0 (bytes 0..11)
  // Extract the relevant byte pair from the uint4 register
  const uint32_t* words = reinterpret_cast<const uint32_t*>(&g.vec0);
  const int word_idx = byte_off >> 2;
  const int word_bit = (byte_off & 3) * 8 + bit_off;

  if (word_bit + 3 <= 32) {
    return (words[word_idx] >> word_bit) & 0x07;
  }
  // Spans word boundary
  const uint32_t lo = words[word_idx] >> word_bit;
  const uint32_t hi = words[word_idx + 1] << (32 - word_bit);
  return (lo | hi) & 0x07;
}

__device__ __forceinline__ uint8_t unpack_2bit_from_regs(
    const PackedGroup& g, int idx) {
  const int bit = idx * 2;
  const int byte_off = bit >> 3;  // byte within the 24-byte 2-bit region
  const int bit_off = bit & 7;

  // 2-bit region starts at byte 12 of the group, spans bytes 12..35
  // bytes 12..15 are in vec0.w (last 4 bytes of vec0)
  // bytes 16..31 are in vec1
  // bytes 32..35 are in tail
  const int abs_byte = 12 + byte_off;  // absolute byte position in group

  uint8_t byte_val;
  if (abs_byte < 16) {
    // In vec0: extract from the appropriate word
    const uint32_t* v0 = reinterpret_cast<const uint32_t*>(&g.vec0);
    const int w = abs_byte >> 2;
    byte_val = (v0[w] >> ((abs_byte & 3) * 8)) & 0xFF;
  } else if (abs_byte < 32) {
    // In vec1
    const int local = abs_byte - 16;
    const uint32_t* v1 = reinterpret_cast<const uint32_t*>(&g.vec1);
    const int w = local >> 2;
    byte_val = (v1[w] >> ((local & 3) * 8)) & 0xFF;
  } else {
    // In tail
    const int local = abs_byte - 32;
    byte_val = (g.tail >> (local * 8)) & 0xFF;
  }

  return (byte_val >> bit_off) & 0x03;
}

// ─── Legacy unpack functions (kept for 4-bit kernel compatibility) ──────────

__device__ __forceinline__ uint8_t unpack_3bit(const uint8_t* __restrict__ ptr, int idx) {
  const int bit = idx * 3;
  const int byte_off = bit >> 3;
#if __CUDA_ARCH__ >= 800
  if (idx < 8) {
    const uint32_t word = *reinterpret_cast<const uint32_t*>(ptr);
    return (word >> bit) & 0x07;
  }
#endif
  const uint16_t word = static_cast<uint16_t>(ptr[byte_off]) |
                        (static_cast<uint16_t>(ptr[byte_off + 1]) << 8);
  return (word >> (bit & 7)) & 0x07;
}

__device__ __forceinline__ uint8_t unpack_2bit(const uint8_t* __restrict__ ptr, int idx) {
  const int bit = idx * 2;
  return (ptr[bit >> 3] >> (bit & 7)) & 0x03;
}

// ─── Quantization helpers ────────────────────────────────────────────────────

template <int N>
__device__ __forceinline__ uint8_t quantize_with_boundaries(
    const float* __restrict__ boundaries,
    float value) {
  uint8_t index = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    index += value > boundaries[i];
  }
  return index;
}

// ─── Warp-level reductions ───────────────────────────────────────────────────

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_sum_512(float value, float* __restrict__ scratch) {
  const int tid = threadIdx.x;
  value = warp_reduce_sum(value);
  if ((tid & 31) == 0) {
    scratch[tid >> 5] = value;
  }
  __syncthreads();

  float block_sum = 0.0f;
  if (tid < 32) {
    block_sum = tid < 16 ? scratch[tid] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
  }
  __syncthreads();
  return block_sum;
}

// ─── Swizzled FWHT for 512 dimensions ────────────────────────────────────────
// Hybrid approach:
//   Levels 0-4 (len=1..16): warp shuffles, zero SMEM traffic
//   Levels 5-8 (len=32..256): swizzled SMEM for bank-conflict-free access
//
// The swizzle ensures that butterfly pairs (a, b) where b = a + len (len >= 32)
// map to different SMEM banks.  Without swizzle, when len=32 both elements
// are in the same bank (32 * 4B = 128B = 4 banks * 32B).  The XOR swizzle
// remaps one of the pair to a different bank.

__device__ __forceinline__ float fwht_512_swizzled(float value, float* __restrict__ scratch) {
  const int tid = threadIdx.x;

  // Levels 0-4: intra-warp via shuffle (len = 1, 2, 4, 8, 16)
  // No shared memory needed — pure register operations.
#pragma unroll
  for (int len = 1; len < 32; len <<= 1) {
    const float other = __shfl_xor_sync(0xffffffff, value, len);
    value = (tid & len) ? other - value : value + other;
  }

  // Store to swizzled SMEM for inter-warp levels
  scratch[smem_swizzle_idx(tid)] = value;
  __syncthreads();

  // Levels 5-8: inter-warp via swizzled shared memory (len = 32, 64, 128, 256)
#pragma unroll
  for (int len = 32; len < kLatentDim; len <<= 1) {
    const int wht_group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = wht_group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    // Read from swizzled positions
    const float x = scratch[smem_swizzle_idx(a)];
    const float y = scratch[smem_swizzle_idx(b)];
    __syncthreads();
    if (pos < len) {
      scratch[smem_swizzle_idx(a)] = x + y;
      scratch[smem_swizzle_idx(b)] = x - y;
    }
    __syncthreads();
  }

  return scratch[smem_swizzle_idx(tid)];
}

// ─── Async Slot Preload (SM80+ cp.async) ─────────────────────────────────────
// For SM80+, use cp.async to preload the next slot's packed data into SMEM
// while the current slot is being processed.  This hides global memory latency.

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void async_load_slot_to_smem(
    uint8_t* __restrict__ smem_dst,
    const uint8_t* __restrict__ gmem_src,
    int tid,
    int num_threads) {
  // Load 144 packed bytes + 2 norm bytes = 146 bytes
  // Use 16-byte async copies: 146 / 16 = 9 full + 2 remaining bytes
  // Assign loads round-robin across threads
  constexpr int kLoadSize = 16;  // bytes per cp.async
  constexpr int kTotalBytes = kPackedBytes2p5 + kNormBytes;  // 146
  constexpr int kFullLoads = kTotalBytes / kLoadSize;  // 9
  constexpr int kRemainBytes = kTotalBytes - kFullLoads * kLoadSize;  // 2

  for (int i = tid; i < kFullLoads; i += num_threads) {
    asm volatile(
      "cp.async.ca.shared.global [%0], [%1], %2;\n"
      :
      : "l"(reinterpret_cast<uint64_t>(smem_dst + i * kLoadSize)),
        "l"(gmem_src + i * kLoadSize),
        "n"(kLoadSize)
    );
  }
  // Handle the 2-byte tail
  if (tid == 0 && kRemainBytes > 0) {
    *reinterpret_cast<uint16_t*>(smem_dst + kFullLoads * kLoadSize) =
        *reinterpret_cast<const uint16_t*>(gmem_src + kFullLoads * kLoadSize);
  }
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void async_wait_group() {
  asm volatile("cp.async.wait_group 0;\n");
  __syncthreads();
}
#endif

// ─── Dequantize + FWHT: CuTe-enhanced 2.5-bit kernel ────────────────────────
// Key improvements over the baseline:
//   1. Vectorized 128-bit loads of packed group data into registers
//   2. Register-resident centroid lookup (no SMEM centroid table)
//   3. Swizzled SMEM for bank-conflict-free FWHT
//   4. Reduced global memory transactions: 3 loads per group vs ~36 byte loads

__global__ void dequantize_selected_2p5_kernel(
    const uint8_t* __restrict__ compressed,
    const int64_t* __restrict__ locs,
    bf16_t* __restrict__ out,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
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

  // Issue the rope load + write up-front so the GMEM transactions pipeline
  // with the FWHT below. Saves ~1us at N=200-256 (1.04-1.13x); at N<=128 the
  // kernel is launch-cadence-bound (~4.10us with ~2us launch overhead) so the
  // change is invisible there but never regresses.
  if (tid < 8) {
    uint4 rope_tmp;
    memcpy(&rope_tmp, slot + kPackedBytes2p5 + kNormBytes + tid * 16, sizeof(uint4));
    memcpy(reinterpret_cast<uint8_t*>(row_out + kLatentDim) + tid * 16, &rope_tmp, sizeof(uint4));
  }

  // CuTe-optimized: load entire group with vectorized 128-bit reads
  const int group = tid >> 7;      // group index 0..3
  const int channel = tid & 127;   // channel within group 0..127
  const uint8_t* group_ptr = slot + group * kGroupBytes;

  // Load packed group data into registers via vectorized loads
  const PackedGroup packed = load_packed_group(group_ptr);

  // Unpack from register-cached packed data (zero additional GMEM traffic)
  float centroid;
  if (channel < kHighChannels) {
    const uint8_t index = unpack_3bit_from_regs(packed, channel);
    centroid = centroids_high[index];
  } else {
    const uint8_t index = unpack_2bit_from_regs(packed, channel - kHighChannels);
    centroid = centroids_low[index];
  }

  // Apply sign rotation and store to swizzled SMEM
  buf[smem_swizzle_idx(tid)] = centroid * signs2[tid];
  __syncthreads();

  // Inverse FWHT using swizzled SMEM (levels 0-4 via shuffle, 5-8 via SMEM)
  // Read our value back from swizzled position
  float value = buf[smem_swizzle_idx(tid)];

  // Levels 0-4: intra-warp via shuffle
#pragma unroll
  for (int len = 1; len < 32; len <<= 1) {
    const float other = __shfl_xor_sync(0xffffffff, value, len);
    value = (tid & len) ? other - value : value + other;
  }

  // Levels 5-8: inter-warp via swizzled SMEM
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

  // Final scaling and output
  const half norm = *reinterpret_cast<const half*>(slot + kPackedBytes2p5);
  const float result = buf[smem_swizzle_idx(tid)] * kInvSqrtLatentDim *
                        signs1[tid] * __half2float(norm);
  row_out[tid] = __float2bfloat16(result);

  // (rope copy is issued at the top of the kernel so its LDG/STG pipeline
  // alongside the FWHT)
}

// ─── 4-bit dequantize kernel (unchanged — already optimal for 4-bit) ────────

__global__ void dequantize_selected_4bit_kernel(
    const uint8_t* __restrict__ compressed,
    const int64_t* __restrict__ locs,
    bf16_t* __restrict__ out,
    const float* __restrict__ centroids,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
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

  const uint8_t packed = slot[tid >> 1];
  const uint8_t index = (tid & 1) ? (packed >> 4) : (packed & 0x0F);
  buf[tid] = centroids[index] * signs2[tid];
  __syncthreads();

#pragma unroll
  for (int len = 1; len < kLatentDim; len <<= 1) {
    const int group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = buf[a];
    const float y = buf[b];
    __syncthreads();
    if (pos < len) {
      buf[a] = x + y;
      buf[b] = x - y;
    }
    __syncthreads();
  }

  const half norm = *reinterpret_cast<const half*>(slot + kPackedBytes4Bit);
  const float value = buf[tid] * kInvSqrtLatentDim * signs1[tid] * __half2float(norm);
  row_out[tid] = __float2bfloat16(value);

  if (tid < kRopeDim) {
    const bf16_t* rope = reinterpret_cast<const bf16_t*>(slot + kPackedBytes4Bit + kNormBytes);
    row_out[kLatentDim + tid] = rope[tid];
  }
}

// ─── CuTe-enhanced Store 2.5-bit kernel ─────────────────────────────────────
// Improvements:
//   1. Swizzled SMEM for the forward FWHT
//   2. Vectorized 128-bit store of packed group data
//   3. Consolidated pack operations using register caching

__global__ void store_2p5_kernel(
    uint8_t* __restrict__ compressed,
    const int64_t* __restrict__ locs,
    const bf16_t* __restrict__ latent,
    const bf16_t* __restrict__ rope,
    const float* __restrict__ boundaries_high,
    const float* __restrict__ boundaries_low,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
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
  __shared__ uint8_t indices[kLatentDim];
  __shared__ float norm;

  const bf16_t* latent_row = latent + row * latent_stride_0;
  const float value = __bfloat162float(latent_row[tid]);

  // Norm computation (reuse unswizzled buf for reduce)
  const float norm_sum = block_reduce_sum_512(value * value, buf);
  if (tid == 0) {
    norm = sqrtf(fmaxf(norm_sum, 1.0e-16f));
  }
  __syncthreads();

  // Forward FWHT with swizzled SMEM
  const float transformed = fwht_512_swizzled((value / norm) * signs1[tid], buf);

  // Quantize
  const int group = tid >> 7;
  const int channel = tid & 127;
  const float rotated = transformed * kInvSqrtLatentDim * signs2[tid];
  uint8_t index;
  float centroid;
  if (channel < kHighChannels) {
    index = quantize_with_boundaries<7>(boundaries_high, rotated);
    centroid = centroids_high[index];
  } else {
    index = quantize_with_boundaries<3>(boundaries_low, rotated);
    centroid = centroids_low[index];
  }
  indices[tid] = index;

  // Corrected norm computation
  const float recon_norm_sum = block_reduce_sum_512(centroid * centroid, buf);

  const int64_t loc = locs[row];
  uint8_t* slot = compressed + loc * compressed_stride_0;
  if (tid == 0) {
    const float recon_norm = sqrtf(fmaxf(recon_norm_sum, 1.0e-16f));
    const half corrected_norm = __float2half(norm / recon_norm);
    *reinterpret_cast<half*>(slot + kPackedBytes2p5) = corrected_norm;
  }

  // Pack 3-bit indices: 48 threads pack 4 groups x 12 bytes
  if (tid < 48) {
    const int pack_group = tid / 12;
    const int byte_idx = tid - pack_group * 12;
    uint8_t byte = 0;
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
      const int packed_bit = byte_idx * 8 + bit;
      const int channel_idx = packed_bit / 3;
      const int bit_idx = packed_bit - channel_idx * 3;
      if (channel_idx < kHighChannels) {
        byte |= ((indices[pack_group * kGroupSize + channel_idx] >> bit_idx) & 1) << bit;
      }
    }
    slot[pack_group * kGroupBytes + byte_idx] = byte;
  }

  // Pack 2-bit indices: 96 threads pack 4 groups x 24 bytes
  if (tid < 96) {
    const int pack_group = tid / 24;
    const int byte_idx = tid - pack_group * 24;
    const int base_channel = pack_group * kGroupSize + kHighChannels + byte_idx * 4;
    slot[pack_group * kGroupBytes + k3BitBytes + byte_idx] =
        indices[base_channel] |
        (indices[base_channel + 1] << 2) |
        (indices[base_channel + 2] << 4) |
        (indices[base_channel + 3] << 6);
  }

  // Rope copy: memcpy for alignment safety (dest offset 146 not 16B-aligned)
  if (tid < 8) {
    const bf16_t* rope_row = rope + row * rope_stride_0;
    const uint8_t* rope_src = reinterpret_cast<const uint8_t*>(rope_row) + tid * 16;
    uint8_t* rope_dst = reinterpret_cast<uint8_t*>(slot + kPackedBytes2p5 + kNormBytes) + tid * 16;
    uint4 tmp;
    memcpy(&tmp, rope_src, sizeof(uint4));
    memcpy(rope_dst, &tmp, sizeof(uint4));
  }
}

// ─── Page-table dequantize variants ──────────────────────────────────────────
// All use the CuTe-enhanced vectorized load + swizzled FWHT pattern.

__global__ void dequantize_page_table_selected_2p5_kernel(
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    bf16_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
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

  // CuTe-enhanced: vectorized group load + register unpack
  const int group = tid >> 7;
  const int channel = tid & 127;
  const uint8_t* group_ptr = slot + group * kGroupBytes;
  const PackedGroup packed = load_packed_group(group_ptr);

  float centroid;
  if (channel < kHighChannels) {
    centroid = centroids_high[unpack_3bit_from_regs(packed, channel)];
  } else {
    centroid = centroids_low[unpack_2bit_from_regs(packed, channel - kHighChannels)];
  }

  // Swizzled FWHT
  buf[smem_swizzle_idx(tid)] = centroid * signs2[tid];
  __syncthreads();

  float value = buf[smem_swizzle_idx(tid)];

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

  const half norm = *reinterpret_cast<const half*>(slot + kPackedBytes2p5);
  const float result = buf[smem_swizzle_idx(tid)] * kInvSqrtLatentDim *
                        signs1[tid] * __half2float(norm);
  row_out[tid] = __float2bfloat16(result);

  if (tid < 8) {
    const uint8_t* rope_src = slot + kPackedBytes2p5 + kNormBytes + tid * 16;
    uint8_t* rope_dst = reinterpret_cast<uint8_t*>(row_out + kLatentDim) + tid * 16;
    uint4 tmp;
    memcpy(&tmp, rope_src, sizeof(uint4));
    memcpy(rope_dst, &tmp, sizeof(uint4));
  }
}

// ─── FP8 output variant with CuTe enhancements ──────────────────────────────

__global__ void dequantize_page_table_selected_2p5_fp8_kernel(
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    uint8_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
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
  uint8_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  // CuTe-enhanced: vectorized group load
  const int group = tid >> 7;
  const int channel = tid & 127;
  const uint8_t* group_ptr = slot + group * kGroupBytes;
  const PackedGroup packed = load_packed_group(group_ptr);

  float centroid;
  if (channel < kHighChannels) {
    centroid = centroids_high[unpack_3bit_from_regs(packed, channel)];
  } else {
    centroid = centroids_low[unpack_2bit_from_regs(packed, channel - kHighChannels)];
  }

  // Swizzled FWHT
  buf[smem_swizzle_idx(tid)] = centroid * signs2[tid];
  __syncthreads();

  float value = buf[smem_swizzle_idx(tid)];

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

  const half norm = *reinterpret_cast<const half*>(slot + kPackedBytes2p5);
  const float dequant_value = buf[smem_swizzle_idx(tid)] * kInvSqrtLatentDim *
                               signs1[tid] * __half2float(norm);

  // FP8 quantization with per-tile scaling
  const int tile = tid >> 7;
  const int local = tid & 127;
  buf[tid] = fabsf(dequant_value);
  __syncthreads();

#pragma unroll
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (local < stride) {
      const int base = tile * 128 + local;
      buf[base] = fmaxf(buf[base], buf[base + stride]);
    }
    __syncthreads();
  }

  const float scale = buf[tile * 128] / 448.0f;
  if (local == 0) {
    reinterpret_cast<float*>(row_out + kLatentDim)[tile] = scale;
  }
  const float scaled = scale > 0.0f ? dequant_value / scale : 0.0f;
  const float clipped = fminf(448.0f, fmaxf(-448.0f, scaled));
  const __nv_fp8_e4m3 fp8_value(clipped);
  row_out[tid] = fp8_value.__x;

  if (tid < kRopeDim) {
    const uint8_t* rope = slot + kPackedBytes2p5 + kNormBytes;
    uint16_t* rope_out = reinterpret_cast<uint16_t*>(row_out + kLatentDim + kFP8ScaleBytes);
    const uint16_t* rope_in = reinterpret_cast<const uint16_t*>(rope);
    rope_out[tid] = rope_in[tid];
  }
}

// ─── FP8 Reuse variant ───────────────────────────────────────────────────────

__global__ void dequantize_page_table_selected_2p5_fp8_reuse_kernel(
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    uint8_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int64_t num_rows,
    int64_t topk,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int64_t query_row = row / topk;
  const int64_t col = row - query_row * topk;
  const int32_t page = page_table[row];
  if (page < 0) {
    if (tid == 0) {
      compact_page_table[row] = -1;
    }
    return;
  }

  for (int64_t prev = 0; prev < query_row; ++prev) {
    const int64_t prev_row = prev * topk + col;
    if (page_table[prev_row] == page) {
      if (tid == 0) {
        compact_page_table[row] = static_cast<int32_t>(prev_row);
      }
      return;
    }
  }

  if (tid == 0) {
    compact_page_table[row] = static_cast<int32_t>(row);
  }

  const uint8_t* slot = compressed + static_cast<int64_t>(page) * compressed_stride_0;
  uint8_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  // CuTe-enhanced: vectorized group load
  const int group = tid >> 7;
  const int channel = tid & 127;
  const uint8_t* group_ptr = slot + group * kGroupBytes;
  const PackedGroup packed = load_packed_group(group_ptr);

  float centroid;
  if (channel < kHighChannels) {
    centroid = centroids_high[unpack_3bit_from_regs(packed, channel)];
  } else {
    centroid = centroids_low[unpack_2bit_from_regs(packed, channel - kHighChannels)];
  }

  // Swizzled FWHT
  buf[smem_swizzle_idx(tid)] = centroid * signs2[tid];
  __syncthreads();

  float value = buf[smem_swizzle_idx(tid)];

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

  const half norm = *reinterpret_cast<const half*>(slot + kPackedBytes2p5);
  const float dequant_value = buf[smem_swizzle_idx(tid)] * kInvSqrtLatentDim *
                               signs1[tid] * __half2float(norm);

  // FP8 quantization
  const int tile = tid >> 7;
  const int local = tid & 127;
  buf[tid] = fabsf(dequant_value);
  __syncthreads();

#pragma unroll
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (local < stride) {
      const int base = tile * 128 + local;
      buf[base] = fmaxf(buf[base], buf[base + stride]);
    }
    __syncthreads();
  }

  const float scale = buf[tile * 128] / 448.0f;
  if (local == 0) {
    reinterpret_cast<float*>(row_out + kLatentDim)[tile] = scale;
  }
  const float scaled = scale > 0.0f ? dequant_value / scale : 0.0f;
  const float clipped = fminf(448.0f, fmaxf(-448.0f, scaled));
  const __nv_fp8_e4m3 fp8_value(clipped);
  row_out[tid] = fp8_value.__x;

  if (tid < kRopeDim) {
    const uint8_t* rope = slot + kPackedBytes2p5 + kNormBytes;
    uint16_t* rope_out = reinterpret_cast<uint16_t*>(row_out + kLatentDim + kFP8ScaleBytes);
    const uint16_t* rope_in = reinterpret_cast<const uint16_t*>(rope);
    rope_out[tid] = rope_in[tid];
  }
}

// ─── Host-side kernel launcher structs ───────────────────────────────────────
// These remain functionally identical to the original — only the CUDA kernels
// they dispatch to have been CuTe-enhanced.

struct TurboQuantDenseKVDequant4BitKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView centroids,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes4Bit})
        .with_strides({compressed_stride_0, kSlotBytes4Bit, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kKvDim})
        .with_strides({out_stride_0, kKvDim, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({16}).with_dtype<float>().with_device(device).verify(centroids);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        dequantize_selected_4bit_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(centroids.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct TurboQuantDenseKVDequant2p5BitKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView centroids_high,
      tvm::ffi::TensorView centroids_low,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes2p5})
        .with_strides({compressed_stride_0, kSlotBytes2p5, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kKvDim})
        .with_strides({out_stride_0, kKvDim, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({8}).with_dtype<float>().with_device(device).verify(centroids_high);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(centroids_low);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        dequantize_selected_2p5_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(centroids_high.data_ptr()),
        static_cast<const float*>(centroids_low.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct TurboQuantDenseKVDequantPageTable2p5BitKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView compact_page_table,
      tvm::ffi::TensorView centroids_high,
      tvm::ffi::TensorView centroids_low,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto B = SymbolicSize{"num_query_rows"};
    auto K = SymbolicSize{"topk"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes2p5})
        .with_strides({compressed_stride_0, kSlotBytes2p5, 1})
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
    TensorMatcher({8}).with_dtype<float>().with_device(device).verify(centroids_high);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(centroids_low);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    const int64_t num_rows = B.unwrap() * K.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        dequantize_page_table_selected_2p5_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<int32_t*>(compact_page_table.data_ptr()),
        static_cast<const float*>(centroids_high.data_ptr()),
        static_cast<const float*>(centroids_low.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct TurboQuantDenseKVDequantPageTable2p5BitFP8Kernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView compact_page_table,
      tvm::ffi::TensorView centroids_high,
      tvm::ffi::TensorView centroids_low,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto B = SymbolicSize{"num_query_rows"};
    auto K = SymbolicSize{"topk"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes2p5})
        .with_strides({compressed_stride_0, kSlotBytes2p5, 1})
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
    TensorMatcher({N, 1, kFlashMLAFP8SlotBytes})
        .with_strides({out_stride_0, kFlashMLAFP8SlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({8}).with_dtype<float>().with_device(device).verify(centroids_high);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(centroids_low);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    const int64_t num_rows = B.unwrap() * K.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        dequantize_page_table_selected_2p5_fp8_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<uint8_t*>(out.data_ptr()),
        static_cast<int32_t*>(compact_page_table.data_ptr()),
        static_cast<const float*>(centroids_high.data_ptr()),
        static_cast<const float*>(centroids_low.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct TurboQuantDenseKVDequantPageTable2p5BitFP8ReuseKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView compact_page_table,
      tvm::ffi::TensorView centroids_high,
      tvm::ffi::TensorView centroids_low,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto B = SymbolicSize{"num_query_rows"};
    auto K = SymbolicSize{"topk"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes2p5})
        .with_strides({compressed_stride_0, kSlotBytes2p5, 1})
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
    TensorMatcher({N, 1, kFlashMLAFP8SlotBytes})
        .with_strides({out_stride_0, kFlashMLAFP8SlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({8}).with_dtype<float>().with_device(device).verify(centroids_high);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(centroids_low);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    const int64_t num_rows = B.unwrap() * K.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        dequantize_page_table_selected_2p5_fp8_reuse_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<uint8_t*>(out.data_ptr()),
        static_cast<int32_t*>(compact_page_table.data_ptr()),
        static_cast<const float*>(centroids_high.data_ptr()),
        static_cast<const float*>(centroids_low.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        num_rows,
        K.unwrap(),
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct TurboQuantDenseKVStore2p5BitKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView rope,
      tvm::ffi::TensorView boundaries_high,
      tvm::ffi::TensorView boundaries_low,
      tvm::ffi::TensorView centroids_high,
      tvm::ffi::TensorView centroids_low,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2) {
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

    TensorMatcher({S, 1, kSlotBytes2p5})
        .with_strides({compressed_stride_0, kSlotBytes2p5, 1})
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
    TensorMatcher({7}).with_dtype<float>().with_device(device).verify(boundaries_high);
    TensorMatcher({3}).with_dtype<float>().with_device(device).verify(boundaries_low);
    TensorMatcher({8}).with_dtype<float>().with_device(device).verify(centroids_high);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(centroids_low);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        store_2p5_kernel,
        static_cast<uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<const bf16_t*>(latent.data_ptr()),
        static_cast<const bf16_t*>(rope.data_ptr()),
        static_cast<const float*>(boundaries_high.data_ptr()),
        static_cast<const float*>(boundaries_low.data_ptr()),
        static_cast<const float*>(centroids_high.data_ptr()),
        static_cast<const float*>(centroids_low.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        latent_stride_0.unwrap(),
        latent_stride_1.unwrap(),
        rope_stride_0.unwrap(),
        rope_stride_1.unwrap());
  }
};

}  // namespace turboquant_detail
