// NVFP4 KV variant of FlashMLA sparse-MLA decode config.
//
// Diff vs upstream FlashMLA csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/config.h:
//   - K_raw buffer typed as uint8_t (half the bytes vs fp8_e4m3)
//   - K_raw buffer sized to D_K/4 per token (vs D_K/2 for fp8)
//   - scales buffer typed as fp8_e4m3 (vs fp8_e8m0) and sized to 32 per token (vs 8)
//   - TMA_K_STRIDE_FOR_DECODING shrunk to match new per-token bytes
//   - new constants for NVFP4 scale layout

#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cutlass/float8.h>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"
#include "nvfp4_dequant.cuh"

namespace sm100::fwd_for_small_topk::head128_nvfp4 {

using namespace cute;
using namespace sm100::fwd_for_small_topk::nvfp4;

template<SparseAttnFwdMode FWD_MODE, int D_QK>
struct KernelTemplate {

using ArgT = SparseFwdArgT<FWD_MODE>;
static constexpr bool IS_DECODE = is_decode_v<FWD_MODE>;
static constexpr bool IS_PREFILL = !IS_DECODE;
using fp8_e4m3 = cutlass::float_e4m3_t;

struct TmaParamsForDecode {
    CUtensorMap tensor_map_q;
    CUtensorMap tensor_map_o;
    CUtensorMap tensor_map_o_accum;
    CUtensorMap tensor_map_kv_nope;       // NVFP4 packed bytes
    CUtensorMap tensor_map_kv_rope;       // BF16 (unchanged)
    CUtensorMap tensor_map_kv_scales;     // NEW: E4M3 per-block scales for kv_nope
    CUtensorMap tensor_map_extra_kv_nope;
    CUtensorMap tensor_map_extra_kv_rope;
    CUtensorMap tensor_map_extra_kv_scales;  // NEW: scales for extra-KV
};
using TmaParams = TmaParamsForDecode;

static_assert(D_QK == 512);

static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;
static constexpr float MAX_INIT_VAL = -1e30;

static constexpr int H_Q = 128;
static constexpr int B_TOPK = 64;
static constexpr int NUM_THREADS = 128*4;
static constexpr int NUM_WORKER_THREADS = (128 + 128 + 1 + 32 + 2 + 128)*2;

static constexpr int NUM_K_BUFS = 3;
static constexpr int NUM_RAW_K_BUFS = 2;
static constexpr int NUM_INDEX_BUFS = 4;

static constexpr int D_NOPE = 448;
static constexpr int D_ROPE = 64;
static constexpr int NVFP4_BLOCK_SIZE = 16;   // NVFP4 has one E4M3 scale per 16 elements

// Per-token NVFP4 K_nope layout: 448 e2m1 nibbles = 224 bytes packed
// Plus 448/16 = 28 E4M3 scale bytes
// Plus 2 * 64 BF16 rope = 128 bytes
// Total per token = 224 + 28 + 128 = 380 bytes (vs 576 for FP8)
//
// Note we pad NUM_NVFP4_SCALES_NOPE up to 32 for 16-byte alignment of the
// SMEM scales buffer; the kernel only references the first 28 entries.
static constexpr int NUM_NVFP4_SCALES_NOPE = D_NOPE / NVFP4_BLOCK_SIZE;    // 28
static constexpr int NUM_NVFP4_SCALES_PADDED = 32;                          // pad to 16-byte alignment
static constexpr int TMA_KV_NOPE_STRIDE = D_NOPE / 2;                       // 224 bytes
static constexpr int TMA_KV_SCALES_STRIDE = NUM_NVFP4_SCALES_PADDED;        // 32 bytes
static constexpr int TMA_KV_ROPE_STRIDE = 2 * D_ROPE * sizeof(__nv_bfloat16);  // 128 bytes

// Total stride per token in HBM = TMA_KV_NOPE_STRIDE + TMA_KV_SCALES_STRIDE + TMA_KV_ROPE_STRIDE = 384
// Round up to 16-byte alignment → 384.
static constexpr int TMA_K_STRIDE_FOR_DECODING = TMA_KV_NOPE_STRIDE + TMA_KV_SCALES_STRIDE + TMA_KV_ROPE_STRIDE;
static_assert(TMA_K_STRIDE_FOR_DECODING == 384);
static_assert(TMA_K_STRIDE_FOR_DECODING % 16 == 0, "TMA stride must be 16-byte aligned");

static constexpr int B_EPI = 64;
static constexpr int B_EPI_SPLITKV = 32;
static constexpr int NUM_EPI_SPLITKV_BUFS = 4;
static_assert((H_Q/2)*D_Q*sizeof(__nv_bfloat16) >= NUM_EPI_SPLITKV_BUFS*(H_Q/2)*(B_EPI_SPLITKV*2)*sizeof(float));

struct tmem_cols {
    static constexpr int O = 0;
    static constexpr int Q = 256;
    static constexpr int P = 384;
};

struct SharedMemoryPlan {
    // Q after BF16 epilogue (unchanged)
    array_aligned<__nv_bfloat16, (H_Q/2)*D_Q> Q;

    // Dequantized K in BF16 (unchanged shape; receives the UMMA input post-dequant)
    array_aligned<__nv_bfloat16, B_TOPK*(D_K/2)> K[NUM_K_BUFS];

    // ----- NVFP4-specific buffers -----
    // K_raw stores the packed NVFP4 nibbles. Half the bytes vs FP8 baseline.
    array_aligned<uint8_t, B_TOPK*(D_K/4)> K_raw_fp4[NUM_RAW_K_BUFS];
    // Per-token E4M3 scales (one per 16-element block) for K_nope.
    CUTE_ALIGNAS(16) fp8_e4m3 scales_fp4[NUM_INDEX_BUFS][B_TOPK][NUM_NVFP4_SCALES_PADDED];
    // ----- End NVFP4-specific -----

    array_aligned<__nv_bfloat16, (H_Q/2)*B_TOPK> S;
    float P_exchange[4][(H_Q/2/2)*(B_TOPK/2)];
    float rowwise_max_buf[128], rowwise_li_buf[128];

    CUTE_ALIGNAS(16) char is_k_valid[NUM_INDEX_BUFS][B_TOPK/8];
    CUTE_ALIGNAS(16) int tma_coord[NUM_INDEX_BUFS][B_TOPK];

    transac_bar_t bar_sQ_full, bar_tQ_empty, bar_tQ_full;
    transac_bar_t bar_tOut_full, bar_tOut_empty;
    transac_bar_t bar_KV_full[NUM_K_BUFS], bar_KV_empty[NUM_K_BUFS];
    transac_bar_t bar_P_empty;
    transac_bar_t bar_QK_done, bar_SV_done;
    transac_bar_t bar_S_O_full;
    transac_bar_t bar_li_full, bar_li_empty;

    transac_bar_t bar_raw_KV_full[NUM_RAW_K_BUFS], bar_raw_KV_empty[NUM_RAW_K_BUFS];
    transac_bar_t bar_valid_coord_scales_full[NUM_INDEX_BUFS], bar_valid_coord_scales_empty[NUM_INDEX_BUFS];

    ku::CLCResponseObj clc_response_obj;
    array_aligned<uint32_t, 1> tmem_start_addr;
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<__nv_bfloat16, __nv_bfloat16, float, H_Q, B_TOPK*2, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<__nv_bfloat16, __nv_bfloat16, float, H_Q, 256, UMMA::Major::K, UMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{},
    Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _16>{}
));

struct barrier_ids {
    static constexpr int WG0_SYNC = 0;
    static constexpr int WG2_SYNC = 1;
    static constexpr int WG2_WARP02_SYNC = 2;
    static constexpr int WG2_WARP13_SYNC = 3;
};

static __device__ void
sparse_attn_fwd_kernel_devfunc(const ArgT &params, const TmaParams &tma_params);

static void run(const ArgT& params);

};

}  // namespace sm100::fwd_for_small_topk::head128_nvfp4
