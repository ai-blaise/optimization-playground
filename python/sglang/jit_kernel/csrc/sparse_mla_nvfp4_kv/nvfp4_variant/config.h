// NVFP4 KV variant of FlashMLA sparse-MLA decode config — NATIVE UMMA path.
//
// Versus the SMEM-dequant scaffolding this replaces, this design:
//   - Has NO BF16 K[] SMEM buffer (K stays as packed FP4 throughout)
//   - Has NO dequant warpgroup at all (warpgroup_idx==1 becomes coord-prep-only)
//   - Uses SM_100 block-scaled UMMA (kind::f8f6f4) that consumes packed FP4 +
//     E4M3 per-block scale as operands directly
//
// SMEM bytes per K row:
//   FP8 baseline    : 512 (BF16 K dequanted) + 256 (FP8 raw)     = 768
//   SMEM-dequant FP4: 512 (BF16 K dequanted) + 128 (FP4 raw)     = 640
//   Native UMMA FP4 :                          128 (FP4) + 16 (E4M3 scales) =  144 (-5x vs FP8)
//
// Plus the dequant pass + its register pressure + its barrier traffic all
// disappear.

#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cutlass/float8.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"
#include "nvfp4_umma_descriptors.cuh"

namespace sm100::fwd_for_small_topk::head128_nvfp4 {

using namespace cute;
using namespace sm100::fwd_for_small_topk::nvfp4;

template<SparseAttnFwdMode FWD_MODE, int D_QK>
struct KernelTemplate {

using ArgT = SparseFwdArgT<FWD_MODE>;
static constexpr bool IS_DECODE = is_decode_v<FWD_MODE>;
static constexpr bool IS_PREFILL = !IS_DECODE;

struct TmaParamsForDecode {
    CUtensorMap tensor_map_q;          // FP8 Q (unchanged from baseline)
    CUtensorMap tensor_map_o;          // BF16 output (unchanged)
    CUtensorMap tensor_map_o_accum;    // FP32 split-K accumulator
    CUtensorMap tensor_map_kv_nope;    // PACKED FP4 latent values
    CUtensorMap tensor_map_kv_scales;  // E4M3 per-block scales (NEW)
    CUtensorMap tensor_map_kv_rope;    // BF16 rope (unchanged)
    CUtensorMap tensor_map_extra_kv_nope;
    CUtensorMap tensor_map_extra_kv_scales;
    CUtensorMap tensor_map_extra_kv_rope;
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

// One fewer worker warp count since warpgroup 1's dequant inner loop is gone.
// Warpgroup 1 still exists but only does coord prep + scale TMA load.
static constexpr int NUM_WORKER_THREADS = (128 + 64 + 1 + 32 + 2 + 128)*2;

// Just ONE K SMEM buffer family (FP4 packed); no K_raw distinction since K is
// never dequanted. Reuse the existing pipeline depth (NUM_K_BUFS=3) for the
// FP4 ping-pong staging.
static constexpr int NUM_K_BUFS = 3;
static constexpr int NUM_INDEX_BUFS = 4;

static constexpr int D_NOPE = 448;
static constexpr int D_ROPE = 64;
static constexpr int NVFP4_BLOCK_SIZE = 16;
static constexpr int NUM_NVFP4_SCALES_NOPE = D_NOPE / NVFP4_BLOCK_SIZE;     // 28
static constexpr int NUM_NVFP4_SCALES_PADDED = 32;
static constexpr int TMA_KV_NOPE_STRIDE = D_NOPE / 2;                       // 224 bytes
static constexpr int TMA_KV_SCALES_STRIDE = NUM_NVFP4_SCALES_PADDED;        // 32 bytes
static constexpr int TMA_KV_ROPE_STRIDE = 2 * D_ROPE * sizeof(__nv_bfloat16);  // 128 bytes
static constexpr int TMA_K_STRIDE_FOR_DECODING = TMA_KV_NOPE_STRIDE + TMA_KV_SCALES_STRIDE + TMA_KV_ROPE_STRIDE;
static_assert(TMA_K_STRIDE_FOR_DECODING == 384);
static_assert(TMA_K_STRIDE_FOR_DECODING % 16 == 0);

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
    array_aligned<__nv_bfloat16, (H_Q/2)*D_Q> Q;                              // Q stays BF16 in epilogue

    // K SMEM: packed FP4 values + E4M3 per-block scales. NO dequant target.
    array_aligned<fp4_e2m1, B_TOPK*(D_K/2)> K[NUM_K_BUFS];                    // (FP4 packed; each fp4_e2m1 is conceptually one nibble; sizeof yields 0.5 bytes packed by compiler)
    CUTE_ALIGNAS(16) fp8_e4m3 K_scales[NUM_K_BUFS][B_TOPK][NUM_NVFP4_SCALES_PADDED];
    CUTE_ALIGNAS(16) fp8_e4m3 V_scales[NUM_K_BUFS][B_TOPK][NUM_NVFP4_SCALES_PADDED];  // MLA shares K/V latent, but UMMA wants separate operand descriptors

    array_aligned<__nv_bfloat16, (H_Q/2)*B_TOPK> S;
    float P_exchange[4][(H_Q/2/2)*(B_TOPK/2)];
    float rowwise_max_buf[128], rowwise_li_buf[128];

    CUTE_ALIGNAS(16) char is_k_valid[NUM_INDEX_BUFS][B_TOPK/8];
    CUTE_ALIGNAS(16) int tma_coord[NUM_INDEX_BUFS][B_TOPK];

    transac_bar_t bar_sQ_full, bar_tQ_empty, bar_tQ_full;
    transac_bar_t bar_tOut_full, bar_tOut_empty;

    // K buffer barriers (combined values + scales). NO separate raw_K barriers
    // because there's no dequant pass to wait on.
    transac_bar_t bar_KV_full[NUM_K_BUFS], bar_KV_empty[NUM_K_BUFS];

    transac_bar_t bar_P_empty;
    transac_bar_t bar_QK_done, bar_SV_done;
    transac_bar_t bar_S_O_full;
    transac_bar_t bar_li_full, bar_li_empty;

    // Per-index barriers still needed for coord prep handoff to TMA loader
    transac_bar_t bar_valid_coord_full[NUM_INDEX_BUFS], bar_valid_coord_empty[NUM_INDEX_BUFS];

    ku::CLCResponseObj clc_response_obj;
    array_aligned<uint32_t, 1> tmem_start_addr;
};

// NVFP4-native UMMA: FP8 Q × FP4 K with E4M3 per-block scale operand.
// Issued via cute::SM100_MMA_F8F6F4_BS_2x1SM_SS — see nvfp4_umma_descriptors.cuh.
using TiledMMA_P = TiledMMA_P_NVFP4<H_Q, B_TOPK*2>;
using TiledMMA_O = TiledMMA_O_NVFP4<H_Q, 256>;

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
