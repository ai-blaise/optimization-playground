// UMMA descriptor helpers for NVFP4 KV sparse-MLA decode.
//
// Replaces the FP8 dequant pipeline entirely. Instead of decompressing FP4 -> BF16
// in SMEM and feeding the existing BF16 UMMA, we use SM_100's block-scaled tensor
// core operations that consume packed FP4 + per-block E4M3 scales as direct UMMA
// operands. K stays in FP4 form right through to the math units.
//
// Reference: NVIDIA PTX 8.5+ tcgen05.mma.kind::f8f6f4 and ::nvf4 block-scaled
// matrix multiplies. DeepGEMM's sm100_fp4_mqa_logits kernel uses the same family
// with MXFP4 (E8M0) scales; we use NVFP4 (E4M3) scales here per directive.
//
// Key UMMA atoms:
//   SM100_MMA_F8F6F4_BS_2x1SM_SS<A=E4M3, B=E2M1, ACC=float, SF=E4M3, M, N, AMaj, BMaj>
//     Mixed FP8 (Q) x FP4 (K) with E4M3 per-block scales on K. The instruction
//     issues one CGA-wide tile multiply with scale-tensor lookups fused into the
//     accumulation step.
//
//   make_instr_desc_block_scaled<A, B, ACC, SF, M, N, AMaj, BMaj>()
//     Builds the runtime tile descriptor that ties the scale-tensor address to
//     the operand. Issued once per persistent CTA, reused across all the
//     top-k slot iterations.

#pragma once

#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/float8.h>
#include <cutlass/numeric_types.h>

namespace sm100::fwd_for_small_topk::nvfp4 {

using namespace cute;

using bf16 = __nv_bfloat16;
using fp8_e4m3 = cutlass::float_e4m3_t;
using fp4_e2m1 = cutlass::float_e2m1_t;

// Block scale layout: one E4M3 scale per 16-element block of the K-major
// dimension. For our 512-dim latent, that's 32 scales per token.
constexpr int kNVFP4ScaleVecSize = 16;

// UMMA tile shape for the P = Q * K^T matmul.
// Q is FP8 (E4M3), K is FP4 (E2M1) with E4M3 per-block scales.
// Output is fp32 accumulator in TMEM (cast to bf16 in epilogue).
// M = H_Q = 128 (number of query heads per 2-CTA cluster)
// N = B_TOPK * 2 = 128 (number of selected slots * 2 CTAs)
// K is consumed in tiles of 64 per UMMA instruction (FP4 + block scale requirement)
template <int M_TILE, int N_TILE>
using TiledMMA_P_NVFP4 = decltype(make_tiled_mma(
    cute::SM100_MMA_F8F6F4_BS_2x1SM_SS_NOELECT<
        fp8_e4m3,   // A operand (Q, FP8)
        fp4_e2m1,   // B operand (K, FP4)
        float,      // ACC
        fp8_e4m3,   // Scale factor type (E4M3 for NVFP4)
        M_TILE,
        N_TILE,
        UMMA::Major::K,
        UMMA::Major::K>{}));

// UMMA tile for the O = P * V matmul.
// V is the same NVFP4 latent storage as K (MLA shares K/V).
// P is the softmaxed score, stored as bf16 in TMEM.
template <int M_TILE, int N_TILE>
using TiledMMA_O_NVFP4 = decltype(make_tiled_mma(
    cute::SM100_MMA_F8F6F4_BS_2x1SM_SS_NOELECT<
        bf16,
        fp4_e2m1,
        float,
        fp8_e4m3,
        M_TILE,
        N_TILE,
        UMMA::Major::K,
        UMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{},
    Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _16>{}));

// Build the runtime instr_desc that ties the K block-scale tensor to the UMMA.
// Returns an opaque uint32_t that's passed to ptx::SM100_MMA_F8F6F4_SS::fma.
__device__ __forceinline__ uint32_t
make_nvfp4_p_instr_desc(uint32_t sf_smem_offset_k) {
    return cute::UMMA::make_instr_desc_block_scaled<
        fp8_e4m3, fp4_e2m1, float, fp8_e4m3,
        128, 128, UMMA::Major::K, UMMA::Major::K
    >(sf_smem_offset_k);
}

__device__ __forceinline__ uint32_t
make_nvfp4_o_instr_desc(uint32_t sf_smem_offset_v) {
    return cute::UMMA::make_instr_desc_block_scaled<
        bf16, fp4_e2m1, float, fp8_e4m3,
        128, 256, UMMA::Major::K, UMMA::Major::MN
    >(sf_smem_offset_v);
}

}  // namespace sm100::fwd_for_small_topk::nvfp4
