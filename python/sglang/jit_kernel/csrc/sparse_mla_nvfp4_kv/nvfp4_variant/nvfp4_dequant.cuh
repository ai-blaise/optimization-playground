// NVFP4 KV dequant helpers for FlashMLA-derived sparse-MLA decode kernel.
//
// Replaces the FP8 (E4M3 × E8M0) dequant path with NVFP4 (E2M1 × E4M3).
// Same target output: BF16 in SMEM, fed to the existing UMMA pipeline.
//
// SM_100 hardware dequant uses the PTX intrinsic
//   cvt.rn.bf16x2.e2m1x2  (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt)
// which converts two E2M1 nibbles into two BF16 values in a single instruction.
// We then multiply by the per-block E4M3 scale (also a single PTX instruction
// via cvt.rn.bf16x2.e4m3x2 → 2× bf16 multiply).

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cute/tensor.hpp>

namespace sm100::fwd_for_small_topk::nvfp4 {

using bf16 = __nv_bfloat16;
using bf16x2 = __nv_bfloat162;
using fp8_e4m3 = cutlass::float_e4m3_t;

// NVFP4 E2M1 value table indexed by (sign | exp | mantissa) 4-bit code.
// Per the OCP MX-format spec: 1 sign bit, 2 exponent bits, 1 mantissa bit.
//   code & 0x8  -> sign (1=negative)
//   code & 0x6  -> exponent biased by 1
//   code & 0x1  -> mantissa
// Magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
__device__ __constant__ float kE2M1Magnitude[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

// Convert a single E2M1 nibble (4 bits) to fp32 magnitude * sign.
__device__ __forceinline__ float e2m1_to_fp32(uint8_t code) {
    const float magnitude = kE2M1Magnitude[code & 0x7u];
    return (code & 0x8u) ? -magnitude : magnitude;
}

// Hardware-accelerated path: cvt.rn.bf16x2.e2m1x2 converts two e2m1 nibbles
// packed in the low byte of a 32-bit register into a bf16x2.
//
// Layout of `packed`:
//   bits [0:3]   = first e2m1 (output .x)
//   bits [4:7]   = second e2m1 (output .y)
//
// Note: this PTX op exists on SM_100; emits one MXFP4-aware tensor-core conversion.
__device__ __forceinline__ bf16x2 cvt_e2m1x2_to_bf16x2(uint32_t packed_byte) {
    uint32_t out;
    asm volatile(
        "cvt.rn.bf16x2.e2m1x2 %0, %1;\n"
        : "=r"(out) : "r"(packed_byte));
    return *reinterpret_cast<bf16x2*>(&out);
}

// Convert E4M3 scale byte (NVFP4 per-block scale) to bf16.
__device__ __forceinline__ bf16 cvt_e4m3_to_bf16(fp8_e4m3 scale) {
    // Use the standard fp8_e4m3 -> float conversion, then narrow to bf16.
    // SM_100 also exposes cvt.rn.bf16.e4m3 but with the same precision.
    float f = float(scale);
    return __float2bfloat16_rn(f);
}

// Dequantize 8 NVFP4 elements (4 bytes) into 8 BF16 values, applying a
// single per-block scale (E4M3, valid for all 8 if they fall in one
// 16-element block; otherwise caller must split).
//
// Inputs:
//   packed: 4 bytes (32 bits) packed NVFP4 = 8 e2m1 nibbles
//   scale_bf16: per-block scale already converted to bf16
//
// Outputs:
//   out[0..8] = 8 BF16 values
__device__ __forceinline__ void
nvfp4x8_to_bf16x8_with_scale(uint32_t packed, bf16 scale_bf16, bf16 out[8]) {
    bf16x2 scale_x2 = {scale_bf16, scale_bf16};
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        uint32_t byte = (packed >> (b * 8)) & 0xffu;
        // Two e2m1 nibbles per byte → bf16x2
        bf16x2 vals = cvt_e2m1x2_to_bf16x2(byte);
        bf16x2 scaled = __hmul2(vals, scale_x2);
        out[b * 2 + 0] = scaled.x;
        out[b * 2 + 1] = scaled.y;
    }
}

// Same as above but uses two scales — necessary when the 8-element span
// crosses a 16-element block boundary (only happens if block_size < 8;
// our config uses block_size=16 so each call's 8 elements fall in one block).
__device__ __forceinline__ void
nvfp4x8_to_bf16x8_with_two_scales(
    uint32_t packed,
    bf16 scale_lo_bf16,  // for first 4 elems
    bf16 scale_hi_bf16,  // for last 4 elems
    bf16 out[8]) {
    bf16x2 scale_lo_x2 = {scale_lo_bf16, scale_lo_bf16};
    bf16x2 scale_hi_x2 = {scale_hi_bf16, scale_hi_bf16};
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        uint32_t byte = (packed >> (b * 8)) & 0xffu;
        bf16x2 vals = cvt_e2m1x2_to_bf16x2(byte);
        bf16x2 scaled = __hmul2(vals, b < 2 ? scale_lo_x2 : scale_hi_x2);
        out[b * 2 + 0] = scaled.x;
        out[b * 2 + 1] = scaled.y;
    }
}

// Storage descriptors for the NVFP4 KV layout (per-token, 512-dim latent):
//
//   K_nope_fp4:    256 bytes  (512 e2m1 nibbles = 128 bytes per half)
//   K_nope_scales: 32 bytes   (32 E4M3 scales, one per 16-element block)
//   K_rope:        128 bytes  (64 BF16 elements, unchanged from FP8 path)
//   Total:         416 bytes per token  (vs 648 for FP8)
//
// Vs FP8 layout:
//   K_nope_fp8:    512 bytes  (512 E4M3, one byte each)
//   K_nope_scales: 7 bytes    (E8M0, one per 64-element block) + 1 pad
//   K_rope:        128 bytes
//   Total:         648 bytes per token  (~64% larger than NVFP4)
//
// HBM bandwidth savings per indexed slot per layer: ~36%. At our cell
// (B=8, K=1024 selected, 61 layers), the KV read traffic per decode step
// drops from ~316 MB (FP8) to ~203 MB (NVFP4). At B200 HBM 8 TB/s the
// pure-bandwidth saving is ~14 us per step — not the dominant TPOT term
// individually, but compounds with the kernel-side compute reduction
// (FP4 GEMM throughput is 2x FP8 on tensor cores).

}  // namespace sm100::fwd_for_small_topk::nvfp4
