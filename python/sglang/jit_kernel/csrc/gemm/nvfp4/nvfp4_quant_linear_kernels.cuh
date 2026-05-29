/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

==============================================================================
Linear-layout (non-swizzled) NVFP4 quantization for trtllm_fp4_block_scale_moe
hidden-states input. Matches flashinfer's `fp4_quantize(is_sf_swizzled_layout=False)`
output shape: SF tensor is [m, K/16] row-major fp8_e4m3, no padding/swizzling.

This is the production target layout for DeepSeek-V3.2-REAP NVFP4 MoE deploy.
Compared to the swizzled CUTLASS variant, this kernel:
  * Writes SF in row-major order matching the trtllm-gen kernel expectation
    when `bA16` block-scale layout is consumed directly.
  * Has a simpler global-store pattern (no tiled M/K reordering).
  * Sets up for iter2 fusion with upstream RMSNorm / downstream prefetch.
==============================================================================*/

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include "nvfp4_quant.cuh"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace host;

// Linear (row-major) SF address helper. Matches flashinfer's non-swizzled
// layout: scales[m, K/16] in fp8_e4m3, stored contiguously.
template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
SGL_DEVICE uint8_t* cvt_quant_to_fp4_get_sf_out_offset_linear(
    int rowIdx, int colIdx, int numCols, SFType* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    // SF vector index (16 elements share one SF in the K dimension).
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    int32_t sf_cols = numCols / CVT_FP4_SF_VEC_SIZE;
    int64_t SFOffset = static_cast<int64_t>(mIdx) * sf_cols + kIdx;

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
#endif
  return nullptr;
}

// Linear-layout FP4 quantization kernel. Mirrors the swizzled variant in
// nvfp4_quant_kernels.cuh but uses cvt_quant_to_fp4_get_sf_out_offset_linear
// so the SF tensor lays out as [m, K/16] row-major (no tile-rotation).
template <class Type, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4_linear(
#else
cvt_fp16_to_fp4_linear(
#endif
    int32_t numRows, int32_t numCols, Type const* in, float const* SFScale,
    uint32_t* out, uint32_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x;
         colIdx < numCols / CVT_FP4_ELTS_PER_THREAD;
         colIdx += blockDim.x) {
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset_linear<uint32_t,
                                                    CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx, numCols, SFout);

      out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
    }
  }
#endif
}

template <typename T>
void invokeFP4QuantizationLinear(int m, int n, T const* input,
                                 float const* SFScale, int64_t* output,
                                 int32_t* SFOutput, bool useUE8M0,
                                 int multiProcessorCount,
                                 cudaStream_t stream) {
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));
  int const numBlocksPerSM = 2048 / block.x;
  dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

  if (useUE8M0) {
    cvt_fp16_to_fp4_linear<T, true><<<grid, block, 0, stream>>>(
        m, n, input, SFScale,
        reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(SFOutput));
  } else {
    cvt_fp16_to_fp4_linear<T, false><<<grid, block, 0, stream>>>(
        m, n, input, SFScale,
        reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(SFOutput));
  }
}

template void invokeFP4QuantizationLinear(int m, int n, half const* input,
                                          float const* SFScale,
                                          int64_t* output, int32_t* SFOutput,
                                          bool useUE8M0,
                                          int multiProcessorCount,
                                          cudaStream_t stream);

template void invokeFP4QuantizationLinear(int m, int n,
                                          __nv_bfloat16 const* input,
                                          float const* SFScale,
                                          int64_t* output, int32_t* SFOutput,
                                          bool useUE8M0,
                                          int multiProcessorCount,
                                          cudaStream_t stream);

inline int getSMVersionLinear(int device_id) {
  int sm_major = 0;
  int sm_minor = 0;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_major,
                                            cudaDevAttrComputeCapabilityMajor,
                                            device_id));
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_minor,
                                            cudaDevAttrComputeCapabilityMinor,
                                            device_id));
  return sm_major * 10 + sm_minor;
}

// Public entry: non-swizzled (linear) layout NVFP4 quantization.
// Matches flashinfer.fp4_quantize(is_sf_swizzled_layout=False) shape:
//   output:    [m, n/2] uint8 (packed E2M1)
//   output_sf: [m, n/16] int32 (the fp8_e4m3 scales reinterpreted as int32
//              4-wide; downstream views as fp8_e4m3fn)
void scaled_fp4_quant_linear_sm100a_sm120a(tvm::ffi::TensorView output,
                                           tvm::ffi::TensorView input,
                                           tvm::ffi::TensorView output_sf,
                                           tvm::ffi::TensorView input_sf) {
  RuntimeCheck(input.device().device_type == kDLCUDA,
               "input must be a CUDA tensor");
  RuntimeCheck(output.device() == input.device(),
               "output and input must be on same device");
  RuntimeCheck(output_sf.device() == input.device(),
               "output_sf and input must be on same device");
  RuntimeCheck(input_sf.device() == input.device(),
               "input_sf and input must be on same device");
  RuntimeCheck(input.dim() == 2, "input must be a 2D tensor");
  RuntimeCheck(output.dim() == 2, "output must be a 2D tensor");
  RuntimeCheck(output_sf.dim() == 2, "output_sf must be a 2D tensor");
  RuntimeCheck(input_sf.numel() == 1, "input_sf must have exactly one element");
  RuntimeCheck(host::is_type<uint8_t>(output.dtype()), "output must be uint8");
  RuntimeCheck(host::is_type<int32_t>(output_sf.dtype()),
               "output_sf must be int32");
  RuntimeCheck(host::is_type<float>(input_sf.dtype()),
               "input_sf must be float32");
  RuntimeCheck(host::is_type<fp16_t>(input.dtype()) ||
                   host::is_type<bf16_t>(input.dtype()),
               "input dtype must be fp16 or bf16");

  const int device_id = input.device().device_id;
  const auto sm_version = getSMVersionLinear(device_id);
  RuntimeCheck(sm_version >= 100,
               "fp4_quant_linear is only supported on sm100+");

  const int32_t m = static_cast<int32_t>(input.size(0));
  const int32_t n = static_cast<int32_t>(input.size(1));

  RuntimeCheck(output.size(0) == m, "output row size mismatch");
  RuntimeCheck(output.size(1) == n / 2, "output column size mismatch");
  RuntimeCheck(n % 16 == 0, "The N dimension must be multiple of 16.");

  // Output SF: [m, n/16] uint8 reinterpreted as int32 (n/16 must be
  // multiple of 4). For DeepSeek hidden_size=7168, n/16 = 448, which is
  // divisible by 4 so this packs cleanly.
  RuntimeCheck((n / 16) % 4 == 0,
               "N/16 must be multiple of 4 for int32 packing.");
  RuntimeCheck(output_sf.size(0) == m, "output_sf row size mismatch");
  RuntimeCheck(output_sf.size(1) == (n / 16) / 4,
               "output_sf column size mismatch (expect n/64 int32 cols)");

  const int multiProcessorCount =
      static_cast<int>(runtime::get_sm_count(device_id));

  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  const cudaStream_t stream = LaunchKernel::resolve_device(input.device());

  constexpr bool useUE8M0 = false;
  if (host::is_type<fp16_t>(input.dtype())) {
    auto input_ptr = reinterpret_cast<half const*>(input.data_ptr());
    invokeFP4QuantizationLinear(m, n, input_ptr, input_sf_ptr, output_ptr,
                                sf_out, useUE8M0, multiProcessorCount, stream);
  } else {
    auto input_ptr = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr());
    invokeFP4QuantizationLinear(m, n, input_ptr, input_sf_ptr, output_ptr,
                                sf_out, useUE8M0, multiProcessorCount, stream);
  }
}
