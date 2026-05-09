/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// GatedNorm CuTe kernel: PyTorch C++ binding.
// Registered via TORCH_LIBRARY_FRAGMENT in common_extension.cc.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "gated_norm_cute.cuh"
#include "utils.h"

void sgl_gated_norm_cute_forward(
    torch::Tensor normed,
    torch::Tensor w_down,
    torch::Tensor w_up,
    torch::Tensor output) {
  // --- Input validation ---
  CHECK_INPUT(normed);
  CHECK_INPUT(w_down);
  CHECK_INPUT(w_up);
  CHECK_INPUT(output);

  CHECK_EQ(normed.dtype(), at::kBFloat16);
  CHECK_EQ(w_down.dtype(), at::kBFloat16);
  CHECK_EQ(w_up.dtype(), at::kBFloat16);
  CHECK_EQ(output.dtype(), at::kBFloat16);

  TORCH_CHECK(normed.dim() >= 2,
              "normed must have at least 2 dimensions, got ", normed.dim());
  TORCH_CHECK(w_down.dim() == 2, "w_down must be 2D, got ", w_down.dim());
  TORCH_CHECK(w_up.dim() == 2, "w_up must be 2D, got ", w_up.dim());

  const int hidden_size = normed.size(-1);
  const int rank = w_down.size(0);

  TORCH_CHECK(w_down.size(1) == hidden_size,
              "w_down shape mismatch: expected [", rank, ", ", hidden_size,
              "], got [", w_down.size(0), ", ", w_down.size(1), "]");
  TORCH_CHECK(w_up.size(0) == hidden_size,
              "w_up shape mismatch: expected [", hidden_size, ", ", rank,
              "], got [", w_up.size(0), ", ", w_up.size(1), "]");
  TORCH_CHECK(w_up.size(1) == rank,
              "w_down/w_up rank mismatch: ", rank, " vs ", w_up.size(1));
  TORCH_CHECK(rank >= 1 && rank <= sgl_gated_norm::kMaxRank,
              "rank must be in [1, ", sgl_gated_norm::kMaxRank,
              "], got ", rank);
  TORCH_CHECK(output.sizes() == normed.sizes(),
              "output shape must match normed shape");

  auto flat_normed = normed.reshape({-1, hidden_size}).contiguous();
  auto flat_output = output.reshape({-1, hidden_size});
  const int num_tokens = flat_normed.size(0);

  if (num_tokens == 0) return;

  // Hold contiguous copies so they outlive the async kernel launch.
  auto w_down_c = w_down.contiguous();
  auto w_up_c = w_up.contiguous();

  const c10::cuda::OptionalCUDAGuard device_guard(normed.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaError_t err = sgl_gated_norm::launch_gated_norm_cute(
      reinterpret_cast<const __nv_bfloat16*>(flat_normed.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w_down_c.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w_up_c.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(flat_output.data_ptr()),
      num_tokens,
      hidden_size,
      rank,
      stream);

  TORCH_CHECK(err == cudaSuccess,
              "gated_norm_cute_forward failed (rank=", rank,
              ", hidden_size=", hidden_size, "): ",
              cudaGetErrorString(err),
              ". If cudaErrorInvalidValue, the rank may exceed the SMEM "
              "budget; fall back to the Triton kernel.");
}
