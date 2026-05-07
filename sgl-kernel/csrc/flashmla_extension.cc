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

#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

#ifndef SGL_KERNEL_ENABLE_SM100A
#include "params.h"

namespace sm100::decode::head64 {

template <ModelType MODEL_TYPE>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams&) {
  TORCH_CHECK(false, "SM100 FlashMLA sparse decode kernel is not built in this sgl-kernel package");
}

template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32>(const SparseAttnDecodeParams&);
template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1>(const SparseAttnDecodeParams&);

}  // namespace sm100::decode::head64

namespace sm100::fwd::head64 {

template <int HEAD_DIM_QK>
void run_fwd_phase1_kernel(const SparseAttnFwdParams&) {
  TORCH_CHECK(false, "SM100 FlashMLA sparse prefill kernel is not built in this sgl-kernel package");
}

template void run_fwd_phase1_kernel<512>(const SparseAttnFwdParams&);
template void run_fwd_phase1_kernel<576>(const SparseAttnFwdParams&);

}  // namespace sm100::fwd::head64

namespace sm100::fwd::head128 {

template <int HEAD_DIM_QK>
void run_fwd_phase1_kernel(const SparseAttnFwdParams&) {
  TORCH_CHECK(false, "SM100 FlashMLA sparse prefill kernel is not built in this sgl-kernel package");
}

template void run_fwd_phase1_kernel<512>(const SparseAttnFwdParams&);
template void run_fwd_phase1_kernel<576>(const SparseAttnFwdParams&);

}  // namespace sm100::fwd::head128

namespace sm100::fwd_for_small_topk::head128 {

template <SparseAttnFwdMode FWD_MODE, int HEAD_DIM_QK>
void run_fwd_for_small_topk_phase1_kernel(const SparseFwdArgT<FWD_MODE>&) {
  TORCH_CHECK(false, "SM100 FlashMLA small-topk kernel is not built in this sgl-kernel package");
}

template void run_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::Prefill, 512>(
    const SparseFwdArgT<SparseAttnFwdMode::Prefill>&);
template void run_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::DecodeWithSplitKV, 512>(
    const SparseFwdArgT<SparseAttnFwdMode::DecodeWithSplitKV>&);

}  // namespace sm100::fwd_for_small_topk::head128
#endif

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  /*
   * From FlashMLA
   */
  m.def(
      "get_mla_decoding_metadata(Tensor seqlens_k, int num_q_tokens_per_head_k, int h_k, int? h_q, bool "
      "is_fp8_kvcache, int? topk) -> Tensor[]");
  m.impl("get_mla_decoding_metadata", torch::kCUDA, &get_mla_decoding_metadata);

  m.def("get_mla_decoding_metadata_dense_fp8(Tensor seqlens_k, int num_heads_per_head_k, int num_heads_k) -> Tensor[]");
  m.impl("get_mla_decoding_metadata_dense_fp8", torch::kCUDA, &get_mla_decoding_metadata_dense_fp8);

  m.def(
      "fwd_kvcache_mla(Tensor q, Tensor kv_cache, int head_size_v, Tensor seqlens_k, Tensor block_table, float "
      "softmax_scale, bool is_causal, Tensor tile_scheduler_metadata, Tensor num_splits, bool is_fp8, Tensor? indices) "
      "-> Tensor[]");
  m.impl("fwd_kvcache_mla", torch::kCUDA, &fwd_kvcache_mla);

#ifdef SGL_KERNEL_ENABLE_SM100A
  m.def(
      "dense_prefill_fwd(Tensor workspace_buffer, Tensor q, Tensor k, Tensor v, Tensor cumulative_seqlen_q, Tensor "
      "cumulative_seqlen_kv, Tensor o, Tensor lse, int mask_mode_code, float softmax_scale, int max_seqlen_q, int "
      "max_seqlen_kv, bool is_varlen) -> ()");
  m.impl("dense_prefill_fwd", torch::kCUDA, &FMHACutlassSM100FwdRun);
#endif

  m.def("sparse_prefill_fwd(Tensor q, Tensor kv, Tensor indices, float sm_scale, int d_v) -> Tensor[]");
  m.impl("sparse_prefill_fwd", torch::kCUDA, &sparse_prefill_fwd);

  m.def(
      "fwd_kvcache_mla_fp8(Tensor q, Tensor kcache, int head_size_v, Tensor seqlens_k, Tensor block_table, float "
      "softmax_scale, bool is_causal, Tensor tile_scheduler_metadata, Tensor num_splits, Tensor? descale_q, Tensor? "
      "descale_k) -> Tensor[]");
  m.impl("fwd_kvcache_mla_fp8", torch::kCUDA, &fwd_kvcache_mla_fp8);
}

REGISTER_EXTENSION(flashmla_ops)
