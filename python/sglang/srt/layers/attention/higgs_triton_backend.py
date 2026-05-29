# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""HIGGS-aware Triton attention backend for the SMC-SD draft path.

Selected automatically by ``_get_attention_backend_from_str`` when the
draft pool is :class:`HiggsMHA2BitTokenToKVPool` and the draft attn
backend string is ``triton``. Reuses :class:`TritonAttnBackend` for
forward_extend + prefill + metadata management, and overrides
``forward_decode`` to call the fused
:func:`decode_attention_fwd_higgs` kernel against the packed (uint8)
K/V buffers — eliminating the 200-600x decode-time materialization
cost of the eager dequant in
:meth:`HiggsMHA2BitTokenToKVPool._get_key_buffer`.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.attention.triton_backend import (
    TritonAttnBackend,
    logit_capping_mod,
)
from sglang.srt.layers.attention.triton_ops.higgs_decode_attention import (
    decode_attention_fwd_higgs,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import HiggsMHA2BitTokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.model_executor.model_runner import ModelRunner


class HiggsTritonAttnBackend(TritonAttnBackend):
    """Triton attention backend with fused HIGGS-MHA-2bit decode.

    All extend / prefill paths fall through to
    :class:`TritonAttnBackend` (the codec stores BF16 K/V at write
    time via :meth:`HiggsMHA2BitTokenToKVPool.set_kv_buffer`, then the
    eager dequant path materializes BF16 again for those non-decode
    steps — extend tokens are O(few) per decode step, and the eager
    cost is amortized).
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(model_runner, skip_prefill, kv_indptr_buf)
        pool = model_runner.token_to_kv_pool
        assert isinstance(pool, HiggsMHA2BitTokenToKVPool), (
            "HiggsTritonAttnBackend requires HiggsMHA2BitTokenToKVPool; "
            f"got {type(pool).__name__}. Wire the pool via "
            "--kv-cache-dtype higgs_2bit (or "
            "--smc-draft-kv-cache-dtype higgs_2bit on the SMC-SD draft)."
        )
        self._higgs_pool = pool

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        sinks: Optional[torch.Tensor] = None,
    ):
        # Reshape Q to flat (token, num_q_heads, head_dim).
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        logits_soft_cap = logit_capping_mod(
            layer.logit_capping_method, layer.logit_cap
        )

        if save_kv_cache:
            # HIGGS pool computes its own per-row FP16 scale; drops
            # external k_scale/v_scale silently (see
            # HiggsMHA2BitTokenToKVPool.set_kv_buffer).
            get_token_to_kv_pool().set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        if (
            layer.sliding_window_size is not None
            and layer.sliding_window_size > -1
        ):
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        attn_logits = self.forward_metadata.attn_logits
        if (
            self.forward_metadata.swa_attn_logits is not None
            and layer.v_head_dim == self.swa_v_head_dim
        ):
            attn_logits = self.forward_metadata.swa_attn_logits

        # Packed (uint8) K/V buffers, no eager materialization.
        k_packed = self._higgs_pool.get_packed_key_buffer(layer.layer_id)
        v_packed = self._higgs_pool.get_packed_value_buffer(layer.layer_id)

        decode_attention_fwd_higgs(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k_packed,
            v_packed,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr,
            kv_indices,
            attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            logit_cap=logits_soft_cap,
            sinks=sinks,
        )
        return o
