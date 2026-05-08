# Copyright 2024-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Integration hooks for warp decode in the FusedMoE layer.

Provides ``maybe_warp_decode_forward`` which checks whether the current
forward pass should use warp decode (small batch, decode mode) and
dispatches accordingly.  Falls through to the normal expert-centric
path when warp decode is not applicable.

When CuTe CUDA kernels are available (via sgl_kernel), they are used
automatically on SM100+ (Blackwell B200) or when explicitly enabled
via SGLANG_WARP_DECODE_CUTE=1. Otherwise falls back to Triton kernels.

Usage in ``FusedMoE.forward_impl``::

    from sglang.srt.layers.moe.warp_decode.integration import (
        maybe_warp_decode_forward,
    )

    def forward_impl(self, hidden_states, topk_output):
        wd_result = maybe_warp_decode_forward(self, hidden_states, topk_output)
        if wd_result is not None:
            return wd_result
        # ... normal path ...

Environment variables:
    SGLANG_ENABLE_WARP_DECODE: Set to ``1`` to enable. Default off.
    SGLANG_WARP_DECODE_MAX_BATCH: Maximum batch size for warp decode
        (default 64). Batches larger than this fall through.
    SGLANG_WARP_DECODE_CUTE: ``1`` = force CuTe, ``0`` = force Triton,
        ``auto`` = CuTe on SM100+, Triton elsewhere. Default ``auto``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopKOutput

logger = logging.getLogger(__name__)

_WARP_DECODE_ENABLED: Optional[bool] = None
_WARP_DECODE_MAX_BATCH: Optional[int] = None


def _load_config() -> None:
    global _WARP_DECODE_ENABLED, _WARP_DECODE_MAX_BATCH
    _WARP_DECODE_ENABLED = os.environ.get("SGLANG_ENABLE_WARP_DECODE", "0") == "1"
    _WARP_DECODE_MAX_BATCH = int(
        os.environ.get("SGLANG_WARP_DECODE_MAX_BATCH", "64")
    )


def _ensure_config() -> None:
    if _WARP_DECODE_ENABLED is None:
        _load_config()


def maybe_warp_decode_forward(
    layer: "FusedMoE",
    hidden_states: torch.Tensor,
    topk_output: "TopKOutput",
) -> Optional[torch.Tensor]:
    """Try to run warp decode; return None if not applicable.

    This function checks:
    1. SGLANG_ENABLE_WARP_DECODE is set
    2. Batch size is within the threshold
    3. Weights are in a supported format (BF16)
    4. The topk_output has standard format (topk_ids + topk_weights)

    If all conditions are met, runs warp decode and returns the output.
    Otherwise returns None to let the caller proceed with the normal path.
    """
    _ensure_config()

    if not _WARP_DECODE_ENABLED:
        return None

    num_tokens = hidden_states.shape[0]
    if num_tokens > _WARP_DECODE_MAX_BATCH or num_tokens == 0:
        return None

    # Check topk output format - we need standard topk_ids and topk_weights
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if not TopKOutputChecker.format_is_standard(topk_output):
        return None

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights

    # Expert parallelism safety: when EP > 1, topk_ids contain global
    # expert IDs but weights are partitioned per rank.  The normal path
    # handles this via the dispatcher/expert_map; warp decode bypasses
    # that, so we fall through unless EP == 1.
    moe_ep_size = getattr(layer, "moe_ep_size", 1)
    if moe_ep_size > 1:
        return None

    # Check weight format: warp decode currently supports BF16 only
    if not hasattr(layer, "w13_weight") or not hasattr(layer, "w2_weight"):
        return None

    w13 = layer.w13_weight
    w2 = layer.w2_weight

    if isinstance(w13, torch.nn.Parameter):
        w13 = w13.data
    if isinstance(w2, torch.nn.Parameter):
        w2 = w2.data

    # Currently only support BF16 weights
    if w13.dtype != torch.bfloat16:
        return None

    from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

    # Determine intermediate size
    intermediate_size = getattr(
        layer, "intermediate_size_per_partition", w13.shape[1] // 2
    )

    # NOTE: Do NOT apply routed_scaling_factor here. The outer
    # DeepseekV2MoE.forward_normal handles scaling after this call
    # returns, either via maybe_fuse_routed_scale_and_shared_add or
    # direct multiplication. Applying it here would double-scale.

    output = warp_decode_moe_packed(
        hidden_states=hidden_states,
        w13=w13,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        intermediate_size=intermediate_size,
        inplace=False,
    )

    return output
