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
"""FusedMoE hook for the small-batch Warp Decode path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopKOutput

def maybe_warp_decode_forward(
    layer: "FusedMoE",
    hidden_states: torch.Tensor,
    topk_output: "TopKOutput",
) -> Optional[torch.Tensor]:
    """Return a Warp Decode output, or ``None`` when the normal path should run."""
    if not envs.SGLANG_ENABLE_WARP_DECODE.get():
        return None

    num_tokens = hidden_states.shape[0]
    if num_tokens == 0 or num_tokens > envs.SGLANG_WARP_DECODE_MAX_BATCH.get():
        return None
    if hidden_states.dtype != torch.bfloat16:
        return None

    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if not TopKOutputChecker.format_is_standard(topk_output):
        return None

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights

    if getattr(layer, "moe_ep_size", 1) > 1:
        return None

    if not hasattr(layer, "w13_weight") or not hasattr(layer, "w2_weight"):
        return None

    w13 = layer.w13_weight
    w2 = layer.w2_weight

    if isinstance(w13, torch.nn.Parameter):
        w13 = w13.data
    if isinstance(w2, torch.nn.Parameter):
        w2 = w2.data

    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return None

    from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

    intermediate_size = getattr(
        layer, "intermediate_size_per_partition", w13.shape[1] // 2
    )

    return warp_decode_moe_packed(
        hidden_states=hidden_states,
        w13=w13,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        intermediate_size=intermediate_size,
        inplace=False,
    )
