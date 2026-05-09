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
"""WarpDecodeRunnerCore: MoE runner backend for warp decode kernels.

Integrates warp decode into sglang's MoE runner framework. When the
batch size is small enough (controlled by SGLANG_WARP_DECODE_MAX_BATCH),
the MoE layer dispatches to warp decode kernels instead of the
expert-centric path.

This runner handles:
  - BF16 weight format (separate gate/up/down or packed w13/w2)
  - Expert parallelism (only processes local experts)
  - Routing weight application
  - Proper integration with sglang's dispatch/combine pipeline
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

logger = logging.getLogger(__name__)

# Environment variable controls
WARP_DECODE_MAX_BATCH = int(
    os.environ.get("SGLANG_WARP_DECODE_MAX_BATCH", "64")
)
WARP_DECODE_ENABLED = os.environ.get(
    "SGLANG_ENABLE_WARP_DECODE", "0"
) == "1"


def is_warp_decode_enabled() -> bool:
    """Check if warp decode is enabled via environment variable."""
    return WARP_DECODE_ENABLED


def should_use_warp_decode(num_tokens: int) -> bool:
    """Check if warp decode should be used for the given batch size.

    Warp decode is beneficial for small-batch decode (B <= 64).
    For larger batches, expert-centric execution is more efficient
    because tokens can share expert computation.
    """
    return WARP_DECODE_ENABLED and num_tokens <= WARP_DECODE_MAX_BATCH


@dataclass
class WarpDecodeQuantInfo(MoeQuantInfo):
    """Quantization info for warp decode MoE runner."""

    # Packed gate+up weights [E, 2*N, K] or separate gate/up
    w13_weight: torch.Tensor
    # Down projection weights [E, D, N]
    w2_weight: torch.Tensor
    # Optional scales for quantized weights
    w13_weight_scale: Optional[torch.Tensor] = None
    w2_weight_scale: Optional[torch.Tensor] = None
    # Global scales for NVFP4
    g1_alphas: Optional[torch.Tensor] = None
    g2_alphas: Optional[torch.Tensor] = None
    # Number of experts
    global_num_experts: int = 256
    local_num_experts: int = 256
    local_expert_offset: int = 0
    # Intermediate size per partition (accounts for TP)
    intermediate_size_per_partition: int = 0


@dataclass
class WarpDecodeRunnerInput(RunnerInput):
    """Input for warp decode runner."""

    hidden_states: torch.Tensor  # [num_tokens, hidden_size]
    topk_ids: torch.Tensor  # [num_tokens, top_k]
    topk_weights: torch.Tensor  # [num_tokens, top_k]

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.WARP_DECODE


@dataclass
class WarpDecodeRunnerOutput(RunnerOutput):
    """Output from warp decode runner."""

    hidden_states: torch.Tensor  # [num_tokens, hidden_size]

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.WARP_DECODE


class WarpDecodeRunnerCore(MoeRunnerCore):
    """MoE runner using warp decode kernels for small-batch decode.

    Falls back to the provided fallback runner for large batches
    or when warp decode is not beneficial.

    Composes with all other custom optimizations (GatedNorm, G1 Gate,
    IndexCache, TurboQuant, LayerSplit, FlashSampling, HiSparse) because
    those operate in the attention phase of the decoder layer, not the
    MoE phase. Warp decode only replaces the expert computation kernel;
    everything else in the layer forward remains unchanged.

    Enable via: --moe-runner-backend warp_decode
    Or via env: SGLANG_ENABLE_WARP_DECODE=1 (for the fast-path hook)
    """

    def __init__(
        self,
        config: MoeRunnerConfig,
        fallback_runner: Optional[MoeRunnerCore] = None,
    ):
        super().__init__(config)
        self.fallback_runner = fallback_runner
        self._max_batch = WARP_DECODE_MAX_BATCH
        logger.info(
            "WarpDecodeRunnerCore initialized: max_batch=%d, "
            "intermediate_size=%s, hidden_size=%s, "
            "fallback=%s",
            self._max_batch,
            config.intermediate_size_per_partition,
            config.hidden_size,
            type(fallback_runner).__name__ if fallback_runner else "none",
        )

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.WARP_DECODE

    def run_from_dispatch(
        self,
        dispatch_output: Any,
        quant_info: MoeQuantInfo,
        config: MoeRunnerConfig,
        hooks: Optional[Any] = None,
    ) -> Any:
        """Direct dispatch path — bypasses pre/post permute.

        Warp decode doesn't need token scatter/gather because each
        program owns one output scalar. This method lets the MoeRunner
        skip the permute pipeline entirely.
        """
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        hidden_states = dispatch_output.hidden_states
        topk_ids = dispatch_output.topk_output.topk_ids
        topk_weights = dispatch_output.topk_output.topk_weights
        num_tokens = hidden_states.shape[0]

        if num_tokens > self._max_batch and self.fallback_runner is not None:
            return None

        if not hasattr(self, "_quant_w13"):
            self._quant_w13 = getattr(quant_info, "w13_weight", None)
            self._quant_w2 = getattr(quant_info, "w2_weight", None)
            self._intermediate = getattr(
                quant_info, "intermediate_size_per_partition",
                config.intermediate_size_per_partition,
            )

        w13 = self._quant_w13 or getattr(quant_info, "w13_weight", None)
        w2 = self._quant_w2 or getattr(quant_info, "w2_weight", None)
        intermediate = self._intermediate or config.intermediate_size_per_partition

        if w13 is None or w2 is None:
            if self.fallback_runner is not None:
                return None
            raise RuntimeError("Warp decode requires w13/w2 weights in quant_info")

        output = warp_decode_moe_packed(
            hidden_states=hidden_states,
            w13=w13,
            w2=w2,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            intermediate_size=intermediate,
            inplace=False,
        )

        from sglang.srt.layers.moe.token_dispatcher.base import CombineInput
        return CombineInput(
            hidden_states=output,
            topk_output=dispatch_output.topk_output,
            format=dispatch_output.format,
        )

    def run(
        self,
        runner_input: RunnerInput,
        quant_info: MoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> RunnerOutput:
        """Run warp decode or fall back to expert-centric.

        Dispatches to CuTe CUDA kernels when available (SM100+ or
        SGLANG_WARP_DECODE_CUTE=1), otherwise uses Triton kernels.
        """
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        if not isinstance(runner_input, WarpDecodeRunnerInput):
            if self.fallback_runner is not None:
                return self.fallback_runner.run(
                    runner_input, quant_info, running_state, hooks
                )
            raise TypeError(
                f"Expected WarpDecodeRunnerInput, got {type(runner_input)}"
            )

        assert isinstance(quant_info, WarpDecodeQuantInfo)

        num_tokens = runner_input.hidden_states.shape[0]

        if num_tokens > self._max_batch and self.fallback_runner is not None:
            return self.fallback_runner.run(
                runner_input, quant_info, running_state, hooks
            )

        topk_ids = runner_input.topk_ids
        if quant_info.local_expert_offset > 0:
            topk_ids = topk_ids - quant_info.local_expert_offset

        output = warp_decode_moe_packed(
            hidden_states=runner_input.hidden_states,
            w13=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_ids=topk_ids,
            topk_weights=runner_input.topk_weights,
            intermediate_size=quant_info.intermediate_size_per_partition,
            inplace=False,
        )

        return WarpDecodeRunnerOutput(hidden_states=output)


# ---------------------------------------------------------------------------
# Standalone warp decode function for direct integration
# ---------------------------------------------------------------------------

def warp_decode_moe_forward(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    intermediate_size: Optional[int] = None,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Standalone warp decode MoE forward pass.

    Can be called directly from model forward() when warp decode
    is preferred over the full MoE runner pipeline.

    Args:
        hidden_states: [num_tokens, hidden_size]
        w13: Packed gate+up weights [E, 2*N, K]
        w2: Down projection weights [E, D, N]
        topk_ids: [num_tokens, top_k]
        topk_weights: [num_tokens, top_k]
        intermediate_size: Override N dimension.
        routed_scaling_factor: Scale factor for routed experts.

    Returns:
        Output [num_tokens, hidden_size]
    """
    from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    return warp_decode_moe_packed(
        hidden_states=hidden_states,
        w13=w13,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        intermediate_size=intermediate_size,
        inplace=False,
    )
