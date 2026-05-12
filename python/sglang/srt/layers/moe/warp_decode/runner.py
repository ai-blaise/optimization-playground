# Copyright 2024-2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""MoE runner backend for Warp Decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend


def is_warp_decode_enabled() -> bool:
    return envs.SGLANG_ENABLE_WARP_DECODE.get()


def should_use_warp_decode(num_tokens: int) -> bool:
    return (
        is_warp_decode_enabled()
        and num_tokens > 0
        and num_tokens <= envs.SGLANG_WARP_DECODE_MAX_BATCH.get()
    )


def _tensor_data(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if isinstance(tensor, torch.nn.Parameter):
        return tensor.data
    return tensor


@dataclass
class WarpDecodeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: Optional[torch.Tensor] = None
    w2_weight_scale: Optional[torch.Tensor] = None
    g1_alphas: Optional[torch.Tensor] = None
    g2_alphas: Optional[torch.Tensor] = None
    global_num_experts: int = 256
    local_num_experts: int = 256
    local_expert_offset: int = 0
    intermediate_size_per_partition: int = 0


@dataclass
class WarpDecodeRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.WARP_DECODE


@dataclass
class WarpDecodeRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.WARP_DECODE


class WarpDecodeRunnerCore(MoeRunnerCore):
    """Run Warp Decode for small BF16 decode batches."""

    def __init__(
        self,
        config: MoeRunnerConfig,
        fallback_runner: Optional[MoeRunnerCore] = None,
    ):
        super().__init__(config)
        self.fallback_runner = fallback_runner
        self._max_batch = envs.SGLANG_WARP_DECODE_MAX_BATCH.get()

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.WARP_DECODE

    def _triton_fallback(
        self,
        dispatch_output: Any,
        quant_info: MoeQuantInfo,
        config: MoeRunnerConfig,
    ) -> Any:
        from sglang.srt.layers.moe.moe_runner.triton import (
            TritonMoeQuantInfo,
            fused_experts_none_to_triton,
        )

        if isinstance(quant_info, TritonMoeQuantInfo):
            return fused_experts_none_to_triton(dispatch_output, quant_info, config)
        raise NotImplementedError(
            "Warp Decode runner only falls back for Triton-compatible MoE "
            f"quant info, got {type(quant_info).__name__}."
        )

    def run_from_dispatch(
        self,
        dispatch_output: Any,
        quant_info: MoeQuantInfo,
        config: MoeRunnerConfig,
        hooks: Optional[Any] = None,
    ) -> Any:
        from sglang.srt.layers.moe.topk import TopKOutputChecker
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        if (
            hidden_states.dtype != torch.bfloat16
            or hidden_states.shape[0] == 0
            or hidden_states.shape[0] > self._max_batch
            or not TopKOutputChecker.format_is_standard(topk_output)
        ):
            return self._triton_fallback(dispatch_output, quant_info, config)

        w13 = _tensor_data(getattr(quant_info, "w13_weight", None))
        w2 = _tensor_data(getattr(quant_info, "w2_weight", None))
        if w13 is None or w2 is None:
            return self._triton_fallback(dispatch_output, quant_info, config)
        if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
            return self._triton_fallback(dispatch_output, quant_info, config)

        intermediate_size = getattr(
            quant_info,
            "intermediate_size_per_partition",
            config.intermediate_size_per_partition,
        )
        output = warp_decode_moe_packed(
            hidden_states=hidden_states,
            w13=w13,
            w2=w2,
            topk_ids=topk_output.topk_ids,
            topk_weights=topk_output.topk_weights,
            intermediate_size=intermediate_size,
            inplace=False,
        )
        return StandardCombineInput(hidden_states=output)

    def run(
        self,
        runner_input: RunnerInput,
        quant_info: MoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> RunnerOutput:
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        if not isinstance(runner_input, WarpDecodeRunnerInput):
            if self.fallback_runner is not None:
                return self.fallback_runner.run(
                    runner_input, quant_info, running_state, hooks
                )
            raise TypeError(
                f"Expected WarpDecodeRunnerInput, got {type(runner_input)}"
            )

        if not isinstance(quant_info, WarpDecodeQuantInfo):
            raise TypeError(f"Expected WarpDecodeQuantInfo, got {type(quant_info)}")

        if (
            runner_input.hidden_states.shape[0] > self._max_batch
            and self.fallback_runner is not None
        ):
            return self.fallback_runner.run(
                runner_input, quant_info, running_state, hooks
            )
        if runner_input.hidden_states.dtype != torch.bfloat16:
            raise TypeError(
                "Warp Decode only supports BF16 hidden states, got "
                f"{runner_input.hidden_states.dtype}."
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


def warp_decode_moe_forward(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    intermediate_size: Optional[int] = None,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
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
