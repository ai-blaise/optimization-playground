from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.fp8_utils import is_blackwell_supported
from sglang.srt.layers.quantization.utils import (
    prepare_static_weights_for_trtllm_fp4_moe,
    reorder_w1w3_to_w3w1,
    replace_parameter,
    swizzle_blockscale,
)
from sglang.srt.utils import next_power_of_2, set_weight_attrs

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A4Nvfp4MoE"]

# Env-var gate for the sgl-native non-swizzled FP4 quantize path on the
# trtllm NVFP4 MoE deploy. Set SGLANG_USE_SGL_NVFP4_QUANT=1 to swap
# flashinfer.fp4_quantize for sglang.jit_kernel.nvfp4.scaled_fp4_quant_linear.
# Default is 0 (flashinfer) for now — flip to default-on once microbench
# confirms parity at production batch sizes.
_USE_SGL_NVFP4_QUANT_LINEAR = (
    os.environ.get("SGLANG_USE_SGL_NVFP4_QUANT", "0") == "1"
)

# Iter3 #15 deploy wire: when an upstream RMSNorm hook fuses the
# residual-add + RMSNorm with the linear NVFP4 quantize, it stashes the
# resulting (fp4, sf) tuple on the hidden_states tensor as
# ``_sglang_pre_quantized_fp4``. The activation lives behind
# ``envs.SGLANG_USE_SGL_NVFP4_FUSED_RMSNORM`` so a deploy can opt in
# without disturbing the unfused path. When the stash is present we
# consume it and skip the redundant `fp4_quantize` call; otherwise the
# original `_USE_SGL_NVFP4_QUANT_LINEAR` / flashinfer path runs.

# Honor the MoE NVFP4-dispatch env at module load (default 0). When set
# alongside DeepEP a2a, the dispatcher is instructed to return NVFP4-packed
# hidden_states + e4m3 block-scale tensors instead of bf16, which halves
# the dispatch bandwidth and lets the per-expert masked GEMM skip its
# internal quantize. Mirrors the modelopt scheme's MOE_NVFP4_DISPATCH wire
# (modelopt_quant.py:274) so the two NVFP4 schemes stay observationally
# consistent under the same env.
_MOE_NVFP4_DISPATCH = envs.SGLANG_MOE_NVFP4_DISPATCH.get()

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


class CompressedTensorsW4A4Nvfp4MoE(CompressedTensorsMoEScheme):

    def __init__(self):
        if not is_blackwell_supported():
            raise ValueError(
                "Current platform does not support NVFP4"
                " quantization. Please use Blackwell and"
                " above."
            )
        self.group_size = 16
        self._is_deepep = get_moe_a2a_backend().is_deepep()
        # The trtllm fast path (`trtllm_fp4_block_scale_moe`) does internal
        # routing on a single rank and ingests un-dispatched tokens; it is
        # mutually exclusive with DeepEP, which pre-dispatches tokens into
        # per-local-expert buckets. When DeepEP is enabled we force the
        # cutlass weight layout (swizzled blockscales + per-expert alphas)
        # so the DeepEP-LL apply path can drive `flashinfer_cutedsl_moe_masked`
        # without an extra weight reshuffle. The runner backend the user
        # passes (e.g. `flashinfer_cutlass`) does not change weight prep
        # here — only the apply-path branch.
        self.use_flashinfer_trtllm = (
            get_moe_runner_backend().is_flashinfer_trtllm()
            and not self._is_deepep
        )
        if self._is_deepep and get_moe_runner_backend().is_flashinfer_trtllm():
            logger.warning_once(
                "CompressedTensorsW4A4Nvfp4MoE: flashinfer_trtllm runner is "
                "incompatible with DeepEP a2a (the trtllm kernel does internal "
                "routing). Forcing cutlass weight layout + flashinfer cutedsl "
                "masked-GEMM apply path."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        # Requires sm100(blackwell) architecture
        return 100

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Weight Global Scales
        w13_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_weight_global_scale", w13_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_weight_global_scale", w2_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        # Input Global Scales
        w13_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_input_global_scale", w13_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_input_global_scale", w2_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # From packed to weight
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        if self.use_flashinfer_trtllm:
            w, s = reorder_w1w3_to_w3w1(
                layer.w13_weight.data, layer.w13_weight_scale.data, dim=-2
            )
            layer.w13_weight = torch.nn.Parameter(w, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(s, requires_grad=False)

        if not torch.allclose(
            layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
        ):
            logger.warning_once(
                "w1_weight_global_scale must match w3_weight_global_scale. "
                "Accuracy may be affected."
            )

        # Take inverse of global scale saved to disk
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            1 / layer.w13_weight_global_scale[:, 0], requires_grad=False
        )

        layer.w2_weight_scale_2 = torch.nn.Parameter(
            1 / layer.w2_weight_global_scale.data, requires_grad=False
        )

        # w13
        if self.use_flashinfer_trtllm:
            w13_input_global_scale = (
                layer.w13_input_global_scale.min()
                .to(torch.float32)
                .expand(layer.num_local_experts)
            )
        else:
            w13_input_global_scale = layer.w13_input_global_scale.min(dim=1).values.to(
                torch.float32
            )
        layer.g1_alphas = torch.nn.Parameter(
            ((1 / w13_input_global_scale) * layer.w13_weight_scale_2),
            requires_grad=False,
        )

        layer.w13_input_scale_quant = torch.nn.Parameter(
            (w13_input_global_scale), requires_grad=False
        )

        # w2
        if self.use_flashinfer_trtllm:
            w2_input_global_scale = (
                layer.w2_input_global_scale.min()
                .to(torch.float32)
                .expand(layer.num_local_experts)
            )
        else:
            w2_input_global_scale = layer.w2_input_global_scale

        layer.g2_alphas = torch.nn.Parameter(
            ((1 / w2_input_global_scale) * layer.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )

        layer.w2_input_scale_quant = torch.nn.Parameter(
            (w2_input_global_scale), requires_grad=False
        )

        # TensorRT-LLM specific processing
        if self.use_flashinfer_trtllm:
            # Prepare static weights for TRT-LLM kernel
            (
                gemm1_weights_fp4_shuffled,
                gemm1_scales_fp4_shuffled,
                gemm2_weights_fp4_shuffled,
                gemm2_scales_fp4_shuffled,
            ) = prepare_static_weights_for_trtllm_fp4_moe(
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                layer.w2_weight.size(-2),  # hidden_size
                layer.w13_weight.size(-2) // 2,  # intermediate_size
                layer.w13_weight.size(0),  # num_experts
            )
            logger.debug("Finished shuffling weights for TRT-LLM MOE")

            replace_parameter(layer, "w13_weight", gemm1_weights_fp4_shuffled)
            replace_parameter(layer, "w2_weight", gemm2_weights_fp4_shuffled)
            replace_parameter(layer, "w13_weight_scale", gemm1_scales_fp4_shuffled)
            replace_parameter(layer, "w2_weight_scale", gemm2_scales_fp4_shuffled)

            # Additional parameter needed for TRT-LLM
            layer.g1_scale_c = torch.nn.Parameter(
                (layer.w2_input_scale_quant * layer.g1_alphas).to(torch.float32),
                requires_grad=False,
            )

            # Pre-sliced [:1] views to avoid re-slicing every decode step
            # in apply_weights. Both fp4_quantize variants (flashinfer
            # cute-dsl and sgl-native) require shape [1] global scales.
            layer.w13_input_scale_quant_slice = layer.w13_input_scale_quant[:1]
        else:
            # swizzle weight scales
            layer.w13_weight_scale = torch.nn.Parameter(
                swizzle_blockscale(layer.w13_weight_scale), requires_grad=False
            )

            layer.w2_weight_scale = torch.nn.Parameter(
                swizzle_blockscale(layer.w2_weight_scale), requires_grad=False
            )

            layer.cutlass_moe_params = CutlassMoEParams(
                CutlassMoEType.BlockscaledFP4,
                layer.w13_weight.device,
                num_experts=layer.num_experts,
                intermediate_size_per_partition=layer.w2_weight.shape[2] * 2,
                hidden_size=layer.w13_weight.shape[2] * 2,
            )

        # When DeepEP is enabled, wire the dispatcher to either NVFP4-quantize
        # activations in-flight (saves bandwidth) or hand bf16 hidden_states to
        # the per-expert masked GEMM. The flashinfer cutedsl moe masked kernel
        # handles both shapes — see ``apply_weights`` DeepEP-LL branch.
        if self._is_deepep and hasattr(layer, "dispatcher"):
            if _MOE_NVFP4_DISPATCH:
                # Matches modelopt_quant.py:1875 — DeepEP pre-quantizes
                # activations using ``layer.w13_input_scale_quant``.
                layer.dispatcher.set_quant_config(
                    {"input_global_scale": layer.w13_input_scale_quant}
                )
            else:
                # bf16 dispatch; per-expert kernel quantizes internally via
                # ``scaled_fp4_grouped_quantize`` (see flashinfer_cutedsl_moe.py).
                layer.dispatcher.set_quant_config(
                    {"dispatcher_output_dtype": "bf16"}
                )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        # DeepEP a2a branch — the dispatch_output is a DeepEPLLDispatchOutput
        # (or DeepEPNormalDispatchOutput) NamedTuple, *not* a StandardDispatchOutput
        # with `.topk_output`. Routing is already complete on the host side; we
        # only need the per-expert NVFP4 BMM + activation + downproj. Delegate
        # to the dedicated handler so we don't trip on `dispatch_output.topk_output`
        # below for the standard path.
        if dispatch_output.format.is_deepep_ll():
            return self._apply_weights_deepep_ll(layer, dispatch_output)
        if dispatch_output.format.is_deepep_normal():
            # The masked-GEMM path requires DeepEP-LL's `[num_local_experts, m, k]`
            # layout. Normal-mode DeepEP returns a flat `[N, k]` packed tensor
            # plus `num_recv_tokens_per_expert` (List[int]) — supporting it
            # would need a custom grouped-GEMM with cumulative offsets which
            # we do not have here. Surface a clear error so the operator can
            # switch to LL mode.
            raise NotImplementedError(
                "DeepEP normal mode is not supported by "
                "CompressedTensorsW4A4Nvfp4MoE; use --deepep-mode low_latency."
            )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        # Zero-rows short-circuit (DP-attention forward_idle case).
        # When this rank has no tokens to process, the upstream topk path
        # generates a StandardTopKOutput (no topk_config) instead of the
        # BypassedTopKOutput the trtllm fast path expects. Return an empty
        # combine_input with the right output shape — collectives still
        # work because they read sizes from the live tensor.
        if x.shape[0] == 0:
            return StandardCombineInput(
                hidden_states=x.new_empty(x.shape, dtype=x.dtype),
            )

        if self.use_flashinfer_trtllm:
            from flashinfer import trtllm_fp4_block_scale_moe

            router_logits = topk_output.router_logits
            topk_config = topk_output.topk_config

            # global_scale must be shape [1] (strict in cute-dsl backend).
            # Pre-sliced [:1] view of w13_input_scale_quant is cached on
            # the layer at load time (process_weights_after_loading) so
            # we don't repeat the slice op every step.
            stash = (
                getattr(x, "_sglang_pre_quantized_fp4", None)
                if envs.SGLANG_USE_SGL_NVFP4_FUSED_RMSNORM.get()
                else None
            )
            if stash is not None:
                # Iter3 deploy wire: upstream prepare_mlp hook fused the
                # post-attention RMSNorm with this kernel's input
                # quantize. (hs_fp4, hs_scale) are already on device with
                # matching layouts; skip the redundant fp4_quantize call.
                hs_fp4, hs_scale = stash
                # Defensive: clear the stash so a follow-on caller (e.g.
                # the moe_cp post-allgather re-run) doesn't accidentally
                # consume the same buffer twice.
                try:
                    del x._sglang_pre_quantized_fp4
                except AttributeError:
                    pass
            elif _USE_SGL_NVFP4_QUANT_LINEAR:
                # Sglang-native non-swizzled NVFP4 quantize. Output layouts
                # match flashinfer's is_sf_swizzled_layout=False path.
                from sglang.jit_kernel.nvfp4 import scaled_fp4_quant_linear

                hs_fp4, hs_scale = scaled_fp4_quant_linear(
                    x, layer.w13_input_scale_quant_slice
                )
            else:
                from sglang.srt.layers.quantization.fp4_utils import (
                    fp4_quantize,
                )

                hs_fp4_bytes, hs_sf_bytes = fp4_quantize(
                    x,
                    layer.w13_input_scale_quant_slice,
                    self.group_size,  # sf_vec_size
                    False,  # use_ue8m0
                    False,  # is_sf_swizzled_layout
                )
                hs_fp4 = hs_fp4_bytes.reshape(x.shape[0], x.shape[1] // 2)
                hs_scale = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(
                    *hs_sf_bytes.shape[:-1], -1
                )

            correction_bias = (
                None
                if topk_config.correction_bias is None
                else topk_config.correction_bias.to(x.dtype)
            )

            assert layer.routing_method_type is not None

            # DeepSeekV3 style routing requires float32 router logits
            if layer.routing_method_type == RoutingMethodType.DeepSeekV3:
                router_logits = router_logits.to(torch.float32)

            routed_scaling_factor = self.moe_runner_config.routed_scaling_factor
            routed_scaling_factor = (
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            )

            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                num_tokens = hs_fp4.shape[0]
                hidden_size = (
                    hs_fp4.shape[-1] * 2
                    if hs_fp4.dtype == torch.uint8
                    else hs_fp4.shape[-1]
                )
                symm_output = torch.empty(
                    num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_fp4.device
                )

            output = trtllm_fp4_block_scale_moe(
                routing_logits=router_logits,
                routing_bias=correction_bias,
                hidden_states=hs_fp4,
                hidden_states_scale=hs_scale,
                gemm1_weights=layer.w13_weight,
                gemm1_weights_scale=layer.w13_weight_scale.view(torch.float8_e4m3fn),
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=layer.w2_weight,
                gemm2_weights_scale=layer.w2_weight_scale.view(torch.float8_e4m3fn),
                gemm2_bias=None,
                output1_scale_scalar=layer.g1_scale_c,
                output1_scale_gate_scalar=layer.g1_alphas,
                output2_scale_scalar=layer.g2_alphas,
                num_experts=layer.num_experts,
                top_k=topk_config.top_k,
                n_group=topk_config.num_expert_group,
                topk_group=topk_config.topk_group,
                intermediate_size=layer.intermediate_size_per_partition,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                local_num_experts=layer.num_local_experts,
                routed_scaling_factor=routed_scaling_factor,
                routing_method_type=layer.routing_method_type,
                do_finalize=True,
                tune_max_num_tokens=next_power_of_2(hs_fp4.shape[0]),
                output=symm_output,
            )[0]
        else:
            from sglang.srt.layers.moe.cutlass_moe import cutlass_moe_fp4

            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

            output = cutlass_moe_fp4(
                a=x,
                a1_gscale=layer.w13_input_scale_quant,
                w1_fp4=layer.w13_weight,
                w1_blockscale=layer.w13_weight_scale,
                w1_alphas=layer.g1_alphas,
                a2_gscale=layer.w2_input_scale_quant,
                w2_fp4=layer.w2_weight,
                w2_blockscale=layer.w2_weight_scale,
                w2_alphas=layer.g2_alphas,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                params=layer.cutlass_moe_params,
                apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            ).to(x.dtype)

        return StandardCombineInput(hidden_states=output)

    def _apply_weights_deepep_ll(
        self,
        layer: torch.nn.Module,
        dispatch_output,
    ):
        """Per-local-expert NVFP4 MoE for the DeepEP low-latency a2a backend.

        DeepEP-LL hands the rank a buffer pre-bucketed into local experts:
            hidden_states: [num_local_experts, max_m, k]  (bf16, or uint8 FP4 when
                NVFP4 dispatch is enabled — then `hidden_states_scale` is e4m3)
            masked_m: [num_local_experts] int32 — actual token count per expert
        Routing (`topk_ids` / `topk_weights`) is already computed; the combine
        step on the way out re-aggregates per-token via DeepEP.

        We pipe this straight into ``flashinfer_cutedsl_moe_masked``: it runs
        two grouped masked NVFP4 GEMMs (W13 + SiLU·Mul + W2) with the
        layer's swizzled blockscales and per-expert alphas, exactly the
        weights the cutlass branch of ``process_weights_after_loading`` lays
        out. Returns a ``DeepEPLLCombineInput`` so the dispatcher combine
        path is a drop-in.
        """
        from sglang.srt.layers.moe.flashinfer_cutedsl_moe import (
            flashinfer_cutedsl_moe_masked,
        )
        from sglang.srt.layers.moe.token_dispatcher import DeepEPLLCombineInput

        assert not self.use_flashinfer_trtllm, (
            "DeepEP path requires cutlass weight layout (swizzled blockscales). "
            "use_flashinfer_trtllm should have been forced False in __init__."
        )

        hidden_states = dispatch_output.hidden_states
        hidden_states_scale = dispatch_output.hidden_states_scale
        masked_m = dispatch_output.masked_m

        # When DeepEP pre-quantized to NVFP4, ``hidden_states_scale`` is non-None
        # and ``input_global_scale`` should be None (kernel skips its internal
        # quantize). When DeepEP dispatched bf16, ``hidden_states_scale`` is None
        # and we hand the kernel the per-expert global scale so it can call
        # ``scaled_fp4_grouped_quantize`` internally.
        use_nvfp4_dispatch = hidden_states_scale is not None
        input_global_scale = (
            None if use_nvfp4_dispatch else layer.w13_input_scale_quant
        )

        # The kernel reads ``hidden_states[1].view(float8_e4m3fn)`` directly —
        # if DeepEP returned an int32-packed UE8M0 layout (FP8-DeepGEMM
        # specific), .view would mis-stride the tensor. Same guard as
        # modelopt's deepep cutedsl path.
        if (
            use_nvfp4_dispatch
            and hidden_states_scale.element_size() != 1
            and hidden_states_scale.stride(-1) != 1
        ):
            raise AssertionError(
                f"NVFP4 dispatch scale has stride(-1)={hidden_states_scale.stride(-1)}, "
                f"dtype={hidden_states_scale.dtype}; "
                ".view(float8_e4m3fn) requires stride(-1)==1. "
                "Try SGLANG_MOE_NVFP4_DISPATCH=0 or check DeepEP version."
            )

        output = flashinfer_cutedsl_moe_masked(
            hidden_states=(hidden_states, hidden_states_scale),
            input_global_scale=input_global_scale,
            w1=layer.w13_weight,
            w1_blockscale=layer.w13_weight_scale,
            w1_alpha=layer.g1_alphas,
            w2=layer.w2_weight,
            a2_global_scale=layer.w2_input_scale_quant,
            w2_blockscale=layer.w2_weight_scale,
            w2_alpha=layer.g2_alphas,
            masked_m=masked_m,
        )

        return DeepEPLLCombineInput(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )
