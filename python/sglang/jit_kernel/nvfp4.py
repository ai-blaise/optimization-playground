from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch
from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_FLOAT4_E2M1_MAX = 6.0
_FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _nvfp4_cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DFLASHINFER_ENABLE_F16",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-DCUTLASS_TEST_LEVEL=0",
        "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "--expt-extended-lambda",
    ]


def _nvfp4_arch_env():
    if not torch.cuda.is_available():
        raise RuntimeError("NVFP4 JIT kernels require CUDA.")
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError(
            f"NVFP4 JIT kernels require compute capability >= 10.0, got {major}.{minor}."
        )
    # NVFP4 kernels use architecture-family-specific instructions and must be
    # compiled for `sm_*a` targets (e.g. sm_100a), not plain sm_100.
    # JIT compilation targets only the current device, unlike AOT fat-binaries;
    # adding extra architectures here would clash with the single SGL_CUDA_ARCH
    # value injected by load_jit().
    return override_jit_cuda_arch(major, minor, suffix="a")


@torch.compiler.disable
def prewarm_nvfp4_jit_modules(
    *,
    include_expert_quant: bool = False,
    include_blockwise_moe: bool = False,
    include_quant_linear: bool = False,
    include_fused_rmsnorm_quant: bool = False,
) -> None:
    """Materialize NVFP4 JIT modules before torch.compile traces the model."""
    _jit_nvfp4_quant_module()
    _jit_nvfp4_scaled_mm_module()
    if include_expert_quant:
        _jit_nvfp4_expert_quant_module()
    if include_blockwise_moe:
        _jit_nvfp4_blockwise_moe_module()
    if include_quant_linear:
        _jit_nvfp4_quant_linear_module()
    if include_fused_rmsnorm_quant:
        _jit_nvfp4_fused_rmsnorm_quant_module()


@cache_once
def _jit_nvfp4_quant_module() -> Module:
    with _nvfp4_arch_env():
        return load_jit(
            "nvfp4_quant",
            cuda_files=[
                "gemm/nvfp4/nvfp4_quant_kernels.cuh",
            ],
            cuda_wrappers=[
                ("scaled_fp4_quant", "scaled_fp4_quant_sm100a_sm120a"),
            ],
            extra_cuda_cflags=_nvfp4_cuda_flags(),
            extra_dependencies=["cutlass"],
        )


@cache_once
def _jit_nvfp4_quant_linear_module() -> Module:
    """Non-swizzled (linear) layout NVFP4 quantize.

    Produces row-major [m, K/16] fp8_e4m3 scales (no tile-rotation),
    matching flashinfer.fp4_quantize(is_sf_swizzled_layout=False). Used as
    the activation pre-quantize for trtllm_fp4_block_scale_moe on the
    NVFP4 MoE production deploy.
    """
    with _nvfp4_arch_env():
        return load_jit(
            "nvfp4_quant_linear",
            cuda_files=[
                "gemm/nvfp4/nvfp4_quant_linear_kernels.cuh",
            ],
            cuda_wrappers=[
                (
                    "scaled_fp4_quant_linear",
                    "scaled_fp4_quant_linear_sm100a_sm120a",
                ),
            ],
            extra_cuda_cflags=_nvfp4_cuda_flags(),
            extra_dependencies=["cutlass"],
        )


@cache_once
def _jit_nvfp4_fused_rmsnorm_quant_module() -> Module:
    """Fused (residual-add + RMSNorm + linear-layout NVFP4 quantize).

    Iter2 primary vector for #15 NVFP4 MoE deploy: collapses the upstream
    `fused_add_rmsnorm` BF16 write+read of hidden_states with the
    downstream `scaled_fp4_quant_linear` BF16 read, eliminating
    ~1.8 MB/layer of HBM traffic on the non-allreduce-fusion path.

    Output FP4 + linear-SF layout matches `scaled_fp4_quant_linear`
    exactly, so trtllm_fp4_block_scale_moe consumes the result without
    layout adjustments. Iter2 secondary vector wires `enable_pdl=True`
    so the kernel can dispatch in the shadow of the preceding store.
    """
    with _nvfp4_arch_env():
        return load_jit(
            "nvfp4_fused_rmsnorm_quant",
            cuda_files=[
                "gemm/nvfp4/nvfp4_fused_rmsnorm_quant_kernels.cuh",
            ],
            cuda_wrappers=[
                (
                    "fused_rmsnorm_scaled_fp4_quant_linear",
                    "fused_rmsnorm_scaled_fp4_quant_linear_sm100a_sm120a",
                ),
                (
                    "fused_rmsnorm_only_to_fp4_linear",
                    "fused_rmsnorm_only_to_fp4_linear_sm100a_sm120a",
                ),
            ],
            extra_cuda_cflags=_nvfp4_cuda_flags(),
            extra_dependencies=["cutlass"],
        )


@cache_once
def _jit_nvfp4_expert_quant_module() -> Module:
    with _nvfp4_arch_env():
        return load_jit(
            "nvfp4_expert_quant",
            cuda_files=[
                "gemm/nvfp4/nvfp4_expert_quant.cuh",
            ],
            cuda_wrappers=[
                ("scaled_fp4_experts_quant", "scaled_fp4_experts_quant_sm100a"),
                (
                    "silu_and_mul_scaled_fp4_experts_quant",
                    "silu_and_mul_scaled_fp4_experts_quant_sm100a",
                ),
            ],
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_nvfp4_cuda_flags(),
        )


@cache_once
def _jit_nvfp4_scaled_mm_module() -> Module:
    with _nvfp4_arch_env():
        return load_jit(
            "nvfp4_scaled_mm",
            cuda_files=[
                "gemm/nvfp4/nvfp4_scaled_mm_kernels.cuh",
                "gemm/nvfp4/nvfp4_scaled_mm_entry.cuh",
            ],
            cuda_wrappers=[("cutlass_scaled_fp4_mm", "cutlass_scaled_fp4_mm")],
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_nvfp4_cuda_flags(),
        )


@cache_once
def _jit_nvfp4_blockwise_moe_module() -> Module:
    with _nvfp4_arch_env():
        return load_jit(
            "nvfp4_blockwise_moe",
            cuda_files=[
                "moe/nvfp4_blockwise_moe.cuh",
            ],
            cuda_wrappers=[
                ("cutlass_fp4_group_mm", "cutlass_fp4_group_mm_sm100a_sm120a")
            ],
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_nvfp4_cuda_flags(),
        )


@debug_kernel_api
def cutlass_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    module = _jit_nvfp4_scaled_mm_module()
    module.cutlass_scaled_fp4_mm(out, a, b, block_scale_a, block_scale_b, alpha)
    return out


@debug_kernel_api
def cutlass_fp4_group_mm(
    a_fp4: torch.Tensor,
    b_fp4: torch.Tensor,
    a_blockscale: torch.Tensor,
    b_blockscale: torch.Tensor,
    alphas: torch.Tensor,
    out_dtype: torch.dtype,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    m_topk = a_fp4.shape[0]
    n = b_fp4.shape[1]
    output = torch.empty((m_topk, n), device=a_fp4.device, dtype=out_dtype)
    num_experts = int(params["expert_offsets"].numel())
    device = a_fp4.device

    # Backward compatibility: older callers may not pass scratch tensors.
    a_ptrs = params.get(
        "a_ptrs", torch.empty((num_experts,), dtype=torch.int64, device=device)
    )
    b_ptrs = params.get(
        "b_ptrs", torch.empty((num_experts,), dtype=torch.int64, device=device)
    )
    out_ptrs = params.get(
        "out_ptrs", torch.empty((num_experts,), dtype=torch.int64, device=device)
    )
    a_scales_ptrs = params.get(
        "a_scales_ptrs", torch.empty((num_experts,), dtype=torch.int64, device=device)
    )
    b_scales_ptrs = params.get(
        "b_scales_ptrs", torch.empty((num_experts,), dtype=torch.int64, device=device)
    )
    alpha_ptrs = params.get(
        "alpha_ptrs", torch.empty((num_experts,), dtype=torch.int64, device=device)
    )
    layout_sfa = params.get(
        "layout_sfa", torch.empty((num_experts, 5), dtype=torch.int64, device=device)
    )
    layout_sfb = params.get(
        "layout_sfb", torch.empty((num_experts, 5), dtype=torch.int64, device=device)
    )

    _cutlass_fp4_group_mm_custom_op(
        output,
        a_fp4,
        b_fp4,
        a_blockscale,
        b_blockscale,
        alphas,
        params["ab_strides"],
        params["c_strides"],
        params["problem_sizes"],
        params["expert_offsets"],
        params["blockscale_offsets"],
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        alpha_ptrs,
        layout_sfa,
        layout_sfb,
    )
    return output


@register_custom_op(
    op_name="scaled_fp4_quant",
    mutates_args=["output", "output_scale"],
)
def _scaled_fp4_quant_custom_op(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    input_global_scale: torch.Tensor,
) -> None:
    module = _jit_nvfp4_quant_module()
    module.scaled_fp4_quant(output, input, output_scale, input_global_scale)


@debug_kernel_api
def scaled_fp4_quant(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize input tensor to FP4 and return packed FP4 tensor + swizzled scales."""
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_n = n // block_size
    rounded_n = ((scale_n + 4 - 1) // 4) * 4
    if rounded_n > scale_n:
        output_scale = torch.zeros(
            (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
        )
    else:
        output_scale = torch.empty(
            (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
        )

    _scaled_fp4_quant_custom_op(input, output, output_scale, input_global_scale)
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


@register_custom_op(
    op_name="scaled_fp4_quant_linear",
    mutates_args=["output", "output_scale"],
)
def _scaled_fp4_quant_linear_custom_op(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    input_global_scale: torch.Tensor,
) -> None:
    module = _jit_nvfp4_quant_linear_module()
    module.scaled_fp4_quant_linear(
        output, input, output_scale, input_global_scale
    )


@debug_kernel_api
def scaled_fp4_quant_linear(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize to NVFP4 with non-swizzled (linear) scale layout.

    Drop-in replacement for flashinfer.fp4_quantize(is_sf_swizzled_layout=False).

    Output shapes:
        packed FP4: [m, n // 2] uint8
        scales:     [m, n // 16] fp8_e4m3 (returned as int32-packed view that
                    callers can reinterpret with .view(torch.float8_e4m3fn))

    Hidden_size = 7168 (DeepSeek) satisfies the alignment constraint
    (n / 16 = 448 which is divisible by 4 for int32 packing). For other
    hidden sizes the kernel asserts at launch.

    The scale tensor is written row-major with stride n/16 columns. No
    padding (in contrast to the swizzled CUTLASS variant which pads M
    up to 128 and K up to 4). This matches the layout that
    trtllm_fp4_block_scale_moe consumes via its `bA16` block-scale config.
    """
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert (
        n // block_size
    ) % 4 == 0, (
        f"n/16={n // block_size} must be divisible by 4 for int32 packing."
    )
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)
    # Allocate scales as int32-packed (4 fp8 per int32). Linear layout has
    # no row/col padding.
    output_scale = torch.empty(
        (m, (n // block_size) // 4), device=device, dtype=torch.int32
    )

    _scaled_fp4_quant_linear_custom_op(
        input, output, output_scale, input_global_scale
    )
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


@register_custom_op(
    op_name="fused_rmsnorm_scaled_fp4_quant_linear",
    mutates_args=["input", "residual", "fp4_output", "sf_output"],
)
def _fused_rmsnorm_scaled_fp4_quant_linear_custom_op(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale: torch.Tensor,
    fp4_output: torch.Tensor,
    sf_output: torch.Tensor,
    eps: float,
    cast_x_before_out_mul: bool,
    enable_pdl: bool,
) -> None:
    module = _jit_nvfp4_fused_rmsnorm_quant_module()
    module.fused_rmsnorm_scaled_fp4_quant_linear(
        input,
        residual,
        weight,
        input_global_scale,
        fp4_output,
        sf_output,
        eps,
        cast_x_before_out_mul,
        enable_pdl,
    )


@debug_kernel_api
def fused_rmsnorm_scaled_fp4_quant_linear(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale: torch.Tensor,
    eps: float = 1e-6,
    *,
    cast_x_before_out_mul: bool = False,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused (residual-add + RMSNorm + linear-layout NVFP4 quantize).

    Equivalent to:

        fused_add_rmsnorm(input, residual, weight, eps,
                          cast_x_before_out_mul=cast_x_before_out_mul)
        fp4, sf = scaled_fp4_quant_linear(input, input_global_scale)

    but does the work in a single CTA-per-row kernel, keeping the
    post-rms BF16 hidden_states in registers instead of round-tripping
    through HBM. Mutates `residual` in place (writes `input + residual`).

    Iter2 primary vector for #15 NVFP4 MoE deploy. Production target
    hidden_size=7168, decode batches 1..256/rank, 58 MoE layers/step.
    On B200, at hidden=7168 the saved BF16 hidden_states write+read is
    ~28 KB/row × 2 × 58 layers / 8 TB/s ≈ 13 us/step at batch=1, plus
    launch-overhead amortization (one kernel instead of two) of
    ~3-5 us/layer.

    Output FP4 + linear-SF layout is identical to
    `scaled_fp4_quant_linear`, so trtllm_fp4_block_scale_moe consumes
    the result unchanged.

    Args:
        input:        [m, n] fp16/bf16 — pre-norm hidden states (read-only).
        residual:     [m, n] same dtype as input — accumulator. Written
                      back as (input + residual) in BF16.
        weight:       [n] same dtype as input — RMSNorm weight.
        input_global_scale: [1] fp32 — passed to the FP4 quantize stage.
        eps:          RMSNorm epsilon.
        cast_x_before_out_mul: HF parity — round to dtype before
                      multiplying by weight. Default False (Llama-style).
        enable_pdl:   Iter2 secondary vector — request programmatic
                      stream serialization so the kernel can start in
                      the shadow of a preceding global store.

    Returns:
        fp4_output:  [m, n//2] uint8 packed E2M1.
        sf_output:   [m, n//16] fp8_e4m3 (int32-packed under the hood).
    """
    assert (
        input.ndim == 2 and residual.ndim == 2
    ), f"input/residual must be 2D, got input={input.shape} residual={residual.shape}"
    m, n = input.shape
    assert residual.shape == input.shape, "residual must match input shape"
    assert weight.ndim == 1 and weight.shape[0] == n, "weight must be [n]"
    block_size = 16
    assert n % block_size == 0, f"last dim must be multiple of 16, got {n}"
    assert (
        n // block_size
    ) % 4 == 0, (
        f"n/16={n // block_size} must be divisible by 4 for int32-packed SF"
    )
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype must be fp16 or bf16, got {input.dtype}"
    assert weight.dtype == input.dtype, "weight dtype must match input"
    assert residual.dtype == input.dtype, "residual dtype must match input"
    assert input_global_scale.numel() == 1, "input_global_scale must be scalar"

    device = input.device
    fp4_output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)
    sf_output = torch.empty(
        (m, (n // block_size) // 4), device=device, dtype=torch.int32
    )

    _fused_rmsnorm_scaled_fp4_quant_linear_custom_op(
        input,
        residual,
        weight,
        input_global_scale,
        fp4_output,
        sf_output,
        float(eps),
        bool(cast_x_before_out_mul),
        bool(enable_pdl),
    )
    sf_output = sf_output.view(torch.float8_e4m3fn)
    return fp4_output, sf_output


@register_custom_op(
    op_name="fused_rmsnorm_only_to_fp4_linear",
    mutates_args=["y_output", "fp4_output", "sf_output"],
)
def _fused_rmsnorm_only_to_fp4_linear_custom_op(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale: torch.Tensor,
    y_output: torch.Tensor,
    fp4_output: torch.Tensor,
    sf_output: torch.Tensor,
    eps: float,
    cast_x_before_out_mul: bool,
    enable_pdl: bool,
) -> None:
    module = _jit_nvfp4_fused_rmsnorm_quant_module()
    module.fused_rmsnorm_only_to_fp4_linear(
        input,
        weight,
        input_global_scale,
        y_output,
        fp4_output,
        sf_output,
        eps,
        cast_x_before_out_mul,
        enable_pdl,
    )


@debug_kernel_api
def fused_rmsnorm_only_to_fp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale: torch.Tensor,
    eps: float = 1e-6,
    *,
    cast_x_before_out_mul: bool = False,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused (RMSNorm + linear-layout NVFP4 quantize), residual-less.

    Iter4 PRIMARY vector for #15 NVFP4 MoE deploy. Equivalent to:

        y = rmsnorm(input, weight, eps,
                    cast_x_before_out_mul=cast_x_before_out_mul)
        fp4, sf = scaled_fp4_quant_linear(y, input_global_scale)

    but does the work in a single CTA-per-row kernel, fusing the
    BF16 post-norm hidden_states write+read between the two ops.
    `input` is NOT mutated; `y` is a fresh BF16 output (consumed by
    the gate/router/shared_experts), and `(fp4, sf)` is the side
    stash consumed by the NVFP4 MoE expert apply_weights via the iter3
    `_sglang_pre_quantized_fp4` hook in
    compressed_tensors_w4a4_nvfp4_moe.py.

    Target call site:
        `CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual`
        in python/sglang/srt/layers/communicator.py, the
        `not use_layer_norm_before_gather` branch where
        `hidden_states = layernorm(hidden_states)` is invoked after
        a DP-attn allgather+scatter. The residual has already been
        folded in at attn_tp_rank==0 (line 997) so a residual-less
        kernel is the right contract.

    Output FP4 + linear-SF layout is identical to
    `scaled_fp4_quant_linear`, so trtllm_fp4_block_scale_moe consumes
    the result unchanged.

    Args:
        input:        [m, n] fp16/bf16 — hidden_states (read-only).
        weight:       [n] same dtype as input — RMSNorm weight.
        input_global_scale: [1] fp32 — passed to the FP4 quantize stage.
        eps:          RMSNorm epsilon.
        cast_x_before_out_mul: HF parity — round to dtype before
                      multiplying by weight. Default False (Llama-style).
        enable_pdl:   Programmatic stream serialization — lets this
                      kernel start in the shadow of the preceding
                      global store (dp_scatter copy or the symmetric
                      allgather completion).

    Returns:
        y_output:    [m, n] same dtype as input — post-norm BF16
                     hidden_states.
        fp4_output:  [m, n//2] uint8 packed E2M1.
        sf_output:   [m, n//16] fp8_e4m3 (int32-packed under the hood).
    """
    assert input.ndim == 2, (
        f"input must be 2D, got input={input.shape}"
    )
    m, n = input.shape
    assert weight.ndim == 1 and weight.shape[0] == n, "weight must be [n]"
    block_size = 16
    assert n % block_size == 0, f"last dim must be multiple of 16, got {n}"
    assert (
        n // block_size
    ) % 4 == 0, (
        f"n/16={n // block_size} must be divisible by 4 for int32-packed SF"
    )
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype must be fp16 or bf16, got {input.dtype}"
    assert weight.dtype == input.dtype, "weight dtype must match input"
    assert input_global_scale.numel() == 1, "input_global_scale must be scalar"

    device = input.device
    y_output = torch.empty((m, n), device=device, dtype=input.dtype)
    fp4_output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)
    sf_output = torch.empty(
        (m, (n // block_size) // 4), device=device, dtype=torch.int32
    )

    _fused_rmsnorm_only_to_fp4_linear_custom_op(
        input,
        weight,
        input_global_scale,
        y_output,
        fp4_output,
        sf_output,
        float(eps),
        bool(cast_x_before_out_mul),
        bool(enable_pdl),
    )
    sf_output = sf_output.view(torch.float8_e4m3fn)
    return y_output, fp4_output, sf_output


def _shuffle_rows_torch(
    input_tensor: torch.Tensor,
    dst2src_map: torch.Tensor,
    output_tensor_shape: tuple[int, int],
) -> torch.Tensor:
    # Keep compatibility when sgl-kernel is slimmed and shuffle_rows may not be present.
    output = input_tensor.index_select(0, dst2src_map.to(dtype=torch.int64))
    return output.view(output_tensor_shape)


@register_custom_op(
    op_name="scaled_fp4_experts_quant",
    mutates_args=["output", "output_scales"],
)
def _scaled_fp4_experts_quant_custom_op(
    output: torch.Tensor,
    output_scales: torch.Tensor,
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
) -> None:
    module = _jit_nvfp4_expert_quant_module()
    module.scaled_fp4_experts_quant(
        output,
        output_scales,
        input_tensor,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
    )


@debug_kernel_api
def scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int,
    expert_map: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize packed MoE activations to NVFP4."""
    assert (
        input_tensor.ndim == 2
    ), f"input.ndim needs to be == 2, but got {input_tensor.ndim}."
    if expert_map is not None:
        m, k = input_tensor.shape
        output_tensor_shape = (m * topk, k)
        input_tensor = _shuffle_rows_torch(
            input_tensor, expert_map, output_tensor_shape
        )

    m_numtopk, k = input_tensor.shape
    max_tokens_per_expert = int(os.environ.get("MODELOPT_MAX_TOKENS_PER_EXPERT", 65536))
    assert m_numtopk <= max_tokens_per_expert * topk, (
        f"m_numtopk must be less than MAX_TOKENS_PER_EXPERT({max_tokens_per_expert})"
        f" for cutlass_moe_fp4, observed m_numtopk = {m_numtopk}. Use"
        " MODELOPT_MAX_TOKENS_PER_EXPERT to set this value."
    )
    scales_k = k // 16
    # output_scales is int32-packed FP8 scales, so second dim is in int32 units.
    padded_k_in_int32 = (scales_k + 3) // 4

    output = torch.empty(
        m_numtopk, k // 2, device=input_tensor.device, dtype=torch.uint8
    )
    if padded_k_in_int32 * 4 > scales_k:
        output_scales = torch.zeros(
            max_tokens_per_expert * topk,
            padded_k_in_int32,
            dtype=torch.int32,
            device=input_tensor.device,
        )
    else:
        output_scales = torch.empty(
            max_tokens_per_expert * topk,
            padded_k_in_int32,
            dtype=torch.int32,
            device=input_tensor.device,
        )

    _scaled_fp4_experts_quant_custom_op(
        output,
        output_scales,
        input_tensor,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
    )
    output_scales = output_scales.view(torch.float8_e4m3fn)
    return output, output_scales


@register_custom_op(
    op_name="scaled_fp4_grouped_quant",
    mutates_args=["output", "output_scales"],
)
def _scaled_fp4_grouped_quant_custom_op(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    output_scales: torch.Tensor,
    input_global_scale: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    l, m, k = input_tensor.shape
    del l, m
    module = _jit_nvfp4_expert_quant_module()
    module.silu_and_mul_scaled_fp4_experts_quant(
        output.view(-1, k // 2),
        output_scales.view(-1, output_scales.shape[-1]),
        input_tensor.view(-1, k),
        input_global_scale,
        mask,
        False,
    )


@debug_kernel_api
def scaled_fp4_grouped_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    mask: torch.Tensor,
):
    """Quantize grouped GEMM inputs to FP4 and return logical (m, k//2, l)."""
    device = input_tensor.device
    l, m, k = input_tensor.shape
    sf_vec_size = 16
    assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

    scale_k = k // sf_vec_size
    padded_k = (scale_k + (4 - 1)) // 4 * 4
    padded_k_int32 = padded_k // 4
    padded_m = (m + (128 - 1)) // 128 * 128
    output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)
    output_scales = torch.empty(
        l, padded_m, padded_k_int32, device=device, dtype=torch.int32
    )

    _scaled_fp4_grouped_quant_custom_op(
        input_tensor,
        output,
        output_scales,
        input_global_scale,
        mask,
    )

    output = output.permute(1, 2, 0)
    output_scales = output_scales.view(torch.float8_e4m3fn).view(
        l, padded_m // 128, padded_k // 4, 32, 4, 4
    )
    output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
    return output, output_scales


@register_custom_op(
    op_name="silu_and_mul_scaled_fp4_grouped_quant",
    mutates_args=["output", "output_scales"],
)
def _silu_and_mul_scaled_fp4_grouped_quant_custom_op(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    output_scales: torch.Tensor,
    input_global_scale: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    l, m, k_by_2 = input_tensor.shape
    del l, m
    module = _jit_nvfp4_expert_quant_module()
    module.silu_and_mul_scaled_fp4_experts_quant(
        output.view(-1, output.shape[-1]),
        output_scales.view(-1, output_scales.shape[-1]),
        input_tensor.view(-1, k_by_2),
        input_global_scale,
        mask,
        True,
    )


@debug_kernel_api
def silu_and_mul_scaled_fp4_grouped_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    mask: torch.Tensor,
):
    """Apply SiLU-and-mul then quantize grouped GEMM inputs to FP4."""
    device = input_tensor.device
    l, m, k_by_2 = input_tensor.shape
    k = k_by_2 // 2
    sf_vec_size = 16
    assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

    scale_k = k // sf_vec_size
    padded_k = (scale_k + (4 - 1)) // 4 * 4
    padded_k_int32 = padded_k // 4
    padded_m = (m + (128 - 1)) // 128 * 128
    output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)
    output_scales = torch.empty(
        l, padded_m, padded_k_int32, device=device, dtype=torch.int32
    )

    _silu_and_mul_scaled_fp4_grouped_quant_custom_op(
        input_tensor,
        output,
        output_scales,
        input_global_scale,
        mask,
    )

    output = output.permute(1, 2, 0)
    output_scales = output_scales.view(torch.float8_e4m3fn).view(
        l, padded_m // 128, padded_k // 4, 32, 4, 4
    )
    output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
    return output, output_scales


@register_custom_op(
    op_name="cutlass_fp4_group_mm",
    mutates_args=[
        "output",
        "a_ptrs",
        "b_ptrs",
        "out_ptrs",
        "a_scales_ptrs",
        "b_scales_ptrs",
        "alpha_ptrs",
        "layout_sfa",
        "layout_sfb",
    ],
)
def _cutlass_fp4_group_mm_custom_op(
    output: torch.Tensor,
    a_fp4: torch.Tensor,
    b_fp4: torch.Tensor,
    a_blockscale: torch.Tensor,
    b_blockscale: torch.Tensor,
    alphas: torch.Tensor,
    ab_strides: torch.Tensor,
    c_strides: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    a_ptrs: torch.Tensor,
    b_ptrs: torch.Tensor,
    out_ptrs: torch.Tensor,
    a_scales_ptrs: torch.Tensor,
    b_scales_ptrs: torch.Tensor,
    alpha_ptrs: torch.Tensor,
    layout_sfa: torch.Tensor,
    layout_sfb: torch.Tensor,
) -> None:
    module = _jit_nvfp4_blockwise_moe_module()
    module.cutlass_fp4_group_mm(
        output,
        a_fp4,
        b_fp4,
        a_blockscale,
        b_blockscale,
        alphas,
        ab_strides,
        c_strides,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        alpha_ptrs,
        layout_sfa,
        layout_sfb,
    )


def suggest_nvfp4_global_scale(x: torch.Tensor) -> torch.Tensor:
    """Utility for tests/benchmarks: return global scale used by NVFP4 quantization."""
    tensor_amax = torch.abs(x).max().to(torch.float32)
    return _FLOAT8_E4M3_MAX * _FLOAT4_E2M1_MAX / tensor_amax
