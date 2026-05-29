"""CUDA binding for HIGGS 2-bit MHA/GQA KV materialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_higgs_mha_2bit_kv_module() -> "Module":
    return load_jit(
        "higgs_mha_2bit_kv",
        cuda_files=["quantization/higgs_mha_2bit_kv.cuh"],
        extra_dependencies=["cutlass"],
        cuda_wrappers=[
            (
                "dequantize_higgs_mha_2bit",
                "higgs_mha_2bit_kv_detail::HiggsMHA2BitDequantKernel::run",
            ),
        ],
    )


@debug_kernel_api
def dequantize_higgs_mha_2bit(
    packed: torch.Tensor,
    out: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    """Materialize HIGGS 2-bit MHA/GQA slots into BF16 rows."""
    assert packed.is_cuda
    assert out.is_cuda
    assert codebook.is_cuda
    assert packed.dtype == torch.uint8
    assert out.dtype == torch.bfloat16
    assert codebook.dtype == torch.float32

    module = _jit_higgs_mha_2bit_kv_module()
    module.dequantize_higgs_mha_2bit(
        packed.contiguous(),
        out,
        codebook.contiguous(),
    )
