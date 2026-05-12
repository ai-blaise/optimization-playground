"""NVFP4/UE8M0 quantization kernels for the NSA Indexer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.nvfp4 import _nvfp4_arch_env, _nvfp4_cuda_flags
from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

_cached_module = None
_cached_key = None


@cache_once
def _jit_nvfp4_indexer_module(
    key_dtype: torch.dtype, indices_dtype: torch.dtype, page_size: int
) -> Module:
    with _nvfp4_arch_env():
        args = make_cpp_args(
            key_dtype, indices_dtype, page_size, is_arch_support_pdl()
        )
        return load_jit(
            "nvfp4_indexer_quant",
            *args,
            cuda_files=["nsa/nvfp4_indexer_quant.cuh"],
            cuda_wrappers=[
                (
                    "fused_store_index_k_cache_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::store_index_k",
                ),
                (
                    "quantize_indexer_q_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::quantize_q",
                ),
            ],
            extra_cuda_cflags=_nvfp4_cuda_flags(),
            extra_dependencies=["cutlass"],
        )


@cache_once
def can_use_nsa_nvfp4_indexer(
    key_dtype: torch.dtype, indices_dtype: torch.dtype, page_size: int
) -> bool:
    try:
        _jit_nvfp4_indexer_module(key_dtype, indices_dtype, page_size)
        return True
    except Exception as e:
        logger.warning("Failed to load NSA NVFP4 indexer JIT kernel: %s", e)
        return False


def _get_module_fast(key_dtype, indices_dtype, page_size):
    global _cached_key, _cached_module
    cache_key = (key_dtype, indices_dtype, page_size)
    if _cached_key == cache_key:
        return _cached_module
    module = _jit_nvfp4_indexer_module(key_dtype, indices_dtype, page_size)
    _cached_key = cache_key
    _cached_module = module
    return module


@debug_kernel_api
def fused_store_index_k_cache_nvfp4(
    key: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    out_cache_loc: torch.Tensor,
    page_size: int = 64,
) -> None:
    if key.dim() != 2:
        key = key.reshape(-1, key.shape[-1])
    if not key.is_contiguous():
        key = key.contiguous()
    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()

    _get_module_fast(
        key.dtype, out_cache_loc.dtype, page_size
    ).fused_store_index_k_cache_nvfp4(key, index_k_with_scale, out_cache_loc)


@debug_kernel_api
def quantize_indexer_q_nvfp4(
    query: torch.Tensor, indices_dtype: torch.dtype = torch.int64, page_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = query.shape
    if query.shape[-1] != 128:
        raise ValueError(f"NSA NVFP4 indexer Q expects head_dim=128, got {query.shape}.")
    query = query.reshape(-1, original_shape[-1])
    if not query.is_contiguous():
        query = query.contiguous()

    values = torch.empty(
        (query.shape[0], original_shape[-1] // 2),
        dtype=torch.uint8,
        device=query.device,
    )
    scales = torch.empty((query.shape[0],), dtype=torch.int32, device=query.device)
    _get_module_fast(query.dtype, indices_dtype, page_size).quantize_indexer_q_nvfp4(
        query, values, scales
    )
    return (
        values.view(*original_shape[:-1], original_shape[-1] // 2),
        scales.view(*original_shape[:-1]),
    )
