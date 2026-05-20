"""Fused bf16->fp8 quantize-and-scatter JIT kernel for DSA IndexCache.

JIT-compiled via TVM FFI. Optimized Python dispatch path caches the
TVM module reference to minimize per-call overhead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

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

# Module-level cache: avoids dict-lookup overhead of cache_once on every call.
# Populated on first successful JIT compilation.
_cached_module = None
_cached_key = None


@cache_once
def _jit_dsa_fused_store_module(
    key_dtype: torch.dtype, indices_dtype: torch.dtype, page_size: int
) -> Module:
    """
    Build a JIT module that exposes:
      module.fused_store_index_k_cache(input_bf16, index_k_with_scale_u8, loc_i64)
    """
    args = make_cpp_args(key_dtype, indices_dtype, page_size, is_arch_support_pdl())
    return load_jit(
        "fused_store_index_k_cache",
        *args,
        cuda_files=["dsa/fused_store_index_cache.cuh"],
        cuda_wrappers=[
            (
                "fused_store_index_k_cache",
                # - Float  = bf16_t (sgl_kernel/type.cuh)
                # - IndicesT = int64_t (out_cache_loc is int64 in SGLang SetKAndS)
                # - kPageSize = 64 (CUDA DSA)
                f"FusedStoreCacheIndexerKernel<{args}>::run",
            )
        ],
    )


@cache_once
def can_use_dsa_fused_store(
    key_dtype: torch.dtype, indices_dtype: torch.dtype, page_size: int
) -> bool:
    try:
        _jit_dsa_fused_store_module(key_dtype, indices_dtype, page_size)
        return True
    except Exception as e:
        logger.warning(f"Failed to load dsa fused store JIT kernel: {e}")
        return False


can_use_nsa_fused_store = can_use_dsa_fused_store


def _get_module_fast(key_dtype, indices_dtype, page_size):
    """Fast module lookup: uses module-level variable cache to skip dict lookup."""
    global _cached_module, _cached_key
    cache_key = (key_dtype, indices_dtype, page_size)
    if _cached_key == cache_key:
        return _cached_module
    mod = _jit_dsa_fused_store_module(key_dtype, indices_dtype, page_size)
    _cached_module = mod
    _cached_key = cache_key
    return mod


@debug_kernel_api
def fused_store_index_k_cache(
    key: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    out_cache_loc: torch.Tensor,
    page_size: int = 64,
) -> None:
    """
    Fused: quantize bf16 key (N,128) -> fp8 + fp32 scale and write into
    DSATokenToKVPool.index_k_with_scale_buffer.

    key:                (num_tokens, 128) bf16 (or reshapeable to it)
    index_k_with_scale: (num_pages, page_size*(128+4)) uint8
    out_cache_loc:      (num_tokens,) int64 token indices in TokenToKVPool
    """
    # Fast path: skip assertions when tensors are already valid.
    # The CUDA kernel's TensorMatcher performs the authoritative validation.
    if key.dim() != 2:
        key = key.view(-1, key.shape[-1])
    if not key.is_contiguous():
        key = key.contiguous()
    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()

    _get_module_fast(key.dtype, out_cache_loc.dtype, page_size).fused_store_index_k_cache(
        key, index_k_with_scale, out_cache_loc
    )
