from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_assign_extend_cache_locs_module(
    req_pool_dtype: torch.dtype, token_dtype: torch.dtype, offset_dtype: torch.dtype
) -> Module:
    args = make_cpp_args(req_pool_dtype, token_dtype, offset_dtype)
    return load_jit(
        "assign_extend_cache_locs",
        *args,
        cuda_files=["speculative/assign_extend_cache_locs.cuh"],
        cuda_wrappers=[
            (
                "assign_extend_cache_locs",
                f"AssignExtendCacheLocs<{args}>::run",
            )
        ],
    )


def assign_extend_cache_locs_cuda(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> None:
    module = _jit_assign_extend_cache_locs_module(
        req_pool_indices.dtype, req_to_token.dtype, start_offset.dtype
    )
    module.assign_extend_cache_locs(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
    )
