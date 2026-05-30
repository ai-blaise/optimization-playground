"""Python bindings for the 2-bit HIGGS dense MLA decode CUDA kernels.

Two entry points are exposed (iter3 dropped the single-pass kernels;
see ``higgs_dense_2bit_mla_decode.cuh`` for the rationale):

* :func:`higgs_dense_2bit_mla_rotate_query` — pre-rotate ``q_nope``
  into a ``float32`` ``q_rotated`` buffer via FWHT_512. Called once
  before split-K decode.
* :func:`higgs_dense_2bit_mla_decode_split` — split-K decode (grid =
  ``(num_rows, num_heads, num_splits)``) for the EDEN2-16 codebook
  variant. Each split processes ``ceil(topk / num_splits)`` slots; a
  merge kernel combines the partial ``(m, l, acc)`` tuples and runs
  the inverse FWHT_512.
* :func:`higgs_dense_2bit_mla_decode_saw_scalar2_split` — split-K
  decode for the SAW scalar2 codec variant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_higgs_dense_2bit_mla_decode_module() -> "Module":
    return load_jit(
        "higgs_dense_2bit_mla_decode",
        cuda_files=["quantization/higgs_dense_2bit_mla_decode.cuh"],
        extra_dependencies=["cutlass"],
        cuda_wrappers=[
            (
                "higgs_dense_2bit_mla_rotate_query",
                "higgs_dense_2bit_mla_detail::"
                "HiggsDense2BitMLARotateQueryKernel::run",
            ),
            (
                "higgs_dense_2bit_mla_decode_split",
                "higgs_dense_2bit_mla_detail::"
                "HiggsDense2BitMLADecodeSplitKernel::run",
            ),
            (
                "higgs_dense_2bit_mla_decode_saw_scalar2_split",
                "higgs_dense_2bit_mla_detail::"
                "HiggsDense2BitMLADecodeSawScalar2SplitKernel::run",
            ),
        ],
    )


@debug_kernel_api
def higgs_dense_2bit_mla_decode_saw_scalar2_split(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    mid: torch.Tensor,
    out: torch.Tensor,
    codebook: torch.Tensor,
    sm_scale: float,
) -> None:
    """Split-K MLA decode for the SAW scalar2 HIGGS slot variant."""
    assert q_nope.is_cuda
    assert q_rope.is_cuda
    assert compressed.is_cuda
    assert page_table.is_cuda
    assert mid.is_cuda
    assert out.is_cuda
    assert q_nope.dtype == torch.bfloat16
    assert q_rope.dtype == torch.bfloat16
    assert compressed.dtype == torch.uint8
    assert page_table.dtype == torch.int32
    assert mid.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert codebook.dtype == torch.float32

    module = _jit_higgs_dense_2bit_mla_decode_module()
    module.higgs_dense_2bit_mla_decode_saw_scalar2_split(
        q_nope.contiguous(),
        q_rope.contiguous(),
        compressed.contiguous(),
        page_table.contiguous(),
        mid,
        out,
        codebook.contiguous(),
        float(sm_scale),
    )


@debug_kernel_api
def higgs_dense_2bit_mla_rotate_query(
    q_nope: torch.Tensor,
    q_rotated: torch.Tensor,
) -> None:
    """Pre-rotate ``q_nope`` into ``q_rotated`` via FWHT_512.

    Used as the first stage of the split-K decode. The result is the
    scaled forward Hadamard rotation of ``q_nope``; subsequent
    :func:`higgs_dense_2bit_mla_decode_split` calls read from this
    buffer and skip the per-block FWHT.

    Args:
      q_nope: ``(num_rows, num_heads, 512)`` ``bfloat16`` input.
      q_rotated: ``(num_rows, num_heads, 512)`` ``float32`` output.
    """
    assert q_nope.is_cuda
    assert q_rotated.is_cuda
    assert q_nope.dtype == torch.bfloat16
    assert q_rotated.dtype == torch.float32

    module = _jit_higgs_dense_2bit_mla_decode_module()
    module.higgs_dense_2bit_mla_rotate_query(
        q_nope.contiguous(),
        q_rotated,
    )


@debug_kernel_api
def higgs_dense_2bit_mla_decode_split(
    q_rotated: torch.Tensor,
    q_rope: torch.Tensor,
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    mid: torch.Tensor,
    out: torch.Tensor,
    codebook: torch.Tensor,
    sm_scale: float,
) -> None:
    """Split-K fused MLA decode against a 2-bit HIGGS packed KV cache.

    Same math as :func:`higgs_dense_2bit_mla_decode` but with the topk
    loop sharded across ``num_splits = mid.shape[2]`` blocks per
    ``(row, head)``. Each split produces a partial ``(m, l, acc)``
    written to ``mid``; a merge kernel combines partials and writes
    the final BF16 output to ``out``.

    Args:
      q_rotated: ``(num_rows, num_heads, 512)`` ``float32`` rotated
        query produced by :func:`higgs_dense_2bit_mla_rotate_query`.
      q_rope: ``(num_rows, num_heads, 64)`` ``bfloat16`` query rope.
      compressed: ``(num_slots, 1, 272)`` ``uint8`` packed KV slots
        (iter4 #16: 258 B payload + 14 B 16-align pad).
      page_table: ``(num_rows, topk)`` ``int32`` slot indices.
      mid: ``(num_rows, num_heads, num_splits, 514)`` ``float32``
        scratch for per-split ``(m, l, acc[0..511])`` partials.
      out: ``(num_rows, num_heads, 512)`` ``bfloat16`` destination.
      codebook: ``(16, 2)`` ``float32`` EDEN2-16 lattice.
      sm_scale: softmax scale factor.
    """
    assert q_rotated.is_cuda
    assert q_rope.is_cuda
    assert compressed.is_cuda
    assert page_table.is_cuda
    assert mid.is_cuda
    assert out.is_cuda
    assert q_rotated.dtype == torch.float32
    assert q_rope.dtype == torch.bfloat16
    assert compressed.dtype == torch.uint8
    assert page_table.dtype == torch.int32
    assert mid.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert codebook.dtype == torch.float32

    module = _jit_higgs_dense_2bit_mla_decode_module()
    module.higgs_dense_2bit_mla_decode_split(
        q_rotated,
        q_rope.contiguous(),
        compressed.contiguous(),
        page_table.contiguous(),
        mid,
        out,
        codebook.contiguous(),
        float(sm_scale),
    )
