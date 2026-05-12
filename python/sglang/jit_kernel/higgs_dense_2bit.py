"""Python bindings for the 2-bit HIGGS dense MLA KV CUDA kernels.

Mirrors ``turboquant_dense_kv.py``: a single JIT-compiled module
exposing the store and dequant entry points. The MLA-decode kernel
lives in a separate JIT module (``higgs_dense_2bit_mla_decode.py``)
to match the TurboQuant split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_higgs_dense_2bit_module() -> "Module":
    return load_jit(
        "higgs_dense_2bit_kv",
        cuda_files=["quantization/higgs_dense_2bit_kv.cuh"],
        extra_dependencies=["cutlass"],
        cuda_wrappers=[
            (
                "store_higgs_dense_2bit",
                "higgs_dense_2bit_detail::HiggsDense2BitStoreKernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit",
                "higgs_dense_2bit_detail::HiggsDense2BitDequantKernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_page_table",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitDequantPageTableKernel::run",
            ),
        ],
    )


@debug_kernel_api
def store_higgs_dense_2bit(
    compressed: torch.Tensor,
    locs: torch.Tensor,
    latent: torch.Tensor,
    rope: torch.Tensor,
    codebook: torch.Tensor,
    codebook_norm_sq: torch.Tensor,
) -> None:
    """Encode ``(N, 1, latent_dim+rope_dim)`` MLA KV into packed slots.

    Args:
      compressed: ``(num_slots, 1, slot_bytes=258)`` ``uint8`` destination.
      locs: ``(N,)`` ``int64`` slot indices into ``compressed``.
      latent: ``(N, 1, 512)`` ``bfloat16`` latent half.
      rope: ``(N, 1, 64)`` ``bfloat16`` rope half.
      codebook: ``(16, 2)`` ``float32`` EDEN2-16 lattice.
      codebook_norm_sq: ``(16,)`` ``float32`` squared codeword norms.
    """
    assert compressed.is_cuda
    assert locs.is_cuda
    assert latent.is_cuda
    assert rope.is_cuda
    assert compressed.dtype == torch.uint8
    assert locs.dtype == torch.int64
    assert latent.dtype == torch.bfloat16
    assert rope.dtype == torch.bfloat16
    assert codebook.dtype == torch.float32
    assert codebook_norm_sq.dtype == torch.float32

    module = _jit_higgs_dense_2bit_module()
    module.store_higgs_dense_2bit(
        compressed,
        locs.contiguous(),
        latent.contiguous() if not latent.is_contiguous() else latent,
        rope.contiguous() if not rope.is_contiguous() else rope,
        codebook.contiguous(),
        codebook_norm_sq.contiguous(),
    )


@debug_kernel_api
def dequantize_higgs_dense_2bit(
    compressed: torch.Tensor,
    locs: torch.Tensor,
    out: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    """Decode packed slots back to BF16 latent + rope.

    Args:
      compressed: ``(num_slots, 1, 258)`` ``uint8`` packed slots.
      locs: ``(N,)`` ``int64`` slot indices to decode.
      out: ``(N, 1, 576)`` ``bfloat16`` destination (latent + rope).
      codebook: ``(16, 2)`` ``float32`` EDEN2-16 lattice.
    """
    assert compressed.is_cuda
    assert locs.is_cuda
    assert out.is_cuda
    assert compressed.dtype == torch.uint8
    assert locs.dtype == torch.int64
    assert out.dtype == torch.bfloat16
    assert codebook.dtype == torch.float32

    module = _jit_higgs_dense_2bit_module()
    module.dequantize_higgs_dense_2bit(
        compressed.contiguous(),
        locs.contiguous(),
        out,
        codebook.contiguous(),
    )


@debug_kernel_api
def dequantize_higgs_dense_2bit_page_table(
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    compact_page_table: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    """Page-table variant of :func:`dequantize_higgs_dense_2bit`.

    Args:
      compressed: ``(num_slots, 1, 258)`` ``uint8`` packed slots.
      page_table: ``(B, K)`` ``int32`` slot indices (-1 marks invalid).
      out: ``(B*K, 1, 576)`` ``bfloat16`` destination.
      compact_page_table: ``(B, K)`` ``int32`` written so that valid
        rows map to themselves and invalid rows map to -1; matches the
        TurboQuant convention.
      codebook: ``(16, 2)`` ``float32`` EDEN2-16 lattice.
    """
    assert compressed.is_cuda
    assert page_table.is_cuda
    assert out.is_cuda
    assert compact_page_table.is_cuda
    assert compressed.dtype == torch.uint8
    assert page_table.dtype == torch.int32
    assert compact_page_table.dtype == torch.int32
    assert out.dtype == torch.bfloat16
    assert codebook.dtype == torch.float32

    module = _jit_higgs_dense_2bit_module()
    module.dequantize_higgs_dense_2bit_page_table(
        compressed.contiguous(),
        page_table,
        out,
        compact_page_table,
        codebook.contiguous(),
    )
