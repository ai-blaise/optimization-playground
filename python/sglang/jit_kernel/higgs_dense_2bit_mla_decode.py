"""Python binding for the 2-bit HIGGS dense MLA decode CUDA kernel."""

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
                "higgs_dense_2bit_mla_decode",
                "higgs_dense_2bit_mla_detail::"
                "HiggsDense2BitMLADecodeKernel::run",
            ),
        ],
    )


@debug_kernel_api
def higgs_dense_2bit_mla_decode(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    codebook: torch.Tensor,
    sm_scale: float,
) -> None:
    """Fused MLA decode against a 2-bit HIGGS packed KV cache.

    Computes ``softmax((q_nope . K_latent + q_rope . K_rope) * sm_scale)
    @ K_latent``, where ``K_latent`` and ``K_rope`` are dequantized
    on-the-fly from the HIGGS-packed slots indexed by ``page_table``.

    Args:
      q_nope: ``(num_rows, num_heads, 512)`` ``bfloat16`` query latent.
      q_rope: ``(num_rows, num_heads, 64)`` ``bfloat16`` query rope.
      compressed: ``(num_slots, 1, 258)`` ``uint8`` packed KV slots.
      page_table: ``(num_rows, topk)`` ``int32`` slot indices
        (``-1`` masks a row).
      out: ``(num_rows, num_heads, 512)`` ``bfloat16`` destination.
      codebook: ``(16, 2)`` ``float32`` EDEN2-16 lattice.
      sm_scale: softmax scale factor.
    """
    assert q_nope.is_cuda
    assert q_rope.is_cuda
    assert compressed.is_cuda
    assert page_table.is_cuda
    assert out.is_cuda
    assert q_nope.dtype == torch.bfloat16
    assert q_rope.dtype == torch.bfloat16
    assert compressed.dtype == torch.uint8
    assert page_table.dtype == torch.int32
    assert out.dtype == torch.bfloat16
    assert codebook.dtype == torch.float32

    module = _jit_higgs_dense_2bit_mla_decode_module()
    module.higgs_dense_2bit_mla_decode(
        q_nope.contiguous(),
        q_rope.contiguous(),
        compressed.contiguous(),
        page_table.contiguous(),
        out,
        codebook.contiguous(),
        float(sm_scale),
    )
