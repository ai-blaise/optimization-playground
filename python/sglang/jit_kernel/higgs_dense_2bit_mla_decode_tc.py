"""Python binding for the Blackwell tensor-core MLA HIGGS-2bit decode kernel.

CuTe + tcgen05.mma variant of :func:`higgs_dense_2bit_mla_decode`. Same
external contract as the scalar baseline — see
:mod:`sglang.jit_kernel.higgs_dense_2bit_mla_decode` for the math.

Iter 1: tensor cores for the q.K^T score MMA only; acc update keeps a
scalar per-warp loop over BF16 SMEM acc. Iter 2 promotes the V matmul
to a second SM100 MMA on TMEM-FP32 acc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_higgs_dense_2bit_mla_decode_tc_module() -> "Module":
    return load_jit(
        "higgs_dense_2bit_mla_decode_tc",
        cuda_files=["quantization/higgs_dense_2bit_mla_decode_tc.cuh"],
        extra_dependencies=["cutlass"],
        cuda_wrappers=[
            (
                "higgs_dense_2bit_mla_decode_tc",
                "higgs_dense_2bit_mla_tc_detail::"
                "HiggsDense2BitMLADecodeTCKernel::run",
            ),
        ],
    )


@debug_kernel_api
def higgs_dense_2bit_mla_decode_tc(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    codebook: torch.Tensor,
    sm_scale: float,
) -> None:
    """Tensor-core single-pass fused MLA decode against a 2-bit HIGGS KV cache.

    Mathematically identical to :func:`higgs_dense_2bit_mla_decode`. Uses
    SM100 tcgen05.mma BF16 tensor cores for the q.K^T score MMA. Requires
    NVIDIA Blackwell (compute capability 10.0).

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

    module = _jit_higgs_dense_2bit_mla_decode_tc_module()
    module.higgs_dense_2bit_mla_decode_tc(
        q_nope.contiguous(),
        q_rope.contiguous(),
        compressed.contiguous(),
        page_table.contiguous(),
        out,
        codebook.contiguous(),
        float(sm_scale),
    )
