"""Python bindings for the 2-bit HIGGS MHA decode CUDA kernel.

Mirrors :mod:`sglang.jit_kernel.higgs_dense_2bit_mla_decode` but for
the SMC-SD draft path: standard MHA / GQA with ``head_dim=128`` and
K + V quantized into independent 34-byte HIGGS slots (vs MLA's single
258-byte slot that fuses latent + rope).

Single entry point:

* :func:`higgs_mha_2bit_decode` -- two-stage split-K decode. Stage 1
  processes ``kBlockH = 16`` Q heads sharing one kv_head over a
  ``1/num_splits`` slice of kv_len per block. Stage 2 merges per-split
  ``(m, l, acc)`` partials and runs InvFWHT_128 + final BF16 write.
  Restores SM occupancy when the per-(row, kv_head) grid is too
  shallow (e.g. small batch + few kv_heads).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# kBlockH and 2 + head_dim must match the kernel-side constants.
_HIGGS_MHA_BLOCK_H = 16
_HIGGS_MHA_HEAD_DIM = 128
_HIGGS_MHA_PARTIAL_PER_HEAD = 2 + _HIGGS_MHA_HEAD_DIM


@cache_once
def _jit_higgs_mha_2bit_decode_module() -> "Module":
    return load_jit(
        "higgs_mha_2bit_decode",
        cuda_files=["quantization/higgs_mha_2bit_decode.cuh"],
        extra_dependencies=["cutlass"],
        cuda_wrappers=[
            (
                "higgs_mha_2bit_decode",
                "higgs_mha_2bit_detail::HiggsMHA2BitDecodeKernel::run",
            ),
        ],
    )


def _choose_num_splits(kv_len_hint: int) -> int:
    """Heuristic: target chunk_size = 128 tokens per split, capped at 16."""
    if kv_len_hint <= 0:
        return 1
    s = (kv_len_hint + 127) // 128
    if s < 1:
        return 1
    if s > 16:
        return 16
    return s


@debug_kernel_api
def higgs_mha_2bit_decode(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    out: torch.Tensor,
    codebook: torch.Tensor,
    sm_scale: float,
) -> None:
    """Split-K fused MHA decode against 2-bit HIGGS-packed K/V buffers.

    Computes ``softmax(q . K^T * sm_scale) @ V`` for one decode step,
    with K and V dequantized on the fly from the HIGGS-packed slots
    indexed by ``kv_indptr`` + ``kv_indices``. K and V are quantized
    independently (each has its own 34-byte slot per (token, kv_head)).

    Args:
      q: ``(num_rows, num_q_heads, 128)`` ``bfloat16`` query.
      k_packed: ``(num_kv_slots, num_kv_heads, 34)`` ``uint8`` K slots.
      v_packed: ``(num_kv_slots, num_kv_heads, 34)`` ``uint8`` V slots.
      kv_indptr: ``(num_rows + 1,)`` ``int32`` row-to-token offsets
        (same layout as SGLang's standard decode metadata).
      kv_indices: ``(total_kv_tokens,)`` ``int32`` token-to-slot map.
      out: ``(num_rows, num_q_heads, 128)`` ``bfloat16`` destination.
      codebook: ``(16, 2)`` ``float32`` EDEN2-16 lattice.
      sm_scale: softmax scale (typically ``1 / sqrt(head_dim)``).
    """
    assert q.is_cuda
    assert k_packed.is_cuda
    assert v_packed.is_cuda
    assert kv_indptr.is_cuda
    assert kv_indices.is_cuda
    assert out.is_cuda
    assert q.dtype == torch.bfloat16
    assert k_packed.dtype == torch.uint8
    assert v_packed.dtype == torch.uint8
    assert kv_indptr.dtype == torch.int32
    assert kv_indices.dtype == torch.int32
    assert out.dtype == torch.bfloat16
    assert codebook.dtype == torch.float32

    num_rows = q.shape[0]
    num_q_heads = q.shape[1]
    num_kv_heads = k_packed.shape[1]
    total_kv = kv_indices.shape[0]

    # Average kv_len drives the split count.
    kv_len_hint = total_kv // max(num_rows, 1)
    num_splits = _choose_num_splits(kv_len_hint)

    # Scratch shape matches the kernel TensorMatcher: (R, K_kv, P, H, 2+D).
    mid = torch.empty(
        (num_rows, num_kv_heads, num_splits, _HIGGS_MHA_BLOCK_H,
         _HIGGS_MHA_PARTIAL_PER_HEAD),
        dtype=torch.float32,
        device=q.device,
    )

    module = _jit_higgs_mha_2bit_decode_module()
    module.higgs_mha_2bit_decode(
        q.contiguous(),
        k_packed.contiguous(),
        v_packed.contiguous(),
        kv_indptr.contiguous(),
        kv_indices.contiguous(),
        mid,
        out,
        codebook.contiguous(),
        float(sm_scale),
    )
