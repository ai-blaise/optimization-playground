"""Python bindings for the HIGGS inline sparse-MLA decode producer (iter8 scaffold).

ai-blaise #19 iter8 PRIMARY vector. Scaffolds the SMEM-resident decode
pipeline that iter9 grafts into the flashinfer cute_dsl monolithic
mla_decode_fp8 template's producer warp, eliminating the
sparse-materialize → trtllm-gen-read HBM round-trip (~302 MiB / layer
write + ~302 MiB / layer read = 604 MiB / step at production shape).

Exposes one entry point that mirrors the signature of
:func:`sglang.jit_kernel.higgs_dense_2bit.dequantize_higgs_dense_2bit_page_table_fp8`
so the iter8 microbench can A/B them at the same call site.

Honest scope note: iter8 ships the kernel + microbench + the
correctness contract (bit-exact output vs the existing dequant
kernel). It is NOT wired into ``_forward_trtllm``. iter9 wires it in
behind ``SGLANG_HIGGS_DSA_INLINE_CUTLASS`` and lifts the SMEM staging
into the CUTLASS template's producer warp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_higgs_inline_sparse_mla_decode_module() -> "Module":
    return load_jit(
        "higgs_inline_sparse_mla_decode",
        cuda_files=["quantization/higgs_inline_sparse_mla_decode.cuh"],
        extra_dependencies=["cutlass"],
        cuda_wrappers=[
            (
                "higgs_inline_sparse_mla_produce_fp8",
                "higgs_inline_sparse_mla_detail::"
                "HiggsInlineSparseMLAProduceFp8Kernel::run",
            ),
        ],
    )


@debug_kernel_api
def higgs_inline_sparse_mla_produce_fp8(
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    compact_page_table: torch.Tensor,
    codebook: torch.Tensor,
    inv_kv_scale: float = 1.0,
) -> None:
    """HIGGS inline sparse-MLA producer — emit FP8 (B*K, 1, 576) from packed slots.

    Iter8 scaffold of the SMEM-resident decode pipeline. Same output
    contract as
    :func:`sglang.jit_kernel.higgs_dense_2bit.dequantize_higgs_dense_2bit_page_table_fp8`
    so the two can be A/B'd at the same call site. The iter8 kernel
    uses cp.async slot prefetch + depth-2 SMEM ping-pong staging (vs
    the iter3 kernel's uncached LDG slot read) — the iter9 path
    inherits the same SMEM-resident pipeline but emits into the cubin
    SMEM rather than gmem, eliminating the round-trip.

    Args:
      compressed: ``(num_slots, 1, 272)`` ``uint8`` packed slots
        (iter4 #16: 258 B payload + 14 B 16-align pad).
      page_table: ``(B, K)`` ``int32`` slot indices (-1 marks invalid).
        Same shape contract as the iter3 dequant fn.
      out: ``(B*K, 1, 576)`` ``torch.float8_e4m3fn`` destination.
      compact_page_table: ``(B, K)`` ``int32`` written so valid rows
        map to themselves and invalid rows map to -1.
      codebook: ``(16, 2)`` ``float32`` EDEN2-16 lattice.
      inv_kv_scale: per-tensor scale applied before the FP8 cast.
    """
    assert compressed.is_cuda
    assert page_table.is_cuda
    assert out.is_cuda
    assert compact_page_table.is_cuda
    assert compressed.dtype == torch.uint8
    assert page_table.dtype == torch.int32
    assert compact_page_table.dtype == torch.int32
    assert out.dtype == torch.float8_e4m3fn
    assert codebook.dtype == torch.float32
    assert page_table.dim() == 2, (
        f"page_table must be (B, K); got {tuple(page_table.shape)}"
    )
    B, K = page_table.shape
    assert out.shape == (B * K, 1, 576), (
        f"out must have shape ({B}*{K}, 1, 576); got {tuple(out.shape)}"
    )
    assert compact_page_table.shape == (B, K), (
        f"compact_page_table must be ({B}, {K}); got "
        f"{tuple(compact_page_table.shape)}"
    )

    module = _jit_higgs_inline_sparse_mla_decode_module()
    module.higgs_inline_sparse_mla_produce_fp8(
        compressed.contiguous(),
        page_table.contiguous(),
        out,
        compact_page_table,
        codebook.contiguous(),
        float(inv_kv_scale),
    )
