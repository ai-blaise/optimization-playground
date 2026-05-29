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
from sglang.srt.layers.quantization.higgs_dense_2bit_kv import (
    get_higgs_dense_2bit_b200_candidate,
)

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
                "store_higgs_dense_2bit_const_codebook",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitStoreConstCodebookKernel::run",
            ),
            (
                "store_higgs_dense_2bit_const_codebook_rope_first",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitStoreConstCodebookRopeFirstKernel::run",
            ),
            (
                "store_higgs_dense_2bit_const_codebook_rope_first_index_pack",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitStoreConstCodebookRopeFirstIndexPackKernel::run",
            ),
            (
                "store_higgs_dense_2bit_const_codebook_index_pack",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitStoreConstCodebookIndexPackKernel::run",
            ),
            (
                "store_higgs_dense_2bit_const_codebook_warp_pack",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitStoreConstCodebookWarpPackKernel::run",
            ),
            (
                "store_higgs_dense_2bit_const_codebook_warp_pack_pre_norm",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitStoreConstCodebookWarpPackPreNormKernel::run",
            ),
            (
                "store_higgs_dense_2bit_const_codebook_warp_pack_fma_score",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitStoreConstCodebookWarpPackFmaScoreKernel::run",
            ),
            (
                "store_higgs_dense_2bit_const_codebook_warp_pack_scale_broadcast",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitStoreConstCodebookWarpPackScaleBroadcastKernel::run",
            ),
            (
                "store_higgs_dense_2bit_const_codebook_warp_pack_rope_first",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitStoreConstCodebookWarpPackRopeFirstKernel::run",
            ),
            (
                "store_higgs_dense_2bit_saw_scalar2",
                "higgs_dense_2bit_detail::HiggsDense2BitStoreSawScalar2Kernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit",
                "higgs_dense_2bit_detail::HiggsDense2BitDequantKernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_const_codebook",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitDequantConstCodebookKernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_saw_scalar2",
                "higgs_dense_2bit_detail::HiggsDense2BitDequantSawScalar2Kernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_vec4",
                "higgs_dense_2bit_detail::HiggsDense2BitDequantVec4Kernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_vec4_ldg_codebook",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitDequantVec4LdgCodebookKernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_pair_lanes_scale_broadcast",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitDequantPairLanesScaleBroadcastKernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_page_table",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitDequantPageTableKernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_page_table_const_codebook",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitDequantPageTableConstCodebookKernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_page_table_saw_scalar2",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitDequantPageTableSawScalar2Kernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_page_table_vec4",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitDequantPageTableVec4Kernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_page_table_vec4_ldg_codebook",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitDequantPageTableVec4LdgCodebookKernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_page_table_pair_lanes_scale_broadcast",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitDequantPageTablePairLanesScaleBroadcastKernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_page_table_fp8",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitDequantPageTableFp8Kernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_page_table_fp8_const_codebook",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitDequantPageTableFp8ConstCodebookKernel::run",
            ),
            (
                "dequantize_higgs_dense_2bit_page_table_fp8_saw_scalar2",
                "higgs_dense_2bit_detail::"
                "HiggsDense2BitDequantPageTableFp8SawScalar2Kernel::run",
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
    candidate = get_higgs_dense_2bit_b200_candidate()
    if candidate.store_variant == "saw_scalar2":
        kernel = module.store_higgs_dense_2bit_saw_scalar2
    elif candidate.store_variant == "const_codebook_warp_pack_pre_norm":
        kernel = module.store_higgs_dense_2bit_const_codebook_warp_pack_pre_norm
    elif candidate.store_variant == "const_codebook_warp_pack_fma_score":
        kernel = module.store_higgs_dense_2bit_const_codebook_warp_pack_fma_score
    elif candidate.store_variant == "const_codebook_warp_pack_scale_broadcast":
        kernel = module.store_higgs_dense_2bit_const_codebook_warp_pack_scale_broadcast
    elif candidate.store_variant == "const_codebook_warp_pack_rope_first":
        kernel = module.store_higgs_dense_2bit_const_codebook_warp_pack_rope_first
    elif candidate.store_variant == "const_codebook_warp_pack":
        kernel = module.store_higgs_dense_2bit_const_codebook_warp_pack
    elif candidate.store_variant == "const_codebook_index_pack":
        kernel = module.store_higgs_dense_2bit_const_codebook_index_pack
    elif candidate.store_variant == "const_codebook_rope_first_index_pack":
        kernel = module.store_higgs_dense_2bit_const_codebook_rope_first_index_pack
    elif candidate.store_variant == "const_codebook_rope_first":
        kernel = module.store_higgs_dense_2bit_const_codebook_rope_first
    elif candidate.store_variant == "const_codebook":
        kernel = module.store_higgs_dense_2bit_const_codebook
    else:
        kernel = module.store_higgs_dense_2bit
    kernel(
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
    candidate = get_higgs_dense_2bit_b200_candidate()
    if candidate.dequant_variant == "saw_scalar2":
        kernel = module.dequantize_higgs_dense_2bit_saw_scalar2
    elif candidate.dequant_variant == "const_codebook":
        kernel = module.dequantize_higgs_dense_2bit_const_codebook
    elif candidate.dequant_variant == "vec4_smem_codebook":
        kernel = module.dequantize_higgs_dense_2bit_vec4
    elif candidate.dequant_variant == "vec4_ldg_codebook":
        kernel = module.dequantize_higgs_dense_2bit_vec4_ldg_codebook
    elif candidate.dequant_variant == "pair_lanes_scale_broadcast":
        kernel = module.dequantize_higgs_dense_2bit_pair_lanes_scale_broadcast
    else:
        kernel = module.dequantize_higgs_dense_2bit
    kernel(
        compressed.contiguous(),
        locs.contiguous(),
        out,
        codebook.contiguous(),
    )


@debug_kernel_api
def dequantize_higgs_dense_2bit_vec4(
    compressed: torch.Tensor,
    locs: torch.Tensor,
    out: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    """Opt-in packed-byte vec4/shared-codebook dequant candidate."""
    assert compressed.is_cuda
    assert locs.is_cuda
    assert out.is_cuda
    assert compressed.dtype == torch.uint8
    assert locs.dtype == torch.int64
    assert out.dtype == torch.bfloat16
    assert codebook.dtype == torch.float32

    module = _jit_higgs_dense_2bit_module()
    module.dequantize_higgs_dense_2bit_vec4(
        compressed.contiguous(),
        locs.contiguous(),
        out,
        codebook.contiguous(),
    )


@debug_kernel_api
def dequantize_higgs_dense_2bit_const_codebook(
    compressed: torch.Tensor,
    locs: torch.Tensor,
    out: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    """Opt-in dequant candidate using fixed EDEN2-16 constant memory."""
    assert compressed.is_cuda
    assert locs.is_cuda
    assert out.is_cuda
    assert compressed.dtype == torch.uint8
    assert locs.dtype == torch.int64
    assert out.dtype == torch.bfloat16
    assert codebook.dtype == torch.float32

    module = _jit_higgs_dense_2bit_module()
    module.dequantize_higgs_dense_2bit_const_codebook(
        compressed.contiguous(),
        locs.contiguous(),
        out,
        codebook.contiguous(),
    )


@debug_kernel_api
def dequantize_higgs_dense_2bit_vec4_ldg_codebook(
    compressed: torch.Tensor,
    locs: torch.Tensor,
    out: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    """Opt-in packed-byte vec4/read-only-codebook dequant candidate."""
    assert compressed.is_cuda
    assert locs.is_cuda
    assert out.is_cuda
    assert compressed.dtype == torch.uint8
    assert locs.dtype == torch.int64
    assert out.dtype == torch.bfloat16
    assert codebook.dtype == torch.float32

    module = _jit_higgs_dense_2bit_module()
    module.dequantize_higgs_dense_2bit_vec4_ldg_codebook(
        compressed.contiguous(),
        locs.contiguous(),
        out,
        codebook.contiguous(),
    )


@debug_kernel_api
def dequantize_higgs_dense_2bit_pair_lanes_scale_broadcast(
    compressed: torch.Tensor,
    locs: torch.Tensor,
    out: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    """Opt-in pair-lane codebook + warp-scale-broadcast dequant candidate."""
    assert compressed.is_cuda
    assert locs.is_cuda
    assert out.is_cuda
    assert compressed.dtype == torch.uint8
    assert locs.dtype == torch.int64
    assert out.dtype == torch.bfloat16
    assert codebook.dtype == torch.float32

    module = _jit_higgs_dense_2bit_module()
    module.dequantize_higgs_dense_2bit_pair_lanes_scale_broadcast(
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
    candidate = get_higgs_dense_2bit_b200_candidate()
    if candidate.page_table_dequant_variant == "saw_scalar2":
        kernel = module.dequantize_higgs_dense_2bit_page_table_saw_scalar2
    elif candidate.page_table_dequant_variant == "const_codebook":
        kernel = module.dequantize_higgs_dense_2bit_page_table_const_codebook
    elif candidate.page_table_dequant_variant == "vec4_smem_codebook":
        kernel = module.dequantize_higgs_dense_2bit_page_table_vec4
    elif candidate.page_table_dequant_variant == "vec4_ldg_codebook":
        kernel = module.dequantize_higgs_dense_2bit_page_table_vec4_ldg_codebook
    elif candidate.page_table_dequant_variant == "pair_lanes_scale_broadcast":
        kernel = (
            module.dequantize_higgs_dense_2bit_page_table_pair_lanes_scale_broadcast
        )
    else:
        kernel = module.dequantize_higgs_dense_2bit_page_table
    kernel(
        compressed.contiguous(),
        page_table,
        out,
        compact_page_table,
        codebook.contiguous(),
    )


@debug_kernel_api
def dequantize_higgs_dense_2bit_page_table_fp8(
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    compact_page_table: torch.Tensor,
    codebook: torch.Tensor,
    inv_kv_scale: float = 1.0,
) -> None:
    """FP8 page-table variant of :func:`dequantize_higgs_dense_2bit_page_table`.

    ai-blaise #19 iter3: writes FP8 e4m3 (576 B/row) instead of BF16
    (1152 B/row), halving the HBM round-trip for the trtllm-gen sparse-MLA
    FP8 cubin path. The kernel applies a per-tensor ``inv_kv_scale``
    before the saturating FP8 cast; downstream attention should pass
    ``k_scale = 1 / inv_kv_scale`` via ``bmm1_scale`` to recover the
    original-range BMM1 inputs.

    Args:
      compressed: ``(num_slots, 1, 258)`` ``uint8`` packed slots.
      page_table: ``(B, K)`` ``int32`` slot indices (-1 marks invalid).
      out: ``(B*K, 1, 576)`` ``torch.float8_e4m3fn`` destination.
      compact_page_table: ``(B, K)`` ``int32`` written so that valid rows
        map to themselves and invalid rows map to -1.
      codebook: ``(16, 2)`` ``float32`` EDEN2-16 lattice.
      inv_kv_scale: per-tensor scale applied before the FP8 cast. 1.0 by
        default (saturating cast).
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

    module = _jit_higgs_dense_2bit_module()
    candidate = get_higgs_dense_2bit_b200_candidate()
    # FP8 variants currently shipped:
    #   * saw_scalar2          — production B200 candidate path
    #   * const_codebook       — compile-time codebook constants
    #   * default (codebook-LDG) — runtime codebook fetch via __ldg
    # The saw_scalar2 variant uses the same 4-warps-per-block coalesced
    # layout as the BF16 sibling and dominates the other variants on
    # large num_rows (≥16384) — it's the variant picked in production
    # via ``get_higgs_dense_2bit_b200_candidate``.
    if candidate.page_table_dequant_variant == "saw_scalar2":
        kernel = module.dequantize_higgs_dense_2bit_page_table_fp8_saw_scalar2
    elif candidate.page_table_dequant_variant == "const_codebook":
        kernel = module.dequantize_higgs_dense_2bit_page_table_fp8_const_codebook
    else:
        kernel = module.dequantize_higgs_dense_2bit_page_table_fp8
    kernel(
        compressed.contiguous(),
        page_table,
        out,
        compact_page_table,
        codebook.contiguous(),
        float(inv_kv_scale),
    )


@debug_kernel_api
def dequantize_higgs_dense_2bit_page_table_vec4(
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    compact_page_table: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    """Opt-in page-table vec4/shared-codebook dequant candidate."""
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
    module.dequantize_higgs_dense_2bit_page_table_vec4(
        compressed.contiguous(),
        page_table,
        out,
        compact_page_table,
        codebook.contiguous(),
    )


@debug_kernel_api
def dequantize_higgs_dense_2bit_page_table_const_codebook(
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    compact_page_table: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    """Opt-in page-table dequant candidate using fixed EDEN2-16 constants."""
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
    module.dequantize_higgs_dense_2bit_page_table_const_codebook(
        compressed.contiguous(),
        page_table,
        out,
        compact_page_table,
        codebook.contiguous(),
    )


@debug_kernel_api
def dequantize_higgs_dense_2bit_page_table_vec4_ldg_codebook(
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    compact_page_table: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    """Opt-in page-table vec4/read-only-codebook dequant candidate."""
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
    module.dequantize_higgs_dense_2bit_page_table_vec4_ldg_codebook(
        compressed.contiguous(),
        page_table,
        out,
        compact_page_table,
        codebook.contiguous(),
    )


@debug_kernel_api
def dequantize_higgs_dense_2bit_page_table_pair_lanes_scale_broadcast(
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    compact_page_table: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    """Opt-in page-table pair-lane + warp-scale-broadcast dequant candidate."""
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
    module.dequantize_higgs_dense_2bit_page_table_pair_lanes_scale_broadcast(
        compressed.contiguous(),
        page_table,
        out,
        compact_page_table,
        codebook.contiguous(),
    )
