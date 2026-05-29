"""Fused HIGGS 2-bit MoE expert-weight dequant (Triton).

Task #15 iter2 — replace the eager-PyTorch HIGGS dequant in the BF16
runtime path of :class:`CompressedTensorsHiggsDense2BitMoE` with a
single fused Triton kernel + the existing fast-Hadamard CUDA kernel.

What the kernel does
--------------------

Input
~~~~~

``packed`` : ``uint8`` tensor of shape ``[E, out_dim, slot_bytes]``
where ``slot_bytes = in_dim/4 + 2 * num_blocks_per_row``. Per-row
layout is contiguous:

* first ``in_dim/4`` bytes are 4-bit codebook indices (two pairs per
  byte; low nibble = even pair, high nibble = odd pair).
* remaining ``2 * num_blocks_per_row`` bytes are little-endian FP16
  scale values, one per FWHT block on the row.

What this kernel produces
~~~~~~~~~~~~~~~~~~~~~~~~~

``rotated`` : ``bfloat16`` tensor of shape ``[E, out_dim, in_dim]``
containing the per-block FWHT-rotated codeword vectors multiplied by
their per-block scale. The caller then runs the existing
``hadamard_transform`` CUDA kernel to undo the rotation, yielding the
final dequantized BF16 weight.

We split the pipeline this way because the HIGGS unpack + codebook
lookup + per-block FP16 scale broadcast is a memory-bound kernel that
Triton handles cleanly, while the FWHT itself is best done with the
existing hand-rolled hadamard CUDA kernel (Tri Dao's). The composition
gives us:

* eager Python loop over FWHT stages (the per-call killer)  ->  single
  fast-Hadamard kernel launch.
* eager torch.embedding for the codebook lookup + scale broadcast
  -> single Triton kernel that fuses unpack + lookup + scale.

Together this is a 50-100x reduction in per-layer dequant cost vs the
eager codec, which is what unblocks the BF16 runtime path of the
HIGGS-2bit-MoE scheme for real deploy.

Why this is the right iter2
---------------------------

The iter1 dispatch (HIGGS-packed checkpoint -> dequant -> NVFP4 quant
-> flashinfer trtllm fp4 kernel) is bit-identical at runtime to the
NVFP4 path, so it doesn't move TPOT. The BF16 runtime path *does*
preserve the 8x weight-memory savings but used the eager codec, which
was unusable in production. This kernel makes the BF16 runtime path
usable, which:

1. Gives us a measurable per-decode-step weight-bandwidth win (HIGGS
   on-device vs NVFP4 on-device: 2.25x denser, so 2.25x less weight
   HBM read per MoE layer in the long run).
2. Lets us cache pre-dequanted BF16 working sets per layer if we
   choose — the dequant cost drops from O(hundreds of ms) to O(ms),
   so re-running per call becomes affordable.

The dequant cost reduction is the iter2 measurable; the e2e TPOT win
from running BF16 trtllm with HIGGS-resident weights is the iter3
target (separate, requires updates to the dispatch wiring).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.higgs_dense_2bit_kv import (
    HIGGS_EDEN2_16,
)


# Flatten the codebook so we can pass it as a constant tensor and let
# the Triton compiler treat it as constant memory. Layout: 32 floats,
# indexed by ``2 * code_idx + axis`` where axis in {0, 1}.
_HIGGS_EDEN2_16_FLAT: Tuple[float, ...] = tuple(
    coord for pair in HIGGS_EDEN2_16 for coord in pair
)


@triton.jit
def _higgs_moe_dequant_unpack_kernel(
    packed_ptr,            # *uint8  [E*out_dim, slot_bytes]
    out_ptr,               # *bf16   [E*out_dim, in_dim]
    codebook_ptr,          # *fp32   [32] = flattened EDEN2-16 (2 floats per code)
    n_rows: tl.constexpr,
    in_dim: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NB_PER_ROW: tl.constexpr,
    PACKED_BYTES: tl.constexpr,
    SLOT_BYTES: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    """Unpack HIGGS codebook indices and apply per-block FP16 scale.

    Layout assumed (matches the iter1 codec exactly):
      slot bytes = [packed_indices : in_dim/4 B][fp16 scales : 2*nb B]
      packed nibble layout: byte b holds two 4-bit indices, low nibble
        = pair 2*b, high nibble = pair 2*b+1. Each pair maps to two
        scalars (x, y) via codebook lookup.

    Per CTA we process ``ROWS_PER_PROGRAM`` rows. We tile the in_dim
    axis as ``in_dim // BLOCK_K`` blocks, but we don't need to fan out
    over blocks because the per-row work is small enough that one CTA
    can cover the whole row span efficiently.

    NOTE: This kernel does *not* apply the inverse FWHT; the caller
    follows up with the existing ``hadamard_transform`` CUDA kernel.
    The output here is the per-block-rotated, scaled codeword tile.
    """
    row_start = tl.program_id(0) * ROWS_PER_PROGRAM
    row_offsets = row_start + tl.arange(0, ROWS_PER_PROGRAM)
    row_mask = row_offsets < n_rows

    # We're going to iterate over scalar positions along the row.
    # Each iteration handles ``BLOCK_K`` scalars = ``BLOCK_K // 2``
    # pairs = ``BLOCK_K // 4`` packed bytes.
    BLOCK_PACKED: tl.constexpr = BLOCK_K // 4
    BLOCK_PAIRS: tl.constexpr = BLOCK_K // 2

    # Load each row's slot pointer base.
    row_packed_base = row_offsets * SLOT_BYTES  # [ROWS]

    # Scale base offset within each row.
    scale_base_in_row = PACKED_BYTES

    # Codebook bcast.
    code_lo_offsets = tl.arange(0, 16) * 2
    code_hi_offsets = tl.arange(0, 16) * 2 + 1
    codebook_x = tl.load(codebook_ptr + code_lo_offsets)  # [16]
    codebook_y = tl.load(codebook_ptr + code_hi_offsets)  # [16]

    # Stream over blocks. For each block we read its FP16 scale once,
    # then unroll BLOCK_PACKED uint8 reads, decode -> two scalars per
    # nibble, multiply by scale, store as bf16.
    # The scales section sits at byte offset ``scale_base_in_row``
    # within each row. We read it as two uint8s and reassemble the
    # fp16 via bitcast (uint16 -> fp16) because the row stride may
    # be odd (e.g. SLOT_BYTES=1806 for in_dim=7168 block=1024), so we
    # can't safely cast the per-row base pointer to ``fp16*``.
    for blk in tl.static_range(NB_PER_ROW):
        scale_byte_offset = scale_base_in_row + 2 * blk
        scale_lo = tl.load(
            packed_ptr + row_packed_base + scale_byte_offset,
            mask=row_mask,
            other=tl.full((), 0, tl.uint8),
        ).to(tl.uint16)
        scale_hi = tl.load(
            packed_ptr + row_packed_base + scale_byte_offset + 1,
            mask=row_mask,
            other=tl.full((), 0, tl.uint8),
        ).to(tl.uint16)
        # Reconstruct fp16 from little-endian byte pair.
        scale_u16 = (scale_hi << 8) | scale_lo  # [ROWS] uint16
        scale_fp16 = scale_u16.to(tl.float16, bitcast=True)
        scale_f32 = scale_fp16.to(tl.float32)  # [ROWS]

        # Block byte range within each row's packed indices.
        blk_byte_start = blk * BLOCK_PACKED
        byte_idx = blk_byte_start + tl.arange(0, BLOCK_PACKED)  # [BLOCK_PACKED]

        # Load packed bytes for all rows x all block bytes.
        # Shape: [ROWS, BLOCK_PACKED]
        ptr = (
            packed_ptr
            + row_packed_base[:, None]
            + byte_idx[None, :]
        )
        bytes_2d = tl.load(
            ptr,
            mask=row_mask[:, None],
            other=tl.full((), 0, tl.uint8),
        ).to(tl.int32)

        # Split into low/high nibble = pair_even / pair_odd.
        idx_lo = bytes_2d & 0x0F                       # [ROWS, BLOCK_PACKED]
        idx_hi = (bytes_2d >> 4) & 0x0F                # [ROWS, BLOCK_PACKED]

        # Look up codewords. We have flat 1-D tables of size 16.
        # Per-row gather: codebook_x[idx_lo], codebook_y[idx_lo] for
        # the even pair, then codebook_x[idx_hi], codebook_y[idx_hi]
        # for the odd pair. We interleave so the storage layout is
        # [code_even.x, code_even.y, code_odd.x, code_odd.y, ...].
        # That matches HIGGS_PAIR_DIM=2 + index-per-pair layout.
        # NOTE: Triton doesn't have an .embedding intrinsic; we
        # compute by emitting ``tl.gather``-style loads via 1-D
        # codebook pointer + index broadcast.
        ce_x = tl.load(codebook_ptr + (idx_lo * 2))       # [ROWS, BLOCK_PACKED]
        ce_y = tl.load(codebook_ptr + (idx_lo * 2 + 1))   # [ROWS, BLOCK_PACKED]
        co_x = tl.load(codebook_ptr + (idx_hi * 2))       # [ROWS, BLOCK_PACKED]
        co_y = tl.load(codebook_ptr + (idx_hi * 2 + 1))   # [ROWS, BLOCK_PACKED]

        # Apply per-block scale (broadcast over BLOCK_PACKED) and
        # cast straight to bf16 — only one f32 mul per element.
        s_b = scale_f32[:, None]
        ce_x_bf = (ce_x * s_b).to(tl.bfloat16)
        ce_y_bf = (ce_y * s_b).to(tl.bfloat16)
        co_x_bf = (co_x * s_b).to(tl.bfloat16)
        co_y_bf = (co_y * s_b).to(tl.bfloat16)

        # Interleave into a contiguous [ROWS, BLOCK_K] tile so we can
        # emit one vectorized BF16 store per CTA instead of four.
        # Per byte j we want flat positions 4j..4j+3 = (ce_x, ce_y,
        # co_x, co_y) i.e. (parity=0 axis=0, parity=0 axis=1,
        # parity=1 axis=0, parity=1 axis=1). Reshape of a Triton
        # tensor is row-major; the innermost ``tl.join`` becomes the
        # innermost flat axis, so we have to join axis as the *inner*
        # dim and parity as the *next* dim out:
        #   pair_x = (ce_x | co_x) over parity        -> (R,P,parity)
        #   pair_y = (ce_y | co_y) over parity        -> (R,P,parity)
        #   pair   = (pair_x | pair_y) over axis      -> (R,P,parity,axis)
        # Row-major flat offset = p*4 + parity*2 + axis.
        pair_x = tl.join(ce_x_bf, co_x_bf)      # [ROWS, BLOCK_PACKED, 2]
        pair_y = tl.join(ce_y_bf, co_y_bf)      # [ROWS, BLOCK_PACKED, 2]
        pair = tl.join(pair_x, pair_y)          # [ROWS, BLOCK_PACKED, 2, 2]
        tile = tl.reshape(pair, (ROWS_PER_PROGRAM, BLOCK_K))

        out_row_base = row_offsets * in_dim + blk * BLOCK_K  # [ROWS]
        out_positions = tl.arange(0, BLOCK_K)                # [BLOCK_K]
        tl.store(
            out_ptr + out_row_base[:, None] + out_positions[None, :],
            tile,
            mask=row_mask[:, None],
        )


def _build_codebook_tensor(device: torch.device) -> torch.Tensor:
    """Flattened EDEN2-16 codebook (32 floats), constant cached on device."""
    return torch.tensor(_HIGGS_EDEN2_16_FLAT, dtype=torch.float32, device=device)


_CODEBOOK_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_codebook(device: torch.device) -> torch.Tensor:
    dev = torch.device(device)
    if dev not in _CODEBOOK_CACHE:
        _CODEBOOK_CACHE[dev] = _build_codebook_tensor(dev)
    return _CODEBOOK_CACHE[dev]


def higgs_moe_2bit_unpack_and_scale(
    packed: torch.Tensor,
    in_dim: int,
    block_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unpack HIGGS-packed expert weights and apply per-block scale.

    Does NOT apply the inverse FWHT; the caller is expected to follow
    up with ``hadamard_transform`` on the returned tensor, reshaped to
    ``(E*out_dim*num_blocks, block_size)``.

    Args:
      packed: ``[E, out_dim, slot_bytes]`` ``uint8`` tensor produced
        by the iter1 :class:`HiggsMoE2BitCodec`.
      in_dim: width of the GEMM row this codec was built for.
      block_size: FWHT block size (power of two, divides in_dim).
      out: optional pre-allocated ``[E, out_dim, in_dim]`` ``bf16``
        buffer; if ``None`` we allocate one.

    Returns:
      ``[E, out_dim, in_dim]`` ``bf16`` tensor of scaled, per-block
      FWHT-rotated codewords.
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"expected uint8 packed tensor, got {packed.dtype}")
    if packed.dim() != 3:
        raise ValueError(f"expected 3-D packed tensor, got {packed.shape}")
    if in_dim % block_size:
        raise ValueError(
            f"in_dim ({in_dim}) must be divisible by block_size ({block_size})"
        )
    if block_size & (block_size - 1):
        raise ValueError(
            f"block_size ({block_size}) must be a power of two"
        )

    e, out_dim, slot_bytes = packed.shape
    packed_bytes = in_dim // 4
    nb_per_row = in_dim // block_size
    expected_slot_bytes = packed_bytes + 2 * nb_per_row
    if slot_bytes != expected_slot_bytes:
        raise ValueError(
            f"slot_bytes mismatch: got {slot_bytes}, expected "
            f"{expected_slot_bytes} for in_dim={in_dim} block={block_size}"
        )

    n_rows = e * out_dim
    if out is None:
        out = torch.empty(e, out_dim, in_dim, dtype=torch.bfloat16, device=packed.device)
    else:
        if out.shape != (e, out_dim, in_dim):
            raise ValueError(
                f"out shape mismatch: got {out.shape}, expected "
                f"{(e, out_dim, in_dim)}"
            )
        if out.dtype != torch.bfloat16:
            raise ValueError(f"out dtype must be bfloat16, got {out.dtype}")

    codebook = _get_codebook(packed.device)

    # Heuristics tuned on DeepSeek-V3.2 per-rank shapes (B200):
    #   w13 [E=32, OUT=4096, IN=7168 block=1024]: best at RPP=4 W=4
    #     -> ~0.65 ms (2.8 TiB/s = 35% HBM peak)
    #   w2  [E=32, OUT=7168, IN=2048 block=2048]: best at RPP=4 W=4
    #     -> ~0.30 ms (3.4 TiB/s = 42% HBM peak)
    # Larger RPP (8+) over-saturates SMEM with the BLOCK_K-wide tile.
    rows_per_program = 4
    num_warps = 4
    grid = ((n_rows + rows_per_program - 1) // rows_per_program,)

    _higgs_moe_dequant_unpack_kernel[grid](
        packed.contiguous(),
        out,
        codebook,
        n_rows=n_rows,
        in_dim=in_dim,
        BLOCK_K=block_size,
        NB_PER_ROW=nb_per_row,
        PACKED_BYTES=packed_bytes,
        SLOT_BYTES=slot_bytes,
        ROWS_PER_PROGRAM=rows_per_program,
        num_warps=num_warps,
    )
    return out


def higgs_moe_2bit_dequant_fast(
    packed: torch.Tensor,
    in_dim: int,
    block_size: int,
    dst_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """End-to-end HIGGS 2-bit MoE dequant -> BF16 weight tensor.

    Composition of:
      1. :func:`higgs_moe_2bit_unpack_and_scale` — Triton kernel that
         emits ``[E, out_dim, in_dim]`` BF16 tile of scaled per-block
         FWHT-rotated codewords.
      2. ``hadamard_transform`` — existing fast-Hadamard CUDA kernel,
         applied per-block to invert the FWHT.

    This is the fast replacement for the eager
    :func:`dequantize_higgs_moe_weights` codec used by the iter1 BF16
    runtime path.

    Args:
      packed: ``[E, out_dim, slot_bytes]`` ``uint8`` HIGGS-packed
        expert weight tensor.
      in_dim: width of the GEMM row.
      block_size: FWHT block size (power of two; divides in_dim).
      dst_dtype: dtype for the returned weight; defaults to bf16.

    Returns:
      ``[E, out_dim, in_dim]`` ``dst_dtype`` weight tensor.
    """
    from sglang.jit_kernel.hadamard import hadamard_transform

    e, out_dim, _ = packed.shape
    nb = in_dim // block_size

    # Step 1: unpack indices + scale per-block (Triton).
    tile = higgs_moe_2bit_unpack_and_scale(
        packed, in_dim=in_dim, block_size=block_size
    )

    # Step 2: inverse FWHT per block. The existing
    # ``hadamard_transform`` treats the trailing dim as the transform
    # axis and emits a transform scaled by ``scale``. Our codec stored
    # the rotated tile via the orthonormal FWHT (which is its own
    # inverse), so the inverse here is the same orthonormal transform.
    # The fast hadamard kernel applies an unscaled Walsh-Hadamard, so
    # we pass ``scale=1/sqrt(block_size)`` to make it orthonormal.
    flat = tile.reshape(e * out_dim * nb, block_size)
    inv_scale = 1.0 / math.sqrt(block_size)
    recovered_flat = hadamard_transform(flat, scale=inv_scale)
    recovered = recovered_flat.reshape(e, out_dim, in_dim)
    if dst_dtype != torch.bfloat16:
        recovered = recovered.to(dst_dtype)
    return recovered
