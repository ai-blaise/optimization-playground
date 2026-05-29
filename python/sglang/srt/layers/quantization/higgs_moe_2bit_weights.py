"""2-bit HIGGS codec for MoE expert weight matrices.

Companion to ``higgs_dense_2bit_kv.py`` and ``higgs_mha_2bit_kv.py``.
Reuses the EDEN2-16 lattice codebook, orthonormal FWHT primitive, and
the 4-bit-pair packing helpers, but targets *static expert weight
matrices* (the ``[num_experts, out_dim, in_dim]`` GEMM tensors that
back gate / up / down projections in an MoE block) instead of dynamic
KV-cache rows.

The compositional contract this file enables (task #15 of the B200
DeepSeek-V3.2 campaign):

* Store the gate / up / down expert weights packed at 2-bit on disk
  via the HIGGS scheme (~2x smaller than NVFP4, ~8x smaller than BF16).
* At ``process_weights_after_loading`` time, dequant once into a fresh
  BF16 buffer, immediately re-quantize via the existing
  ``prepare_static_weights_for_trtllm_fp4_moe`` pipeline so that
  flashinfer's ``trtllm_fp4_block_scale_moe`` kernel sees its native
  NVFP4 input format. The BF16 buffer is dropped at the end of
  preparation; on-GPU footprint matches the NVFP4 path.
* Followup (out of scope for this commit but the file is laid out for
  it): expose a runtime dequant-to-BF16 path that feeds
  ``trtllm_bf16_moe`` directly, trading GEMM throughput for HIGGS-level
  GPU memory savings.

Per-row format (one HIGGS "slot" per expert row)
------------------------------------------------

* Rotate each ``in_dim``-wide row by an orthonormal FWHT (in_dim must
  be a power of two).
* Normalize by ``s = ||rotated|| / sqrt(in_dim)`` so each post-rotation
  coordinate has unit variance, matching the EDEN2-16 calibration.
* Quantize each 2-D pair to its nearest EDEN2-16 codeword (4-bit index)
  and pack two indices per byte. For ``in_dim=H`` that is ``H/4`` bytes
  per row.
* Store ``s`` as an FP16 scalar (2 bytes per row).

Per-expert byte budget for an MoE block with ``hidden_size=7168``,
``intermediate_size=2048``, ``num_experts=256`` (DeepSeek-V3.2 REAP):

* w13 ``[E, 2*I, H]``: ``E * 2*I * (H/4 + 2)`` bytes
  = ``256 * 4096 * (1792 + 2)`` = 1.88 GiB
* w2 ``[E, H, I]``:    ``E * H * (I/4 + 2)`` bytes
  = ``256 * 7168 * (512 + 2)`` = 941 MiB
* Total: ~2.82 GiB vs ~5.64 GiB (NVFP4) and ~22.5 GiB (BF16)

Compared to the NVFP4 baseline this is a 2x storage savings; compared
to BF16 it is 8x.

This file deliberately stays codec-only — it does not register schemes
or wire CLI flags. The scheme that consumes it lives at
``compressed_tensors/schemes/compressed_tensors_higgs_dense_2bit_moe.py``
and the config-side dispatch lives at
``quantization_config_dispatch.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.layers.quantization.higgs_dense_2bit_kv import (
    HIGGS_EDEN2_16,
    HIGGS_NORM_BYTES,
    HIGGS_PAIR_DIM,
    _fwht,
    pack_higgs_2bit_indices,
    unpack_higgs_2bit_indices,
)


@dataclass(frozen=True)
class HiggsMoE2BitConfig:
    """Configuration for the 2-bit HIGGS MoE expert-weight codec.

    Attributes:
      in_dim: The input dimension of the GEMM row being quantized
        (``hidden_size`` for w13, ``intermediate_size`` for w2). Must be
        a power of two and a multiple of the EDEN2 pair dim (2).
      block_size: Number of contiguous scalars rotated together inside
        one HIGGS slot. Defaults to ``in_dim`` (one block per row) which
        matches the dense MLA latent convention. Must divide ``in_dim``
        and be a power of two. The codebook calibration is least
        violated when ``block_size`` is at least 64; sub-row blocking
        is exposed for future hidden-size-7168 experiments where a
        single 7168-wide FWHT is too wide for one CTA.
    """

    in_dim: int
    block_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.in_dim % HIGGS_PAIR_DIM:
            raise ValueError(
                f"in_dim ({self.in_dim}) must be a multiple of "
                f"HIGGS pair dim ({HIGGS_PAIR_DIM})."
            )
        block = self.block_size if self.block_size is not None else self.in_dim
        # Only the FWHT block must be a power of two. ``in_dim`` itself
        # need not be — DeepSeek-V3.2's hidden_size=7168 (7*2^10) is
        # quantized with block_size=1024 (7 blocks per row).
        if block & (block - 1):
            raise ValueError(
                f"block_size ({block}) must be a power of two."
            )
        if self.in_dim % block:
            raise ValueError(
                f"in_dim ({self.in_dim}) must be a multiple of "
                f"block_size ({block})."
            )

    @property
    def effective_block_size(self) -> int:
        return self.block_size if self.block_size is not None else self.in_dim

    @property
    def num_blocks_per_row(self) -> int:
        return self.in_dim // self.effective_block_size

    @property
    def num_pairs_per_block(self) -> int:
        return self.effective_block_size // HIGGS_PAIR_DIM

    @property
    def num_pairs_per_row(self) -> int:
        return self.in_dim // HIGGS_PAIR_DIM

    @property
    def packed_bytes_per_row(self) -> int:
        # 4 bits per pair, 2 pairs per byte.
        return self.num_pairs_per_row // 2

    @property
    def scale_bytes_per_row(self) -> int:
        return HIGGS_NORM_BYTES * self.num_blocks_per_row

    @property
    def slot_bytes_per_row(self) -> int:
        return self.packed_bytes_per_row + self.scale_bytes_per_row


class HiggsMoE2BitCodec:
    """Reference (eager-PyTorch) HIGGS 2-bit MoE weight codec.

    Quantizes/dequantizes 3D tensors ``[num_experts, out_dim, in_dim]``
    by treating each row as one HIGGS slot. The pack format is laid out
    contiguously per row: ``[packed_indices : packed_bytes_per_row B]
    [scales : scale_bytes_per_row B]`` so a 2-bit MoE-weight tensor has
    final shape ``[num_experts, out_dim, slot_bytes_per_row]`` of
    ``uint8``.
    """

    def __init__(self, config: HiggsMoE2BitConfig, device: torch.device) -> None:
        self.config = config
        self.device = torch.device(device)
        self.codebook = torch.tensor(
            HIGGS_EDEN2_16, dtype=torch.float32, device=self.device
        )
        self.codebook_norm_sq = (
            (self.codebook * self.codebook).sum(dim=-1).contiguous()
        )

    @property
    def slot_bytes_per_row(self) -> int:
        return self.config.slot_bytes_per_row

    # -- rotation primitives ------------------------------------------------

    def _reshape_to_blocks(self, rows: torch.Tensor) -> torch.Tensor:
        """Reshape ``(..., in_dim)`` to ``(..., num_blocks, block_size)``."""
        block = self.config.effective_block_size
        nb = self.config.num_blocks_per_row
        return rows.reshape(*rows.shape[:-1], nb, block)

    def rotate(self, rows: torch.Tensor) -> torch.Tensor:
        """Block-orthonormal FWHT on the trailing dim."""
        if self.config.num_blocks_per_row == 1:
            return _fwht(rows)
        blocks = self._reshape_to_blocks(rows)
        rotated = _fwht(blocks)
        return rotated.reshape(*rows.shape)

    def inverse_rotate(self, rotated: torch.Tensor) -> torch.Tensor:
        return self.rotate(rotated)

    # -- nearest-neighbor lookup -------------------------------------------

    def _quantize_pairs(self, normalized: torch.Tensor) -> torch.Tensor:
        """Map ``(..., in_dim)`` rows to ``(..., num_pairs_per_row)`` indices."""
        pairs = normalized.reshape(
            *normalized.shape[:-1],
            self.config.num_pairs_per_row,
            HIGGS_PAIR_DIM,
        )
        scores = 2.0 * torch.matmul(pairs, self.codebook.T) - self.codebook_norm_sq
        return torch.argmax(scores, dim=-1).to(torch.uint8).contiguous()

    def _values_for_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codewords, then flatten the pair dim back to in_dim."""
        values = self.codebook[indices.long()]
        return values.flatten(-2).contiguous()

    # -- 3-D weight tensor encode / decode ---------------------------------

    def compress(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode ``[E, out_dim, in_dim]`` BF16/FP16/FP32 weight tensor.

        Args:
          weight: ``[E, out_dim, in_dim]`` floating tensor.

        Returns:
          packed: ``[E, out_dim, slot_bytes_per_row]`` ``uint8``.
          scales_view: ``[E, out_dim, num_blocks_per_row]`` ``float16``
            view aliased into ``packed`` — written to ``packed`` already,
            returned for callers that want a typed handle.
        """
        if weight.shape[-1] != self.config.in_dim:
            raise ValueError(
                f"expected in_dim={self.config.in_dim}, got {weight.shape[-1]}"
            )
        e, out_dim, _ = weight.shape
        flat = weight.reshape(e * out_dim, self.config.in_dim).to(torch.float32)
        rotated = self.rotate(flat)
        block = self.config.effective_block_size
        nb = self.config.num_blocks_per_row

        block_view = rotated.reshape(e * out_dim, nb, block)
        # Per-block L2 norm, then divide by sqrt(block_size) so each
        # post-normalize coordinate matches the EDEN2-16 unit-variance
        # calibration.
        scale_f = (
            torch.linalg.vector_norm(block_view, dim=-1).clamp_min(1e-8)
            / math.sqrt(block)
        )
        normalized_block = block_view / scale_f.unsqueeze(-1)
        normalized = normalized_block.reshape(e * out_dim, self.config.in_dim)

        indices = self._quantize_pairs(normalized)
        packed_idx = pack_higgs_2bit_indices(indices)  # (E*out_dim, packed_B)

        scale_bytes = (
            scale_f.to(torch.float16)
            .contiguous()
            .view(torch.uint8)
            .reshape(e * out_dim, self.config.scale_bytes_per_row)
        )
        slot = torch.cat((packed_idx, scale_bytes), dim=-1)
        packed = slot.reshape(e, out_dim, self.config.slot_bytes_per_row).contiguous()

        scales_view = (
            packed[..., self.config.packed_bytes_per_row :]
            .reshape(e, out_dim, self.config.scale_bytes_per_row)
            .view(torch.float16)
            .reshape(e, out_dim, nb)
        )
        return packed, scales_view

    def decompress(
        self, packed: torch.Tensor, dst_dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        """Decode ``[E, out_dim, slot_bytes_per_row]`` packed tensor.

        Args:
          packed: ``[E, out_dim, slot_bytes_per_row]`` ``uint8`` tensor.
          dst_dtype: dtype for the returned weight tensor.

        Returns:
          ``[E, out_dim, in_dim]`` weight tensor in ``dst_dtype``.
        """
        if packed.shape[-1] != self.config.slot_bytes_per_row:
            raise ValueError(
                f"expected slot_bytes={self.config.slot_bytes_per_row}, "
                f"got {packed.shape[-1]}"
            )
        e, out_dim, _ = packed.shape
        flat = packed.reshape(e * out_dim, self.config.slot_bytes_per_row)
        index_bytes = flat[:, : self.config.packed_bytes_per_row]
        scale_bytes = flat[:, self.config.packed_bytes_per_row :]
        indices = unpack_higgs_2bit_indices(
            index_bytes, self.config.num_pairs_per_row
        ).long()
        values = self._values_for_indices(indices)  # (E*out_dim, in_dim) fp32
        nb = self.config.num_blocks_per_row
        block = self.config.effective_block_size
        scales = (
            scale_bytes.contiguous()
            .view(torch.float16)
            .reshape(e * out_dim, nb)
            .to(torch.float32)
        )
        rotated = (
            values.reshape(e * out_dim, nb, block) * scales.unsqueeze(-1)
        ).reshape(e * out_dim, self.config.in_dim)
        recovered = self.inverse_rotate(rotated)
        return recovered.reshape(e, out_dim, self.config.in_dim).to(dst_dtype)


def quantize_moe_weights_to_higgs(
    weight: torch.Tensor,
    block_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, HiggsMoE2BitConfig]:
    """Quantize a BF16/FP16/FP32 MoE weight tensor to HIGGS 2-bit.

    Args:
      weight: ``[E, out_dim, in_dim]`` floating tensor.
      block_size: Optional sub-row block size for the FWHT (defaults to
        ``in_dim``).
      device: Device to materialize the codebook on; defaults to
        ``weight.device``.

    Returns:
      packed: ``[E, out_dim, slot_bytes_per_row]`` ``uint8`` tensor.
      config: The :class:`HiggsMoE2BitConfig` used for the encode, so
        the corresponding decode call can recover the same shape.
    """
    if weight.dim() != 3:
        raise ValueError(
            f"expected [E, out_dim, in_dim] tensor, got shape {weight.shape}"
        )
    in_dim = weight.shape[-1]
    cfg = HiggsMoE2BitConfig(in_dim=in_dim, block_size=block_size)
    dev = torch.device(device) if device is not None else weight.device
    codec = HiggsMoE2BitCodec(cfg, dev)
    packed, _ = codec.compress(weight)
    return packed, cfg


def dequantize_higgs_moe_weights(
    packed: torch.Tensor,
    config: HiggsMoE2BitConfig,
    dst_dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Inverse of :func:`quantize_moe_weights_to_higgs`."""
    dev = torch.device(device) if device is not None else packed.device
    codec = HiggsMoE2BitCodec(config, dev)
    return codec.decompress(packed, dst_dtype=dst_dtype)
