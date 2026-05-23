"""2-bit HIGGS MHA KV codec for the SMC-SD draft model.

Companion to ``higgs_dense_2bit_kv.py``. Reuses the EDEN2-16 lattice
codebook, the orthonormal FWHT primitive, and the 4-bit-pair packing
helpers, but targets standard multi-head attention K/V caches (the
SMC-SD draft model GLM-4-9B-FP8-OMP, ``head_dim=128``) instead of the
DeepSeek MLA latent (``latent_dim=512`` + rope split).

Format (per K-head or V-head, per token)
----------------------------------------

* Rotate the head_dim vector by an orthonormal FWHT of order
  ``head_dim`` (power-of-two, 128 for GLM).
* Normalize by ``s = ||rotated|| / sqrt(head_dim)`` so each post-rotation
  coordinate has unit variance, matching the EDEN2-16 calibration.
* Quantize each 2-D pair to its nearest EDEN2-16 codeword (4-bit index)
  and pack two indices per byte. For ``head_dim=128`` that is 32 bytes
  of indices.
* Store ``s`` as an FP16 scalar (2 bytes).

Slot layout (per K-head, per V-head, per token)::

    [packed_bytes = head_dim/4 B][scale = 2 B]

For ``head_dim=128`` the slot is 34 B, vs 128 B for FP8 (3.76x smaller)
and 256 B for BF16 (7.53x smaller). K and V are quantized independently
so the per-token-per-layer storage for a GQA group with
``num_kv_heads_local=1`` is ``2 * 34 = 68 B``.

The codec is intended for the materialize-on-fetch attention path: the
pool's ``_get_key_buffer`` / ``_get_value_buffer`` decompress the layer
cache on demand and the existing dense Triton decode kernel consumes the
materialized BF16 tensor. A fused dequant-inside-attention kernel is a
follow-on (rule-7 conformant) optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
class HiggsMHA2BitConfig:
    """Configuration for 2-bit HIGGS MHA KV.

    Attributes:
      head_dim: per-head dimension of the K/V vector. Must be a power of
        two and a multiple of the EDEN2 pair dim (2).
    """

    head_dim: int

    def __post_init__(self) -> None:
        if self.head_dim & (self.head_dim - 1):
            raise ValueError(
                f"head_dim ({self.head_dim}) must be a power of two."
            )
        if self.head_dim % HIGGS_PAIR_DIM:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be a multiple of "
                f"HIGGS pair dim ({HIGGS_PAIR_DIM})."
            )

    @property
    def num_pairs(self) -> int:
        return self.head_dim // HIGGS_PAIR_DIM

    @property
    def packed_bytes(self) -> int:
        # 4 bits per pair, 2 pairs per byte.
        return self.num_pairs // 2

    @property
    def slot_bytes(self) -> int:
        return self.packed_bytes + HIGGS_NORM_BYTES


class HiggsMHA2BitCodec:
    """Reference (eager-PyTorch) HIGGS 2-bit MHA codec.

    Compresses/decompresses per-head K or V rows in the SMC-SD draft KV
    pool. K and V do not share scales — each row is independently
    rotated, normalized, and packed.
    """

    def __init__(self, config: HiggsMHA2BitConfig, device: torch.device) -> None:
        self.config = config
        self.device = torch.device(device)
        # Codebook G: shape (16, 2).
        self.codebook = torch.tensor(
            HIGGS_EDEN2_16, dtype=torch.float32, device=self.device
        )
        # ||G_i||^2: shape (16,).
        self.codebook_norm_sq = (
            (self.codebook * self.codebook).sum(dim=-1).contiguous()
        )

    @property
    def slot_bytes(self) -> int:
        return self.config.slot_bytes

    def rotate(self, head_rows: torch.Tensor) -> torch.Tensor:
        return _fwht(head_rows)

    def inverse_rotate(self, rotated: torch.Tensor) -> torch.Tensor:
        return _fwht(rotated)

    def _quantize_pairs(self, normalized_rot: torch.Tensor) -> torch.Tensor:
        pairs = normalized_rot.reshape(
            *normalized_rot.shape[:-1], self.config.num_pairs, HIGGS_PAIR_DIM
        )
        scores = 2.0 * torch.matmul(pairs, self.codebook.T) - self.codebook_norm_sq
        return torch.argmax(scores, dim=-1).to(torch.uint8).contiguous()

    def _values_for_indices(self, indices: torch.Tensor) -> torch.Tensor:
        values = self.codebook[indices.long()]
        return values.flatten(-2).contiguous()

    def compress(self, cache: torch.Tensor) -> torch.Tensor:
        """Encode ``(N, H, head_dim)`` K or V rows into packed slots.

        Args:
          cache: ``(N, H, head_dim)`` tensor in any float dtype. Cast
            to float32 internally for the rotation + quantization math.

        Returns:
          ``(N, H, slot_bytes)`` ``uint8`` packed tensor.
        """
        assert cache.shape[-1] == self.config.head_dim, (
            f"expected head_dim={self.config.head_dim}, "
            f"got {cache.shape[-1]}"
        )
        n, h = cache.shape[0], cache.shape[1]
        flat = cache.reshape(n * h, self.config.head_dim).to(torch.float32)
        rotated = self.rotate(flat)
        scale_f = (
            torch.linalg.vector_norm(rotated, dim=-1).clamp_min(1e-8)
            / math.sqrt(self.config.head_dim)
        )
        normalized = rotated / scale_f[:, None]
        indices = self._quantize_pairs(normalized)
        packed_idx = pack_higgs_2bit_indices(indices)
        scale_bytes = (
            scale_f.to(torch.float16)
            .contiguous()
            .view(torch.uint8)
            .reshape(n * h, HIGGS_NORM_BYTES)
        )
        slots = torch.cat((packed_idx, scale_bytes), dim=-1)
        return slots.reshape(n, h, self.config.slot_bytes)

    def decompress(
        self, packed: torch.Tensor, dst_dtype: torch.dtype
    ) -> torch.Tensor:
        """Decode ``(N, H, slot_bytes)`` packed slots into K or V rows.

        Args:
          packed: ``(N, H, slot_bytes)`` ``uint8`` packed tensor.
          dst_dtype: dtype for the returned K/V rows (typically bf16).
        """
        assert packed.shape[-1] == self.config.slot_bytes, (
            f"expected slot_bytes={self.config.slot_bytes}, "
            f"got {packed.shape[-1]}"
        )
        n, h = packed.shape[0], packed.shape[1]
        flat = packed.reshape(n * h, self.config.slot_bytes)
        index_bytes = flat[:, : self.config.packed_bytes]
        scale_bytes = flat[:, self.config.packed_bytes :]
        indices = unpack_higgs_2bit_indices(index_bytes, self.config.num_pairs).long()
        values = self._values_for_indices(indices)  # fp32, (N*H, head_dim)
        scale = (
            scale_bytes.contiguous()
            .view(torch.float16)
            .reshape(n * h)
            .to(torch.float32)
        )
        rotated = values * scale[:, None]
        out = self.inverse_rotate(rotated).to(dst_dtype)
        return out.reshape(n, h, self.config.head_dim)
