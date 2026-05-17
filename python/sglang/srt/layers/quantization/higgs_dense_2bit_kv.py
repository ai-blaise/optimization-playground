"""2-bit HIGGS dense MLA KV codec (reference Python).

Implements the HIGGS quantization scheme of Pletka et al.
(`Cache Me If You Must` — https://arxiv.org/abs/2501.19392) for the
DeepSeek V3.2-style dense MLA KV layout, as a drop-in alternative to the
2.5-bit TurboQuant codec living next to this file.

Algorithm summary
-----------------

* Per token, the 512-d latent vector is rotated by a fast Hadamard
  transform of order ``HIGGS_HADAMARD_ORDER = 512`` (one block per
  token; "block-diagonal" with a single block, which is the natural
  groupsize for an MLA latent that is exactly Hadamard-friendly).
* The L2 norm of the rotated block is stored as a single FP16 scale
  (2 bytes per token).
* The normalized rotated vector is split into 256 pairs of dimension
  2. Each pair is mapped to the nearest codeword of the EDEN2-16
  lattice via ``argmax(2 * x . G - ||G||^2)``. The resulting 4-bit
  index per pair packs as one byte per two pairs (128 bytes total).
* Decode reverses the steps: codebook lookup -> scale -> inverse
  Hadamard.

Slot layout (kSlotBytes = 258)
------------------------------

  [packed 4-bit pair indices: 128 B] [norm scale fp16: 2 B] [rope: 128 B]

That is 16 bytes / slot smaller than the existing 2.5-bit TurboQuant
slot (``kSlotBytes2p5 = 274``). The savings come from the fact that
HIGGS uses a *uniform 2-bit* code (no high/low split) and one single
scale per token.

Acceptance gates the reference codec is verified against:

* cos_sim(latent, dequant(quant(latent))) > 0.95 for random
  unit-variance latents at ``latent_dim=512`` (mirrors
  ``test_codec_2p5_round_trip_preserves_rope``).
* pack/unpack round trip is bit-exact.
* slot bytes == 258.

The CUDA kernel in ``python/sglang/jit_kernel/csrc/quantization/
higgs_dense_2bit_kv.cuh`` mirrors this math; differences are kernel
fast-paths (FWHT factorization, codebook in registers, fused store)
not algorithmic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

# Eden-2 lattice with 16 entries. Source: AquaKV repo
# (aquakv/grids/EDEN2-16.pt). Original gist:
#   https://gist.github.com/galqiwi/d8fdeb2c6603ad3e54d72a0801416ad3
# Each row is a 2-d codeword; index in [0, 15] => 4 bits per pair => 2
# bits / scalar.
HIGGS_EDEN2_16: Tuple[Tuple[float, float], ...] = (
    (-0.8996632695198059, -1.6360418796539307),
    (-0.961183488368988, 1.5999565124511719),
    (-1.882026195526123, 0.678778350353241),
    (0.36300793290138245, -1.9667866230010986),
    (-0.6814072728157043, -0.576818585395813),
    (0.7270012497901917, 0.6186859607696533),
    (0.3359416127204895, 1.8371193408966064),
    (1.859930396080017, 0.036668598651885986),
    (0.17208248376846313, -0.9401724338531494),
    (-1.7599700689315796, -0.6244229674339294),
    (-0.8993809223175049, 0.32267823815345764),
    (0.839488685131073, -0.3017036020755768),
    (1.5314953327178955, 1.2942044734954834),
    (-0.0011779458727687597, 0.00022069070837460458),
    (1.4274526834487915, -1.207889199256897),
    (-0.16123905777931213, 0.8787511587142944),
)

HIGGS_PAIR_DIM = 2
HIGGS_CODEBOOK_SIZE = 16
HIGGS_BITS_PER_INDEX = 4  # log2(16); each index covers two scalars
HIGGS_HADAMARD_ORDER = 512  # block size for the rotation
HIGGS_NORM_BYTES = 2  # fp16 scale per token


def select_higgs_mla_decode_num_splits(
    num_rows: int, num_heads: int, topk: int
) -> int:
    """B200 auto policy for HIGGS fused MLA split-K decode."""

    row_head_ctas = int(num_rows) * int(num_heads)
    topk = int(topk)

    if topk >= 4096:
        if row_head_ctas <= 8:
            return 128
        if row_head_ctas <= 16:
            return 80
        if row_head_ctas == 32:
            return 56
        if row_head_ctas == 64:
            return 72
        if row_head_ctas == 128:
            return 64
        if row_head_ctas == 256:
            return 40
        return 32
    if topk >= 2048:
        if row_head_ctas <= 8:
            return 96
        if row_head_ctas <= 16:
            return 64
        if row_head_ctas == 32:
            return 56
        if row_head_ctas == 64:
            return 48
        if row_head_ctas == 128:
            return 36
        return 32
    if topk >= 1024:
        if row_head_ctas <= 16:
            return 64
        if row_head_ctas == 64:
            return 48
    return 32


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Fast Walsh-Hadamard transform on the trailing dim.

    Operates on the last axis of an n-d float tensor whose trailing
    size is a power of two and returns a tensor scaled by ``1/sqrt(n)``
    so the transform is its own inverse (``H/sqrt(n)`` is involutory:
    ``H @ H == n * I``).

    The naive aliased version (slice-write into the same storage that
    backs the next iteration's pair reads) silently miscompiles for
    ``n >= 8`` after PyTorch normalises strides; we write into a fresh
    buffer per stage to keep the math correct.
    """
    n = x.shape[-1]
    if n <= 1:
        return x.clone()
    if n & (n - 1):
        raise ValueError(f"FWHT requires power-of-2 dim, got {n}")
    shape = x.shape
    y = x.contiguous().reshape(-1, n)
    h = 1
    while h < n:
        view = y.view(-1, n // (2 * h), 2, h)
        a = view[:, :, 0, :]
        b = view[:, :, 1, :]
        nxt = torch.empty_like(view)
        nxt[:, :, 0, :] = a + b
        nxt[:, :, 1, :] = a - b
        y = nxt.reshape(-1, n)
        h *= 2
    return (y / math.sqrt(n)).reshape(shape)


def pack_higgs_2bit_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codebook indices, two per byte.

    Args:
      indices: ``(..., num_pairs)`` ``uint8`` tensor of values in
        ``[0, 15]``.

    Returns:
      ``(..., num_pairs // 2)`` ``uint8`` tensor.
    """
    indices = indices.to(torch.uint8)
    if indices.shape[-1] % 2:
        indices = torch.nn.functional.pad(indices, (0, 1))
    lo = indices[..., 0::2] & 0x0F
    hi = (indices[..., 1::2] & 0x0F) << 4
    return (lo | hi).contiguous()


def unpack_higgs_2bit_indices(
    packed: torch.Tensor, num_pairs: int
) -> torch.Tensor:
    """Inverse of :func:`pack_higgs_2bit_indices`."""
    packed = packed.to(torch.uint8)
    needed = (num_pairs + 1) // 2
    p = packed[..., :needed]
    out = torch.empty(
        *p.shape[:-1], needed * 2, dtype=torch.uint8, device=p.device
    )
    out[..., 0::2] = p & 0x0F
    out[..., 1::2] = (p >> 4) & 0x0F
    return out[..., :num_pairs].contiguous()


@dataclass(frozen=True)
class HiggsDense2BitConfig:
    """Configuration for 2-bit HIGGS dense MLA KV.

    Attributes:
      latent_dim: latent (LoRA-rank) dimension. Must equal
        ``HIGGS_HADAMARD_ORDER`` and be a power of two.
      rope_dim: RoPE head dim, stored alongside packed latent.
      rope_dtype: dtype the rope half is stored as in the slot.
    """

    latent_dim: int = 512
    rope_dim: int = 64
    rope_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if self.latent_dim != HIGGS_HADAMARD_ORDER:
            raise ValueError(
                f"HIGGS 2-bit dense KV is fixed to latent_dim="
                f"{HIGGS_HADAMARD_ORDER}; got {self.latent_dim}."
            )
        if self.latent_dim % HIGGS_PAIR_DIM:
            raise ValueError(
                f"latent_dim ({self.latent_dim}) must be a multiple of "
                f"HIGGS pair dim ({HIGGS_PAIR_DIM})."
            )

    @property
    def num_pairs(self) -> int:
        return self.latent_dim // HIGGS_PAIR_DIM

    @property
    def packed_bytes(self) -> int:
        # 4 bits per pair, 2 pairs per byte.
        return self.num_pairs // 2

    @property
    def latent_bytes(self) -> int:
        return self.packed_bytes + HIGGS_NORM_BYTES

    @property
    def rope_bytes(self) -> int:
        return self.rope_dim * torch.tensor([], dtype=self.rope_dtype).element_size()

    @property
    def slot_bytes(self) -> int:
        return self.latent_bytes + self.rope_bytes


class HiggsDense2BitCodec:
    """Reference (eager-PyTorch) HIGGS 2-bit codec.

    Used for unit tests and as the cos-similarity oracle the CUDA
    kernel is validated against.
    """

    def __init__(self, config: HiggsDense2BitConfig, device: torch.device) -> None:
        self.config = config
        self.device = torch.device(device)
        # Codebook G: shape (16, 2).
        self.codebook = torch.tensor(
            HIGGS_EDEN2_16, dtype=torch.float32, device=self.device
        )
        # ||G_i||^2: shape (16,).
        self.codebook_norm_sq = (self.codebook * self.codebook).sum(dim=-1).contiguous()
        # 1/sqrt(N) prefactor used by the orthonormal Hadamard.
        self._fwht_scale = 1.0 / math.sqrt(config.latent_dim)

    @property
    def slot_bytes(self) -> int:
        return self.config.slot_bytes

    def rotate(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward block-Hadamard rotation along the last axis."""
        return _fwht(latent)

    def inverse_rotate(self, rotated: torch.Tensor) -> torch.Tensor:
        """Inverse Hadamard rotation. FWHT is self-inverse up to scale."""
        return _fwht(rotated)

    def _quantize_pairs(self, normalized_rot: torch.Tensor) -> torch.Tensor:
        """Map each 2-D pair to its nearest codeword index.

        Args:
          normalized_rot: ``(..., latent_dim)`` rotated tensor, already
            divided by its block L2 norm.

        Returns:
          ``(..., num_pairs)`` ``uint8`` index tensor.
        """
        pairs = normalized_rot.reshape(
            *normalized_rot.shape[:-1], self.config.num_pairs, HIGGS_PAIR_DIM
        )
        # nearest neighbor in 2-D <=> argmax(2 x.G^T - ||G||^2).
        scores = 2.0 * torch.matmul(pairs, self.codebook.T) - self.codebook_norm_sq
        return torch.argmax(scores, dim=-1).to(torch.uint8).contiguous()

    def _values_for_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook vectors and flatten the pair dim.

        Args:
          indices: ``(..., num_pairs)`` integer indices in [0, 15].

        Returns:
          ``(..., latent_dim)`` float32 codeword values.
        """
        values = self.codebook[indices.long()]
        return values.flatten(-2).contiguous()

    def compress(
        self, latent: torch.Tensor, rope: torch.Tensor
    ) -> torch.Tensor:
        """Encode ``(N, 1, latent_dim)`` BF16 latent + rope into packed slots.

        Args:
          latent: ``(N, 1, latent_dim)`` BF16.
          rope: ``(N, 1, rope_dim)`` BF16.

        Returns:
          ``(N, 1, slot_bytes)`` packed ``uint8`` slot tensor.
        """
        n = latent.shape[0]
        flat = latent.reshape(n, self.config.latent_dim).to(torch.float32)
        rotated = self.rotate(flat)
        # rotated has the same L2 norm as the input (orthonormal FWHT).
        # The EDEN2-16 codebook is calibrated for per-coordinate inputs
        # ~ N(0, 1). After we divide rotated by ||rotated||, each entry
        # is ~ N(0, 1/sqrt(N)); multiplying by sqrt(N) restores N(0, 1).
        # Equivalently, divide ``rotated`` by ``s = ||rotated|| /
        # sqrt(N)`` once, and store ``s`` as the per-token scale so that
        # decode reconstructs ``rotated_recon = s * G[idx]`` and
        # ``InvFWHT(rotated_recon)`` recovers the original magnitude.
        scale_f = (
            torch.linalg.vector_norm(rotated, dim=-1).clamp_min(1e-8)
            / math.sqrt(self.config.latent_dim)
        )
        normalized = rotated / scale_f[:, None]
        indices = self._quantize_pairs(normalized)
        packed_idx = pack_higgs_2bit_indices(indices)  # (N, packed_bytes)
        scale = scale_f.to(torch.float16).contiguous().view(torch.uint8).reshape(
            n, HIGGS_NORM_BYTES
        )
        latent_bytes = torch.cat((packed_idx, scale), dim=-1)
        rope_bytes = (
            rope.reshape(n, self.config.rope_dim)
            .to(self.config.rope_dtype)
            .contiguous()
            .view(torch.uint8)
        )
        return torch.cat((latent_bytes, rope_bytes), dim=-1).reshape(
            n, 1, self.config.slot_bytes
        )

    def decompress(
        self, compressed: torch.Tensor, dst_dtype: torch.dtype
    ) -> torch.Tensor:
        """Decode ``(N, 1, slot_bytes)`` into ``(N, 1, latent+rope)``.

        Args:
          compressed: ``(N, 1, slot_bytes)`` ``uint8`` packed slots.
          dst_dtype: dtype the returned latent + rope tensor is cast to.
        """
        n = compressed.shape[0]
        flat = compressed.reshape(n, self.config.slot_bytes)
        packed = flat[:, : self.config.packed_bytes]
        scale_bytes = flat[:, self.config.packed_bytes : self.config.latent_bytes]
        indices = unpack_higgs_2bit_indices(packed, self.config.num_pairs).long()
        values = self._values_for_indices(indices)  # fp32, (N, latent_dim)
        scale = (
            scale_bytes.contiguous()
            .view(torch.float16)
            .reshape(n)
            .to(torch.float32)
        )
        rotated = values * scale[:, None]
        latent = self.inverse_rotate(rotated).to(dst_dtype)
        rope = (
            flat[:, self.config.latent_bytes :]
            .contiguous()
            .view(self.config.rope_dtype)
            .reshape(n, self.config.rope_dim)
            .to(dst_dtype)
        )
        return torch.cat((latent, rope), dim=-1).reshape(
            n, 1, self.config.latent_dim + self.config.rope_dim
        )
