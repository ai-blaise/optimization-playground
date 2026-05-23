"""WINA activation-sparsity loader + masking helpers.

Inference-time integration for models sparsified by
`infrastructure/scripts/sglang-reap/wina_sparsify_glm_draft.py`. The sparsifier
writes a `wina_config` block into `config.json` and a sidecar
`wina_norms.safetensors` containing the per-input-column L2 norms of the
norm-bearing projection in each fused pair (K in attention, gate_up in MLP).
Model code consumes this via :func:`load_wina_assets` at init time and applies
:func:`wina_topk_mask` inside the forward pass on the activation feeding each
linear.

For each mask point the per-token criterion is::

    s_i = |x_i| * norm_i       (norm-bearing projection in the fused pair)
    s_i = |x_i|                (magnitude-only — second projection of the pair)

The top-K = round((1 - sparsity) * hidden_dim) columns by ``s_i`` survive; the
rest are zeroed before the linear runs. Mask compute is per-row and lives in
the activation's dtype/device (norms are bf16 and broadcast to fp32 internally
to match the sparsifier's float32 score path).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from safetensors import safe_open

WINA_CONFIG_KEY = "wina_config"


@dataclass(frozen=True)
class WinaConfig:
    """Parsed ``wina_config`` block from a sparsified model's ``config.json``.

    Fields mirror the sparsifier's metadata schema (v1.0). ``projections`` is a
    list of dicts describing each linear that participates in the mask, in the
    order the sparsifier wrote them. Only the projections with
    ``applies_column_norm=True`` have tensors in ``norms_file``; the rest use
    plain magnitude criterion at inference.
    """

    method: str
    version: str
    sparsity: float
    mask_by: str
    norms_file: str
    num_layers: int
    projections: tuple[dict, ...]

    @property
    def is_topk(self) -> bool:
        return self.mask_by == "topk"


def parse_wina_config(hf_config: object) -> Optional[WinaConfig]:
    """Return the parsed ``wina_config`` block or ``None`` when absent."""
    raw = getattr(hf_config, WINA_CONFIG_KEY, None)
    if raw is None and isinstance(hf_config, Mapping):
        raw = hf_config.get(WINA_CONFIG_KEY)
    if raw is None:
        return None
    return WinaConfig(
        method=raw["method"],
        version=raw["version"],
        sparsity=float(raw["sparsity"]),
        mask_by=raw["mask_by"],
        norms_file=raw["norms_file"],
        num_layers=int(raw["num_layers"]),
        projections=tuple(raw["projections"]),
    )


def projection_norm_key(layer_idx: int, projection_name: str) -> str:
    """Stable key under which the sparsifier stores each norm tensor.

    Must match ``wina_sparsify_glm_draft.projection_norm_key`` byte-for-byte.
    """
    return f"wina.layers.{layer_idx}.{projection_name}.in_norm"


def load_wina_assets(
    model_path: str,
    config: WinaConfig,
) -> dict[str, torch.Tensor]:
    """Load every per-input-column norm tensor produced by the sparsifier.

    Returns a flat dict mapping :func:`projection_norm_key` to a 1-D bf16
    tensor of length ``hidden_in`` for that projection. Tensors stay on CPU
    here; callers move them to the device.
    """
    norms_path = os.path.join(model_path, config.norms_file)
    if not os.path.isfile(norms_path):
        raise FileNotFoundError(
            f"WINA norms file {norms_path!r} not found; the sparsifier writes "
            f"this alongside config.json."
        )
    norms: dict[str, torch.Tensor] = {}
    with safe_open(norms_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            norms[key] = handle.get_tensor(key)
    return norms


def wina_topk_mask(
    x: torch.Tensor,
    norm: Optional[torch.Tensor],
    sparsity: float,
) -> torch.Tensor:
    """Per-row top-K mask matching the sparsifier's criterion.

    Uses :func:`torch.topk` + scatter so ties resolve identically to the
    reference implementation in ``wina_sparsify_glm_draft.run_self_test``
    (each row keeps exactly ``keep`` columns).

    Args:
      x: activation of shape ``(..., hidden_in)``.
      norm: 1-D tensor of length ``hidden_in`` (bf16/fp32) when the mask point
        is norm-bearing, else ``None`` for plain magnitude criterion.
      sparsity: fraction of input columns to drop per token (0 < sparsity < 1).

    Returns:
      Boolean mask of the same shape as ``x``; ``True`` means keep.
    """
    hidden_in = x.shape[-1]
    keep = max(1, round((1.0 - sparsity) * hidden_in))
    if keep >= hidden_in:
        return torch.ones_like(x, dtype=torch.bool)
    score = x.abs().to(torch.float32)
    if norm is not None:
        score = score * norm.to(torch.float32).to(x.device)
    top_idx = torch.topk(score, keep, dim=-1).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    return mask.scatter(-1, top_idx, True)


def apply_wina_mask(
    x: torch.Tensor,
    norm: Optional[torch.Tensor],
    sparsity: float,
) -> torch.Tensor:
    """Zero out the dropped columns of ``x`` per :func:`wina_topk_mask`."""
    if sparsity <= 0.0:
        return x
    mask = wina_topk_mask(x, norm, sparsity)
    return x * mask.to(x.dtype)
