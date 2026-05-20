"""DSA indexer quantization layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

INDEXER_AUTO_QUANT_METHOD = "auto"
INDEXER_FP8_QUANT_METHOD = "fp8_e4m3"
INDEXER_NVFP4_QUANT_METHOD = "nvfp4_e2m1_ue8m0"
INDEXER_DISABLED_QUANT_METHOD = "disabled"

SUPPORTED_INDEXER_QUANT_METHODS = (
    INDEXER_FP8_QUANT_METHOD,
    INDEXER_NVFP4_QUANT_METHOD,
    INDEXER_DISABLED_QUANT_METHOD,
)

DSA_INDEXER_QUANTIZATION_CHOICES = (
    INDEXER_AUTO_QUANT_METHOD,
    *SUPPORTED_INDEXER_QUANT_METHODS,
)


@dataclass(frozen=True)
class DSAIndexerCacheLayout:
    quant_method: str
    index_head_dim: int
    value_bytes: int
    scale_bytes: int
    quant_block_size: int

    @property
    def token_bytes(self) -> int:
        return self.value_bytes + self.scale_bytes

    def page_bytes(self, page_size: int) -> int:
        return page_size * self.token_bytes


def get_dsa_indexer_quant_method(server_args: Any) -> str:
    cli_method = getattr(
        server_args,
        "dsa_indexer_quantization",
        getattr(server_args, "nsa_indexer_quantization", None),
    )
    if cli_method and cli_method != INDEXER_AUTO_QUANT_METHOD:
        return cli_method

    declared = getattr(server_args, "indexer_quantization_declared", None)
    if isinstance(declared, dict):
        method = declared.get("quant_method")
        if method in SUPPORTED_INDEXER_QUANT_METHODS:
            return method

    return INDEXER_FP8_QUANT_METHOD


def get_dsa_indexer_cache_layout(
    quant_method: Optional[str], index_head_dim: int
) -> DSAIndexerCacheLayout:
    method = quant_method or INDEXER_FP8_QUANT_METHOD
    if method == INDEXER_AUTO_QUANT_METHOD:
        method = INDEXER_FP8_QUANT_METHOD

    if index_head_dim != 128:
        raise ValueError(
            f"DSA indexer cache layout only supports head_dim=128, got {index_head_dim}."
        )

    if method == INDEXER_FP8_QUANT_METHOD:
        return DSAIndexerCacheLayout(
            quant_method=method,
            index_head_dim=index_head_dim,
            value_bytes=index_head_dim,
            scale_bytes=4,
            quant_block_size=128,
        )
    if method == INDEXER_NVFP4_QUANT_METHOD:
        return DSAIndexerCacheLayout(
            quant_method=method,
            index_head_dim=index_head_dim,
            value_bytes=index_head_dim // 2,
            scale_bytes=4,
            quant_block_size=32,
        )
    if method == INDEXER_DISABLED_QUANT_METHOD:
        return get_dsa_indexer_cache_layout(INDEXER_FP8_QUANT_METHOD, index_head_dim)

    raise ValueError(f"Unsupported DSA indexer quantization method: {method!r}.")

# Backward-compatible NSA names used by existing ai-blaise configs/tests.
NSA_INDEXER_QUANTIZATION_CHOICES = DSA_INDEXER_QUANTIZATION_CHOICES
NSAIndexerCacheLayout = DSAIndexerCacheLayout
get_nsa_indexer_quant_method = get_dsa_indexer_quant_method
get_nsa_indexer_cache_layout = get_dsa_indexer_cache_layout
