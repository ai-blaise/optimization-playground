from dataclasses import dataclass
from typing import Optional

import pytest

from sglang.srt.layers.attention.nsa.indexer_quantization import (
    INDEXER_AUTO_QUANT_METHOD,
    INDEXER_FP8_QUANT_METHOD,
    INDEXER_NVFP4_QUANT_METHOD,
    get_nsa_indexer_cache_layout,
    get_nsa_indexer_quant_method,
)


def test_fp8_indexer_layout_matches_existing_indexcache_slot():
    layout = get_nsa_indexer_cache_layout(INDEXER_FP8_QUANT_METHOD, 128)
    assert layout.value_bytes == 128
    assert layout.scale_bytes == 4
    assert layout.token_bytes == 132
    assert layout.page_bytes(64) == 64 * 132
    assert layout.quant_block_size == 128


def test_nvfp4_indexer_layout_matches_deepgemm_paged_slot():
    layout = get_nsa_indexer_cache_layout(INDEXER_NVFP4_QUANT_METHOD, 128)
    assert layout.value_bytes == 64
    assert layout.scale_bytes == 4
    assert layout.token_bytes == 68
    assert layout.page_bytes(64) == 64 * 68
    assert layout.quant_block_size == 32


def test_auto_layout_keeps_fp8_default():
    assert (
        get_nsa_indexer_cache_layout(INDEXER_AUTO_QUANT_METHOD, 128).quant_method
        == INDEXER_FP8_QUANT_METHOD
    )


def test_indexer_layout_rejects_unsupported_head_dim():
    with pytest.raises(ValueError, match="head_dim=128"):
        get_nsa_indexer_cache_layout(INDEXER_NVFP4_QUANT_METHOD, 64)


@dataclass
class _Args:
    nsa_indexer_quantization: str = "auto"
    indexer_quantization_declared: Optional[dict] = None


def test_indexer_quant_method_prefers_cli_over_config():
    args = _Args(
        nsa_indexer_quantization=INDEXER_NVFP4_QUANT_METHOD,
        indexer_quantization_declared={"quant_method": INDEXER_FP8_QUANT_METHOD},
    )
    assert get_nsa_indexer_quant_method(args) == INDEXER_NVFP4_QUANT_METHOD


def test_indexer_quant_method_uses_config_when_cli_auto():
    args = _Args(
        indexer_quantization_declared={"quant_method": INDEXER_NVFP4_QUANT_METHOD}
    )
    assert get_nsa_indexer_quant_method(args) == INDEXER_NVFP4_QUANT_METHOD
