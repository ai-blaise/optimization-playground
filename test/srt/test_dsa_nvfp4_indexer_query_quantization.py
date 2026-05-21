from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.indexer_quantization import (
    INDEXER_FP8_QUANT_METHOD,
    INDEXER_NVFP4_QUANT_METHOD,
)

_DEFAULT_TOKEN_MAP = object()


def _indexer_stub():
    return SimpleNamespace(
        block_size=128,
        scale_fmt="ue8m0",
        _uses_nvfp4_indexer=Indexer._uses_nvfp4_indexer,
    )


def _forward_batch(quant_method):
    return SimpleNamespace(
        out_cache_loc=SimpleNamespace(dtype=torch.int64),
        token_to_kv_pool=SimpleNamespace(
            indexer_quantization=quant_method,
            page_size=64,
        ),
    )


class _Nvfp4HisaIndexerStub:
    dsa_indexer_mode = "indexcache-hisa"
    enable_hisparse = False
    enable_dsa_nvfp4_hisa = True
    hisa_block_size = 128
    hisa_block_topk = 64
    hisa_compression_ratio = 4.0
    hisa_execution_mode = "optimized"
    hisa_min_seq_len = 65536
    index_topk = 1024

    _get_dense_short_relative_topk = Indexer._get_dense_short_relative_topk
    _get_hisa_token_to_batch_idx = Indexer._get_hisa_token_to_batch_idx


def _nvfp4_hisa_indexer_stub():
    return _Nvfp4HisaIndexerStub()


def _mode(*, decode=False, target_verify=False, draft_extend=False):
    return SimpleNamespace(
        is_decode_or_idle=lambda: decode,
        is_target_verify=lambda: target_verify,
        is_draft_extend=lambda include_v2=False: draft_extend,
    )


class _Metadata:
    def __init__(self, q_len, seq_len, token_to_batch_idx=_DEFAULT_TOKEN_MAP):
        self._q_len = q_len
        self._seq_len = seq_len
        self._token_to_batch_idx = token_to_batch_idx

    def get_token_to_batch_idx(self):
        if self._token_to_batch_idx is _DEFAULT_TOKEN_MAP:
            return torch.zeros((self._q_len,), dtype=torch.int32)
        return self._token_to_batch_idx

    def get_dsa_extend_len_cpu(self):
        return [self._q_len]

    def get_indexer_seq_len_cpu(self):
        return torch.tensor([self._seq_len], dtype=torch.int32)

    def get_page_table_1(self):
        return torch.arange(self._seq_len, dtype=torch.int32).unsqueeze(0)


def test_nvfp4_query_quantization_returns_packed_query_tuple(monkeypatch):
    q_values = object()
    q_scales = object()

    monkeypatch.setattr(
        dsa_indexer,
        "can_use_dsa_nvfp4_indexer",
        lambda dtype, indices_dtype, page_size: True,
    )
    monkeypatch.setattr(
        dsa_indexer,
        "quantize_indexer_q_nvfp4",
        lambda query, indices_dtype, page_size: (q_values, q_scales),
    )

    result = Indexer._quantize_query_for_indexer(
        _indexer_stub(),
        torch.empty((1, 128), dtype=torch.bfloat16),
        _forward_batch(INDEXER_NVFP4_QUANT_METHOD),
        act_quant=lambda query, block_size, scale_fmt: (_ for _ in ()).throw(
            AssertionError("FP8 quantization should not run for NVFP4 indexers")
        ),
    )

    assert result == ((q_values, q_scales), None)


def test_fp8_query_quantization_keeps_existing_contract():
    q_values = object()
    q_scales = object()

    result = Indexer._quantize_query_for_indexer(
        _indexer_stub(),
        torch.empty((1, 128), dtype=torch.bfloat16),
        _forward_batch(INDEXER_FP8_QUANT_METHOD),
        act_quant=lambda query, block_size, scale_fmt: (q_values, q_scales),
    )

    assert result == (q_values, q_scales)


def test_nvfp4_hisa_stays_eligible_without_dense_kernel(monkeypatch):
    monkeypatch.setattr(dsa_indexer, "_is_cuda", True)
    monkeypatch.setattr(dsa_indexer, "_has_deep_gemm_kernel", lambda name: False)

    result = Indexer._should_use_hisa_nvfp4_paged(
        _nvfp4_hisa_indexer_stub(),
        SimpleNamespace(forward_mode=_mode(target_verify=True)),
        _Metadata(q_len=1, seq_len=1024),
        torch.tensor([1024], dtype=torch.int32),
    )

    assert result is True


def test_nvfp4_hisa_stays_eligible_without_metadata_token_map(monkeypatch):
    monkeypatch.setattr(dsa_indexer, "_is_cuda", True)
    monkeypatch.setattr(dsa_indexer, "_has_deep_gemm_kernel", lambda name: False)

    result = Indexer._should_use_hisa_nvfp4_paged(
        _nvfp4_hisa_indexer_stub(),
        SimpleNamespace(forward_mode=_mode(target_verify=True)),
        _Metadata(q_len=1, seq_len=1024, token_to_batch_idx=None),
        torch.tensor([1024], dtype=torch.int32),
    )

    assert result is True


def test_nvfp4_hisa_keeps_query_gate_when_dense_kernel_exists(monkeypatch):
    monkeypatch.setattr(dsa_indexer, "_is_cuda", True)
    monkeypatch.setattr(dsa_indexer, "_has_deep_gemm_kernel", lambda name: True)

    result = Indexer._should_use_hisa_nvfp4_paged(
        _nvfp4_hisa_indexer_stub(),
        SimpleNamespace(forward_mode=_mode(decode=True)),
        _Metadata(q_len=1, seq_len=1024),
        torch.tensor([1024], dtype=torch.int32),
    )

    assert result is False


def test_nvfp4_hisa_uses_fused_torch_path_without_deepgemm_fp4_mqa(monkeypatch):
    topk_relative = torch.arange(1024, dtype=torch.int32).unsqueeze(0)

    monkeypatch.setattr(dsa_indexer, "_has_deep_gemm_kernel", lambda name: False)
    monkeypatch.setattr(
        dsa_indexer,
        "nvfp4_hisa_indexer_paged_deepgemm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("DeepGEMM FP4 MQA path should not be used")
        ),
    )
    monkeypatch.setattr(
        dsa_indexer,
        "nvfp4_hisa_indexer_paged_torch",
        lambda *args, **kwargs: topk_relative,
    )

    result = Indexer._get_topk_hisa_nvfp4_paged(
        _nvfp4_hisa_indexer_stub(),
        (torch.empty((1, 32, 64)), torch.empty((1, 32, 1))),
        torch.empty((1, 1), dtype=torch.int8),
        torch.arange(2048, dtype=torch.int32).unsqueeze(0),
        torch.tensor([2048], dtype=torch.int32),
        torch.empty((1, 32)),
        _Metadata(q_len=1, seq_len=2048),
    )

    assert torch.equal(result, topk_relative)
