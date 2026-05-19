from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.flashsampling.runtime import (
    FlashSamplingInfo,
    FlashSamplingRuntime,
    _should_use_dense_greedy_path_on_blackwell,
    get_flashsampling_lm_head_metadata,
    get_flashsampling_info,
)
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL


def _sampling_info(**overrides):
    defaults = dict(
        temperatures=torch.ones(2, 1),
        top_ps=torch.ones(2),
        top_ks=torch.full((2,), TOP_K_ALL, dtype=torch.int32),
        min_ps=torch.zeros(2),
        is_all_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=16,
        device="cpu",
        penalizer_orchestrator=MagicMock(is_required=False),
    )
    defaults.update(overrides)
    return SamplingBatchInfo(**defaults)


def _server_args(**overrides):
    defaults = dict(
        enable_flashsampling=True,
        flashsampling_fallback="auto",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _forward_mode(**overrides):
    defaults = dict(
        is_decode=lambda: True,
        is_extend=lambda: False,
        is_target_verify=lambda: False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _lm_head():
    shard_indices = SimpleNamespace(
        padded_org_vocab_start_index=0,
        org_vocab_end_index=16,
    )
    return SimpleNamespace(
        weight=torch.empty(16, 8, dtype=torch.bfloat16),
        num_embeddings=16,
        num_added_embeddings=0,
        shard_indices=shard_indices,
        quant_method=UnquantizedEmbeddingMethod(),
    )


def test_flashsampling_accepts_temperature_only_batch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    info = get_flashsampling_info(
        hidden_states=torch.empty(2, 8, dtype=torch.bfloat16),
        lm_head=_lm_head(),
        sampling_info=_sampling_info(),
        server_args=_server_args(),
        forward_mode=_forward_mode(),
        extend_return_logprob=False,
        final_logit_softcapping=None,
        logit_scale=None,
        use_fp32_lm_head=False,
        use_attn_tp_group=False,
        do_dp_attention_lm_head_gather=False,
    )

    assert isinstance(info, FlashSamplingInfo)
    assert info.valid_vocab_size == 16


def test_flashsampling_rejects_logits_processing_batch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    info = get_flashsampling_info(
        hidden_states=torch.empty(2, 8, dtype=torch.bfloat16),
        lm_head=_lm_head(),
        sampling_info=_sampling_info(need_top_p_sampling=True),
        server_args=_server_args(),
        forward_mode=_forward_mode(),
        extend_return_logprob=False,
        final_logit_softcapping=None,
        logit_scale=None,
        use_fp32_lm_head=False,
        use_attn_tp_group=False,
        do_dp_attention_lm_head_gather=False,
    )

    assert info is None


def test_flashsampling_rejects_small_batch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    info = get_flashsampling_info(
        hidden_states=torch.empty(2, 8, dtype=torch.bfloat16),
        lm_head=_lm_head(),
        sampling_info=_sampling_info(),
        server_args=_server_args(flashsampling_min_batch_size=4),
        forward_mode=_forward_mode(),
        extend_return_logprob=False,
        final_logit_softcapping=None,
        logit_scale=None,
        use_fp32_lm_head=False,
        use_attn_tp_group=False,
        do_dp_attention_lm_head_gather=False,
    )

    assert info is None


def test_flashsampling_rejects_large_batch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    info = get_flashsampling_info(
        hidden_states=torch.empty(4, 8, dtype=torch.bfloat16),
        lm_head=_lm_head(),
        sampling_info=_sampling_info(),
        server_args=_server_args(flashsampling_max_batch_size=2),
        forward_mode=_forward_mode(),
        extend_return_logprob=False,
        final_logit_softcapping=None,
        logit_scale=None,
        use_fp32_lm_head=False,
        use_attn_tp_group=False,
        do_dp_attention_lm_head_gather=False,
    )

    assert info is None


def test_blackwell_greedy_dense_dispatch_policy():
    assert not _should_use_dense_greedy_path_on_blackwell(
        batch_size=64,
        valid_vocab_size=16160,
        hidden_size=7168,
        is_all_greedy=True,
        cc_major=10,
    )
    assert _should_use_dense_greedy_path_on_blackwell(
        batch_size=72,
        valid_vocab_size=16160,
        hidden_size=7168,
        is_all_greedy=True,
        cc_major=10,
    )
    assert not _should_use_dense_greedy_path_on_blackwell(
        batch_size=72,
        valid_vocab_size=16160,
        hidden_size=7168,
        is_all_greedy=False,
        cc_major=10,
    )
    assert not _should_use_dense_greedy_path_on_blackwell(
        batch_size=128,
        valid_vocab_size=151936,
        hidden_size=2048,
        is_all_greedy=True,
        cc_major=10,
    )


def test_flashsampling_rejects_extend_batch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    info = get_flashsampling_info(
        hidden_states=torch.empty(2, 8, dtype=torch.bfloat16),
        lm_head=_lm_head(),
        sampling_info=_sampling_info(),
        server_args=_server_args(flashsampling_fallback="error"),
        forward_mode=_forward_mode(is_decode=lambda: False, is_extend=lambda: True),
        extend_return_logprob=False,
        final_logit_softcapping=None,
        logit_scale=None,
        use_fp32_lm_head=False,
        use_attn_tp_group=False,
        do_dp_attention_lm_head_gather=False,
    )

    assert info is None


def test_flashsampling_rejects_quantized_lm_head(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    lm_head = _lm_head()
    lm_head.quant_method = object()

    info = get_flashsampling_info(
        hidden_states=torch.empty(2, 8, dtype=torch.bfloat16),
        lm_head=lm_head,
        sampling_info=_sampling_info(),
        server_args=_server_args(),
        forward_mode=_forward_mode(),
        extend_return_logprob=False,
        final_logit_softcapping=None,
        logit_scale=None,
        use_fp32_lm_head=False,
        use_attn_tp_group=False,
        do_dp_attention_lm_head_gather=False,
    )

    assert info is None


def test_flashsampling_lm_head_metadata_uses_unpadded_vocab_range():
    weight, vocab_start_index, valid_vocab_size = get_flashsampling_lm_head_metadata(
        _lm_head()
    )

    assert weight.shape == (16, 8)
    assert vocab_start_index == 0
    assert valid_vocab_size == 16


def test_flashsampling_temperature_scaling_stays_on_tensor():
    info = FlashSamplingInfo(
        hidden_states=torch.empty(2, 8),
        lm_head_weight=torch.empty(16, 8),
        vocab_start_index=0,
        valid_vocab_size=16,
        use_attn_tp_group=False,
        logit_scale=2.0,
    )
    sampling_info = _sampling_info(temperatures=torch.full((2, 1), 0.6))

    temperature = FlashSamplingRuntime._temperature_tensor(info, sampling_info)

    assert temperature.shape == torch.Size([])
    assert torch.isclose(temperature, torch.tensor(0.3))
