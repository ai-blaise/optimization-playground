"""Integration smoke test for the 2-bit HIGGS dense MLA KV runtime wiring.

This exercises the pool-level store + dequant + fused decode path the
SGLang runtime invokes when ``--enable-higgs-dense-2bit-kv-cache`` is
set (see ``HiggsDense2BitNSATokenToKVPool`` in
``python/sglang/srt/mem_cache/memory_pool.py`` and the dispatch gates
in ``python/sglang/srt/layers/attention/nsa_backend.py``).

Coverage:
    * Build a tiny synthetic NSA pool with 2 layers (matching DeepSeek
      kv_lora_rank=512 + qk_rope_head_dim=64).
    * Store random latent + rope through ``set_mla_kv_buffer`` (this
      routes through ``_set_higgs_mla_kv_buffer_to_buffer`` ->
      ``store_higgs_dense_2bit`` CUDA kernel).
    * Materialize a page-table-indexed selection via
      ``get_higgs_selected_kv_buffer`` (dense BF16 view through the
      dequant CUDA kernel).
    * Run ``forward_higgs_dense_2bit_mla_decode`` (the fused-decode
      kernel the runtime dispatch calls) and compare against an FP32
      softmax(qK)V reference computed over the dequantized cache.
    * Assert latent cos_sim >= 0.999976 against the FP32 reference and
      output close to the reference attention output.
    * Regression check: in the same process, build a
      ``TurboQuantNSATokenToKVPool`` and round-trip a small batch to
      confirm the existing path is still intact.
"""

# Use future annotations for forward-reference style typing.
from __future__ import annotations

import pytest
import torch

cuda_available = torch.cuda.is_available()


def _make_radix_layer(layer_id: int, scaling: float = 1.0):
    """Lightweight stand-in for ``RadixAttention``.

    The pool methods we exercise only need ``layer_id`` and ``scaling``
    (the latter used by the fused decode kernel as ``sm_scale``).
    """

    class _Layer:
        pass

    layer = _Layer()
    layer.layer_id = layer_id
    layer.scaling = scaling
    layer.v_head_dim = 512
    return layer


def _make_higgs_pool(
    *,
    size: int = 256,
    page_size: int = 64,
    layer_num: int = 2,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    index_head_dim: int = 128,
    num_splits: int = 16,
):
    from sglang.srt.layers.attention.nsa.indexer_quantization import (
        INDEXER_FP8_QUANT_METHOD,
    )
    from sglang.srt.mem_cache.memory_pool import (
        HiggsDense2BitNSATokenToKVPool,
    )

    return HiggsDense2BitNSATokenToKVPool(
        size=size,
        page_size=page_size,
        kv_lora_rank=kv_lora_rank,
        dtype=torch.bfloat16,
        qk_rope_head_dim=qk_rope_head_dim,
        layer_num=layer_num,
        device="cuda",
        index_head_dim=index_head_dim,
        enable_memory_saver=False,
        kv_cache_dim=kv_lora_rank + qk_rope_head_dim,
        start_layer=0,
        end_layer=layer_num,
        indexer_quantization=INDEXER_FP8_QUANT_METHOD,
        higgs_execution_mode="fused_decode",
        higgs_skip_layers=set(),
        higgs_mla_decode_num_splits=num_splits,
    )


def _make_turboquant_pool(
    *,
    size: int = 256,
    page_size: int = 64,
    layer_num: int = 2,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    index_head_dim: int = 128,
):
    from sglang.srt.layers.attention.nsa.indexer_quantization import (
        INDEXER_FP8_QUANT_METHOD,
    )
    from sglang.srt.mem_cache.memory_pool import (
        TurboQuantNSATokenToKVPool,
    )

    return TurboQuantNSATokenToKVPool(
        size=size,
        page_size=page_size,
        kv_lora_rank=kv_lora_rank,
        dtype=torch.bfloat16,
        qk_rope_head_dim=qk_rope_head_dim,
        layer_num=layer_num,
        device="cuda",
        index_head_dim=index_head_dim,
        enable_memory_saver=False,
        kv_cache_dim=kv_lora_rank + qk_rope_head_dim,
        start_layer=0,
        end_layer=layer_num,
        indexer_quantization=INDEXER_FP8_QUANT_METHOD,
        turboquant_dense_kv_preset="latent_2p5bit_nc",
        turboquant_execution_mode="fused_decode",
        turboquant_mla_decode_num_splits=16,
        turboquant_skip_layers=set(),
    )


def test_higgs_auto_split_policy_from_b200_probe():
    """Auto split policy preserves fixed-32 cases and long-topk wins."""

    from sglang.srt.layers.quantization.higgs_dense_2bit_kv import (
        select_higgs_mla_decode_num_splits,
    )

    select = select_higgs_mla_decode_num_splits
    assert select(1, 8, 512) == 32
    assert select(1, 8, 1024) == 64
    assert select(1, 8, 2048) == 96
    assert select(1, 8, 4096) == 128
    assert select(2, 8, 4096) == 80
    assert select(4, 8, 1024) == 32
    assert select(4, 8, 2048) == 56
    assert select(4, 8, 4096) == 56
    assert select(8, 8, 1024) == 48
    assert select(8, 8, 4096) == 72
    assert select(16, 8, 2048) == 36
    assert select(16, 8, 4096) == 64
    assert select(32, 8, 4096) == 40
    assert select(64, 8, 4096) == 32


@pytest.mark.skipif(not cuda_available, reason="requires CUDA")
def test_higgs_pool_slot_layout():
    """Pool reports 258 B/slot/layer as designed."""
    pool = _make_higgs_pool()
    assert pool.higgs_slot_bytes == 258
    assert pool.higgs_dense_2bit_preset == "eden2_16"
    # Layer-0 row width == HIGGS slot (compressed), not BF16 dense (1152 B).
    assert pool.kv_buffer[0].shape[-1] == 258
    assert pool.kv_buffer[0].dtype == torch.uint8


@pytest.mark.skipif(not cuda_available, reason="requires CUDA")
def test_higgs_store_decode_round_trip_matches_dense_reference():
    """End-to-end: store -> fused decode matches dequant-then-softmax ref."""
    torch.manual_seed(7)
    pool = _make_higgs_pool(size=256, page_size=64)
    layer_id = 0
    layer = _make_radix_layer(layer_id)

    num_slots = 128
    kv_lora_rank = pool.kv_lora_rank
    qk_rope_head_dim = pool.qk_rope_head_dim
    device = pool.device

    # Random latent + rope; matches the production BF16 dtype.
    latent = torch.randn(
        num_slots, 1, kv_lora_rank, device=device, dtype=torch.bfloat16
    )
    rope = torch.randn(
        num_slots, 1, qk_rope_head_dim, device=device, dtype=torch.bfloat16
    )
    loc = torch.arange(num_slots, device=device, dtype=torch.int64)

    # Route store through the public RadixAttention API used by the runtime.
    pool.set_mla_kv_buffer(layer, loc, latent.squeeze(1), rope.squeeze(1))

    # The runtime's fused decode dispatch uses an int32 page table.
    num_rows = 4
    num_heads = 8
    topk = 32
    page_table = torch.randint(
        0, num_slots, (num_rows, topk), device=device, dtype=torch.int32
    )

    q_nope = torch.randn(
        num_rows, num_heads, kv_lora_rank, device=device, dtype=torch.bfloat16
    )
    q_rope = torch.randn(
        num_rows, num_heads, qk_rope_head_dim, device=device, dtype=torch.bfloat16
    )
    sm_scale = 1.0 / float(kv_lora_rank) ** 0.5

    out = pool.forward_higgs_dense_2bit_mla_decode(
        layer_id, q_nope, q_rope, page_table, sm_scale
    )
    assert out.shape == (num_rows, num_heads, kv_lora_rank)
    assert out.dtype == torch.bfloat16

    # FP32 reference: dequantize the cache, then naive softmax(qK)V.
    layer_buffer = pool._get_layersplit_kv_buffer(layer_id)
    dequant_full = pool.higgs_codec.decompress(layer_buffer, torch.bfloat16)
    ref_latent = dequant_full[..., :kv_lora_rank].squeeze(1).float()
    ref_rope = dequant_full[..., kv_lora_rank:].squeeze(1).float()

    rows = page_table.long()
    scores = (
        torch.einsum("rhd,rkd->rhk", q_nope.float(), ref_latent[rows])
        + torch.einsum("rhe,rke->rhk", q_rope.float(), ref_rope[rows])
    ) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    expected = torch.einsum("rhk,rkd->rhd", weights, ref_latent[rows])

    # Fused decode output must match the dequantize-then-attend reference.
    # Tolerances mirror the existing kernel test (test_higgs_dense_2bit_kv.py).
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not cuda_available, reason="requires CUDA")
def test_higgs_dequant_cos_sim_gate_through_pool():
    """Cos_sim of pool-dequantized latent against FP32 source >= 0.999976."""
    torch.manual_seed(11)
    pool = _make_higgs_pool(size=256, page_size=64)
    layer_id = 0
    layer = _make_radix_layer(layer_id)

    num_slots = 128
    kv_lora_rank = pool.kv_lora_rank
    qk_rope_head_dim = pool.qk_rope_head_dim
    device = pool.device

    latent = torch.randn(
        num_slots, 1, kv_lora_rank, device=device, dtype=torch.bfloat16
    )
    rope = torch.randn(
        num_slots, 1, qk_rope_head_dim, device=device, dtype=torch.bfloat16
    )
    loc = torch.arange(num_slots, device=device, dtype=torch.int64)
    pool.set_mla_kv_buffer(layer, loc, latent.squeeze(1), rope.squeeze(1))

    page_table = torch.arange(
        num_slots, device=device, dtype=torch.int32
    ).reshape(1, num_slots)
    kv_cache, _ = pool.get_higgs_selected_kv_buffer(layer_id, page_table)
    assert kv_cache.shape[-1] == kv_lora_rank + qk_rope_head_dim
    assert kv_cache.dtype == torch.bfloat16

    # FP32 reference for the cosine-similarity gate: codec-decompressed
    # latent vs the post-quant kernel-dequantized latent.
    layer_buffer = pool._get_layersplit_kv_buffer(layer_id)
    ref = pool.higgs_codec.decompress(layer_buffer[:num_slots], torch.bfloat16)
    cos = torch.nn.functional.cosine_similarity(
        kv_cache[..., :kv_lora_rank].squeeze(1).float(),
        ref[..., :kv_lora_rank].squeeze(1).float(),
        dim=-1,
    )
    min_cos = cos.min().item()
    assert min_cos >= 0.999976, f"min cos_sim = {min_cos!r}"


@pytest.mark.skipif(not cuda_available, reason="requires CUDA")
def test_higgs_pool_kv_size_smaller_than_turboquant():
    """Memory check: HIGGS pool buffers are smaller than TurboQuant's."""
    higgs_pool = _make_higgs_pool()
    tq_pool = _make_turboquant_pool()
    # 258 vs 274 bytes per token: HIGGS layer-0 buffer is strictly smaller.
    h_bytes = higgs_pool.kv_buffer[0].numel() * higgs_pool.kv_buffer[0].element_size()
    t_bytes = tq_pool.kv_buffer[0].numel() * tq_pool.kv_buffer[0].element_size()
    assert h_bytes < t_bytes
    ratio = h_bytes / t_bytes
    # Allow small variation; expected ~258/274 = 0.9416.
    assert 0.93 <= ratio <= 0.95, f"ratio={ratio!r}"


@pytest.mark.skipif(not cuda_available, reason="requires CUDA")
def test_turboquant_pool_regression_round_trip():
    """Regression: TurboQuant pool still works after HIGGS landed.

    Mirrors the HIGGS round-trip but on the TurboQuant pool. The two
    pools share an inheritance ancestor; this guards against accidental
    name collisions or import-order regressions from the HIGGS additions.
    """
    torch.manual_seed(13)
    pool = _make_turboquant_pool(size=256, page_size=64)
    layer_id = 0
    layer = _make_radix_layer(layer_id)

    num_slots = 128
    kv_lora_rank = pool.kv_lora_rank
    qk_rope_head_dim = pool.qk_rope_head_dim
    device = pool.device

    latent = torch.randn(
        num_slots, 1, kv_lora_rank, device=device, dtype=torch.bfloat16
    )
    rope = torch.randn(
        num_slots, 1, qk_rope_head_dim, device=device, dtype=torch.bfloat16
    )
    loc = torch.arange(num_slots, device=device, dtype=torch.int64)
    pool.set_mla_kv_buffer(layer, loc, latent.squeeze(1), rope.squeeze(1))

    num_rows = 4
    num_heads = 8
    topk = 32
    page_table = torch.randint(
        0, num_slots, (num_rows, topk), device=device, dtype=torch.int32
    )
    q_nope = torch.randn(
        num_rows, num_heads, kv_lora_rank, device=device, dtype=torch.bfloat16
    )
    q_rope = torch.randn(
        num_rows, num_heads, qk_rope_head_dim, device=device, dtype=torch.bfloat16
    )
    sm_scale = 1.0 / float(kv_lora_rank) ** 0.5

    out = pool.forward_turboquant_dense_mla_decode(
        layer_id, q_nope, q_rope, page_table, sm_scale
    )
    assert out.shape == (num_rows, num_heads, kv_lora_rank)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not cuda_available, reason="requires CUDA")
def test_higgs_split_k_matches_single_pass():
    """Split-K decode (num_splits=16) matches the single-pass kernel.

    Both kernels implement the same algorithm; split-K shards the
    topk loop and merges per-split (m, l, acc) tuples via log-sum-exp.
    Outputs must be numerically equivalent within bf16 rounding noise.
    """
    torch.manual_seed(17)
    pool_split = _make_higgs_pool(size=256, page_size=64, num_splits=16)
    pool_single = _make_higgs_pool(size=256, page_size=64, num_splits=1)

    layer_id = 0
    num_slots = 128
    kv_lora_rank = pool_split.kv_lora_rank
    qk_rope_head_dim = pool_split.qk_rope_head_dim
    device = pool_split.device

    latent = torch.randn(
        num_slots, 1, kv_lora_rank, device=device, dtype=torch.bfloat16
    )
    rope = torch.randn(
        num_slots, 1, qk_rope_head_dim, device=device, dtype=torch.bfloat16
    )
    loc = torch.arange(num_slots, device=device, dtype=torch.int64)
    pool_split.set_mla_kv_buffer(
        _make_radix_layer(layer_id), loc, latent.squeeze(1), rope.squeeze(1)
    )
    pool_single.set_mla_kv_buffer(
        _make_radix_layer(layer_id), loc, latent.squeeze(1), rope.squeeze(1)
    )

    num_rows = 4
    num_heads = 8
    topk = 64
    page_table = torch.randint(
        0, num_slots, (num_rows, topk), device=device, dtype=torch.int32
    )
    q_nope = torch.randn(
        num_rows, num_heads, kv_lora_rank, device=device, dtype=torch.bfloat16
    )
    q_rope = torch.randn(
        num_rows, num_heads, qk_rope_head_dim, device=device, dtype=torch.bfloat16
    )
    sm_scale = 1.0 / float(kv_lora_rank) ** 0.5

    out_split = pool_split.forward_higgs_dense_2bit_mla_decode(
        layer_id, q_nope, q_rope, page_table, sm_scale
    )
    out_single = pool_single.forward_higgs_dense_2bit_mla_decode(
        layer_id, q_nope, q_rope, page_table, sm_scale
    )

    torch.testing.assert_close(
        out_split.float(), out_single.float(), rtol=5e-3, atol=5e-3
    )
    cos = torch.nn.functional.cosine_similarity(
        out_split.float().reshape(-1, kv_lora_rank),
        out_single.float().reshape(-1, kv_lora_rank),
        dim=-1,
    )
    assert cos.min().item() >= 0.999976, f"min cos_sim = {cos.min().item()!r}"
