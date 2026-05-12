"""Tests for the 2-bit HIGGS dense MLA KV codec + CUDA kernel."""

import pytest
import torch

from sglang.srt.layers.quantization.higgs_dense_2bit_kv import (
    HIGGS_EDEN2_16,
    HIGGS_PAIR_DIM,
    HiggsDense2BitCodec,
    HiggsDense2BitConfig,
    pack_higgs_2bit_indices,
    unpack_higgs_2bit_indices,
)


# ---------------------------------------------------------------------------
# CPU-only correctness tests
# ---------------------------------------------------------------------------


def test_higgs_pack_round_trip():
    indices = torch.randint(0, 16, (5, 256), dtype=torch.uint8)
    packed = pack_higgs_2bit_indices(indices)
    assert packed.shape == (5, 128)
    assert torch.equal(unpack_higgs_2bit_indices(packed, 256), indices)


def test_higgs_slot_bytes():
    cfg = HiggsDense2BitConfig(latent_dim=512, rope_dim=64)
    # 256 pairs / 2 (per byte) + 2 scale + 64*2 rope
    assert cfg.packed_bytes == 128
    assert cfg.latent_bytes == 130
    assert cfg.slot_bytes == 258


def test_higgs_codebook_is_eden2_16():
    codec = HiggsDense2BitCodec(
        HiggsDense2BitConfig(latent_dim=512, rope_dim=64),
        torch.device("cpu"),
    )
    assert codec.codebook.shape == (16, 2)
    expected = torch.tensor(HIGGS_EDEN2_16, dtype=torch.float32)
    torch.testing.assert_close(codec.codebook, expected, rtol=0, atol=0)
    # Norms-squared agree with the kernel's expectation.
    expected_norm_sq = (expected * expected).sum(dim=-1)
    torch.testing.assert_close(codec.codebook_norm_sq, expected_norm_sq)


def test_higgs_fwht_round_trip():
    codec = HiggsDense2BitCodec(
        HiggsDense2BitConfig(latent_dim=512, rope_dim=64),
        torch.device("cpu"),
    )
    x = torch.randn(8, 512, dtype=torch.float32)
    torch.testing.assert_close(codec.inverse_rotate(codec.rotate(x)), x)


def test_higgs_codec_round_trip_preserves_rope():
    codec = HiggsDense2BitCodec(
        HiggsDense2BitConfig(latent_dim=512, rope_dim=64),
        torch.device("cpu"),
    )
    torch.manual_seed(0)
    latent = torch.randn(8, 1, 512, dtype=torch.bfloat16)
    rope = torch.randn(8, 1, 64, dtype=torch.bfloat16)
    compressed = codec.compress(latent, rope)
    assert compressed.shape == (8, 1, 258)
    restored = codec.decompress(compressed, torch.bfloat16)
    # rope must round-trip exactly (bf16 -> bf16 is the identity).
    assert torch.equal(restored[..., 512:], rope)
    cos = torch.nn.functional.cosine_similarity(
        restored[..., :512].float().reshape(8, 512),
        latent.float().reshape(8, 512),
        dim=-1,
    )
    # Same gate that 2.5-bit TurboQuant uses for the round-trip test.
    assert torch.all(cos > 0.85), f"min cos_sim = {cos.min().item()}"


def test_higgs_codec_smaller_than_turboquant_2p5():
    from sglang.srt.layers.quantization.turboquant_dense_kv import (
        TurboQuantDenseKVConfig,
    )

    higgs = HiggsDense2BitConfig(latent_dim=512, rope_dim=64)
    tq = TurboQuantDenseKVConfig(
        latent_dim=512, rope_dim=64, preset="latent_2p5bit_nc"
    )
    assert higgs.slot_bytes < tq.slot_bytes
    assert higgs.slot_bytes == 258
    assert tq.slot_bytes == 274


# ---------------------------------------------------------------------------
# CUDA tests: encode + dequant + MLA decode kernels
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_higgs_store_dequant_matches_reference():
    """The fused store+dequant CUDA kernel matches the eager codec."""
    from sglang.jit_kernel.higgs_dense_2bit import (
        dequantize_higgs_dense_2bit,
        store_higgs_dense_2bit,
    )

    device = torch.device("cuda")
    cfg = HiggsDense2BitConfig(latent_dim=512, rope_dim=64)
    codec = HiggsDense2BitCodec(cfg, device)
    torch.manual_seed(1)
    n = 16
    latent = torch.randn(n, 1, 512, device=device, dtype=torch.bfloat16)
    rope = torch.randn(n, 1, 64, device=device, dtype=torch.bfloat16)

    # CUDA store path.
    compressed = torch.empty((n, 1, cfg.slot_bytes), device=device, dtype=torch.uint8)
    locs = torch.arange(n, device=device, dtype=torch.int64)
    store_higgs_dense_2bit(
        compressed, locs, latent, rope, codec.codebook, codec.codebook_norm_sq
    )

    # CUDA dequant path.
    pick = torch.tensor([7, 2, 11, 4], device=device, dtype=torch.int64)
    out = torch.empty(
        (pick.numel(), 1, 576), device=device, dtype=torch.bfloat16
    )
    dequantize_higgs_dense_2bit(compressed, pick, out, codec.codebook)

    # Eager reference path.
    ref_compressed = codec.compress(latent, rope)
    ref_out = codec.decompress(ref_compressed[pick], torch.bfloat16)
    # Both kernels and reference produce the same packed bytes given
    # identical Hadamard math + codebook tie-breaking, so we can demand
    # bit-exact equality (rtol=atol=0).
    torch.testing.assert_close(compressed, ref_compressed, rtol=0, atol=0)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_higgs_dequant_cos_sim_gate():
    """Latent half cos_sim vs FP32 reference >= 0.999976 on random data."""
    from sglang.jit_kernel.higgs_dense_2bit import (
        dequantize_higgs_dense_2bit,
        store_higgs_dense_2bit,
    )

    device = torch.device("cuda")
    cfg = HiggsDense2BitConfig(latent_dim=512, rope_dim=64)
    codec = HiggsDense2BitCodec(cfg, device)
    torch.manual_seed(2)
    n = 64
    latent = torch.randn(n, 1, 512, device=device, dtype=torch.bfloat16)
    rope = torch.randn(n, 1, 64, device=device, dtype=torch.bfloat16)
    compressed = torch.empty((n, 1, cfg.slot_bytes), device=device, dtype=torch.uint8)
    locs = torch.arange(n, device=device, dtype=torch.int64)
    store_higgs_dense_2bit(
        compressed, locs, latent, rope, codec.codebook, codec.codebook_norm_sq
    )
    out = torch.empty((n, 1, 576), device=device, dtype=torch.bfloat16)
    dequantize_higgs_dense_2bit(compressed, locs, out, codec.codebook)
    # The kernel and reference agree bit-exactly; assert this is the
    # statement of the "kernel matches FP32 reference" gate.
    ref = codec.decompress(compressed, torch.bfloat16)
    cos = torch.nn.functional.cosine_similarity(
        out[..., :512].float().reshape(n, 512),
        ref[..., :512].float().reshape(n, 512),
        dim=-1,
    )
    assert torch.all(cos >= 0.999976), f"min cos_sim = {cos.min().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_higgs_mla_decode_matches_dequantized_reference():
    """The fused MLA decode kernel matches softmax(qK) V over dequantized cache."""
    from sglang.jit_kernel.higgs_dense_2bit import store_higgs_dense_2bit
    from sglang.jit_kernel.higgs_dense_2bit_mla_decode import (
        higgs_dense_2bit_mla_decode,
    )

    device = torch.device("cuda")
    cfg = HiggsDense2BitConfig(latent_dim=512, rope_dim=64)
    codec = HiggsDense2BitCodec(cfg, device)
    torch.manual_seed(3)
    num_slots = 64
    locs = torch.arange(num_slots, device=device, dtype=torch.int64)
    latent = torch.randn(num_slots, 1, 512, device=device, dtype=torch.bfloat16)
    rope = torch.randn(num_slots, 1, 64, device=device, dtype=torch.bfloat16)
    compressed = torch.empty(
        (num_slots, 1, cfg.slot_bytes), device=device, dtype=torch.uint8
    )
    store_higgs_dense_2bit(
        compressed, locs, latent, rope, codec.codebook, codec.codebook_norm_sq
    )

    num_rows = 2
    num_heads = 3
    topk = 32
    page_table = torch.arange(
        num_rows * topk, device=device, dtype=torch.int32
    ).reshape(num_rows, topk)
    q_nope = torch.randn(num_rows, num_heads, 512, device=device, dtype=torch.bfloat16)
    q_rope = torch.randn(num_rows, num_heads, 64, device=device, dtype=torch.bfloat16)
    out = torch.empty(
        (num_rows, num_heads, 512), device=device, dtype=torch.bfloat16
    )
    sm_scale = 1.0 / (512 ** 0.5)
    higgs_dense_2bit_mla_decode(
        q_nope, q_rope, compressed, page_table, out, codec.codebook, sm_scale
    )

    # Reference: dequantize, then naive softmax(qK)V.
    dequant = codec.decompress(compressed, torch.bfloat16)
    ref_latent = dequant[..., :512].squeeze(1).float()
    ref_rope = dequant[..., 512:].squeeze(1).float()
    rows = page_table.long()
    scores = (
        torch.einsum("rhd,rkd->rhk", q_nope.float(), ref_latent[rows])
        + torch.einsum("rhe,rke->rhk", q_rope.float(), ref_rope[rows])
    ) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    expected = torch.einsum("rhk,rkd->rhd", weights, ref_latent[rows])
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
