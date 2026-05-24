"""Tests for the 2-bit HIGGS dense MLA KV codec + CUDA kernel."""

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.quantization.higgs_dense_2bit_kv import (
    HIGGS_EDEN2_16,
    HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV,
    HIGGS_PAIR_DIM,
    HiggsDense2BitCodec,
    HiggsDense2BitConfig,
    get_higgs_dense_2bit_b200_candidate,
    higgs_dense_2bit_b200_candidate_metadata,
    iter_higgs_dense_2bit_b200_candidates,
    pack_higgs_2bit_indices,
    select_higgs_mla_decode_num_splits,
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
    assert codec.codebook.shape == (16, HIGGS_PAIR_DIM)
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


def test_higgs_b200_candidate_registry_has_opt_in_candidates():
    candidates = iter_higgs_dense_2bit_b200_candidates(include_production=False)

    assert len(candidates) >= 4
    assert {
        "splitk_aggressive_small_batch",
        "splitk_scratch_capped",
        "splitk_ikp_stage1_balanced",
        "store_const_codebook",
        "store_const_codebook_rope_first",
        "store_const_codebook_index_pack",
        "store_const_codebook_rope_first_index_pack",
        "store_const_codebook_warp_pack",
        "store_const_codebook_warp_pack_pre_norm",
        "store_const_codebook_warp_pack_fma_score",
        "store_const_codebook_warp_pack_scale_broadcast",
        "store_const_codebook_warp_pack_rope_first",
        "store_saw_scalar2",
        "dequant_const_codebook",
        "dequant_vec4_smem_codebook",
        "dequant_vec4_ldg_codebook",
        "dequant_pair_lanes_scale_broadcast",
        "page_table_dequant_const_codebook",
        "page_table_dequant_vec4_smem_codebook",
        "page_table_dequant_vec4_ldg_codebook",
        "page_table_dequant_pair_lanes_scale_broadcast",
    }.issubset({candidate.name for candidate in candidates})
    assert all(candidate.requires_b200 for candidate in candidates)
    assert all(candidate.requires_ikp for candidate in candidates)


def test_higgs_b200_candidate_selector_defaults_to_production(monkeypatch):
    monkeypatch.delenv(HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, raising=False)

    candidate = get_higgs_dense_2bit_b200_candidate()
    assert candidate.name == "production"
    assert candidate.store_variant == "const_codebook_warp_pack"

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, "dequant_vec4_smem_codebook"
    )
    assert (
        get_higgs_dense_2bit_b200_candidate().dequant_variant
        == "vec4_smem_codebook"
    )

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, "store_const_codebook"
    )
    assert (
        get_higgs_dense_2bit_b200_candidate().store_variant == "const_codebook"
    )

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, "store_const_codebook_rope_first"
    )
    assert (
        get_higgs_dense_2bit_b200_candidate().store_variant
        == "const_codebook_rope_first"
    )

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, "store_const_codebook_rope_first_index_pack"
    )
    assert (
        get_higgs_dense_2bit_b200_candidate().store_variant
        == "const_codebook_rope_first_index_pack"
    )

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, "store_const_codebook_index_pack"
    )
    assert (
        get_higgs_dense_2bit_b200_candidate().store_variant
        == "const_codebook_index_pack"
    )

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, "store_const_codebook_warp_pack"
    )
    assert (
        get_higgs_dense_2bit_b200_candidate().store_variant
        == "const_codebook_warp_pack"
    )

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV,
        "store_const_codebook_warp_pack_pre_norm",
    )
    assert (
        get_higgs_dense_2bit_b200_candidate().store_variant
        == "const_codebook_warp_pack_pre_norm"
    )

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV,
        "store_const_codebook_warp_pack_fma_score",
    )
    assert (
        get_higgs_dense_2bit_b200_candidate().store_variant
        == "const_codebook_warp_pack_fma_score"
    )

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV,
        "store_const_codebook_warp_pack_scale_broadcast",
    )
    assert (
        get_higgs_dense_2bit_b200_candidate().store_variant
        == "const_codebook_warp_pack_scale_broadcast"
    )

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV,
        "store_const_codebook_warp_pack_rope_first",
    )
    assert (
        get_higgs_dense_2bit_b200_candidate().store_variant
        == "const_codebook_warp_pack_rope_first"
    )

    monkeypatch.setenv(HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, "store_saw_scalar2")
    candidate = get_higgs_dense_2bit_b200_candidate()
    assert candidate.store_variant == "saw_scalar2"
    assert candidate.dequant_variant == "saw_scalar2"
    assert candidate.page_table_dequant_variant == "saw_scalar2"

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, "dequant_vec4_ldg_codebook"
    )
    assert (
        get_higgs_dense_2bit_b200_candidate().dequant_variant
        == "vec4_ldg_codebook"
    )

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV,
        "dequant_pair_lanes_scale_broadcast",
    )
    assert (
        get_higgs_dense_2bit_b200_candidate().dequant_variant
        == "pair_lanes_scale_broadcast"
    )

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, "page_table_dequant_const_codebook"
    )
    assert (
        get_higgs_dense_2bit_b200_candidate().page_table_dequant_variant
        == "const_codebook"
    )

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV,
        "page_table_dequant_pair_lanes_scale_broadcast",
    )
    assert (
        get_higgs_dense_2bit_b200_candidate().page_table_dequant_variant
        == "pair_lanes_scale_broadcast"
    )


def test_higgs_b200_candidate_selector_rejects_unknown(monkeypatch):
    monkeypatch.setenv(HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, "not_a_candidate")

    with pytest.raises(ValueError, match="Unknown HIGGS dense 2-bit B200"):
        get_higgs_dense_2bit_b200_candidate()


def test_higgs_split_policy_candidates_are_opt_in(monkeypatch):
    monkeypatch.delenv(HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, raising=False)
    assert select_higgs_mla_decode_num_splits(1, 8, 4096) == 128
    assert select_higgs_mla_decode_num_splits(16, 8, 4096) == 64

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, "splitk_aggressive_small_batch"
    )
    assert select_higgs_mla_decode_num_splits(1, 8, 4096) == 160
    assert select_higgs_mla_decode_num_splits(16, 8, 4096) == 72

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, "splitk_scratch_capped"
    )
    assert select_higgs_mla_decode_num_splits(1, 8, 4096) == 64
    assert select_higgs_mla_decode_num_splits(16, 8, 4096) == 40

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, "hf_config_fixed_split64"
    )
    assert select_higgs_mla_decode_num_splits(1, 8, 512) == 64
    assert select_higgs_mla_decode_num_splits(64, 8, 4096) == 64

    monkeypatch.setenv(
        HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, "splitk_ikp_stage1_balanced"
    )
    assert select_higgs_mla_decode_num_splits(1, 8, 2048) == 80
    assert select_higgs_mla_decode_num_splits(1, 8, 4096) == 128
    assert select_higgs_mla_decode_num_splits(64, 8, 4096) == 48


def test_higgs_candidate_can_be_selected_from_hf_config(monkeypatch):
    from sglang.srt.layers.quantization.quantization_config_dispatch import (
        apply_quantization_config_dispatch,
    )

    monkeypatch.delenv(HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, raising=False)
    server_args = SimpleNamespace(
        enable_higgs_dense_2bit_kv_cache=False,
        enable_turboquant_dense_kv_cache=False,
    )
    hf_config = SimpleNamespace(
        quantization_config={
            "kv_cache_scheme": {
                "quant_method": "higgs_dense_2bit",
                "b200_candidate": "store_saw_scalar2",
            }
        }
    )

    apply_quantization_config_dispatch(server_args, hf_config)

    assert server_args.enable_higgs_dense_2bit_kv_cache
    assert get_higgs_dense_2bit_b200_candidate().name == "store_saw_scalar2"
    os.environ.pop(HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV, None)


def test_higgs_b200_candidate_metadata_is_json_ready():
    metadata = higgs_dense_2bit_b200_candidate_metadata()

    assert metadata["selector_env"] == HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV
    assert isinstance(metadata["candidates"], list)
    assert any(
        candidate["name"] == "production" and candidate["production_default"]
        for candidate in metadata["candidates"]
    )
    assert any(
        candidate["name"] == "hf_config_fixed_split64"
        and candidate["hf_config_fields"]["kv_cache_scheme.mla_decode_num_splits"]
        == 64
        for candidate in metadata["candidates"]
    )


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
