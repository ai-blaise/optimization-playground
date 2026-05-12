import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    try:
        from sglang.jit_kernel.nvfp4_indexer import (
            _hisa_block_topk_counts,
            dequantize_indexer_nvfp4,
            fused_store_index_k_cache_nvfp4,
            hisa_map_candidate_indices_indexer_cache_nvfp4,
            hisa_precompute_block_reps_indexer_cache_nvfp4,
            nvfp4_hisa_indexer_from_dequant,
            nvfp4_hisa_indexer_paged_deepgemm,
            nvfp4_hisa_indexer_paged_deepgemm_precomputed,
            nvfp4_hisa_indexer_paged_torch,
            quantize_indexer_q_nvfp4,
        )
    except ModuleNotFoundError:
        _hisa_block_topk_counts = None
        dequantize_indexer_nvfp4 = None
        fused_store_index_k_cache_nvfp4 = None
        hisa_map_candidate_indices_indexer_cache_nvfp4 = None
        hisa_precompute_block_reps_indexer_cache_nvfp4 = None
        nvfp4_hisa_indexer_from_dequant = None
        nvfp4_hisa_indexer_paged_deepgemm = None
        nvfp4_hisa_indexer_paged_deepgemm_precomputed = None
        nvfp4_hisa_indexer_paged_torch = None
        quantize_indexer_q_nvfp4 = None
else:
    _hisa_block_topk_counts = None
    dequantize_indexer_nvfp4 = None
    fused_store_index_k_cache_nvfp4 = None
    hisa_map_candidate_indices_indexer_cache_nvfp4 = None
    hisa_precompute_block_reps_indexer_cache_nvfp4 = None
    nvfp4_hisa_indexer_from_dequant = None
    nvfp4_hisa_indexer_paged_deepgemm = None
    nvfp4_hisa_indexer_paged_deepgemm_precomputed = None
    nvfp4_hisa_indexer_paged_torch = None
    quantize_indexer_q_nvfp4 = None

pytestmark = pytest.mark.skipif(
    torch is None or quantize_indexer_q_nvfp4 is None,
    reason="torch and SGLang runtime dependencies are required",
)

PREFIX_LENGTHS_BY_K = {
    2048: (
        1,
        2047,
        2048,
        2049,
        4096,
        8192,
        8193,
        16 * 1024,
        32 * 1024,
        64 * 1024,
        128 * 1024,
    ),
    1024: (
        1,
        1023,
        1024,
        1025,
        4096,
        8192,
        8193,
        16 * 1024,
        32 * 1024,
        64 * 1024,
        128 * 1024,
    ),
}


def _nvfp4_supported() -> bool:
    if torch is None:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def _dense_reference_topk(q, k, weights, prefix_len, topk):
    scores = torch.einsum("qhd,kd->qkh", q.float(), k[:prefix_len].float())
    scores = torch.relu(scores) * weights.float().unsqueeze(1)
    scores = scores.sum(dim=-1)
    keep = min(topk, prefix_len)
    out = torch.full((q.shape[0], topk), -1, dtype=torch.int32)
    out[:, :keep] = torch.topk(scores.cpu(), k=keep, dim=-1, sorted=False).indices.to(
        torch.int32
    )
    return out


def _build_case(prefix_len: int, heads: int = 8):
    torch.manual_seed(1000 + prefix_len)
    q = (torch.randn((1, heads, 128), device="cuda") * 0.25).to(torch.bfloat16)
    k = (torch.randn((prefix_len, 128), device="cuda") * 0.25).to(torch.bfloat16)
    weights = torch.randn((1, heads), device="cuda", dtype=torch.float32)
    q_values, q_scales = quantize_indexer_q_nvfp4(q)
    page_size = 64
    pages = (prefix_len + page_size - 1) // page_size
    cache = torch.zeros((pages, page_size * 68), device="cuda", dtype=torch.uint8)
    fused_store_index_k_cache_nvfp4(
        k,
        cache,
        torch.arange(prefix_len, device="cuda", dtype=torch.int64),
        page_size=page_size,
    )
    page_table = torch.arange(pages, device="cuda", dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([prefix_len], device="cuda", dtype=torch.int32)
    token_to_batch_idx = torch.zeros((1,), device="cuda", dtype=torch.int32)
    return q, weights, (q_values, q_scales), cache, page_table, seq_lens, token_to_batch_idx


def _dequant_cache(cache, prefix_len):
    k_values = cache[:, : 64 * 64].reshape(-1, 64)[:prefix_len]
    k_scales = cache[:, 64 * 64 : 64 * 68].reshape(-1, 4)[:prefix_len]
    return dequantize_indexer_nvfp4(
        k_values.cpu(), k_scales.contiguous().view(torch.int32).cpu().reshape(-1)
    )


def _sorted_valid_indices(x):
    valid = x[x >= 0]
    return torch.sort(valid).values


@pytest.mark.skipif(not _nvfp4_supported(), reason="NVFP4 requires Blackwell.")
@pytest.mark.parametrize("topk", (2048, 1024))
def test_nvfp4_hisa_defaults_to_incumbent_when_t_le_k(topk):
    for prefix_len in (1, topk - 1, topk):
        q, weights, q_fp4, cache, page_table, seq_lens, token_to_batch_idx = _build_case(
            prefix_len
        )
        assert (
            nvfp4_hisa_indexer_paged_torch(
                q_fp4,
                cache,
                page_table,
                seq_lens,
                weights,
                token_to_batch_idx,
                topk_tokens=topk,
            )
            is None
        )
        q_deq = dequantize_indexer_nvfp4(q_fp4[0].cpu(), q_fp4[1].cpu())
        ref = _dense_reference_topk(
            q_deq, _dequant_cache(cache, prefix_len), weights.cpu(), prefix_len, topk
        )
        assert ref.shape == (1, topk)
        assert int((ref >= 0).sum().item()) == prefix_len


@pytest.mark.skipif(not _nvfp4_supported(), reason="NVFP4 requires Blackwell.")
@pytest.mark.parametrize("topk", (2048, 1024))
def test_nvfp4_hisa_gpu_matches_pytorch_reference_required_prefixes(topk):
    for prefix_len in PREFIX_LENGTHS_BY_K[topk]:
        if prefix_len <= topk:
            continue
        _, weights, q_fp4, cache, page_table, seq_lens, token_to_batch_idx = _build_case(
            prefix_len
        )
        actual = nvfp4_hisa_indexer_paged_torch(
            q_fp4,
            cache,
            page_table,
            seq_lens,
            weights,
            token_to_batch_idx,
            topk_tokens=topk,
            fallback_to_dense_if_short=False,
        )
        assert actual is not None

        expected = nvfp4_hisa_indexer_from_dequant(
            dequantize_indexer_nvfp4(q_fp4[0].cpu(), q_fp4[1].cpu()),
            [_dequant_cache(cache, prefix_len)],
            weights.cpu(),
            torch.tensor([prefix_len], dtype=torch.int64),
            torch.zeros((1,), dtype=torch.int64),
            topk_tokens=topk,
            fallback_to_dense_if_short=False,
        )
        assert expected is not None
        torch.testing.assert_close(
            _sorted_valid_indices(actual.cpu()), _sorted_valid_indices(expected)
        )


def test_nvfp4_hisa_compression_ratio_4to1_block_budget_table():
    block_counts = torch.tensor((64, 128, 256, 512), dtype=torch.int32)
    selected, width = _hisa_block_topk_counts(
        block_counts,
        block_size=128,
        topk_tokens=2048,
        compression_ratio=4.0,
    )
    torch.testing.assert_close(
        selected, torch.tensor((16, 32, 64, 128), dtype=torch.int32)
    )
    assert width == 128


@pytest.mark.skipif(not _nvfp4_supported(), reason="NVFP4 requires Blackwell.")
def test_nvfp4_hisa_uses_compression_ratio_4to1_contract():
    _, weights, q_fp4, cache, page_table, seq_lens, token_to_batch_idx = _build_case(
        8193
    )
    out = nvfp4_hisa_indexer_paged_torch(
        q_fp4,
        cache,
        page_table,
        seq_lens,
        weights,
        token_to_batch_idx,
        block_size=128,
        block_topk=64,
        compression_ratio=4.0,
        topk_tokens=2048,
        fallback_to_dense_if_short=False,
    )
    assert out is not None
    assert out.shape == (1, 2048)


@pytest.mark.skipif(not _nvfp4_supported(), reason="NVFP4 requires Blackwell.")
def test_nvfp4_hisa_precomputed_reps_match_dequantized_mean_pool():
    prefix_len = 8193
    _, _, _, cache, page_table, seq_lens, _ = _build_case(prefix_len)
    _, max_blocks, reps = hisa_precompute_block_reps_indexer_cache_nvfp4(
        cache, page_table, seq_lens, return_float_reps=True
    )
    k_deq = _dequant_cache(cache, prefix_len).cuda()
    expected = []
    for block_id in range(max_blocks):
        start = block_id * 128
        end = min(start + 128, prefix_len)
        expected.append(k_deq[start:end].mean(dim=0))
    expected = torch.stack(expected, dim=0).unsqueeze(0)
    torch.testing.assert_close(reps, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _nvfp4_supported(), reason="NVFP4 requires Blackwell.")
def test_nvfp4_hisa_precomputed_deepgemm_matches_inline_deepgemm():
    pytest.importorskip("deep_gemm")
    _, weights, q_fp4, cache, page_table, seq_lens, token_to_batch_idx = _build_case(
        8193, heads=64
    )
    rep_fp4, max_blocks = hisa_precompute_block_reps_indexer_cache_nvfp4(
        cache, page_table, seq_lens
    )
    try:
        inline = nvfp4_hisa_indexer_paged_deepgemm(
            q_fp4,
            cache,
            page_table,
            seq_lens,
            weights,
            token_to_batch_idx,
            topk_tokens=2048,
            fallback_to_dense_if_short=False,
        )
        precomputed = nvfp4_hisa_indexer_paged_deepgemm_precomputed(
            q_fp4,
            cache,
            page_table,
            seq_lens,
            weights,
            token_to_batch_idx,
            rep_fp4,
            max_blocks,
            topk_tokens=2048,
            fallback_to_dense_if_short=False,
        )
    except RuntimeError as exc:
        if "PyObjectSlot" in str(exc) or "no interpreter set" in str(exc):
            pytest.skip(f"DeepGEMM ABI is not usable in this venv: {exc}")
        raise
    assert inline is not None and precomputed is not None
    torch.testing.assert_close(
        _sorted_valid_indices(precomputed.cpu()), _sorted_valid_indices(inline.cpu())
    )


@pytest.mark.skipif(not _nvfp4_supported(), reason="NVFP4 requires Blackwell.")
def test_nvfp4_hisa_map_all_candidates_matches_selected_blocks():
    top_blocks = torch.tensor([[0, 2, 63, -1]], device="cuda", dtype=torch.int32)
    prefix_lens = torch.tensor([8192], device="cuda", dtype=torch.int32)
    actual = hisa_map_candidate_indices_indexer_cache_nvfp4(
        top_blocks, prefix_lens, topk_tokens=512
    )
    expected = torch.cat(
        (
            torch.arange(0, 128, dtype=torch.int32),
            torch.arange(256, 384, dtype=torch.int32),
            torch.arange(8064, 8192, dtype=torch.int32),
            torch.full((128,), -1, dtype=torch.int32),
        )
    ).view(1, 512)
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not _nvfp4_supported(), reason="NVFP4 requires Blackwell.")
def test_nvfp4_hisa_map_all_candidates_masks_past_prefix():
    top_blocks = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    prefix_lens = torch.tensor([192], device="cuda", dtype=torch.int32)
    actual = hisa_map_candidate_indices_indexer_cache_nvfp4(
        top_blocks, prefix_lens, topk_tokens=256
    )
    expected = torch.arange(0, 256, dtype=torch.int32)
    expected[192:] = -1
    torch.testing.assert_close(actual.cpu(), expected.view(1, 256))
