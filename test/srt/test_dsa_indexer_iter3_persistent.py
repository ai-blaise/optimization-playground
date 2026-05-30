"""Correctness regression test for the iter3 DSA NVFP4 indexer
persistent-block candidate_score kernel.

Verifies that ``hisa_candidate_score_persistent_indexer_cache_nvfp4``
(kTileN=8 with each CTA processing tiles_per_split tile-blocks of one
row) produces logits and candidate_indices that match the iter2 tile-N
``hisa_candidate_score_tilen_indexer_cache_nvfp4`` kernel within FP32
reduction-order round-off on the production shape grid.

The persistent kernel is currently kept as an opt-in path
(SGLANG_NSA_NVFP4_HISA_CANDIDATE_SCORE_PERSISTENT) because at the
production shape grid it does not beat the iter2 tile-N kernel — Q HBM
amortization is no longer the bottleneck after iter2. This test
exists to guarantee numerical correctness so the kernel is safe to
revive if a future shape mix changes the trade-off.
"""

from __future__ import annotations

import unittest

import torch


class TestDSAIndexerIter3PersistentCandidateScore(unittest.TestCase):
    """Persistent kernel == iter2 tile-N kernel modulo FP32 round-off."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        if torch.cuda.get_device_capability(0)[0] < 9:
            raise unittest.SkipTest("SM90+ required for NVFP4 indexer build")

    def _build_case(self, batch, n_heads, head_dim, prefix, page=64, block=128, seed=2027):
        from sglang.jit_kernel.nvfp4_indexer import (
            _hisa_block_topk_counts,
            fused_store_index_k_cache_nvfp4,
            hisa_block_score_indexer_cache_nvfp4,
            hisa_block_topk_indexer_cache_nvfp4,
            hisa_mean_pool_indexer_cache_nvfp4,
            quantize_indexer_q_nvfp4,
        )

        device = torch.device("cuda")
        torch.manual_seed(seed)

        q = (
            torch.randn(batch, n_heads, head_dim, dtype=torch.bfloat16, device=device)
            * 0.5
        )
        qv, qs = quantize_indexer_q_nvfp4(q, indices_dtype=torch.int32)

        pages_per_seq = (prefix + page - 1) // page
        total = batch * pages_per_seq * page
        keys = (
            torch.randn(total, head_dim, dtype=torch.bfloat16, device=device) * 0.5
        )
        cache = torch.zeros(
            batch * pages_per_seq + 4,
            (head_dim // 2 + 4) * page,
            dtype=torch.uint8,
            device=device,
        )
        out_cache_loc = torch.arange(total, dtype=torch.int32, device=device)
        fused_store_index_k_cache_nvfp4(keys, cache, out_cache_loc, page_size=page)

        page_table = torch.arange(
            batch * pages_per_seq, device=device, dtype=torch.int32
        ).reshape(batch, pages_per_seq)
        seq_lens = torch.full((batch,), prefix, dtype=torch.int32, device=device)
        weights = (
            torch.randn(batch, n_heads, device=device, dtype=torch.float32) * 0.1
        )
        ttb = torch.arange(batch, device=device, dtype=torch.int32)
        mb = (prefix + block - 1) // block
        reps = hisa_mean_pool_indexer_cache_nvfp4(cache, page_table, seq_lens, mb)
        bs = hisa_block_score_indexer_cache_nvfp4(
            qv, qs, reps, weights, seq_lens, ttb, page_table_dtype=page_table.dtype
        )
        bc_per_row = torch.div(
            seq_lens.to(torch.int32) + block - 1, block, rounding_mode="floor"
        ).to(torch.int32).index_select(0, ttb.long())
        btc, eff = _hisa_block_topk_counts(
            bc_per_row, block_size=block, topk_tokens=1024, compression_ratio=4.0
        )
        top_blocks = hisa_block_topk_indexer_cache_nvfp4(
            bs,
            bc_per_row,
            block_topk=eff,
            block_topk_counts=btc,
            page_table_dtype=page_table.dtype,
        )
        return qv, qs, cache, page_table, seq_lens, weights, top_blocks, ttb, eff

    def _run(self, batch, n_heads, head_dim, prefix):
        from sglang.jit_kernel.nvfp4_indexer import (
            hisa_candidate_score_persistent_indexer_cache_nvfp4,
            hisa_candidate_score_tilen_indexer_cache_nvfp4,
        )

        (
            qv, qs, cache, page_table, seq_lens, weights, top_blocks, ttb, eff
        ) = self._build_case(batch, n_heads, head_dim, prefix)

        tilen_logits, tilen_indices = hisa_candidate_score_tilen_indexer_cache_nvfp4(
            qv, qs, cache, page_table, seq_lens, weights, top_blocks, ttb
        )
        persist_logits, persist_indices = (
            hisa_candidate_score_persistent_indexer_cache_nvfp4(
                qv, qs, cache, page_table, seq_lens, weights, top_blocks, ttb
            )
        )

        self.assertTrue(
            torch.equal(tilen_indices, persist_indices),
            f"candidate_indices disagree on batch={batch} n_heads={n_heads} "
            f"prefix={prefix} eff_block_topk={eff}",
        )
        self.assertTrue(
            torch.equal(
                torch.isneginf(tilen_logits), torch.isneginf(persist_logits)
            ),
            f"-INFINITY pattern disagrees on batch={batch} n_heads={n_heads} "
            f"prefix={prefix}",
        )
        mask = torch.isfinite(tilen_logits) & torch.isfinite(persist_logits)
        diff = (tilen_logits[mask] - persist_logits[mask]).abs()
        base = tilen_logits[mask].abs().clamp_min(1e-6)
        rel_max = (diff / base).max().item() if mask.any() else 0.0
        self.assertLess(
            rel_max,
            5e-3,
            f"candidate_score relative diff {rel_max:.2e} too large for "
            f"batch={batch} n_heads={n_heads} prefix={prefix}",
        )

    def test_n_heads_64_fast_path(self):
        # candidate_len = 8 * 128 = 1024; total_tiles = 128;
        # tiles_per_split ~ 32 → splits_per_row = 4.
        self._run(batch=8, n_heads=64, head_dim=128, prefix=4096)

    def test_n_heads_64_candidate_path_2048(self):
        self._run(batch=4, n_heads=64, head_dim=128, prefix=8192)

    def test_n_heads_64_long_prefix_4096(self):
        self._run(batch=4, n_heads=64, head_dim=128, prefix=16384)

    def test_n_heads_64_very_long_prefix_8192(self):
        self._run(batch=4, n_heads=64, head_dim=128, prefix=32768)

    def test_n_heads_8_legacy_path(self):
        # n_heads = 8 exercises heads_per_warp = 1; warp_id == head index.
        self._run(batch=4, n_heads=8, head_dim=128, prefix=4096)

    def test_batched_production_grid(self):
        # Match the iter2 perf table's largest production shape.
        self._run(batch=64, n_heads=64, head_dim=128, prefix=32768)

    def test_small_prefix_single_split(self):
        # total_tiles = candidate_len / kTileN may be <= kTilesPerCTA so
        # splits_per_row collapses to 1 — exercises the degenerate single
        # split case.
        self._run(batch=4, n_heads=64, head_dim=128, prefix=2048)


if __name__ == "__main__":
    unittest.main()
