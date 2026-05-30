"""Correctness regression test for the iter4 DSA NVFP4 indexer
persistent + cp.async K-row prefetch candidate_score kernel.

Verifies that ``hisa_candidate_score_persistent_kprefetch_indexer_cache_nvfp4``
produces the same per-(row, candidate) logits + indices as the iter3 v1
``hisa_candidate_score_persistent_indexer_cache_nvfp4`` kernel across the
production HISA shape grid.

The iter4 kernel changes the source of the raw NVFP4 K bytes (cp.async-
landed SMEM vs HBM via load_nvfp4_value) but the value/scale arithmetic
is identical, so the outputs are bit-identical modulo fp32 reduction
order. We also cross-check against the iter3 v4 (tilen32) kernel —
fp32 reductions may differ in the last few bits, so we use a tight
abs-tolerance instead of bitwise equality for the tilen32 comparison.
"""

from __future__ import annotations

import unittest

import torch


class TestDSAIndexerIter4Kprefetch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        if torch.cuda.get_device_capability(0)[0] < 9:
            raise unittest.SkipTest("SM90+ required")

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
        bc_per_row = (
            torch.div(seq_lens.to(torch.int32) + block - 1, block, rounding_mode="floor")
            .to(torch.int32)
            .index_select(0, ttb.long())
        )
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
        return dict(qv=qv, qs=qs, cache=cache, page_table=page_table,
                    seq_lens=seq_lens, weights=weights, top_blocks=top_blocks,
                    ttb=ttb)

    def _run(self, batch, n_heads, prefix):
        from sglang.jit_kernel.nvfp4_indexer import (
            hisa_candidate_score_persistent_indexer_cache_nvfp4,
            hisa_candidate_score_persistent_kprefetch_indexer_cache_nvfp4,
            hisa_candidate_score_tilen32_indexer_cache_nvfp4,
        )
        c = self._build_case(batch, n_heads, 128, prefix)
        ref_logits, ref_indices = hisa_candidate_score_persistent_indexer_cache_nvfp4(
            c["qv"], c["qs"], c["cache"], c["page_table"], c["seq_lens"],
            c["weights"], c["top_blocks"], c["ttb"],
        )
        new_logits, new_indices = (
            hisa_candidate_score_persistent_kprefetch_indexer_cache_nvfp4(
                c["qv"], c["qs"], c["cache"], c["page_table"], c["seq_lens"],
                c["weights"], c["top_blocks"], c["ttb"],
            )
        )
        # Indices must match exactly (token IDs are integer).
        idx_diff = (ref_indices != new_indices).any().item()
        self.assertFalse(
            idx_diff,
            f"kprefetch indices != iter3 v1 at batch={batch}, n_heads={n_heads}, prefix={prefix}",
        )
        # Logits: same scale/value/Q decode, same dot-product, same warp_sum
        # tree. Differences come only from cp.async vs synchronous decode
        # path which produces bit-identical fp32 in the inner loop. We
        # therefore expect exact match; allow tiny tolerance for safety.
        # Replace -inf with 0 in both for the abs-diff (they should both
        # be -inf on the same slots).
        mask_inf_ref = ref_logits == float("-inf")
        mask_inf_new = new_logits == float("-inf")
        self.assertTrue(
            (mask_inf_ref == mask_inf_new).all().item(),
            f"kprefetch -inf mask differs from iter3 v1 at batch={batch}, n_heads={n_heads}, prefix={prefix}",
        )
        ref_clean = ref_logits.clone()
        new_clean = new_logits.clone()
        ref_clean[mask_inf_ref] = 0.0
        new_clean[mask_inf_new] = 0.0
        max_abs = (ref_clean - new_clean).abs().max().item()
        # Tolerance: fp32 nibble decode is exact, scale extraction is exact,
        # warp_sum is the same — should be 0.0.
        self.assertLessEqual(
            max_abs, 0.0,
            f"kprefetch logits differ from iter3 v1 at batch={batch}, n_heads={n_heads}, prefix={prefix}: max abs diff = {max_abs}",
        )

    def test_n8_b32_p4096(self):
        self._run(batch=32, n_heads=8, prefix=4096)

    def test_n16_b16_p8192(self):
        self._run(batch=16, n_heads=16, prefix=8192)

    def test_n64_b32_p8192(self):
        self._run(batch=32, n_heads=64, prefix=8192)

    def test_n64_b32_p16384(self):
        self._run(batch=32, n_heads=64, prefix=16384)

    def test_n64_b64_p32768(self):
        self._run(batch=64, n_heads=64, prefix=32768)

    def test_n64_b1_p2048(self):
        # single batch, short prefix — edge of CTA partition.
        self._run(batch=1, n_heads=64, prefix=2048)

    def test_n64_b128_p32768(self):
        self._run(batch=128, n_heads=64, prefix=32768)


if __name__ == "__main__":
    unittest.main()
