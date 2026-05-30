"""Correctness regression test for the iter4 DSA NVFP4 indexer
predecode-scale mean_pool kernel.

Verifies that ``hisa_mean_pool_predecode_indexer_cache_nvfp4`` produces
the same per-(batch, hisa_block, dim) representations as the iter2
``hisa_mean_pool_indexer_cache_nvfp4`` kernel modulo FP32 accumulation
order, across the production HISA shape grid.

The iter4 kernel pre-decodes the per-token scale word into an fp32 SMEM
table during the staging pass so the inner per-dim sum loop becomes a
branchless 1-byte + 1-fp32 SMEM chain. Outputs must be bit-identical
because the only change is WHERE the uint32->fp32 conversion happens —
the values fed into ``decode_e2m1_nibble`` are identical.
"""

from __future__ import annotations

import unittest

import torch


class TestDSAIndexerIter4MeanPoolPredecode(unittest.TestCase):
    """Predecode kernel == iter2 cooperative-uint4 kernel, bit-identical."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        if torch.cuda.get_device_capability(0)[0] < 9:
            raise unittest.SkipTest("SM90+ required for NVFP4 indexer build")

    def _build_cache(self, batch, prefix, page=64, head_dim=128, seed=42):
        from sglang.jit_kernel.nvfp4_indexer import (
            fused_store_index_k_cache_nvfp4,
        )

        device = torch.device("cuda")
        torch.manual_seed(seed)
        pages_per_seq = (prefix + page - 1) // page
        total = batch * pages_per_seq * page
        keys = (
            torch.randn(total, head_dim, dtype=torch.bfloat16, device=device)
            * 0.5
        )
        cache = torch.zeros(
            batch * pages_per_seq + 4,
            (head_dim // 2 + 4) * page,
            dtype=torch.uint8,
            device=device,
        )
        out_cache_loc = torch.arange(total, dtype=torch.int32, device=device)
        fused_store_index_k_cache_nvfp4(
            keys, cache, out_cache_loc, page_size=page
        )
        page_table = torch.arange(
            batch * pages_per_seq, device=device, dtype=torch.int32
        ).reshape(batch, pages_per_seq)
        seq_lens = torch.full(
            (batch,), prefix, dtype=torch.int32, device=device
        )
        return cache, page_table, seq_lens

    def _run_shape(self, batch, prefix, page=64, block=128, head_dim=128):
        from sglang.jit_kernel.nvfp4_indexer import (
            hisa_mean_pool_indexer_cache_nvfp4,
            hisa_mean_pool_predecode_indexer_cache_nvfp4,
        )

        cache, page_table, seq_lens = self._build_cache(
            batch, prefix, page=page, head_dim=head_dim
        )
        max_blocks = (prefix + block - 1) // block
        ref = hisa_mean_pool_indexer_cache_nvfp4(
            cache, page_table, seq_lens, max_blocks, page_size=page
        )
        new = hisa_mean_pool_predecode_indexer_cache_nvfp4(
            cache, page_table, seq_lens, max_blocks, page_size=page
        )
        self.assertEqual(ref.shape, new.shape)
        diff = (ref - new).abs().max().item()
        self.assertEqual(
            diff,
            0.0,
            f"predecode != iter2 at shape batch={batch}, prefix={prefix}: "
            f"max abs diff={diff}",
        )

    def test_small_shape(self):
        self._run_shape(batch=2, prefix=4096)

    def test_production_32_8192(self):
        self._run_shape(batch=32, prefix=8192)

    def test_production_32_16384(self):
        self._run_shape(batch=32, prefix=16384)

    def test_production_64_32768(self):
        self._run_shape(batch=64, prefix=32768)

    def test_short_unaligned_prefix(self):
        # 130 tokens -> one full HISA block (128 tokens) plus a tail of 2.
        # Exercises the token_count < kBlockSize branch.
        self._run_shape(batch=4, prefix=130)

    def test_single_page_prefix(self):
        # 64 tokens -> one HISA block, one page (kMaxPagesPerBlock=2 but
        # only the first page contributes; second page is invalid).
        self._run_shape(batch=4, prefix=64)

    def test_dispatch_default_predecode(self):
        """The module-level _hisa_mean_pool_call dispatch must use the
        predecode kernel by default (SGLANG_NSA_NVFP4_HISA_MEAN_POOL_PREDECODE
        defaults to True)."""
        from sglang.jit_kernel import nvfp4_indexer

        self.assertTrue(
            nvfp4_indexer._hisa_nvfp4_mean_pool_predecode,
            "iter4 predecode mean_pool must be the production default",
        )

        cache, page_table, seq_lens = self._build_cache(batch=4, prefix=2048)
        max_blocks = (2048 + 128 - 1) // 128
        ref = nvfp4_indexer.hisa_mean_pool_indexer_cache_nvfp4(
            cache, page_table, seq_lens, max_blocks
        )
        dispatched = nvfp4_indexer._hisa_mean_pool_call(
            cache, page_table, seq_lens, max_blocks
        )
        self.assertEqual((ref - dispatched).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
