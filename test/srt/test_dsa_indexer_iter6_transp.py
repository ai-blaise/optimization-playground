"""Correctness regression test for the iter6 DSA NVFP4 indexer mean_pool
transposed-value variant.

Verifies that ``hisa_mean_pool_predecode_transp_indexer_cache_nvfp4``
produces reps that are bit-identical to the iter5 SECONDARY fma2 kernel
at the production shape grid.

The only structural change from iter5 SECONDARY is the SMEM value-byte
layout (transposed to [dim_byte][token-local], 8 KB extra SMEM) and the
inner-loop LDS instruction (1 LDS.b16 per pair instead of 2 LDS.b8).
Both kernels do the exact same set of fp32 multiplications and additions
in the same partition (sum0 = even-i, sum1 = odd-i, then sum0 + sum1),
so the final sum matches bit-for-bit.
"""

from __future__ import annotations

import unittest

import torch


class TestDSAIndexerIter6MeanPoolTransp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        if torch.cuda.get_device_capability(0)[0] < 9:
            raise unittest.SkipTest("SM90+ required for NVFP4 indexer build")

    def _build_case(self, batch, prefix, page=64, block=128, seed=2027):
        from sglang.jit_kernel.nvfp4_indexer import (
            fused_store_index_k_cache_nvfp4,
        )

        device = torch.device("cuda")
        torch.manual_seed(seed)
        head_dim = 128
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
        max_blocks = (prefix + block - 1) // block
        return cache, page_table, seq_lens, max_blocks

    def _run(self, batch, prefix):
        from sglang.jit_kernel.nvfp4_indexer import (
            hisa_mean_pool_predecode_fma2_indexer_cache_nvfp4,
            hisa_mean_pool_predecode_transp_indexer_cache_nvfp4,
        )

        cache, page_table, seq_lens, max_blocks = self._build_case(batch, prefix)

        ref = hisa_mean_pool_predecode_fma2_indexer_cache_nvfp4(
            cache, page_table, seq_lens, max_blocks
        )
        transp = hisa_mean_pool_predecode_transp_indexer_cache_nvfp4(
            cache, page_table, seq_lens, max_blocks
        )
        max_diff = (ref - transp).abs().max().item()
        self.assertEqual(
            ref.shape, transp.shape,
            f"shape mismatch on batch={batch} prefix={prefix}",
        )
        # iter6 PRIMARY uses identical decode/accumulation as iter5
        # SECONDARY -- only the SMEM address mapping differs. The result
        # must be bit-identical (max_diff == 0.0).
        self.assertEqual(
            max_diff,
            0.0,
            f"mean_pool transp must be bit-identical to fma2: diff "
            f"{max_diff:.2e} on batch={batch} prefix={prefix}",
        )

    def test_short_prefix(self):
        self._run(batch=4, prefix=4096)

    def test_medium_prefix(self):
        self._run(batch=8, prefix=8192)

    def test_long_prefix(self):
        self._run(batch=4, prefix=16384)

    def test_very_long_prefix(self):
        self._run(batch=4, prefix=32768)

    def test_production_grid_32(self):
        self._run(batch=32, prefix=32768)

    def test_production_grid_64(self):
        self._run(batch=64, prefix=32768)

    def test_production_grid_128(self):
        self._run(batch=128, prefix=32768)

    def test_single_token_partial_block(self):
        # Prefix that ends in the middle of a block to exercise the
        # odd-token scalar tail path in the transposed inner loop.
        self._run(batch=4, prefix=8127)  # 63 full blocks + 63 tokens

    def test_against_iter4_predecode(self):
        """iter6 PRIMARY must also be bit-identical to iter4 predecode
        (since iter5 SECONDARY was bit-identical to iter4 predecode and
        iter6 PRIMARY is bit-identical to iter5 SECONDARY)."""
        from sglang.jit_kernel.nvfp4_indexer import (
            hisa_mean_pool_predecode_indexer_cache_nvfp4,
            hisa_mean_pool_predecode_transp_indexer_cache_nvfp4,
        )

        cache, page_table, seq_lens, max_blocks = self._build_case(64, 32768)
        ref = hisa_mean_pool_predecode_indexer_cache_nvfp4(
            cache, page_table, seq_lens, max_blocks
        )
        transp = hisa_mean_pool_predecode_transp_indexer_cache_nvfp4(
            cache, page_table, seq_lens, max_blocks
        )
        # iter4 -> iter5 SECONDARY is bit-identical (by iter5 SECONDARY
        # test); iter5 SECONDARY -> iter6 PRIMARY is bit-identical (above
        # tests). Therefore iter4 -> iter6 PRIMARY must be bit-identical.
        max_diff = (ref - transp).abs().max().item()
        self.assertEqual(
            max_diff,
            0.0,
            f"iter6 transp must be bit-identical to iter4 predecode: "
            f"diff {max_diff:.2e}",
        )


if __name__ == "__main__":
    unittest.main()
