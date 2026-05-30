"""Correctness regression test for the iter5 DSA NVFP4 indexer mean_pool
FMA-pair variant.

Verifies that ``hisa_mean_pool_predecode_fma2_indexer_cache_nvfp4``
produces reps that are bit-identical to the iter4 predecode kernel at
the production shape grid.

The only structural change from iter4 is the SMEM scales layout
(transposed [scale_group][token-local]) and an explicit 2-iter unroll
with two parallel fp32 accumulators. Both kernels do the same set of
fp32 multiplications and additions; the FMA pair just exposes more ILP
to the SM_100 schedulers. Because the accumulation order is identical
(sum0 + sum1 with sum0 accumulating even-i, sum1 accumulating odd-i),
the final sum matches bit-for-bit modulo the final sum0+sum1 ordering,
which is associative for the production magnitudes (fp32 has 24-bit
mantissa, the sum of 128 values of magnitude <= 6 * 2^7 fits well
inside that).
"""

from __future__ import annotations

import unittest

import torch


class TestDSAIndexerIter5MeanPoolFMA2(unittest.TestCase):
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
            hisa_mean_pool_predecode_indexer_cache_nvfp4,
        )

        cache, page_table, seq_lens, max_blocks = self._build_case(batch, prefix)

        ref = hisa_mean_pool_predecode_indexer_cache_nvfp4(
            cache, page_table, seq_lens, max_blocks
        )
        fma2 = hisa_mean_pool_predecode_fma2_indexer_cache_nvfp4(
            cache, page_table, seq_lens, max_blocks
        )
        max_diff = (ref - fma2).abs().max().item()
        self.assertEqual(
            ref.shape, fma2.shape,
            f"shape mismatch on batch={batch} prefix={prefix}",
        )
        self.assertLess(
            max_diff,
            1e-5,
            f"mean_pool diff {max_diff:.2e} too large for "
            f"batch={batch} prefix={prefix}",
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
        # odd-token scalar tail path in the FMA-pair loop.
        self._run(batch=4, prefix=8127)  # 63 full blocks + 63 tokens


if __name__ == "__main__":
    unittest.main()
