"""Correctness regression test for the iter7 PRIMARY persistent
block_score variant.

Verifies that ``hisa_block_score_persistent_indexer_cache_nvfp4`` is
bit-identical (modulo fp32 ordering) to the base
``hisa_block_score_indexer_cache_nvfp4`` at the production shape grid.

The two kernels do the same dot computation, same warp_sum reduction
order, same weighted reduction epilogue, and same -INFINITY padding.
The only difference is that the persistent variant predecodes the Q
tile to fp32 SMEM once per row and reuses it across all max_blocks
iters (vs the base kernel which redoes the dequant in every
(row, block_id) CTA).
"""

from __future__ import annotations

import unittest

import torch


class TestDSAIndexerIter7BlockScorePersistent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        if torch.cuda.get_device_capability(0)[0] < 9:
            raise unittest.SkipTest("SM90+ required for NVFP4 indexer build")

    def _build_case(self, batch, prefix, page=64, block=128, seed=4071):
        from sglang.jit_kernel.nvfp4_indexer import (
            _hisa_block_topk_counts,
            fused_store_index_k_cache_nvfp4,
            hisa_block_topk_indexer_cache_nvfp4,
            hisa_mean_pool_indexer_cache_nvfp4,
            quantize_indexer_q_nvfp4,
        )

        device = torch.device("cuda")
        torch.manual_seed(seed)
        head_dim = 128
        n_heads = 64
        q = (
            torch.randn(batch, n_heads, head_dim, dtype=torch.bfloat16, device=device)
            * 0.5
        )
        qv, qs = quantize_indexer_q_nvfp4(q, indices_dtype=torch.int32)
        pages = (prefix + page - 1) // page
        total = batch * pages * page
        keys = torch.randn(total, head_dim, dtype=torch.bfloat16, device=device) * 0.5
        cache = torch.zeros(
            batch * pages + 4,
            (head_dim // 2 + 4) * page,
            dtype=torch.uint8,
            device=device,
        )
        out_cache_loc = torch.arange(total, dtype=torch.int32, device=device)
        fused_store_index_k_cache_nvfp4(keys, cache, out_cache_loc, page_size=page)
        page_table = torch.arange(
            batch * pages, device=device, dtype=torch.int32
        ).reshape(batch, pages)
        seq_lens = torch.full((batch,), prefix, dtype=torch.int32, device=device)
        weights = (
            torch.randn(batch, n_heads, device=device, dtype=torch.float32) * 0.1
        )
        ttb = torch.arange(batch, device=device, dtype=torch.int32)
        mb = (prefix + block - 1) // block
        reps = hisa_mean_pool_indexer_cache_nvfp4(
            cache, page_table, seq_lens, mb
        )
        return dict(qv=qv, qs=qs, reps=reps, weights=weights, seq_lens=seq_lens,
                    ttb=ttb, page_table=page_table, mb=mb)

    def _run(self, batch, prefix):
        from sglang.jit_kernel.nvfp4_indexer import (
            hisa_block_score_indexer_cache_nvfp4,
            hisa_block_score_persistent_indexer_cache_nvfp4,
        )

        case = self._build_case(batch, prefix)
        qv = case["qv"]
        qs = case["qs"]
        reps = case["reps"]
        weights = case["weights"]
        seq_lens = case["seq_lens"]
        ttb = case["ttb"]
        page_table = case["page_table"]

        ref = hisa_block_score_indexer_cache_nvfp4(
            qv, qs, reps, weights, seq_lens, ttb,
            page_table_dtype=page_table.dtype,
        )
        persistent = hisa_block_score_persistent_indexer_cache_nvfp4(
            qv, qs, reps, weights, seq_lens, ttb,
            page_table_dtype=page_table.dtype,
        )
        # The two kernels do the same FFMAs in the same warp_sum order;
        # the only difference is that the persistent variant reads Q from
        # SMEM (predecoded once) instead of dequantizing in the inner
        # loop. The dequant -> fp32 result is identical; the dot is
        # identical; the warp_sum is identical. Expect very close to
        # bit-identical, with at most fp32-round differences due to the
        # compiler optionally reassociating the predecoded-fp32 multiply
        # vs the inline-dequant multiply.
        max_diff = (ref - persistent).abs().max().item()
        self.assertEqual(
            ref.shape, persistent.shape,
            f"shape mismatch on batch={batch} prefix={prefix}",
        )
        # Use a small tolerance for fp32 round-off only. The base kernel
        # does (NVFP4 -> fp32) * fp32 per inner iter; the persistent
        # kernel does fp32 * fp32 with the predecoded value. Both
        # paths go through the same decode_e2m1_nibble + FMA, so the
        # arithmetic should match bit-for-bit.
        self.assertLess(
            max_diff,
            1e-3,
            f"block_score persistent diff {max_diff:.2e} too large for "
            f"batch={batch} prefix={prefix}",
        )

    def test_short_prefix(self):
        self._run(batch=4, prefix=4096)

    def test_medium_prefix(self):
        self._run(batch=8, prefix=8192)

    def test_long_prefix(self):
        self._run(batch=4, prefix=16384)

    def test_production_grid_32(self):
        self._run(batch=32, prefix=32768)

    def test_production_grid_64(self):
        self._run(batch=64, prefix=32768)

    def test_partial_block_padding(self):
        # Prefix that doesn't fill all max_blocks; verifies -INFINITY
        # padding path for block_id >= block_count.
        self._run(batch=4, prefix=8127)


if __name__ == "__main__":
    unittest.main()
