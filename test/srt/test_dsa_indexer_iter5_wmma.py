"""Correctness regression test for the iter5 DSA NVFP4 indexer WMMA
candidate_score variant (kTileN=32, mma.m16n8k8 fp32 Stage B).

Verifies that ``hisa_candidate_score_tilen32_wmma_indexer_cache_nvfp4``
produces logits and candidate_indices that match the iter3 tilen32
scalar baseline within fp32 reduction-order round-off on the production
shape grid.

Stage B is replaced by a B200 SM_100 tensor-core mma.m16n8k8 over fp16
inputs (Q and K cast from the fp32 NVFP4 decode) into an fp32
accumulator. Because the NVFP4 e2m1 / ue8m0 codebook is fully
representable in fp16 within the production dynamic range, the cast is
lossless and the dot is bit-identical to the scalar reduction modulo
accumulation order.
"""

from __future__ import annotations

import unittest

import torch


class TestDSAIndexerIter5CandidateScoreWMMA(unittest.TestCase):
    """WMMA tilen32 kernel matches scalar tilen32 modulo FP32 round-off."""

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
            hisa_candidate_score_tilen32_indexer_cache_nvfp4,
            hisa_candidate_score_tilen32_wmma_indexer_cache_nvfp4,
        )

        (
            qv, qs, cache, page_table, seq_lens, weights, top_blocks, ttb, eff
        ) = self._build_case(batch, n_heads, head_dim, prefix)

        ref_logits, ref_indices = (
            hisa_candidate_score_tilen32_indexer_cache_nvfp4(
                qv, qs, cache, page_table, seq_lens, weights, top_blocks, ttb
            )
        )
        wmma_logits, wmma_indices = (
            hisa_candidate_score_tilen32_wmma_indexer_cache_nvfp4(
                qv, qs, cache, page_table, seq_lens, weights, top_blocks, ttb
            )
        )

        # candidate_indices must match bit-for-bit (no FP arithmetic involved).
        self.assertTrue(
            torch.equal(ref_indices, wmma_indices),
            f"candidate_indices disagree on batch={batch} n_heads={n_heads} "
            f"prefix={prefix} eff_block_topk={eff}",
        )
        self.assertTrue(
            torch.equal(
                torch.isneginf(ref_logits), torch.isneginf(wmma_logits)
            ),
            f"-INFINITY pattern disagrees on batch={batch} n_heads={n_heads} "
            f"prefix={prefix}",
        )
        mask = torch.isfinite(ref_logits) & torch.isfinite(wmma_logits)
        diff = (ref_logits[mask] - wmma_logits[mask]).abs()
        base = ref_logits[mask].abs().clamp_min(1e-6)
        rel_max = (diff / base).max().item() if mask.any() else 0.0
        # 5e-3 matches the iter3 tilen32 cross-shape tolerance. Empirically
        # the WMMA dot is bit-identical at every production cell because
        # NVFP4 values are exactly representable in fp16.
        self.assertLess(
            rel_max,
            5e-3,
            f"candidate_score relative diff {rel_max:.2e} too large for "
            f"batch={batch} n_heads={n_heads} prefix={prefix}",
        )

    def test_n_heads_64_fast_path(self):
        self._run(batch=8, n_heads=64, head_dim=128, prefix=4096)

    def test_n_heads_64_candidate_path_2048(self):
        self._run(batch=4, n_heads=64, head_dim=128, prefix=8192)

    def test_n_heads_64_long_prefix_4096(self):
        self._run(batch=4, n_heads=64, head_dim=128, prefix=16384)

    def test_n_heads_64_very_long_prefix_8192(self):
        self._run(batch=4, n_heads=64, head_dim=128, prefix=32768)

    def test_n_heads_8_tp_shard(self):
        # TP-shard fallback path: kMaxHeads=8 instantiation.
        self._run(batch=4, n_heads=8, head_dim=128, prefix=4096)

    def test_n_heads_16_smc_draft(self):
        # SMC-SD draft path: kMaxHeads=16 instantiation.
        self._run(batch=4, n_heads=16, head_dim=128, prefix=4096)

    def test_batched_production_grid(self):
        # Match the iter3 perf table's largest production shape.
        self._run(batch=64, n_heads=64, head_dim=128, prefix=32768)

    def test_batched_production_grid_128(self):
        # The largest production cell where WMMA is intended to win.
        self._run(batch=128, n_heads=64, head_dim=128, prefix=32768)


class TestDSAIndexerIter5CandidateScoreWMMADispatcher(unittest.TestCase):
    """The default-off env var leaves the autotune routing unchanged; the
    opt-in env var routes large-batch n_heads<=64 cells to the WMMA kernel
    while keeping smaller cells on the scalar tilen16/8 paths."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def test_default_off_keeps_scalar(self):
        import importlib
        import os

        os.environ.pop("SGLANG_NSA_NVFP4_HISA_CAND_SCORE_WMMA", None)
        from sglang.jit_kernel import nvfp4_indexer as ni
        importlib.reload(ni)
        self.assertFalse(ni._hisa_nvfp4_candidate_score_wmma)

    def test_env_var_opts_in(self):
        import importlib
        import os

        os.environ["SGLANG_NSA_NVFP4_HISA_CAND_SCORE_WMMA"] = "1"
        try:
            from sglang.jit_kernel import nvfp4_indexer as ni
            importlib.reload(ni)
            self.assertTrue(ni._hisa_nvfp4_candidate_score_wmma)
        finally:
            os.environ.pop("SGLANG_NSA_NVFP4_HISA_CAND_SCORE_WMMA", None)
            from sglang.jit_kernel import nvfp4_indexer as ni
            importlib.reload(ni)


if __name__ == "__main__":
    unittest.main()
