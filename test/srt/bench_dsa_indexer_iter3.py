"""Microbench harness for DSA NVFP4 indexer iter3.

Measures per-call latency (us) of:
  1. hisa_candidate_score_indexer_cache_nvfp4 (iter1 baseline, per-candidate)
  2. hisa_candidate_score_tilen_indexer_cache_nvfp4 (iter2 kTileN=8)
  3. hisa_candidate_score_tilen16_indexer_cache_nvfp4 (iter2 kTileN=16)
  4. hisa_candidate_score_tilen32_indexer_cache_nvfp4 (iter3 vec4 kTileN=32)
  5. hisa_candidate_score_persistent_indexer_cache_nvfp4 (iter3 vec1)
  6. hisa_mean_pool_indexer_cache_nvfp4 (iter2 mean_pool)
  7. hisa_block_score_indexer_cache_nvfp4 (iter1 fused-head)
  8. Full per-layer pipeline (mean_pool + block_score + block_topk + candidate_score)

Shape grid matches the iter2 commit body and the prompt:
  - (batch, prefix) in [(32, 8192), (32, 16384), (64, 32768), (128, 32768)]
  - n_heads = 64, head_dim = 128, index_topk = 1024,
    hisa block_size = 128, page_size = 64, compression_ratio = 4.0

Reports us/call using CUDA events with warmup + median over many trials.

Production exact replica: see test_dsa_indexer_iter2_tilen.py for the
identical setup pipeline. This harness ONLY measures the candidate_score
kernel + the full pipeline.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import Callable

import torch


def _make_case(batch, n_heads, head_dim, prefix, page=64, block=128, seed=2027):
    """Build a one-shot input case identical to the regression test."""
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
    # Production schedule with compression_ratio=4.0
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
    return {
        "qv": qv,
        "qs": qs,
        "cache": cache,
        "page_table": page_table,
        "seq_lens": seq_lens,
        "weights": weights,
        "top_blocks": top_blocks,
        "ttb": ttb,
        "eff_block_topk": eff,
        "reps": reps,
        "block_scores": bs,
        "max_blocks": mb,
        "block_size": block,
        "page_size": page,
    }


def _bench_callable(fn: Callable, n_warmup: int = 5, n_iter: int = 50) -> float:
    """Measure fn() per-call us using CUDA events. Returns median."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    return statistics.median(times_us)


def _shape_grid():
    return [
        (32, 8192),
        (32, 16384),
        (64, 32768),
        (128, 32768),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        type=str,
        default="iter2_tilen8,iter2_tilen16,iter3_tilen32,iter3_persistent",
        help=(
            "Comma-separated list of variants to measure. Options: "
            "iter1_per_cand, iter2_tilen8, iter2_tilen16, iter3_tilen32, "
            "iter3_persistent"
        ),
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=64,
        help="Number of indexer heads.",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=128,
        help="Head dim. (fixed at 128 by kIndexerHeadDim)",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=50,
        help="Iterations per measurement.",
    )
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=10,
        help="Warmup iterations.",
    )
    parser.add_argument(
        "--full_pipe",
        action="store_true",
        help="Also measure mean_pool + block_score + candidate_score full pipeline.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1
    if torch.cuda.get_device_capability(0)[0] < 9:
        print("SM90+ required", file=sys.stderr)
        return 1

    variants = args.variants.split(",")
    print(f"# Device: {torch.cuda.get_device_name(0)}")
    print(f"# Variants: {variants}")
    print(f"# n_heads={args.n_heads}, head_dim={args.head_dim}")
    print()

    from sglang.jit_kernel import nvfp4_indexer
    from sglang.jit_kernel.nvfp4_indexer import (
        hisa_block_score_indexer_cache_nvfp4,
        hisa_candidate_score_indexer_cache_nvfp4,
        hisa_candidate_score_tilen_indexer_cache_nvfp4,
        hisa_candidate_score_tilen16_indexer_cache_nvfp4,
        hisa_mean_pool_indexer_cache_nvfp4,
    )

    has_iter3_persist = hasattr(
        nvfp4_indexer, "hisa_candidate_score_persistent_indexer_cache_nvfp4"
    )
    has_iter3_tilen32 = hasattr(
        nvfp4_indexer, "hisa_candidate_score_tilen32_indexer_cache_nvfp4"
    )

    fn_map = {
        "iter1_per_cand": hisa_candidate_score_indexer_cache_nvfp4,
        "iter2_tilen8": hisa_candidate_score_tilen_indexer_cache_nvfp4,
        "iter2_tilen16": hisa_candidate_score_tilen16_indexer_cache_nvfp4,
    }
    if has_iter3_persist:
        from sglang.jit_kernel.nvfp4_indexer import (
            hisa_candidate_score_persistent_indexer_cache_nvfp4,
        )
        fn_map["iter3_persistent"] = (
            hisa_candidate_score_persistent_indexer_cache_nvfp4
        )
    if has_iter3_tilen32:
        from sglang.jit_kernel.nvfp4_indexer import (
            hisa_candidate_score_tilen32_indexer_cache_nvfp4,
        )
        fn_map["iter3_tilen32"] = (
            hisa_candidate_score_tilen32_indexer_cache_nvfp4
        )
    missing = [v for v in variants if v not in fn_map]
    if missing:
        print(
            f"# WARN: variants not wired in nvfp4_indexer.py: {missing}; "
            "skipping"
        )
        variants = [v for v in variants if v not in missing]

    # Header
    print(
        f"{'variant':<22} {'shape':<14} {'cand_score_us':>16} "
        f"{'mean_pool_us':>14} {'block_score_us':>16} {'full_pipe_us':>14}"
    )

    for batch, prefix in _shape_grid():
        case = _make_case(batch, args.n_heads, args.head_dim, prefix)
        qv = case["qv"]
        qs = case["qs"]
        cache = case["cache"]
        page_table = case["page_table"]
        seq_lens = case["seq_lens"]
        weights = case["weights"]
        top_blocks = case["top_blocks"]
        ttb = case["ttb"]
        max_blocks = case["max_blocks"]

        # Per-variant candidate_score measurement.
        for v in variants:
            if v not in fn_map:
                continue
            cand_fn = fn_map[v]
            cand_us = _bench_callable(
                lambda fn=cand_fn: fn(
                    qv, qs, cache, page_table, seq_lens, weights, top_blocks, ttb
                ),
                n_warmup=args.n_warmup,
                n_iter=args.n_iter,
            )

            # Full pipeline: mean_pool + block_score + candidate_score (cand_fn).
            if args.full_pipe:
                mp_us = _bench_callable(
                    lambda: hisa_mean_pool_indexer_cache_nvfp4(
                        cache, page_table, seq_lens, max_blocks
                    ),
                    n_warmup=args.n_warmup,
                    n_iter=args.n_iter,
                )
                reps = hisa_mean_pool_indexer_cache_nvfp4(
                    cache, page_table, seq_lens, max_blocks
                )
                bs_us = _bench_callable(
                    lambda: hisa_block_score_indexer_cache_nvfp4(
                        qv, qs, reps, weights, seq_lens, ttb,
                        page_table_dtype=page_table.dtype,
                    ),
                    n_warmup=args.n_warmup,
                    n_iter=args.n_iter,
                )
                def _full_pipe():
                    _reps = hisa_mean_pool_indexer_cache_nvfp4(
                        cache, page_table, seq_lens, max_blocks
                    )
                    _bs = hisa_block_score_indexer_cache_nvfp4(
                        qv, qs, _reps, weights, seq_lens, ttb,
                        page_table_dtype=page_table.dtype,
                    )
                    cand_fn(
                        qv, qs, cache, page_table, seq_lens, weights, top_blocks, ttb
                    )
                pipe_us = _bench_callable(
                    _full_pipe,
                    n_warmup=args.n_warmup,
                    n_iter=args.n_iter,
                )
            else:
                mp_us = float("nan")
                bs_us = float("nan")
                pipe_us = float("nan")

            shape_str = f"{batch}/{prefix}"
            print(
                f"{v:<22} {shape_str:<14} {cand_us:>15.1f} "
                f"{mp_us:>13.1f} {bs_us:>15.1f} {pipe_us:>13.1f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
