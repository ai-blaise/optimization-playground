"""iter5 PRIMARY WMMA cand_score microbench.

Compares iter3 tilen32 scalar Stage B vs iter5 tilen32 WMMA Stage B at
the full production shape grid: n_heads in {8,16,64} x batch in
{1,8,16,32,64,128} x prefix in {2048,4096,8192,16384,32768}, plus
the pipeline-level mean_pool + block_score + cand_score wall time at
n_heads=64.
"""

import argparse
import os
import sys

import torch

os.environ["SGLANG_NSA_NVFP4_HISA_CAND_SCORE_WMMA"] = "1"


def _make_case(batch, n_heads, head_dim, prefix, seed=12345, page=64, block=128):
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
    q = torch.randn(batch, n_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.5
    qv, qs = quantize_indexer_q_nvfp4(q, indices_dtype=torch.int32)
    pages = (prefix + page - 1) // page
    total = batch * pages * page
    keys = torch.randn(total, head_dim, dtype=torch.bfloat16, device=device) * 0.5
    cache = torch.zeros(
        batch * pages + 4, (head_dim // 2 + 4) * page, dtype=torch.uint8, device=device
    )
    out_cache_loc = torch.arange(total, dtype=torch.int32, device=device)
    fused_store_index_k_cache_nvfp4(keys, cache, out_cache_loc, page_size=page)
    page_table = torch.arange(batch * pages, device=device, dtype=torch.int32).reshape(
        batch, pages
    )
    seq_lens = torch.full((batch,), prefix, dtype=torch.int32, device=device)
    weights = torch.randn(batch, n_heads, device=device, dtype=torch.float32) * 0.1
    ttb = torch.arange(batch, device=device, dtype=torch.int32)
    mb = (prefix + block - 1) // block
    reps = hisa_mean_pool_indexer_cache_nvfp4(cache, page_table, seq_lens, mb)
    bs = hisa_block_score_indexer_cache_nvfp4(
        qv, qs, reps, weights, seq_lens, ttb, page_table_dtype=page_table.dtype
    )
    bc = torch.div(seq_lens.to(torch.int32) + block - 1, block, rounding_mode="floor").to(
        torch.int32
    ).index_select(0, ttb.long())
    btc, eff = _hisa_block_topk_counts(
        bc, block_size=block, topk_tokens=1024, compression_ratio=4.0
    )
    top_blocks = hisa_block_topk_indexer_cache_nvfp4(
        bs, bc, block_topk=eff, block_topk_counts=btc, page_table_dtype=page_table.dtype
    )
    return dict(
        qv=qv,
        qs=qs,
        cache=cache,
        page_table=page_table,
        seq_lens=seq_lens,
        weights=weights,
        top_blocks=top_blocks,
        ttb=ttb,
        eff=eff,
        max_blocks=mb,
    )


def _bench(fn, n_warmup=8, n_iter=30):
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / n_iter  # us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_heads", type=int, default=64)
    parser.add_argument("--full_pipe", action="store_true")
    parser.add_argument(
        "--shapes",
        type=str,
        default=(
            "1/2048,1/4096,1/8192,8/4096,8/8192,16/8192,16/16384,"
            "32/8192,32/16384,32/32768,64/32768,128/32768"
        ),
    )
    parser.add_argument("--n_warmup", type=int, default=8)
    parser.add_argument("--n_iter", type=int, default=30)
    args = parser.parse_args()

    from sglang.jit_kernel.nvfp4_indexer import (
        hisa_block_score_indexer_cache_nvfp4,
        hisa_candidate_score_tilen32_indexer_cache_nvfp4,
        hisa_candidate_score_tilen32_wmma_indexer_cache_nvfp4,
        hisa_mean_pool_indexer_cache_nvfp4,
        hisa_mean_pool_predecode_indexer_cache_nvfp4,
    )

    shapes = []
    for s in args.shapes.split(","):
        b, p = s.split("/")
        shapes.append((int(b), int(p)))

    print(f"# Device: {torch.cuda.get_device_name(0)}")
    print(f"# n_heads={args.n_heads}, head_dim=128")
    print()
    print(
        f"{'shape':<14} {'scalar_us':>12} {'wmma_us':>12} {'speedup':>10}"
        f"{(' ' + 'pipe_scalar' + ' pipe_wmma  pipe_speedup') if args.full_pipe else ''}"
    )

    for batch, prefix in shapes:
        case = _make_case(batch, args.n_heads, 128, prefix)
        qv = case["qv"]
        qs = case["qs"]
        cache = case["cache"]
        page_table = case["page_table"]
        seq_lens = case["seq_lens"]
        weights = case["weights"]
        top_blocks = case["top_blocks"]
        ttb = case["ttb"]
        mb = case["max_blocks"]

        scalar_us = _bench(
            lambda: hisa_candidate_score_tilen32_indexer_cache_nvfp4(
                qv, qs, cache, page_table, seq_lens, weights, top_blocks, ttb
            ),
            n_warmup=args.n_warmup,
            n_iter=args.n_iter,
        )
        wmma_us = _bench(
            lambda: hisa_candidate_score_tilen32_wmma_indexer_cache_nvfp4(
                qv, qs, cache, page_table, seq_lens, weights, top_blocks, ttb
            ),
            n_warmup=args.n_warmup,
            n_iter=args.n_iter,
        )
        speedup = scalar_us / max(wmma_us, 1e-9)

        suffix = ""
        if args.full_pipe:
            def _pipe_scalar():
                _reps = hisa_mean_pool_predecode_indexer_cache_nvfp4(
                    cache, page_table, seq_lens, mb
                )
                _bs = hisa_block_score_indexer_cache_nvfp4(
                    qv,
                    qs,
                    _reps,
                    weights,
                    seq_lens,
                    ttb,
                    page_table_dtype=page_table.dtype,
                )
                hisa_candidate_score_tilen32_indexer_cache_nvfp4(
                    qv,
                    qs,
                    cache,
                    page_table,
                    seq_lens,
                    weights,
                    top_blocks,
                    ttb,
                )

            def _pipe_wmma():
                _reps = hisa_mean_pool_predecode_indexer_cache_nvfp4(
                    cache, page_table, seq_lens, mb
                )
                _bs = hisa_block_score_indexer_cache_nvfp4(
                    qv,
                    qs,
                    _reps,
                    weights,
                    seq_lens,
                    ttb,
                    page_table_dtype=page_table.dtype,
                )
                hisa_candidate_score_tilen32_wmma_indexer_cache_nvfp4(
                    qv,
                    qs,
                    cache,
                    page_table,
                    seq_lens,
                    weights,
                    top_blocks,
                    ttb,
                )

            pipe_s_us = _bench(_pipe_scalar, n_warmup=args.n_warmup, n_iter=args.n_iter)
            pipe_w_us = _bench(_pipe_wmma, n_warmup=args.n_warmup, n_iter=args.n_iter)
            pipe_sp = pipe_s_us / max(pipe_w_us, 1e-9)
            suffix = f" {pipe_s_us:>10.1f} {pipe_w_us:>10.1f} {pipe_sp:>13.3f}"

        label = f"{batch}/{prefix}"
        print(
            f"{label:<14} {scalar_us:>12.1f} {wmma_us:>12.1f} {speedup:>10.3f}"
            f"{suffix}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
