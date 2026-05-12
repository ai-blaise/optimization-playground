#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from sglang.jit_kernel.nvfp4_indexer import (
    _hisa_block_topk_counts,
    dequantize_indexer_nvfp4,
    fused_store_index_k_cache_nvfp4,
    hisa_precompute_block_reps_indexer_cache_nvfp4,
    nvfp4_hisa_indexer_paged_deepgemm,
    nvfp4_hisa_indexer_paged_deepgemm_precomputed,
    nvfp4_hisa_indexer_paged_torch,
    quantize_indexer_q_nvfp4,
)


def _require_blackwell() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0):
        raise RuntimeError("NVFP4 IndexCache benchmarks require a Blackwell GPU.")


def _build_case(prefix_len: int, heads: int, query_rows: int, seed: int):
    torch.manual_seed(seed + prefix_len)
    q = (torch.randn((query_rows, heads, 128), device="cuda") * 0.25).to(
        torch.bfloat16
    )
    k = (torch.randn((prefix_len, 128), device="cuda") * 0.25).to(torch.bfloat16)
    weights = torch.randn((query_rows, heads), device="cuda", dtype=torch.float32)
    q_fp4 = quantize_indexer_q_nvfp4(q)
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
    token_to_batch_idx = torch.zeros((query_rows,), device="cuda", dtype=torch.int32)
    return q_fp4, cache, page_table, seq_lens, weights, token_to_batch_idx


def _time_cuda(fn, warmup: int, iters: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), samples


def _incumbent_dense_torch(q_fp4, cache, seq_lens, weights, topk: int):
    prefix_len = int(seq_lens[0].item())
    q = dequantize_indexer_nvfp4(q_fp4[0], q_fp4[1])
    values = cache[:, : 64 * 64].reshape(-1, 64)[:prefix_len]
    scales = cache[:, 64 * 64 : 64 * 68].reshape(-1, 4)[:prefix_len]
    k = dequantize_indexer_nvfp4(
        values, scales.contiguous().view(torch.int32).reshape(-1)
    )
    scores = torch.einsum("qhd,kd->qkh", q.float(), k.float())
    scores = torch.relu(scores) * weights.float().unsqueeze(1)
    scores = scores.sum(dim=-1)
    keep = min(topk, prefix_len)
    return torch.topk(scores, k=keep, dim=-1, sorted=False).indices


def _incumbent_deepgemm_paged(
    q_fp4, cache, page_table, seq_lens, weights, topk: int
):
    import deep_gemm

    kernel = getattr(deep_gemm, "fp8_fp4_paged_mqa_logits")
    page_size = 64
    query_rows = q_fp4[0].shape[0]
    page_table = page_table.expand(query_rows, -1).contiguous()
    max_seq_len = page_table.shape[1] * page_size
    context_lens = seq_lens.view(1, 1).expand(query_rows, -1).contiguous()
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, page_size, deep_gemm.get_num_sms()
    )
    q_values, q_scales = q_fp4
    logits = kernel(
        (q_values.view(torch.int8).unsqueeze(1), q_scales.unsqueeze(1)),
        cache.view(cache.shape[0], page_size, 1, 68),
        weights,
        context_lens,
        page_table,
        schedule_metadata,
        max_seq_len,
        clean_logits=False,
    )
    keep = min(topk, int(seq_lens[0].item()))
    return torch.topk(logits, k=keep, dim=-1, sorted=False).indices


def _summarize_counts(counts: torch.Tensor | None):
    if counts is None:
        return None
    counts_cpu = counts.cpu()
    if counts_cpu.numel() <= 16:
        return counts_cpu.tolist()
    unique, freq = torch.unique(counts_cpu, return_counts=True)
    return {
        "num_rows": int(counts_cpu.numel()),
        "min": int(counts_cpu.min().item()),
        "max": int(counts_cpu.max().item()),
        "unique": [
            {"value": int(value.item()), "count": int(count.item())}
            for value, count in zip(unique, freq)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix-lengths",
        default="1024,2048,4096,8192,8193,16384,32768,65536,131072",
    )
    parser.add_argument("--topks", default="2048,1024")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--query-rows", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hisa-candidate-scorer",
        choices=("scalar", "deepgemm", "precomputed"),
        default="scalar",
    )
    parser.add_argument(
        "--hisa-compression-ratio",
        type=float,
        default=4.0,
        help="Use 0 to benchmark the legacy fixed --hisa-block-topk budget.",
    )
    parser.add_argument("--json-out")
    args = parser.parse_args()

    _require_blackwell()
    results = []
    for topk in [int(x) for x in args.topks.split(",") if x]:
        for prefix_len in [int(x) for x in args.prefix_lengths.split(",") if x]:
            case = _build_case(prefix_len, args.heads, args.query_rows, args.seed)
            q_fp4, cache, page_table, seq_lens, weights, token_to_batch_idx = case
            deepgemm_error = None
            try:
                incumbent_ms, _ = _time_cuda(
                    lambda: _incumbent_deepgemm_paged(
                        q_fp4, cache, page_table, seq_lens, weights, topk
                    ),
                    args.warmup,
                    args.iters,
                )
            except Exception as exc:
                incumbent_ms = None
                deepgemm_error = repr(exc)
            dense_torch_ms, _ = _time_cuda(
                lambda: _incumbent_dense_torch(q_fp4, cache, seq_lens, weights, topk),
                max(1, args.warmup // 2),
                max(1, args.iters // 2),
            )
            precompute_reps_ms = None
            precomputed_reps = None
            precomputed_max_blocks = None
            prepared_prefix_lens = None
            prepared_block_counts = None
            prepared_block_starts = None
            prepared_block_ends = None
            prepared_candidate_context_lens = None
            prepared_candidate_schedule_metadata = None
            compression_ratio = (
                args.hisa_compression_ratio
                if args.hisa_compression_ratio > 0
                else None
            )
            effective_block_topk = 64
            prepared_prefix_lens = seq_lens.reshape(-1).index_select(
                0, token_to_batch_idx.long()
            ).to(torch.int32)
            prepared_block_counts = torch.div(
                prepared_prefix_lens + 127, 128, rounding_mode="floor"
            ).to(torch.int32)
            if compression_ratio is not None:
                prepared_block_topk_counts, effective_block_topk = (
                    _hisa_block_topk_counts(
                        prepared_block_counts,
                        block_size=128,
                        topk_tokens=topk,
                        compression_ratio=compression_ratio,
                    )
                )
            else:
                prepared_block_topk_counts = None
            if args.hisa_candidate_scorer == "precomputed":
                precompute_reps_ms, _ = _time_cuda(
                    lambda: hisa_precompute_block_reps_indexer_cache_nvfp4(
                        cache, page_table, seq_lens
                    ),
                    max(1, args.warmup // 2),
                    max(1, args.iters // 2),
                )
                precomputed_reps, precomputed_max_blocks = (
                    hisa_precompute_block_reps_indexer_cache_nvfp4(
                        cache, page_table, seq_lens
                    )
                )
                prepared_block_starts = (
                    token_to_batch_idx.to(torch.int32) * precomputed_max_blocks
                )
                prepared_block_ends = prepared_block_starts + prepared_block_counts
                prepared_candidate_context_lens = torch.full(
                    (q_fp4[0].shape[0], 1),
                    effective_block_topk * 128,
                    device=q_fp4[0].device,
                    dtype=torch.int32,
                )
                import deep_gemm
                prepared_candidate_schedule_metadata = (
                    deep_gemm.get_paged_mqa_logits_metadata(
                        prepared_candidate_context_lens, 64, deep_gemm.get_num_sms()
                    )
                )
            if args.hisa_candidate_scorer == "deepgemm":
                hisa_call = lambda: nvfp4_hisa_indexer_paged_deepgemm(
                    q_fp4,
                    cache,
                    page_table,
                    seq_lens,
                    weights,
                    token_to_batch_idx,
                    compression_ratio=compression_ratio,
                    topk_tokens=topk,
                    fallback_to_dense_if_short=False,
                )
            elif args.hisa_candidate_scorer == "precomputed":
                hisa_call = lambda: nvfp4_hisa_indexer_paged_deepgemm_precomputed(
                    q_fp4,
                    cache,
                    page_table,
                    seq_lens,
                    weights,
                    token_to_batch_idx,
                    precomputed_reps,
                    precomputed_max_blocks,
                    compression_ratio=compression_ratio,
                    topk_tokens=topk,
                    fallback_to_dense_if_short=False,
                    prepared_prefix_lens=prepared_prefix_lens,
                    prepared_block_counts=prepared_block_counts,
                    prepared_block_starts=prepared_block_starts,
                    prepared_block_ends=prepared_block_ends,
                    prepared_candidate_context_lens=prepared_candidate_context_lens,
                    prepared_candidate_schedule_metadata=prepared_candidate_schedule_metadata,
                )
            else:
                hisa_call = lambda: nvfp4_hisa_indexer_paged_torch(
                    q_fp4,
                    cache,
                    page_table,
                    seq_lens,
                    weights,
                    token_to_batch_idx,
                    compression_ratio=compression_ratio,
                    topk_tokens=topk,
                    fallback_to_dense_if_short=False,
                )
            hisa_ms, _ = _time_cuda(
                hisa_call,
                args.warmup,
                args.iters,
            )
            row = {
                "prefix_len": prefix_len,
                "topk": topk,
                "query_rows": args.query_rows,
                "incumbent_deepgemm_paged_ms": incumbent_ms,
                "incumbent_deepgemm_error": deepgemm_error,
                "incumbent_dense_torch_ms": dense_torch_ms,
                "hisa_nvfp4_ms": hisa_ms,
                "hisa_candidate_scorer": args.hisa_candidate_scorer,
                "hisa_compression_ratio": compression_ratio,
                "hisa_effective_block_topk": effective_block_topk,
                "hisa_block_topk_counts": _summarize_counts(
                    prepared_block_topk_counts
                ),
                "precompute_reps_ms": precompute_reps_ms,
                "speedup_vs_deepgemm": incumbent_ms / hisa_ms
                if incumbent_ms and hisa_ms
                else None,
                "speedup_vs_dense_torch": dense_torch_ms / hisa_ms
                if hisa_ms
                else None,
            }
            print(json.dumps(row, sort_keys=True))
            results.append(row)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
