#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.nsa.bench_nvfp4_hisa_indexer import (
    _build_case,
    _hisa_block_topk_counts,
    _indexcache_nvfp4_paged,
    _require_blackwell,
    _time_cuda,
)
from sglang.jit_kernel.nvfp4_indexer import (
    hisa_block_topk_indexer_cache_nvfp4,
    hisa_block_topk_map_all_indexer_cache_nvfp4,
    hisa_candidate_pages_indexer_cache_nvfp4,
    hisa_candidate_keys_topk_map_indexer_cache_nvfp4,
    hisa_deepgemm_candidate_keys_indexer_cache_nvfp4,
    hisa_deepgemm_candidate_keys_row_split_from_blocks_indexer_cache_nvfp4,
    hisa_deepgemm_candidate_topk_cooperative_from_blocks_indexer_cache_nvfp4,
    hisa_deepgemm_candidate_topk_cluster_from_blocks_indexer_cache_nvfp4,
    hisa_deepgemm_candidate_logits_indexer_cache_nvfp4,
    hisa_fused_mask_topk_map_indexer_cache_nvfp4,
    hisa_precompute_block_reps_indexer_cache_nvfp4,
    hisa_map_candidate_indices_indexer_cache_nvfp4,
    hisa_selector_parallel_score_candidates_indexer_cache_nvfp4,
    hisa_selector_parallel_select_blocks_indexer_cache_nvfp4,
    nvfp4_hisa_indexer_paged_cluster_fused_precomputed,
    nvfp4_hisa_indexer_paged_collective_key_precomputed,
    nvfp4_hisa_indexer_paged_deepgemm_precomputed,
    nvfp4_hisa_indexer_paged_megakernel_precomputed,
    nvfp4_hisa_indexer_paged_parallel_collective_precomputed,
    nvfp4_hisa_indexer_paged_parallel_megakernel_precomputed,
)


def _ints_csv(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def _median_cuda(fn: Callable[[], object], warmup: int, iters: int) -> float:
    median, _ = _time_cuda(fn, warmup, iters)
    return median


def _tensor_set_equal_fraction(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return 0.0
    left_sorted = torch.sort(left.to(torch.int32), dim=1).values
    right_sorted = torch.sort(right.to(torch.int32), dim=1).values
    return float((left_sorted == right_sorted).all(dim=1).float().mean().item())


def _prefix_exact_fraction(left: torch.Tensor, right: torch.Tensor, limit: int) -> float:
    if left.shape[0] != right.shape[0]:
        return 0.0
    k = min(limit, left.shape[1], right.shape[1])
    return float((left[:, :k] == right[:, :k]).all(dim=1).float().mean().item())


def _assert_candidate_keys_topk_map_smoke() -> None:
    keys = torch.tensor(
        [[1, 2, 3, 4] + [-1] * 124],
        device="cuda",
        dtype=torch.int32,
    )
    blocks = torch.tensor([[5]], device="cuda", dtype=torch.int32)
    out = hisa_candidate_keys_topk_map_indexer_cache_nvfp4(
        keys,
        blocks,
        4,
        page_table_dtype=torch.int32,
    )
    torch.cuda.synchronize()
    expected = torch.tensor([[640, 641, 642, 643]], dtype=torch.int32)
    if not torch.equal(out.cpu(), expected):
        raise RuntimeError(
            "candidate key topk/map smoke failed: "
            f"got {out.cpu().tolist()}, expected {expected.tolist()}"
        )


def _prepare_hisa(
    q_fp4,
    cache,
    page_table,
    seq_lens,
    token_to_batch_idx,
    topk: int,
    compression_ratio: float,
    warmup: int,
    iters: int,
):
    precompute_ms = _median_cuda(
        lambda: hisa_precompute_block_reps_indexer_cache_nvfp4(
            cache, page_table, seq_lens
        ),
        max(1, warmup // 2),
        max(1, iters // 2),
    )
    reps, max_blocks = hisa_precompute_block_reps_indexer_cache_nvfp4(
        cache, page_table, seq_lens
    )
    prefix_lens = seq_lens.reshape(-1).index_select(
        0, token_to_batch_idx.long()
    ).to(torch.int32)
    block_counts = torch.div(prefix_lens + 127, 128, rounding_mode="floor").to(
        torch.int32
    )
    block_topk_counts, effective_block_topk = _hisa_block_topk_counts(
        block_counts,
        block_size=128,
        topk_tokens=topk,
        compression_ratio=compression_ratio,
    )
    block_starts = token_to_batch_idx.to(torch.int32) * max_blocks
    candidate_len = effective_block_topk * 128

    import deep_gemm

    candidate_context_lens = torch.full(
        (q_fp4[0].shape[0], 1),
        candidate_len,
        device=q_fp4[0].device,
        dtype=torch.int32,
    )
    candidate_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        candidate_context_lens, 64, deep_gemm.get_num_sms()
    )
    return {
        "precompute_ms": precompute_ms,
        "reps": reps,
        "max_blocks": max_blocks,
        "prefix_lens": prefix_lens,
        "block_counts": block_counts,
        "block_topk_counts": block_topk_counts,
        "effective_block_topk": effective_block_topk,
        "candidate_len": candidate_len,
        "block_starts": block_starts,
        "block_ends": block_starts + block_counts,
        "candidate_context_lens": candidate_context_lens,
        "candidate_schedule_metadata": candidate_schedule_metadata,
    }


def _parallel_selector_call(
    q_fp4,
    cache,
    page_table,
    seq_lens,
    weights,
    token_to_batch_idx,
    prepared,
    topk: int,
    compression_ratio: float,
):
    return nvfp4_hisa_indexer_paged_deepgemm_precomputed(
        q_fp4,
        cache,
        page_table,
        seq_lens,
        weights,
        token_to_batch_idx,
        prepared["reps"],
        prepared["max_blocks"],
        compression_ratio=compression_ratio,
        topk_tokens=topk,
        fallback_to_dense_if_short=False,
        prepared_prefix_lens=prepared["prefix_lens"],
        prepared_block_counts=prepared["block_counts"],
        prepared_block_starts=prepared["block_starts"],
        prepared_block_ends=prepared["block_ends"],
        prepared_candidate_context_lens=prepared["candidate_context_lens"],
        prepared_candidate_schedule_metadata=prepared["candidate_schedule_metadata"],
        prepared_block_topk_counts=prepared["block_topk_counts"],
        prepared_effective_block_topk=prepared["effective_block_topk"],
    )


def _serial_probe_call(
    q_fp4,
    cache,
    page_table,
    seq_lens,
    weights,
    token_to_batch_idx,
    prepared,
    topk: int,
    compression_ratio: float,
):
    return nvfp4_hisa_indexer_paged_megakernel_precomputed(
        q_fp4,
        cache,
        page_table,
        seq_lens,
        weights,
        token_to_batch_idx,
        prepared["reps"],
        prepared["max_blocks"],
        compression_ratio=compression_ratio,
        topk_tokens=topk,
        fallback_to_dense_if_short=False,
        prepared_prefix_lens=prepared["prefix_lens"],
        prepared_block_counts=prepared["block_counts"],
        prepared_block_topk_counts=prepared["block_topk_counts"],
        prepared_effective_block_topk=prepared["effective_block_topk"],
    )


def _parallel_probe_call(
    q_fp4,
    cache,
    page_table,
    seq_lens,
    weights,
    token_to_batch_idx,
    prepared,
    topk: int,
    compression_ratio: float,
):
    return nvfp4_hisa_indexer_paged_parallel_megakernel_precomputed(
        q_fp4,
        cache,
        page_table,
        seq_lens,
        weights,
        token_to_batch_idx,
        prepared["reps"],
        prepared["max_blocks"],
        compression_ratio=compression_ratio,
        topk_tokens=topk,
        fallback_to_dense_if_short=False,
        prepared_prefix_lens=prepared["prefix_lens"],
        prepared_block_counts=prepared["block_counts"],
        prepared_block_topk_counts=prepared["block_topk_counts"],
        prepared_effective_block_topk=prepared["effective_block_topk"],
    )


def _cluster_fused_probe_call(
    q_fp4,
    cache,
    page_table,
    seq_lens,
    weights,
    token_to_batch_idx,
    prepared,
    topk: int,
    compression_ratio: float,
):
    return nvfp4_hisa_indexer_paged_cluster_fused_precomputed(
        q_fp4,
        cache,
        page_table,
        seq_lens,
        weights,
        token_to_batch_idx,
        prepared["reps"],
        prepared["max_blocks"],
        compression_ratio=compression_ratio,
        topk_tokens=topk,
        fallback_to_dense_if_short=False,
        prepared_prefix_lens=prepared["prefix_lens"],
        prepared_block_counts=prepared["block_counts"],
        prepared_block_topk_counts=prepared["block_topk_counts"],
        prepared_effective_block_topk=prepared["effective_block_topk"],
    )


def _parallel_collective_probe_call(
    q_fp4,
    cache,
    page_table,
    seq_lens,
    weights,
    token_to_batch_idx,
    prepared,
    topk: int,
    compression_ratio: float,
):
    return nvfp4_hisa_indexer_paged_parallel_collective_precomputed(
        q_fp4,
        cache,
        page_table,
        seq_lens,
        weights,
        token_to_batch_idx,
        prepared["reps"],
        prepared["max_blocks"],
        compression_ratio=compression_ratio,
        topk_tokens=topk,
        fallback_to_dense_if_short=False,
        prepared_prefix_lens=prepared["prefix_lens"],
        prepared_block_counts=prepared["block_counts"],
        prepared_block_topk_counts=prepared["block_topk_counts"],
        prepared_effective_block_topk=prepared["effective_block_topk"],
        prepared_candidate_context_lens=prepared["candidate_context_lens"],
        prepared_candidate_schedule_metadata=prepared["candidate_schedule_metadata"],
    )


def _collective_key_probe_call(
    q_fp4,
    cache,
    page_table,
    seq_lens,
    weights,
    token_to_batch_idx,
    prepared,
    topk: int,
    compression_ratio: float,
):
    return nvfp4_hisa_indexer_paged_collective_key_precomputed(
        q_fp4,
        cache,
        page_table,
        seq_lens,
        weights,
        token_to_batch_idx,
        prepared["reps"],
        prepared["max_blocks"],
        compression_ratio=compression_ratio,
        topk_tokens=topk,
        fallback_to_dense_if_short=False,
        prepared_prefix_lens=prepared["prefix_lens"],
        prepared_block_counts=prepared["block_counts"],
        prepared_block_starts=prepared["block_starts"],
        prepared_block_ends=prepared["block_ends"],
        prepared_candidate_context_lens=prepared["candidate_context_lens"],
        prepared_candidate_schedule_metadata=prepared["candidate_schedule_metadata"],
        prepared_block_topk_counts=prepared["block_topk_counts"],
        prepared_effective_block_topk=prepared["effective_block_topk"],
    )


def _stage_microbench(
    q_fp4,
    cache,
    page_table,
    weights,
    token_to_batch_idx,
    prepared,
    topk: int,
    compression_ratio: float,
    warmup: int,
    iters: int,
) -> dict:
    import deep_gemm

    q_values, q_scales = q_fp4
    rep_values, rep_scales = prepared["reps"]

    def block_score_call():
        return deep_gemm.fp8_fp4_mqa_logits(
            (q_values.view(torch.int8), q_scales),
            (rep_values.view(torch.int8), rep_scales),
            weights,
            prepared["block_starts"].to(torch.int32),
            prepared["block_ends"].to(torch.int32),
            clean_logits=False,
            max_seqlen_k=prepared["max_blocks"],
            logits_dtype=torch.float32,
        )

    block_score_ms = _median_cuda(block_score_call, warmup, iters)
    block_scores = block_score_call()

    if prepared["candidate_len"] == topk:
        def block_topk_map_all_call():
            return hisa_block_topk_map_all_indexer_cache_nvfp4(
                block_scores,
                prepared["block_counts"],
                block_topk=prepared["effective_block_topk"],
                block_topk_counts=prepared["block_topk_counts"],
                prefix_lens=prepared["prefix_lens"],
                page_table_dtype=page_table.dtype,
            )

        block_topk_map_all_ms = _median_cuda(block_topk_map_all_call, warmup, iters)
        return {
            "block_score_ms": block_score_ms,
            "block_topk_map_all_ms": block_topk_map_all_ms,
            "stage_sum_ms": block_score_ms + block_topk_map_all_ms,
        }

    def block_topk_call():
        return hisa_block_topk_indexer_cache_nvfp4(
            block_scores,
            prepared["block_counts"],
            block_topk=prepared["effective_block_topk"],
            block_topk_counts=prepared["block_topk_counts"],
            page_table_dtype=page_table.dtype,
        )

    block_topk_ms = _median_cuda(block_topk_call, warmup, iters)
    top_blocks = block_topk_call()

    def candidate_pages_call():
        return hisa_candidate_pages_indexer_cache_nvfp4(
            top_blocks, page_table, token_to_batch_idx
        )

    candidate_pages_ms = _median_cuda(candidate_pages_call, warmup, iters)
    candidate_page_table = candidate_pages_call()

    def candidate_logits_call():
        return deep_gemm.fp8_fp4_paged_mqa_logits(
            (q_values.view(torch.int8).unsqueeze(1), q_scales.unsqueeze(1)),
            cache.view(cache.shape[0], 64, 1, 68),
            weights,
            prepared["candidate_context_lens"],
            candidate_page_table,
            prepared["candidate_schedule_metadata"],
            prepared["candidate_len"],
            clean_logits=False,
            logits_dtype=torch.float32,
        )

    candidate_logits_ms = _median_cuda(candidate_logits_call, warmup, iters)
    logits = candidate_logits_call()

    def fused_mask_topk_map_call():
        return hisa_fused_mask_topk_map_indexer_cache_nvfp4(
            logits,
            top_blocks,
            prepared["prefix_lens"],
            min(topk, prepared["candidate_len"]),
            page_table_dtype=page_table.dtype,
        )

    fused_mask_topk_map_ms = _median_cuda(fused_mask_topk_map_call, warmup, iters)
    return {
        "block_score_ms": block_score_ms,
        "block_topk_ms": block_topk_ms,
        "candidate_pages_ms": candidate_pages_ms,
        "candidate_logits_ms": candidate_logits_ms,
        "fused_mask_topk_map_ms": fused_mask_topk_map_ms,
        "stage_sum_ms": sum(
            [
                block_score_ms,
                block_topk_ms,
                candidate_pages_ms,
                candidate_logits_ms,
                fused_mask_topk_map_ms,
            ]
        ),
    }


def _parallel_probe_stage_microbench(
    q_fp4,
    cache,
    page_table,
    seq_lens,
    weights,
    token_to_batch_idx,
    prepared,
    topk: int,
    warmup: int,
    iters: int,
) -> dict:
    def select_blocks_call():
        return hisa_selector_parallel_select_blocks_indexer_cache_nvfp4(
            q_fp4,
            seq_lens,
            weights,
            token_to_batch_idx,
            prepared["reps"],
            prepared["max_blocks"],
            prepared["block_counts"],
            prepared["block_topk_counts"],
            prepared["effective_block_topk"],
            page_table_dtype=page_table.dtype,
        )

    select_blocks_ms = _median_cuda(select_blocks_call, warmup, iters)
    top_blocks = select_blocks_call()

    if prepared["candidate_len"] == topk:
        def map_all_call():
            return hisa_map_candidate_indices_indexer_cache_nvfp4(
                top_blocks,
                prepared["prefix_lens"],
                topk,
                page_table_dtype=page_table.dtype,
            )

        map_all_ms = _median_cuda(map_all_call, warmup, iters)
        return {
            "select_blocks_ms": select_blocks_ms,
            "map_all_ms": map_all_ms,
            "stage_sum_ms": select_blocks_ms + map_all_ms,
        }

    def score_candidates_call():
        return hisa_selector_parallel_score_candidates_indexer_cache_nvfp4(
            q_fp4,
            cache,
            page_table,
            seq_lens,
            weights,
            token_to_batch_idx,
            top_blocks,
        )

    score_candidates_ms = _median_cuda(score_candidates_call, warmup, iters)
    logits = score_candidates_call()

    def fused_mask_topk_map_call():
        return hisa_fused_mask_topk_map_indexer_cache_nvfp4(
            logits,
            top_blocks,
            prepared["prefix_lens"],
            min(topk, prepared["candidate_len"]),
            page_table_dtype=page_table.dtype,
        )

    fused_mask_topk_map_ms = _median_cuda(fused_mask_topk_map_call, warmup, iters)
    return {
        "select_blocks_ms": select_blocks_ms,
        "score_candidates_ms": score_candidates_ms,
        "fused_mask_topk_map_ms": fused_mask_topk_map_ms,
        "stage_sum_ms": (
            select_blocks_ms + score_candidates_ms + fused_mask_topk_map_ms
        ),
    }


def _parallel_collective_stage_microbench(
    q_fp4,
    cache,
    page_table,
    seq_lens,
    weights,
    token_to_batch_idx,
    prepared,
    topk: int,
    warmup: int,
    iters: int,
) -> dict:
    import deep_gemm

    def select_blocks_call():
        return hisa_selector_parallel_select_blocks_indexer_cache_nvfp4(
            q_fp4,
            seq_lens,
            weights,
            token_to_batch_idx,
            prepared["reps"],
            prepared["max_blocks"],
            prepared["block_counts"],
            prepared["block_topk_counts"],
            prepared["effective_block_topk"],
            page_table_dtype=page_table.dtype,
        )

    select_blocks_ms = _median_cuda(select_blocks_call, warmup, iters)
    top_blocks = select_blocks_call()

    if prepared["candidate_len"] == topk:
        def map_all_call():
            return hisa_map_candidate_indices_indexer_cache_nvfp4(
                top_blocks,
                prepared["prefix_lens"],
                topk,
                page_table_dtype=page_table.dtype,
            )

        map_all_ms = _median_cuda(map_all_call, warmup, iters)
        return {
            "select_blocks_ms": select_blocks_ms,
            "map_all_ms": map_all_ms,
            "stage_sum_ms": select_blocks_ms + map_all_ms,
        }

    def candidate_pages_call():
        return hisa_candidate_pages_indexer_cache_nvfp4(
            top_blocks, page_table, token_to_batch_idx
        )

    candidate_pages_ms = _median_cuda(candidate_pages_call, warmup, iters)
    candidate_page_table = candidate_pages_call()

    q_values, q_scales = q_fp4

    def candidate_logits_call():
        return deep_gemm.fp8_fp4_paged_mqa_logits(
            (q_values.view(torch.int8).unsqueeze(1), q_scales.unsqueeze(1)),
            cache.view(cache.shape[0], 64, 1, 68),
            weights,
            prepared["candidate_context_lens"],
            candidate_page_table,
            prepared["candidate_schedule_metadata"],
            prepared["candidate_len"],
            clean_logits=False,
            logits_dtype=torch.float32,
        )

    candidate_logits_ms = _median_cuda(candidate_logits_call, warmup, iters)
    logits = candidate_logits_call()

    def fused_mask_topk_map_call():
        return hisa_fused_mask_topk_map_indexer_cache_nvfp4(
            logits,
            top_blocks,
            prepared["prefix_lens"],
            min(topk, prepared["candidate_len"]),
            page_table_dtype=page_table.dtype,
        )

    fused_mask_topk_map_ms = _median_cuda(fused_mask_topk_map_call, warmup, iters)
    return {
        "select_blocks_ms": select_blocks_ms,
        "candidate_pages_ms": candidate_pages_ms,
        "candidate_logits_ms": candidate_logits_ms,
        "fused_mask_topk_map_ms": fused_mask_topk_map_ms,
        "stage_sum_ms": (
            select_blocks_ms
            + candidate_pages_ms
            + candidate_logits_ms
            + fused_mask_topk_map_ms
        ),
    }


def _jit_deepgemm_candidate_probe_microbench(
    q_fp4,
    cache,
    page_table,
    weights,
    token_to_batch_idx,
    prepared,
    topk: int,
    warmup: int,
    iters: int,
) -> dict | None:
    if prepared["candidate_len"] == topk:
        return None

    import deep_gemm

    q_values, q_scales = q_fp4
    rep_values, rep_scales = prepared["reps"]
    block_scores = deep_gemm.fp8_fp4_mqa_logits(
        (q_values.view(torch.int8), q_scales),
        (rep_values.view(torch.int8), rep_scales),
        weights,
        prepared["block_starts"].to(torch.int32),
        prepared["block_ends"].to(torch.int32),
        clean_logits=False,
        max_seqlen_k=prepared["max_blocks"],
        logits_dtype=torch.float32,
    )
    top_blocks = hisa_block_topk_indexer_cache_nvfp4(
        block_scores,
        prepared["block_counts"],
        block_topk=prepared["effective_block_topk"],
        block_topk_counts=prepared["block_topk_counts"],
        page_table_dtype=page_table.dtype,
    )
    candidate_page_table = hisa_candidate_pages_indexer_cache_nvfp4(
        top_blocks, page_table, token_to_batch_idx
    )

    def external_candidate_logits_call():
        return deep_gemm.fp8_fp4_paged_mqa_logits(
            (q_values.view(torch.int8).unsqueeze(1), q_scales.unsqueeze(1)),
            cache.view(cache.shape[0], 64, 1, 68),
            weights,
            prepared["candidate_context_lens"],
            candidate_page_table,
            prepared["candidate_schedule_metadata"],
            prepared["candidate_len"],
            clean_logits=False,
            logits_dtype=torch.float32,
        )

    def jit_candidate_logits_call():
        return hisa_deepgemm_candidate_logits_indexer_cache_nvfp4(
            q_fp4,
            cache,
            weights,
            prepared["candidate_context_lens"],
            candidate_page_table,
            prepared["candidate_schedule_metadata"],
            prepared["candidate_len"],
        )

    jit_candidate_logits_ms = _median_cuda(jit_candidate_logits_call, warmup, iters)
    external_logits = external_candidate_logits_call()
    jit_logits = jit_candidate_logits_call()
    diff = (jit_logits - external_logits).abs()
    return {
        "jit_candidate_logits_ms": jit_candidate_logits_ms,
        "jit_vs_external_max_abs_diff": float(diff.max().item()),
        "jit_vs_external_mean_abs_diff": float(diff.mean().item()),
    }


def _jit_deepgemm_candidate_key_probe_microbench(
    q_fp4,
    cache,
    page_table,
    weights,
    token_to_batch_idx,
    prepared,
    staged_out: torch.Tensor,
    topk: int,
    warmup: int,
    iters: int,
) -> dict | None:
    if prepared["candidate_len"] == topk:
        return None

    import deep_gemm

    q_values, q_scales = q_fp4
    rep_values, rep_scales = prepared["reps"]
    block_scores = deep_gemm.fp8_fp4_mqa_logits(
        (q_values.view(torch.int8), q_scales),
        (rep_values.view(torch.int8), rep_scales),
        weights,
        prepared["block_starts"].to(torch.int32),
        prepared["block_ends"].to(torch.int32),
        clean_logits=False,
        max_seqlen_k=prepared["max_blocks"],
        logits_dtype=torch.float32,
    )
    top_blocks = hisa_block_topk_indexer_cache_nvfp4(
        block_scores,
        prepared["block_counts"],
        block_topk=prepared["effective_block_topk"],
        block_topk_counts=prepared["block_topk_counts"],
        page_table_dtype=page_table.dtype,
    )
    candidate_page_table = hisa_candidate_pages_indexer_cache_nvfp4(
        top_blocks, page_table, token_to_batch_idx
    )

    def candidate_keys_call():
        return hisa_deepgemm_candidate_keys_indexer_cache_nvfp4(
            q_fp4,
            cache,
            weights,
            prepared["candidate_context_lens"],
            candidate_page_table,
            prepared["candidate_schedule_metadata"],
            top_blocks,
            prepared["prefix_lens"],
            prepared["candidate_len"],
        )

    candidate_keys_ms = _median_cuda(candidate_keys_call, warmup, iters)
    candidate_keys = candidate_keys_call()

    def keys_topk_map_call():
        return hisa_candidate_keys_topk_map_indexer_cache_nvfp4(
            candidate_keys,
            top_blocks,
            topk,
            page_table_dtype=page_table.dtype,
        )

    keys_topk_map_ms = _median_cuda(keys_topk_map_call, warmup, iters)
    key_out = keys_topk_map_call()
    return {
        "candidate_keys_ms": candidate_keys_ms,
        "candidate_keys_topk_map_ms": keys_topk_map_ms,
        "stage_sum_ms": candidate_keys_ms + keys_topk_map_ms,
        "vs_staged_exact_order_equal": bool(torch.equal(key_out, staged_out)),
        "vs_staged_row_set_equal_fraction": _tensor_set_equal_fraction(
            key_out, staged_out
        ),
        "vs_staged_prefix16_exact_fraction": _prefix_exact_fraction(
            key_out, staged_out, 16
        ),
    }


def _row_split_candidate_key_probe_microbench(
    q_fp4,
    cache,
    page_table,
    weights,
    token_to_batch_idx,
    prepared,
    staged_out,
    topk: int,
    row_splits: int,
    warmup: int,
    iters: int,
):
    import deep_gemm

    q_values, q_scales = q_fp4
    rep_values, rep_scales = prepared["reps"]
    block_scores = deep_gemm.fp8_fp4_mqa_logits(
        (q_values.view(torch.int8), q_scales),
        (rep_values.view(torch.int8), rep_scales),
        weights,
        prepared["block_starts"].to(torch.int32),
        prepared["block_ends"].to(torch.int32),
        clean_logits=False,
        max_seqlen_k=prepared["max_blocks"],
        logits_dtype=torch.float32,
    )
    top_blocks = hisa_block_topk_indexer_cache_nvfp4(
        block_scores,
        prepared["block_counts"],
        block_topk=prepared["effective_block_topk"],
        block_topk_counts=prepared["block_topk_counts"],
        page_table_dtype=page_table.dtype,
    )

    def candidate_keys_call():
        return hisa_deepgemm_candidate_keys_row_split_from_blocks_indexer_cache_nvfp4(
            q_fp4,
            cache,
            weights,
            page_table,
            token_to_batch_idx,
            top_blocks,
            prepared["prefix_lens"],
            prepared["candidate_len"],
            row_splits=row_splits,
        )

    candidate_keys_ms = _median_cuda(candidate_keys_call, warmup, iters)
    candidate_keys = candidate_keys_call()

    def keys_topk_map_call():
        return hisa_candidate_keys_topk_map_indexer_cache_nvfp4(
            candidate_keys,
            top_blocks,
            topk,
            page_table_dtype=page_table.dtype,
        )

    keys_topk_map_ms = _median_cuda(keys_topk_map_call, warmup, iters)
    key_out = keys_topk_map_call()
    return {
        "row_splits": row_splits,
        "candidate_keys_ms": candidate_keys_ms,
        "candidate_keys_topk_map_ms": keys_topk_map_ms,
        "stage_sum_ms": candidate_keys_ms + keys_topk_map_ms,
        "vs_staged_exact_order_equal": bool(torch.equal(key_out, staged_out)),
        "vs_staged_row_set_equal_fraction": _tensor_set_equal_fraction(
            key_out, staged_out
        ),
        "vs_staged_prefix16_exact_fraction": _prefix_exact_fraction(
            key_out, staged_out, 16
        ),
    }


def _fused_candidate_topk_cluster_probe_microbench(
    q_fp4,
    cache,
    page_table,
    weights,
    token_to_batch_idx,
    prepared,
    staged_out,
    topk: int,
    row_splits: int,
    use_cluster: bool,
    warmup: int,
    iters: int,
):
    import deep_gemm

    q_values, q_scales = q_fp4
    rep_values, rep_scales = prepared["reps"]
    block_scores = deep_gemm.fp8_fp4_mqa_logits(
        (q_values.view(torch.int8), q_scales),
        (rep_values.view(torch.int8), rep_scales),
        weights,
        prepared["block_starts"].to(torch.int32),
        prepared["block_ends"].to(torch.int32),
        clean_logits=False,
        max_seqlen_k=prepared["max_blocks"],
        logits_dtype=torch.float32,
    )
    top_blocks = hisa_block_topk_indexer_cache_nvfp4(
        block_scores,
        prepared["block_counts"],
        block_topk=prepared["effective_block_topk"],
        block_topk_counts=prepared["block_topk_counts"],
        page_table_dtype=page_table.dtype,
    )

    def candidate_topk_call():
        return hisa_deepgemm_candidate_topk_cluster_from_blocks_indexer_cache_nvfp4(
            q_fp4,
            cache,
            weights,
            page_table,
            token_to_batch_idx,
            top_blocks,
            prepared["prefix_lens"],
            prepared["candidate_len"],
            topk,
            row_splits=row_splits,
            use_cluster=use_cluster,
        )

    candidate_topk_ms = _median_cuda(candidate_topk_call, warmup, iters)
    topk_out = candidate_topk_call()
    return {
        "row_splits": row_splits,
        "use_cluster": use_cluster,
        "candidate_score_topk_map_ms": candidate_topk_ms,
        "stage_sum_ms": candidate_topk_ms,
        "vs_staged_exact_order_equal": bool(torch.equal(topk_out, staged_out)),
        "vs_staged_row_set_equal_fraction": _tensor_set_equal_fraction(
            topk_out, staged_out
        ),
        "vs_staged_prefix16_exact_fraction": _prefix_exact_fraction(
            topk_out, staged_out, 16
        ),
    }


def _fused_candidate_topk_cooperative_probe_microbench(
    q_fp4,
    cache,
    page_table,
    weights,
    token_to_batch_idx,
    prepared,
    staged_out,
    topk: int,
    warmup: int,
    iters: int,
):
    import deep_gemm

    q_values, q_scales = q_fp4
    rep_values, rep_scales = prepared["reps"]
    block_scores = deep_gemm.fp8_fp4_mqa_logits(
        (q_values.view(torch.int8), q_scales),
        (rep_values.view(torch.int8), rep_scales),
        weights,
        prepared["block_starts"].to(torch.int32),
        prepared["block_ends"].to(torch.int32),
        clean_logits=False,
        max_seqlen_k=prepared["max_blocks"],
        logits_dtype=torch.float32,
    )
    top_blocks = hisa_block_topk_indexer_cache_nvfp4(
        block_scores,
        prepared["block_counts"],
        block_topk=prepared["effective_block_topk"],
        block_topk_counts=prepared["block_topk_counts"],
        page_table_dtype=page_table.dtype,
    )

    def candidate_topk_call():
        return hisa_deepgemm_candidate_topk_cooperative_from_blocks_indexer_cache_nvfp4(
            q_fp4,
            cache,
            weights,
            prepared["candidate_context_lens"],
            page_table,
            token_to_batch_idx,
            prepared["candidate_schedule_metadata"],
            top_blocks,
            prepared["prefix_lens"],
            prepared["candidate_len"],
            topk,
        )

    candidate_topk_ms = _median_cuda(candidate_topk_call, warmup, iters)
    topk_out = candidate_topk_call()
    return {
        "candidate_score_topk_map_ms": candidate_topk_ms,
        "stage_sum_ms": candidate_topk_ms,
        "vs_staged_exact_order_equal": bool(torch.equal(topk_out, staged_out)),
        "vs_staged_row_set_equal_fraction": _tensor_set_equal_fraction(
            topk_out, staged_out
        ),
        "vs_staged_prefix16_exact_fraction": _prefix_exact_fraction(
            topk_out, staged_out, 16
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Microbenchmark NVFP4 HISA selector shapes: current parallel/staged "
            "selector versus the serial cuBLASDx correctness probe."
        )
    )
    parser.add_argument("--prefix-lengths", default="4096,8192,16384,32768,65536")
    parser.add_argument("--topks", default="1024")
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--query-rows", type=int, default=1024)
    parser.add_argument("--hisa-compression-ratio", type=float, default=4.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-dense", action="store_true")
    parser.add_argument("--include-cluster-fused-probe", action="store_true")
    parser.add_argument("--include-collective-key-probe", action="store_true")
    parser.add_argument(
        "--include-jit-deepgemm-candidate-probe",
        action="store_true",
        help=(
            "Time the SGLang-owned DeepGEMM-source candidate scorer. This is a "
            "diagnostic primitive, not the final fused selector."
        ),
    )
    parser.add_argument(
        "--include-jit-deepgemm-candidate-key-probe",
        action="store_true",
        help=(
            "Time the collective FP4 candidate scorer variant that emits masked "
            "sortable keys instead of logits. This is diagnostic fusion work."
        ),
    )
    parser.add_argument(
        "--include-row-split-candidate-key-probe",
        action="store_true",
        help=(
            "Time the row-owned SM100 FP4 candidate-key scorer probe. This keeps "
            "batched candidate scoring but uses multiple CTAs per row for future "
            "in-kernel top-k fusion work."
        ),
    )
    parser.add_argument(
        "--include-fused-candidate-topk-cluster-probe",
        action="store_true",
        help=(
            "Time the cluster-launched row-split FP4 candidate scorer with candidate "
            "top-k/map fused into the same launch. This still assumes preselected "
            "HISA blocks and is a candidate-scope fusion probe."
        ),
    )
    parser.add_argument(
        "--include-fused-candidate-topk-row-probe",
        action="store_true",
        help=(
            "Time the one-CTA-per-row FP4 candidate scorer with candidate top-k/map "
            "fused into the same launch. This still assumes preselected HISA blocks."
        ),
    )
    parser.add_argument(
        "--include-fused-candidate-topk-cooperative-probe",
        action="store_true",
        help=(
            "Time the DeepGEMM-shaped persistent FP4 candidate scorer with a "
            "cooperative-grid barrier and candidate top-k/map fused into the "
            "same kernel launch. This still assumes preselected HISA blocks."
        ),
    )
    parser.add_argument("--row-splits", default="1,2,4,8")
    parser.add_argument("--skip-parallel-probe", action="store_true")
    parser.add_argument("--skip-parallel-collective-probe", action="store_true")
    parser.add_argument("--skip-serial-probe", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    _require_blackwell()
    _assert_candidate_keys_topk_map_smoke()
    results = []
    for topk in _ints_csv(args.topks):
        for prefix_len in _ints_csv(args.prefix_lengths):
            q_fp4, cache, page_table, seq_lens, weights, token_to_batch_idx = (
                _build_case(prefix_len, args.heads, args.query_rows, args.seed)
            )
            prepared = _prepare_hisa(
                q_fp4,
                cache,
                page_table,
                seq_lens,
                token_to_batch_idx,
                topk,
                args.hisa_compression_ratio,
                args.warmup,
                args.iters,
            )

            parallel_fn = lambda: _parallel_selector_call(
                q_fp4,
                cache,
                page_table,
                seq_lens,
                weights,
                token_to_batch_idx,
                prepared,
                topk,
                args.hisa_compression_ratio,
            )
            parallel_ms = _median_cuda(parallel_fn, args.warmup, args.iters)
            parallel_out = parallel_fn()
            stage_times = _stage_microbench(
                q_fp4,
                cache,
                page_table,
                weights,
                token_to_batch_idx,
                prepared,
                topk,
                args.hisa_compression_ratio,
                args.warmup,
                args.iters,
            )

            jit_candidate_probe_stage_times = None
            if args.include_jit_deepgemm_candidate_probe:
                jit_candidate_probe_stage_times = (
                    _jit_deepgemm_candidate_probe_microbench(
                        q_fp4,
                        cache,
                        page_table,
                        weights,
                        token_to_batch_idx,
                        prepared,
                        topk,
                        args.warmup,
                        args.iters,
                    )
                )
            jit_candidate_key_probe_stage_times = None
            if args.include_jit_deepgemm_candidate_key_probe:
                jit_candidate_key_probe_stage_times = (
                    _jit_deepgemm_candidate_key_probe_microbench(
                        q_fp4,
                        cache,
                        page_table,
                        weights,
                        token_to_batch_idx,
                        prepared,
                        parallel_out,
                        topk,
                        args.warmup,
                        args.iters,
                    )
                )
            row_split_candidate_key_probe_stage_times = None
            if args.include_row_split_candidate_key_probe:
                row_split_candidate_key_probe_stage_times = []
                for row_splits in _ints_csv(args.row_splits):
                    row_split_candidate_key_probe_stage_times.append(
                        _row_split_candidate_key_probe_microbench(
                            q_fp4,
                            cache,
                            page_table,
                            weights,
                            token_to_batch_idx,
                            prepared,
                            parallel_out,
                            topk,
                            row_splits,
                            args.warmup,
                            args.iters,
                        )
                    )
            fused_candidate_topk_cluster_probe_stage_times = None
            if args.include_fused_candidate_topk_cluster_probe:
                fused_candidate_topk_cluster_probe_stage_times = (
                    _fused_candidate_topk_cluster_probe_microbench(
                        q_fp4,
                        cache,
                        page_table,
                        weights,
                        token_to_batch_idx,
                        prepared,
                        parallel_out,
                        topk,
                        4,
                        True,
                        args.warmup,
                        args.iters,
                    )
                )
            fused_candidate_topk_row_probe_stage_times = None
            if args.include_fused_candidate_topk_row_probe:
                fused_candidate_topk_row_probe_stage_times = (
                    _fused_candidate_topk_cluster_probe_microbench(
                        q_fp4,
                        cache,
                        page_table,
                        weights,
                        token_to_batch_idx,
                        prepared,
                        parallel_out,
                        topk,
                        1,
                        False,
                        args.warmup,
                        args.iters,
                    )
                )
            fused_candidate_topk_cooperative_probe_stage_times = None
            if args.include_fused_candidate_topk_cooperative_probe:
                fused_candidate_topk_cooperative_probe_stage_times = (
                    _fused_candidate_topk_cooperative_probe_microbench(
                        q_fp4,
                        cache,
                        page_table,
                        weights,
                        token_to_batch_idx,
                        prepared,
                        parallel_out,
                        topk,
                        args.warmup,
                        args.iters,
                    )
                )
            collective_key_probe_ms = None
            collective_key_probe_supported = False
            collective_key_probe_exact_order_equal = None
            collective_key_probe_row_set_equal_fraction = None
            collective_key_probe_prefix16_exact_fraction = None
            collective_key_probe_over_staged_x = None
            if args.include_collective_key_probe:
                collective_key_probe_out = _collective_key_probe_call(
                    q_fp4,
                    cache,
                    page_table,
                    seq_lens,
                    weights,
                    token_to_batch_idx,
                    prepared,
                    topk,
                    args.hisa_compression_ratio,
                )
                if collective_key_probe_out is not None:
                    collective_key_probe_supported = True
                    collective_key_probe_ms = _median_cuda(
                        lambda: _collective_key_probe_call(
                            q_fp4,
                            cache,
                            page_table,
                            seq_lens,
                            weights,
                            token_to_batch_idx,
                            prepared,
                            topk,
                            args.hisa_compression_ratio,
                        ),
                        args.warmup,
                        args.iters,
                    )
                    collective_key_probe_exact_order_equal = bool(
                        torch.equal(collective_key_probe_out, parallel_out)
                    )
                    collective_key_probe_row_set_equal_fraction = (
                        _tensor_set_equal_fraction(
                            collective_key_probe_out, parallel_out
                        )
                    )
                    collective_key_probe_prefix16_exact_fraction = (
                        _prefix_exact_fraction(
                            collective_key_probe_out, parallel_out, 16
                        )
                    )
                    collective_key_probe_over_staged_x = (
                        collective_key_probe_ms / parallel_ms if parallel_ms else None
                    )

            parallel_probe_ms = None
            parallel_probe_supported = False
            parallel_probe_exact_order_equal = None
            parallel_probe_row_set_equal_fraction = None
            parallel_probe_prefix16_exact_fraction = None
            parallel_probe_over_staged_x = None
            parallel_probe_stage_times = None
            if not args.skip_parallel_probe:
                parallel_probe_out = _parallel_probe_call(
                    q_fp4,
                    cache,
                    page_table,
                    seq_lens,
                    weights,
                    token_to_batch_idx,
                    prepared,
                    topk,
                    args.hisa_compression_ratio,
                )
                if parallel_probe_out is not None:
                    parallel_probe_supported = True
                    parallel_probe_ms = _median_cuda(
                        lambda: _parallel_probe_call(
                            q_fp4,
                            cache,
                            page_table,
                            seq_lens,
                            weights,
                            token_to_batch_idx,
                            prepared,
                            topk,
                            args.hisa_compression_ratio,
                        ),
                        args.warmup,
                        args.iters,
                    )
                    parallel_probe_exact_order_equal = bool(
                        torch.equal(parallel_probe_out, parallel_out)
                    )
                    parallel_probe_row_set_equal_fraction = _tensor_set_equal_fraction(
                        parallel_probe_out, parallel_out
                    )
                    parallel_probe_prefix16_exact_fraction = _prefix_exact_fraction(
                        parallel_probe_out, parallel_out, 16
                    )
                    parallel_probe_over_staged_x = (
                        parallel_probe_ms / parallel_ms if parallel_ms else None
                    )
                    parallel_probe_stage_times = _parallel_probe_stage_microbench(
                        q_fp4,
                        cache,
                        page_table,
                        seq_lens,
                        weights,
                        token_to_batch_idx,
                        prepared,
                        topk,
                        args.warmup,
                        args.iters,
                    )

            cluster_fused_probe_ms = None
            cluster_fused_probe_supported = False
            cluster_fused_probe_exact_order_equal = None
            cluster_fused_probe_row_set_equal_fraction = None
            cluster_fused_probe_prefix16_exact_fraction = None
            cluster_fused_probe_over_staged_x = None
            if args.include_cluster_fused_probe:
                cluster_fused_probe_out = _cluster_fused_probe_call(
                    q_fp4,
                    cache,
                    page_table,
                    seq_lens,
                    weights,
                    token_to_batch_idx,
                    prepared,
                    topk,
                    args.hisa_compression_ratio,
                )
                if cluster_fused_probe_out is not None:
                    cluster_fused_probe_supported = True
                    cluster_fused_probe_ms = _median_cuda(
                        lambda: _cluster_fused_probe_call(
                            q_fp4,
                            cache,
                            page_table,
                            seq_lens,
                            weights,
                            token_to_batch_idx,
                            prepared,
                            topk,
                            args.hisa_compression_ratio,
                        ),
                        args.warmup,
                        args.iters,
                    )
                    cluster_fused_probe_exact_order_equal = bool(
                        torch.equal(cluster_fused_probe_out, parallel_out)
                    )
                    cluster_fused_probe_row_set_equal_fraction = (
                        _tensor_set_equal_fraction(cluster_fused_probe_out, parallel_out)
                    )
                    cluster_fused_probe_prefix16_exact_fraction = (
                        _prefix_exact_fraction(
                            cluster_fused_probe_out, parallel_out, 16
                        )
                    )
                    cluster_fused_probe_over_staged_x = (
                        cluster_fused_probe_ms / parallel_ms if parallel_ms else None
                    )

            dense_ms = None
            if args.include_dense:
                dense_ms = _median_cuda(
                    lambda: _indexcache_nvfp4_paged(
                        q_fp4, cache, page_table, seq_lens, weights, topk
                    ),
                    args.warmup,
                    args.iters,
                )

            collective_probe_ms = None
            collective_probe_supported = False
            collective_probe_exact_order_equal = None
            collective_probe_row_set_equal_fraction = None
            collective_probe_prefix16_exact_fraction = None
            collective_probe_over_staged_x = None
            collective_probe_stage_times = None
            if not args.skip_parallel_collective_probe:
                collective_probe_out = _parallel_collective_probe_call(
                    q_fp4,
                    cache,
                    page_table,
                    seq_lens,
                    weights,
                    token_to_batch_idx,
                    prepared,
                    topk,
                    args.hisa_compression_ratio,
                )
                if collective_probe_out is not None:
                    collective_probe_supported = True
                    collective_probe_ms = _median_cuda(
                        lambda: _parallel_collective_probe_call(
                            q_fp4,
                            cache,
                            page_table,
                            seq_lens,
                            weights,
                            token_to_batch_idx,
                            prepared,
                            topk,
                            args.hisa_compression_ratio,
                        ),
                        args.warmup,
                        args.iters,
                    )
                    collective_probe_exact_order_equal = bool(
                        torch.equal(collective_probe_out, parallel_out)
                    )
                    collective_probe_row_set_equal_fraction = _tensor_set_equal_fraction(
                        collective_probe_out, parallel_out
                    )
                    collective_probe_prefix16_exact_fraction = _prefix_exact_fraction(
                        collective_probe_out, parallel_out, 16
                    )
                    collective_probe_over_staged_x = (
                        collective_probe_ms / parallel_ms if parallel_ms else None
                    )
                    collective_probe_stage_times = _parallel_collective_stage_microbench(
                        q_fp4,
                        cache,
                        page_table,
                        seq_lens,
                        weights,
                        token_to_batch_idx,
                        prepared,
                        topk,
                        args.warmup,
                        args.iters,
                    )

            serial_ms = None
            serial_supported = False
            serial_exact_order_equal = None
            serial_row_set_equal_fraction = None
            serial_prefix16_exact_fraction = None
            serial_over_parallel_x = None
            if not args.skip_serial_probe:
                serial_out = _serial_probe_call(
                    q_fp4,
                    cache,
                    page_table,
                    seq_lens,
                    weights,
                    token_to_batch_idx,
                    prepared,
                    topk,
                    args.hisa_compression_ratio,
                )
                if serial_out is not None:
                    serial_supported = True
                    serial_ms = _median_cuda(
                        lambda: _serial_probe_call(
                            q_fp4,
                            cache,
                            page_table,
                            seq_lens,
                            weights,
                            token_to_batch_idx,
                            prepared,
                            topk,
                            args.hisa_compression_ratio,
                        ),
                        args.warmup,
                        args.iters,
                    )
                    serial_exact_order_equal = bool(torch.equal(serial_out, parallel_out))
                    serial_row_set_equal_fraction = _tensor_set_equal_fraction(
                        serial_out, parallel_out
                    )
                    serial_prefix16_exact_fraction = _prefix_exact_fraction(
                        serial_out, parallel_out, 16
                    )
                    serial_over_parallel_x = (
                        serial_ms / parallel_ms if parallel_ms else None
                    )

            row = {
                "prefix_len": prefix_len,
                "topk": topk,
                "heads": args.heads,
                "query_rows": args.query_rows,
                "hisa_compression_ratio": args.hisa_compression_ratio,
                "hisa_effective_block_topk": prepared["effective_block_topk"],
                "hisa_candidate_len": prepared["candidate_len"],
                "max_blocks": prepared["max_blocks"],
                "precompute_reps_ms": prepared["precompute_ms"],
                "parallel_staged_selector_ms": parallel_ms,
                "parallel_stage_times": stage_times,
                "jit_deepgemm_candidate_probe_supported": (
                    jit_candidate_probe_stage_times is not None
                ),
                "jit_deepgemm_candidate_probe_stage_times": (
                    jit_candidate_probe_stage_times
                ),
                "jit_deepgemm_candidate_key_probe_supported": (
                    jit_candidate_key_probe_stage_times is not None
                ),
                "jit_deepgemm_candidate_key_probe_stage_times": (
                    jit_candidate_key_probe_stage_times
                ),
                "row_split_candidate_key_probe_supported": (
                    row_split_candidate_key_probe_stage_times is not None
                ),
                "row_split_candidate_key_probe_stage_times": (
                    row_split_candidate_key_probe_stage_times
                ),
                "fused_candidate_topk_cluster_probe_supported": (
                    fused_candidate_topk_cluster_probe_stage_times is not None
                ),
                "fused_candidate_topk_cluster_probe_stage_times": (
                    fused_candidate_topk_cluster_probe_stage_times
                ),
                "fused_candidate_topk_row_probe_supported": (
                    fused_candidate_topk_row_probe_stage_times is not None
                ),
                "fused_candidate_topk_row_probe_stage_times": (
                    fused_candidate_topk_row_probe_stage_times
                ),
                "fused_candidate_topk_cooperative_probe_supported": (
                    fused_candidate_topk_cooperative_probe_stage_times is not None
                ),
                "fused_candidate_topk_cooperative_probe_stage_times": (
                    fused_candidate_topk_cooperative_probe_stage_times
                ),
                "collective_key_probe_supported": collective_key_probe_supported,
                "collective_key_probe_ms": collective_key_probe_ms,
                "collective_key_probe_over_staged_x": (
                    collective_key_probe_over_staged_x
                ),
                "collective_key_probe_vs_staged_exact_order_equal": (
                    collective_key_probe_exact_order_equal
                ),
                "collective_key_probe_vs_staged_row_set_equal_fraction": (
                    collective_key_probe_row_set_equal_fraction
                ),
                "collective_key_probe_vs_staged_prefix16_exact_fraction": (
                    collective_key_probe_prefix16_exact_fraction
                ),
                "parallel_probe_supported": parallel_probe_supported,
                "parallel_probe_ms": parallel_probe_ms,
                "parallel_probe_over_staged_x": parallel_probe_over_staged_x,
                "parallel_probe_stage_times": parallel_probe_stage_times,
                "parallel_probe_vs_staged_exact_order_equal": (
                    parallel_probe_exact_order_equal
                ),
                "parallel_probe_vs_staged_row_set_equal_fraction": (
                    parallel_probe_row_set_equal_fraction
                ),
                "parallel_probe_vs_staged_prefix16_exact_fraction": (
                    parallel_probe_prefix16_exact_fraction
                ),
                "cluster_fused_probe_supported": cluster_fused_probe_supported,
                "cluster_fused_probe_ms": cluster_fused_probe_ms,
                "cluster_fused_probe_over_staged_x": (
                    cluster_fused_probe_over_staged_x
                ),
                "cluster_fused_probe_vs_staged_exact_order_equal": (
                    cluster_fused_probe_exact_order_equal
                ),
                "cluster_fused_probe_vs_staged_row_set_equal_fraction": (
                    cluster_fused_probe_row_set_equal_fraction
                ),
                "cluster_fused_probe_vs_staged_prefix16_exact_fraction": (
                    cluster_fused_probe_prefix16_exact_fraction
                ),
                "parallel_collective_probe_supported": collective_probe_supported,
                "parallel_collective_probe_ms": collective_probe_ms,
                "parallel_collective_probe_over_staged_x": (
                    collective_probe_over_staged_x
                ),
                "parallel_collective_probe_stage_times": (
                    collective_probe_stage_times
                ),
                "parallel_collective_probe_vs_staged_exact_order_equal": (
                    collective_probe_exact_order_equal
                ),
                "parallel_collective_probe_vs_staged_row_set_equal_fraction": (
                    collective_probe_row_set_equal_fraction
                ),
                "parallel_collective_probe_vs_staged_prefix16_exact_fraction": (
                    collective_probe_prefix16_exact_fraction
                ),
                "dense_indexcache_ms": dense_ms,
                "serial_probe_supported": serial_supported,
                "serial_probe_ms": serial_ms,
                "serial_over_parallel_x": serial_over_parallel_x,
                "serial_vs_parallel_exact_order_equal": serial_exact_order_equal,
                "serial_vs_parallel_row_set_equal_fraction": (
                    serial_row_set_equal_fraction
                ),
                "serial_vs_parallel_prefix16_exact_fraction": (
                    serial_prefix16_exact_fraction
                ),
            }
            print(json.dumps(row, sort_keys=True))
            results.append(row)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
