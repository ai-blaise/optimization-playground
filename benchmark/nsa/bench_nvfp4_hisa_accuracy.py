#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.nsa.bench_nvfp4_hisa_indexer import (
    _build_case,
    _hisa_block_topk_counts,
    _indexcache_nvfp4_paged,
    _require_blackwell,
)
from sglang.jit_kernel.nvfp4_indexer import (
    hisa_precompute_block_reps_indexer_cache_nvfp4,
    nvfp4_hisa_indexer_paged_collective_key_precomputed,
    nvfp4_hisa_indexer_paged_deepgemm_precomputed,
    nvfp4_hisa_indexer_paged_megakernel_precomputed,
)


def _ints_csv(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def _row_sets(tensor: torch.Tensor) -> list[set[int]]:
    cpu = tensor.detach().cpu()
    return [set(int(x) for x in row.tolist() if int(x) >= 0) for row in cpu]


def _compare_sets(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    pred_sets = _row_sets(pred)
    ref_sets = _row_sets(ref)
    overlaps = []
    recalls = []
    precisions = []
    full_rows = 0
    pred_valid = 0
    ref_valid = 0
    for pred_set, ref_set in zip(pred_sets, ref_sets):
        overlap = len(pred_set & ref_set)
        overlaps.append(overlap)
        pred_valid += len(pred_set)
        ref_valid += len(ref_set)
        recalls.append(overlap / len(ref_set) if ref_set else 1.0)
        precisions.append(overlap / len(pred_set) if pred_set else 1.0)
        full_rows += int(pred_set == ref_set)
    rows = max(1, len(pred_sets))
    return {
        "exact_order_equal": bool(torch.equal(pred, ref)),
        "row_set_equal_fraction": full_rows / rows,
        "mean_overlap": sum(overlaps) / rows,
        "min_overlap": min(overlaps) if overlaps else 0,
        "mean_recall": sum(recalls) / rows,
        "min_recall": min(recalls) if recalls else 1.0,
        "mean_precision": sum(precisions) / rows,
        "pred_valid_mean": pred_valid / rows,
        "ref_valid_mean": ref_valid / rows,
    }


def _compare_ordered_prefix(pred: torch.Tensor, ref: torch.Tensor, limits: Iterable[int]) -> dict:
    out = {}
    for limit in limits:
        k = min(limit, pred.shape[1], ref.shape[1])
        out[f"prefix_{k}_exact_fraction"] = (
            (pred[:, :k] == ref[:, :k]).all(dim=1).float().mean().item()
        )
    return out


def _prepare_hisa(
    q_fp4,
    cache,
    page_table,
    seq_lens,
    token_to_batch_idx,
    topk: int,
    compression_ratio: float,
):
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
    context_lens = torch.full(
        (q_fp4[0].shape[0], 1),
        effective_block_topk * 128,
        device=q_fp4[0].device,
        dtype=torch.int32,
    )
    import deep_gemm

    return {
        "reps": reps,
        "max_blocks": max_blocks,
        "prefix_lens": prefix_lens,
        "block_counts": block_counts,
        "block_topk_counts": block_topk_counts,
        "effective_block_topk": effective_block_topk,
        "block_starts": token_to_batch_idx.to(torch.int32) * max_blocks,
        "context_lens": context_lens,
        "schedule_metadata": deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, 64, deep_gemm.get_num_sms()
        ),
    }


def _current_hisa(
    q_fp4,
    cache,
    page_table,
    seq_lens,
    weights,
    token_to_batch_idx,
    prepared,
    topk: int,
    compression_ratio: float,
) -> torch.Tensor:
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
        prepared_block_ends=prepared["block_starts"] + prepared["block_counts"],
        prepared_candidate_context_lens=prepared["context_lens"],
        prepared_candidate_schedule_metadata=prepared["schedule_metadata"],
        prepared_block_topk_counts=prepared["block_topk_counts"],
        prepared_effective_block_topk=prepared["effective_block_topk"],
    )


def _megakernel_hisa(
    q_fp4,
    cache,
    page_table,
    seq_lens,
    weights,
    token_to_batch_idx,
    prepared,
    topk: int,
    compression_ratio: float,
) -> torch.Tensor | None:
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


def _collective_key_hisa(
    q_fp4,
    cache,
    page_table,
    seq_lens,
    weights,
    token_to_batch_idx,
    prepared,
    topk: int,
    compression_ratio: float,
) -> torch.Tensor | None:
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
        prepared_block_ends=prepared["block_starts"] + prepared["block_counts"],
        prepared_candidate_context_lens=prepared["context_lens"],
        prepared_candidate_schedule_metadata=prepared["schedule_metadata"],
        prepared_block_topk_counts=prepared["block_topk_counts"],
        prepared_effective_block_topk=prepared["effective_block_topk"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure NVFP4 HISA selector overlap against dense IndexCache."
    )
    parser.add_argument("--prefix-lengths", default="4096,8192,16384,32768,65536")
    parser.add_argument("--topks", default="1024")
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--query-rows", type=int, default=256)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--hisa-compression-ratio", type=float, default=4.0)
    parser.add_argument(
        "--include-serial-megakernel-probe",
        "--include-megakernel",
        dest="include_serial_megakernel_probe",
        action="store_true",
        help=(
            "Also compare the training-style serial megakernel probe against "
            "current HISA. This is not the target inference implementation."
        ),
    )
    parser.add_argument(
        "--include-collective-key-probe",
        action="store_true",
        help=(
            "Also compare the collective FP4 candidate score-key probe against "
            "current HISA and dense IndexCache."
        ),
    )
    parser.add_argument("--json-out")
    args = parser.parse_args()

    _require_blackwell()
    results = []
    for seed in _ints_csv(args.seeds):
        for topk in _ints_csv(args.topks):
            for prefix_len in _ints_csv(args.prefix_lengths):
                case = _build_case(prefix_len, args.heads, args.query_rows, seed)
                q_fp4, cache, page_table, seq_lens, weights, token_to_batch_idx = case
                dense = _indexcache_nvfp4_paged(
                    q_fp4, cache, page_table, seq_lens, weights, topk
                ).to(torch.int32)
                prepared = _prepare_hisa(
                    q_fp4,
                    cache,
                    page_table,
                    seq_lens,
                    token_to_batch_idx,
                    topk,
                    args.hisa_compression_ratio,
                )
                current = _current_hisa(
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
                row = {
                    "seed": seed,
                    "prefix_len": prefix_len,
                    "topk": topk,
                    "query_rows": args.query_rows,
                    "heads": args.heads,
                    "hisa_compression_ratio": args.hisa_compression_ratio,
                    "hisa_effective_block_topk": prepared["effective_block_topk"],
                    "hisa_candidate_len": prepared["effective_block_topk"] * 128,
                    "current_hisa_vs_dense": _compare_sets(current, dense),
                }
                if args.include_serial_megakernel_probe:
                    mega = _megakernel_hisa(
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
                    if mega is None:
                        row["megakernel_supported"] = False
                    else:
                        row["megakernel_supported"] = True
                        row["megakernel_vs_dense"] = _compare_sets(mega, dense)
                        row["megakernel_vs_current_hisa"] = {
                            **_compare_sets(mega, current),
                            **_compare_ordered_prefix(mega, current, (16, 128, topk)),
                        }
                if args.include_collective_key_probe:
                    collective_key = _collective_key_hisa(
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
                    if collective_key is None:
                        row["collective_key_supported"] = False
                    else:
                        row["collective_key_supported"] = True
                        row["collective_key_vs_dense"] = _compare_sets(
                            collective_key, dense
                        )
                        row["collective_key_vs_current_hisa"] = {
                            **_compare_sets(collective_key, current),
                            **_compare_ordered_prefix(
                                collective_key, current, (16, 128, topk)
                            ),
                        }
                print(json.dumps(row, sort_keys=True))
                results.append(row)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
