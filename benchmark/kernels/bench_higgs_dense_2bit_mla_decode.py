"""CUDA-event microbench for the HIGGS dense 2-bit MLA decode kernels.

Times the split-K decode path across the production shapes the REAP
B200 deploy uses (head_dim=128, kv_lora_rank=512, top_k=2048, batch B
in {1, 4, 8, 16}, num_heads=128//tp for TP=8 → 16 heads/rank). The
single-pass kernel was dropped in iter3 (#16) because the split-K
path is 12× faster at B=1 and 4× faster at B=16 — see
``higgs_dense_2bit_mla_decode.cuh`` for measurements.

Runs the same input twice (warmup + N iterations) and reports median
us-per-call. Used to compare iter2/iter3 changes against the
baseline.

Usage:
    python benchmark/kernels/bench_higgs_dense_2bit_mla_decode.py \\
        --iters 100 --warmup 20 --batches 1,4,8,16

Output is one line per (variant, B) so a sed-friendly diff between
``bench-before.txt`` and ``bench-after.txt`` makes the regression /
speedup obvious.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch


def _build_inputs(num_rows, num_heads, top_k, num_slots, device):
    from sglang.srt.layers.quantization.higgs_dense_2bit_kv import (
        HiggsDense2BitConfig,
    )
    cfg = HiggsDense2BitConfig(latent_dim=512, rope_dim=64)
    slot_bytes = cfg.slot_bytes
    q_nope = torch.randn(
        num_rows, num_heads, 512, device=device, dtype=torch.bfloat16
    )
    q_rope = torch.randn(
        num_rows, num_heads, 64, device=device, dtype=torch.bfloat16
    )
    # The values inside ``compressed`` matter for correctness but not for
    # microbench timing — a uniform random fill is enough to exercise the
    # full dequant + softmax loop.
    compressed = torch.randint(
        0, 255, (num_slots, 1, slot_bytes), device=device, dtype=torch.uint8
    )
    # ``page_table`` uses every slot at least once. Linear permutation
    # mimics a topk page-table that points into a fully populated KV
    # cache.
    flat = torch.arange(num_rows * top_k, device=device, dtype=torch.int32)
    page_table = (flat % num_slots).reshape(num_rows, top_k).contiguous()
    out = torch.empty(
        (num_rows, num_heads, 512), device=device, dtype=torch.bfloat16
    )
    # Codebook: EDEN2-16 lattice — real values don't matter for timing.
    codebook = torch.randn(16, 2, device=device, dtype=torch.float32)
    return q_nope, q_rope, compressed, page_table, out, codebook


def _bench_split(num_rows, num_heads, top_k, num_splits, iters, warmup, device):
    from sglang.jit_kernel.higgs_dense_2bit_mla_decode import (
        higgs_dense_2bit_mla_decode_split,
        higgs_dense_2bit_mla_rotate_query,
    )

    num_slots = max(2048, top_k * num_rows)
    q_nope, q_rope, compressed, page_table, out, codebook = _build_inputs(
        num_rows, num_heads, top_k, num_slots, device
    )
    q_rotated = torch.empty(
        (num_rows, num_heads, 512), device=device, dtype=torch.float32
    )
    mid = torch.empty(
        (num_rows, num_heads, num_splits, 514),
        device=device, dtype=torch.float32,
    )
    sm_scale = 1.0 / (576 ** 0.5)

    for _ in range(warmup):
        higgs_dense_2bit_mla_rotate_query(q_nope, q_rotated)
        higgs_dense_2bit_mla_decode_split(
            q_rotated, q_rope, compressed, page_table, mid, out, codebook,
            sm_scale,
        )
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        higgs_dense_2bit_mla_rotate_query(q_nope, q_rotated)
        higgs_dense_2bit_mla_decode_split(
            q_rotated, q_rope, compressed, page_table, mid, out, codebook,
            sm_scale,
        )
        ends[i].record()
    torch.cuda.synchronize()
    times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    return {
        "median_us": statistics.median(times_us),
        "min_us": min(times_us),
        "max_us": max(times_us),
        "mean_us": statistics.mean(times_us),
        "p10_us": statistics.quantiles(times_us, n=10)[0] if iters >= 10 else min(times_us),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batches", type=str, default="1,4,8,16")
    ap.add_argument("--num-heads", type=int, default=16)  # DP=TP=8, 128 heads / 8
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--num-splits", type=int, default=16)
    ap.add_argument(
        "--variants", type=str, default="split",
        help="Comma list of variants to bench: split (single dropped in iter3 #16)",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    batches = [int(b) for b in args.batches.split(",")]
    variants = args.variants.split(",")

    results = []
    for B in batches:
        for variant in variants:
            if variant == "split":
                r = _bench_split(
                    num_rows=B, num_heads=args.num_heads,
                    top_k=args.top_k, num_splits=args.num_splits,
                    iters=args.iters, warmup=args.warmup, device=device,
                )
            else:
                raise ValueError(
                    f"unknown variant {variant}; only 'split' is supported "
                    f"(single-pass was dropped in iter3 #16)"
                )
            r["variant"] = variant
            r["B"] = B
            r["num_heads"] = args.num_heads
            r["top_k"] = args.top_k
            if variant == "split":
                r["num_splits"] = args.num_splits
            results.append(r)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Plain table, sed-friendly.
        print(f"# B={batches} num_heads={args.num_heads} top_k={args.top_k}")
        hdr = "{:>8} {:>3} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "variant", "B", "median_us", "p10_us", "mean_us", "min_us", "max_us"
        )
        print(hdr)
        for r in results:
            print(
                "{:>8} {:>3} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
                    r["variant"], r["B"], r["median_us"], r["p10_us"],
                    r["mean_us"], r["min_us"], r["max_us"],
                )
            )


if __name__ == "__main__":
    main()
