#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from statistics import geometric_mean

import torch

from sglang.jit_kernel.gated_norm import (
    _TORCH_MM_MIN_TOKENS_ENV,
    _should_use_torch_mm,
    gated_norm_forward,
)


@contextmanager
def env_override(name: str, value: str | None):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def elapsed_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_case(
    tokens: int,
    hidden_size: int,
    rank: int,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, float | int | bool]:
    torch.manual_seed(seed + tokens * 17 + rank)
    normed = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    w_down = (
        torch.randn(rank, hidden_size, device="cuda", dtype=torch.bfloat16)
        / hidden_size**0.5
    )
    w_up = (
        torch.randn(hidden_size, rank, device="cuda", dtype=torch.bfloat16)
        / rank**0.5
    )
    current_out = torch.empty_like(normed)
    candidate_out = torch.empty_like(normed)

    with env_override(_TORCH_MM_MIN_TOKENS_ENV, "-1"):
        current = lambda: gated_norm_forward(normed, w_down, w_up, out=current_out)
        current()

    candidate_uses_torch_mm = _should_use_torch_mm(tokens, rank)
    candidate = lambda: gated_norm_forward(normed, w_down, w_up, out=candidate_out)
    candidate()
    torch.testing.assert_close(candidate_out, current_out, atol=2e-2, rtol=2e-2)

    with env_override(_TORCH_MM_MIN_TOKENS_ENV, "-1"):
        current_ms = elapsed_ms(current, warmup, iters)
    candidate_ms = elapsed_ms(candidate, warmup, iters)
    speedup = current_ms / candidate_ms
    return {
        "tokens": tokens,
        "hidden_size": hidden_size,
        "rank": rank,
        "current_ms": current_ms,
        "candidate_ms": candidate_ms,
        "speedup": speedup,
        "candidate_uses_torch_mm": candidate_uses_torch_mm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark BF16 REAP GatedNorm paths.")
    parser.add_argument("--tokens", default="1,8,64,256,512,1024,2048,4096")
    parser.add_argument("--ranks", default="8,32,64")
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2030)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("bench_gated_norm.py requires CUDA")

    rows = []
    for rank in parse_int_list(args.ranks):
        for tokens in parse_int_list(args.tokens):
            row = run_case(
                tokens=tokens,
                hidden_size=args.hidden_size,
                rank=rank,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    decode_regressions = [
        max((row["candidate_ms"] / row["current_ms"]) - 1.0, 0.0)
        for row in rows
        if not row["candidate_uses_torch_mm"]
    ]
    prefill_speedups = [
        row["speedup"] for row in rows if row["candidate_uses_torch_mm"]
    ]
    all_speedups = [row["speedup"] for row in rows]
    summary = {
        "gatednorm_min_speedup": min(all_speedups),
        "gatednorm_best_speedup": max(all_speedups),
        "gatednorm_decode_regression_pct": max(decode_regressions, default=0.0) * 100.0,
        "gatednorm_prefill_speedup": geometric_mean(prefill_speedups)
        if prefill_speedups
        else 1.0,
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    for key, value in summary.items():
        print(f"{key}={value:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
