#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from statistics import geometric_mean

REPO_ROOT = Path(__file__).resolve().parents[2]
GATED_NORM_MODULE_PATH = REPO_ROOT / "python/sglang/jit_kernel/gated_norm.py"
KERNEL_API_LOGGING_PATH = REPO_ROOT / "python/sglang/kernel_api_logging.py"

SCRIPT_MODES = {
    "production": "Current production selector.",
    "incumbent": "Pre-optimization selector: no CuTe-first and sigmoid*mul fuse floor 1024.",
    "torch_mm_all": "Force torch.mm/cuBLAS path for all shapes.",
    "torch_mm_rank8_decode": "Local probe lowering rank>=8 torch.mm threshold to one token.",
    "torch_mm_lowrank_prefill16": "Local probe pulling low-rank prefill earlier.",
    "sigmoid_fuse_256": "Local probe lowering fused sigmoid*mul floor to 256 tokens.",
    "sigmoid_fuse_384": "Local probe lowering fused sigmoid*mul floor to 384 tokens.",
    "sigmoid_fuse_448": "Local probe lowering fused sigmoid*mul floor to 448 tokens.",
    "sigmoid_fuse_480": "Local probe lowering fused sigmoid*mul floor to 480 tokens.",
    "cute_first_supported": "Local probe trying CuTe before torch.mm for all CuTe-accepted shapes.",
    "cute_no_decline_experimental": "Local probe bypassing Python CuTe decline guards with runtime fallback.",
}


def _load_module_from_path(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_gated_norm_module():
    if "sglang" not in sys.modules:
        sglang_module = types.ModuleType("sglang")
        sglang_module.__path__ = [str(REPO_ROOT / "python/sglang")]
        sys.modules["sglang"] = sglang_module
    if "sglang.jit_kernel" not in sys.modules:
        jit_kernel_module = types.ModuleType("sglang.jit_kernel")
        jit_kernel_module.__path__ = [str(REPO_ROOT / "python/sglang/jit_kernel")]
        sys.modules["sglang.jit_kernel"] = jit_kernel_module

    _load_module_from_path("sglang.kernel_api_logging", KERNEL_API_LOGGING_PATH)
    return _load_module_from_path(
        "sglang.jit_kernel.gated_norm", GATED_NORM_MODULE_PATH
    )


@contextlib.contextmanager
def env_override(name: str, value: str | None) -> Iterator[None]:
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


@contextlib.contextmanager
def dispatch_mode(gated_norm, mode: str) -> Iterator[None]:
    if mode not in SCRIPT_MODES:
        valid = ", ".join(sorted(SCRIPT_MODES))
        raise ValueError(f"unknown GatedNorm benchmark mode {mode!r}; valid: {valid}")

    old_should_use_torch_mm = gated_norm._should_use_torch_mm
    old_should_try_cute = gated_norm._should_try_cute_before_torch_mm
    old_cute_declines = gated_norm._cute_declines_shape
    old_cute_forward = gated_norm._gated_norm_cute_forward
    fuse_env = gated_norm._SIGMOID_MUL_FUSE_MIN_TOKENS_ENV

    def threshold_policy(thresholds: tuple[tuple[int, int], ...]):
        def should_use(num_tokens: int, rank: int) -> bool:
            for rank_floor, tokens in thresholds:
                if rank >= rank_floor:
                    return num_tokens >= tokens
            return old_should_use_torch_mm(num_tokens, rank)

        return should_use

    try:
        if mode == "incumbent":
            gated_norm._should_try_cute_before_torch_mm = lambda _tokens, _rank: False
            with env_override(fuse_env, "1024"):
                yield
        elif mode == "production":
            with env_override(fuse_env, None):
                yield
        elif mode == "torch_mm_all":
            gated_norm._should_try_cute_before_torch_mm = lambda _tokens, _rank: False
            gated_norm._should_use_torch_mm = lambda _tokens, _rank: True
            with env_override(fuse_env, None):
                yield
        elif mode == "torch_mm_rank8_decode":
            gated_norm._should_try_cute_before_torch_mm = lambda _tokens, _rank: False
            gated_norm._should_use_torch_mm = threshold_policy(
                ((64, 1), (32, 1), (16, 1), (8, 1), (5, 64), (1, 1))
            )
            with env_override(fuse_env, None):
                yield
        elif mode == "torch_mm_lowrank_prefill16":
            gated_norm._should_try_cute_before_torch_mm = lambda _tokens, _rank: False
            gated_norm._should_use_torch_mm = threshold_policy(
                ((64, 1), (32, 1), (16, 1), (8, 8), (5, 16), (1, 1))
            )
            with env_override(fuse_env, None):
                yield
        elif mode == "sigmoid_fuse_256":
            with env_override(fuse_env, "256"):
                yield
        elif mode == "sigmoid_fuse_384":
            with env_override(fuse_env, "384"):
                yield
        elif mode == "sigmoid_fuse_448":
            with env_override(fuse_env, "448"):
                yield
        elif mode == "sigmoid_fuse_480":
            with env_override(fuse_env, "480"):
                yield
        elif mode == "cute_first_supported":
            gated_norm._should_try_cute_before_torch_mm = lambda _tokens, _rank: True
            with env_override(fuse_env, None):
                yield
        elif mode == "cute_no_decline_experimental":
            gated_norm._should_try_cute_before_torch_mm = lambda _tokens, _rank: True
            gated_norm._cute_declines_shape = lambda _tokens, _rank: False

            def runtime_fallback(*args, **kwargs):
                try:
                    return old_cute_forward(*args, **kwargs)
                except RuntimeError:
                    return False

            gated_norm._gated_norm_cute_forward = runtime_fallback
            with env_override(fuse_env, None):
                yield
    finally:
        gated_norm._should_use_torch_mm = old_should_use_torch_mm
        gated_norm._should_try_cute_before_torch_mm = old_should_try_cute
        gated_norm._cute_declines_shape = old_cute_declines
        gated_norm._gated_norm_cute_forward = old_cute_forward


def parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def elapsed_ms(fn, warmup: int, iters: int) -> float:
    import torch

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
    gated_norm,
    baseline_mode: str,
    mode: str,
    tokens: int,
    hidden_size: int,
    rank: int,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, float | int | bool | str]:
    import torch

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
    baseline_out = torch.empty_like(normed)
    mode_out = torch.empty_like(normed)

    with dispatch_mode(gated_norm, baseline_mode):
        baseline_uses_torch_mm = gated_norm._should_use_torch_mm(tokens, rank)

        def baseline() -> None:
            gated_norm.gated_norm_forward(normed, w_down, w_up, out=baseline_out)

        baseline()
    with dispatch_mode(gated_norm, mode):
        mode_uses_torch_mm = gated_norm._should_use_torch_mm(tokens, rank)

        def candidate() -> None:
            gated_norm.gated_norm_forward(normed, w_down, w_up, out=mode_out)

        candidate()
    torch.testing.assert_close(mode_out, baseline_out, atol=2e-2, rtol=2e-2)

    with dispatch_mode(gated_norm, baseline_mode):
        baseline_ms = elapsed_ms(baseline, warmup, iters)
    with dispatch_mode(gated_norm, mode):
        mode_ms = elapsed_ms(candidate, warmup, iters)
    speedup = baseline_ms / mode_ms
    return {
        "mode": mode,
        "baseline_mode": baseline_mode,
        "tokens": tokens,
        "hidden_size": hidden_size,
        "rank": rank,
        "baseline_ms": baseline_ms,
        "mode_ms": mode_ms,
        "speedup": speedup,
        "baseline_uses_torch_mm": baseline_uses_torch_mm,
        "mode_uses_torch_mm": mode_uses_torch_mm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark BF16 REAP GatedNorm paths.")
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="Print local playground benchmark modes and exit.",
    )
    parser.add_argument(
        "--modes",
        default="production",
        help="Comma-separated local benchmark modes.",
    )
    parser.add_argument(
        "--candidates",
        help="Deprecated alias for --modes, kept for old artifact reproduction.",
    )
    parser.add_argument("--baseline-mode", default="incumbent", choices=sorted(SCRIPT_MODES))
    parser.add_argument("--tokens", default="1,8,64,256,512,1024,2048,4096")
    parser.add_argument("--ranks", default="8,32,64")
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2030)
    args = parser.parse_args()

    if args.list_modes:
        print(json.dumps({"modes": SCRIPT_MODES}, indent=2, sort_keys=True))
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("bench_gated_norm.py requires CUDA")

    gated_norm = _load_gated_norm_module()
    mode_text = args.candidates if args.candidates is not None else args.modes
    mode_names = [item.strip() for item in mode_text.split(",") if item.strip()]
    if not mode_names:
        raise ValueError("--modes must include at least one mode")

    rows = []
    for mode in mode_names:
        if mode not in SCRIPT_MODES:
            valid = ", ".join(sorted(SCRIPT_MODES))
            raise ValueError(f"unknown GatedNorm benchmark mode {mode!r}; valid: {valid}")
        for rank in parse_int_list(args.ranks):
            for tokens in parse_int_list(args.tokens):
                row = run_case(
                    gated_norm=gated_norm,
                    baseline_mode=args.baseline_mode,
                    mode=mode,
                    tokens=tokens,
                    hidden_size=args.hidden_size,
                    rank=rank,
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=args.seed,
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    regressions = [max((row["mode_ms"] / row["baseline_ms"]) - 1.0, 0.0) for row in rows]
    all_speedups = [row["speedup"] for row in rows]
    summary = {
        "gatednorm_modes": sorted({str(row["mode"]) for row in rows}),
        "gatednorm_baseline_mode": args.baseline_mode,
        "gatednorm_min_speedup": min(all_speedups),
        "gatednorm_best_speedup": max(all_speedups),
        "gatednorm_geomean_speedup": geometric_mean(all_speedups),
        "gatednorm_max_regression_pct": max(regressions, default=0.0) * 100.0,
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}={value:.6f}", flush=True)
        else:
            print(f"{key}={json.dumps(value, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
