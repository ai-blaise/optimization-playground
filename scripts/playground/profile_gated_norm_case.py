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
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
GATED_NORM_MODULE_PATH = REPO_ROOT / "python/sglang/jit_kernel/gated_norm.py"
KERNEL_API_LOGGING_PATH = REPO_ROOT / "python/sglang/kernel_api_logging.py"

SCRIPT_CANDIDATES = {
    "torch_mm_all",
    "torch_mm_rank8_decode",
    "torch_mm_lowrank_prefill16",
    "sigmoid_fuse_256",
    "sigmoid_fuse_384",
    "sigmoid_fuse_448",
    "sigmoid_fuse_480",
    "cute_first_supported",
    "cute_no_decline_experimental",
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
def _env_override(name: str, value: str | None) -> Iterator[None]:
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
def _dispatch_mode(gated_norm, mode: str, candidate: str | None) -> Iterator[None]:
    old_should_use_torch_mm: Callable[[int, int], bool] = gated_norm._should_use_torch_mm
    old_should_try_cute: Callable[[int, int], bool] = gated_norm._should_try_cute_before_torch_mm
    old_cute_declines: Callable[[int, int], bool] = gated_norm._cute_declines_shape
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
            with _env_override(fuse_env, "1024"):
                yield
        elif mode == "production":
            with _env_override(fuse_env, None):
                yield
        elif mode == "candidate":
            if not candidate:
                raise ValueError("--candidate is required when --mode=candidate")
            if candidate not in SCRIPT_CANDIDATES:
                valid = ", ".join(sorted(SCRIPT_CANDIDATES))
                raise ValueError(f"unknown script-local candidate {candidate!r}; valid: {valid}")
            if candidate == "torch_mm_all":
                gated_norm._should_try_cute_before_torch_mm = lambda _tokens, _rank: False
                gated_norm._should_use_torch_mm = lambda _tokens, _rank: True
                with _env_override(fuse_env, None):
                    yield
            elif candidate == "torch_mm_rank8_decode":
                gated_norm._should_try_cute_before_torch_mm = lambda _tokens, _rank: False
                gated_norm._should_use_torch_mm = threshold_policy(
                    ((64, 1), (32, 1), (16, 1), (8, 1), (5, 64), (1, 1))
                )
                with _env_override(fuse_env, None):
                    yield
            elif candidate == "torch_mm_lowrank_prefill16":
                gated_norm._should_try_cute_before_torch_mm = lambda _tokens, _rank: False
                gated_norm._should_use_torch_mm = threshold_policy(
                    ((64, 1), (32, 1), (16, 1), (8, 8), (5, 16), (1, 1))
                )
                with _env_override(fuse_env, None):
                    yield
            elif candidate == "sigmoid_fuse_256":
                with _env_override(fuse_env, "256"):
                    yield
            elif candidate == "sigmoid_fuse_384":
                with _env_override(fuse_env, "384"):
                    yield
            elif candidate == "sigmoid_fuse_448":
                with _env_override(fuse_env, "448"):
                    yield
            elif candidate == "sigmoid_fuse_480":
                with _env_override(fuse_env, "480"):
                    yield
            elif candidate == "cute_first_supported":
                gated_norm._should_try_cute_before_torch_mm = lambda _tokens, _rank: True
                with _env_override(fuse_env, None):
                    yield
            elif candidate == "cute_no_decline_experimental":
                gated_norm._should_try_cute_before_torch_mm = lambda _tokens, _rank: True
                gated_norm._cute_declines_shape = lambda _tokens, _rank: False

                def runtime_fallback(*args, **kwargs):
                    try:
                        return old_cute_forward(*args, **kwargs)
                    except RuntimeError:
                        return False

                gated_norm._gated_norm_cute_forward = runtime_fallback
                with _env_override(fuse_env, None):
                    yield
        else:
            raise ValueError(f"unknown mode: {mode}")
    finally:
        gated_norm._should_use_torch_mm = old_should_use_torch_mm
        gated_norm._should_try_cute_before_torch_mm = old_should_try_cute
        gated_norm._cute_declines_shape = old_cute_declines
        gated_norm._gated_norm_cute_forward = old_cute_forward


def _elapsed_ms(fn, warmup: int, iters: int) -> float:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile one BF16 GatedNorm dispatch case.")
    parser.add_argument("--mode", choices=("incumbent", "production", "candidate"), required=True)
    parser.add_argument("--candidate", default="")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Start/stop CUDA profiling around the measured loop for nsys --capture-range=cudaProfilerApi.",
    )
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("profile_gated_norm_case.py requires CUDA")

    gated_norm = _load_gated_norm_module()

    torch.manual_seed(args.seed + args.tokens * 17 + args.rank)
    normed = torch.randn(
        args.tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    w_down = torch.randn(
        args.rank, args.hidden_size, device="cuda", dtype=torch.bfloat16
    ) / args.hidden_size**0.5
    w_up = torch.randn(
        args.hidden_size, args.rank, device="cuda", dtype=torch.bfloat16
    ) / args.rank**0.5
    output = torch.empty_like(normed)

    with torch.no_grad():
        z = torch.mm(normed, w_down.t())
        F.silu(z, inplace=True)
        logits = torch.mm(z, w_up.t())
        reference = normed * torch.sigmoid(logits)

    with _dispatch_mode(gated_norm, args.mode, args.candidate or None):
        def fn() -> None:
            gated_norm.gated_norm_forward(normed, w_down, w_up, out=output)

        fn()
        torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
        elapsed = _elapsed_ms(fn, args.warmup, args.iters)

        if args.capture:
            for _ in range(args.warmup):
                fn()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(
                f"gated_norm:{args.mode}:{args.candidate or 'default'}:"
                f"r{args.rank}:t{args.tokens}"
            )
            for _ in range(args.iters):
                fn()
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
            torch.cuda.synchronize()

    print(
        json.dumps(
            {
                "mode": args.mode,
                "candidate": args.candidate or None,
                "tokens": args.tokens,
                "rank": args.rank,
                "hidden_size": args.hidden_size,
                "warmup": args.warmup,
                "iters": args.iters,
                "elapsed_ms": elapsed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
