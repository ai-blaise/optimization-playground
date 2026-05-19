#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATED_NORM_MODULE_PATH = REPO_ROOT / "python/sglang/jit_kernel/gated_norm.py"
KERNEL_API_LOGGING_PATH = REPO_ROOT / "python/sglang/kernel_api_logging.py"
DEFAULT_FLASH_ROOT = Path("/root/b200-run-20260518/repos/Megatron-LM")


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


def _load_current_gated_norm():
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


def _install_flashtraining_utils_stub(flash_root: Path) -> None:
    if "megatron" not in sys.modules:
        megatron_module = types.ModuleType("megatron")
        megatron_module.__path__ = [str(flash_root / "megatron")]
        sys.modules["megatron"] = megatron_module
    if "megatron.core" not in sys.modules:
        core_module = types.ModuleType("megatron.core")
        core_module.__path__ = [str(flash_root / "megatron/core")]
        sys.modules["megatron.core"] = core_module
    if "megatron.core.utils" not in sys.modules:
        utils_module = types.ModuleType("megatron.core.utils")

        def null_decorator(fn=None, *args, **kwargs):
            if fn is None:
                return lambda wrapped: wrapped
            return fn

        utils_module.null_decorator = null_decorator
        sys.modules["megatron.core.utils"] = utils_module


def _load_flashtraining_gated_norm(flash_root: Path):
    _install_flashtraining_utils_stub(flash_root)
    path = flash_root / "megatron/core/fusions/gated_norm.py"
    return _load_module_from_path("flashtraining_gated_norm_ref", path)


def _git_ref(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def _elapsed_ms(fn, warmup: int, iters: int) -> float:
    import torch

    with torch.no_grad():
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


def _make_reference(normed, w_down, w_up):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        z = torch.mm(normed, w_down.t())
        F.silu(z, inplace=True)
        logits = torch.mm(z, w_up.t())
        return normed * torch.sigmoid(logits)


def _run_case(current, flash, args, tokens: int, rank: int) -> dict[str, object]:
    import torch

    torch.manual_seed(args.seed + tokens * 17 + rank)
    normed = torch.randn(
        tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    w_down = torch.randn(
        rank, args.hidden_size, device="cuda", dtype=torch.bfloat16
    ) / args.hidden_size**0.5
    w_up = torch.randn(
        args.hidden_size, rank, device="cuda", dtype=torch.bfloat16
    ) / rank**0.5
    reference = _make_reference(normed, w_down, w_up)

    if args.current_preallocate_out:
        current_out = torch.empty_like(normed)

        def current_fn() -> torch.Tensor:
            return current.gated_norm_forward(normed, w_down, w_up, out=current_out)

    else:

        def current_fn() -> torch.Tensor:
            return current.gated_norm_forward(normed, w_down, w_up)

    def flash_fn() -> torch.Tensor:
        return flash.apply_gated_norm(normed, w_down, w_up)

    current_result = current_fn()
    flash_result = flash_fn()
    torch.cuda.synchronize()
    torch.testing.assert_close(current_result, reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(flash_result, reference, atol=2e-2, rtol=2e-2)

    current_ms = _elapsed_ms(current_fn, args.warmup, args.iters)
    flash_ms = _elapsed_ms(flash_fn, args.warmup, args.iters)
    delta_pct = (flash_ms - current_ms) / flash_ms * 100.0
    return {
        "tokens": tokens,
        "rank": rank,
        "hidden_size": args.hidden_size,
        "current_ms": current_ms,
        "flashtraining_ms": flash_ms,
        "current_speedup_vs_flashtraining": flash_ms / current_ms,
        "current_delta_pct_vs_flashtraining": delta_pct,
        "current_correct": True,
        "flashtraining_correct": True,
        "current_preallocate_out": args.current_preallocate_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare optimization-playground GatedNorm to "
            "ai-blaise/Megatron-LM flashtraining."
        )
    )
    parser.add_argument("--flash-root", type=Path, default=DEFAULT_FLASH_ROOT)
    parser.add_argument("--tokens", default="1,8,16,64,256,512,1024,2048,4096")
    parser.add_argument("--ranks", default="1,5,8,16,32,40,48,64")
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=160)
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument(
        "--current-preallocate-out",
        action="store_true",
        help=(
            "Use the SGLang out= path. Default compares allocation-returning "
            "public forwards."
        ),
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GatedNorm flashtraining comparison")

    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        "/root/b200-run-20260518/metrics/gated_norm_flashtraining_torch_extensions",
    )
    current = _load_current_gated_norm()
    flash = _load_flashtraining_gated_norm(args.flash_root)
    header = {
        "comparator": (
            "ai-blaise/Megatron-LM "
            "megatron.core.fusions.gated_norm.apply_gated_norm"
        ),
        "flashtraining_root": str(args.flash_root),
        "flashtraining_ref": _git_ref(args.flash_root),
        "optimization_playground_ref": _git_ref(REPO_ROOT),
        "shape_matrix": {
            "tokens": _parse_int_list(args.tokens),
            "ranks": _parse_int_list(args.ranks),
            "hidden_size": args.hidden_size,
        },
    }
    print(json.dumps({"metadata": header}, sort_keys=True), flush=True)
    for rank in _parse_int_list(args.ranks):
        for tokens in _parse_int_list(args.tokens):
            row = _run_case(current, flash, args, tokens, rank)
            print(json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
