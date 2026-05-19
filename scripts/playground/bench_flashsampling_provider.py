#!/usr/bin/env python3
"""Kernel-only FlashSampling provider smoke benchmark."""

import argparse
import json
import math
import pathlib
import sys
import types

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", nargs="+", default=["triton", "target"])
    parser.add_argument("--vocab-size", type=int, default=16160)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 64])
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=pathlib.Path)
    parser.add_argument("--target-block-v", type=int)
    parser.add_argument("--target-block-d", type=int)
    parser.add_argument("--target-num-warps", type=int)
    parser.add_argument("--target-num-stages", type=int)
    parser.add_argument(
        "--direct-module-import",
        action="store_true",
        help=(
            "Bypass flashsampling.__init__ and import provider modules directly. "
            "Use only when the local editable sglang-kernel package is broken."
        ),
    )
    return parser.parse_args()


def prepare_direct_import():
    package_name = "sglang.srt.layers.flashsampling"
    if package_name in sys.modules:
        return

    import sglang.srt.layers as layers_package

    package = types.ModuleType(package_name)
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    package.__path__ = [
        str(repo_root / "python" / "sglang" / "srt" / "layers" / "flashsampling")
    ]
    sys.modules[package_name] = package
    setattr(layers_package, "flashsampling", package)


def apply_target_overrides(args):
    import sglang.srt.layers.flashsampling.target_kernel_blackwell as blackwell

    overrides = {}
    if args.target_block_v is not None:
        blackwell._BLOCK_V_BLACKWELL = args.target_block_v
        overrides["BLOCK_V"] = args.target_block_v
    if args.target_block_d is not None:
        blackwell._BLOCK_D_BLACKWELL = args.target_block_d
        overrides["BLOCK_D"] = args.target_block_d
    if args.target_num_warps is not None:
        blackwell._NUM_WARPS_BLACKWELL = args.target_num_warps
        overrides["num_warps"] = args.target_num_warps
    if args.target_num_stages is not None:
        blackwell._num_stages_blackwell = lambda _block_h: args.target_num_stages
        overrides["num_stages"] = args.target_num_stages
    return overrides


def load_provider(name: str, args):
    if name == "triton":
        from sglang.srt.layers.flashsampling.core import fused_mm_sample_triton

        return "triton", fused_mm_sample_triton
    if name == "target_generic":
        from sglang.srt.layers.flashsampling.target_kernel import fused_mm_sample_target

        return "target_generic", fused_mm_sample_target
    if name == "target":
        if torch.cuda.get_device_capability()[0] >= 10:
            from sglang.srt.layers.flashsampling.target_kernel_blackwell import (
                fused_mm_sample_blackwell,
            )

            overrides = apply_target_overrides(args)
            if overrides:
                suffix = "_" + "_".join(
                    f"{key}{value}" for key, value in sorted(overrides.items())
                )
            else:
                suffix = ""
            return "target_blackwell" + suffix, fused_mm_sample_blackwell
        from sglang.srt.layers.flashsampling.target_kernel import fused_mm_sample_target

        return "target", fused_mm_sample_target
    raise ValueError(f"unknown provider {name!r}")


def measure(fn, warmup: int, iters: int) -> float:
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


def main():
    args = parse_args()
    if args.direct_module_import:
        prepare_direct_import()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(17)

    weights = (
        torch.randn(
            args.vocab_size,
            args.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / math.sqrt(args.hidden_size)
    ).contiguous()
    temperature = torch.tensor(1.0, device=device, dtype=torch.float32)
    providers = [load_provider(name, args) for name in args.providers]

    rows = []
    for batch_size in args.batch_sizes:
        hidden_states = torch.randn(
            batch_size,
            args.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        ).contiguous()
        expected = (hidden_states @ weights.T).float().argmax(dim=-1, keepdim=True)
        for provider_name, provider_fn in providers:
            seed = [0]

            def call():
                seed[0] += 1
                return provider_fn(
                    weights=weights,
                    hidden_states=hidden_states,
                    num_samples=1,
                    temperature=temperature,
                    seed=seed[0],
                    greedy_sampling=True,
                    valid_vocab_size=args.vocab_size,
                )

            samples = call()
            ms = measure(call, args.warmup, args.iters)
            bytes_read = (
                args.vocab_size * args.hidden_size
                + batch_size * args.hidden_size
            ) * 2
            rows.append(
                {
                    "provider": provider_name,
                    "batch_size": batch_size,
                    "ms": ms,
                    "approx_read_tb_s": bytes_read / (ms / 1000) / 1e12,
                    "greedy_correct": bool(torch.equal(samples.cpu(), expected.cpu())),
                    "shape": {
                        "vocab": args.vocab_size,
                        "hidden": args.hidden_size,
                    },
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "device": args.device,
                    "direct_module_import": args.direct_module_import,
                    "target_overrides": {
                        "BLOCK_V": args.target_block_v,
                        "BLOCK_D": args.target_block_d,
                        "num_warps": args.target_num_warps,
                        "num_stages": args.target_num_stages,
                    },
                }
            )

    print("provider,batch_size,ms,approx_read_TB_s,greedy_correct")
    for row in rows:
        print(
            f"{row['provider']},{row['batch_size']},{row['ms']:.6f},"
            f"{row['approx_read_tb_s']:.3f},{row['greedy_correct']}"
        )

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
