#!/usr/bin/env python3
"""Compare dense MLA KV compression candidates for DeepSeek NSA.

This benchmark intentionally isolates the dense MLA KV storage problem before
wiring a candidate into the NSA runtime.  It compares:

* NVFP4 dense MLA KV: packed E2M1 data plus per-block FP8 scales.
* dense TurboQuant 2.5-bit: the current compressed dense MLA KV path.

Both candidates operate on the DeepSeek NSA dense MLA dimensions used by the
target REAP model: 512 latent channels plus 64 rope channels.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from types import SimpleNamespace
from typing import Callable

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil
from sglang.srt.layers.quantization.turboquant_dense_kv import (
    TurboQuantDenseKVConfig,
)
from sglang.srt.mem_cache.memory_pool import TurboQuantNSATokenToKVPool
from sglang.srt.utils import is_sm90_supported, is_sm100_supported


LATENT_DIM = 512
ROPE_DIM = 64
KV_DIM = LATENT_DIM + ROPE_DIM


def _cuda_time_ms(fn: Callable[[], None], warmup: int, iters: int) -> float:
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


def _mean_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        torch.nn.functional.cosine_similarity(
            a.float().reshape(a.shape[0], -1),
            b.float().reshape(b.shape[0], -1),
            dim=-1,
        )
        .mean()
        .item()
    )


def _candidate_accuracy(reference: torch.Tensor, recovered: torch.Tensor) -> dict:
    latent_ref = reference[..., :LATENT_DIM]
    latent_out = recovered[..., :LATENT_DIM]
    rope_ref = reference[..., LATENT_DIM:]
    rope_out = recovered[..., LATENT_DIM:]
    return {
        "latent_cosine_mean": _mean_cosine(latent_ref, latent_out),
        "latent_mse": torch.mean((latent_ref.float() - latent_out.float()) ** 2).item(),
        "rope_exact": bool(torch.equal(rope_ref, rope_out)),
        "rope_mse": torch.mean((rope_ref.float() - rope_out.float()) ** 2).item(),
    }


def _make_turboquant_pool(num_tokens: int) -> TurboQuantNSATokenToKVPool:
    size = ((num_tokens + 63) // 64) * 64
    return TurboQuantNSATokenToKVPool(
        size=size,
        page_size=64,
        kv_lora_rank=LATENT_DIM,
        dtype=torch.bfloat16,
        qk_rope_head_dim=ROPE_DIM,
        layer_num=1,
        device="cuda",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=KV_DIM,
        turboquant_dense_kv_preset="latent_2p5bit_nc",
        turboquant_execution_mode="fused_decode",
    )


def benchmark_nvfp4(
    kv: torch.Tensor,
    loc: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict:
    if not (is_sm90_supported() or is_sm100_supported()):
        raise RuntimeError("NVFP4 KV quantization requires an SM90+ GPU")

    global_scale = torch.ones(1, dtype=torch.float32, device=kv.device)
    fp4, scales, _ = NVFP4KVQuantizeUtil.quantize(kv, global_scale)

    holder: dict[str, torch.Tensor] = {"fp4": fp4, "scales": scales}

    def store_once() -> None:
        holder["fp4"], holder["scales"], _ = NVFP4KVQuantizeUtil.quantize(
            kv, global_scale
        )

    def recover_once() -> None:
        holder["recovered"] = NVFP4KVQuantizeUtil.dequantize(
            holder["fp4"][loc],
            holder["scales"][loc],
            global_scale,
            dtype=torch.bfloat16,
        )

    store_ms = _cuda_time_ms(store_once, warmup, iters)
    recover_ms = _cuda_time_ms(recover_once, warmup, iters)
    recover_once()
    torch.cuda.synchronize()

    bytes_per_token = fp4[0].nbytes + scales[0].nbytes
    recovered = holder["recovered"]
    reference = kv[loc]
    return {
        "candidate": "nvfp4_dense_mla",
        "bytes_per_token_layer": bytes_per_token,
        "store_ms": store_ms,
        "selected_recover_ms": recover_ms,
        "selected_tokens": int(loc.numel()),
        **_candidate_accuracy(reference, recovered),
    }


def benchmark_turboquant(
    latent: torch.Tensor,
    rope: torch.Tensor,
    loc: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict:
    pool = _make_turboquant_pool(latent.shape[0])
    all_loc = torch.arange(latent.shape[0], device="cuda", dtype=torch.int64)

    def store_once() -> None:
        pool.set_mla_kv_buffer(SimpleNamespace(layer_id=0), all_loc, latent, rope)

    page_table = loc.reshape(1, -1).to(torch.int32).contiguous()

    def recover_once() -> None:
        holder["recovered"], holder["page_table"] = pool.get_turboquant_selected_kv_buffer(
            0,
            page_table,
        )

    holder: dict[str, torch.Tensor] = {}
    store_ms = _cuda_time_ms(store_once, warmup, iters)
    recover_ms = _cuda_time_ms(recover_once, warmup, iters)
    recover_once()
    torch.cuda.synchronize()

    reference = torch.cat((latent, rope), dim=-1)[loc]
    return {
        "candidate": "turboquant_dense_mla_2p5",
        "bytes_per_token_layer": TurboQuantDenseKVConfig(
            latent_dim=LATENT_DIM,
            rope_dim=ROPE_DIM,
            preset="latent_2p5bit_nc",
        ).slot_bytes,
        "store_ms": store_ms,
        "selected_recover_ms": recover_ms,
        "selected_tokens": int(loc.numel()),
        **_candidate_accuracy(reference, holder["recovered"]),
    }


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    latent = torch.randn(
        args.num_tokens,
        1,
        LATENT_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    rope = torch.randn(
        args.num_tokens,
        1,
        ROPE_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv = torch.cat((latent, rope), dim=-1).contiguous()
    loc = torch.randperm(args.num_tokens, device="cuda", dtype=torch.int64)[
        : args.selected_tokens
    ].contiguous()

    candidates = [
        benchmark_nvfp4(kv, loc, args.warmup, args.iters),
        benchmark_turboquant(latent, rope, loc, args.warmup, args.iters),
    ]
    bf16_bytes = KV_DIM * torch.tensor([], dtype=torch.bfloat16).element_size()
    for result in candidates:
        result["relative_to_bf16"] = result["bytes_per_token_layer"] / bf16_bytes
    by_name = {result["candidate"]: result for result in candidates}
    nvfp4 = by_name["nvfp4_dense_mla"]
    turboquant = by_name["turboquant_dense_mla_2p5"]

    return {
        "shape": {
            "num_tokens": args.num_tokens,
            "selected_tokens": args.selected_tokens,
            "latent_dim": LATENT_DIM,
            "rope_dim": ROPE_DIM,
            "bf16_bytes_per_token_layer": bf16_bytes,
        },
        "candidates": candidates,
        "comparison": {
            "turboquant_bytes_saved_vs_nvfp4_per_token_layer": (
                nvfp4["bytes_per_token_layer"] - turboquant["bytes_per_token_layer"]
            ),
            "turboquant_percent_smaller_than_nvfp4": (
                1.0
                - turboquant["bytes_per_token_layer"]
                / nvfp4["bytes_per_token_layer"]
            )
            * 100.0,
            "turboquant_selected_recover_speedup_vs_nvfp4": (
                nvfp4["selected_recover_ms"] / turboquant["selected_recover_ms"]
            ),
            "turboquant_store_speedup_vs_nvfp4": (
                nvfp4["store_ms"] / turboquant["store_ms"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--selected-tokens", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=pathlib.Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.selected_tokens > args.num_tokens:
        raise SystemExit("--selected-tokens cannot exceed --num-tokens")

    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n")


if __name__ == "__main__":
    main()
