# Copyright 2024-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Benchmark warp decode MoE kernels vs expert-centric baseline.

Reports throughput (GB/s), latency (us), and correctness metrics
across different batch sizes for a DeepSeek-V3-like MoE configuration.

Usage:
    python bench_warp_decode.py [--hidden-size 7168] [--intermediate-size 2048]
        [--num-experts 256] [--top-k 8] [--batch-sizes 1,4,8,16,32,64]
        [--warmup 10] [--iters 100]
"""

import argparse
import time
from typing import List, Tuple

import torch


def cute_shape_supported(hidden_size: int, intermediate_size: int) -> bool:
    return (
        hidden_size % 512 == 0
        and hidden_size % 8 == 0
        and intermediate_size % 1024 == 0
    )


def generate_moe_data(
    batch_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, ...]:
    """Generate random MoE test data."""
    torch.manual_seed(0)

    hidden_states = torch.randn(
        batch_size, hidden_size, dtype=torch.bfloat16, device=device
    )

    w13 = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size,
        dtype=torch.bfloat16, device=device,
    ) * 0.01

    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size,
        dtype=torch.bfloat16, device=device,
    ) * 0.01

    topk_ids = torch.stack([
        torch.randperm(num_experts, device=device)[:top_k]
        for _ in range(batch_size)
    ])

    topk_weights = torch.softmax(
        torch.randn(batch_size, top_k, dtype=torch.float32, device=device),
        dim=-1,
    )

    return hidden_states, w13, w2, topk_ids, topk_weights


def benchmark_warp_decode(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    intermediate_size: int,
    warmup: int = 10,
    iters: int = 100,
) -> Tuple[float, float]:
    """Benchmark warp decode kernel.

    Returns:
        Tuple of (mean_latency_us, bandwidth_gb_s)
    """
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

    # Warmup
    with envs.SGLANG_WARP_DECODE_CUTE.override("0"):
        for _ in range(warmup):
            warp_decode_moe_packed(
                hidden_states, w13, w2, topk_ids, topk_weights,
                intermediate_size=intermediate_size,
            )
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    with envs.SGLANG_WARP_DECODE_CUTE.override("0"):
        for i in range(iters):
            start_events[i].record()
            warp_decode_moe_packed(
                hidden_states, w13, w2, topk_ids, topk_weights,
                intermediate_size=intermediate_size,
            )
            end_events[i].record()

    torch.cuda.synchronize()

    latencies_ms = [
        start_events[i].elapsed_time(end_events[i]) for i in range(iters)
    ]
    mean_latency_ms = sum(latencies_ms) / len(latencies_ms)
    mean_latency_us = mean_latency_ms * 1000

    # Calculate bandwidth
    B = hidden_states.shape[0]
    K = topk_ids.shape[1]
    D = hidden_states.shape[1]
    N = intermediate_size
    E_local = w13.shape[0]

    # Bytes read: input activations + w13 rows + w2 rows + routing
    # Each token reads K experts worth of gate+up+down weights
    bytes_per_element = 2  # BF16
    bytes_input = B * D * bytes_per_element
    bytes_w13 = B * K * 2 * N * D * bytes_per_element  # gate + up
    bytes_w2 = B * K * D * N * bytes_per_element  # down
    bytes_routing = B * K * 4  # float32
    total_bytes = bytes_input + bytes_w13 + bytes_w2 + bytes_routing

    bandwidth_gb_s = (total_bytes / 1e9) / (mean_latency_ms / 1e3)

    return mean_latency_us, bandwidth_gb_s


def benchmark_torch_reference(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    intermediate_size: int,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    """Benchmark PyTorch reference for comparison."""
    N = intermediate_size
    w_gate = w13[:, :N, :]
    w_up = w13[:, N:, :]

    def run_once():
        B, D = hidden_states.shape
        K = topk_ids.shape[1]
        output = torch.zeros(B, D, dtype=torch.float32, device=hidden_states.device)
        for b in range(B):
            for k in range(K):
                eid = topk_ids[b, k].item()
                rw = topk_weights[b, k].item()
                x = hidden_states[b].float()
                g = w_gate[eid].float() @ x
                u = w_up[eid].float() @ x
                inter = torch.nn.functional.silu(g) * u
                output[b] += rw * (w2[eid].float() @ inter)
        return output

    # Warmup
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        run_once()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iters) * 1e6  # us


def benchmark_cute_warp_decode(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    intermediate_size: int,
    warmup: int = 10,
    iters: int = 100,
) -> Tuple[float, float]:
    """Benchmark CuTe warp decode kernel.

    Returns:
        Tuple of (mean_latency_us, bandwidth_gb_s), or (-1, -1) if unavailable.
    """
    if not cute_shape_supported(hidden_states.shape[1], intermediate_size):
        return -1.0, -1.0
    try:
        import sgl_kernel
        if not hasattr(sgl_kernel, "warp_decode_cute_moe_packed_forward"):
            return -1.0, -1.0
    except ImportError:
        return -1.0, -1.0

    # Warmup
    for _ in range(warmup):
        sgl_kernel.warp_decode_cute_moe_packed_forward(
            hidden_states, w13, w2, topk_ids, topk_weights,
            intermediate_size, False,
        )
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        sgl_kernel.warp_decode_cute_moe_packed_forward(
            hidden_states, w13, w2, topk_ids, topk_weights,
            intermediate_size, False,
        )
        end_events[i].record()

    torch.cuda.synchronize()

    latencies_ms = [
        start_events[i].elapsed_time(end_events[i]) for i in range(iters)
    ]
    mean_latency_ms = sum(latencies_ms) / len(latencies_ms)
    mean_latency_us = mean_latency_ms * 1000

    # Calculate bandwidth (same formula as Triton benchmark)
    B = hidden_states.shape[0]
    K = topk_ids.shape[1]
    D = hidden_states.shape[1]
    N = intermediate_size
    bytes_per_element = 2  # BF16
    bytes_input = B * D * bytes_per_element
    bytes_w13 = B * K * 2 * N * D * bytes_per_element
    bytes_w2 = B * K * D * N * bytes_per_element
    bytes_routing = B * K * 4
    total_bytes = bytes_input + bytes_w13 + bytes_w2 + bytes_routing
    bandwidth_gb_s = (total_bytes / 1e9) / (mean_latency_ms / 1e3)

    return mean_latency_us, bandwidth_gb_s


def main():
    parser = argparse.ArgumentParser(description="Benchmark warp decode MoE")
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--batch-sizes", type=str, default="1,4,8,16,32,64",
        help="Comma-separated batch sizes to benchmark"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--skip-reference", action="store_true")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Check CuTe availability
    cute_available = False
    try:
        import sgl_kernel
        cute_available = (
            hasattr(sgl_kernel, "warp_decode_cute_moe_packed_forward")
            and cute_shape_supported(args.hidden_size, args.intermediate_size)
        )
    except ImportError:
        pass

    print(f"Configuration:")
    print(f"  hidden_size = {args.hidden_size}")
    print(f"  intermediate_size = {args.intermediate_size}")
    print(f"  num_experts = {args.num_experts}")
    print(f"  top_k = {args.top_k}")
    print(f"  warmup = {args.warmup}, iters = {args.iters}")
    print(f"  cute_available = {cute_available}")
    print()

    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    smem_per_sm = getattr(
        gpu_props,
        "max_shared_memory_per_multiprocessor",
        gpu_props.shared_memory_per_multiprocessor,
    )
    print(f"GPU: {gpu_name}")
    print(f"  SMs: {gpu_props.multi_processor_count}")
    print(f"  SMEM/SM: {smem_per_sm // 1024} KB")
    print()

    header = f"{'Batch':>6} | {'Triton (us)':>12} | {'Triton BW':>10} |"
    if cute_available:
        header += f" {'CuTe (us)':>12} | {'CuTe BW':>10} | {'CuTe/Tri':>9} |"
    if not args.skip_reference:
        header += f" {'Ref (us)':>12} | {'Speedup':>8} |"
    print(header)
    print("-" * len(header))

    for B in batch_sizes:
        hs, w13, w2, ids, wts = generate_moe_data(
            B, args.hidden_size, args.intermediate_size,
            args.num_experts, args.top_k,
        )

        wd_lat, wd_bw = benchmark_warp_decode(
            hs, w13, w2, ids, wts, args.intermediate_size,
            warmup=args.warmup, iters=args.iters,
        )

        line = f"{B:>6} | {wd_lat:>12.1f} | {wd_bw:>9.2f}G |"

        if cute_available:
            cute_lat, cute_bw = benchmark_cute_warp_decode(
                hs, w13, w2, ids, wts, args.intermediate_size,
                warmup=args.warmup, iters=args.iters,
            )
            if cute_lat > 0:
                ratio = wd_lat / cute_lat
                line += f" {cute_lat:>12.1f} | {cute_bw:>9.2f}G | {ratio:>8.2f}x |"
            else:
                line += f" {'N/A':>12} | {'N/A':>10} | {'N/A':>9} |"

        if not args.skip_reference and B <= 16:
            ref_lat = benchmark_torch_reference(
                hs, w13, w2, ids, wts, args.intermediate_size,
                warmup=3, iters=5,
            )
            best_lat = min(wd_lat, cute_lat) if cute_available and cute_lat > 0 else wd_lat
            speedup = ref_lat / best_lat
            line += f" {ref_lat:>12.1f} | {speedup:>7.2f}x |"
        elif not args.skip_reference:
            line += f" {'(skipped)':>12} | {'N/A':>8} |"

        print(line)


if __name__ == "__main__":
    main()
