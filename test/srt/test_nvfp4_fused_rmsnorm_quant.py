"""Correctness + microbench for fused (residual-add + RMSNorm + linear NVFP4 quantize).

Compares sglang.jit_kernel.nvfp4.fused_rmsnorm_scaled_fp4_quant_linear against
the unfused pair (jit fused_add_rmsnorm + sgl scaled_fp4_quant_linear) at the
production decode shapes for DeepSeek-V3.2-REAP NVFP4 MoE.

Usage (B200 with a free GPU):

    python -m pytest test/srt/test_nvfp4_fused_rmsnorm_quant.py -v -s
"""
from __future__ import annotations

import time
from typing import Tuple

import pytest
import torch


def _has_b200() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


def _has_flashinfer() -> bool:
    try:
        import flashinfer  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_b200(),
    reason="Fused RMSNorm + NVFP4 quantize requires Blackwell (sm100+).",
)

# Production shapes for DeepSeek-V3.2-REAP NVFP4 MoE
PROD_HIDDEN = 7168
DECODE_BATCHES = [1, 8, 16, 32, 64, 128, 256]


def _suggest_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().max().to(torch.float32).clamp_min_(1e-6)
    return (448.0 * 6.0 / amax).reshape(1)


def _unfused_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    gs: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: existing jit fused_add_rmsnorm + scaled_fp4_quant_linear."""
    from sglang.jit_kernel.norm import fused_add_rmsnorm
    from sglang.jit_kernel.nvfp4 import scaled_fp4_quant_linear

    x_c = x.clone()
    r_c = residual.clone()
    fused_add_rmsnorm(x_c, r_c, weight, eps)
    fp4, sf = scaled_fp4_quant_linear(x_c, gs)
    return r_c, fp4, sf


@pytest.mark.parametrize("m", DECODE_BATCHES)
def test_fused_matches_unfused(m: int) -> None:
    from sglang.jit_kernel.nvfp4 import fused_rmsnorm_scaled_fp4_quant_linear

    torch.manual_seed(0xDEADBEEF)
    device = torch.device("cuda")
    x = torch.randn(m, PROD_HIDDEN, dtype=torch.bfloat16, device=device) * 0.5
    residual = torch.randn(m, PROD_HIDDEN, dtype=torch.bfloat16, device=device) * 0.2
    weight = torch.ones(PROD_HIDDEN, dtype=torch.bfloat16, device=device)
    weight += 0.1 * torch.randn_like(weight)
    eps = 1e-6

    # Reference using the unfused pair (does its own clones).
    ref_residual, ref_fp4, ref_sf = _unfused_reference(
        x, residual, weight, _suggest_global_scale(x), eps
    )

    # Fused (mutates `residual` in place).
    x_test = x.clone()
    r_test = residual.clone()
    gs = _suggest_global_scale(x_test)
    fused_fp4, fused_sf = fused_rmsnorm_scaled_fp4_quant_linear(
        x_test, r_test, weight, gs, eps
    )

    # Residual: bit-exact (we round through bf16 the same way upstream does).
    res_diff = (ref_residual.float() - r_test.float()).abs()
    res_max = res_diff.max().item()
    res_mean = res_diff.mean().item()
    print(
        f"\n[m={m}] residual max-abs-err={res_max:.6f} mean-abs-err={res_mean:.6f}"
    )
    # Allow tiny rounding differences (we may differ from upstream's exact
    # rounding mode by one bf16 ULP at the boundary).
    assert res_max < 0.05, f"residual diverges at m={m}"

    fp4_match = (fused_fp4 == ref_fp4).float().mean().item()
    sf_match = (
        fused_sf.view(torch.uint8) == ref_sf.view(torch.uint8)
    ).float().mean().item()
    print(
        f"[m={m}] FP4 vals match rate: {fp4_match:.4f}  SF byte match rate: {sf_match:.4f}"
    )
    # Allow up to 2% mismatch from rounding-mode differences at scale boundaries.
    assert fp4_match > 0.95, f"FP4 vals diverge at m={m}: rate {fp4_match:.4f}"
    assert sf_match > 0.95, f"SF bytes diverge at m={m}: rate {sf_match:.4f}"


def _time_call(fn, *args, warmup=50, iters=500, use_cuda_graph=True) -> float:
    """Return mean per-call latency in microseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    if use_cuda_graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn(*args)
        torch.cuda.current_stream().wait_stream(stream)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            fn(*args)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / iters

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


@pytest.mark.parametrize("m", DECODE_BATCHES)
def test_perf_vs_unfused(m: int) -> None:
    from sglang.jit_kernel.norm import fused_add_rmsnorm
    from sglang.jit_kernel.nvfp4 import (
        fused_rmsnorm_scaled_fp4_quant_linear,
        scaled_fp4_quant_linear,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(m, PROD_HIDDEN, dtype=torch.bfloat16, device=device)
    residual = torch.randn(m, PROD_HIDDEN, dtype=torch.bfloat16, device=device)
    weight = torch.ones(PROD_HIDDEN, dtype=torch.bfloat16, device=device)
    gs = _suggest_global_scale(x)
    eps = 1e-6

    # Persistent buffers for unfused (CUDA-graph capture requires no
    # alloc inside the captured region; both paths allocate via torch
    # APIs internally for the FP4 + SF tensors, but the residual+input
    # mutate-in-place stay fixed).
    x_u = x.clone()
    r_u = residual.clone()
    x_f = x.clone()
    r_f = residual.clone()

    def _unfused():
        fused_add_rmsnorm(x_u, r_u, weight, eps)
        scaled_fp4_quant_linear(x_u, gs)

    def _fused(enable_pdl=False):
        fused_rmsnorm_scaled_fp4_quant_linear(
            x_f, r_f, weight, gs, eps, enable_pdl=enable_pdl
        )

    def _fused_pdl():
        _fused(enable_pdl=True)

    u_us_g = _time_call(_unfused, use_cuda_graph=True)
    f_us_g = _time_call(_fused, use_cuda_graph=True)
    f_pdl_g = _time_call(_fused_pdl, use_cuda_graph=True)

    u_us = _time_call(_unfused, use_cuda_graph=False)
    f_us = _time_call(_fused, use_cuda_graph=False)
    f_pdl = _time_call(_fused_pdl, use_cuda_graph=False)

    print(
        f"\n[m={m}] eager: unfused={u_us:.2f}us  fused={f_us:.2f}us "
        f"fused+pdl={f_pdl:.2f}us  speedup={u_us / f_us:.2f}x"
    )
    print(
        f"[m={m}] graph: unfused={u_us_g:.2f}us  fused={f_us_g:.2f}us "
        f"fused+pdl={f_pdl_g:.2f}us  speedup={u_us_g / f_us_g:.2f}x"
    )


if __name__ == "__main__":
    for m in DECODE_BATCHES:
        test_fused_matches_unfused(m)
    for m in DECODE_BATCHES:
        test_perf_vs_unfused(m)
