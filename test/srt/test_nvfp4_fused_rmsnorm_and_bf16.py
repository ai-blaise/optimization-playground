"""Correctness + microbench for fused (residual-add + RMSNorm + FP4 +
BF16 hidden_states) — iter4 SECONDARY for #15 NVFP4 MoE deploy.

Tests `sglang.jit_kernel.nvfp4.fused_rmsnorm_to_fp4_and_bf16_linear`
against the unfused pair (jit fused_add_rmsnorm + sgl
scaled_fp4_quant_linear) at production decode shapes.

Target call site:
    python/sglang/srt/layers/communicator.py
    CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
    `use_layer_norm_before_gather=True` branch at L1024 (the DSv3.2-REAP
    DP=TP=8 + attn_tp_size=1 deploy path).

Usage:

    python -m pytest test/srt/test_nvfp4_fused_rmsnorm_and_bf16.py -v -s
"""
from __future__ import annotations

from typing import Tuple

import pytest
import torch


def _has_b200() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


pytestmark = pytest.mark.skipif(
    not _has_b200(),
    reason="Fused RMSNorm + FP4 + BF16 requires Blackwell (sm100+).",
)

PROD_HIDDEN = 7168
DECODE_BATCHES = [1, 8, 16, 32, 64, 128, 256, 512]


def _suggest_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().max().to(torch.float32).clamp_min_(1e-6)
    return (448.0 * 6.0 / amax).reshape(1)


def _unfused_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    gs: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: fused_add_rmsnorm + scaled_fp4_quant_linear pair."""
    from sglang.jit_kernel.norm import fused_add_rmsnorm
    from sglang.jit_kernel.nvfp4 import scaled_fp4_quant_linear

    x_c = x.clone()
    r_c = residual.clone()
    fused_add_rmsnorm(x_c, r_c, weight, eps)
    # x_c is now the BF16 post-norm hidden_states
    fp4, sf = scaled_fp4_quant_linear(x_c, gs)
    return x_c, r_c, fp4, sf


@pytest.mark.parametrize("cast_x_before_out_mul", [False, True])
@pytest.mark.parametrize("m", DECODE_BATCHES)
def test_fused_fp4_and_bf16_matches_unfused(
    m: int, cast_x_before_out_mul: bool
) -> None:
    from sglang.jit_kernel.nvfp4 import fused_rmsnorm_to_fp4_and_bf16_linear

    torch.manual_seed(0xCAFEBEEF + m + (1 if cast_x_before_out_mul else 0))
    device = torch.device("cuda")
    x = torch.randn(m, PROD_HIDDEN, dtype=torch.bfloat16, device=device) * 0.5
    residual = (
        torch.randn(m, PROD_HIDDEN, dtype=torch.bfloat16, device=device) * 0.2
    )
    weight = torch.ones(PROD_HIDDEN, dtype=torch.bfloat16, device=device)
    weight += 0.1 * torch.randn_like(weight)
    gs = _suggest_global_scale(x + residual)
    eps = 1e-6

    if cast_x_before_out_mul:
        # Reference for cast_x=True: compute in plain torch since
        # fused_add_rmsnorm only supports Llama-style on the cuda path.
        x_f = (x + residual).float()
        rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
        normed = (x_f * rms).to(x.dtype)
        ref_y = (normed.float() * weight.float()).to(x.dtype)
        ref_residual = (x + residual).to(x.dtype)
        from sglang.jit_kernel.nvfp4 import scaled_fp4_quant_linear
        ref_fp4, ref_sf = scaled_fp4_quant_linear(ref_y, gs)
    else:
        ref_y, ref_residual, ref_fp4, ref_sf = _unfused_reference(
            x, residual, weight, gs, eps
        )

    x_test = x.clone()
    r_test = residual.clone()
    fused_y, fused_fp4, fused_sf = fused_rmsnorm_to_fp4_and_bf16_linear(
        x_test, r_test, weight, gs, eps,
        cast_x_before_out_mul=cast_x_before_out_mul,
    )

    # y bit-exact (up to bf16 ULP)
    y_diff = (ref_y.float() - fused_y.float()).abs()
    y_max = y_diff.max().item()
    y_mean = y_diff.mean().item()
    print(
        f"\n[m={m} cast_x={cast_x_before_out_mul}] "
        f"y max-abs-err={y_max:.6f} mean={y_mean:.6f}"
    )
    assert y_max < 0.1, f"y diverges at m={m}: {y_max}"

    # Residual is x + residual_old, also bit-exact in BF16
    res_diff = (ref_residual.float() - r_test.float()).abs()
    res_max = res_diff.max().item()
    print(
        f"[m={m} cast_x={cast_x_before_out_mul}] "
        f"residual max-abs-err={res_max:.6f}"
    )
    assert res_max < 0.05, f"residual diverges at m={m}"

    fp4_match = (fused_fp4 == ref_fp4).float().mean().item()
    sf_match = (
        fused_sf.view(torch.uint8) == ref_sf.view(torch.uint8)
    ).float().mean().item()
    print(
        f"[m={m} cast_x={cast_x_before_out_mul}] "
        f"FP4 match={fp4_match:.4f}  SF match={sf_match:.4f}"
    )
    assert fp4_match > 0.95, f"FP4 diverges at m={m}: {fp4_match:.4f}"
    assert sf_match > 0.95, f"SF diverges at m={m}: {sf_match:.4f}"


def _time_call(fn, *args, warmup=50, iters=500, use_cuda_graph=True) -> float:
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
def test_perf_vs_unfused_fp4_and_bf16(m: int) -> None:
    from sglang.jit_kernel.norm import fused_add_rmsnorm
    from sglang.jit_kernel.nvfp4 import (
        fused_rmsnorm_to_fp4_and_bf16_linear,
        scaled_fp4_quant_linear,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(m, PROD_HIDDEN, dtype=torch.bfloat16, device=device)
    residual = torch.randn(m, PROD_HIDDEN, dtype=torch.bfloat16, device=device)
    weight = torch.ones(PROD_HIDDEN, dtype=torch.bfloat16, device=device)
    gs = _suggest_global_scale(x)
    eps = 1e-6

    x_u = x.clone()
    r_u = residual.clone()
    x_f = x.clone()
    r_f = residual.clone()

    def _unfused():
        fused_add_rmsnorm(x_u, r_u, weight, eps)
        scaled_fp4_quant_linear(x_u, gs)

    def _fused(enable_pdl=False):
        fused_rmsnorm_to_fp4_and_bf16_linear(
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

    step_save_g = (u_us_g - f_us_g) * 58
    step_save_pdl_g = (u_us_g - f_pdl_g) * 58

    print(
        f"\n[m={m}] eager: unfused={u_us:.2f}us  fused={f_us:.2f}us "
        f"fused+pdl={f_pdl:.2f}us  speedup={u_us / f_us:.2f}x"
    )
    print(
        f"[m={m}] graph: unfused={u_us_g:.2f}us  fused={f_us_g:.2f}us "
        f"fused+pdl={f_pdl_g:.2f}us  speedup={u_us_g / f_us_g:.2f}x  "
        f"step_save_g={step_save_g:.1f}us  step_save_pdl_g={step_save_pdl_g:.1f}us"
    )
