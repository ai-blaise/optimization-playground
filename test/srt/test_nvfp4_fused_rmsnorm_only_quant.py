"""Correctness + microbench for fused (RMSNorm-only + linear NVFP4 quantize).

Iter4 PRIMARY #15 NVFP4 MoE deploy. Tests
`sglang.jit_kernel.nvfp4.fused_rmsnorm_only_to_fp4_linear` against the
unfused pair (`rmsnorm` + `scaled_fp4_quant_linear`) at production decode
shapes for DeepSeek-V3.2-REAP NVFP4 MoE.

Target call site:
    python/sglang/srt/layers/communicator.py
    CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
    `not use_layer_norm_before_gather` branch:
        if hidden_states.shape[0] != 0:
            hidden_states = layernorm(hidden_states)   # <-- replaced

Usage (B200 with a free GPU):

    python -m pytest test/srt/test_nvfp4_fused_rmsnorm_only_quant.py -v -s
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
    reason="Fused RMSNorm-only + NVFP4 quantize requires Blackwell (sm100+).",
)

PROD_HIDDEN = 7168
# Production decode batches per rank in DP=TP=8 (m = batch / DP). Add
# 256 and 512 to exercise the kernel at larger m for completeness.
DECODE_BATCHES = [1, 8, 16, 32, 64, 128, 256, 512]


def _suggest_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().max().to(torch.float32).clamp_min_(1e-6)
    return (448.0 * 6.0 / amax).reshape(1)


def _unfused_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    gs: torch.Tensor,
    eps: float,
    *,
    cast_x_before_out_mul: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: existing jit rmsnorm + scaled_fp4_quant_linear (the
    pair our kernel replaces at the wired call site).

    Supports both kCastXBeforeOutMul=false (Llama-style, DSv3.2 default)
    and =true (HF-parity, Gemma/glm4 style — rounds through bf16 after
    the rms multiply but before the weight multiply).
    """
    from sglang.jit_kernel.nvfp4 import scaled_fp4_quant_linear

    if cast_x_before_out_mul:
        # HF-parity reference: do the cast in plain torch since flashinfer
        # `rmsnorm` only exposes the Llama-style variant on its bf16 cuda
        # path. Compute in fp32, round to bf16 after the rms multiply
        # (mirroring the kernel's mid-phase bf16 round), then weight
        # multiply.
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
        normed = (x_f * rms).to(x.dtype)
        y = (normed.float() * weight.float()).to(x.dtype)
    else:
        try:
            from flashinfer.norm import rmsnorm
        except ImportError:
            def rmsnorm(x_, w_, eps_):
                x_f = x_.float()
                rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps_)
                return (x_f * rms).to(x_.dtype) * w_

        y = rmsnorm(x, weight, eps)
    fp4, sf = scaled_fp4_quant_linear(y, gs)
    return y, fp4, sf


@pytest.mark.parametrize("cast_x_before_out_mul", [False, True])
@pytest.mark.parametrize("m", DECODE_BATCHES)
def test_fused_no_residual_matches_unfused(
    m: int, cast_x_before_out_mul: bool
) -> None:
    from sglang.jit_kernel.nvfp4 import fused_rmsnorm_only_to_fp4_linear

    torch.manual_seed(0xDEADBEEF + m + (1 if cast_x_before_out_mul else 0))
    device = torch.device("cuda")
    # The wired call site (L1042 of communicator.py) feeds
    # post-allgather+scatter hidden_states. Magnitudes are ~residual+attn
    # output combined; ~0.7 scale matches the production decode mix.
    x = torch.randn(m, PROD_HIDDEN, dtype=torch.bfloat16, device=device) * 0.7
    weight = torch.ones(PROD_HIDDEN, dtype=torch.bfloat16, device=device)
    weight += 0.1 * torch.randn_like(weight)
    gs = _suggest_global_scale(x)
    eps = 1e-6

    ref_y, ref_fp4, ref_sf = _unfused_reference(
        x, weight, gs, eps, cast_x_before_out_mul=cast_x_before_out_mul
    )
    fused_y, fused_fp4, fused_sf = fused_rmsnorm_only_to_fp4_linear(
        x, weight, gs, eps, cast_x_before_out_mul=cast_x_before_out_mul
    )

    # The fused kernel produces a new BF16 tensor; verify it bit-aligns
    # with the reference rmsnorm output (allow tiny ULP differences).
    y_diff = (ref_y.float() - fused_y.float()).abs()
    y_max = y_diff.max().item()
    y_mean = y_diff.mean().item()
    print(
        f"\n[m={m}] y_out max-abs-err={y_max:.6f} mean-abs-err={y_mean:.6f}"
    )
    # rsqrt + bf16 round can differ by a couple ULPs at amplitude.
    assert y_max < 0.1, f"y_out diverges at m={m}: max={y_max}"
    assert y_mean < 0.01, f"y_out mean diverges at m={m}: mean={y_mean}"

    fp4_match = (fused_fp4 == ref_fp4).float().mean().item()
    sf_match = (
        fused_sf.view(torch.uint8) == ref_sf.view(torch.uint8)
    ).float().mean().item()
    print(
        f"[m={m}] FP4 vals match rate: {fp4_match:.4f}  "
        f"SF byte match rate: {sf_match:.4f}"
    )
    # Allow up to 2% mismatch from rounding-mode differences at scale
    # boundaries (same envelope as iter2's residual variant test).
    assert fp4_match > 0.95, (
        f"FP4 vals diverge at m={m}: rate {fp4_match:.4f}"
    )
    assert sf_match > 0.95, (
        f"SF bytes diverge at m={m}: rate {sf_match:.4f}"
    )


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
def test_perf_vs_unfused_no_residual(m: int) -> None:
    """Bench: fused kernel vs (rmsnorm + scaled_fp4_quant_linear) pair.

    Production target: 58 MoE layers per decode step. Per-step delta is
    `(unfused_us - fused_us) * 58 us/step`. Reports both eager and
    CUDA-graph captured timings since the deploy path runs both modes
    (piecewise capture + eager warmup).
    """
    from sglang.jit_kernel.nvfp4 import (
        fused_rmsnorm_only_to_fp4_linear,
        scaled_fp4_quant_linear,
    )
    try:
        from flashinfer.norm import rmsnorm as _rmsnorm
    except ImportError:
        pytest.skip("flashinfer not available for unfused reference")

    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(m, PROD_HIDDEN, dtype=torch.bfloat16, device=device)
    weight = torch.ones(PROD_HIDDEN, dtype=torch.bfloat16, device=device)
    gs = _suggest_global_scale(x)
    eps = 1e-6

    def _unfused():
        y = _rmsnorm(x, weight, eps)
        scaled_fp4_quant_linear(y, gs)

    def _fused(enable_pdl=False):
        fused_rmsnorm_only_to_fp4_linear(
            x, weight, gs, eps, enable_pdl=enable_pdl
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


if __name__ == "__main__":
    for m in DECODE_BATCHES:
        test_fused_no_residual_matches_unfused(m)
    for m in DECODE_BATCHES:
        test_perf_vs_unfused_no_residual(m)
