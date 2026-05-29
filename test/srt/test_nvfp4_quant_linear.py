"""Microbench + correctness for the sgl-native non-swizzled NVFP4 quantize.

Compares sglang.jit_kernel.nvfp4.scaled_fp4_quant_linear against
flashinfer.fp4_quantize(is_sf_swizzled_layout=False) at production decode
shapes for DeepSeek-V3.2-REAP NVFP4 MoE.

Usage (on a node with a free B200 GPU; will not run while production
serving pod owns all GPUs):

    python -m pytest test/srt/test_nvfp4_quant_linear.py -v -s
"""
from __future__ import annotations

import time
from typing import Optional

import pytest
import torch


def _has_flashinfer() -> bool:
    try:
        import flashinfer  # noqa: F401

        return True
    except ImportError:
        return False


def _has_b200() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


pytestmark = pytest.mark.skipif(
    not _has_b200(),
    reason="Linear NVFP4 quantize requires Blackwell (sm100+).",
)


# Production shapes for DeepSeek-V3.2-REAP NVFP4 MoE
# hidden_size = 7168, decode batches typically 1..128 per DP rank
PROD_HIDDEN = 7168
DECODE_BATCHES = [1, 8, 16, 32, 64, 128, 256]


def _suggest_global_scale(x: torch.Tensor) -> torch.Tensor:
    """Reasonable input global scale: 448*6 / amax(x)."""
    amax = x.abs().max().to(torch.float32).clamp_min_(1e-6)
    return (448.0 * 6.0 / amax).reshape(1)


@pytest.mark.parametrize("m", DECODE_BATCHES)
def test_linear_quant_matches_flashinfer(m: int) -> None:
    if not _has_flashinfer():
        pytest.skip("flashinfer not installed")

    from sglang.jit_kernel.nvfp4 import scaled_fp4_quant_linear

    torch.manual_seed(0xDEADBEEF)
    device = torch.device("cuda")
    x = torch.randn(m, PROD_HIDDEN, dtype=torch.bfloat16, device=device)
    gs = _suggest_global_scale(x)

    # Reference: flashinfer non-swizzled
    from flashinfer import fp4_quantize as fi_fp4_quantize

    ref_out, ref_sf = fi_fp4_quantize(
        x,
        gs,
        16,  # sf_vec_size
        False,  # use_ue8m0
        False,  # is_sf_swizzled_layout
    )
    ref_sf = ref_sf.view(torch.float8_e4m3fn).reshape(m, PROD_HIDDEN // 16)

    # Test: sgl-native linear
    sgl_out, sgl_sf = scaled_fp4_quant_linear(x, gs)
    sgl_out_2d = sgl_out.reshape(m, PROD_HIDDEN // 2)
    sgl_sf_2d = sgl_sf.reshape(m, PROD_HIDDEN // 16)

    # Bit-exact comparison: both kernels do warp-level amax + same scaling,
    # but may disagree on FP8 rounding mode. We compare the FP4 vals first
    # then the scales separately with a small tolerance.
    matches = (sgl_out_2d == ref_out).float().mean().item()
    sf_matches = (sgl_sf_2d.view(torch.uint8) == ref_sf.view(torch.uint8)).float().mean().item()
    print(
        f"\n[m={m}] FP4 vals match rate: {matches:.4f}, SF byte match rate: {sf_matches:.4f}"
    )
    # Allow up to 2% mismatch from rounding-mode differences at scale boundaries
    assert matches > 0.98, f"FP4 vals diverge at m={m}: match rate {matches:.4f}"
    assert sf_matches > 0.98, f"SF bytes diverge at m={m}: match rate {sf_matches:.4f}"


def _time_kernel(
    fn,
    *args,
    warmup: int = 50,
    iters: int = 500,
    use_cuda_graph: bool = True,
) -> float:
    """Returns mean kernel time in microseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    if use_cuda_graph:
        # Capture & replay to mimic production CUDA graph deploy
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
def test_linear_quant_perf_vs_flashinfer(m: int) -> None:
    if not _has_flashinfer():
        pytest.skip("flashinfer not installed")

    from sglang.jit_kernel.nvfp4 import scaled_fp4_quant_linear

    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(m, PROD_HIDDEN, dtype=torch.bfloat16, device=device)
    gs = _suggest_global_scale(x)

    from flashinfer import fp4_quantize as fi_fp4_quantize

    def _flashinfer_call():
        fi_fp4_quantize(x, gs, 16, False, False)

    def _sgl_call():
        scaled_fp4_quant_linear(x, gs)

    fi_us = _time_kernel(_flashinfer_call, use_cuda_graph=False)
    sgl_us = _time_kernel(_sgl_call, use_cuda_graph=False)
    fi_g_us = _time_kernel(_flashinfer_call, use_cuda_graph=True)
    sgl_g_us = _time_kernel(_sgl_call, use_cuda_graph=True)

    print(
        f"\n[m={m}] eager: flashinfer={fi_us:.2f}us  sgl-linear={sgl_us:.2f}us  "
        f"ratio={fi_us / sgl_us:.2f}x"
    )
    print(
        f"[m={m}] graph: flashinfer={fi_g_us:.2f}us  sgl-linear={sgl_g_us:.2f}us  "
        f"ratio={fi_g_us / sgl_g_us:.2f}x"
    )


if __name__ == "__main__":
    # Quick smoke run without pytest
    for m in DECODE_BATCHES:
        test_linear_quant_matches_flashinfer(m)
    for m in DECODE_BATCHES:
        test_linear_quant_perf_vs_flashinfer(m)
