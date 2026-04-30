import itertools
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.gated_norm import gated_norm_forward
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)


DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = [(1, 128), (7, 512), (2, 3, 1024), (4, 7168)]
RANKS = [1, 8, 32, 64]


def _reference(normed: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    z = normed.float() @ w_down.float().t()
    gate = torch.sigmoid(F.silu(z) @ w_up.float().t())
    return (normed.float() * gate).to(normed.dtype)


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 2e-4, 2e-4
    return 2e-2, 2e-2


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,rank", list(itertools.product(SHAPES, RANKS)))
def test_gated_norm_forward(dtype: torch.dtype, shape: tuple[int, ...], rank: int) -> None:
    torch.manual_seed(2026)
    normed = torch.randn(shape, device="cuda", dtype=dtype)
    hidden_size = shape[-1]
    w_down = torch.randn(rank, hidden_size, device="cuda", dtype=dtype) / hidden_size**0.5
    w_up = torch.randn(hidden_size, rank, device="cuda", dtype=dtype) / rank**0.5

    actual = gated_norm_forward(normed, w_down, w_up)
    expected = _reference(normed, w_down, w_up)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gated_norm_forward_out(dtype: torch.dtype) -> None:
    torch.manual_seed(2027)
    normed = torch.randn(9, 256, device="cuda", dtype=dtype)
    w_down = torch.randn(16, 256, device="cuda", dtype=dtype) / 16
    w_up = torch.randn(256, 16, device="cuda", dtype=dtype) / 4
    out = torch.empty_like(normed)

    result = gated_norm_forward(normed, w_down, w_up, out=out)
    assert result is out
    expected = _reference(normed, w_down, w_up)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)


def test_gated_norm_rank_guard() -> None:
    normed = torch.empty(1, 128, device="cuda", dtype=torch.bfloat16)
    w_down = torch.empty(65, 128, device="cuda", dtype=torch.bfloat16)
    w_up = torch.empty(128, 65, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="rank"):
        gated_norm_forward(normed, w_down, w_up)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
