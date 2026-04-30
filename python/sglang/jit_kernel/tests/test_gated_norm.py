import itertools
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.gated_norm import gated_norm_forward
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)


SHAPES = [(1, 128), (7, 512), (2, 3, 1024), (4, 7168)]
RANKS = [1, 8, 32, 64]


def _reference(normed: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    z = normed.float() @ w_down.float().t()
    gate = torch.sigmoid(F.silu(z) @ w_up.float().t())
    return (normed.float() * gate).to(normed.dtype)


@pytest.mark.parametrize("shape,rank", list(itertools.product(SHAPES, RANKS)))
def test_gated_norm_forward(shape: tuple[int, ...], rank: int) -> None:
    torch.manual_seed(2026)
    normed = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    hidden_size = shape[-1]
    w_down = (
        torch.randn(rank, hidden_size, device="cuda", dtype=torch.bfloat16)
        / hidden_size**0.5
    )
    w_up = (
        torch.randn(hidden_size, rank, device="cuda", dtype=torch.bfloat16)
        / rank**0.5
    )

    actual = gated_norm_forward(normed, w_down, w_up)
    expected = _reference(normed, w_down, w_up)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_gated_norm_forward_out() -> None:
    torch.manual_seed(2027)
    normed = torch.randn(9, 256, device="cuda", dtype=torch.bfloat16)
    w_down = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16) / 16
    w_up = torch.randn(256, 16, device="cuda", dtype=torch.bfloat16) / 4
    out = torch.empty_like(normed)

    result = gated_norm_forward(normed, w_down, w_up, out=out)
    assert result is out
    expected = _reference(normed, w_down, w_up)
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


def test_gated_norm_torch_mm_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SGLANG_GATED_NORM_TORCH_MM_MIN_TOKENS", "1")
    torch.manual_seed(2028)
    normed = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    w_down = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16) / 8
    w_up = torch.randn(128, 8, device="cuda", dtype=torch.bfloat16) / 8

    actual = gated_norm_forward(normed, w_down, w_up)
    expected = _reference(normed, w_down, w_up)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_gated_norm_rejects_non_bf16() -> None:
    normed = torch.empty(1, 128, device="cuda", dtype=torch.float16)
    w_down = torch.empty(8, 128, device="cuda", dtype=torch.float16)
    w_up = torch.empty(128, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(TypeError, match="bf16"):
        gated_norm_forward(normed, w_down, w_up)


def test_gated_norm_rank_guard() -> None:
    normed = torch.empty(1, 128, device="cuda", dtype=torch.bfloat16)
    w_down = torch.empty(65, 128, device="cuda", dtype=torch.bfloat16)
    w_up = torch.empty(128, 65, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="rank"):
        gated_norm_forward(normed, w_down, w_up)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
