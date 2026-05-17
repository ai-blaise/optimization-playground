import sys

import pytest
import torch

from sgl_kernel import g1_gate_forward, g1_gate_forward_fused


@pytest.mark.parametrize("shape", [(1,), (17,), (4, 257), (2, 3, 7168)])
def test_g1_gate_forward(shape: tuple[int, ...]) -> None:
    torch.manual_seed(2026)
    linear_out = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    attn_out = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    output, gate = g1_gate_forward(linear_out, attn_out)
    expected_gate = torch.sigmoid(linear_out.float()).to(torch.bfloat16)
    expected_output = (attn_out.float() * torch.sigmoid(linear_out.float())).to(torch.bfloat16)

    torch.testing.assert_close(gate, expected_gate, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(output, expected_output, atol=2e-2, rtol=2e-2)


def test_g1_gate_forward_out_params() -> None:
    linear_out = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16)
    attn_out = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(attn_out)
    gate = torch.empty_like(attn_out)

    result_output, result_gate = g1_gate_forward(linear_out, attn_out, output=output, gate=gate)
    assert result_output is output
    assert result_gate is gate
    expected_gate = torch.sigmoid(linear_out.float()).to(torch.bfloat16)
    expected_output = (attn_out.float() * torch.sigmoid(linear_out.float())).to(torch.bfloat16)

    torch.testing.assert_close(gate, expected_gate, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(output, expected_output, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("shape", [(1,), (17,), (4, 257), (2, 3, 7168)])
def test_g1_gate_forward_fused_matches_full_kernel(shape: tuple[int, ...]) -> None:
    torch.manual_seed(2027)
    linear_out = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    attn_out = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    expected, _ = g1_gate_forward(linear_out, attn_out)
    actual = g1_gate_forward_fused(linear_out, attn_out)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_g1_gate_forward_fused_out_param() -> None:
    linear_out = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16)
    attn_out = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(attn_out)

    result = g1_gate_forward_fused(linear_out, attn_out, output=output)
    assert result is output
    expected, _ = g1_gate_forward(linear_out, attn_out)

    torch.testing.assert_close(output, expected, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
