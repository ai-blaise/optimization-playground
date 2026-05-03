"""End-to-end wiring tests for the G1 attention-output gate and GatedNorm
glue inside `sglang.srt.models.deepseek_v2`.

These are pure-Python tests that exercise the model-side helpers
(`_apply_g1_gate`, `_g1_gate_pre_hook`, `_maybe_apply_gated_norm`) on
synthetic tensors. They confirm the wiring math without requiring the
full DeepseekV2 model or a quantized checkpoint.
"""

from __future__ import annotations

import sys
import unittest
import weakref

import pytest
import torch
import torch.nn as nn


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def _ref_g1(attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (attn_out.float() * torch.sigmoid(gate.float())).to(attn_out.dtype)


def _ref_gated_norm(
    x: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    z = torch.matmul(flat.float(), w_down.float().t())
    activation = nn.functional.silu(z).to(w_up.dtype)
    logits = torch.matmul(activation, w_up.t())
    torch.sigmoid(logits, out=logits)
    return (flat * logits).to(x.dtype).reshape(x.shape)


class TestG1GateHelper:
    """`_apply_g1_gate` should produce `attn_out * sigmoid(gate)`."""

    @pytest.mark.parametrize("shape", [(4, 7168), (1, 16384), (8, 2048)])
    def test_bf16_cuda_fast_path_matches_reference(
        self, device: torch.device, shape: tuple[int, ...]
    ) -> None:
        from sglang.srt.models.deepseek_v2 import _apply_g1_gate

        torch.manual_seed(2026)
        attn_out = torch.randn(shape, device=device, dtype=torch.bfloat16)
        gate = torch.randn(shape, device=device, dtype=torch.bfloat16)

        got = _apply_g1_gate(attn_out, gate)
        expected = _ref_g1(attn_out, gate)
        torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)

    def test_fallback_path_used_for_fp32(self, device: torch.device) -> None:
        from sglang.srt.models.deepseek_v2 import _apply_g1_gate

        attn_out = torch.randn(8, 256, device=device, dtype=torch.float32)
        gate = torch.randn(8, 256, device=device, dtype=torch.float32)
        got = _apply_g1_gate(attn_out, gate)
        expected = _ref_g1(attn_out, gate)
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)

    def test_zero_gate_halves_output(self, device: torch.device) -> None:
        """sigmoid(0) = 0.5 -> gated output should be half the input."""
        from sglang.srt.models.deepseek_v2 import _apply_g1_gate

        attn_out = torch.ones(4, 64, device=device, dtype=torch.bfloat16)
        gate = torch.zeros(4, 64, device=device, dtype=torch.bfloat16)
        got = _apply_g1_gate(attn_out, gate)
        torch.testing.assert_close(
            got, torch.full_like(attn_out, 0.5), atol=2e-2, rtol=2e-2
        )


class _StubOwner:
    pass


class TestG1ForwardPreHook:
    """`_g1_gate_pre_hook` must substitute the first positional argument
    with `_apply_g1_gate(arg, owner._g1_pending_gate)` and clear the gate."""

    def test_hook_substitutes_args_and_clears_gate(
        self, device: torch.device
    ) -> None:
        from sglang.srt.models.deepseek_v2 import _g1_gate_pre_hook

        owner = _StubOwner()
        gate = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
        owner._g1_pending_gate = gate
        attn_out = torch.randn(4, 128, device=device, dtype=torch.bfloat16)

        proj = nn.Linear(128, 64, bias=False).to(device).to(torch.bfloat16)
        proj._g1_owner_ref = weakref.ref(owner)
        proj.register_forward_pre_hook(_g1_gate_pre_hook, with_kwargs=True)

        out = proj(attn_out)
        # Hook should have applied attn_out * sigmoid(gate) before the matmul
        expected_input = (attn_out.float() * torch.sigmoid(gate.float())).to(
            torch.bfloat16
        )
        expected_out = expected_input @ proj.weight.t()
        torch.testing.assert_close(out, expected_out, atol=2e-2, rtol=2e-2)
        # Gate must be cleared so a re-entrant call cannot reuse it
        assert owner._g1_pending_gate is None

    def test_hook_no_op_when_no_pending_gate(
        self, device: torch.device
    ) -> None:
        from sglang.srt.models.deepseek_v2 import _g1_gate_pre_hook

        owner = _StubOwner()
        owner._g1_pending_gate = None
        attn_out = torch.randn(2, 64, device=device, dtype=torch.bfloat16)
        proj = nn.Linear(64, 32, bias=False).to(device).to(torch.bfloat16)
        proj._g1_owner_ref = weakref.ref(owner)
        proj.register_forward_pre_hook(_g1_gate_pre_hook, with_kwargs=True)

        out = proj(attn_out)
        torch.testing.assert_close(out, attn_out @ proj.weight.t())


class TestGatedNormHelper:
    """`_maybe_apply_gated_norm` should produce
    `x * sigmoid(silu(x @ w_down.T) @ w_up.T)`."""

    def _stub_owner_with_method(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer

        owner = _StubOwner()
        owner._maybe_apply_gated_norm = (
            DeepseekV2DecoderLayer._maybe_apply_gated_norm.__get__(owner)
        )
        return owner

    @pytest.mark.parametrize(
        "hidden,rank,n_tokens",
        [(7168, 16, 4), (7168, 16, 1), (4096, 8, 12)],
    )
    def test_bf16_cuda_matches_reference(
        self,
        device: torch.device,
        hidden: int,
        rank: int,
        n_tokens: int,
    ) -> None:
        owner = self._stub_owner_with_method()
        torch.manual_seed(2026)
        x = torch.randn(n_tokens, hidden, device=device, dtype=torch.bfloat16)
        gate_down = nn.Linear(hidden, rank, bias=False).to(device).to(torch.bfloat16)
        gate_up = nn.Linear(rank, hidden, bias=False).to(device).to(torch.bfloat16)

        got = owner._maybe_apply_gated_norm(x, gate_down, gate_up)
        expected = _ref_gated_norm(x, gate_down.weight, gate_up.weight)
        torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)

    def test_no_op_when_either_gate_missing(
        self, device: torch.device
    ) -> None:
        owner = self._stub_owner_with_method()
        x = torch.randn(4, 256, device=device, dtype=torch.bfloat16)
        gate_down = nn.Linear(256, 8, bias=False).to(device).to(torch.bfloat16)
        gate_up = nn.Linear(8, 256, bias=False).to(device).to(torch.bfloat16)

        # No-op cases: either projection None
        assert owner._maybe_apply_gated_norm(x, None, gate_up) is x
        assert owner._maybe_apply_gated_norm(x, gate_down, None) is x
        assert owner._maybe_apply_gated_norm(x, None, None) is x

    def test_tuple_input_passthrough(self, device: torch.device) -> None:
        """Quantized-input paths pass tuple(weight, scale); GN must passthrough."""
        owner = self._stub_owner_with_method()
        x = torch.randn(4, 256, device=device, dtype=torch.bfloat16)
        scale = torch.randn(4, device=device, dtype=torch.float32)
        gate_down = nn.Linear(256, 8, bias=False).to(device).to(torch.bfloat16)
        gate_up = nn.Linear(8, 256, bias=False).to(device).to(torch.bfloat16)

        result = owner._maybe_apply_gated_norm((x, scale), gate_down, gate_up)
        assert result is not None
        assert isinstance(result, tuple)
        assert result[0] is x

    def test_zero_input_yields_zero(self, device: torch.device) -> None:
        """`x * sigmoid(silu(0) @ w_up) = 0 * sigmoid(0) = 0`."""
        owner = self._stub_owner_with_method()
        x = torch.zeros(4, 64, device=device, dtype=torch.bfloat16)
        gate_down = nn.Linear(64, 4, bias=False).to(device).to(torch.bfloat16)
        gate_up = nn.Linear(4, 64, bias=False).to(device).to(torch.bfloat16)
        got = owner._maybe_apply_gated_norm(x, gate_down, gate_up)
        torch.testing.assert_close(got, torch.zeros_like(x))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
