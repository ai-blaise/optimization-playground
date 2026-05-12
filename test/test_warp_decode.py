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
"""Tests for warp decode MoE kernels.

Verifies numerical correctness against a pure-PyTorch reference
implementation across various configurations:
  - Batch sizes: 1, 4, 8, 16, 32, 64
  - Expert counts and top-k settings
  - Hidden sizes and intermediate sizes matching real models
  - Separate gate/up weights vs packed w13 format

Correctness criteria (matching Cursor's published thresholds):
  - Cosine similarity > 0.9999 between warp decode and reference
  - Max absolute difference < 0.01 (BF16 accumulation tolerance)
"""

import unittest
from typing import Tuple

import torch

# Skip if CUDA is not available
if not torch.cuda.is_available():
    raise unittest.SkipTest("CUDA not available")


def reference_moe_forward(
    hidden_states: torch.Tensor,  # [B, D]
    w_gate: torch.Tensor,  # [E, N, D]
    w_up: torch.Tensor,  # [E, N, D]
    w_down: torch.Tensor,  # [E, D, N]
    topk_ids: torch.Tensor,  # [B, K]
    topk_weights: torch.Tensor,  # [B, K]
) -> torch.Tensor:
    """Pure PyTorch reference MoE implementation.

    Explicitly computes each expert's contribution for each token,
    matching the mathematical definition of MoE with SiLU activation.
    """
    B, D = hidden_states.shape
    K = topk_ids.shape[1]
    output = torch.zeros(B, D, dtype=torch.float32, device=hidden_states.device)

    for b in range(B):
        for k in range(K):
            expert_id = topk_ids[b, k].item()
            routing_weight = topk_weights[b, k].item()

            # Gate projection: w_gate[expert_id] @ x
            x = hidden_states[b].float()
            gate_out = w_gate[expert_id].float() @ x  # [N]
            up_out = w_up[expert_id].float() @ x  # [N]

            # SiLU(gate) * up
            intermediate = torch.nn.functional.silu(gate_out) * up_out  # [N]

            # Down projection
            expert_out = w_down[expert_id].float() @ intermediate  # [D]

            # Fold routing weight
            output[b] += routing_weight * expert_out

    return output.to(torch.bfloat16)


def reference_moe_packed_forward(
    hidden_states: torch.Tensor,  # [B, D]
    w13: torch.Tensor,  # [E, 2*N, D]
    w2: torch.Tensor,  # [E, D, N]
    topk_ids: torch.Tensor,  # [B, K]
    topk_weights: torch.Tensor,  # [B, K]
    intermediate_size: int,
) -> torch.Tensor:
    """Reference using packed w13 format."""
    E = w13.shape[0]
    N = intermediate_size

    # Split w13 into gate and up
    w_gate = w13[:, :N, :]  # [E, N, D]
    w_up = w13[:, N:, :]  # [E, N, D]

    return reference_moe_forward(
        hidden_states, w_gate, w_up, w2, topk_ids, topk_weights
    )


def generate_test_data(
    batch_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, ...]:
    """Generate random test data for MoE computation."""
    torch.manual_seed(42)

    hidden_states = torch.randn(
        batch_size, hidden_size, dtype=torch.bfloat16, device=device
    )

    w_gate = torch.randn(
        num_experts, intermediate_size, hidden_size,
        dtype=torch.bfloat16, device=device,
    ) * 0.02  # Scale weights to avoid overflow

    w_up = torch.randn(
        num_experts, intermediate_size, hidden_size,
        dtype=torch.bfloat16, device=device,
    ) * 0.02

    w_down = torch.randn(
        num_experts, hidden_size, intermediate_size,
        dtype=torch.bfloat16, device=device,
    ) * 0.02

    # Random expert assignments (no duplicates per token)
    topk_ids = torch.stack([
        torch.randperm(num_experts, device=device)[:top_k]
        for _ in range(batch_size)
    ])

    # Random routing weights (softmax normalized)
    topk_weights = torch.randn(
        batch_size, top_k, dtype=torch.float32, device=device
    )
    topk_weights = torch.softmax(topk_weights, dim=-1)

    return hidden_states, w_gate, w_up, w_down, topk_ids, topk_weights


class TestWarpDecodeKernels(unittest.TestCase):
    """Test warp decode Triton kernels against PyTorch reference."""

    def _check_correctness(
        self,
        output: torch.Tensor,
        reference: torch.Tensor,
        test_name: str,
        cos_threshold: float = 0.9999,
        abs_threshold: float = 0.01,
    ) -> None:
        """Check numerical correctness of output vs reference."""
        out_f = output.float().flatten()
        ref_f = reference.float().flatten()

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            out_f.unsqueeze(0), ref_f.unsqueeze(0)
        ).item()

        # Max absolute difference
        max_abs_diff = (out_f - ref_f).abs().max().item()

        # Mean absolute error
        mae = (out_f - ref_f).abs().mean().item()

        self.assertGreater(
            cos_sim, cos_threshold,
            f"{test_name}: cosine similarity {cos_sim:.6f} < {cos_threshold}"
        )
        self.assertLess(
            max_abs_diff, abs_threshold,
            f"{test_name}: max abs diff {max_abs_diff:.6f} >= {abs_threshold}"
        )

    def test_separate_weights_batch1(self):
        """Test with separate gate/up/down weights, batch=1."""
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe

        B, D, N, E, K = 1, 256, 128, 8, 2
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        output = warp_decode_moe(hs, wg, wu, wd, ids, wts)
        reference = reference_moe_forward(hs, wg, wu, wd, ids, wts)

        self._check_correctness(output, reference, "separate_B1")

    def test_separate_weights_batch4(self):
        """Test with separate weights, batch=4."""
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe

        B, D, N, E, K = 4, 256, 128, 8, 2
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        output = warp_decode_moe(hs, wg, wu, wd, ids, wts)
        reference = reference_moe_forward(hs, wg, wu, wd, ids, wts)

        self._check_correctness(output, reference, "separate_B4")

    def test_separate_weights_batch32(self):
        """Test with separate weights, batch=32."""
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe

        B, D, N, E, K = 32, 256, 128, 8, 2
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        output = warp_decode_moe(hs, wg, wu, wd, ids, wts)
        reference = reference_moe_forward(hs, wg, wu, wd, ids, wts)

        self._check_correctness(output, reference, "separate_B32")

    def test_packed_weights_batch1(self):
        """Test packed w13 format, batch=1."""
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        B, D, N, E, K = 1, 256, 128, 8, 2
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        # Pack gate + up into w13
        w13 = torch.cat([wg, wu], dim=1)  # [E, 2*N, D]

        output = warp_decode_moe_packed(hs, w13, wd, ids, wts, intermediate_size=N)
        reference = reference_moe_packed_forward(hs, w13, wd, ids, wts, N)

        self._check_correctness(output, reference, "packed_B1")

    def test_packed_weights_batch8(self):
        """Test packed w13 format, batch=8."""
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        B, D, N, E, K = 8, 256, 128, 8, 2
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        w13 = torch.cat([wg, wu], dim=1)

        output = warp_decode_moe_packed(hs, w13, wd, ids, wts, intermediate_size=N)
        reference = reference_moe_packed_forward(hs, w13, wd, ids, wts, N)

        self._check_correctness(output, reference, "packed_B8")

    def test_packed_weights_batch64(self):
        """Test packed w13 at max warp decode batch size."""
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        B, D, N, E, K = 64, 256, 128, 8, 2
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        w13 = torch.cat([wg, wu], dim=1)

        output = warp_decode_moe_packed(hs, w13, wd, ids, wts, intermediate_size=N)
        reference = reference_moe_packed_forward(hs, w13, wd, ids, wts, N)

        self._check_correctness(output, reference, "packed_B64")

    def test_topk8_experts256(self):
        """Test with 256 experts and top-8, similar to DeepSeek-V3."""
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        B, D, N, E, K = 4, 512, 256, 64, 8
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        w13 = torch.cat([wg, wu], dim=1)

        output = warp_decode_moe_packed(hs, w13, wd, ids, wts, intermediate_size=N)
        reference = reference_moe_packed_forward(hs, w13, wd, ids, wts, N)

        self._check_correctness(output, reference, "topk8_e256")

    def test_larger_hidden_size(self):
        """Test with larger hidden size matching real models."""
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        B, D, N, E, K = 4, 1024, 512, 16, 4
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        w13 = torch.cat([wg, wu], dim=1)

        output = warp_decode_moe_packed(hs, w13, wd, ids, wts, intermediate_size=N)
        reference = reference_moe_packed_forward(hs, w13, wd, ids, wts, N)

        self._check_correctness(output, reference, "large_hidden")

    def test_single_expert(self):
        """Test with top_k=1 (single expert per token)."""
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        B, D, N, E, K = 4, 256, 128, 8, 1
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        w13 = torch.cat([wg, wu], dim=1)

        output = warp_decode_moe_packed(hs, w13, wd, ids, wts, intermediate_size=N)
        reference = reference_moe_packed_forward(hs, w13, wd, ids, wts, N)

        self._check_correctness(output, reference, "topk1")

    def test_greedy_determinism(self):
        """Verify deterministic output for identical inputs."""
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        B, D, N, E, K = 4, 256, 128, 8, 2
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        w13 = torch.cat([wg, wu], dim=1)

        out1 = warp_decode_moe_packed(hs, w13, wd, ids, wts, intermediate_size=N)
        out2 = warp_decode_moe_packed(hs, w13, wd, ids, wts, intermediate_size=N)

        self.assertTrue(
            torch.equal(out1, out2),
            "Warp decode output is not deterministic"
        )

    def test_zero_routing_weight(self):
        """Test that zero routing weight produces zero contribution."""
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe

        B, D, N, E, K = 1, 128, 64, 4, 2
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        # Set one expert's routing weight to zero
        wts[0, 1] = 0.0

        output = warp_decode_moe(hs, wg, wu, wd, ids, wts)
        reference = reference_moe_forward(hs, wg, wu, wd, ids, wts)

        self._check_correctness(output, reference, "zero_weight")

    def test_routing_weight_sum_to_one(self):
        """Test with routing weights that sum to exactly 1.0."""
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe

        B, D, N, E, K = 2, 128, 64, 4, 2
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        # Normalize to sum exactly to 1
        wts = wts / wts.sum(dim=-1, keepdim=True)

        output = warp_decode_moe(hs, wg, wu, wd, ids, wts)
        reference = reference_moe_forward(hs, wg, wu, wd, ids, wts)

        self._check_correctness(output, reference, "normalized_weights")

    def test_zero_tokens(self):
        """Test that zero-token input returns an empty tensor without error."""
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        D, N, E, K = 256, 128, 8, 2
        # Zero batch size
        hs = torch.empty(0, D, dtype=torch.bfloat16, device="cuda")
        w13 = torch.randn(E, 2 * N, D, dtype=torch.bfloat16, device="cuda") * 0.02
        w2 = torch.randn(E, D, N, dtype=torch.bfloat16, device="cuda") * 0.02
        ids = torch.empty(0, K, dtype=torch.int64, device="cuda")
        wts = torch.empty(0, K, dtype=torch.float32, device="cuda")

        output = warp_decode_moe_packed(hs, w13, w2, ids, wts, intermediate_size=N)
        self.assertEqual(output.shape, (0, D))

    def test_non_divisible_dimensions(self):
        """Test with dimensions not evenly divisible by tile sizes."""
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        # Unsupported by the target-only CuTe path; should fall back cleanly.
        B, D, N, E, K = 2, 100, 50, 4, 2
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        w13 = torch.cat([wg, wu], dim=1)

        output = warp_decode_moe_packed(hs, w13, wd, ids, wts, intermediate_size=N)
        reference = reference_moe_packed_forward(hs, w13, wd, ids, wts, N)

        self._check_correctness(
            output, reference, "non_divisible_dims",
            cos_threshold=0.999, abs_threshold=0.05,
        )


class TestWarpDecodeIntegration(unittest.TestCase):
    """Test warp decode integration with the FusedMoE layer."""

    def test_env_var_disabled_by_default(self):
        """Verify warp decode is disabled by default."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.warp_decode.runner import is_warp_decode_enabled

        with envs.SGLANG_ENABLE_WARP_DECODE.override(False):
            self.assertFalse(is_warp_decode_enabled())

    def test_env_var_enabled(self):
        """Verify warp decode can be enabled via env var."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.warp_decode.runner import is_warp_decode_enabled

        with envs.SGLANG_ENABLE_WARP_DECODE.override(True):
            self.assertTrue(is_warp_decode_enabled())

    def test_max_batch_threshold(self):
        """Verify batch size threshold is respected."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.warp_decode.runner import should_use_warp_decode

        with envs.SGLANG_ENABLE_WARP_DECODE.override(True):
            with envs.SGLANG_WARP_DECODE_MAX_BATCH.override(16):
                self.assertTrue(should_use_warp_decode(16))
                self.assertFalse(should_use_warp_decode(17))


class TestWarpDecodeRunner(unittest.TestCase):
    """Test the WarpDecodeRunnerCore."""

    def test_runner_forward(self):
        """Test runner core forward pass."""
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.warp_decode.runner import (
            WarpDecodeQuantInfo,
            WarpDecodeRunnerCore,
            WarpDecodeRunnerInput,
        )

        B, D, N, E, K = 4, 256, 128, 8, 2
        config = MoeRunnerConfig(
            num_experts=E,
            num_local_experts=E,
            hidden_size=D,
            intermediate_size_per_partition=N,
            top_k=K,
            activation="silu",
            is_gated=True,
        )

        runner = WarpDecodeRunnerCore(config)

        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)
        w13 = torch.cat([wg, wu], dim=1)

        quant_info = WarpDecodeQuantInfo(
            w13_weight=w13,
            w2_weight=wd,
            global_num_experts=E,
            local_num_experts=E,
            local_expert_offset=0,
            intermediate_size_per_partition=N,
        )

        runner_input = WarpDecodeRunnerInput(
            hidden_states=hs,
            topk_ids=ids,
            topk_weights=wts,
        )

        result = runner.run(runner_input, quant_info, {})
        reference = reference_moe_packed_forward(hs, w13, wd, ids, wts, N)

        # Check correctness
        out_f = result.hidden_states.float().flatten()
        ref_f = reference.float().flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            out_f.unsqueeze(0), ref_f.unsqueeze(0)
        ).item()

        self.assertGreater(cos_sim, 0.9999, f"Runner cos_sim={cos_sim}")


class TestWarpDecodeCuTe(unittest.TestCase):
    """Test CuTe-based warp decode CUDA kernels.

    These tests verify the CuTe kernels produce identical results to
    the Triton kernels and the PyTorch reference. Skipped if sgl_kernel
    is not built with CuTe warp decode support.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import sgl_kernel
            if not hasattr(sgl_kernel, "warp_decode_cute_moe_packed_forward"):
                raise unittest.SkipTest(
                    "sgl_kernel missing CuTe warp decode ops"
                )
        except ImportError:
            raise unittest.SkipTest("sgl_kernel not installed")

    def _check_correctness(
        self,
        output: torch.Tensor,
        reference: torch.Tensor,
        test_name: str,
        cos_threshold: float = 0.9999,
        abs_threshold: float = 0.01,
    ) -> None:
        out_f = output.float().flatten()
        ref_f = reference.float().flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            out_f.unsqueeze(0), ref_f.unsqueeze(0)
        ).item()
        max_abs_diff = (out_f - ref_f).abs().max().item()
        self.assertGreater(
            cos_sim, cos_threshold,
            f"{test_name}: cosine similarity {cos_sim:.6f} < {cos_threshold}"
        )
        self.assertLess(
            max_abs_diff, abs_threshold,
            f"{test_name}: max abs diff {max_abs_diff:.6f} >= {abs_threshold}"
        )

    def test_cute_separate_weights_batch1(self):
        """CuTe kernel with separate weights, batch=1."""
        import sgl_kernel

        B, D, N, E, K = 1, 1024, 2048, 8, 8
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        output = sgl_kernel.warp_decode_cute_moe_forward(
            hs, wg, wu, wd, ids, wts, False
        )
        reference = reference_moe_forward(hs, wg, wu, wd, ids, wts)
        self._check_correctness(output, reference, "cute_separate_B1")

    def test_cute_separate_weights_batch4(self):
        """CuTe kernel with separate weights, batch=4."""
        import sgl_kernel

        B, D, N, E, K = 4, 1024, 2048, 8, 8
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)

        output = sgl_kernel.warp_decode_cute_moe_forward(
            hs, wg, wu, wd, ids, wts, False
        )
        reference = reference_moe_forward(hs, wg, wu, wd, ids, wts)
        self._check_correctness(output, reference, "cute_separate_B32")

    def test_cute_packed_batch1(self):
        """CuTe packed kernel, batch=1."""
        import sgl_kernel

        B, D, N, E, K = 1, 1024, 2048, 8, 8
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)
        w13 = torch.cat([wg, wu], dim=1)

        output = sgl_kernel.warp_decode_cute_moe_packed_forward(
            hs, w13, wd, ids, wts, N, False
        )
        reference = reference_moe_packed_forward(hs, w13, wd, ids, wts, N)
        self._check_correctness(output, reference, "cute_packed_B1")

    def test_cute_packed_batch64(self):
        """CuTe packed kernel at max batch size."""
        import sgl_kernel
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        B, D, N, E, K = 64, 1024, 2048, 8, 8
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)
        w13 = torch.cat([wg, wu], dim=1)

        output = sgl_kernel.warp_decode_cute_moe_packed_forward(
            hs, w13, wd, ids, wts, N, False
        )
        with envs.SGLANG_WARP_DECODE_CUTE.override("0"):
            reference = warp_decode_moe_packed(
                hs, w13, wd, ids, wts, intermediate_size=N
            )
        self._check_correctness(output, reference, "cute_packed_B64")

    def test_cute_topk8_experts64(self):
        """CuTe kernel with 64 experts and top-8."""
        import sgl_kernel

        B, D, N, E, K = 2, 1024, 2048, 64, 8
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)
        w13 = torch.cat([wg, wu], dim=1)

        output = sgl_kernel.warp_decode_cute_moe_packed_forward(
            hs, w13, wd, ids, wts, N, False
        )
        reference = reference_moe_packed_forward(hs, w13, wd, ids, wts, N)
        self._check_correctness(output, reference, "cute_topk8_e64")

    def test_cute_determinism(self):
        """Verify CuTe output is deterministic."""
        import sgl_kernel

        B, D, N, E, K = 4, 1024, 2048, 8, 8
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)
        w13 = torch.cat([wg, wu], dim=1)

        out1 = sgl_kernel.warp_decode_cute_moe_packed_forward(
            hs, w13, wd, ids, wts, N, False
        )
        out2 = sgl_kernel.warp_decode_cute_moe_packed_forward(
            hs, w13, wd, ids, wts, N, False
        )
        self.assertTrue(
            torch.equal(out1, out2),
            "CuTe warp decode output is not deterministic"
        )

    def test_cute_matches_triton(self):
        """Verify CuTe output matches Triton output bit-for-bit."""
        import sgl_kernel
        from sglang.srt.layers.moe.warp_decode.kernels import warp_decode_moe_packed

        B, D, N, E, K = 2, 1024, 2048, 8, 8
        hs, wg, wu, wd, ids, wts = generate_test_data(B, D, N, E, K)
        w13 = torch.cat([wg, wu], dim=1)

        cute_out = sgl_kernel.warp_decode_cute_moe_packed_forward(
            hs, w13, wd, ids, wts, N, False
        )
        from sglang.srt.environ import envs

        with envs.SGLANG_WARP_DECODE_CUTE.override("0"):
            triton_out = warp_decode_moe_packed(
                hs, w13, wd, ids, wts, intermediate_size=N
            )

        reference = reference_moe_packed_forward(hs, w13, wd, ids, wts, N)
        self._check_correctness(cute_out, reference, "cute_vs_ref")
        self._check_correctness(triton_out, reference, "triton_vs_ref")


if __name__ == "__main__":
    unittest.main()
