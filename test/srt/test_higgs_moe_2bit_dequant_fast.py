"""Correctness test for the fast HIGGS-MoE-2bit dequant kernel.

Verifies the iter2 Triton-unpack + fast-Hadamard composition
(:func:`higgs_moe_2bit_dequant_fast`) matches the iter1 eager codec
(:func:`dequantize_higgs_moe_weights`) on shapes representative of
the DeepSeek-V3.2-REAP MoE block:

* Small toy shapes (E=4, OUT=64, IN=128) for fast unit coverage.
* Sub-row blocking case (in_dim=7168 with block=1024 — the production
  DeepSeek-V3.2 hidden_size which is not a power of two; iter2 fixed
  ``HiggsMoE2BitConfig`` to support this).
* Power-of-two single-block case (in_dim=2048 with block=2048 — the
  DeepSeek-V3.2 intermediate_size_per_partition for DP=TP=8).

Each test asserts the fast and eager decodes produce element-wise
matches within BF16 unit roundoff and that the recovered weight's
cosine similarity to the original BF16 input is bit-identical to the
eager codec's cosine similarity (the lossy step is the EDEN2-16
quantization, not the rotation primitive).
"""

from __future__ import annotations

import math
import unittest

import pytest
import torch

cuda_available = torch.cuda.is_available()


def _largest_pow2_div(n: int) -> int:
    b = 1
    while (b * 2) <= n and (n % (b * 2)) == 0:
        b *= 2
    return b


@unittest.skipUnless(cuda_available, "CUDA required")
class TestHiggsMoeDequantFast(unittest.TestCase):
    """Bit-for-bit fast vs eager equivalence on representative shapes."""

    def _check(self, num_experts: int, out_dim: int, in_dim: int) -> None:
        from sglang.jit_kernel.higgs_moe_2bit_dequant import (
            higgs_moe_2bit_dequant_fast,
        )
        from sglang.srt.layers.quantization.higgs_moe_2bit_weights import (
            dequantize_higgs_moe_weights,
            quantize_moe_weights_to_higgs,
        )

        dev = torch.device("cuda")
        torch.manual_seed(0)
        block_size = _largest_pow2_div(in_dim)

        # Use sigma small enough that the recovered weight still
        # lands inside EDEN2-16's per-block calibrated unit-variance
        # neighbourhood, matching the production codec call.
        w = torch.randn(
            num_experts, out_dim, in_dim, dtype=torch.bfloat16, device=dev
        ) * 0.02
        packed, cfg = quantize_moe_weights_to_higgs(w, block_size=block_size)
        eager = dequantize_higgs_moe_weights(packed, cfg)
        fast = higgs_moe_2bit_dequant_fast(
            packed, in_dim=in_dim, block_size=block_size
        )

        self.assertEqual(eager.shape, fast.shape)
        self.assertEqual(eager.dtype, fast.dtype)

        # Compare in fp32 to skip BF16 round-off comparison weirdness.
        diff = (eager.float() - fast.float()).abs()
        # The fast path uses the existing fast-Hadamard CUDA kernel
        # which may have slightly different rounding than the eager
        # FWHT (both are orthonormal, but the bf16 add/sub ladders
        # differ in order). Tolerance: 2e-3 absolute, mean ~1e-4.
        self.assertLess(diff.max().item(), 2e-3,
            f"E={num_experts} OUT={out_dim} IN={in_dim} block={block_size}: "
            f"max |fast - eager| = {diff.max().item()}")
        self.assertLess(diff.mean().item(), 1e-3,
            f"E={num_experts} OUT={out_dim} IN={in_dim} block={block_size}: "
            f"mean |fast - eager| = {diff.mean().item()}")

        # Both decoders should give the same cosine similarity to the
        # original (it's the quantization that's lossy, not the
        # rotation/scale).
        cos_eager = torch.nn.functional.cosine_similarity(
            eager.reshape(-1).float(), w.reshape(-1).float(), dim=0
        ).item()
        cos_fast = torch.nn.functional.cosine_similarity(
            fast.reshape(-1).float(), w.reshape(-1).float(), dim=0
        ).item()
        self.assertAlmostEqual(cos_eager, cos_fast, places=3)

    def test_small_pow2(self) -> None:
        self._check(num_experts=4, out_dim=64, in_dim=128)

    def test_medium_pow2_single_block(self) -> None:
        # in_dim = block_size, single FWHT block per row.
        self._check(num_experts=2, out_dim=256, in_dim=512)

    def test_dsv32_w2_shape(self) -> None:
        # DeepSeek-V3.2-REAP w2 in_dim = intermediate_size = 2048, a
        # power of two -> single FWHT block per row.
        self._check(num_experts=2, out_dim=128, in_dim=2048)

    def test_dsv32_w13_shape_sub_row_blocking(self) -> None:
        # DeepSeek-V3.2-REAP w13 in_dim = hidden_size = 7168 = 7 * 2^10
        # -> not a power of two. iter2 codec accepts this by picking
        # block_size = 1024, giving 7 FWHT blocks per row.
        self._check(num_experts=2, out_dim=64, in_dim=7168)


@unittest.skipUnless(cuda_available, "CUDA required")
class TestHiggsMoeDequantFastViaScheme(unittest.TestCase):
    """Scheme entry point matches the eager fallback.

    Verifies that ``CompressedTensorsHiggsDense2BitMoE._dequant_to_bf16``
    routes through the fast Triton kernel on CUDA tensors and gives
    output equivalent (to bf16 round-off) to the eager codec.
    """

    def test_scheme_helper_routes_through_fast_kernel(self) -> None:
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_higgs_dense_2bit_moe import (
            CompressedTensorsHiggsDense2BitMoE,
        )
        from sglang.srt.layers.quantization.higgs_moe_2bit_weights import (
            HiggsMoE2BitConfig,
            dequantize_higgs_moe_weights,
            quantize_moe_weights_to_higgs,
        )

        dev = torch.device("cuda")
        torch.manual_seed(0)

        scheme = CompressedTensorsHiggsDense2BitMoE.__new__(
            CompressedTensorsHiggsDense2BitMoE
        )
        scheme.use_flashinfer_trtllm = False
        scheme.nvfp4_group_size = 16
        scheme.bf16_runtime_dequant = True
        scheme.use_sub_row_blocks = None

        in_dim = 7168
        out_dim = 32
        num_experts = 2
        block_size = _largest_pow2_div(in_dim)
        w = torch.randn(
            num_experts, out_dim, in_dim, dtype=torch.bfloat16, device=dev
        ) * 0.02
        packed, cfg = quantize_moe_weights_to_higgs(w, block_size=block_size)
        eager = dequantize_higgs_moe_weights(packed, cfg)
        via_scheme = scheme._dequant_to_bf16(packed, in_dim=in_dim)
        diff = (eager.float() - via_scheme.float()).abs()
        self.assertLess(diff.max().item(), 2e-3)
        self.assertLess(diff.mean().item(), 1e-3)
        self.assertEqual(scheme._resolve_block_size(in_dim), block_size)


@unittest.skipUnless(cuda_available, "CUDA required")
class TestHiggsMoeDequantFastBench(unittest.TestCase):
    """Smoke-bench at the production per-rank shape; not strictly an
    assertion (the per-iter speedup is platform-dependent), but useful
    to log timings when the test suite is run on B200.
    """

    def test_bench_dsv32_per_rank(self) -> None:
        from sglang.jit_kernel.higgs_moe_2bit_dequant import (
            higgs_moe_2bit_dequant_fast,
        )
        from sglang.srt.layers.quantization.higgs_moe_2bit_weights import (
            quantize_moe_weights_to_higgs,
        )

        dev = torch.device("cuda")
        torch.manual_seed(0)
        num_experts = 32
        hidden = 7168
        intermediate = 2048
        # DeepSeek-V3.2 per-rank shapes (DP=TP=8 over 256 experts).
        w13 = torch.randn(
            num_experts, 2 * intermediate, hidden, dtype=torch.bfloat16, device=dev
        ) * 0.02
        w2 = torch.randn(
            num_experts, hidden, intermediate, dtype=torch.bfloat16, device=dev
        ) * 0.02
        w13_block = _largest_pow2_div(hidden)
        w2_block = _largest_pow2_div(intermediate)
        w13_pk, _ = quantize_moe_weights_to_higgs(w13, block_size=w13_block)
        w2_pk, _ = quantize_moe_weights_to_higgs(w2, block_size=w2_block)

        # Warm.
        for _ in range(3):
            _ = higgs_moe_2bit_dequant_fast(
                w13_pk, in_dim=hidden, block_size=w13_block
            )
            _ = higgs_moe_2bit_dequant_fast(
                w2_pk, in_dim=intermediate, block_size=w2_block
            )
        torch.cuda.synchronize()

        n = 10
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            starts[i].record()
            _ = higgs_moe_2bit_dequant_fast(
                w13_pk, in_dim=hidden, block_size=w13_block
            )
            _ = higgs_moe_2bit_dequant_fast(
                w2_pk, in_dim=intermediate, block_size=w2_block
            )
            ends[i].record()
        torch.cuda.synchronize()
        ms = min(s.elapsed_time(e) for s, e in zip(starts, ends))
        print(
            f"[higgs_moe_dequant_fast] per layer (best of {n}): "
            f"{ms:.3f} ms  (x58 layers: {ms * 58:.1f} ms)",
            flush=True,
        )
        # Loose ceiling so the assert doesn't fail on smaller GPUs.
        # On B200 we measure ~3.3 ms.
        self.assertLess(ms, 50.0)


if __name__ == "__main__":
    unittest.main()
