import math
import unittest

import torch

from sglang.srt.layers.flashsampling.core import (
    LOCAL_INDEX_DTYPE,
    fused_mm_sample_triton,
)
from sglang.srt.layers.flashsampling.tp_info import TPInfo
from sglang.srt.layers.flashsampling.runtime import FlashSamplingRuntime
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, suite="stage-b-test-1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FlashSampling.")
class TestFlashSamplingKernel(unittest.TestCase):
    def test_greedy_matches_dense_argmax(self):
        torch.cuda.set_device(0)
        torch.manual_seed(0)
        vocab_size, hidden_size, batch_size = 512, 128, 8
        weights = torch.randn(
            vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        hidden_states = torch.randn(
            batch_size, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        temperature = torch.tensor(1.0, device="cuda")

        samples = fused_mm_sample_triton(
            weights,
            hidden_states,
            1,
            temperature,
            seed=0,
            greedy_sampling=True,
            valid_vocab_size=vocab_size,
        )
        expected = (hidden_states @ weights.T).float().argmax(dim=-1, keepdim=True)

        self.assertTrue(torch.equal(samples.cpu(), expected.cpu()))

    def test_blackwell_target_provider_matches_dense_argmax_and_scores(self):
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability(0)[0] < 10:
            self.skipTest("Blackwell target provider requires SM100 or newer.")

        from sglang.srt.layers.flashsampling.target_kernel_blackwell import (
            fused_mm_sample_blackwell,
        )

        torch.manual_seed(3)
        vocab_size, hidden_size, batch_size = 512, 128, 8
        weights = torch.randn(
            vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        hidden_states = torch.randn(
            batch_size, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        temperature = torch.tensor(1.0, device="cuda")

        samples, scores = fused_mm_sample_blackwell(
            weights,
            hidden_states,
            1,
            temperature,
            seed=0,
            greedy_sampling=True,
            return_scores=True,
            valid_vocab_size=vocab_size,
        )
        expected_scores, expected_samples = (hidden_states @ weights.T).float().max(
            dim=-1, keepdim=True
        )

        self.assertTrue(torch.equal(samples.cpu(), expected_samples.cpu()))
        self.assertLessEqual((scores - expected_scores).abs().max().item(), 0.30)

    def test_blackwell_block_h_uses_small_tile_for_warmup_buckets(self):
        from sglang.srt.layers.flashsampling.target_kernel_blackwell import (
            _block_h_blackwell,
        )

        self.assertEqual(_block_h_blackwell(1), 16)
        self.assertEqual(_block_h_blackwell(2), 8)
        self.assertEqual(_block_h_blackwell(8), 8)
        self.assertEqual(_block_h_blackwell(32), 32)
        self.assertEqual(_block_h_blackwell(64), 64)

    def test_debug_logits_match_dense_matmul(self):
        torch.cuda.set_device(0)
        torch.manual_seed(1)
        vocab_size, hidden_size, batch_size = 1024, 128, 16
        weights = (
            torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
            / math.sqrt(hidden_size)
        )
        hidden_states = torch.randn(
            batch_size, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        temperature = torch.tensor(1.0, device="cuda")

        samples, logits = fused_mm_sample_triton(
            weights,
            hidden_states,
            1,
            temperature,
            seed=7,
            greedy_sampling=True,
            return_logits=True,
            valid_vocab_size=vocab_size,
        )
        expected_logits = (hidden_states @ weights.T).float()
        expected_samples = expected_logits.argmax(dim=-1, keepdim=True)

        self.assertLessEqual((logits - expected_logits).abs().max().item(), 0.30)
        self.assertTrue(torch.equal(samples.cpu(), expected_samples.cpu()))

    def test_local_manual_tp_path_uses_vocab_start_index(self):
        torch.cuda.set_device(0)
        torch.manual_seed(2)
        vocab_size, hidden_size, batch_size = 512, 128, 8
        vocab_start_index = 4096
        weights = torch.randn(
            vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        hidden_states = torch.randn(
            batch_size, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        temperature = torch.tensor(1.0, device="cuda")

        samples = fused_mm_sample_triton(
            weights,
            hidden_states,
            1,
            temperature,
            seed=0,
            greedy_sampling=True,
            tp=TPInfo(rank=1, size=1),
            valid_vocab_size=vocab_size,
            vocab_start_index=vocab_start_index,
        )
        expected = (
            (hidden_states @ weights.T).float().argmax(dim=-1, keepdim=True)
            + vocab_start_index
        )

        self.assertTrue(torch.equal(samples.cpu(), expected.cpu()))

    def test_flat_logits_sample_in_range_and_use_seed(self):
        torch.cuda.set_device(0)
        vocab_size, hidden_size, batch_size = 128, 64, 8192
        weights = torch.zeros(
            vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        hidden_states = torch.zeros(
            batch_size, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        temperature = torch.tensor(1.0, device="cuda")

        first = fused_mm_sample_triton(
            weights,
            hidden_states,
            1,
            temperature,
            seed=11,
            valid_vocab_size=vocab_size,
        )
        second = fused_mm_sample_triton(
            weights,
            hidden_states,
            1,
            temperature,
            seed=12,
            valid_vocab_size=vocab_size,
        )

        self.assertFalse(torch.equal(first.cpu(), second.cpu()))
        self.assertGreaterEqual(int(first.min()), 0)
        self.assertLess(int(first.max()), vocab_size)

        counts = torch.bincount(first.flatten().cpu(), minlength=vocab_size).float()
        expected = batch_size / vocab_size
        max_rel = ((counts - expected).abs() / expected).max().item()
        self.assertLessEqual(max_rel, 1.50)

    def test_local_workspace_uses_compact_index_dtype(self):
        torch.cuda.set_device(0)
        runtime = FlashSamplingRuntime()

        workspaces = runtime._local_workspaces(torch.device("cuda:0"), 4, 8)

        self.assertEqual(workspaces["maxs_idx_workspace"].dtype, LOCAL_INDEX_DTYPE)
