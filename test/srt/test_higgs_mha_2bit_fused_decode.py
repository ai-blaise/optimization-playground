"""Correctness test for the fused HIGGS-MHA-2bit decode attention kernel.

Compares the fused kernel
(:func:`decode_attention_fwd_higgs`) against a numpy/torch oracle:
``HiggsMHA2BitCodec.decompress`` followed by a vanilla
``softmax(QK^T / sqrt(d)) V`` attention.

The test exercises the GLM draft model's attention shape
(``head_dim=128``, ``num_q_heads=32``, ``num_kv_heads=2`` = GQA fan-in
16) at several batch / sequence-length combinations relevant to the
SMC-SD draft path.

A parallel test confirms the in-kernel FWHT_128 matches the eager
codec ``_fwht`` exactly (sanity-checks the butterfly + 1/sqrt(128)
scaling).
"""

from __future__ import annotations

import math
import unittest

import pytest
import torch

from sglang.srt.layers.quantization.higgs_dense_2bit_kv import _fwht
from sglang.srt.layers.quantization.higgs_mha_2bit_kv import (
    HiggsMHA2BitCodec,
    HiggsMHA2BitConfig,
)

cuda_available = torch.cuda.is_available()


def _reference_decode_attention(
    q: torch.Tensor,  # (B, num_q_heads, head_dim) bf16
    k_full: torch.Tensor,  # (B, S, num_kv_heads, head_dim) bf16
    v_full: torch.Tensor,  # (B, S, num_kv_heads, head_dim) bf16
    scaling: float,
) -> torch.Tensor:
    """Eager BF16 GQA decode attention oracle."""

    batch, num_q_heads, head_dim = q.shape
    _, seq_len, num_kv_heads, _ = k_full.shape
    kv_group_num = num_q_heads // num_kv_heads
    assert kv_group_num * num_kv_heads == num_q_heads
    # Broadcast K/V across the Q group fan-in.
    k = k_full.repeat_interleave(kv_group_num, dim=2)  # (B, S, num_q, head_dim)
    v = v_full.repeat_interleave(kv_group_num, dim=2)
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    # qk: (B, num_q, S)
    qk = torch.einsum("bhd,bshd->bhs", qf, kf) * scaling
    p = torch.softmax(qk, dim=-1)
    o = torch.einsum("bhs,bshd->bhd", p, vf)
    return o.to(q.dtype)


class TestFwhtMatch(unittest.TestCase):
    """Compare the in-kernel FWHT_128 vs the eager codec ``_fwht``."""

    @unittest.skipIf(not cuda_available, "CUDA required")
    def test_fwht_butterfly_matches_eager(self):
        from sglang.srt.layers.attention.triton_ops.higgs_decode_attention import (
            HIGGS_HEAD_DIM,
            _fwht_butterfly_128,
        )
        import triton

        torch.manual_seed(0)
        block_n = 16
        x = torch.randn(block_n, HIGGS_HEAD_DIM, device="cuda", dtype=torch.float32)
        expected = _fwht(x.cpu()).to("cuda")

        out = torch.empty_like(x)

        import triton.language as tl

        @triton.jit
        def _runner(X, Y, BLOCK_N: tl.constexpr):
            x_t = tl.load(
                X + tl.arange(0, BLOCK_N)[:, None] * HIGGS_HEAD_DIM
                + tl.arange(0, HIGGS_HEAD_DIM)[None, :]
            )
            y = _fwht_butterfly_128(x_t, BLOCK_N)
            tl.store(
                Y + tl.arange(0, BLOCK_N)[:, None] * HIGGS_HEAD_DIM
                + tl.arange(0, HIGGS_HEAD_DIM)[None, :],
                y,
            )

        _runner[(1,)](x, out, BLOCK_N=block_n)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out, expected, atol=2e-5, rtol=1e-4,
            msg="In-kernel FWHT_128 disagrees with the eager codec _fwht.",
        )


class TestFusedHiggsDecode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not cuda_available:
            raise unittest.SkipTest("CUDA required for fused HIGGS decode")
        cls.device = torch.device("cuda")
        torch.manual_seed(0xC0DEC)

    def _run_case(self, batch: int, seq_len: int):
        from sglang.srt.layers.attention.triton_ops.higgs_decode_attention import (
            HIGGS_HEAD_DIM,
            decode_attention_fwd_higgs,
        )

        head_dim = HIGGS_HEAD_DIM
        num_q_heads = 32
        num_kv_heads = 2
        scaling = 1.0 / math.sqrt(head_dim)

        codec = HiggsMHA2BitCodec(
            HiggsMHA2BitConfig(head_dim=head_dim), device=self.device
        )

        # Synthesize K/V for the full (B, S, H_kv, D) cache.
        k_full = torch.randn(
            batch, seq_len, num_kv_heads, head_dim,
            device=self.device, dtype=torch.bfloat16,
        )
        v_full = torch.randn(
            batch, seq_len, num_kv_heads, head_dim,
            device=self.device, dtype=torch.bfloat16,
        )
        # Compress + decompress to put both reference and fused on the
        # *same dequantized cache content* (otherwise we'd compare
        # against the lossless K/V, which the kernel cannot recover —
        # the codec is lossy).
        k_packed = codec.compress(
            k_full.reshape(batch * seq_len, num_kv_heads, head_dim)
        ).reshape(batch * seq_len, num_kv_heads, codec.slot_bytes)
        v_packed = codec.compress(
            v_full.reshape(batch * seq_len, num_kv_heads, head_dim)
        ).reshape(batch * seq_len, num_kv_heads, codec.slot_bytes)
        # Dequantize once for the reference path.
        k_ref = codec.decompress(
            k_packed.reshape(batch * seq_len, num_kv_heads, codec.slot_bytes),
            torch.bfloat16,
        ).reshape(batch, seq_len, num_kv_heads, head_dim)
        v_ref = codec.decompress(
            v_packed.reshape(batch * seq_len, num_kv_heads, codec.slot_bytes),
            torch.bfloat16,
        ).reshape(batch, seq_len, num_kv_heads, head_dim)

        # Q for the decode step (one token per batch row).
        q = torch.randn(
            batch, num_q_heads, head_dim,
            device=self.device, dtype=torch.bfloat16,
        )

        # Reference attention over the dequantized cache.
        o_ref = _reference_decode_attention(q, k_ref, v_ref, scaling)

        # Build SGLang-style page-table (kv_indptr / kv_indices) and
        # split metadata for the fused kernel.
        # k_packed is laid out (batch * seq_len, num_kv_heads, 34);
        # kv_indices for batch b is range(b*seq_len, (b+1)*seq_len).
        kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=self.device)
        kv_indptr[1:] = seq_len * torch.arange(
            1, batch + 1, dtype=torch.int32, device=self.device
        )
        kv_indices = torch.arange(
            batch * seq_len, dtype=torch.int64, device=self.device
        )
        max_kv_splits = 8
        num_kv_splits = torch.full(
            (batch,), max_kv_splits, dtype=torch.int32, device=self.device
        )

        attn_logits = torch.zeros(
            batch, num_q_heads, max_kv_splits, head_dim,
            device=self.device, dtype=torch.bfloat16,
        )
        attn_lse = torch.zeros(
            batch, num_q_heads, max_kv_splits,
            device=self.device, dtype=torch.float32,
        )
        o_fused = torch.empty_like(o_ref)

        decode_attention_fwd_higgs(
            q,
            k_packed,
            v_packed,
            o_fused,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            scaling,
        )
        torch.cuda.synchronize()

        # bf16 attention has ~1e-2 abs noise after S~1k tokens of
        # softmax-weighted accumulation; tighten if you see better.
        torch.testing.assert_close(
            o_fused, o_ref, atol=5e-2, rtol=5e-2,
            msg=(
                f"Fused HIGGS decode mismatched ref oracle: "
                f"batch={batch}, seq_len={seq_len}, "
                f"max_diff={(o_fused - o_ref).abs().max().item():.4e}"
            ),
        )
        # Cosine similarity per (batch, head); should be > 0.99
        flat_f = o_fused.reshape(batch * 32, head_dim).to(torch.float32)
        flat_r = o_ref.reshape(batch * 32, head_dim).to(torch.float32)
        cos = torch.nn.functional.cosine_similarity(flat_f, flat_r, dim=-1).min()
        self.assertGreater(
            cos.item(), 0.99,
            f"Cosine similarity {cos.item():.4f} < 0.99 "
            f"at batch={batch}, seq_len={seq_len}",
        )

    def test_decode_small(self):
        self._run_case(batch=1, seq_len=256)

    def test_decode_medium(self):
        self._run_case(batch=4, seq_len=1024)

    def test_decode_large(self):
        self._run_case(batch=8, seq_len=2048)


class TestHiggsMHAKVKernels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not cuda_available:
            raise unittest.SkipTest("CUDA required for HIGGS MHA KV kernels")
        cls.device = torch.device("cuda")
        torch.manual_seed(0x2B17)

    def test_triton_store_and_cuda_dequant_match_codec(self):
        from sglang.jit_kernel.higgs_mha_2bit_kv import dequantize_higgs_mha_2bit
        from sglang.srt.layers.attention.triton_ops.higgs_mha_kv_pack import (
            store_higgs_mha_2bit_triton,
        )

        head_dim = 128
        num_rows = 257
        num_slots = 300
        num_kv_heads = 2
        codec = HiggsMHA2BitCodec(
            HiggsMHA2BitConfig(head_dim=head_dim), device=self.device
        )
        cache = torch.randn(
            num_rows,
            num_kv_heads,
            head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        locs = torch.randperm(num_slots, device=self.device, dtype=torch.int64)[
            :num_rows
        ]
        packed = torch.zeros(
            num_slots,
            num_kv_heads,
            codec.slot_bytes,
            device=self.device,
            dtype=torch.uint8,
        )
        store_higgs_mha_2bit_triton(packed, locs, cache)

        expected = torch.zeros_like(packed)
        expected[locs] = codec.compress(cache)
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
        out = torch.empty(
            num_slots,
            num_kv_heads,
            head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        dequantize_higgs_mha_2bit(packed, out, codec.codebook)
        ref = codec.decompress(expected, torch.bfloat16)
        selected = out[locs].reshape(num_rows * num_kv_heads, head_dim)
        selected_ref = ref[locs].reshape(num_rows * num_kv_heads, head_dim)
        cos = torch.nn.functional.cosine_similarity(
            selected.float(), selected_ref.float(), dim=-1
        )
        self.assertGreater(cos.min().item(), 0.99)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
