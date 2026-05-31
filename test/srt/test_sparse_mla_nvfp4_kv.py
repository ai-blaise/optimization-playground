"""Correctness + perf harness for the NVFP4-KV sparse-MLA decode kernel.

Compares the new native-UMMA NVFP4 path against the FP8 trtllm-gen baseline.
Equivalence target: bit-exact-modulo-FP4-roundoff (the quantization step is
the only source of divergence; the UMMA math is identical modulo the
input precision).

Run inside the deepseek-v32 pod:

  kubectl -n dynamo-system exec <pod> -- python3 \\
    -m sglang.jit_kernel.test_sparse_mla_nvfp4_kv

Or invoke specific tests directly. See `pytest -k nvfp4` once integrated into
the SGLang test runner.
"""

import torch
import pytest

# Production shape match (matches our active deploy):
B = 8                    # per-rank batch
NUM_HEADS = 128          # tp_q_head_num at TP=8 is 16 per rank, but kernel sees H_Q=128 for 2-CTA
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_DIM_QK = KV_LORA_RANK + QK_ROPE_HEAD_DIM
NUM_PAGES = 256
PAGE_SIZE = 64
SPARSE_TOP_K = 1024


def _build_fp8_baseline_inputs(seed=0):
    """Construct FP8 KV + Q the way the SGLang trtllm path produces them."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Q (FP8 e4m3)
    q = torch.randn(B, 1, NUM_HEADS, HEAD_DIM_QK, dtype=torch.bfloat16,
                    device="cuda", generator=g).to(torch.float8_e4m3fn)
    # FP8 KV cache (single combined buffer in trtllm-gen layout)
    kv_fp8 = torch.randn(NUM_PAGES, 1, PAGE_SIZE, HEAD_DIM_QK,
                          dtype=torch.bfloat16, device="cuda", generator=g
                         ).to(torch.float8_e4m3fn)
    block_tables = torch.arange(NUM_PAGES * B, dtype=torch.int32, device="cuda"
                                 ).reshape(B, NUM_PAGES) % NUM_PAGES
    seq_lens = torch.full((B,), NUM_PAGES * PAGE_SIZE, dtype=torch.int32, device="cuda")
    topk_indices = torch.randint(0, NUM_PAGES * PAGE_SIZE,
                                  (B, SPARSE_TOP_K), dtype=torch.int32, device="cuda")
    return q, kv_fp8, block_tables, seq_lens, topk_indices


def _quantize_to_nvfp4(kv_bf16: torch.Tensor):
    """Quantize a BF16 KV tensor to NVFP4 (packed e2m1) + per-block E4M3 scales.

    Uses the existing NVFP4KVQuantizeUtil from sglang.srt.layers.quantization.kvfp4_tensor.
    """
    from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil
    # NVFP4KVQuantizeUtil.quantize returns (packed_uint8, fp8_scales, _)
    packed, scales, _ = NVFP4KVQuantizeUtil.quantize(kv_bf16.contiguous(), scale=None)
    return packed, scales


@pytest.mark.gpu
def test_nvfp4_kv_decode_runs():
    """Sanity: kernel loads + executes at production shape; returns expected output shape."""
    from sglang.jit_kernel.sparse_mla_nvfp4_kv import sparse_mla_nvfp4_kv

    q, kv_fp8_baseline, block_tables, seq_lens, topk_indices = _build_fp8_baseline_inputs()

    # Quantize the BF16 baseline to NVFP4 layout
    kv_bf16 = kv_fp8_baseline.to(torch.bfloat16)
    kv_nope_bf16 = kv_bf16[..., :KV_LORA_RANK].contiguous()
    kv_rope_bf16 = kv_bf16[..., KV_LORA_RANK:].contiguous()

    kv_nope_packed, kv_scales = _quantize_to_nvfp4(kv_nope_bf16)

    # Pad scales to NUM_NVFP4_SCALES_PADDED=32 (per config.h)
    if kv_scales.shape[-1] < 32:
        pad = 32 - kv_scales.shape[-1]
        kv_scales = torch.nn.functional.pad(kv_scales, (0, pad))

    out = sparse_mla_nvfp4_kv(
        query=q,
        kv_nope=kv_nope_packed,
        kv_scales=kv_scales,
        kv_rope=kv_rope_bf16,
        block_tables=block_tables,
        seq_lens=seq_lens,
        topk_indices=topk_indices,
        sparse_top_k=SPARSE_TOP_K,
        sm_scale=1.0 / (HEAD_DIM_QK ** 0.5),
    )
    assert out.shape == (B, NUM_HEADS, KV_LORA_RANK)
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.gpu
def test_nvfp4_kv_decode_numerics_vs_fp8():
    """Correctness: NVFP4 path output close to FP8 baseline within FP4 quantization noise."""
    import flashinfer.decode

    q, kv_fp8, block_tables, seq_lens, topk_indices = _build_fp8_baseline_inputs()
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    # FP8 reference (trtllm-gen cubin)
    out_fp8 = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_fp8,
        workspace_buffer=workspace,
        qk_nope_head_dim=KV_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=NUM_PAGES * PAGE_SIZE,
        sparse_mla_top_k=SPARSE_TOP_K,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
        backend="trtllm-gen",
    )

    # NVFP4 path
    from sglang.jit_kernel.sparse_mla_nvfp4_kv import sparse_mla_nvfp4_kv
    kv_bf16 = kv_fp8.to(torch.bfloat16)
    kv_nope_packed, kv_scales = _quantize_to_nvfp4(kv_bf16[..., :KV_LORA_RANK].contiguous())
    if kv_scales.shape[-1] < 32:
        kv_scales = torch.nn.functional.pad(kv_scales, (0, 32 - kv_scales.shape[-1]))

    out_nvfp4 = sparse_mla_nvfp4_kv(
        query=q,
        kv_nope=kv_nope_packed,
        kv_scales=kv_scales,
        kv_rope=kv_bf16[..., KV_LORA_RANK:].contiguous(),
        block_tables=block_tables,
        seq_lens=seq_lens,
        topk_indices=topk_indices,
        sparse_top_k=SPARSE_TOP_K,
        sm_scale=1.0 / (HEAD_DIM_QK ** 0.5),
    )

    # Quantization-noise tolerance: NVFP4 has ~3-bit effective precision per element
    # times the per-block E4M3 scale. Output diff should be within ~5% RMS.
    diff = (out_fp8.float() - out_nvfp4.float())
    rms = (diff ** 2).mean().sqrt().item()
    ref_rms = (out_fp8.float() ** 2).mean().sqrt().item()
    rel_err = rms / max(ref_rms, 1e-6)
    print(f"FP8 vs NVFP4 RMS diff: {rms:.4f} (ref RMS {ref_rms:.4f}, rel err {rel_err*100:.2f}%)")
    assert rel_err < 0.05, f"NVFP4 output diverges {rel_err*100:.2f}% from FP8 baseline (expected <5%)"


@pytest.mark.gpu
def test_nvfp4_kv_decode_perf_vs_fp8():
    """Perf: NVFP4 path should be faster than FP8 baseline on B200 (less HBM traffic + 2x FP4 throughput)."""
    import flashinfer.decode
    import time

    q, kv_fp8, block_tables, seq_lens, topk_indices = _build_fp8_baseline_inputs()
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    # Warmup
    for _ in range(10):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q, kv_cache=kv_fp8, workspace_buffer=workspace,
            qk_nope_head_dim=KV_LORA_RANK, kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=NUM_PAGES * PAGE_SIZE,
            sparse_mla_top_k=SPARSE_TOP_K, backend="trtllm-gen",
        )
    torch.cuda.synchronize()

    # FP8 baseline timing
    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q, kv_cache=kv_fp8, workspace_buffer=workspace,
            qk_nope_head_dim=KV_LORA_RANK, kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=NUM_PAGES * PAGE_SIZE,
            sparse_mla_top_k=SPARSE_TOP_K, backend="trtllm-gen",
        )
    torch.cuda.synchronize()
    t_fp8 = (time.perf_counter() - t0) / iters * 1e6
    print(f"FP8 trtllm-gen: {t_fp8:.1f} us/call")

    # NVFP4 timing
    from sglang.jit_kernel.sparse_mla_nvfp4_kv import sparse_mla_nvfp4_kv
    kv_bf16 = kv_fp8.to(torch.bfloat16)
    kv_nope_packed, kv_scales = _quantize_to_nvfp4(kv_bf16[..., :KV_LORA_RANK].contiguous())
    if kv_scales.shape[-1] < 32:
        kv_scales = torch.nn.functional.pad(kv_scales, (0, 32 - kv_scales.shape[-1]))
    kv_rope = kv_bf16[..., KV_LORA_RANK:].contiguous()

    # Warmup
    for _ in range(10):
        sparse_mla_nvfp4_kv(query=q, kv_nope=kv_nope_packed, kv_scales=kv_scales,
                             kv_rope=kv_rope, block_tables=block_tables,
                             seq_lens=seq_lens, topk_indices=topk_indices,
                             sparse_top_k=SPARSE_TOP_K, sm_scale=1.0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        sparse_mla_nvfp4_kv(query=q, kv_nope=kv_nope_packed, kv_scales=kv_scales,
                             kv_rope=kv_rope, block_tables=block_tables,
                             seq_lens=seq_lens, topk_indices=topk_indices,
                             sparse_top_k=SPARSE_TOP_K, sm_scale=1.0)
    torch.cuda.synchronize()
    t_nvfp4 = (time.perf_counter() - t0) / iters * 1e6
    print(f"NVFP4 native: {t_nvfp4:.1f} us/call")
    print(f"Speedup: {t_fp8 / t_nvfp4:.2f}x")

    # Target: NVFP4 should be at least 1.2x faster than FP8 (per the design's
    # HBM bandwidth + UMMA throughput math)
    assert t_nvfp4 < t_fp8 * 0.85, (
        f"NVFP4 ({t_nvfp4:.1f} us) not faster than FP8 ({t_fp8:.1f} us); "
        f"expected at least 1.2x speedup. Likely kernel needs further optimization."
    )


if __name__ == "__main__":
    test_nvfp4_kv_decode_runs()
    test_nvfp4_kv_decode_numerics_vs_fp8()
    test_nvfp4_kv_decode_perf_vs_fp8()
    print("All NVFP4 KV decode tests passed.")
