"""Python wrapper for the sparse-MLA NVFP4-KV decode kernel.

Native-UMMA path: tcgen05.mma.kind::f8f6f4.block_scale_vec::1X consumes
packed NVFP4 K + per-block E4M3 scale operand directly. No SMEM dequant.

Drop-in replacement for `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`
when KV is stored as NVFP4 in the pool. Same logical contract:
  query: [B, q_len=1, num_heads, head_dim_qk] FP8 e4m3
  kv_nope: [num_pages, page_size, D_NOPE/2] uint8 packed FP4
  kv_scales: [num_pages, page_size, num_blocks_per_token] uint8 E4M3
  kv_rope: [num_pages, page_size, D_ROPE] bf16
  block_tables: [B, max_pages_per_seq] int32
  topk_indices: [B, sparse_top_k] int32

Build is via tvm-ffi JIT on first call, same pattern as
sglang.jit_kernel.higgs_inline_sparse_mla_decode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_sparse_mla_nvfp4_kv_module() -> "Module":
    return load_jit(
        "sparse_mla_nvfp4_kv",
        cuda_files=[
            "sparse_mla_nvfp4_kv/nvfp4_variant/phase1.cuh",
            "sparse_mla_nvfp4_kv/nvfp4_variant/config.h",
            "sparse_mla_nvfp4_kv/nvfp4_variant/nvfp4_umma_descriptors.cuh",
            "sparse_mla_nvfp4_kv/nvfp4_variant/common_subroutine.h",
            "sparse_mla_nvfp4_kv/helpers.h",
            "sparse_mla_nvfp4_kv/params.h",
            "sparse_mla_nvfp4_kv/utils.h",
            "sparse_mla_nvfp4_kv/defines.h",
        ],
        extra_dependencies=["cutlass", "cute"],
        cuda_wrappers=[
            (
                "sparse_mla_nvfp4_kv_decode",
                "sm100::fwd_for_small_topk::head128_nvfp4::"
                "SparseAttnDecodeNVFP4Kernel::run",
            ),
            (
                "sparse_mla_nvfp4_kv_combine",
                "sm100::fwd_for_small_topk::head128_nvfp4::"
                "SparseAttnCombineNVFP4Kernel::run",
            ),
        ],
    )


# Layout constants (must match config.h)
D_K = 512                  # kv_lora_rank
D_ROPE = 64                # qk_rope_head_dim
NVFP4_BLOCK_SIZE = 16
NUM_NVFP4_SCALES_PADDED = 32   # padded scale entries per token
TMA_K_STRIDE_FOR_DECODING = 384  # NVFP4 K_nope (224) + scales (32) + rope (128)


@debug_kernel_api
def sparse_mla_nvfp4_kv_decode(
    query: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_scales: torch.Tensor,
    kv_rope: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    workspace: torch.Tensor,
    *,
    sparse_top_k: int,
    sm_scale: float,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Run the NVFP4-KV sparse-MLA decode kernel.

    Args:
      query: (B, q_len=1, num_heads, head_dim_qk=576) float8_e4m3fn — Q+rope concat
      kv_nope: (num_pages, page_size, D_NOPE/2 = 224) uint8 — packed FP4 latent values
      kv_scales: (num_pages, page_size, NUM_NVFP4_SCALES_PADDED = 32) uint8 — E4M3 per-block scales
      kv_rope: (num_pages, page_size, D_ROPE = 64) bfloat16 — rope component
      block_tables: (B, max_pages_per_seq) int32
      seq_lens: (B,) int32
      topk_indices: (B, sparse_top_k) int32 — sparse-MLA top-k selection
      workspace: (workspace_bytes,) uint8 — scratch for splitkv combine

    Returns:
      output: (B, num_heads, D_V = 512) bfloat16
    """
    # Shape + dtype contracts
    assert query.is_cuda and query.dtype == torch.float8_e4m3fn
    assert query.dim() == 4 and query.shape[-1] == D_K + D_ROPE, (
        f"query last dim must be D_K+D_ROPE={D_K + D_ROPE}, got {query.shape[-1]}"
    )

    assert kv_nope.is_cuda and kv_nope.dtype == torch.uint8
    assert kv_nope.dim() == 3 and kv_nope.shape[-1] == D_K // 2 + 0, (
        f"kv_nope last dim must be {D_K // 2} (FP4 packed); got {kv_nope.shape[-1]}"
    )

    assert kv_scales.is_cuda and kv_scales.dtype == torch.uint8
    assert kv_scales.dim() == 3 and kv_scales.shape[-1] == NUM_NVFP4_SCALES_PADDED, (
        f"kv_scales last dim must be {NUM_NVFP4_SCALES_PADDED}; got {kv_scales.shape[-1]}"
    )

    assert kv_rope.is_cuda and kv_rope.dtype == torch.bfloat16
    assert kv_rope.dim() == 3 and kv_rope.shape[-1] == D_ROPE

    assert block_tables.is_cuda and block_tables.dtype == torch.int32
    assert seq_lens.is_cuda and seq_lens.dtype == torch.int32
    assert topk_indices.is_cuda and topk_indices.dtype == torch.int32
    assert workspace.is_cuda and workspace.dtype == torch.uint8

    B = query.shape[0]
    num_heads = query.shape[2]
    out = torch.empty(B, num_heads, D_K, dtype=torch.bfloat16, device=query.device)

    module = _jit_sparse_mla_nvfp4_kv_module()
    module.sparse_mla_nvfp4_kv_decode(
        query.contiguous(),
        kv_nope,
        kv_scales,
        kv_rope,
        block_tables.contiguous(),
        seq_lens.contiguous(),
        topk_indices.contiguous(),
        out,
        workspace,
        int(sparse_top_k),
        float(sm_scale),
        float(q_scale),
    )
    return out


def compute_workspace_size(
    batch_size: int, num_heads: int = 128, num_sm_parts: int = 16
) -> int:
    """Compute the workspace buffer size for the splitkv combine stage.

    Heuristic from FlashMLA: ~256 bytes per (split * batch * head) +
    metadata. We allocate generously since the workspace is reused
    across decode steps.
    """
    sched_meta_bytes = 4096
    splitkv_accum_bytes = num_sm_parts * batch_size * num_heads * D_K * 4  # FP32
    return sched_meta_bytes + splitkv_accum_bytes


# Convenience: combined entry that allocates workspace internally.
def sparse_mla_nvfp4_kv(
    query: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_scales: torch.Tensor,
    kv_rope: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    sparse_top_k: int,
    sm_scale: float,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if workspace is None:
        B = query.shape[0]
        num_heads = query.shape[2]
        workspace = torch.empty(
            compute_workspace_size(B, num_heads),
            dtype=torch.uint8,
            device=query.device,
        )
    return sparse_mla_nvfp4_kv_decode(
        query, kv_nope, kv_scales, kv_rope,
        block_tables, seq_lens, topk_indices, workspace,
        sparse_top_k=sparse_top_k, sm_scale=sm_scale,
    )
