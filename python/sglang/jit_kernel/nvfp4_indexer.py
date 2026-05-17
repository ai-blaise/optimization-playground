"""NVFP4/UE8M0 quantization kernels for the NSA Indexer."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.nvfp4 import _nvfp4_arch_env, _nvfp4_cuda_flags
from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

_cached_module = None
_cached_key = None
def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "off", "no", "")


# Use the fused exact radix top-k tail by default. Disable via env to fall back
# to the mask + torch.topk + map reference path for A/B testing.
_hisa_fused_topk = _env_bool("SGLANG_NSA_HISA_FUSED_TOPK", True)


_hisa_profile_path = os.environ.get(
    "SGLANG_NSA_NVFP4_HISA_PROFILE_PATH"
) or os.environ.get("SGLANG_NSA_HISA_PROFILE_PATH")
_hisa_profile_sync = os.environ.get(
    "SGLANG_NSA_NVFP4_HISA_PROFILE_SYNC", ""
).lower() in ("1", "true", "yes", "on")
_hisa_profile_lock = threading.Lock()
_NVFP4_E2M1_CODEBOOK = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# DeepGEMM splits the head count into two TMEM loads of kNumHeads / 2.
# The current Blackwell FP4 MQA kernels only support TMEM load widths 32/64,
# so 32 heads would instantiate an unsupported 16-wide load and JIT-fail.
_DEEPGEMM_FP4_MQA_HEAD_COUNTS = (64,)


def _deepgemm_fp4_mqa_supports(q_values: torch.Tensor) -> bool:
    return (
        q_values.dim() == 3
        and q_values.shape[1] in _DEEPGEMM_FP4_MQA_HEAD_COUNTS
        and q_values.shape[-1] == 64
    )


@cache_once
def _jit_nvfp4_indexer_module(
    key_dtype: torch.dtype, indices_dtype: torch.dtype, page_size: int
) -> Module:
    with _nvfp4_arch_env():
        args = make_cpp_args(
            key_dtype, indices_dtype, page_size, is_arch_support_pdl()
        )
        return load_jit(
            "nvfp4_indexer_quant",
            *args,
            cuda_files=["nsa/nvfp4_indexer_quant.cuh"],
            cuda_wrappers=[
                (
                    "fused_store_index_k_cache_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::store_index_k",
                ),
                (
                    "quantize_indexer_q_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::quantize_q",
                ),
                (
                    "hisa_mean_pool_indexer_cache_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::hisa_mean_pool",
                ),
                (
                    "hisa_candidate_score_indexer_cache_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::hisa_candidate_score",
                ),
                (
                    "hisa_block_score_indexer_cache_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::hisa_block_score",
                ),
                (
                    "hisa_block_topk_indexer_cache_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::hisa_block_topk",
                ),
                (
                    "hisa_block_topk_map_all_indexer_cache_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::hisa_block_topk_map_all",
                ),
                (
                    "hisa_candidate_pages_indexer_cache_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::hisa_candidate_pages",
                ),
                (
                    "hisa_mask_candidate_logits_indexer_cache_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::hisa_mask_candidate_logits",
                ),
                (
                    "hisa_mask_logits_indexer_cache_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::hisa_mask_logits",
                ),
                (
                    "hisa_map_topk_indices_indexer_cache_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::hisa_map_topk_indices",
                ),
                (
                    "hisa_map_candidate_indices_indexer_cache_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::hisa_map_candidate_indices",
                ),
                (
                    "hisa_fused_mask_topk_map_indexer_cache_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::hisa_fused_mask_topk_map",
                ),
            ],
            extra_cuda_cflags=_nvfp4_cuda_flags(),
            extra_dependencies=["cutlass"],
        )


@cache_once
def can_use_nsa_nvfp4_indexer(
    key_dtype: torch.dtype, indices_dtype: torch.dtype, page_size: int
) -> bool:
    try:
        _jit_nvfp4_indexer_module(key_dtype, indices_dtype, page_size)
        return True
    except Exception as e:
        logger.warning("Failed to load NSA NVFP4 indexer JIT kernel: %s", e)
        return False


def _get_module_fast(key_dtype, indices_dtype, page_size):
    global _cached_key, _cached_module
    cache_key = (key_dtype, indices_dtype, page_size)
    if _cached_key == cache_key:
        return _cached_module
    module = _jit_nvfp4_indexer_module(key_dtype, indices_dtype, page_size)
    _cached_key = cache_key
    _cached_module = module
    return module


@debug_kernel_api
def fused_store_index_k_cache_nvfp4(
    key: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    out_cache_loc: torch.Tensor,
    page_size: int = 64,
) -> None:
    if key.dim() != 2:
        key = key.reshape(-1, key.shape[-1])
    if not key.is_contiguous():
        key = key.contiguous()
    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()

    _get_module_fast(
        key.dtype, out_cache_loc.dtype, page_size
    ).fused_store_index_k_cache_nvfp4(key, index_k_with_scale, out_cache_loc)


@debug_kernel_api
def quantize_indexer_q_nvfp4(
    query: torch.Tensor, indices_dtype: torch.dtype = torch.int64, page_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = query.shape
    if query.shape[-1] != 128:
        raise ValueError(f"NSA NVFP4 indexer Q expects head_dim=128, got {query.shape}.")
    query = query.reshape(-1, original_shape[-1])
    if not query.is_contiguous():
        query = query.contiguous()

    values = torch.empty(
        (query.shape[0], original_shape[-1] // 2),
        dtype=torch.uint8,
        device=query.device,
    )
    scales = torch.empty((query.shape[0],), dtype=torch.int32, device=query.device)
    _get_module_fast(query.dtype, indices_dtype, page_size).quantize_indexer_q_nvfp4(
        query, values, scales
    )
    return (
        values.view(*original_shape[:-1], original_shape[-1] // 2),
        scales.view(*original_shape[:-1]),
    )


@debug_kernel_api
def hisa_mean_pool_indexer_cache_nvfp4(
    index_k_with_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_blocks: int,
    page_size: int = 64,
) -> torch.Tensor:
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    reps = torch.empty(
        (page_table.shape[0], max_blocks, 128),
        dtype=torch.float32,
        device=index_k_with_scale.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_mean_pool_indexer_cache_nvfp4(
        index_k_with_scale, page_table, seq_lens.to(torch.int32), reps
    )
    return reps


@debug_kernel_api
def hisa_candidate_score_indexer_cache_nvfp4(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    top_blocks: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weights.dtype != torch.float32:
        weights = weights.float()
    if q_values.dtype != torch.uint8:
        raise ValueError(f"NVFP4 HISA q values must be uint8, got {q_values.dtype}.")
    if q_scales.dtype != torch.int32:
        q_scales = q_scales.to(torch.int32)
    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    weights = weights.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not top_blocks.is_contiguous():
        top_blocks = top_blocks.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()

    candidate_len = top_blocks.shape[1] * 128
    logits = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.float32, device=q_values.device
    )
    candidate_indices = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.int32, device=q_values.device
    )
    if q_values.shape[1] > 8:
        logits.zero_()
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_candidate_score_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale,
        page_table,
        seq_lens.to(torch.int32),
        weights,
        top_blocks.to(torch.int32),
        token_to_batch_idx.to(torch.int32),
        logits,
        candidate_indices,
    )
    return logits, candidate_indices


@debug_kernel_api
def hisa_block_score_indexer_cache_nvfp4(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    reps: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    if weights.dtype != torch.float32:
        weights = weights.float()
    if q_values.dtype != torch.uint8:
        raise ValueError(f"NVFP4 HISA q values must be uint8, got {q_values.dtype}.")
    if q_scales.dtype != torch.int32:
        q_scales = q_scales.to(torch.int32)
    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    reps = reps.contiguous()
    weights = weights.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()
    block_scores = torch.empty(
        (q_values.shape[0], reps.shape[1]), dtype=torch.float32, device=q_values.device
    )
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_block_score_indexer_cache_nvfp4(
        q_values,
        q_scales,
        reps,
        weights,
        seq_lens.to(torch.int32),
        token_to_batch_idx.to(torch.int32),
        block_scores,
    )
    return block_scores


@debug_kernel_api
def hisa_block_topk_indexer_cache_nvfp4(
    block_scores: torch.Tensor,
    block_counts: torch.Tensor,
    block_topk: int,
    block_topk_counts: Optional[torch.Tensor] = None,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    block_scores = block_scores.contiguous()
    if block_counts.dtype != torch.int32:
        block_counts = block_counts.to(torch.int32)
    block_counts = block_counts.contiguous()
    if block_topk_counts is None:
        block_topk_counts = torch.full_like(block_counts, block_topk)
    elif block_topk_counts.dtype != torch.int32:
        block_topk_counts = block_topk_counts.to(torch.int32)
    block_topk_counts = block_topk_counts.contiguous()
    top_blocks = torch.empty(
        (block_scores.shape[0], block_topk),
        dtype=torch.int32,
        device=block_scores.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_block_topk_indexer_cache_nvfp4(
        block_scores, block_counts, block_topk_counts, top_blocks
    )
    return top_blocks


@debug_kernel_api
def hisa_block_topk_map_all_indexer_cache_nvfp4(
    block_scores: torch.Tensor,
    block_counts: torch.Tensor,
    block_topk: int,
    prefix_lens: torch.Tensor,
    block_topk_counts: Optional[torch.Tensor] = None,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    block_scores = block_scores.contiguous()
    if block_counts.dtype != torch.int32:
        block_counts = block_counts.to(torch.int32)
    block_counts = block_counts.contiguous()
    if block_topk_counts is None:
        block_topk_counts = torch.full_like(block_counts, block_topk)
    elif block_topk_counts.dtype != torch.int32:
        block_topk_counts = block_topk_counts.to(torch.int32)
    block_topk_counts = block_topk_counts.contiguous()
    if prefix_lens.dtype != torch.int32:
        prefix_lens = prefix_lens.to(torch.int32)
    prefix_lens = prefix_lens.contiguous()
    topk_indices = torch.empty(
        (block_scores.shape[0], block_topk * 128),
        dtype=torch.int32,
        device=block_scores.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_block_topk_map_all_indexer_cache_nvfp4(
        block_scores, block_counts, block_topk_counts, prefix_lens, topk_indices
    )
    return topk_indices


@debug_kernel_api
def hisa_candidate_pages_indexer_cache_nvfp4(
    top_blocks: torch.Tensor,
    page_table: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    page_size: int = 64,
) -> torch.Tensor:
    if top_blocks.dtype != torch.int32:
        top_blocks = top_blocks.to(torch.int32)
    if token_to_batch_idx.dtype != torch.int32:
        token_to_batch_idx = token_to_batch_idx.to(torch.int32)
    top_blocks = top_blocks.contiguous()
    page_table = page_table.contiguous()
    token_to_batch_idx = token_to_batch_idx.contiguous()
    candidate_page_table = torch.empty(
        (top_blocks.shape[0], top_blocks.shape[1] * 2),
        dtype=page_table.dtype,
        device=page_table.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_candidate_pages_indexer_cache_nvfp4(
        top_blocks, page_table, token_to_batch_idx, candidate_page_table
    )
    return candidate_page_table


@debug_kernel_api
def hisa_mask_candidate_logits_indexer_cache_nvfp4(
    logits: torch.Tensor,
    top_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    if not logits.is_contiguous():
        logits = logits.contiguous()
    if top_blocks.dtype != torch.int32:
        top_blocks = top_blocks.to(torch.int32)
    if prefix_lens.dtype != torch.int32:
        prefix_lens = prefix_lens.to(torch.int32)
    top_blocks = top_blocks.contiguous()
    prefix_lens = prefix_lens.contiguous()
    candidate_indices = torch.empty_like(logits, dtype=torch.int32)
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_mask_candidate_logits_indexer_cache_nvfp4(
        logits, top_blocks, prefix_lens, candidate_indices
    )
    return candidate_indices


@debug_kernel_api
def hisa_mask_logits_indexer_cache_nvfp4(
    logits: torch.Tensor,
    top_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> None:
    if not logits.is_contiguous():
        raise ValueError("NVFP4 HISA logits mask requires a contiguous logits tensor.")
    if top_blocks.dtype != torch.int32:
        top_blocks = top_blocks.to(torch.int32)
    if prefix_lens.dtype != torch.int32:
        prefix_lens = prefix_lens.to(torch.int32)
    top_blocks = top_blocks.contiguous()
    prefix_lens = prefix_lens.contiguous()
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_mask_logits_indexer_cache_nvfp4(logits, top_blocks, prefix_lens)


@debug_kernel_api
def hisa_map_topk_indices_indexer_cache_nvfp4(
    relevant_topk_indices: torch.Tensor,
    top_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    if relevant_topk_indices.dtype != torch.int64:
        relevant_topk_indices = relevant_topk_indices.to(torch.int64)
    if top_blocks.dtype != torch.int32:
        top_blocks = top_blocks.to(torch.int32)
    if prefix_lens.dtype != torch.int32:
        prefix_lens = prefix_lens.to(torch.int32)
    relevant_topk_indices = relevant_topk_indices.contiguous()
    top_blocks = top_blocks.contiguous()
    prefix_lens = prefix_lens.contiguous()
    topk_indices = torch.empty(
        relevant_topk_indices.shape,
        dtype=torch.int32,
        device=relevant_topk_indices.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_map_topk_indices_indexer_cache_nvfp4(
        relevant_topk_indices, top_blocks, prefix_lens, topk_indices
    )
    return topk_indices


@debug_kernel_api
def hisa_fused_mask_topk_map_indexer_cache_nvfp4(
    logits: torch.Tensor,
    top_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    topk_tokens: int,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    """Fused exact mask + radix top-k + position-to-token map."""
    if not logits.is_contiguous():
        raise ValueError("NVFP4 HISA fused mask+topk+map requires contiguous logits.")
    if top_blocks.dtype != torch.int32:
        top_blocks = top_blocks.to(torch.int32)
    if prefix_lens.dtype != torch.int32:
        prefix_lens = prefix_lens.to(torch.int32)
    top_blocks = top_blocks.contiguous()
    prefix_lens = prefix_lens.contiguous()
    topk_indices = torch.empty(
        (logits.shape[0], topk_tokens),
        dtype=torch.int32,
        device=logits.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_fused_mask_topk_map_indexer_cache_nvfp4(
        logits, top_blocks, prefix_lens, topk_indices
    )
    return topk_indices


@debug_kernel_api
def hisa_map_candidate_indices_indexer_cache_nvfp4(
    top_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    topk_tokens: int,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    if top_blocks.dtype != torch.int32:
        top_blocks = top_blocks.to(torch.int32)
    if prefix_lens.dtype != torch.int32:
        prefix_lens = prefix_lens.to(torch.int32)
    top_blocks = top_blocks.contiguous()
    prefix_lens = prefix_lens.contiguous()
    topk_indices = torch.empty(
        (top_blocks.shape[0], topk_tokens),
        dtype=torch.int32,
        device=top_blocks.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_map_candidate_indices_indexer_cache_nvfp4(
        top_blocks, prefix_lens, topk_indices
    )
    return topk_indices


def hisa_precompute_block_reps_indexer_cache_nvfp4(
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int = 128,
    page_size: int = 64,
    return_float_reps: bool = False,
) -> tuple[tuple[torch.Tensor, torch.Tensor], int] | tuple[tuple[torch.Tensor, torch.Tensor], int, torch.Tensor]:
    if block_size != 128:
        raise ValueError("NVFP4 HISA precomputed reps require block_size=128.")
    seq_lens_flat = seq_lens.reshape(-1).to(device=page_table.device, dtype=torch.int32)
    max_blocks = int(((seq_lens_flat.max().item() + block_size - 1) // block_size))
    reps = hisa_mean_pool_indexer_cache_nvfp4(
        index_k_with_scale_buffer,
        page_table,
        seq_lens_flat,
        max_blocks,
        page_size=page_size,
    )
    rep_values, rep_scales = quantize_indexer_q_nvfp4(
        reps.reshape(-1, reps.shape[-1]).to(torch.bfloat16),
        indices_dtype=page_table.dtype,
        page_size=page_size,
    )
    rep_fp4 = (rep_values, rep_scales)
    if return_float_reps:
        return rep_fp4, max_blocks, reps
    return rep_fp4, max_blocks


def _profile_start(device: torch.device) -> Optional[float]:
    if not _hisa_profile_path:
        return None
    if _hisa_profile_sync and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _profile_end(
    stage: str, start: Optional[float], device: torch.device, **record
) -> None:
    if start is None or not _hisa_profile_path:
        return
    if _hisa_profile_sync and device.type == "cuda":
        torch.cuda.synchronize(device)
    payload = {
        "time": time.time(),
        "path": f"hisa_nvfp4:{stage}",
        "duration_ms": (time.perf_counter() - start) * 1000.0,
        **record,
    }
    directory = os.path.dirname(_hisa_profile_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with _hisa_profile_lock:
        with open(_hisa_profile_path, "a") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


def dequantize_indexer_nvfp4(
    values: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Dequantize IndexCache NVFP4/UE8M0 rows to fp32."""

    if values.shape[-1] != 64:
        raise ValueError(
            f"NVFP4 IndexCache values expect 64 packed bytes, got {values.shape}."
        )
    values_u8 = values.to(torch.uint8)
    low = values_u8 & 0x0F
    high = (values_u8 >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(*values.shape[:-1], 128)

    codebook = torch.tensor(
        _NVFP4_E2M1_CODEBOOK, device=values.device, dtype=torch.float32
    )
    magnitudes = codebook[(codes & 0x07).long()]
    signed = torch.where((codes & 0x08).bool(), -magnitudes, magnitudes)

    scale_words = scales.to(torch.int64) & 0xFFFFFFFF
    exponents = torch.stack(
        tuple(((scale_words >> shift) & 0xFF) for shift in (0, 8, 16, 24)),
        dim=-1,
    ).to(torch.float32)
    group_scales = torch.pow(torch.full_like(exponents, 2.0), exponents - 127.0)
    return signed * torch.repeat_interleave(group_scales, 32, dim=-1)


def _unpack_paged_nvfp4_cache(
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int = 64,
) -> list[torch.Tensor]:
    rows_by_batch = []
    for batch_idx in range(page_table.shape[0]):
        seq_len = int(seq_lens.reshape(-1)[batch_idx].item())
        num_pages = (seq_len + page_size - 1) // page_size
        pages = page_table[batch_idx, :num_pages].long().clamp_min(0)
        packed_pages = index_k_with_scale_buffer.index_select(0, pages)
        values = packed_pages[:, : page_size * 64].reshape(-1, 64)[:seq_len]
        scale_bytes = packed_pages[:, page_size * 64 : page_size * 68]
        scales = scale_bytes.reshape(-1, 4)[:seq_len].contiguous()
        scales = scales.view(torch.int32).reshape(-1)
        rows_by_batch.append(dequantize_indexer_nvfp4(values, scales))
    return rows_by_batch


def _mean_pool_blocks(k_rows: torch.Tensor, block_size: int) -> torch.Tensor:
    block_count = (k_rows.shape[0] + block_size - 1) // block_size
    reps = []
    for block_id in range(block_count):
        start = block_id * block_size
        end = min(start + block_size, k_rows.shape[0])
        reps.append(k_rows[start:end].mean(dim=0))
    if not reps:
        return k_rows.new_zeros((0, k_rows.shape[-1]))
    return torch.stack(reps, dim=0)


def _weighted_relu_dsa_score(
    q_rows: torch.Tensor, k_rows: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    logits = torch.einsum("qhd,kd->qkh", q_rows.float(), k_rows.float())
    logits = torch.relu(logits) * weights.float().unsqueeze(1)
    return logits.sum(dim=-1)


def _forced_hisa_blocks(block_count: int, device: torch.device) -> torch.Tensor:
    if block_count <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    forced = [0]
    forced.append(block_count - 1)
    return torch.tensor(sorted(set(forced)), device=device, dtype=torch.long)


def _hisa_block_topk_counts(
    block_counts: torch.Tensor,
    *,
    block_size: int,
    topk_tokens: int,
    compression_ratio: Optional[float],
) -> tuple[torch.Tensor, int]:
    if compression_ratio is None or compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive for dynamic HISA budgets.")
    if abs(compression_ratio - round(compression_ratio)) < 1e-6:
        ratio = int(round(compression_ratio))
        selected = torch.div(block_counts + ratio - 1, ratio, rounding_mode="floor")
    else:
        selected = torch.ceil(block_counts.float() / compression_ratio).to(torch.int32)
    selected = torch.minimum(selected, block_counts)
    selected = torch.where(block_counts > 0, selected, torch.zeros_like(selected))
    max_selected = int(selected.max().item()) if selected.numel() else 0
    return selected.to(torch.int32), max(1, max_selected)


def _select_hisa_blocks(
    block_scores: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    block_topk: int,
    block_topk_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    selected = torch.full(
        (block_scores.shape[0], block_topk),
        -1,
        device=block_scores.device,
        dtype=torch.int32,
    )
    for row in range(block_scores.shape[0]):
        block_count = int(block_counts[row].item())
        if block_count == 0:
            continue
        scores = block_scores[row, :block_count].clone()
        scores[_forced_hisa_blocks(block_count, block_scores.device)] = float("inf")
        row_topk = (
            int(block_topk_counts[row].item())
            if block_topk_counts is not None
            else block_topk
        )
        keep = min(row_topk, block_topk, block_count)
        selected[row, :keep] = torch.topk(scores, k=keep, sorted=False).indices.to(
            torch.int32
        )
    return selected


def _select_hisa_blocks_vectorized(
    block_scores: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    block_topk: int,
    block_topk_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if block_scores.shape[1] == 0:
        return torch.full(
            (block_scores.shape[0], block_topk),
            -1,
            device=block_scores.device,
            dtype=torch.int32,
        )
    scores = block_scores.clone()
    rows = torch.arange(scores.shape[0], device=scores.device)
    valid = block_counts > 0
    if bool(valid.any().item()):
        valid_rows = rows[valid]
        counts = block_counts[valid].long()
        scores[valid_rows, 0] = float("inf")
        scores[valid_rows, counts - 1] = float("inf")
    if block_topk_counts is not None and bool((block_topk_counts != block_topk).any().item()):
        return _select_hisa_blocks(
            block_scores,
            block_counts,
            block_topk=block_topk,
            block_topk_counts=block_topk_counts,
        )
    keep = min(block_topk, scores.shape[1])
    top_blocks = torch.topk(scores, k=keep, dim=-1, sorted=False).indices.to(torch.int32)
    if keep == block_topk:
        return top_blocks
    selected = torch.full(
        (scores.shape[0], block_topk),
        -1,
        device=scores.device,
        dtype=torch.int32,
    )
    selected[:, :keep] = top_blocks
    return selected


def nvfp4_hisa_indexer_from_dequant(
    q: torch.Tensor,
    k_by_batch: list[torch.Tensor],
    weights: torch.Tensor,
    prefix_lens: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    *,
    block_size: int = 128,
    block_topk: int = 64,
    compression_ratio: Optional[float] = 4.0,
    topk_tokens: int = 2048,
    fallback_to_dense_if_short: bool = True,
) -> Optional[torch.Tensor]:
    """Reference HISA selection over already-dequantized NVFP4 rows."""

    prefix_lens = prefix_lens.reshape(-1).to(device=q.device, dtype=torch.int64)
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q.device, dtype=torch.long
    )
    if fallback_to_dense_if_short and bool(torch.all(prefix_lens <= topk_tokens).item()):
        return None

    q = q.float()
    weights = weights.float()
    block_counts_by_batch = [
        (rows.shape[0] + block_size - 1) // block_size for rows in k_by_batch
    ]
    max_blocks = max(block_counts_by_batch, default=0)

    stage = _profile_start(q.device)
    reps_by_batch = [_mean_pool_blocks(rows.to(q.device), block_size) for rows in k_by_batch]
    _profile_end("mean_pool", stage, q.device, batches=len(k_by_batch), max_blocks=max_blocks)

    block_scores = q.new_full((q.shape[0], max_blocks), float("-inf"))
    block_counts = torch.empty((q.shape[0],), device=q.device, dtype=torch.int32)
    stage = _profile_start(q.device)
    for batch_idx, reps in enumerate(reps_by_batch):
        row_mask = token_to_batch_idx == batch_idx
        if not bool(row_mask.any().item()) or reps.numel() == 0:
            continue
        rows = row_mask.nonzero(as_tuple=False).flatten()
        scores = _weighted_relu_dsa_score(
            q.index_select(0, rows), reps, weights.index_select(0, rows)
        )
        block_scores[rows, : scores.shape[1]] = scores
        block_counts[rows] = scores.shape[1]
    _profile_end("block_score", stage, q.device, rows=int(q.shape[0]), max_blocks=max_blocks)

    stage = _profile_start(q.device)
    if compression_ratio is None or compression_ratio <= 0:
        block_topk_counts = None
        effective_block_topk = block_topk
    else:
        block_topk_counts, effective_block_topk = _hisa_block_topk_counts(
            block_counts,
            block_size=block_size,
            topk_tokens=topk_tokens,
            compression_ratio=compression_ratio,
        )
    top_blocks = _select_hisa_blocks(
        block_scores,
        block_counts,
        block_topk=effective_block_topk,
        block_topk_counts=block_topk_counts,
    )
    _profile_end(
        "block_topk",
        stage,
        q.device,
        rows=int(q.shape[0]),
        block_topk=effective_block_topk,
        compression_ratio=compression_ratio,
    )

    out = torch.full((q.shape[0], topk_tokens), -1, device=q.device, dtype=torch.int32)
    for row in range(q.shape[0]):
        prefix_len = int(prefix_lens[row].item())
        if fallback_to_dense_if_short and prefix_len <= topk_tokens:
            continue
        batch_idx = int(token_to_batch_idx[row].item())
        k_rows = k_by_batch[batch_idx].to(q.device)
        valid_blocks = top_blocks[row][top_blocks[row] >= 0].long()
        if valid_blocks.numel() == 0:
            continue

        stage = _profile_start(q.device)
        candidate_ranges = []
        for block_id in valid_blocks.tolist():
            start = block_id * block_size
            end = min(start + block_size, prefix_len)
            if start < end:
                candidate_ranges.append(torch.arange(start, end, device=q.device))
        if not candidate_ranges:
            _profile_end("candidate_dequant", stage, q.device, row=row, candidates=0)
            continue
        candidate_indices = torch.cat(candidate_ranges).unique(sorted=False)
        candidate_k = k_rows.index_select(0, candidate_indices)
        _profile_end(
            "candidate_dequant",
            stage,
            q.device,
            row=row,
            candidates=int(candidate_indices.numel()),
        )

        stage = _profile_start(q.device)
        candidate_scores = _weighted_relu_dsa_score(
            q[row : row + 1], candidate_k, weights[row : row + 1]
        ).squeeze(0)
        _profile_end(
            "candidate_score",
            stage,
            q.device,
            row=row,
            candidates=int(candidate_indices.numel()),
        )

        stage = _profile_start(q.device)
        keep = min(topk_tokens, candidate_scores.numel())
        rel = torch.topk(candidate_scores, k=keep, sorted=False).indices
        out[row, :keep] = candidate_indices.index_select(0, rel).to(torch.int32)
        _profile_end("candidate_topk", stage, q.device, row=row, topk=int(keep))

    stage = _profile_start(q.device)
    _profile_end("map/store", stage, q.device, rows=int(q.shape[0]), topk_tokens=topk_tokens)
    return out


def nvfp4_hisa_indexer_paged_torch(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    *,
    block_size: int = 128,
    block_topk: int = 64,
    compression_ratio: Optional[float] = 4.0,
    topk_tokens: int = 2048,
    fallback_to_dense_if_short: bool = True,
) -> Optional[torch.Tensor]:
    q_values, q_scales = q_fp4
    if q_values.dim() != 3 or q_values.shape[-1] != 64:
        raise ValueError(
            f"NVFP4 HISA q values expect [tokens, heads, 64], got {q_values.shape}."
        )
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.long
    )
    seq_lens_flat = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    prefix_lens = seq_lens_flat.index_select(0, token_to_batch_idx)
    if fallback_to_dense_if_short and bool(torch.all(prefix_lens <= topk_tokens).item()):
        return None

    max_blocks = int(((seq_lens_flat.max().item() + block_size - 1) // block_size))

    stage = _profile_start(q_values.device)
    reps = hisa_mean_pool_indexer_cache_nvfp4(
        index_k_with_scale_buffer,
        page_table,
        seq_lens_flat,
        max_blocks,
    )
    _profile_end(
        "mean_pool",
        stage,
        q_values.device,
        batches=int(page_table.shape[0]),
        max_blocks=max_blocks,
        fused_cuda=True,
    )

    block_counts = torch.div(
        prefix_lens.to(torch.int32) + block_size - 1,
        block_size,
        rounding_mode="floor",
    ).to(torch.int32)
    stage = _profile_start(q_values.device)
    block_scores = hisa_block_score_indexer_cache_nvfp4(
        q_values,
        q_scales,
        reps,
        weights,
        seq_lens_flat,
        token_to_batch_idx.to(torch.int32),
        page_table_dtype=page_table.dtype,
    )
    _profile_end(
        "block_score",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        max_blocks=max_blocks,
        fused_cuda=True,
        packed_q=True,
    )

    stage = _profile_start(q_values.device)
    if compression_ratio is None or compression_ratio <= 0:
        block_topk_counts = None
        effective_block_topk = block_topk
    else:
        block_topk_counts, effective_block_topk = _hisa_block_topk_counts(
            block_counts,
            block_size=block_size,
            topk_tokens=topk_tokens,
            compression_ratio=compression_ratio,
        )
    candidate_len = effective_block_topk * block_size
    if candidate_len == topk_tokens:
        topk_indices = hisa_block_topk_map_all_indexer_cache_nvfp4(
            block_scores,
            block_counts,
            block_topk=effective_block_topk,
            block_topk_counts=block_topk_counts,
            prefix_lens=prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "block_topk_map_all",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            block_topk=effective_block_topk,
            compression_ratio=compression_ratio,
            fused_cuda=True,
        )
        return topk_indices

    top_blocks = hisa_block_topk_indexer_cache_nvfp4(
        block_scores,
        block_counts,
        block_topk=effective_block_topk,
        block_topk_counts=block_topk_counts,
        page_table_dtype=page_table.dtype,
    )
    _profile_end(
        "block_topk",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        block_topk=effective_block_topk,
        compression_ratio=compression_ratio,
        fused_cuda=True,
    )

    stage = _profile_start(q_values.device)
    candidate_logits, candidate_indices = hisa_candidate_score_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale_buffer,
        page_table,
        seq_lens_flat,
        weights,
        top_blocks,
        token_to_batch_idx.to(torch.int32),
    )
    _profile_end(
        "candidate_score",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=int(candidate_logits.shape[-1]),
        fused_dequant=True,
        packed_q=True,
    )
    stage = _profile_start(q_values.device)
    _profile_end(
        "candidate_dequant",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=int(candidate_logits.shape[-1]),
        fused_into="candidate_score",
    )

    stage = _profile_start(q_values.device)
    keep = min(topk_tokens, candidate_logits.shape[-1])
    relevant_topk_indices = torch.topk(
        candidate_logits, k=keep, dim=-1, sorted=False
    ).indices
    topk_indices = torch.gather(candidate_indices, dim=1, index=relevant_topk_indices)
    _profile_end(
        "candidate_topk",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=int(candidate_logits.shape[-1]),
        topk_tokens=int(keep),
    )

    if keep < topk_tokens:
        padding = torch.full(
            (topk_indices.shape[0], topk_tokens - keep),
            -1,
            device=topk_indices.device,
            dtype=torch.int32,
        )
        topk_indices = torch.cat((topk_indices, padding), dim=1)
    stage = _profile_start(q_values.device)
    _profile_end("map/store", stage, q_values.device, rows=int(q_values.shape[0]), topk_tokens=topk_tokens)
    return topk_indices.to(torch.int32)


def nvfp4_hisa_indexer_paged_deepgemm_precomputed(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    block_rep_fp4: tuple[torch.Tensor, torch.Tensor],
    max_blocks: int,
    *,
    block_size: int = 128,
    block_topk: int = 64,
    compression_ratio: Optional[float] = 4.0,
    topk_tokens: int = 2048,
    fallback_to_dense_if_short: bool = True,
    prepared_prefix_lens: Optional[torch.Tensor] = None,
    prepared_block_counts: Optional[torch.Tensor] = None,
    prepared_block_starts: Optional[torch.Tensor] = None,
    prepared_block_ends: Optional[torch.Tensor] = None,
    prepared_candidate_context_lens: Optional[torch.Tensor] = None,
    prepared_candidate_schedule_metadata: Optional[object] = None,
    prepared_block_topk_counts: Optional[torch.Tensor] = None,
    prepared_effective_block_topk: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if block_size != 128:
        raise ValueError("DeepGEMM-backed NVFP4 HISA path requires block_size=128.")

    q_values, q_scales = q_fp4
    rep_values, rep_scales = block_rep_fp4
    if not _deepgemm_fp4_mqa_supports(q_values):
        return None
    import deep_gemm

    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    seq_lens_flat = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    if prepared_prefix_lens is None:
        prefix_lens = seq_lens_flat.index_select(0, token_to_batch_idx.long())
    else:
        prefix_lens = prepared_prefix_lens.to(device=q_values.device, dtype=torch.int32)
    if fallback_to_dense_if_short and bool(torch.all(prefix_lens <= topk_tokens).item()):
        return None

    if prepared_block_counts is None:
        block_counts = torch.div(
            prefix_lens.to(torch.int32) + block_size - 1,
            block_size,
            rounding_mode="floor",
        ).to(torch.int32)
    else:
        block_counts = prepared_block_counts.to(device=q_values.device, dtype=torch.int32)
    if prepared_block_starts is None:
        block_starts = token_to_batch_idx * max_blocks
    else:
        block_starts = prepared_block_starts.to(device=q_values.device, dtype=torch.int32)
    if prepared_block_ends is None:
        block_ends = block_starts + block_counts
    else:
        block_ends = prepared_block_ends.to(device=q_values.device, dtype=torch.int32)
    stage = _profile_start(q_values.device)
    block_scores = deep_gemm.fp8_fp4_mqa_logits(
        (q_values.view(torch.int8), q_scales),
        (rep_values.view(torch.int8), rep_scales),
        weights,
        block_starts.to(torch.int32),
        block_ends.to(torch.int32),
        clean_logits=False,
        max_seqlen_k=max_blocks,
        logits_dtype=torch.float32,
    )
    _profile_end(
        "blockscore_precomputed",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        max_blocks=max_blocks,
        backend="deepgemm_fp4_mqa",
        packed_q=True,
    )

    stage = _profile_start(q_values.device)
    if prepared_block_topk_counts is not None and prepared_effective_block_topk is not None:
        # Hot-path fast path: caller already computed per-row counts AND the
        # max on host side, so we avoid the .item() sync inside this function.
        block_topk_counts = prepared_block_topk_counts.to(
            device=q_values.device, dtype=torch.int32
        )
        effective_block_topk = int(prepared_effective_block_topk)
    elif compression_ratio is None or compression_ratio <= 0:
        block_topk_counts = None
        effective_block_topk = block_topk
    else:
        block_topk_counts, effective_block_topk = _hisa_block_topk_counts(
            block_counts,
            block_size=block_size,
            topk_tokens=topk_tokens,
            compression_ratio=compression_ratio,
        )
    candidate_len = effective_block_topk * block_size
    if candidate_len == topk_tokens:
        topk_indices = hisa_block_topk_map_all_indexer_cache_nvfp4(
            block_scores,
            block_counts,
            block_topk=effective_block_topk,
            block_topk_counts=block_topk_counts,
            prefix_lens=prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "block_topk_map_all",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            block_topk=effective_block_topk,
            compression_ratio=compression_ratio,
            fused_cuda=True,
        )
        return topk_indices

    top_blocks = hisa_block_topk_indexer_cache_nvfp4(
        block_scores,
        block_counts,
        block_topk=effective_block_topk,
        block_topk_counts=block_topk_counts,
        page_table_dtype=page_table.dtype,
    )
    _profile_end(
        "block_topk",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        block_topk=effective_block_topk,
        compression_ratio=compression_ratio,
        fused_cuda=True,
    )

    stage = _profile_start(q_values.device)
    candidate_page_table = hisa_candidate_pages_indexer_cache_nvfp4(
        top_blocks, page_table, token_to_batch_idx
    )
    _profile_end(
        "candidate_pages",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        block_topk=effective_block_topk,
        candidate_pages=int(candidate_page_table.shape[1]),
        fused_cuda=True,
    )

    if prepared_candidate_context_lens is None:
        context_lens = torch.full(
            (q_values.shape[0], 1),
            candidate_len,
            device=q_values.device,
            dtype=torch.int32,
        )
    else:
        context_lens = prepared_candidate_context_lens.to(
            device=q_values.device, dtype=torch.int32
        )
    schedule_metadata = prepared_candidate_schedule_metadata
    if schedule_metadata is None:
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, 64, deep_gemm.get_num_sms()
        )
    stage = _profile_start(q_values.device)
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(
        (q_values.view(torch.int8).unsqueeze(1), q_scales.unsqueeze(1)),
        index_k_with_scale_buffer.view(index_k_with_scale_buffer.shape[0], 64, 1, 68),
        weights,
        context_lens,
        candidate_page_table,
        schedule_metadata,
        candidate_len,
        clean_logits=False,
        logits_dtype=torch.float32,
    )
    _profile_end(
        "candidate_logits",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=candidate_len,
    )

    keep = min(topk_tokens, candidate_len)
    if _hisa_fused_topk and keep <= candidate_len and top_blocks.shape[1] <= 256:
        # Fused mask + radix-topk + token-id map in a single launch.
        stage = _profile_start(q_values.device)
        topk_indices = hisa_fused_mask_topk_map_indexer_cache_nvfp4(
            logits,
            top_blocks,
            prefix_lens.to(torch.int32),
            keep,
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "fused_mask_topk_map",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            topk_tokens=int(keep),
            fused_cuda=True,
        )
    else:
        stage = _profile_start(q_values.device)
        hisa_mask_logits_indexer_cache_nvfp4(
            logits,
            top_blocks,
            prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "mask_map",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            fused_cuda=True,
        )

        stage = _profile_start(q_values.device)
        relevant_topk_indices = torch.topk(logits, k=keep, dim=-1, sorted=False).indices
        topk_indices = hisa_map_topk_indices_indexer_cache_nvfp4(
            relevant_topk_indices,
            top_blocks,
            prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "final_topk_gather",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            topk_tokens=int(keep),
        )

    if keep < topk_tokens:
        padding = torch.full(
            (topk_indices.shape[0], topk_tokens - keep),
            -1,
            device=topk_indices.device,
            dtype=torch.int32,
        )
        topk_indices = torch.cat((topk_indices, padding), dim=1)
    stage = _profile_start(q_values.device)
    _profile_end(
        "map/store",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        topk_tokens=topk_tokens,
    )
    return topk_indices.to(torch.int32)


def nvfp4_hisa_indexer_paged_deepgemm(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    *,
    block_size: int = 128,
    block_topk: int = 64,
    compression_ratio: Optional[float] = 4.0,
    topk_tokens: int = 2048,
    fallback_to_dense_if_short: bool = True,
) -> Optional[torch.Tensor]:
    if block_size != 128:
        raise ValueError("DeepGEMM-backed NVFP4 HISA path requires block_size=128.")

    q_values, q_scales = q_fp4
    if not _deepgemm_fp4_mqa_supports(q_values):
        return None
    import deep_gemm

    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    seq_lens_flat = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    prefix_lens = seq_lens_flat.index_select(0, token_to_batch_idx.long())
    if fallback_to_dense_if_short and bool(torch.all(prefix_lens <= topk_tokens).item()):
        return None

    max_blocks = int(((seq_lens_flat.max().item() + block_size - 1) // block_size))
    stage = _profile_start(q_values.device)
    reps = hisa_mean_pool_indexer_cache_nvfp4(
        index_k_with_scale_buffer,
        page_table,
        seq_lens_flat,
        max_blocks,
    )
    _profile_end(
        "mean_pool",
        stage,
        q_values.device,
        batches=int(page_table.shape[0]),
        max_blocks=max_blocks,
        fused_cuda=True,
    )

    block_counts = torch.div(
        prefix_lens.to(torch.int32) + block_size - 1,
        block_size,
        rounding_mode="floor",
    ).to(torch.int32)
    stage = _profile_start(q_values.device)
    rep_values, rep_scales = quantize_indexer_q_nvfp4(
        reps.reshape(-1, reps.shape[-1]).to(torch.bfloat16),
        indices_dtype=page_table.dtype,
    )
    _profile_end(
        "block_rep_quant",
        stage,
        q_values.device,
        rows=int(rep_values.shape[0]),
    )

    block_starts = token_to_batch_idx * max_blocks
    block_ends = block_starts + block_counts
    stage = _profile_start(q_values.device)
    block_scores = deep_gemm.fp8_fp4_mqa_logits(
        (q_values.view(torch.int8), q_scales),
        (rep_values.view(torch.int8), rep_scales),
        weights,
        block_starts.to(torch.int32),
        block_ends.to(torch.int32),
        clean_logits=False,
        max_seqlen_k=max_blocks,
        logits_dtype=torch.float32,
    )
    _profile_end(
        "block_score",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        max_blocks=max_blocks,
        backend="deepgemm_fp4_mqa",
        packed_q=True,
    )

    stage = _profile_start(q_values.device)
    if compression_ratio is None or compression_ratio <= 0:
        block_topk_counts = None
        effective_block_topk = block_topk
    else:
        block_topk_counts, effective_block_topk = _hisa_block_topk_counts(
            block_counts,
            block_size=block_size,
            topk_tokens=topk_tokens,
            compression_ratio=compression_ratio,
        )
    candidate_len = effective_block_topk * block_size
    if candidate_len == topk_tokens:
        topk_indices = hisa_block_topk_map_all_indexer_cache_nvfp4(
            block_scores,
            block_counts,
            block_topk=effective_block_topk,
            block_topk_counts=block_topk_counts,
            prefix_lens=prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "block_topk_map_all",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            block_topk=effective_block_topk,
            compression_ratio=compression_ratio,
            fused_cuda=True,
        )
        return topk_indices

    top_blocks = hisa_block_topk_indexer_cache_nvfp4(
        block_scores,
        block_counts,
        block_topk=effective_block_topk,
        block_topk_counts=block_topk_counts,
        page_table_dtype=page_table.dtype,
    )
    _profile_end(
        "block_topk",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        block_topk=effective_block_topk,
        compression_ratio=compression_ratio,
        fused_cuda=True,
    )

    stage = _profile_start(q_values.device)
    candidate_page_table = hisa_candidate_pages_indexer_cache_nvfp4(
        top_blocks, page_table, token_to_batch_idx
    )
    _profile_end(
        "candidate_table",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        block_topk=effective_block_topk,
        candidate_pages=int(candidate_page_table.shape[1]),
        fused_cuda=True,
    )

    context_lens = torch.full(
        (q_values.shape[0], 1),
        candidate_len,
        device=q_values.device,
        dtype=torch.int32,
    )
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, 64, deep_gemm.get_num_sms()
    )
    stage = _profile_start(q_values.device)
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(
        (q_values.view(torch.int8).unsqueeze(1), q_scales.unsqueeze(1)),
        index_k_with_scale_buffer.view(index_k_with_scale_buffer.shape[0], 64, 1, 68),
        weights,
        context_lens,
        candidate_page_table,
        schedule_metadata,
        candidate_len,
        clean_logits=False,
        logits_dtype=torch.float32,
    )
    _profile_end(
        "candidate_deepgemm",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=candidate_len,
    )

    keep = min(topk_tokens, candidate_len)
    if _hisa_fused_topk and keep <= candidate_len and top_blocks.shape[1] <= 256:
        stage = _profile_start(q_values.device)
        topk_indices = hisa_fused_mask_topk_map_indexer_cache_nvfp4(
            logits,
            top_blocks,
            prefix_lens.to(torch.int32),
            keep,
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "fused_mask_topk_map",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            topk_tokens=int(keep),
            fused_cuda=True,
        )
    else:
        stage = _profile_start(q_values.device)
        hisa_mask_logits_indexer_cache_nvfp4(
            logits,
            top_blocks,
            prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "candidate_mask_map",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            fused_cuda=True,
        )

        stage = _profile_start(q_values.device)
        relevant_topk_indices = torch.topk(logits, k=keep, dim=-1, sorted=False).indices
        topk_indices = hisa_map_topk_indices_indexer_cache_nvfp4(
            relevant_topk_indices,
            top_blocks,
            prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "candidate_topk",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            topk_tokens=int(keep),
        )

    if keep < topk_tokens:
        padding = torch.full(
            (topk_indices.shape[0], topk_tokens - keep),
            -1,
            device=topk_indices.device,
            dtype=torch.int32,
        )
        topk_indices = torch.cat((topk_indices, padding), dim=1)
    stage = _profile_start(q_values.device)
    _profile_end(
        "map/store",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        topk_tokens=topk_tokens,
    )
    return topk_indices.to(torch.int32)
