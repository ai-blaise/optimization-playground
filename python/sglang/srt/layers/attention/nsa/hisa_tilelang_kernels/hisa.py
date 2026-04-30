import contextlib
import fcntl
import json
import os
import tempfile
import threading
import time
from typing import Optional

import torch

from sglang.srt.layers.attention.nsa.hisa_tilelang_kernels.block_sparse_mqa_fp8 import (
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
)
from sglang.srt.layers.attention.nsa.hisa_tilelang_kernels.clean_and_maintain_logits import (
    clean_and_maintain_logits_interface,
)
from sglang.srt.layers.attention.nsa.hisa_tilelang_kernels.fp8_block_mean_pooling import (
    fp8_native_block_mean_pooling_interface,
)
from sglang.srt.layers.attention.nsa.hisa_tilelang_kernels.pool_mqa_fp8 import (
    pool_mqa_attn_return_logits_fp8_interface,
)


_COMPILED_SHAPES = set()
_DEEPGEMM_METADATA_CACHE = {}
_HISA_STAGE_PROFILE_ENABLED = os.environ.get(
    "SGLANG_NSA_HISA_PROFILE_STAGES", ""
).lower() in ("1", "true", "yes", "on")
_HISA_STAGE_PROFILE_PATH = os.environ.get(
    "SGLANG_NSA_HISA_STAGE_PROFILE_PATH"
) or os.environ.get("SGLANG_NSA_HISA_PROFILE_PATH")
_HISA_STAGE_PROFILE_LOCK = threading.Lock()


def _stage_profile_start(device: torch.device) -> Optional[float]:
    if not _HISA_STAGE_PROFILE_ENABLED or not _HISA_STAGE_PROFILE_PATH:
        return None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _stage_profile_end(
    stage: str, start: Optional[float], device: torch.device, **record
) -> None:
    if start is None or not _HISA_STAGE_PROFILE_PATH:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    payload = {
        "time": time.time(),
        "pid": os.getpid(),
        "path": f"hisa_stage:{stage}",
        "duration_ms": (time.perf_counter() - start) * 1000.0,
        **record,
    }
    directory = os.path.dirname(_HISA_STAGE_PROFILE_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with _HISA_STAGE_PROFILE_LOCK:
        with open(_HISA_STAGE_PROFILE_PATH, "a") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


@contextlib.contextmanager
def _compile_lock():
    lock_path = os.environ.get(
        "SGLANG_HISA_TILELANG_COMPILE_LOCK",
        os.path.join(tempfile.gettempdir(), "sglang_hisa_tilelang_compile.lock"),
    )
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _pad_k_to_block(
    k: torch.Tensor, k_scale: torch.Tensor, k_block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    target_len = max(k_block_size, 1 << (k.shape[0] - 1).bit_length())
    remainder = target_len % k_block_size
    if remainder != 0:
        target_len += k_block_size - remainder
    if k.shape[0] == target_len:
        return k, k_scale

    pad_len = target_len - k.shape[0]
    k_padding = torch.zeros(
        (pad_len, k.shape[1]),
        dtype=k.dtype,
        device=k.device,
    )
    scale_padding = torch.zeros(
        (pad_len,),
        dtype=k_scale.dtype,
        device=k_scale.device,
    )
    return torch.cat((k, k_padding), dim=0), torch.cat(
        (k_scale, scale_padding),
        dim=0,
    )


def hisa_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    k_block_size: int,
    block_topk: int,
    topk_tokens: int,
    return_block_topk: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    k, k_scale = _pad_k_to_block(k, k_scale, k_block_size)
    shape_key = (
        tuple(q.shape),
        tuple(k.shape),
        k_block_size,
        block_topk,
        topk_tokens,
    )
    if shape_key not in _COMPILED_SHAPES:
        with _compile_lock():
            result = _hisa_indexer_impl(
                q,
                k,
                k_scale,
                weights,
                cu_seqlen_ks,
                cu_seqlen_ke,
                k_block_size=k_block_size,
                block_topk=block_topk,
                topk_tokens=topk_tokens,
                return_block_topk=return_block_topk,
            )
        _COMPILED_SHAPES.add(shape_key)
        return result

    return _hisa_indexer_impl(
        q,
        k,
        k_scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        k_block_size=k_block_size,
        block_topk=block_topk,
        topk_tokens=topk_tokens,
        return_block_topk=return_block_topk,
    )


def _hisa_indexer_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    k_block_size: int,
    block_topk: int,
    topk_tokens: int,
    return_block_topk: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    blocked_k_fp8, blocked_k_scale = fp8_native_block_mean_pooling_interface(
        k,
        k_scale,
        k_block_size,
    )

    cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
    cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size

    block_k_score = pool_mqa_attn_return_logits_fp8_interface(
        q,
        blocked_k_fp8,
        blocked_k_scale,
        weights,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
    )

    block_topk_eff = min(block_topk, block_k_score.shape[-1])
    topk_block_indices = _topk_block_indices(
        block_k_score,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
        block_topk_eff,
        k_block_size,
    )

    block_sparse_logits = fp8_native_block_sparse_mqa_attn_return_logits_interface(
        q,
        k,
        k_scale,
        topk_block_indices,
        k_block_size,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )

    relevant_topk_indices = _topk_token_indices(block_sparse_logits, topk_tokens)

    absolute_topk_block_indices = torch.gather(
        topk_block_indices,
        dim=-1,
        index=(relevant_topk_indices // k_block_size),
    )
    topk_indices = absolute_topk_block_indices * k_block_size + (
        relevant_topk_indices % k_block_size
    )
    topk_indices = _mask_topk_indices(topk_indices, cu_seqlen_ks, cu_seqlen_ke)
    if return_block_topk:
        return topk_indices, topk_block_indices
    return topk_indices


def hisa_indexer_paged(
    q: torch.Tensor,
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    *,
    k_block_size: int,
    block_topk: int,
    topk_tokens: int,
    block_metadata: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    return_block_topk: bool = False,
    topk_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if block_metadata is None:
        block_metadata = build_hisa_block_metadata(page_table, seq_lens, k_block_size)
    block_pages, block_token_counts, block_offsets = block_metadata
    prefix_lens = cu_seqlen_ke - cu_seqlen_ks
    cu_block_ks = block_offsets[token_to_batch_idx.long()].to(torch.int32)
    cu_block_ke = cu_block_ks + torch.div(
        prefix_lens + k_block_size - 1,
        k_block_size,
        rounding_mode="floor",
    ).to(torch.int32)

    shape_key = (
        "paged",
        tuple(q.shape),
        tuple(block_pages.shape),
        k_block_size,
        block_topk,
        topk_tokens,
    )
    if shape_key not in _COMPILED_SHAPES:
        with _compile_lock():
            result = _hisa_indexer_paged_impl(
                q,
                index_k_with_scale_buffer,
                block_pages,
                block_token_counts,
                weights,
                cu_block_ks,
                cu_block_ke,
                prefix_lens,
                k_block_size=k_block_size,
                block_topk=block_topk,
                topk_tokens=topk_tokens,
                return_block_topk=return_block_topk,
                topk_offsets=topk_offsets,
            )
        _COMPILED_SHAPES.add(shape_key)
        return result

    return _hisa_indexer_paged_impl(
        q,
        index_k_with_scale_buffer,
        block_pages,
        block_token_counts,
        weights,
        cu_block_ks,
        cu_block_ke,
        prefix_lens,
        k_block_size=k_block_size,
        block_topk=block_topk,
        topk_tokens=topk_tokens,
        return_block_topk=return_block_topk,
        topk_offsets=topk_offsets,
    )


def _hisa_indexer_paged_impl(
    q: torch.Tensor,
    index_k_with_scale_buffer: torch.Tensor,
    block_pages: torch.Tensor,
    block_token_counts: torch.Tensor,
    weights: torch.Tensor,
    cu_block_ks: torch.Tensor,
    cu_block_ke: torch.Tensor,
    prefix_lens: torch.Tensor,
    *,
    k_block_size: int,
    block_topk: int,
    topk_tokens: int,
    return_block_topk: bool = False,
    topk_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    from sglang.jit_kernel.hisa_topk_map import hisa_paged_block_mean_pool_cuda

    stage_start = _stage_profile_start(q.device)
    blocked_k_fp8, blocked_k_scale = hisa_paged_block_mean_pool_cuda(
        index_k_with_scale_buffer,
        block_pages,
        block_token_counts,
        k_block_size,
    )
    _stage_profile_end(
        "paged_mean_pool",
        stage_start,
        q.device,
        rows=int(q.shape[0]),
        blocks=int(block_pages.shape[0]),
        block_size=int(k_block_size),
    )

    stage_start = _stage_profile_start(q.device)
    block_k_score = pool_mqa_attn_return_logits_fp8_interface(
        q,
        blocked_k_fp8,
        blocked_k_scale,
        weights,
        cu_block_ks,
        cu_block_ke,
    )
    _stage_profile_end(
        "block_score",
        stage_start,
        q.device,
        rows=int(q.shape[0]),
        blocks=int(block_k_score.shape[-1]),
    )

    block_topk_eff = min(block_topk, block_k_score.shape[-1])
    stage_start = _stage_profile_start(q.device)
    topk_block_indices = _topk_block_indices(
        block_k_score,
        cu_block_ks,
        cu_block_ke,
        block_topk_eff,
        k_block_size,
    )
    _stage_profile_end(
        "block_topk",
        stage_start,
        q.device,
        rows=int(q.shape[0]),
        blocks=int(block_k_score.shape[-1]),
        block_topk=int(block_topk_eff),
    )

    stage_start = _stage_profile_start(q.device)
    deepgemm_candidate = _deepgemm_paged_candidate_logits(
        q,
        index_k_with_scale_buffer,
        block_pages,
        topk_block_indices,
        k_block_size,
        weights,
        cu_block_ks,
        prefix_lens,
    )
    _stage_profile_end(
        "candidate_logits",
        stage_start,
        q.device,
        rows=int(q.shape[0]),
        candidate_len=int(block_topk_eff * k_block_size),
    )

    if deepgemm_candidate is None:
        raise RuntimeError("Paged HISA requires DeepGEMM candidate logits on CUDA.")

    block_sparse_logits, candidate_rel_blocks = deepgemm_candidate
    if topk_tokens == 2048:
        stage_start = _stage_profile_start(q.device)
        topk_indices = _topk_candidate_indices(
            block_sparse_logits,
            candidate_rel_blocks,
            prefix_lens,
            k_block_size,
            topk_offsets=topk_offsets,
        )
        _stage_profile_end(
            "candidate_topk",
            stage_start,
            q.device,
            rows=int(q.shape[0]),
            candidate_len=int(block_sparse_logits.shape[-1]),
            topk_tokens=int(topk_tokens),
            topk_offsets=topk_offsets is not None,
        )
    else:
        stage_start = _stage_profile_start(q.device)
        _mask_candidate_logits(
            block_sparse_logits,
            candidate_rel_blocks,
            prefix_lens,
            k_block_size,
        )
        _stage_profile_end(
            "candidate_mask",
            stage_start,
            q.device,
            rows=int(q.shape[0]),
            candidate_len=int(block_sparse_logits.shape[-1]),
        )
        stage_start = _stage_profile_start(q.device)
        relevant_topk_indices = _topk_token_indices(block_sparse_logits, topk_tokens)
        _stage_profile_end(
            "candidate_torch_topk",
            stage_start,
            q.device,
            rows=int(q.shape[0]),
            candidate_len=int(block_sparse_logits.shape[-1]),
            topk_tokens=int(topk_tokens),
        )
        stage_start = _stage_profile_start(q.device)
        topk_indices = _map_candidate_topk_indices(
            relevant_topk_indices,
            candidate_rel_blocks,
            prefix_lens,
            k_block_size,
        )
        _stage_profile_end(
            "candidate_map",
            stage_start,
            q.device,
            rows=int(q.shape[0]),
            topk_tokens=int(topk_tokens),
        )

    if topk_offsets is None:
        stage_start = _stage_profile_start(q.device)
        topk_indices = _mask_relative_topk_indices(topk_indices, prefix_lens)
        _stage_profile_end(
            "mask_relative",
            stage_start,
            q.device,
            rows=int(q.shape[0]),
            topk_tokens=int(topk_indices.shape[-1]),
        )
    if return_block_topk:
        return topk_indices, topk_block_indices
    return topk_indices


def _deepgemm_paged_candidate_logits(
    q: torch.Tensor,
    index_k_with_scale_buffer: torch.Tensor,
    block_pages: torch.Tensor,
    topk_block_indices: torch.Tensor,
    k_block_size: int,
    weights: torch.Tensor,
    cu_block_ks: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    try:
        import deep_gemm
    except ImportError:
        return None

    page_size = 64
    if k_block_size % page_size != 0:
        return None

    topk = topk_block_indices.shape[1]
    candidate_len = topk * k_block_size
    safe_block_indices = topk_block_indices.reshape(-1).long().clamp(
        0, block_pages.shape[0] - 1
    )
    candidate_page_table = block_pages[safe_block_indices]
    candidate_page_table = candidate_page_table.reshape(q.shape[0], -1).contiguous()
    safe_candidate_page_table = candidate_page_table.clamp_min(0)
    candidate_lens, schedule_metadata = _deepgemm_candidate_metadata(
        deep_gemm,
        q,
        candidate_len,
        page_size,
    )
    kv_cache = index_k_with_scale_buffer.view(
        index_k_with_scale_buffer.shape[0],
        page_size,
        1,
        q.shape[-1] + 4,
    )
    logits = deep_gemm.fp8_paged_mqa_logits(
        q.unsqueeze(1),
        kv_cache,
        weights,
        candidate_lens,
        safe_candidate_page_table,
        schedule_metadata,
        candidate_len,
        clean_logits=False,
    )

    candidate_rel_blocks = (topk_block_indices - cu_block_ks[:, None]).to(torch.int32)
    return logits, candidate_rel_blocks


def _mask_candidate_logits(
    logits: torch.Tensor,
    candidate_rel_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    k_block_size: int,
) -> None:
    from sglang.jit_kernel.hisa_topk_map import hisa_mask_candidate_logits_cuda

    hisa_mask_candidate_logits_cuda(
        logits,
        candidate_rel_blocks,
        prefix_lens,
        k_block_size,
    )


def _deepgemm_candidate_metadata(
    deep_gemm,
    q: torch.Tensor,
    candidate_len: int,
    page_size: int,
) -> tuple[torch.Tensor, object]:
    device = q.device
    cache_key = (
        device.type,
        device.index,
        q.shape[0],
        candidate_len,
        page_size,
    )
    cached = _DEEPGEMM_METADATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    candidate_lens = torch.full(
        (q.shape[0], 1),
        candidate_len,
        dtype=torch.int32,
        device=q.device,
    )
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        candidate_lens,
        page_size,
        deep_gemm.get_num_sms(),
    )
    cached = (candidate_lens, schedule_metadata)
    _DEEPGEMM_METADATA_CACHE[cache_key] = cached
    return cached


def _map_candidate_topk_indices(
    relevant_topk_indices: torch.Tensor,
    candidate_rel_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    k_block_size: int,
) -> torch.Tensor:
    from sglang.jit_kernel.hisa_topk_map import hisa_map_candidate_topk_cuda

    return hisa_map_candidate_topk_cuda(
        relevant_topk_indices,
        candidate_rel_blocks,
        prefix_lens,
        k_block_size,
    )


def _topk_candidate_indices(
    logits: torch.Tensor,
    candidate_rel_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    k_block_size: int,
    topk_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from sglang.jit_kernel.hisa_topk_map import hisa_candidate_topk_cuda

    return hisa_candidate_topk_cuda(
        logits,
        candidate_rel_blocks,
        prefix_lens,
        k_block_size,
        topk_offsets=topk_offsets,
    )


def _topk_candidate_page_indices(
    logits: torch.Tensor,
    candidate_rel_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    page_table_1: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    k_block_size: int,
) -> torch.Tensor:
    from sglang.jit_kernel.hisa_topk_map import hisa_candidate_topk_paged_cuda

    return hisa_candidate_topk_paged_cuda(
        logits,
        candidate_rel_blocks,
        prefix_lens,
        page_table_1,
        token_to_batch_idx,
        k_block_size,
    )


def _topk_block_indices(
    block_k_score: torch.Tensor,
    cu_block_ks: torch.Tensor,
    cu_block_ke: torch.Tensor,
    block_topk: int,
    k_block_size: int,
) -> torch.Tensor:
    if (
        block_topk <= 64
        and block_k_score.shape[-1] <= 2048
        and block_k_score.is_cuda
        and cu_block_ks.dtype == torch.int32
        and cu_block_ke.dtype == torch.int32
    ):
        from sglang.jit_kernel.hisa_topk_map import hisa_block_topk_cuda

        return hisa_block_topk_cuda(
            block_k_score,
            cu_block_ks,
            cu_block_ke,
            block_topk,
            k_block_size,
        )

    clean_and_maintain_logits_interface(
        block_k_score,
        cu_block_ks,
        cu_block_ke,
    )
    topk_block_indices = torch.topk(
        block_k_score.bfloat16(),
        k=block_topk,
        dim=-1,
        sorted=False,
    ).indices
    topk_block_indices = torch.maximum(
        topk_block_indices,
        cu_block_ks[:, None],
    )
    return torch.minimum(
        topk_block_indices,
        (cu_block_ke - 1)[:, None],
    )


def _map_block_sparse_topk_indices(
    relevant_topk_indices: torch.Tensor,
    topk_block_indices: torch.Tensor,
    cu_block_ks: torch.Tensor,
    k_block_size: int,
) -> torch.Tensor:
    absolute_topk_block_indices = torch.gather(
        topk_block_indices,
        dim=-1,
        index=(relevant_topk_indices // k_block_size),
    )
    topk_indices = (absolute_topk_block_indices - cu_block_ks[:, None]) * k_block_size
    return topk_indices + (relevant_topk_indices % k_block_size)


def build_hisa_block_metadata(
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    k_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    page_size = 64
    pages_per_block = k_block_size // page_size
    assert k_block_size % page_size == 0

    max_blocks = (page_table.shape[1] + pages_per_block - 1) // pages_per_block
    max_pages = max_blocks * pages_per_block
    if max_pages == page_table.shape[1]:
        padded_page_table = page_table
    else:
        padded_page_table = torch.full(
            (page_table.shape[0], max_pages),
            -1,
            dtype=page_table.dtype,
            device=page_table.device,
        )
        padded_page_table[:, : page_table.shape[1]] = page_table
    block_pages = padded_page_table.reshape(
        page_table.shape[0],
        max_blocks,
        pages_per_block,
    )
    block_ids = torch.arange(max_blocks, device=page_table.device)
    block_pages = block_pages.reshape(-1, pages_per_block).contiguous().to(torch.int32)

    token_counts = seq_lens[:, None] - block_ids[None, :] * k_block_size
    token_counts = token_counts.clamp_(min=0, max=k_block_size)
    token_counts = token_counts.reshape(-1).contiguous().to(torch.int32)

    block_offsets = torch.arange(
        page_table.shape[0],
        dtype=torch.int32,
        device=page_table.device,
    )
    return block_pages, token_counts, block_offsets * max_blocks


def _topk_token_indices(logits: torch.Tensor, topk_tokens: int) -> torch.Tensor:
    topk_tokens_eff = min(topk_tokens, logits.shape[-1])
    if topk_tokens_eff == 2048:
        from sgl_kernel import fast_topk_v2

        lengths = torch.full(
            (logits.shape[0],),
            logits.shape[1],
            dtype=torch.int32,
            device=logits.device,
        )
        row_starts = torch.zeros_like(lengths)
        return fast_topk_v2(
            logits,
            lengths,
            topk_tokens_eff,
            row_starts=row_starts,
        )

    return torch.topk(
        logits,
        k=topk_tokens_eff,
        dim=-1,
    ).indices


def _mask_topk_indices(
    topk_indices: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    topk_indices = topk_indices.to(torch.int32)
    topk_indices -= cu_seqlen_ks[:, None]
    mask_lo = topk_indices >= 0
    mask_hi = topk_indices - (cu_seqlen_ke - cu_seqlen_ks)[:, None] < 0
    mask = mask_lo & mask_hi
    topk_indices = topk_indices.masked_fill(~mask, -1)
    return topk_indices


def _mask_relative_topk_indices(
    topk_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> torch.Tensor:
    topk_indices = topk_indices.to(torch.int32)
    mask = (topk_indices >= 0) & (topk_indices < prefix_lens[:, None])
    topk_indices = topk_indices.masked_fill(~mask, -1)
    return topk_indices
