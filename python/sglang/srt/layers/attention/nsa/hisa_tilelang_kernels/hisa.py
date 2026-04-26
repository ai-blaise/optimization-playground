import contextlib
import fcntl
import os
import tempfile

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
from sglang.srt.layers.attention.nsa.hisa_tilelang_kernels.paged_triton_kernels import (
    fp8_paged_block_mean_pooling_interface,
    fp8_paged_block_sparse_mqa_attn_return_logits_interface,
)
from sglang.srt.layers.attention.nsa.hisa_tilelang_kernels.pool_mqa_fp8 import (
    pool_mqa_attn_return_logits_fp8_interface,
)


_COMPILED_SHAPES = set()


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
) -> torch.Tensor:
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
) -> torch.Tensor:
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

    clean_and_maintain_logits_interface(
        block_k_score,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
    )

    block_topk_eff = min(block_topk, block_k_score.shape[-1])
    topk_block_indices = torch.topk(
        block_k_score.bfloat16(),
        k=block_topk_eff,
        dim=-1,
        sorted=False,
    ).indices
    _force_boundary_blocks_(
        topk_block_indices,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
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
    return _mask_topk_indices(topk_indices, cu_seqlen_ks, cu_seqlen_ke)


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
    return_block_topk: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    block_pages, block_token_counts, block_offsets = _build_hisa_block_metadata(
        page_table,
        seq_lens,
        k_block_size,
    )
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
    return_block_topk: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    blocked_k_fp8, blocked_k_scale = fp8_paged_block_mean_pooling_interface(
        index_k_with_scale_buffer,
        block_pages,
        block_token_counts,
        k_block_size,
    )

    block_k_score = pool_mqa_attn_return_logits_fp8_interface(
        q,
        blocked_k_fp8,
        blocked_k_scale,
        weights,
        cu_block_ks,
        cu_block_ke,
    )

    clean_and_maintain_logits_interface(
        block_k_score,
        cu_block_ks,
        cu_block_ke,
    )

    block_topk_eff = min(block_topk, block_k_score.shape[-1])
    topk_block_indices = torch.topk(
        block_k_score.bfloat16(),
        k=block_topk_eff,
        dim=-1,
        sorted=False,
    ).indices
    _force_boundary_blocks_(
        topk_block_indices,
        cu_block_ks,
        cu_block_ke,
    )

    block_sparse_logits = (
        fp8_paged_block_sparse_mqa_attn_return_logits_interface(
            q,
            index_k_with_scale_buffer,
            block_pages,
            topk_block_indices,
            k_block_size,
            weights,
            cu_block_ks,
            prefix_lens,
        )
    )

    relevant_topk_indices = _topk_token_indices(block_sparse_logits, topk_tokens)

    absolute_topk_block_indices = torch.gather(
        topk_block_indices,
        dim=-1,
        index=(relevant_topk_indices // k_block_size),
    )
    topk_indices = (absolute_topk_block_indices - cu_block_ks[:, None]) * k_block_size
    topk_indices = topk_indices + (relevant_topk_indices % k_block_size)
    topk_indices = _mask_relative_topk_indices(topk_indices, prefix_lens)
    if return_block_topk:
        return topk_indices, topk_block_indices
    return topk_indices


def _build_hisa_block_metadata(
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    k_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    page_size = 64
    pages_per_block = k_block_size // page_size
    assert k_block_size % page_size == 0

    max_blocks = page_table.shape[1] // pages_per_block
    max_pages = max_blocks * pages_per_block
    block_pages = page_table[:, :max_pages].reshape(
        page_table.shape[0],
        max_blocks,
        pages_per_block,
    )
    block_ids = torch.arange(max_blocks, device=page_table.device)
    block_pages = block_pages.reshape(-1, pages_per_block).contiguous().to(torch.int32)

    token_counts = seq_lens[:, None] - block_ids[None, :] * k_block_size
    token_counts = token_counts.clamp_(min=1, max=k_block_size)
    token_counts = token_counts.reshape(-1).contiguous().to(torch.int32)

    block_offsets = torch.arange(
        page_table.shape[0],
        dtype=torch.int32,
        device=page_table.device,
    )
    return block_pages, token_counts, block_offsets * max_blocks


def _force_boundary_blocks_(
    topk_block_indices: torch.Tensor,
    cu_block_ks: torch.Tensor,
    cu_block_ke: torch.Tensor,
) -> None:
    if topk_block_indices.shape[1] == 0:
        return

    first_blocks = cu_block_ks.to(topk_block_indices.dtype)
    _force_block_(topk_block_indices, first_blocks, 0)

    if topk_block_indices.shape[1] > 1:
        last_blocks = (cu_block_ke - 1).to(topk_block_indices.dtype)
        _force_block_(topk_block_indices, last_blocks, 1)


def _force_block_(
    topk_block_indices: torch.Tensor,
    target_blocks: torch.Tensor,
    slot: int,
) -> None:
    present = (topk_block_indices == target_blocks[:, None]).any(dim=1)
    topk_block_indices[:, slot] = torch.where(
        present,
        topk_block_indices[:, slot],
        target_blocks,
    )


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
