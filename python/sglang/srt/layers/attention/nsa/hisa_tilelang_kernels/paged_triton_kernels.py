import torch
import triton
import triton.language as tl


def fp8_paged_block_mean_pooling_interface(
    index_k_with_scale_buffer: torch.Tensor,
    block_pages: torch.Tensor,
    block_token_counts: torch.Tensor,
    k_block_size: int,
):
    page_size = 64
    index_dim = 128
    pages_per_block = k_block_size // page_size
    assert k_block_size % page_size == 0
    assert block_pages.shape[1] == pages_per_block

    num_blocks = block_pages.shape[0]
    blocked_k = torch.empty(
        (num_blocks, index_dim),
        dtype=torch.float8_e4m3fn,
        device=index_k_with_scale_buffer.device,
    )
    blocked_k_scale = torch.empty(
        (num_blocks,),
        dtype=torch.float32,
        device=index_k_with_scale_buffer.device,
    )

    _fp8_paged_block_mean_pooling_kernel[(num_blocks,)](
        index_k_with_scale_buffer.view(torch.float8_e4m3fn),
        index_k_with_scale_buffer.view(torch.float32),
        block_pages,
        block_token_counts,
        blocked_k,
        blocked_k_scale,
        PAGE_BYTES=index_k_with_scale_buffer.shape[1],
        PAGE_FLOATS=index_k_with_scale_buffer.shape[1] // 4,
        SCALE_OFFSET=page_size * index_dim // 4,
        PAGE_SIZE=page_size,
        INDEX_DIM=index_dim,
        K_BLOCK_SIZE=k_block_size,
        BLOCK_T=32,
        BLOCK_D=128,
    )
    return blocked_k, blocked_k_scale


@triton.jit
def _fp8_paged_block_mean_pooling_kernel(
    buf_fp8,
    buf_f32,
    block_pages,
    block_token_counts,
    blocked_k,
    blocked_k_scale,
    PAGE_BYTES: tl.constexpr,
    PAGE_FLOATS: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    INDEX_DIM: tl.constexpr,
    K_BLOCK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_id = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    token_count = tl.load(block_token_counts + block_id)

    for start in range(0, K_BLOCK_SIZE, BLOCK_T):
        offs_t = start + tl.arange(0, BLOCK_T)
        page_slot = offs_t // PAGE_SIZE
        token_in_page = offs_t % PAGE_SIZE
        page = tl.load(block_pages + block_id * (K_BLOCK_SIZE // PAGE_SIZE) + page_slot)

        k = tl.load(
            buf_fp8
            + page[:, None] * PAGE_BYTES
            + token_in_page[:, None] * INDEX_DIM
            + offs_d[None, :],
            mask=offs_t[:, None] < token_count,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(
            buf_f32 + page * PAGE_FLOATS + SCALE_OFFSET + token_in_page,
            mask=offs_t < token_count,
            other=0.0,
        )
        acc += tl.sum(k * scale[:, None], axis=0)

    mean = acc / token_count.to(tl.float32)
    abs_max = tl.max(tl.abs(mean), axis=0)
    out_scale = tl.maximum(abs_max * (1.0 / 448.0), 1.0e-10)
    tl.store(blocked_k + block_id * INDEX_DIM + offs_d, mean / out_scale)
    tl.store(blocked_k_scale + block_id, out_scale)


def fp8_paged_block_sparse_mqa_attn_return_logits_interface(
    q: torch.Tensor,
    index_k_with_scale_buffer: torch.Tensor,
    block_pages: torch.Tensor,
    topk_block_index: torch.Tensor,
    kv_block_size: int,
    weights: torch.Tensor,
    cu_seqlen_blocked_ks: torch.Tensor,
    prefix_lens: torch.Tensor,
):
    page_size = 64
    seq_len, heads, index_dim = q.shape
    topk = topk_block_index.shape[1]
    assert kv_block_size % page_size == 0
    assert heads == 64
    assert index_dim == 128

    logits = torch.empty(
        (seq_len, topk * kv_block_size),
        dtype=torch.float32,
        device=q.device,
    )
    grid = (seq_len, topk)
    _fp8_paged_block_sparse_mqa_kernel[grid](
        q,
        index_k_with_scale_buffer.view(torch.float8_e4m3fn),
        index_k_with_scale_buffer.view(torch.float32),
        block_pages,
        topk_block_index,
        weights,
        cu_seqlen_blocked_ks,
        prefix_lens,
        logits,
        PAGE_BYTES=index_k_with_scale_buffer.shape[1],
        PAGE_FLOATS=index_k_with_scale_buffer.shape[1] // 4,
        SCALE_OFFSET=page_size * index_dim // 4,
        PAGE_SIZE=page_size,
        INDEX_DIM=index_dim,
        KV_BLOCK_SIZE=kv_block_size,
        TOPK=topk,
        BLOCK_T=kv_block_size,
        HEADS=heads,
    )
    return logits


@triton.jit
def _fp8_paged_block_sparse_mqa_kernel(
    q,
    buf_fp8,
    buf_f32,
    block_pages,
    topk_block_index,
    weights,
    cu_seqlen_blocked_ks,
    prefix_lens,
    logits,
    PAGE_BYTES: tl.constexpr,
    PAGE_FLOATS: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    INDEX_DIM: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HEADS: tl.constexpr,
):
    query_id = tl.program_id(0)
    topk_id = tl.program_id(1)

    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, INDEX_DIM)
    offs_h = tl.arange(0, HEADS)

    topk_block_id = tl.load(topk_block_index + query_id * TOPK + topk_id)
    first_block = tl.load(cu_seqlen_blocked_ks + query_id)
    prefix_len = tl.load(prefix_lens + query_id)
    local_block_id = topk_block_id - first_block
    page_slot = offs_t // PAGE_SIZE
    token_in_page = offs_t % PAGE_SIZE
    page = tl.load(block_pages + topk_block_id * (KV_BLOCK_SIZE // PAGE_SIZE) + page_slot)

    k = tl.load(
        buf_fp8
        + page[:, None] * PAGE_BYTES
        + token_in_page[:, None] * INDEX_DIM
        + offs_d[None, :],
    )
    q_tile = tl.load(
        q + query_id * HEADS * INDEX_DIM + offs_h[:, None] * INDEX_DIM + offs_d[None, :]
    )
    dot = tl.dot(k, tl.trans(q_tile))
    scale = tl.load(buf_f32 + page * PAGE_FLOATS + SCALE_OFFSET + token_in_page)
    w = tl.load(weights + query_id * HEADS + offs_h)
    score = tl.sum(tl.maximum(dot * scale[:, None], 0.0) * w[None, :], axis=1)

    token_offset = local_block_id * KV_BLOCK_SIZE + offs_t
    valid = (topk_block_id >= first_block) & (token_offset >= 0) & (
        token_offset < prefix_len
    )
    score = tl.where(valid, score, -float("inf"))
    out = query_id * TOPK * KV_BLOCK_SIZE + topk_id * KV_BLOCK_SIZE
    tl.store(logits + out + offs_t, score)
