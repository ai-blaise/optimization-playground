import tilelang
from tilelang import language as T
import torch


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_block_sparse_mqa_attn_return_logits(
    IndexQ,
    IndexK,
    IndexKScale,
    TopKBlockIndex,
    Weights,
    CuSeqLenKS,
    CuSeqLenKE,
    heads: int = 64,
    index_dim: int = 128,
    kv_block_size: int = 128,
    topk: int = 64,
    block_N: int = 128,
    num_stages: int = 1,
    threads: int = 256,
):
    fp8_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32
    topk_index_dtype = T.int64

    seq_len, seq_len_kv = T.const("seq_len, seq_len_kv")

    H_per_block = heads
    block_N = min(block_N, kv_block_size // 2)
    assert kv_block_size % block_N == 0, "block_N must divide kv_block_size"

    IndexQ: T.Tensor[[seq_len * heads, index_dim], fp8_dtype]
    IndexK: T.Tensor[[seq_len_kv, index_dim], fp8_dtype]
    IndexKScale: T.Tensor[[seq_len_kv], accum_dtype]
    TopKBlockIndex: T.Tensor[[seq_len, topk], topk_index_dtype]
    Weights: T.Tensor[[seq_len, heads], accum_dtype]
    CuSeqLenKS: T.Tensor[[seq_len], index_dtype]
    CuSeqLenKE: T.Tensor[[seq_len], index_dtype]

    Logits = T.empty((seq_len, topk * kv_block_size), accum_dtype)

    with T.Kernel(seq_len, threads=threads) as bx:
        index_q_shared = T.alloc_shared([H_per_block, index_dim], fp8_dtype)
        index_k_shared = T.alloc_shared([block_N, index_dim], fp8_dtype)
        # Shared (zero-init'd) — see note in the hisa source about serial-topk
        # loop making shared slightly faster than fragment here.
        scale_shared = T.alloc_shared([block_N], accum_dtype)

        s = T.alloc_fragment([block_N, H_per_block], accum_dtype)
        s_reshaped = T.reshape(s, (block_N, H_per_block // heads, heads))
        logits = T.alloc_fragment([block_N, H_per_block // heads], accum_dtype)
        weights = T.alloc_fragment([H_per_block // heads, heads], accum_dtype)

        seq_len_i = bx

        cu_k_s_min = CuSeqLenKS[seq_len_i]
        cu_k_e_max = CuSeqLenKE[seq_len_i]

        T.copy(IndexQ[seq_len_i * heads : seq_len_i * heads + H_per_block, :], index_q_shared)
        T.copy(Weights[seq_len_i, :], weights)

        for n_i in T.serial(topk):
            topk_block_id = T.cast(TopKBlockIndex[seq_len_i, n_i], index_dtype)
            block_s = topk_block_id * kv_block_size
            for b_i in T.Pipelined(kv_block_size // block_N, num_stages=num_stages):
                block_s_i = block_s + b_i * block_N

                T.copy(IndexK[block_s_i : block_s_i + block_N, :], index_k_shared)
                for bn_i in T.Parallel(block_N):
                    scale_shared[bn_i] = IndexKScale[block_s_i + bn_i]

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, H_per_block // heads, heads):
                    s_reshaped[bn_i, bq_i, h_i] = T.max(s_reshaped[bn_i, bq_i, h_i] * scale_shared[bn_i], 0) * weights[bq_i, h_i]

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for i_i in T.Parallel(block_N):
                    k_i = block_s_i + i_i
                    if k_i < cu_k_s_min or k_i >= cu_k_e_max:
                        logits[i_i, 0] = -T.infinity(accum_dtype)

                for bn_i in T.Parallel(block_N):
                    Logits[seq_len_i, n_i * kv_block_size + b_i * block_N + bn_i] = logits[bn_i, 0]

    return Logits


def fp8_native_block_sparse_mqa_attn_return_logits_interface(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    topk_block_index: torch.Tensor,
    kv_block_size: int,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    seq_len, heads, index_dim = q.shape
    topk = topk_block_index.shape[1]
    logits = fp8_native_block_sparse_mqa_attn_return_logits(
        q.view(seq_len * heads, index_dim),
        k,
        k_scale,
        topk_block_index,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        heads=heads,
        index_dim=index_dim,
        kv_block_size=kv_block_size,
        topk=topk,
    )
    return logits
