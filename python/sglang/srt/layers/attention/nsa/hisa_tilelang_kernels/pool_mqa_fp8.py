"""Stage-1 kernel: prefill pool-MQA over pooled (blocked) K.

Input: fp8 Q ``[M, H, D]`` + fp8 BlockedK ``[Nb, D]`` + per-block f32 scale
``[Nb]`` + f32 Weights ``[M, H]`` + per-query ``cu_seqlen_blocked_ks/ke [M]``.

For each query ``m`` and pool block ``n`` in ``[cu_seqlen_blocked_ks[m],
cu_seqlen_blocked_ke[m])``:
  ``logits[m, n] = sum_h ReLU(Q[m, h] . BlockedK[n]) * BlockedKScale[n] * Weights[m, h]``

Out-of-range entries in the raw kernel output are undefined — caller should
zero-init the buffer or apply a separate mask kernel.
"""

import tilelang
from tilelang import language as T
import torch


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def pool_mqa_attn_return_logits_fp8(
    IndexQ,
    IndexBlockedK,
    IndexBlockedKScale,
    Logits,
    Weights,
    CuSeqLenBlockedKS,
    CuSeqLenBlockedKE,
    heads: int = 64,
    index_dim: int = 128,
    block_N: int = 256,
    num_stages: int = 3,
    threads: int = 512,
    block_Q: int = 0,
):
    # block_Q is the tile size for queries; `0` means "derive from heads".
    if block_Q == 0:
        block_Q = 128 // heads
    fp8_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len, seq_len_blocked_kv = T.const("seq_len, seq_len_blocked_kv")

    IndexQ: T.Tensor[[seq_len * heads, index_dim], fp8_dtype]
    IndexBlockedK: T.Tensor[[seq_len_blocked_kv, index_dim], fp8_dtype]
    IndexBlockedKScale: T.Tensor[[seq_len_blocked_kv], accum_dtype]
    Logits: T.Tensor[[seq_len, seq_len_blocked_kv], accum_dtype]
    Weights: T.Tensor[[seq_len, heads], accum_dtype]
    CuSeqLenBlockedKS: T.Tensor[[seq_len], index_dtype]
    CuSeqLenBlockedKE: T.Tensor[[seq_len], index_dtype]

    with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
        index_q_shared = T.alloc_shared([block_Q * heads, index_dim], fp8_dtype)
        index_k_shared = T.alloc_shared([block_N, index_dim], fp8_dtype)
        index_k_scale_fragment = T.alloc_fragment([block_N], accum_dtype)
        s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
        s_reshaped = T.reshape(s, (block_N, block_Q, heads))
        logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
        weights = T.alloc_fragment([block_Q, heads], accum_dtype)

        seq_len_i = bx * block_Q

        cu_k_s_min = T.alloc_var(index_dtype)
        cu_k_e_max = T.alloc_var(index_dtype)
        cu_k_s_min = 2147483647
        cu_k_e_max = -2147483648

        for bq_i in T.serial(block_Q):
            cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenBlockedKS[seq_len_i + bq_i], seq_len_blocked_kv))
        for bq_i in T.serial(block_Q):
            cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenBlockedKE[seq_len_i + bq_i], seq_len_blocked_kv))

        T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
        T.copy(Weights[seq_len_i, 0], weights)

        for nbn_i in T.Pipelined(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages):
            T.copy(IndexBlockedK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)
            T.copy(IndexBlockedKScale[cu_k_s_min + nbn_i * block_N], index_k_scale_fragment)

            T.gemm(
                index_k_shared,
                index_q_shared,
                s,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullCol,
            )

            for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                s_reshaped[bn_i, bq_i, h_i] = T.max(s_reshaped[bn_i, bq_i, h_i] * index_k_scale_fragment[bn_i], 0) * weights[bq_i, h_i]

            T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

            for bq_i, bn_i in T.Parallel(block_Q, block_N):
                Logits[seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i] = logits[bn_i, bq_i]


def pool_mqa_attn_return_logits_fp8_interface(
    q_fp8: torch.Tensor,
    blocked_kv_fp8: torch.Tensor,
    blocked_kv_scale: torch.Tensor,
    weights_f32: torch.Tensor,
    cu_seqlen_blocked_ks: torch.Tensor,
    cu_seqlen_blocked_ke: torch.Tensor,
    block_N: int = 256,
):
    """Raw kernel invocation; zero-inits logits so positions the kernel
    doesn't touch are 0 (matches the ref)."""
    seq_len, heads, index_dim = q_fp8.shape
    seq_len_blocked_kv = blocked_kv_fp8.shape[0]

    logits = torch.zeros([seq_len, seq_len_blocked_kv], device=q_fp8.device, dtype=torch.float32)
    pool_mqa_attn_return_logits_fp8(
        q_fp8.view(seq_len * heads, index_dim),
        blocked_kv_fp8,
        blocked_kv_scale,
        logits,
        weights_f32,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
        heads=heads,
        index_dim=index_dim,
        block_N=block_N,
    )
    return logits
