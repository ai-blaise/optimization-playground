import tilelang
from tilelang import language as T
import torch


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_block_mean_pooling(
    K,
    KScale,
    dim: int = 128,
    pooling_block_size: int = 128,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 256,
):
    dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    FP8_MAX_INV = 1.0 / 448.0

    seq_len_k = T.const("seq_len_k")

    K: T.Tensor[[seq_len_k, dim], dtype]
    KScale: T.Tensor[[seq_len_k], accum_dtype]

    num_blocks = T.ceildiv(seq_len_k, pooling_block_size)
    BlockedK = T.empty((num_blocks, dim), dtype)
    BlockedKScale = T.empty((num_blocks,), accum_dtype)

    with T.Kernel(num_blocks, threads=threads) as bx:
        index_k = T.alloc_fragment([block_N, dim], dtype)
        scale = T.alloc_fragment([block_N], accum_dtype)
        acc = T.alloc_fragment([dim], accum_dtype)
        max_abs = T.alloc_fragment([1], accum_dtype)
        T.fill(acc, 0.0)

        k_start = bx * pooling_block_size
        k_end = T.min(k_start + pooling_block_size, seq_len_k)
        cur_pooling_block_size = k_end - k_start

        for b_i in T.serial(T.ceildiv(cur_pooling_block_size, block_N)):
            T.fill(index_k, 0.0)

            tl_block_s = k_start + b_i * block_N
            tl_block_e = T.min(k_start + (b_i + 1) * block_N, k_end)
            T.copy(K[tl_block_s : tl_block_s + block_N, :], index_k)
            for bn_i in T.Parallel(block_N):
                scale[bn_i] = KScale[tl_block_s + bn_i]

            for bn_i, d_i in T.Parallel(block_N, dim):
                index_k[bn_i, d_i] = index_k[bn_i, d_i] * scale[bn_i]

            cur_tl_block_size = tl_block_e - tl_block_s
            for n_i in T.parallel(block_N):
                for d_i in T.parallel(dim):
                    if n_i >= cur_tl_block_size:
                        index_k[n_i, d_i] = T.cast(0, accum_dtype)

            T.reduce_sum(index_k, acc, dim=0, clear=False)

        inv_count = T.cast(1.0, accum_dtype) / T.cast(cur_pooling_block_size, accum_dtype)
        for d_i in T.Parallel(dim):
            acc[d_i] = acc[d_i] * inv_count

        # Re-quantize f32 mean to fp8 with a per-block scale.
        T.reduce_absmax(acc, max_abs, dim=0, clear=True)
        block_scale = T.max(max_abs[0] * T.cast(FP8_MAX_INV, accum_dtype), T.cast(1e-10, accum_dtype))
        inv_block_scale = T.cast(1.0, accum_dtype) / block_scale

        for d_i in T.Parallel(dim):
            BlockedK[bx, d_i] = T.cast(acc[d_i] * inv_block_scale, dtype)
        BlockedKScale[bx] = block_scale

    return BlockedK, BlockedKScale


def fp8_native_block_mean_pooling_interface(k: torch.Tensor, k_scale: torch.Tensor, k_block_size: int):
    return fp8_native_block_mean_pooling(k, k_scale, dim=k.shape[1], pooling_block_size=k_block_size)
