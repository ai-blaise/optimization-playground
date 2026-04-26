import tilelang
from tilelang import language as T
import torch


@tilelang.jit
def clean_and_maintain_logits_(
    Logits,
    CuSeqLenKS,
    CuSeqLenKE,
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len, seq_len_kv = T.const("seq_len, seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    Logits: T.Tensor[[seq_len, seq_len_kv], dtype]
    CuSeqLenKS: T.Tensor[[seq_len], indices_dtype]
    CuSeqLenKE: T.Tensor[[seq_len], indices_dtype]

    with T.Kernel(seq_len, threads=threads) as bx:
        tx = T.thread_binding(0, threads, thread="threadIdx.x")
        cu_k_s = CuSeqLenKS[bx]
        cu_k_e = CuSeqLenKE[bx]

        for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
            for k_i in T.serial(block_K // threads):
                idx = n_i * block_K + k_i * threads + tx
                if idx == cu_k_s or idx == cu_k_e - 1:
                    Logits[bx, idx] = T.infinity(dtype)
                if idx < cu_k_s or idx >= cu_k_e:
                    Logits[bx, idx] = -T.infinity(dtype)


def clean_and_maintain_logits_interface(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    """In-place: applies +inf/-inf mask based on per-row [ks, ke)."""
    clean_and_maintain_logits_(logits, cu_seqlen_ks, cu_seqlen_ke)
    return logits
