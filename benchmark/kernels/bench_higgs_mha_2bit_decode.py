"""3-path microbench: fused HIGGS-MHA-2bit decode vs FP8 baseline vs
materialize-on-fetch HIGGS.

Targets the SMC-SD draft attention shape (head_dim=128,
num_q_heads=32, num_kv_heads=2 GQA), sweeps batch x seq_len, and
reports the per-decode-step kernel latency in microseconds.

Run on a B200 (or any SM90+ CUDA box with Triton):

    python benchmark/kernels/bench_higgs_mha_2bit_decode.py

The intended interpretation:
* ``baseline-fp8``: dense Triton GQA decode against an FP8 KV cache
  (existing :func:`decode_attention_fwd`). This is the latency target
  the fused kernel needs to stay within 2x of.
* ``higgs-mof``: the regressed "materialize-on-fetch" path that
  decompresses the full layer cache to BF16 via
  :meth:`HiggsMHA2BitCodec.decompress` on every decode step.
* ``higgs-fused``: the new fused decode kernel; should be << ``higgs-mof``
  and within ~2x of ``baseline-fp8``.
"""

from __future__ import annotations

import argparse
import math
import statistics
from typing import Callable, Dict, List, Tuple

import torch

try:
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        decode_attention_fwd,
    )
    from sglang.srt.layers.attention.triton_ops.higgs_decode_attention import (
        HIGGS_HEAD_DIM,
        decode_attention_fwd_higgs,
    )
    from sglang.srt.layers.quantization.higgs_mha_2bit_kv import (
        HiggsMHA2BitCodec,
        HiggsMHA2BitConfig,
    )
except ImportError as exc:
    raise SystemExit(
        f"Microbench requires sglang in the importable path: {exc}"
    )


def _time_kernel(fn: Callable[[], None], n_iters: int = 50, n_warmup: int = 5) -> float:
    """Return the median per-call latency in microseconds."""
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(n_iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)  # ms -> us
    return statistics.median(times)


def _make_inputs(
    batch: int, seq_len: int, device: torch.device
) -> Tuple[Dict[str, torch.Tensor], HiggsMHA2BitCodec]:
    head_dim = HIGGS_HEAD_DIM
    num_q_heads = 32
    num_kv_heads = 2

    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_bf16 = torch.randn(
        batch * seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v_bf16 = torch.randn_like(k_bf16)

    codec = HiggsMHA2BitCodec(HiggsMHA2BitConfig(head_dim=head_dim), device=device)
    k_packed = codec.compress(k_bf16)
    v_packed = codec.compress(v_bf16)

    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = seq_len * torch.arange(
        1, batch + 1, dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(
        batch * seq_len, dtype=torch.int64, device=device
    )

    max_kv_splits = 8
    num_kv_splits = torch.full(
        (batch,), max_kv_splits, dtype=torch.int32, device=device
    )

    return (
        {
            "q": q,
            "k_bf16": k_bf16,
            "v_bf16": v_bf16,
            "k_packed": k_packed,
            "v_packed": v_packed,
            "kv_indptr": kv_indptr,
            "kv_indices": kv_indices,
            "num_kv_splits": num_kv_splits,
            "max_kv_splits": max_kv_splits,
            "head_dim": head_dim,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "batch": batch,
            "seq_len": seq_len,
        },
        codec,
    )


def _make_workspace(inputs: Dict[str, torch.Tensor]):
    batch = inputs["batch"]
    num_q_heads = inputs["num_q_heads"]
    head_dim = inputs["head_dim"]
    max_kv_splits = inputs["max_kv_splits"]
    attn_logits = torch.zeros(
        batch,
        num_q_heads,
        max_kv_splits,
        head_dim,
        device=inputs["q"].device,
        dtype=torch.bfloat16,
    )
    attn_lse = torch.zeros(
        batch,
        num_q_heads,
        max_kv_splits,
        device=inputs["q"].device,
        dtype=torch.float32,
    )
    o = torch.empty(
        batch,
        num_q_heads,
        head_dim,
        device=inputs["q"].device,
        dtype=torch.bfloat16,
    )
    return attn_logits, attn_lse, o


def bench_baseline_bf16(inputs: Dict[str, torch.Tensor]) -> float:
    # Reference dense BF16 GQA decode (proxy for FP8 — same Triton
    # kernel, same launch shape; FP8 K/V differ only in dtype).
    attn_logits, attn_lse, o = _make_workspace(inputs)
    scaling = 1.0 / math.sqrt(inputs["head_dim"])

    # The dense kernel expects K/V buffers (M, H, head_dim) BF16.
    k_buf = inputs["k_bf16"]
    v_buf = inputs["v_bf16"]

    def _call():
        decode_attention_fwd(
            inputs["q"],
            k_buf,
            v_buf,
            o,
            inputs["kv_indptr"],
            inputs["kv_indices"],
            attn_logits,
            attn_lse,
            inputs["num_kv_splits"],
            inputs["max_kv_splits"],
            scaling,
            1.0,
            1.0,
        )

    return _time_kernel(_call)


def bench_higgs_materialize_on_fetch(
    inputs: Dict[str, torch.Tensor], codec: HiggsMHA2BitCodec
) -> float:
    # The regressed path: decompress the full packed layer cache on
    # every decode step (this is what
    # :meth:`HiggsMHA2BitTokenToKVPool._get_key_buffer` does today
    # before this patch).
    attn_logits, attn_lse, o = _make_workspace(inputs)
    scaling = 1.0 / math.sqrt(inputs["head_dim"])

    def _call():
        k_buf = codec.decompress(inputs["k_packed"], torch.bfloat16)
        v_buf = codec.decompress(inputs["v_packed"], torch.bfloat16)
        decode_attention_fwd(
            inputs["q"],
            k_buf,
            v_buf,
            o,
            inputs["kv_indptr"],
            inputs["kv_indices"],
            attn_logits,
            attn_lse,
            inputs["num_kv_splits"],
            inputs["max_kv_splits"],
            scaling,
            1.0,
            1.0,
        )

    return _time_kernel(_call, n_iters=10, n_warmup=2)


def bench_higgs_fused(inputs: Dict[str, torch.Tensor]) -> float:
    attn_logits, attn_lse, o = _make_workspace(inputs)
    scaling = 1.0 / math.sqrt(inputs["head_dim"])

    def _call():
        decode_attention_fwd_higgs(
            inputs["q"],
            inputs["k_packed"],
            inputs["v_packed"],
            o,
            inputs["kv_indptr"],
            inputs["kv_indices"],
            attn_logits,
            attn_lse,
            inputs["num_kv_splits"],
            inputs["max_kv_splits"],
            scaling,
        )

    return _time_kernel(_call)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=str,
        default="1x1024,4x1024,16x1024,32x1024,8x4096,32x4096",
        help="Comma-separated batch x seq_len list.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    shapes: List[Tuple[int, int]] = []
    for s in args.shapes.split(","):
        b, t = s.split("x")
        shapes.append((int(b), int(t)))

    print(f"{'batch':>6} {'seq_len':>8} {'fp8 baseline (us)':>20} "
          f"{'higgs-mof (us)':>20} {'higgs-fused (us)':>20} "
          f"{'fused/baseline':>16}")
    print("-" * 100)
    for batch, seq_len in shapes:
        inputs, codec = _make_inputs(batch, seq_len, device)
        t_baseline = bench_baseline_bf16(inputs)
        t_mof = bench_higgs_materialize_on_fetch(inputs, codec)
        t_fused = bench_higgs_fused(inputs)
        ratio = t_fused / t_baseline
        print(
            f"{batch:>6} {seq_len:>8} {t_baseline:>20.2f} "
            f"{t_mof:>20.2f} {t_fused:>20.2f} {ratio:>16.2f}x"
        )


if __name__ == "__main__":
    main()
