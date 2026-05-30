"""CUDA-event microbench for the HIGGS DSA + trtllm-gen sparse-MLA pipeline.

ai-blaise #19 iter4 vector B: side-stream HIGGS dequant inside
``_forward_trtllm``. Measures the dequant + trtllm-gen sequence end-to-end
in a single-stream baseline and a side-stream-dequant variant. Records
per-iteration latency via CUDA events so the side-stream variant captures
the actual cross-stream overlap (or lack thereof) on B200.

Production decode shape (REAP B200 deploy, DP=TP=8):
- B = 128 (batch_size after DP attention all-gather)
- num_heads = 16 (128 attention heads / TP=8)
- head_dim = 576 (kv_lora_rank=512 + qk_rope=64)
- top_k = 2048 (sparse_mla_top_k)
- page_size = 64 (trtllm DSA paged KV layout)
- kv dtype: FP8 e4m3fn (Vector A path enabled)

Usage:
    python bench_higgs_trtllm_dsa_iter4.py --iters 100 --warmup 20

Reports two lines:
    - baseline: dequant + trtllm-gen on a single stream (iter3 path)
    - sidestream: dequant on a dedicated stream, trtllm-gen waits via
      event (iter4 vector B path)

The arithmetic delta = (baseline - sidestream) median * 61 layers gives
the projected per-step TPOT savings.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch


def _build_inputs(B, num_heads, head_dim, top_k, num_slots, page_size, device, kv_dtype):
    """Build production-shaped tensors for the dequant + trtllm pipeline."""
    from sglang.srt.layers.quantization.higgs_dense_2bit_kv import (
        HiggsDense2BitConfig,
    )

    cfg = HiggsDense2BitConfig(latent_dim=512, rope_dim=64)
    slot_bytes = cfg.slot_bytes
    # Iter4 (#16) bumped the HIGGS dense slot stride from 258 to 272
    # (258 B payload + 14 B 16-align pad for ``cp.async.16``).
    assert slot_bytes == 272, f"unexpected slot_bytes {slot_bytes}"

    # Compressed HIGGS K cache for one layer (per the production layout
    # of HiggsDense2BitDSATokenToKVPool).
    compressed = torch.randint(
        0, 255, (num_slots, 1, slot_bytes), device=device, dtype=torch.uint8
    )

    # page_table from indexer top-k. -1 marks padding (none here so we
    # measure full dequant work).
    flat = torch.arange(B * top_k, device=device, dtype=torch.int32)
    page_table = (flat % num_slots).reshape(B, top_k).contiguous()

    # Codebook for EDEN2-16 lattice. Random values; correctness not
    # required for microbench.
    codebook = torch.randn(16, 2, device=device, dtype=torch.float32)

    # Compact buffers — these are the actual targets the dequant writes
    # and the trtllm-gen reads. Allocate once and reuse to mimic the
    # production memory_pool reuse pattern.
    compact_page_table = torch.empty_like(page_table)
    kv_compact = torch.empty(
        (B * top_k, 1, head_dim), device=device, dtype=kv_dtype
    )

    # Query — FP8 saturating cast as in the iter3 vector A path.
    q_bf16 = torch.randn(B, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn) if kv_dtype == torch.float8_e4m3fn else q_bf16

    seq_lens = torch.full((B,), top_k, dtype=torch.int32, device=device)

    return {
        "compressed": compressed,
        "page_table": page_table,
        "kv_compact": kv_compact,
        "compact_page_table": compact_page_table,
        "codebook": codebook,
        "q": q_fp8,
        "seq_lens": seq_lens,
    }


def _dequant_and_attn(
    inputs,
    page_size,
    B,
    num_heads,
    head_dim,
    top_k,
    side_stream=None,
    use_fp8=True,
):
    """Run dequant + trtllm-gen sparse-MLA one shot."""
    from sglang.jit_kernel.higgs_dense_2bit import (
        dequantize_higgs_dense_2bit_page_table,
        dequantize_higgs_dense_2bit_page_table_fp8,
    )
    import flashinfer.decode

    if side_stream is not None:
        current_stream = torch.cuda.current_stream()
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            if use_fp8:
                dequantize_higgs_dense_2bit_page_table_fp8(
                    inputs["compressed"],
                    inputs["page_table"],
                    inputs["kv_compact"],
                    inputs["compact_page_table"],
                    inputs["codebook"],
                    1.0,
                )
            else:
                dequantize_higgs_dense_2bit_page_table(
                    inputs["compressed"],
                    inputs["page_table"],
                    inputs["kv_compact"],
                    inputs["compact_page_table"],
                    inputs["codebook"],
                )
            dequant_event = side_stream.record_event()
        current_stream.wait_event(dequant_event)
    else:
        if use_fp8:
            dequantize_higgs_dense_2bit_page_table_fp8(
                inputs["compressed"],
                inputs["page_table"],
                inputs["kv_compact"],
                inputs["compact_page_table"],
                inputs["codebook"],
                1.0,
            )
        else:
            dequantize_higgs_dense_2bit_page_table(
                inputs["compressed"],
                inputs["page_table"],
                inputs["kv_compact"],
                inputs["compact_page_table"],
                inputs["codebook"],
            )

    # Reshape compact → paged for trtllm-gen.
    kv_paged = inputs["kv_compact"].view(
        B * top_k // page_size, page_size, head_dim
    ).unsqueeze(1)
    block_tables = inputs["compact_page_table"].unsqueeze(1)

    out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=inputs["q"].view(B, 1, num_heads, head_dim),
        kv_cache=kv_paged,
        workspace_buffer=_workspace_buffer(),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=block_tables,
        seq_lens=inputs["seq_lens"],
        max_seq_len=top_k,
        sparse_mla_top_k=top_k,
        bmm1_scale=1.0,
        backend="trtllm-gen",
    )
    return out


_WORKSPACE = None


def _workspace_buffer():
    global _WORKSPACE
    if _WORKSPACE is None:
        # 128 MiB workspace, ample for production shapes.
        _WORKSPACE = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return _WORKSPACE


def _bench_one(name, side_stream, inputs, page_size, B, num_heads, head_dim, top_k,
               iters, warmup, use_fp8):
    # Warmup.
    for _ in range(warmup):
        _dequant_and_attn(
            inputs, page_size, B, num_heads, head_dim, top_k,
            side_stream=side_stream, use_fp8=use_fp8,
        )
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        _dequant_and_attn(
            inputs, page_size, B, num_heads, head_dim, top_k,
            side_stream=side_stream, use_fp8=use_fp8,
        )
        ends[i].record()
    torch.cuda.synchronize()

    times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    return {
        "name": name,
        "median_us": statistics.median(times_us),
        "min_us": min(times_us),
        "max_us": max(times_us),
        "mean_us": statistics.mean(times_us),
        "p10_us": statistics.quantiles(times_us, n=10)[0] if iters >= 10 else min(times_us),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=576)
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--num-slots", type=int, default=65536)
    ap.add_argument("--use-fp8", action="store_true", default=True)
    ap.add_argument("--use-bf16", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    kv_dtype = torch.bfloat16 if args.use_bf16 else torch.float8_e4m3fn
    use_fp8 = not args.use_bf16

    inputs = _build_inputs(
        args.B, args.num_heads, args.head_dim, args.top_k,
        args.num_slots, args.page_size, device, kv_dtype,
    )

    # Initialize JIT modules + workspace once.
    _dequant_and_attn(
        inputs, args.page_size, args.B, args.num_heads, args.head_dim, args.top_k,
        side_stream=None, use_fp8=use_fp8,
    )
    torch.cuda.synchronize()

    side_stream = torch.cuda.Stream()

    baseline = _bench_one(
        "baseline", None, inputs, args.page_size,
        args.B, args.num_heads, args.head_dim, args.top_k,
        args.iters, args.warmup, use_fp8,
    )
    sidestream = _bench_one(
        "sidestream", side_stream, inputs, args.page_size,
        args.B, args.num_heads, args.head_dim, args.top_k,
        args.iters, args.warmup, use_fp8,
    )

    results = [baseline, sidestream]
    delta_us = baseline["median_us"] - sidestream["median_us"]
    delta_pct = 100.0 * delta_us / baseline["median_us"]
    layers = 61  # DSV3.2-REAP-345B MoE layers
    projected_tpot_savings_ms = delta_us * layers / 1000.0

    if args.json:
        print(json.dumps({
            "results": results,
            "delta_us": delta_us,
            "delta_pct": delta_pct,
            "projected_tpot_savings_ms_61L": projected_tpot_savings_ms,
            "kv_dtype": "fp8_e4m3" if use_fp8 else "bf16",
            "shape": dict(B=args.B, num_heads=args.num_heads, head_dim=args.head_dim,
                          top_k=args.top_k, page_size=args.page_size),
        }, indent=2))
    else:
        print(f"# B={args.B} num_heads={args.num_heads} head_dim={args.head_dim} "
              f"top_k={args.top_k} page_size={args.page_size} "
              f"kv_dtype={'fp8_e4m3' if use_fp8 else 'bf16'}")
        hdr = "{:>12} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "name", "median_us", "p10_us", "mean_us", "min_us", "max_us"
        )
        print(hdr)
        for r in results:
            print(
                "{:>12} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
                    r["name"], r["median_us"], r["p10_us"],
                    r["mean_us"], r["min_us"], r["max_us"],
                )
            )
        print(f"# delta (baseline - sidestream) median: {delta_us:.2f} us "
              f"({delta_pct:+.1f}%)")
        print(f"# projected TPOT savings @ 61L = {projected_tpot_savings_ms:.2f} ms")


if __name__ == "__main__":
    main()
