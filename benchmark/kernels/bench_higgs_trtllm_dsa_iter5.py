"""CUDA-event microbench for the HIGGS DSA + trtllm-gen ping-pong pipeline.

ai-blaise #19 iter5 (primary vector): ping-pong compact dequant buffers.
This bench measures a **two-layer alternating** dequant + trtllm-gen
sequence — the smallest setup that exposes the cross-layer write/read
aliasing hazard the ping-pong is designed to break. The single-layer
microbench in :file:`bench_higgs_trtllm_dsa_iter4.py` cannot capture
the ping-pong benefit because the side-stream's
``wait_stream(current_stream)`` only blocks on intra-iteration work.

Three variants are timed:

  1. ``baseline``     — single stream, single compact buffer. No overlap.
  2. ``sidestream``   — iter4 vector B: side-stream dequant, single
                        compact buffer. Side stream's ``wait_stream`` still
                        pulls in the prior trtllm-gen via main-stream
                        queue ordering — overlap window is tiny.
  3. ``pingpong``     — iter5 primary vector: side-stream dequant +
                        TWO compact buffers, indexed by ``layer & 1``.
                        Eliminates the buffer aliasing hazard between
                        adjacent layers. Whether this delivers wall-clock
                        savings depends on the ``wait_stream`` sync
                        relaxation; see report annotations.

Production decode shape (REAP B200 deploy, DP=TP=8):
- B = 128, num_heads = 16, head_dim = 576
- top_k = 2048, page_size = 64
- kv dtype: FP8 e4m3fn

The bench reports per-iteration (= one pair of consecutive layers)
median latency, plus the projected per-step TPOT delta at 61 MoE
layers from the variant-vs-baseline median delta * 30 (one pair per
two layers).

Usage:
    python bench_higgs_trtllm_dsa_iter5.py --iters 100 --warmup 20
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch


def _build_inputs(
    B,
    num_heads,
    head_dim,
    top_k,
    num_slots,
    page_size,
    device,
    kv_dtype,
):
    """Build production-shaped tensors. Returns two parities of compact bufs.

    The two ``compressed`` buffers simulate two consecutive layers (the
    layer-buffer in the production pool is per-layer). The two
    ``kv_compact_pp`` + ``compact_page_table_pp`` slots simulate the
    iter5 ping-pong pool attrs.
    """
    from sglang.srt.layers.quantization.higgs_dense_2bit_kv import (
        HiggsDense2BitConfig,
    )

    cfg = HiggsDense2BitConfig(latent_dim=512, rope_dim=64)
    slot_bytes = cfg.slot_bytes
    assert slot_bytes == 258, f"unexpected slot_bytes {slot_bytes}"

    # Two HIGGS-packed K caches, one per simulated layer.
    compressed = [
        torch.randint(
            0, 255, (num_slots, 1, slot_bytes), device=device, dtype=torch.uint8
        )
        for _ in range(2)
    ]

    flat = torch.arange(B * top_k, device=device, dtype=torch.int32)
    page_table = (flat % num_slots).reshape(B, top_k).contiguous()

    codebook = torch.randn(16, 2, device=device, dtype=torch.float32)

    # Single compact buf (baseline + sidestream variants).
    compact_page_table_single = torch.empty_like(page_table)
    kv_compact_single = torch.empty(
        (B * top_k, 1, head_dim), device=device, dtype=kv_dtype
    )

    # Two compact bufs (pingpong variant).
    compact_page_table_pp = [
        torch.empty_like(page_table) for _ in range(2)
    ]
    kv_compact_pp = [
        torch.empty(
            (B * top_k, 1, head_dim), device=device, dtype=kv_dtype
        )
        for _ in range(2)
    ]

    q_bf16 = torch.randn(B, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    q_fp8 = (
        q_bf16.to(torch.float8_e4m3fn)
        if kv_dtype == torch.float8_e4m3fn
        else q_bf16
    )
    seq_lens = torch.full((B,), top_k, dtype=torch.int32, device=device)

    return {
        "compressed": compressed,  # list[2]
        "page_table": page_table,
        "kv_compact_single": kv_compact_single,
        "compact_page_table_single": compact_page_table_single,
        "kv_compact_pp": kv_compact_pp,  # list[2]
        "compact_page_table_pp": compact_page_table_pp,  # list[2]
        "codebook": codebook,
        "q": q_fp8,
        "seq_lens": seq_lens,
    }


_WORKSPACE = None


def _workspace_buffer():
    global _WORKSPACE
    if _WORKSPACE is None:
        _WORKSPACE = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return _WORKSPACE


def _dequant_and_attn_layer(
    *,
    layer_id,
    inputs,
    page_size,
    B,
    num_heads,
    head_dim,
    top_k,
    use_fp8,
    side_stream,
    variant,
):
    """One layer's dequant + trtllm-gen.

    variant is one of: 'baseline', 'sidestream', 'pingpong'.
    """
    from sglang.jit_kernel.higgs_dense_2bit import (
        dequantize_higgs_dense_2bit_page_table,
        dequantize_higgs_dense_2bit_page_table_fp8,
    )
    import flashinfer.decode

    compressed = inputs["compressed"][layer_id & 1]

    if variant == "pingpong":
        parity = layer_id & 1
        kv_compact = inputs["kv_compact_pp"][parity]
        compact_pt = inputs["compact_page_table_pp"][parity]
    else:
        kv_compact = inputs["kv_compact_single"]
        compact_pt = inputs["compact_page_table_single"]

    def _do_dequant():
        if use_fp8:
            dequantize_higgs_dense_2bit_page_table_fp8(
                compressed,
                inputs["page_table"],
                kv_compact,
                compact_pt,
                inputs["codebook"],
                1.0,
            )
        else:
            dequantize_higgs_dense_2bit_page_table(
                compressed,
                inputs["page_table"],
                kv_compact,
                compact_pt,
                inputs["codebook"],
            )

    if variant == "baseline":
        _do_dequant()
        dequant_event = None
    else:
        current_stream = torch.cuda.current_stream()
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            _do_dequant()
            dequant_event = side_stream.record_event()
        current_stream.wait_event(dequant_event)

    # Reshape compact → paged.
    kv_paged = (
        kv_compact.view(B * top_k // page_size, page_size, head_dim).unsqueeze(1)
    )
    block_tables = compact_pt.unsqueeze(1)

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


def _bench_two_layer(
    name,
    variant,
    side_stream,
    inputs,
    page_size,
    B,
    num_heads,
    head_dim,
    top_k,
    iters,
    warmup,
    use_fp8,
):
    """Bench a back-to-back two-layer dequant + trtllm-gen sequence."""

    # Warmup.
    for _ in range(warmup):
        _dequant_and_attn_layer(
            layer_id=0,
            inputs=inputs,
            page_size=page_size,
            B=B,
            num_heads=num_heads,
            head_dim=head_dim,
            top_k=top_k,
            use_fp8=use_fp8,
            side_stream=side_stream,
            variant=variant,
        )
        _dequant_and_attn_layer(
            layer_id=1,
            inputs=inputs,
            page_size=page_size,
            B=B,
            num_heads=num_heads,
            head_dim=head_dim,
            top_k=top_k,
            use_fp8=use_fp8,
            side_stream=side_stream,
            variant=variant,
        )
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        _dequant_and_attn_layer(
            layer_id=0,
            inputs=inputs,
            page_size=page_size,
            B=B,
            num_heads=num_heads,
            head_dim=head_dim,
            top_k=top_k,
            use_fp8=use_fp8,
            side_stream=side_stream,
            variant=variant,
        )
        _dequant_and_attn_layer(
            layer_id=1,
            inputs=inputs,
            page_size=page_size,
            B=B,
            num_heads=num_heads,
            head_dim=head_dim,
            top_k=top_k,
            use_fp8=use_fp8,
            side_stream=side_stream,
            variant=variant,
        )
        ends[i].record()
    torch.cuda.synchronize()

    times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    return {
        "name": name,
        "variant": variant,
        "median_us_per_pair": statistics.median(times_us),
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
    ap.add_argument("--use-bf16", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    kv_dtype = torch.bfloat16 if args.use_bf16 else torch.float8_e4m3fn
    use_fp8 = not args.use_bf16

    inputs = _build_inputs(
        args.B,
        args.num_heads,
        args.head_dim,
        args.top_k,
        args.num_slots,
        args.page_size,
        device,
        kv_dtype,
    )

    # Initialize JIT modules + workspace once.
    _dequant_and_attn_layer(
        layer_id=0,
        inputs=inputs,
        page_size=args.page_size,
        B=args.B,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        top_k=args.top_k,
        use_fp8=use_fp8,
        side_stream=None,
        variant="baseline",
    )
    torch.cuda.synchronize()

    side_stream = torch.cuda.Stream()

    baseline = _bench_two_layer(
        "baseline",
        "baseline",
        None,
        inputs,
        args.page_size,
        args.B,
        args.num_heads,
        args.head_dim,
        args.top_k,
        args.iters,
        args.warmup,
        use_fp8,
    )
    sidestream = _bench_two_layer(
        "sidestream",
        "sidestream",
        side_stream,
        inputs,
        args.page_size,
        args.B,
        args.num_heads,
        args.head_dim,
        args.top_k,
        args.iters,
        args.warmup,
        use_fp8,
    )
    pingpong = _bench_two_layer(
        "pingpong",
        "pingpong",
        side_stream,
        inputs,
        args.page_size,
        args.B,
        args.num_heads,
        args.head_dim,
        args.top_k,
        args.iters,
        args.warmup,
        use_fp8,
    )

    results = [baseline, sidestream, pingpong]
    # Projected TPOT delta at 61 layers: median delta us per pair * 30.5 pairs.
    layers = 61
    pairs = layers / 2.0

    def _delta(other):
        delta = baseline["median_us_per_pair"] - other["median_us_per_pair"]
        return delta, 100.0 * delta / baseline["median_us_per_pair"], delta * pairs / 1000.0

    side_delta_us, side_delta_pct, side_tpot_ms = _delta(sidestream)
    pp_delta_us, pp_delta_pct, pp_tpot_ms = _delta(pingpong)

    if args.json:
        print(json.dumps({
            "results": results,
            "deltas_vs_baseline": {
                "sidestream": {
                    "delta_us_per_pair": side_delta_us,
                    "delta_pct": side_delta_pct,
                    "projected_tpot_savings_ms_61L": side_tpot_ms,
                },
                "pingpong": {
                    "delta_us_per_pair": pp_delta_us,
                    "delta_pct": pp_delta_pct,
                    "projected_tpot_savings_ms_61L": pp_tpot_ms,
                },
            },
            "kv_dtype": "fp8_e4m3" if use_fp8 else "bf16",
            "shape": dict(B=args.B, num_heads=args.num_heads, head_dim=args.head_dim,
                          top_k=args.top_k, page_size=args.page_size),
        }, indent=2))
    else:
        print(f"# B={args.B} num_heads={args.num_heads} head_dim={args.head_dim} "
              f"top_k={args.top_k} page_size={args.page_size} "
              f"kv_dtype={'fp8_e4m3' if use_fp8 else 'bf16'}")
        print("{:>12} {:>16} {:>10} {:>10} {:>10} {:>10}".format(
            "name", "median_us/pair", "p10_us", "mean_us", "min_us", "max_us"
        ))
        for r in results:
            print(
                "{:>12} {:>16.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
                    r["name"],
                    r["median_us_per_pair"],
                    r["p10_us"],
                    r["mean_us"],
                    r["min_us"],
                    r["max_us"],
                )
            )
        print(
            f"# sidestream vs baseline: {side_delta_us:.2f} us/pair "
            f"({side_delta_pct:+.1f}%); proj TPOT @ 61L = {side_tpot_ms:.2f} ms"
        )
        print(
            f"# pingpong vs baseline:   {pp_delta_us:.2f} us/pair "
            f"({pp_delta_pct:+.1f}%); proj TPOT @ 61L = {pp_tpot_ms:.2f} ms"
        )


if __name__ == "__main__":
    main()
