"""CUDA-event microbench for the HIGGS DSA + trtllm-gen dedicated-stream pipeline.

ai-blaise #19 iter6: trtllm-gen on dedicated CUDA stream.

This bench simulates the **inter-layer work pattern** of the production
decode loop. Per layer L the production main stream does:

  1. ``kv_proj_L`` (Q/K/V projection — small GEMM)
  2. ``set_mla_kv_buffer_L`` (compact write to per-layer K cache)
  3. ``transform_page_table_1_L`` (small reindex)
  4. ``dequant_L`` (HIGGS sparse-materialize)        # ~3 ms HBM
  5. ``trtllm_gen_attn_L`` (sparse-MLA)              # ~3 ms HBM
  6. ``o_proj_L`` + ``mlp_L``                         # small / not modeled

Steps 4 and 5 are the two big HBM-bound kernels. The other steps are
~us-scale GEMMs. To make the simulation honest, we model the small
GEMMs as a torch.bmm or a small launch that consumes one main-stream
launch slot but ~no wall-clock time.

Four variants are timed:

  1. ``baseline``        — single stream, single compact buffer.
                           Serial: kv_proj → set_kv → page_table_1 →
                           dequant → trtllm-gen → o_proj → mlp.
  2. ``sidestream``      — iter4 vector B: dequant on side_stream, single
                           compact buffer. trtllm-gen back on main.
  3. ``pingpong``        — iter5 primary vector: side-stream dequant +
                           TWO compact buffers, indexed by ``layer & 1``.
                           Side-stream's ``wait_stream(main)`` still
                           pulls in the prior trtllm-gen via main-stream
                           FIFO ordering.
  4. ``dedicated_stream`` — iter6 primary vector: side-stream dequant +
                            ping-pong buffers + trtllm-gen launched on
                            its OWN dedicated stream. Main-stream queue
                            after the dedicated stream wire-up only
                            contains the small set_mla_kv_buffer +
                            page_table_1 + kv_proj kernels; the
                            side-stream's wait_stream(main) for layer
                            N+1 only blocks on those (no prior
                            trtllm-gen tail). Layer N+1's dequant runs
                            concurrently with layer N's trtllm-gen.

Production decode shape (REAP B200 deploy, DP=TP=8):
- B = 128, num_heads = 16, head_dim = 576
- top_k = 2048, page_size = 64
- kv dtype: FP8 e4m3fn

The bench reports per-iteration (= four consecutive layers) median
latency, plus the projected per-step TPOT delta at 61 MoE layers from
the variant-vs-baseline median delta * (61/4).

Usage:
    python bench_higgs_trtllm_dsa_iter6.py --iters 100 --warmup 20
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
    """Build production-shaped tensors for a 4-layer cycle.

    Four HIGGS-packed K caches (one per simulated layer in the cycle).
    Two ping-pong compact slots (since layers alternate parity).
    """
    from sglang.srt.layers.quantization.higgs_dense_2bit_kv import (
        HiggsDense2BitConfig,
    )

    cfg = HiggsDense2BitConfig(latent_dim=512, rope_dim=64)
    slot_bytes = cfg.slot_bytes
    assert slot_bytes == 272, f"unexpected slot_bytes {slot_bytes}"

    # Four HIGGS-packed K caches, one per simulated layer in the 4-layer cycle.
    compressed = [
        torch.randint(
            0, 255, (num_slots, 1, slot_bytes), device=device, dtype=torch.uint8
        )
        for _ in range(4)
    ]

    flat = torch.arange(B * top_k, device=device, dtype=torch.int32)
    page_table = (flat % num_slots).reshape(B, top_k).contiguous()

    codebook = torch.randn(16, 2, device=device, dtype=torch.float32)

    # Single compact buf (baseline + sidestream variants).
    compact_page_table_single = torch.empty_like(page_table)
    kv_compact_single = torch.empty(
        (B * top_k, 1, head_dim), device=device, dtype=kv_dtype
    )

    # Two compact bufs (pingpong + dedicated_stream variants).
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

    # ai-blaise #19 iter6: small main-stream simulation tensors. These
    # model the kv_proj GEMM + set_mla_kv_buffer write that happens on
    # main between layers in production. Numbers chosen to roughly
    # match production: kv_proj is B x 576 -> B x 576 (small at B=128),
    # set_mla_kv_buffer copies B compressed slots (272 B each).
    q_proj_small = torch.randn(576, 576, device=device, dtype=torch.bfloat16)
    q_proj_in = torch.randn(B, 576, device=device, dtype=torch.bfloat16)
    k_cache_layer = torch.zeros(
        num_slots, 1, slot_bytes, device=device, dtype=torch.uint8
    )
    k_locs = torch.arange(B, device=device, dtype=torch.int64)
    k_new = torch.randint(
        0, 255, (B, 1, slot_bytes), device=device, dtype=torch.uint8
    )

    return {
        "compressed": compressed,  # list[4]
        "page_table": page_table,
        "kv_compact_single": kv_compact_single,
        "compact_page_table_single": compact_page_table_single,
        "kv_compact_pp": kv_compact_pp,  # list[2]
        "compact_page_table_pp": compact_page_table_pp,  # list[2]
        "codebook": codebook,
        "q": q_fp8,
        "seq_lens": seq_lens,
        # iter6 main-stream sim:
        "q_proj_small": q_proj_small,
        "q_proj_in": q_proj_in,
        "k_cache_layer": k_cache_layer,
        "k_locs": k_locs,
        "k_new": k_new,
    }


_WORKSPACE = None


def _workspace_buffer():
    global _WORKSPACE
    if _WORKSPACE is None:
        _WORKSPACE = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return _WORKSPACE


def _simulate_main_stream_inter_layer_work(inputs, layer_id, mode="kv_proj_and_set"):
    """Simulate the small main-stream work between layers (kv_proj + set_kv).

    The production decode does a few GEMMs per layer on the main stream
    (Q/K/V projection, K cache write, page_table_1 transform). These are
    individually small (sub-us GPU time at B=128 since the dimensions are
    small) but they occupy main-stream launch slots and pull the side
    stream's wait_stream synchronization point forward.

    We simulate this with a small bmm + index_copy. Realistic enough
    that the relative ordering matters.
    """
    if mode == "kv_proj_and_set":
        # Small GEMM (faux Q projection): B=128, n=576, k=576.
        # On B200 this is ~10us.
        q_proj = inputs["q_proj_small"]
        q_in = inputs["q_proj_in"]
        # Out gets reused, just record-stream effect.
        _ = torch.matmul(q_in, q_proj)
        # Faux set_mla_kv_buffer write — small index_copy.
        inputs["k_cache_layer"][inputs["k_locs"]] = inputs["k_new"]
    return None


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
    trtllm_stream,
    variant,
):
    """One layer's inter-layer-work + dequant + trtllm-gen.

    variant is one of:
      - 'baseline'
      - 'sidestream'
      - 'pingpong'
      - 'dedicated_stream'

    Returns the trtllm-gen output (kept alive on whichever stream ran it).
    """
    from sglang.jit_kernel.higgs_dense_2bit import (
        dequantize_higgs_dense_2bit_page_table,
        dequantize_higgs_dense_2bit_page_table_fp8,
    )
    import flashinfer.decode

    compressed = inputs["compressed"][layer_id & 3]

    if variant in ("pingpong", "dedicated_stream"):
        parity = layer_id & 1
        kv_compact = inputs["kv_compact_pp"][parity]
        compact_pt = inputs["compact_page_table_pp"][parity]
    else:
        kv_compact = inputs["kv_compact_single"]
        compact_pt = inputs["compact_page_table_single"]

    # ai-blaise #19 iter6: simulate the small main-stream work per layer
    # (kv_proj + set_mla_kv_buffer). These are launched on main; the
    # side stream's wait_stream(main) waits for them. With dedicated
    # trtllm stream, prior trtllm-gen is NOT on main, so this wait is
    # cheap. Without dedicated stream, prior trtllm-gen is on main, so
    # this wait blocks until prior trtllm-gen finishes (FIFO ordering).
    _simulate_main_stream_inter_layer_work(inputs, layer_id)

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
        main_stream = torch.cuda.current_stream()
        side_stream.wait_stream(main_stream)
        with torch.cuda.stream(side_stream):
            _do_dequant()
            dequant_event = side_stream.record_event()

    # Reshape compact → paged.
    kv_paged = (
        kv_compact.view(B * top_k // page_size, page_size, head_dim).unsqueeze(1)
    )
    block_tables = compact_pt.unsqueeze(1)

    # ai-blaise #19 iter6: choose where trtllm-gen runs.
    if variant == "dedicated_stream":
        # trtllm-gen on its own stream. The stream waits on the
        # dequant_event (records cross-stream causal dep). Main stream
        # only ran the small kv_proj + set_kv, so the next layer's
        # side-stream wait_stream(main) is much shorter.
        main_stream = torch.cuda.current_stream()
        # The main stream's pending work (small kernels) must be visible
        # to trtllm before it reads K cache. wait_stream(main) is cheap
        # because main hasn't queued the prior trtllm-gen.
        trtllm_stream.wait_stream(main_stream)
        # Causal dep: trtllm-gen must wait for the dequant write.
        trtllm_stream.wait_event(dequant_event)
        with torch.cuda.stream(trtllm_stream):
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
        # NOTE(iter6): do NOT main_stream.wait_event(trtllm_done) here.
        # In production, `out` flows to o_proj on the main stream — that's
        # where the main-stream sync to trtllm-gen happens. In this bench
        # we approximate by syncing only once at the END of the 4-layer
        # sequence (in _bench_four_layer), simulating that each layer's
        # `out` is consumed lazily by downstream ops on the same trtllm
        # stream or via a deferred main-stream consumer.
    else:
        # baseline / sidestream / pingpong: trtllm-gen on main.
        if dequant_event is not None:
            torch.cuda.current_stream().wait_event(dequant_event)
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


def _bench_four_layer(
    name,
    variant,
    side_stream,
    trtllm_stream,
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
    """Bench a 4-layer back-to-back dequant + trtllm-gen sequence."""

    # Warmup.
    for _ in range(warmup):
        for lid in range(4):
            _dequant_and_attn_layer(
                layer_id=lid,
                inputs=inputs,
                page_size=page_size,
                B=B,
                num_heads=num_heads,
                head_dim=head_dim,
                top_k=top_k,
                use_fp8=use_fp8,
                side_stream=side_stream,
                trtllm_stream=trtllm_stream,
                variant=variant,
            )
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        for lid in range(4):
            _dequant_and_attn_layer(
                layer_id=lid,
                inputs=inputs,
                page_size=page_size,
                B=B,
                num_heads=num_heads,
                head_dim=head_dim,
                top_k=top_k,
                use_fp8=use_fp8,
                side_stream=side_stream,
                trtllm_stream=trtllm_stream,
                variant=variant,
            )
        # For dedicated_stream: end-of-sequence sync of trtllm onto main
        # so the ``ends[i].record()`` event correctly captures the tail of
        # the last trtllm-gen launch. In production this is the final
        # ``main.wait_event(trtllm_done)`` before the next pipeline stage
        # (e.g. logits + sampling) consumes the per-layer outputs.
        if variant == "dedicated_stream":
            torch.cuda.current_stream().wait_stream(trtllm_stream)
        ends[i].record()
    torch.cuda.synchronize()

    times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    return {
        "name": name,
        "variant": variant,
        "median_us_per_4layer": statistics.median(times_us),
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
        trtllm_stream=None,
        variant="baseline",
    )
    torch.cuda.synchronize()

    side_stream = torch.cuda.Stream()
    trtllm_stream = torch.cuda.Stream()

    baseline = _bench_four_layer(
        "baseline",
        "baseline",
        None,
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
    sidestream = _bench_four_layer(
        "sidestream",
        "sidestream",
        side_stream,
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
    pingpong = _bench_four_layer(
        "pingpong",
        "pingpong",
        side_stream,
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
    dedicated = _bench_four_layer(
        "dedicated_stream",
        "dedicated_stream",
        side_stream,
        trtllm_stream,
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

    results = [baseline, sidestream, pingpong, dedicated]
    # Projected TPOT delta at 61 layers: median delta us per 4layer * (61/4).
    layers = 61
    fours = layers / 4.0

    def _delta(other):
        delta = baseline["median_us_per_4layer"] - other["median_us_per_4layer"]
        return (
            delta,
            100.0 * delta / baseline["median_us_per_4layer"],
            delta * fours / 1000.0,
        )

    side_delta_us, side_delta_pct, side_tpot_ms = _delta(sidestream)
    pp_delta_us, pp_delta_pct, pp_tpot_ms = _delta(pingpong)
    ded_delta_us, ded_delta_pct, ded_tpot_ms = _delta(dedicated)

    if args.json:
        print(json.dumps({
            "results": results,
            "deltas_vs_baseline": {
                "sidestream": {
                    "delta_us_per_4layer": side_delta_us,
                    "delta_pct": side_delta_pct,
                    "projected_tpot_savings_ms_61L": side_tpot_ms,
                },
                "pingpong": {
                    "delta_us_per_4layer": pp_delta_us,
                    "delta_pct": pp_delta_pct,
                    "projected_tpot_savings_ms_61L": pp_tpot_ms,
                },
                "dedicated_stream": {
                    "delta_us_per_4layer": ded_delta_us,
                    "delta_pct": ded_delta_pct,
                    "projected_tpot_savings_ms_61L": ded_tpot_ms,
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
        print("{:>18} {:>18} {:>10} {:>10} {:>10} {:>10}".format(
            "name", "median_us/4layer", "p10_us", "mean_us", "min_us", "max_us"
        ))
        for r in results:
            print(
                "{:>18} {:>18.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
                    r["name"],
                    r["median_us_per_4layer"],
                    r["p10_us"],
                    r["mean_us"],
                    r["min_us"],
                    r["max_us"],
                )
            )
        print(
            f"# sidestream vs baseline:       {side_delta_us:+8.2f} us/4layer "
            f"({side_delta_pct:+6.1f}%); proj TPOT @ 61L = {side_tpot_ms:+6.2f} ms"
        )
        print(
            f"# pingpong vs baseline:         {pp_delta_us:+8.2f} us/4layer "
            f"({pp_delta_pct:+6.1f}%); proj TPOT @ 61L = {pp_tpot_ms:+6.2f} ms"
        )
        print(
            f"# dedicated_stream vs baseline: {ded_delta_us:+8.2f} us/4layer "
            f"({ded_delta_pct:+6.1f}%); proj TPOT @ 61L = {ded_tpot_ms:+6.2f} ms"
        )


if __name__ == "__main__":
    main()
