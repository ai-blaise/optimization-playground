"""CUDA-event microbench for the HIGGS DSA + depth-4 ping-pong + dual trtllm-gen.

ai-blaise #19 iter7 (tertiary vector): depth-4 ping-pong buffer rotation
and dual dedicated trtllm-gen streams.

This bench is a strict superset of bench_higgs_trtllm_dsa_iter6.py. It
extends the inter-layer simulation from a 4-layer cycle to an 8-layer
cycle so the depth-4 + dual-stream interleaving is fully observable
(two cycles of (A0, B1, A2, B3) produces all four stream/slot pairings
back-to-back), and adds two new variants on top of iter6's four.

Six variants are timed in this bench:

  1. ``baseline``        — single stream, single compact buffer
                           (serial: kv_proj -> set_kv -> page_table_1 ->
                           dequant -> trtllm-gen -> o_proj -> mlp).
  2. ``sidestream``      — iter4 vector B: dequant on side_stream,
                           single compact buffer. trtllm-gen back on main.
  3. ``pingpong``        — iter5 primary vector: side-stream dequant +
                           TWO compact buffers, indexed by ``layer & 1``.
                           Side-stream's wait_stream(main) still pulls
                           in the prior trtllm-gen via main-stream FIFO.
  4. ``dedicated_stream`` — iter6 primary vector: side-stream dequant +
                            depth-2 ping-pong buffers + trtllm-gen
                            launched on its OWN dedicated stream
                            (single stream, no dual).
  5. ``depth4``          — iter7 base: side-stream dequant + dedicated
                            trtllm-gen stream + DEPTH-4 ping-pong slots
                            (``layer & 3`` parity). Single trtllm stream
                            still — so back-to-back trtllm-gens
                            serialize on its FIFO, but the deeper slot
                            rotation lets layer N+2 dequant write to a
                            slot disjoint from N's still-running
                            trtllm-gen (depth-2 had only 2 slots so
                            layer N+2 was forced to wait for layer N).
  6. ``depth4_dual``     — iter7 TERTIARY primary vector: depth-4 +
                            DUAL dedicated trtllm-gen streams. Even
                            layers run trtllm-gen on stream A, odd on
                            stream B. Two adjacent trtllm-gen kernels
                            execute concurrently on disjoint streams,
                            collapsing the FIFO stall.

Production decode shape (REAP B200 deploy, DP=TP=8):
- B = 128, num_heads = 16, head_dim = 576
- top_k = 2048, page_size = 64
- kv dtype: FP8 e4m3fn

The bench reports per-iteration (= eight consecutive layers) median
latency, plus the projected per-step TPOT delta at 61 MoE layers from
the variant-vs-baseline median delta * (61/8).

Usage:
    python bench_higgs_trtllm_dsa_iter7.py --iters 100 --warmup 30
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
    """Build production-shaped tensors for an 8-layer cycle.

    Eight HIGGS-packed K caches (one per simulated layer in the cycle).
    Four ping-pong compact slots for depth-4 + dual_stream variants;
    two for the depth-2 baseline variants; one for the truly-baseline
    variant.
    """
    from sglang.srt.layers.quantization.higgs_dense_2bit_kv import (
        HiggsDense2BitConfig,
    )

    cfg = HiggsDense2BitConfig(latent_dim=512, rope_dim=64)
    slot_bytes = cfg.slot_bytes
    assert slot_bytes == 272, f"unexpected slot_bytes {slot_bytes}"

    # Eight HIGGS-packed K caches, one per simulated layer in the
    # 8-layer cycle. Real production has 61 layers but the iter6/iter7
    # cross-layer effects fully cycle in 8 (4 ping-pong slots x 2
    # trtllm streams) so 8 captures the steady-state behaviour.
    compressed = [
        torch.randint(
            0, 255, (num_slots, 1, slot_bytes), device=device, dtype=torch.uint8
        )
        for _ in range(8)
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
    compact_page_table_pp2 = [
        torch.empty_like(page_table) for _ in range(2)
    ]
    kv_compact_pp2 = [
        torch.empty(
            (B * top_k, 1, head_dim), device=device, dtype=kv_dtype
        )
        for _ in range(2)
    ]

    # Four compact bufs (depth4 + depth4_dual variants).
    compact_page_table_pp4 = [
        torch.empty_like(page_table) for _ in range(4)
    ]
    kv_compact_pp4 = [
        torch.empty(
            (B * top_k, 1, head_dim), device=device, dtype=kv_dtype
        )
        for _ in range(4)
    ]

    q_bf16 = torch.randn(B, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    q_fp8 = (
        q_bf16.to(torch.float8_e4m3fn)
        if kv_dtype == torch.float8_e4m3fn
        else q_bf16
    )
    seq_lens = torch.full((B,), top_k, dtype=torch.int32, device=device)

    # iter6/iter7 main-stream simulation: small kv_proj GEMM +
    # set_mla_kv_buffer write that happens on main between layers.
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
        "compressed": compressed,  # list[8]
        "page_table": page_table,
        "kv_compact_single": kv_compact_single,
        "compact_page_table_single": compact_page_table_single,
        "kv_compact_pp2": kv_compact_pp2,  # list[2]
        "compact_page_table_pp2": compact_page_table_pp2,  # list[2]
        "kv_compact_pp4": kv_compact_pp4,  # list[4]
        "compact_page_table_pp4": compact_page_table_pp4,  # list[4]
        "codebook": codebook,
        "q": q_fp8,
        "seq_lens": seq_lens,
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


def _simulate_main_stream_inter_layer_work(inputs):
    """Simulate the small main-stream work between layers.

    See bench_higgs_trtllm_dsa_iter6.py for the rationale.
    """
    q_proj = inputs["q_proj_small"]
    q_in = inputs["q_proj_in"]
    _ = torch.matmul(q_in, q_proj)
    inputs["k_cache_layer"][inputs["k_locs"]] = inputs["k_new"]


def _select_slots(variant, layer_id, inputs):
    """Pick the compact KV + page_table scratch slots for a variant/layer.

    Returns (kv_compact, compact_pt).
    """
    if variant in ("baseline", "sidestream"):
        return inputs["kv_compact_single"], inputs["compact_page_table_single"]
    if variant in ("pingpong", "dedicated_stream"):
        parity = layer_id & 1
        return (
            inputs["kv_compact_pp2"][parity],
            inputs["compact_page_table_pp2"][parity],
        )
    # depth4 / depth4_dual
    parity = layer_id & 3
    return (
        inputs["kv_compact_pp4"][parity],
        inputs["compact_page_table_pp4"][parity],
    )


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
    trtllm_stream_a,
    trtllm_stream_b,
    variant,
):
    """One layer's inter-layer-work + dequant + trtllm-gen.

    variant is one of:
      - 'baseline'
      - 'sidestream'
      - 'pingpong'
      - 'dedicated_stream'
      - 'depth4'              (iter7 base: depth-4 slots, single trtllm stream)
      - 'depth4_dual'         (iter7 TERTIARY: depth-4 + dual trtllm streams)

    Returns the trtllm-gen output (kept alive on whichever stream ran it).
    """
    from sglang.jit_kernel.higgs_dense_2bit import (
        dequantize_higgs_dense_2bit_page_table,
        dequantize_higgs_dense_2bit_page_table_fp8,
    )
    import flashinfer.decode

    # Cycle 8 distinct compressed K caches.
    compressed = inputs["compressed"][layer_id & 7]
    kv_compact, compact_pt = _select_slots(variant, layer_id, inputs)

    # Simulate the small main-stream work (kv_proj + set_mla_kv_buffer).
    _simulate_main_stream_inter_layer_work(inputs)

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

    # Reshape compact -> paged.
    kv_paged = (
        kv_compact.view(B * top_k // page_size, page_size, head_dim).unsqueeze(1)
    )
    block_tables = compact_pt.unsqueeze(1)

    if variant in ("dedicated_stream", "depth4", "depth4_dual"):
        main_stream = torch.cuda.current_stream()
        # Pick the active trtllm-gen stream.
        if variant == "depth4_dual":
            # Even layer -> stream A; odd layer -> stream B.
            active_trtllm = (
                trtllm_stream_a if (layer_id & 1) == 0 else trtllm_stream_b
            )
        else:
            # Single dedicated trtllm-gen stream (iter6 baseline path
            # for variants 4/5).
            active_trtllm = trtllm_stream_a

        active_trtllm.wait_stream(main_stream)
        active_trtllm.wait_event(dequant_event)
        with torch.cuda.stream(active_trtllm):
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
        # NOTE: as in iter6 bench, do not main.wait_event(trtllm_done)
        # here. The end-of-cycle sync in _bench_n_layer captures the
        # tail of the last trtllm-gen on whichever stream ran it.
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


def _bench_n_layer(
    name,
    variant,
    side_stream,
    trtllm_stream_a,
    trtllm_stream_b,
    inputs,
    page_size,
    B,
    num_heads,
    head_dim,
    top_k,
    iters,
    warmup,
    use_fp8,
    n_layers,
):
    """Bench an n-layer back-to-back dequant + trtllm-gen sequence."""

    # Warmup.
    for _ in range(warmup):
        for lid in range(n_layers):
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
                trtllm_stream_a=trtllm_stream_a,
                trtllm_stream_b=trtllm_stream_b,
                variant=variant,
            )
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        for lid in range(n_layers):
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
                trtllm_stream_a=trtllm_stream_a,
                trtllm_stream_b=trtllm_stream_b,
                variant=variant,
            )
        # End-of-cycle sync: capture tails of both trtllm streams (only
        # one is non-None for non-dual variants, so the second wait is
        # a no-op there).
        if variant in ("dedicated_stream", "depth4", "depth4_dual"):
            torch.cuda.current_stream().wait_stream(trtllm_stream_a)
            if variant == "depth4_dual":
                torch.cuda.current_stream().wait_stream(trtllm_stream_b)
        ends[i].record()
    torch.cuda.synchronize()

    times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    return {
        "name": name,
        "variant": variant,
        "median_us_per_cycle": statistics.median(times_us),
        "min_us": min(times_us),
        "max_us": max(times_us),
        "mean_us": statistics.mean(times_us),
        "p10_us": (
            statistics.quantiles(times_us, n=10)[0]
            if iters >= 10
            else min(times_us)
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=576)
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--num-slots", type=int, default=65536)
    ap.add_argument("--n-layers", type=int, default=8)
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

    # JIT warmup pass through the baseline path.
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
        trtllm_stream_a=None,
        trtllm_stream_b=None,
        variant="baseline",
    )
    torch.cuda.synchronize()

    side_stream = torch.cuda.Stream()
    trtllm_stream_a = torch.cuda.Stream()
    trtllm_stream_b = torch.cuda.Stream()

    variants = [
        ("baseline", "baseline"),
        ("sidestream", "sidestream"),
        ("pingpong", "pingpong"),
        ("dedicated_stream", "dedicated_stream"),
        ("depth4", "depth4"),
        ("depth4_dual", "depth4_dual"),
    ]
    results = []
    for name, variant in variants:
        r = _bench_n_layer(
            name,
            variant,
            side_stream,
            trtllm_stream_a,
            trtllm_stream_b,
            inputs,
            args.page_size,
            args.B,
            args.num_heads,
            args.head_dim,
            args.top_k,
            args.iters,
            args.warmup,
            use_fp8,
            args.n_layers,
        )
        results.append(r)

    baseline = results[0]
    layers = 61
    cycles = layers / float(args.n_layers)

    def _delta(other):
        d = baseline["median_us_per_cycle"] - other["median_us_per_cycle"]
        return (
            d,
            100.0 * d / baseline["median_us_per_cycle"],
            d * cycles / 1000.0,
        )

    deltas = {}
    for r in results:
        if r["name"] == "baseline":
            continue
        d_us, d_pct, d_tpot = _delta(r)
        deltas[r["name"]] = {
            "delta_us_per_cycle": d_us,
            "delta_pct": d_pct,
            "projected_tpot_savings_ms_61L": d_tpot,
        }

    if args.json:
        print(
            json.dumps(
                {
                    "results": results,
                    "deltas_vs_baseline": deltas,
                    "kv_dtype": "fp8_e4m3" if use_fp8 else "bf16",
                    "shape": dict(
                        B=args.B,
                        num_heads=args.num_heads,
                        head_dim=args.head_dim,
                        top_k=args.top_k,
                        page_size=args.page_size,
                        n_layers=args.n_layers,
                    ),
                },
                indent=2,
            )
        )
    else:
        print(
            f"# B={args.B} num_heads={args.num_heads} head_dim={args.head_dim} "
            f"top_k={args.top_k} page_size={args.page_size} "
            f"n_layers={args.n_layers} "
            f"kv_dtype={'fp8_e4m3' if use_fp8 else 'bf16'}"
        )
        print(
            "{:>18} {:>20} {:>10} {:>10} {:>10} {:>10}".format(
                "name",
                f"median_us/{args.n_layers}layer",
                "p10_us",
                "mean_us",
                "min_us",
                "max_us",
            )
        )
        for r in results:
            print(
                "{:>18} {:>20.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
                    r["name"],
                    r["median_us_per_cycle"],
                    r["p10_us"],
                    r["mean_us"],
                    r["min_us"],
                    r["max_us"],
                )
            )
        print()
        for name, d in deltas.items():
            print(
                f"# {name:>18} vs baseline: {d['delta_us_per_cycle']:+8.2f} "
                f"us/{args.n_layers}layer ({d['delta_pct']:+6.1f}%); "
                f"proj TPOT @ 61L = {d['projected_tpot_savings_ms_61L']:+6.2f} ms"
            )


if __name__ == "__main__":
    main()
