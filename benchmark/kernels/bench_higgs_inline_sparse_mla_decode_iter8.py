"""HIGGS inline sparse-MLA decode producer microbench (#19 iter8/iter9).

ai-blaise #19 iter8 scaffold (dad3bdfca) + iter9 PRIMARY correctness
fix. Times the producer ``higgs_inline_sparse_mla_produce_fp8``
(cp.async slot prefetch + depth-2 SMEM ping-pong staging) against the
iter3 production path ``dequantize_higgs_dense_2bit_page_table_fp8``
(uncached LDG slot read). Both kernels run a 512-thread CTA — the
iter9 fix reverted the iter8 128-thread launch (which left a
latent-tile correctness gap, see ``notes/higgs_dsa_iter9_recon.md``)
to restore bit-exactness; the cp.async win is independent of the FWHT
lane count.

Production shape (REAP B200 deploy, DP=TP=8):
- B = 128, top_k = 2048 — num_rows = 262144
- num_slots = 65536 (~2 layers worth of context)
- kSlotBytes = 272 (HIGGS payload + iter4 #16 16-align pad)
- output: (B*K, 1, 576) FP8 e4m3 = 144 MiB / layer-step

Modes:
  --correctness : slot-by-slot FP8 diff vs iter3 (max_diff, % bit-exact)
  default       : wall-clock median latency + projected TPOT delta

Iter8 honest scope (unchanged in iter9 PRIMARY): both kernels write
the SAME 302 MiB to gmem (the dequant *write* still happens; only the
iter9 SECONDARY in-cubin lift, queued separately, eliminates it). The
standalone microbench measures only the kernel-launch + slot-read +
decode-pipeline efficiency delta.

Usage:
    python bench_higgs_inline_sparse_mla_decode_iter8.py \\
        --iters 200 --warmup 50
    python bench_higgs_inline_sparse_mla_decode_iter8.py --correctness
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from sglang.jit_kernel.higgs_dense_2bit import (
    dequantize_higgs_dense_2bit_page_table_fp8,
)
from sglang.jit_kernel.higgs_inline_sparse_mla_decode import (
    higgs_inline_sparse_mla_produce_fp8,
)


def _build_inputs(
    B: int,
    K: int,
    num_slots: int,
    device: torch.device,
):
    """Build production-shaped HIGGS inputs.

    Compressed slots are random uint8 (the codebook handles any 4-bit
    pattern; the fp16 scale at slot+128 is a random half too). Page
    table is a uniform random permutation of valid slot ids (no -1
    holes — production runs after the iter2 page<0 early exit so
    invalid rows are a small fraction in long-context).
    """
    torch.manual_seed(0xC0FFEE)
    kSlotBytes = 272
    compressed = torch.randint(
        0, 256, (num_slots, 1, kSlotBytes), dtype=torch.uint8, device=device,
    )
    # Fill the fp16 scale slot at offset kPackedBytes=128 with a small
    # positive half (avoid Inf/NaN from random bytes).
    scale_h = torch.empty((num_slots,), dtype=torch.float16, device=device)
    scale_h.uniform_(0.05, 0.2)
    scale_bytes = scale_h.view(torch.uint8).view(num_slots, 2)
    compressed[:, 0, 128:130] = scale_bytes
    # Fill the bf16 rope slot at offset kPackedBytes + kNormBytes = 130
    # (64 bf16 = 128 bytes) with finite values. Random uint8 there can
    # encode bf16 NaN/Inf patterns, which cast to FP8 NaN with
    # non-deterministic bit payloads — diffs against the iter3 baseline
    # would then be NaN at the same positions for both kernels but with
    # different bit reps, producing spurious correctness failures.
    rope_safe = torch.empty(
        (num_slots, 64), dtype=torch.bfloat16, device=device,
    )
    rope_safe.uniform_(-1.0, 1.0)
    compressed[:, 0, 130:130 + 128] = (
        rope_safe.view(torch.uint8).view(num_slots, 128)
    )

    page_table_2d = torch.randint(
        0, num_slots, (B, K), dtype=torch.int32, device=device,
    )

    # Output buffer.
    out_fp8 = torch.empty(
        (B * K, 1, 576), dtype=torch.float8_e4m3fn, device=device,
    )

    # Codebook (16, 2) — production EDEN2-16 lattice; here a placeholder
    # is OK (we measure latency, not correctness).
    codebook = torch.randn((16, 2), dtype=torch.float32, device=device)
    codebook = codebook / codebook.abs().max() * 0.5

    return {
        "compressed": compressed,
        "page_table_2d": page_table_2d,
        "compact_page_table_2d": torch.empty_like(page_table_2d),
        "out_fp8": out_fp8,
        "codebook": codebook,
    }


def _time_kernel(fn, iters: int, warmup: int, device: torch.device) -> float:
    """Wall-clock median latency (us) across ``iters`` iterations."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    times_us = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times_us)


def _run_iter3(inputs):
    dequantize_higgs_dense_2bit_page_table_fp8(
        inputs["compressed"],
        inputs["page_table_2d"],
        inputs["out_fp8"],
        inputs["compact_page_table_2d"],
        inputs["codebook"],
        inv_kv_scale=1.0,
    )


def _run_inline(inputs):
    higgs_inline_sparse_mla_produce_fp8(
        inputs["compressed"],
        inputs["page_table_2d"],
        inputs["out_fp8"],
        inputs["compact_page_table_2d"],
        inputs["codebook"],
        inv_kv_scale=1.0,
    )


def _correctness_diff(
    B: int,
    K: int,
    num_slots: int,
    device: torch.device,
) -> dict:
    """Slot-by-slot FP8 bit-identity vs iter3 for the inline producer.

    Returns the diff stats for both the latent tile [0:512] and the
    rope tile [512:576] separately, since the iter8 scaffold had a
    bit-exact rope (max_diff = 0) and a 48.5% bit-exact latent
    (max_diff = 0.875). The iter9 PRIMARY fix should drive the latent
    tile to bit-exactness (max_diff = 0, 100% bit-exact) without
    touching the rope path.
    """
    iter3_inputs = _build_inputs(B, K, num_slots, device)
    iter9_inputs = _build_inputs(B, K, num_slots, device)

    _run_iter3(iter3_inputs)
    _run_inline(iter9_inputs)
    torch.cuda.synchronize(device)

    iter3_out = iter3_inputs["out_fp8"]
    iter9_out = iter9_inputs["out_fp8"]

    # Convert to fp32 for diff math (FP8 e4m3 doesn't support sub).
    iter3_f = iter3_out.float()
    iter9_f = iter9_out.float()

    diff = (iter3_f - iter9_f).abs()
    latent_diff = diff[:, :, :512]
    rope_diff = diff[:, :, 512:576]

    def _stats(d: torch.Tensor) -> dict:
        flat = d.reshape(-1)
        n = flat.numel()
        bit_exact = (flat == 0).sum().item()
        return {
            "max_diff": float(flat.max().item()),
            "mean_diff": float(flat.mean().item()),
            "frac_bit_exact": bit_exact / max(1, n),
            "num_elements": n,
        }

    return {
        "latent": _stats(latent_diff),
        "rope": _stats(rope_diff),
        "all": _stats(diff),
    }


def _run_correctness(args, device):
    shapes = [
        (64, 1024),
        (64, 2048),
        (128, 1024),
        (128, 2048),
        (256, 2048),
    ]
    print(
        "=== HIGGS inline sparse-MLA produce FP8 (iter9 PRIMARY) — "
        "bit-identity vs iter3 ===\n"
    )
    print(
        "shape (B,K)    | tile    | max_diff  mean_diff  frac_bit_exact"
        "   num_elements"
    )
    print("-" * 84)
    all_pass = True
    for B, K in shapes:
        diff = _correctness_diff(B, K, args.num_slots, device)
        for tile_name in ("latent", "rope", "all"):
            s = diff[tile_name]
            ok = s["max_diff"] <= 1.0 / 16  # one FP8 e4m3 lsb @ |x|<=1
            all_pass = all_pass and ok
            mark = "OK " if ok else "BAD"
            print(
                f"({B:3d},{K:5d}) | {tile_name:6s} | "
                f"{s['max_diff']:8.4f}  {s['mean_diff']:9.5f}  "
                f"{s['frac_bit_exact']*100:6.2f}%       "
                f"{s['num_elements']:>12d}  {mark}"
            )
    print()
    if all_pass:
        print("PASS — iter9 PRIMARY latent tile is bit-exact (or <=1 fp8 lsb) vs iter3.")
    else:
        print("FAIL — latent tile diff exceeds 1 fp8 lsb; the iter9 fix has not landed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=2048)
    parser.add_argument("--num_slots", type=int, default=65536)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="run the slot-by-slot bit-identity check across production shapes",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=61,
        help="layer count used for projected TPOT delta (61 for REAP-345B)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for the inline producer microbench")
    device = torch.device("cuda:0")

    if args.correctness:
        _run_correctness(args, device)
        return

    inputs = _build_inputs(args.batch, args.top_k, args.num_slots, device)

    iter3_us = _time_kernel(
        lambda: _run_iter3(inputs), args.iters, args.warmup, device
    )
    inline_us = _time_kernel(
        lambda: _run_inline(inputs), args.iters, args.warmup, device
    )

    delta_us = iter3_us - inline_us
    delta_pct = (delta_us / iter3_us) * 100.0 if iter3_us > 0 else 0.0
    proj_tpot_ms_iter3 = iter3_us * args.num_layers / 1000.0
    proj_tpot_ms_inline = inline_us * args.num_layers / 1000.0
    proj_tpot_delta_ms = (iter3_us - inline_us) * args.num_layers / 1000.0

    if args.json:
        print(
            json.dumps(
                {
                    "shape": {
                        "B": args.batch,
                        "K": args.top_k,
                        "num_slots": args.num_slots,
                        "num_rows": args.batch * args.top_k,
                        "num_layers": args.num_layers,
                    },
                    "median_us": {
                        "iter3_existing": iter3_us,
                        "inline_producer": inline_us,
                    },
                    "delta_us": delta_us,
                    "delta_pct": delta_pct,
                    "proj_tpot_ms": {
                        "iter3_existing": proj_tpot_ms_iter3,
                        "inline_producer": proj_tpot_ms_inline,
                        "delta": proj_tpot_delta_ms,
                    },
                }
            )
        )
    else:
        print(
            f"=== HIGGS inline sparse-MLA produce FP8 (iter9 PRIMARY) ===\n"
            f"shape : B={args.batch} K={args.top_k} "
            f"num_slots={args.num_slots} num_rows={args.batch * args.top_k}\n"
            f"iters : {args.iters} (warmup {args.warmup})\n\n"
            f"variant            median_us  proj TPOT ms @ {args.num_layers}L\n"
            f"iter3_existing   {iter3_us:13.2f}  {proj_tpot_ms_iter3:6.2f}\n"
            f"inline_producer  {inline_us:13.2f}  {proj_tpot_ms_inline:6.2f}\n"
            f"\n"
            f"delta vs iter3 : {delta_us:+.2f} us ({delta_pct:+.2f}%)\n"
            f"proj TPOT delta : {proj_tpot_delta_ms:+.2f} ms across "
            f"{args.num_layers} layers\n"
            f"\n"
            f"Honest scope: both kernels write the same 302 MiB to gmem;\n"
            f"the standalone microbench measures kernel-launch + slot-read +\n"
            f"decode-pipeline efficiency delta only, NOT the in-cubin\n"
            f"round-trip elimination that the iter9 SECONDARY unlocks.\n"
        )


if __name__ == "__main__":
    main()
