"""HIGGS inline sparse-MLA decode producer microbench (#19 iter8 scaffold).

ai-blaise #19 iter8 PRIMARY vector. Times the iter8 scaffold producer
``higgs_inline_sparse_mla_produce_fp8`` (cp.async slot prefetch +
depth-2 SMEM ping-pong, scaffold of the producer the iter9 in-cubin
CUTLASS path will graft into the flashinfer cute_dsl monolithic
mla_decode_fp8 template) against the existing iter3 production path
``dequantize_higgs_dense_2bit_page_table_fp8`` (uncached LDG slot
read, 512-thread CTA).

Production shape (REAP B200 deploy, DP=TP=8):
- B = 128, top_k = 2048 — num_rows = 262144
- num_slots = 65536 (~2 layers worth of context)
- kSlotBytes = 272 (HIGGS payload + iter4 #16 16-align pad)
- output: (B*K, 1, 576) FP8 e4m3 = 144 MiB / layer-step

Iter8 scaffold honest scope. Both kernels write the SAME 302 MiB to
gmem (the dequant *write* still happens; only iter9's in-cubin lift
eliminates it). The standalone microbench therefore measures *only*
the kernel-launch + slot-read + decode-pipeline efficiency delta, not
the architectural HBM round-trip elimination. A modest speedup (or
parity-within-noise) is the expected outcome; the value of the iter8
scaffold is the SMEM-resident pipeline being ready for the iter9 graft.

Usage:
    python bench_higgs_inline_sparse_mla_decode_iter8.py \
        --iters 200 --warmup 50
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=2048)
    parser.add_argument("--num_slots", type=int, default=65536)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=61,
        help="layer count used for projected TPOT delta (61 for REAP-345B)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for the iter8 producer microbench")
    device = torch.device("cuda:0")

    inputs = _build_inputs(args.batch, args.top_k, args.num_slots, device)

    # Existing iter3 production path.
    def run_iter3():
        dequantize_higgs_dense_2bit_page_table_fp8(
            inputs["compressed"],
            inputs["page_table_2d"],
            inputs["out_fp8"],
            inputs["compact_page_table_2d"],
            inputs["codebook"],
            inv_kv_scale=1.0,
        )

    # Iter8 scaffold producer.
    def run_iter8():
        higgs_inline_sparse_mla_produce_fp8(
            inputs["compressed"],
            inputs["page_table_2d"],
            inputs["out_fp8"],
            inputs["compact_page_table_2d"],
            inputs["codebook"],
            inv_kv_scale=1.0,
        )

    iter3_us = _time_kernel(run_iter3, args.iters, args.warmup, device)
    iter8_us = _time_kernel(run_iter8, args.iters, args.warmup, device)

    delta_us = iter3_us - iter8_us
    delta_pct = (delta_us / iter3_us) * 100.0 if iter3_us > 0 else 0.0
    proj_tpot_ms_iter3 = iter3_us * args.num_layers / 1000.0
    proj_tpot_ms_iter8 = iter8_us * args.num_layers / 1000.0
    proj_tpot_delta_ms = (iter3_us - iter8_us) * args.num_layers / 1000.0

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
                        "iter8_scaffold": iter8_us,
                    },
                    "delta_us": delta_us,
                    "delta_pct": delta_pct,
                    "proj_tpot_ms": {
                        "iter3_existing": proj_tpot_ms_iter3,
                        "iter8_scaffold": proj_tpot_ms_iter8,
                        "delta": proj_tpot_delta_ms,
                    },
                }
            )
        )
    else:
        print(
            f"=== HIGGS inline sparse-MLA produce FP8 (iter8 scaffold) ===\n"
            f"shape : B={args.batch} K={args.top_k} "
            f"num_slots={args.num_slots} num_rows={args.batch * args.top_k}\n"
            f"iters : {args.iters} (warmup {args.warmup})\n\n"
            f"variant            median_us  proj TPOT ms @ {args.num_layers}L\n"
            f"iter3_existing  {iter3_us:13.2f}  {proj_tpot_ms_iter3:6.2f}\n"
            f"iter8_scaffold  {iter8_us:13.2f}  {proj_tpot_ms_iter8:6.2f}\n"
            f"\n"
            f"delta vs iter3 : {delta_us:+.2f} us ({delta_pct:+.2f}%)\n"
            f"proj TPOT delta : {proj_tpot_delta_ms:+.2f} ms across "
            f"{args.num_layers} layers\n"
            f"\n"
            f"Honest scope: both kernels write the same 302 MiB to gmem;\n"
            f"the standalone microbench measures kernel-launch + slot-read +\n"
            f"decode-pipeline efficiency delta only, NOT the in-cubin\n"
            f"round-trip elimination that iter9 unlocks.\n"
        )


if __name__ == "__main__":
    main()
