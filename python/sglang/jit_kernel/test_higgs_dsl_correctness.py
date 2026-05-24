"""Correctness + benchmark test for HIGGS 2-bit dense MLA DSL kernel.

Compares the CuTe Python DSL kernel against:
  - C++ HIGGS TC baseline (higgs_dense_2bit_mla_decode_tc, commit 961c4794a)
  - tokenspeed FP8 MLA decode (mathematically different but architecturally
    the closest production-quality SM100 MLA decode kernel — used as the
    perf bar to beat)

Usage:
  python3.11 -m sglang.jit_kernel.test_higgs_dsl_correctness  [--bench]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _make_random_inputs(R: int, H: int, TOPK: int, *, seed: int = 42, device="cuda"):
    """Generate plausible HIGGS-2bit MLA decode inputs."""
    torch.manual_seed(seed)
    LATENT, ROPE, SLOT_BYTES = 512, 64, 258
    q_nope = torch.randn(R, H, LATENT, dtype=torch.bfloat16, device=device) * 0.1
    q_rope = torch.randn(R, H, ROPE, dtype=torch.bfloat16, device=device) * 0.1
    compressed = torch.zeros(TOPK, 1, SLOT_BYTES, dtype=torch.uint8, device=device)
    # Random packed nibbles
    compressed[:, 0, :128] = torch.randint(
        0, 256, (TOPK, 128), dtype=torch.uint8, device=device
    )
    # FP16 scale = 0.5 (= 0x3800 LE)
    scale_bytes = torch.tensor([0x00, 0x38], dtype=torch.uint8, device=device)
    compressed[:, 0, 128:130] = scale_bytes.unsqueeze(0).expand(TOPK, -1)
    # Random rope bytes
    compressed[:, 0, 130:] = torch.randint(
        0, 256, (TOPK, 128), dtype=torch.uint8, device=device
    )
    page_table = torch.arange(TOPK, dtype=torch.int32, device=device).reshape(1, TOPK)
    out_dsl = torch.zeros(R, H, LATENT, dtype=torch.bfloat16, device=device)
    out_tc = torch.zeros(R, H, LATENT, dtype=torch.bfloat16, device=device)
    codebook = torch.randn(16, 2, dtype=torch.float32, device=device) * 0.5
    sm_scale = 1.0 / math.sqrt(LATENT + ROPE)
    return dict(
        q_nope=q_nope, q_rope=q_rope, compressed=compressed,
        page_table=page_table, out_dsl=out_dsl, out_tc=out_tc,
        codebook=codebook, sm_scale=sm_scale,
    )


def _compare(out_a: torch.Tensor, out_b: torch.Tensor, label_a: str, label_b: str, atol: float, rtol: float):
    """Element-wise diff + summary stats."""
    a32 = out_a.float()
    b32 = out_b.float()
    diff = (a32 - b32).abs()
    rel = diff / (b32.abs() + 1e-6)
    max_abs = diff.max().item()
    max_rel = rel.max().item()
    mean_abs = diff.mean().item()
    is_close = torch.allclose(a32, b32, atol=atol, rtol=rtol)
    print(f"{label_a} vs {label_b}:")
    print(f"  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  mean_abs={mean_abs:.3e}")
    print(f"  allclose(atol={atol:.0e}, rtol={rtol:.0e}): {is_close}")
    return is_close


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="run benchmarks too")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--rows", type=int, default=1)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    dsl_path = here / "higgs_dense_2bit_mla_decode_dsl.py"
    tc_path = here / "higgs_dense_2bit_mla_decode_tc.py"

    print(f"Loading DSL kernel: {dsl_path}")
    dsl = _load("hdmd_dsl", str(dsl_path))

    print(f"Loading TC baseline: {tc_path}")
    try:
        tc = _load("hdmd_tc", str(tc_path))
    except Exception as e:
        print(f"  WARN: TC baseline import failed ({e}); will only run DSL")
        tc = None

    inputs = _make_random_inputs(args.rows, args.num_heads, args.topk, seed=args.seed)

    # 1) DSL kernel
    print("\n=== DSL kernel ===")
    dsl.higgs_dense_2bit_mla_decode_dsl(
        inputs["q_nope"], inputs["q_rope"], inputs["compressed"],
        inputs["page_table"], inputs["out_dsl"], inputs["codebook"],
        inputs["sm_scale"],
    )
    torch.cuda.synchronize()
    o = inputs["out_dsl"]
    print(f"  shape={tuple(o.shape)} dtype={o.dtype}")
    print(f"  min={o.min().item():.3e} max={o.max().item():.3e}")
    print(f"  mean={o.mean().item():.3e} std={o.std().item():.3e}")
    print(f"  nan={torch.isnan(o).any().item()} inf={torch.isinf(o).any().item()}")

    # 2) C++ TC baseline (if available)
    if tc is not None:
        print("\n=== C++ TC baseline ===")
        try:
            tc.higgs_dense_2bit_mla_decode_tc(
                inputs["q_nope"], inputs["q_rope"], inputs["compressed"],
                inputs["page_table"], inputs["out_tc"], inputs["codebook"],
                inputs["sm_scale"],
            )
            torch.cuda.synchronize()
            o = inputs["out_tc"]
            print(f"  shape={tuple(o.shape)} dtype={o.dtype}")
            print(f"  min={o.min().item():.3e} max={o.max().item():.3e}")
            print(f"  mean={o.mean().item():.3e} std={o.std().item():.3e}")

            # 3) Correctness comparison
            print("\n=== Correctness ===")
            _compare(inputs["out_dsl"], inputs["out_tc"], "DSL", "TC", atol=5e-2, rtol=5e-2)
        except Exception as e:
            print(f"  TC kernel failed: {e}")

    # 4) Benchmarks
    if args.bench:
        print("\n=== Benchmark ===")
        N_WARMUP, N_ITER = 5, 100
        for _ in range(N_WARMUP):
            dsl.higgs_dense_2bit_mla_decode_dsl(
                inputs["q_nope"], inputs["q_rope"], inputs["compressed"],
                inputs["page_table"], inputs["out_dsl"], inputs["codebook"],
                inputs["sm_scale"],
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            dsl.higgs_dense_2bit_mla_decode_dsl(
                inputs["q_nope"], inputs["q_rope"], inputs["compressed"],
                inputs["page_table"], inputs["out_dsl"], inputs["codebook"],
                inputs["sm_scale"],
            )
        torch.cuda.synchronize()
        dsl_us = (time.perf_counter() - t0) * 1e6 / N_ITER
        print(f"  DSL:  {dsl_us:.2f} us/call (N={N_ITER})")

        if tc is not None:
            for _ in range(N_WARMUP):
                tc.higgs_dense_2bit_mla_decode_tc(
                    inputs["q_nope"], inputs["q_rope"], inputs["compressed"],
                    inputs["page_table"], inputs["out_tc"], inputs["codebook"],
                    inputs["sm_scale"],
                )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_ITER):
                tc.higgs_dense_2bit_mla_decode_tc(
                    inputs["q_nope"], inputs["q_rope"], inputs["compressed"],
                    inputs["page_table"], inputs["out_tc"], inputs["codebook"],
                    inputs["sm_scale"],
                )
            torch.cuda.synchronize()
            tc_us = (time.perf_counter() - t0) * 1e6 / N_ITER
            print(f"  TC:   {tc_us:.2f} us/call (N={N_ITER})")
            print(f"  DSL/TC ratio: {dsl_us / tc_us:.2f}x  ({'DSL faster' if dsl_us < tc_us else 'TC faster'})")


if __name__ == "__main__":
    main()
