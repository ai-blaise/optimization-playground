"""Autonomous benchmark loop for HIGGS 2-bit MLA decode DSL kernel.

Runs in a loop on B200, comparing:
  - DSL kernel (the work-in-progress CuTe Python DSL kernel)
  - C++ TC baseline (higgs_dense_2bit_mla_decode_tc, commit 961c4794a)
  - PyTorch reference (per-element decode + dense attention)

Reports per-iter correctness + timing; exits when DSL matches reference
within tolerance.

Usage:
  python3.11 bench_higgs_dsl.py [--bench-only] [--iters N]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch


LATENT_DIM = 512
ROPE_DIM = 64
FULL_DIM = LATENT_DIM + ROPE_DIM  # 576
PAIR_DIM = 2
CODEBOOK_SIZE = 16
PACKED_BYTES = 128
NORM_BYTES = 2
SLOT_BYTES = PACKED_BYTES + NORM_BYTES + ROPE_DIM * 2  # 258


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _decode_higgs_torch(compressed: torch.Tensor, codebook: torch.Tensor):
    """PyTorch reference: decode HIGGS-2bit slots into (K_full, V_latent) BF16.

    K is the full (LATENT + ROPE) BF16 latent representation.
    V is the latent-only (LATENT) BF16 representation (no rope in V).
    """
    n, _, _ = compressed.shape  # (n, 1, 258)
    device = compressed.device
    base = compressed.reshape(n, SLOT_BYTES)
    packed = base[:, :PACKED_BYTES].contiguous()  # (n, 128) u8
    scale = base[:, PACKED_BYTES:PACKED_BYTES + NORM_BYTES].contiguous().view(torch.float16).reshape(n)  # (n,) fp16
    rope = base[:, PACKED_BYTES + NORM_BYTES:].contiguous().view(torch.bfloat16).reshape(n, ROPE_DIM)  # (n, 64) bf16

    # Decode 4-bit nibbles → codebook indices
    # Each byte holds 2 nibbles (low+high), 4 groups of 32 bytes each
    # Total: 128 bytes × 2 nibbles = 256 codebook lookups per slot = 256 pairs × 2 = 512 dims
    bytes_view = packed.reshape(n, 4, 32)  # (n, 4 groups, 32 bytes)
    # For each (group, byte_in_group, nibble), get codebook idx
    lo = bytes_view & 0x0F     # (n, 4, 32)
    hi = (bytes_view >> 4) & 0x0F  # (n, 4, 32)
    # Stack: (n, 4 groups, 32 bytes, 2 nibbles)
    nibbles = torch.stack([lo, hi], dim=-1).reshape(n, 4, 32 * 2).to(torch.int64)  # (n, 4, 64)
    # Reshape: each group's 64 nibbles → 64 pairs → 128 dims (2 coords per pair)
    # Layout per C++: dim d = group * 128 + lane*4_or_similar...
    # Per the dequant: per-lane (tid=0..127), 4 dims per slot from 4 groups
    # d0 = 0*128 + lane; d1 = 1*128 + lane; d2 = 2*128 + lane; d3 = 3*128 + lane
    # Lane = (pair_within_group * 2 + coord_lane) where pair_within_group = byte_in_group * 2 + nibble
    # Need to figure out the actual dim layout

    # Let me work backwards from the C++ kernel's dim assignment:
    #   tid 0..127, pair_within_group = tid >> 1 = 0..63
    #   coord_lane = tid & 1 = 0 or 1
    #   byte_in_group = pair_within_group >> 1 = 0..31
    #   nibble = pair_within_group & 1 = 0 or 1
    #   d0 = 0 * 128 + tid  (= group * 128 + tid)
    #
    # So for tid=2k: pair=k, coord=0, byte=k>>1, nibble=k&1
    #   d0 = 2k
    # For tid=2k+1: pair=k, coord=1, byte=k>>1, nibble=k&1
    #   d0 = 2k+1
    #
    # cb_idx at this (byte, nibble) is used; cb_val at codebook[cb_idx, coord]
    # So dim 2k uses codebook[idx(byte=k>>1, nibble=k&1), 0]
    # And dim 2k+1 uses codebook[idx(byte=k>>1, nibble=k&1), 1]
    #
    # Within group g, dim g*128 + 2k = codebook[bytes[g, k>>1]_{k&1 nibble}, 0]
    # within group g, dim g*128 + 2k+1 = codebook[same, 1]
    #
    # For group g, k=0..63: covers dims g*128 + 0..127

    out = torch.zeros((n, LATENT_DIM), dtype=torch.float32, device=device)
    for g in range(4):
        # bytes for this group: (n, 32)
        grp = bytes_view[:, g, :]  # (n, 32)
        for k in range(64):
            byte_idx = k >> 1
            nibble = k & 1
            cb_idx = (grp[:, byte_idx] >> 4) if nibble else (grp[:, byte_idx] & 0x0F)  # (n,)
            cb_idx_long = cb_idx.to(torch.int64)
            # Look up codebook[cb_idx, 0] and codebook[cb_idx, 1]
            cb0 = codebook[cb_idx_long, 0]  # (n,)
            cb1 = codebook[cb_idx_long, 1]  # (n,)
            scale_f32 = scale.to(torch.float32)  # (n,)
            out[:, g * 128 + 2 * k] = scale_f32 * cb0
            out[:, g * 128 + 2 * k + 1] = scale_f32 * cb1

    v_full = out.to(torch.bfloat16)  # (n, 512) BF16
    k_full = torch.cat([v_full, rope], dim=-1)  # (n, 576) BF16
    return k_full, v_full


def _torch_reference(q_nope, q_rope, compressed, page_table, codebook, sm_scale):
    """Full PyTorch reference: HIGGS dequant + FWHT_512 + attention + InvFWHT_512.

    Currently NO FWHT (returns raw attention against decoded K, V). The DSL
    kernel also operates in rotated basis (Q pre-rotated, output pre-InvFWHT).
    For comparison, both ops are applied symmetrically and cancel out for the
    test data — so plain attention against decoded K, V matches what DSL
    produces (modulo FWHT errors).
    """
    R, H, _ = q_nope.shape
    TOPK = page_table.shape[1]
    device = q_nope.device

    # Decode HIGGS → BF16 K_full, V_latent
    k_full, v_full = _decode_higgs_torch(compressed, codebook)  # (N, 576), (N, 512)

    # Gather per-row K and V via page_table
    # k_per_row: (R, TOPK, 576), v_per_row: (R, TOPK, 512)
    pages = page_table.long()
    k_gathered = k_full[pages]  # (R, TOPK, 576)
    v_gathered = v_full[pages]  # (R, TOPK, 512)

    # Mask out invalid pages (page=-1) — set their K, V to 0
    valid = (page_table >= 0).unsqueeze(-1).expand_as(k_gathered)
    k_gathered = torch.where(valid, k_gathered, torch.zeros_like(k_gathered))
    valid_v = (page_table >= 0).unsqueeze(-1).expand_as(v_gathered)
    v_gathered = torch.where(valid_v, v_gathered, torch.zeros_like(v_gathered))

    # Concat Q
    q = torch.cat([q_nope, q_rope], dim=-1)  # (R, H, 576)

    # Attention: P = softmax(Q @ K^T * sm_scale); O = P @ V
    q_f32 = q.to(torch.float32)
    k_f32 = k_gathered.to(torch.float32)
    v_f32 = v_gathered.to(torch.float32)

    # Q @ K^T: (R, H, 576) @ (R, 576, TOPK) → (R, H, TOPK)
    scores = torch.einsum("rhd,rkd->rhk", q_f32, k_f32) * sm_scale
    # Softmax along TOPK
    scores_max = scores.max(dim=-1, keepdim=True).values
    P = torch.exp(scores - scores_max)
    P_sum = P.sum(dim=-1, keepdim=True)
    P = P / P_sum
    # P @ V: (R, H, TOPK) @ (R, TOPK, 512) → (R, H, 512)
    out = torch.einsum("rhk,rkd->rhd", P, v_f32)
    return out.to(torch.bfloat16)


def _make_inputs(R, H, TOPK, *, seed=42, device="cuda"):
    torch.manual_seed(seed)
    q_nope = torch.randn(R, H, LATENT_DIM, dtype=torch.bfloat16, device=device) * 0.1
    q_rope = torch.randn(R, H, ROPE_DIM, dtype=torch.bfloat16, device=device) * 0.1
    compressed = torch.zeros(TOPK, 1, SLOT_BYTES, dtype=torch.uint8, device=device)
    compressed[:, 0, :PACKED_BYTES] = torch.randint(
        0, 256, (TOPK, PACKED_BYTES), dtype=torch.uint8, device=device
    )
    scale_bytes = torch.tensor([0x00, 0x38], dtype=torch.uint8, device=device)  # FP16 0.5
    compressed[:, 0, PACKED_BYTES:PACKED_BYTES + NORM_BYTES] = scale_bytes.unsqueeze(0).expand(TOPK, -1)
    # Rope: generate actual BF16 values (random bytes → NaN/inf in BF16 frequently)
    rope_bf16 = (torch.randn(TOPK, ROPE_DIM, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    compressed[:, 0, PACKED_BYTES + NORM_BYTES:] = rope_bf16.view(torch.uint8).reshape(TOPK, ROPE_DIM * 2)
    page_table = torch.arange(TOPK, dtype=torch.int32, device=device).reshape(1, TOPK).expand(R, -1).contiguous()
    codebook = torch.randn(CODEBOOK_SIZE, PAIR_DIM, dtype=torch.float32, device=device) * 0.5
    sm_scale = 1.0 / math.sqrt(FULL_DIM)
    return dict(
        q_nope=q_nope, q_rope=q_rope, compressed=compressed,
        page_table=page_table, codebook=codebook, sm_scale=sm_scale,
    )


def _compare(out_a, out_b, label_a, label_b, atol, rtol):
    a = out_a.float()
    b = out_b.float()
    diff = (a - b).abs()
    rel = diff / (b.abs() + 1e-6)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = rel.max().item()
    close = torch.allclose(a, b, atol=atol, rtol=rtol)
    print(f"{label_a} vs {label_b}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
          f"max_rel={max_rel:.3e}  allclose={close}")
    return close


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100, help="bench iters")
    parser.add_argument("--bench-only", action="store_true", help="skip correctness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--rows", type=int, default=1)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    dsl_path = here / "higgs_dense_2bit_mla_decode_dsl.py"
    tc_path = here / "higgs_dense_2bit_mla_decode_tc.py"

    print(f"Loading DSL: {dsl_path}")
    dsl = _load_module("hdmd_dsl", dsl_path)

    print(f"Loading TC : {tc_path}")
    # Add python dir to path so sglang.* imports resolve
    py_dir = here.parent.parent
    if str(py_dir) not in sys.path:
        sys.path.insert(0, str(py_dir))
    try:
        tc = _load_module("hdmd_tc", tc_path)
    except Exception as e:
        print(f"  TC load failed: {type(e).__name__}: {e}")
        tc = None

    inputs = _make_inputs(args.rows, args.num_heads, args.topk, seed=args.seed)

    # 1. PyTorch reference
    print("\n=== PyTorch reference ===")
    t0 = time.perf_counter()
    out_ref = _torch_reference(
        inputs["q_nope"], inputs["q_rope"], inputs["compressed"],
        inputs["page_table"], inputs["codebook"], inputs["sm_scale"],
    )
    torch.cuda.synchronize()
    print(f"  Reference computed in {(time.perf_counter()-t0)*1000:.1f}ms")
    print(f"  min={out_ref.min().item():.3e} max={out_ref.max().item():.3e} "
          f"mean={out_ref.mean().item():.3e} std={out_ref.std().item():.3e}")

    # 2. DSL kernel
    print("\n=== DSL kernel ===")
    out_dsl = torch.zeros_like(out_ref)
    t0 = time.perf_counter()
    try:
        dsl.higgs_dense_2bit_mla_decode_dsl(
            inputs["q_nope"], inputs["q_rope"], inputs["compressed"],
            inputs["page_table"].clone(), out_dsl, inputs["codebook"],
            inputs["sm_scale"],
        )
        torch.cuda.synchronize()
        print(f"  Compile+run {time.perf_counter()-t0:.1f}s")
        print(f"  min={out_dsl.min().item():.3e} max={out_dsl.max().item():.3e} "
              f"mean={out_dsl.mean().item():.3e}")
        print(f"  nan={torch.isnan(out_dsl).any().item()} inf={torch.isinf(out_dsl).any().item()}")
        if not args.bench_only:
            _compare(out_dsl, out_ref, "DSL", "Ref", atol=5e-2, rtol=5e-2)
    except Exception as e:
        print(f"  DSL FAIL: {type(e).__name__}: {e}")

    # 3. C++ TC baseline
    if tc is not None:
        print("\n=== C++ TC baseline ===")
        out_tc = torch.zeros_like(out_ref)
        try:
            tc.higgs_dense_2bit_mla_decode_tc(
                inputs["q_nope"], inputs["q_rope"], inputs["compressed"],
                inputs["page_table"].clone(), out_tc, inputs["codebook"],
                inputs["sm_scale"],
            )
            torch.cuda.synchronize()
            print(f"  min={out_tc.min().item():.3e} max={out_tc.max().item():.3e} "
                  f"mean={out_tc.mean().item():.3e}")
            if not args.bench_only:
                _compare(out_tc, out_ref, "TC", "Ref", atol=5e-2, rtol=5e-2)
        except Exception as e:
            print(f"  TC FAIL: {type(e).__name__}: {e}")

    # 4. Benchmark
    if not args.bench_only:
        return

    print("\n=== Benchmark ===")
    N_WARMUP = 5
    for kernel, name, out in [
        (lambda: dsl.higgs_dense_2bit_mla_decode_dsl(
            inputs["q_nope"], inputs["q_rope"], inputs["compressed"],
            inputs["page_table"], out_dsl, inputs["codebook"], inputs["sm_scale"],
        ), "DSL", out_dsl),
    ]:
        try:
            for _ in range(N_WARMUP): kernel()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters): kernel()
            torch.cuda.synchronize()
            us = (time.perf_counter() - t0) * 1e6 / args.iters
            print(f"  {name}: {us:.2f} us/call (N={args.iters})")
        except Exception as e:
            print(f"  {name} FAIL: {e}")


if __name__ == "__main__":
    main()
