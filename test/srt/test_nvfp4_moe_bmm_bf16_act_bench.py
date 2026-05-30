"""Standalone microbench scaffold for #15 iter7 BF16-act NVFP4 MoE bmm path.

Tests the flashinfer trtllm_fp4_block_scale_moe gate-relaxation
that unlocks the 58 in-cubin Bmm_Bfloat16_E2m1E2m1_*.cubin SM_100
variants. See notes/nvfp4_moe_iter7_recon.md.

Reference vs candidate paths:

    # Reference (iter1-3 + iter4 PRIMARY production path):
    hs_fp4, hs_scale = scaled_fp4_quant_linear(x_bf16, layer.w13_input_scale_quant)
    out_ref = trtllm_fp4_block_scale_moe(
        hidden_states=hs_fp4,
        hidden_states_scale=hs_scale,
        # ... weights, biases, etc.
    )

    # Candidate (iter7 patch series, gated on SGLANG_USE_TRTLLM_BF16_ACT_FP4_MOE):
    out_cand = trtllm_fp4_block_scale_moe_bf16_act(
        hidden_states=x_bf16,         # raw BF16, no pre-quantize
        hidden_states_scale=None,     # cubin epilogue derives SF on the fly
        # ... weights, biases, etc.
    )

    # Acceptance gates:
    #   * out_cand vs out_ref: max abs diff <= 2.5e-2 (BF16 epilogue noise)
    #   * per-token-cosine >= 0.999 across all rows
    #   * end-to-end us per layer (candidate) < end-to-end us per layer (reference)

Coverage:

  * Production shape (DSv3.2-REAP): m_global in {1, 8, 16, 32, 64, 128, 256, 512},
    H=7168, intermediate=4096, num_experts=256, top_k=8.
  * Microbench iters: 1000 warmup + 5000 measure (per row, per path).
  * Trials: 3 per row for median + min/max bracket (matches iter6 bench
    discipline).

Status: SCAFFOLD ONLY. The candidate symbol
trtllm_fp4_block_scale_moe_bf16_act is NOT bound in iter7 (no Stage B/C
yet). The reference column is runnable today and serves as the iter8
acceptance baseline; the candidate column will be wired by iter8.

Usage when fully wired:

    SGLANG_USE_TRTLLM_BF16_ACT_FP4_MOE=1 python test/srt/test_nvfp4_moe_bmm_bf16_act_bench.py --bench
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import torch

PROD_HIDDEN = 7168
PROD_INTERMEDIATE = 4096
PROD_NUM_EXPERTS = 256
PROD_TOP_K = 8
PROD_DECODE_BATCHES = [1, 8, 16, 32, 64, 128, 256, 512]


def _make_synthetic_layer(
    num_tokens: int,
    hidden: int = PROD_HIDDEN,
    intermediate: int = PROD_INTERMEDIATE,
    num_experts: int = PROD_NUM_EXPERTS,
    top_k: int = PROD_TOP_K,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Build a minimal DSv3.2-REAP-shaped synthetic MoE layer for bench.

    Returns a dict carrying x_bf16, w13_weight (uint8 packed FP4),
    w13_weight_scale (FP8 group scales), w13_input_scale_quant (BF16
    global scale), and the analogous w2_* tensors. All randomly
    initialized — output correctness is not tested here, only the
    per-call latency. Bit-exactness vs the reference path is the iter8
    Stage D verifier.
    """
    x_bf16 = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    # Packed FP4 weights: (num_experts, intermediate * 2, hidden // 2) uint8
    w13_weight = torch.randint(
        0, 256, (num_experts, intermediate * 2, hidden // 2), dtype=torch.uint8, device=device
    )
    # E4M3 group scales: (num_experts, intermediate * 2, hidden // 16)
    w13_weight_scale = torch.randn(
        num_experts, intermediate * 2, hidden // 16, dtype=dtype, device=device
    ).to(torch.float8_e4m3fn)
    w13_input_scale_quant = torch.ones(1, dtype=torch.float32, device=device)
    w2_weight = torch.randint(
        0, 256, (num_experts, hidden, intermediate // 2), dtype=torch.uint8, device=device
    )
    w2_weight_scale = torch.randn(
        num_experts, hidden, intermediate // 16, dtype=dtype, device=device
    ).to(torch.float8_e4m3fn)
    w2_input_scale_quant = torch.ones(1, dtype=torch.float32, device=device)
    return dict(
        x_bf16=x_bf16,
        w13_weight=w13_weight,
        w13_weight_scale=w13_weight_scale,
        w13_input_scale_quant=w13_input_scale_quant,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
        w2_input_scale_quant=w2_input_scale_quant,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
    )


def _run_reference_path(layer, router_logits, correction_bias) -> torch.Tensor:
    """iter1-3 + iter4 PRIMARY production path: pre-quantize + bmm."""
    from sglang.jit_kernel.nvfp4 import scaled_fp4_quant_linear
    from flashinfer import trtllm_fp4_block_scale_moe

    hs_fp4, hs_scale = scaled_fp4_quant_linear(
        layer["x_bf16"], layer["w13_input_scale_quant"]
    )
    # NOTE: full call signature elided for scaffold; see
    # compressed_tensors_w4a4_nvfp4_moe.py L425 for the production call site.
    raise NotImplementedError(
        "Reference path full wire is in iter8 Stage D; scaffold returns "
        "placeholder to keep the file importable for iter7 CI smoke."
    )


def _run_candidate_path(layer, router_logits, correction_bias) -> torch.Tensor:
    """iter7 patched path: BF16 input direct, in-cubin SF generation."""
    try:
        from sglang.srt.external_kernels.flashinfer import (
            trtllm_fp4_block_scale_moe_bf16_act,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Candidate symbol not bound. The iter7 patch series is at the "
            "Stage A scaffold checkpoint. Stage B builds the vendored "
            "launcher and exports this symbol. See "
            "notes/nvfp4_moe_iter7_recon.md."
        ) from exc
    raise NotImplementedError("Candidate path wire is iter8 Stage C.")


def _bench_row(layer, iters: int = 5000, warmup: int = 1000) -> dict:
    """Microbench one row. Returns dict with reference + candidate us."""
    # SCAFFOLD: actual bench loop is iter8 Stage D.
    return dict(
        reference_us=float("nan"),
        candidate_us=float("nan"),
        delta_us=float("nan"),
        max_abs_diff=float("nan"),
        per_token_cosine_min=float("nan"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="Run microbench (iter8+)")
    parser.add_argument(
        "--correctness", action="store_true", help="Run bit-exactness check (iter8+)"
    )
    parser.add_argument(
        "--row-batches",
        type=int,
        nargs="+",
        default=PROD_DECODE_BATCHES,
        help="Per-rank m_local batch sizes to sweep",
    )
    args = parser.parse_args()

    if not (args.bench or args.correctness):
        # Default: print the iter7 scaffold status so CI smoke catches obvious breakage.
        print(
            "iter7 scaffold: candidate symbol not bound; pass --bench or "
            "--correctness once iter8 Stages B/C/D land."
        )
        return 0

    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping bench.")
        return 0

    print("# #15 iter7 NVFP4 MoE BF16-act bmm bench (SCAFFOLD)")
    print("#")
    print("# Per-row columns:")
    print("#   m_local | reference_us | candidate_us | delta_us | max_abs_diff | cosine_min")
    print("#" + "-" * 75)
    for m_local in args.row_batches:
        layer = _make_synthetic_layer(num_tokens=m_local)
        row = _bench_row(layer)
        print(
            f"  {m_local:7d} | {row['reference_us']:11.2f} | "
            f"{row['candidate_us']:11.2f} | {row['delta_us']:8.2f} | "
            f"{row['max_abs_diff']:12.6g} | {row['per_token_cosine_min']:9.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
