# Copyright 2024-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Warp Decode kernels for small-batch BF16 MoE decode."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_CUTE_AVAILABLE = False
_CUTE_EXTENSION_PATH: Optional[Path] = None
_CUTE_GATE_UP_TILE_K = 1024
_CUTE_DOWN_TILE_D = 8
_CUTE_DOWN_TILE_N = 2048
_CUTE_TOP_K = 8
try:
    import sgl_kernel

    if hasattr(sgl_kernel, "warp_decode_cute_moe_packed_forward"):
        _CUTE_AVAILABLE = True
        common_ops = getattr(sgl_kernel, "common_ops", None)
        common_ops_file = getattr(common_ops, "__file__", None)
        if common_ops_file is not None:
            _CUTE_EXTENSION_PATH = Path(common_ops_file)
        logger.info("CuTe warp decode kernels available via sgl_kernel")
except ImportError:
    pass


def _loaded_common_ops_arch() -> Optional[str]:
    if _CUTE_EXTENSION_PATH is None:
        return None
    return _CUTE_EXTENSION_PATH.resolve().parent.name


def _cute_extension_matches_device() -> bool:
    if not torch.cuda.is_available():
        return False
    cc = torch.cuda.get_device_capability()
    loaded_arch = _loaded_common_ops_arch()
    if cc[0] >= 10:
        return loaded_arch == "sm100"
    if cc[0] == 9:
        return loaded_arch == "sm90"
    return False


def _should_use_cute() -> bool:
    if not _CUTE_AVAILABLE:
        return False
    if not _cute_extension_matches_device():
        return False
    cute_mode = envs.SGLANG_WARP_DECODE_CUTE.get()
    if cute_mode == "0":
        return False
    if cute_mode == "1":
        return True
    # "auto": use CuTe on SM100+ (Blackwell), Triton elsewhere
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 10:  # SM100+
            return True
    return False


def _shape_supports_cute(
    hidden_size: int, intermediate_size: int, top_k: int
) -> bool:
    return (
        top_k == _CUTE_TOP_K
        and hidden_size % _CUTE_GATE_UP_TILE_K == 0
        and hidden_size % _CUTE_DOWN_TILE_D == 0
        and intermediate_size % _CUTE_DOWN_TILE_N == 0
    )


def _can_use_cute(hidden_size: int, intermediate_size: int, top_k: int) -> bool:
    return _should_use_cute() and _shape_supports_cute(
        hidden_size, intermediate_size, top_k
    )


def _triton_block_sizes(
    hidden_size: int, intermediate_size: int, top_k: int
) -> tuple[int, int, int, int]:
    if top_k == 8 and hidden_size >= 1024 and intermediate_size >= 2048:
        return (
            min(4, intermediate_size),
            min(256, hidden_size),
            min(4, hidden_size),
            min(128, intermediate_size),
        )
    return (
        min(32, intermediate_size),
        min(128, hidden_size),
        min(32, hidden_size),
        min(128, intermediate_size),
    )


# ---------------------------------------------------------------------------
# Kernel 1: Gate/Up projection
# ---------------------------------------------------------------------------
# Grid: (num_intermediate_neurons_per_block, num_tokens * top_k)
# Each program instance handles BLOCK_N intermediate neurons for one
# (token, expert) pair. It streams over the hidden dimension in
# BLOCK_K chunks, accumulating gate_proj and up_proj dot products.
# ---------------------------------------------------------------------------

@triton.jit
def _warp_decode_gate_up_kernel(
    # Activation input [num_tokens, hidden_size]
    x_ptr,
    x_stride_t: int,
    x_stride_d: int,
    # Gate weight [num_experts, intermediate_size, hidden_size] (BF16)
    w_gate_ptr,
    w_gate_stride_e: int,
    w_gate_stride_n: int,
    w_gate_stride_k: int,
    # Up weight [num_experts, intermediate_size, hidden_size] (BF16)
    w_up_ptr,
    w_up_stride_e: int,
    w_up_stride_n: int,
    w_up_stride_k: int,
    # Output intermediate [num_tokens * top_k, intermediate_size]
    out_ptr,
    out_stride_te: int,
    out_stride_n: int,
    # Expert IDs [num_tokens, top_k]
    expert_ids_ptr,
    expert_ids_stride_t: int,
    expert_ids_stride_k: int,
    # Dimensions
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_tokens: int,
    # Block sizes (constexpr for Triton)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Gate/Up fused kernel: one program = BLOCK_N intermediate neurons
    for one (token, expert) pair."""
    # Program indices
    pid_n = tl.program_id(0)  # which block of intermediate neurons
    pid_te = tl.program_id(1)  # which (token, expert) pair

    # Decode token and expert indices
    token_idx = pid_te // top_k
    k_idx = pid_te % top_k

    # Bounds check
    if token_idx >= num_tokens:
        return

    # Load expert ID for this (token, k) pair
    expert_id = tl.load(
        expert_ids_ptr
        + token_idx * expert_ids_stride_t
        + k_idx * expert_ids_stride_k
    )

    # Intermediate neuron offsets for this block
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < intermediate_size

    # Initialize gate and up accumulators in FP32
    gate_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Stream over hidden dimension in BLOCK_K chunks
    for k_start in range(0, hidden_size, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < hidden_size

        # Load activation slice: x[token_idx, k_start:k_start+BLOCK_K]
        x_vals = tl.load(
            x_ptr + token_idx * x_stride_t + k_offsets * x_stride_d,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        # Load gate weight: w_gate[expert_id, n_offsets, k_offsets]
        # Shape: [BLOCK_N, BLOCK_K]
        w_gate_vals = tl.load(
            w_gate_ptr
            + expert_id * w_gate_stride_e
            + n_offsets[:, None] * w_gate_stride_n
            + k_offsets[None, :] * w_gate_stride_k,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Load up weight: w_up[expert_id, n_offsets, k_offsets]
        w_up_vals = tl.load(
            w_up_ptr
            + expert_id * w_up_stride_e
            + n_offsets[:, None] * w_up_stride_n
            + k_offsets[None, :] * w_up_stride_k,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Dot product: accumulate gate_acc and up_acc
        # gate_acc[n] += sum_k(w_gate[n,k] * x[k])
        gate_acc += tl.sum(w_gate_vals * x_vals[None, :], axis=1)
        up_acc += tl.sum(w_up_vals * x_vals[None, :], axis=1)

    # Apply SiLU(gate) * up
    # SiLU(x) = x * sigmoid(x)
    gate_silu = gate_acc * tl.sigmoid(gate_acc)
    intermediate = gate_silu * up_acc

    # Write output: out[token_idx * top_k + k_idx, n_offsets]
    te_idx = token_idx * top_k + k_idx
    tl.store(
        out_ptr + te_idx * out_stride_te + n_offsets * out_stride_n,
        intermediate.to(tl.bfloat16),
        mask=n_mask,
    )


# ---------------------------------------------------------------------------
# Kernel 1b: Gate/Up projection for PACKED weights (gate+up concatenated)
# ---------------------------------------------------------------------------
# For models that store gate and up weights concatenated as w13:
#   w13[expert, 2*intermediate_size, hidden_size]
# where first half is gate, second half is up.
# ---------------------------------------------------------------------------

@triton.jit
def _warp_decode_gate_up_packed_kernel(
    # Activation input [num_tokens, hidden_size]
    x_ptr,
    x_stride_t: int,
    x_stride_d: int,
    # Packed gate+up weight [num_experts, 2*intermediate_size, hidden_size]
    w13_ptr,
    w13_stride_e: int,
    w13_stride_n: int,
    w13_stride_k: int,
    # Output intermediate [num_tokens * top_k, intermediate_size]
    out_ptr,
    out_stride_te: int,
    out_stride_n: int,
    # Expert IDs [num_tokens, top_k]
    expert_ids_ptr,
    expert_ids_stride_t: int,
    expert_ids_stride_k: int,
    # Dimensions
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_tokens: int,
    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Gate/Up kernel for packed w13 weights."""
    pid_n = tl.program_id(0)
    pid_te = tl.program_id(1)

    token_idx = pid_te // top_k
    k_idx = pid_te % top_k

    if token_idx >= num_tokens:
        return

    expert_id = tl.load(
        expert_ids_ptr
        + token_idx * expert_ids_stride_t
        + k_idx * expert_ids_stride_k
    )

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < intermediate_size

    # Gate rows are at [0, intermediate_size), up rows at [intermediate_size, 2*intermediate_size)
    gate_row_offsets = n_offsets
    up_row_offsets = n_offsets + intermediate_size

    gate_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, hidden_size, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < hidden_size

        x_vals = tl.load(
            x_ptr + token_idx * x_stride_t + k_offsets * x_stride_d,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        # Load gate weights from first half
        w_gate = tl.load(
            w13_ptr
            + expert_id * w13_stride_e
            + gate_row_offsets[:, None] * w13_stride_n
            + k_offsets[None, :] * w13_stride_k,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Load up weights from second half
        w_up = tl.load(
            w13_ptr
            + expert_id * w13_stride_e
            + up_row_offsets[:, None] * w13_stride_n
            + k_offsets[None, :] * w13_stride_k,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        gate_acc += tl.sum(w_gate * x_vals[None, :], axis=1)
        up_acc += tl.sum(w_up * x_vals[None, :], axis=1)

    gate_silu = gate_acc * tl.sigmoid(gate_acc)
    intermediate = gate_silu * up_acc

    te_idx = token_idx * top_k + k_idx
    tl.store(
        out_ptr + te_idx * out_stride_te + n_offsets * out_stride_n,
        intermediate.to(tl.bfloat16),
        mask=n_mask,
    )


# ---------------------------------------------------------------------------
# Kernel 2: Down projection + expert combine
# ---------------------------------------------------------------------------
# Grid: (hidden_size_blocks, num_tokens)
# Each program computes BLOCK_D output dimensions for one token.
# Loops over top-k experts, folding routing weights into the accumulator.
# No cross-program synchronization is needed.
# ---------------------------------------------------------------------------

@triton.jit
def _warp_decode_down_kernel(
    # Intermediate activations [num_tokens * top_k, intermediate_size]
    intermediate_ptr,
    intermediate_stride_te: int,
    intermediate_stride_n: int,
    # Down projection weight [num_experts, hidden_size, intermediate_size]
    w_down_ptr,
    w_down_stride_e: int,
    w_down_stride_d: int,
    w_down_stride_n: int,
    # Routing weights [num_tokens, top_k]
    routing_weights_ptr,
    routing_stride_t: int,
    routing_stride_k: int,
    # Expert IDs [num_tokens, top_k]
    expert_ids_ptr,
    expert_ids_stride_t: int,
    expert_ids_stride_k: int,
    # Output [num_tokens, hidden_size]
    out_ptr,
    out_stride_t: int,
    out_stride_d: int,
    # Dimensions
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_tokens: int,
    # Block sizes
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Down projection kernel: one program = BLOCK_D output dims for one token."""
    pid_d = tl.program_id(0)  # which block of output dimensions
    pid_t = tl.program_id(1)  # which token

    if pid_t >= num_tokens:
        return

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < hidden_size

    # Accumulator for this token's output dimensions
    out_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Loop over top-k experts, folding routing weights
    for k_idx in range(top_k):
        # Load routing weight for this (token, expert)
        routing_weight = tl.load(
            routing_weights_ptr
            + pid_t * routing_stride_t
            + k_idx * routing_stride_k
        ).to(tl.float32)

        # Load expert ID
        expert_id = tl.load(
            expert_ids_ptr
            + pid_t * expert_ids_stride_t
            + k_idx * expert_ids_stride_k
        )

        # Intermediate activation index for this (token, expert)
        te_idx = pid_t * top_k + k_idx

        # Stream over intermediate dimension
        expert_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for n_start in range(0, intermediate_size, BLOCK_N):
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offsets < intermediate_size

            # Load intermediate activation slice
            inter_vals = tl.load(
                intermediate_ptr
                + te_idx * intermediate_stride_te
                + n_offsets * intermediate_stride_n,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)

            # Load down weight: w_down[expert_id, d_offsets, n_offsets]
            w_down_vals = tl.load(
                w_down_ptr
                + expert_id * w_down_stride_e
                + d_offsets[:, None] * w_down_stride_d
                + n_offsets[None, :] * w_down_stride_n,
                mask=d_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # Dot product
            expert_acc += tl.sum(w_down_vals * inter_vals[None, :], axis=1)

        # Fold routing weight into accumulator
        out_acc += routing_weight * expert_acc

    # Write output
    tl.store(
        out_ptr + pid_t * out_stride_t + d_offsets * out_stride_d,
        out_acc.to(tl.bfloat16),
        mask=d_mask,
    )


def warp_decode_moe(
    hidden_states: torch.Tensor,  # [num_tokens, hidden_size]
    w_gate: torch.Tensor,  # [num_experts, intermediate_size, hidden_size]
    w_up: torch.Tensor,  # [num_experts, intermediate_size, hidden_size]
    w_down: torch.Tensor,  # [num_experts, hidden_size, intermediate_size]
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    topk_weights: torch.Tensor,  # [num_tokens, top_k]
    inplace: bool = False,
) -> torch.Tensor:
    """Run warp decode MoE with separate gate/up/down weights.

    Dispatches to CuTe CUDA kernels when available (SM100+ or
    SGLANG_WARP_DECODE_CUTE=1), otherwise uses Triton kernels.

    Args:
        hidden_states: Input activations [num_tokens, hidden_size].
        w_gate: Gate projection weights [E, N, K] in BF16.
        w_up: Up projection weights [E, N, K] in BF16.
        w_down: Down projection weights [E, D, N] in BF16.
        topk_ids: Expert IDs per token [num_tokens, top_k].
        topk_weights: Routing weights per token [num_tokens, top_k].
        inplace: If True, add result to hidden_states in-place.

    Returns:
        Output tensor [num_tokens, hidden_size].
    """
    # Input validation
    assert hidden_states.ndim == 2, f"hidden_states must be 2D, got {hidden_states.ndim}D"
    assert hidden_states.is_cuda, "hidden_states must be on CUDA"
    assert hidden_states.dtype == torch.bfloat16, (
        f"hidden_states must be bfloat16, got {hidden_states.dtype}"
    )

    num_tokens, hidden_size = hidden_states.shape

    # Edge case: zero tokens returns empty output
    if num_tokens == 0:
        if inplace:
            return hidden_states
        return torch.empty_like(hidden_states)

    top_k = topk_ids.shape[1]
    # CuTe dispatch
    if _can_use_cute(hidden_size, w_gate.shape[1], top_k):
        return sgl_kernel.warp_decode_cute_moe_forward(
            hidden_states, w_gate, w_up, w_down,
            topk_ids, topk_weights, inplace,
        )

    assert w_gate.dtype == torch.bfloat16, f"w_gate must be bfloat16, got {w_gate.dtype}"
    assert w_up.dtype == torch.bfloat16, f"w_up must be bfloat16, got {w_up.dtype}"
    assert w_down.dtype == torch.bfloat16, f"w_down must be bfloat16, got {w_down.dtype}"
    num_experts, intermediate_size, w_k = w_gate.shape

    assert w_k == hidden_size, (
        f"w_gate hidden dim {w_k} != hidden_size {hidden_size}"
    )
    assert w_up.shape == w_gate.shape, (
        f"w_up shape {w_up.shape} != w_gate shape {w_gate.shape}"
    )
    assert w_down.shape == (num_experts, hidden_size, intermediate_size), (
        f"w_down shape {w_down.shape} != expected "
        f"({num_experts}, {hidden_size}, {intermediate_size})"
    )
    assert topk_ids.shape[0] == num_tokens
    assert topk_weights.shape == topk_ids.shape

    # Allocate intermediate buffer [num_tokens * top_k, intermediate_size]
    intermediate = torch.empty(
        (num_tokens * top_k, intermediate_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )

    BLOCK_N, BLOCK_K, BLOCK_D, BLOCK_N_DOWN = _triton_block_sizes(
        hidden_size, intermediate_size, top_k
    )

    # Round up to power of 2 for Triton
    BLOCK_N = triton.next_power_of_2(BLOCK_N)
    BLOCK_K = triton.next_power_of_2(BLOCK_K)
    BLOCK_D = triton.next_power_of_2(BLOCK_D)
    BLOCK_N_DOWN = triton.next_power_of_2(BLOCK_N_DOWN)

    # Kernel 1: Gate/Up
    grid_gate_up = (
        triton.cdiv(intermediate_size, BLOCK_N),
        num_tokens * top_k,
    )
    _warp_decode_gate_up_kernel[grid_gate_up](
        x_ptr=hidden_states,
        x_stride_t=hidden_states.stride(0),
        x_stride_d=hidden_states.stride(1),
        w_gate_ptr=w_gate,
        w_gate_stride_e=w_gate.stride(0),
        w_gate_stride_n=w_gate.stride(1),
        w_gate_stride_k=w_gate.stride(2),
        w_up_ptr=w_up,
        w_up_stride_e=w_up.stride(0),
        w_up_stride_n=w_up.stride(1),
        w_up_stride_k=w_up.stride(2),
        out_ptr=intermediate,
        out_stride_te=intermediate.stride(0),
        out_stride_n=intermediate.stride(1),
        expert_ids_ptr=topk_ids,
        expert_ids_stride_t=topk_ids.stride(0),
        expert_ids_stride_k=topk_ids.stride(1),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        num_tokens=num_tokens,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Allocate output
    if inplace:
        output = hidden_states
    else:
        output = torch.empty_like(hidden_states)

    # Kernel 2: Down + combine
    grid_down = (
        triton.cdiv(hidden_size, BLOCK_D),
        num_tokens,
    )
    _warp_decode_down_kernel[grid_down](
        intermediate_ptr=intermediate,
        intermediate_stride_te=intermediate.stride(0),
        intermediate_stride_n=intermediate.stride(1),
        w_down_ptr=w_down,
        w_down_stride_e=w_down.stride(0),
        w_down_stride_d=w_down.stride(1),
        w_down_stride_n=w_down.stride(2),
        routing_weights_ptr=topk_weights,
        routing_stride_t=topk_weights.stride(0),
        routing_stride_k=topk_weights.stride(1),
        expert_ids_ptr=topk_ids,
        expert_ids_stride_t=topk_ids.stride(0),
        expert_ids_stride_k=topk_ids.stride(1),
        out_ptr=output,
        out_stride_t=output.stride(0),
        out_stride_d=output.stride(1),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        num_tokens=num_tokens,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N_DOWN,
    )

    return output


def warp_decode_moe_packed(
    hidden_states: torch.Tensor,  # [num_tokens, hidden_size]
    w13: torch.Tensor,  # [num_experts, 2 * intermediate_size, hidden_size]
    w2: torch.Tensor,  # [num_experts, hidden_size, intermediate_size]
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    topk_weights: torch.Tensor,  # [num_tokens, top_k]
    intermediate_size: Optional[int] = None,
    inplace: bool = False,
) -> torch.Tensor:
    """Run warp decode MoE with packed gate+up (w13) and down (w2) weights.

    Dispatches to CuTe CUDA kernels when available (SM100+ or
    SGLANG_WARP_DECODE_CUTE=1), otherwise uses Triton kernels.

    This is the standard sglang weight layout where gate and up projections
    are concatenated along the output dimension.

    Args:
        hidden_states: Input activations [num_tokens, hidden_size].
        w13: Packed gate+up weights [E, 2*N, K] in BF16.
        w2: Down projection weights [E, D, N] in BF16.
        topk_ids: Expert IDs per token [num_tokens, top_k].
        topk_weights: Routing weights per token [num_tokens, top_k].
        intermediate_size: Override for N (default: w13.shape[1] // 2).
        inplace: If True, add result to hidden_states in-place.

    Returns:
        Output tensor [num_tokens, hidden_size].
    """
    # Input validation
    assert hidden_states.ndim == 2, f"hidden_states must be 2D, got {hidden_states.ndim}D"
    assert hidden_states.is_cuda, "hidden_states must be on CUDA"
    assert hidden_states.dtype == torch.bfloat16, (
        f"hidden_states must be bfloat16, got {hidden_states.dtype}"
    )

    num_tokens, hidden_size = hidden_states.shape

    # Edge case: zero tokens returns empty output
    if num_tokens == 0:
        if inplace:
            return hidden_states
        return torch.empty_like(hidden_states)

    isize = intermediate_size if intermediate_size is not None else (w13.shape[1] // 2)
    top_k = topk_ids.shape[1]
    # CuTe dispatch
    if _can_use_cute(hidden_size, isize, top_k):
        return sgl_kernel.warp_decode_cute_moe_packed_forward(
            hidden_states, w13, w2,
            topk_ids, topk_weights,
            isize, inplace,
        )

    assert w13.dtype == torch.bfloat16, f"w13 must be bfloat16, got {w13.dtype}"
    assert w2.dtype == torch.bfloat16, f"w2 must be bfloat16, got {w2.dtype}"

    if intermediate_size is None:
        intermediate_size = w13.shape[1] // 2

    assert w13.shape[2] == hidden_size, (
        f"w13 hidden dim {w13.shape[2]} != hidden_size {hidden_size}"
    )
    assert w13.shape[1] == 2 * intermediate_size, (
        f"w13 intermediate dim {w13.shape[1]} != 2 * intermediate_size "
        f"{2 * intermediate_size}"
    )
    assert topk_ids.shape[0] == num_tokens
    assert topk_weights.shape == (num_tokens, top_k)

    # Intermediate buffer
    intermediate = torch.empty(
        (num_tokens * top_k, intermediate_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )

    BLOCK_N, BLOCK_K, BLOCK_D, BLOCK_N_DOWN = _triton_block_sizes(
        hidden_size, intermediate_size, top_k
    )

    BLOCK_N = triton.next_power_of_2(BLOCK_N)
    BLOCK_K = triton.next_power_of_2(BLOCK_K)
    BLOCK_D = triton.next_power_of_2(BLOCK_D)
    BLOCK_N_DOWN = triton.next_power_of_2(BLOCK_N_DOWN)

    # Kernel 1: Gate/Up with packed weights
    grid_gate_up = (
        triton.cdiv(intermediate_size, BLOCK_N),
        num_tokens * top_k,
    )
    _warp_decode_gate_up_packed_kernel[grid_gate_up](
        x_ptr=hidden_states,
        x_stride_t=hidden_states.stride(0),
        x_stride_d=hidden_states.stride(1),
        w13_ptr=w13,
        w13_stride_e=w13.stride(0),
        w13_stride_n=w13.stride(1),
        w13_stride_k=w13.stride(2),
        out_ptr=intermediate,
        out_stride_te=intermediate.stride(0),
        out_stride_n=intermediate.stride(1),
        expert_ids_ptr=topk_ids,
        expert_ids_stride_t=topk_ids.stride(0),
        expert_ids_stride_k=topk_ids.stride(1),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        num_tokens=num_tokens,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Output
    if inplace:
        output = hidden_states
    else:
        output = torch.empty_like(hidden_states)

    # Kernel 2: Down + combine
    grid_down = (
        triton.cdiv(hidden_size, BLOCK_D),
        num_tokens,
    )
    _warp_decode_down_kernel[grid_down](
        intermediate_ptr=intermediate,
        intermediate_stride_te=intermediate.stride(0),
        intermediate_stride_n=intermediate.stride(1),
        w_down_ptr=w2,
        w_down_stride_e=w2.stride(0),
        w_down_stride_d=w2.stride(1),
        w_down_stride_n=w2.stride(2),
        routing_weights_ptr=topk_weights,
        routing_stride_t=topk_weights.stride(0),
        routing_stride_k=topk_weights.stride(1),
        expert_ids_ptr=topk_ids,
        expert_ids_stride_t=topk_ids.stride(0),
        expert_ids_stride_k=topk_ids.stride(1),
        out_ptr=output,
        out_stride_t=output.stride(0),
        out_stride_d=output.stride(1),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        num_tokens=num_tokens,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N_DOWN,
    )

    return output
