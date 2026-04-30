from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernel_api_logging import debug_kernel_api


_MAX_BLOCK_H = 128
_MAX_BLOCK_R = 64
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _next_power_of_2(value: int, maximum: int) -> int:
    return min(triton.next_power_of_2(value), maximum)


def _validate_gated_norm_inputs(
    normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    out: Optional[torch.Tensor],
) -> tuple[int, int]:
    if not normed.is_cuda or not w_down.is_cuda or not w_up.is_cuda:
        raise RuntimeError("gated_norm_forward requires CUDA tensors")
    if normed.dim() < 2:
        raise ValueError(f"normed must have at least 2 dimensions, got {tuple(normed.shape)}")
    if normed.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"normed must have dtype bf16, fp16, or fp32; got {normed.dtype}")
    if w_down.dtype != normed.dtype or w_up.dtype != normed.dtype:
        raise TypeError(
            "w_down and w_up must have the same dtype as normed: "
            f"normed={normed.dtype}, w_down={w_down.dtype}, w_up={w_up.dtype}"
        )

    hidden_size = normed.shape[-1]
    if w_down.dim() != 2 or w_up.dim() != 2:
        raise ValueError("w_down and w_up must be rank-2 tensors")
    if w_down.shape[1] != hidden_size:
        raise ValueError(
            f"w_down must have shape [rank, hidden_size]; got {tuple(w_down.shape)} "
            f"for hidden_size={hidden_size}"
        )
    if w_up.shape[0] != hidden_size:
        raise ValueError(
            f"w_up must have shape [hidden_size, rank]; got {tuple(w_up.shape)} "
            f"for hidden_size={hidden_size}"
        )
    rank = w_down.shape[0]
    if rank != w_up.shape[1]:
        raise ValueError(
            "w_down and w_up must agree on the gating rank: "
            f"got {w_down.shape[0]} and {w_up.shape[1]}"
        )
    if rank <= 0 or rank > _MAX_BLOCK_R:
        raise ValueError(f"gated_norm_forward supports 1 <= rank <= {_MAX_BLOCK_R}; got {rank}")

    if out is not None:
        if not out.is_cuda:
            raise RuntimeError("out must be a CUDA tensor")
        if out.shape != normed.shape:
            raise ValueError(f"out shape must match normed shape: {tuple(out.shape)} != {tuple(normed.shape)}")
        if out.dtype != normed.dtype:
            raise TypeError(f"out dtype must match normed dtype: {out.dtype} != {normed.dtype}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    return hidden_size, rank


@triton.jit
def _gated_norm_forward_kernel(
    y_ptr,
    w_down_ptr,
    w_up_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    rank: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    token_idx = tl.program_id(0)
    h_offsets = tl.arange(0, BLOCK_H)
    r_offsets = tl.arange(0, BLOCK_R)
    r_mask = r_offsets < rank

    z = tl.zeros((BLOCK_R,), tl.float32)
    for h_start in tl.range(0, hidden_size, BLOCK_H):
        h = h_start + h_offsets
        h_mask = h < hidden_size
        y = tl.load(y_ptr + token_idx * hidden_size + h, mask=h_mask, other=0.0)
        w_down = tl.load(
            w_down_ptr + r_offsets[:, None] * hidden_size + h[None, :],
            mask=r_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        z += tl.sum(w_down.to(tl.float32) * y[None, :].to(tl.float32), axis=1)

    sigmoid_z = 1.0 / (1.0 + tl.exp(-z))
    activation = tl.where(r_mask, z * sigmoid_z, 0.0)
    for h_start in tl.range(0, hidden_size, BLOCK_H):
        h = h_start + h_offsets
        h_mask = h < hidden_size
        w_up = tl.load(
            w_up_ptr + h[:, None] * rank + r_offsets[None, :],
            mask=h_mask[:, None] & r_mask[None, :],
            other=0.0,
        )
        logits = tl.sum(w_up.to(tl.float32) * activation[None, :], axis=1)
        gate = 1.0 / (1.0 + tl.exp(-logits))
        y = tl.load(y_ptr + token_idx * hidden_size + h, mask=h_mask, other=0.0)
        tl.store(output_ptr + token_idx * hidden_size + h, y * gate, mask=h_mask)


@debug_kernel_api
def gated_norm_forward(
    normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply forward-only GatedNorm to an already normalized tensor.

    Formula:
        output = normed * sigmoid(silu(normed @ w_down.T) @ w_up.T)

    The implementation is the forward kernel from the Blaise Megatron-LM
    GatedNorm prototype, adapted for SGLang inference use without backward or
    autograd integration.
    """

    hidden_size, rank = _validate_gated_norm_inputs(normed, w_down, w_up, out)
    flat_normed = normed.reshape(-1, hidden_size).contiguous()
    output = (
        out
        if out is not None
        else torch.empty_like(normed, memory_format=torch.contiguous_format)
    )
    flat_output = output.reshape(-1, hidden_size)

    num_tokens = flat_normed.shape[0]
    if num_tokens == 0:
        return output

    block_h = _next_power_of_2(hidden_size, _MAX_BLOCK_H)
    block_r = _next_power_of_2(rank, _MAX_BLOCK_R)
    _gated_norm_forward_kernel[(num_tokens,)](
        flat_normed,
        w_down.contiguous(),
        w_up.contiguous(),
        flat_output,
        hidden_size,
        rank,
        BLOCK_H=block_h,
        BLOCK_R=block_r,
        num_warps=4,
    )
    return output


apply_gated_norm = gated_norm_forward
