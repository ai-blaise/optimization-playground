"""Non-persistent FlashSampling kernel for Blackwell (SM100, B200/GB200).

Same Triton non-persistent design as the H200 target kernel, with
SM100-specific tuning: adaptive pipeline stages for 227 KB max SMEM. This is
the Triton baseline, not a CuTe/CZS implementation.

Validated on B200 (SM 10.0, Triton 3.4, driver 590.48):
- Greedy: bit-exact vs torch reference at V=16160/D=7168 (REAP shape),
  BS=1/4/64
- Sampling: 100% top-1 agreement over 200 Gumbel trials
- Perf: 4.33-4.36 TB/s, matches cuBLAS at BS=1, +9% at BS=64

BF16 weights only; a separate FP4 variant would be needed for native
NVFP4 inference.
"""

import torch
import triton
import triton.language as tl

from .core import (
    MIN_BLOCK_SIZE_V,
    LOCAL_INDEX_DTYPE,
    bsz_h,
    fused_mm_sample_triton,
    num_sms_cached,
    set_torch_allocator_for_tma_descriptors_cached,
    _local_reduce_samples_triton,
)
from .tp_info import TP1, TPInfo


@triton.jit
def _gumbel_noise_blackwell(seed, pid_v, pid_h, sample_idx, noise_offsets):
    return -tl.log(
        -tl.log(
            tl.rand(
                seed + pid_v * 100 + pid_h * 1_000 + sample_idx * 10_000,
                noise_offsets,
            )
        )
    )


@triton.jit
def flashsample_blackwell_kernel(
    weights_ptr,
    hidden_states_ptr,
    max_out_ptr,
    max_out_idx_ptr,
    vocab_size,
    hidden_size: tl.constexpr,
    n_hidden_states: int,
    num_samples: tl.constexpr,
    temperature_ptr,
    seed: int,
    max_grid_size_v: tl.constexpr,
    tp_rank: tl.constexpr,
    num_pid_v: tl.constexpr,
    GREEDY_SAMPLING: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_h = tl.program_id(1)

    if not GREEDY_SAMPLING:
        temperature = tl.load(temperature_ptr)

    v_start = pid_v * BLOCK_SIZE_V
    h_start = pid_h * BLOCK_SIZE_H

    offsets_v = v_start + tl.arange(0, BLOCK_SIZE_V)
    mask_v = offsets_v < vocab_size

    w_desc = tl.make_tensor_descriptor(
        weights_ptr,
        shape=[vocab_size, hidden_size],
        strides=[hidden_size, 1],
        block_shape=[BLOCK_SIZE_V, BLOCK_SIZE_D],
    )
    h_desc = tl.make_tensor_descriptor(
        hidden_states_ptr,
        shape=[n_hidden_states, hidden_size],
        strides=[hidden_size, 1],
        block_shape=[BLOCK_SIZE_H, BLOCK_SIZE_D],
    )

    logits_blk = tl.zeros((BLOCK_SIZE_V, BLOCK_SIZE_H), dtype=tl.float32)
    for d_start in range(0, hidden_size, BLOCK_SIZE_D):
        w_blk = w_desc.load([v_start, d_start])
        h_blk = h_desc.load([h_start, d_start])
        logits_blk = tl.dot(w_blk, h_blk.T, acc=logits_blk)

    logits_blk = tl.where(mask_v[:, None], logits_blk, -float("inf"))

    if not GREEDY_SAMPLING:
        logits_blk = logits_blk / temperature

    for sample_idx in range(num_samples):
        noise_size: tl.constexpr = BLOCK_SIZE_V * BLOCK_SIZE_H
        noise_offsets = tl.arange(0, noise_size).reshape((BLOCK_SIZE_V, BLOCK_SIZE_H))
        if not GREEDY_SAMPLING:
            gumbel_max, gumbel_max_idx_local = tl.max(
                logits_blk + _gumbel_noise_blackwell(
                    seed, pid_v + tp_rank * num_pid_v, pid_h, sample_idx, noise_offsets,
                ),
                axis=0,
                return_indices=True,
            )
        else:
            gumbel_max, gumbel_max_idx_local = tl.max(logits_blk, axis=0, return_indices=True)

        gumbel_max_idx_global = gumbel_max_idx_local + v_start

        offsets_h_out = h_start + tl.arange(0, BLOCK_SIZE_H)
        mask_h_out = offsets_h_out < n_hidden_states
        base_offset = (
            sample_idx * max_grid_size_v * n_hidden_states
            + pid_v * n_hidden_states
            + offsets_h_out
        )
        tl.store(max_out_ptr + base_offset, gumbel_max, mask=mask_h_out)
        tl.store(max_out_idx_ptr + base_offset, gumbel_max_idx_global, mask=mask_h_out)


_BLOCK_V_BLACKWELL = MIN_BLOCK_SIZE_V
_BLOCK_D_BLACKWELL = 128
_NUM_WARPS_BLACKWELL = 8
_B200_MAX_SMEM = 232448


def _num_stages_blackwell(block_h: int) -> int:
    smem_per_stage = (_BLOCK_V_BLACKWELL + block_h) * _BLOCK_D_BLACKWELL * 2
    for stages in (5, 4, 3, 2):
        if smem_per_stage * stages <= _B200_MAX_SMEM:
            return stages
    return 1


def _block_h_blackwell(
    num_hidden_states: int, greedy_sampling: bool = True
) -> int:
    if num_hidden_states == 1 and not greedy_sampling:
        return 8
    if 1 < num_hidden_states <= 8:
        return 8
    return max(16, bsz_h(num_hidden_states))


def _max_nonpersistent_tiles_blackwell(num_sms: int, greedy_sampling: bool) -> int:
    if greedy_sampling:
        return num_sms
    return 2 * num_sms


def _launch_num_stages_blackwell(block_h: int, two_wave_nongreedy: bool) -> int:
    if two_wave_nongreedy:
        return 2
    return _num_stages_blackwell(block_h)


def fused_mm_sample_blackwell(
    weights: torch.Tensor,
    hidden_states: torch.Tensor,
    num_samples: int,
    temperature: torch.Tensor,
    seed: int,
    greedy_sampling: bool = False,
    tp: "TPInfo" = TP1,
    return_logits: bool = False,
    return_scores: bool = False,
    valid_vocab_size: int | None = None,
    vocab_start_index: int | None = None,
    maxs_workspace: torch.Tensor | None = None,
    maxs_idx_workspace: torch.Tensor | None = None,
    logits_out_workspace: torch.Tensor | None = None,
):
    """FlashSampling for Blackwell (SM100) TP-sharded vocab shapes.

    Uses the non-persistent target kernel for production TP-sharded shapes and
    falls back to the persistent FlashSampling kernel outside the target gate.
    """
    V_local, D = weights.shape
    V = valid_vocab_size if valid_vocab_size is not None else V_local
    H = hidden_states.shape[0]
    NUM_SMS = num_sms_cached(weights.device.index)

    num_pid_v = triton.cdiv(V, _BLOCK_V_BLACKWELL)
    BLOCK_H = _block_h_blackwell(H, greedy_sampling)
    num_pid_h = triton.cdiv(H, BLOCK_H)
    total_tiles = num_pid_v * num_pid_h
    # Non-greedy sampling pays extra RNG/log cost in the persistent fallback on
    # Blackwell. B200 measurements show the non-persistent target kernel is
    # faster through two SM waves for H=72..128, while greedy regresses there.
    max_nonpersistent_tiles = _max_nonpersistent_tiles_blackwell(
        NUM_SMS, greedy_sampling
    )
    uses_two_wave_nongreedy = (
        not greedy_sampling and NUM_SMS < total_tiles <= max_nonpersistent_tiles
    )

    if total_tiles > max_nonpersistent_tiles or return_logits or tp.size > 1:
        return fused_mm_sample_triton(
            weights=weights,
            hidden_states=hidden_states,
            num_samples=num_samples,
            temperature=temperature,
            seed=seed,
            greedy_sampling=greedy_sampling,
            tp=tp,
            return_logits=return_logits,
            return_scores=return_scores,
            valid_vocab_size=valid_vocab_size,
            vocab_start_index=vocab_start_index,
            maxs_workspace=maxs_workspace,
            maxs_idx_workspace=maxs_idx_workspace,
            logits_out_workspace=logits_out_workspace,
        )

    set_torch_allocator_for_tma_descriptors_cached()

    max_grid_v = triton.cdiv(V, MIN_BLOCK_SIZE_V)
    maxs_shape = (num_samples, max_grid_v, H)

    if maxs_workspace is not None:
        maxs = maxs_workspace
    else:
        maxs = torch.empty(maxs_shape, dtype=torch.bfloat16, device=weights.device)

    if maxs_idx_workspace is not None:
        maxs_idx = maxs_idx_workspace
    else:
        maxs_idx = torch.empty(maxs_shape, dtype=LOCAL_INDEX_DTYPE, device=weights.device)

    flashsample_blackwell_kernel[(num_pid_v, num_pid_h)](
        weights_ptr=weights,
        hidden_states_ptr=hidden_states,
        max_out_ptr=maxs,
        max_out_idx_ptr=maxs_idx,
        vocab_size=V,
        hidden_size=D,
        n_hidden_states=H,
        num_samples=num_samples,
        temperature_ptr=temperature,
        seed=seed,
        max_grid_size_v=max_grid_v,
        tp_rank=tp.rank,
        num_pid_v=num_pid_v,
        GREEDY_SAMPLING=greedy_sampling,
        BLOCK_SIZE_V=_BLOCK_V_BLACKWELL,
        BLOCK_SIZE_D=_BLOCK_D_BLACKWELL,
        BLOCK_SIZE_H=BLOCK_H,
        num_warps=_NUM_WARPS_BLACKWELL,
        num_stages=_launch_num_stages_blackwell(
            BLOCK_H, uses_two_wave_nongreedy
        ),
    )

    if vocab_start_index is None:
        vocab_start_index = tp.rank * V

    maxs_sliced = maxs[:, :num_pid_v, :]
    maxs_idx_sliced = maxs_idx[:, :num_pid_v, :]

    if return_scores:
        from .core import _local_reduce

        samples, max_values = _local_reduce(
            maxs_sliced, maxs_idx_sliced, vocab_start_index
        )
        return samples, max_values

    return _local_reduce_samples_triton(maxs_sliced, maxs_idx_sliced, vocab_start_index)


def fused_mm_sample_adaptive(
    weights: torch.Tensor,
    hidden_states: torch.Tensor,
    num_samples: int,
    temperature: torch.Tensor,
    seed: int,
    greedy_sampling: bool = False,
    tp: "TPInfo" = TP1,
    return_logits: bool = False,
    return_scores: bool = False,
    valid_vocab_size: int | None = None,
    vocab_start_index: int | None = None,
    maxs_workspace: torch.Tensor | None = None,
    maxs_idx_workspace: torch.Tensor | None = None,
    logits_out_workspace: torch.Tensor | None = None,
):
    """Auto-select H200 or Blackwell kernel based on device capability."""
    cc = torch.cuda.get_device_capability(weights.device.index)
    if cc[0] >= 10:
        return fused_mm_sample_blackwell(
            weights=weights,
            hidden_states=hidden_states,
            num_samples=num_samples,
            temperature=temperature,
            seed=seed,
            greedy_sampling=greedy_sampling,
            tp=tp,
            return_logits=return_logits,
            return_scores=return_scores,
            valid_vocab_size=valid_vocab_size,
            vocab_start_index=vocab_start_index,
            maxs_workspace=maxs_workspace,
            maxs_idx_workspace=maxs_idx_workspace,
            logits_out_workspace=logits_out_workspace,
        )
    else:
        from .target_kernel import fused_mm_sample_target
        return fused_mm_sample_target(
            weights=weights,
            hidden_states=hidden_states,
            num_samples=num_samples,
            temperature=temperature,
            seed=seed,
            greedy_sampling=greedy_sampling,
            tp=tp,
            return_logits=return_logits,
            return_scores=return_scores,
            valid_vocab_size=valid_vocab_size,
            vocab_start_index=vocab_start_index,
            maxs_workspace=maxs_workspace,
            maxs_idx_workspace=maxs_idx_workspace,
            logits_out_workspace=logits_out_workspace,
        )
