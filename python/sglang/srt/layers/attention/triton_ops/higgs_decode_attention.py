# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""Fused HIGGS-MHA-2bit dequant-inside-attention decode kernel.

Used by the SMC-SD draft path when ``--smc-draft-kv-cache-dtype higgs_2bit``
selects :class:`HiggsMHA2BitTokenToKVPool`. Streams packed slots
(:func:`HiggsMHA2BitCodec.compress`) into the attention tile and
reconstructs BF16 K/V *inside the kernel* — codebook lookup, scale
multiply, and an in-register FWHT_128 — so the per-decode cost is the
attention math itself plus a tile of bit-unpacking, not a full per-layer
materialization of the BF16 cache.

Slot layout (per K-head or V-head, per token, ``head_dim=128``):

    [packed_indices: 32 B][scale: fp16 = 2 B] = 34 B

References:
- Codec format: :mod:`sglang.srt.layers.quantization.higgs_mha_2bit_kv`.
- Eager FWHT reference: ``_fwht`` in :mod:`...higgs_dense_2bit_kv`.
- Baseline dense Triton decode (the shape the HIGGS kernel mirrors):
  ``_fwd_grouped_kernel_stage1`` in
  :mod:`sglang.srt.layers.attention.triton_ops.decode_attention`.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _MIN_BLOCK_KV,
    _fwd_kernel_stage2,
    tanh,
)
from sglang.srt.layers.quantization.higgs_dense_2bit_kv import HIGGS_NORM_BYTES

logger = logging.getLogger(__name__)


# Slot format constants (must match HiggsMHA2BitCodec on head_dim=128).
# Constants referenced inside @triton.jit functions must be wrapped in
# triton.language.constexpr so the JIT compiler accepts them as
# compile-time globals (Triton >=3.5 disallows bare-int globals in JIT).
# Plain-int aliases are kept for host-side arithmetic + asserts that need a
# regular Python int.
_HIGGS_HEAD_DIM_INT = 128
_HIGGS_PAIR_DIM_INT = 2
_HIGGS_NUM_PAIRS_INT = _HIGGS_HEAD_DIM_INT // _HIGGS_PAIR_DIM_INT  # 64
_HIGGS_PACKED_BYTES_INT = _HIGGS_NUM_PAIRS_INT // 2  # 32
HIGGS_HEAD_DIM = tl.constexpr(_HIGGS_HEAD_DIM_INT)
HIGGS_PAIR_DIM = tl.constexpr(_HIGGS_PAIR_DIM_INT)
HIGGS_NUM_PAIRS = tl.constexpr(_HIGGS_NUM_PAIRS_INT)
HIGGS_PACKED_BYTES = tl.constexpr(_HIGGS_PACKED_BYTES_INT)
HIGGS_SLOT_BYTES = _HIGGS_PACKED_BYTES_INT + HIGGS_NORM_BYTES  # 34, host-side int
HIGGS_INV_SQRT_HEAD_DIM = 1.0 / math.sqrt(_HIGGS_HEAD_DIM_INT)


@triton.jit
def _eden2_xy(idx):
    # EDEN2-16 codebook lookup. ``idx`` is uint8/int32 in [0, 15];
    # returns (x, y) pair via two chains of selects. Values are the
    # exact float32 codebook from
    # :mod:`...higgs_dense_2bit_kv.HIGGS_EDEN2_16`.
    x = (
        tl.where(idx == 0, -0.8996632695198059, 0.0)
        + tl.where(idx == 1, -0.961183488368988, 0.0)
        + tl.where(idx == 2, -1.882026195526123, 0.0)
        + tl.where(idx == 3, 0.36300793290138245, 0.0)
        + tl.where(idx == 4, -0.6814072728157043, 0.0)
        + tl.where(idx == 5, 0.7270012497901917, 0.0)
        + tl.where(idx == 6, 0.3359416127204895, 0.0)
        + tl.where(idx == 7, 1.859930396080017, 0.0)
        + tl.where(idx == 8, 0.17208248376846313, 0.0)
        + tl.where(idx == 9, -1.7599700689315796, 0.0)
        + tl.where(idx == 10, -0.8993809223175049, 0.0)
        + tl.where(idx == 11, 0.839488685131073, 0.0)
        + tl.where(idx == 12, 1.5314953327178955, 0.0)
        + tl.where(idx == 13, -0.0011779458727687597, 0.0)
        + tl.where(idx == 14, 1.4274526834487915, 0.0)
        + tl.where(idx == 15, -0.16123905777931213, 0.0)
    )
    y = (
        tl.where(idx == 0, -1.6360418796539307, 0.0)
        + tl.where(idx == 1, 1.5999565124511719, 0.0)
        + tl.where(idx == 2, 0.678778350353241, 0.0)
        + tl.where(idx == 3, -1.9667866230010986, 0.0)
        + tl.where(idx == 4, -0.576818585395813, 0.0)
        + tl.where(idx == 5, 0.6186859607696533, 0.0)
        + tl.where(idx == 6, 1.8371193408966064, 0.0)
        + tl.where(idx == 7, 0.036668598651885986, 0.0)
        + tl.where(idx == 8, -0.9401724338531494, 0.0)
        + tl.where(idx == 9, -0.6244229674339294, 0.0)
        + tl.where(idx == 10, 0.32267823815345764, 0.0)
        + tl.where(idx == 11, -0.3017036020755768, 0.0)
        + tl.where(idx == 12, 1.2942044734954834, 0.0)
        + tl.where(idx == 13, 0.00022069070837460458, 0.0)
        + tl.where(idx == 14, -1.207889199256897, 0.0)
        + tl.where(idx == 15, 0.8787511587142944, 0.0)
    )
    return x, y


@triton.jit
def _fwht_stage(rotated, BLOCK_N: tl.constexpr, N_BLOCKS: tl.constexpr,
                STRIDE: tl.constexpr):
    # One butterfly stage with the given ``STRIDE``. Mirrors the eager
    # codec ``_fwht``:
    #
    #   view = x.reshape(BLOCK_N, n_blocks, 2, stride)
    #   a, b = view[:, :, 0, :], view[:, :, 1, :]
    #   out[:, :, 0, :] = a + b
    #   out[:, :, 1, :] = a - b
    #
    # ``tl.join(a + b, a - b)`` creates a tensor of shape
    # ``(BLOCK_N, n_blocks, stride, 2)`` — the new size-2 axis is the
    # *innermost*. We swap it back to second-from-innermost via
    # ``tl.trans`` so the recombined flat ordering matches the eager
    # codec exactly.
    # Triton 3.5+ disallows tensor[constexpr_int_idx]; use tl.split after
    # permuting the size-2 pair axis to be innermost.
    v = tl.reshape(rotated, (BLOCK_N, N_BLOCKS, 2, STRIDE))
    v_perm = tl.trans(v, (0, 1, 3, 2))  # (BLOCK_N, N_BLOCKS, STRIDE, 2)
    a, b = tl.split(v_perm)  # each (BLOCK_N, N_BLOCKS, STRIDE)
    joined = tl.join(a + b, a - b)  # (BLOCK_N, N_BLOCKS, STRIDE, 2)
    swapped = tl.trans(joined, (0, 1, 3, 2))  # (BLOCK_N, N_BLOCKS, 2, STRIDE)
    return tl.reshape(swapped, (BLOCK_N, 128))


@triton.jit
def _fwht_butterfly_128(rotated, BLOCK_N: tl.constexpr):
    # Seven-stage orthonormal FWHT_128 along the last axis of
    # ``rotated`` (shape ``(BLOCK_N, 128)``, fp32). Returns the
    # 1/sqrt(128)-scaled result so the transform is self-inverse.
    rotated = _fwht_stage(rotated, BLOCK_N, 64, 1)
    rotated = _fwht_stage(rotated, BLOCK_N, 32, 2)
    rotated = _fwht_stage(rotated, BLOCK_N, 16, 4)
    rotated = _fwht_stage(rotated, BLOCK_N, 8, 8)
    rotated = _fwht_stage(rotated, BLOCK_N, 4, 16)
    rotated = _fwht_stage(rotated, BLOCK_N, 2, 32)
    rotated = _fwht_stage(rotated, BLOCK_N, 1, 64)
    return rotated * 0.08838834764831845  # 1.0 / sqrt(128)


@triton.jit
def _dequant_higgs_kv_tile(
    Packed,  # uint8*, base ptr of the packed K or V buffer
    ScaleView,  # fp16*, same underlying memory viewed as fp16
    row_byte_offs,  # int64 (BLOCK_N,) — byte offset to slot start for each token
    row_mask,  # bool (BLOCK_N,) — true if row is in-bounds
    BLOCK_N: tl.constexpr,
):
    # Returns a ``(BLOCK_N, HIGGS_HEAD_DIM)`` fp32 tile of dequantized
    # K (or V) rows for a single KV head. Each row dequantizes
    # independently — codebook lookup -> scale -> FWHT_128.

    # Step 1: load the 32 packed bytes per row.
    offs_b = tl.arange(0, HIGGS_PACKED_BYTES)  # 0..31
    byte_ptrs = Packed + row_byte_offs[:, None] + offs_b[None, :]
    bytes_loaded = tl.load(
        byte_ptrs,
        mask=row_mask[:, None],
        other=0,
    )

    # Step 2: unpack each byte into two 4-bit indices. lo=(b & 0xF)
    # covers pair index 0, 2, 4, ...; hi=(b >> 4) & 0xF covers
    # pair index 1, 3, 5, ... (matches HiggsMHA2BitCodec).
    lo_idx = (bytes_loaded & 0x0F).to(tl.int32)
    hi_idx = ((bytes_loaded >> 4) & 0x0F).to(tl.int32)

    # Step 3: codebook lookup. Each pair contributes two scalars
    # (x = first component, y = second component) into adjacent
    # head_dim positions (2i, 2i+1).
    lo_x, lo_y = _eden2_xy(lo_idx)  # (BLOCK_N, 32) each
    hi_x, hi_y = _eden2_xy(hi_idx)

    # Interleave into the head_dim layout:
    #   dim 4k+0 = lo_idx[k].x
    #   dim 4k+1 = lo_idx[k].y
    #   dim 4k+2 = hi_idx[k].x
    #   dim 4k+3 = hi_idx[k].y
    # k = 0..31 (= HIGGS_PACKED_BYTES). 32 * 4 = 128 = head_dim.
    lo_pairs = tl.join(lo_x, lo_y)  # (BLOCK_N, 32, 2) — last dim: [x, y]
    hi_pairs = tl.join(hi_x, hi_y)  # (BLOCK_N, 32, 2)
    # tl.join(lo_pairs, hi_pairs) lays out as (BLOCK_N, 32, xy=2, lh=2),
    # which flattens to [lo_x[k], hi_x[k], lo_y[k], hi_y[k]] per k —
    # swapping positions 4k+1 and 4k+2 relative to the codec layout
    # ([pair[2k].x, pair[2k].y, pair[2k+1].x, pair[2k+1].y]). Transpose
    # the last two dims so the lo/hi axis sits before the x/y axis;
    # then the flatten produces [lo_x, lo_y, hi_x, hi_y] = the codec
    # layout.
    pair_groups = tl.join(lo_pairs, hi_pairs)  # (BLOCK_N, 32, 2_xy, 2_lh)
    pair_groups = tl.trans(pair_groups, (0, 1, 3, 2))  # (BLOCK_N, 32, 2_lh, 2_xy)
    rotated = tl.reshape(pair_groups, (BLOCK_N, HIGGS_HEAD_DIM))

    # Step 4: load per-row FP16 scale. The packed buffer ``ScaleView``
    # aliases the same memory as ``Packed`` but with element type
    # fp16; each slot is 34 B = 17 fp16, scale sits at fp16-index 16
    # (= byte offset 32 inside the slot). ``row_byte_offs`` is
    # byte-addressed into ``Packed``; convert to fp16-index by halving.
    row_fp16_offs = row_byte_offs // 2
    scale_f16 = tl.load(
        ScaleView + row_fp16_offs + (HIGGS_PACKED_BYTES // 2),
        mask=row_mask,
        other=0.0,
    )
    scale_f = scale_f16.to(tl.float32)
    rotated = rotated * scale_f[:, None]

    # Step 5: in-register FWHT_128 (orthonormal, self-inverse with
    # 1/sqrt(128) prefactor).
    return _fwht_butterfly_128(rotated, BLOCK_N)


@triton.jit
def _higgs_fwd_grouped_kernel_stage1(
    Q,
    K_Packed,
    V_Packed,
    K_Scale_View,
    V_Scale_View,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    # GQA decode kernel mirror of ``_fwd_grouped_kernel_stage1`` adapted
    # for HIGGS-packed K/V. One program handles one
    # (batch, q-head-block, kv-split) triple. All Q heads in a block
    # share the same KV head (= cur_head_id // cdiv(kv_group_num, BLOCK_H)).
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_q = (
        cur_batch * stride_qbs
        + cur_head[:, None] * stride_qh
        + offs_d[None, :]
    )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(
            Q + offs_q,
            mask=(mask_h[:, None]) & (mask_d[None, :]),
            other=0.0,
        )
        q_bf16 = q.to(tl.bfloat16)

        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            row_mask = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=row_mask,
                other=0,
            )

            # Byte offset to the start of (token_loc, cur_kv_head) slot
            # inside K_Packed. stride_buf_kbs / kh are in bytes
            # because K_Packed is uint8.
            row_byte_offs_k = (
                kv_loc.to(tl.int64) * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
            )
            row_byte_offs_v = (
                kv_loc.to(tl.int64) * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
            )

            k_tile = _dequant_higgs_kv_tile(
                K_Packed,
                K_Scale_View,
                row_byte_offs_k,
                row_mask,
                BLOCK_N=BLOCK_N,
            )  # (BLOCK_N, head_dim) fp32

            # qk = q @ k.T, broadcasting K over the BLOCK_H Q heads.
            # k.T shape: (head_dim, BLOCK_N).
            k_bf16 = k_tile.to(tl.bfloat16)
            k_T = tl.trans(k_bf16)  # (head_dim, BLOCK_N)
            qk = tl.dot(q_bf16, k_T)
            qk *= sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                qk,
                float("-inf"),
            )

            v_tile = _dequant_higgs_kv_tile(
                V_Packed,
                V_Scale_View,
                row_byte_offs_v,
                row_mask,
                BLOCK_N=BLOCK_N,
            )  # (BLOCK_N, head_dim) fp32
            v_bf16 = v_tile.to(tl.bfloat16)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(tl.bfloat16), v_bf16)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _higgs_decode_grouped_att_m_fwd(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    att_out: torch.Tensor,
    att_lse: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale_withk: float,
    logit_cap: float,
):
    """Stage 1 launcher for the fused HIGGS GQA decode kernel.

    Args:
      q: ``(num_tokens, num_q_heads, head_dim)`` BF16.
      k_packed: ``(num_total_kv_tokens, num_kv_heads, HIGGS_SLOT_BYTES)``
        ``uint8`` packed HIGGS slots.
      v_packed: same shape as ``k_packed``.
      att_out: ``(num_tokens, num_q_heads, max_kv_splits, head_dim)`` BF16.
      att_lse: ``(num_tokens, num_q_heads, max_kv_splits)`` FP32.
      kv_indptr, kv_indices: standard SGLang page-table.
      num_kv_splits, max_kv_splits: per-batch + ceiling on KV split count.
      sm_scale_withk: ``layer.scaling`` (typically ``1/sqrt(head_dim)``).
      logit_cap: optional tanh logit capping value (0 disables).
    """

    Lk = _HIGGS_HEAD_DIM_INT
    Lv = _HIGGS_HEAD_DIM_INT
    assert q.shape[-1] == Lk, (
        f"HIGGS-MHA decode expects head_dim={Lk}; got {q.shape[-1]}"
    )
    assert k_packed.shape[-1] == HIGGS_SLOT_BYTES, (
        f"K_Packed last-dim must be {HIGGS_SLOT_BYTES} (uint8 slot); "
        f"got {k_packed.shape[-1]}"
    )
    assert v_packed.shape[-1] == HIGGS_SLOT_BYTES, (
        f"V_Packed last-dim must be {HIGGS_SLOT_BYTES} (uint8 slot); "
        f"got {v_packed.shape[-1]}"
    )
    assert k_packed.dtype == torch.uint8 and v_packed.dtype == torch.uint8

    batch, num_q_heads = q.shape[0], q.shape[1]
    num_kv_heads = k_packed.shape[1]
    kv_group_num = num_q_heads // num_kv_heads

    BLOCK_DMODEL = Lk
    BLOCK_DV = Lv
    BLOCK_N = 32
    BLOCK_H = 16

    grid = (
        batch,
        triton.cdiv(num_q_heads, min(BLOCK_H, kv_group_num)),
        max_kv_splits,
    )

    # Aliased fp16 view of the packed buffer for in-kernel scale loads.
    # The codec lays each slot out as 17 fp16 (= 34 bytes); scale is
    # the 17th fp16. ``view(torch.float16)`` reinterprets bytes
    # without copying; ``reshape`` is a no-op since the original
    # layout is contiguous on the last dim.
    k_scale_view = k_packed.view(torch.float16)
    v_scale_view = v_packed.view(torch.float16)

    _higgs_fwd_grouped_kernel_stage1[grid](
        q,
        k_packed,
        v_packed,
        k_scale_view,
        v_scale_view,
        sm_scale_withk,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_packed.stride(0),
        k_packed.stride(1),
        v_packed.stride(0),
        v_packed.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=num_q_heads,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        Lk=Lk,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )


def decode_attention_fwd_higgs(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    o: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale_withk: float,
    logit_cap: float = 0.0,
    sinks: Optional[torch.Tensor] = None,
):
    """Fused HIGGS-MHA-2bit GQA decode entry point.

    Mirrors :func:`decode_attention_fwd` but takes ``k_packed`` /
    ``v_packed`` (uint8, 34 B/head/token) directly and dequantizes in
    the attention tile. Stage-2 reuses the dense decode reducer.

    Args:
      q: ``(num_tokens, num_q_heads, head_dim)`` BF16.
      k_packed, v_packed: ``(total_kv_tokens, num_kv_heads, 34)`` uint8.
      o: ``(num_tokens, num_q_heads, head_dim)`` BF16 output.
      kv_indptr, kv_indices: page-table indirection.
      attn_logits: ``(num_tokens, num_q_heads, max_kv_splits, head_dim)``.
      attn_lse: ``(num_tokens, num_q_heads, max_kv_splits)``.
      num_kv_splits: ``(num_tokens,)`` int32.
      max_kv_splits: upper bound on KV splits.
      sm_scale_withk: ``layer.scaling``.
      logit_cap: tanh logit cap (0 disables).
      sinks: optional per-q-head attention sink (passthrough to stage2).
    """

    _higgs_decode_grouped_att_m_fwd(
        q,
        k_packed,
        v_packed,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale_withk,
        logit_cap,
    )

    # Stage 2: identical to the dense decode reducer, with Lv pinned
    # to ``HIGGS_HEAD_DIM`` instead of inferred from V_Packed (whose
    # trailing dim is the 34-byte slot, not head_dim).
    batch, num_q_heads, _ = q.shape
    BLOCK_DV = triton.next_power_of_2(_HIGGS_HEAD_DIM_INT)
    grid = (batch, num_q_heads)
    _fwd_kernel_stage2[grid](
        attn_logits,
        attn_lse,
        o,
        1.0,  # v_scale: identity (HIGGS dequant already in-kernel).
        kv_indptr,
        num_kv_splits,
        sinks,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=max_kv_splits,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=_HIGGS_HEAD_DIM_INT,
        HAS_SINK=sinks is not None,
        num_warps=4,
        num_stages=2,
    )
