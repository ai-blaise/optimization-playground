"""Triton packer for HIGGS 2-bit MHA/GQA KV rows."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.higgs_decode_attention import (
    HIGGS_HEAD_DIM,
    HIGGS_PACKED_BYTES,
    _fwht_butterfly_128,
)


def _require_supported_tensors(
    packed: torch.Tensor, locs: torch.Tensor, cache: torch.Tensor
) -> None:
    assert packed.is_cuda
    assert locs.is_cuda
    assert cache.is_cuda
    assert packed.dtype == torch.uint8
    assert locs.dtype in (torch.int32, torch.int64)
    assert cache.dtype == torch.bfloat16
    assert packed.shape[1] == cache.shape[1]
    assert packed.shape[2] == 34
    assert cache.shape[2] == 128


@triton.jit
def _update_best(best, best_idx, score, idx: tl.constexpr):
    take = score > best
    best = tl.where(take, score, best)
    best_idx = tl.where(take, idx, best_idx)
    return best, best_idx


@triton.jit
def _nearest_eden2_idx(x0, x1):
    best = tl.full(x0.shape, -3.4028234663852886e38, tl.float32)
    best_idx = tl.zeros(x0.shape, tl.int32)
    score = (
        2.0 * (x0 * -0.8996632695198059 + x1 * -1.6360418796539307)
        - 3.486027240753174
    )
    best, best_idx = _update_best(best, best_idx, score, 0)
    score = (
        2.0 * (x0 * -0.961183488368988 + x1 * 1.5999565124511719)
        - 3.483734607696533
    )
    best, best_idx = _update_best(best, best_idx, score, 1)
    score = (
        2.0 * (x0 * -1.882026195526123 + x1 * 0.678778350353241)
        - 4.002762794494629
    )
    best, best_idx = _update_best(best, best_idx, score, 2)
    score = (
        2.0 * (x0 * 0.36300793290138245 + x1 * -1.9667866230010986)
        - 4.000024318695068
    )
    best, best_idx = _update_best(best, best_idx, score, 3)
    score = (
        2.0 * (x0 * -0.6814072728157043 + x1 * -0.576818585395813)
        - 0.7970355749130249
    )
    best, best_idx = _update_best(best, best_idx, score, 4)
    score = (
        2.0 * (x0 * 0.7270012497901917 + x1 * 0.6186859607696533)
        - 0.9113031625747681
    )
    best, best_idx = _update_best(best, best_idx, score, 5)
    score = (
        2.0 * (x0 * 0.3359416127204895 + x1 * 1.8371193408966064)
        - 3.4878642559051514
    )
    best, best_idx = _update_best(best, best_idx, score, 6)
    score = (
        2.0 * (x0 * 1.859930396080017 + x1 * 0.036668598651885986)
        - 3.4606857299804688
    )
    best, best_idx = _update_best(best, best_idx, score, 7)
    score = (
        2.0 * (x0 * 0.17208248376846313 + x1 * -0.9401724338531494)
        - 0.913536548614502
    )
    best, best_idx = _update_best(best, best_idx, score, 8)
    score = (
        2.0 * (x0 * -1.7599700689315796 + x1 * -0.6244229674339294)
        - 3.487398624420166
    )
    best, best_idx = _update_best(best, best_idx, score, 9)
    score = (
        2.0 * (x0 * -0.8993809223175049 + x1 * 0.32267823815345764)
        - 0.9130073189735413
    )
    best, best_idx = _update_best(best, best_idx, score, 10)
    score = (
        2.0 * (x0 * 0.839488685131073 + x1 * -0.3017036020755768)
        - 0.795766294002533
    )
    best, best_idx = _update_best(best, best_idx, score, 11)
    score = (
        2.0 * (x0 * 1.5314953327178955 + x1 * 1.2942044734954834)
        - 4.020443439483643
    )
    best, best_idx = _update_best(best, best_idx, score, 12)
    score = (
        2.0 * (x0 * -0.0011779458727687597 + x1 * 0.00022069070837460458)
        - 1.4362608454e-06
    )
    best, best_idx = _update_best(best, best_idx, score, 13)
    score = (
        2.0 * (x0 * 1.4274526834487915 + x1 * -1.207889199256897)
        - 3.496617555618286
    )
    best, best_idx = _update_best(best, best_idx, score, 14)
    score = (
        2.0 * (x0 * -0.16123905777931213 + x1 * 0.8787511587142944)
        - 0.7982016801834106
    )
    best, best_idx = _update_best(best, best_idx, score, 15)
    return best_idx


@triton.jit
def _store_higgs_mha_2bit_kernel(
    Packed,
    PackedScale,
    Locs,
    Cache,
    num_rows: tl.constexpr,
    packed_stride_0: tl.constexpr,
    packed_stride_1: tl.constexpr,
    cache_stride_0: tl.constexpr,
    cache_stride_1: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    head = tl.program_id(1)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < num_rows
    dims = tl.arange(0, HIGGS_HEAD_DIM)
    x = tl.load(
        Cache + rows[:, None] * cache_stride_0 + head * cache_stride_1 + dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    rotated = _fwht_butterfly_128(x, BLOCK_N)
    sum_sq = tl.sum(rotated * rotated, axis=1)
    scale = tl.sqrt(tl.maximum(sum_sq, 1.0e-32)) * 0.08838834764831845
    normalized = rotated / tl.maximum(scale[:, None], 1.0e-30)

    grouped = tl.reshape(normalized, (BLOCK_N, HIGGS_PACKED_BYTES, 2, 2))
    coord0, coord1 = tl.split(grouped)
    lo_x, hi_x = tl.split(coord0)
    lo_y, hi_y = tl.split(coord1)
    lo_idx = _nearest_eden2_idx(lo_x, lo_y)
    hi_idx = _nearest_eden2_idx(hi_x, hi_y)
    packed_byte = (lo_idx | (hi_idx << 4)).to(tl.uint8)

    locs = tl.load(Locs + rows, mask=row_mask, other=0).to(tl.int64)
    byte_offsets = tl.arange(0, HIGGS_PACKED_BYTES)
    row_byte_offsets = locs * packed_stride_0 + head * packed_stride_1
    tl.store(
        Packed + row_byte_offsets[:, None] + byte_offsets[None, :],
        packed_byte,
        mask=row_mask[:, None],
    )
    tl.store(
        PackedScale + (row_byte_offsets // 2) + (HIGGS_PACKED_BYTES // 2),
        scale.to(tl.float32),
        mask=row_mask,
    )


def store_higgs_mha_2bit_triton(
    packed: torch.Tensor,
    locs: torch.Tensor,
    cache: torch.Tensor,
    *,
    block_n: int = 16,
) -> None:
    """Pack BF16 GQA KV rows into EDEN2/FWHT HIGGS slots."""
    _require_supported_tensors(packed, locs, cache)
    grid = (triton.cdiv(cache.shape[0], block_n), cache.shape[1])
    _store_higgs_mha_2bit_kernel[grid](
        packed,
        packed.view(torch.float16),
        locs.contiguous(),
        cache.contiguous(),
        cache.shape[0],
        packed.stride(0),
        packed.stride(1),
        cache.stride(0),
        cache.stride(1),
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=3,
    )
