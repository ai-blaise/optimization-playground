from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.distributed import (
    GroupCoordinator,
    get_attn_context_model_parallel_rank,
    get_attn_context_model_parallel_world_size,
    get_attn_cp_group,
    get_attn_tensor_model_parallel_rank,
    get_attn_tensor_model_parallel_world_size,
    get_attn_tp_group,
)
from sglang.srt.distributed import get_moe_dp_group as _get_moe_dp_group
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.utils import get_bool_env_var, is_hip

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_ATTN_DP_RANK: Optional[int] = None
_ATTN_DP_SIZE: Optional[int] = None
_LOCAL_ATTN_DP_SIZE: Optional[int] = None
_LOCAL_ATTN_DP_RANK: Optional[int] = None
_ENABLE_DP_ATTENTION_FLAG: bool = False

_is_hip = is_hip()
_USE_ROCM700A_WA = _is_hip and get_bool_env_var("SGLANG_USE_ROCM700A")


class DpPaddingMode(IntEnum):

    # Padding tokens to max length and then gather tokens using `all_gather_into_tensor`
    MAX_LEN = auto()
    # Padding tokens to sum length and then gather tokens using `all_reduce`
    SUM_LEN = auto()

    def is_max_len(self):
        return self == DpPaddingMode.MAX_LEN

    def is_sum_len(self):
        return self == DpPaddingMode.SUM_LEN

    @classmethod
    def get_dp_padding_mode(
        cls, is_extend_in_batch, global_num_tokens: List[int]
    ) -> DpPaddingMode:
        dp_size = get_attention_dp_size()

        # When is_extend_in_batch and dp_size > 1, use SUM_LEN to avoid padding
        # overhead from uneven token distribution.
        # For dp_size=1, max_len equals sum_len, so prefer MAX_LEN mode
        # to enable symmetric memory optimization (needed for DSA CP, etc.).
        if is_extend_in_batch and dp_size > 1:
            return DpPaddingMode.SUM_LEN

        # we choose the mode that minimizes the communication cost
        # prefer MAX_LEN when communication cost is equal to enable symmetric memory
        max_len = max(global_num_tokens)
        sum_len = sum(global_num_tokens)
        if sum_len * 2 >= max_len * dp_size:
            return cls.MAX_LEN
        else:
            return cls.SUM_LEN

    @classmethod
    def get_default_mode_in_cuda_graph(cls) -> DpPaddingMode:
        # TODO(kkhuang-amd): noqa, temporary work-around for rocm 7.0.0 alpha
        # it can be safely removed later, once RCCL fixed
        if _USE_ROCM700A_WA:
            return cls.SUM_LEN
        else:
            return cls.MAX_LEN


class _DpGatheredBufferWrapper:

    _hidden_size: int
    _dtype: torch.dtype
    _device: torch.device
    _global_dp_buffer_len: int
    _local_dp_buffer_len: int
    _dp_max_padding: bool
    _global_num_tokens: Optional[List[int]]
    _is_extend_in_batch: bool

    @classmethod
    def set_metadata(cls, hidden_size: int, dtype: torch.dtype, device: torch.device):
        cls._hidden_size = hidden_size
        cls._dtype = dtype
        cls._device = device

    @classmethod
    def set_dp_buffer_len(
        cls,
        global_dp_buffer_len: int,
        local_dp_buffer_len: int,
        dp_max_padding: bool,
        global_num_tokens: Optional[List[int]] = None,
    ):
        cls._global_dp_buffer_len = global_dp_buffer_len
        cls._local_dp_buffer_len = local_dp_buffer_len
        cls._dp_max_padding = dp_max_padding
        cls._global_num_tokens = global_num_tokens

    @classmethod
    def get_global_dp_buffer(
        cls,
        group: GroupCoordinator,
        size_ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # When the caller passes a `size_ref` tensor (the local per-rank
        # hidden states), derive the global buffer length as
        # `size_ref.shape[0] * world_size`. Inside torch.compile this keeps
        # the buffer dim symbolic and the captured graph scales correctly
        # across piecewise CUDA graph sizes. Outside compile it returns the
        # same int as the legacy path.
        if size_ref is not None:
            length = size_ref.shape[0] * group.world_size
        else:
            length = cls._global_dp_buffer_len
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (length, cls._hidden_size),
                dtype=cls._dtype,
                device=cls._device,
            )
        return buffer

    @classmethod
    def get_local_dp_buffer(
        cls,
        group: GroupCoordinator,
        size_ref: Optional[torch.Tensor] = None,
        size_ref_is_global: bool = False,
    ) -> torch.Tensor:
        # When `size_ref_is_global=True`, derive the local length from the
        # passed *global* tensor's first dim divided by world_size — this is
        # what the post-attention reduce_scatter needs so the output buffer
        # scales symbolically with the captured graph instead of being baked
        # to the warmup-time class var value.
        if size_ref is not None:
            if size_ref_is_global:
                length = size_ref.shape[0] // group.world_size
            else:
                length = size_ref.shape[0]
        else:
            length = cls._local_dp_buffer_len
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (length, cls._hidden_size),
                dtype=cls._dtype,
                device=cls._device,
            )
        return buffer

    @classmethod
    def get_global_dp_buffer_len(cls) -> int:
        return cls._global_dp_buffer_len

    @classmethod
    def get_local_dp_buffer_len(cls) -> int:
        return cls._local_dp_buffer_len

    @classmethod
    def get_dp_global_num_tokens(cls) -> List[int]:
        return cls._global_num_tokens

    @classmethod
    def get_dp_hidden_size(cls) -> int:
        return cls._hidden_size

    @classmethod
    def get_dp_dtype(cls) -> torch.dtype:
        return cls._dtype

    @classmethod
    def get_dp_device(cls) -> torch.device:
        return cls._device

    @classmethod
    def set_is_extend_in_batch(cls, is_extend_in_batch: bool):
        cls._is_extend_in_batch = is_extend_in_batch

    @classmethod
    def get_is_extend_in_batch(cls) -> bool:
        return cls._is_extend_in_batch

    @classmethod
    def is_dp_max_padding(cls) -> bool:
        return cls._dp_max_padding


def set_dp_buffer_len(
    global_dp_buffer_len: int,
    local_dp_buffer_len: int,
    dp_max_padding: bool,
    global_num_tokens: Optional[List[int]] = None,
):
    _DpGatheredBufferWrapper.set_dp_buffer_len(
        global_dp_buffer_len, local_dp_buffer_len, dp_max_padding, global_num_tokens
    )


def get_global_dp_buffer(
    group: GroupCoordinator,
    size_ref: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_global_dp_buffer(
        group=group, size_ref=size_ref
    )


def get_local_dp_buffer(
    group: GroupCoordinator,
    size_ref: Optional[torch.Tensor] = None,
    size_ref_is_global: bool = False,
) -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_local_dp_buffer(
        group=group,
        size_ref=size_ref,
        size_ref_is_global=size_ref_is_global,
    )


def get_global_dp_buffer_len() -> int:
    return _DpGatheredBufferWrapper.get_global_dp_buffer_len()


def get_local_dp_buffer_len() -> int:
    return _DpGatheredBufferWrapper.get_local_dp_buffer_len()


def get_dp_global_num_tokens() -> List[int]:
    return _DpGatheredBufferWrapper.get_dp_global_num_tokens()


def get_dp_hidden_size() -> int:
    return _DpGatheredBufferWrapper.get_dp_hidden_size()


def get_dp_dtype() -> torch.dtype:
    return _DpGatheredBufferWrapper.get_dp_dtype()


def get_dp_device() -> torch.device:
    return _DpGatheredBufferWrapper.get_dp_device()


def set_is_extend_in_batch(is_extend_in_batch: bool):
    _DpGatheredBufferWrapper.set_is_extend_in_batch(is_extend_in_batch)


def get_is_extend_in_batch() -> bool:
    return _DpGatheredBufferWrapper.get_is_extend_in_batch()


def is_dp_max_padding() -> bool:
    return _DpGatheredBufferWrapper.is_dp_max_padding()


def compute_dp_attention_world_info(
    enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size: int = 1
):
    attn_dp_size = dp_size if enable_dp_attention else 1
    attn_tp_size = tp_size // attn_dp_size // attn_cp_size
    attn_tp_rank = tp_rank % attn_tp_size

    if not enable_dp_attention:
        attn_dp_rank = 0
    else:
        # Rank layout is (dp, cp, tp) where tp is the fastest-changing dim:
        # tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size + attn_tp_rank
        attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)

    return attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size


def compute_dp_attention_local_info(
    enable_dp_attention, tp_rank, tp_size, dp_size, moe_dense_tp_size
):
    if not enable_dp_attention:
        return tp_rank, tp_size, 0

    local_tp_size = moe_dense_tp_size if moe_dense_tp_size else tp_size
    local_tp_rank = tp_rank % local_tp_size
    local_dp_size = max(1, dp_size // (tp_size // local_tp_size))

    local_attn_tp_size = local_tp_size // local_dp_size
    local_attn_dp_rank = local_tp_rank // local_attn_tp_size
    local_attn_tp_rank = local_tp_rank % local_attn_tp_size

    return local_attn_tp_rank, local_attn_tp_size, local_attn_dp_rank


def initialize_dp_attention(
    server_args: ServerArgs,
    model_config: ModelConfig,
):
    global _ATTN_DP_RANK, _ATTN_DP_SIZE
    global _LOCAL_ATTN_DP_SIZE, _LOCAL_ATTN_DP_RANK, _ENABLE_DP_ATTENTION_FLAG
    enable_dp_attention = server_args.enable_dp_attention
    dp_size = server_args.dp_size
    moe_dense_tp_size = server_args.moe_dense_tp_size
    attn_cp_size = server_args.attn_cp_size

    _ENABLE_DP_ATTENTION_FLAG = enable_dp_attention

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    _, _, _ATTN_DP_RANK, _ = compute_dp_attention_world_info(
        enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size
    )
    _, _, _LOCAL_ATTN_DP_RANK = compute_dp_attention_local_info(
        enable_dp_attention, tp_rank, tp_size, dp_size, moe_dense_tp_size
    )

    if enable_dp_attention:
        _ATTN_DP_SIZE = dp_size
        if moe_dense_tp_size is None:
            _LOCAL_ATTN_DP_SIZE = _ATTN_DP_SIZE
        else:
            _LOCAL_ATTN_DP_SIZE = max(1, dp_size // (tp_size // moe_dense_tp_size))
    else:
        _ATTN_DP_SIZE = 1
        _LOCAL_ATTN_DP_SIZE = 1

    _DpGatheredBufferWrapper.set_metadata(
        hidden_size=model_config.hidden_size,
        dtype=model_config.dtype,
        device=torch.device(server_args.device),
    )


def is_dp_attention_enabled() -> bool:
    return _ENABLE_DP_ATTENTION_FLAG


def is_allocation_symmetric() -> bool:
    return not is_dp_attention_enabled() or is_dp_max_padding()


def get_attention_tp_group() -> GroupCoordinator:
    return get_attn_tp_group()


def get_attention_tp_rank() -> int:
    return get_attn_tensor_model_parallel_rank()


def get_attention_tp_size() -> int:
    return get_attn_tensor_model_parallel_world_size()


def get_attention_cp_group() -> GroupCoordinator:
    return get_attn_cp_group()


def get_attention_cp_rank() -> int:
    return get_attn_context_model_parallel_rank()


def get_attention_cp_size() -> int:
    return get_attn_context_model_parallel_world_size()


def get_attention_dp_rank() -> int:
    assert _ATTN_DP_RANK is not None, "dp attention not initialized!"
    return _ATTN_DP_RANK


def get_attention_dp_size() -> int:
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"
    return _ATTN_DP_SIZE


def get_local_attention_dp_rank() -> int:
    assert _LOCAL_ATTN_DP_RANK is not None, "dp attention not initialized!"
    return _LOCAL_ATTN_DP_RANK


def get_local_attention_dp_size() -> int:
    assert _LOCAL_ATTN_DP_SIZE is not None, "dp attention not initialized!"
    return _LOCAL_ATTN_DP_SIZE


@contextmanager
def disable_dp_size():
    """Disable DP attention metadata inside speculative draft workers.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.
    """
    global _ATTN_DP_RANK, _ATTN_DP_SIZE
    global _LOCAL_ATTN_DP_SIZE, _LOCAL_ATTN_DP_RANK, _ENABLE_DP_ATTENTION_FLAG
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"
    assert _ATTN_DP_RANK is not None, "dp attention not initialized!"
    assert _LOCAL_ATTN_DP_SIZE is not None, "dp attention not initialized!"
    assert _LOCAL_ATTN_DP_RANK is not None, "dp attention not initialized!"

    old_dp_rank = _ATTN_DP_RANK
    old_dp_size = _ATTN_DP_SIZE
    old_local_dp_rank = _LOCAL_ATTN_DP_RANK
    old_local_dp_size = _LOCAL_ATTN_DP_SIZE
    old_enable_dp_attention_flag = _ENABLE_DP_ATTENTION_FLAG
    _ATTN_DP_RANK = 0
    _ATTN_DP_SIZE = 1
    _LOCAL_ATTN_DP_RANK = 0
    _LOCAL_ATTN_DP_SIZE = 1
    _ENABLE_DP_ATTENTION_FLAG = False
    try:
        yield
    finally:
        _ATTN_DP_RANK = old_dp_rank
        _ATTN_DP_SIZE = old_dp_size
        _LOCAL_ATTN_DP_RANK = old_local_dp_rank
        _LOCAL_ATTN_DP_SIZE = old_local_dp_size
        _ENABLE_DP_ATTENTION_FLAG = old_enable_dp_attention_flag


def get_dp_local_info(forward_batch: ForwardBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    # `get_dp_local_info` is only called in global DP gather and scatter. We use global DP rank here.
    dp_rank = get_attention_dp_rank()

    if forward_batch.dp_local_start_pos is None:
        if forward_batch.dp_local_start_pos_token_gpu is not None:
            # B200 SM_100 workaround (#14 fix followup 9): read the precomputed
            # local-DP-rank prefix value (materialized at FB construction via
            # cudaMemcpyHostToDevice) instead of running cumsum + .to() inside
            # capture. dp_local_start_pos_token_gpu is the prefix over
            # global_num_tokens_gpu (token-count path used by gather/scatter).
            # Single-element select on the already-allocated
            # global_num_tokens_gpu is a stride view, no kernel launch.
            forward_batch.dp_local_start_pos = (
                forward_batch.dp_local_start_pos_token_gpu
            )
            forward_batch.dp_local_num_tokens = (
                forward_batch.global_num_tokens_gpu[dp_rank]
            )
        else:
            # Fallback for paths that don't precompute the prefix (e.g.
            # EAGLE draft worker cuda graph capture). This branch keeps the
            # legacy float32-cast cumsum behavior — fine for non-B200 or
            # non-PCG flows where the launch-config bug doesn't surface.
            _src = forward_batch.global_num_tokens_gpu
            cumtokens = torch.cumsum(_src.to(torch.float32), dim=0).to(_src.dtype)
            if dp_rank == 0:
                local_start_pos = torch.zeros_like(cumtokens[0])
            else:
                local_start_pos = cumtokens[dp_rank - 1]
            forward_batch.dp_local_start_pos = local_start_pos
            forward_batch.dp_local_num_tokens = (
                forward_batch.global_num_tokens_gpu[dp_rank]
            )

    return forward_batch.dp_local_start_pos, forward_batch.dp_local_num_tokens


def get_dp_local_slice_cpu(
    forward_batch: ForwardBatch,
    can_run_graph: bool,
    cuda_graph_batch: Optional[int],
) -> Tuple[int, int]:
    # CPU (start, length) slice for DP-local data in a rank-padded buffer.
    # Returns Python ints (no D2H sync) and handles the cuda-graph-padded layout.
    global_num_tokens = forward_batch.global_num_tokens_cpu
    dp_rank = get_attention_dp_rank()
    local_num_tokens = global_num_tokens[dp_rank]
    if can_run_graph:
        local_start_pos = dp_rank * cuda_graph_batch
    else:
        local_start_pos = sum(global_num_tokens[:dp_rank])
    return local_start_pos, local_num_tokens


@triton.jit
def memcpy_triton_kernel(
    dst_ptr,
    src_ptr,
    offset_ptr,
    sz_ptr,
    offset_src: tl.constexpr,
    chunk_size,  # multiplied for offset and sz
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size

    start_index = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = start_index + offs < sz

    if offset_src:
        data = tl.load(src_ptr + offset + start_index + offs, mask=mask)
        tl.store(dst_ptr + start_index + offs, data, mask=mask)
    else:
        data = tl.load(src_ptr + start_index + offs, mask=mask)
        tl.store(dst_ptr + offset + start_index + offs, data, mask=mask)


def prod(x):
    return functools.reduce(lambda a, b: a * b, x, 1)


def memcpy_triton(dst, src, dim, offset, sz, offset_src):
    max_size = min(src.numel(), dst.numel())
    assert dim == 0, "dim != 0 unsupported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same shape"
    chunk_size = prod(src.shape[1:])
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(max_size, BLOCK_SIZE),)

    memcpy_triton_kernel[grid](dst, src, offset, sz, offset_src, chunk_size, BLOCK_SIZE)


@triton.jit
def fill_zero_triton_kernel(
    dst_ptr,
    n_elem,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    start_index = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = start_index + offs < n_elem
    tl.store(dst_ptr + start_index + offs, 0, mask=mask)


def fill_zero_triton(dst: torch.Tensor):
    # B200 SM_100 workaround (#14 fix followup 12): in-place zero-fill via a
    # triton kernel, used in place of `Tensor.fill_(0)`, `Tensor.mul_(0)`, and
    # `Tensor.copy_(zero_buffer)` in the dp_attention gather/scatter paths.
    # All three of those PyTorch elementwise APIs dispatch through a launch
    # path that hits `cudaErrorInvalidConfiguration` on small leading-dim
    # tensors (e.g. (1, 7168) BF16) when replayed from a PCG-captured graph
    # on B200 (SM_100). Triton has its own grid/block launch path that does
    # not hit this bug (`memcpy_triton` is confirmed safe at the same shapes).
    n_elem = dst.numel()
    if n_elem == 0:
        return
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n_elem, BLOCK_SIZE),)
    fill_zero_triton_kernel[grid](dst, n_elem, BLOCK_SIZE)


def _dp_gather_via_all_reduce(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    # B200 SM_100 workaround: Tensor.fill_(0) on numel==0 tensors hits
    # cudaErrorInvalidConfiguration (same kernel-launch-config family as the
    # cumsum bug). When all DP ranks have 0 tokens (forward_idle ladder),
    # global_tokens has shape (0, hidden_size); skip fill + all_reduce.
    if global_tokens.numel() == 0:
        return
    # B200 SM_100 workaround (#14 fix followup 12): `fill_(0)` (original),
    # `mul_(0)` (followup 8), and `copy_(zero_buffer)` (followup 10) all
    # hit `cudaErrorInvalidConfiguration` on small leading dims (e.g.
    # (1, 7168) BF16) when replayed from a PCG-captured graph on B200
    # SM_100. They share the same PyTorch elementwise launch dispatcher;
    # same-dtype same-device same-shape `copy_` does *not* fall through
    # to `cudaMemcpyAsync` as followup 10 assumed. Use the triton
    # zero-fill kernel `fill_zero_triton` instead, which uses triton's
    # own launch path (the `memcpy_triton` data-copy a few lines below
    # succeeds at the same shape via the same triton dispatch, confirming
    # this path is known-good on B200 SM_100).
    fill_zero_triton(global_tokens)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()

    if local_tokens.shape[0] > 0 and (is_partial or get_attention_tp_rank() == 0):
        if not torch.compiler.is_compiling():
            # Dynamo cannot trace `is not` on UntypedStorage; keep the alias
            # safety check in eager mode only.
            assert (
                local_tokens.untyped_storage() is not global_tokens.untyped_storage()
            ), "aliasing between global_tokens and local_tokens not allowed"

        memcpy_triton(
            global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False
        )

    # Input IDs are in int 32. We should use inplace_all_reduce for local case because of custom all reduce.
    NUM_GPUS_PER_NODE = 8
    if (
        not local_tokens.dtype.is_floating_point
        and get_tensor_model_parallel_world_size() <= NUM_GPUS_PER_NODE
    ):
        from sglang.srt.distributed.parallel_state import inplace_all_reduce

        inplace_all_reduce(global_tokens, group_name=get_tp_group().unique_name)

    else:
        global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)


def _dp_gather_via_all_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    # B200 SM_100 workaround for empty-tensor fill_: see _dp_gather_via_all_reduce.
    if global_tokens.numel() == 0:
        return
    if get_attention_tp_size() == 1:
        get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
        return

    if not is_partial:
        if get_attention_tp_rank() != 0 and local_tokens.numel() > 0:
            # B200 SM_100 workaround (#14 fix followup 12): replace
            # `local_tokens.copy_(zero_buffer)` (followup 10) with the
            # triton zero-fill kernel. See `_dp_gather_via_all_reduce`
            # for the full rationale on why the triton launch path
            # bypasses the small-leading-dim launch-config bug.
            fill_zero_triton(local_tokens)
    scattered_local_tokens = local_tokens.tensor_split(get_attention_tp_size())[
        get_attention_tp_rank()
    ]
    get_attention_tp_group().reduce_scatter_tensor(scattered_local_tokens, local_tokens)
    get_tp_group().all_gather_into_tensor(global_tokens, scattered_local_tokens)


def _dp_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    if forward_batch.dp_padding_mode.is_max_len():
        _dp_gather_via_all_gather(
            global_tokens, local_tokens, forward_batch, is_partial
        )
    else:
        _dp_gather_via_all_reduce(
            global_tokens, local_tokens, forward_batch, is_partial
        )


def dp_gather_partial(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=True)


def dp_gather_replicate(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=False)


def dp_gather_partial_fp4(
    values_global: torch.Tensor,
    values_local: torch.Tensor,
    scales_global: torch.Tensor,
    scales_local: torch.Tensor,
    forward_batch: ForwardBatch,
) -> None:
    """Parallel DP allgather of (FP4 packed bytes, UE8M0 scale bytes).

    Iter5 #15 NVFP4 MoE companion to ``dp_gather_partial``: unlocks the
    iter4 SECONDARY wire (communicator.py L1024) by making the local-rows
    FP4 stash from ``cvt_fused_rmsnorm_to_fp4_and_bf16_linear`` survive
    the DP-allgather. After this call the gathered (fp4, sf) attached to
    the post-gather BF16 hidden_states matches the layout the NVFP4 MoE
    expert ``apply_weights`` consumer expects:
      - ``values_global``: ``[M_global, hidden // 2]`` uint8
      - ``scales_global``: ``[M_global, hidden // 16]`` fp8_e4m3 (caller
        must pass an int32-typed buffer of shape ``[M_global,
        (hidden // 16) // 4]`` for the underlying NCCL collective; the
        caller re-views as fp8_e4m3 after).

    Both allgathers fire on the same TP group as ``dp_gather_partial``,
    fused under ``ncclGroupStart`` / ``ncclGroupEnd`` so they share a
    single launch overhead in the captured CUDA graph (per-launch cost
    matters for the iter5 deploy-mode break-even — bench delta at
    production decode m_local=16 / m_global=128 swings from -10us
    REGRESSION (serial) to +0.7us SAVED (grouped)).

    NCCL is dtype-agnostic at the bytes level — uint8 + int32 work
    natively.

    Supported deploy config (DSv3.2-REAP DP=TP=8, attn_tp_size=1,
    max_len padding under CUDA-graph decode). Other configs raise.
    """
    # Local-rows shape sanity: leading dim must match the BF16 allgather.
    assert values_local.shape[0] == scales_local.shape[0]
    assert values_global.shape[0] == scales_global.shape[0]
    assert values_global.shape[0] == values_local.shape[0] * get_tp_group().world_size
    assert values_local.is_contiguous() and values_global.is_contiguous()
    assert scales_local.is_contiguous() and scales_global.is_contiguous()

    if get_attention_tp_size() != 1:
        raise NotImplementedError(
            "dp_gather_partial_fp4 requires attn_tp_size == 1 (deploy "
            "config). attn_tp_size > 1 would need a reduce_scatter on "
            "FP4 bytes which is not well-defined."
        )
    if not forward_batch.dp_padding_mode.is_max_len():
        # SUM_LEN mode would need the rank-offset memcpy + allreduce-or-
        # allgather pattern. allreduce sums bytes which corrupts FP4; a
        # per-rank-padded allgather would have to know the global pad
        # length. Defer to iter6 when prefill needs it; current deploy
        # decode path uses MAX_LEN.
        raise NotImplementedError(
            "dp_gather_partial_fp4 requires max_len dp_padding_mode "
            "(decode); sum_len (prefill) deferred to iter6."
        )

    tp = get_tp_group()
    # Prefer the grouped fast path: when the tp_group has an active
    # pynccl_comm (the deploy default outside torchcomms_ncclx), wrap
    # both allgathers in a ncclGroupStart/End pair so the captured
    # graph fires them under a single launch. Saves ~15-20us per layer
    # at production m=128 vs serial allgathers (graph-captured pynccl
    # bench, NCCL_NVLS_ENABLE=0, dp=2 on B200).
    pynccl_comm = getattr(tp, "pynccl_comm", None)
    torchcomms_ncclx_comm = getattr(tp, "torchcomms_ncclx_comm", None)
    can_group = (
        pynccl_comm is not None
        and not pynccl_comm.disabled
        and (torchcomms_ncclx_comm is None or torchcomms_ncclx_comm.disabled)
    )
    if can_group:
        with pynccl_comm.change_state(enable=True):
            pynccl_comm.group_start()
            pynccl_comm.all_gather(values_global, values_local)
            pynccl_comm.all_gather(scales_global, scales_local)
            pynccl_comm.group_end()
    else:
        # Fall back to the (slower, double-launch) torch.distributed
        # path. torchcomms_ncclx path doesn't expose ncclGroupStart;
        # in that case the iter5 wire is honest-negative and the
        # iter4 SECONDARY env flag should be off for that deploy.
        tp.all_gather_into_tensor(values_global, values_local)
        tp.all_gather_into_tensor(scales_global, scales_local)


def dp_scatter(
    local_tokens: torch.Tensor,  # output
    global_tokens: torch.Tensor,  # input
    forward_batch: ForwardBatch,
):
    # local_num_tokens is not necessarily the same as local_tokens.shape[0],
    # since local_tokens may be padded for cuda graph
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    # B200 SM_100 workaround (#14 fix followup 12): replace
    # `local_tokens.copy_(zero_buffer)` (followup 10) with the triton
    # zero-fill kernel. See `_dp_gather_via_all_reduce` for the full
    # rationale on why the triton launch path bypasses the small-
    # leading-dim launch-config bug.
    if local_tokens.numel() > 0:
        fill_zero_triton(local_tokens)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0:
        if not torch.compiler.is_compiling():
            # Same as above; gated on torch.compile state.
            assert (
                local_tokens.untyped_storage() is not global_tokens.untyped_storage()
            ), "aliasing between local_tokens and global_tokens not allowed"

        memcpy_triton(
            local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True
        )


def dp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    if get_tensor_model_parallel_world_size() == get_attention_dp_size():
        get_tp_group().reduce_scatter_tensor(output, input)
    else:
        scattered_local_tokens = input.tensor_split(
            get_tensor_model_parallel_world_size()
        )[get_tensor_model_parallel_rank()]
        get_tp_group().reduce_scatter_tensor(scattered_local_tokens, input)
        get_attention_tp_group().all_gather_into_tensor(output, scattered_local_tokens)


def attn_tp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attention_tp_group().reduce_scatter_tensor(output, input)


def attn_cp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attention_cp_group().reduce_scatter_tensor(output, input)


def attn_tp_all_reduce(input: torch.Tensor):
    return get_attention_tp_group().all_reduce(input)


def attn_tp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attention_tp_group().all_gather_into_tensor(output, input)


def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attention_cp_group().all_gather_into_tensor(output, input)


def get_moe_cp_group() -> GroupCoordinator:
    """Returns the MOE_DP group, which includes CP partners when attn_cp_size > moe_dp_size."""
    return _get_moe_dp_group()


def get_moe_cp_rank() -> int:
    return _get_moe_dp_group().rank_in_group


def get_moe_cp_size() -> int:
    return _get_moe_dp_group().world_size


def is_enable_moe_cp_allgather() -> bool:
    """True when moe_dp_size < attn_cp_size, requiring allgather across CP ranks before MoE."""
    from sglang.srt.server_args import get_global_server_args

    sa = get_global_server_args()
    return sa.attn_cp_size > sa.moe_dp_size


def moe_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return _get_moe_dp_group().all_gather_into_tensor(output, input)


def attn_tp_all_gather(output_list: List[torch.Tensor], input: torch.Tensor):
    return get_attention_tp_group().all_gather(input, output_tensor_list=output_list)
