from functools import lru_cache
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from .tensor_parallel_reduce import allocate_symm_mem_outputs, tp_post_kernel_reduce
from .tp_info import TP1, TPInfo

try:
    import nvtx
except ImportError:

    class _NvtxStub:
        @staticmethod
        def annotate(*_args, **_kwargs):
            def decorator(fn):
                return fn

            return decorator

    nvtx = _NvtxStub()


@lru_cache(maxsize=1)
def num_sms_cached(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


MIN_BLOCK_SIZE_V = 128
LOCAL_INDEX_DTYPE = torch.int32


@nvtx.annotate()
def fused_mm_sample_triton(
    weights: torch.Tensor,  # [V_local, D] (may be a TP shard)
    hidden_states: torch.Tensor,  # [n_hidden_states, D]
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
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
    assert torch.cuda.is_available(), "fused_mm_sample_triton requires CUDA"
    weight_vocab_size, D = weights.shape  # noqa: N806
    V = valid_vocab_size if valid_vocab_size is not None else weight_vocab_size  # noqa: N806
    H, D2 = hidden_states.shape  # noqa: N806
    if D2 != D:
        raise ValueError(
            f"hidden_states second dimension ({D2}) must match weights second dimension ({D})"
        )

    # The kernel uses TMA descriptors which need a runtime allocator. Some
    # autotuner configs (notably the ones picked on B200/sm_100) request global
    # scratch from this allocator; without it Triton raises a RuntimeError at launch.
    set_torch_allocator_for_tma_descriptors_cached()

    NUM_SMS = num_sms_cached(weights.device.index)  # noqa: N806

    max_grid_size_v = triton.cdiv(V, MIN_BLOCK_SIZE_V)
    if tp.size > 1:
        if (
            maxs_workspace is not None
            or maxs_idx_workspace is not None
            or logits_out_workspace is not None
        ):
            raise ValueError("FlashSampling workspaces are only supported for local TP")
        maxs, maxs_idx, symm_mem_hdl, storage_offset_maxs_idx = allocate_symm_mem_outputs(
            num_samples=num_samples,
            max_grid_size_v=max_grid_size_v,
            H=H,
        )
        kernel_maxs = maxs[tp.rank]
        kernel_maxs_idx = maxs_idx[tp.rank]
        symm_mem_buffer_ptrs = symm_mem_hdl.buffer_ptrs_dev
    else:
        maxs_shape = (num_samples, max_grid_size_v, H)
        if maxs_workspace is None:
            maxs = torch.empty(
                maxs_shape,
                dtype=torch.bfloat16,
                device=weights.device,
            )
        else:
            _check_workspace(
                maxs_workspace,
                maxs_shape,
                torch.bfloat16,
                weights.device,
                "maxs_workspace",
            )
            maxs = maxs_workspace
        if maxs_idx_workspace is None:
            maxs_idx = torch.empty_like(maxs, dtype=LOCAL_INDEX_DTYPE)
        else:
            _check_workspace(
                maxs_idx_workspace,
                maxs_shape,
                LOCAL_INDEX_DTYPE,
                weights.device,
                "maxs_idx_workspace",
            )
            maxs_idx = maxs_idx_workspace
        kernel_maxs = maxs
        kernel_maxs_idx = maxs_idx
        storage_offset_maxs_idx = 0
        symm_mem_buffer_ptrs = maxs

    # logits_out is only read when RETURN_LOGITS=True. For the common path
    # (return_logits=False), allocating a (V, H) fp32 buffer per call is wasted
    # HBM (155 MB per decode step at Qwen3-1.7B / H=256) for a buffer the
    # kernel never touches. Pass a 1-element dummy in that case so the kernel
    # still has a valid pointer to receive.
    logits_shape = (V, H) if return_logits else (1,)
    if logits_out_workspace is None:
        logits_out = torch.empty(logits_shape, dtype=torch.float32, device=weights.device)
    else:
        _check_workspace(
            logits_out_workspace,
            logits_shape,
            torch.float32,
            weights.device,
            "logits_out_workspace",
        )
        logits_out = logits_out_workspace

    grid_size = {"v": None}

    def grid(meta):
        grid_size_v = triton.cdiv(V, meta["BLOCK_SIZE_V"])
        grid_size_h = triton.cdiv(H, meta["BLOCK_SIZE_H"])
        grid_size["v"] = grid_size_v
        num_tiles = grid_size_v * grid_size_h
        # Persistent kernel: launch min(NUM_SMS, num_tiles) programs
        return (min(NUM_SMS, num_tiles),)

    fused_mm_sample_triton_kernel[grid](
        weights_ptr=weights,
        hidden_states_ptr=hidden_states,
        max_out_ptr=kernel_maxs,
        max_out_idx_ptr=kernel_maxs_idx,
        symm_mem_buffer_ptrs=symm_mem_buffer_ptrs,
        vocab_size=V,
        hidden_size=D,
        n_hidden_states=H,
        num_samples=num_samples,
        temperature_ptr=temperature,
        seed=seed,
        max_grid_size_v=max_grid_size_v,
        storage_offset_maxs_idx=storage_offset_maxs_idx,
        tp_rank=tp.rank,
        tp_world_size=tp.size,
        logits_out_ptr=logits_out,
        # Gate WS on having enough V tiles so the persistent loop runs more
        # than one iteration even for the largest autotune BLOCK_SIZE_V (=2 *
        # MIN_BLOCK_SIZE_V). On B200/Triton 3.6, NVWSInsertAref crashes when
        # num_tiles=1 (e.g. tiny test vocabs); production V (>=128k) is far
        # above the threshold.
        WARP_SPECIALIZE=supports_warp_specialization_cached() and V > 2 * MIN_BLOCK_SIZE_V,
        NUM_SMS=NUM_SMS,
        GREEDY_SAMPLING=greedy_sampling,
        RETURN_LOGITS=return_logits,
        FAN_OUT_TP=tp.size > 1,
    )

    assert grid_size["v"] is not None

    if tp.size > 1:
        if return_scores:
            raise RuntimeError(
                "return_scores is only supported for local FlashSampling TP reduction."
            )
        samples = tp_post_kernel_reduce(
            local_maxs=maxs,
            local_maxs_idx=maxs_idx,
            symm_mem_hdl=symm_mem_hdl,
            grid_size_v=grid_size["v"],
        )
    else:
        # Local reduction across V-tiles on this rank.
        if vocab_start_index is None:
            vocab_start_index = tp.rank * V
        if return_scores:
            samples, max_values = _local_reduce(
                maxs[:, : grid_size["v"], :],
                maxs_idx[:, : grid_size["v"], :],
                vocab_start_index,
            )
        else:
            samples = _local_reduce_samples_triton(
                maxs[:, : grid_size["v"], :],
                maxs_idx[:, : grid_size["v"], :],
                vocab_start_index,
            )

    if return_logits:
        if return_scores:
            return samples, max_values, logits_out.T
        return samples, logits_out.T  # [n_hidden_states, num_samples], [H, V]
    if return_scores:
        return samples, max_values
    return samples  # [n_hidden_states, num_samples]


def _check_workspace(
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> None:
    if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(
            f"{name} must have shape={shape}, dtype={dtype}, and device={device}; "
            f"got shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _supports_warp_specialization_for_cc(cc_major: int) -> bool:
    # Triton 3.7 FlashSampling warp-specialized persistent kernels fail the
    # PassManager on Blackwell serving and fallback shapes. Keep the optimization
    # on Hopper while SM100 uses the compile-safe non-warp-specialized path.
    return cc_major == 9


@lru_cache(maxsize=1)
def supports_warp_specialization_cached():
    is_cuda = triton.runtime.driver.active.get_current_target().backend == "cuda"
    return is_cuda and _supports_warp_specialization_for_cc(
        torch.cuda.get_device_capability()[0]
    )


@torch.compile(fullgraph=True)
def _local_reduce(
    maxs: torch.Tensor,  # [num_samples, n_tiles, H]
    maxs_idx: torch.Tensor,  # [num_samples, n_tiles, H]
    vocab_start_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce across V-tiles (dim=1) on this rank and adjust to global vocab indices."""
    idxs = maxs.max(dim=1).indices  # [num_samples, H]
    samples = maxs_idx.gather(1, idxs.unsqueeze(1)).squeeze(1)  # [num_samples, H]
    max_values = maxs.gather(1, idxs.unsqueeze(1)).squeeze(1)  # [num_samples, H]
    samples += vocab_start_index
    return samples.T.contiguous(), max_values.T.contiguous()  # [H, num_samples]


@torch.compile(fullgraph=True)
def _local_reduce_samples(
    maxs: torch.Tensor,  # [num_samples, n_tiles, H]
    maxs_idx: torch.Tensor,  # [num_samples, n_tiles, H]
    vocab_start_index: int,
) -> torch.Tensor:
    idxs = maxs.max(dim=1).indices  # [num_samples, H]
    samples = maxs_idx.gather(1, idxs.unsqueeze(1)).squeeze(1)  # [num_samples, H]
    samples += vocab_start_index
    return samples.T.contiguous()  # [H, num_samples]


def _local_reduce_samples_triton(
    maxs: torch.Tensor,  # [num_samples, n_tiles, H]
    maxs_idx: torch.Tensor,  # [num_samples, n_tiles, H]
    vocab_start_index: int,
) -> torch.Tensor:
    if maxs.shape[0] != 1:
        return _local_reduce_samples(maxs, maxs_idx, vocab_start_index)

    _, n_tiles, H = maxs.shape  # noqa: N806
    samples = torch.empty((H, 1), dtype=maxs_idx.dtype, device=maxs.device)
    block_tiles = triton.next_power_of_2(n_tiles)
    _local_reduce_samples_kernel[(H,)](
        maxs,
        maxs_idx,
        samples,
        n_tiles,
        H,
        vocab_start_index,
        BLOCK_TILES=block_tiles,
    )
    return samples


@triton.jit
def _local_reduce_samples_kernel(
    maxs_ptr,
    maxs_idx_ptr,
    samples_ptr,
    n_tiles: tl.constexpr,
    H,  # noqa: N803
    vocab_start_index,
    BLOCK_TILES: tl.constexpr,  # noqa: N803
):
    h = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_TILES)
    mask = offsets < n_tiles
    values = tl.load(maxs_ptr + offsets * H + h, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    _, tile = tl.max(values, axis=0, return_indices=True)
    sample = tl.load(maxs_idx_ptr + tile * H + h) + vocab_start_index
    tl.store(samples_ptr + h, sample)


@triton.jit
def _compute_tile_pid(tile_id, num_pid_in_group, num_pid_v, GROUP_SIZE_V):  # noqa: N803
    """Compute pid_v, pid_h from tile_id using grouped ordering for L2 cache efficiency."""
    group_id = tile_id // num_pid_in_group
    first_pid_v = group_id * GROUP_SIZE_V
    group_size_v = tl.minimum(num_pid_v - first_pid_v, GROUP_SIZE_V)
    pid_v = first_pid_v + (tile_id % group_size_v)
    pid_h = (tile_id % num_pid_in_group) // group_size_v
    return pid_v, pid_h


def metadata_fn(
    grid: tuple,
    metadata: NamedTuple,
    args: dict,
):
    """Set a compact Triton profiling name for this autotuned kernel."""
    grid_x, grid_y, grid_z = unpack_grid(grid)
    num_warps = metadata.num_warps
    num_stages = metadata.num_stages
    cluster_x, cluster_y, cluster_z = unpack_grid((metadata.num_ctas,))
    shared_memory = metadata.shared
    return {
        "name": f"fused_mm_sample_triton_<grid:{grid_x}x{grid_y}x{grid_z}>_<cluster:{cluster_x}x{cluster_y}x{cluster_z}>_<warps:{num_warps}>_<shared:{shared_memory}>_<stages:{num_stages}>",
    }


def unpack_grid(grid):
    if len(grid) == 1:
        return grid[0], 1, 1
    if len(grid) == 2:
        return grid[0], grid[1], 1
    if len(grid) == 3:
        return grid[0], grid[1], grid[2]


def get_autotuning_configs() -> list[triton.Config]:
    cc = torch.cuda.get_device_capability()
    is_dev_machine: bool = cc == (8, 6)  # RTX 3090 config
    if is_dev_machine:
        return [
            triton.Config(
                {"BLOCK_SIZE_V": MIN_BLOCK_SIZE_V, "BLOCK_SIZE_D": 32, "GROUP_SIZE_V": 4},
                num_warps=4,
                num_stages=2,
                # Persistent kernel: grid = NUM_SMS, so only 1 block per SM.
                # No occupancy benefit from limiting registers, so let ptxas
                # use the full register file instead of spilling to local memory.
                maxnreg=255,
            )
        ]
    return [
        triton.Config(
            {"BLOCK_SIZE_V": bsz_v, "BLOCK_SIZE_D": bsz_d, "GROUP_SIZE_V": 4},
            num_warps=num_warps,
            num_stages=num_stages,
            maxnreg=maxnreg,
        )
        for bsz_v in [MIN_BLOCK_SIZE_V, 2 * MIN_BLOCK_SIZE_V]
        for bsz_d in [64, 128]
        for num_warps in [8]
        # Warp specialization adds warps (8+4=384 threads). Keep candidates
        # within the 65536-register file while giving high-H shapes room to
        # avoid local-memory spills.
        for maxnreg in [128, 170]
        for num_stages in [4]
    ]


@triton.autotune(
    configs=get_autotuning_configs(),
    key=["vocab_size", "hidden_size", "BLOCK_SIZE_H", "num_samples", "GREEDY_SAMPLING"],
    cache_results=True,
)
@triton.heuristics(values={"BLOCK_SIZE_H": lambda args: bsz_h(args["n_hidden_states"])})
@triton.jit(launch_metadata=metadata_fn)
def fused_mm_sample_triton_kernel(
    weights_ptr,  # [V, D]
    hidden_states_ptr,  # [n_hidden_states, D]
    max_out_ptr,  # [num_samples, max_grid_size_v, n_hidden_states]
    max_out_idx_ptr,  # [num_samples, max_grid_size_v, n_hidden_states]
    symm_mem_buffer_ptrs,  # [tp_world_size] uint64 base pointers when FAN_OUT_TP=True
    vocab_size,  # V
    hidden_size: tl.constexpr,  # D
    n_hidden_states: int,
    num_samples: tl.constexpr,
    temperature_ptr,  # scalar (0-d tensor)
    seed: int,
    storage_offset_maxs_idx: tl.constexpr,
    tp_rank: tl.constexpr,
    tp_world_size: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_D: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_H: tl.constexpr,  # noqa: N803
    GROUP_SIZE_V: tl.constexpr,  # noqa: N803
    max_grid_size_v: tl.constexpr,
    logits_out_ptr,  # [V, n_hidden_states] float32 (or dummy when RETURN_LOGITS=False)
    WARP_SPECIALIZE: tl.constexpr,  # noqa: N803
    NUM_SMS: tl.constexpr,  # noqa: N803
    GREEDY_SAMPLING: tl.constexpr,  # noqa: N803
    RETURN_LOGITS: tl.constexpr,  # noqa: N803
    FAN_OUT_TP: tl.constexpr,  # noqa: N803
):
    """Persistent kernel for fused matmul + Gumbel-max sampling.

    Each SM processes multiple tiles in a loop, staying persistent on the SM
    rather than exiting after processing a single tile.
    """
    if not GREEDY_SAMPLING:
        temperature = tl.load(temperature_ptr)
    start_pid = tl.program_id(axis=0)
    num_pid_v = tl.cdiv(vocab_size, BLOCK_SIZE_V)
    num_pid_h = tl.cdiv(n_hidden_states, BLOCK_SIZE_H)
    num_tiles = num_pid_v * num_pid_h
    num_pid_in_group = GROUP_SIZE_V * num_pid_h

    # TMA descriptors are used for the matmul LOADS (where the bandwidth wins
    # really matter) but NOT for the small per-tile output stores. The output
    # stores were originally TMA descriptors with shape=[num_samples, V_tiles,
    # H] and block_shape=[1, 1, BLOCK_SIZE_H]. That 3D-with-two-singleton-dims
    # pattern silently no-ops most stores on Blackwell (sm_100): only 2/1187
    # V-tile slots get written, leaving the rest uninitialized. The Triton
    # tutorial 09-persistent-matmul only uses TMA stores in the canonical 2D
    # form, so we sidestep the issue by switching the output stores to plain
    # tl.store with computed offsets.
    w_desc = tl.make_tensor_descriptor(
        weights_ptr,
        shape=[vocab_size, hidden_size],
        strides=[hidden_size, 1],
        block_shape=[BLOCK_SIZE_V, BLOCK_SIZE_D],
    )
    hidden_states_desc = tl.make_tensor_descriptor(
        hidden_states_ptr,
        shape=[n_hidden_states, hidden_size],
        strides=[hidden_size, 1],
        block_shape=[BLOCK_SIZE_H, BLOCK_SIZE_D],
    )
    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue (workaround for Blackwell pipelining bug)
    tile_id_c = start_pid - NUM_SMS

    # Persistent loop: each SM processes multiple tiles
    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE
    ):
        # Compute pid_v, pid_h from tile_id using grouped ordering for L2 cache
        pid_v, pid_h = _compute_tile_pid(tile_id, num_pid_in_group, num_pid_v, GROUP_SIZE_V)

        v_start = pid_v * BLOCK_SIZE_V
        h_start = pid_h * BLOCK_SIZE_H

        offsets_v = v_start + tl.arange(0, BLOCK_SIZE_V)
        mask_v = offsets_v < vocab_size

        logits_blk = tl.zeros((BLOCK_SIZE_V, BLOCK_SIZE_H), dtype=tl.float32)

        # Compute a block of logits logits_blk
        for d_start in range(0, hidden_size, BLOCK_SIZE_D):
            # load weights tile [BLOCK_SIZE_V, BLOCK_SIZE_D]
            w_blk = w_desc.load([v_start, d_start])
            # load hidden_states tile [BLOCK_SIZE_H, BLOCK_SIZE_D]
            hidden_states_blk = hidden_states_desc.load([h_start, d_start])
            logits_blk = tl.dot(w_blk, hidden_states_blk.T, acc=logits_blk)

        # Optionally store raw logits to GMEM before masking and temperature.
        if RETURN_LOGITS:
            offsets_h = h_start + tl.arange(0, BLOCK_SIZE_H)
            logits_ptrs = logits_out_ptr + offsets_v[:, None] * n_hidden_states + offsets_h[None, :]
            logits_mask = mask_v[:, None] & (offsets_h < n_hidden_states)[None, :]
            tl.store(logits_ptrs, logits_blk, mask=logits_mask)

        # Later we will take max over logits + noise, but rows outside the mask
        # should not be considered. Setting them to -inf achieves this.
        logits_blk = tl.where(mask_v[:, None], logits_blk, -float("inf"))

        if not GREEDY_SAMPLING:
            logits_blk = logits_blk / temperature  # [Vblk, n_hidden_states]

        # Epilogue: use tile_id_c to break dependency with prologue
        tile_id_c += NUM_SMS
        pid_v_c, pid_h_c = _compute_tile_pid(tile_id_c, num_pid_in_group, num_pid_v, GROUP_SIZE_V)
        v_start_c = pid_v_c * BLOCK_SIZE_V
        h_start_c = pid_h_c * BLOCK_SIZE_H

        for sample_idx in range(num_samples):
            noise_size: tl.constexpr = BLOCK_SIZE_V * BLOCK_SIZE_H
            noise_offsets = tl.arange(0, noise_size).reshape((BLOCK_SIZE_V, BLOCK_SIZE_H))
            if not GREEDY_SAMPLING:
                gumbel_max, gumbel_max_idx_local = tl.max(
                    logits_blk
                    + _gumbel_noise(
                        seed,
                        pid_v_c + tp_rank * num_pid_v,
                        pid_h_c,
                        sample_idx,
                        noise_offsets,
                    ),
                    axis=0,
                    return_indices=True,
                )
            else:
                gumbel_max, gumbel_max_idx_local = tl.max(logits_blk, axis=0, return_indices=True)

            gumbel_max_idx_global = gumbel_max_idx_local + v_start_c
            if FAN_OUT_TP:
                gumbel_max_idx_global += tp_rank * vocab_size

            # Plain tl.store (no TMA) for the small per-tile outputs. See note
            # above the descriptor block.
            offsets_h_out = h_start_c + tl.arange(0, BLOCK_SIZE_H)
            mask_h_out = offsets_h_out < n_hidden_states
            base_offset = (
                sample_idx * max_grid_size_v * n_hidden_states
                + pid_v_c * n_hidden_states
                + offsets_h_out
            )
            if FAN_OUT_TP:
                buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))
                source_rank_base_offset = (
                    tp_rank * num_samples * max_grid_size_v * n_hidden_states + base_offset
                )
                for peer_rank in tl.static_range(0, tp_world_size):
                    peer_base = tl.load(buffer_ptrs + peer_rank)
                    peer_maxs_ptr = peer_base.to(tl.pointer_type(tl.bfloat16))
                    peer_maxs_idx_ptr = peer_base.to(tl.pointer_type(tl.int64))
                    tl.store(
                        peer_maxs_ptr + source_rank_base_offset,
                        gumbel_max,
                        mask=mask_h_out,
                    )
                    tl.store(
                        peer_maxs_idx_ptr + storage_offset_maxs_idx + source_rank_base_offset,
                        gumbel_max_idx_global,
                        mask=mask_h_out,
                    )
            else:
                tl.store(max_out_ptr + base_offset, gumbel_max, mask=mask_h_out)
                tl.store(max_out_idx_ptr + base_offset, gumbel_max_idx_global, mask=mask_h_out)


@triton.jit
def _gumbel_noise(seed, pid_v, pid_h, sample_idx, noise_offsets):
    # Note: Each tile (v, h) and sample needs a different seed,
    # otherwise they all create the same noise, leading to sampling artifacts.
    return -tl.log(
        -tl.log(
            tl.rand(
                seed + pid_v * 100 + pid_h * 1_000 + sample_idx * 10_000,
                noise_offsets,
            )
        )
    )


@lru_cache(maxsize=1)
def set_torch_allocator_for_tma_descriptors_cached():
    """From https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html"""
    # TMA descriptors require a global memory allocation
    triton.set_allocator(alloc_on_cuda)


def alloc_on_cuda(size: int, alignment: int, stream: int | None):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def bsz_h(H: int) -> int:  # noqa: N803
    if H <= 16:
        return 16
    elif H <= 32:
        return 32
    return 64
