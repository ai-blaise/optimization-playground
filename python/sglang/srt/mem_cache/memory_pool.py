"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

import abc
import dataclasses
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.jit_kernel.kvcache import can_use_store_cache, store_cache
from sglang.jit_kernel.turboquant_dense_kv import (
    dequantize_page_table_selected_2p5,
    dequantize_page_table_selected_2p5_fp8,
    dequantize_page_table_selected_2p5_fp8_reuse,
    dequantize_selected_2p5,
    dequantize_selected_4bit,
    store_2p5,
)
from sglang.jit_kernel.turboquant_dense_mla_decode import (
    turboquant_dense_mla_decode_2p5_split_rotated,
    turboquant_dense_mla_rotate_query,
)
from sglang.jit_kernel.higgs_dense_2bit import (
    dequantize_higgs_dense_2bit,
    dequantize_higgs_dense_2bit_page_table,
    dequantize_higgs_dense_2bit_page_table_fp8,
    store_higgs_dense_2bit,
)
from sglang.jit_kernel.higgs_dense_2bit_mla_decode import (
    higgs_dense_2bit_mla_decode_saw_scalar2_split,
    higgs_dense_2bit_mla_decode_split,
    higgs_dense_2bit_mla_rotate_query,
)
from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import index_buf_accessor
from sglang.srt.layers.attention.dsa.indexer_quantization import (
    INDEXER_FP8_QUANT_METHOD,
    get_dsa_indexer_cache_layout,
)
from sglang.srt.layers.attention.dsa.quant_k_cache import (
    quantize_k_cache,
    quantize_k_cache_separate,
)
from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
from sglang.srt.layers.quantization.turboquant_dense_kv import (
    TurboQuantDenseKVCodec,
    TurboQuantDenseKVConfig,
)
from sglang.srt.layers.quantization.higgs_dense_2bit_kv import (
    HiggsDense2BitCodec,
    HiggsDense2BitConfig,
    get_higgs_dense_2bit_b200_candidate,
    select_higgs_mla_decode_num_splits,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    maybe_init_custom_mem_pool,
    set_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton_fp8_quant,
    set_mla_kv_scale_buffer_triton,
)
from sglang.srt.platforms import current_platform
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_cpu_has_amx_support = cpu_has_amx_support()
_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()


def get_tensor_size_bytes(t: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(t, list):
        return sum(get_tensor_size_bytes(x) for x in t)
    return np.prod(t.shape) * t.dtype.itemsize


def _set_kv_buffer_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    row_dim: int,  # head_num * head_dim
    store_dtype: torch.dtype,
    device_module: Any,
    alt_stream: Optional[torch.cuda.Stream] = None,
    same_kv_dim: bool = True,
) -> None:
    row_bytes = row_dim * store_dtype.itemsize
    if (_is_cuda or _is_hip) and same_kv_dim and can_use_store_cache(row_bytes):
        return store_cache(
            k.view(-1, row_dim),
            v.view(-1, row_dim),
            k_cache.view(-1, row_dim),
            v_cache.view(-1, row_dim),
            indices,
            row_bytes=row_bytes,
        )

    if _is_cpu and _cpu_has_amx_support:
        return torch.ops.sgl_kernel.store_cache_cpu(
            k,
            v,
            k_cache,
            v_cache,
            indices,
            row_dim,
        )

    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    if get_is_capture_mode() and alt_stream is not None:
        current_stream = device_module.current_stream()
        alt_stream.wait_stream(current_stream)
        k_cache[indices] = k
        with device_module.stream(alt_stream):
            v_cache[indices] = v
        current_stream.wait_stream(alt_stream)
    else:  # fallback to naive implementation
        k_cache[indices] = k
        v_cache[indices] = v


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        # +1 padding row at index 0: cuda-graph padded batches default
        # req_pool_indices to 0, so dummy reads/writes land here harmlessly.
        self._alloc_size = size + 1
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len), dtype=torch.int32, device=device
            )
        self.free_slots = list(range(1, self._alloc_size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def copy_block_table(
        self,
        src_req_pool_idx: int,
        dst_req_pool_idx: int,
        seq_len: int,
        token_to_kv_pool_allocator=None,
    ):
        if seq_len <= 0:
            return
        indices = self.req_to_token[src_req_pool_idx, :seq_len].to(
            dtype=torch.int64, copy=True
        )
        self.write(
            (dst_req_pool_idx, slice(0, seq_len)),
            indices.to(dtype=torch.int32),
        )
        if token_to_kv_pool_allocator is not None:
            token_to_kv_pool_allocator.inc_ref(indices)

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        # NOTE: this check is relaxed temporarily
        # https://github.com/sgl-project/sglang/pull/20476
        # if not any(r.is_dllm() for r in reqs):
        #     assert (
        #         sum(1 for i in reusing if reqs[i].inflight_middle_chunks > 0) <= 1
        #     ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
            for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))


class MambaPool:
    @dataclass(frozen=True, kw_only=True)
    class State:
        conv: List[torch.Tensor]
        temporal: torch.Tensor

        def at_layer_idx(self, layer: int):
            kwargs = {}
            # Use fields instead of vars to avoid torch.compile graph break
            for f in fields(self):
                name = f.name
                v = getattr(self, name)
                if name in ("conv", "intermediate_conv_window"):
                    kwargs[name] = [conv[layer] for conv in v]
                else:
                    kwargs[name] = v[layer]

            return type(self)(**kwargs)

        def mem_usage_bytes(self):
            return sum(
                get_tensor_size_bytes(getattr(self, f.name))
                for f in dataclasses.fields(self)
            )

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):
        intermediate_ssm: torch.Tensor
        intermediate_conv_window: List[torch.Tensor]

    def __init__(
        self,
        *,
        size: int,
        spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        conv_state_shape = cache_params.shape.conv
        temporal_state_shape = cache_params.shape.temporal
        conv_dtype = cache_params.dtype.conv
        ssm_dtype = cache_params.dtype.temporal
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        num_mamba_layers = len(mamba_layer_ids)

        self.size = size
        self.device = device

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        with (
            self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ),
        ):
            conv_state = [
                torch.zeros(
                    size=(num_mamba_layers, size + 1) + conv_shape,
                    dtype=conv_dtype,
                    device=device,
                )
                for conv_shape in conv_state_shape
            ]

            if _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    _init_npu_conv_state,
                )

                conv_state = _init_npu_conv_state(
                    conv_state[0], conv_state_shape, speculative_num_draft_tokens
                )

            if _is_cpu and _cpu_has_amx_support:
                from sglang.srt.layers.amx_utils import _init_amx_conv_state

                # CPU uses a different layout of conv_state for kernel optimization
                conv_state = _init_amx_conv_state(conv_state)

            temporal_state = torch.zeros(
                size=(num_mamba_layers, size + 1) + temporal_state_shape,
                dtype=ssm_dtype,
                device=device,
            )
            if speculative_num_draft_tokens is not None:
                if _is_npu:
                    temporal_state = temporal_state.transpose(-1, -2)
                    temporal_state_shape = (
                        *temporal_state_shape[:-2],
                        temporal_state_shape[-1],
                        temporal_state_shape[-2],
                    )
                # Cache intermediate SSM states per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, HV, K, V]
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device="cuda",
                )
                # Cache intermediate conv windows (last K-1 inputs) per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, dim, K-1]
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            conv_shape[0],
                            conv_shape[1],
                        ),
                        dtype=conv_dtype,
                        device="cuda",
                    )
                    for conv_shape in conv_state_shape
                ]
                self.mamba_cache = self.SpeculativeState(
                    conv=conv_state,
                    temporal=temporal_state,
                    intermediate_ssm=intermediate_ssm_state_cache,
                    intermediate_conv_window=intermediate_conv_window_cache,
                )
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                    f"intermediate_ssm_state_cache size: {get_tensor_size_bytes(intermediate_ssm_state_cache) / GB:.2f}GB "
                    f"intermediate_conv_window_cache size: {get_tensor_size_bytes(intermediate_conv_window_cache) / GB:.2f}GB "
                )
            else:
                self.mamba_cache = self.State(conv=conv_state, temporal=temporal_state)
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                )
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.free_slots = torch.arange(
                1, self.size + 1, dtype=torch.int64, device=self.device
            )
            self.mem_usage = self.mamba_cache.mem_usage_bytes() / GB
            self.num_mamba_layers = num_mamba_layers

    def get_speculative_mamba2_params_all_layers(self) -> SpeculativeState:
        assert isinstance(self.mamba_cache, self.SpeculativeState)
        return self.mamba_cache

    def mamba2_layer_cache(self, layer_id: int):
        return self.mamba_cache.at_layer_idx(layer_id)

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def clear_slots(self, indices: torch.Tensor):
        """Zero out mamba state at the given pool indices. Must run on forward stream."""
        need_size = len(indices)
        for i in range(len(self.mamba_cache.conv)):
            t = self.mamba_cache.conv[i]
            z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
                t.shape[0], need_size, *t.shape[2:]
            )
            t[:, indices] = z
        t = self.mamba_cache.temporal
        z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
            t.shape[0], need_size, *t.shape[2:]
        )
        t[:, indices] = z

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        self.free_slots = torch.cat((self.free_slots, free_index))

    def clear(self):
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )

    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):
        for i in range(len(self.mamba_cache.conv)):
            self.mamba_cache.conv[i][:, dst_indices] = self.mamba_cache.conv[i][
                :, src_indices
            ]
        self.mamba_cache.temporal[:, dst_indices] = self.mamba_cache.temporal[
            :, src_indices
        ]

    def get_cpu_copy(self, indices):
        current_platform.synchronize()
        conv_cpu = [
            conv[:, indices].to("cpu", non_blocking=True)
            for conv in self.mamba_cache.conv
        ]
        temporal_cpu = self.mamba_cache.temporal[:, indices].to(
            "cpu", non_blocking=True
        )
        current_platform.synchronize()
        return conv_cpu, temporal_cpu

    def load_cpu_copy(self, mamba_cache_cpu, indices):
        conv_cpu, temporal_cpu = mamba_cache_cpu
        current_platform.synchronize()
        for i, conv in enumerate(self.mamba_cache.conv):
            conv[:, indices] = conv_cpu[i].to(conv.device, non_blocking=True)
        self.mamba_cache.temporal[:, indices] = temporal_cpu.to(
            self.mamba_cache.temporal.device, non_blocking=True
        )
        current_platform.synchronize()

    def get_contiguous_buf_infos(self):
        """
        Get buffer info for RDMA registration.
        Only returns conv and temporal state buffers, excluding intermediate buffers
        used for speculative decoding (intermediate_ssm, intermediate_conv_window).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            # Skip intermediate buffers used only for speculative decoding
            # These buffers have different size (spec_state_size + 1) and should not be transferred
            if field in ("intermediate_ssm", "intermediate_conv_window"):
                continue
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)
        data_ptrs, data_lens, item_lens = [], [], []

        for _, state_tensor in enumerate(state_tensors):
            data_ptrs += [
                state_tensor[i].data_ptr() for i in range(self.num_mamba_layers)
            ]
            data_lens += [state_tensor[i].nbytes for i in range(self.num_mamba_layers)]
            item_lens += [
                state_tensor[i][0].nbytes for i in range(self.num_mamba_layers)
            ]
        return data_ptrs, data_lens, item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each state tensor.

        For mamba state, the layout is:
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

        The 3rd dimension (index 2) is the one that gets sliced by TP.
        Returns the size of this dimension for each tensor (repeated for each layer).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)

        dim_per_tensor = []
        for state_tensor in state_tensors:
            # state_tensor shape: [num_layers, size+1, sliceable_dim, ...]
            # The sliceable dimension is at index 2 (after num_layers and size)
            sliceable_dim = state_tensor.shape[2]
            # Repeat for each layer since we have per-layer data_ptrs
            dim_per_tensor += [sliceable_dim] * self.num_mamba_layers
        return dim_per_tensor


class HybridReqToTokenPool(ReqToTokenPool):
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        *,
        size: int,
        mamba_size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            mamba_size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def _init_mamba_pool(
        self,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
    ):
        self.mamba_pool = MambaPool(
            size=mamba_size,
            spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}

        self.device = device
        req_pool_size = self.req_to_token.shape[0]
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            req_pool_size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (req_pool_size, self.mamba_ping_pong_track_buffer_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            )

    def register_layer_transfer_counter(
        self, layer_transfer_counter: "LayerDoneCounter"
    ):
        self.layer_transfer_counter = layer_transfer_counter

    # For chunk prefill req, we do not need to allocate mamba cache,
    # We could use allocated mamba cache instead.
    def alloc(self, reqs: List["Req"]) -> Optional[List[int]]:
        select_index = super().alloc(reqs)
        if select_index is None:
            return None

        mamba_indices: list[torch.Tensor] = []
        mamba_ping_pong_track_buffers: list[torch.Tensor] = []
        for req in reqs:
            if req.mamba_pool_idx is not None:  # for radix cache / continuing chunked
                pass
            else:
                mid = self.mamba_pool.alloc(1)
                assert (
                    mid is not None
                ), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size. {mid=}, {self.mamba_pool.size=}, {self.mamba_pool.available_size()=}, {len(reqs)=}"
                req.mamba_pool_idx = mid[0]
                req.mamba_needs_clear = True
            mamba_indices.append(req.mamba_pool_idx)
            if self.enable_mamba_extra_buffer:
                if req.mamba_ping_pong_track_buffer is None:
                    req.mamba_ping_pong_track_buffer = self.mamba_pool.alloc(
                        self.mamba_ping_pong_track_buffer_size
                    )
                    assert (
                        req.mamba_ping_pong_track_buffer is not None
                    ), "Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
                    req.mamba_next_track_idx = 0
                mamba_ping_pong_track_buffers.append(req.mamba_ping_pong_track_buffer)
        assert len(select_index) == len(
            mamba_indices
        ), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size."
        if self.enable_mamba_extra_buffer:
            assert len(select_index) == len(
                mamba_ping_pong_track_buffers
            ), f"Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
        mamba_index_tensor = torch.stack(mamba_indices).to(dtype=torch.int32)
        self.req_index_to_mamba_index_mapping[select_index] = mamba_index_tensor
        if self.enable_mamba_extra_buffer:
            ping_pong_tensor = torch.stack(mamba_ping_pong_track_buffers)
            self.req_index_to_mamba_ping_pong_track_buffer_mapping[select_index] = (
                ping_pong_tensor
            )
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def mamba2_layer_cache(self, layer_id: int):
        assert layer_id in self.mamba_map
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.mamba_pool.mamba2_layer_cache(self.mamba_map[layer_id])

    def get_speculative_mamba2_params_all_layers(self) -> MambaPool.SpeculativeState:
        return self.mamba_pool.get_speculative_mamba2_params_all_layers()

    def get_state_buf_infos(self):
        return self.mamba_pool.get_contiguous_buf_infos()

    def get_state_dim_per_tensor(self):
        return self.mamba_pool.get_state_dim_per_tensor()

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        if self.mamba_ping_pong_track_buffer_size == 2:
            return 1 - mamba_next_track_idx
        else:
            return mamba_next_track_idx

    def free_mamba_cache(
        self, req: "Req", mamba_ping_pong_track_buffer_to_keep: Optional[int] = None
    ):
        mamba_index = req.mamba_pool_idx
        assert mamba_index is not None, "double free? mamba_index is None"
        self.mamba_pool.free(mamba_index.unsqueeze(0))
        req.mamba_pool_idx = None

        if self.enable_mamba_extra_buffer:
            mamba_ping_pong_track_buffer_to_free = (
                self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx]
            )
            if mamba_ping_pong_track_buffer_to_keep is not None:
                assert mamba_ping_pong_track_buffer_to_keep in [
                    0,
                    1,
                ], f"mamba_ping_pong_track_buffer_to_keep must be 0 or 1, {mamba_ping_pong_track_buffer_to_keep=}"
                # Avoid Python-list advanced indexing on a device tensor.
                # The ping-pong buffer size is either 2 (normal) or 1 (spec decode).
                if self.mamba_ping_pong_track_buffer_size == 2:
                    idx_to_free = 1 - mamba_ping_pong_track_buffer_to_keep
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[
                            idx_to_free : idx_to_free + 1
                        ]
                    )
                else:
                    assert self.mamba_ping_pong_track_buffer_size == 1, (
                        f"Unexpected mamba_ping_pong_track_buffer_size="
                        f"{self.mamba_ping_pong_track_buffer_size}"
                    )
                    assert mamba_ping_pong_track_buffer_to_keep == 0, (
                        "mamba_ping_pong_track_buffer_to_keep must be 0 when "
                        "mamba_ping_pong_track_buffer_size is 1"
                    )
                    # Keep the only slot, so free nothing.
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[0:0]
                    )
            self.mamba_pool.free(mamba_ping_pong_track_buffer_to_free)

    def clear(self):
        logger.info("Reset HybridReqToTokenPool")
        super().clear()
        self.mamba_pool.clear()
        self.req_index_to_mamba_index_mapping.zero_()
        if self.enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping.zero_()


class KVCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

    def _finalize_allocation_log(self, num_tokens: int):
        """Common logging and mem_usage computation for KV cache allocation.
        Supports both tuple (K, V) size returns and single KV size returns.
        """
        kv_size_bytes = self.get_kv_size_bytes()
        if isinstance(kv_size_bytes, tuple):
            k_size, v_size = kv_size_bytes
            k_size_GB = k_size / GB
            v_size_GB = v_size / GB
            logger.info(
                f"KV Cache is allocated. dtype: {self.dtype}, #tokens: {num_tokens}, K size: {k_size_GB:.2f} GB, V size: {v_size_GB:.2f} GB"
            )
            self.mem_usage = k_size_GB + v_size_GB
        else:
            kv_size_GB = kv_size_bytes / GB
            logger.info(
                f"KV Cache is allocated. dtype: {self.dtype}, #tokens: {num_tokens}, KV size: {kv_size_GB:.2f} GB"
            )
            self.mem_usage = kv_size_GB

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError()

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool


class MHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None else head_dim
        )

        self._create_buffers()

        self.device_module = torch.get_device_module(self.device)

        _use_alt_stream = _is_cuda or current_platform.is_cuda_alike()
        self.alt_stream = (
            self.device_module.Stream()
            if _use_alt_stream and enable_alt_stream
            else None
        )

        if enable_kv_cache_copy:
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        self._finalize_allocation_log(size)

        # for store_cache JIT kernel
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = self.head_dim == self.v_head_dim

    def _init_kv_copy_and_warmup(self):
        # Heuristics for KV copy tiling
        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

        stride_bytes = int(self.data_strides[0].item())
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        # Calculate num_locs_upper to avoid large Triton specialization (e.g. 8192)
        chunk_upper = 128 if bytes_per_tile >= _KV_COPY_TILE_SIZE_LARGE else 256

        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
            "num_locs_upper": chunk_upper,
        }

        dummy_loc = torch.zeros(chunk_upper, dtype=torch.int64, device=self.device)
        grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])

        copy_all_layer_kv_cache_tiled[grid](
            self.data_ptrs,
            self.data_strides,
            dummy_loc,
            dummy_loc,
            1,
            chunk_upper,
            BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
            num_warps=self._kv_copy_config["num_warps"],
            num_stages=2,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.k_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.v_head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_cpu_copy(self, indices, mamba_indices=None):
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        current_platform.synchronize()

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
            device_module=self.device_module,
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # Catch stale indices here instead of as illegal-addr or silent KV corruption.
        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")

        if envs.SGLANG_NATIVE_MOVE_KV_CACHE.get():
            move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)
            return

        N = tgt_loc.numel()
        if N == 0:
            return

        assert (
            self._kv_copy_config is not None
        ), "KV copy not initialized. Set enable_kv_cache_copy=True in __init__"

        cfg = self._kv_copy_config
        cap = int(cfg.get("num_locs_upper", 256))
        grid = (self.data_ptrs.numel(), cfg["byte_tiles"])

        if N <= cap:
            upper = next_power_of_2(N)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                N,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )
            return

        # Huge N: chunk, but each chunk's upper is still pow2(<= cap)
        for start in range(0, N, cap):
            end = min(start + cap, N)
            chunk_len = end - start
            upper = next_power_of_2(chunk_len)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc[start:end],
                src_loc[start:end],
                chunk_len,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )


class NoOpMHATokenToKVPool(MHATokenToKVPool):
    """KV cache pool that skips physical K/V buffer allocation.

    Used in embedding-mode prefill-only workloads with the FA
    fa_skip_kv_cache path, where no layer reads or writes KV cache because
    attention uses raw K/V via flash_attn_varlen_func. Other prefill-only paths
    such as scoring/MIS may benefit from the same idea later, but some still
    stage K/V through paged cache today.

    This class keeps the scheduler's view of pool capacity (self.size is
    honored for admission) but allocates only (page_size, head_num, head_dim)
    placeholder tensors per layer to satisfy any code paths that dereference
    the buffers.

    Callers MUST ensure no real set_kv_buffer/get_*_buffer calls happen against
    this pool; those paths raise loudly so misuse is visible.
    """

    def _create_buffers(self):
        # Allocate minimal placeholder buffers. They exist purely so that code
        # paths holding `k_buffer` / `v_buffer` references (pointer tables,
        # layer-transfer counters, stride arithmetic) keep working without
        # None-guards scattered across the codebase. Shape is
        # [page_size, head_num, head_dim] per layer so that the unconditional
        # `key_cache.view(-1, page_size, head_num, head_dim)` in the FA backend
        # at the top of forward_extend succeeds regardless of --page-size.
        # Total footprint is still on the order of KB vs GBs for a real pool.
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_buffer = [
                torch.zeros(
                    (self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (self.page_size, self.head_num, self.v_head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _finalize_allocation_log(self, num_tokens: int):
        self.mem_usage = 0.0
        placeholder_bytes = (
            2
            * self.layer_num
            * self.page_size
            * self.head_num
            * max(self.head_dim, self.v_head_dim)
            * self.store_dtype.itemsize
        )
        logger.info(
            f"KV Cache skipped (no-op pool). Logical #tokens: {num_tokens}, "
            f"physical K/V size: ~{placeholder_bytes / 1024:.1f} KB placeholder"
        )

    def get_kv_size_bytes(self):
        # Report zero so downstream memory accounting matches reality.
        return (0, 0)

    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError(
            "NoOpMHATokenToKVPool.set_kv_buffer was called. This pool is only "
            "valid in prefill-only modes (e.g. --is-embedding, scoring) with "
            "the FA backend's fa_skip_kv_cache path active; the attention "
            "backend must never write to it. Check that the workload truly "
            "performs no decode and that the FA backend's fa_skip_kv_cache "
            "preconditions are met."
        )

    def get_key_buffer(self, layer_id: int):
        # Return the placeholder. The FA backend reads this before taking the
        # fa_skip_kv_cache branch (which does not use it); the placeholder shape
        # is (page_size, head_num, head_dim) so downstream .view() calls succeed.
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # no-op; embedding mode has no KV cache to move
        return


class MHATokenToKVPoolFP4(MHATokenToKVPool):

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = self.head_num
                k = self.head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8
                self.k_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.k_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.k_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.k_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_v_nope_fp4 = self.v_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_v_nope_fp4_sf = self.v_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_v_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_v_nope_fp4, cache_v_nope_fp4_sf
            )
            return cache_v_nope_fp4_dequant
        return self.v_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)
            cache_v, cache_v_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_v)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

            cache_k_fp4_sf = cache_k_fp4_sf.view(self.store_dtype)
            cache_v_fp4_sf = cache_v_fp4_sf.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:
            # Overlap the copy of K and V cache for small batch size
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v

                self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf


class HiggsMHA2BitTokenToKVPool(MHATokenToKVPool):
    """MHA KV pool that stores K/V rows as 2-bit HIGGS packed slots.

    Used by the SMC-SD draft model to compress the per-head KV footprint
    by 3.76x vs FP8 and 7.53x vs BF16. The codec lives in
    ``sglang.srt.layers.quantization.higgs_mha_2bit_kv``.

    The decode path materializes the packed buffer back to the requested
    dtype (BF16 by default) on demand inside ``_get_key_buffer`` /
    ``_get_value_buffer``. This keeps the existing dense Triton decode
    kernel unchanged; a fused dequant-inside-attention kernel is a
    follow-on optimization.
    """

    def _create_buffers(self):
        from sglang.srt.layers.quantization.higgs_mha_2bit_kv import (
            HiggsMHA2BitCodec,
            HiggsMHA2BitConfig,
        )

        self._higgs_k_config = HiggsMHA2BitConfig(head_dim=self.head_dim)
        self._higgs_v_config = HiggsMHA2BitConfig(head_dim=self.v_head_dim)
        self._higgs_k_codec = HiggsMHA2BitCodec(
            self._higgs_k_config, torch.device(self.device)
        )
        self._higgs_v_codec = HiggsMHA2BitCodec(
            self._higgs_v_config, torch.device(self.device)
        )
        # Force uint8 storage so ``index_put_`` works on the packed rows.
        self.store_dtype = torch.uint8

        k_slot_bytes = self._higgs_k_config.slot_bytes
        v_slot_bytes = self._higgs_v_config.slot_bytes
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                m = self.size + self.page_size
                self.k_buffer = [
                    torch.zeros(
                        (m, self.head_num, k_slot_bytes),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, self.head_num, v_slot_bytes),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _dequant_packed_buffer(
        self,
        packed: torch.Tensor,
        codec: Any,
        head_dim: int,
    ) -> torch.Tensor:
        if packed.is_cuda and self.dtype == torch.bfloat16:
            from sglang.jit_kernel.higgs_mha_2bit_kv import (
                dequantize_higgs_mha_2bit,
            )

            out = torch.empty(
                (*packed.shape[:2], head_dim),
                dtype=self.dtype,
                device=packed.device,
            )
            dequantize_higgs_mha_2bit(packed, out, codec.codebook)
            return out
        return codec.decompress(packed, self.dtype)

    def _get_key_buffer(self, layer_id: int):
        # Fused HIGGS decode reads packed slots directly. This materialization
        # path remains for extend/prefill and non-fused fallback backends.
        packed = self.k_buffer[layer_id - self.start_layer]
        return self._dequant_packed_buffer(packed, self._higgs_k_codec, self.head_dim)

    def _get_value_buffer(self, layer_id: int):
        packed = self.v_buffer[layer_id - self.start_layer]
        return self._dequant_packed_buffer(
            packed, self._higgs_v_codec, self.v_head_dim
        )

    def get_packed_key_buffer(self, layer_id: int) -> torch.Tensor:
        """Return the packed (uint8) K buffer for the fused decode path."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.k_buffer[layer_id - self.start_layer]

    def get_packed_value_buffer(self, layer_id: int) -> torch.Tensor:
        """Return the packed (uint8) V buffer for the fused decode path."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.v_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        # The codec computes its own per-row FP16 scale (||rotated|| /
        # sqrt(head_dim)). External per-tensor k_scale / v_scale are
        # redundant — applying them at store time without an equivalent
        # multiply on decompress would shift the recovered K/V by
        # 1/scale, biasing attention. The codec's per-row scale already
        # carries the magnitude information needed for accurate
        # reconstruction. k_scale / v_scale are accepted for API parity
        # with MHATokenToKVPool and silently dropped.
        del k_scale, v_scale
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        k_buffer = self.k_buffer[layer_id - self.start_layer]
        v_buffer = self.v_buffer[layer_id - self.start_layer]
        if (
            cache_k.is_cuda
            and cache_v.is_cuda
            and cache_k.dtype == torch.bfloat16
            and cache_v.dtype == torch.bfloat16
        ):
            from sglang.srt.layers.attention.triton_ops.higgs_mha_kv_pack import (
                store_higgs_mha_2bit_triton,
            )

            store_higgs_mha_2bit_triton(k_buffer, loc, cache_k)
            store_higgs_mha_2bit_triton(v_buffer, loc, cache_v)
            return

        packed_k = self._higgs_k_codec.compress(cache_k)
        packed_v = self._higgs_v_codec.compress(cache_v)
        k_buffer[loc] = packed_k
        v_buffer[loc] = packed_v


class HybridLinearKVPool(KVCache):
    """KV cache with separate pools for full and linear attention layers."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
        mamba_pool: MambaPool,
        enable_memory_saver: bool = False,
        # TODO: refactor mla related args
        use_mla: bool = False,
        kv_lora_rank: int = None,
        qk_rope_head_dim: int = None,
        start_layer: Optional[int] = None,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = page_size
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self.head_num = head_num
        self.head_dim = head_dim
        self.mamba_pool = mamba_pool
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose
        self.use_mla = use_mla
        if not use_mla:

            TokenToKVPoolClass = MHATokenToKVPool

            if current_platform.is_out_of_tree():
                TokenToKVPoolClass = current_platform.get_mha_kv_pool_cls()
            elif _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMHATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=enable_memory_saver,
            )
        else:

            TokenToKVPoolClass = MLATokenToKVPool

            if current_platform.is_out_of_tree():
                TokenToKVPoolClass = current_platform.get_mla_kv_pool_cls()
            elif _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMLATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                layer_num=self.full_layer_nums,
                device=device,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                enable_memory_saver=enable_memory_saver,
            )
        self.full_attention_layer_id_mapping = {
            id: i for i, id in enumerate(full_attention_layer_ids)
        }
        if use_mla:
            self.mem_usage = self.get_kv_size_bytes() / GB
        else:
            k_size, v_size = self.get_kv_size_bytes()
            self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        return self.full_kv_pool.get_kv_size_bytes()

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    def get_state_buf_infos(self):
        mamba_data_ptrs, mamba_data_lens, mamba_item_lens = (
            self.mamba_pool.get_contiguous_buf_infos()
        )
        return mamba_data_ptrs, mamba_data_lens, mamba_item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each mamba state tensor."""
        return self.mamba_pool.get_state_dim_per_tensor()

    def maybe_get_custom_mem_pool(self):
        return self.full_kv_pool.maybe_get_custom_mem_pool()

    def _transfer_full_attention_id(self, layer_id: int):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(
                f"{layer_id=} not in full attention layers: {self.full_attention_layer_id_mapping.keys()}"
            )
        return self.full_attention_layer_id_mapping[layer_id]

    def register_layer_transfer_counter(
        self, layer_transfer_counter: "LayerDoneCounter"
    ):
        self.layer_transfer_counter = layer_transfer_counter
        # The layer-wise wait logic is executed at the Hybrid LinearPool level;
        # no additional wait is needed in the full_kv_pool
        self.full_kv_pool.register_layer_transfer_counter(None)

    def _wait_for_layer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_key_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_buffer(layer_id)

    def prefetch_layersplit_kv_buffer(
        self,
        layer_id: int,
        use_staging: bool = False,
        active_rows: Optional[int] = None,
    ) -> None:
        if not self.use_mla:
            return
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        self.full_kv_pool.prefetch_layersplit_kv_buffer(
            layer_id, use_staging, active_rows
        )

    def prefetch_layersplit_index_k_with_scale_buffer(self, layer_id: int) -> None:
        if not self.use_mla:
            return
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        if hasattr(self.full_kv_pool, "prefetch_layersplit_index_k_with_scale_buffer"):
            self.full_kv_pool.prefetch_layersplit_index_k_with_scale_buffer(layer_id)

    def set_layersplit_active_rows(self, active_rows: Optional[int]) -> None:
        if self.use_mla:
            self.full_kv_pool.set_layersplit_active_rows(active_rows)

    @contextmanager
    def _transfer_id_context(self, layer: RadixAttention):

        @contextmanager
        def _patch_layer_id(layer):
            original_layer_id = layer.layer_id
            layer.layer_id = self._transfer_full_attention_id(layer.layer_id)
            try:
                yield
            finally:
                layer.layer_id = original_layer_id

        with _patch_layer_id(layer):
            yield

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        layer_id = self._transfer_full_attention_id(layer.layer_id)
        if not self.use_mla:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
        else:
            with self._transfer_id_context(layer):
                self.full_kv_pool.set_kv_buffer(
                    layer,
                    loc,
                    cache_k,
                    cache_v,
                )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)

    def get_cpu_copy(self, indices, mamba_indices=None):
        kv_cpu = self.full_kv_pool.get_cpu_copy(indices)
        mamba_cpu = (
            self.mamba_pool.get_cpu_copy(mamba_indices)
            if mamba_indices is not None
            else None
        )
        return kv_cpu, mamba_cpu

    def load_cpu_copy(self, cache_cpu, indices, mamba_indices=None):
        kv_cpu, mamba_cpu = cache_cpu
        self.full_kv_pool.load_cpu_copy(kv_cpu, indices)
        if mamba_cpu is not None and mamba_indices is not None:
            self.mamba_pool.load_cpu_copy(mamba_cpu, mamba_indices)

    def get_v_head_dim(self):
        return self.full_kv_pool.get_value_buffer(0).shape[-1]

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        assert self.use_mla, "set_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            self.full_kv_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        assert self.use_mla, "get_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            return self.full_kv_pool.get_mla_kv_buffer(layer, loc, dst_dtype)


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_dsa: bool = False,
        override_kv_cache_dim: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_dsa = use_dsa
        self.dsa_kv_cache_store_fp8 = (
            use_dsa
            and dtype == torch.float8_e4m3fn
            and override_kv_cache_dim is not None
        )
        # When override_kv_cache_dim is provided with dsa model, we assume the
        # override kv cache dim is correct and use it directly.
        self.kv_cache_dim = (
            override_kv_cache_dim
            if self.dsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )
        self.layersplit_policy = self._build_layersplit_policy() if use_dsa else None
        self.layersplit_kv_buffer = None
        self._layersplit_kv_prefetch_layer_id = None
        self._layersplit_kv_prefetch_buffer = None
        self._layersplit_kv_prefetch_work = None
        self._layersplit_kv_prefetch_owner_staging = False
        self._layersplit_kv_prefetch_active_rows = None
        self._layersplit_active_rows = None
        self._layersplit_decode_materialized = False

        self._create_buffers()

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        if not use_dsa:
            # DSA will allocate indexer KV cache later and then log the total size
            self._finalize_allocation_log(size)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                self.kv_buffer = []
                for local_layer_idx in range(self.layer_num):
                    layer_id = self.start_layer + local_layer_idx
                    rows = (
                        m
                        if self.layersplit_owns_layer(layer_id)
                        else self.page_size
                    )
                    self.kv_buffer.append(
                        torch.zeros(
                            (rows, 1, self.kv_cache_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                    )
                if self.layersplit_policy is not None:
                    self.layersplit_kv_buffer = torch.zeros(
                        (m, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )

    def _clear_buffers(self):
        self._wait_layersplit_kv_prefetch()
        del self.kv_buffer
        if self.layersplit_kv_buffer is not None:
            del self.layersplit_kv_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        if self.layersplit_kv_buffer is not None:
            kv_size_bytes += get_tensor_size_bytes(self.layersplit_kv_buffer)
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def _build_layersplit_policy(self):
        from sglang.srt.layers.attention.dsa.layersplit import LayerSplitPolicy
        from sglang.srt.layers.dp_attention import (
            get_attention_cp_rank,
            get_attention_cp_size,
        )
        from sglang.srt.server_args import get_global_server_args

        try:
            server_args = get_global_server_args()
        except ValueError:
            return None
        if (
            getattr(server_args, "dsa_prefill_cp_kv_storage_mode", "replicated")
            != "layersplit"
        ):
            return None
        cp_size = get_attention_cp_size()
        if cp_size <= 1:
            return None
        return LayerSplitPolicy(
            cp_rank=get_attention_cp_rank(),
            cp_size=cp_size,
            start_layer=self.start_layer,
            end_layer=self.start_layer + self.layer_num,
            layout=server_args.dsa_prefill_cp_layersplit_layout,
        )

    def layersplit_owns_layer(self, layer_id: int) -> bool:
        if self.layersplit_policy is None:
            return True
        if self._layersplit_decode_materialized:
            return True
        return self.layersplit_policy.owns_layer(layer_id)

    def _wait_layersplit_kv_prefetch(self) -> Optional[torch.Tensor]:
        work = self._layersplit_kv_prefetch_work
        if work is not None:
            work.wait()
        buffer = self._layersplit_kv_prefetch_buffer
        self._layersplit_kv_prefetch_layer_id = None
        self._layersplit_kv_prefetch_buffer = None
        self._layersplit_kv_prefetch_work = None
        self._layersplit_kv_prefetch_owner_staging = False
        self._layersplit_kv_prefetch_active_rows = None
        return buffer

    def _finish_layersplit_kv_prefetch_for_update(
        self, layer_id: int
    ) -> Optional[torch.Tensor]:
        if self._layersplit_kv_prefetch_layer_id != layer_id:
            return None
        active_rows = self._layersplit_kv_prefetch_active_rows
        owner_staging = self._layersplit_kv_prefetch_owner_staging
        kv_buffer = self._wait_layersplit_kv_prefetch()
        assert kv_buffer is not None
        self._layersplit_kv_prefetch_layer_id = layer_id
        self._layersplit_kv_prefetch_buffer = kv_buffer
        self._layersplit_kv_prefetch_work = None
        self._layersplit_kv_prefetch_owner_staging = owner_staging
        self._layersplit_kv_prefetch_active_rows = active_rows
        return kv_buffer

    def _layersplit_active_rows_from_loc(self, loc: torch.Tensor) -> Optional[int]:
        if loc.numel() == 0:
            return None
        return self._layersplit_active_rows

    def set_layersplit_active_rows(self, active_rows: Optional[int]) -> None:
        if active_rows is None:
            self._layersplit_active_rows = None
        else:
            self._layersplit_active_rows = min(
                active_rows, self.size + self.page_size
            )

    def _refresh_layersplit_data_ptrs(self) -> None:
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )

    def _ensure_layersplit_decode_kv_buffer(self, layer_idx: int) -> None:
        assert self.layersplit_kv_buffer is not None
        if self.kv_buffer[layer_idx].shape[0] == self.layersplit_kv_buffer.shape[0]:
            return
        self.kv_buffer[layer_idx] = torch.zeros_like(self.layersplit_kv_buffer)

    def materialize_layersplit_for_decode(
        self, active_rows: Optional[int] = None
    ) -> None:
        policy = self.layersplit_policy
        if policy is None or self._layersplit_decode_materialized:
            return
        assert self.layersplit_kv_buffer is not None
        self._wait_layersplit_kv_prefetch()
        if active_rows is None:
            active_rows = self._layersplit_active_rows

        from sglang.srt.layers.dp_attention import get_attention_cp_group

        for layer_id in range(self.start_layer, self.start_layer + self.layer_num):
            layer_idx = layer_id - self.start_layer
            if self.layer_transfer_counter is not None:
                self.layer_transfer_counter.wait_until(layer_idx)
            owner_rank = policy.owner_rank(layer_id)
            if owner_rank != policy.cp_rank:
                self._ensure_layersplit_decode_kv_buffer(layer_idx)
            kv_buffer = self.kv_buffer[layer_idx]
            broadcast_buffer = self._layersplit_broadcast_buffer(
                kv_buffer, active_rows
            )
            get_attention_cp_group().broadcast(broadcast_buffer, src=owner_rank)
        self._refresh_layersplit_data_ptrs()
        self._layersplit_decode_materialized = True

    def _layersplit_broadcast_buffer(
        self, kv_buffer: torch.Tensor, active_rows: Optional[int]
    ) -> torch.Tensor:
        if active_rows is None or active_rows >= kv_buffer.shape[0]:
            return kv_buffer
        return kv_buffer[:active_rows]

    def _layersplit_broadcast_async(
        self, buffer: torch.Tensor, owner_rank: int
    ) -> Optional[Any]:
        from sglang.srt.layers.dp_attention import get_attention_cp_group

        return get_attention_cp_group().broadcast_async(buffer, src=owner_rank)

    def _has_compatible_layersplit_kv_prefetch(
        self, layer_id: int, active_rows: Optional[int]
    ) -> bool:
        if self._layersplit_kv_prefetch_layer_id != layer_id:
            return False
        prefetched_rows = self._layersplit_kv_prefetch_active_rows
        if active_rows is None:
            return prefetched_rows is None
        return prefetched_rows is None or prefetched_rows >= active_rows

    def prefetch_layersplit_kv_buffer(
        self,
        layer_id: int,
        use_staging: bool = False,
        active_rows: Optional[int] = None,
    ) -> None:
        policy = self.layersplit_policy
        if policy is None or self._layersplit_decode_materialized:
            return
        if active_rows is None:
            active_rows = self._layersplit_active_rows
        if self._has_compatible_layersplit_kv_prefetch(layer_id, active_rows):
            return
        if self._layersplit_kv_prefetch_layer_id is not None:
            self._wait_layersplit_kv_prefetch()

        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        layer_idx = layer_id - self.start_layer
        owner_rank = policy.owner_rank(layer_id)
        if owner_rank == policy.cp_rank:
            kv_buffer = self.kv_buffer[layer_idx]
            if use_staging:
                assert self.layersplit_kv_buffer is not None
                broadcast_buffer = self._layersplit_broadcast_buffer(
                    kv_buffer, active_rows
                )
                self.layersplit_kv_buffer[: broadcast_buffer.shape[0]].copy_(
                    broadcast_buffer
                )
                kv_buffer = self.layersplit_kv_buffer
        else:
            kv_buffer = self.layersplit_kv_buffer

        broadcast_buffer = self._layersplit_broadcast_buffer(kv_buffer, active_rows)
        self._layersplit_kv_prefetch_layer_id = layer_id
        self._layersplit_kv_prefetch_buffer = kv_buffer
        self._layersplit_kv_prefetch_active_rows = active_rows
        self._layersplit_kv_prefetch_owner_staging = (
            use_staging and owner_rank == policy.cp_rank
        )
        self._layersplit_kv_prefetch_work = self._layersplit_broadcast_async(
            broadcast_buffer, owner_rank
        )

    def _get_layersplit_kv_buffer(self, layer_id: int) -> torch.Tensor:
        layer_idx = layer_id - self.start_layer
        policy = self.layersplit_policy
        if policy is None or self._layersplit_decode_materialized:
            return self.kv_buffer[layer_idx]
        if self._layersplit_kv_prefetch_layer_id == layer_id:
            owner_staging = self._layersplit_kv_prefetch_owner_staging
            kv_buffer = self._wait_layersplit_kv_prefetch()
            assert kv_buffer is not None
            if owner_staging:
                return self.kv_buffer[layer_idx]
            return kv_buffer
        owner_rank = policy.owner_rank(layer_id)
        if owner_rank == policy.cp_rank:
            kv_buffer = self.kv_buffer[layer_idx]
        else:
            kv_buffer = self.layersplit_kv_buffer

        from sglang.srt.layers.dp_attention import get_attention_cp_group

        return get_attention_cp_group().broadcast(kv_buffer, src=owner_rank)

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        kv_buffer = self._get_layersplit_kv_buffer(layer_id)
        if self.store_dtype != self.dtype:
            return kv_buffer.view(self.dtype)

        return kv_buffer

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        kv_buffer = self._get_layersplit_kv_buffer(layer_id)
        if self.store_dtype != self.dtype:
            return kv_buffer[..., : self.kv_lora_rank].view(self.dtype)
        return kv_buffer[..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        kv_buffer = self._get_layersplit_kv_buffer(layer_id)
        if self.store_dtype != self.dtype:
            kv_buffer = kv_buffer.view(self.dtype)
        return kv_buffer, kv_buffer[..., : self.kv_lora_rank]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        layer_idx = layer_id - self.start_layer
        active_rows = self._layersplit_active_rows_from_loc(loc)
        if not self.layersplit_owns_layer(layer_id):
            self.prefetch_layersplit_kv_buffer(layer_id, active_rows=active_rows)
            return
        assert not self.dsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        if self._layersplit_kv_prefetch_layer_id == layer_id:
            self._finish_layersplit_kv_prefetch_for_update(layer_id)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_idx][loc] = cache_k.view(self.store_dtype)
        else:
            self.kv_buffer[layer_idx][loc] = cache_k
        self.prefetch_layersplit_kv_buffer(layer_id, active_rows=active_rows)

    def _set_mla_kv_buffer_to_buffer(
        self,
        kv_buffer: torch.Tensor,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        if _is_hip and self.use_dsa and self.dtype == fp8_dtype:
            set_mla_kv_buffer_triton_fp8_quant(
                kv_buffer,
                loc,
                cache_k_nope,
                cache_k_rope,
                fp8_dtype,
            )
        elif self.dsa_kv_cache_store_fp8:
            cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
                cache_k_nope, cache_k_rope
            )
            set_mla_kv_buffer_triton(
                kv_buffer,
                loc,
                cache_k_nope_fp8,
                cache_k_rope_fp8,
            )
        else:
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                kv_buffer,
                loc,
                cache_k_nope,
                cache_k_rope,
            )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id
        layer_idx = layer_id - self.start_layer
        active_rows = self._layersplit_active_rows_from_loc(loc)
        if not self.layersplit_owns_layer(layer_id):
            kv_buffer = self._finish_layersplit_kv_prefetch_for_update(layer_id)
            if kv_buffer is not None:
                self._set_mla_kv_buffer_to_buffer(
                    kv_buffer, loc, cache_k_nope, cache_k_rope
                )
            else:
                self.prefetch_layersplit_kv_buffer(layer_id, active_rows=active_rows)
            return
        has_prefetched_kv = self._layersplit_kv_prefetch_layer_id == layer_id
        if has_prefetched_kv:
            self._finish_layersplit_kv_prefetch_for_update(layer_id)

        self._set_mla_kv_buffer_to_buffer(
            self.kv_buffer[layer_idx], loc, cache_k_nope, cache_k_rope
        )
        if not has_prefetched_kv:
            self.prefetch_layersplit_kv_buffer(layer_id, active_rows=active_rows)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        # get k nope and k rope from the kv buffer, and optionally cast them to dst_dtype.
        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope

    def get_cpu_copy(self, indices, mamba_indices=None):
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        current_platform.synchronize()


class MLATokenToKVPoolFP4(MLATokenToKVPool):

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = 1  # head_num
                k = self.kv_cache_dim  # head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8

                self.kv_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.kv_scale_buffer = [
                    torch.zeros(
                        (m, k // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer
        del self.kv_scale_buffer

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.kv_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.kv_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant

        return self.kv_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        assert not self.dsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_fp4, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k_fp4.view(
                self.store_dtype
            )
            self.kv_scale_buffer[layer_id - self.start_layer][loc] = (
                cache_k_fp4_sf.view(self.store_dtype)
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id

        if self.dsa_kv_cache_store_fp8:
            # original cache_k: (num_tokens, num_heads 1, hidden 576); we unsqueeze the page_size=1 dim here
            # TODO no need to cat
            cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
            cache_k = cache_k.view(self.store_dtype)
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k
        else:
            if cache_k_nope.dtype != self.dtype:
                from sglang.srt.layers.quantization.kvfp4_tensor import (
                    KVFP4QuantizeUtil,
                )

                cache_k_nope_fp4, cache_k_nope_fp4_sf = (
                    KVFP4QuantizeUtil.batched_quantize(cache_k_nope)
                )
                cache_k_rope_fp4, cache_k_rope_fp4_sf = (
                    KVFP4QuantizeUtil.batched_quantize(cache_k_rope)
                )

            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4,
                cache_k_rope_fp4,
            )
            set_mla_kv_scale_buffer_triton(
                self.kv_scale_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4_sf,
                cache_k_rope_fp4_sf,
            )


class DSATokenToKVPool(MLATokenToKVPool):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8
    rope_storage_dtype = torch.bfloat16  # rope is always stored in bf16

    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_buf_size: Optional[int] = None,
        indexer_quantization: str = INDEXER_FP8_QUANT_METHOD,
    ):

        override_dim = (
            kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
        )

        super().__init__(
            size,
            page_size,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            use_dsa=True,
            override_kv_cache_dim=override_dim,
        )
        # self.index_k_dtype = torch.float8_e4m3fn
        # self.index_k_scale_dtype = torch.float32
        self.index_head_dim = index_head_dim
        if index_buf_size is None:
            index_buf_size = size
        # num head == 1 and head dim == 128 for index_k in DSA
        assert index_head_dim == 128
        self.indexer_cache_layout = get_dsa_indexer_cache_layout(
            indexer_quantization, index_head_dim
        )
        self.indexer_quantization = self.indexer_cache_layout.quant_method
        self.quant_block_size = self.indexer_cache_layout.quant_block_size

        if _is_hip:
            if aiter_can_use_preshuffle_paged_mqa():
                assert (
                    self.page_size % 16 == 0
                ), f"HIP preshuffle requires page_size to be a multiple of 16, got {self.page_size}"
            else:
                assert (
                    self.page_size == 1
                ), f"HIP legacy DSA path requires page_size == 1, got {self.page_size}"
        else:
            assert self.page_size == 64
        index_rows = (index_buf_size + page_size + 1) // self.page_size
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self.index_k_with_scale_buffer = []
            for local_layer_idx in range(layer_num):
                layer_id = self.start_layer + local_layer_idx
                rows = index_rows if self.layersplit_owns_layer(layer_id) else 1
                self.index_k_with_scale_buffer.append(
                    torch.zeros(
                        (rows, self.indexer_cache_layout.page_bytes(self.page_size)),
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=device,
                    )
                )
            self.layersplit_index_k_with_scale_buffer = None
            if self.layersplit_policy is not None:
                self.layersplit_index_k_with_scale_buffer = torch.zeros(
                    (
                        index_rows,
                        self.indexer_cache_layout.page_bytes(self.page_size),
                    ),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=device,
                )
        self._layersplit_index_prefetch_layer_id = None
        self._layersplit_index_prefetch_buffer = None
        self._layersplit_index_prefetch_work = None
        self._finalize_allocation_log(size)

    def _layersplit_index_active_rows(
        self, active_rows: Optional[int]
    ) -> Optional[int]:
        if active_rows is None:
            return None
        assert self.layersplit_index_k_with_scale_buffer is not None
        return min(
            triton.cdiv(active_rows, self.page_size) + 1,
            self.layersplit_index_k_with_scale_buffer.shape[0],
        )

    def _ensure_layersplit_decode_index_buffer(self, layer_idx: int) -> None:
        assert self.layersplit_index_k_with_scale_buffer is not None
        if (
            self.index_k_with_scale_buffer[layer_idx].shape[0]
            == self.layersplit_index_k_with_scale_buffer.shape[0]
        ):
            return
        self.index_k_with_scale_buffer[layer_idx] = torch.zeros_like(
            self.layersplit_index_k_with_scale_buffer
        )

    def materialize_layersplit_for_decode(
        self, active_rows: Optional[int] = None
    ) -> None:
        policy = self.layersplit_policy
        if policy is None or self._layersplit_decode_materialized:
            return
        if active_rows is None:
            active_rows = self._layersplit_active_rows
        self._wait_layersplit_index_prefetch()
        super().materialize_layersplit_for_decode(active_rows)
        assert self.layersplit_index_k_with_scale_buffer is not None
        index_rows = self._layersplit_index_active_rows(active_rows)

        from sglang.srt.layers.dp_attention import get_attention_cp_group

        for layer_id in range(self.start_layer, self.start_layer + self.layer_num):
            layer_idx = layer_id - self.start_layer
            if self.layer_transfer_counter is not None:
                self.layer_transfer_counter.wait_until(layer_idx)
            owner_rank = policy.owner_rank(layer_id)
            if owner_rank != policy.cp_rank:
                self._ensure_layersplit_decode_index_buffer(layer_idx)
            index_buffer = self.index_k_with_scale_buffer[layer_idx]
            broadcast_buffer = (
                index_buffer
                if index_rows is None
                else index_buffer[:index_rows]
            )
            get_attention_cp_group().broadcast(broadcast_buffer, src=owner_rank)

    def _wait_layersplit_index_prefetch(self) -> Optional[torch.Tensor]:
        work = self._layersplit_index_prefetch_work
        if work is not None:
            work.wait()
        buffer = self._layersplit_index_prefetch_buffer
        self._layersplit_index_prefetch_layer_id = None
        self._layersplit_index_prefetch_buffer = None
        self._layersplit_index_prefetch_work = None
        return buffer

    def prefetch_layersplit_index_k_with_scale_buffer(self, layer_id: int) -> None:
        policy = self.layersplit_policy
        if policy is None or self._layersplit_decode_materialized:
            return
        if self._layersplit_index_prefetch_layer_id == layer_id:
            return
        if self._layersplit_index_prefetch_layer_id is not None:
            self._wait_layersplit_index_prefetch()

        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        layer_idx = layer_id - self.start_layer
        owner_rank = policy.owner_rank(layer_id)
        if owner_rank == policy.cp_rank:
            index_k_buffer = self.index_k_with_scale_buffer[layer_idx]
        else:
            index_k_buffer = self.layersplit_index_k_with_scale_buffer

        self._layersplit_index_prefetch_layer_id = layer_id
        self._layersplit_index_prefetch_buffer = index_k_buffer
        self._layersplit_index_prefetch_work = self._layersplit_broadcast_async(
            index_k_buffer, owner_rank
        )

    def _get_layersplit_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        layer_idx = layer_id - self.start_layer
        policy = self.layersplit_policy
        if policy is None or self._layersplit_decode_materialized:
            return self.index_k_with_scale_buffer[layer_idx]
        if self._layersplit_index_prefetch_layer_id == layer_id:
            index_k_buffer = self._wait_layersplit_index_prefetch()
            assert index_k_buffer is not None
            return index_k_buffer
        owner_rank = policy.owner_rank(layer_id)
        if owner_rank == policy.cp_rank:
            index_k_buffer = self.index_k_with_scale_buffer[layer_idx]
        else:
            index_k_buffer = self.layersplit_index_k_with_scale_buffer

        from sglang.srt.layers.dp_attention import get_attention_cp_group

        return get_attention_cp_group().broadcast(index_k_buffer, src=owner_rank)

    def _clear_buffers(self):
        del self.kv_buffer
        del self.index_k_with_scale_buffer

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_layersplit_index_k_with_scale_buffer(layer_id)

    def get_local_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if not self.layersplit_owns_layer(layer_id):
            raise ValueError(f"LayerSplit rank does not own layer {layer_id}.")
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self._get_layersplit_index_k_with_scale_buffer(layer_id)
        return index_buf_accessor.GetK.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self._get_layersplit_index_k_with_scale_buffer(layer_id)
        return index_buf_accessor.GetS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_and_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self._get_layersplit_index_k_with_scale_buffer(layer_id)
        return (
            index_buf_accessor.GetK.execute(
                self, buf, seq_len=seq_len, page_indices=page_indices
            ),
            index_buf_accessor.GetS.execute(
                self, buf, seq_len=seq_len, page_indices=page_indices
            ),
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        """
        Fused method to get both index K and scale data in a single call using Triton.
        More efficient than calling get_index_k_continuous and get_index_k_scale_continuous separately.

        :param layer_id: Layer index
        :param seq_len: Sequence length
        :param page_indices: Page indices tensor
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self._get_layersplit_index_k_with_scale_buffer(layer_id)
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        if not self.layersplit_owns_layer(layer_id):
            self.prefetch_layersplit_index_k_with_scale_buffer(layer_id)
            return
        if self._layersplit_index_prefetch_layer_id == layer_id:
            self._wait_layersplit_index_prefetch()
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )
        if not self._layersplit_decode_materialized:
            self.prefetch_layersplit_index_k_with_scale_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        kv_buffer = self._get_layersplit_kv_buffer(layer_id)
        if self.store_dtype != self.dtype:
            kv_buffer = kv_buffer.view(self.dtype)
        return kv_buffer, kv_buffer[..., : self.kv_lora_rank]

    def get_cpu_copy(self, indices):
        # DSA keeps a page-indexed index_k_with_scale_buffer alongside kv_buffer.
        # Retract frees the slots/pages and they get reused by other reqs'
        # set_index_k_scale_buffer, so we must offload it here too -- otherwise
        # resume restores kv_buffer but leaves foreign index/scale in place and
        # DSA attention reads garbage at those token positions.
        kv_cache_cpu = super().get_cpu_copy(indices)

        page_indices = indices[:: self.page_size] // self.page_size
        torch.cuda.synchronize()
        index_k_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.page_size)
        for layer_id in range(self.layer_num):
            index_k_cpu.append([])
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = self.index_k_with_scale_buffer[layer_id][
                    chunk_page_indices
                ].to("cpu", non_blocking=True)
                index_k_cpu[-1].append(idx_cpu)
        torch.cuda.synchronize()

        return {"kv": kv_cache_cpu, "index_k": index_k_cpu}

    def load_cpu_copy(self, kv_cache_cpu_dict, indices):
        super().load_cpu_copy(kv_cache_cpu_dict["kv"], indices)

        page_indices = indices[:: self.page_size] // self.page_size
        index_k_cpu = kv_cache_cpu_dict["index_k"]
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.page_size)
        for layer_id in range(self.layer_num):
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = index_k_cpu[layer_id][i // page_chunk_size]
                assert idx_cpu.shape[0] == len(chunk_page_indices)
                idx_chunk = idx_cpu.to(
                    self.index_k_with_scale_buffer[0].device, non_blocking=True
                )
                self.index_k_with_scale_buffer[layer_id][chunk_page_indices] = idx_chunk
        torch.cuda.synchronize()

    def get_state_buf_infos(self):
        data_ptrs = [
            self.index_k_with_scale_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        data_lens = [
            self.index_k_with_scale_buffer[i].nbytes for i in range(self.layer_num)
        ]
        item_lens = [
            self.index_k_with_scale_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        kv_size_bytes = super().get_kv_size_bytes()
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        if self.layersplit_index_k_with_scale_buffer is not None:
            kv_size_bytes += get_tensor_size_bytes(
                self.layersplit_index_k_with_scale_buffer
            )
        return kv_size_bytes


class TurboQuantDSATokenToKVPool(DSATokenToKVPool):
    """DSA KV pool with compressed dense MLA storage.

    PD disaggregation transfers raw contiguous pool buffers. Like the NVFP4 MLA
    pool, TurboQuant keeps the dense KV bytes in ``kv_buffer`` and exposes DSA
    indexer state separately through ``get_state_buf_infos`` inherited from
    ``DSATokenToKVPool``. This lets prefill send compressed dense KV bytes plus
    indexer state to decode without materializing BF16 dense KV during transfer.
    """

    def __init__(
        self,
        *args,
        turboquant_dense_kv_preset: str,
        turboquant_execution_mode: str,
        turboquant_mla_decode_num_splits: int = 16,
        turboquant_skip_layers: Optional[set[int]] = None,
        **kwargs,
    ):
        self.turboquant_dense_kv_preset = turboquant_dense_kv_preset
        self.turboquant_execution_mode = turboquant_execution_mode
        self.turboquant_mla_decode_num_splits = int(
            turboquant_mla_decode_num_splits
        )
        self.turboquant_skip_layers = turboquant_skip_layers or set()
        super().__init__(*args, **kwargs)

    def _create_buffers(self):
        self.store_dtype = torch.uint8
        self.turboquant_codec = TurboQuantDenseKVCodec(
            TurboQuantDenseKVConfig(
                latent_dim=self.kv_lora_rank,
                rope_dim=self.qk_rope_head_dim,
                preset=self.turboquant_dense_kv_preset,
            ),
            torch.device(self.device),
        )
        self.turboquant_slot_bytes = self.turboquant_codec.slot_bytes
        dtype_bytes = torch.tensor([], dtype=self.dtype).element_size()
        self.turboquant_flashmla_fp8_slot_bytes = (
            self.kv_lora_rank
            + (self.kv_lora_rank // 128) * 4
            + self.qk_rope_head_dim * dtype_bytes
        )

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                m = self.size + self.page_size
                self.kv_buffer = []
                for local_layer_idx in range(self.layer_num):
                    layer_id = self.start_layer + local_layer_idx
                    rows = (
                        m
                        if self.layersplit_owns_layer(layer_id)
                        else self.page_size
                    )
                    if layer_id in self.turboquant_skip_layers:
                        self.kv_buffer.append(
                            torch.zeros(
                                (rows, 1, self.kv_cache_dim),
                                dtype=self.dtype,
                                device=self.device,
                            )
                        )
                    else:
                        self.kv_buffer.append(
                            torch.zeros(
                                (rows, 1, self.turboquant_slot_bytes),
                                dtype=torch.uint8,
                                device=self.device,
                            )
                        )
                if self.layersplit_policy is not None:
                    self.layersplit_kv_buffer = torch.zeros(
                        (m, 1, self.turboquant_slot_bytes),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                self._deq_buffer = (
                    torch.zeros(
                        (m, 1, self.kv_cache_dim),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    if self.turboquant_execution_mode == "materialize"
                    else None
                )

        self._tq_active = [0] * self.layer_num
        self._tq_dirty = [True] * self.layer_num
        self._tq_deq_layer_idx: Optional[int] = None
        self._tq_selected_buffer: Optional[torch.Tensor] = None
        self._tq_selected_fp8_buffer: Optional[torch.Tensor] = None
        self._tq_compact_page_table: Optional[torch.Tensor] = None
        self._tq_full_page_table: Optional[torch.Tensor] = None
        self._tq_full_compact_page_table: Optional[torch.Tensor] = None
        self._tq_full_page_table_filled = 0
        self._tq_mla_decode_mid: Optional[torch.Tensor] = None
        self._tq_mla_q_rotated: Optional[torch.Tensor] = None
        fp16_bytes = (self.kv_lora_rank + self.qk_rope_head_dim) * dtype_bytes
        logger.info(
            "TurboQuant dense MLA KV cache enabled: preset=%s, %d bytes/token "
            "(baseline dense=%d bytes/token)",
            self.turboquant_dense_kv_preset,
            self.turboquant_slot_bytes,
            fp16_bytes,
        )

    def _clear_buffers(self):
        self._wait_layersplit_kv_prefetch()
        del self.kv_buffer
        if self.layersplit_kv_buffer is not None:
            del self.layersplit_kv_buffer
        if self._deq_buffer is not None:
            del self._deq_buffer

    def _uses_turboquant_layer(self, layer_id: int) -> bool:
        return layer_id not in self.turboquant_skip_layers

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        layer_idx = layer_id - self.start_layer
        if not self._uses_turboquant_layer(layer_id):
            return self._get_layersplit_kv_buffer(layer_id)

        if self._deq_buffer is None:
            raise RuntimeError(
                "TurboQuant fused_decode does not materialize the full dense KV cache."
            )
        if self._tq_dirty[layer_idx] or self._tq_deq_layer_idx != layer_idx:
            n = self._tq_active[layer_idx]
            if n > 0:
                self._deq_buffer[:n] = self.turboquant_codec.decompress(
                    self._get_layersplit_kv_buffer(layer_id)[:n],
                    self.dtype,
                )
            self._tq_dirty[layer_idx] = False
            self._tq_deq_layer_idx = layer_idx
        return self._deq_buffer

    def get_value_buffer(self, layer_id: int):
        key_buf = self.get_key_buffer(layer_id)
        return key_buf[..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        key_buf = self.get_key_buffer(layer_id)
        return key_buf, key_buf[..., : self.kv_lora_rank]

    def get_turboquant_selected_kv_buffer(
        self,
        layer_id: int,
        page_table: torch.Tensor,
        fp8_layout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_idx = layer_id - self.start_layer
        if not self._uses_turboquant_layer(layer_id):
            layer_buffer = self._get_layersplit_kv_buffer(layer_id)
            return layer_buffer, page_table
        layer_buffer = self._get_layersplit_kv_buffer(layer_id)

        if (
            fp8_layout
            and self.turboquant_codec.bits == 2.5
            and self.dtype == torch.bfloat16
            and page_table.is_cuda
            and page_table.dtype == torch.int32
            and page_table.is_contiguous()
        ):
            active_rows = 0
            full_rows = 0
            if page_table.shape[0] > 8:
                active_rows = self._tq_active[layer_idx]
                if active_rows == 0:
                    active_rows = int(page_table.max().item()) + 1
                full_rows = triton.cdiv(active_rows, self.page_size) * self.page_size
            if full_rows > 0 and full_rows * 4 < page_table.numel():
                if (
                    self._tq_selected_fp8_buffer is None
                    or self._tq_selected_fp8_buffer.shape[0] < full_rows
                ):
                    self._tq_selected_fp8_buffer = torch.empty(
                        (
                            full_rows,
                            1,
                            self.turboquant_flashmla_fp8_slot_bytes,
                        ),
                        dtype=torch.float8_e4m3fn,
                        device=self.device,
                    )
                if (
                    self._tq_full_page_table is None
                    or self._tq_full_page_table.numel() < full_rows
                ):
                    self._tq_full_page_table = torch.empty(
                        (1, full_rows), dtype=torch.int32, device=self.device
                    )
                    self._tq_full_compact_page_table = torch.empty_like(
                        self._tq_full_page_table
                    )
                    self._tq_full_page_table_filled = 0
                full_page_table = self._tq_full_page_table[:, :full_rows]
                if self._tq_full_page_table_filled < full_rows:
                    torch.arange(
                        self._tq_full_page_table_filled,
                        full_rows,
                        dtype=torch.int32,
                        device=self.device,
                        out=self._tq_full_page_table[
                            0, self._tq_full_page_table_filled : full_rows
                        ],
                    )
                    self._tq_full_page_table_filled = full_rows
                full_compact_page_table = self._tq_full_compact_page_table[
                    :, :full_rows
                ]
                kv_cache = self._tq_selected_fp8_buffer[:full_rows]
                dequantize_page_table_selected_2p5_fp8(
                    layer_buffer,
                    full_page_table,
                    kv_cache.view(torch.uint8),
                    full_compact_page_table,
                    self.turboquant_codec.centroids_high,
                    self.turboquant_codec.centroids_low,
                    self.turboquant_codec.signs1,
                    self.turboquant_codec.signs2,
                )
                return kv_cache, page_table
            if (
                self._tq_selected_fp8_buffer is None
                or self._tq_selected_fp8_buffer.shape[0] < page_table.numel()
            ):
                self._tq_selected_fp8_buffer = torch.empty(
                    (
                        page_table.numel(),
                        1,
                        self.turboquant_flashmla_fp8_slot_bytes,
                    ),
                    dtype=torch.float8_e4m3fn,
                    device=self.device,
                )
            if (
                self._tq_compact_page_table is None
                or self._tq_compact_page_table.shape != page_table.shape
            ):
                self._tq_compact_page_table = torch.empty_like(page_table)
            kv_cache = self._tq_selected_fp8_buffer[: page_table.numel()]
            fp8_dequant_fn = (
                dequantize_page_table_selected_2p5_fp8_reuse
                if page_table.shape[0] <= 8
                else dequantize_page_table_selected_2p5_fp8
            )
            fp8_dequant_fn(
                layer_buffer,
                page_table,
                kv_cache.view(torch.uint8),
                self._tq_compact_page_table,
                self.turboquant_codec.centroids_high,
                self.turboquant_codec.centroids_low,
                self.turboquant_codec.signs1,
                self.turboquant_codec.signs2,
            )
            return kv_cache, self._tq_compact_page_table

        if (
            self.turboquant_codec.bits == 2.5
            and self.dtype == torch.bfloat16
            and page_table.is_cuda
            and page_table.dtype == torch.int32
            and page_table.is_contiguous()
        ):
            if (
                self._tq_selected_buffer is None
                or self._tq_selected_buffer.shape[0] < page_table.numel()
            ):
                self._tq_selected_buffer = torch.empty(
                    (page_table.numel(), 1, self.kv_cache_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
            if (
                self._tq_compact_page_table is None
                or self._tq_compact_page_table.shape != page_table.shape
            ):
                self._tq_compact_page_table = torch.empty_like(page_table)
            kv_cache = self._tq_selected_buffer[: page_table.numel()]
            dequantize_page_table_selected_2p5(
                layer_buffer,
                page_table,
                kv_cache,
                self._tq_compact_page_table,
                self.turboquant_codec.centroids_high,
                self.turboquant_codec.centroids_low,
                self.turboquant_codec.signs1,
                self.turboquant_codec.signs2,
            )
            return kv_cache, self._tq_compact_page_table

        mask = page_table >= 0
        flat_loc = page_table.clamp_min(0).reshape(-1).long()
        if (
            self.turboquant_codec.bits == 4
            and self.dtype == torch.bfloat16
            and flat_loc.is_cuda
        ):
            if (
                self._tq_selected_buffer is None
                or self._tq_selected_buffer.shape[0] < flat_loc.numel()
            ):
                self._tq_selected_buffer = torch.empty(
                    (flat_loc.numel(), 1, self.kv_cache_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
            kv_cache = self._tq_selected_buffer[: flat_loc.numel()]
            dequantize_selected_4bit(
                layer_buffer,
                flat_loc,
                kv_cache,
                self.turboquant_codec.centroids,
                self.turboquant_codec.signs1,
                self.turboquant_codec.signs2,
            )
        elif (
            self.turboquant_codec.bits == 2.5
            and self.dtype == torch.bfloat16
            and flat_loc.is_cuda
        ):
            if (
                self._tq_selected_buffer is None
                or self._tq_selected_buffer.shape[0] < flat_loc.numel()
            ):
                self._tq_selected_buffer = torch.empty(
                    (flat_loc.numel(), 1, self.kv_cache_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
            kv_cache = self._tq_selected_buffer[: flat_loc.numel()]
            dequantize_selected_2p5(
                layer_buffer,
                flat_loc,
                kv_cache,
                self.turboquant_codec.centroids_high,
                self.turboquant_codec.centroids_low,
                self.turboquant_codec.signs1,
                self.turboquant_codec.signs2,
            )
        else:
            kv_cache = self.turboquant_codec.decompress(
                layer_buffer[flat_loc],
                self.dtype,
            )
        compact_page_table = torch.arange(
            page_table.numel(),
            dtype=page_table.dtype,
            device=page_table.device,
        ).reshape(page_table.shape)
        compact_page_table = compact_page_table.masked_fill(~mask, -1)
        return kv_cache, compact_page_table

    def forward_turboquant_dense_mla_decode(
        self,
        layer_id: int,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        page_table: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        layer_idx = layer_id - self.start_layer
        layer_buffer = self._get_layersplit_kv_buffer(layer_id)
        out = torch.empty(
            (q_nope.shape[0], q_nope.shape[1], self.kv_lora_rank),
            dtype=self.dtype,
            device=q_nope.device,
        )
        num_splits = self.turboquant_mla_decode_num_splits
        mid_shape = (
            q_nope.shape[0],
            q_nope.shape[1],
            num_splits,
            self.kv_lora_rank + 2,
        )
        if (
            self._tq_mla_decode_mid is None
            or self._tq_mla_decode_mid.shape != mid_shape
        ):
            self._tq_mla_decode_mid = torch.empty(
                mid_shape,
                dtype=torch.float32,
                device=q_nope.device,
            )
        q_rotated_shape = q_nope.shape
        if (
            self._tq_mla_q_rotated is None
            or self._tq_mla_q_rotated.shape != q_rotated_shape
        ):
            self._tq_mla_q_rotated = torch.empty(
                q_rotated_shape,
                dtype=torch.float32,
                device=q_nope.device,
            )
        turboquant_dense_mla_rotate_query(
            q_nope,
            self._tq_mla_q_rotated,
            self.turboquant_codec.signs1,
            self.turboquant_codec.signs2,
        )
        turboquant_dense_mla_decode_2p5_split_rotated(
            self._tq_mla_q_rotated,
            q_rope,
            layer_buffer,
            page_table,
            self._tq_mla_decode_mid,
            out,
            self.turboquant_codec.centroids_high,
            self.turboquant_codec.centroids_low,
            self.turboquant_codec.signs1,
            self.turboquant_codec.signs2,
            sm_scale,
        )
        return out

    def get_kv_size_bytes(self):
        kv_size_bytes = super().get_kv_size_bytes()
        if self.turboquant_execution_mode == "materialize":
            kv_size_bytes += get_tensor_size_bytes(self._deq_buffer)
        return kv_size_bytes

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        self.set_mla_kv_buffer(
            layer,
            loc,
            cache_k[..., : self.kv_lora_rank],
            cache_k[..., self.kv_lora_rank :],
        )

    def _set_turboquant_mla_kv_buffer_to_buffer(
        self,
        kv_buffer: torch.Tensor,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        if (
            self.turboquant_codec.bits == 2.5
            and self.dtype == torch.bfloat16
            and loc.is_cuda
            and loc.dtype == torch.int64
            and cache_k_nope.is_cuda
            and cache_k_rope.is_cuda
            and cache_k_nope.dtype == torch.bfloat16
            and cache_k_rope.dtype == torch.bfloat16
        ):
            if cache_k_nope.dim() == 2:
                cache_k_nope = cache_k_nope.unsqueeze(1)
            if cache_k_rope.dim() == 2:
                cache_k_rope = cache_k_rope.unsqueeze(1)
            store_2p5(
                kv_buffer,
                loc,
                cache_k_nope,
                cache_k_rope,
                self.turboquant_codec.boundaries_high,
                self.turboquant_codec.boundaries_low,
                self.turboquant_codec.centroids_high,
                self.turboquant_codec.centroids_low,
                self.turboquant_codec.signs1,
                self.turboquant_codec.signs2,
            )
        else:
            kv_buffer[loc] = self.turboquant_codec.compress(
                cache_k_nope,
                cache_k_rope,
            )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id
        layer_idx = layer_id - self.start_layer
        active_rows = self._layersplit_active_rows_from_loc(loc)
        if not self.layersplit_owns_layer(layer_id):
            kv_buffer = self._finish_layersplit_kv_prefetch_for_update(layer_id)
            if kv_buffer is not None:
                if self._uses_turboquant_layer(layer_id):
                    self._set_turboquant_mla_kv_buffer_to_buffer(
                        kv_buffer, loc, cache_k_nope, cache_k_rope
                    )
                else:
                    self._set_mla_kv_buffer_to_buffer(
                        kv_buffer, loc, cache_k_nope, cache_k_rope
                    )
            else:
                self.prefetch_layersplit_kv_buffer(layer_id, active_rows=active_rows)
            return

        if not self._uses_turboquant_layer(layer_id):
            super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)
            return

        has_prefetched_kv = self._layersplit_kv_prefetch_layer_id == layer_id
        if has_prefetched_kv:
            self._finish_layersplit_kv_prefetch_for_update(layer_id)
        self._set_turboquant_mla_kv_buffer_to_buffer(
            self.kv_buffer[layer_idx], loc, cache_k_nope, cache_k_rope
        )
        self._tq_dirty[layer_idx] = True
        if self.turboquant_execution_mode == "materialize" and loc.numel() > 0:
            self._tq_active[layer_idx] = max(
                self._tq_active[layer_idx],
                int(loc.max().item()) + 1,
            )
        if not has_prefetched_kv:
            self.prefetch_layersplit_kv_buffer(layer_id, active_rows=active_rows)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        layer_id = layer.layer_id
        if not self._uses_turboquant_layer(layer_id):
            return super().get_mla_kv_buffer(layer, loc, dst_dtype)

        dst_dtype = dst_dtype or self.dtype
        kv = self.turboquant_codec.decompress(
            self._get_layersplit_kv_buffer(layer_id)[loc], dst_dtype
        )
        return kv[..., : self.kv_lora_rank], kv[..., self.kv_lora_rank :]


class HiggsDense2BitDSATokenToKVPool(DSATokenToKVPool):
    """DSA KV pool with 2-bit HIGGS compressed dense MLA storage.

    Mirrors :class:`TurboQuantDSATokenToKVPool` 1:1 but stores 258 B/token/layer
    instead of TurboQuant's 274 B/token/layer (-5.84%) using EDEN2-16 lattice
    quantization (Pletka et al. arXiv 2501.19392 + AquaKV reference).

    Two-attribute compatibility shim with the TurboQuant dispatch:
      * ``turboquant_execution_mode`` mirrors as ``"fused_decode"`` so the
        existing decode-dispatch gate at ``dsa_backend.py`` recognizes this
        pool as a compressed-dense pool through a single field, while the
        ``higgs_execution_mode`` field provides the dedicated HIGGS view.
    """

    def __init__(
        self,
        *args,
        higgs_execution_mode: str = "fused_decode",
        higgs_skip_layers: Optional[set[int]] = None,
        higgs_mla_decode_num_splits: int = 0,
        **kwargs,
    ):
        self.higgs_execution_mode = higgs_execution_mode
        # Compatibility alias: backends gate on `turboquant_execution_mode`
        # to decide "this pool stores compressed-dense KV"; we surface the
        # same field so any future backend that checks the alias works
        # uniformly. The actual dispatch in `dsa_backend.py` also checks
        # the pool type / `higgs_dense_2bit_preset` to disambiguate paths.
        self.turboquant_execution_mode = higgs_execution_mode
        self.higgs_dense_2bit_preset = "eden2_16"
        self.higgs_skip_layers = higgs_skip_layers or set()
        self.higgs_mla_decode_num_splits = int(higgs_mla_decode_num_splits)
        if self.higgs_mla_decode_num_splits < 0:
            raise ValueError(
                "higgs_mla_decode_num_splits must be >= 0; "
                f"got {self.higgs_mla_decode_num_splits}."
            )
        # Lazily-allocated scratch buffers for the split-K decode path.
        # Shapes depend on (batch, heads, num_splits) and the BF16 query;
        # we allocate on first call and only reallocate when shapes
        # change. Matches TurboQuant's pattern at line 2973 above.
        self._higgs_mla_decode_mid: Optional[torch.Tensor] = None
        self._higgs_mla_q_rotated: Optional[torch.Tensor] = None
        super().__init__(*args, **kwargs)

    def _create_buffers(self):
        self.store_dtype = torch.uint8
        self.higgs_codec = HiggsDense2BitCodec(
            HiggsDense2BitConfig(
                latent_dim=self.kv_lora_rank,
                rope_dim=self.qk_rope_head_dim,
            ),
            torch.device(self.device),
        )
        self.higgs_slot_bytes = self.higgs_codec.slot_bytes
        dtype_bytes = torch.tensor([], dtype=self.dtype).element_size()

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                m = self.size + self.page_size
                self.kv_buffer = []
                for local_layer_idx in range(self.layer_num):
                    layer_id = self.start_layer + local_layer_idx
                    rows = (
                        m
                        if self.layersplit_owns_layer(layer_id)
                        else self.page_size
                    )
                    if layer_id in self.higgs_skip_layers:
                        self.kv_buffer.append(
                            torch.zeros(
                                (rows, 1, self.kv_cache_dim),
                                dtype=self.dtype,
                                device=self.device,
                            )
                        )
                    else:
                        self.kv_buffer.append(
                            torch.zeros(
                                (rows, 1, self.higgs_slot_bytes),
                                dtype=torch.uint8,
                                device=self.device,
                            )
                        )
                if self.layersplit_policy is not None:
                    self.layersplit_kv_buffer = torch.zeros(
                        (m, 1, self.higgs_slot_bytes),
                        dtype=torch.uint8,
                        device=self.device,
                    )

        self._higgs_selected_buffer: Optional[torch.Tensor] = None
        # ai-blaise #19 iter3: separate FP8 scratch for the trtllm-gen FP8
        # sparse-MLA cubin path (Vector A). Reused across MoE layers like
        # the BF16 sibling buffer above.
        self._higgs_selected_buffer_fp8: Optional[torch.Tensor] = None
        self._higgs_compact_page_table: Optional[torch.Tensor] = None
        # ai-blaise #19 iter5 (primary vector): ping-pong slots for the
        # compact dequant scratch and the compact_page_table. With a
        # single slot, layer N+1's side-stream dequant cannot overlap
        # with layer N's trtllm-gen sparse-MLA read (write-after-read
        # hazard on the shared buffer). Two slots keyed on
        # ``layer_id & 1`` mean adjacent layers touch disjoint scratch
        # so the side-stream dequant can run concurrently with the
        # prior-layer attention kernel on the main stream.
        #
        # Allocation policy: lazy + per-slot, in
        # :meth:`get_higgs_selected_kv_buffer`. The slots share dtype +
        # shape semantics with the legacy single-slot attrs above; on
        # the ``slot_parity == 0`` path we keep the legacy attrs alive
        # for back-compat (no behavior change when the iter5 env flag
        # is off). When the env flag is on, both slots [0] and [1] are
        # populated lazily on first access.
        self._higgs_selected_buffer_pp: list[Optional[torch.Tensor]] = [None, None]
        self._higgs_selected_buffer_fp8_pp: list[Optional[torch.Tensor]] = [
            None,
            None,
        ]
        self._higgs_compact_page_table_pp: list[Optional[torch.Tensor]] = [
            None,
            None,
        ]
        fp16_bytes = (self.kv_lora_rank + self.qk_rope_head_dim) * dtype_bytes
        logger.info(
            "HIGGS dense 2-bit MLA KV cache enabled: %d bytes/token "
            "(baseline dense=%d bytes/token, savings=%.2f%%)",
            self.higgs_slot_bytes,
            fp16_bytes,
            100.0 * (fp16_bytes - self.higgs_slot_bytes) / fp16_bytes,
        )

    def _clear_buffers(self):
        self._wait_layersplit_kv_prefetch()
        del self.kv_buffer
        if self.layersplit_kv_buffer is not None:
            del self.layersplit_kv_buffer

    def _uses_higgs_layer(self, layer_id: int) -> bool:
        return layer_id not in self.higgs_skip_layers

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if not self._uses_higgs_layer(layer_id):
            return self._get_layersplit_kv_buffer(layer_id)

        # HIGGS fused_decode does NOT materialize the full dense KV cache.
        # Callers that need the dense view should go through
        # ``get_mla_kv_buffer`` (single-loc decompress) or
        # ``get_higgs_selected_kv_buffer`` (page-table decompress).
        raise RuntimeError(
            "HIGGS dense 2-bit fused_decode does not materialize the full "
            "dense KV cache; use get_higgs_selected_kv_buffer or "
            "get_mla_kv_buffer."
        )

    def get_value_buffer(self, layer_id: int):
        key_buf = self.get_key_buffer(layer_id)
        return key_buf[..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        key_buf = self.get_key_buffer(layer_id)
        return key_buf, key_buf[..., : self.kv_lora_rank]

    def get_higgs_selected_kv_buffer(
        self,
        layer_id: int,
        page_table: torch.Tensor,
        fp8_layout: bool = False,
        fp8_inv_kv_scale: float = 1.0,
        slot_parity: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize the latent+rope view for a page_table-indexed selection.

        Returns ``(kv_cache, compact_page_table)``. Output dtype:
          * ``fp8_layout=False`` (default): BF16, width ``kv_cache_dim``.
          * ``fp8_layout=True``: ``torch.float8_e4m3fn``, width
            ``kv_cache_dim``. ai-blaise #19 iter3 vector A path — feeds the
            trtllm-gen sparse-MLA ``QkvE4m3OBfloat16`` cubin set with a
            576 B/row buffer (vs 1152 B/row BF16, -50% HBM traffic). The
            kernel applies ``fp8_inv_kv_scale`` as a per-tensor multiplier
            before the saturating FP8 cast; downstream attention should
            pass ``k_scale = 1 / fp8_inv_kv_scale`` via ``bmm1_scale``.

        ai-blaise #19 iter5 (primary vector) ``slot_parity``: selects
        between two ping-pong scratch slots (0 or 1) so adjacent layers
        write/read disjoint scratch and the side-stream dequant in
        ``_forward_trtllm`` can overlap with the prior layer's
        trtllm-gen kernel without aliasing. Default 0 preserves the
        legacy single-slot behavior bit-for-bit when callers don't pass
        a parity; the layer-driven parity comes from the call site in
        :class:`DeepseekSparseAttnBackend._forward_trtllm` and is
        guarded by ``envs.SGLANG_HIGGS_DSA_TRTLLM_PINGPONG``.
        """
        if not self._uses_higgs_layer(layer_id):
            layer_buffer = self._get_layersplit_kv_buffer(layer_id)
            return layer_buffer, page_table
        layer_buffer = self._get_layersplit_kv_buffer(layer_id)

        if (
            self.dtype == torch.bfloat16
            and page_table.is_cuda
            and page_table.dtype == torch.int32
            and page_table.is_contiguous()
        ):
            # ai-blaise #19 iter5: select the ping-pong slot. ``parity=0``
            # callers (legacy) keep using the original ``_higgs_*``
            # singleton attrs so the iter3/iter4 paths are bit-for-bit
            # unchanged when ``SGLANG_HIGGS_DSA_TRTLLM_PINGPONG`` is off.
            # ``parity=1`` always routes to the ``_pp[1]`` slot.
            parity = int(slot_parity) & 1
            if parity == 0:
                compact_pt = self._higgs_compact_page_table
                if (
                    compact_pt is None
                    or compact_pt.shape != page_table.shape
                ):
                    compact_pt = torch.empty_like(page_table)
                    self._higgs_compact_page_table = compact_pt
            else:
                compact_pt = self._higgs_compact_page_table_pp[1]
                if (
                    compact_pt is None
                    or compact_pt.shape != page_table.shape
                ):
                    compact_pt = torch.empty_like(page_table)
                    self._higgs_compact_page_table_pp[1] = compact_pt
            if fp8_layout:
                # FP8 fast path: half the HBM traffic (576 vs 1152 B/row).
                if parity == 0:
                    sel_buf = self._higgs_selected_buffer_fp8
                    if (
                        sel_buf is None
                        or sel_buf.shape[0] < page_table.numel()
                    ):
                        sel_buf = torch.empty(
                            (page_table.numel(), 1, self.kv_cache_dim),
                            dtype=torch.float8_e4m3fn,
                            device=self.device,
                        )
                        self._higgs_selected_buffer_fp8 = sel_buf
                else:
                    sel_buf = self._higgs_selected_buffer_fp8_pp[1]
                    if (
                        sel_buf is None
                        or sel_buf.shape[0] < page_table.numel()
                    ):
                        sel_buf = torch.empty(
                            (page_table.numel(), 1, self.kv_cache_dim),
                            dtype=torch.float8_e4m3fn,
                            device=self.device,
                        )
                        self._higgs_selected_buffer_fp8_pp[1] = sel_buf
                kv_cache = sel_buf[: page_table.numel()]
                dequantize_higgs_dense_2bit_page_table_fp8(
                    layer_buffer,
                    page_table,
                    kv_cache,
                    compact_pt,
                    self.higgs_codec.codebook,
                    fp8_inv_kv_scale,
                )
                return kv_cache, compact_pt
            if parity == 0:
                sel_buf = self._higgs_selected_buffer
                if (
                    sel_buf is None
                    or sel_buf.shape[0] < page_table.numel()
                ):
                    sel_buf = torch.empty(
                        (page_table.numel(), 1, self.kv_cache_dim),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    self._higgs_selected_buffer = sel_buf
            else:
                sel_buf = self._higgs_selected_buffer_pp[1]
                if (
                    sel_buf is None
                    or sel_buf.shape[0] < page_table.numel()
                ):
                    sel_buf = torch.empty(
                        (page_table.numel(), 1, self.kv_cache_dim),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    self._higgs_selected_buffer_pp[1] = sel_buf
            kv_cache = sel_buf[: page_table.numel()]
            dequantize_higgs_dense_2bit_page_table(
                layer_buffer,
                page_table,
                kv_cache,
                compact_pt,
                self.higgs_codec.codebook,
            )
            return kv_cache, compact_pt

        # Fallback: eager codec path for non-standard layouts (e.g. CPU).
        # No ping-pong here: the fallback allocates fresh tensors per
        # call so there is no aliasing hazard to begin with.
        mask = page_table >= 0
        flat_loc = page_table.clamp_min(0).reshape(-1).long()
        kv_cache = self.higgs_codec.decompress(
            layer_buffer[flat_loc],
            self.dtype,
        )
        if fp8_layout:
            kv_cache = (kv_cache.float() * fp8_inv_kv_scale).to(
                torch.float8_e4m3fn
            )
        compact_page_table = torch.arange(
            page_table.numel(),
            dtype=page_table.dtype,
            device=page_table.device,
        ).reshape(page_table.shape)
        compact_page_table = compact_page_table.masked_fill(~mask, -1)
        return kv_cache, compact_page_table

    def get_higgs_selected_kv_buffer_trtllm(
        self,
        layer_id: int,
        page_table: torch.Tensor,
        page_size: int,
        fp8_layout: bool = False,
        fp8_inv_kv_scale: float = 1.0,
        slot_parity: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """trtllm DSA decode view of the HIGGS-packed latent+rope cache.

        Sparse materialization adapter for
        :func:`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` with
        ``sparse_mla_top_k > 0``. The trtllm-gen kernel is closed-source CUBIN
        (cannot dequant in-kernel), so we materialize the *selected* HIGGS
        slots dictated by ``page_table`` into a compact BF16 buffer laid out
        as ``(num_compact_pages, page_size, kv_cache_dim)``, then re-emit
        ``block_tables`` pointing into that compact buffer.

        Why this works: when ``sparse_mla_top_k > 0``, the trtllm kernel
        treats each ``block_tables[b, t, k]`` value as a flat token index
        ``flat_idx`` into a paged dense buffer of shape
        ``(num_pages, page_size, kv_cache_dim)`` and reads
        ``kv_cache[flat_idx // page_size, flat_idx % page_size, :]``. The
        existing :meth:`get_higgs_selected_kv_buffer` already writes the
        selected slots row-by-row into a compact ``(B*K, 1, kv_cache_dim)``
        buffer with ``compact_page_table[b, k] = b * K + k`` (or ``-1`` for
        invalid). Reshaping that compact buffer to
        ``(B*K // page_size, page_size, kv_cache_dim)`` makes the indices
        line up: the flat indices the kernel reads land in the right page,
        row pair by construction.

        Memory cost: ``batch * sparse_mla_top_k * 576 * 2 B`` (BF16) per
        layer per step — ``B=128, K=2048 -> ~301 MiB`` reused across all
        61 MoE layers via the shared ``_higgs_selected_buffer``. The dequant
        cost is the same ``B*K`` HIGGS-slot inverse transforms the
        flashmla_sparse / tokenspeed_mla HIGGS paths already pay; the only
        new cost is the BF16-page-shape reshape (zero copies — same storage,
        different stride view) and the kernel dispatch path.

        Bottleneck analysis (ai-blaise #19 iter2). On B200 with 3 TB/s HBM,
        the materialize+read round-trip is the dominant cost on this path:

            dequant write : 128 * 2048 * 576 B BF16 = 301.99 MiB / layer
            kernel read   : 128 * 2048 * 576 B BF16 = 301.99 MiB / layer
            per layer     : 603.98 MiB total transfer
            per step (61L): 36.84 GB total transfer
            HBM bound     : ~12.3 ms TPOT minimum for materialization alone

        The observed HIGGS+tokenspeed_mla TPOT gap vs FP8-trtllm baseline
        (38.6 ms vs 25.9 ms = 12.7 ms) is consistent with the HIGGS dequant
        round-trip cost. Replacing tokenspeed_mla (Triton attention) with
        the trtllm-gen kernel does NOT close that gap — the attention math
        is roughly HBM-bound by the same materialized BF16 buffer in both
        paths, and the gap is the materialization itself.

        Two structural follow-ups remain (iter3 / iter4):

          (a) FP8 materialization. The flashinfer trtllm-gen sparse-MLA
              cubin set includes ``QkvE4m3OBfloat16`` variants (~48 of 96
              sparse cubins) gated on ``kv_cache.dtype == float8_e4m3fn``.
              Materializing into FP8 instead of BF16 halves the dequant
              write + kernel read traffic to ~302 MiB/step — closes
              ~6 ms of the gap. Requires:
                - new CUDA kernel
                  ``higgs_dense_2bit_dequant_page_table_fp8_kernel`` that
                  writes FP8 (``__nv_fp8_e4m3``) for the latent tile and
                  downcasts the BF16 rope tile to FP8 inline, producing the
                  576-byte fully-FP8 slot the trtllm-gen FP8 cubin expects;
                - query-side FP8 quant via
                  :func:`mla_quantize_and_rope_for_fp8` (same path the
                  existing FP8-baseline ``_forward_trtllm`` branch uses);
                - ``_fuse_rope_for_trtllm_mla`` to recognize HIGGS+trtllm
                  so the prepare-stage rope skip applies and the fused
                  rope+quant kernel handles q_rope at trtllm-call time.
          (b) Cross-layer dequant pipelining. The DSA NVFP4 indexer for
              layer N+1 has no data dependency on layer N's attention
              output (it runs on the residual stream after MoE). The
              dequant for layer N's selected slots can be issued on a
              separate CUDA stream concurrent with layer N+1's indexer
              kernel — overlaps ~half of the materialization with the
              indexer and recovers another ~3 ms when the indexer and
              dequant are similar latency.

        # NOTE(ai-blaise #19): we materialize one BF16 slot per page_table
        # entry rather than deduping repeated indices. Each query's top-k
        # selects from its OWN past context (per-request req_to_token
        # mapping); two different queries' page_table entries never point
        # at the same physical slot. There are no cross-query duplicates,
        # and within a single query top-k indices are distinct by
        # construction, so a write-side dedup buys nothing on real
        # workloads.

        # NOTE(ai-blaise #19): an alternative composition would have been
        # to teach the trtllm CUBIN to consume HIGGS-packed slots directly
        # (a custom fetch indirection). That kernel is closed-source so we
        # cannot extend it; this sparse-materialize-then-call path is the
        # tightest possible "compose, don't branch" answer for trtllm DSA.

        # ai-blaise #19 iter2 landed change: the page_table dequant CUDA
        # kernel set in :file:`higgs_dense_2bit_kv.cuh` now early-exits
        # for rows with ``page < 0`` (top-k padding). That recovers the
        # dequant work for K-end padding when ``seq_len < sparse_mla_top_k``
        # (short-context decoders), which is the largest fraction of work
        # the iter1 implementation was wasting on guaranteed-unread slots.
        # For long contexts (``seq_len >= 2048``) the indexer fills all
        # K slots with valid entries so the early-exit saves nothing; that
        # regime hits the bottleneck above and needs (a) or (b).

        Args:
          layer_id: layer index into the per-layer KV buffer list.
          page_table: ``(qo_len, sparse_mla_top_k)`` ``int32`` token-loc page
            table from the indexer (``-1`` marks padding/invalid).
          page_size: trtllm KV page size (must be 32 or 64; DSA uses 64).

        Returns:
          ``(kv_cache_paged, compact_page_table)`` where
          ``kv_cache_paged`` has shape
          ``(qo_len * sparse_mla_top_k // page_size, page_size, kv_cache_dim)``
          BF16, and ``compact_page_table`` has shape
          ``(qo_len, sparse_mla_top_k)`` ``int32`` (``-1`` preserved).
        """
        assert page_size in (32, 64), (
            "trtllm DSA decode only supports page_size 32 or 64; "
            f"got page_size={page_size}."
        )
        assert page_table.dim() == 2, (
            f"page_table must be 2-D (qo_len, top_k); got {page_table.shape}."
        )
        total = page_table.numel()
        if total % page_size != 0:
            raise ValueError(
                "HIGGS trtllm DSA path requires page_table.numel() "
                f"({total}) to be a multiple of page_size ({page_size}). "
                "The DSA indexer should pad top-k to a 64-aligned value."
            )
        kv_compact, compact_page_table = self.get_higgs_selected_kv_buffer(
            layer_id,
            page_table,
            fp8_layout=fp8_layout,
            fp8_inv_kv_scale=fp8_inv_kv_scale,
            slot_parity=slot_parity,
        )
        # kv_compact is (total, 1, kv_cache_dim); the underlying buffer is
        # contiguous on the last two axes, so .view() to the paged layout
        # is zero-copy. ai-blaise #19 iter3: in FP8 mode kv_compact is
        # ``torch.float8_e4m3fn`` at 576 B/row (vs 1152 B/row BF16); the
        # trtllm-gen ``QkvE4m3OBfloat16`` cubin is selected automatically
        # by the kernel launcher based on ``kv_cache.dtype``.
        kv_cache_paged = kv_compact.view(
            total // page_size, page_size, self.kv_cache_dim
        )
        return kv_cache_paged, compact_page_table

    def forward_higgs_dense_2bit_mla_decode(
        self,
        layer_id: int,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        page_table: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        """Fused dense MLA decode on the 2-bit HIGGS packed slots.

        Mirrors :meth:`TurboQuantDSATokenToKVPool.forward_turboquant_dense_mla_decode`.
        Always runs the split-K path: FWHT_512 of ``q_nope`` into a
        ``q_rotated`` scratch, then the topk loop is sharded across
        ``num_splits`` blocks per ``(row, head)`` with a merge-and-
        inverse-FWHT stage. The legacy single-pass kernels were
        retired in iter3 (#16) because the split-K path is uniformly
        faster (12× at B=1, 4× at B=16 on B200 sm_100, topk=2048).
        ``num_splits`` is clamped to at least 2 so the dropped
        single-pass branch is unreachable.
        """
        layer_buffer = self._get_layersplit_kv_buffer(layer_id)
        out = torch.empty(
            (q_nope.shape[0], q_nope.shape[1], self.kv_lora_rank),
            dtype=self.dtype,
            device=q_nope.device,
        )
        candidate = get_higgs_dense_2bit_b200_candidate()
        num_splits = self._select_higgs_mla_decode_num_splits(
            q_nope.shape[0], q_nope.shape[1], page_table.shape[1]
        )
        # Iter3 (#16): single-pass kernels dropped — always need >= 2
        # splits to keep stage2 merge well-defined.
        num_splits = max(int(num_splits), 2)
        if candidate.name == "store_saw_scalar2":
            if page_table.shape[1] >= 1024:
                num_splits = max(num_splits, 16)
            mid_shape = (
                q_nope.shape[0],
                q_nope.shape[1],
                num_splits,
                self.kv_lora_rank + 2,
            )
            if (
                self._higgs_mla_decode_mid is None
                or self._higgs_mla_decode_mid.shape != mid_shape
            ):
                self._higgs_mla_decode_mid = torch.empty(
                    mid_shape,
                    dtype=torch.float32,
                    device=q_nope.device,
                )
            higgs_dense_2bit_mla_decode_saw_scalar2_split(
                q_nope,
                q_rope,
                layer_buffer,
                page_table,
                self._higgs_mla_decode_mid,
                out,
                self.higgs_codec.codebook,
                sm_scale,
            )
            return out

        mid_shape = (
            q_nope.shape[0],
            q_nope.shape[1],
            num_splits,
            self.kv_lora_rank + 2,
        )
        if (
            self._higgs_mla_decode_mid is None
            or self._higgs_mla_decode_mid.shape != mid_shape
        ):
            self._higgs_mla_decode_mid = torch.empty(
                mid_shape,
                dtype=torch.float32,
                device=q_nope.device,
            )
        q_rotated_shape = q_nope.shape
        if (
            self._higgs_mla_q_rotated is None
            or self._higgs_mla_q_rotated.shape != q_rotated_shape
        ):
            self._higgs_mla_q_rotated = torch.empty(
                q_rotated_shape,
                dtype=torch.float32,
                device=q_nope.device,
            )
        higgs_dense_2bit_mla_rotate_query(
            q_nope,
            self._higgs_mla_q_rotated,
        )
        higgs_dense_2bit_mla_decode_split(
            self._higgs_mla_q_rotated,
            q_rope,
            layer_buffer,
            page_table,
            self._higgs_mla_decode_mid,
            out,
            self.higgs_codec.codebook,
            sm_scale,
        )
        return out

    @staticmethod
    def _auto_higgs_mla_decode_num_splits(
        num_rows: int, num_heads: int, topk: int
    ) -> int:
        return select_higgs_mla_decode_num_splits(num_rows, num_heads, topk)

    def _select_higgs_mla_decode_num_splits(
        self, num_rows: int, num_heads: int, topk: int
    ) -> int:
        if self.higgs_mla_decode_num_splits > 0:
            return self.higgs_mla_decode_num_splits
        return self._auto_higgs_mla_decode_num_splits(num_rows, num_heads, topk)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        self.set_mla_kv_buffer(
            layer,
            loc,
            cache_k[..., : self.kv_lora_rank],
            cache_k[..., self.kv_lora_rank :],
        )

    def _set_higgs_mla_kv_buffer_to_buffer(
        self,
        kv_buffer: torch.Tensor,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        if (
            self.dtype == torch.bfloat16
            and loc.is_cuda
            and loc.dtype == torch.int64
            and cache_k_nope.is_cuda
            and cache_k_rope.is_cuda
            and cache_k_nope.dtype == torch.bfloat16
            and cache_k_rope.dtype == torch.bfloat16
        ):
            if cache_k_nope.dim() == 2:
                cache_k_nope = cache_k_nope.unsqueeze(1)
            if cache_k_rope.dim() == 2:
                cache_k_rope = cache_k_rope.unsqueeze(1)
            store_higgs_dense_2bit(
                kv_buffer,
                loc,
                cache_k_nope,
                cache_k_rope,
                self.higgs_codec.codebook,
                self.higgs_codec.codebook_norm_sq,
            )
        else:
            kv_buffer[loc] = self.higgs_codec.compress(
                cache_k_nope,
                cache_k_rope,
            )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id
        layer_idx = layer_id - self.start_layer
        active_rows = self._layersplit_active_rows_from_loc(loc)
        if not self.layersplit_owns_layer(layer_id):
            kv_buffer = self._finish_layersplit_kv_prefetch_for_update(layer_id)
            if kv_buffer is not None:
                if self._uses_higgs_layer(layer_id):
                    self._set_higgs_mla_kv_buffer_to_buffer(
                        kv_buffer, loc, cache_k_nope, cache_k_rope
                    )
                else:
                    self._set_mla_kv_buffer_to_buffer(
                        kv_buffer, loc, cache_k_nope, cache_k_rope
                    )
            else:
                self.prefetch_layersplit_kv_buffer(layer_id, active_rows=active_rows)
            return

        if not self._uses_higgs_layer(layer_id):
            super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)
            return

        has_prefetched_kv = self._layersplit_kv_prefetch_layer_id == layer_id
        if has_prefetched_kv:
            self._finish_layersplit_kv_prefetch_for_update(layer_id)
        self._set_higgs_mla_kv_buffer_to_buffer(
            self.kv_buffer[layer_idx], loc, cache_k_nope, cache_k_rope
        )
        if not has_prefetched_kv:
            self.prefetch_layersplit_kv_buffer(layer_id, active_rows=active_rows)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        layer_id = layer.layer_id
        if not self._uses_higgs_layer(layer_id):
            return super().get_mla_kv_buffer(layer, loc, dst_dtype)

        dst_dtype = dst_dtype or self.dtype
        kv = self.higgs_codec.decompress(
            self._get_layersplit_kv_buffer(layer_id)[loc], dst_dtype
        )
        return kv[..., : self.kv_lora_rank], kv[..., self.kv_lora_rank :]


def move_kv_cache_native(
    k_buffer: List[torch.Tensor],
    v_buffer: List[torch.Tensor],
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
):
    if tgt_loc.numel() == 0:
        return

    tgt_loc_flat = tgt_loc.view(-1).long()
    src_loc_flat = src_loc.view(-1).long()
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        k_cache[tgt_loc_flat] = k_cache[src_loc_flat]
        v_cache[tgt_loc_flat] = v_cache[src_loc_flat]


@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    """2D tiled kernel. Safe for in-place copy."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < stride
    tl.multiple_of(byte_off, 16)

    loc_idx = tl.arange(0, num_locs_upper)
    mask_loc = loc_idx < num_locs

    src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc, other=0)
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc, other=0)

    src_ptr = base_ptr + src[:, None] * stride + byte_off[None, :]
    tgt_ptr = base_ptr + tgt[:, None] * stride + byte_off[None, :]

    mask = mask_loc[:, None] & mask_byte[None, :]
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)

# Backward-compatible NSA class names for existing ai-blaise tests and scripts.
NSATokenToKVPool = DSATokenToKVPool
TurboQuantNSATokenToKVPool = TurboQuantDSATokenToKVPool
HiggsDense2BitNSATokenToKVPool = HiggsDense2BitDSATokenToKVPool
