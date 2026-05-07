import json
import logging
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sglang.srt.distributed.utils import StatelessProcessGroup
from sglang.srt.environ import temp_set_env
from sglang.srt.utils.common import get_current_device_stream_fast

logger = logging.getLogger(__name__)


def parse_torchcomms_ncclx_hints(raw_hints: str) -> Dict[str, str]:
    if not raw_hints:
        return {}

    raw_hints = raw_hints.strip()
    if raw_hints.startswith("{"):
        parsed = json.loads(raw_hints)
        if not isinstance(parsed, dict):
            raise ValueError("torchcomms NCCLX hints JSON must be an object")
        return {str(key): str(value) for key, value in parsed.items()}

    hints = {}
    for item in raw_hints.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(
                "torchcomms NCCLX hints must be JSON or comma-separated key=value pairs"
            )
        hints[key.strip()] = value.strip()
    return hints


class TorchCommsNcclxCommunicator:
    def __init__(
        self,
        group: Union[ProcessGroup, StatelessProcessGroup],
        device: Union[int, str, torch.device],
        name: str,
        hints: Optional[Dict[str, str]] = None,
        timeout: Optional[timedelta] = None,
        abort_process_on_timeout_or_error: bool = False,
        enable_rdma_registration: bool = False,
    ):
        if isinstance(group, StatelessProcessGroup):
            self.rank = group.rank
            self.world_size = group.world_size
        else:
            assert dist.is_initialized()
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)

        self.group = group
        self.name = name
        self.hints = hints or {}
        self.timeout = timeout

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        self.available = False
        self.disabled = True
        self.comm: Optional[Any] = None

        if self.world_size == 1:
            return

        import torchcomms

        if not torchcomms.is_backend_built("ncclx"):
            raise RuntimeError(
                "torchcomms is installed without the NCCLX backend. "
                "Install a torchcomms build that includes ncclx."
            )

        if enable_rdma_registration:
            try:
                from torchcomms import _comms_ncclx as ncclx

                ncclx.init_caching_allocator_hook()
            except Exception as exc:
                logger.warning(
                    "Failed to initialize torchcomms NCCLX CUDA allocator hook: %s",
                    exc,
                )

        store = None
        if not isinstance(group, StatelessProcessGroup):
            from torch.distributed import PrefixStore, distributed_c10d

            store = PrefixStore(name, distributed_c10d._get_default_store())

        with temp_set_env(
            TORCHCOMM_RANK=self.rank,
            TORCHCOMM_SIZE=self.world_size,
            TORCHCOMM_BOOTSTRAP_RANKSIZE_QUERY_METHOD="env",
        ):
            self.comm = torchcomms.new_comm(
                "ncclx",
                self.device,
                store=store,
                name=name,
                hints=self.hints or None,
                timeout=timeout,
                abort_process_on_timeout_or_error=(
                    abort_process_on_timeout_or_error
                ),
            )

        self.available = True
        self.disabled = False
        if self.rank == 0:
            backend_version = self._optional_backend_value(
                self.comm.get_backend_version
            )
            logger.info(
                "sglang is using torchcomms NCCLX backend version %s for %s",
                backend_version,
                name,
            )

    @staticmethod
    def _optional_backend_value(method: Callable[[], Any]) -> Any:
        try:
            return method()
        except RuntimeError as exc:
            return {"unavailable": str(exc)}

    def _check_tensor(self, tensor: torch.Tensor) -> None:
        assert tensor.device == self.device, (
            f"this torchcomms NCCLX communicator is created for {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        assert tensor.is_contiguous(), "torchcomms NCCLX requires contiguous tensors"

    def _check_tensors(self, tensors: Iterable[torch.Tensor]) -> None:
        for tensor in tensors:
            self._check_tensor(tensor)

    def _backend_method(self, method_name: str) -> Callable[..., Any]:
        backend = self.comm.get_backend_impl()
        method = getattr(backend, method_name, None)
        if method is None:
            raise RuntimeError(
                f"torchcomms NCCLX backend does not expose {method_name}"
            )
        return method

    @staticmethod
    def _to_torchcomms_reduce_op(op: ReduceOp) -> Any:
        import torchcomms

        for name in ("SUM", "PRODUCT", "MIN", "MAX", "AVG", "BAND", "BOR", "BXOR"):
            if hasattr(ReduceOp, name) and op == getattr(ReduceOp, name):
                return getattr(torchcomms.ReduceOp, name)
        raise ValueError(f"Unsupported torchcomms NCCLX reduce op: {op}")

    def _current_stream_context(self):
        stream = get_current_device_stream_fast()
        return torch.cuda.stream(stream)

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM):
        if self.disabled:
            return
        self._check_tensor(tensor)
        with self._current_stream_context():
            self.comm.all_reduce(
                tensor, self._to_torchcomms_reduce_op(op), async_op=False
            )

    def outplace_all_reduce(
        self,
        in_tensor: torch.Tensor,
        out_tensor: Optional[torch.Tensor] = None,
        op: ReduceOp = ReduceOp.SUM,
    ) -> Optional[torch.Tensor]:
        if self.disabled:
            return None
        self._check_tensor(in_tensor)
        if out_tensor is None:
            out_tensor = torch.empty_like(in_tensor)
        else:
            self._check_tensor(out_tensor)
        out_tensor.copy_(in_tensor)
        self.all_reduce(out_tensor, op=op)
        return out_tensor

    def reduce(
        self, tensor: torch.Tensor, dst: int, op: ReduceOp = ReduceOp.SUM
    ) -> None:
        if self.disabled:
            return
        self._check_tensor(tensor)
        with self._current_stream_context():
            self.comm.reduce(
                tensor, dst, self._to_torchcomms_reduce_op(op), async_op=False
            )

    def all_gather(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: Optional[List[int]] = None,
    ):
        if self.disabled:
            return
        self._check_tensor(output_tensor)
        self._check_tensor(input_tensor)
        with self._current_stream_context():
            if sizes is None:
                self.comm.all_gather_single(output_tensor, input_tensor, async_op=False)
            else:
                output_list = list(output_tensor.split(sizes, dim=0))
                self.comm.all_gather_v(output_list, input_tensor, async_op=False)

    def cp_all_gather_into_tensor(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        stream: torch.cuda.Stream,
        sizes: Optional[List[int]] = None,
    ):
        self._check_tensor(output_tensor)
        self._check_tensor(input_tensor)
        with torch.cuda.stream(stream):
            if sizes is None:
                self.comm.all_gather_single(output_tensor, input_tensor, async_op=False)
            else:
                output_list = list(output_tensor.split(sizes, dim=0))
                self.comm.all_gather_v(output_list, input_tensor, async_op=False)

    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        sizes: Optional[List[int]] = None,
    ):
        if self.disabled:
            return
        self._check_tensor(output_tensor)
        self._check_tensor(input_tensor)
        tc_op = self._to_torchcomms_reduce_op(op)
        with self._current_stream_context():
            if sizes is None:
                self.comm.reduce_scatter_single(
                    output_tensor, input_tensor, tc_op, async_op=False
                )
            else:
                input_list = list(input_tensor.split(sizes, dim=0))
                self.comm.reduce_scatter_v(
                    output_tensor, input_list, tc_op, async_op=False
                )

    def all_to_all_single(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor
    ) -> None:
        if self.disabled:
            return
        self._check_tensor(output_tensor)
        self._check_tensor(input_tensor)
        with self._current_stream_context():
            self.comm.all_to_all_single(output_tensor, input_tensor, async_op=False)

    def all_to_all_v_single(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
    ) -> None:
        if self.disabled:
            return
        self._check_tensor(output_tensor)
        self._check_tensor(input_tensor)
        with self._current_stream_context():
            self.comm.all_to_all_v_single(
                output_tensor,
                input_tensor,
                output_split_sizes,
                input_split_sizes,
                async_op=False,
            )

    def all_to_all(
        self,
        output_tensor_list: List[torch.Tensor],
        input_tensor_list: List[torch.Tensor],
    ) -> None:
        if self.disabled:
            return
        self._check_tensors(output_tensor_list)
        self._check_tensors(input_tensor_list)
        with self._current_stream_context():
            self.comm.all_to_all(
                output_tensor_list, input_tensor_list, async_op=False
            )

    def send(self, tensor: torch.Tensor, dst: int):
        if self.disabled:
            return
        self._check_tensor(tensor)
        with self._current_stream_context():
            self.comm.send(tensor, dst, async_op=False)

    def recv(self, tensor: torch.Tensor, src: int):
        if self.disabled:
            return
        self._check_tensor(tensor)
        with self._current_stream_context():
            self.comm.recv(tensor, src, async_op=False)

    def broadcast(self, tensor: torch.Tensor, src: int):
        if self.disabled:
            return
        self._check_tensor(tensor)
        with self._current_stream_context():
            self.comm.broadcast(tensor, src, async_op=False)

    def broadcast_async(self, tensor: torch.Tensor, src: int):
        if self.disabled:
            return None
        self._check_tensor(tensor)
        with self._current_stream_context():
            return self.comm.broadcast(tensor, src, async_op=True)

    def barrier(self) -> None:
        if self.disabled:
            return
        with self._current_stream_context():
            self.comm.barrier(async_op=False)

    def scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor_list: List[torch.Tensor],
        src: int,
    ) -> None:
        if self.disabled:
            return
        self._check_tensor(output_tensor)
        self._check_tensors(input_tensor_list)
        with self._current_stream_context():
            self.comm.scatter(
                output_tensor, input_tensor_list, src, async_op=False
            )

    def gather(
        self,
        output_tensor_list: List[torch.Tensor],
        input_tensor: torch.Tensor,
        dst: int,
    ) -> None:
        if self.disabled:
            return
        self._check_tensors(output_tensor_list)
        self._check_tensor(input_tensor)
        with self._current_stream_context():
            self.comm.gather(output_tensor_list, input_tensor, dst, async_op=False)

    def gather_single(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, dst: int
    ) -> None:
        if self.disabled:
            return
        self._check_tensor(output_tensor)
        self._check_tensor(input_tensor)
        with self._current_stream_context():
            self.comm.gather_single(output_tensor, input_tensor, dst, async_op=False)

    def split(self, ranks: List[int], name: str):
        if self.disabled:
            return None
        return self.comm.split(
            ranks, name, hints=self.hints or None, timeout=self.timeout
        )

    def new_window(self, tensor: Optional[torch.Tensor] = None):
        if self.disabled:
            return None
        if tensor is not None:
            self._check_tensor(tensor)
        return self.comm.new_window(tensor)

    def get_device_transport(self):
        if self.comm is None:
            return None
        return self.comm.get_device_transport()

    @property
    def mem_allocator(self):
        if self.comm is None:
            return None
        return self.comm.mem_allocator

    def batch_op_create(self):
        if self.comm is None:
            return None
        return self.comm.batch_op_create()

    def register_pre_hook(self, callback: Callable[..., None]):
        if self.comm is None:
            return None
        return self.comm.register_pre_hook(callback)

    def register_post_hook(self, callback: Callable[..., None]):
        if self.comm is None:
            return None
        return self.comm.register_post_hook(callback)

    def register_abort_hook(self, callback: Callable[[], None]):
        if self.comm is None:
            return None
        return self.comm.register_abort_hook(callback)

    def all_gather_p_init(self, output_tensor: torch.Tensor):
        if self.disabled:
            return None
        self._check_tensor(output_tensor)
        return self.comm.all_gather_p_init(
            output_tensor, hints=self.hints or None, timeout=self.timeout
        )

    def all_gather_p_exec(self, handle: Any, input_tensor: torch.Tensor):
        if self.disabled:
            return None
        self._check_tensor(input_tensor)
        with self._current_stream_context():
            return self.comm.all_gather_p_exec(
                handle, input_tensor, async_op=False
            )

    def all_gather_p_free(self, handle: Any) -> None:
        if self.comm is not None:
            self.comm.all_gather_p_free(handle)

    def device_alltoallv_single(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: torch.Tensor,
        input_split_sizes: torch.Tensor,
    ) -> None:
        if self.disabled:
            return
        self._check_tensor(output_tensor)
        self._check_tensor(input_tensor)
        self._check_tensor(output_split_sizes)
        self._check_tensor(input_split_sizes)
        with self._current_stream_context():
            self._backend_method("device_alltoallv_single")(
                output_tensor,
                input_tensor,
                output_split_sizes,
                input_split_sizes,
                async_op=False,
            )

    def alltoallv_dynamic_dispatch(
        self,
        output_tensor_list: List[torch.Tensor],
        output_chunk_sizes_per_rank: torch.Tensor,
        input_tensor: torch.Tensor,
        input_chunk_sizes: torch.Tensor,
        input_chunk_indices: torch.Tensor,
        input_chunk_count_per_rank: torch.Tensor,
        hidden_dim: int,
    ) -> None:
        if self.disabled:
            return
        self._check_tensors(output_tensor_list)
        self._check_tensor(output_chunk_sizes_per_rank)
        self._check_tensor(input_tensor)
        self._check_tensor(input_chunk_sizes)
        self._check_tensor(input_chunk_indices)
        self._check_tensor(input_chunk_count_per_rank)
        with self._current_stream_context():
            self._backend_method("alltoallv_dynamic_dispatch")(
                output_tensor_list,
                output_chunk_sizes_per_rank,
                input_tensor,
                input_chunk_sizes,
                input_chunk_indices,
                input_chunk_count_per_rank,
                hidden_dim,
                async_op=False,
            )

    def alltoallv_dynamic_combine(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        input_chunk_sizes: torch.Tensor,
        input_chunk_indices: torch.Tensor,
        input_chunk_count_per_rank: torch.Tensor,
        hidden_dim: int,
    ) -> None:
        if self.disabled:
            return
        self._check_tensor(output_tensor)
        self._check_tensor(input_tensor)
        self._check_tensor(input_chunk_sizes)
        self._check_tensor(input_chunk_indices)
        self._check_tensor(input_chunk_count_per_rank)
        with self._current_stream_context():
            self._backend_method("alltoallv_dynamic_combine")(
                output_tensor,
                input_tensor,
                input_chunk_sizes,
                input_chunk_indices,
                input_chunk_count_per_rank,
                hidden_dim,
                async_op=False,
            )

    def reduce_scatter_quantized(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        seed: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
    ) -> None:
        if self.disabled:
            return
        self._check_tensor(output_tensor)
        self._check_tensor(input_tensor)
        self._check_tensor(seed)
        tc_op = self._to_torchcomms_reduce_op(op)
        with self._current_stream_context():
            self._backend_method("reduce_scatter_quantized")(
                output_tensor,
                input_tensor,
                tc_op,
                seed,
                async_op=False,
            )

    def comm_dump(self) -> Optional[Dict[str, str]]:
        if self.comm is None:
            return None
        method = getattr(self.comm.get_backend_impl(), "comm_dump", None)
        if method is None:
            return None
        return self._optional_backend_value(method)

    def backend_info(self) -> Dict[str, Any]:
        if self.comm is None:
            return {
                "available": False,
                "disabled": self.disabled,
                "name": self.name,
            }
        info = {
            "available": self.available,
            "disabled": self.disabled,
            "name": self.comm.get_name(),
            "backend": self.comm.get_backend(),
            "backend_version": self._optional_backend_value(
                self.comm.get_backend_version
            ),
            "rank": self.comm.get_rank(),
            "world_size": self.comm.get_size(),
            "abort_enabled": self.comm.abort_enabled(),
            "is_aborted": self.comm.is_aborted(),
        }
        comm_dump = self.comm_dump()
        if comm_dump is not None:
            info["comm_dump"] = comm_dump
        return info

    def get_init_handle(self):
        if self.comm is None:
            return None
        return self.comm.get_init_handle()

    def reconfigure(self, uuid: int, init_handles: Union[List[str], set]):
        if self.comm is None:
            return None
        return self.comm.reconfigure(
            uuid, init_handles, timeout=self.timeout, hints=self.hints or None
        )

    def abort(self) -> None:
        if self.comm is not None:
            self.comm.abort()

    def abort_enabled(self) -> bool:
        return bool(self.comm is not None and self.comm.abort_enabled())

    def is_aborted(self) -> bool:
        return bool(self.comm is not None and self.comm.is_aborted())

    def group_start(self):
        return None

    def group_end(self):
        return None

    def finalize(self):
        if self.comm is not None:
            self.comm.finalize()
            self.comm = None
        self.available = False
        self.disabled = True

    @contextmanager
    def change_state(self, enable: Optional[bool] = None):
        if enable is None:
            enable = self.available
        old_disable = self.disabled
        self.disabled = not enable
        try:
            yield
        finally:
            self.disabled = old_disable
