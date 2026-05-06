import json
import logging
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

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
                from torchcomms import ncclx

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
            )

        self.available = True
        self.disabled = False
        if self.rank == 0:
            logger.info(
                "sglang is using torchcomms NCCLX backend version %s for %s",
                self.comm.get_backend_version(),
                name,
            )

    def _check_tensor(self, tensor: torch.Tensor) -> None:
        assert tensor.device == self.device, (
            f"this torchcomms NCCLX communicator is created for {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        assert tensor.is_contiguous(), "torchcomms NCCLX requires contiguous tensors"

    @staticmethod
    def _to_torchcomms_reduce_op(op: ReduceOp) -> Any:
        import torchcomms

        if op == ReduceOp.SUM:
            return torchcomms.ReduceOp.SUM
        if op == ReduceOp.PRODUCT:
            return torchcomms.ReduceOp.PRODUCT
        if op == ReduceOp.MIN:
            return torchcomms.ReduceOp.MIN
        if op == ReduceOp.MAX:
            return torchcomms.ReduceOp.MAX
        if op == ReduceOp.AVG:
            return torchcomms.ReduceOp.AVG
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
