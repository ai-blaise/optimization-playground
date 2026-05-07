import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_ncclx_module():
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.device = lambda value: value
    torch.cuda = types.SimpleNamespace(
        Stream=object, stream=lambda stream: contextlib.nullcontext()
    )

    dist = types.ModuleType("torch.distributed")
    dist.ProcessGroup = object
    dist.ReduceOp = types.SimpleNamespace(
        SUM="sum",
        PRODUCT="product",
        MIN="min",
        MAX="max",
        AVG="avg",
        BAND="band",
        BOR="bor",
        BXOR="bxor",
    )
    torch.distributed = dist

    environ = types.ModuleType("sglang.srt.environ")

    @contextlib.contextmanager
    def temp_set_env(**_kwargs):
        yield

    environ.temp_set_env = temp_set_env

    common = types.ModuleType("sglang.srt.utils.common")
    common.get_current_device_stream_fast = lambda: None

    modules = {
        "torch": torch,
        "torch.distributed": dist,
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.environ": environ,
        "sglang.srt.utils": types.ModuleType("sglang.srt.utils"),
        "sglang.srt.utils.common": common,
        "sglang.srt.distributed": types.ModuleType("sglang.srt.distributed"),
        "sglang.srt.distributed.utils": types.ModuleType(
            "sglang.srt.distributed.utils"
        ),
    }
    modules["sglang.srt.distributed.utils"].StatelessProcessGroup = object
    old = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        path = (
            ROOT
            / "python"
            / "sglang"
            / "srt"
            / "distributed"
            / "device_communicators"
            / "torchcomms_ncclx.py"
        )
        spec = importlib.util.spec_from_file_location("torchcomms_ncclx_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_torchcomms_ncclx_hints_parser():
    parse_torchcomms_ncclx_hints = load_ncclx_module().parse_torchcomms_ncclx_hints

    assert parse_torchcomms_ncclx_hints("") == {}
    assert parse_torchcomms_ncclx_hints('{"foo": "bar", "count": 4}') == {
        "foo": "bar",
        "count": "4",
    }
    assert parse_torchcomms_ncclx_hints("transport=rdma,mode=low_latency") == {
        "transport": "rdma",
        "mode": "low_latency",
    }

    with pytest.raises(ValueError):
        parse_torchcomms_ncclx_hints("transport")


def test_torchcomms_ncclx_server_args_are_registered():
    source = (ROOT / "python" / "sglang" / "srt" / "server_args.py").read_text()
    assert "--device-collective-backend" in source
    assert 'choices=["default", "ncclx"]' in source
    assert "--torchcomms-ncclx-hints" in source
    assert "--torchcomms-ncclx-strict" in source
    assert "--enable-torchcomms-ncclx-rdma" in source
    assert "--torchcomms-ncclx-abort-on-timeout" in source


def test_torchcomms_ncclx_uses_packaged_backend_module():
    source = (
        ROOT
        / "python"
        / "sglang"
        / "srt"
        / "distributed"
        / "device_communicators"
        / "torchcomms_ncclx.py"
    ).read_text()
    assert "from torchcomms import _comms_ncclx as ncclx" in source
    assert "tc_op = self._to_torchcomms_reduce_op(op)" in source


def test_torchcomms_ncclx_wrapper_exposes_documented_collectives():
    communicator = load_ncclx_module().TorchCommsNcclxCommunicator
    for method in [
        "all_reduce",
        "reduce",
        "all_gather",
        "cp_all_gather_into_tensor",
        "reduce_scatter",
        "all_to_all_single",
        "all_to_all_v_single",
        "all_to_all",
        "broadcast",
        "scatter",
        "gather",
        "gather_single",
        "send",
        "recv",
        "barrier",
        "split",
        "new_window",
        "get_device_transport",
        "batch_op_create",
        "register_pre_hook",
        "register_post_hook",
        "register_abort_hook",
        "all_gather_p_init",
        "all_gather_p_exec",
        "all_gather_p_free",
        "device_alltoallv_single",
        "alltoallv_dynamic_dispatch",
        "alltoallv_dynamic_combine",
        "reduce_scatter_quantized",
        "backend_info",
        "comm_dump",
        "abort",
        "abort_enabled",
        "is_aborted",
        "reconfigure",
    ]:
        assert hasattr(communicator, method)
