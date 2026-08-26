# SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang.srt.snapshot_hooks covering both dependency branches.

Import-light by design: the module under test is loaded standalone via
importlib with torch / dynamo_runtime / scheduler stand-ins injected
through sys.modules, one subprocess per scenario so signal-handler and
module state cannot leak between tests. The neutral-branch tests require
the criu_snapshot_hooks package (ai-blaise/criu-snapshots) and are
skipped when it is not installed.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "snapshot_hooks.py"
)

requires_neutral_package = pytest.mark.skipif(
    importlib.util.find_spec("criu_snapshot_hooks") is None,
    reason="criu_snapshot_hooks is not installed",
)

_PRELUDE = """
import importlib.util
import json
import os
import signal
import sys
import tempfile
import time
import types
from pathlib import Path

REGISTERED_SIGNALS = {}
CALLS = []


def stub_signals():
    signal.signal = lambda signum, handler: REGISTERED_SIGNALS.__setitem__(
        signum, handler
    )
    signal.getsignal = lambda signum: None


def stub_torch(dist_attrs=None):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        synchronize=lambda: CALLS.append("cuda.synchronize"),
        empty_cache=lambda: CALLS.append("cuda.empty_cache"),
        current_device=lambda: 0,
    )
    torch.device = lambda spec: spec
    torch.zeros = lambda *args, **kwargs: "zeros"
    dist = types.ModuleType("torch.distributed")
    dist.is_initialized = lambda: False
    dist.all_reduce = lambda t: CALLS.append("dist.all_reduce")
    dist.barrier = lambda: CALLS.append("dist.barrier")
    dist.init_process_group = lambda **kwargs: CALLS.append(
        ("init_process_group", kwargs)
    )
    dist.destroy_process_group = lambda: CALLS.append("destroy_process_group")
    for name, value in (dist_attrs or {}).items():
        setattr(dist, name, value)
    torch.distributed = dist
    sys.modules["torch"] = torch
    sys.modules["torch.distributed"] = dist
    return dist


def stub_scheduler():
    scheduler = types.ModuleType("sglang.srt.managers.scheduler")
    scheduler.drain_inflight = lambda deadline_s: CALLS.append(
        ("drain_inflight", deadline_s)
    )
    scheduler.resume_inflight = lambda: CALLS.append("resume_inflight")
    sys.modules["sglang.srt.managers.scheduler"] = scheduler


def stub_kv_router():
    runtime = types.ModuleType("dynamo_runtime")
    runtime.kv_router = types.SimpleNamespace(
        detach_current_replica=lambda timeout_s: CALLS.append(
            ("kv_detach", timeout_s)
        ),
        attach_current_replica=lambda: CALLS.append("kv_attach"),
    )
    sys.modules["dynamo_runtime"] = runtime


def block_neutral_package():
    sys.modules["criu_snapshot_hooks"] = None


def load_module():
    spec = importlib.util.spec_from_file_location(
        "snapshot_hooks", os.environ["SNAPSHOT_HOOKS_MODULE"]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wait_for(path, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"{path} did not appear within {timeout_s}s")
"""


def _run(code: str) -> None:
    env = os.environ.copy()
    env["SNAPSHOT_HOOKS_MODULE"] = str(_MODULE_PATH)
    subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(code)],
        check=True,
        env=env,
    )


def test_legacy_branch_registers_inline_handlers_and_ack_files():
    _run("""
block_neutral_package()
stub_signals()
stub_torch()

module = load_module()
assert module.NEUTRAL_PACKAGE is None
assert not REGISTERED_SIGNALS

expected_pre = getattr(signal, "SIGRTMIN", 34) + 5
assert module.PRE_SNAPSHOT_SIGNAL == expected_pre
assert module.POST_RESUME_SIGNAL == expected_pre + 1

tmp = Path(tempfile.mkdtemp())
module.READY_FILE = str(tmp / "pre_snapshot.ready")
module.READY_ERR_FILE = module.READY_FILE + ".err"
module.RESUME_FILE = str(tmp / "post_restore.done")

module.install()
assert set(REGISTERED_SIGNALS) == {
    module.PRE_SNAPSHOT_SIGNAL,
    module.POST_RESUME_SIGNAL,
}

REGISTERED_SIGNALS[module.PRE_SNAPSHOT_SIGNAL](module.PRE_SNAPSHOT_SIGNAL, None)
ready = json.loads(Path(module.READY_FILE).read_text(encoding="utf-8"))
assert ready["pid"] == os.getpid()
assert ready["ranks"] is None
assert Path(module.READY_FILE + "." + str(os.getpid())).exists()
assert not Path(module.READY_ERR_FILE).exists()

REGISTERED_SIGNALS[module.POST_RESUME_SIGNAL](module.POST_RESUME_SIGNAL, None)
assert Path(module.RESUME_FILE).exists()
assert Path(module.RESUME_FILE + "." + str(os.getpid())).exists()
""")


def test_legacy_branch_writes_error_ack_on_quiesce_failure():
    _run("""
block_neutral_package()
stub_signals()
stub_torch()

runtime = types.ModuleType("dynamo_runtime")


def _fail(timeout_s):
    raise RuntimeError("router detach timed out")


runtime.kv_router = types.SimpleNamespace(detach_current_replica=_fail)
sys.modules["dynamo_runtime"] = runtime

module = load_module()
tmp = Path(tempfile.mkdtemp())
module.READY_FILE = str(tmp / "pre_snapshot.ready")
module.READY_ERR_FILE = module.READY_FILE + ".err"
module.RESUME_FILE = str(tmp / "post_restore.done")
module.install()

REGISTERED_SIGNALS[module.PRE_SNAPSHOT_SIGNAL](module.PRE_SNAPSHOT_SIGNAL, None)
assert not Path(module.READY_FILE).exists()
error = Path(module.READY_ERR_FILE).read_text(encoding="utf-8")
assert "router detach timed out" in error
""")


@requires_neutral_package
def test_neutral_branch_signalfile_trigger_drains_and_acks():
    _run("""
stub_signals()
stub_torch()

control_dir = Path(tempfile.mkdtemp())
os.environ["DYN_SNAPSHOT_CONTROL_DIR"] = str(control_dir)
os.environ["SGLANG_SNAPSHOT_HOOKS"] = "1"

module = load_module()
import criu_snapshot_hooks

assert module.NEUTRAL_PACKAGE is criu_snapshot_hooks
assert module.PRE_SNAPSHOT_SIGNAL == criu_snapshot_hooks.DEFAULT_QUIESCE_SIGNAL == 39
assert module.POST_RESUME_SIGNAL == criu_snapshot_hooks.DEFAULT_RESUME_SIGNAL == 40
assert set(REGISTERED_SIGNALS) == {39, 40}

ready = json.loads(
    (control_dir / "ready-for-checkpoint").read_text(encoding="utf-8")
)
assert ready["pid"] == os.getpid()
assert ready["contract"] == criu_snapshot_hooks.HOOK_CONTRACT_VERSION

module.install()
assert set(REGISTERED_SIGNALS) == {39, 40}

(control_dir / "quiesce-requested").touch()
wait_for(control_dir / "quiesced")
assert not (control_dir / "quiesce-requested").exists()
assert not (control_dir / "error").exists()
assert "cuda.empty_cache" in CALLS

payload = {
    "epoch": 1,
    "rendezvous": {
        "masterAddr": "10.9.9.9",
        "masterPort": 29501,
        "rank": 5,
        "worldSize": 16,
        "backend": "nccl",
    },
}
(control_dir / "resume.json").write_text(json.dumps(payload), encoding="utf-8")
(control_dir / "resume-requested").touch()
wait_for(control_dir / "resumed")
assert not (control_dir / "error").exists()

init_calls = [c for c in CALLS if isinstance(c, tuple) and c[0] == "init_process_group"]
assert init_calls == [
    ("init_process_group", {"backend": "nccl", "rank": 5, "world_size": 16})
]
assert os.environ["MASTER_ADDR"] == "10.9.9.9"
assert os.environ["MASTER_PORT"] == "29501"
assert "dist.all_reduce" in CALLS and "dist.barrier" in CALLS
""")


@requires_neutral_package
def test_neutral_callbacks_delegate_teardown_and_keep_sglang_residue():
    _run("""
stub_signals()
initialized = {"value": True}
stub_torch(
    {
        "is_initialized": lambda: initialized["value"],
        "get_backend": lambda: "nccl",
        "get_world_size": lambda: 8,
        "get_rank": lambda: 3,
    }
)
stub_scheduler()
stub_kv_router()

os.environ["DYN_SNAPSHOT_CONTROL_DIR"] = tempfile.mkdtemp()
os.environ["MASTER_ADDR"] = "10.0.0.1"
os.environ["MASTER_PORT"] = "29500"

module = load_module()
module._on_quiesce()

assert CALLS.index(("kv_detach", 2.0)) < CALLS.index(("drain_inflight", 5.0))
assert CALLS.index(("drain_inflight", 5.0)) < CALLS.index("destroy_process_group")
assert module._pre_snapshot_state["nccl"] == {
    "backend": "nccl",
    "world_size": 8,
    "rank": 3,
    "master_addr": "10.0.0.1",
    "master_port": "29500",
}
assert "cuda.empty_cache" in CALLS

initialized["value"] = False
os.environ["MASTER_ADDR"] = ""
os.environ["MASTER_PORT"] = ""
module._on_resume(None)

init_calls = [c for c in CALLS if isinstance(c, tuple) and c[0] == "init_process_group"]
assert init_calls == [
    ("init_process_group", {"backend": "nccl", "rank": 3, "world_size": 8})
]
assert os.environ["MASTER_ADDR"] == "10.0.0.1"
assert os.environ["MASTER_PORT"] == "29500"
assert CALLS.index("resume_inflight") < CALLS.index("kv_attach")
assert module._pre_snapshot_state == {}

module._pre_snapshot_state["nccl"] = {
    "backend": "nccl",
    "world_size": 8,
    "rank": 3,
    "master_addr": "stale",
    "master_port": "0",
}
module._on_resume(
    {
        "epoch": 2,
        "rendezvous": {
            "masterAddr": "10.2.2.2",
            "masterPort": 29502,
            "rank": 1,
            "worldSize": 4,
        },
    }
)
init_calls = [c for c in CALLS if isinstance(c, tuple) and c[0] == "init_process_group"]
assert init_calls[-1] == (
    "init_process_group",
    {"backend": "nccl", "rank": 1, "world_size": 4},
)
assert os.environ["MASTER_ADDR"] == "10.2.2.2"
assert module._pre_snapshot_state == {}
""")
