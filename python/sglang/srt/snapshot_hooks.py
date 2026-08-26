# SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-snapshot / post-restore hooks for SGLang under Dynamo.

Wired in by setting ``SGLANG_SNAPSHOT_HOOKS=1`` in the runtime image. When
the framework-neutral ``criu_snapshot_hooks`` package
(``ai-blaise/criu-snapshots``) is importable, this module is thin glue
over it: the package owns the trigger mechanisms — the ``SIGRTMIN+5`` /
``SIGRTMIN+6`` signal pair plus the upstream-Dynamo-compatible signalFile
trigger — the control-directory acknowledgment protocol,
``torch.distributed`` teardown/rebuild, and POSIX semaphore preservation.
This module contributes only the SGLang residue (KV-router detach/attach,
scheduler drain/unpause) as the ``on_quiesce`` / ``on_resume`` callbacks.
Without the package, the original inline signal handlers below stay in
force; the legacy path retires when the runtime image ships
criu-snapshot-hooks.

Hard constraints — see
``ai-blaise/criu-snapshots/docs/hard-constraints.md``:

* NCCL state is not snapshottable; quiesce MUST destroy every process
  group, and resume MUST rebuild them.
* Quiesce must complete within ~5s. ``cuda-checkpoint --action lock`` has
  its own 10s timeout layered on top.
* The hooks must be idempotent; the agent may retry triggers.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import signal
import sys
import time
from types import ModuleType
from typing import TYPE_CHECKING, Any

try:
    import criu_snapshot_hooks
except ImportError:
    criu_snapshot_hooks = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from criu_snapshot_hooks.hooks import Hooks
    from criu_snapshot_hooks.mechanisms import SignalFileMechanism, SignalMechanism

NEUTRAL_PACKAGE: ModuleType | None = criu_snapshot_hooks

logger = logging.getLogger(__name__)

READY_FILE = "/var/run/dynamo/pre_snapshot.ready"
READY_ERR_FILE = READY_FILE + ".err"
RESUME_FILE = "/var/run/dynamo/post_restore.done"

# Python's signal.SIGRTMIN (glibc value 34) does not exist on macOS; the
# contract constants pin the identical Linux numbers 39/40.
_LINUX_SIGRTMIN = 34

if NEUTRAL_PACKAGE is not None:
    PRE_SNAPSHOT_SIGNAL: int = int(NEUTRAL_PACKAGE.DEFAULT_QUIESCE_SIGNAL)
    POST_RESUME_SIGNAL: int = int(NEUTRAL_PACKAGE.DEFAULT_RESUME_SIGNAL)
else:
    PRE_SNAPSHOT_SIGNAL = int(getattr(signal, "SIGRTMIN", _LINUX_SIGRTMIN)) + 5
    POST_RESUME_SIGNAL = int(getattr(signal, "SIGRTMIN", _LINUX_SIGRTMIN)) + 6

_RENDEZVOUS_KEYS = ("master_addr", "master_port", "rank", "world_size", "backend")

_pre_snapshot_state: dict[str, Any] = {}

_hooks: Hooks | None = None
_signal_mechanism: SignalMechanism | None = None
_signal_file_mechanism: SignalFileMechanism | None = None


def _neutral() -> ModuleType:
    """Return the neutral package; only reachable on the delegated path."""
    if NEUTRAL_PACKAGE is None:
        raise RuntimeError("criu_snapshot_hooks is not importable")
    return NEUTRAL_PACKAGE


def _import_kv_router() -> Any | None:
    try:
        runtime = importlib.import_module("dynamo_runtime")
    except ModuleNotFoundError as exc:
        if exc.name == "dynamo_runtime":
            return None
        raise
    return runtime.kv_router


def _drain_kv_router(timeout_s: float) -> None:
    """Detach this replica from the Dynamo KV router event queue.

    Imported lazily because some production images embed the Dynamo sidecar
    path differently from the worker runtime. Missing Dynamo bindings are a
    no-op; present bindings must drain successfully.
    """
    kv_router = _import_kv_router()
    if kv_router is None:
        logger.info(
            "dynamo_runtime.kv_router is not available; skipping KV router drain"
        )
        return

    kv_router.detach_current_replica(timeout_s=timeout_s)


def _attach_kv_router() -> None:
    kv_router = _import_kv_router()
    if kv_router is None:
        logger.info(
            "dynamo_runtime.kv_router is not available; skipping KV router attach"
        )
        return

    kv_router.attach_current_replica()


def _drain_inflight(deadline_s: float) -> None:
    scheduler = sys.modules.get("sglang.srt.managers.scheduler")
    if scheduler is None:
        logger.info("scheduler module is not loaded; relying on controller-side drain")
        return
    drain = getattr(scheduler, "drain_inflight", None)
    if drain is None:
        logger.info(
            "scheduler.drain_inflight is not registered; relying on controller-side drain"
        )
        return
    drain(deadline_s=deadline_s)


def _resume_inflight() -> None:
    scheduler = sys.modules.get("sglang.srt.managers.scheduler")
    if scheduler is None:
        logger.info(
            "scheduler module is not loaded; relying on controller-side unpause"
        )
        return
    resume = getattr(scheduler, "resume_inflight", None)
    if resume is None:
        logger.info(
            "scheduler.resume_inflight is not registered; relying on controller-side unpause"
        )
        return
    resume()


def _warm_collectives() -> None:
    # Warm the rings / trees so the first request-path collective is
    # not the one paying the bootstrap latency.
    import torch
    import torch.distributed as dist

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    t = torch.zeros(1, device=device)
    dist.all_reduce(t)
    dist.barrier()


def _capture_process_group_state() -> dict[str, Any] | None:
    """Record rendezvous material for an in-place resume on the source replica."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return None
    return {
        "backend": dist.get_backend(),
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "master_addr": os.environ.get("MASTER_ADDR", ""),
        "master_port": os.environ.get("MASTER_PORT", ""),
    }


def _rendezvous_from_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map the agent's ``resume.json`` payload onto ``torch_dist`` rebuild keys."""
    if not payload:
        return None
    rendezvous = payload.get("rendezvous")
    if isinstance(rendezvous, dict):
        return {
            "master_addr": rendezvous["masterAddr"],
            "master_port": rendezvous["masterPort"],
            "rank": rendezvous["rank"],
            "world_size": rendezvous["worldSize"],
            "backend": rendezvous.get("backend", "nccl"),
        }
    if all(key in payload for key in _RENDEZVOUS_KEYS):
        return {key: payload[key] for key in _RENDEZVOUS_KEYS}
    return None


def _on_quiesce() -> None:
    """Quiesce callback: SGLang drain residue around delegated teardown.

    Runs under ``criu_snapshot_hooks.hooks.Hooks``, which serializes the
    flows and writes the ``quiesced`` / ``error`` acknowledgments.
    """
    import torch

    _drain_kv_router(timeout_s=2.0)
    _drain_inflight(deadline_s=5.0)
    torch.cuda.synchronize()

    state = _capture_process_group_state()
    if state is not None:
        _pre_snapshot_state["nccl"] = state
    if _neutral().torch_dist.destroy_process_groups():
        torch.cuda.synchronize()

    # Surrender any cuBLAS / cuDNN workspace handles held by torch's
    # caching allocator. cuda-checkpoint can dump the underlying VRAM
    # regardless, but explicit release keeps the snapshot smaller.
    torch.cuda.empty_cache()


def _on_resume(payload: dict[str, Any] | None) -> None:
    """Resume callback: delegated rebuild followed by SGLang re-attach residue.

    Prefers the agent's resume payload (fresh rendezvous after a gang
    restore) and falls back to the state captured at quiesce for the
    in-place resume on the source replica.
    """
    rendezvous = _rendezvous_from_payload(payload)
    captured = _pre_snapshot_state.pop("nccl", None)
    if rendezvous is None:
        rendezvous = captured
    if rendezvous is not None:
        _neutral().torch_dist.rebuild_process_groups(rendezvous)
        _warm_collectives()
    _resume_inflight()
    _attach_kv_router()


def _install_neutral() -> None:
    """Bind the neutral mechanisms around the SGLang residue callbacks."""
    global _hooks, _signal_mechanism, _signal_file_mechanism

    package = _neutral()
    if _hooks is None:
        hooks = package.hooks.Hooks(on_quiesce=_on_quiesce, on_resume=_on_resume)
        _signal_mechanism = package.mechanisms.SignalMechanism(
            hooks,
            quiesce_signal=PRE_SNAPSHOT_SIGNAL,
            resume_signal=POST_RESUME_SIGNAL,
        )
        _signal_file_mechanism = package.mechanisms.SignalFileMechanism(hooks)
        _hooks = hooks
    _signal_mechanism.install()
    _signal_file_mechanism.start()
    # The legacy path has always declared the worker snapshot-eligible at
    # this point: scheduler-init side-effect import, handlers live, before
    # any traffic flows.
    _hooks.control.write_ready()
    logger.info(
        "CRIU snapshot hooks installed via criu_snapshot_hooks "
        "(signals %d/%d + signalFile, control dir %s)",
        PRE_SNAPSHOT_SIGNAL,
        POST_RESUME_SIGNAL,
        _hooks.control.path,
    )


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.rename(tmp, path)


def _write_ready(payload: dict[str, Any]) -> None:
    """Atomically signal the agent that pre_snapshot finished."""
    payload = dict(payload, pid=os.getpid())
    _atomic_write_json(READY_FILE, payload)
    _atomic_write_json(f"{READY_FILE}.{os.getpid()}", payload)


def _write_ready_error(message: str) -> None:
    for path in (READY_ERR_FILE, f"{READY_ERR_FILE}.{os.getpid()}"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(message)


def _destroy_nccl() -> dict[str, Any] | None:
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return None
    state = {
        "backend": dist.get_backend(),
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "master_addr": os.environ.get("MASTER_ADDR", ""),
        "master_port": os.environ.get("MASTER_PORT", ""),
    }
    dist.destroy_process_group()
    torch.cuda.synchronize()
    return state


def _restore_nccl(state: dict[str, Any]) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = state["master_addr"]
    os.environ["MASTER_PORT"] = state["master_port"]
    dist.init_process_group(
        backend=state["backend"],
        world_size=state["world_size"],
        rank=state["rank"],
    )
    _warm_collectives()


def _pre_snapshot_handler(signum: int, frame: Any) -> None:
    """SIGRTMIN+5: drain, destroy NCCL, signal ready.

    Designed to complete in well under 5 seconds for the deepseek-v32-reap
    profile; the ``deadline_s`` and ``timeout_s`` defaults are calibrated
    for the standard SGLang scheduler queue depth.
    """
    del signum, frame
    import torch  # noqa: WPS433

    try:
        _drain_kv_router(timeout_s=2.0)
        _drain_inflight(deadline_s=5.0)
        torch.cuda.synchronize()

        nccl_state = _destroy_nccl()
        if nccl_state is not None:
            _pre_snapshot_state["nccl"] = nccl_state

        # Surrender any cuBLAS / cuDNN workspace handles held by torch's
        # caching allocator. cuda-checkpoint can dump the underlying VRAM
        # regardless, but explicit release keeps the snapshot smaller.
        torch.cuda.empty_cache()

        _write_ready({"ts": time.time(), "ranks": _pre_snapshot_state.get("nccl")})
        logger.info("pre_snapshot complete; agent may proceed")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("pre_snapshot failed")
        _write_ready_error(repr(exc))


def _post_resume_handler(signum: int, frame: Any) -> None:
    """SIGRTMIN+6: NCCL rebind, KV router re-attach.

    Fires twice per snapshot operation: once on the SOURCE replica
    immediately after cuda-checkpoint restores the GPU state in place
    (non-destructive snapshot), and once on the RESTORED replica after
    ``criu restore`` + the CUDA plugin's RESUME_DEVICES_LATE hook.
    """
    del signum, frame
    try:
        nccl_state = _pre_snapshot_state.pop("nccl", None)
        if nccl_state is not None:
            _restore_nccl(nccl_state)
        _attach_kv_router()

        payload = str(time.time())
        for path in (RESUME_FILE, f"{RESUME_FILE}.{os.getpid()}"):
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)
        logger.info("post_resume complete; replica is serving")
    except Exception:  # pylint: disable=broad-except
        logger.exception("post_resume failed")
        os._exit(1)  # noqa: WPS437 (intentional: fast-fail for K8s restart)


def _install_legacy() -> None:
    """Register the inline signal handlers used before criu-snapshot-hooks."""
    os.makedirs(os.path.dirname(READY_FILE), exist_ok=True)
    signal.signal(PRE_SNAPSHOT_SIGNAL, _pre_snapshot_handler)
    signal.signal(POST_RESUME_SIGNAL, _post_resume_handler)
    logger.info(
        "CRIU snapshot hooks installed (SIGRTMIN+5=pre_snapshot, SIGRTMIN+6=post_resume)"
    )


def install() -> None:
    """Register the snapshot trigger mechanisms. Idempotent."""
    if NEUTRAL_PACKAGE is not None:
        _install_neutral()
    else:
        _install_legacy()


if os.environ.get("SGLANG_SNAPSHOT_HOOKS") == "1":
    install()
