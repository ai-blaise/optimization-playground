# SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-snapshot / post-restore hooks for SGLang under Dynamo.

Wired in by setting ``SGLANG_SNAPSHOT_HOOKS=1`` in the runtime image. The
hooks register ``SIGRTMIN+5`` (pre_snapshot) and ``SIGRTMIN+6``
(post_resume) handlers in the SGLang worker process. The agent in
``ai-blaise/criu-snapshots`` raises these signals around the
``cuda-checkpoint`` and CRIU phases.

Hard constraints — see
``ai-blaise/criu-snapshots/docs/hard-constraints.md``:

* NCCL state is not snapshottable; the pre_snapshot handler MUST destroy
  every process group, and the post_resume handler MUST rebuild them.
* The pre_snapshot handler must complete within ~5s. ``cuda-checkpoint
  --action lock`` has its own 10s timeout layered on top.
* The hooks must be idempotent; the agent may retry signal delivery.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)

READY_FILE = "/var/run/dynamo/pre_snapshot.ready"
READY_ERR_FILE = READY_FILE + ".err"
RESUME_FILE = "/var/run/dynamo/post_restore.done"

PRE_SNAPSHOT_SIGNAL = signal.SIGRTMIN + 5
POST_RESUME_SIGNAL = signal.SIGRTMIN + 6

_pre_snapshot_state: dict[str, Any] = {}


def _write_ready(payload: dict[str, Any]) -> None:
    """Atomically signal the agent that pre_snapshot finished."""
    tmp = READY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.rename(tmp, READY_FILE)


def _write_ready_error(message: str) -> None:
    with open(READY_ERR_FILE, "w", encoding="utf-8") as f:
        f.write(message)


def _drain_kv_router(timeout_s: float) -> None:
    """Detach this replica from the Dynamo KV router event queue.

    Imported lazily because the dynamo_runtime package may not be on the
    import path during unit tests. A missing import is treated as fatal —
    the replica is part of a Dynamo deployment, the runtime must be there.
    """
    from dynamo_runtime import kv_router  # noqa: WPS433

    kv_router.detach_current_replica(timeout_s=timeout_s)


def _drain_inflight(deadline_s: float) -> None:
    scheduler = sys.modules.get("sglang.srt.managers.scheduler")
    if scheduler is None:
        logger.info("scheduler module is not loaded; relying on controller-side drain")
        return
    drain = getattr(scheduler, "drain_inflight", None)
    if drain is None:
        logger.info("scheduler.drain_inflight is not registered; relying on controller-side drain")
        return
    drain(deadline_s=deadline_s)


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
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = state["master_addr"]
    os.environ["MASTER_PORT"] = state["master_port"]
    dist.init_process_group(
        backend=state["backend"],
        world_size=state["world_size"],
        rank=state["rank"],
    )
    # Warm the rings / trees so the first request-path collective is
    # not the one paying the bootstrap latency.
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    t = torch.zeros(1, device=device)
    dist.all_reduce(t)
    dist.barrier()


def _attach_kv_router() -> None:
    from dynamo_runtime import kv_router  # noqa: WPS433

    kv_router.attach_current_replica()


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

        with open(RESUME_FILE, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
        logger.info("post_resume complete; replica is serving")
    except Exception:  # pylint: disable=broad-except
        logger.exception("post_resume failed")
        os._exit(1)  # noqa: WPS437 (intentional: fast-fail for K8s restart)


def install() -> None:
    """Register the two signal handlers. Idempotent."""
    os.makedirs(os.path.dirname(READY_FILE), exist_ok=True)
    signal.signal(PRE_SNAPSHOT_SIGNAL, _pre_snapshot_handler)
    signal.signal(POST_RESUME_SIGNAL, _post_resume_handler)
    logger.info(
        "CRIU snapshot hooks installed (SIGRTMIN+5=pre_snapshot, SIGRTMIN+6=post_resume)"
    )


if os.environ.get("SGLANG_SNAPSHOT_HOOKS") == "1":
    install()
