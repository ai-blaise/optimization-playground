# SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocessing adjustments for CRIU-enabled SGLang workers.

When the framework-neutral ``criu_snapshot_hooks`` package is importable,
preservation delegates to ``criu_snapshot_hooks.semaphores``; the inline
fallback retires when the runtime image ships criu-snapshot-hooks.
"""

from __future__ import annotations

import logging
import os
from types import ModuleType

try:
    import criu_snapshot_hooks
except ImportError:
    criu_snapshot_hooks = None  # type: ignore[assignment]

NEUTRAL_PACKAGE: ModuleType | None = criu_snapshot_hooks

logger = logging.getLogger(__name__)

_INSTALLED = False


def preserve_posix_semaphores_for_criu() -> None:
    """Keep Python semaphore names linkable while CRIU checkpoints the worker."""
    if os.environ.get("SGLANG_CRIU_KEEP_POSIX_SEMAPHORES") != "1":
        return

    if NEUTRAL_PACKAGE is not None:
        NEUTRAL_PACKAGE.semaphores.preserve_posix_semaphores()
        return

    global _INSTALLED
    if _INSTALLED:
        return

    import multiprocessing.synchronize as mp_synchronize

    if not hasattr(mp_synchronize.SemLock, "_cleanup"):
        return

    def _preserve_cleanup(name: str) -> None:
        logger.debug("preserving POSIX semaphore %s for CRIU", name)

    mp_synchronize.SemLock._cleanup = staticmethod(_preserve_cleanup)
    _INSTALLED = True
    logger.info("CRIU POSIX semaphore preservation is enabled")


__all__ = ["preserve_posix_semaphores_for_criu"]
