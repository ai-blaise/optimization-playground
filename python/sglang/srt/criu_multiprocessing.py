# SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocessing adjustments for CRIU-enabled SGLang workers."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False


def preserve_posix_semaphores_for_criu() -> None:
    """Keep Python semaphore names linkable while CRIU checkpoints the worker."""
    if os.environ.get("SGLANG_CRIU_KEEP_POSIX_SEMAPHORES") != "1":
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
