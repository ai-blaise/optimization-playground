# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


def clone_scheduler_runtime(scheduler: Any) -> Any:
    """Create an isolated scheduler runtime from a scheduler template or runtime."""
    return deepcopy(scheduler)


def get_or_create_request_scheduler(batch: Req, scheduler_template: Any) -> Any:
    """Return the request-local scheduler, cloning the template when absent."""
    if batch.scheduler is None:
        batch.scheduler = clone_scheduler_runtime(scheduler_template)
    return batch.scheduler
