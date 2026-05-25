# SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "python" if not pythonpath else f"python:{pythonpath}"
    return env


def test_criu_semaphore_preservation_keeps_spawn_semaphore_linked():
    code = """
import gc
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path

from _multiprocessing import sem_unlink
from multiprocessing import resource_tracker

module_path = Path("python/sglang/srt/criu_multiprocessing.py").resolve()
spec = importlib.util.spec_from_file_location("criu_multiprocessing", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

os.environ["SGLANG_CRIU_KEEP_POSIX_SEMAPHORES"] = "1"
mp.set_start_method("spawn", force=True)
module.preserve_posix_semaphores_for_criu()

sem = mp.get_context("spawn").Semaphore(1)
name = sem._semlock.name
assert name
path = Path("/dev/shm") / ("sem." + name.lstrip("/"))
assert path.exists(), path
del sem
gc.collect()
try:
    assert path.exists(), path
finally:
    try:
        resource_tracker.unregister(name, "semaphore")
        sem_unlink(name)
    except FileNotFoundError:
        pass
"""
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        env=_pythonpath_env(),
    )


def test_criu_semaphore_preservation_is_opt_in():
    code = """
import importlib.util
import multiprocessing.synchronize as mp_synchronize
import os
from pathlib import Path

module_path = Path("python/sglang/srt/criu_multiprocessing.py").resolve()
spec = importlib.util.spec_from_file_location("criu_multiprocessing", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

os.environ.pop("SGLANG_CRIU_KEEP_POSIX_SEMAPHORES", None)
before = mp_synchronize.SemLock._cleanup
module.preserve_posix_semaphores_for_criu()
assert mp_synchronize.SemLock._cleanup is before
"""
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        env=_pythonpath_env(),
    )
