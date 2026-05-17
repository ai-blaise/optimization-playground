#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODULE = REPO_ROOT / "docs/proofs/gated_norm_mma_czs_module.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CZS over the B200 GatedNorm tensor-op proof artifact."
    )
    parser.add_argument("--module", type=Path, default=DEFAULT_MODULE)
    parser.add_argument(
        "--czs-bin",
        default=os.environ.get("CZS_BIN", "/root/work/CZS/build-rel/src/czs"),
        help="Path to the czs CLI, or set CZS_BIN.",
    )
    args = parser.parse_args()

    module_path = args.module.resolve()
    if not module_path.is_file():
        raise FileNotFoundError(module_path)

    cmd = [args.czs_bin, "prove", "--json", str(module_path)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sys.stdout.write(proc.stdout)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
