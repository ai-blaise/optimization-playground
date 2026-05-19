#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODULE = REPO_ROOT / "docs/proofs/gated_norm_mma_czs_module.json"
CUTE_PROOFED_CANDIDATES = {"cute_first_supported", "production"}


def _run_czs(czs_bin: str, module_path: Path, dry_run: bool) -> int:
    module_path = module_path.resolve()
    if not module_path.is_file():
        raise FileNotFoundError(module_path)

    cmd = [czs_bin, "prove", "--json", str(module_path)]
    if dry_run:
        print(" ".join(cmd))
        return 0
    proc = subprocess.run(
        cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    sys.stdout.write(proc.stdout)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CZS over the B200 GatedNorm tensor-op proof artifact."
    )
    parser.add_argument("--module", type=Path, default=DEFAULT_MODULE)
    parser.add_argument(
        "--candidate",
        help=(
            "Compatibility option for old playground artifacts. "
            "cute_first_supported and production both run --module."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the CZS command without executing it.",
    )
    parser.add_argument(
        "--czs-bin",
        default=os.environ.get("CZS_BIN", "/root/work/CZS/build-rel/src/czs"),
        help="Path to the czs CLI, or set CZS_BIN.",
    )
    args = parser.parse_args()

    if args.candidate and args.candidate not in CUTE_PROOFED_CANDIDATES:
        valid = ", ".join(sorted(CUTE_PROOFED_CANDIDATES))
        print(
            f"{args.candidate}: no registered GatedNorm CZS proof; valid: {valid}",
            file=sys.stderr,
        )
        return 2

    return _run_czs(args.czs_bin, args.module, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
