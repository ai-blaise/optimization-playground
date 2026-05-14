#!/usr/bin/env bash
# Bundle optimization-playground handoff state for a new device.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
BUNDLE_DIR="$(mktemp -d -t op-handoff.XXXXXX)"
OUT="${ROOT}/optimization-playground-handoff-${STAMP}.tar.gz"

cleanup() {
  rm -rf "$BUNDLE_DIR"
}
trap cleanup EXIT

mkdir -p "$BUNDLE_DIR"/{agentmemory,git-state,repo-docs}

echo "== Capturing git state =="
if [ "${OP_HANDOFF_CAPTURE_GIT:-0}" = "1" ]; then
  python3 - "$ROOT" "$BUNDLE_DIR/git-state" <<'PY'
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])

commands = {
    "remotes.txt": ["git", "-C", str(root), "remote", "-v"],
    "branch.txt": ["git", "-C", str(root), "branch", "--show-current"],
    "head.txt": ["git", "-C", str(root), "rev-parse", "HEAD"],
    "recent-log.txt": ["git", "-C", str(root), "log", "-20", "--oneline"],
    "status-short.txt": [
        "git",
        "-C",
        str(root),
        "status",
        "--short",
        "--ignore-submodules=all",
    ],
    "worktrees.txt": ["git", "-C", str(root), "worktree", "list"],
}

for filename, command in commands.items():
    path = out_dir / filename
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        text = result.stdout
        if result.stderr:
            text += "\n[stderr]\n" + result.stderr
        if result.returncode:
            text += f"\n[exit_code] {result.returncode}\n"
    except subprocess.TimeoutExpired as exc:
        text = f"[timeout] {' '.join(command)}\n"
        if exc.stdout:
            text += str(exc.stdout)
        if exc.stderr:
            text += "\n[stderr]\n" + str(exc.stderr)
    path.write_text(text)
PY
else
  cat > "$BUNDLE_DIR/git-state/README.txt" <<'EOF'
Git command capture skipped by default to keep handoff packing fast and avoid
local submodule/filesystem stalls. Run with OP_HANDOFF_CAPTURE_GIT=1 to capture
remotes, branch, HEAD, recent log, status, and worktree metadata.
EOF
fi

echo "== Capturing handoff docs =="
cp -a "$ROOT/handoff/README.md" "$BUNDLE_DIR/repo-docs/README.md"
cp -a "$ROOT/handoff/MANIFEST.md" "$BUNDLE_DIR/repo-docs/MANIFEST.md"

MEMORY_JSON="${AGENTMEMORY_STANDALONE_JSON:-$HOME/.agentmemory/standalone.json}"
echo "== Capturing shared agentmemory from $MEMORY_JSON =="
if [ -f "$MEMORY_JSON" ]; then
  cp -a "$MEMORY_JSON" "$BUNDLE_DIR/agentmemory/standalone.json"
  python3 - "$MEMORY_JSON" > "$BUNDLE_DIR/agentmemory/summary.txt" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
print(f"path: {path}")
print(f"entries: {len(data)}")
for key in sorted(data)[-50:]:
    print(key)
PY
else
  echo "No standalone agentmemory file found at $MEMORY_JSON" \
    > "$BUNDLE_DIR/agentmemory/summary.txt"
fi

cat > "$BUNDLE_DIR/MANIFEST.txt" <<EOF
optimization-playground handoff bundle
======================================
Created: $(date -Iseconds)
Source machine: $(hostname) ($(uname -m))
Repo root: $ROOT
Repo head: $(head -1 "$BUNDLE_DIR/git-state/head.txt" 2>/dev/null || echo unknown)
Repo branch: $(head -1 "$BUNDLE_DIR/git-state/branch.txt" 2>/dev/null || echo unknown)
Agentmemory source: $MEMORY_JSON

Contents:
- agentmemory/standalone.json : shared local memory snapshot, if present
- agentmemory/summary.txt     : entry count and recent keys
- git-state/                  : lightweight HEAD file by default; richer Git
                                metadata when OP_HANDOFF_CAPTURE_GIT=1
- repo-docs/                  : handoff README and manifest copied at pack time

Restore policy:
- Merge agentmemory append-only.
- Never delete destination memories.
- Preserve conflicting source keys under timestamped conflict keys.
- Clone or update the repo separately from GitHub; Git remains code source of truth.
EOF

echo "== Creating $OUT =="
tar czf "$OUT" -C "$BUNDLE_DIR" .
echo "DONE: $OUT ($(du -h "$OUT" | cut -f1))"
