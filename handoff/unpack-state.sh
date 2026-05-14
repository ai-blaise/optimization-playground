#!/usr/bin/env bash
# Restore optimization-playground handoff state on a new device.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash handoff/unpack-state.sh <optimization-playground-handoff.tar.gz>" >&2
  exit 1
fi

BUNDLE="$1"
WORK="$(mktemp -d -t op-restore.XXXXXX)"
DEST_BASE="${OP_HANDOFF_BASE:-$HOME/Documents/Codex/optimization-playground-handoff}"
REPO_URL="${OP_HANDOFF_REPO_URL:-https://github.com/ai-blaise/optimization-playground.git}"
MEMORY_JSON="${AGENTMEMORY_STANDALONE_JSON:-$HOME/.agentmemory/standalone.json}"

cleanup() {
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "== Extracting bundle =="
tar xzf "$BUNDLE" -C "$WORK"
cat "$WORK/MANIFEST.txt"
echo

echo "== Preflight checks =="
command -v git >/dev/null || { echo "ERROR: git is required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 is required" >&2; exit 1; }

echo "== Restoring shared agentmemory with append-only merge =="
mkdir -p "$(dirname "$MEMORY_JSON")"
if [ -f "$WORK/agentmemory/standalone.json" ]; then
  python3 - "$WORK/agentmemory/standalone.json" "$MEMORY_JSON" <<'PY'
import datetime
import fcntl
import json
import os
import pathlib
import shutil
import sys
import tempfile

source = pathlib.Path(sys.argv[1])
dest = pathlib.Path(sys.argv[2])
lock_path = pathlib.Path(str(dest) + ".lock")
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def load(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())

with lock_path.open("w") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    src = load(source)
    dst = load(dest)
    backup = None
    if dest.exists():
        backup = dest.with_suffix(dest.suffix + f".bak-{stamp}")
        shutil.copy2(dest, backup)

    added = 0
    conflicts = 0
    for key, value in src.items():
        if key not in dst:
            dst[key] = value
            added += 1
        elif dst[key] != value:
            conflict_key = f"{key}__handoff_conflict_{stamp}"
            suffix = 1
            while conflict_key in dst:
                suffix += 1
                conflict_key = f"{key}__handoff_conflict_{stamp}_{suffix}"
            dst[conflict_key] = value
            conflicts += 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=dest.name + ".", suffix=".tmp", dir=str(dest.parent)
    )
    with os.fdopen(fd, "w") as tmp:
        json.dump(dst, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
    os.replace(tmp_name, dest)
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

print(f"source_entries={len(src)}")
print(f"dest_entries={len(dst)}")
print(f"added={added}")
print(f"conflicts_preserved={conflicts}")
if backup:
    print(f"backup={backup}")
print(f"dest={dest}")
PY
else
  echo "No agentmemory snapshot present in bundle; keeping existing $MEMORY_JSON"
fi

echo "== Cloning or verifying primary repo =="
mkdir -p "$DEST_BASE"
cd "$DEST_BASE"
if [ ! -d optimization-playground/.git ]; then
  git clone "$REPO_URL" optimization-playground
else
  echo "Repo already exists: $DEST_BASE/optimization-playground"
fi

if [ "${OP_HANDOFF_CLONE_ADJACENT:-0}" = "1" ]; then
  echo "== Cloning adjacent repos =="
  for spec in \
    "Megatron-LM https://github.com/ai-blaise/Megatron-LM.git" \
    "infrastructure https://github.com/ai-blaise/infrastructure.git" \
    "autoinfer https://github.com/ai-blaise/autoinfer.git" \
    "dynamo-prod-k8s https://github.com/ai-blaise/dynamo-prod-k8s.git" \
    "TileKernels https://github.com/deepseek-ai/TileKernels.git" \
    "cutlass https://github.com/NVIDIA/cutlass.git" \
    "intra-kernel-profiler https://github.com/yao-jz/intra-kernel-profiler.git" \
    "cutest https://github.com/aturker1/cutest.git"
  do
    name="${spec%% *}"
    url="${spec#* }"
    if [ ! -d "$name/.git" ]; then
      git clone "$url" "$name" || true
    fi
  done
fi

echo
echo "== Restore verification =="
python3 - "$MEMORY_JSON" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if path.exists():
    data = json.loads(path.read_text())
    print(f"agentmemory: {path} entries={len(data)}")
else:
    print(f"agentmemory: {path} missing")
PY
(
  cd "$DEST_BASE/optimization-playground"
  echo "repo: $(pwd)"
  git remote -v | sed -n '1,4p'
  git log -1 --oneline
  git status --short
)

cat <<'POST'

Restore complete.

Next agent steps:
1. Read handoff/README.md and handoff/MANIFEST.md.
2. Read recent shared agentmemory entries for optimization-playground.
3. Write a restore-complete memory entry.
4. Pull latest upstream state before starting implementation work.
POST
