#!/usr/bin/env bash
# Run all optimization-playground handoff setup helpers.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$HERE/setup-prereqs.sh"
bash "$HERE/setup-github.sh"
bash "$HERE/setup-gcloud.sh"
bash "$HERE/setup-huggingface.sh"

echo
echo "Setup checks complete. Restore with:"
echo "  bash handoff/unpack-state.sh /path/to/optimization-playground-handoff-*.tar.gz"
