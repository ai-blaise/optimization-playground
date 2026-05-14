#!/usr/bin/env bash
# Install/verify Hugging Face Hub CLI auth.

set -euo pipefail

echo "== Hugging Face =="

python3 -m pip install --user --upgrade huggingface_hub

user_bin="$(python3 -m site --user-base)/bin"
export PATH="$user_bin:$PATH"

if command -v hf >/dev/null 2>&1; then
  hf_cli=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  hf_cli=huggingface-cli
else
  echo "ERROR: hf CLI not found after installing huggingface_hub." >&2
  exit 1
fi

if [ ! -s "$HOME/.cache/huggingface/token" ]; then
  "$hf_cli" login
fi

"$hf_cli" whoami
echo "HF cache: ${HF_HOME:-$HOME/.cache/huggingface}"
