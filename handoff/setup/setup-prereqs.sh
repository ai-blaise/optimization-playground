#!/usr/bin/env bash
# Verify/install baseline command-line prerequisites.

set -euo pipefail

echo "== Prerequisites =="

if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! xcode-select -p >/dev/null 2>&1; then
    echo "Installing Xcode Command Line Tools. Complete the GUI prompt, then rerun."
    xcode-select --install || true
    exit 1
  fi
  if ! command -v brew >/dev/null 2>&1; then
    echo "Installing Homebrew."
    NONINTERACTIVE=1 /bin/bash -c \
      "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ -x /opt/homebrew/bin/brew ]; then
      eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
  fi
fi

command -v git >/dev/null || {
  echo "ERROR: install git before continuing." >&2
  exit 1
}

command -v python3 >/dev/null || {
  echo "ERROR: install Python 3 before continuing." >&2
  exit 1
}

python3 -m ensurepip --upgrade >/dev/null 2>&1 || true

if command -v docker >/dev/null 2>&1; then
  docker --version || true
else
  echo "Docker is optional for this handoff, but needed for some build paths."
fi

echo "git: $(git --version)"
echo "python: $(python3 --version)"
