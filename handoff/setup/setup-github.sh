#!/usr/bin/env bash
# Install/verify GitHub CLI and git identity.

set -euo pipefail

echo "== GitHub =="

if ! command -v gh >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install gh
  else
    echo "ERROR: install GitHub CLI: https://cli.github.com/" >&2
    exit 1
  fi
fi

if ! gh auth status >/dev/null 2>&1; then
  gh auth login --hostname github.com --git-protocol ssh
fi

if [ -z "$(git config --global user.name 2>/dev/null || true)" ]; then
  git config --global user.name "$(gh api user --jq .login)"
fi

if [ -z "$(git config --global user.email 2>/dev/null || true)" ]; then
  gh_user="$(gh api user --jq .login)"
  gh_id="$(gh api user --jq .id)"
  git config --global user.email "${gh_id}+${gh_user}@users.noreply.github.com"
fi

gh auth status
echo "git identity: $(git config --global user.name) <$(git config --global user.email)>"
