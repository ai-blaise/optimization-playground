#!/usr/bin/env bash
# Install/verify Google Cloud SDK auth for the project VMs.

set -euo pipefail

echo "== Google Cloud =="

if ! command -v gcloud >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install --cask google-cloud-sdk
    if [ -d /opt/homebrew/share/google-cloud-sdk/bin ]; then
      export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
    fi
  else
    echo "ERROR: install gcloud: https://cloud.google.com/sdk/docs/install" >&2
    exit 1
  fi
fi

if [ -z "$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)" ]; then
  gcloud auth login
fi

if [ ! -f "$HOME/.config/gcloud/application_default_credentials.json" ]; then
  gcloud auth application-default login
fi

if gcloud projects describe blaise-478114 >/dev/null 2>&1; then
  gcloud config set project blaise-478114 >/dev/null
fi

gcloud auth list | sed -n '1,6p'
gcloud config list | sed -n '1,12p'
