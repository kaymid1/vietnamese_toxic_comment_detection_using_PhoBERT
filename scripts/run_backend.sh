#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -d "venv" ]]; then
  source "venv/bin/activate"
elif [[ -d ".venv" ]]; then
  source ".venv/bin/activate"
else
  echo "No venv found. Create one with: python3 -m venv venv"
  exit 1
fi

export PYTHONUNBUFFERED=1

uvicorn backend.app:app --host 0.0.0.0 --port 8000
