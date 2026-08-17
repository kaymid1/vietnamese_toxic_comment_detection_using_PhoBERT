#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x "$REPO_ROOT/.venv/bin/python3" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python3"
elif [[ -x "$REPO_ROOT/venv/bin/python3" ]]; then
  PYTHON_BIN="$REPO_ROOT/venv/bin/python3"
else
  echo "No venv found."
  echo "Create one with: python3 -m venv .venv"
  echo "Then install runtime dependencies using the canonical split requirements:"
  echo "  ./.venv/bin/python3 -m pip install -r requirements-ml.txt"
  echo "  ./.venv/bin/python3 -m pip install -r requirements-base.txt"
  exit 1
fi

export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
