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

NGROK_CONFIG_PATH="${NGROK_CONFIG_PATH:-$HOME/.ngrok/ngrok.yml}"
if [[ ! -f "$NGROK_CONFIG_PATH" ]]; then
  echo "[WARN] ngrok config not found at $NGROK_CONFIG_PATH"
else
  echo "[INFO] ngrok config: $NGROK_CONFIG_PATH"
fi

if [[ -z "${NGROK_AUTHTOKEN:-}" ]]; then
  echo "[WARN] NGROK_AUTHTOKEN not set in environment (using existing ngrok config if available)"
else
  echo "[INFO] NGROK_AUTHTOKEN is set"
fi

echo "[INFO] Starting FastAPI server on 0.0.0.0:8000..."
uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

cleanup() {
  echo "[INFO] Shutting down..."
  kill "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[INFO] Starting ngrok tunnel..."
ngrok http --domain=living-rare-ram.ngrok-free.app 8000
