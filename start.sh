#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$REPO_ROOT/.runtime"
mkdir -p "$RUNTIME_DIR"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
WEBHOOK_PORT="${WEBHOOK_PORT:-9001}"
WEBHOOK_NGROK_DOMAIN="${WEBHOOK_NGROK_DOMAIN:-living-rare-ram.ngrok-free.app}"
START_WEBHOOK="${START_WEBHOOK:-1}"
START_WEBHOOK_NGROK="${START_WEBHOOK_NGROK:-1}"
START_FRONTEND_NGROK="${START_FRONTEND_NGROK:-0}"
SKIP_WEBHOOK_SETTINGS_SYNC="${SKIP_WEBHOOK_SETTINGS_SYNC:-0}"

if [[ "$WEBHOOK_NGROK_DOMAIN" =~ ^https?:// ]]; then
  WEBHOOK_NGROK_URL="$WEBHOOK_NGROK_DOMAIN"
else
  WEBHOOK_NGROK_URL="https://$WEBHOOK_NGROK_DOMAIN"
fi

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif [[ -x "$REPO_ROOT/venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERR] Python not found: $PYTHON_BIN"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "[ERR] npm not found in PATH."
  exit 1
fi

if [[ ( "$START_WEBHOOK" == "1" && "$START_WEBHOOK_NGROK" == "1" ) || "$START_FRONTEND_NGROK" == "1" ]]; then
  if ! command -v ngrok >/dev/null 2>&1; then
    echo "[ERR] ngrok not found in PATH."
    exit 1
  fi
fi

if [[ "$START_WEBHOOK" == "1" && "$SKIP_WEBHOOK_SETTINGS_SYNC" != "1" ]]; then
  echo "[INFO] Syncing local Kaggle webhook settings..."
  "$PYTHON_BIN" - "$REPO_ROOT" "http://127.0.0.1:$WEBHOOK_PORT/kaggle/trigger" "http://127.0.0.1:$WEBHOOK_PORT/kaggle/status" "$WEBHOOK_NGROK_URL" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
sys.path.insert(0, str(repo_root))

from backend.system_settings import DEFAULT_SETTINGS_DB_PATH, update_system_settings

DEFAULT_SETTINGS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
update_system_settings(
    DEFAULT_SETTINGS_DB_PATH,
    {
        "KAGGLE_WEBHOOK_URL": sys.argv[2],
        "KAGGLE_STATUS_WEBHOOK_URL": sys.argv[3],
        "KAGGLE_BUNDLE_PUBLIC_BASE_URL": sys.argv[4],
    },
    updated_by="start.sh",
)
PY
else
  echo "[INFO] Skipping Kaggle webhook settings sync."
fi

cleanup() {
  echo "[INFO] Shutting down child processes..."
  kill "${BACKEND_PID:-}" >/dev/null 2>&1 || true
  kill "${FRONTEND_PID:-}" >/dev/null 2>&1 || true
  kill "${WEBHOOK_PID:-}" >/dev/null 2>&1 || true
  kill "${NGROK_WEBHOOK_PID:-}" >/dev/null 2>&1 || true
  kill "${NGROK_FRONTEND_PID:-}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[INFO] Starting backend on :$BACKEND_PORT"
(
  cd "$REPO_ROOT"
  exec "$PYTHON_BIN" -m uvicorn backend.app:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload
) >"$RUNTIME_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

if [[ "$START_WEBHOOK" == "1" ]]; then
  echo "[INFO] Starting Kaggle webhook receiver on :$WEBHOOK_PORT"
  (
    cd "$REPO_ROOT"
    exec "$PYTHON_BIN" -m uvicorn backend.kaggle_webhook_receiver:app --host 0.0.0.0 --port "$WEBHOOK_PORT" --reload
  ) >"$RUNTIME_DIR/webhook-receiver.log" 2>&1 &
  WEBHOOK_PID=$!
fi

echo "[INFO] Starting frontend on :$FRONTEND_PORT"
(
  cd "$REPO_ROOT/comprehensive_ui"
  if [[ ! -d node_modules ]]; then
    npm install
  fi
  exec npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT"
) >"$RUNTIME_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

if [[ "$START_WEBHOOK" == "1" && "$START_WEBHOOK_NGROK" == "1" ]]; then
  echo "[INFO] Starting ngrok webhook tunnel https://$WEBHOOK_NGROK_DOMAIN -> localhost:$WEBHOOK_PORT"
  (
    cd "$REPO_ROOT"
    exec ngrok http --url="$WEBHOOK_NGROK_URL" --pooling-enabled=true "$WEBHOOK_PORT"
  ) >"$RUNTIME_DIR/ngrok-webhook.log" 2>&1 &
  NGROK_WEBHOOK_PID=$!
fi

if [[ "$START_FRONTEND_NGROK" == "1" ]]; then
  echo "[INFO] Starting ngrok frontend tunnel -> localhost:$FRONTEND_PORT"
  (
    cd "$REPO_ROOT"
    exec ngrok http "$FRONTEND_PORT"
  ) >"$RUNTIME_DIR/ngrok-frontend.log" 2>&1 &
  NGROK_FRONTEND_PID=$!
fi

echo "[INFO] Services started."
echo "[INFO] Backend:   http://127.0.0.1:$BACKEND_PORT"
echo "[INFO] Frontend:  http://127.0.0.1:$FRONTEND_PORT"
echo "[INFO] Webhook:   http://127.0.0.1:$WEBHOOK_PORT"
echo "[INFO] Ngrok URL: $WEBHOOK_NGROK_URL"
echo "[INFO] Logs dir:  $RUNTIME_DIR"

SERVICE_PIDS=("$BACKEND_PID" "$FRONTEND_PID")
if [[ -n "${WEBHOOK_PID:-}" ]]; then
  SERVICE_PIDS+=("$WEBHOOK_PID")
fi
if [[ -n "${NGROK_WEBHOOK_PID:-}" ]]; then
  SERVICE_PIDS+=("$NGROK_WEBHOOK_PID")
fi
if [[ -n "${NGROK_FRONTEND_PID:-}" ]]; then
  SERVICE_PIDS+=("$NGROK_FRONTEND_PID")
fi

# Bash 3.2, which is still the macOS system shell, does not support wait -n.
# Polling keeps this launcher compatible with the default macOS Bash.
while true; do
  for pid in "${SERVICE_PIDS[@]}"; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "[ERR] Service process exited: PID=$pid"
      exit 1
    fi
  done
  sleep 2
done
