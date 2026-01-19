#!/usr/bin/env bash
set -euo pipefail

if ! command -v ngrok >/dev/null 2>&1; then
  echo "ngrok not found in PATH. Install or add it to PATH first."
  exit 1
fi

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

if [[ -f "$NGROK_CONFIG_PATH" ]]; then
  ngrok start --all --config "$NGROK_CONFIG_PATH"
else
  ngrok start --all
fi
