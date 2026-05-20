#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/redbot/redbot}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python3}"
HOST="${OPERATOR_APP_HOST:-0.0.0.0}"
PORT="${OPERATOR_APP_PORT:-8080}"

cd "$ROOT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "operator app failed: python not found or not executable: $PYTHON_BIN"
  exit 1
fi

exec "$PYTHON_BIN" -m uvicorn src.operator_app.backend.app:app --host "$HOST" --port "$PORT"
