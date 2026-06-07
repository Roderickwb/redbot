#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/redbot/redbot}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python3}"
HOST="${OPERATOR_APP_HOST:-0.0.0.0}"
PORT="${OPERATOR_APP_PORT:-8080}"
APP_SERVICE="${APP_SERVICE:-redbot-operator-app.service}"

cd "$ROOT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "operator app failed: python not found or not executable: $PYTHON_BIN"
  exit 1
fi

if systemctl cat "$APP_SERVICE" >/dev/null 2>&1; then
  echo "systemd operator app detected; restarting $APP_SERVICE"
  if [[ "$(id -u)" -eq 0 ]]; then
    systemctl restart "$APP_SERVICE"
  else
    sudo systemctl restart "$APP_SERVICE"
  fi
  sleep 3
  systemctl --no-pager --full status "$APP_SERVICE" || true
  exit 0
fi

EXISTING_PIDS="$(pgrep -f "uvicorn src.operator_app.backend.app:app" || true)"
if [[ -n "$EXISTING_PIDS" ]]; then
  echo "existing operator app found; stopping: $EXISTING_PIDS"
  kill -INT $EXISTING_PIDS 2>/dev/null || true
  sleep 2
  EXISTING_PIDS="$(pgrep -f "uvicorn src.operator_app.backend.app:app" || true)"
  if [[ -n "$EXISTING_PIDS" ]]; then
    echo "operator app still running after SIGINT; sending SIGTERM"
    kill -TERM $EXISTING_PIDS 2>/dev/null || true
    sleep 2
  fi
fi

echo "operator app starting on http://$HOST:$PORT"
exec "$PYTHON_BIN" -m uvicorn src.operator_app.backend.app:app --host "$HOST" --port "$PORT"
