#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/redbot/redbot}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python3}"
BOT_SERVICE="${BOT_SERVICE:-redbot.service}"
MAX_CANDLE_AGE_MIN="${MAX_CANDLE_AGE_MIN:-30}"
RESTART_COOLDOWN_SEC="${RESTART_COOLDOWN_SEC:-1800}"
UPDATE_LOCK="${UPDATE_LOCK:-/tmp/redbot-update-in-progress}"
STATE_FILE="${STATE_FILE:-/run/redbot-runtime-watchdog.last_restart}"

cd "$ROOT_DIR"

log() {
  echo "[redbot-watchdog] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
}

if [[ -e "$UPDATE_LOCK" ]]; then
  log "update in progress; watchdog waits"
  exit 0
fi

if ! systemctl is-active --quiet "$BOT_SERVICE"; then
  log "$BOT_SERVICE is not active; leaving explicit service stop untouched"
  exit 0
fi

set +e
HEALTH_OUTPUT="$("$PYTHON_BIN" -m src.maintenance.runtime_healthcheck --max-age-min "$MAX_CANDLE_AGE_MIN" 2>&1)"
HEALTH_RC=$?
set -e
log "health rc=$HEALTH_RC $HEALTH_OUTPUT"

if [[ "$HEALTH_RC" -eq 0 ]]; then
  exit 0
fi

NOW_EPOCH="$(date +%s)"
LAST_RESTART=0
if [[ -f "$STATE_FILE" ]]; then
  LAST_RESTART="$(cat "$STATE_FILE" 2>/dev/null || echo 0)"
fi
if ! [[ "$LAST_RESTART" =~ ^[0-9]+$ ]]; then
  LAST_RESTART=0
fi

if (( NOW_EPOCH - LAST_RESTART < RESTART_COOLDOWN_SEC )); then
  log "runtime unhealthy, but restart cooldown is active"
  exit 0
fi

log "runtime unhealthy; restarting $BOT_SERVICE"
echo "$NOW_EPOCH" > "$STATE_FILE"
systemctl restart "$BOT_SERVICE"
sleep 15
systemctl is-active --quiet "$BOT_SERVICE"
