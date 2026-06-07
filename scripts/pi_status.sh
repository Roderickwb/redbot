#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/redbot/redbot}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python3}"
BOT_SERVICE="${BOT_SERVICE:-redbot.service}"
APP_SERVICE="${APP_SERVICE:-redbot-operator-app.service}"
WATCHDOG_TIMER="${WATCHDOG_TIMER:-redbot-watchdog.timer}"
LEARNING_TIMER="${LEARNING_TIMER:-redbot-learning.timer}"
VERBOSE="${VERBOSE:-0}"

if [[ "${1:-}" == "--verbose" ]]; then
  VERBOSE=1
fi

cd "$ROOT_DIR"

echo "== Red Bot Runtime Status =="

SERVICE_LINES=()
for unit in "$BOT_SERVICE" "$APP_SERVICE" "$WATCHDOG_TIMER" "$LEARNING_TIMER"; do
  if systemctl cat "$unit" >/dev/null 2>&1; then
    SERVICE_LINES+=("$unit=$(systemctl is-active "$unit" 2>/dev/null || true)/$(systemctl is-enabled "$unit" 2>/dev/null || true)")
  else
    SERVICE_LINES+=("$unit=not-installed")
  fi
done
echo "Services: ${SERVICE_LINES[*]}"

BOT_PID="$(pgrep -f "python3? -m src.main" | head -n 1 || true)"
if [[ -n "$BOT_PID" ]]; then
  echo "Bot: RUNNING | pid=$BOT_PID"
else
  echo "Bot: STOPPED"
fi

set +e
"$PYTHON_BIN" -m src.maintenance.runtime_healthcheck --max-age-min 30 --compact
HEALTH_RC=$?
set -e

echo
echo "== Trading & Learning =="
"$PYTHON_BIN" -m src.analysis.operator_cockpit --summary

if [[ "$VERBOSE" == "1" ]] && systemctl cat "$BOT_SERVICE" >/dev/null 2>&1; then
  echo
  echo "== Raw candle health =="
  "$PYTHON_BIN" -m src.maintenance.runtime_healthcheck --max-age-min 30 || true
  echo
  echo "== Recent bot service log =="
  journalctl -u "$BOT_SERVICE" -n 30 --no-pager
else
  echo
  echo "Technische details: bash scripts/pi_status.sh --verbose"
fi

exit "$HEALTH_RC"
