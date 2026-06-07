#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/redbot/redbot}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python3}"
BOT_SERVICE="${BOT_SERVICE:-redbot.service}"
APP_SERVICE="${APP_SERVICE:-redbot-operator-app.service}"
WATCHDOG_TIMER="${WATCHDOG_TIMER:-redbot-watchdog.timer}"

cd "$ROOT_DIR"

echo "== Red Bot Runtime Status =="
echo "root: $ROOT_DIR"
echo

for unit in "$BOT_SERVICE" "$APP_SERVICE" "$WATCHDOG_TIMER"; do
  if systemctl cat "$unit" >/dev/null 2>&1; then
    echo "$unit: $(systemctl is-active "$unit" 2>/dev/null || true) / $(systemctl is-enabled "$unit" 2>/dev/null || true)"
  else
    echo "$unit: not installed"
  fi
done

echo
echo "== Bot process =="
pgrep -af "python3? -m src.main" || echo "no src.main process found"

echo
echo "== Candle freshness =="
set +e
"$PYTHON_BIN" -m src.maintenance.runtime_healthcheck --max-age-min 30
HEALTH_RC=$?
set -e
echo "health_exit_code: $HEALTH_RC"

if systemctl cat "$BOT_SERVICE" >/dev/null 2>&1; then
  echo
  echo "== Recent bot service log =="
  journalctl -u "$BOT_SERVICE" -n 30 --no-pager
fi
