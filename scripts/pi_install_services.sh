#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/redbot/redbot}"
RUN_USER="${RUN_USER:-redbot}"
UNIT_SOURCE_DIR="$ROOT_DIR/scripts/systemd"

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

if [[ ! -x "$ROOT_DIR/venv/bin/python3" ]]; then
  echo "install failed: missing executable $ROOT_DIR/venv/bin/python3"
  exit 1
fi

for unit in redbot.service redbot-operator-app.service redbot-watchdog.service redbot-watchdog.timer; do
  if [[ ! -f "$UNIT_SOURCE_DIR/$unit" ]]; then
    echo "install failed: missing $UNIT_SOURCE_DIR/$unit"
    exit 1
  fi
done

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

render_unit() {
  local unit="$1"
  sed \
    -e "s|__ROOT_DIR__|$ROOT_DIR|g" \
    -e "s|__RUN_USER__|$RUN_USER|g" \
    "$UNIT_SOURCE_DIR/$unit" > "$TMP_DIR/$unit"
}

echo "== Install Red Bot systemd services =="
echo "root: $ROOT_DIR"
echo "user: $RUN_USER"

for unit in redbot.service redbot-operator-app.service redbot-watchdog.service redbot-watchdog.timer; do
  render_unit "$unit"
done

"${SUDO[@]}" systemctl stop redbot-watchdog.timer redbot.service redbot-operator-app.service 2>/dev/null || true

LEGACY_BOT_PIDS="$(pgrep -f "$ROOT_DIR/venv/bin/python3 -m src.main" || true)"
if [[ -n "$LEGACY_BOT_PIDS" ]]; then
  echo "stopping legacy bot process: $LEGACY_BOT_PIDS"
  kill -TERM $LEGACY_BOT_PIDS 2>/dev/null || true
  sleep 3
fi

LEGACY_APP_PIDS="$(pgrep -f "uvicorn src.operator_app.backend.app:app" || true)"
if [[ -n "$LEGACY_APP_PIDS" ]]; then
  echo "stopping legacy operator app process: $LEGACY_APP_PIDS"
  kill -TERM $LEGACY_APP_PIDS 2>/dev/null || true
  sleep 3
fi

mkdir -p "$ROOT_DIR/logs"
for unit in redbot.service redbot-operator-app.service redbot-watchdog.service redbot-watchdog.timer; do
  "${SUDO[@]}" install -m 0644 "$TMP_DIR/$unit" "/etc/systemd/system/$unit"
done

"${SUDO[@]}" systemctl daemon-reload
"${SUDO[@]}" systemctl enable --now redbot.service redbot-operator-app.service redbot-watchdog.timer
sleep 5

echo
echo "== Installed status =="
systemctl --no-pager --full status redbot.service redbot-operator-app.service redbot-watchdog.timer || true
echo
echo "Services installed. Future power restores will start the bot automatically."
