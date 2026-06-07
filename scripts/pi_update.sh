#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/redbot/redbot}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python3}"
BOT_PATTERN="${BOT_PATTERN:-src.main}"
BOT_LOG="${BOT_LOG:-$ROOT_DIR/logs/redbot_main.log}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-master}"
SEND_COCKPIT="${SEND_COCKPIT:-1}"
SYSTEMD_SERVICE="${SYSTEMD_SERVICE:-redbot.service}"
UPDATE_LOCK="${UPDATE_LOCK:-/tmp/redbot-update-in-progress}"

cd "$ROOT_DIR"
touch "$UPDATE_LOCK"
trap 'rm -f "$UPDATE_LOCK"' EXIT

systemctl_cmd() {
  if [[ "$(id -u)" -eq 0 ]]; then
    systemctl "$@"
  else
    sudo systemctl "$@"
  fi
}

service_installed() {
  systemctl cat "$SYSTEMD_SERVICE" >/dev/null 2>&1
}

bot_pids() {
  pgrep -f "$BOT_PATTERN" || true
}

bot_status() {
  local pids
  pids="$(bot_pids)"
  if service_installed; then
    echo "systemd: $(systemctl is-active "$SYSTEMD_SERVICE" 2>/dev/null || true)"
  fi
  if [[ -z "$pids" ]]; then
    echo "bot: STOPPED"
    return 1
  fi
  echo "bot: RUNNING"
  echo "$pids" | while read -r pid; do
    [[ -z "$pid" ]] && continue
    ps -p "$pid" -o pid=,etime=,pcpu=,pmem=,cmd=
  done
}

bot_count() {
  local pids
  pids="$(bot_pids)"
  [[ -z "$pids" ]] && echo 0 && return 0
  echo "$pids" | wc -l
}

stop_bot() {
  local pids
  echo "== Stop bot =="
  if service_installed; then
    echo "systemd service detected; stopping $SYSTEMD_SERVICE"
    systemctl_cmd stop "$SYSTEMD_SERVICE"
    sleep 2
  fi

  pids="$(bot_pids)"
  if [[ -z "$pids" ]]; then
    echo "bot already stopped"
    return 0
  fi

  echo "$pids" | while read -r pid; do
    [[ -z "$pid" ]] && continue
    ps -p "$pid" -o pid=,etime=,cmd=
  done
  echo "$pids" | xargs -r kill -INT

  for _ in $(seq 1 20); do
    sleep 1
    [[ -z "$(bot_pids)" ]] && echo "bot stopped cleanly" && return 0
  done

  echo "bot still running after SIGINT; sending SIGTERM"
  bot_pids | xargs -r kill -TERM

  for _ in $(seq 1 10); do
    sleep 1
    [[ -z "$(bot_pids)" ]] && echo "bot stopped after SIGTERM" && return 0
  done

  echo "bot did not stop cleanly. I will not use SIGKILL automatically."
  echo "Manual last resort: pkill -f '$BOT_PATTERN'"
  exit 1
}

start_bot() {
  echo "== Start bot =="
  local existing_count
  existing_count="$(bot_count)"
  if service_installed; then
    if [[ "$existing_count" -gt "0" ]]; then
      echo "existing bot process found before systemd start; stopping it first"
      stop_bot
    fi
    systemctl_cmd start "$SYSTEMD_SERVICE"
    sleep 5
    if ! systemctl is-active --quiet "$SYSTEMD_SERVICE"; then
      echo "systemd start failed for $SYSTEMD_SERVICE"
      journalctl -u "$SYSTEMD_SERVICE" -n 40 --no-pager || true
      exit 1
    fi
    bot_status || true
    return 0
  fi

  if [[ "$existing_count" == "1" ]]; then
    echo "bot already running"
    bot_status || true
    return 0
  fi
  if [[ "$existing_count" -gt "1" ]]; then
    echo "start repair: found $existing_count bot processes; stopping all before starting one clean instance"
    stop_bot
  fi
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "start failed: python not found or not executable: $PYTHON_BIN"
    exit 1
  fi
  mkdir -p "$(dirname "$BOT_LOG")"
  nohup "$PYTHON_BIN" -m src.main >> "$BOT_LOG" 2>&1 &
  echo "bot started pid=$! log=$BOT_LOG"
  sleep 3
  local count
  count="$(bot_count)"
  if [[ "$count" != "1" ]]; then
    echo "start warning: expected 1 bot process, found $count"
    bot_status || true
    exit 1
  fi
}

ensure_single_bot() {
  local count
  count="$(bot_count)"
  if service_installed; then
    if systemctl is-active --quiet "$SYSTEMD_SERVICE" && [[ "$count" == "1" ]]; then
      return 0
    fi
    echo "final repair: systemd runtime is not a single healthy process; restarting cleanly"
    stop_bot
    start_bot
    return 0
  fi

  if [[ "$count" == "1" ]]; then
    return 0
  fi
  if [[ "$count" == "0" ]]; then
    echo "final repair: bot is stopped; starting one instance"
    start_bot
    return 0
  fi
  echo "final repair: found $count bot processes; restarting cleanly"
  stop_bot
  start_bot
}

echo "== Red Bot Update =="
echo "root: $ROOT_DIR"
echo "remote: $REMOTE/$BRANCH"
echo

stop_bot
echo

echo "== Git pull =="
git pull "$REMOTE" "$BRANCH"
echo

echo "== Smoke check =="
SEND_COCKPIT="$SEND_COCKPIT" "$ROOT_DIR/scripts/pi_smoke_check.sh"
echo

start_bot
echo

ensure_single_bot
echo

echo "== Final bot status =="
bot_status || true
echo
echo "PI UPDATE OK"
