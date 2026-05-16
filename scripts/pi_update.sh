#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/redbot/redbot}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python3}"
BOT_PATTERN="${BOT_PATTERN:-python -m src.main}"
BOT_LOG="${BOT_LOG:-$ROOT_DIR/logs/redbot_main.log}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-master}"
SEND_COCKPIT="${SEND_COCKPIT:-1}"

cd "$ROOT_DIR"

bot_pids() {
  pgrep -f "$BOT_PATTERN" || true
}

bot_status() {
  local pids
  pids="$(bot_pids)"
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

stop_bot() {
  local pids
  pids="$(bot_pids)"
  if [[ -z "$pids" ]]; then
    echo "== Stop bot =="
    echo "bot already stopped"
    return 0
  fi

  echo "== Stop bot =="
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
  if bot_status >/tmp/redbot_bot_status.txt 2>&1; then
    echo "bot already running"
    cat /tmp/redbot_bot_status.txt
    return 0
  fi
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "start failed: python not found or not executable: $PYTHON_BIN"
    exit 1
  fi
  mkdir -p "$(dirname "$BOT_LOG")"
  nohup "$PYTHON_BIN" -m src.main >> "$BOT_LOG" 2>&1 &
  echo "bot started pid=$! log=$BOT_LOG"
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

echo "== Final bot status =="
bot_status || true
echo
echo "PI UPDATE OK"
