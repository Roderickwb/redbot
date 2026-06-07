#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/redbot/redbot}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python3}"
UPDATE_LOCK="${UPDATE_LOCK:-/tmp/redbot-update-in-progress}"
LEARNING_LOCK="${LEARNING_LOCK:-/tmp/redbot-learning-cycle.lock}"
OUTPUT_FILE="${OUTPUT_FILE:-/tmp/redbot_learning_cycle.json}"
LOG_FILE="${LOG_FILE:-/tmp/redbot_learning_cycle.log}"

cd "$ROOT_DIR"

if [[ -f "$UPDATE_LOCK" ]]; then
  echo "learning cycle skipped: Pi update in progress"
  exit 0
fi

exec 9>"$LEARNING_LOCK"
if ! flock -n 9; then
  echo "learning cycle skipped: previous cycle still running"
  exit 0
fi

if "$PYTHON_BIN" -m src.analysis.daily_analysis_job > "$OUTPUT_FILE" 2> "$LOG_FILE"; then
  "$PYTHON_BIN" - "$OUTPUT_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    report = json.load(f)

steps = report.get("steps") or {}
learning = (steps.get("strategy_learning") or {}).get("result") or {}
labels = learning.get("label_stats") or {}
outcomes = (steps.get("adaptive_restriction_outcomes") or {}).get("result") or {}
summary = outcomes.get("summary") or {}

print(
    "learning cycle:"
    f" status={report.get('status')}"
    f" labels_updated={labels.get('updated', 0)}"
    f" pending_loaded={labels.get('loaded', 0)}"
    f" restrictions={summary.get('active_restrictions', 0)}"
    f" labeled_events={summary.get('labeled_events', 0)}"
    f" ready={summary.get('ready_for_review', 0)}"
    f" delta_R={summary.get('delta_r', 0.0)}"
)
PY
else
  echo "learning cycle failed; last output:"
  tail -n 40 "$LOG_FILE" || true
  tail -n 80 "$OUTPUT_FILE" || true
  exit 1
fi
