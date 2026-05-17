#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/redbot/redbot}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python3}"
SEND_COCKPIT="${SEND_COCKPIT:-1}"

cd "$ROOT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "SMOKE CHECK FAILED: python not found or not executable: $PYTHON_BIN"
  exit 1
fi

echo "== Red Bot Pi Smoke Check =="
echo "root: $ROOT_DIR"
echo "python: $PYTHON_BIN"
echo

echo "== Compile critical modules =="
"$PYTHON_BIN" -m py_compile \
  src/analysis/safety_control.py \
  src/analysis/daily_analysis_job.py \
  src/analysis/daily_control_report.py \
  src/analysis/operator_cockpit.py \
  src/analysis/operator_app_snapshot.py \
  src/analysis/operator_decisions.py \
  src/analysis/risk_advice_history.py \
  src/analysis/live_readiness_gate.py \
  src/analysis/pre_gpt_gate_report.py \
  src/analysis/strategy_profile_proposer.py \
  src/analysis/coin_profile_generator.py \
  src/analysis/run_daily_coin_profiles.py \
  src/strategy/trend_strategy_4h.py
echo "compile: OK"
echo

echo "== Safety status =="
"$PYTHON_BIN" -m src.analysis.safety_control status > /tmp/redbot_safety_status.json
"$PYTHON_BIN" - <<'PY'
import json
with open("/tmp/redbot_safety_status.json", "r", encoding="utf-8") as f:
    d = json.load(f)
m = d.get("meltdown") or {}
print("safety:", d.get("status"))
print("kill_switch_active:", d.get("kill_switch_active"))
print("meltdown_active:", m.get("active"))
print("live_entry_orders_allowed:", d.get("live_entry_orders_allowed"))
print("live_enforcement_allowed:", d.get("live_enforcement_allowed"))
print("reason:", d.get("reason"))
PY
echo

echo "== Daily analysis job =="
"$PYTHON_BIN" -m src.analysis.daily_analysis_job --dry-run-labels --cleanup-registry > /tmp/redbot_daily_analysis_job.json
"$PYTHON_BIN" - <<'PY'
import json
with open("/tmp/redbot_daily_analysis_job.json", "r", encoding="utf-8") as f:
    d = json.load(f)
print("daily_status:", d.get("status"))
print("failed_steps:", d.get("failed_steps", []))
steps = d.get("steps") or {}
for name in (
    "safety_control",
    "ml_edge_model",
    "risk_guard_report",
    "risk_advice_history",
    "live_readiness_gate",
    "pre_gpt_gate_report",
    "daily_control_report",
    "operator_cockpit",
    "operator_decisions",
    "operator_app_snapshot",
):
    step = steps.get(name) or {}
    result = step.get("result") or {}
    print(f"{name}: {step.get('status')}", end="")
    if name == "operator_app_snapshot":
        summary = result.get("summary") or {}
        print(f" status={result.get('status')} cards={result.get('cards')} live_effect={summary.get('live_effect')}")
    elif name == "operator_decisions":
        summary = result.get("summary") or {}
        print(f" status={result.get('status')} total={summary.get('total')} live_effect={summary.get('live_effect')}")
    elif name == "operator_cockpit":
        decision = (result.get("daily_decision") or {}).get("label")
        print(f" decision={decision}")
    elif name == "daily_control_report":
        print(f" status={result.get('status')} blockers={result.get('blockers')} approvals={result.get('approval_queue')}")
    elif name == "risk_guard_report":
        summary = result.get("summary") or {}
        issue = (summary.get("primary_issue") or {}).get("guard")
        print(f" verdict={summary.get('verdict')} triggers={summary.get('guard_triggers')} issue={issue}")
    elif name == "risk_advice_history":
        summary = result.get("summary") or {}
        print(
            f" verdict={summary.get('verdict')} "
            f"tracked={summary.get('tracked_symbols')} "
            f"days={summary.get('days_observed')} "
            f"stable_down={summary.get('stable_data_down_symbols')}"
        )
    elif name == "live_readiness_gate":
        summary = result.get("summary") or {}
        print(
            f" eligible={summary.get('eligible_for_live_wiring')} "
            f"review={summary.get('ready_for_operator_review')} "
            f"blocked={summary.get('blocked')} waiting={summary.get('waiting')} "
            f"calibration={summary.get('calibration_only')}"
        )
    elif name == "pre_gpt_gate_report":
        summary = result.get("summary") or {}
        print(f" verdict={summary.get('verdict')} skip={summary.get('would_skip_gpt')} net_R={summary.get('estimated_net_saved_r')}")
    else:
        print("")
if d.get("failed_steps"):
    raise SystemExit(2)
PY
echo

echo "== Operator cockpit =="
if [[ "$SEND_COCKPIT" == "1" ]]; then
  "$PYTHON_BIN" -m src.analysis.operator_cockpit --send > /tmp/redbot_operator_cockpit.txt
else
  "$PYTHON_BIN" -m src.analysis.operator_cockpit > /tmp/redbot_operator_cockpit.txt
fi
cat /tmp/redbot_operator_cockpit.txt
echo

echo "SMOKE CHECK OK"
