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
  src/database_manager/database_manager.py \
  src/ai/gpt_trend_decider.py \
  src/analysis/ml_edge_model.py \
  src/analysis/indicator_edge_report.py \
  src/analysis/exit_management_report.py \
  src/analysis/position_lifecycle_report.py \
  src/analysis/recommendation_aggregator.py \
  src/analysis/operator_decision_resolver.py \
  src/analysis/recommendation_quality_tracker.py \
  src/analysis/daily_analysis_job.py \
  src/analysis/daily_control_report.py \
  src/analysis/operator_cockpit.py \
  src/analysis/operator_app_snapshot.py \
  src/analysis/operator_decisions.py \
  src/operator_app/backend/app.py \
  src/operator_app/backend/auth.py \
  src/operator_app/backend/data.py \
  src/operator_app/backend/schemas.py \
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
    "indicator_edge_report",
    "exit_management_report",
    "position_lifecycle_report",
    "risk_guard_report",
    "risk_advice_history",
    "live_readiness_gate",
    "recommendation_aggregator",
    "operator_decisions",
    "recommendation_quality_tracker",
    "pre_gpt_gate_report",
    "daily_control_report",
    "operator_cockpit",
    "operator_app_snapshot",
):
    step = steps.get(name) or {}
    result = step.get("result") or {}
    print(f"{name}: {step.get('status')}", end="")
    if name == "ml_edge_model":
        readiness = result.get("readiness") or {}
        metrics = result.get("metrics") or {}
        prediction_summary = result.get("prediction_summary") or {}
        print(
            f" readiness={readiness.get('status')} "
            f"model={result.get('model_status')} rows={readiness.get('rows')} "
            f"auc={metrics.get('classification_auc')} "
            f"acc={metrics.get('classification_accuracy')} "
            f"mae_R={metrics.get('regression_mae_r')} "
            f"avg_pred_R={prediction_summary.get('avg_predicted_r')}"
        )
    elif name == "indicator_edge_report":
        summary = result.get("summary") or {}
        top = result.get("top_feature") or summary.get("top_feature") or {}
        print(
            f" status={result.get('status')} ranked={summary.get('ranked_features')} "
            f"top={top.get('feature')} edge_R={top.get('edge_r')}"
        )
    elif name == "exit_management_report":
        summary = result.get("summary") or {}
        print(
            f" status={result.get('status')} positions={summary.get('positions_loaded')} "
            f"closed={summary.get('closed_positions')} tp1={summary.get('positions_with_tp1_proxy')} "
            f"pnl={summary.get('total_realized_pnl_eur')} verdict={summary.get('verdict')}"
        )
    elif name == "position_lifecycle_report":
        summary = result.get("summary") or {}
        print(
            f" status={result.get('status')} masters={summary.get('master_trades')} "
            f"open={summary.get('open_masters')} partial={summary.get('partial_masters')} "
            f"closed={summary.get('closed_masters')} issues={summary.get('issue_count')} "
            f"high={summary.get('high_issues')} verdict={summary.get('verdict')}"
        )
    elif name == "operator_app_snapshot":
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
    elif name == "recommendation_aggregator":
        summary = result.get("summary") or {}
        resolver = summary.get("operator_resolution") or {}
        print(
            f" status={result.get('status')} review={summary.get('needs_operator_review')} "
            f"auto_context={summary.get('auto_accept_as_context')} "
            f"wait={summary.get('wait_more_evidence')} blocked={summary.get('blocked')} "
            f"resolved_active={resolver.get('active')} suppressed={resolver.get('suppressed')} "
            f"pending_live={resolver.get('pending_live_gate')}"
        )
    elif name == "recommendation_quality_tracker":
        summary = result.get("summary") or {}
        print(
            f" status={result.get('status')} tracked={summary.get('tracked_items')} "
            f"days={summary.get('days_observed')} attention={summary.get('needs_attention')} "
            f"unstable={summary.get('unstable')}"
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
