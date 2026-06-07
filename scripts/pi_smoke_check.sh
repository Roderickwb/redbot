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
  src/analysis/loss_diagnosis_report.py \
  src/analysis/entry_rule_candidate_simulator.py \
  src/analysis/per_coin_learning_loop.py \
  src/analysis/adaptive_restrictions.py \
  src/analysis/adaptive_restriction_outcome_tracker.py \
  src/maintenance/runtime_healthcheck.py \
  src/analysis/indicator_edge_report.py \
  src/analysis/learning_context_integrator.py \
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
  src/exchange/kraken/kraken_mixed_client.py \
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

echo "== Kraken candle rounding =="
"$PYTHON_BIN" - <<'PY'
from datetime import datetime, timezone
from src.exchange.kraken.kraken_mixed_client import _round_bar_end_timestamp

def rounded(iso_utc: str, interval: int) -> str:
    dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    end_ms = _round_bar_end_timestamp(dt.timestamp(), interval)
    return datetime.fromtimestamp(end_ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")

assert rounded("2026-06-07T13:38:55Z", 15) == "2026-06-07T13:45:00Z"
assert rounded("2026-06-07T21:02:03Z", 15) == "2026-06-07T21:15:00Z"
assert rounded("2026-06-07T23:59:59Z", 15) == "2026-06-08T00:00:00Z"
assert rounded("2026-06-07T21:02:03Z", 240) == "2026-06-08T00:00:00Z"
print("rounding: OK")
PY
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
    "loss_diagnosis_report",
    "entry_rule_candidate_simulator",
    "per_coin_learning_loop",
    "indicator_edge_report",
    "learning_context_integrator",
    "exit_management_report",
    "position_lifecycle_report",
    "risk_guard_report",
    "risk_advice_history",
    "live_readiness_gate",
    "recommendation_aggregator",
    "adaptive_restrictions",
    "adaptive_restriction_outcomes",
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
    elif name == "loss_diagnosis_report":
        summary = result.get("summary") or {}
        top_loss = result.get("top_loss") or summary.get("top_loss") or {}
        top_opp = result.get("top_opportunity") or summary.get("top_opportunity") or {}
        print(
            f" status={result.get('status')} opened={summary.get('opened_rows')} "
            f"total_R={summary.get('total_R')} candidates={summary.get('candidate_count')} "
            f"top_loss={top_loss.get('dimension')}:{top_loss.get('value')} "
            f"top_opp={top_opp.get('dimension')}:{top_opp.get('value')}"
        )
    elif name == "entry_rule_candidate_simulator":
        summary = result.get("summary") or {}
        best = result.get("best_candidate") or {}
        print(
            f" status={result.get('status')} cluster={summary.get('dimension')}:{summary.get('value')} "
            f"best={best.get('rule_id')} net_R={best.get('estimated_net_R')} "
            f"affected={best.get('affected_trades')}"
        )
    elif name == "per_coin_learning_loop":
        summary = result.get("summary") or {}
        print(
            f" status={result.get('status')} symbols={summary.get('symbols')} "
            f"actionable={summary.get('actionable')} under={summary.get('underperforming')} "
            f"opp={summary.get('opportunity')} risk_down={summary.get('risk_down_candidates')}"
        )
    elif name == "indicator_edge_report":
        summary = result.get("summary") or {}
        top = result.get("top_feature") or summary.get("top_feature") or {}
        print(
            f" status={result.get('status')} ranked={summary.get('ranked_features')} "
            f"top={top.get('feature')} edge_R={top.get('edge_r')}"
        )
    elif name == "learning_context_integrator":
        print(
            f" status={result.get('status')} profiles={result.get('profiles_updated')} "
            f"indicator={result.get('indicator_top_feature')} ml={result.get('ml_model_status')} "
            f"live_effect={result.get('live_effect')}"
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
    elif name == "adaptive_restrictions":
        summary = result.get("summary") or {}
        print(
            f" status={result.get('status')} active={summary.get('active_restrictions')} "
            f"approved={summary.get('approved_items')} supported={summary.get('approved_supported_items')} "
            f"unsupported={summary.get('approved_unsupported_items')} "
            f"coin={summary.get('coin_restrictions')} cluster={summary.get('cluster_restrictions')} "
            f"reduced_risk={summary.get('reduced_risk')} strict={summary.get('strict_confirmation')} "
            f"cooldown={summary.get('conditional_cooldown')} live_effect={summary.get('live_effect')}"
        )
    elif name == "adaptive_restriction_outcomes":
        summary = result.get("summary") or {}
        print(
            f" status={result.get('status')} active={summary.get('active_restrictions')} "
            f"with_events={summary.get('restrictions_with_events')} "
            f"labeled={summary.get('restrictions_with_labeled_outcomes')} "
            f"ready={summary.get('ready_for_review')} collecting={summary.get('collecting')} "
            f"applied={summary.get('applied_events')} sizing={summary.get('sizing_adjustments')} "
            f"skips={summary.get('pre_gpt_skips')} obs_R={summary.get('observed_r')}"
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
