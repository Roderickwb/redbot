# ============================================================
# src/analysis/daily_control_report.py
# ============================================================
"""
Daily control report.

Builds one compact operator-facing view on top of the daily analysis stack.
This report is intentionally read-only: it decides what needs attention, what
is merely collecting evidence, and what is ready for human approval.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Iterable, Optional

from dotenv import load_dotenv

from src.notifier.telegram_notifier import TelegramNotifier


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "daily_control")
DEFAULT_LATEST_FILE = "latest_daily_control_report.json"

DEFAULT_DAILY_JOB_REPORT = os.path.join("analysis", "daily", "latest_daily_analysis_job.json")
DEFAULT_ADVISOR_REPORT = os.path.join("analysis", "bot_advisor", "latest_bot_advice.json")
DEFAULT_REGISTRY_SUMMARY = os.path.join("analysis", "recommendations", "latest_recommendation_registry_summary.json")
DEFAULT_EXPERIMENT_PLAN = os.path.join("analysis", "experiments", "latest_experiment_plan.json")
DEFAULT_SHADOW_RESULTS = os.path.join("analysis", "experiments", "latest_shadow_experiment_results.json")
DEFAULT_PROMOTION_GATE = os.path.join("analysis", "promotion_gate", "latest_promotion_gate_report.json")
DEFAULT_APPROVAL_INBOX = os.path.join("analysis", "approvals", "latest_approval_inbox.json")
DEFAULT_SHADOW_LIVE_BRIDGE = os.path.join("analysis", "shadow_live", "latest_shadow_live_bridge_report.json")
DEFAULT_RISK_POLICY_REPORT = os.path.join("analysis", "risk", "latest_risk_policy_report.json")
DEFAULT_RISK_ADVICE_HISTORY = os.path.join("analysis", "risk", "latest_risk_advice_history_report.json")
DEFAULT_RISK_STRATEGY_BRIDGE = os.path.join("analysis", "risk", "latest_risk_strategy_bridge_report.json")
DEFAULT_RISK_BRIDGE_OUTCOMES = os.path.join("analysis", "risk", "latest_risk_bridge_outcome_report.json")
DEFAULT_RISK_BRIDGE_HISTORY = os.path.join("analysis", "risk", "latest_risk_bridge_history_report.json")
DEFAULT_RISK_GUARD_REPORT = os.path.join("analysis", "risk", "latest_risk_guard_report.json")
DEFAULT_ML_EDGE_REPORT = os.path.join("analysis", "ml_models", "latest_edge_model_report.json")
DEFAULT_INDICATOR_EDGE_REPORT = os.path.join("analysis", "indicator_edge", "latest_indicator_edge_report.json")
DEFAULT_EXIT_MANAGEMENT_REPORT = os.path.join("analysis", "exits", "latest_exit_management_report.json")
DEFAULT_POSITION_LIFECYCLE_REPORT = os.path.join("analysis", "positions", "latest_position_lifecycle_report.json")
DEFAULT_RECOMMENDATION_AGGREGATOR = os.path.join("analysis", "recommendations", "latest_recommendation_aggregator.json")
DEFAULT_RECOMMENDATION_QUALITY = os.path.join("analysis", "recommendations", "latest_recommendation_quality_report.json")
DEFAULT_ALERT_REPORT = os.path.join("analysis", "bot_alerts", "latest_bot_alerts_report.json")
DEFAULT_MARKET_REGIME_REPORT = os.path.join("analysis", "market_regime", "latest_market_regime.json")
DEFAULT_GPT_DECISION_REPORT = os.path.join("analysis", "gpt_decisions", "latest_gpt_decision_report.json")
DEFAULT_PRE_GPT_GATE_REPORT = os.path.join("analysis", "gpt_decisions", "latest_pre_gpt_gate_report.json")
DEFAULT_SAFETY_CONTROL_REPORT = os.path.join("analysis", "safety", "latest_safety_control_report.json")
DEFAULT_LIVE_READINESS_GATE = os.path.join("analysis", "live_readiness", "latest_live_readiness_gate.json")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {"_missing": True, "_path": path}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e), "_path": path}


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class DailyControlReport:
    def __init__(
        self,
        daily_job_path: str = DEFAULT_DAILY_JOB_REPORT,
        advisor_path: str = DEFAULT_ADVISOR_REPORT,
        registry_path: str = DEFAULT_REGISTRY_SUMMARY,
        experiment_path: str = DEFAULT_EXPERIMENT_PLAN,
        shadow_results_path: str = DEFAULT_SHADOW_RESULTS,
        promotion_gate_path: str = DEFAULT_PROMOTION_GATE,
        approval_inbox_path: str = DEFAULT_APPROVAL_INBOX,
        shadow_live_path: str = DEFAULT_SHADOW_LIVE_BRIDGE,
        risk_policy_path: str = DEFAULT_RISK_POLICY_REPORT,
        risk_advice_history_path: str = DEFAULT_RISK_ADVICE_HISTORY,
        risk_strategy_bridge_path: str = DEFAULT_RISK_STRATEGY_BRIDGE,
        risk_bridge_outcomes_path: str = DEFAULT_RISK_BRIDGE_OUTCOMES,
        risk_bridge_history_path: str = DEFAULT_RISK_BRIDGE_HISTORY,
        risk_guard_path: str = DEFAULT_RISK_GUARD_REPORT,
        ml_edge_path: str = DEFAULT_ML_EDGE_REPORT,
        indicator_edge_path: str = DEFAULT_INDICATOR_EDGE_REPORT,
        exit_management_path: str = DEFAULT_EXIT_MANAGEMENT_REPORT,
        position_lifecycle_path: str = DEFAULT_POSITION_LIFECYCLE_REPORT,
        recommendation_aggregator_path: str = DEFAULT_RECOMMENDATION_AGGREGATOR,
        recommendation_quality_path: str = DEFAULT_RECOMMENDATION_QUALITY,
        alerts_path: str = DEFAULT_ALERT_REPORT,
        market_regime_path: str = DEFAULT_MARKET_REGIME_REPORT,
        gpt_decision_path: str = DEFAULT_GPT_DECISION_REPORT,
        pre_gpt_gate_path: str = DEFAULT_PRE_GPT_GATE_REPORT,
        safety_control_path: str = DEFAULT_SAFETY_CONTROL_REPORT,
        live_readiness_path: str = DEFAULT_LIVE_READINESS_GATE,
    ):
        self.daily_job_path = daily_job_path
        self.advisor_path = advisor_path
        self.registry_path = registry_path
        self.experiment_path = experiment_path
        self.shadow_results_path = shadow_results_path
        self.promotion_gate_path = promotion_gate_path
        self.approval_inbox_path = approval_inbox_path
        self.shadow_live_path = shadow_live_path
        self.risk_policy_path = risk_policy_path
        self.risk_advice_history_path = risk_advice_history_path
        self.risk_strategy_bridge_path = risk_strategy_bridge_path
        self.risk_bridge_outcomes_path = risk_bridge_outcomes_path
        self.risk_bridge_history_path = risk_bridge_history_path
        self.risk_guard_path = risk_guard_path
        self.ml_edge_path = ml_edge_path
        self.indicator_edge_path = indicator_edge_path
        self.exit_management_path = exit_management_path
        self.position_lifecycle_path = position_lifecycle_path
        self.recommendation_aggregator_path = recommendation_aggregator_path
        self.recommendation_quality_path = recommendation_quality_path
        self.alerts_path = alerts_path
        self.market_regime_path = market_regime_path
        self.gpt_decision_path = gpt_decision_path
        self.pre_gpt_gate_path = pre_gpt_gate_path
        self.safety_control_path = safety_control_path
        self.live_readiness_path = live_readiness_path

    def build_report(self) -> dict:
        reports = {
            "daily_job": load_json(self.daily_job_path),
            "advisor": load_json(self.advisor_path),
            "registry": load_json(self.registry_path),
            "experiments": load_json(self.experiment_path),
            "shadow_results": load_json(self.shadow_results_path),
            "promotion_gate": load_json(self.promotion_gate_path),
            "approval_inbox": load_json(self.approval_inbox_path),
            "shadow_live": load_json(self.shadow_live_path),
            "risk_policy": load_json(self.risk_policy_path),
            "risk_advice_history": load_json(self.risk_advice_history_path),
            "risk_strategy_bridge": load_json(self.risk_strategy_bridge_path),
            "risk_bridge_outcomes": load_json(self.risk_bridge_outcomes_path),
            "risk_bridge_history": load_json(self.risk_bridge_history_path),
            "risk_guard": load_json(self.risk_guard_path),
            "ml_edge": load_json(self.ml_edge_path),
            "indicator_edge": load_json(self.indicator_edge_path),
            "exit_management": load_json(self.exit_management_path),
            "position_lifecycle": load_json(self.position_lifecycle_path),
            "recommendation_aggregator": load_json(self.recommendation_aggregator_path),
            "recommendation_quality": load_json(self.recommendation_quality_path),
            "alerts": load_json(self.alerts_path),
            "market_regime": load_json(self.market_regime_path),
            "gpt_decisions": load_json(self.gpt_decision_path),
            "pre_gpt_gate": load_json(self.pre_gpt_gate_path),
            "safety_control": load_json(self.safety_control_path),
            "live_readiness": load_json(self.live_readiness_path),
        }

        blockers = self._blockers(reports)
        approval_queue = self._approval_queue(reports)
        experiment_status = self._experiment_status(reports)
        promotion_status = self._promotion_status(reports)
        approval_status = self._approval_status(reports)
        shadow_live_status = self._shadow_live_status(reports)
        risk_policy_status = self._risk_policy_status(reports)
        risk_advice_history_status = self._risk_advice_history_status(reports)
        risk_strategy_status = self._risk_strategy_status(reports)
        risk_outcome_status = self._risk_outcome_status(reports)
        risk_history_status = self._risk_history_status(reports)
        risk_guard_status = self._risk_guard_status(reports)
        exit_management_status = self._exit_management_status(reports)
        position_lifecycle_status = self._position_lifecycle_status(reports)
        gpt_efficiency_status = self._gpt_efficiency_status(reports)
        pre_gpt_gate_status = self._pre_gpt_gate_status(reports)
        safety_status = self._safety_status(reports)
        live_readiness_status = self._live_readiness_status(reports)
        learning_status = self._learning_status(reports)
        recommendation_status = self._recommendation_status(reports)
        recommendation_quality_status = self._recommendation_quality_status(reports)
        operating_state = self._operating_state(reports, blockers, approval_queue)
        next_actions = self._next_actions(blockers, approval_queue, experiment_status, promotion_status, approval_status, shadow_live_status, risk_policy_status, risk_advice_history_status, risk_strategy_status, risk_outcome_status, risk_history_status, risk_guard_status, exit_management_status, position_lifecycle_status, gpt_efficiency_status, pre_gpt_gate_status, safety_status, live_readiness_status, learning_status, recommendation_quality_status)

        return {
            "created_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": operating_state["status"],
            "operator_summary": operating_state,
            "blockers": blockers,
            "approval_queue": approval_queue,
            "learning_status": learning_status,
            "recommendation_status": recommendation_status,
            "recommendation_quality_status": recommendation_quality_status,
            "experiment_status": experiment_status,
            "promotion_status": promotion_status,
            "approval_status": approval_status,
            "shadow_live_status": shadow_live_status,
            "risk_policy_status": risk_policy_status,
            "risk_advice_history_status": risk_advice_history_status,
            "risk_strategy_status": risk_strategy_status,
            "risk_outcome_status": risk_outcome_status,
            "risk_history_status": risk_history_status,
            "risk_guard_status": risk_guard_status,
            "exit_management_status": exit_management_status,
            "position_lifecycle_status": position_lifecycle_status,
            "gpt_efficiency_status": gpt_efficiency_status,
            "pre_gpt_gate_status": pre_gpt_gate_status,
            "safety_status": safety_status,
            "live_readiness_status": live_readiness_status,
            "next_actions": next_actions,
            "sources": {
                "daily_job": self.daily_job_path,
                "advisor": self.advisor_path,
                "registry": self.registry_path,
                "experiment_plan": self.experiment_path,
                "shadow_results": self.shadow_results_path,
                "promotion_gate": self.promotion_gate_path,
                "approval_inbox": self.approval_inbox_path,
                "shadow_live_bridge": self.shadow_live_path,
                "risk_policy": self.risk_policy_path,
                "risk_advice_history": self.risk_advice_history_path,
                "risk_strategy_bridge": self.risk_strategy_bridge_path,
                "risk_bridge_outcomes": self.risk_bridge_outcomes_path,
                "risk_bridge_history": self.risk_bridge_history_path,
                "risk_guard": self.risk_guard_path,
                "ml_edge_model": self.ml_edge_path,
                "indicator_edge": self.indicator_edge_path,
                "exit_management": self.exit_management_path,
                "position_lifecycle": self.position_lifecycle_path,
                "recommendation_aggregator": self.recommendation_aggregator_path,
                "recommendation_quality": self.recommendation_quality_path,
                "alerts": self.alerts_path,
                "market_regime": self.market_regime_path,
                "gpt_decisions": self.gpt_decision_path,
                "pre_gpt_gate": self.pre_gpt_gate_path,
                "safety_control": self.safety_control_path,
                "live_readiness": self.live_readiness_path,
            },
        }

    def _blockers(self, reports: dict) -> list[dict]:
        blockers = []
        for name, report in reports.items():
            if name in {"live_readiness", "risk_advice_history", "indicator_edge", "exit_management", "position_lifecycle", "recommendation_aggregator", "recommendation_quality"} and report.get("_missing"):
                continue
            if report.get("_missing"):
                blockers.append({
                    "level": "high",
                    "area": "pipeline",
                    "finding": f"{name} report is missing.",
                    "action": f"Run the daily analysis job or inspect {report.get('_path')}.",
                })
            elif report.get("_error"):
                blockers.append({
                    "level": "high",
                    "area": "pipeline",
                    "finding": f"{name} report could not be read.",
                    "action": "Inspect the JSON output and logs.",
                    "error": report.get("_error"),
                })

        daily_job = reports.get("daily_job", {}) or {}
        failed_steps = daily_job.get("failed_steps", []) or []
        for step in failed_steps:
            blockers.append({
                "level": "high",
                "area": "daily_job",
                "finding": f"Daily analysis step failed: {step}.",
                "action": "Inspect analysis/daily/latest_daily_analysis_job.json and the daily cron log.",
            })

        alerts = reports.get("alerts", {}) or {}
        for alert in alerts.get("alerts", []) or []:
            if alert.get("level") == "ALERT":
                blockers.append({
                    "level": "high",
                    "area": "runtime",
                    "finding": alert.get("message"),
                    "action": "Fix runtime/data health before approving strategy changes.",
                    "code": alert.get("code"),
                })

        safety = reports.get("safety_control", {}) or {}
        if safety.get("kill_switch_active"):
            blockers.append({
                "level": "high",
                "area": "safety",
                "finding": f"Kill-switch is active: {safety.get('reason')}",
                "action": "Live entry orders are blocked until the kill-switch is cleared explicitly.",
                "code": "KILL_SWITCH_ACTIVE",
            })
        meltdown = safety.get("meltdown") or {}
        if meltdown.get("active"):
            blockers.append({
                "level": "high",
                "area": "safety",
                "finding": f"Meltdown manager is active: {meltdown.get('reason')}",
                "action": "New entries are paused by the market/portfolio meltdown guard.",
                "code": "MELTDOWN_ACTIVE",
            })
        return blockers

    def _approval_queue(self, reports: dict) -> list[dict]:
        advisor = reports.get("advisor", {}) or {}
        recommendations = advisor.get("recommendations", []) or []
        items = []
        for rec in recommendations:
            if not rec.get("requires_human_approval"):
                continue
            items.append({
                "priority": rec.get("priority"),
                "area": rec.get("area"),
                "finding": rec.get("finding"),
                "recommendation": rec.get("recommendation"),
            })
        items.sort(key=lambda item: {"high": 3, "medium": 2, "low": 1}.get(str(item.get("priority")), 0), reverse=True)
        return items[:10]

    def _learning_status(self, reports: dict) -> dict:
        ml_edge = reports.get("ml_edge", {}) or {}
        readiness = ml_edge.get("readiness", {}) or {}
        model = ml_edge.get("model", {}) or {}
        registry = reports.get("registry", {}) or {}
        hypothesis_summary = registry.get("hypothesis_summary", {}) or {}
        market = reports.get("market_regime", {}) or {}
        indicator_edge = reports.get("indicator_edge", {}) or {}
        indicator_summary = indicator_edge.get("summary", {}) or {}

        metrics = model.get("metrics", {}) or {}
        prediction_summary = model.get("prediction_summary", {}) or {}
        meta = ml_edge.get("meta", {}) or {}

        return {
            "ml_edge": {
                "status": readiness.get("status") or model.get("status") or "unknown",
                "rows": _safe_int(readiness.get("rows")),
                "positive": _safe_int(readiness.get("positive")),
                "non_positive": _safe_int(readiness.get("non_positive")),
                "model_status": model.get("status"),
                "model_reason": model.get("reason"),
                "model_path": model.get("model_path"),
                "feature_set": meta.get("feature_set") or model.get("feature_version"),
                "feature_contract": meta.get("feature_contract", {}),
                "metrics": metrics,
                "prediction_summary": prediction_summary,
                "regression_mae_r": _safe_float(metrics.get("regression_mae_r")),
                "regression_rmse_r": _safe_float(metrics.get("regression_rmse_r")),
                "classification_accuracy": _safe_float(metrics.get("classification_accuracy")),
                "classification_auc": metrics.get("classification_auc"),
                "avg_predicted_r": _safe_float(prediction_summary.get("avg_predicted_r")),
                "avg_actual_r": _safe_float(prediction_summary.get("avg_actual_r")),
                "avg_p_positive": _safe_float(prediction_summary.get("avg_p_positive")),
                "reason": readiness.get("reason"),
            },
            "indicator_edge": {
                "status": indicator_edge.get("status"),
                "usable_rows": _safe_int(indicator_summary.get("usable_rows")),
                "ranked_features": _safe_int(indicator_summary.get("ranked_features")),
                "symbols_ranked": _safe_int(indicator_summary.get("symbols_ranked")),
                "top_feature": indicator_summary.get("top_feature"),
                "weak_feature": indicator_summary.get("weak_feature"),
                "live_effect": bool((indicator_edge.get("meta") or {}).get("live_effect")),
            },
            "hypotheses": {
                "total": _safe_int(hypothesis_summary.get("total")),
                "promotable": _safe_int(hypothesis_summary.get("promotable")),
                "by_stability": hypothesis_summary.get("by_stability", {}),
            },
            "market_regime": {
                "regime": market.get("regime"),
                "risk_mode": market.get("risk_mode"),
                "directional_bias": market.get("directional_bias"),
                "breadth": market.get("breadth", {}),
            },
        }

    def _recommendation_status(self, reports: dict) -> dict:
        rec = reports.get("recommendation_aggregator", {}) or {}
        summary = rec.get("summary", {}) or {}
        return {
            "status": rec.get("status"),
            "total": _safe_int(summary.get("total")),
            "needs_operator_review": _safe_int(summary.get("needs_operator_review")),
            "auto_accept_as_context": _safe_int(summary.get("auto_accept_as_context")),
            "wait_more_evidence": _safe_int(summary.get("wait_more_evidence")),
            "blocked": _safe_int(summary.get("blocked")),
            "by_status": summary.get("by_status", {}),
            "top_review": summary.get("top_review", [])[:3],
            "live_effect": bool(summary.get("live_effect")),
        }

    def _recommendation_quality_status(self, reports: dict) -> dict:
        quality = reports.get("recommendation_quality", {}) or {}
        summary = quality.get("summary", {}) or {}
        return {
            "status": summary.get("status") or quality.get("status"),
            "tracked_items": _safe_int(summary.get("tracked_items")),
            "observed_today": _safe_int(summary.get("observed_today")),
            "days_observed": _safe_int(summary.get("days_observed")),
            "needs_attention": _safe_int(summary.get("needs_attention")),
            "needs_review": _safe_int(summary.get("needs_review")),
            "unstable": _safe_int(summary.get("unstable")),
            "stale_unresolved": _safe_int(summary.get("stale_unresolved")),
            "stable_unresolved": _safe_int(summary.get("stable_unresolved")),
            "stable_context": _safe_int(summary.get("stable_context")),
            "operator_handled": _safe_int(summary.get("operator_handled")),
            "by_quality": summary.get("by_quality", {}),
            "top_attention": summary.get("top_attention", [])[:5],
            "live_effect": bool(summary.get("live_effect")),
        }

    def _experiment_status(self, reports: dict) -> dict:
        plan = reports.get("experiments", {}) or {}
        shadow = reports.get("shadow_results", {}) or {}
        plan_summary = plan.get("summary", {}) or {}
        shadow_summary = shadow.get("summary", {}) or {}
        by_verdict = shadow_summary.get("by_verdict", {}) or {}

        return {
            "planned": _safe_int(plan_summary.get("total")),
            "ready_for_approval": _safe_int(plan_summary.get("ready_for_approval")),
            "approved_for_shadow": _safe_int(plan_summary.get("approved_for_shadow")),
            "by_status": plan_summary.get("by_status", {}),
            "by_type": plan_summary.get("by_type", {}),
            "shadow_replay_matches": _safe_int(shadow_summary.get("replay_matches")),
            "shadow_forward_matches": _safe_int(shadow_summary.get("forward_matches")),
            "verdicts": by_verdict,
            "promising_replay_needs_forward": _safe_int(by_verdict.get("promising_replay_needs_forward")),
            "protection_confirmed": _safe_int(by_verdict.get("protection_confirmed")),
        }

    def _promotion_status(self, reports: dict) -> dict:
        promotion = reports.get("promotion_gate", {}) or {}
        summary = promotion.get("summary", {}) or {}
        return {
            "total": _safe_int(summary.get("total")),
            "ready_for_human_review": _safe_int(summary.get("ready_for_human_review")),
            "confirmed_protection": _safe_int(summary.get("confirmed_protection")),
            "blocked": _safe_int(summary.get("blocked")),
            "waiting": _safe_int(summary.get("waiting")),
            "by_status": summary.get("by_status", {}),
            "by_type": summary.get("by_type", {}),
        }

    def _approval_status(self, reports: dict) -> dict:
        inbox = reports.get("approval_inbox", {}) or {}
        summary = inbox.get("summary", {}) or {}
        return {
            "total": _safe_int(summary.get("total")),
            "review_for_approval": _safe_int(summary.get("review_for_approval")),
            "reject_candidate": _safe_int(summary.get("reject_candidate")),
            "review_for_rejection": _safe_int(summary.get("review_for_rejection")),
            "wait": _safe_int(summary.get("wait")),
            "no_action_keep_protection": _safe_int(summary.get("no_action_keep_protection")),
            "by_action": summary.get("by_action", {}),
            "lifecycle": inbox.get("lifecycle_summary", {}),
        }

    def _shadow_live_status(self, reports: dict) -> dict:
        shadow_live = reports.get("shadow_live", {}) or {}
        summary = shadow_live.get("summary", {}) or {}
        return {
            "active_shadow_policies": _safe_int(summary.get("active_shadow_policies")),
            "loaded_events": _safe_int(summary.get("loaded_events")),
            "matches": _safe_int(summary.get("matches")),
            "by_shadow_action": summary.get("by_shadow_action", {}),
        }

    def _risk_policy_status(self, reports: dict) -> dict:
        risk_policy = reports.get("risk_policy", {}) or {}
        summary = risk_policy.get("summary", {}) or {}
        market = risk_policy.get("market_context", {}) or {}
        return {
            "total_symbols": _safe_int(summary.get("total_symbols")),
            "average_risk_multiplier": _safe_float(summary.get("average_risk_multiplier"), 1.0),
            "average_long_risk_multiplier": _safe_float(summary.get("average_long_risk_multiplier"), 1.0),
            "average_short_risk_multiplier": _safe_float(summary.get("average_short_risk_multiplier"), 1.0),
            "risk_down": len(summary.get("risk_down_symbols", []) or []),
            "data_driven_risk_down": _safe_int(summary.get("data_driven_risk_down")),
            "market_context_only": _safe_int(summary.get("market_context_only")),
            "review_only": _safe_int(summary.get("review_only")),
            "risk_up": _safe_int(summary.get("risk_up")),
            "long_risk_down": len(summary.get("long_risk_down_symbols", []) or []),
            "short_risk_down": len(summary.get("short_risk_down_symbols", []) or []),
            "cap_new_longs": len(summary.get("cap_new_long_symbols", []) or []),
            "by_policy_mode": summary.get("by_policy_mode", {}),
            "by_long_policy_mode": summary.get("by_long_policy_mode", {}),
            "by_short_policy_mode": summary.get("by_short_policy_mode", {}),
            "market_regime": summary.get("market_regime") or market.get("regime"),
            "promotion_blocked": _safe_int(summary.get("promotion_blocked")),
            "approval_reject_candidates": _safe_int(summary.get("approval_reject_candidates")),
            "ml_edge_status": summary.get("ml_edge_status"),
            "live_enforcement": bool((risk_policy.get("meta") or {}).get("live_enforcement")),
        }

    def _risk_advice_history_status(self, reports: dict) -> dict:
        history = reports.get("risk_advice_history", {}) or {}
        summary = history.get("summary", {}) or {}
        return {
            "tracked_symbols": _safe_int(summary.get("tracked_symbols")),
            "days_observed": _safe_int(summary.get("days_observed")),
            "stable_days_required": _safe_int(summary.get("stable_days_required")),
            "data_driven_risk_down_symbols": _safe_int(summary.get("data_driven_risk_down_symbols")),
            "stable_data_down_symbols": _safe_int(summary.get("stable_data_down_symbols")),
            "market_context_only_symbols": _safe_int(summary.get("market_context_only_symbols")),
            "review_only_symbols": _safe_int(summary.get("review_only_symbols")),
            "risk_up_symbols": _safe_int(summary.get("risk_up_symbols")),
            "verdict": summary.get("verdict"),
            "live_enforcement": bool((history.get("meta") or {}).get("live_enforcement")),
        }

    def _risk_strategy_status(self, reports: dict) -> dict:
        bridge = reports.get("risk_strategy_bridge", {}) or {}
        summary = bridge.get("summary", {}) or {}
        return {
            "loaded_decisions": _safe_int(summary.get("loaded_decisions")),
            "opened_trades": _safe_int(summary.get("opened_trades")),
            "would_adjust_open_trades": _safe_int(summary.get("would_adjust_open_trades")),
            "average_adjusted_open_multiplier": _safe_float(summary.get("average_adjusted_open_multiplier")),
            "by_risk_shadow_action": summary.get("by_risk_shadow_action", {}),
            "live_enforcement": bool((bridge.get("meta") or {}).get("live_enforcement")),
        }

    def _risk_outcome_status(self, reports: dict) -> dict:
        outcome = reports.get("risk_bridge_outcomes", {}) or {}
        summary = outcome.get("summary", {}) or {}
        return {
            "adjusted_with_labeled_outcomes": _safe_int(summary.get("adjusted_with_labeled_outcomes")),
            "adjusted_loss_trades": _safe_int(summary.get("adjusted_loss_trades")),
            "adjusted_winner_trades": _safe_int(summary.get("adjusted_winner_trades")),
            "adjusted_avg_cf_r": _safe_float(summary.get("adjusted_avg_cf_r")),
            "estimated_saved_r": _safe_float(summary.get("estimated_saved_r")),
            "estimated_missed_r": _safe_float(summary.get("estimated_missed_r")),
            "estimated_net_saved_r": _safe_float(summary.get("estimated_net_saved_r")),
            "verdict": summary.get("verdict"),
            "live_enforcement": bool((outcome.get("meta") or {}).get("live_enforcement")),
        }

    def _risk_history_status(self, reports: dict) -> dict:
        history = reports.get("risk_bridge_history", {}) or {}
        summary = history.get("summary", {}) or {}
        return {
            "unique_adjusted_labeled_events": _safe_int(summary.get("unique_adjusted_labeled_events")),
            "days_observed": _safe_int(summary.get("days_observed")),
            "adjusted_loss_trades": _safe_int(summary.get("adjusted_loss_trades")),
            "adjusted_winner_trades": _safe_int(summary.get("adjusted_winner_trades")),
            "adjusted_avg_cf_r": _safe_float(summary.get("adjusted_avg_cf_r")),
            "estimated_saved_r": _safe_float(summary.get("estimated_saved_r")),
            "estimated_missed_r": _safe_float(summary.get("estimated_missed_r")),
            "estimated_net_saved_r": _safe_float(summary.get("estimated_net_saved_r")),
            "verdict": summary.get("verdict"),
            "live_enforcement": bool((history.get("meta") or {}).get("live_enforcement")),
        }

    def _risk_guard_status(self, reports: dict) -> dict:
        guard = reports.get("risk_guard", {}) or {}
        summary = guard.get("summary", {}) or {}
        return {
            "loaded_open_trades": _safe_int(summary.get("loaded_open_trades")),
            "days_observed": _safe_int(summary.get("days_observed")),
            "weeks_observed": _safe_int(summary.get("weeks_observed")),
            "guard_triggers": _safe_int(summary.get("guard_triggers")),
            "estimated_saved_r": _safe_float(summary.get("estimated_saved_r")),
            "estimated_missed_r": _safe_float(summary.get("estimated_missed_r")),
            "estimated_net_saved_r": _safe_float(summary.get("estimated_net_saved_r")),
            "verdict": summary.get("verdict"),
            "primary_issue": summary.get("primary_issue"),
            "calibration_advice": summary.get("calibration_advice", [])[:5],
            "by_guard": summary.get("by_guard", {}),
            "live_enforcement": bool((guard.get("meta") or {}).get("live_enforcement")),
        }

    def _exit_management_status(self, reports: dict) -> dict:
        exits = reports.get("exit_management", {}) or {}
        summary = exits.get("summary", {}) or {}
        data_quality = exits.get("data_quality", {}) or {}
        return {
            "status": exits.get("status"),
            "positions_loaded": _safe_int(summary.get("positions_loaded")),
            "closed_positions": _safe_int(summary.get("closed_positions")),
            "open_or_partial_positions": _safe_int(summary.get("open_or_partial_positions")),
            "positions_with_tp1_proxy": _safe_int(summary.get("positions_with_tp1_proxy")),
            "tp1_proxy_rate_pct": _safe_float(summary.get("tp1_proxy_rate_pct")),
            "total_realized_pnl_eur": _safe_float(summary.get("total_realized_pnl_eur")),
            "avg_realized_pnl_eur": _safe_float(summary.get("avg_realized_pnl_eur")),
            "win_rate_pct": _safe_float(summary.get("win_rate_pct")),
            "avg_hold_hours_closed": _safe_float(summary.get("avg_hold_hours_closed")),
            "verdict": summary.get("verdict"),
            "reason_available": bool(summary.get("reason_available") or data_quality.get("reason_available")),
            "close_reason_missing": _safe_int(data_quality.get("close_reason_missing")),
            "by_exit_path": summary.get("by_exit_path", {}),
            "live_effect": bool((exits.get("meta") or {}).get("live_effect")),
        }

    def _position_lifecycle_status(self, reports: dict) -> dict:
        lifecycle = reports.get("position_lifecycle", {}) or {}
        summary = lifecycle.get("summary", {}) or {}
        return {
            "status": summary.get("status") or lifecycle.get("status"),
            "verdict": summary.get("verdict"),
            "master_trades": _safe_int(summary.get("master_trades")),
            "child_trades": _safe_int(summary.get("child_trades")),
            "open_masters": _safe_int(summary.get("open_masters")),
            "partial_masters": _safe_int(summary.get("partial_masters")),
            "closed_masters": _safe_int(summary.get("closed_masters")),
            "issue_count": _safe_int(summary.get("issue_count")),
            "high_issues": _safe_int(summary.get("high_issues")),
            "medium_issues": _safe_int(summary.get("medium_issues")),
            "low_issues": _safe_int(summary.get("low_issues")),
            "by_issue_code": summary.get("by_issue_code", {}),
            "top_issues": summary.get("top_issues", [])[:5],
            "live_effect": bool((lifecycle.get("meta") or {}).get("live_effect")),
        }

    def _gpt_efficiency_status(self, reports: dict) -> dict:
        gpt = reports.get("gpt_decisions", {}) or {}
        totals = gpt.get("totals", {}) or {}
        open_rate = _safe_float(totals.get("open_rate_pct"))
        hold_rate = _safe_float(totals.get("hold_rate_pct"))
        decisions = _safe_int(totals.get("events"))

        if decisions == 0:
            verdict = "no_labeled_gpt_decisions"
        elif decisions >= 100 and hold_rate >= 90.0 and open_rate <= 10.0:
            verdict = "mostly_hold_review_cost"
        else:
            verdict = "normal"

        return {
            "decisions": decisions,
            "hold": _safe_int(totals.get("hold")),
            "open": _safe_int(totals.get("open")),
            "hold_rate_pct": hold_rate,
            "open_rate_pct": open_rate,
            "cf_avg_r": _safe_float(totals.get("cf_avg_r")),
            "zero_conf_pct": _safe_float(totals.get("zero_conf_pct")),
            "verdict": verdict,
        }

    def _pre_gpt_gate_status(self, reports: dict) -> dict:
        gate = reports.get("pre_gpt_gate", {}) or {}
        summary = gate.get("summary", {}) or {}
        return {
            "evaluated_decisions": _safe_int(summary.get("evaluated_decisions")),
            "would_skip_gpt": _safe_int(summary.get("would_skip_gpt")),
            "call_reduction_pct": _safe_float(summary.get("call_reduction_pct")),
            "skipped_holds": _safe_int(summary.get("skipped_holds")),
            "skipped_opens": _safe_int(summary.get("skipped_opens")),
            "skipped_open_winners": _safe_int(summary.get("skipped_open_winners")),
            "skipped_open_losers": _safe_int(summary.get("skipped_open_losers")),
            "estimated_net_saved_r": _safe_float(summary.get("estimated_net_saved_r")),
            "verdict": summary.get("verdict"),
            "live_enforcement": bool((gate.get("meta") or {}).get("live_enforcement")),
        }

    def _safety_status(self, reports: dict) -> dict:
        safety = reports.get("safety_control", {}) or {}
        return {
            "status": safety.get("status") or "UNKNOWN",
            "kill_switch_active": bool(safety.get("kill_switch_active")),
            "live_entry_orders_allowed": bool(safety.get("live_entry_orders_allowed", True)),
            "live_enforcement_allowed": bool(safety.get("live_enforcement_allowed")),
            "live_risk_wiring_allowed": bool(safety.get("live_risk_wiring_allowed")),
            "live_strategy_wiring_allowed": bool(safety.get("live_strategy_wiring_allowed")),
            "reason": safety.get("reason"),
            "updated_utc": safety.get("updated_utc"),
            "meltdown_active": bool((safety.get("meltdown") or {}).get("active")),
            "meltdown_reason": (safety.get("meltdown") or {}).get("reason"),
            "meltdown_updated_utc": (safety.get("meltdown") or {}).get("updated_utc"),
            "audit_events": _safe_int((safety.get("audit") or {}).get("events")),
            "output_path": safety.get("output_path"),
        }

    def _live_readiness_status(self, reports: dict) -> dict:
        readiness = reports.get("live_readiness", {}) or {}
        summary = readiness.get("summary", {}) or {}
        return {
            "total": _safe_int(summary.get("total")),
            "eligible_for_live_wiring": _safe_int(summary.get("eligible_for_live_wiring")),
            "ready_for_operator_review": _safe_int(summary.get("ready_for_operator_review")),
            "approved_but_safety_locked": _safe_int(summary.get("approved_but_safety_locked")),
            "blocked": _safe_int(summary.get("blocked")),
            "waiting": _safe_int(summary.get("waiting")),
            "calibration_only": _safe_int(summary.get("calibration_only")),
            "by_status": summary.get("by_status", {}),
            "by_area": summary.get("by_area", {}),
            "read_only": bool((readiness.get("meta") or {}).get("read_only", True)),
            "live_enforcement": bool((readiness.get("meta") or {}).get("live_enforcement")),
        }

    def _operating_state(self, reports: dict, blockers: list[dict], approval_queue: list[dict]) -> dict:
        advisor = reports.get("advisor", {}) or {}
        registry = reports.get("registry", {}) or {}
        experiments = reports.get("experiments", {}) or {}
        shadow = reports.get("shadow_results", {}) or {}
        promotion = reports.get("promotion_gate", {}) or {}
        approval = reports.get("approval_inbox", {}) or {}
        shadow_live = reports.get("shadow_live", {}) or {}
        risk_policy = reports.get("risk_policy", {}) or {}
        risk_advice_history = reports.get("risk_advice_history", {}) or {}
        risk_strategy = reports.get("risk_strategy_bridge", {}) or {}
        risk_outcomes = reports.get("risk_bridge_outcomes", {}) or {}
        risk_history = reports.get("risk_bridge_history", {}) or {}
        risk_guard = reports.get("risk_guard", {}) or {}
        exit_management = reports.get("exit_management", {}) or {}
        position_lifecycle = reports.get("position_lifecycle", {}) or {}
        gpt_decisions = reports.get("gpt_decisions", {}) or {}
        pre_gpt_gate = reports.get("pre_gpt_gate", {}) or {}
        safety = reports.get("safety_control", {}) or {}
        live_readiness = reports.get("live_readiness", {}) or {}
        recommendation_quality = reports.get("recommendation_quality", {}) or {}

        if blockers:
            status = "ACTION_NEEDED"
            headline = "Runtime or pipeline blockers need attention before approvals."
        elif approval_queue:
            status = "REVIEW"
            headline = "There are strategy recommendations waiting for human review."
        else:
            status = "WATCH"
            headline = "Pipeline is collecting evidence; no immediate approval needed."

        return {
            "status": status,
            "headline": headline,
            "advisor_status": advisor.get("status"),
            "advisor_summary": advisor.get("summary", {}),
            "registry": {
                "total": registry.get("total", 0),
                "by_status": registry.get("by_status", {}),
                "active": len(registry.get("active", []) or []),
            },
            "experiments": experiments.get("summary", {}),
            "shadow_results": shadow.get("summary", {}),
            "promotion_gate": promotion.get("summary", {}),
            "approval_inbox": approval.get("summary", {}),
            "shadow_live": shadow_live.get("summary", {}),
            "risk_policy": risk_policy.get("summary", {}),
            "risk_advice_history": risk_advice_history.get("summary", {}),
            "risk_strategy_bridge": risk_strategy.get("summary", {}),
            "risk_bridge_outcomes": risk_outcomes.get("summary", {}),
            "risk_bridge_history": risk_history.get("summary", {}),
            "risk_guard": risk_guard.get("summary", {}),
            "exit_management": exit_management.get("summary", {}),
            "position_lifecycle": position_lifecycle.get("summary", {}),
            "gpt_decisions": gpt_decisions.get("totals", {}),
            "pre_gpt_gate": pre_gpt_gate.get("summary", {}),
            "live_readiness": live_readiness.get("summary", {}),
            "recommendation_quality": recommendation_quality.get("summary", {}),
            "safety_control": {
                "status": safety.get("status"),
                "kill_switch_active": safety.get("kill_switch_active"),
                "meltdown_active": (safety.get("meltdown") or {}).get("active"),
                "live_entry_orders_allowed": safety.get("live_entry_orders_allowed"),
                "live_enforcement_allowed": safety.get("live_enforcement_allowed"),
            },
        }

    def _next_actions(
        self,
        blockers: list[dict],
        approval_queue: list[dict],
        experiment_status: dict,
        promotion_status: dict,
        approval_status: dict,
        shadow_live_status: dict,
        risk_policy_status: dict,
        risk_advice_history_status: dict,
        risk_strategy_status: dict,
        risk_outcome_status: dict,
        risk_history_status: dict,
        risk_guard_status: dict,
        exit_management_status: dict,
        position_lifecycle_status: dict,
        gpt_efficiency_status: dict,
        pre_gpt_gate_status: dict,
        safety_status: dict,
        live_readiness_status: dict,
        learning_status: dict,
        recommendation_quality_status: dict,
    ) -> list[str]:
        if blockers:
            return [
                "Fix high-priority runtime/pipeline blockers first.",
                "Do not approve strategy experiments while health blockers are active.",
            ]

        actions = []
        if safety_status.get("kill_switch_active"):
            actions.append("Kill-switch is active; clear it explicitly only after verifying why it was enabled.")
        if safety_status.get("meltdown_active"):
            actions.append("Meltdown manager is active; wait for market/portfolio re-entry conditions before new entries.")
        elif not safety_status.get("live_entry_orders_allowed", True):
            actions.append("Live entry orders are disabled by safety state; inspect safety control before auto mode.")
        if not safety_status.get("live_enforcement_allowed"):
            actions.append("Live enforcement wiring is disabled by safety state; keep autonomous changes read-only.")
        if live_readiness_status.get("eligible_for_live_wiring"):
            actions.append("Live readiness has eligible candidates; require explicit operator approval before wiring.")
        if live_readiness_status.get("ready_for_operator_review"):
            actions.append("Live readiness has candidates for operator review; keep them read-only until approved.")
        if live_readiness_status.get("approved_but_safety_locked"):
            actions.append("Live readiness has safety-locked candidates; do not wire while safety blocks enforcement.")
        if approval_status.get("review_for_approval"):
            actions.append("Approval inbox has experiments ready for explicit approve/reject review.")
        if approval_status.get("reject_candidate"):
            actions.append("Approval inbox has repeat-blocked experiments that are reject candidates.")
        if approval_status.get("review_for_rejection"):
            actions.append("Approval inbox has experiments that should likely be rejected or left blocked.")
        if shadow_live_status.get("active_shadow_policies"):
            actions.append("Review shadow-live matches before considering any live behavior change.")
        if risk_policy_status.get("cap_new_longs"):
            actions.append("Risk policy is recommending long-risk caps; keep it read-only until live wiring is approved.")
        if risk_policy_status.get("data_driven_risk_down"):
            actions.append("Risk policy has data-driven risk-down advice; keep it read-only until enough history confirms it.")
        elif risk_policy_status.get("risk_down"):
            actions.append("Risk policy is mostly market-context risk-down; treat it as caution, not coin-specific proof.")
        if risk_policy_status.get("risk_up"):
            actions.append("Risk-up advice requires explicit human approval and must stay disabled.")
        if risk_advice_history_status.get("verdict") == "stable_data_down_candidates":
            actions.append("Risk advice history has stable data-down candidates; keep read-only until bridge outcomes confirm them.")
        elif risk_advice_history_status.get("tracked_symbols"):
            actions.append("Risk advice history is collecting multi-day stability; repeated same-day runs are not counted as new evidence.")
        if risk_strategy_status.get("would_adjust_open_trades"):
            actions.append("Risk strategy bridge would reduce recent open trade sizing; review before live wiring.")
        if risk_outcome_status.get("verdict") == "risk_down_helpful":
            actions.append("Risk bridge outcomes suggest risk-down helped; keep collecting evidence before live wiring.")
        elif risk_outcome_status.get("verdict") == "risk_down_too_strict":
            actions.append("Risk bridge outcomes suggest risk-down may be too strict; do not wire live yet.")
        if risk_history_status.get("verdict") == "stable_risk_down_helpful":
            actions.append("Risk bridge history is stable-positive; prepare human review before any live wiring.")
        elif risk_history_status.get("verdict") == "risk_down_too_strict":
            actions.append("Risk bridge history says risk-down is too strict; keep it shadow-only.")
        elif risk_history_status.get("unique_adjusted_labeled_events"):
            actions.append("Risk bridge history is accumulating unique labeled outcomes; avoid judging repeated runs as new evidence.")
        if risk_guard_status.get("verdict") == "guards_look_helpful":
            actions.append("Risk guards look helpful in shadow replay; keep collecting evidence before live wiring.")
        elif risk_guard_status.get("verdict") == "guards_too_strict":
            issue = risk_guard_status.get("primary_issue") or {}
            guard_name = issue.get("guard")
            if guard_name:
                actions.append(f"Risk guard {guard_name} looks too strict in shadow replay; tune threshold before live wiring.")
            else:
                actions.append("Risk guards look too strict in shadow replay; do not wire live.")
        elif risk_guard_status.get("guard_triggers"):
            actions.append("Risk guards are seeing pressure in shadow mode; keep them read-only while sample grows.")
        if exit_management_status.get("verdict") == "exit_logging_needs_reason_field":
            actions.append("Exit report needs explicit close-reason logging before exit-rule tuning.")
        elif exit_management_status.get("verdict") == "exit_logging_collecting_reasons":
            actions.append("Exit report now logs close reasons; collect fresh exits before tuning exit rules.")
        elif exit_management_status.get("verdict") == "exit_data_ready_for_review":
            actions.append("Exit management has enough data for review; keep exit changes manual/read-only.")
        elif exit_management_status.get("positions_loaded"):
            actions.append("Exit management is collecting position lifecycle evidence before tuning exits.")
        if position_lifecycle_status.get("high_issues"):
            actions.append("Position lifecycle has high-integrity issues; fix bookkeeping before app/autonomy decisions.")
        elif position_lifecycle_status.get("medium_issues"):
            actions.append("Position lifecycle has review items; keep app actions read-only until checked.")
        elif position_lifecycle_status.get("master_trades"):
            actions.append("Position lifecycle is being tracked for app readiness.")
        if gpt_efficiency_status.get("verdict") == "mostly_hold_review_cost":
            actions.append("GPT decisions are mostly HOLD; evaluate a shadow pre-GPT gate before reducing live calls.")
        if pre_gpt_gate_status.get("verdict") == "promising_for_shadow":
            actions.append("Pre-GPT gate shadow looks promising; keep collecting evidence before live call reduction.")
        elif pre_gpt_gate_status.get("verdict") == "too_risky":
            actions.append("Pre-GPT gate shadow would skip too many useful calls; do not reduce GPT calls live.")
        elif pre_gpt_gate_status.get("would_skip_gpt"):
            actions.append("Pre-GPT gate shadow is measuring possible GPT call savings; keep it read-only.")
        if promotion_status.get("ready_for_human_review"):
            actions.append("Review promotion-gate candidates before approving any experiment.")
        if promotion_status.get("blocked"):
            actions.append("Do not promote blocked experiments; their evidence failed the promotion gate.")
        if experiment_status.get("ready_for_approval"):
            actions.append("Review ready experiments and approve/reject explicitly via experiment_planner.")
        if experiment_status.get("promising_replay_needs_forward"):
            actions.append("Keep relax-entry experiments shadow-only until they get forward matches.")
        if experiment_status.get("protection_confirmed"):
            actions.append("Keep confirmed protection patterns; avoid broad loosening that includes them.")
        ml = learning_status.get("ml_edge", {})
        if ml.get("status") == "insufficient_data":
            actions.append("Keep collecting structured labeled events before using ML predictions.")
        elif ml.get("model_status") == "trained":
            actions.append("ML edge model is trained shadow-only; review validation metrics before any live use.")
        elif ml.get("model_status") == "dependency_missing":
            actions.append("ML has enough data, but training dependencies are missing; verify sklearn/joblib on the Pi.")
        elif ml.get("status") == "ready":
            actions.append("ML has enough data but no trained model output yet; inspect the ML edge model report.")
        if recommendation_quality_status.get("unstable"):
            actions.append("Recommendation quality tracker found unstable advice; use history before approving.")
        elif recommendation_quality_status.get("needs_attention"):
            actions.append("Recommendation quality tracker has items needing review in the app.")
        elif recommendation_quality_status.get("tracked_items"):
            actions.append("Recommendation quality tracker is building multi-day memory for app decisions.")
        if approval_queue:
            actions.append("Review human-approval recommendations; no automatic live changes are made.")
        if not actions:
            actions.append("Let the bot keep collecting evidence and review the next daily control report.")
        return actions


def format_control_message(report: dict, max_actions: int = 5, max_approvals: int = 4) -> str:
    summary = report.get("operator_summary", {}) or {}
    learning = report.get("learning_status", {}) or {}
    ml = learning.get("ml_edge", {}) or {}
    experiments = report.get("experiment_status", {}) or {}
    promotion = report.get("promotion_status", {}) or {}
    approval = report.get("approval_status", {}) or {}
    shadow_live = report.get("shadow_live_status", {}) or {}
    risk_policy = report.get("risk_policy_status", {}) or {}
    risk_advice_history = report.get("risk_advice_history_status", {}) or {}
    risk_strategy = report.get("risk_strategy_status", {}) or {}
    risk_outcomes = report.get("risk_outcome_status", {}) or {}
    risk_history = report.get("risk_history_status", {}) or {}
    risk_guard = report.get("risk_guard_status", {}) or {}
    exits = report.get("exit_management_status", {}) or {}
    lifecycle = report.get("position_lifecycle_status", {}) or {}
    rec_quality = report.get("recommendation_quality_status", {}) or {}
    live_readiness = report.get("live_readiness_status", {}) or {}

    lines = [
        f"Daily Control [{report.get('status')}]",
        str(summary.get("headline") or ""),
        (
            f"ML rows={ml.get('rows', 0)} pos={ml.get('positive', 0)} "
            f"non_pos={ml.get('non_positive', 0)} status={ml.get('status')}"
        ),
        (
            f"Experiments planned={experiments.get('planned', 0)} "
            f"ready={experiments.get('ready_for_approval', 0)} "
            f"forward_matches={experiments.get('shadow_forward_matches', 0)}"
        ),
        (
            f"Promotion ready={promotion.get('ready_for_human_review', 0)} "
            f"confirmed_protection={promotion.get('confirmed_protection', 0)} "
            f"blocked={promotion.get('blocked', 0)}"
        ),
        (
            f"Approval inbox approve={approval.get('review_for_approval', 0)} "
            f"reject_candidate={approval.get('reject_candidate', 0)} "
            f"reject={approval.get('review_for_rejection', 0)} "
            f"wait={approval.get('wait', 0)}"
        ),
        (
            f"Shadow-live policies={shadow_live.get('active_shadow_policies', 0)} "
            f"matches={shadow_live.get('matches', 0)}"
        ),
        (
            f"Risk policy symbols={risk_policy.get('total_symbols', 0)} "
            f"long_down={risk_policy.get('long_risk_down', 0)} "
            f"short_down={risk_policy.get('short_risk_down', 0)} "
            f"cap_longs={risk_policy.get('cap_new_longs', 0)} "
            f"data_down={risk_policy.get('data_driven_risk_down', 0)} "
            f"market_only={risk_policy.get('market_context_only', 0)} "
            f"avg_long={risk_policy.get('average_long_risk_multiplier', 1.0)} "
            f"avg_short={risk_policy.get('average_short_risk_multiplier', 1.0)}"
        ),
        (
            f"Risk advice history tracked={risk_advice_history.get('tracked_symbols', 0)} "
            f"days={risk_advice_history.get('days_observed', 0)} "
            f"stable_down={risk_advice_history.get('stable_data_down_symbols', 0)} "
            f"verdict={risk_advice_history.get('verdict')}"
        ),
        (
            f"Risk bridge decisions={risk_strategy.get('loaded_decisions', 0)} "
            f"opens={risk_strategy.get('opened_trades', 0)} "
            f"adjusted_opens={risk_strategy.get('would_adjust_open_trades', 0)} "
            f"avg_open_mult={risk_strategy.get('average_adjusted_open_multiplier', 0.0)}"
        ),
        (
            f"Risk outcomes labeled={risk_outcomes.get('adjusted_with_labeled_outcomes', 0)} "
            f"net_saved_R={risk_outcomes.get('estimated_net_saved_r', 0.0)} "
            f"verdict={risk_outcomes.get('verdict')}"
        ),
        (
            f"Risk history unique={risk_history.get('unique_adjusted_labeled_events', 0)} "
            f"days={risk_history.get('days_observed', 0)} "
            f"net_saved_R={risk_history.get('estimated_net_saved_r', 0.0)} "
            f"verdict={risk_history.get('verdict')}"
        ),
        (
            f"Risk guards trades={risk_guard.get('loaded_open_trades', 0)} "
            f"triggers={risk_guard.get('guard_triggers', 0)} "
            f"net_saved_R={risk_guard.get('estimated_net_saved_r', 0.0)} "
            f"verdict={risk_guard.get('verdict')} "
            f"issue={(risk_guard.get('primary_issue') or {}).get('guard')}"
        ),
        (
            f"Exits positions={exits.get('positions_loaded', 0)} "
            f"closed={exits.get('closed_positions', 0)} "
            f"tp1={exits.get('positions_with_tp1_proxy', 0)} "
            f"win={exits.get('win_rate_pct', 0.0)}% "
            f"pnl={exits.get('total_realized_pnl_eur', 0.0)} "
            f"verdict={exits.get('verdict')}"
        ),
        (
            f"Lifecycle masters={lifecycle.get('master_trades', 0)} "
            f"open={lifecycle.get('open_masters', 0)} "
            f"partial={lifecycle.get('partial_masters', 0)} "
            f"closed={lifecycle.get('closed_masters', 0)} "
            f"issues={lifecycle.get('issue_count', 0)} "
            f"high={lifecycle.get('high_issues', 0)} "
            f"verdict={lifecycle.get('verdict')}"
        ),
        (
            f"Recommendation quality tracked={rec_quality.get('tracked_items', 0)} "
            f"days={rec_quality.get('days_observed', 0)} "
            f"attention={rec_quality.get('needs_attention', 0)} "
            f"unstable={rec_quality.get('unstable', 0)}"
        ),
        (
            f"Live readiness eligible={live_readiness.get('eligible_for_live_wiring', 0)} "
            f"review={live_readiness.get('ready_for_operator_review', 0)} "
            f"blocked={live_readiness.get('blocked', 0)} "
            f"waiting={live_readiness.get('waiting', 0)} "
            f"calibration={live_readiness.get('calibration_only', 0)}"
        ),
    ]

    blockers = report.get("blockers", []) or []
    if blockers:
        lines.append("Blockers:")
        for item in blockers[:max_actions]:
            lines.append(f"- {item.get('level', '').upper()} {item.get('area')}: {item.get('finding')}")

    approvals = report.get("approval_queue", []) or []
    if approvals:
        lines.append("Approval queue:")
        for item in approvals[:max_approvals]:
            lines.append(f"- {str(item.get('priority')).upper()} {item.get('area')}: {item.get('finding')}")

    actions = report.get("next_actions", []) or []
    lines.append("Next:")
    for action in actions[:max_actions]:
        lines.append(f"- {action}")
    return "\n".join(line for line in lines if line)


def send_telegram(report: dict) -> bool:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    TelegramNotifier(token, chat_id).safe_send(format_control_message(report))
    return True


def run_daily_control_report(output_dir: str = DEFAULT_OUTPUT_DIR, send: bool = False) -> dict:
    report = DailyControlReport().build_report()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    report["telegram_sent"] = send_telegram(report) if send else False
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build daily control report.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--send", action="store_true", help="Send compact control report to Telegram.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_daily_control_report(output_dir=args.output_dir, send=args.send)
    result = {
        "status": report.get("status"),
        "blockers": len(report.get("blockers", []) or []),
        "approval_queue": len(report.get("approval_queue", []) or []),
        "next_actions": report.get("next_actions", []),
        "output_path": report.get("output_path"),
        "telegram_sent": report.get("telegram_sent", False),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
