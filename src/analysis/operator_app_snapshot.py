# ============================================================
# src/analysis/operator_app_snapshot.py
# ============================================================
"""Operator app snapshot.

Single read-only JSON surface for a future operator app. It aggregates the
existing cockpit/control/safety/approval/decision reports and documents which
operator actions are allowed in v1. This module does not change live trading
behavior and does not enable live autonomy.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "operator_app")
DEFAULT_LATEST_FILE = "latest_operator_app_snapshot.json"

DEFAULT_SOURCES = {
    "operator_cockpit": os.path.join("analysis", "operator_cockpit", "latest_operator_cockpit.json"),
    "daily_control": os.path.join("analysis", "daily_control", "latest_daily_control_report.json"),
    "safety_control": os.path.join("analysis", "safety", "latest_safety_control_report.json"),
    "approval_inbox": os.path.join("analysis", "approvals", "latest_approval_inbox.json"),
    "operator_decisions": os.path.join("analysis", "operator_decisions", "latest_operator_decisions.json"),
    "live_readiness": os.path.join("analysis", "live_readiness", "latest_live_readiness_gate.json"),
    "risk_advice_history": os.path.join("analysis", "risk", "latest_risk_advice_history_report.json"),
    "exit_management": os.path.join("analysis", "exits", "latest_exit_management_report.json"),
    "position_lifecycle": os.path.join("analysis", "positions", "latest_position_lifecycle_report.json"),
    "recommendation_aggregator": os.path.join("analysis", "recommendations", "latest_recommendation_aggregator.json"),
    "recommendation_quality": os.path.join("analysis", "recommendations", "latest_recommendation_quality_report.json"),
}

APP_ALLOWED_ACTIONS = {
    "approve": "Record approval intent only; no live effect in v1.",
    "reject": "Record rejection intent only; no live effect in v1.",
    "wait": "Keep item pending for more evidence.",
    "freeze": "Record operator freeze intent for review queues; live behavior remains unchanged in v1.",
    "snooze": "Hide or defer item in the app until expiry.",
    "note": "Attach human context to a recommendation or safety item.",
}

APP_FORBIDDEN_ACTIONS = {
    "enable_live_enforcement": "Live enforcement must not be enabled from app v1.",
    "risk_up_live": "Risk-up requires a later explicit autonomy design and approval gate.",
    "entry_rule_live": "Entry-rule changes require promotion/live-readiness gates first.",
    "ml_live": "ML remains shadow-only until data volume and validation are sufficient.",
    "clear_kill_switch": "Kill-switch clearing must stay explicit operator/admin flow, not app v1 quick action.",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: str, default: Any) -> Any:
    if not path or not os.path.exists(path):
        return {"_missing": True, "_path": path}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e), "_path": path}


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _source_status(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "invalid"
    if payload.get("_missing"):
        return "missing"
    if payload.get("_error"):
        return "error"
    return "ok"


class OperatorAppSnapshot:
    def __init__(self, sources: Optional[dict[str, str]] = None):
        self.sources = dict(DEFAULT_SOURCES)
        if sources:
            self.sources.update(sources)

    def build(self) -> dict:
        loaded = {name: _load_json(path, {}) for name, path in self.sources.items()}
        cockpit = loaded.get("operator_cockpit") or {}
        control = loaded.get("daily_control") or {}
        safety = loaded.get("safety_control") or cockpit.get("safety") or {}
        approvals = loaded.get("approval_inbox") or {}
        decisions = loaded.get("operator_decisions") or {}
        live_readiness = loaded.get("live_readiness") or {}
        risk_advice_history = loaded.get("risk_advice_history") or {}
        exit_management = loaded.get("exit_management") or {}
        position_lifecycle = loaded.get("position_lifecycle") or {}
        recommendations = loaded.get("recommendation_aggregator") or {}
        recommendation_quality = loaded.get("recommendation_quality") or {}

        source_health = {
            name: {"status": _source_status(payload), "path": self.sources.get(name)}
            for name, payload in loaded.items()
        }
        critical_sources = {"operator_cockpit", "daily_control", "safety_control"}
        missing_or_error = [
            name for name, item in source_health.items()
            if item["status"] != "ok" and name in critical_sources
        ]

        app_status = self._app_status(cockpit, safety, missing_or_error)
        summary = self._summary(cockpit, control, safety, approvals, decisions, live_readiness, risk_advice_history, exit_management, position_lifecycle, recommendations, recommendation_quality)

        return {
            "created_utc": _utc_now(),
            "status": app_status,
            "summary": summary,
            "cards": self._cards(cockpit, safety, approvals, decisions, live_readiness, risk_advice_history, exit_management, position_lifecycle, recommendations, recommendation_quality),
            "actions": {
                "allowed_v1": APP_ALLOWED_ACTIONS,
                "forbidden_v1": APP_FORBIDDEN_ACTIONS,
                "live_effect": False,
                "decision_store": os.path.join("analysis", "operator_decisions", "operator_decisions.jsonl"),
            },
            "source_health": source_health,
            "sources": self.sources,
            "raw": {
                "operator_cockpit": cockpit,
                "daily_control": control,
                "safety_control": safety,
                "approval_inbox": approvals,
                "operator_decisions": decisions,
                "live_readiness": live_readiness,
                "risk_advice_history": risk_advice_history,
                "exit_management": exit_management,
                "position_lifecycle": position_lifecycle,
                "recommendation_aggregator": recommendations,
                "recommendation_quality": recommendation_quality,
            },
        }

    def _app_status(self, cockpit: dict, safety: dict, missing_or_error: list[str]) -> str:
        meltdown = safety.get("meltdown") or {}
        meltdown_active = bool(safety.get("meltdown_active") or meltdown.get("active"))
        if safety.get("kill_switch_active") or meltdown_active:
            return "ACTION_NEEDED"
        if cockpit.get("status") in {"ACTION_NEEDED", "REVIEW"}:
            return str(cockpit.get("status"))
        if missing_or_error:
            return "DEGRADED"
        return str(cockpit.get("status") or "WATCH")

    def _summary(self, cockpit: dict, control: dict, safety: dict, approvals: dict, decisions: dict, live_readiness: dict, risk_advice_history: dict, exit_management: dict, position_lifecycle: dict, recommendations: dict, recommendation_quality: dict) -> dict:
        approval_summary = approvals.get("summary") or {}
        decision_summary = decisions.get("summary") or {}
        readiness_summary = live_readiness.get("summary") or {}
        risk_summary = risk_advice_history.get("summary") or {}
        exit_summary = exit_management.get("summary") or {}
        lifecycle_summary = position_lifecycle.get("summary") or {}
        recommendation_summary = recommendations.get("summary") or {}
        quality_summary = recommendation_quality.get("summary") or {}
        return {
            "daily_decision": (cockpit.get("daily_decision") or {}).get("label"),
            "cockpit_status": cockpit.get("status"),
            "control_status": control.get("status"),
            "kill_switch_active": bool(safety.get("kill_switch_active")),
            "meltdown_active": bool(safety.get("meltdown_active") or (safety.get("meltdown") or {}).get("active")),
            "live_entry_orders_allowed": bool(safety.get("live_entry_orders_allowed", True)),
            "live_enforcement_allowed": bool(safety.get("live_enforcement_allowed")),
            "approval_items": _safe_int(approval_summary.get("total")),
            "approval_ready": _safe_int(approval_summary.get("review_for_approval")),
            "approval_reject_candidates": _safe_int(approval_summary.get("reject_candidate")),
            "operator_decisions": _safe_int(decision_summary.get("total")),
            "live_eligible": _safe_int(readiness_summary.get("eligible_for_live_wiring")),
            "live_review": _safe_int(readiness_summary.get("ready_for_operator_review")),
            "risk_advice_verdict": risk_summary.get("verdict"),
            "stable_data_down_symbols": _safe_int(risk_summary.get("stable_data_down_symbols")),
            "exit_positions": _safe_int(exit_summary.get("positions_loaded")),
            "exit_closed": _safe_int(exit_summary.get("closed_positions")),
            "exit_verdict": exit_summary.get("verdict"),
            "lifecycle_issues": _safe_int(lifecycle_summary.get("issue_count")),
            "lifecycle_high_issues": _safe_int(lifecycle_summary.get("high_issues")),
            "lifecycle_verdict": lifecycle_summary.get("verdict"),
            "recommendation_review": _safe_int(recommendation_summary.get("needs_operator_review")),
            "recommendation_auto_context": _safe_int(recommendation_summary.get("auto_accept_as_context")),
            "recommendation_wait": _safe_int(recommendation_summary.get("wait_more_evidence")),
            "recommendation_blocked": _safe_int(recommendation_summary.get("blocked")),
            "recommendation_quality_tracked": _safe_int(quality_summary.get("tracked_items")),
            "recommendation_quality_attention": _safe_int(quality_summary.get("needs_attention")),
            "recommendation_quality_unstable": _safe_int(quality_summary.get("unstable")),
            "live_effect": False,
        }

    def _cards(self, cockpit: dict, safety: dict, approvals: dict, decisions: dict, live_readiness: dict, risk_advice_history: dict, exit_management: dict, position_lifecycle: dict, recommendations: dict, recommendation_quality: dict) -> list[dict]:
        return [
            {
                "id": "cockpit",
                "title": "Cockpit",
                "status": cockpit.get("status", "UNKNOWN"),
                "headline": (cockpit.get("daily_decision") or {}).get("label") or "No cockpit decision available.",
                "data": {
                    "live_changes": cockpit.get("live_changes", {}),
                    "bot_health": cockpit.get("bot_health", {}),
                    "next_actions": cockpit.get("next_actions", [])[:6],
                },
            },
            {
                "id": "safety",
                "title": "Safety",
                "status": safety.get("status", "UNKNOWN"),
                "headline": safety.get("reason") or "Safety state loaded.",
                "data": safety,
            },
            {
                "id": "approval_inbox",
                "title": "Approval Inbox",
                "status": approvals.get("status", "UNKNOWN"),
                "headline": "Operator decisions are recorded read-only in v1.",
                "data": approvals.get("summary", {}),
            },
            {
                "id": "live_readiness",
                "title": "Live Readiness",
                "status": "READ_ONLY",
                "headline": "Live wiring remains disabled until explicit approval design is complete.",
                "data": live_readiness.get("summary", {}),
            },
            {
                "id": "risk_advice_history",
                "title": "Risk Advice History",
                "status": risk_advice_history.get("status", "OK"),
                "headline": (risk_advice_history.get("summary") or {}).get("verdict") or "No verdict.",
                "data": risk_advice_history.get("summary", {}),
            },
            {
                "id": "exit_management",
                "title": "Exit Management",
                "status": exit_management.get("status", "UNKNOWN"),
                "headline": (exit_management.get("summary") or {}).get("verdict") or "Exit report not available.",
                "data": exit_management.get("summary", {}),
            },
            {
                "id": "position_lifecycle",
                "title": "Position Lifecycle",
                "status": position_lifecycle.get("status", "UNKNOWN"),
                "headline": (position_lifecycle.get("summary") or {}).get("verdict") or "Lifecycle report not available.",
                "data": position_lifecycle.get("summary", {}),
            },
            {
                "id": "recommendations",
                "title": "Recommendations",
                "status": recommendations.get("status", "WATCH"),
                "headline": "Bundled decision items; no live effect in v1.",
                "data": recommendations.get("summary", {}),
            },
            {
                "id": "recommendation_quality",
                "title": "Recommendation Quality",
                "status": recommendation_quality.get("status", "WATCH"),
                "headline": "Multi-day memory for recommendation stability.",
                "data": recommendation_quality.get("summary", {}),
            },
            {
                "id": "operator_decisions",
                "title": "Operator Decisions",
                "status": decisions.get("status", "OK"),
                "headline": "Append-only human decisions; no live effect.",
                "data": decisions.get("summary", {}),
            },
        ]


def run_operator_app_snapshot(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    snapshot = OperatorAppSnapshot().build()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    snapshot["output_path"] = output_path
    _write_json(output_path, snapshot)
    return snapshot


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the read-only operator app snapshot.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_operator_app_snapshot(output_dir=args.output_dir)
    print(json.dumps({
        "status": payload.get("status"),
        "summary": payload.get("summary", {}),
        "cards": len(payload.get("cards", [])),
        "output_path": payload.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
