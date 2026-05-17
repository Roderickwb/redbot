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
    "live_readiness": os.path.join("analysis", "live_readiness", "latest_live_readiness_gate_report.json"),
    "risk_advice_history": os.path.join("analysis", "risk", "latest_risk_advice_history_report.json"),
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

        source_health = {
            name: {"status": _source_status(payload), "path": self.sources.get(name)}
            for name, payload in loaded.items()
        }
        missing_or_error = [name for name, item in source_health.items() if item["status"] != "ok"]

        app_status = self._app_status(cockpit, safety, missing_or_error)
        summary = self._summary(cockpit, control, safety, approvals, decisions, live_readiness, risk_advice_history)

        return {
            "created_utc": _utc_now(),
            "status": app_status,
            "summary": summary,
            "cards": self._cards(cockpit, safety, approvals, decisions, live_readiness, risk_advice_history),
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
            },
        }

    def _app_status(self, cockpit: dict, safety: dict, missing_or_error: list[str]) -> str:
        if safety.get("kill_switch_active") or safety.get("meltdown_active"):
            return "ACTION_NEEDED"
        if cockpit.get("status") in {"ACTION_NEEDED", "REVIEW"}:
            return str(cockpit.get("status"))
        if missing_or_error:
            return "DEGRADED"
        return str(cockpit.get("status") or "WATCH")

    def _summary(self, cockpit: dict, control: dict, safety: dict, approvals: dict, decisions: dict, live_readiness: dict, risk_advice_history: dict) -> dict:
        approval_summary = approvals.get("summary") or {}
        decision_summary = decisions.get("summary") or {}
        readiness_summary = live_readiness.get("summary") or {}
        risk_summary = risk_advice_history.get("summary") or {}
        return {
            "daily_decision": (cockpit.get("daily_decision") or {}).get("label"),
            "cockpit_status": cockpit.get("status"),
            "control_status": control.get("status"),
            "kill_switch_active": bool(safety.get("kill_switch_active")),
            "meltdown_active": bool(safety.get("meltdown_active")),
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
            "live_effect": False,
        }

    def _cards(self, cockpit: dict, safety: dict, approvals: dict, decisions: dict, live_readiness: dict, risk_advice_history: dict) -> list[dict]:
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