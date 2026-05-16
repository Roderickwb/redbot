# ============================================================
# src/analysis/live_readiness_gate.py
# ============================================================
"""
Read-only live readiness gate.

This module centralizes the question: "Is any shadow/risk/learning signal mature
enough to consider live wiring?" It never enables live behavior. It only marks
candidate areas as blocked, waiting, or ready for operator review.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "live_readiness")
DEFAULT_LATEST_FILE = "latest_live_readiness_gate.json"

DEFAULT_SAFETY_CONTROL = os.path.join("analysis", "safety", "latest_safety_control_report.json")
DEFAULT_PROMOTION_GATE = os.path.join("analysis", "promotion_gate", "latest_promotion_gate_report.json")
DEFAULT_RISK_BRIDGE_HISTORY = os.path.join("analysis", "risk", "latest_risk_bridge_history_report.json")
DEFAULT_RISK_GUARD = os.path.join("analysis", "risk", "latest_risk_guard_report.json")
DEFAULT_ML_EDGE = os.path.join("analysis", "ml_models", "latest_edge_model_report.json")
DEFAULT_PRE_GPT_GATE = os.path.join("analysis", "gpt_decisions", "latest_pre_gpt_gate_report.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except Exception:
        return default


def _load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e), "_path": path}


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class LiveReadinessGate:
    def __init__(
        self,
        safety_path: str = DEFAULT_SAFETY_CONTROL,
        promotion_gate_path: str = DEFAULT_PROMOTION_GATE,
        risk_bridge_history_path: str = DEFAULT_RISK_BRIDGE_HISTORY,
        risk_guard_path: str = DEFAULT_RISK_GUARD,
        ml_edge_path: str = DEFAULT_ML_EDGE,
        pre_gpt_gate_path: str = DEFAULT_PRE_GPT_GATE,
    ):
        self.safety_path = safety_path
        self.promotion_gate_path = promotion_gate_path
        self.risk_bridge_history_path = risk_bridge_history_path
        self.risk_guard_path = risk_guard_path
        self.ml_edge_path = ml_edge_path
        self.pre_gpt_gate_path = pre_gpt_gate_path

    def build_report(self) -> dict:
        reports = {
            "safety": _load_json(self.safety_path, {}),
            "promotion_gate": _load_json(self.promotion_gate_path, {"summary": {}, "decisions": []}),
            "risk_bridge_history": _load_json(self.risk_bridge_history_path, {"summary": {}}),
            "risk_guard": _load_json(self.risk_guard_path, {"summary": {}}),
            "ml_edge": _load_json(self.ml_edge_path, {"readiness": {}, "model": {}}),
            "pre_gpt_gate": _load_json(self.pre_gpt_gate_path, {"summary": {}}),
        }
        safety = self._safety_context(reports.get("safety") or {})
        decisions = [
            self._risk_down_decision(reports.get("risk_bridge_history") or {}, safety),
            self._risk_guard_decision(reports.get("risk_guard") or {}, safety),
            self._promotion_decision(reports.get("promotion_gate") or {}, safety),
            self._ml_decision(reports.get("ml_edge") or {}, safety),
            self._pre_gpt_decision(reports.get("pre_gpt_gate") or {}, safety),
        ]

        return {
            "created_utc": _utc_now(),
            "meta": {
                "read_only": True,
                "live_enforcement": False,
                "safety_live_enforcement_allowed": safety.get("live_enforcement_allowed"),
                "safety_live_risk_wiring_allowed": safety.get("live_risk_wiring_allowed"),
                "safety_live_strategy_wiring_allowed": safety.get("live_strategy_wiring_allowed"),
            },
            "safety": safety,
            "summary": self._summary(decisions),
            "decisions": decisions,
            "sources": {
                "safety": self.safety_path,
                "promotion_gate": self.promotion_gate_path,
                "risk_bridge_history": self.risk_bridge_history_path,
                "risk_guard": self.risk_guard_path,
                "ml_edge": self.ml_edge_path,
                "pre_gpt_gate": self.pre_gpt_gate_path,
            },
        }

    def _safety_context(self, safety: dict) -> dict:
        meltdown = safety.get("meltdown") or {}
        return {
            "status": safety.get("status") or "UNKNOWN",
            "kill_switch_active": bool(safety.get("kill_switch_active")),
            "meltdown_active": bool(meltdown.get("active")),
            "live_entry_orders_allowed": bool(safety.get("live_entry_orders_allowed", True)),
            "live_enforcement_allowed": bool(safety.get("live_enforcement_allowed")),
            "live_risk_wiring_allowed": bool(safety.get("live_risk_wiring_allowed")),
            "live_strategy_wiring_allowed": bool(safety.get("live_strategy_wiring_allowed")),
            "reason": safety.get("reason"),
            "meltdown_reason": meltdown.get("reason"),
        }

    def _risk_down_decision(self, history: dict, safety: dict) -> dict:
        summary = history.get("summary", {}) or {}
        verdict = summary.get("verdict")
        evidence = {
            "unique_adjusted_labeled_events": _safe_int(summary.get("unique_adjusted_labeled_events")),
            "days_observed": _safe_int(summary.get("days_observed")),
            "estimated_net_saved_r": _safe_float(summary.get("estimated_net_saved_r")),
            "estimated_saved_r": _safe_float(summary.get("estimated_saved_r")),
            "estimated_missed_r": _safe_float(summary.get("estimated_missed_r")),
            "verdict": verdict,
        }
        if self._safety_blocks(safety):
            status, reason = "blocked", "safety_blocks_live_changes"
            action = "Keep risk sizing read-only until kill-switch/meltdown state is clear."
        elif verdict == "stable_risk_down_helpful":
            status, reason = "ready_for_operator_review", "risk_down_history_is_stable_positive"
            action = "Operator may review risk-down live wiring, starting with conservative sizing only."
        elif verdict == "risk_down_too_strict":
            status, reason = "blocked", "risk_down_history_says_too_strict"
            action = "Do not wire risk-down live; shadow evidence suggests it may cut too much."
        else:
            status, reason = "waiting_for_more_evidence", verdict or "risk_down_history_not_ready"
            action = "Keep collecting unique labeled adjusted outcomes before live wiring."
        return self._decision("risk", "risk_down_sizing", status, reason, action, evidence, safety)

    def _risk_guard_decision(self, guard: dict, safety: dict) -> dict:
        summary = guard.get("summary", {}) or {}
        verdict = summary.get("verdict")
        evidence = {
            "loaded_open_trades": _safe_int(summary.get("loaded_open_trades")),
            "guard_triggers": _safe_int(summary.get("guard_triggers")),
            "estimated_net_saved_r": _safe_float(summary.get("estimated_net_saved_r")),
            "estimated_saved_r": _safe_float(summary.get("estimated_saved_r")),
            "estimated_missed_r": _safe_float(summary.get("estimated_missed_r")),
            "verdict": verdict,
        }
        if self._safety_blocks(safety):
            status, reason = "blocked", "safety_blocks_live_changes"
            action = "Keep risk guards read-only until kill-switch/meltdown state is clear."
        elif verdict == "guards_look_helpful":
            status, reason = "ready_for_operator_review", "risk_guards_shadow_positive"
            action = "Operator may review risk guard thresholds before any live wiring."
        elif verdict == "guards_too_strict":
            status, reason = "blocked", "risk_guards_too_strict"
            action = "Do not wire these guards live; thresholds need tuning or more evidence."
        else:
            status, reason = "waiting_for_more_evidence", verdict or "risk_guards_not_ready"
            action = "Keep risk guards in shadow while more open-trade outcomes accumulate."
        return self._decision("risk", "risk_guards", status, reason, action, evidence, safety)

    def _promotion_decision(self, promotion: dict, safety: dict) -> dict:
        summary = promotion.get("summary", {}) or {}
        ready = _safe_int(summary.get("ready_for_human_review"))
        blocked = _safe_int(summary.get("blocked"))
        waiting = _safe_int(summary.get("waiting"))
        evidence = {
            "total": _safe_int(summary.get("total")),
            "ready_for_human_review": ready,
            "confirmed_protection": _safe_int(summary.get("confirmed_protection")),
            "blocked": blocked,
            "waiting": waiting,
            "by_status": summary.get("by_status", {}),
        }
        if self._safety_blocks(safety):
            status, reason = "blocked", "safety_blocks_live_changes"
            action = "Do not promote experiments while safety blocks live changes."
        elif ready:
            status, reason = "ready_for_operator_review", "promotion_gate_has_ready_candidates"
            action = "Review promotion-gate candidates explicitly before any live rule wiring."
        elif blocked:
            status, reason = "blocked", "promotion_gate_blocked_candidates"
            action = "Do not promote blocked experiments."
        else:
            status, reason = "waiting_for_more_evidence", "promotion_gate_waiting_or_empty"
            action = "Keep experiments in shadow until promotion gate marks them ready."
        return self._decision("strategy", "promotion_gate_experiments", status, reason, action, evidence, safety)

    def _ml_decision(self, ml_edge: dict, safety: dict) -> dict:
        readiness = ml_edge.get("readiness", {}) or {}
        model = ml_edge.get("model", {}) or {}
        status_value = readiness.get("status") or model.get("status")
        model_status = model.get("status") or ml_edge.get("model_status")
        evidence = {
            "readiness": status_value,
            "model_status": model_status,
            "rows": _safe_int(readiness.get("rows")),
            "positive": _safe_int(readiness.get("positive")),
            "non_positive": _safe_int(readiness.get("non_positive")),
            "reason": readiness.get("reason"),
        }
        if self._safety_blocks(safety):
            status, reason = "blocked", "safety_blocks_live_changes"
            action = "Do not use ML in live decisions while safety blocks live changes."
        elif status_value in {"ready", "trained"} and model_status not in {None, "not_trained"}:
            status, reason = "ready_for_operator_review", "ml_edge_model_ready_for_validation"
            action = "Validate ML metrics and drift before considering any live influence."
        else:
            status, reason = "waiting_for_more_evidence", status_value or "ml_not_ready"
            action = "Keep collecting structured labeled events before using ML predictions."
        return self._decision("learning", "ml_edge_model", status, reason, action, evidence, safety)

    def _pre_gpt_decision(self, gate: dict, safety: dict) -> dict:
        summary = gate.get("summary", {}) or {}
        verdict = summary.get("verdict")
        evidence = {
            "evaluated_decisions": _safe_int(summary.get("evaluated_decisions")),
            "would_skip_gpt": _safe_int(summary.get("would_skip_gpt")),
            "call_reduction_pct": _safe_float(summary.get("call_reduction_pct")),
            "skipped_opens": _safe_int(summary.get("skipped_opens")),
            "estimated_net_saved_r": _safe_float(summary.get("estimated_net_saved_r")),
            "verdict": verdict,
        }
        if self._safety_blocks(safety):
            status, reason = "blocked", "safety_blocks_live_changes"
            action = "Keep pre-GPT gating read-only while safety blocks live changes."
        elif verdict == "promising_for_shadow":
            status, reason = "ready_for_operator_review", "pre_gpt_gate_shadow_promising"
            action = "Review false-skip risk before reducing live GPT calls."
        elif verdict == "too_risky":
            status, reason = "blocked", "pre_gpt_gate_too_risky"
            action = "Do not reduce GPT calls live; shadow gate skips useful decisions."
        else:
            status, reason = "waiting_for_more_evidence", verdict or "pre_gpt_gate_not_ready"
            action = "Keep pre-GPT gate in shadow; GPT cost optimization is not live-ready."
        return self._decision("cost", "pre_gpt_gate", status, reason, action, evidence, safety)

    def _decision(
        self,
        area: str,
        candidate_type: str,
        status: str,
        reason: str,
        next_action: str,
        evidence: dict,
        safety: dict,
    ) -> dict:
        return {
            "id": f"{area}:{candidate_type}",
            "area": area,
            "candidate_type": candidate_type,
            "status": status,
            "reason": reason,
            "next_action": next_action,
            "evidence": evidence,
            "safety": {
                "live_enforcement_allowed": safety.get("live_enforcement_allowed"),
                "kill_switch_active": safety.get("kill_switch_active"),
                "meltdown_active": safety.get("meltdown_active"),
            },
        }

    def _safety_blocks(self, safety: dict) -> bool:
        return bool(safety.get("kill_switch_active") or safety.get("meltdown_active"))

    def _summary(self, decisions: list[dict]) -> dict:
        by_status = Counter(row.get("status") or "unknown" for row in decisions)
        by_area = Counter(row.get("area") or "unknown" for row in decisions)
        return {
            "total": len(decisions),
            "by_status": dict(by_status),
            "by_area": dict(by_area),
            "eligible_for_live_wiring": by_status.get("eligible_for_live_wiring", 0),
            "ready_for_operator_review": by_status.get("ready_for_operator_review", 0),
            "approved_but_safety_locked": by_status.get("approved_but_safety_locked", 0),
            "blocked": by_status.get("blocked", 0),
            "waiting": by_status.get("waiting_for_more_evidence", 0),
        }


def run_live_readiness_gate(
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    report = LiveReadinessGate().build_report()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build read-only live readiness gate report.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_live_readiness_gate(output_dir=args.output_dir)
    print(json.dumps({
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
