"""Aggregate learning/risk signals into operator-sized recommendations.

This is the layer above detailed reports. It prevents the operator app from
showing dozens of micro-signals by grouping them into a small set of review,
wait, auto-context, or blocked items. It is read-only and has no live effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

DEFAULT_OUTPUT_DIR = os.path.join("analysis", "recommendations")
DEFAULT_LATEST_FILE = "latest_recommendation_aggregator.json"

DEFAULT_INDICATOR_EDGE = os.path.join("analysis", "indicator_edge", "latest_indicator_edge_report.json")
DEFAULT_LIVE_READINESS = os.path.join("analysis", "live_readiness", "latest_live_readiness_gate.json")
DEFAULT_RISK_ADVICE_HISTORY = os.path.join("analysis", "risk", "latest_risk_advice_history_report.json")
DEFAULT_RISK_BRIDGE_HISTORY = os.path.join("analysis", "risk", "latest_risk_bridge_history_report.json")
DEFAULT_RISK_GUARD = os.path.join("analysis", "risk", "latest_risk_guard_report.json")
DEFAULT_PRE_GPT_GATE = os.path.join("analysis", "gpt_decisions", "latest_pre_gpt_gate_report.json")
DEFAULT_ML_EDGE = os.path.join("analysis", "ml_models", "latest_edge_model_report.json")
DEFAULT_EXIT_MANAGEMENT = os.path.join("analysis", "exits", "latest_exit_management_report.json")
DEFAULT_POSITION_LIFECYCLE = os.path.join("analysis", "positions", "latest_position_lifecycle_report.json")

STATUS_AUTO_CONTEXT = "auto_accept_as_context"
STATUS_WAIT = "wait_more_evidence"
STATUS_REVIEW = "needs_operator_review"
STATUS_BLOCKED = "blocked"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: str, default: Optional[dict] = None) -> dict:
    if not os.path.exists(path):
        return default or {"_missing": True, "_path": path}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e), "_path": path}


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


def _stable_id(area: str, candidate_type: str, subject: str = "") -> str:
    raw = f"{area}|{candidate_type}|{subject}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


class RecommendationAggregator:
    def __init__(
        self,
        indicator_edge_path: str = DEFAULT_INDICATOR_EDGE,
        live_readiness_path: str = DEFAULT_LIVE_READINESS,
        risk_advice_history_path: str = DEFAULT_RISK_ADVICE_HISTORY,
        risk_bridge_history_path: str = DEFAULT_RISK_BRIDGE_HISTORY,
        risk_guard_path: str = DEFAULT_RISK_GUARD,
        pre_gpt_gate_path: str = DEFAULT_PRE_GPT_GATE,
        ml_edge_path: str = DEFAULT_ML_EDGE,
        exit_management_path: str = DEFAULT_EXIT_MANAGEMENT,
        position_lifecycle_path: str = DEFAULT_POSITION_LIFECYCLE,
    ):
        self.paths = {
            "indicator_edge": indicator_edge_path,
            "live_readiness": live_readiness_path,
            "risk_advice_history": risk_advice_history_path,
            "risk_bridge_history": risk_bridge_history_path,
            "risk_guard": risk_guard_path,
            "pre_gpt_gate": pre_gpt_gate_path,
            "ml_edge": ml_edge_path,
            "exit_management": exit_management_path,
            "position_lifecycle": position_lifecycle_path,
        }

    def build_report(self) -> dict:
        reports = {name: _load_json(path, {}) for name, path in self.paths.items()}
        items = []
        items.extend(self._from_live_readiness(reports.get("live_readiness") or {}))
        items.extend(self._from_risk_advice_history(reports.get("risk_advice_history") or {}))
        items.extend(self._from_risk_bridge_history(reports.get("risk_bridge_history") or {}))
        items.extend(self._from_risk_guard(reports.get("risk_guard") or {}))
        items.extend(self._from_pre_gpt_gate(reports.get("pre_gpt_gate") or {}))
        items.extend(self._from_ml_edge(reports.get("ml_edge") or {}))
        items.extend(self._from_indicator_edge(reports.get("indicator_edge") or {}))
        items.extend(self._from_exit_management(reports.get("exit_management") or {}))
        items.extend(self._from_position_lifecycle(reports.get("position_lifecycle") or {}))

        items = self._dedupe(items)
        items.sort(key=self._sort_key)
        summary = self._summary(items)
        return {
            "created_utc": _utc_now(),
            "status": "REVIEW" if summary.get("needs_operator_review") else "WATCH",
            "meta": {
                "read_only": True,
                "live_effect": False,
                "max_operator_review_items": 5,
            },
            "summary": summary,
            "items": items,
            "operator_review_items": [item for item in items if item.get("status") == STATUS_REVIEW][:5],
            "sources": self.paths,
        }

    def _from_live_readiness(self, report: dict) -> list[dict]:
        summary = report.get("summary", {}) or {}
        review = _safe_int(summary.get("ready_for_operator_review"))
        blocked = _safe_int(summary.get("blocked"))
        if review:
            return [self._item(
                area="autonomy",
                candidate_type="live_readiness_batch",
                status=STATUS_REVIEW,
                title="Live-readiness has review candidates",
                headline=f"{review} candidates are ready for operator review, with live wiring still disabled.",
                why="The live-readiness gate found mature shadow candidates, but app v1 and safety state keep them read-only.",
                default_action="wait",
                evidence={"review": review, "blocked": blocked, "summary": summary},
            )]
        if blocked:
            return [self._item(
                area="autonomy",
                candidate_type="live_readiness_blocked",
                status=STATUS_BLOCKED,
                title="Live-readiness has blocked candidates",
                headline=f"{blocked} candidates are blocked by evidence or safety gates.",
                why="Blocked candidates should not become live behavior.",
                default_action="reject_or_freeze",
                evidence={"blocked": blocked, "summary": summary},
            )]
        return []

    def _from_risk_advice_history(self, report: dict) -> list[dict]:
        summary = report.get("summary", {}) or {}
        verdict = summary.get("verdict")
        stable = _safe_int(summary.get("stable_data_down_symbols"))
        days = _safe_int(summary.get("days_observed"))
        if verdict == "stable_data_down_candidates" and stable:
            return [self._item(
                area="risk",
                candidate_type="risk_down_advice_batch",
                status=STATUS_WAIT,
                title="Risk-down advice is stable but not live-ready",
                headline=f"{stable} symbols have stable data-down advice across {days} days.",
                why="Advice is stable, but bridge outcomes currently say risk-down may be too strict, so this stays read-only.",
                default_action="wait",
                evidence={"stable_symbols": stable, "days": days, "top": summary.get("top_stable_data_down", [])[:5]},
            )]
        return []

    def _from_risk_bridge_history(self, report: dict) -> list[dict]:
        summary = report.get("summary", {}) or {}
        verdict = summary.get("verdict")
        if verdict == "risk_down_too_strict":
            return [self._item(
                area="risk",
                candidate_type="risk_down_too_strict",
                status=STATUS_BLOCKED,
                title="Risk-down looks too strict",
                headline="Risk bridge history says current risk-down shadow policy may cut too much.",
                why="This conflicts with data-down candidates, so risk-down must remain shadow-only until sizing logic is recalibrated.",
                default_action="freeze",
                evidence=summary,
            )]
        if verdict == "stable_risk_down_helpful":
            return [self._item(
                area="risk",
                candidate_type="risk_down_helpful",
                status=STATUS_REVIEW,
                title="Risk-down may be helpful",
                headline="Risk bridge history is stable-positive and can be reviewed as a future conservative live candidate.",
                why="This still requires explicit operator review and safety wiring; no live effect in app v1.",
                default_action="wait",
                evidence=summary,
            )]
        return []

    def _from_risk_guard(self, report: dict) -> list[dict]:
        summary = report.get("summary", {}) or {}
        verdict = summary.get("verdict")
        issue = summary.get("primary_issue") or {}
        if verdict == "guards_too_strict":
            guard = issue.get("guard") or "risk_guard"
            return [self._item(
                area="risk",
                candidate_type="guard_threshold_calibration",
                subject=guard,
                status=STATUS_REVIEW,
                title="Risk guard threshold needs calibration",
                headline=f"Guard {guard} looks too strict in shadow replay.",
                why="This is a threshold/design review item, not a live enforcement candidate.",
                default_action="wait",
                evidence={"verdict": verdict, "primary_issue": issue, "summary": summary},
            )]
        return []

    def _from_pre_gpt_gate(self, report: dict) -> list[dict]:
        summary = report.get("summary", {}) or {}
        verdict = summary.get("verdict")
        if verdict == "promising_for_shadow":
            return [self._item(
                area="cost",
                candidate_type="pre_gpt_gate_shadow",
                status=STATUS_WAIT,
                title="Pre-GPT gate is promising in shadow",
                headline=f"Potential call reduction {summary.get('call_reduction_pct')}%, net_R={summary.get('estimated_net_saved_r')}.",
                why="The cost-saving gate needs more false-skip validation before it can affect live GPT calls.",
                default_action="wait",
                evidence=summary,
            )]
        if verdict == "too_risky":
            return [self._item(
                area="cost",
                candidate_type="pre_gpt_gate_too_risky",
                status=STATUS_BLOCKED,
                title="Pre-GPT gate is too risky",
                headline="Shadow pre-GPT gate would skip too many useful decisions.",
                why="Do not reduce GPT calls live while useful opens are still skipped.",
                default_action="reject_or_freeze",
                evidence=summary,
            )]
        return []

    def _from_ml_edge(self, report: dict) -> list[dict]:
        readiness = report.get("readiness", {}) or {}
        model = report.get("model", {}) or {}
        if readiness.get("status") == "ready" and model.get("status") == "trained":
            metrics = model.get("metrics", {}) or {}
            return [self._item(
                area="learning",
                candidate_type="ml_edge_shadow_model",
                status=STATUS_AUTO_CONTEXT,
                title="ML edge model is trained for shadow context",
                headline=f"ML shadow model trained with auc={metrics.get('classification_auc')} and mae_R={metrics.get('regression_mae_r')}.",
                why="This is accepted as read-only evidence/context, not as live decision authority.",
                default_action="auto_accept_context",
                evidence={"readiness": readiness, "metrics": metrics, "prediction_summary": model.get("prediction_summary", {})},
            )]
        return []

    def _from_indicator_edge(self, report: dict) -> list[dict]:
        summary = report.get("summary", {}) or {}
        top = summary.get("top_feature") or {}
        usable = _safe_int(summary.get("usable_rows"))
        ranked = _safe_int(summary.get("ranked_features"))
        edge = abs(_safe_float(top.get("edge_r")))
        if usable >= 250 and ranked and edge >= 0.5:
            return [self._item(
                area="learning",
                candidate_type="indicator_edge_context",
                subject=str(top.get("feature") or "top_feature"),
                status=STATUS_AUTO_CONTEXT,
                title="Indicator edge found useful context",
                headline=f"Top feature {top.get('feature')} has edge_R={top.get('edge_r')} across {usable} rows.",
                why="This can enrich coin/profile/GPT context, but it should not directly alter live scoring yet.",
                default_action="auto_accept_context",
                evidence={"top_feature": top, "weak_feature": summary.get("weak_feature"), "usable_rows": usable, "ranked_features": ranked},
            )]
        if usable:
            return [self._item(
                area="learning",
                candidate_type="indicator_edge_collecting",
                status=STATUS_WAIT,
                title="Indicator edge is collecting evidence",
                headline=f"Indicator edge ranked {ranked} features across {usable} rows.",
                why="Evidence is not strong enough for a bundled context recommendation yet.",
                default_action="wait",
                evidence={"top_feature": top, "usable_rows": usable, "ranked_features": ranked},
            )]
        return []

    def _from_exit_management(self, report: dict) -> list[dict]:
        summary = report.get("summary", {}) or {}
        verdict = summary.get("verdict")
        positions = _safe_int(summary.get("positions_loaded"))
        closed = _safe_int(summary.get("closed_positions"))
        if verdict == "exit_logging_needs_reason_field":
            return [self._item(
                area="exits",
                candidate_type="exit_reason_instrumentation",
                status=STATUS_WAIT,
                title="Exit report needs explicit close reasons",
                headline=f"Exit report found {closed} closed positions, but close reasons are inferred from child trades.",
                why="Before tuning TP/SL/trailing rules, the bot should persist explicit exit reasons so the app can judge exits with context.",
                default_action="wait",
                evidence={"positions": positions, "closed": closed, "summary": summary},
            )]
        if verdict == "exit_logging_collecting_reasons":
            return [self._item(
                area="exits",
                candidate_type="exit_reason_collection",
                status=STATUS_WAIT,
                title="Exit reason logging is collecting fresh data",
                headline=f"Exit report has {closed} closed positions, but older rows still miss explicit close reasons.",
                why="New exits can now be judged by reason; wait for fresh reason-labeled exits before tuning TP/SL/trailing rules.",
                default_action="wait",
                evidence={"positions": positions, "closed": closed, "summary": summary},
            )]
        if verdict == "exit_data_ready_for_review":
            return [self._item(
                area="exits",
                candidate_type="exit_management_review",
                status=STATUS_REVIEW,
                title="Exit management is ready for review",
                headline=f"Exit report has {closed} closed positions with win={summary.get('win_rate_pct')}% and pnl={summary.get('total_realized_pnl_eur')}.",
                why="This is a human review item for later exit-rule tuning; no live behavior changes in app v1.",
                default_action="wait",
                evidence=summary,
            )]
        if positions:
            return [self._item(
                area="exits",
                candidate_type="exit_management_collecting",
                status=STATUS_WAIT,
                title="Exit management is collecting evidence",
                headline=f"Exit report is tracking {positions} positions and {closed} closed positions.",
                why="Sample is still building before exit rules should be tuned.",
                default_action="wait",
                evidence=summary,
            )]
        return []

    def _from_position_lifecycle(self, report: dict) -> list[dict]:
        summary = report.get("summary", {}) or {}
        high = _safe_int(summary.get("high_issues"))
        medium = _safe_int(summary.get("medium_issues"))
        issues = _safe_int(summary.get("issue_count"))
        if high:
            return [self._item(
                area="positions",
                candidate_type="position_lifecycle_integrity",
                status=STATUS_REVIEW,
                title="Position lifecycle has integrity issues",
                headline=f"Lifecycle report found {high} high issues across {summary.get('master_trades')} master trades.",
                why="The app and future autonomy should not rely on position state until high-integrity bookkeeping issues are understood.",
                default_action="freeze",
                evidence=summary,
            )]
        if medium:
            return [self._item(
                area="positions",
                candidate_type="position_lifecycle_review",
                status=STATUS_WAIT,
                title="Position lifecycle has review items",
                headline=f"Lifecycle report found {medium} medium review items.",
                why="These are not live blockers, but they should stay visible before app decisions become operational.",
                default_action="wait",
                evidence=summary,
            )]
        if summary.get("master_trades"):
            return [self._item(
                area="positions",
                candidate_type="position_lifecycle_ok",
                status=STATUS_AUTO_CONTEXT,
                title="Position lifecycle is tracked",
                headline=f"Lifecycle report is tracking {summary.get('master_trades')} master trades with {issues} issues.",
                why="This is accepted as app-readiness context with no live effect.",
                default_action="auto_accept_context",
                evidence=summary,
            )]
        return []

    def _item(
        self,
        area: str,
        candidate_type: str,
        status: str,
        title: str,
        headline: str,
        why: str,
        default_action: str,
        evidence: dict,
        subject: str = "",
    ) -> dict:
        return {
            "id": _stable_id(area, candidate_type, subject),
            "area": area,
            "candidate_type": candidate_type,
            "subject": subject,
            "status": status,
            "title": title,
            "headline": headline,
            "why": why,
            "default_action": default_action,
            "allowed_actions_v1": ["approve", "reject", "wait", "freeze", "snooze", "note"] if status == STATUS_REVIEW else ["wait", "snooze", "note"],
            "live_effect": False,
            "evidence": evidence,
        }

    def _dedupe(self, items: list[dict]) -> list[dict]:
        seen = set()
        result = []
        for item in items:
            item_id = item.get("id")
            if item_id in seen:
                continue
            seen.add(item_id)
            result.append(item)
        return result

    def _sort_key(self, item: dict) -> tuple[int, str, str]:
        priority = {
            STATUS_REVIEW: 0,
            STATUS_BLOCKED: 1,
            STATUS_WAIT: 2,
            STATUS_AUTO_CONTEXT: 3,
        }.get(str(item.get("status")), 9)
        return (priority, str(item.get("area")), str(item.get("candidate_type")))

    def _summary(self, items: list[dict]) -> dict:
        by_status = Counter(item.get("status") or "unknown" for item in items)
        by_area = Counter(item.get("area") or "unknown" for item in items)
        review_items = [item for item in items if item.get("status") == STATUS_REVIEW]
        return {
            "total": len(items),
            "by_status": dict(by_status),
            "by_area": dict(by_area),
            "needs_operator_review": by_status.get(STATUS_REVIEW, 0),
            "auto_accept_as_context": by_status.get(STATUS_AUTO_CONTEXT, 0),
            "wait_more_evidence": by_status.get(STATUS_WAIT, 0),
            "blocked": by_status.get(STATUS_BLOCKED, 0),
            "top_review": review_items[:3],
            "live_effect": False,
        }


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_recommendation_aggregator(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    report = RecommendationAggregator().build_report()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build read-only recommendation aggregator report.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_recommendation_aggregator(output_dir=args.output_dir)
    print(json.dumps({
        "status": report.get("status"),
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
