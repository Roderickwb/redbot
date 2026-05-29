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

from src.analysis.operator_decision_resolver import run_operator_decision_resolver

DEFAULT_OUTPUT_DIR = os.path.join("analysis", "recommendations")
DEFAULT_LATEST_FILE = "latest_recommendation_aggregator.json"

DEFAULT_INDICATOR_EDGE = os.path.join("analysis", "indicator_edge", "latest_indicator_edge_report.json")
DEFAULT_LIVE_READINESS = os.path.join("analysis", "live_readiness", "latest_live_readiness_gate.json")
DEFAULT_RISK_ADVICE_HISTORY = os.path.join("analysis", "risk", "latest_risk_advice_history_report.json")
DEFAULT_RISK_BRIDGE_HISTORY = os.path.join("analysis", "risk", "latest_risk_bridge_history_report.json")
DEFAULT_RISK_GUARD = os.path.join("analysis", "risk", "latest_risk_guard_report.json")
DEFAULT_PRE_GPT_GATE = os.path.join("analysis", "gpt_decisions", "latest_pre_gpt_gate_report.json")
DEFAULT_ML_EDGE = os.path.join("analysis", "ml_models", "latest_edge_model_report.json")
DEFAULT_LOSS_DIAGNOSIS = os.path.join("analysis", "loss_diagnosis", "latest_loss_diagnosis_report.json")
DEFAULT_ENTRY_RULE_CANDIDATES = os.path.join("analysis", "entry_rules", "latest_entry_rule_candidate_simulator.json")
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
        loss_diagnosis_path: str = DEFAULT_LOSS_DIAGNOSIS,
        entry_rule_candidates_path: str = DEFAULT_ENTRY_RULE_CANDIDATES,
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
            "loss_diagnosis": loss_diagnosis_path,
            "entry_rule_candidates": entry_rule_candidates_path,
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
        items.extend(self._from_loss_diagnosis(reports.get("loss_diagnosis") or {}))
        items.extend(self._from_entry_rule_candidates(reports.get("entry_rule_candidates") or {}))
        items.extend(self._from_indicator_edge(reports.get("indicator_edge") or {}))
        items.extend(self._from_exit_management(reports.get("exit_management") or {}))
        items.extend(self._from_position_lifecycle(reports.get("position_lifecycle") or {}))

        items = self._dedupe(items)
        resolver_report = run_operator_decision_resolver(items=items)
        suppressed_items = resolver_report.get("suppressed_items", []) or []
        resolved_items = resolver_report.get("resolved_items", []) or []
        items = resolver_report.get("active_items", []) or items
        items.sort(key=self._sort_key)
        summary = self._summary(items)
        summary["operator_resolution"] = resolver_report.get("summary", {})
        return {
            "created_utc": _utc_now(),
            "status": "REVIEW" if summary.get("needs_operator_review") else "WATCH",
            "meta": {
                "read_only": True,
                "live_effect": False,
                "operator_review_policy": "show_all_actionable_items",
            },
            "summary": summary,
            "items": items,
            "operator_review_items": [item for item in items if item.get("status") == STATUS_REVIEW],
            "resolved_items": resolved_items,
            "suppressed_items": suppressed_items,
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
                effect_level="risk_down_live",
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
                effect_level="risk_down_live",
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
                effect_level="risk_down_live",
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
                effect_level="risk_down_live",
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
                effect_level="risk_down_live",
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
                effect_level="strategy_live",
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
                effect_level="shadow_only",
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
                effect_level="shadow_only",
                evidence=summary,
            )]
        return []

    def _from_loss_diagnosis(self, report: dict) -> list[dict]:
        summary = report.get("summary", {}) or {}
        candidates = report.get("improvement_candidates", []) or []
        if not candidates:
            return []
        top = candidates[0] or {}
        evidence = top.get("evidence") or {}
        status = STATUS_REVIEW if top.get("kind") == "problem" else STATUS_WAIT
        return [self._item(
            area=top.get("area") or "diagnosis",
            candidate_type="loss_diagnosis_candidate",
            subject=str(top.get("title") or "top_candidate"),
            status=status,
            title=top.get("title") or "Loss diagnosis found a candidate",
            headline=top.get("problem_or_opportunity") or "Loss diagnosis found a testable improvement candidate.",
            why="This is the central PnL diagnosis: it ranks where opened trades lose or find edge before changing rules.",
            default_action="wait",
            effect_level="strategy_live",
            evidence={
                "candidate": top,
                "cluster": evidence,
                "summary": summary,
                "top_loss": summary.get("top_loss"),
                "top_opportunity": summary.get("top_opportunity"),
            },
        )]

    def _from_entry_rule_candidates(self, report: dict) -> list[dict]:
        summary = report.get("summary", {}) or {}
        best = report.get("best_candidate") or {}
        if not best or _safe_float(best.get("estimated_net_R")) <= 0:
            return []
        cluster = report.get("source_cluster") or {}
        return [self._item(
            area="entry",
            candidate_type="entry_rule_candidate",
            subject=str(best.get("rule_id") or "entry_rule"),
            status=STATUS_REVIEW,
            title=best.get("title") or "Entry rule candidate is ready for paper test",
            headline=f"Replay estimate: {best.get('rule_id')} improves the diagnosed cluster by {best.get('estimated_net_R')} R.",
            why="The bot simulated concrete entry-rule variants against the top loss cluster and found a positive candidate.",
            default_action="wait",
            effect_level="strategy_live",
            evidence={
                "summary": summary,
                "best_candidate": best,
                "source_cluster": cluster,
                "candidates": report.get("candidates", [])[:6],
            },
        )]

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
                effect_level="context_live",
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
                effect_level="context_live",
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
                effect_level="context_live",
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
                effect_level="strategy_live",
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
                effect_level="strategy_live",
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
                effect_level="strategy_live",
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
                effect_level="strategy_live",
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
                effect_level="shadow_only",
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
                effect_level="shadow_only",
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
                effect_level="context_live",
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
        effect_level: str = "shadow_only",
    ) -> dict:
        item = {
            "id": _stable_id(area, candidate_type, subject),
            "area": area,
            "candidate_type": candidate_type,
            "subject": subject,
            "status": status,
            "title": title,
            "headline": headline,
            "why": why,
            "default_action": default_action,
            "effect_level": effect_level,
            "allowed_next_steps": self._allowed_next_steps(effect_level, status),
            "allowed_actions_v1": self._allowed_actions(effect_level, status),
            "live_effect": False,
            "evidence": evidence,
        }
        item.update(self._improvement_fields(item))
        item.update(self._operator_card_fields(item))
        return item

    def _improvement_fields(self, item: dict) -> dict:
        """Normalize module signals into one autonomy-loop vocabulary."""
        ctype = str(item.get("candidate_type") or "")
        status = str(item.get("status") or "")
        effect = str(item.get("effect_level") or "")

        if ctype == "indicator_edge_context":
            return {
                "candidate_kind": "opportunity",
                "improvement_area": "entry_context",
                "autonomy_stage": "context_integrated",
                "learning_question": "Welke indicator/timeframe combinaties voorspellen betere entries?",
                "proposed_change": "Gebruik sterke indicator-edge als context in coin profiles en GPT; niet als harde entryregel.",
                "test_plan": "Blijf meten of setups met deze feature betere counterfactual R en paper outcomes houden.",
                "current_use": "coin_profile/GPT context",
                "missing_use": "geen harde entry scoring of live filter",
            }
        if ctype == "ml_edge_shadow_model":
            return {
                "candidate_kind": "opportunity",
                "improvement_area": "entry_quality",
                "autonomy_stage": "shadow_context",
                "learning_question": "Kan ML setupkwaliteit voorspellen voordat of nadat GPT beslist?",
                "proposed_change": "Gebruik ML voorlopig als context/signaal; geen entry, sizing of skip-beslissing.",
                "test_plan": "Vergelijk ML-score buckets tegen latere R en gemiste/gewonnen trades.",
                "current_use": "rapport + learned_context",
                "missing_use": "geen actieve entry gate, sizing of GPT-call skip",
            }
        if ctype == "loss_diagnosis_candidate":
            evidence = item.get("evidence") or {}
            candidate = evidence.get("candidate") or {}
            return {
                "candidate_kind": candidate.get("kind") or "problem",
                "improvement_area": candidate.get("area") or "diagnosis",
                "autonomy_stage": "diagnosis",
                "learning_question": "Waar verliest of verdient de bot nu echt R/PnL?",
                "proposed_change": candidate.get("proposed_change") or "Maak een testbare hypothese voor dit cluster.",
                "test_plan": candidate.get("test_plan") or "Shadow/paper vergelijk met baseline.",
                "current_use": "loss diagnosis / operator review",
                "missing_use": "nog geen filter, sizing of entryregel actief",
            }
        if ctype == "entry_rule_candidate":
            evidence = item.get("evidence") or {}
            best = evidence.get("best_candidate") or {}
            return {
                "candidate_kind": "problem",
                "improvement_area": "entry_rule",
                "autonomy_stage": "candidate_simulated",
                "learning_question": "Welke concrete entryregel verbetert het grootste verliescluster?",
                "proposed_change": best.get("title") or "Start een gerichte paper-test voor deze entryregel.",
                "test_plan": "Paper/shadow vergelijk kandidaatregel tegen baseline voordat strategiegedrag wijzigt.",
                "current_use": "candidate simulation / operator review",
                "missing_use": "nog niet actief in entrylogica",
            }
        if ctype in {"pre_gpt_gate_shadow", "pre_gpt_gate_too_risky"}:
            return {
                "candidate_kind": "opportunity" if status != STATUS_BLOCKED else "problem",
                "improvement_area": "cost_and_entry_filter",
                "autonomy_stage": "shadow_test",
                "learning_question": "Kan de bot dure GPT-calls overslaan zonder goede entries te missen?",
                "proposed_change": "Test een pre-GPT filter in shadow; pas later eventueel GPT-call skippen.",
                "test_plan": "Meet false skips, skipped opens en netto R tegenover baseline GPT-flow.",
                "current_use": "shadow report",
                "missing_use": "geen live GPT-call skip",
            }
        if ctype == "guard_threshold_calibration":
            return {
                "candidate_kind": "problem",
                "improvement_area": "entry_guard",
                "autonomy_stage": "validation_candidate",
                "learning_question": "Blokkeert de daglimiet te veel goede entries?",
                "proposed_change": "Valideer een ruimere of slimmere max_daily_opens guard tegen baseline.",
                "test_plan": "Replay/paper vergelijk: huidige guard versus kandidaatguard op bescherming en gemiste R.",
                "current_use": "operator review",
                "missing_use": "geen aangepaste live/paper guard zolang validatie niet rond is",
            }
        if ctype in {"risk_down_advice_batch", "risk_down_helpful", "risk_down_too_strict", "live_readiness_batch"}:
            return {
                "candidate_kind": "problem" if ctype == "risk_down_too_strict" else "opportunity",
                "improvement_area": "risk_sizing",
                "autonomy_stage": "live_gate_candidate" if ctype == "live_readiness_batch" else "validation",
                "learning_question": "Moet de bot per coin/regime minder risico nemen om verliesclusters te beperken?",
                "proposed_change": "Test of conservatiever risico verlies verlaagt zonder te veel positieve expectancy te missen.",
                "test_plan": "Bridge history vergelijkt adjusted sizing met baseline outcomes en netto R.",
                "current_use": "operator review/shadow bridge",
                "missing_use": "geen live risk sizing zolang live gate niet expliciet groen is",
            }
        if ctype in {"exit_reason_instrumentation", "exit_reason_collection", "exit_management_collecting", "exit_management_review"}:
            return {
                "candidate_kind": "problem",
                "improvement_area": "exit_management",
                "autonomy_stage": "evidence_collection",
                "learning_question": "Verliest de bot door te late, te vroege of slecht gelabelde exits?",
                "proposed_change": "Gebruik verse exit reasons om later TP/SL/trailing hypotheses te maken.",
                "test_plan": "Wacht op genoeg reason-labeled exits en vergelijk exit types met realized R.",
                "current_use": "exit report/app",
                "missing_use": "nog geen TP/SL/trailing wijziging",
            }
        if ctype.startswith("position_lifecycle"):
            return {
                "candidate_kind": "health",
                "improvement_area": "position_state",
                "autonomy_stage": "readiness_check",
                "learning_question": "Is positie-administratie betrouwbaar genoeg voor app/autonomie?",
                "proposed_change": "Gebruik lifecycle status als gate voor latere autonomie.",
                "test_plan": "Blijf checken op orphan/duplicate/partial-close issues.",
                "current_use": "health/app readiness",
                "missing_use": "geen tradingbeslissing",
            }
        return {
            "candidate_kind": "problem" if status == STATUS_BLOCKED else "opportunity",
            "improvement_area": effect or "unknown",
            "autonomy_stage": "review",
            "learning_question": "Welke verbetering probeert dit signaal te bereiken?",
            "proposed_change": item.get("headline") or item.get("why") or "",
            "test_plan": "Meer bewijs verzamelen en vergelijken met baseline.",
            "current_use": "recommendation",
            "missing_use": "geen live effect",
        }

    def _operator_card_fields(self, item: dict) -> dict:
        """Render-ready operator copy for app cards.

        The app should display these fields first and only fall back to raw
        report language when a new recommendation type has no template yet.
        """
        ctype = item.get("candidate_type")
        evidence = item.get("evidence") or {}
        status = item.get("status")
        effect_level = item.get("effect_level")

        if ctype == "guard_threshold_calibration":
            issue = evidence.get("primary_issue") or {}
            summary = evidence.get("summary") or {}
            guard = issue.get("guard") or "daglimiet"
            return self._operator_fields(
                title="Daglimiet voor trades lijkt te streng",
                question="Mag de bot dit doorzetten naar validatie voor een mogelijke ruimere daglimiet?",
                summary=f"De replay ziet dat {guard} vaak ingrijpt. Dat kan bescherming zijn, maar ook kansen blokkeren.",
                consequence="Akkoord betekent: validatiefase voor deze daglimiet. Er verandert nu niets live.",
                current_phase="shadow",
                target_phase="validation",
                returns_as="live-gate voorstel voor daglimiet aanpassen",
                evidence=[
                    ("Guard", guard),
                    ("Triggers", self._fmt_value(summary.get("guard_triggers") or summary.get("triggers"))),
                    ("Shadow-resultaat", self._fmt_r(summary.get("guard_net_saved_r") or summary.get("net_saved_r"))),
                    ("Issue-effect", self._fmt_r(issue.get("net_r") or issue.get("net_saved_r"))),
                    ("Bot-oordeel", "guard lijkt te streng"),
                    ("Live effect nu", "geen"),
                    ("Volgende stap", "validatiefase, geen live wijziging"),
                ],
                actions=[
                    ("approve", "Door naar validatie"),
                    ("wait", "Meer bewijs"),
                    ("reject", "Afwijzen"),
                    ("freeze", "Parkeren"),
                ],
                recommended_action="approve",
            )

        if ctype == "loss_diagnosis_candidate":
            candidate = evidence.get("candidate") or {}
            cluster = evidence.get("cluster") or {}
            kind = candidate.get("kind") or "problem"
            return self._operator_fields(
                title="Grootste verlies/kans cluster gevonden",
                question="Mag de bot dit cluster doorzetten naar een gerichte shadow/paper test?",
                summary=candidate.get("problem_or_opportunity") or item.get("headline") or "",
                consequence="Akkoord betekent: testbare hypothese voorbereiden. Er verandert nu niets live.",
                current_phase="diagnosis",
                target_phase="validation",
                returns_as="gerichte entry/risk/coin validatie",
                evidence=[
                    ("Type", kind),
                    ("Gebied", candidate.get("area")),
                    ("Cluster", f"{cluster.get('dimension')}={cluster.get('value')}"),
                    ("Trades", self._fmt_value(cluster.get("count"))),
                    ("Netto R", self._fmt_r(cluster.get("net_R"))),
                    ("Gem. R", self._fmt_r(cluster.get("avg_R"))),
                    ("Winrate", self._fmt_pct(cluster.get("win_rate_pct"))),
                    ("Voorstel", candidate.get("proposed_change")),
                    ("Live effect nu", "geen"),
                ],
                actions=[
                    ("approve", "Door naar test"),
                    ("wait", "Meer bewijs"),
                    ("reject", "Afwijzen"),
                    ("freeze", "Parkeren"),
                ],
                recommended_action="approve" if kind == "problem" else "wait",
            )

        if ctype == "entry_rule_candidate":
            summary = evidence.get("summary") or {}
            best = evidence.get("best_candidate") or {}
            cluster = evidence.get("source_cluster") or {}
            return self._operator_fields(
                title="Entryregel klaar voor paper-test",
                question="Mag de bot deze kandidaatregel in paper/shadow tegen de baseline testen?",
                summary=f"De simulator ziet +{best.get('estimated_net_R')} R voor deze variant op het grootste verliescluster.",
                consequence="Akkoord betekent: kandidaatregel voorbereiden voor paper-test. Er verandert nu niets live.",
                current_phase="validation",
                target_phase="validation",
                returns_as="paper-test resultaat met baselinevergelijking",
                evidence=[
                    ("Cluster", f"{cluster.get('dimension')}={cluster.get('value')}" if cluster else f"{summary.get('dimension')}={summary.get('value')}"),
                    ("Regel", best.get("rule_id")),
                    ("Effect", self._fmt_r(best.get("estimated_net_R"))),
                    ("Affected trades", self._fmt_value(best.get("affected_trades"))),
                    ("Geblokkeerde/geraakte verliezers", self._fmt_value(best.get("blocked_or_adjusted_losers"))),
                    ("Gemiste/geraakte winnaars", self._fmt_value(best.get("missed_or_adjusted_winners"))),
                    ("Live effect nu", "geen"),
                ],
                actions=[
                    ("approve", "Start paper-test"),
                    ("wait", "Meer bewijs"),
                    ("reject", "Afwijzen"),
                    ("freeze", "Parkeren"),
                ],
                recommended_action="approve",
            )

        if ctype == "live_readiness_batch":
            summary = evidence.get("summary") or {}
            return self._operator_fields(
                title="Risico-omlaag klaar voor review",
                question="Mag de bot deze risico-omlaag kandidaten klaarzetten voor de live-gate, zonder nu al live gedrag te wijzigen?",
                summary="De bot ziet volwassen shadow-kandidaten voor conservatiever risicobeheer, maar live wiring staat nog uit.",
                consequence="Akkoord betekent: klaarzetten voor live-gate review. Pas na die gate kan risico-omlaag live effect krijgen.",
                current_phase="validation",
                target_phase="live_gate",
                returns_as="risk-down live voorstel",
                evidence=[
                    ("Review-kandidaten", self._fmt_value(evidence.get("review") or summary.get("ready_for_operator_review"))),
                    ("Geblokkeerd", self._fmt_value(evidence.get("blocked") or summary.get("blocked"))),
                    ("Type wijziging", "risico omlaag, geen risk-up"),
                    ("Live wiring", "uit"),
                    ("Live effect nu", "geen"),
                    ("Volgende stap", "live-gate review voorbereiden"),
                ],
                actions=[
                    ("approve", "Klaarzetten live-gate"),
                    ("wait", "Meer bewijs"),
                    ("reject", "Afwijzen"),
                    ("freeze", "Parkeren"),
                ],
                recommended_action="wait",
            )

        if ctype == "risk_down_advice_batch":
            return self._operator_fields(
                title="Risico-omlaag advies is stabiel, maar nog niet live-klaar",
                question="Wil je wachten tot bridge-resultaten bevestigen dat dit live verstandig is?",
                summary="Meerdere coins geven al meerdere dagen een risico-omlaag signaal.",
                consequence="Er verandert niets live. De bot wacht op bevestiging dat risico-omlaag niet te streng uitpakt.",
                current_phase="shadow",
                target_phase="validation",
                returns_as="risk-down validatievoorstel",
                evidence=[
                    ("Coins met signaal", self._fmt_value(evidence.get("stable_symbols"))),
                    ("Dagen gemeten", self._fmt_value(evidence.get("days"))),
                    ("Bot-oordeel", "stabiel, maar nog niet live-klaar"),
                    ("Live effect nu", "geen"),
                    ("Volgende stap", "bridge-resultaten afwachten"),
                ],
                actions=[
                    ("wait", "Meer bewijs"),
                    ("freeze", "Parkeren"),
                ],
                recommended_action="wait",
            )

        if ctype == "risk_down_too_strict":
            return self._operator_fields(
                title="Risicoverlaging lijkt nu te streng",
                question="Wil je dit blokkeren tot de sizing-logica opnieuw is gekalibreerd?",
                summary="De bridge-resultaten botsen met het risico-omlaag advies.",
                consequence="Er verandert niets live. Afwijzen of parkeren voorkomt dat dit als actief live-kandidaat terugkomt.",
                current_phase="blocked",
                target_phase="blocked",
                returns_as="nieuw voorstel alleen bij sterker bewijs",
                evidence=[
                    ("Bot-oordeel", "risico omlaag lijkt te streng"),
                    ("Netto shadow-effect", self._fmt_r(evidence.get("net_saved_r") or evidence.get("history_net_saved_r"))),
                    ("Live effect nu", "geen"),
                    ("Volgende stap", "niet doorzetten tot conflict is opgelost"),
                ],
                actions=[
                    ("reject", "Afwijzen"),
                    ("freeze", "Parkeren"),
                    ("note", "Notitie"),
                ],
                recommended_action="freeze",
            )

        if ctype in {"exit_reason_instrumentation", "exit_reason_collection", "exit_management_collecting"}:
            summary = evidence.get("summary") or evidence
            return self._operator_fields(
                title="Exit-data wordt beter bruikbaar",
                question="Wil je wachten tot nieuwe exits genoeg reden-labels hebben voor TP/SL/trailing tuning?",
                summary="Nieuwe trades krijgen betere exit-redenen, maar oude data is nog incompleet.",
                consequence="Geen TP/SL-wijziging nu. De bot verzamelt eerst betere exit-data.",
                current_phase="shadow",
                target_phase="validation",
                returns_as="exit-rule validatievoorstel",
                evidence=[
                    ("Posities bekeken", self._fmt_value(evidence.get("positions") or summary.get("positions_loaded"))),
                    ("Gesloten trades", self._fmt_value(evidence.get("closed") or summary.get("closed_positions"))),
                    ("Bot-oordeel", "exit-redenen worden verzameld"),
                    ("Live effect nu", "geen"),
                    ("Volgende stap", "wachten op nieuwe reden-labels"),
                ],
                actions=[
                    ("wait", "Meer bewijs"),
                    ("freeze", "Parkeren"),
                ],
                recommended_action="wait",
            )

        if ctype == "pre_gpt_gate_shadow":
            return self._operator_fields(
                title="GPT-besparing blijft in shadow-test",
                question="Blijft dit als shadow-test doorlopen tot bewezen is dat goede trades niet worden gemist?",
                summary="De bot ziet mogelijke GPT-call besparing, maar dit mag live GPT-calls nog niet beperken.",
                consequence="Geen live effect. De gate blijft alleen meten en vergelijken.",
                current_phase="shadow",
                target_phase="validation",
                returns_as="pre-GPT gate validatievoorstel",
                evidence=[
                    ("Mogelijke call-reductie", self._fmt_pct(evidence.get("call_reduction_pct"))),
                    ("Shadow-effect", self._fmt_r(evidence.get("estimated_net_saved_r"))),
                    ("Skipped opens", self._fmt_value(evidence.get("skipped_opens"))),
                    ("Live effect nu", "geen"),
                    ("Volgende stap", "meer false-skip bewijs"),
                ],
                actions=[
                    ("wait", "Meer bewijs"),
                    ("freeze", "Parkeren"),
                ],
                recommended_action="wait",
            )

        if ctype == "indicator_edge_context":
            top = evidence.get("top_feature") or {}
            return self._operator_fields(
                title="Indicator-analyse vond bruikbare context",
                question="Mag deze indicator-context automatisch mee blijven wegen in GPT/profielen?",
                summary="De analyse vond een indicator-signaal met duidelijke edge in de historische labels.",
                consequence="Dit is context voor GPT/profielen. Het wijzigt live scoring of orders niet direct.",
                current_phase="context",
                target_phase="context",
                returns_as="context update",
                evidence=[
                    ("Top indicator", top.get("feature") or "-"),
                    ("Edge", self._fmt_r(top.get("edge_r"))),
                    ("Datapunten", self._fmt_value(evidence.get("usable_rows"))),
                    ("Features vergeleken", self._fmt_value(evidence.get("ranked_features"))),
                    ("Live effect nu", "alleen context"),
                ],
                actions=[
                    ("note", "Notitie"),
                    ("freeze", "Parkeren"),
                ],
                recommended_action="auto_accept_context",
            )

        if ctype == "ml_edge_shadow_model":
            metrics = evidence.get("metrics") or {}
            return self._operator_fields(
                title="ML-model draait als context/shadow-bewijs",
                question="Mag dit model als bewijs/context blijven meelopen, zonder live beslisrecht?",
                summary="Het ML-model is trainbaar en geeft extra context, maar blijft shadow-only.",
                consequence="Geen live effect. ML krijgt pas later invloed na aparte validatie en approval.",
                current_phase="shadow",
                target_phase="shadow",
                returns_as="ML-validatievoorstel bij sterker bewijs",
                evidence=[
                    ("AUC", self._fmt_value(metrics.get("classification_auc"))),
                    ("MAE R", self._fmt_value(metrics.get("regression_mae_r"))),
                    ("Live effect nu", "geen"),
                    ("Volgende stap", "shadow-context blijven verzamelen"),
                ],
                actions=[
                    ("note", "Notitie"),
                    ("freeze", "Parkeren"),
                ],
                recommended_action="auto_accept_context",
            )

        if ctype == "position_lifecycle_ok":
            return self._operator_fields(
                title="Positie-administratie is gezond",
                question="Mag deze status als app-readiness context blijven meelopen?",
                summary="De lifecycle-check ziet geen hoge issues in de positie-administratie.",
                consequence="Geen live effect. Dit ondersteunt vertrouwen in app/trade-overzichten.",
                current_phase="context",
                target_phase="context",
                returns_as="app-readiness context",
                evidence=[
                    ("Master trades", self._fmt_value(evidence.get("master_trades"))),
                    ("Open", self._fmt_value(evidence.get("open_positions"))),
                    ("Issues", self._fmt_value(evidence.get("issue_count"))),
                    ("Bot-oordeel", "positielogica klopt"),
                    ("Live effect nu", "geen"),
                ],
                actions=[
                    ("note", "Notitie"),
                    ("freeze", "Parkeren"),
                ],
                recommended_action="auto_accept_context",
            )

        return self._operator_fields(
            title=item.get("title") or item.get("candidate_type") or "Aanbeveling",
            question="Welke vervolgstap wil je voor deze aanbeveling?",
            summary=item.get("headline") or "",
            consequence=item.get("why") or "Deze klik wordt opgeslagen als operatorbesluit.",
            current_phase=self._phase_from_effect(effect_level),
            target_phase=self._target_phase_from_effect(effect_level, status),
            returns_as="vervolgvoorstel bij nieuw bewijs",
            evidence=[
                ("Status", status),
                ("Effectniveau", effect_level),
                ("Live effect nu", "geen"),
            ],
            actions=[(action, self._default_action_label(action)) for action in item.get("allowed_actions_v1", [])],
            recommended_action=item.get("default_action") or "wait",
        )

    def _operator_fields(
        self,
        title: str,
        question: str,
        summary: str,
        consequence: str,
        current_phase: str,
        target_phase: str,
        returns_as: str,
        evidence: list[tuple[str, Any]],
        actions: list[tuple[str, str]],
        recommended_action: str,
    ) -> dict:
        clean_evidence = [
            {"label": str(label), "value": self._fmt_value(value)}
            for label, value in evidence
            if value not in (None, "", "-")
        ]
        return {
            "operator_title": title,
            "operator_question": question,
            "operator_summary": summary,
            "operator_consequence": consequence,
            "current_phase": current_phase,
            "target_phase": target_phase,
            "phase_label": self._phase_label(current_phase),
            "target_phase_label": self._phase_label(target_phase),
            "phase_transition_label": f"{self._phase_label(current_phase)} → {self._phase_label(target_phase)}",
            "returns_as": returns_as,
            "live_effect_now": "none",
            "operator_evidence": clean_evidence,
            "operator_actions": [{"id": action_id, "label": label} for action_id, label in actions],
            "recommended_action": recommended_action,
        }

    def _phase_label(self, phase: str) -> str:
        return {
            "diagnosis": "Diagnose",
            "context": "Context",
            "shadow": "Shadow",
            "validation": "Validatie",
            "live_gate": "Live-gate",
            "live": "Live",
            "blocked": "Geblokkeerd",
        }.get(str(phase or ""), str(phase or "Onbekend"))

    def _phase_from_effect(self, effect_level: str) -> str:
        if effect_level == "context_live":
            return "context"
        if effect_level in {"risk_down_live", "strategy_live"}:
            return "validation"
        return "shadow"

    def _target_phase_from_effect(self, effect_level: str, status: str) -> str:
        if status == STATUS_BLOCKED:
            return "blocked"
        if effect_level in {"risk_down_live", "strategy_live"}:
            return "live_gate"
        if effect_level == "context_live":
            return "context"
        return "validation"

    def _default_action_label(self, action: str) -> str:
        return {
            "approve": "Akkoord",
            "wait": "Meer bewijs",
            "reject": "Afwijzen",
            "freeze": "Parkeren",
            "snooze": "Later",
            "note": "Notitie",
        }.get(action, action)

    def _fmt_pct(self, value: Any) -> str:
        if value in (None, ""):
            return "-"
        try:
            return f"{float(value):.2f}%"
        except Exception:
            return str(value)

    def _fmt_r(self, value: Any) -> str:
        if value in (None, ""):
            return "-"
        try:
            return f"{float(value):.4f} R"
        except Exception:
            return str(value)

    def _fmt_value(self, value: Any) -> str:
        if value in (None, ""):
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)

    def _allowed_actions(self, effect_level: str, status: str) -> list[str]:
        if status == STATUS_BLOCKED:
            return ["reject", "freeze", "note"]
        if status == STATUS_AUTO_CONTEXT:
            return ["note", "freeze"]
        if effect_level in {"risk_down_live", "strategy_live"}:
            return ["approve", "reject", "wait", "freeze", "snooze", "note"]
        return ["approve", "reject", "wait", "freeze", "snooze", "note"] if status == STATUS_REVIEW else ["wait", "snooze", "note"]

    def _allowed_next_steps(self, effect_level: str, status: str) -> list[str]:
        if status == STATUS_BLOCKED:
            return ["reject_or_freeze"]
        if effect_level == "context_live":
            return ["auto_context", "approve_context_live", "wait_more_evidence", "freeze_topic"]
        if effect_level == "shadow_only":
            return ["continue_shadow", "wait_more_evidence", "freeze_topic"]
        if effect_level == "risk_down_live":
            return ["approve_pending_live_gate", "wait_more_evidence", "reject", "freeze_topic"]
        if effect_level == "strategy_live":
            return ["approve_pending_strict_live_gate", "wait_more_evidence", "reject", "freeze_topic"]
        return ["wait_more_evidence", "note"]

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
            "approved_context_live": 4,
            "approved_shadow": 4,
            "approved_pending_live_gate": 4,
        }.get(str(item.get("status")), 9)
        return (priority, str(item.get("area")), str(item.get("candidate_type")))

    def _summary(self, items: list[dict]) -> dict:
        by_status = Counter(item.get("status") or "unknown" for item in items)
        by_area = Counter(item.get("area") or "unknown" for item in items)
        by_candidate_kind = Counter(item.get("candidate_kind") or "unknown" for item in items)
        by_improvement_area = Counter(item.get("improvement_area") or "unknown" for item in items)
        by_stage = Counter(item.get("autonomy_stage") or "unknown" for item in items)
        review_items = [item for item in items if item.get("status") == STATUS_REVIEW]
        return {
            "total": len(items),
            "by_status": dict(by_status),
            "by_area": dict(by_area),
            "by_candidate_kind": dict(by_candidate_kind),
            "by_improvement_area": dict(by_improvement_area),
            "by_autonomy_stage": dict(by_stage),
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
