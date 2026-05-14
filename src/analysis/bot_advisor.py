# ============================================================
# src/analysis/bot_advisor.py
# ============================================================
"""
Bot advisor layer.

This module combines the existing analysis reports and turns them into
actionable recommendations. It does not change trading behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Iterable, Optional

from dotenv import load_dotenv

from src.analysis.bot_alerts_reporter import format_alert_message
from src.analysis.recommendation_registry import RecommendationRegistry
from src.notifier.telegram_notifier import TelegramNotifier


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "bot_advisor")
DEFAULT_LATEST_FILE = "latest_bot_advice.json"

DEFAULT_LEARNING_REPORT = os.path.join("analysis", "strategy_events", "latest_strategy_learning_report.json")
DEFAULT_PROFILE_PROPOSALS = os.path.join("analysis", "strategy_events", "latest_strategy_profile_proposals.json")
DEFAULT_GPT_REPORT = os.path.join("analysis", "gpt_decisions", "latest_gpt_decision_report.json")
DEFAULT_CHART_REPORT = os.path.join("analysis", "chart_vision", "latest_chart_vision_report.json")
DEFAULT_ALERT_REPORT = os.path.join("analysis", "bot_alerts", "latest_bot_alerts_report.json")
DEFAULT_MARKET_REGIME_REPORT = os.path.join("analysis", "market_regime", "latest_market_regime.json")
DEFAULT_OPPORTUNITY_REPORT = os.path.join("analysis", "opportunities", "latest_opportunity_report.json")
DEFAULT_SHADOW_REPORT = os.path.join("analysis", "shadow_models", "latest_shadow_model_report.json")
DEFAULT_SHADOW_EXPERIMENT_REPORT = os.path.join("analysis", "experiments", "latest_shadow_experiment_results.json")
DEFAULT_ML_EDGE_REPORT = os.path.join("analysis", "ml_models", "latest_edge_model_report.json")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
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


class BotAdvisor:
    def __init__(
        self,
        learning_report_path: str = DEFAULT_LEARNING_REPORT,
        profile_proposals_path: str = DEFAULT_PROFILE_PROPOSALS,
        gpt_report_path: str = DEFAULT_GPT_REPORT,
        chart_report_path: str = DEFAULT_CHART_REPORT,
        alert_report_path: str = DEFAULT_ALERT_REPORT,
        market_regime_path: str = DEFAULT_MARKET_REGIME_REPORT,
        opportunity_path: str = DEFAULT_OPPORTUNITY_REPORT,
        shadow_report_path: str = DEFAULT_SHADOW_REPORT,
        shadow_experiment_path: str = DEFAULT_SHADOW_EXPERIMENT_REPORT,
        ml_edge_report_path: str = DEFAULT_ML_EDGE_REPORT,
    ):
        self.learning_report_path = learning_report_path
        self.profile_proposals_path = profile_proposals_path
        self.gpt_report_path = gpt_report_path
        self.chart_report_path = chart_report_path
        self.alert_report_path = alert_report_path
        self.market_regime_path = market_regime_path
        self.opportunity_path = opportunity_path
        self.shadow_report_path = shadow_report_path
        self.shadow_experiment_path = shadow_experiment_path
        self.ml_edge_report_path = ml_edge_report_path

    def build_advice(self) -> dict:
        reports = {
            "learning": load_json(self.learning_report_path),
            "profiles": load_json(self.profile_proposals_path),
            "gpt": load_json(self.gpt_report_path),
            "chart_vision": load_json(self.chart_report_path),
            "alerts": load_json(self.alert_report_path),
            "market_regime": load_json(self.market_regime_path),
            "opportunities": load_json(self.opportunity_path),
            "shadow_models": load_json(self.shadow_report_path),
            "shadow_experiments": load_json(self.shadow_experiment_path),
            "ml_edge_model": load_json(self.ml_edge_report_path),
        }

        recommendations = []
        recommendations.extend(self._missing_report_recommendations(reports))
        recommendations.extend(self._runtime_recommendations(reports["alerts"]))
        recommendations.extend(self._gpt_decision_recommendations(reports["gpt"]))
        recommendations.extend(self._chart_vision_recommendations(reports["chart_vision"]))
        recommendations.extend(self._market_regime_recommendations(reports["market_regime"]))
        recommendations.extend(self._opportunity_recommendations(reports["opportunities"]))
        recommendations.extend(self._shadow_model_recommendations(reports["shadow_models"]))
        recommendations.extend(self._shadow_experiment_recommendations(reports["shadow_experiments"]))
        recommendations.extend(self._ml_edge_model_recommendations(reports["ml_edge_model"]))
        recommendations.extend(self._profile_recommendations(reports["profiles"]))
        recommendations.extend(self._learning_recommendations(reports["learning"]))

        recommendations.sort(key=lambda item: self._priority_rank(item.get("priority")), reverse=True)
        return {
            "created_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": self._overall_status(recommendations),
            "summary": self._summary(recommendations),
            "recommendations": recommendations,
            "sources": {
                "learning_report": self.learning_report_path,
                "profile_proposals": self.profile_proposals_path,
                "gpt_report": self.gpt_report_path,
                "chart_vision_report": self.chart_report_path,
                "alert_report": self.alert_report_path,
                "market_regime_report": self.market_regime_path,
                "opportunity_report": self.opportunity_path,
                "shadow_model_report": self.shadow_report_path,
                "shadow_experiment_report": self.shadow_experiment_path,
                "ml_edge_model_report": self.ml_edge_report_path,
            },
        }

    def _missing_report_recommendations(self, reports: dict) -> list[dict]:
        items = []
        for name, report in reports.items():
            if report.get("_missing"):
                items.append(self._rec(
                    priority="high",
                    area="pipeline",
                    finding=f"{name} report is missing.",
                    recommendation=f"Run or schedule the reporter that creates {report.get('_path')}.",
                    requires_human_approval=False,
                    evidence={"path": report.get("_path")},
                ))
            elif report.get("_error"):
                items.append(self._rec(
                    priority="high",
                    area="pipeline",
                    finding=f"{name} report could not be read.",
                    recommendation="Inspect the JSON file and reporter logs.",
                    requires_human_approval=False,
                    evidence={"path": report.get("_path"), "error": report.get("_error")},
                ))
        return items

    def _runtime_recommendations(self, alert_report: dict) -> list[dict]:
        if alert_report.get("_missing") or alert_report.get("_error"):
            return []
        items = []
        for alert in alert_report.get("alerts", []) or []:
            code = alert.get("code")
            level = alert.get("level")
            priority = "high" if level == "ALERT" else "medium"
            items.append(self._rec(
                priority=priority,
                area="runtime",
                finding=f"{code}: {alert.get('message')}",
                recommendation=self._runtime_recommendation_for(code),
                requires_human_approval=False,
                evidence=alert,
            ))
        return items

    def _gpt_decision_recommendations(self, gpt_report: dict) -> list[dict]:
        if gpt_report.get("_missing") or gpt_report.get("_error"):
            return []
        totals = gpt_report.get("totals", {}) or {}
        meta = gpt_report.get("meta", {}) or {}
        items = []
        loaded = _safe_int(meta.get("loaded_gpt_decisions"))
        zero_conf_pct = _safe_float(totals.get("zero_conf_pct"))
        missing_scores_pct = _safe_float(totals.get("missing_scores_pct"))
        cf_avg_r = _safe_float(totals.get("cf_avg_r"))
        holds_positive = len((gpt_report.get("attention_cases") or {}).get("holds_with_positive_counterfactual", []) or [])
        scored_events = max(0, loaded - _safe_int(totals.get("missing_scores")))

        if loaded < 100:
            items.append(self._rec(
                priority="medium",
                area="gpt_decision",
                finding=f"GPT decision analytics sample is still small ({loaded} labeled decisions).",
                recommendation="Keep collecting structured GPT decisions before changing prompt thresholds.",
                requires_human_approval=False,
                evidence={"loaded_gpt_decisions": loaded},
            ))

        if zero_conf_pct >= 10.0:
            items.append(self._rec(
                priority="high",
                area="gpt_runtime",
                finding=f"GPT zero-confidence rate is high ({zero_conf_pct}%).",
                recommendation="Treat this as infrastructure noise first: inspect timeouts/API latency before evaluating strategy quality.",
                requires_human_approval=False,
                evidence={"zero_conf_pct": zero_conf_pct},
            ))
        elif zero_conf_pct >= 5.0:
            items.append(self._rec(
                priority="medium",
                area="gpt_runtime",
                finding=f"GPT zero-confidence rate is elevated ({zero_conf_pct}%).",
                recommendation="Monitor after the timeout/retry change; expected target is below a few percent.",
                requires_human_approval=False,
                evidence={"zero_conf_pct": zero_conf_pct},
            ))

        if missing_scores_pct >= 25.0:
            items.append(self._rec(
                priority="low",
                area="gpt_decision",
                finding=f"Many historical GPT events are missing structured scores ({missing_scores_pct}%).",
                recommendation="Treat this as transition noise from old events. Use structured-only/newer reports for decision quality.",
                requires_human_approval=False,
                evidence={"missing_scores_pct": missing_scores_pct, "scored_events": scored_events, "loaded_gpt_decisions": loaded},
            ))

        if scored_events < 100:
            items.append(self._rec(
                priority="medium",
                area="gpt_decision",
                finding=f"Structured GPT sample is still small ({scored_events} scored events).",
                recommendation="Do not tune prompt thresholds from GPT analytics yet; collect at least 100 scored/labeled events.",
                requires_human_approval=False,
                evidence={"scored_events": scored_events, "target": 100},
            ))

        if holds_positive >= 10:
            items.append(self._rec(
                priority="medium",
                area="gpt_decision",
                finding=f"{holds_positive} HOLD decisions had strongly positive counterfactual outcomes.",
                recommendation="Investigate whether GPT is too conservative for specific vetoes/coins before loosening rules.",
                requires_human_approval=True,
                evidence={"attention_case": "holds_with_positive_counterfactual", "count": holds_positive},
            ))

        if cf_avg_r < -0.25 and loaded >= 100:
            items.append(self._rec(
                priority="medium",
                area="gpt_decision",
                finding=f"Overall GPT decision counterfactual average R is negative ({cf_avg_r}).",
                recommendation="Keep risk conservative and use per-veto/per-symbol breakdown before changing entry rules.",
                requires_human_approval=True,
                evidence={"cf_avg_r": cf_avg_r},
            ))
        return items

    def _chart_vision_recommendations(self, chart_report: dict) -> list[dict]:
        if chart_report.get("_missing") or chart_report.get("_error"):
            return []
        meta = chart_report.get("meta", {}) or {}
        totals = chart_report.get("totals", {}) or {}
        cases = chart_report.get("attention_cases", {}) or {}
        items = []
        loaded = _safe_int(meta.get("loaded_events"))
        cf_avg_r = _safe_float(totals.get("cf_avg_r"))
        noisy_positive = len(cases.get("chop_or_noisy_but_positive_cf", []) or [])
        clean_negative = len(cases.get("clean_or_pullback_but_negative_cf", []) or [])

        if loaded < 100:
            items.append(self._rec(
                priority="medium",
                area="chart_vision",
                finding=f"Chart Vision QA sample is still small ({loaded} structured/labeled events).",
                recommendation="Do not recalibrate chart labels yet; wait for at least 100 structured labeled events.",
                requires_human_approval=False,
                evidence={"loaded_events": loaded},
            ))

        if noisy_positive >= 5 and cf_avg_r > 0:
            items.append(self._rec(
                priority="medium",
                area="chart_vision",
                finding=f"Chart Vision may be too strict: {noisy_positive} chop/noisy cases had positive counterfactual outcomes.",
                recommendation="Collect more samples; then split noisy/chop into true range chop versus messy directional trend.",
                requires_human_approval=True,
                evidence={"chop_or_noisy_but_positive_cf": noisy_positive, "cf_avg_r": cf_avg_r},
            ))

        if clean_negative >= 5:
            items.append(self._rec(
                priority="medium",
                area="chart_vision",
                finding=f"Clean/pullback labels may be too optimistic: {clean_negative} cases had negative counterfactual outcomes.",
                recommendation="Review candle quality and late-trend thresholds before allowing looser entries.",
                requires_human_approval=True,
                evidence={"clean_or_pullback_but_negative_cf": clean_negative},
            ))
        return items

    def _market_regime_recommendations(self, regime_report: dict) -> list[dict]:
        if regime_report.get("_missing") or regime_report.get("_error"):
            return []
        regime = regime_report.get("regime", "unknown")
        risk_mode = regime_report.get("risk_mode", "normal")
        breadth = regime_report.get("breadth", {}) or {}
        flags = regime_report.get("flags", []) or []
        items = []

        if regime == "risk_off":
            items.append(self._rec(
                priority="medium",
                area="market_regime",
                finding="Market regime is risk_off.",
                recommendation="Keep long entries selective; review whether future risk engine should cap new longs in this regime.",
                requires_human_approval=True,
                evidence={"risk_mode": risk_mode, "breadth": breadth, "flags": flags},
            ))
        elif regime == "chop":
            items.append(self._rec(
                priority="medium",
                area="market_regime",
                finding="Market regime is chop.",
                recommendation="Expect more false starts; use chart vision and GPT veto analytics before loosening entries.",
                requires_human_approval=False,
                evidence={"risk_mode": risk_mode, "breadth": breadth, "flags": flags},
            ))
        elif regime == "unknown":
            items.append(self._rec(
                priority="low",
                area="market_regime",
                finding="Market regime has insufficient data.",
                recommendation="Verify candles_kraken has enough 4h candles for anchors and tracked pairs.",
                requires_human_approval=False,
                evidence={"breadth": breadth, "flags": flags},
            ))
        return items

    def _opportunity_recommendations(self, opportunity_report: dict) -> list[dict]:
        if opportunity_report.get("_missing") or opportunity_report.get("_error"):
            return []
        meta = opportunity_report.get("meta", {}) or {}
        totals = opportunity_report.get("totals", {}) or {}
        cases = opportunity_report.get("attention_cases", {}) or {}
        by_regime_direction = opportunity_report.get("by_regime_direction", {}) or {}
        pattern_summary = opportunity_report.get("pattern_summary", {}) or {}
        pattern_contrast = opportunity_report.get("pattern_contrast", []) or []
        pattern_feature_contrast = opportunity_report.get("pattern_feature_contrast", []) or []
        items = []

        loaded = _safe_int(meta.get("loaded_candidates"))
        hold_rate = _safe_float(totals.get("hold_rate_pct"))
        cf_avg_r = _safe_float(totals.get("cf_avg_r"))
        held_positive = len(cases.get("held_positive_opportunities", []) or [])
        held_large = len(cases.get("held_large_positive_opportunities", []) or [])
        risk_off_short = by_regime_direction.get("risk_off|short", {}) or {}

        if loaded < 100:
            items.append(self._rec(
                priority="medium",
                area="opportunities",
                finding=f"Opportunity sample is still small ({loaded} labeled candidates).",
                recommendation="Use this report for diagnostics, but wait for more labeled long/short candidates before loosening live rules.",
                requires_human_approval=False,
                evidence={"loaded_candidates": loaded},
            ))

        if loaded >= 20 and hold_rate >= 90.0 and cf_avg_r > 0.25:
            items.append(self._rec(
                priority="medium",
                area="opportunities",
                finding=f"Candidates are mostly held ({hold_rate}%) while counterfactual average R is positive ({cf_avg_r}).",
                recommendation="Review vetoes/chart labels by direction and regime before changing live rules.",
                requires_human_approval=True,
                evidence={"hold_rate_pct": hold_rate, "cf_avg_r": cf_avg_r},
            ))

        if held_large >= 5:
            top_pattern = (pattern_summary.get("held_large_positive") or [{}])[0]
            items.append(self._rec(
                priority="medium",
                area="opportunities",
                finding=f"{held_large} held candidates later had >= 1.0R counterfactual outcomes.",
                recommendation="Inspect common veto/structure/regime patterns; likely candidate for shadow prompt or chart-label tuning.",
                requires_human_approval=True,
                evidence={"held_large_positive_opportunities": held_large, "top_pattern": top_pattern},
            ))
        elif held_positive >= 10:
            items.append(self._rec(
                priority="low",
                area="opportunities",
                finding=f"{held_positive} held candidates later had positive counterfactual outcomes.",
                recommendation="Track whether this persists after chart-label and prompt changes.",
                requires_human_approval=False,
                evidence={"held_positive_opportunities": held_positive},
            ))

        if risk_off_short:
            ro_short_events = _safe_int(risk_off_short.get("events"))
            ro_short_hold_rate = _safe_float(risk_off_short.get("hold_rate_pct"))
            ro_short_cf = _safe_float(risk_off_short.get("cf_avg_r"))
            if ro_short_events >= 20 and ro_short_hold_rate >= 90.0 and ro_short_cf > 0.25:
                items.append(self._rec(
                    priority="medium",
                    area="opportunities",
                    finding=f"Risk-off shorts are mostly held ({ro_short_hold_rate}%) while cf_avg_r is positive ({ro_short_cf}).",
                    recommendation="Investigate short-specific vetoes and chart labels; this is the first place to tune after enough samples.",
                    requires_human_approval=True,
                    evidence={"regime_direction": "risk_off|short", "events": ro_short_events, "cf_avg_r": ro_short_cf},
                ))

        protected_patterns = pattern_summary.get("protected_holds") or []
        large_patterns = pattern_summary.get("held_large_positive") or []
        if protected_patterns and large_patterns:
            items.append(self._rec(
                priority="low",
                area="opportunities",
                finding="Opportunity report now has contrast patterns for missed winners and protected holds.",
                recommendation="Use pattern_summary to tune only patterns with positive edge, not the whole direction/regime.",
                requires_human_approval=False,
                evidence={
                    "top_held_large_positive": large_patterns[:3],
                    "top_protected_holds": protected_patterns[:3],
                },
            ))

        mixed_patterns = [
            row for row in pattern_contrast
            if row.get("interpretation") == "mixed_high_value_high_risk"
        ]
        conservative_patterns = [
            row for row in pattern_contrast
            if row.get("interpretation") == "possible_too_conservative"
        ]
        if mixed_patterns:
            items.append(self._rec(
                priority="medium",
                area="opportunities",
                finding="Top missed-opportunity patterns are mixed: they also protected the bot often.",
                recommendation="Do not loosen these patterns globally; add finer chart features such as directional continuation, breakdown quality and retest quality.",
                requires_human_approval=True,
                evidence={"patterns": mixed_patterns[:5]},
            ))
        if conservative_patterns:
            items.append(self._rec(
                priority="medium",
                area="opportunities",
                finding="Some patterns look possibly too conservative.",
                recommendation="Review these first for shadow-rule or prompt experiments.",
                requires_human_approval=True,
                evidence={"patterns": conservative_patterns[:5]},
            ))

        if pattern_feature_contrast:
            first = pattern_feature_contrast[0]
            items.append(self._rec(
                priority="low",
                area="opportunities",
                finding="Feature contrast is available for mixed opportunity patterns.",
                recommendation="Use feature deltas to decide which chart features should split true chop from directional continuation.",
                requires_human_approval=False,
                evidence={
                    "pattern": first.get("pattern"),
                    "held_large_positive": first.get("held_large_positive"),
                    "protected_holds": first.get("protected_holds"),
                    "numeric": first.get("numeric"),
                },
            ))
        return items

    def _shadow_model_recommendations(self, shadow_report: dict) -> list[dict]:
        if shadow_report.get("_missing") or shadow_report.get("_error"):
            return []
        meta = shadow_report.get("meta", {}) or {}
        rules = shadow_report.get("rules", {}) or {}
        recs = shadow_report.get("recommendations", []) or []
        discovered = shadow_report.get("discovered_patterns", {}) or {}
        hypotheses = shadow_report.get("generated_hypotheses", []) or []
        items = []

        loaded = _safe_int(meta.get("loaded_rows"))
        if loaded < 100:
            items.append(self._rec(
                priority="medium",
                area="shadow_models",
                finding=f"Shadow model sample is still limited ({loaded} rows).",
                recommendation="Keep shadow rules observational only until the structured dataset grows.",
                requires_human_approval=False,
                evidence={"loaded_rows": loaded},
            ))

        promising = [
            rec for rec in recs
            if rec.get("verdict") == "promising_shadow_rule"
        ]
        rejected = [
            rec for rec in recs
            if rec.get("verdict") == "reject_or_tighten"
        ]
        mixed = [
            rec for rec in recs
            if rec.get("verdict") == "mixed"
        ]

        for rec in promising[:3]:
            items.append(self._rec(
                priority="medium",
                area="shadow_models",
                finding=(
                    f"Shadow rule looks promising: {rec.get('rule')} "
                    f"(matches={rec.get('matches')}, cf_avg_r={rec.get('cf_avg_r')})."
                ),
                recommendation="Do not enable live yet; inspect sample cases and require more out-of-sample matches.",
                requires_human_approval=True,
                evidence=rec,
            ))

        for rec in rejected[:3]:
            items.append(self._rec(
                priority="medium",
                area="shadow_models",
                finding=(
                    f"Shadow rule likely hurts performance: {rec.get('rule')} "
                    f"(matches={rec.get('matches')}, cf_avg_r={rec.get('cf_avg_r')}, loss_rate={rec.get('cf_loss_rate_pct')}%)."
                ),
                recommendation="Keep this protection in place or tighten it; do not promote this rule.",
                requires_human_approval=False,
                evidence=rec,
            ))

        if mixed:
            items.append(self._rec(
                priority="low",
                area="shadow_models",
                finding=f"{len(mixed)} shadow rules are mixed.",
                recommendation="Split mixed rules by symbol, market regime and chop subtype before considering changes.",
                requires_human_approval=True,
                evidence={"rules": mixed[:5]},
            ))

        promising_patterns = []
        protective_patterns = []
        for group_name, patterns in discovered.items():
            for pattern in patterns or []:
                row = {"group": group_name, **pattern}
                if pattern.get("interpretation") == "promising_pattern":
                    promising_patterns.append(row)
                elif pattern.get("interpretation") == "protective_hold_pattern":
                    protective_patterns.append(row)

        promising_patterns.sort(key=lambda row: (row.get("cf_avg_r", 0), row.get("matches", 0)), reverse=True)
        protective_patterns.sort(key=lambda row: (row.get("cf_loss_rate_pct", 0), -row.get("cf_avg_r", 0)), reverse=True)

        if promising_patterns:
            top = promising_patterns[0]
            items.append(self._rec(
                priority="medium",
                area="shadow_models",
                finding=(
                    f"Discovered promising HOLD pattern: {top.get('pattern')} "
                    f"(matches={top.get('matches')}, cf_avg_r={top.get('cf_avg_r')})."
                ),
                recommendation="Use this pattern as a candidate for a future shadow rule; require more samples before live changes.",
                requires_human_approval=True,
                evidence={"top_patterns": promising_patterns[:5]},
            ))

        if protective_patterns:
            top = protective_patterns[0]
            items.append(self._rec(
                priority="low",
                area="shadow_models",
                finding=(
                    f"Discovered protective HOLD pattern: {top.get('pattern')} "
                    f"(loss_rate={top.get('cf_loss_rate_pct')}%, cf_avg_r={top.get('cf_avg_r')})."
                ),
                recommendation="Keep this protection; do not loosen broad rules that include this pattern.",
                requires_human_approval=False,
                evidence={"top_patterns": protective_patterns[:5]},
            ))

        relax_hypotheses = [
            item for item in hypotheses
            if item.get("hypothesis_type") == "allow_or_relax_hold"
        ]
        protect_hypotheses = [
            item for item in hypotheses
            if item.get("hypothesis_type") == "protect_or_block"
        ]

        if relax_hypotheses:
            top = relax_hypotheses[0]
            items.append(self._rec(
                priority="medium",
                area="shadow_hypotheses",
                finding=(
                    f"Generated relax/allow hypothesis: {top.get('rule_id')} "
                    f"(confidence={top.get('confidence')}, matches={top.get('matches')}, cf_avg_r={top.get('cf_avg_r')})."
                ),
                recommendation="Keep this as shadow-only until it repeats; then consider approving a controlled dry-run experiment.",
                requires_human_approval=True,
                evidence={"hypotheses": relax_hypotheses[:5]},
            ))

        if protect_hypotheses:
            top = protect_hypotheses[0]
            items.append(self._rec(
                priority="low",
                area="shadow_hypotheses",
                finding=(
                    f"Generated protection hypothesis: {top.get('rule_id')} "
                    f"(confidence={top.get('confidence')}, matches={top.get('matches')}, loss_rate={top.get('cf_loss_rate_pct')}%)."
                ),
                recommendation="Use this to avoid loosening broad rules that include this risky pattern.",
                requires_human_approval=False,
                evidence={"hypotheses": protect_hypotheses[:5]},
            ))

        if rules and not items:
            items.append(self._rec(
                priority="low",
                area="shadow_models",
                finding=f"Shadow evaluator ran {len(rules)} rules without strong action signals.",
                recommendation="Keep collecting structured events and review again after the next daily cycle.",
                requires_human_approval=False,
                evidence={"loaded_rows": loaded, "rule_count": len(rules)},
            ))
        return items

    def _shadow_experiment_recommendations(self, shadow_experiment_report: dict) -> list[dict]:
        if shadow_experiment_report.get("_missing") or shadow_experiment_report.get("_error"):
            return []
        summary = shadow_experiment_report.get("summary", {}) or {}
        results = shadow_experiment_report.get("results", []) or []
        overlap_groups = shadow_experiment_report.get("overlap_groups", []) or []
        items = []

        by_verdict = summary.get("by_verdict", {}) or {}
        replay_promising = _safe_int(by_verdict.get("promising_replay_needs_forward"))
        protection_confirmed = _safe_int(by_verdict.get("protection_confirmed"))
        protective_replay = _safe_int(by_verdict.get("protective_replay_needs_forward"))
        overlap_count = len(overlap_groups)

        if replay_promising:
            top = self._top_shadow_result(results, "promising_replay_needs_forward")
            items.append(self._rec(
                priority="medium",
                area="shadow_experiments",
                finding=f"{replay_promising} relax experiments look promising on replay but lack forward confirmation.",
                recommendation="Do not approve live or shadow promotion yet; keep collecting forward-shadow matches.",
                requires_human_approval=False,
                evidence={
                    "verdict": "promising_replay_needs_forward",
                    "count": replay_promising,
                    "top_result": top,
                },
            ))

        if protection_confirmed:
            top = self._top_shadow_result(results, "protection_confirmed")
            items.append(self._rec(
                priority="low",
                area="shadow_experiments",
                finding=f"{protection_confirmed} protection experiments are confirmed by forward shadow data.",
                recommendation="Keep these protections; do not loosen broader rules that include these patterns.",
                requires_human_approval=False,
                evidence={
                    "verdict": "protection_confirmed",
                    "count": protection_confirmed,
                    "top_result": top,
                },
            ))

        if protective_replay:
            top = self._top_shadow_result(results, "protective_replay_needs_forward")
            items.append(self._rec(
                priority="low",
                area="shadow_experiments",
                finding=f"{protective_replay} protection experiments are replay-positive but need forward confirmation.",
                recommendation="Keep observing; do not use replay-only protection evidence as an independent live rule.",
                requires_human_approval=False,
                evidence={
                    "verdict": "protective_replay_needs_forward",
                    "count": protective_replay,
                    "top_result": top,
                },
            ))

        if overlap_count:
            items.append(self._rec(
                priority="low",
                area="shadow_experiments",
                finding=f"{overlap_count} shadow experiment overlap groups detected.",
                recommendation="Treat overlapping experiments as one evidence family; do not count duplicate patterns independently.",
                requires_human_approval=False,
                evidence={
                    "overlap_groups": overlap_groups[:5],
                    "overlap_count": overlap_count,
                },
            ))
        return items

    def _top_shadow_result(self, results: list[dict], verdict: str) -> dict:
        matches = [
            result for result in results
            if ((result.get("verdict") or {}).get("primary") == verdict)
        ]
        matches.sort(
            key=lambda item: (
                _safe_int((item.get("forward_shadow_results") or {}).get("matches")),
                _safe_float((item.get("forward_shadow_results") or {}).get("cf_avg_r")),
                _safe_int((item.get("replay_results") or {}).get("matches")),
                _safe_float((item.get("replay_results") or {}).get("cf_avg_r")),
            ),
            reverse=True,
        )
        if not matches:
            return {}
        top = matches[0]
        return {
            "experiment_id": top.get("experiment_id"),
            "pattern": top.get("pattern"),
            "experiment_type": top.get("experiment_type"),
            "verdict": top.get("verdict"),
            "replay_results": top.get("replay_results"),
            "forward_shadow_results": top.get("forward_shadow_results"),
        }

    def _ml_edge_model_recommendations(self, ml_report: dict) -> list[dict]:
        if ml_report.get("_missing") or ml_report.get("_error"):
            return []
        readiness = ml_report.get("readiness", {}) or {}
        model = ml_report.get("model", {}) or {}
        summary = ml_report.get("dataset_summary", {}) or {}
        items = []

        status = readiness.get("status")
        loaded = _safe_int(readiness.get("rows"))
        positive = _safe_int(readiness.get("positive"))
        non_positive = _safe_int(readiness.get("non_positive"))
        if status == "insufficient_data":
            items.append(self._rec(
                priority="low",
                area="ml_edge_model",
                finding=(
                    f"ML Edge Model is not trained yet: rows={loaded}, "
                    f"positive={positive}, non_positive={non_positive}."
                ),
                recommendation="Keep collecting structured labeled events; model remains shadow-only and inactive.",
                requires_human_approval=False,
                evidence={"readiness": readiness, "dataset_summary": summary},
            ))
            return items

        model_status = model.get("status")
        if model_status == "trained":
            metrics = model.get("metrics", {}) or {}
            auc = _safe_float(metrics.get("classification_auc"))
            mae = _safe_float(metrics.get("regression_mae_r"))
            priority = "medium" if auc >= 0.6 else "low"
            items.append(self._rec(
                priority=priority,
                area="ml_edge_model",
                finding=f"ML Edge Model trained: AUC={metrics.get('classification_auc')} MAE_R={mae}.",
                recommendation="Use predictions only for shadow comparison until out-of-sample stability is proven.",
                requires_human_approval=True,
                evidence={"metrics": metrics, "prediction_summary": model.get("prediction_summary", {})},
            ))
        elif model_status == "dependency_missing":
            items.append(self._rec(
                priority="medium",
                area="ml_edge_model",
                finding=f"ML Edge Model dependencies missing: {model.get('reason')}",
                recommendation="Install/verify sklearn and joblib in the runtime if ML training should run on this machine.",
                requires_human_approval=False,
                evidence={"model": model},
            ))
        elif model_status == "failed":
            items.append(self._rec(
                priority="medium",
                area="ml_edge_model",
                finding=f"ML Edge Model failed: {model.get('reason')}",
                recommendation="Inspect ml_edge_model report and traceback/logs before relying on ML output.",
                requires_human_approval=False,
                evidence={"model": model},
            ))
        return items

    def _profile_recommendations(self, profiles_report: dict) -> list[dict]:
        if profiles_report.get("_missing") or profiles_report.get("_error"):
            return []
        summary = profiles_report.get("summary", {}) or {}
        items = []
        risk_down = summary.get("risk_down_symbols", []) or []
        filter_review = summary.get("filter_review_symbols", []) or []
        range_candidates = summary.get("range_breakout_candidates", []) or []

        if risk_down:
            items.append(self._rec(
                priority="medium",
                area="risk",
                finding=f"{len(risk_down)} symbols are proposed for risk-down.",
                recommendation="Risk-down is safe to keep enabled; inspect only if major coins are repeatedly capped.",
                requires_human_approval=False,
                evidence={"symbols": [r.get("symbol") for r in risk_down[:10]]},
            ))

        if filter_review:
            items.append(self._rec(
                priority="medium",
                area="filters",
                finding=f"{len(filter_review)} symbols show possible over-filtering.",
                recommendation="Use GPT/chart reports to determine whether filter strictness or chart vision labels caused missed opportunities.",
                requires_human_approval=True,
                evidence={"symbols": [r.get("symbol") for r in filter_review[:10]]},
            ))

        if range_candidates:
            items.append(self._rec(
                priority="low",
                area="chart_vision",
                finding=f"{len(range_candidates)} range breakout candidates detected.",
                recommendation="Do not trade range blindly; use this as input for a future range-breakout submode.",
                requires_human_approval=True,
                evidence={"symbols": [r.get("symbol") for r in range_candidates[:10]]},
            ))
        return items

    def _learning_recommendations(self, learning_report: dict) -> list[dict]:
        if learning_report.get("_missing") or learning_report.get("_error"):
            return []
        label_stats = learning_report.get("label_stats", {}) or {}
        report = learning_report.get("report", {}) or {}
        loaded = _safe_int((report.get("meta") or {}).get("loaded_labeled_events"))
        waiting = _safe_int(label_stats.get("waiting_for_candles"))
        items = []

        if waiting > 0:
            items.append(self._rec(
                priority="low",
                area="learning",
                finding=f"{waiting} events are waiting for enough future candles.",
                recommendation="Normal if recent events are inside the lookahead window; investigate only if it keeps growing.",
                requires_human_approval=False,
                evidence={"waiting_for_candles": waiting},
            ))

        if loaded < 500:
            items.append(self._rec(
                priority="medium",
                area="learning",
                finding=f"Learning sample is still limited ({loaded} labeled events).",
                recommendation="Keep changes conservative until labeled event count grows.",
                requires_human_approval=False,
                evidence={"loaded_labeled_events": loaded},
            ))
        return items

    def _runtime_recommendation_for(self, code: Optional[str]) -> str:
        mapping = {
            "GPT_ZERO_CONF_HIGH": "Inspect OpenAI latency/timeouts and reduce prompt payload only if timeout change is insufficient.",
            "GPT_ZERO_CONF_WARN": "Monitor next daily digest after timeout/retry change.",
            "GPT_TIMEOUTS": "Check if timeouts cluster around hourly candle processing.",
            "GPT_STRUCTURED_OUTPUT_MISSING": "Use structured-only reports for recent decisions; older events may miss scores.",
            "CANDLES_STALE": "Check Kraken data client and candle polling immediately.",
            "DISK_LOW": "Free disk space before running long backfills or analysis.",
            "WAL_EXISTS": "Confirm DB journal mode and investigate unexpected WAL file growth.",
            "LEARNING_PROFILES_LOW": "Run strategy learning job and verify coin_profiles writes.",
        }
        return mapping.get(str(code), "Inspect the related report and logs.")

    def _summary(self, recommendations: list[dict]) -> dict:
        return {
            "total": len(recommendations),
            "high": sum(1 for r in recommendations if r.get("priority") == "high"),
            "medium": sum(1 for r in recommendations if r.get("priority") == "medium"),
            "low": sum(1 for r in recommendations if r.get("priority") == "low"),
            "requires_human_approval": sum(1 for r in recommendations if r.get("requires_human_approval")),
        }

    def _overall_status(self, recommendations: list[dict]) -> str:
        if any(r.get("priority") == "high" for r in recommendations):
            return "ACTION_NEEDED"
        if any(r.get("priority") == "medium" for r in recommendations):
            return "WATCH"
        return "OK"

    def _priority_rank(self, priority: Any) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(str(priority), 0)

    def _rec(
        self,
        priority: str,
        area: str,
        finding: str,
        recommendation: str,
        requires_human_approval: bool,
        evidence: Optional[dict] = None,
    ) -> dict:
        return {
            "priority": priority,
            "area": area,
            "finding": finding,
            "recommendation": recommendation,
            "requires_human_approval": requires_human_approval,
            "evidence": evidence or {},
        }


def format_advice_message(advice: dict, max_items: int = 8) -> str:
    summary = advice.get("summary", {}) or {}
    lines = [
        f"Bot Advisor [{advice.get('status')}]",
        f"Findings: total={summary.get('total', 0)} | high={summary.get('high', 0)} | medium={summary.get('medium', 0)} | approval={summary.get('requires_human_approval', 0)}",
    ]
    for rec in (advice.get("recommendations") or [])[:max_items]:
        lines.append(
            f"- {str(rec.get('priority', '')).upper()} {rec.get('area')}: {rec.get('finding')} -> {rec.get('recommendation')}"
        )
    if not advice.get("recommendations"):
        lines.append("- No recommendations.")
    return "\n".join(lines)


def send_telegram(advice: dict) -> bool:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    TelegramNotifier(token, chat_id).safe_send(format_advice_message(advice))
    return True


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build combined bot advisor report.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--send", action="store_true", help="Send advice summary to Telegram.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    advisor = BotAdvisor()
    advice = advisor.build_advice()
    output_path = os.path.join(args.output_dir, DEFAULT_LATEST_FILE)
    write_json(output_path, advice)
    registry_summary = sync_registry(advice)
    sent = send_telegram(advice) if args.send else False

    result = {
        "status": advice.get("status"),
        "summary": advice.get("summary"),
        "output_path": output_path,
        "registry": {
            "total": registry_summary.get("total", 0),
            "by_status": registry_summary.get("by_status", {}),
            "active": len(registry_summary.get("active", [])),
        },
        "telegram_sent": sent,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def sync_registry(advice: dict) -> dict:
    return RecommendationRegistry().sync_from_advice(advice)


if __name__ == "__main__":
    raise SystemExit(main())
