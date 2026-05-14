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
DEFAULT_ML_EDGE_REPORT = os.path.join("analysis", "ml_models", "latest_edge_model_report.json")
DEFAULT_ALERT_REPORT = os.path.join("analysis", "bot_alerts", "latest_bot_alerts_report.json")
DEFAULT_MARKET_REGIME_REPORT = os.path.join("analysis", "market_regime", "latest_market_regime.json")


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
        ml_edge_path: str = DEFAULT_ML_EDGE_REPORT,
        alerts_path: str = DEFAULT_ALERT_REPORT,
        market_regime_path: str = DEFAULT_MARKET_REGIME_REPORT,
    ):
        self.daily_job_path = daily_job_path
        self.advisor_path = advisor_path
        self.registry_path = registry_path
        self.experiment_path = experiment_path
        self.shadow_results_path = shadow_results_path
        self.promotion_gate_path = promotion_gate_path
        self.approval_inbox_path = approval_inbox_path
        self.ml_edge_path = ml_edge_path
        self.alerts_path = alerts_path
        self.market_regime_path = market_regime_path

    def build_report(self) -> dict:
        reports = {
            "daily_job": load_json(self.daily_job_path),
            "advisor": load_json(self.advisor_path),
            "registry": load_json(self.registry_path),
            "experiments": load_json(self.experiment_path),
            "shadow_results": load_json(self.shadow_results_path),
            "promotion_gate": load_json(self.promotion_gate_path),
            "approval_inbox": load_json(self.approval_inbox_path),
            "ml_edge": load_json(self.ml_edge_path),
            "alerts": load_json(self.alerts_path),
            "market_regime": load_json(self.market_regime_path),
        }

        blockers = self._blockers(reports)
        approval_queue = self._approval_queue(reports)
        experiment_status = self._experiment_status(reports)
        promotion_status = self._promotion_status(reports)
        approval_status = self._approval_status(reports)
        learning_status = self._learning_status(reports)
        operating_state = self._operating_state(reports, blockers, approval_queue)
        next_actions = self._next_actions(blockers, approval_queue, experiment_status, promotion_status, approval_status, learning_status)

        return {
            "created_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": operating_state["status"],
            "operator_summary": operating_state,
            "blockers": blockers,
            "approval_queue": approval_queue,
            "learning_status": learning_status,
            "experiment_status": experiment_status,
            "promotion_status": promotion_status,
            "approval_status": approval_status,
            "next_actions": next_actions,
            "sources": {
                "daily_job": self.daily_job_path,
                "advisor": self.advisor_path,
                "registry": self.registry_path,
                "experiment_plan": self.experiment_path,
                "shadow_results": self.shadow_results_path,
                "promotion_gate": self.promotion_gate_path,
                "approval_inbox": self.approval_inbox_path,
                "ml_edge_model": self.ml_edge_path,
                "alerts": self.alerts_path,
                "market_regime": self.market_regime_path,
            },
        }

    def _blockers(self, reports: dict) -> list[dict]:
        blockers = []
        for name, report in reports.items():
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

        return {
            "ml_edge": {
                "status": readiness.get("status") or model.get("status") or "unknown",
                "rows": _safe_int(readiness.get("rows")),
                "positive": _safe_int(readiness.get("positive")),
                "non_positive": _safe_int(readiness.get("non_positive")),
                "model_status": model.get("status"),
                "reason": readiness.get("reason"),
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
            "review_for_rejection": _safe_int(summary.get("review_for_rejection")),
            "wait": _safe_int(summary.get("wait")),
            "no_action_keep_protection": _safe_int(summary.get("no_action_keep_protection")),
            "by_action": summary.get("by_action", {}),
        }

    def _operating_state(self, reports: dict, blockers: list[dict], approval_queue: list[dict]) -> dict:
        advisor = reports.get("advisor", {}) or {}
        registry = reports.get("registry", {}) or {}
        experiments = reports.get("experiments", {}) or {}
        shadow = reports.get("shadow_results", {}) or {}
        promotion = reports.get("promotion_gate", {}) or {}
        approval = reports.get("approval_inbox", {}) or {}

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
        }

    def _next_actions(
        self,
        blockers: list[dict],
        approval_queue: list[dict],
        experiment_status: dict,
        promotion_status: dict,
        approval_status: dict,
        learning_status: dict,
    ) -> list[str]:
        if blockers:
            return [
                "Fix high-priority runtime/pipeline blockers first.",
                "Do not approve strategy experiments while health blockers are active.",
            ]

        actions = []
        if approval_status.get("review_for_approval"):
            actions.append("Approval inbox has experiments ready for explicit approve/reject review.")
        if approval_status.get("review_for_rejection"):
            actions.append("Approval inbox has experiments that should likely be rejected or left blocked.")
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
            f"reject={approval.get('review_for_rejection', 0)} "
            f"wait={approval.get('wait', 0)}"
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
