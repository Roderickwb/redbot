# ============================================================
# src/analysis/operator_cockpit.py
# ============================================================
"""
Operator cockpit.

One compact human-facing view on top of the analysis stack. This module is the
daily surface for the operator: health, live-change status, urgent actions, and
the few learning/risk signals that matter. It is read-only.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Iterable, Optional

from dotenv import load_dotenv

from src.notifier.telegram_notifier import TelegramNotifier


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "operator_cockpit")
DEFAULT_LATEST_FILE = "latest_operator_cockpit.json"
DEFAULT_CONTROL_REPORT = os.path.join("analysis", "daily_control", "latest_daily_control_report.json")


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


class OperatorCockpit:
    def __init__(self, control_path: str = DEFAULT_CONTROL_REPORT):
        self.control_path = control_path

    def build(self) -> dict:
        control = load_json(self.control_path)
        if control.get("_missing") or control.get("_error"):
            return self._missing_control(control)

        blockers = control.get("blockers", []) or []
        approvals = control.get("approval_queue", []) or []
        learning = control.get("learning_status", {}) or {}
        experiments = control.get("experiment_status", {}) or {}
        promotion = control.get("promotion_status", {}) or {}
        approval_status = control.get("approval_status", {}) or {}
        risk_policy = control.get("risk_policy_status", {}) or {}
        risk_strategy = control.get("risk_strategy_status", {}) or {}
        risk_outcome = control.get("risk_outcome_status", {}) or {}
        risk_history = control.get("risk_history_status", {}) or {}
        shadow_live = control.get("shadow_live_status", {}) or {}

        live_changes = self._live_changes(control, shadow_live, risk_policy, risk_strategy, risk_outcome, risk_history)
        health = self._health(blockers)
        action_needed = self._action_needed(blockers, approvals, approval_status, promotion, risk_history)
        status = self._status(health, action_needed, control.get("status"))
        learning_summary = self._learning_summary(learning, experiments, promotion, approval_status)
        risk_summary = self._risk_summary(risk_policy, risk_strategy, risk_outcome, risk_history)
        daily_decision = self._daily_decision(live_changes, health, action_needed, learning_summary, risk_summary)

        return {
            "created_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "daily_decision": daily_decision,
            "live_changes": live_changes,
            "bot_health": health,
            "action_needed": action_needed,
            "blockers": blockers[:5],
            "operator_summary": {
                "blockers": len(blockers),
                "approval_queue": len(approvals),
                "urgent": action_needed.get("urgent", False),
                "live_enforcement": live_changes.get("live_enforcement", False),
            },
            "learning": learning_summary,
            "risk": risk_summary,
            "next_actions": self._next_actions(control, daily_decision),
            "sources": {
                "daily_control": self.control_path,
            },
        }

    def _missing_control(self, control: dict) -> dict:
        return {
            "created_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "ACTION_NEEDED",
            "daily_decision": {
                "label": "TODAY: STOP / FIX",
                "severity": "stop",
                "reason": "Daily control report is missing or unreadable.",
            },
            "live_changes": {"status": "UNKNOWN", "live_enforcement": False},
            "bot_health": {
                "status": "BROKEN",
                "blockers": 1,
                "finding": "Daily control report is missing or unreadable.",
            },
            "action_needed": {
                "status": "YES",
                "urgent": True,
                "headline": "Run daily_analysis_job or daily_control_report first.",
            },
            "blockers": [{
                "level": "high",
                "area": "pipeline",
                "finding": control.get("_error") or f"Missing {control.get('_path')}",
            }],
            "operator_summary": {"blockers": 1, "approval_queue": 0, "urgent": True, "live_enforcement": False},
            "learning": {},
            "risk": {},
            "next_actions": ["Run python -m src.analysis.daily_analysis_job --dry-run-labels --cleanup-registry."],
            "sources": {"daily_control": self.control_path},
        }

    def _live_changes(self, control: dict, shadow_live: dict, risk_policy: dict, risk_strategy: dict, risk_outcome: dict, risk_history: dict) -> dict:
        live_enforcement = any([
            bool(risk_policy.get("live_enforcement")),
            bool(risk_strategy.get("live_enforcement")),
            bool(risk_outcome.get("live_enforcement")),
            bool(risk_history.get("live_enforcement")),
        ])
        active_shadow = _safe_int(shadow_live.get("active_shadow_policies"))
        if live_enforcement:
            status = "ACTIVE"
            headline = "Live enforcement is enabled somewhere in the risk/adaptation layer."
        else:
            status = "NONE"
            headline = "No automatic live strategy or risk changes are enabled."
        return {
            "status": status,
            "headline": headline,
            "live_enforcement": live_enforcement,
            "active_shadow_policies": active_shadow,
            "control_status": control.get("status"),
        }

    def _health(self, blockers: list[dict]) -> dict:
        if blockers:
            return {
                "status": "ACTION_NEEDED",
                "blockers": len(blockers),
                "finding": blockers[0].get("finding"),
            }
        return {
            "status": "OK",
            "blockers": 0,
            "finding": "No runtime or pipeline blockers.",
        }

    def _action_needed(self, blockers: list[dict], approvals: list[dict], approval_status: dict, promotion: dict, risk_history: dict) -> dict:
        if blockers:
            return {
                "status": "YES",
                "urgent": True,
                "headline": "Fix pipeline/runtime blockers before strategy work.",
            }
        if _safe_int(promotion.get("ready_for_human_review")):
            return {
                "status": "YES",
                "urgent": False,
                "headline": "Promotion gate has candidates ready for human review.",
            }
        if _safe_int(approval_status.get("review_for_approval")):
            return {
                "status": "YES",
                "urgent": False,
                "headline": "Approval inbox has candidates for explicit approval/rejection.",
            }
        if risk_history.get("verdict") in {"stable_risk_down_helpful", "risk_down_too_strict"}:
            return {
                "status": "REVIEW",
                "urgent": False,
                "headline": f"Risk bridge history verdict is {risk_history.get('verdict')}.",
            }
        if approvals:
            return {
                "status": "OPTIONAL_REVIEW",
                "urgent": False,
                "headline": "There are non-urgent recommendations to review.",
            }
        return {
            "status": "NO",
            "urgent": False,
            "headline": "No operator action needed today.",
        }

    def _status(self, health: dict, action: dict, fallback: Optional[str]) -> str:
        if health.get("status") != "OK":
            return "ACTION_NEEDED"
        if action.get("urgent"):
            return "ACTION_NEEDED"
        if action.get("status") in {"YES", "REVIEW"}:
            return "REVIEW"
        if action.get("status") == "OPTIONAL_REVIEW":
            return "WATCH"
        return fallback or "WATCH"

    def _learning_summary(self, learning: dict, experiments: dict, promotion: dict, approval: dict) -> dict:
        ml = learning.get("ml_edge", {}) or {}
        hypotheses = learning.get("hypotheses", {}) or {}
        market = learning.get("market_regime", {}) or {}
        return {
            "ml_status": ml.get("status"),
            "ml_rows": _safe_int(ml.get("rows")),
            "ml_positive": _safe_int(ml.get("positive")),
            "ml_non_positive": _safe_int(ml.get("non_positive")),
            "hypotheses_promotable": _safe_int(hypotheses.get("promotable")),
            "experiments_planned": _safe_int(experiments.get("planned")),
            "experiments_ready": _safe_int(experiments.get("ready_for_approval")),
            "promotion_blocked": _safe_int(promotion.get("blocked")),
            "promotion_ready": _safe_int(promotion.get("ready_for_human_review")),
            "approval_reject_candidates": _safe_int(approval.get("reject_candidate")),
            "market_regime": market.get("regime"),
            "market_bias": market.get("directional_bias"),
        }

    def _risk_summary(self, risk_policy: dict, risk_strategy: dict, risk_outcome: dict, risk_history: dict) -> dict:
        return {
            "policy_symbols": _safe_int(risk_policy.get("total_symbols")),
            "risk_down": _safe_int(risk_policy.get("risk_down")),
            "cap_new_longs": _safe_int(risk_policy.get("cap_new_longs")),
            "avg_long_multiplier": _safe_float(risk_policy.get("average_long_risk_multiplier"), 1.0),
            "avg_short_multiplier": _safe_float(risk_policy.get("average_short_risk_multiplier"), 1.0),
            "recent_open_trades": _safe_int(risk_strategy.get("opened_trades")),
            "recent_adjusted_opens": _safe_int(risk_strategy.get("would_adjust_open_trades")),
            "latest_labeled_adjusted": _safe_int(risk_outcome.get("adjusted_with_labeled_outcomes")),
            "latest_net_saved_r": _safe_float(risk_outcome.get("estimated_net_saved_r")),
            "history_unique": _safe_int(risk_history.get("unique_adjusted_labeled_events")),
            "history_days": _safe_int(risk_history.get("days_observed")),
            "history_net_saved_r": _safe_float(risk_history.get("estimated_net_saved_r")),
            "history_verdict": risk_history.get("verdict"),
            "live_enforcement": bool(risk_history.get("live_enforcement")),
        }

    def _daily_decision(self, live_changes: dict, health: dict, action_needed: dict, learning: dict, risk: dict) -> dict:
        if health.get("status") != "OK":
            return {
                "label": "TODAY: STOP / FIX",
                "severity": "stop",
                "reason": health.get("finding") or "Bot health is not OK.",
            }
        if live_changes.get("live_enforcement"):
            return {
                "label": "TODAY: STOP / REVIEW LIVE CHANGES",
                "severity": "stop",
                "reason": "Live enforcement is enabled; verify this was intentional.",
            }
        if action_needed.get("status") == "YES":
            return {
                "label": "TODAY: REVIEW REQUIRED",
                "severity": "review",
                "reason": action_needed.get("headline"),
            }
        if risk.get("history_verdict") in {"stable_risk_down_helpful", "risk_down_too_strict"}:
            return {
                "label": "TODAY: REVIEW REQUIRED",
                "severity": "review",
                "reason": f"Risk bridge history verdict is {risk.get('history_verdict')}.",
            }
        return {
            "label": "TODAY: NO ACTION REQUIRED",
            "severity": "watch",
            "reason": "No blockers, no live changes, and no approval-ready item.",
        }

    def _next_actions(self, control: dict, daily_decision: dict) -> list[str]:
        actions = control.get("next_actions", []) or []
        if daily_decision.get("severity") == "watch":
            return ["No action required today. Let the bot keep collecting evidence."]
        return actions[:6]


def format_cockpit_message(cockpit: dict) -> str:
    learning = cockpit.get("learning", {}) or {}
    risk = cockpit.get("risk", {}) or {}
    live = cockpit.get("live_changes", {}) or {}
    health = cockpit.get("bot_health", {}) or {}
    action = cockpit.get("action_needed", {}) or {}
    decision = cockpit.get("daily_decision", {}) or {}

    lines = [
        f"RED BOT COCKPIT [{cockpit.get('status')}]",
        str(decision.get("label") or "TODAY: UNKNOWN"),
        f"Reason: {decision.get('reason')}",
        f"Live changes: {live.get('status', 'UNKNOWN')}",
        f"Bot health: {health.get('status', 'UNKNOWN')} (blockers={health.get('blockers', 0)})",
        f"Action needed: {action.get('status', 'UNKNOWN')} - {action.get('headline')}",
        "",
        "Learning:",
        (
            f"- ML: {learning.get('ml_status')} "
            f"rows={learning.get('ml_rows', 0)} "
            f"pos={learning.get('ml_positive', 0)} "
            f"non_pos={learning.get('ml_non_positive', 0)}"
        ),
        (
            f"- Experiments: planned={learning.get('experiments_planned', 0)} "
            f"ready={learning.get('experiments_ready', 0)} "
            f"blocked={learning.get('promotion_blocked', 0)}"
        ),
        (
            f"- Market: {learning.get('market_regime')} "
            f"bias={learning.get('market_bias')}"
        ),
        "",
        "Risk:",
        (
            f"- Policy: symbols={risk.get('policy_symbols', 0)} "
            f"risk_down={risk.get('risk_down', 0)} "
            f"cap_longs={risk.get('cap_new_longs', 0)}"
        ),
        (
            f"- Recent bridge: opens={risk.get('recent_open_trades', 0)} "
            f"adjusted={risk.get('recent_adjusted_opens', 0)} "
            f"latest_net_R={risk.get('latest_net_saved_r', 0.0)}"
        ),
        (
            f"- History: unique={risk.get('history_unique', 0)} "
            f"days={risk.get('history_days', 0)} "
            f"net_R={risk.get('history_net_saved_r', 0.0)} "
            f"verdict={risk.get('history_verdict')}"
        ),
    ]

    next_actions = cockpit.get("next_actions", []) or []
    if next_actions:
        lines.extend(["", "Next:"])
        for item in next_actions[:5]:
            lines.append(f"- {item}")
    lines.extend(["", str(decision.get("label") or "TODAY: UNKNOWN")])
    return "\n".join(line for line in lines if line is not None)


def send_telegram(cockpit: dict) -> bool:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    TelegramNotifier(token, chat_id).safe_send(format_cockpit_message(cockpit))
    return True


def run_operator_cockpit(
    control_path: str = DEFAULT_CONTROL_REPORT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    send: bool = False,
) -> dict:
    cockpit = OperatorCockpit(control_path=control_path).build()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    cockpit["output_path"] = output_path
    write_json(output_path, cockpit)
    cockpit["telegram_sent"] = send_telegram(cockpit) if send else False
    return cockpit


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the compact Red Bot operator cockpit.")
    parser.add_argument("--control-report", type=str, default=DEFAULT_CONTROL_REPORT, help="Daily control report JSON.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--send", action="store_true", help="Send cockpit digest to Telegram.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the human cockpit text.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cockpit = run_operator_cockpit(
        control_path=args.control_report,
        output_dir=args.output_dir,
        send=args.send,
    )
    if args.json:
        print(json.dumps({
            "status": cockpit.get("status"),
            "daily_decision": cockpit.get("daily_decision"),
            "live_changes": cockpit.get("live_changes"),
            "bot_health": cockpit.get("bot_health"),
            "action_needed": cockpit.get("action_needed"),
            "learning": cockpit.get("learning"),
            "risk": cockpit.get("risk"),
            "next_actions": cockpit.get("next_actions"),
            "output_path": cockpit.get("output_path"),
            "telegram_sent": cockpit.get("telegram_sent"),
        }, indent=2, ensure_ascii=False))
    else:
        print(format_cockpit_message(cockpit))
        print(f"\noutput_path: {cockpit.get('output_path')}")
        print(f"telegram_sent: {cockpit.get('telegram_sent', False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
