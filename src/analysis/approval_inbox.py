# ============================================================
# src/analysis/approval_inbox.py
# ============================================================
"""
Approval inbox.

Creates a compact human decision queue from the experiment planner and
promotion gate. This module does not approve, reject, or change trading rules.
It only shows which experiments can be acted on and provides the exact command
to use when a human decides.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Iterable, Optional


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "approvals")
DEFAULT_LATEST_FILE = "latest_approval_inbox.json"
DEFAULT_LIFECYCLE_FILE = "approval_lifecycle.json"
DEFAULT_EXPERIMENT_PLAN = os.path.join("analysis", "experiments", "latest_experiment_plan.json")
DEFAULT_PROMOTION_GATE = os.path.join("analysis", "promotion_gate", "latest_promotion_gate_report.json")
REJECT_CANDIDATE_BLOCKED_COUNT = 3


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


class ApprovalInbox:
    def __init__(
        self,
        experiment_plan_path: str = DEFAULT_EXPERIMENT_PLAN,
        promotion_gate_path: str = DEFAULT_PROMOTION_GATE,
        lifecycle_path: str = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_LIFECYCLE_FILE),
    ):
        self.experiment_plan_path = experiment_plan_path
        self.promotion_gate_path = promotion_gate_path
        self.lifecycle_path = lifecycle_path

    def build_report(self) -> dict:
        plan = _load_json(self.experiment_plan_path, {"experiments": [], "summary": {}})
        promotion = _load_json(self.promotion_gate_path, {"decisions": [], "summary": {}})
        lifecycle = _load_json(self.lifecycle_path, {"items": {}})
        if plan.get("_error") or promotion.get("_error"):
            items = []
            lifecycle_summary = {}
            errors = [
                report.get("_error")
                for report in (plan, promotion)
                if report.get("_error")
            ]
        else:
            errors = []
            items = self._items(
                experiments=plan.get("experiments", []) or [],
                decisions=promotion.get("decisions", []) or [],
                lifecycle=lifecycle,
            )
            lifecycle_summary = self._lifecycle_summary(lifecycle)
            self._write_lifecycle(lifecycle)

        return {
            "created_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "ERROR" if errors else "OK",
            "errors": errors,
            "summary": self._summary(items),
            "lifecycle_summary": lifecycle_summary,
            "items": items,
            "sources": {
                "experiment_plan": self.experiment_plan_path,
                "promotion_gate": self.promotion_gate_path,
                "lifecycle": self.lifecycle_path,
            },
        }

    def _items(self, experiments: list[dict], decisions: list[dict], lifecycle: dict) -> list[dict]:
        decision_by_id = {
            item.get("experiment_id"): item
            for item in decisions
            if item.get("experiment_id")
        }
        lifecycle_items = lifecycle.setdefault("items", {})
        now_ms = int(time.time() * 1000)
        now_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        seen_ids = set()
        items = []
        for experiment in experiments:
            exp_id = experiment.get("id")
            seen_ids.add(exp_id)
            promotion = decision_by_id.get(exp_id, {})
            evidence = experiment.get("evidence", {}) or {}
            decision = experiment.get("decision", {}) or {}
            state = self._update_lifecycle_state(
                lifecycle_items=lifecycle_items,
                experiment=experiment,
                promotion=promotion,
                now_ms=now_ms,
                now_local=now_local,
            )
            action = self._action_for(experiment, promotion, state)
            items.append({
                "experiment_id": exp_id,
                "action": action,
                "priority": self._priority_for(action, experiment, promotion),
                "experiment_type": experiment.get("experiment_type"),
                "pattern": evidence.get("pattern"),
                "experiment_status": experiment.get("status"),
                "promotion_status": promotion.get("status", "unknown"),
                "promotion_reason": promotion.get("reason"),
                "blocked_count": state.get("blocked_count", 0),
                "last_action": state.get("last_action"),
                "proposal": experiment.get("proposal"),
                "next_action": promotion.get("next_action") or experiment.get("next_action"),
                "metrics": {
                    "replay": promotion.get("replay", {}),
                    "forward": promotion.get("forward", {}),
                    "hypothesis": {
                        "seen_count": evidence.get("seen_count"),
                        "age_days": evidence.get("age_days"),
                        "matches": evidence.get("matches"),
                        "cf_avg_r": evidence.get("cf_avg_r"),
                        "cf_loss_rate_pct": evidence.get("cf_loss_rate_pct"),
                    },
                },
                "commands": {
                    "approve": self._command_with_note(
                        decision.get("approve_command"),
                        f"approved via approval inbox; promotion={promotion.get('status', 'unknown')}",
                    ),
                    "reject": self._command_with_note(
                        decision.get("reject_command"),
                        f"rejected via approval inbox; promotion={promotion.get('status', 'unknown')}",
                    ),
                },
            })

        self._mark_missing_lifecycle_items(lifecycle_items, seen_ids, now_ms=now_ms, now_local=now_local)
        items.sort(
            key=lambda item: (
                self._action_rank(item.get("action")),
                self._priority_rank(item.get("priority")),
                _safe_int(item.get("blocked_count")),
                _safe_int((item.get("metrics", {}).get("forward") or {}).get("matches")),
                abs(_safe_float((item.get("metrics", {}).get("forward") or {}).get("cf_avg_r"))),
            ),
            reverse=True,
        )
        return items

    def _action_for(self, experiment: dict, promotion: dict, state: dict) -> str:
        promotion_status = promotion.get("status")
        experiment_status = experiment.get("status")
        if promotion_status == "ready_for_human_review":
            return "review_for_approval"
        if promotion_status == "blocked" and _safe_int(state.get("blocked_count")) >= REJECT_CANDIDATE_BLOCKED_COUNT:
            return "reject_candidate"
        if promotion_status in {"blocked", "needs_review"}:
            return "review_for_rejection"
        if experiment_status == "approved_for_shadow":
            return "already_approved_for_shadow"
        if promotion_status == "confirmed_protection":
            return "no_action_keep_protection"
        return "wait"

    def _priority_for(self, action: str, experiment: dict, promotion: dict) -> str:
        if action == "review_for_approval":
            return "high"
        if action == "reject_candidate":
            return "high"
        if action == "review_for_rejection":
            return "medium"
        if action == "already_approved_for_shadow":
            return "medium"
        return experiment.get("priority") or "low"

    def _command_with_note(self, command: Optional[str], note: str) -> Optional[str]:
        if not command:
            return None
        escaped = note.replace('"', "'")
        return f'{command} --note "{escaped}"'

    def _summary(self, items: list[dict]) -> dict:
        by_action = {}
        by_promotion_status = {}
        for item in items:
            by_action[item.get("action", "unknown")] = by_action.get(item.get("action", "unknown"), 0) + 1
            by_promotion_status[item.get("promotion_status", "unknown")] = by_promotion_status.get(item.get("promotion_status", "unknown"), 0) + 1
        return {
            "total": len(items),
            "by_action": by_action,
            "by_promotion_status": by_promotion_status,
            "review_for_approval": by_action.get("review_for_approval", 0),
            "reject_candidate": by_action.get("reject_candidate", 0),
            "review_for_rejection": by_action.get("review_for_rejection", 0),
            "wait": by_action.get("wait", 0),
            "no_action_keep_protection": by_action.get("no_action_keep_protection", 0),
        }

    def _update_lifecycle_state(
        self,
        lifecycle_items: dict,
        experiment: dict,
        promotion: dict,
        now_ms: int,
        now_local: str,
    ) -> dict:
        exp_id = experiment.get("id")
        state = lifecycle_items.setdefault(exp_id, {
            "experiment_id": exp_id,
            "first_seen_ts": now_ms,
            "first_seen_local": now_local,
            "seen_count": 0,
            "blocked_count": 0,
            "missing_count": 0,
        })
        promotion_status = promotion.get("status", "unknown")
        state["seen_count"] = _safe_int(state.get("seen_count")) + 1
        state["last_seen_ts"] = now_ms
        state["last_seen_local"] = now_local
        state["missing_count"] = 0
        state["experiment_type"] = experiment.get("experiment_type")
        state["pattern"] = (experiment.get("evidence") or {}).get("pattern")
        state["experiment_status"] = experiment.get("status")
        state["promotion_status"] = promotion_status
        state["promotion_reason"] = promotion.get("reason")

        if promotion_status == "blocked":
            state["blocked_count"] = _safe_int(state.get("blocked_count")) + 1
        elif promotion_status == "ready_for_human_review":
            state["blocked_count"] = 0

        state["last_action"] = (
            "reject_candidate"
            if promotion_status == "blocked" and _safe_int(state.get("blocked_count")) >= REJECT_CANDIDATE_BLOCKED_COUNT
            else None
        )
        return state

    def _mark_missing_lifecycle_items(self, lifecycle_items: dict, seen_ids: set[str], now_ms: int, now_local: str) -> None:
        for exp_id, state in lifecycle_items.items():
            if exp_id in seen_ids:
                continue
            state["missing_count"] = _safe_int(state.get("missing_count")) + 1
            state["last_missing_ts"] = now_ms
            state["last_missing_local"] = now_local

    def _write_lifecycle(self, lifecycle: dict) -> None:
        lifecycle["meta"] = {
            "updated_ts": int(time.time() * 1000),
            "updated_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reject_candidate_blocked_count": REJECT_CANDIDATE_BLOCKED_COUNT,
        }
        write_json(self.lifecycle_path, lifecycle)

    def _lifecycle_summary(self, lifecycle: dict) -> dict:
        items = list((lifecycle.get("items") or {}).values())
        return {
            "tracked": len(items),
            "reject_candidate_blocked_count": REJECT_CANDIDATE_BLOCKED_COUNT,
            "blocked_ge_threshold": sum(
                1 for item in items
                if _safe_int(item.get("blocked_count")) >= REJECT_CANDIDATE_BLOCKED_COUNT
            ),
        }

    def _action_rank(self, action: Any) -> int:
        return {
            "review_for_approval": 5,
            "reject_candidate": 4,
            "review_for_rejection": 3,
            "already_approved_for_shadow": 3,
            "wait": 2,
            "no_action_keep_protection": 1,
        }.get(str(action), 0)

    def _priority_rank(self, priority: Any) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(str(priority), 0)


def run_approval_inbox(
    experiment_plan_path: str = DEFAULT_EXPERIMENT_PLAN,
    promotion_gate_path: str = DEFAULT_PROMOTION_GATE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    report = ApprovalInbox(
        experiment_plan_path=experiment_plan_path,
        promotion_gate_path=promotion_gate_path,
        lifecycle_path=os.path.join(output_dir, DEFAULT_LIFECYCLE_FILE),
    ).build_report()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build approval inbox from experiment and promotion reports.")
    parser.add_argument("--experiment-plan", type=str, default=DEFAULT_EXPERIMENT_PLAN, help="Experiment plan JSON.")
    parser.add_argument("--promotion-gate", type=str, default=DEFAULT_PROMOTION_GATE, help="Promotion gate JSON.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_approval_inbox(
        experiment_plan_path=args.experiment_plan,
        promotion_gate_path=args.promotion_gate,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "status": report.get("status"),
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
