"""Resolve operator decisions against recommendation items.

The resolver is the bridge between append-only operator intent and the
recommendation queue. It does not change live trading behavior. It only marks
recommendations as active, accepted, waiting, rejected, frozen, or pending a
future live gate.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

DEFAULT_DECISIONS_PATH = os.path.join("analysis", "operator_decisions", "latest_operator_decisions.json")
DEFAULT_OUTPUT_DIR = os.path.join("analysis", "operator_decisions")
DEFAULT_LATEST_FILE = "latest_operator_decision_resolver.json"

STATUS_APPROVED_CONTEXT = "approved_context_live"
STATUS_APPROVED_SHADOW = "approved_shadow"
STATUS_APPROVED_PENDING_LIVE_GATE = "approved_pending_live_gate"
STATUS_WAIT = "wait_more_evidence"
STATUS_REJECTED = "rejected_by_operator"
STATUS_FROZEN = "frozen_by_operator"
STATUS_SNOOZED = "snoozed_by_operator"
STATUS_NOTED = "noted_by_operator"

HIDDEN_STATUSES = {
    STATUS_APPROVED_CONTEXT,
    STATUS_APPROVED_SHADOW,
    STATUS_REJECTED,
    STATUS_FROZEN,
    STATUS_SNOOZED,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: str, default: Any) -> Any:
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _decision_key(source_type: str, source_id: str) -> str:
    return f"{source_type}:{source_id}"


def _parse_utc(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _is_expired(decision: dict, now: Optional[datetime] = None) -> bool:
    expires = _parse_utc(str(decision.get("expires_utc") or ""))
    if not expires:
        return False
    now = now or datetime.now(timezone.utc)
    return expires <= now


class OperatorDecisionResolver:
    def __init__(self, decisions_path: str = DEFAULT_DECISIONS_PATH):
        self.decisions_path = decisions_path
        self.decisions_report = _load_json(decisions_path, {})
        self.latest_by_source = self._latest_decisions()

    def _latest_decisions(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for item in self.decisions_report.get("latest_by_source", []) or []:
            if _is_expired(item):
                continue
            key = _decision_key(str(item.get("source_type") or "recommendation"), str(item.get("source_id") or ""))
            result[key] = item
        return result

    def resolve_items(self, items: list[dict]) -> dict:
        resolved: list[dict] = []
        active: list[dict] = []
        suppressed: list[dict] = []
        for item in items:
            resolved_item = self.resolve_item(dict(item))
            resolved.append(resolved_item)
            if resolved_item.get("hidden_from_review"):
                suppressed.append(resolved_item)
            else:
                active.append(resolved_item)
        return {
            "created_utc": _utc_now(),
            "status": "OK",
            "active_items": active,
            "suppressed_items": suppressed,
            "resolved_items": resolved,
            "summary": self.summary(resolved, active, suppressed),
            "decisions_path": self.decisions_path,
            "live_effect": False,
        }

    def resolve_item(self, item: dict) -> dict:
        key = _decision_key("recommendation", str(item.get("id") or ""))
        decision = self.latest_by_source.get(key)
        if not decision:
            item["operator_resolution"] = {
                "status": "open",
                "action": "",
                "live_effect": False,
            }
            return item

        action = str(decision.get("action") or "").lower()
        effect_level = str(item.get("effect_level") or "shadow_only")
        resolution_status = self._resolution_status(action, effect_level)
        item["operator_resolution"] = {
            "status": resolution_status,
            "action": action,
            "decision_id": decision.get("decision_id"),
            "created_utc": decision.get("created_utc"),
            "reason": decision.get("reason"),
            "effect_level": effect_level,
            "live_effect": False,
        }
        item["operator_decision"] = {
            "decision_id": decision.get("decision_id"),
            "action": action,
            "reason": decision.get("reason"),
            "created_utc": decision.get("created_utc"),
        }
        if resolution_status in HIDDEN_STATUSES:
            item["hidden_from_review"] = True
        if resolution_status in {
            STATUS_APPROVED_CONTEXT,
            STATUS_APPROVED_SHADOW,
            STATUS_APPROVED_PENDING_LIVE_GATE,
            STATUS_WAIT,
        }:
            item["status"] = resolution_status
        return item

    def _resolution_status(self, action: str, effect_level: str) -> str:
        if action == "reject":
            return STATUS_REJECTED
        if action == "freeze":
            return STATUS_FROZEN
        if action == "snooze":
            return STATUS_SNOOZED
        if action == "wait":
            return STATUS_WAIT
        if action == "note":
            return STATUS_NOTED
        if action == "approve":
            if effect_level == "context_live":
                return STATUS_APPROVED_CONTEXT
            if effect_level == "shadow_only":
                return STATUS_APPROVED_SHADOW
            if effect_level in {"risk_down_live", "strategy_live"}:
                return STATUS_APPROVED_PENDING_LIVE_GATE
            return STATUS_APPROVED_SHADOW
        return "open"

    def summary(self, resolved: list[dict], active: list[dict], suppressed: list[dict]) -> dict:
        by_resolution = Counter(
            (item.get("operator_resolution") or {}).get("status", "open")
            for item in resolved
        )
        by_effect = Counter(item.get("effect_level") or "unknown" for item in resolved)
        return {
            "total": len(resolved),
            "active": len(active),
            "suppressed": len(suppressed),
            "by_resolution": dict(sorted(by_resolution.items())),
            "by_effect_level": dict(sorted(by_effect.items())),
            "pending_live_gate": by_resolution.get(STATUS_APPROVED_PENDING_LIVE_GATE, 0),
            "approved_context_live": by_resolution.get(STATUS_APPROVED_CONTEXT, 0),
            "approved_shadow": by_resolution.get(STATUS_APPROVED_SHADOW, 0),
            "rejected": by_resolution.get(STATUS_REJECTED, 0),
            "frozen": by_resolution.get(STATUS_FROZEN, 0),
            "live_effect": False,
        }


def run_operator_decision_resolver(
    items: Optional[list[dict]] = None,
    decisions_path: str = DEFAULT_DECISIONS_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    report = OperatorDecisionResolver(decisions_path=decisions_path).resolve_items(items or [])
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    _write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve operator decisions against an optional recommendation report.")
    parser.add_argument("--recommendations-path", type=str, default=os.path.join("analysis", "recommendations", "latest_recommendation_aggregator.json"))
    parser.add_argument("--decisions-path", type=str, default=DEFAULT_DECISIONS_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)
    recommendations = _load_json(args.recommendations_path, {})
    items = recommendations.get("items", []) or []
    report = run_operator_decision_resolver(items=items, decisions_path=args.decisions_path, output_dir=args.output_dir)
    print(json.dumps({
        "status": report.get("status"),
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
