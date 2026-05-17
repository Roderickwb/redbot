# ============================================================
# src/analysis/operator_decisions.py
# ============================================================
"""Append-only operator decision store.

This module is intentionally read-only for live trading behavior. It records
operator intent for the future cockpit/app layer, but it does not approve,
modify, enable, or disable strategy/risk/live enforcement by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "operator_decisions")
DEFAULT_HISTORY_FILE = "operator_decisions.jsonl"
DEFAULT_LATEST_FILE = "latest_operator_decisions.json"

ALLOWED_ACTIONS = {"approve", "reject", "wait", "freeze", "snooze", "note"}
LIVE_BLOCKED_ACTIONS = {
    "enable_live_enforcement",
    "risk_up_live",
    "entry_rule_live",
    "ml_live",
    "clear_kill_switch",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: str, default: Any) -> Any:
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e), "_path": path}


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
            except Exception:
                continue
    return rows


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _append_jsonl(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _source_snapshot(source_path: str) -> tuple[dict, str]:
    snapshot = _load_json(source_path, {}) if source_path else {}
    return snapshot, _stable_hash(snapshot) if snapshot else ""


class OperatorDecisionStore:
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir
        self.history_path = os.path.join(output_dir, DEFAULT_HISTORY_FILE)
        self.latest_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)

    def record(
        self,
        *,
        source_id: str,
        source_type: str,
        action: str,
        operator: str = "human",
        reason: str = "",
        scope: str = "recommendation",
        source_path: str = "",
        expires_utc: str = "",
    ) -> dict:
        action = str(action).strip().lower()
        if action in LIVE_BLOCKED_ACTIONS:
            raise ValueError(f"Action is explicitly forbidden in v1: {action}")
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported action: {action}. Allowed: {sorted(ALLOWED_ACTIONS)}")

        source_snapshot, evidence_hash = _source_snapshot(source_path)
        created_utc = _utc_now()
        base = {
            "source_id": source_id,
            "source_type": source_type,
            "action": action,
            "scope": scope,
            "operator": operator,
            "reason": reason,
            "created_utc": created_utc,
            "expires_utc": expires_utc,
            "evidence_hash": evidence_hash,
        }
        decision_id = f"opdec_{_stable_hash({**base, 'ts': time.time_ns()})}"
        item = {
            "decision_id": decision_id,
            **base,
            "status": "recorded",
            "live_effect": False,
            "live_allowed": False,
            "source_path": source_path,
            "source_snapshot": source_snapshot,
        }
        _append_jsonl(self.history_path, item)
        self.write_report()
        return item

    def write_report(self) -> dict:
        decisions = _load_jsonl(self.history_path)
        by_action = Counter(str(d.get("action", "unknown")) for d in decisions)
        by_source_type = Counter(str(d.get("source_type", "unknown")) for d in decisions)
        last_by_source: dict[str, dict] = {}
        for item in decisions:
            key = f"{item.get('source_type')}:{item.get('source_id')}"
            last_by_source[key] = item

        report = {
            "status": "OK",
            "summary": {
                "total": len(decisions),
                "by_action": dict(sorted(by_action.items())),
                "by_source_type": dict(sorted(by_source_type.items())),
                "active_sources": len(last_by_source),
                "live_effect": False,
            },
            "recent": decisions[-25:][::-1],
            "latest_by_source": list(last_by_source.values())[-100:][::-1],
            "history_path": self.history_path,
            "output_path": self.latest_path,
        }
        _write_json(self.latest_path, report)
        return report


def run_operator_decisions(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    return OperatorDecisionStore(output_dir=output_dir).write_report()


def record_operator_decision(
    *,
    source_id: str,
    source_type: str,
    action: str,
    operator: str = "human",
    reason: str = "",
    scope: str = "recommendation",
    source_path: str = "",
    expires_utc: str = "",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    return OperatorDecisionStore(output_dir=output_dir).record(
        source_id=source_id,
        source_type=source_type,
        action=action,
        operator=operator,
        reason=reason,
        scope=scope,
        source_path=source_path,
        expires_utc=expires_utc,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record or summarize append-only operator decisions.")
    parser.add_argument("command", nargs="?", choices=["report", "record"], default="report")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-id", type=str, default="")
    parser.add_argument("--source-type", type=str, default="recommendation")
    parser.add_argument("--action", type=str, default="note")
    parser.add_argument("--operator", type=str, default="human")
    parser.add_argument("--reason", type=str, default="")
    parser.add_argument("--scope", type=str, default="recommendation")
    parser.add_argument("--source-path", type=str, default="")
    parser.add_argument("--expires-utc", type=str, default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.command == "record":
            if not args.source_id:
                raise ValueError("--source-id is required for record")
            payload = record_operator_decision(
                source_id=args.source_id,
                source_type=args.source_type,
                action=args.action,
                operator=args.operator,
                reason=args.reason,
                scope=args.scope,
                source_path=args.source_path,
                expires_utc=args.expires_utc,
                output_dir=args.output_dir,
            )
        else:
            payload = run_operator_decisions(output_dir=args.output_dir)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"status": "ERROR", "error": str(e)}, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
