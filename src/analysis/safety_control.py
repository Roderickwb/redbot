# ============================================================
# src/analysis/safety_control.py
# ============================================================
"""
Live safety state, audit log and kill-switch controls.

The safety state is intentionally file-based so it can be inspected and changed
without a database write path. Trading modules may use this as a hard guard
before live entry orders. Analysis modules use it for cockpit visibility.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "safety")
DEFAULT_STATE_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "live_safety_state.json")
DEFAULT_MELTDOWN_STATE_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "latest_meltdown_state.json")
DEFAULT_AUDIT_LOG = os.path.join("analysis", "audit", "live_change_audit.jsonl")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ms_now() -> int:
    return int(time.time() * 1000)


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e)}


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def append_audit_event(
    event_type: str,
    actor: str = "system",
    reason: str = "",
    payload: Optional[dict] = None,
    audit_path: str = DEFAULT_AUDIT_LOG,
) -> dict:
    event = {
        "created_ts": _ms_now(),
        "created_utc": _utc_now(),
        "event_type": event_type,
        "actor": actor,
        "reason": reason,
        "host": socket.gethostname(),
        "payload": payload or {},
    }
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def write_meltdown_state(
    active: bool,
    reason: Optional[str] = None,
    source: str = "meltdown_manager",
    output_path: str = DEFAULT_MELTDOWN_STATE_FILE,
) -> dict:
    payload = {
        "updated_ts": _ms_now(),
        "updated_utc": _utc_now(),
        "active": bool(active),
        "reason": reason,
        "source": source,
    }
    _write_json(output_path, payload)
    return payload


class SafetyControl:
    def __init__(self, state_path: str = DEFAULT_STATE_FILE, audit_path: str = DEFAULT_AUDIT_LOG):
        self.state_path = state_path
        self.audit_path = audit_path

    def default_state(self) -> dict:
        return {
            "schema_version": 1,
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "kill_switch_active": False,
            "live_enforcement_allowed": False,
            "live_entry_orders_allowed": True,
            "live_risk_wiring_allowed": False,
            "live_strategy_wiring_allowed": False,
            "reason": "default safe state",
            "actor": "system",
            "last_event": None,
        }

    def load_state(self) -> dict:
        state = self.default_state()
        saved = _read_json(self.state_path)
        if saved.get("_error"):
            state["state_error"] = saved.get("_error")
            state["kill_switch_active"] = True
            state["live_entry_orders_allowed"] = False
            state["reason"] = "safety state unreadable; fail closed for live entries"
            return state
        state.update(saved)
        return state

    def save_state(self, state: dict, event_type: str, actor: str, reason: str) -> dict:
        state["updated_utc"] = _utc_now()
        state["actor"] = actor
        state["reason"] = reason
        event = append_audit_event(
            event_type=event_type,
            actor=actor,
            reason=reason,
            payload={
                "state_path": self.state_path,
                "kill_switch_active": state.get("kill_switch_active"),
                "live_entry_orders_allowed": state.get("live_entry_orders_allowed"),
                "live_enforcement_allowed": state.get("live_enforcement_allowed"),
            },
            audit_path=self.audit_path,
        )
        state["last_event"] = event
        _write_json(self.state_path, state)
        return state

    def status(self) -> dict:
        state = self.load_state()
        audit_summary = self.audit_summary()
        meltdown = _read_json(DEFAULT_MELTDOWN_STATE_FILE)
        if meltdown.get("_error"):
            meltdown = {"active": False, "error": meltdown.get("_error")}
        live_blocked = bool(state.get("kill_switch_active")) or not bool(state.get("live_entry_orders_allowed", True))
        return {
            "status": "KILL_SWITCH_ACTIVE" if live_blocked else "OK",
            "kill_switch_active": bool(state.get("kill_switch_active")),
            "live_entry_orders_allowed": bool(state.get("live_entry_orders_allowed", True)),
            "live_enforcement_allowed": bool(state.get("live_enforcement_allowed")),
            "live_risk_wiring_allowed": bool(state.get("live_risk_wiring_allowed")),
            "live_strategy_wiring_allowed": bool(state.get("live_strategy_wiring_allowed")),
            "reason": state.get("reason"),
            "updated_utc": state.get("updated_utc"),
            "state_path": self.state_path,
            "audit_path": self.audit_path,
            "audit": audit_summary,
            "meltdown": {
                "active": bool(meltdown.get("active")),
                "reason": meltdown.get("reason"),
                "updated_utc": meltdown.get("updated_utc"),
                "source": meltdown.get("source"),
                "error": meltdown.get("error"),
            },
        }

    def activate_kill_switch(self, reason: str, actor: str = "operator") -> dict:
        state = self.load_state()
        state["kill_switch_active"] = True
        state["live_entry_orders_allowed"] = False
        return self.save_state(state, "kill_switch_activated", actor=actor, reason=reason)

    def clear_kill_switch(self, reason: str, actor: str = "operator") -> dict:
        state = self.load_state()
        state["kill_switch_active"] = False
        state["live_entry_orders_allowed"] = True
        return self.save_state(state, "kill_switch_cleared", actor=actor, reason=reason)

    def set_live_enforcement(self, allowed: bool, reason: str, actor: str = "operator") -> dict:
        state = self.load_state()
        state["live_enforcement_allowed"] = bool(allowed)
        event_type = "live_enforcement_allowed" if allowed else "live_enforcement_disabled"
        return self.save_state(state, event_type, actor=actor, reason=reason)

    def rollback_to_safe_state(self, reason: str, actor: str = "operator") -> dict:
        state = self.load_state()
        state["kill_switch_active"] = True
        state["live_entry_orders_allowed"] = False
        state["live_enforcement_allowed"] = False
        state["live_risk_wiring_allowed"] = False
        state["live_strategy_wiring_allowed"] = False
        return self.save_state(state, "rollback_to_safe_state", actor=actor, reason=reason)

    def live_entry_allowed(self) -> tuple[bool, str]:
        status = self.status()
        if status.get("kill_switch_active"):
            return False, f"kill_switch_active: {status.get('reason')}"
        if not status.get("live_entry_orders_allowed"):
            return False, f"live_entry_orders_not_allowed: {status.get('reason')}"
        return True, "live entry orders allowed"

    def audit_summary(self, limit: int = 20) -> dict:
        if not os.path.exists(self.audit_path):
            return {"events": 0, "recent": []}
        recent = []
        total = 0
        try:
            with open(self.audit_path, "r", encoding="utf-8") as f:
                for line in f:
                    total += 1
                    if len(recent) >= limit:
                        recent.pop(0)
                    try:
                        recent.append(json.loads(line))
                    except Exception:
                        pass
        except Exception as e:
            return {"events": total, "error": str(e), "recent": recent[-limit:]}
        return {"events": total, "recent": recent[-limit:]}


def run_safety_control_report(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    control = SafetyControl(state_path=os.path.join(output_dir, "live_safety_state.json"))
    report = control.status()
    output_path = os.path.join(output_dir, "latest_safety_control_report.json")
    report["output_path"] = output_path
    _write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or change Red Bot live safety state.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Print current safety status.")

    kill = sub.add_parser("kill", help="Activate live-entry kill switch.")
    kill.add_argument("--reason", required=True)
    kill.add_argument("--actor", default="operator")

    clear = sub.add_parser("clear", help="Clear live-entry kill switch.")
    clear.add_argument("--reason", required=True)
    clear.add_argument("--actor", default="operator")

    allow = sub.add_parser("allow-live-enforcement", help="Allow future live enforcement wiring.")
    allow.add_argument("--reason", required=True)
    allow.add_argument("--actor", default="operator")

    disable = sub.add_parser("disable-live-enforcement", help="Disable live enforcement wiring.")
    disable.add_argument("--reason", required=True)
    disable.add_argument("--actor", default="operator")

    rollback = sub.add_parser("rollback", help="Disable live enforcement and activate kill-switch.")
    rollback.add_argument("--reason", required=True)
    rollback.add_argument("--actor", default="operator")

    args = parser.parse_args(list(argv) if argv is not None else None)
    control = SafetyControl()

    if args.command == "kill":
        result = control.activate_kill_switch(reason=args.reason, actor=args.actor)
    elif args.command == "clear":
        result = control.clear_kill_switch(reason=args.reason, actor=args.actor)
    elif args.command == "allow-live-enforcement":
        result = control.set_live_enforcement(True, reason=args.reason, actor=args.actor)
    elif args.command == "disable-live-enforcement":
        result = control.set_live_enforcement(False, reason=args.reason, actor=args.actor)
    elif args.command == "rollback":
        result = control.rollback_to_safe_state(reason=args.reason, actor=args.actor)
    else:
        result = control.status()

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
