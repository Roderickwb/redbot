# ============================================================
# src/analysis/risk_strategy_bridge.py
# ============================================================
"""
Risk-to-strategy bridge in shadow mode.

Applies the read-only risk policy to recent GPT decisions and records what
would have happened to trade sizing. This module does not block trades, change
orders, or write strategy settings.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "risk")
DEFAULT_LATEST_FILE = "latest_risk_strategy_bridge_report.json"
DEFAULT_RISK_POLICY = os.path.join("analysis", "risk", "latest_risk_policy_report.json")


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


def _parse_json(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _direction_for(algo_signal: Any, trend_dir: Any, gpt_action: Any = None) -> str:
    action = str(gpt_action or "")
    if action == "OPEN_LONG":
        return "long"
    if action == "OPEN_SHORT":
        return "short"
    signal = str(algo_signal or "")
    if signal == "long_candidate":
        return "long"
    if signal == "short_candidate":
        return "short"
    trend = str(trend_dir or "")
    if trend == "bull":
        return "long"
    if trend == "bear":
        return "short"
    return "unknown"


class RiskStrategyBridge:
    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        risk_policy_path: str = DEFAULT_RISK_POLICY,
    ):
        self.db = db or DatabaseManager(db_path=DB_FILE)
        self.risk_policy_path = risk_policy_path

    def build_report(self, hours: int = 24, limit: int = 1000) -> dict:
        risk_policy = _load_json(self.risk_policy_path, {"policies": []})
        policies = {
            item.get("symbol"): item
            for item in (risk_policy.get("policies") or [])
            if item.get("symbol")
        }
        events = self._load_recent_events(hours=hours, limit=limit)
        decisions = [
            self._shadow_decision(event, policies.get(event.get("symbol"), {}))
            for event in events
        ]

        return {
            "meta": {
                "created_utc": _utc_now(),
                "hours": hours,
                "limit": limit,
                "risk_policy_path": self.risk_policy_path,
                "loaded_events": len(events),
                "loaded_policies": len(policies),
                "read_only": True,
                "live_enforcement": False,
            },
            "summary": self._summary(decisions),
            "decisions": decisions[:200],
        }

    def _load_recent_events(self, hours: int, limit: int) -> list[dict]:
        cutoff_ms = int(time.time() * 1000) - int(hours) * 3600 * 1000
        rows = self.db.execute_query(
            """
            SELECT
              id, timestamp, symbol, strategy_name, strategy_version,
              event_type, trend_dir, algo_signal, gpt_action, gpt_confidence,
              features_json
            FROM strategy_events
            WHERE event_type='gpt_decision'
              AND timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (cutoff_ms, int(limit)),
        )
        cols = [
            "id", "timestamp", "symbol", "strategy_name", "strategy_version",
            "event_type", "trend_dir", "algo_signal", "gpt_action", "gpt_confidence",
            "features_json",
        ]
        return [dict(zip(cols, row)) for row in rows] if rows else []

    def _shadow_decision(self, event: dict, policy: dict) -> dict:
        features = _parse_json(event.get("features_json"))
        direction = _direction_for(event.get("algo_signal"), event.get("trend_dir"), event.get("gpt_action"))
        directional_policy = (policy.get("directional_policy") or {}).get(direction) or {}
        multiplier = _safe_float(
            directional_policy.get("risk_multiplier"),
            _safe_float(policy.get("risk_multiplier"), 1.0),
        )
        action = str(event.get("gpt_action") or "")
        opened = action in {"OPEN_LONG", "OPEN_SHORT"}
        shadow_action = self._shadow_action(opened=opened, multiplier=multiplier)

        return {
            "event_id": event.get("id"),
            "timestamp": event.get("timestamp"),
            "timestamp_utc": self._ts_to_utc(event.get("timestamp")),
            "symbol": event.get("symbol"),
            "direction": direction,
            "gpt_action": action,
            "gpt_confidence": _safe_float(event.get("gpt_confidence")),
            "opened_trade": opened,
            "risk_shadow_action": shadow_action,
            "original_size_multiplier": 1.0 if opened else 0.0,
            "adjusted_size_multiplier": multiplier if opened else 0.0,
            "policy_mode": directional_policy.get("policy_mode") or policy.get("policy_mode"),
            "policy_reasons": directional_policy.get("reasons", []),
            "primary_veto": features.get("primary_veto"),
            "live_enforcement": False,
        }

    def _shadow_action(self, opened: bool, multiplier: float) -> str:
        if not opened:
            return "observe_hold"
        if multiplier <= 0.25:
            return "would_min_size_or_block"
        if multiplier < 1.0:
            return "would_reduce_size"
        return "would_allow_full_size"

    def _summary(self, decisions: list[dict]) -> dict:
        by_action = Counter(item.get("risk_shadow_action") for item in decisions)
        by_direction = Counter(item.get("direction") for item in decisions)
        opened = [item for item in decisions if item.get("opened_trade")]
        adjusted = [item for item in opened if item.get("risk_shadow_action") != "would_allow_full_size"]
        return {
            "loaded_decisions": len(decisions),
            "opened_trades": len(opened),
            "would_adjust_open_trades": len(adjusted),
            "by_risk_shadow_action": dict(by_action),
            "by_direction": dict(by_direction),
            "top_adjusted_symbols": [
                {"symbol": symbol, "events": count}
                for symbol, count in Counter(item.get("symbol") for item in adjusted).most_common(10)
            ],
            "average_adjusted_open_multiplier": (
                round(sum(_safe_float(item.get("adjusted_size_multiplier")) for item in opened) / len(opened), 3)
                if opened else 0.0
            ),
        }

    def _ts_to_utc(self, timestamp_ms: Any) -> Optional[str]:
        ts = _safe_int(timestamp_ms)
        if not ts:
            return None
        return datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def run_risk_strategy_bridge(
    hours: int = 24,
    limit: int = 1000,
    risk_policy_path: str = DEFAULT_RISK_POLICY,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    db = DatabaseManager(db_path=DB_FILE)
    try:
        report = RiskStrategyBridge(db=db, risk_policy_path=risk_policy_path).build_report(hours=hours, limit=limit)
    finally:
        db.close_connection()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Apply risk policy to recent GPT decisions in shadow mode.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window.")
    parser.add_argument("--limit", type=int, default=1000, help="Max recent GPT decisions.")
    parser.add_argument("--risk-policy", type=str, default=DEFAULT_RISK_POLICY, help="Risk policy JSON.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_risk_strategy_bridge(
        hours=args.hours,
        limit=args.limit,
        risk_policy_path=args.risk_policy,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
