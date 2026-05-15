# ============================================================
# src/analysis/shadow_live_bridge.py
# ============================================================
"""
Shadow-to-live bridge.

Runs approved experiments against recent live strategy decisions in shadow mode.
This module is strictly read-only: it does not alter GPT decisions, open trades,
block trades, or change risk. It only records what an approved shadow policy
would have done on recent events.
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


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "shadow_live")
DEFAULT_LATEST_FILE = "latest_shadow_live_bridge_report.json"
DEFAULT_EXPERIMENT_PLAN = os.path.join("analysis", "experiments", "latest_experiment_plan.json")


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


def _parse_json(raw: Any) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


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


def _direction_for(algo_signal: Any, trend_dir: Any) -> str:
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


class ShadowLiveBridge:
    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        experiment_plan_path: str = DEFAULT_EXPERIMENT_PLAN,
    ):
        self.db = db or DatabaseManager(db_path=DB_FILE)
        self.experiment_plan_path = experiment_plan_path

    def build_report(self, hours: int = 24, limit: int = 1000) -> dict:
        plan = _load_json(self.experiment_plan_path, {"experiments": []})
        experiments = [
            item for item in (plan.get("experiments") or [])
            if item.get("status") == "approved_for_shadow"
        ]
        events = self._load_recent_events(hours=hours, limit=limit)
        matches = self._matches(experiments, events)

        return {
            "meta": {
                "created_utc": _utc_now(),
                "experiment_plan_path": self.experiment_plan_path,
                "hours": hours,
                "limit": limit,
                "loaded_events": len(events),
                "active_shadow_policies": len(experiments),
            },
            "summary": self._summary(experiments, events, matches),
            "active_policies": [self._compact_policy(item) for item in experiments],
            "matches": matches[:100],
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
              AND features_json IS NOT NULL
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

    def _matches(self, experiments: list[dict], events: list[dict]) -> list[dict]:
        matches = []
        for event in events:
            row_patterns = self._event_patterns(event)
            for experiment in experiments:
                pattern = ((experiment.get("evidence") or {}).get("pattern") or "").strip()
                if not pattern or pattern not in row_patterns:
                    continue
                matches.append(self._match(event, experiment, pattern))
        return matches

    def _match(self, event: dict, experiment: dict, pattern: str) -> dict:
        features = _parse_json(event.get("features_json"))
        chart_1h = ((features.get("chart_features") or {}).get("1h") or {})
        regime = features.get("market_regime") or {}
        return {
            "event_id": event.get("id"),
            "timestamp": event.get("timestamp"),
            "timestamp_utc": self._ts_to_utc(event.get("timestamp")),
            "symbol": event.get("symbol"),
            "direction": _direction_for(event.get("algo_signal"), event.get("trend_dir")),
            "gpt_action": event.get("gpt_action"),
            "gpt_confidence": _safe_float(event.get("gpt_confidence")),
            "experiment_id": experiment.get("id"),
            "experiment_type": experiment.get("experiment_type"),
            "pattern": pattern,
            "shadow_action": self._shadow_action(experiment, event),
            "current_primary_veto": features.get("primary_veto"),
            "market_regime": regime.get("regime"),
            "chart_1h": {
                "structure_label": chart_1h.get("structure_label"),
                "chop_subtype": chart_1h.get("chop_subtype"),
                "entry_timing": chart_1h.get("entry_timing"),
                "breakout_pressure": _safe_int(chart_1h.get("breakout_pressure")),
                "breakdown_pressure": _safe_int(chart_1h.get("breakdown_pressure")),
            },
            "guardrails": experiment.get("guardrails", []),
        }

    def _event_patterns(self, event: dict) -> set[str]:
        features = _parse_json(event.get("features_json"))
        regime = (features.get("market_regime") or {}).get("regime") or "missing"
        chart_1h = ((features.get("chart_features") or {}).get("1h") or {})
        profile = features.get("coin_profile") or {}
        direction = _direction_for(event.get("algo_signal"), event.get("trend_dir"))
        veto = features.get("primary_veto") or "missing"
        structure = chart_1h.get("structure_label") or "missing"
        chop_subtype = chart_1h.get("chop_subtype") or "missing"
        entry_timing = chart_1h.get("entry_timing") or "missing"
        pressure = self._pressure_bucket(direction, chart_1h)
        risk_mult = _safe_float(profile.get("risk_multiplier"), 1.0)
        flags = set(profile.get("flags") or [])

        drawdown_flag = "drawdown_flag" if "DRAWDOWN_RISK" in flags else "no_drawdown_flag"
        cf_flag = "cf_negative_flag" if "COUNTERFACTUAL_EDGE_NEGATIVE" in flags else "no_cf_negative_flag"

        return {
            f"{direction}|{regime}|{veto}",
            f"{direction}|{regime}|{structure}|{chop_subtype}|{entry_timing}",
            f"{direction}|{regime}|{veto}|{pressure}",
            f"{direction}|{veto}|risk_mult_{risk_mult:.2f}|{drawdown_flag}|{cf_flag}",
        }

    def _pressure_bucket(self, direction: str, chart_1h: dict) -> str:
        if direction == "short":
            value = _safe_int(chart_1h.get("breakdown_pressure"))
        elif direction == "long":
            value = _safe_int(chart_1h.get("breakout_pressure"))
        else:
            value = max(
                _safe_int(chart_1h.get("breakdown_pressure")),
                _safe_int(chart_1h.get("breakout_pressure")),
            )
        if value >= 75:
            return "p75"
        if value >= 50:
            return "p50"
        if value >= 25:
            return "p25"
        return "p0"

    def _shadow_action(self, experiment: dict, event: dict) -> str:
        exp_type = experiment.get("experiment_type")
        action = event.get("gpt_action")
        if exp_type == "shadow_relax_entry_rule":
            if action == "HOLD":
                return "would_allow_candidate"
            return "would_not_change_open"
        if exp_type == "shadow_protection_rule":
            if action in {"OPEN_LONG", "OPEN_SHORT"}:
                return "would_block_open"
            return "would_keep_hold"
        return "observe_only"

    def _compact_policy(self, experiment: dict) -> dict:
        evidence = experiment.get("evidence") or {}
        return {
            "experiment_id": experiment.get("id"),
            "experiment_type": experiment.get("experiment_type"),
            "status": experiment.get("status"),
            "pattern": evidence.get("pattern"),
            "source": experiment.get("source"),
            "guardrails": experiment.get("guardrails", []),
        }

    def _summary(self, experiments: list[dict], events: list[dict], matches: list[dict]) -> dict:
        by_action = Counter(match.get("shadow_action") for match in matches)
        by_policy = Counter(match.get("experiment_id") for match in matches)
        by_symbol = Counter(match.get("symbol") for match in matches)
        return {
            "active_shadow_policies": len(experiments),
            "loaded_events": len(events),
            "matches": len(matches),
            "by_shadow_action": dict(by_action),
            "top_policies": [
                {"experiment_id": key, "matches": value}
                for key, value in by_policy.most_common(10)
            ],
            "top_symbols": [
                {"symbol": key, "matches": value}
                for key, value in by_symbol.most_common(10)
            ],
        }

    def _ts_to_utc(self, timestamp_ms: Any) -> Optional[str]:
        ts = _safe_int(timestamp_ms)
        if not ts:
            return None
        return datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def run_shadow_live_bridge(
    hours: int = 24,
    limit: int = 1000,
    experiment_plan_path: str = DEFAULT_EXPERIMENT_PLAN,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    db = DatabaseManager(db_path=DB_FILE)
    try:
        report = ShadowLiveBridge(
            db=db,
            experiment_plan_path=experiment_plan_path,
        ).build_report(hours=hours, limit=limit)
    finally:
        db.close_connection()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run approved experiments against recent live decisions in shadow mode.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window for recent GPT decisions.")
    parser.add_argument("--limit", type=int, default=1000, help="Max recent GPT decisions to inspect.")
    parser.add_argument("--experiment-plan", type=str, default=DEFAULT_EXPERIMENT_PLAN, help="Experiment plan JSON.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_shadow_live_bridge(
        hours=args.hours,
        limit=args.limit,
        experiment_plan_path=args.experiment_plan,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
