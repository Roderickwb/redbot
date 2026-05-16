# ============================================================
# src/analysis/pre_gpt_gate_report.py
# ============================================================
"""
Shadow analysis for a future pre-GPT gate.

This is read-only. It estimates which GPT calls could have been skipped by
cheap pre-GPT context signals and measures the likely saved calls versus missed
open trades. It does not change live strategy behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "gpt_decisions")
DEFAULT_LATEST_FILE = "latest_pre_gpt_gate_report.json"


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


def _pct(part: int, total: int) -> float:
    return round(part / total * 100.0, 2) if total else 0.0


def _parse_json(raw: Any) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _contains_any(text: Any, needles: tuple[str, ...]) -> bool:
    haystack = str(text or "").lower()
    return any(needle in haystack for needle in needles)


class PreGptGateReport:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager(db_path=DB_FILE)

    def build_report(self, limit: int = 5000) -> dict:
        events = self._load_events(limit=limit)
        rows = []
        totals = {
            "evaluated_decisions": 0,
            "would_skip_gpt": 0,
            "would_keep_gpt": 0,
            "skipped_holds": 0,
            "skipped_opens": 0,
            "skipped_open_winners": 0,
            "skipped_open_losers": 0,
            "estimated_saved_r": 0.0,
            "estimated_missed_r": 0.0,
        }
        by_reason: Counter[str] = Counter()
        by_symbol: Counter[str] = Counter()
        by_action: Counter[str] = Counter()
        examples = []

        for event in events:
            row = self._evaluate_event(event)
            rows.append(row)
            totals["evaluated_decisions"] += 1
            by_action[row["gpt_action"]] += 1

            if row["would_skip_gpt"]:
                totals["would_skip_gpt"] += 1
                by_symbol[row["symbol"]] += 1
                for reason in row["reasons"]:
                    by_reason[reason] += 1

                if row["gpt_action"] == "HOLD":
                    totals["skipped_holds"] += 1
                elif row["gpt_action"] in ("OPEN_LONG", "OPEN_SHORT"):
                    totals["skipped_opens"] += 1
                    if row["cf_r"] > 0:
                        totals["skipped_open_winners"] += 1
                        totals["estimated_missed_r"] += row["cf_r"]
                    elif row["cf_r"] < 0:
                        totals["skipped_open_losers"] += 1
                        totals["estimated_saved_r"] += abs(row["cf_r"])

                if len(examples) < 25:
                    examples.append(row)
            else:
                totals["would_keep_gpt"] += 1

        totals["estimated_saved_r"] = round(totals["estimated_saved_r"], 4)
        totals["estimated_missed_r"] = round(totals["estimated_missed_r"], 4)
        totals["estimated_net_saved_r"] = round(totals["estimated_saved_r"] - totals["estimated_missed_r"], 4)
        totals["call_reduction_pct"] = _pct(totals["would_skip_gpt"], totals["evaluated_decisions"])
        totals["skipped_open_rate_pct"] = _pct(totals["skipped_opens"], totals["would_skip_gpt"])
        totals["verdict"] = self._verdict(totals)

        return {
            "meta": {
                "limit": limit,
                "loaded_events": len(events),
                "live_enforcement": False,
            },
            "summary": totals,
            "by_reason": [{"reason": key, "events": value} for key, value in by_reason.most_common(20)],
            "by_symbol": [{"symbol": key, "events": value} for key, value in by_symbol.most_common(20)],
            "by_action": dict(by_action),
            "examples": examples,
            "rules": [
                "risk_off_long_candidate",
                "profile_risk_minimal_or_negative_edge",
                "chart_local_chop_or_noisy",
                "market_cash_bias_with_weak_profile",
            ],
        }

    def _load_events(self, limit: int) -> list[dict]:
        rows = self.db.execute_query(
            """
            SELECT id, timestamp, symbol, gpt_action, gpt_confidence, features_json, outcome_json
            FROM strategy_events
            WHERE event_type = 'gpt_decision'
              AND outcome_status = 'labeled'
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        cols = ["id", "timestamp", "symbol", "gpt_action", "gpt_confidence", "features_json", "outcome_json"]
        return [dict(zip(cols, row)) for row in rows] if rows else []

    def _evaluate_event(self, event: dict) -> dict:
        features = _parse_json(event.get("features_json"))
        outcome = _parse_json(event.get("outcome_json"))
        market = features.get("market_regime") or {}
        profile = features.get("coin_profile") or {}
        chart = features.get("chart_features") or {}
        chart_1h = chart.get("1h") or {}
        chart_4h = chart.get("4h") or {}
        counterfactual = outcome.get("counterfactual_trade") or {}
        action = str(event.get("gpt_action") or "UNKNOWN")
        symbol = str(event.get("symbol") or "UNKNOWN")
        reasons = []

        direction = self._direction(features, action)
        market_regime = str(market.get("regime") or "").lower()
        market_bias = str(market.get("directional_bias") or "").lower()
        risk_mult = _safe_float(profile.get("risk_multiplier"), 1.0)
        expectancy = _safe_float(profile.get("expectancy_R"))
        flags = {str(flag).lower() for flag in (profile.get("flags") or [])}
        structure_1h = chart_1h.get("structure_label") or chart_1h.get("label")
        structure_4h = chart_4h.get("structure_label") or chart_4h.get("label")

        if direction == "long" and (market_regime == "risk_off" or market_bias == "short_or_cash"):
            reasons.append("risk_off_long_candidate")
        if risk_mult <= 0.5 or "counterfactual_edge_negative" in flags or expectancy <= -0.25:
            reasons.append("profile_risk_minimal_or_negative_edge")
        if _contains_any(structure_1h, ("local_chop", "chop", "noisy")) or _contains_any(structure_4h, ("local_chop", "chop", "noisy")):
            reasons.append("chart_local_chop_or_noisy")
        if market_bias == "short_or_cash" and risk_mult < 1.0 and direction == "long":
            reasons.append("market_cash_bias_with_weak_profile")

        would_skip = bool(reasons)
        cf_r = _safe_float(counterfactual.get("r_multiple"))

        return {
            "event_id": event.get("id"),
            "timestamp": event.get("timestamp"),
            "symbol": symbol,
            "gpt_action": action,
            "direction": direction,
            "would_skip_gpt": would_skip,
            "reasons": reasons,
            "risk_multiplier": risk_mult,
            "profile_expectancy_R": expectancy,
            "market_regime": market_regime,
            "market_bias": market_bias,
            "structure_1h": structure_1h,
            "structure_4h": structure_4h,
            "cf_r": round(cf_r, 4),
            "outcome_label": outcome.get("label"),
            "cf_label": counterfactual.get("label"),
        }

    def _direction(self, features: dict, action: str) -> str:
        if action == "OPEN_LONG":
            return "long"
        if action == "OPEN_SHORT":
            return "short"
        chart = features.get("chart_features") or {}
        for timeframe in ("1h", "4h"):
            intended = ((chart.get(timeframe) or {}).get("intended_direction") or "").lower()
            if intended in ("long", "short"):
                return intended
        return "unknown"

    def _verdict(self, totals: dict) -> str:
        skipped = _safe_int(totals.get("would_skip_gpt"))
        evaluated = _safe_int(totals.get("evaluated_decisions"))
        skipped_opens = _safe_int(totals.get("skipped_opens"))
        net = _safe_float(totals.get("estimated_net_saved_r"))

        if evaluated < 100:
            return "insufficient_sample"
        if skipped < 50:
            return "low_savings"
        if skipped_opens == 0 and totals.get("call_reduction_pct", 0) >= 10.0:
            return "promising_for_shadow"
        if net > 0 and totals.get("call_reduction_pct", 0) >= 10.0:
            return "promising_for_shadow"
        if net < 0:
            return "too_risky"
        return "collect_more_evidence"


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_pre_gpt_gate_report(output_dir: str = DEFAULT_OUTPUT_DIR, limit: int = 5000) -> dict:
    db = DatabaseManager(db_path=DB_FILE)
    try:
        report = PreGptGateReport(db=db).build_report(limit=limit)
    finally:
        db.close_connection()

    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build shadow pre-GPT gate efficiency report.")
    parser.add_argument("--limit", type=int, default=5000, help="Max labeled GPT decisions to read.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_pre_gpt_gate_report(output_dir=args.output_dir, limit=args.limit)
    print(json.dumps({
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
