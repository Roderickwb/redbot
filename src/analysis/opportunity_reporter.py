# ============================================================
# src/analysis/opportunity_reporter.py
# ============================================================
"""
Read-only opportunity intelligence report.

This is the generic analysis layer for candidate decisions:
- long_candidate and short_candidate
- HOLD versus OPEN
- chart labels, GPT vetoes, market regime and later outcomes
- per coin, per direction and per regime

It does not change trading behavior. It exists to feed advisor/learning layers.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Iterable, Optional

from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "opportunities")
DEFAULT_LATEST_FILE = "latest_opportunity_report.json"


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


def _avg(total: float, count: int) -> float:
    return round(total / count, 4) if count else 0.0


def _new_bucket() -> dict:
    return {
        "events": 0,
        "holds": 0,
        "opens": 0,
        "open_long": 0,
        "open_short": 0,
        "cf_count": 0,
        "cf_total_r": 0.0,
        "cf_positive": 0,
        "cf_loss": 0,
        "missed_opportunity": 0,
        "skip_protected": 0,
        "open_followed_through": 0,
        "open_went_against": 0,
        "open_mixed_volatility": 0,
        "open_no_followthrough": 0,
        "missing_counterfactual": 0,
    }


class OpportunityReporter:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager(db_path=DB_FILE)

    def build_report(
        self,
        limit: int = 5000,
        hours: Optional[int] = None,
        direction: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> dict:
        events = self._load_events(limit=limit, hours=hours, direction=direction, regime=regime)
        report = self._summarize(events)
        report["meta"] = {
            "loaded_candidates": len(events),
            "limit": limit,
            "hours": hours,
            "direction": direction,
            "regime": regime,
        }
        return report

    def _load_events(
        self,
        limit: int,
        hours: Optional[int],
        direction: Optional[str],
        regime: Optional[str],
    ) -> list[dict]:
        params: list[Any] = []
        where = [
            "event_type='gpt_decision'",
            "algo_signal IN ('long_candidate', 'short_candidate')",
            "outcome_status='labeled'",
        ]
        if direction:
            where.append("algo_signal=?")
            params.append("long_candidate" if direction == "long" else "short_candidate")
        if regime:
            where.append("json_extract(features_json,'$.market_regime.regime')=?")
            params.append(regime)
        if hours:
            where.append("timestamp >= (strftime('%s','now') * 1000) - ?")
            params.append(int(hours) * 3600 * 1000)

        params.append(int(limit))
        rows = self.db.execute_query(
            f"""
            SELECT
              id, timestamp, symbol, algo_signal, trend_dir, gpt_action, gpt_confidence,
              features_json, outcome_json
            FROM strategy_events
            WHERE {" AND ".join(where)}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            tuple(params),
        )
        cols = [
            "id", "timestamp", "symbol", "algo_signal", "trend_dir", "gpt_action",
            "gpt_confidence", "features_json", "outcome_json",
        ]
        return [dict(zip(cols, row)) for row in rows] if rows else []

    def _summarize(self, events: list[dict]) -> dict:
        totals = _new_bucket()
        by_symbol = defaultdict(_new_bucket)
        by_direction = defaultdict(_new_bucket)
        by_symbol_direction = defaultdict(_new_bucket)
        by_veto = defaultdict(_new_bucket)
        by_structure_1h = defaultdict(_new_bucket)
        by_entry_timing_1h = defaultdict(_new_bucket)
        by_market_regime = defaultdict(_new_bucket)
        by_regime_direction = defaultdict(_new_bucket)

        attention_cases = {
            "held_positive_opportunities": [],
            "held_large_positive_opportunities": [],
            "protected_holds": [],
            "bad_opens": [],
            "good_opens": [],
            "opened_weak_entries": [],
        }

        for event in events:
            features = self._parse_json(event.get("features_json"))
            outcome = self._parse_json(event.get("outcome_json"))
            direction = self._direction_for(event)
            chart_1h = ((features.get("chart_features") or {}).get("1h") or {})
            regime = features.get("market_regime") or {}
            primary_veto = features.get("primary_veto") or "missing"
            structure = chart_1h.get("structure_label") or "missing"
            timing = chart_1h.get("entry_timing") or "missing"
            market_regime = regime.get("regime") or "missing"
            symbol = event.get("symbol") or "UNKNOWN"

            buckets = [
                totals,
                by_symbol[symbol],
                by_direction[direction],
                by_symbol_direction[f"{symbol}|{direction}"],
                by_veto[primary_veto],
                by_structure_1h[structure],
                by_entry_timing_1h[timing],
                by_market_regime[market_regime],
                by_regime_direction[f"{market_regime}|{direction}"],
            ]
            for bucket in buckets:
                self._add_event(bucket, event, outcome)

            self._collect_attention_cases(attention_cases, event, features, outcome)

        return {
            "totals": self._finalize_bucket(totals),
            "by_symbol": self._finalize_mapping(by_symbol),
            "by_direction": self._finalize_mapping(by_direction),
            "by_symbol_direction": self._finalize_mapping(by_symbol_direction),
            "by_primary_veto": self._finalize_mapping(by_veto),
            "by_structure_1h": self._finalize_mapping(by_structure_1h),
            "by_entry_timing_1h": self._finalize_mapping(by_entry_timing_1h),
            "by_market_regime": self._finalize_mapping(by_market_regime),
            "by_regime_direction": self._finalize_mapping(by_regime_direction),
            "top_attention": self._top_attention(by_symbol_direction, by_veto, by_structure_1h, by_entry_timing_1h),
            "attention_cases": attention_cases,
        }

    def _add_event(self, bucket: dict, event: dict, outcome: dict) -> None:
        bucket["events"] += 1
        action = event.get("gpt_action")
        label = outcome.get("label")
        counterfactual = outcome.get("counterfactual_trade") or {}
        cf_label = counterfactual.get("label")
        cf_r = _safe_float(counterfactual.get("r_multiple"))

        if action == "HOLD":
            bucket["holds"] += 1
        elif action in ("OPEN_LONG", "OPEN_SHORT"):
            bucket["opens"] += 1
            if action == "OPEN_LONG":
                bucket["open_long"] += 1
            else:
                bucket["open_short"] += 1

        if label in bucket:
            bucket[label] += 1

        if cf_label:
            bucket["cf_count"] += 1
            bucket["cf_total_r"] += cf_r
            if cf_label in ("cf_win", "cf_tp1_then_positive", "cf_small_win"):
                bucket["cf_positive"] += 1
            elif cf_label == "cf_loss":
                bucket["cf_loss"] += 1
        else:
            bucket["missing_counterfactual"] += 1

    def _collect_attention_cases(self, cases: dict, event: dict, features: dict, outcome: dict) -> None:
        counterfactual = outcome.get("counterfactual_trade") or {}
        cf_r = _safe_float(counterfactual.get("r_multiple"))
        action = event.get("gpt_action")
        scores = features.get("scores") or {}
        chart_1h = ((features.get("chart_features") or {}).get("1h") or {})
        row = {
            "id": event.get("id"),
            "timestamp": event.get("timestamp"),
            "symbol": event.get("symbol"),
            "direction": self._direction_for(event),
            "trend_dir": event.get("trend_dir"),
            "action": action,
            "confidence": event.get("gpt_confidence"),
            "primary_veto": features.get("primary_veto"),
            "structure_1h": chart_1h.get("structure_label"),
            "entry_timing_1h": chart_1h.get("entry_timing"),
            "directional_continuation": chart_1h.get("directional_continuation"),
            "market_regime": (features.get("market_regime") or {}).get("regime"),
            "entry_score": _safe_int(scores.get("entry")),
            "cf_label": counterfactual.get("label"),
            "cf_r": round(cf_r, 4),
            "outcome_label": outcome.get("label"),
        }

        if action == "HOLD" and cf_r >= 0.5:
            cases["held_positive_opportunities"].append(row)
        if action == "HOLD" and cf_r >= 1.0:
            cases["held_large_positive_opportunities"].append(row)
        if action == "HOLD" and cf_r <= -0.5:
            cases["protected_holds"].append(row)
        if action in ("OPEN_LONG", "OPEN_SHORT") and cf_r <= -0.5:
            cases["bad_opens"].append(row)
        if action in ("OPEN_LONG", "OPEN_SHORT") and cf_r >= 0.5:
            cases["good_opens"].append(row)
        if action in ("OPEN_LONG", "OPEN_SHORT") and _safe_int(scores.get("entry")) < 60:
            cases["opened_weak_entries"].append(row)

        for key in cases:
            cases[key] = cases[key][:25]

    def _direction_for(self, event: dict) -> str:
        signal = str(event.get("algo_signal") or "")
        if signal == "long_candidate":
            return "long"
        if signal == "short_candidate":
            return "short"
        return "unknown"

    def _finalize_mapping(self, mapping: dict[str, dict]) -> dict:
        return {
            key: self._finalize_bucket(value)
            for key, value in sorted(mapping.items(), key=lambda item: item[0])
        }

    def _finalize_bucket(self, bucket: dict) -> dict:
        result = dict(bucket)
        events = result["events"]
        holds = result["holds"]
        cf_count = result["cf_count"]
        result["hold_rate_pct"] = _pct(holds, events)
        result["open_rate_pct"] = _pct(result["opens"], events)
        result["missed_rate_pct"] = _pct(result["missed_opportunity"], holds)
        result["protection_rate_pct"] = _pct(result["skip_protected"], holds)
        result["cf_positive_rate_pct"] = _pct(result["cf_positive"], cf_count)
        result["cf_loss_rate_pct"] = _pct(result["cf_loss"], cf_count)
        result["cf_avg_r"] = _avg(result["cf_total_r"], cf_count)
        result["cf_total_r"] = round(result["cf_total_r"], 4)
        return result

    def _top_attention(self, by_symbol_direction: dict, by_veto: dict, by_structure: dict, by_timing: dict) -> dict:
        finalized_symbol_direction = self._finalize_mapping(by_symbol_direction)
        finalized_veto = self._finalize_mapping(by_veto)
        finalized_structure = self._finalize_mapping(by_structure)
        finalized_timing = self._finalize_mapping(by_timing)
        return {
            "symbol_direction_best_cf": self._top(finalized_symbol_direction, "cf_avg_r", reverse=True),
            "symbol_direction_worst_cf": self._top(finalized_symbol_direction, "cf_avg_r", reverse=False),
            "symbol_direction_most_missed": self._top(finalized_symbol_direction, "missed_opportunity", reverse=True),
            "veto_best_cf": self._top(finalized_veto, "cf_avg_r", reverse=True),
            "veto_worst_cf": self._top(finalized_veto, "cf_avg_r", reverse=False),
            "structure_best_cf": self._top(finalized_structure, "cf_avg_r", reverse=True),
            "structure_worst_cf": self._top(finalized_structure, "cf_avg_r", reverse=False),
            "entry_timing_best_cf": self._top(finalized_timing, "cf_avg_r", reverse=True),
            "entry_timing_worst_cf": self._top(finalized_timing, "cf_avg_r", reverse=False),
        }

    def _top(self, mapping: dict, metric: str, reverse: bool = True) -> list[dict]:
        rows = [
            {"name": name, metric: values.get(metric, 0), "events": values.get("events", 0)}
            for name, values in mapping.items()
        ]
        rows.sort(key=lambda row: row[metric], reverse=reverse)
        return [row for row in rows if row[metric] != 0][:10]

    def _parse_json(self, raw: Any) -> dict:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            return {}


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build generic opportunity analytics report.")
    parser.add_argument("--limit", type=int, default=5000, help="Max labeled candidate decisions to read.")
    parser.add_argument("--hours", type=int, default=None, help="Optional lookback window in hours.")
    parser.add_argument("--direction", choices=["long", "short"], default=None, help="Optional direction filter.")
    parser.add_argument("--regime", type=str, default=None, help="Optional market regime filter, e.g. risk_off.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    db = DatabaseManager(db_path=DB_FILE)
    try:
        reporter = OpportunityReporter(db=db)
        report = reporter.build_report(
            limit=args.limit,
            hours=args.hours,
            direction=args.direction,
            regime=args.regime,
        )
    finally:
        db.close_connection()

    output_path = os.path.join(args.output_dir, DEFAULT_LATEST_FILE)
    write_json(output_path, report)
    result = {
        "loaded_candidates": report.get("meta", {}).get("loaded_candidates", 0),
        "output_path": output_path,
        "hold_rate_pct": report.get("totals", {}).get("hold_rate_pct", 0),
        "open_rate_pct": report.get("totals", {}).get("open_rate_pct", 0),
        "cf_avg_r": report.get("totals", {}).get("cf_avg_r", 0),
        "attention_cases": {
            key: len(value)
            for key, value in (report.get("attention_cases") or {}).items()
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
