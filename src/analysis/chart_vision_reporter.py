# ============================================================
# src/analysis/chart_vision_reporter.py
# ============================================================
"""
Read-only QA report for the Chart Vision Layer.

This checks whether computed labels such as chop/noisy/pullback/clean_trend
line up with later outcomes and counterfactual trade results.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Iterable, Optional

from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "chart_vision")
DEFAULT_LATEST_FILE = "latest_chart_vision_report.json"


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
        "hold": 0,
        "open": 0,
        "cf_count": 0,
        "cf_total_r": 0.0,
        "cf_positive": 0,
        "cf_loss": 0,
        "missed_opportunity": 0,
        "skip_protected": 0,
        "range_breakout_up": 0,
        "range_breakout_down": 0,
        "range_no_breakout": 0,
        "avg_entry_score_total": 0.0,
        "avg_entry_score_count": 0,
        "avg_trend_score_total": 0.0,
        "avg_trend_score_count": 0,
    }


class ChartVisionReporter:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager(db_path=DB_FILE)

    def build_report(self, limit: int = 5000, symbol: Optional[str] = None) -> dict:
        events = self._load_events(limit=limit, symbol=symbol)
        report = self._summarize(events)
        report["meta"] = {
            "loaded_events": len(events),
            "limit": limit,
            "symbol": symbol,
        }
        return report

    def _load_events(self, limit: int, symbol: Optional[str]) -> list[dict]:
        params: list[Any] = []
        where = [
            "event_type='gpt_decision'",
            "outcome_status='labeled'",
            "json_extract(features_json,'$.chart_features.1h.structure_label') IS NOT NULL",
        ]
        if symbol:
            where.append("symbol = ?")
            params.append(symbol)

        params.append(int(limit))
        rows = self.db.execute_query(
            f"""
            SELECT id, timestamp, symbol, gpt_action, gpt_confidence, features_json, outcome_json
            FROM strategy_events
            WHERE {" AND ".join(where)}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            tuple(params),
        )
        cols = ["id", "timestamp", "symbol", "gpt_action", "gpt_confidence", "features_json", "outcome_json"]
        return [dict(zip(cols, row)) for row in rows] if rows else []

    def _summarize(self, events: list[dict]) -> dict:
        totals = _new_bucket()
        by_symbol = defaultdict(_new_bucket)
        by_structure_1h = defaultdict(_new_bucket)
        by_entry_timing_1h = defaultdict(_new_bucket)
        by_candle_quality_1h = defaultdict(_new_bucket)
        by_structure_4h = defaultdict(_new_bucket)
        by_entry_timing_4h = defaultdict(_new_bucket)
        attention_cases = {
            "chop_or_noisy_but_positive_cf": [],
            "clean_or_pullback_but_negative_cf": [],
            "late_trend_positive_cf": [],
            "noisy_opens": [],
        }

        for event in events:
            features = self._parse_json(event.get("features_json"))
            outcome = self._parse_json(event.get("outcome_json"))
            chart = features.get("chart_features") or {}
            one_h = chart.get("1h") or {}
            four_h = chart.get("4h") or {}

            buckets = [
                totals,
                by_symbol[event.get("symbol") or "UNKNOWN"],
                by_structure_1h[one_h.get("structure_label") or "missing"],
                by_entry_timing_1h[one_h.get("entry_timing") or "missing"],
                by_candle_quality_1h[one_h.get("last_candle_quality") or "missing"],
                by_structure_4h[four_h.get("structure_label") or "missing"],
                by_entry_timing_4h[four_h.get("entry_timing") or "missing"],
            ]
            for bucket in buckets:
                self._add_event(bucket, event, features, outcome)

            self._collect_attention_case(attention_cases, event, features, outcome)

        return {
            "totals": self._finalize_bucket(totals),
            "by_symbol": self._finalize_mapping(by_symbol),
            "by_structure_1h": self._finalize_mapping(by_structure_1h),
            "by_entry_timing_1h": self._finalize_mapping(by_entry_timing_1h),
            "by_candle_quality_1h": self._finalize_mapping(by_candle_quality_1h),
            "by_structure_4h": self._finalize_mapping(by_structure_4h),
            "by_entry_timing_4h": self._finalize_mapping(by_entry_timing_4h),
            "top_attention": self._build_attention_lists(by_symbol, by_structure_1h, by_entry_timing_1h),
            "attention_cases": attention_cases,
        }

    def _add_event(self, bucket: dict, event: dict, features: dict, outcome: dict) -> None:
        bucket["events"] += 1
        action = event.get("gpt_action")
        label = outcome.get("label")
        counterfactual = outcome.get("counterfactual_trade") or {}
        cf_label = counterfactual.get("label")
        cf_r = _safe_float(counterfactual.get("r_multiple"))
        scores = features.get("scores") or {}

        if action == "HOLD":
            bucket["hold"] += 1
        elif action in ("OPEN_LONG", "OPEN_SHORT"):
            bucket["open"] += 1

        if cf_label:
            bucket["cf_count"] += 1
            bucket["cf_total_r"] += cf_r
            if cf_label in ("cf_win", "cf_tp1_then_positive", "cf_small_win"):
                bucket["cf_positive"] += 1
            if cf_label == "cf_loss":
                bucket["cf_loss"] += 1

        if label in bucket:
            bucket[label] += 1

        entry_score = scores.get("entry")
        if entry_score is not None:
            bucket["avg_entry_score_total"] += _safe_float(entry_score)
            bucket["avg_entry_score_count"] += 1

        trend_score = scores.get("trend")
        if trend_score is not None:
            bucket["avg_trend_score_total"] += _safe_float(trend_score)
            bucket["avg_trend_score_count"] += 1

    def _collect_attention_case(self, cases: dict, event: dict, features: dict, outcome: dict) -> None:
        chart = features.get("chart_features") or {}
        one_h = chart.get("1h") or {}
        structure = one_h.get("structure_label")
        timing = one_h.get("entry_timing")
        quality = one_h.get("last_candle_quality")
        counterfactual = outcome.get("counterfactual_trade") or {}
        cf_r = _safe_float(counterfactual.get("r_multiple"))

        row = {
            "id": event.get("id"),
            "timestamp": event.get("timestamp"),
            "symbol": event.get("symbol"),
            "action": event.get("gpt_action"),
            "confidence": event.get("gpt_confidence"),
            "structure_1h": structure,
            "entry_timing_1h": timing,
            "last_candle_quality_1h": quality,
            "cf_label": counterfactual.get("label"),
            "cf_r": round(cf_r, 4),
            "outcome_label": outcome.get("label"),
            "entry_score": (features.get("scores") or {}).get("entry"),
            "primary_veto": features.get("primary_veto"),
        }

        if (structure == "chop" or timing == "noisy") and cf_r >= 0.75:
            cases["chop_or_noisy_but_positive_cf"].append(row)
        if (structure in ("clean_trend", "pullback") or timing == "clean") and cf_r <= -0.75:
            cases["clean_or_pullback_but_negative_cf"].append(row)
        if structure == "late_trend" and cf_r >= 0.75:
            cases["late_trend_positive_cf"].append(row)
        if event.get("gpt_action") in ("OPEN_LONG", "OPEN_SHORT") and timing == "noisy":
            cases["noisy_opens"].append(row)

        for key in cases:
            cases[key] = cases[key][:20]

    def _finalize_mapping(self, mapping: dict[str, dict]) -> dict:
        return {
            key: self._finalize_bucket(value)
            for key, value in sorted(mapping.items(), key=lambda item: item[0])
        }

    def _finalize_bucket(self, bucket: dict) -> dict:
        result = dict(bucket)
        events = result["events"]
        cf_count = result["cf_count"]
        result["hold_rate_pct"] = _pct(result["hold"], events)
        result["open_rate_pct"] = _pct(result["open"], events)
        result["cf_avg_r"] = _avg(result["cf_total_r"], cf_count)
        result["cf_positive_rate_pct"] = _pct(result["cf_positive"], cf_count)
        result["cf_loss_rate_pct"] = _pct(result["cf_loss"], cf_count)
        result["avg_entry_score"] = _avg(result["avg_entry_score_total"], result["avg_entry_score_count"])
        result["avg_trend_score"] = _avg(result["avg_trend_score_total"], result["avg_trend_score_count"])
        result["cf_total_r"] = round(result["cf_total_r"], 4)
        del result["avg_entry_score_total"]
        del result["avg_entry_score_count"]
        del result["avg_trend_score_total"]
        del result["avg_trend_score_count"]
        return result

    def _build_attention_lists(self, by_symbol: dict, by_structure: dict, by_timing: dict) -> dict:
        finalized_symbols = self._finalize_mapping(by_symbol)
        finalized_structure = self._finalize_mapping(by_structure)
        finalized_timing = self._finalize_mapping(by_timing)
        return {
            "symbols_worst_cf_avg_r": self._top(finalized_symbols, "cf_avg_r", reverse=False, only_negative=True),
            "symbols_best_cf_avg_r": self._top(finalized_symbols, "cf_avg_r", reverse=True),
            "structure_best_cf_avg_r": self._top(finalized_structure, "cf_avg_r", reverse=True),
            "structure_worst_cf_avg_r": self._top(finalized_structure, "cf_avg_r", reverse=False, only_negative=True),
            "entry_timing_best_cf_avg_r": self._top(finalized_timing, "cf_avg_r", reverse=True),
            "entry_timing_worst_cf_avg_r": self._top(finalized_timing, "cf_avg_r", reverse=False, only_negative=True),
        }

    def _top(self, mapping: dict, metric: str, reverse: bool = True, only_negative: bool = False) -> list[dict]:
        rows = [
            {"name": name, metric: values.get(metric, 0), "events": values.get("events", 0)}
            for name, values in mapping.items()
        ]
        rows.sort(key=lambda row: row[metric], reverse=reverse)
        if only_negative:
            rows = [row for row in rows if row[metric] < 0]
        else:
            rows = [row for row in rows if row[metric] != 0]
        return rows[:10]

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
    parser = argparse.ArgumentParser(description="Build Chart Vision QA report.")
    parser.add_argument("--limit", type=int, default=5000, help="Max labeled GPT events to read.")
    parser.add_argument("--symbol", type=str, default=None, help="Optional symbol filter.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    db = DatabaseManager(db_path=DB_FILE)
    try:
        reporter = ChartVisionReporter(db=db)
        report = reporter.build_report(limit=args.limit, symbol=args.symbol)
    finally:
        db.close_connection()

    output_path = os.path.join(args.output_dir, DEFAULT_LATEST_FILE)
    write_json(output_path, report)
    result = {
        "loaded_events": report.get("meta", {}).get("loaded_events", 0),
        "output_path": output_path,
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
