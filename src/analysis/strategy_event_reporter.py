# ============================================================
# src/analysis/strategy_event_reporter.py
# ============================================================
"""
Read-only summary report for labeled strategy_events.

This is the first reporting layer for the new learning flow. It deliberately
does not depend on the older trade-centric analysis modules.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager

logger = logging.getLogger("strategy_event_reporter")


def _new_bucket() -> Dict[str, Any]:
    return {
        "events": 0,
        "skips": 0,
        "missed_opportunity": 0,
        "skip_protected": 0,
        "skip_correct_no_move": 0,
        "volatile_after_skip": 0,
        "range_breakout_up": 0,
        "range_breakout_down": 0,
        "range_no_breakout": 0,
        "range_volatile_breakout": 0,
        "trade_open": 0,
        "trade_profitable": 0,
        "trade_losing": 0,
        "trade_breakeven": 0,
        "trade_still_open": 0,
        "trade_pnl_eur": 0.0,
    }


def _new_range_bucket() -> Dict[str, Any]:
    return {
        "events": 0,
        "range_breakout_up": 0,
        "range_breakout_down": 0,
        "range_no_breakout": 0,
        "range_volatile_breakout": 0,
    }


class StrategyEventReporter:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager(db_path=DB_FILE)

    def build_report(self, limit: int = 5000, symbol: Optional[str] = None) -> Dict[str, Any]:
        events = self._load_labeled_events(limit=limit, symbol=symbol)

        by_symbol = defaultdict(_new_bucket)
        by_skip_reason = defaultdict(_new_bucket)
        by_event_type = defaultdict(_new_bucket)
        skip_summary = defaultdict(_new_bucket)
        gpt_hold_summary = defaultdict(_new_bucket)
        trade_open_summary = defaultdict(_new_bucket)
        range_summary = defaultdict(_new_range_bucket)

        for event in events:
            outcome = self._parse_outcome(event.get("outcome_json"))

            self._add_event(by_symbol[event.get("symbol") or "UNKNOWN"], event, outcome)
            self._add_event(by_event_type[event.get("event_type") or "UNKNOWN"], event, outcome)

            skip_reason = event.get("skip_reason")
            if skip_reason:
                self._add_event(by_skip_reason[skip_reason], event, outcome)

            if event.get("event_type") == "skip":
                self._add_event(skip_summary[skip_reason or "UNKNOWN"], event, outcome)

            if skip_reason == "gpt_hold" or event.get("gpt_action") == "HOLD":
                self._add_event(gpt_hold_summary[event.get("symbol") or "UNKNOWN"], event, outcome)

            if event.get("event_type") == "trade_open":
                self._add_event(trade_open_summary[event.get("symbol") or "UNKNOWN"], event, outcome)

            if skip_reason == "trend_range":
                self._add_range_event(range_summary[event.get("symbol") or "UNKNOWN"], outcome)

        report = {
            "meta": {
                "loaded_labeled_events": len(events),
                "limit": limit,
                "symbol": symbol,
            },
            "totals": self._finalize_bucket(self._combine_buckets(by_event_type.values())),
            "by_symbol": self._finalize_mapping(by_symbol),
            "by_skip_reason": self._finalize_mapping(by_skip_reason),
            "by_event_type": self._finalize_mapping(by_event_type),
            "skip_summary": self._finalize_mapping(skip_summary),
            "gpt_hold_summary": self._finalize_mapping(gpt_hold_summary),
            "trade_open_summary": self._finalize_mapping(trade_open_summary),
            "range_summary": self._finalize_range_mapping(range_summary),
        }
        report["top_attention"] = self._build_attention_lists(report)
        return report

    def _load_labeled_events(self, limit: int, symbol: Optional[str]) -> list[Dict[str, Any]]:
        params: list[Any] = []
        where = ["outcome_status = 'labeled'"]
        if symbol:
            where.append("symbol = ?")
            params.append(symbol)

        params.append(int(limit))
        rows = self.db.execute_query(
            f"""
            SELECT id, timestamp, symbol, strategy_name, event_type, decision_stage,
                   skip_reason, trend_dir, gpt_action, trade_id, outcome_json
            FROM strategy_events
            WHERE {" AND ".join(where)}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            tuple(params),
        )
        cols = [
            "id", "timestamp", "symbol", "strategy_name", "event_type", "decision_stage",
            "skip_reason", "trend_dir", "gpt_action", "trade_id", "outcome_json",
        ]
        return [dict(zip(cols, row)) for row in rows] if rows else []

    def _add_event(self, bucket: Dict[str, Any], event: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        bucket["events"] += 1

        label = outcome.get("label")
        if event.get("event_type") == "skip":
            bucket["skips"] += 1

        if label in bucket:
            bucket[label] += 1

        realized = outcome.get("realized_trade") or {}
        realized_label = realized.get("label")
        if event.get("event_type") == "trade_open":
            bucket["trade_open"] += 1
            if realized_label in bucket:
                bucket[realized_label] += 1
            try:
                bucket["trade_pnl_eur"] += float(realized.get("pnl_eur") or 0.0)
            except Exception:
                pass

    def _add_range_event(self, bucket: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        bucket["events"] += 1
        label = outcome.get("label")
        if label in bucket:
            bucket[label] += 1

    def _combine_buckets(self, buckets: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        total = _new_bucket()
        for bucket in buckets:
            for key, value in bucket.items():
                total[key] += value
        return total

    def _finalize_mapping(self, mapping: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {
            key: self._finalize_bucket(value)
            for key, value in sorted(mapping.items(), key=lambda item: item[0])
        }

    def _finalize_range_mapping(self, mapping: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {
            key: self._finalize_range_bucket(value)
            for key, value in sorted(mapping.items(), key=lambda item: item[0])
        }

    def _finalize_bucket(self, bucket: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(bucket)
        skips = result["skips"]
        trade_done = result["trade_profitable"] + result["trade_losing"] + result["trade_breakeven"]

        result["missed_rate_pct"] = round(result["missed_opportunity"] / skips * 100.0, 2) if skips else 0.0
        result["skip_protection_rate_pct"] = round(result["skip_protected"] / skips * 100.0, 2) if skips else 0.0
        result["trade_winrate_pct"] = round(result["trade_profitable"] / trade_done * 100.0, 2) if trade_done else 0.0
        result["trade_pnl_eur"] = round(result["trade_pnl_eur"], 6)
        return result

    def _finalize_range_bucket(self, bucket: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(bucket)
        events = result["events"]
        breakouts = (
            result["range_breakout_up"]
            + result["range_breakout_down"]
            + result["range_volatile_breakout"]
        )
        result["range_breakout_rate_pct"] = round(breakouts / events * 100.0, 2) if events else 0.0
        return result

    def _build_attention_lists(self, report: Dict[str, Any]) -> Dict[str, Any]:
        by_symbol = report["by_symbol"]
        skip_summary = report["skip_summary"]
        trade_summary = report["trade_open_summary"]
        range_summary = report["range_summary"]

        return {
            "symbols_most_missed_skips": self._top_items(by_symbol, "missed_opportunity"),
            "symbols_worst_trade_pnl": self._top_items(trade_summary, "trade_pnl_eur", reverse=False),
            "skip_reasons_most_missed": self._top_items(skip_summary, "missed_opportunity"),
            "skip_reasons_best_protection": self._top_items(skip_summary, "skip_protected"),
            "range_symbols_most_breakouts": self._top_items(range_summary, "range_breakout_rate_pct"),
        }

    def _top_items(self, mapping: Dict[str, Dict[str, Any]], metric: str, reverse: bool = True) -> list[Dict[str, Any]]:
        rows = [
            {"name": name, metric: values.get(metric, 0), "events": values.get("events", 0)}
            for name, values in mapping.items()
        ]
        rows.sort(key=lambda row: row[metric], reverse=reverse)
        if metric == "trade_pnl_eur" and not reverse:
            return [row for row in rows if row[metric] < 0][:10]
        return [row for row in rows if row[metric] != 0][:10]

    def _parse_outcome(self, raw: Any) -> Dict[str, Any]:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            return {}


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize labeled strategy_events.")
    parser.add_argument("--limit", type=int, default=5000, help="Max labeled events to read.")
    parser.add_argument("--symbol", type=str, default=None, help="Optional symbol filter, e.g. XBT-EUR.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    reporter = StrategyEventReporter()
    report = reporter.build_report(limit=args.limit, symbol=args.symbol)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
