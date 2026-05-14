# ============================================================
# src/analysis/ml_training_dataset.py
# ============================================================
"""
Build an ML-ready dataset from labeled strategy_events.

This module is deliberately read-only for the trading system:
- it reads labeled decisions and outcomes;
- flattens stable strategy/GPT/chart/profile fields;
- writes JSONL rows for later model training and shadow evaluation.

No live trading behavior is changed here.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "ml_training")
DEFAULT_LATEST_JSONL = "latest_strategy_event_dataset.jsonl"
DEFAULT_LATEST_SUMMARY = "latest_strategy_event_dataset_summary.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
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


class MlTrainingDatasetBuilder:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager(db_path=DB_FILE)

    def build_dataset(
        self,
        limit: int = 10000,
        hours: Optional[int] = None,
        event_type: Optional[str] = None,
        structured_only: bool = False,
    ) -> dict:
        events = self._load_events(
            limit=limit,
            hours=hours,
            event_type=event_type,
            structured_only=structured_only,
        )
        rows = [self._event_to_row(event) for event in events]
        summary = self._summarize(rows)
        return {
            "meta": {
                "loaded_events": len(events),
                "rows": len(rows),
                "limit": limit,
                "hours": hours,
                "event_type": event_type,
                "structured_only": structured_only,
            },
            "summary": summary,
            "rows": rows,
        }

    def _load_events(
        self,
        limit: int,
        hours: Optional[int],
        event_type: Optional[str],
        structured_only: bool,
    ) -> list[dict]:
        params: list[Any] = []
        where = [
            "outcome_status='labeled'",
            "features_json IS NOT NULL",
        ]
        if event_type:
            where.append("event_type=?")
            params.append(event_type)
        if structured_only:
            where.append("json_extract(features_json,'$.scores.entry') IS NOT NULL")
        if hours:
            where.append("timestamp >= (strftime('%s','now') * 1000) - ?")
            params.append(int(hours) * 3600 * 1000)

        params.append(int(limit))
        rows = self.db.execute_query(
            f"""
            SELECT
              id, timestamp, symbol, strategy_name, strategy_version,
              event_type, decision_stage, skip_reason, trend_dir,
              algo_signal, gpt_action, gpt_confidence,
              trade_id, coin_profile_bias, coin_profile_risk,
              coin_profile_expectancy, coin_profile_n_trades,
              features_json, outcome_json
            FROM strategy_events
            WHERE {" AND ".join(where)}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            tuple(params),
        )
        cols = [
            "id", "timestamp", "symbol", "strategy_name", "strategy_version",
            "event_type", "decision_stage", "skip_reason", "trend_dir",
            "algo_signal", "gpt_action", "gpt_confidence",
            "trade_id", "coin_profile_bias", "coin_profile_risk",
            "coin_profile_expectancy", "coin_profile_n_trades",
            "features_json", "outcome_json",
        ]
        return [dict(zip(cols, row)) for row in rows] if rows else []

    def _event_to_row(self, event: dict) -> dict:
        features = _parse_json(event.get("features_json"))
        outcome = _parse_json(event.get("outcome_json"))
        chart_1h = ((features.get("chart_features") or {}).get("1h") or {})
        chart_4h = ((features.get("chart_features") or {}).get("4h") or {})
        regime = features.get("market_regime") or {}
        profile = features.get("coin_profile") or {}
        scores = features.get("scores") or {}
        counterfactual = outcome.get("counterfactual_trade") or {}
        realized_trade = outcome.get("realized_trade") or {}

        cf_r = _safe_float(counterfactual.get("r_multiple"))
        realized_pnl = _safe_float(realized_trade.get("pnl_eur"))
        outcome_label = outcome.get("label")
        cf_label = counterfactual.get("label")
        direction = _direction_for(event.get("algo_signal"), event.get("trend_dir"))
        action = event.get("gpt_action")

        targets = {
            "cf_r": round(cf_r, 6),
            "cf_positive": cf_r > 0,
            "cf_large_positive": cf_r >= 1.0,
            "cf_loss": cf_label == "cf_loss" or cf_r <= -0.5,
            "missed_opportunity": outcome_label == "missed_opportunity",
            "skip_protected": outcome_label == "skip_protected",
            "opened_trade": action in ("OPEN_LONG", "OPEN_SHORT"),
            "realized_pnl_eur": round(realized_pnl, 6),
            "realized_profitable": realized_trade.get("label") == "trade_profitable",
        }

        return {
            "id": event.get("id"),
            "timestamp": event.get("timestamp"),
            "symbol": event.get("symbol"),
            "strategy_name": event.get("strategy_name"),
            "strategy_version": event.get("strategy_version"),
            "event_type": event.get("event_type"),
            "decision_stage": event.get("decision_stage"),
            "direction": direction,
            "algo_signal": event.get("algo_signal"),
            "gpt_action": action,
            "gpt_confidence": _safe_float(event.get("gpt_confidence")),
            "primary_veto": features.get("primary_veto") or event.get("skip_reason"),
            "learning_effect": features.get("learning_effect"),
            "outcome_label": outcome_label,
            "counterfactual_label": cf_label,
            "targets": targets,
            "features": {
                "scores": {
                    "trend": _safe_int(scores.get("trend")),
                    "entry": _safe_int(scores.get("entry")),
                    "risk": _safe_int(scores.get("risk")),
                    "learning": _safe_int(scores.get("learning")),
                    "sentiment": _safe_int(scores.get("sentiment")),
                },
                "market_regime": {
                    "regime": regime.get("regime"),
                    "risk_mode": regime.get("risk_mode"),
                    "directional_bias": regime.get("directional_bias"),
                    "risk_multiplier": _safe_float(regime.get("risk_multiplier"), 1.0),
                    "bull_pct": _safe_float((regime.get("breadth") or {}).get("bull_pct")),
                    "bear_pct": _safe_float((regime.get("breadth") or {}).get("bear_pct")),
                    "range_pct": _safe_float((regime.get("breadth") or {}).get("range_pct")),
                    "flags": regime.get("flags", []),
                },
                "coin_profile": {
                    "source": profile.get("source"),
                    "learning_confidence": profile.get("learning_confidence"),
                    "risk_multiplier": _safe_float(profile.get("risk_multiplier"), 1.0),
                    "bias": profile.get("bias"),
                    "n_trades": _safe_int(profile.get("n_trades")),
                    "expectancy_R": _safe_float(profile.get("expectancy_R")),
                    "flags": profile.get("flags", []),
                    "metrics": profile.get("learning_metrics", {}),
                },
                "chart_1h": self._flatten_chart(chart_1h),
                "chart_4h": self._flatten_chart(chart_4h),
            },
        }

    def _flatten_chart(self, chart: dict) -> dict:
        return {
            "structure_label": chart.get("structure_label"),
            "chop_subtype": chart.get("chop_subtype"),
            "entry_timing": chart.get("entry_timing"),
            "last_candle_quality": chart.get("last_candle_quality"),
            "directional_continuation": bool(chart.get("directional_continuation")),
            "continuation_pressure": _safe_int(chart.get("continuation_pressure")),
            "breakout_pressure": _safe_int(chart.get("breakout_pressure")),
            "breakdown_pressure": _safe_int(chart.get("breakdown_pressure")),
            "ema20_distance_pct": _safe_float(chart.get("ema20_distance_pct")),
            "ema50_distance_pct": _safe_float(chart.get("ema50_distance_pct")),
            "ema_spread_pct": _safe_float(chart.get("ema_spread_pct")),
            "atr_pct": _safe_float(chart.get("atr_pct")),
            "trend_age_bars": _safe_int(chart.get("trend_age_bars")),
            "pullback_depth_pct": _safe_float(chart.get("pullback_depth_pct")),
            "recent_doji_count": _safe_int(chart.get("recent_doji_count")),
            "recent_opposing_wick_count": _safe_int(chart.get("recent_opposing_wick_count")),
            "recent_directional_body_count": _safe_int(chart.get("recent_directional_body_count")),
            "recent_directional_close_count": _safe_int(chart.get("recent_directional_close_count")),
            "macd_hist": _safe_float(chart.get("macd_hist")),
            "macd_hist_slope": _safe_float(chart.get("macd_hist_slope")),
            "rsi": _safe_float(chart.get("rsi"), 50.0),
        }

    def _summarize(self, rows: list[dict]) -> dict:
        by_action = Counter(row.get("gpt_action") or "missing" for row in rows)
        by_direction = Counter(row.get("direction") or "missing" for row in rows)
        by_outcome = Counter(row.get("outcome_label") or "missing" for row in rows)
        by_veto = Counter(row.get("primary_veto") or "missing" for row in rows)
        by_regime = Counter(
            ((row.get("features") or {}).get("market_regime") or {}).get("regime") or "missing"
            for row in rows
        )
        by_chop_subtype = Counter(
            ((row.get("features") or {}).get("chart_1h") or {}).get("chop_subtype") or "missing"
            for row in rows
        )
        by_symbol = defaultdict(lambda: {"events": 0, "cf_total_r": 0.0, "cf_count": 0, "cf_large_positive": 0})

        for row in rows:
            target = row.get("targets") or {}
            symbol = row.get("symbol") or "UNKNOWN"
            by_symbol[symbol]["events"] += 1
            by_symbol[symbol]["cf_total_r"] += _safe_float(target.get("cf_r"))
            by_symbol[symbol]["cf_count"] += 1
            if target.get("cf_large_positive"):
                by_symbol[symbol]["cf_large_positive"] += 1

        return {
            "rows": len(rows),
            "by_action": dict(by_action),
            "by_direction": dict(by_direction),
            "by_outcome": dict(by_outcome),
            "by_veto": dict(by_veto.most_common(20)),
            "by_regime": dict(by_regime),
            "by_chop_subtype": dict(by_chop_subtype),
            "top_symbols_by_events": self._top_symbols(by_symbol, "events", reverse=True),
            "top_symbols_by_cf_avg_r": self._top_symbols(by_symbol, "cf_avg_r", reverse=True),
            "bottom_symbols_by_cf_avg_r": self._top_symbols(by_symbol, "cf_avg_r", reverse=False),
        }

    def _top_symbols(self, by_symbol: dict, metric: str, reverse: bool) -> list[dict]:
        rows = []
        for symbol, values in by_symbol.items():
            cf_count = _safe_int(values.get("cf_count"))
            cf_avg = round(_safe_float(values.get("cf_total_r")) / cf_count, 4) if cf_count else 0.0
            rows.append({
                "symbol": symbol,
                "events": _safe_int(values.get("events")),
                "cf_avg_r": cf_avg,
                "cf_large_positive": _safe_int(values.get("cf_large_positive")),
            })
        rows.sort(key=lambda row: row.get(metric, 0), reverse=reverse)
        return rows[:10]


def write_dataset(output_dir: str, payload: dict) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, DEFAULT_LATEST_JSONL)
    summary_path = os.path.join(output_dir, DEFAULT_LATEST_SUMMARY)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in payload.get("rows", []):
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "meta": payload.get("meta", {}),
        "summary": payload.get("summary", {}),
        "jsonl_path": jsonl_path,
        "summary_path": summary_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return jsonl_path, summary_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build ML-ready strategy event dataset.")
    parser.add_argument("--limit", type=int, default=10000, help="Max labeled events to read.")
    parser.add_argument("--hours", type=int, default=None, help="Optional lookback window in hours.")
    parser.add_argument("--event-type", type=str, default=None, help="Optional event_type filter, e.g. gpt_decision.")
    parser.add_argument("--structured-only", action="store_true", help="Only include rows with structured GPT scores.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    db = DatabaseManager(db_path=DB_FILE)
    try:
        payload = MlTrainingDatasetBuilder(db=db).build_dataset(
            limit=args.limit,
            hours=args.hours,
            event_type=args.event_type,
            structured_only=args.structured_only,
        )
    finally:
        db.close_connection()

    jsonl_path, summary_path = write_dataset(args.output_dir, payload)
    result = {
        "rows": payload.get("meta", {}).get("rows", 0),
        "jsonl_path": jsonl_path,
        "summary_path": summary_path,
        "by_action": payload.get("summary", {}).get("by_action", {}),
        "by_direction": payload.get("summary", {}).get("by_direction", {}),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
