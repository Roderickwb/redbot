# ============================================================
# src/analysis/strategy_event_outcome_labeler.py
# ============================================================
"""
Outcome labeling for strategy_events.

This job turns raw strategy events into learning cases by looking at what
happened after the event in candles_kraken.

Default mode is dry-run. Use --apply to write outcome_status/outcome_json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from src.config.config import DB_FILE, yaml_config
from src.database_manager.database_manager import DatabaseManager

logger = logging.getLogger("strategy_event_outcome_labeler")


@dataclass
class OutcomeConfig:
    interval: str = "5m"
    lookahead_hours: float = 8.0
    move_threshold_pct: float = 2.0
    adverse_threshold_pct: float = 2.0
    max_events: int = 1000


def _analysis_cfg() -> OutcomeConfig:
    cfg = yaml_config.get("analysis", {}) or {}
    return OutcomeConfig(
        interval=str(cfg.get("hold_eval_interval", "5m")),
        lookahead_hours=float(cfg.get("hold_lookahead_hours", 8.0)),
        move_threshold_pct=float(cfg.get("hold_missed_move_pct", 2.0)),
        adverse_threshold_pct=float(cfg.get("hold_missed_move_pct", 2.0)),
        max_events=int(cfg.get("outcome_label_max_events", 1000)),
    )


class StrategyEventOutcomeLabeler:
    def __init__(self, db: Optional[DatabaseManager] = None, config: Optional[OutcomeConfig] = None):
        self.db = db or DatabaseManager(db_path=DB_FILE)
        self.config = config or _analysis_cfg()

    def label_pending_events(self, apply: bool = False, limit: Optional[int] = None) -> Dict[str, int]:
        limit = int(limit or self.config.max_events)
        events = self._load_pending_events(limit=limit)

        stats = {
            "loaded": len(events),
            "labeled": 0,
            "waiting_for_candles": 0,
            "no_candles": 0,
            "no_direction": 0,
            "updated": 0,
        }

        for event in events:
            outcome = self._build_outcome(event)
            status = outcome["status"]
            if status == "waiting_for_candles":
                stats["waiting_for_candles"] += 1
                continue
            if status == "no_candles":
                stats["no_candles"] += 1
                if apply:
                    self._update_event_outcome(event["id"], status="no_candles", outcome=outcome)
                    stats["updated"] += 1
                continue
            if status == "no_direction":
                stats["no_direction"] += 1

            stats["labeled"] += 1
            if apply:
                self._update_event_outcome(event["id"], status="labeled", outcome=outcome)
                stats["updated"] += 1

        logger.info("[strategy_event_outcomes] %s", stats)
        return stats

    def _load_pending_events(self, limit: int) -> list[Dict[str, Any]]:
        cutoff_ts = int(time.time() * 1000) - int(self.config.lookahead_hours * 3600 * 1000)
        rows = self.db.execute_query(
            """
            SELECT
                id, timestamp, symbol, strategy_name, event_type, decision_stage,
                skip_reason, trend_dir, price, algo_signal, gpt_action, trade_id
            FROM strategy_events
            WHERE COALESCE(outcome_status, 'pending') = 'pending'
              AND timestamp <= ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (cutoff_ts, limit),
        )
        cols = [
            "id", "timestamp", "symbol", "strategy_name", "event_type", "decision_stage",
            "skip_reason", "trend_dir", "price", "algo_signal", "gpt_action", "trade_id",
        ]
        return [dict(zip(cols, row)) for row in rows] if rows else []

    def _build_outcome(self, event: Dict[str, Any]) -> Dict[str, Any]:
        direction = self._infer_direction(event)
        start_ts = int(event["timestamp"])
        end_ts = start_ts + int(self.config.lookahead_hours * 3600 * 1000)

        candles = self._fetch_candles(event["symbol"], start_ts, end_ts)
        if not candles:
            return self._base_outcome(event, direction, status="no_candles")

        latest_candle_ts = int(candles[-1]["timestamp"])
        min_required_ts = end_ts - self._interval_ms()
        if latest_candle_ts < min_required_ts:
            return self._base_outcome(
                event,
                direction,
                status="waiting_for_candles",
                extra={
                    "latest_candle_ts": latest_candle_ts,
                    "required_end_ts": end_ts,
                    "min_required_ts": min_required_ts,
                },
            )

        entry_price = self._entry_price(event, candles)
        if entry_price <= 0:
            return self._base_outcome(event, direction, status="invalid_entry_price")

        high_after = max(float(c["high"]) for c in candles)
        low_after = min(float(c["low"]) for c in candles)
        close_after = float(candles[-1]["close"])

        up_pct = (high_after - entry_price) / entry_price * 100.0
        down_pct = (entry_price - low_after) / entry_price * 100.0
        close_pct = (close_after - entry_price) / entry_price * 100.0

        if direction == "long":
            favorable_pct = up_pct
            adverse_pct = down_pct
        elif direction == "short":
            favorable_pct = down_pct
            adverse_pct = up_pct
        else:
            favorable_pct = max(up_pct, down_pct)
            adverse_pct = min(up_pct, down_pct)

        event_for_label = dict(event)
        event_for_label["_up_pct"] = up_pct
        event_for_label["_down_pct"] = down_pct
        label = self._label_event(event_for_label, direction, favorable_pct, adverse_pct)
        outcome = self._base_outcome(event, direction, status="labeled")
        outcome.update({
            "label": label,
            "interval": self.config.interval,
            "lookahead_hours": self.config.lookahead_hours,
            "move_threshold_pct": self.config.move_threshold_pct,
            "entry_price": round(entry_price, 10),
            "high_after": round(high_after, 10),
            "low_after": round(low_after, 10),
            "close_after": round(close_after, 10),
            "up_pct": round(up_pct, 4),
            "down_pct": round(down_pct, 4),
            "close_pct": round(close_pct, 4),
            "favorable_pct": round(favorable_pct, 4),
            "adverse_pct": round(adverse_pct, 4),
            "n_candles": len(candles),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "first_candle_ts": int(candles[0]["timestamp"]),
            "last_candle_ts": latest_candle_ts,
        })

        if event.get("event_type") == "trade_open" and event.get("trade_id"):
            outcome["realized_trade"] = self._load_realized_trade_outcome(int(event["trade_id"]))

        return outcome

    def _label_event(self, event: Dict[str, Any], direction: Optional[str], favorable_pct: float, adverse_pct: float) -> str:
        event_type = event.get("event_type")
        skip_reason = event.get("skip_reason")

        up_hit = event.get("_up_pct", 0.0) >= self.config.move_threshold_pct
        down_hit = event.get("_down_pct", 0.0) >= self.config.move_threshold_pct

        if direction is None:
            if event.get("skip_reason") == "trend_range":
                if up_hit and down_hit:
                    return "range_volatile_breakout"
                if up_hit:
                    return "range_breakout_up"
                if down_hit:
                    return "range_breakout_down"
                return "range_no_breakout"
            if favorable_pct >= self.config.move_threshold_pct:
                return "directionless_large_move_after_event"
            return "directionless_no_large_move"

        favorable_hit = favorable_pct >= self.config.move_threshold_pct
        adverse_hit = adverse_pct >= self.config.adverse_threshold_pct

        if event_type == "trade_open":
            if favorable_hit and not adverse_hit:
                return "open_followed_through"
            if adverse_hit and not favorable_hit:
                return "open_went_against"
            if favorable_hit and adverse_hit:
                return "open_mixed_volatility"
            return "open_no_followthrough"

        if skip_reason == "gpt_hold" or event_type == "skip":
            if favorable_hit and not adverse_hit:
                return "missed_opportunity"
            if adverse_hit and not favorable_hit:
                return "skip_protected"
            if favorable_hit and adverse_hit:
                return "volatile_after_skip"
            return "skip_correct_no_move"

        if event_type == "gpt_decision":
            if favorable_hit:
                return "gpt_direction_followed_through"
            if adverse_hit:
                return "gpt_direction_failed"
            return "gpt_no_followthrough"

        return "labeled"

    def _infer_direction(self, event: Dict[str, Any]) -> Optional[str]:
        gpt_action = event.get("gpt_action")
        if gpt_action == "OPEN_LONG":
            return "long"
        if gpt_action == "OPEN_SHORT":
            return "short"

        algo_signal = event.get("algo_signal")
        if algo_signal == "long_candidate":
            return "long"
        if algo_signal == "short_candidate":
            return "short"

        trend_dir = event.get("trend_dir")
        if trend_dir == "bull":
            return "long"
        if trend_dir == "bear":
            return "short"

        return None

    def _entry_price(self, event: Dict[str, Any], candles: list[Dict[str, Any]]) -> float:
        price = event.get("price")
        try:
            if price is not None and float(price) > 0:
                return float(price)
        except Exception:
            pass
        return float(candles[0]["close"])

    def _load_realized_trade_outcome(self, trade_id: int) -> Dict[str, Any]:
        rows = self.db.execute_query(
            """
            SELECT id, timestamp, symbol, side, price, amount, position_id,
                   position_type, status, pnl_eur, fees, trade_cost
            FROM trades
            WHERE id = ? AND is_master = 1
            LIMIT 1
            """,
            (trade_id,),
        )
        if not rows:
            return {"status": "master_not_found", "trade_id": trade_id}

        cols = [
            "id", "timestamp", "symbol", "side", "price", "amount", "position_id",
            "position_type", "status", "pnl_eur", "fees", "trade_cost",
        ]
        master = dict(zip(cols, rows[0]))
        trade_status = master.get("status")
        pnl_eur = float(master.get("pnl_eur") or 0.0)
        fees = float(master.get("fees") or 0.0)
        trade_cost = float(master.get("trade_cost") or 0.0)

        child_rows = self.db.execute_query(
            """
            SELECT id, timestamp, side, price, amount, status, pnl_eur, fees, trade_cost
            FROM trades
            WHERE position_id = ? AND is_master = 0
            ORDER BY timestamp ASC
            """,
            (master.get("position_id"),),
        )
        child_cols = ["id", "timestamp", "side", "price", "amount", "status", "pnl_eur", "fees", "trade_cost"]
        children = [dict(zip(child_cols, row)) for row in child_rows] if child_rows else []

        if trade_status != "closed":
            realized_label = "trade_still_open"
        elif pnl_eur > 0:
            realized_label = "trade_profitable"
        elif pnl_eur < 0:
            realized_label = "trade_losing"
        else:
            realized_label = "trade_breakeven"

        roi_pct = (pnl_eur / trade_cost * 100.0) if trade_cost > 0 else None
        return {
            "status": trade_status,
            "label": realized_label,
            "trade_id": trade_id,
            "position_id": master.get("position_id"),
            "pnl_eur": round(pnl_eur, 6),
            "fees": round(fees, 6),
            "trade_cost": round(trade_cost, 6),
            "roi_pct": round(roi_pct, 4) if roi_pct is not None else None,
            "child_count": len(children),
            "children": children,
        }

    def _interval_ms(self) -> int:
        intervals = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
        }
        return intervals.get(self.config.interval, 5 * 60 * 1000)

    def _fetch_candles(self, symbol: str, start_ts: int, end_ts: int) -> list[Dict[str, Any]]:
        rows = self.db.execute_query(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM candles_kraken
            WHERE market = ?
              AND interval = ?
              AND timestamp >= ?
              AND timestamp <= ?
            ORDER BY timestamp ASC
            """,
            (symbol, self.config.interval, start_ts, end_ts),
        )
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        return [dict(zip(cols, row)) for row in rows] if rows else []

    def _base_outcome(
        self,
        event: Dict[str, Any],
        direction: Optional[str],
        status: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = {
            "status": status,
            "event_id": event.get("id"),
            "symbol": event.get("symbol"),
            "event_type": event.get("event_type"),
            "decision_stage": event.get("decision_stage"),
            "skip_reason": event.get("skip_reason"),
            "direction": direction,
        }
        if extra:
            data.update(extra)
        return data

    def _update_event_outcome(self, event_id: int, status: str, outcome: Dict[str, Any]) -> None:
        self.db.execute_query(
            """
            UPDATE strategy_events
            SET outcome_status = ?, outcome_json = ?
            WHERE id = ?
            """,
            (status, json.dumps(outcome, ensure_ascii=False), int(event_id)),
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Label outcomes for pending strategy_events.")
    parser.add_argument("--apply", action="store_true", help="Write labels to strategy_events.")
    parser.add_argument("--limit", type=int, default=None, help="Max events to process.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    labeler = StrategyEventOutcomeLabeler()
    stats = labeler.label_pending_events(apply=args.apply, limit=args.limit)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
