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
    sl_atr_mult: float = 1.3
    tp1_atr_mult: float = 2.0
    tp1_portion_pct: float = 0.5
    trailing_atr_mult: float = 0.9
    breakeven_after_tp1: bool = True


def _analysis_cfg() -> OutcomeConfig:
    cfg = yaml_config.get("analysis", {}) or {}
    strategy_cfg = yaml_config.get("trend_strategy_4h", {}) or {}
    return OutcomeConfig(
        interval=str(cfg.get("hold_eval_interval", "5m")),
        lookahead_hours=float(cfg.get("hold_lookahead_hours", 8.0)),
        move_threshold_pct=float(cfg.get("hold_missed_move_pct", 2.0)),
        adverse_threshold_pct=float(cfg.get("hold_missed_move_pct", 2.0)),
        max_events=int(cfg.get("outcome_label_max_events", 1000)),
        sl_atr_mult=float(strategy_cfg.get("sl_atr_mult", 1.3)),
        tp1_atr_mult=float(strategy_cfg.get("tp1_atr_mult", 2.0)),
        tp1_portion_pct=float(strategy_cfg.get("tp1_portion_pct", 0.5)),
        trailing_atr_mult=float(strategy_cfg.get("trailing_atr_mult", 0.9)),
        breakeven_after_tp1=bool(strategy_cfg.get("breakeven_after_tp1", True)),
    )


class StrategyEventOutcomeLabeler:
    def __init__(self, db: Optional[DatabaseManager] = None, config: Optional[OutcomeConfig] = None):
        self.db = db or DatabaseManager(db_path=DB_FILE)
        self.config = config or _analysis_cfg()

    def label_pending_events(
        self,
        apply: bool = False,
        limit: Optional[int] = None,
        relabel: bool = False,
    ) -> Dict[str, int]:
        limit = int(limit or self.config.max_events)
        events = self._load_events(limit=limit, relabel=relabel)

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

    def _load_events(self, limit: int, relabel: bool = False) -> list[Dict[str, Any]]:
        cutoff_ts = int(time.time() * 1000) - int(self.config.lookahead_hours * 3600 * 1000)
        status_clause = "1 = 1" if relabel else "COALESCE(outcome_status, 'pending') = 'pending'"
        rows = self.db.execute_query(
            f"""
            SELECT
                id, timestamp, symbol, strategy_name, event_type, decision_stage,
                skip_reason, trend_dir, price, algo_signal, gpt_action, trade_id, atr_1h
            FROM strategy_events
            WHERE {status_clause}
              AND timestamp <= ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (cutoff_ts, limit),
        )
        cols = [
            "id", "timestamp", "symbol", "strategy_name", "event_type", "decision_stage",
            "skip_reason", "trend_dir", "price", "algo_signal", "gpt_action", "trade_id", "atr_1h",
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

        counterfactual = self._simulate_counterfactual_trade(
            direction=direction,
            entry_price=entry_price,
            atr_value=event.get("atr_1h"),
            candles=candles,
        )
        if counterfactual:
            outcome["counterfactual_trade"] = counterfactual

        if event.get("event_type") == "trade_open" and event.get("trade_id"):
            outcome["realized_trade"] = self._load_realized_trade_outcome(int(event["trade_id"]))

        return outcome

    def _simulate_counterfactual_trade(
        self,
        direction: Optional[str],
        entry_price: float,
        atr_value: Any,
        candles: list[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if direction not in ("long", "short"):
            return None

        try:
            atr = float(atr_value)
        except Exception:
            return {"status": "no_atr", "direction": direction}

        if entry_price <= 0 or atr <= 0:
            return {"status": "invalid_risk_input", "direction": direction}

        risk_per_unit = atr * self.config.sl_atr_mult
        if risk_per_unit <= 0:
            return {"status": "invalid_risk_input", "direction": direction}

        tp1_portion = min(max(float(self.config.tp1_portion_pct), 0.0), 1.0)
        remaining_portion = 1.0
        realized_r = 0.0
        tp1_hit = False
        tp1_hit_ts: Optional[int] = None
        trail_active = False
        breakeven_applied = False
        exit_price: Optional[float] = None
        exit_reason: Optional[str] = None

        if direction == "long":
            stop = entry_price - risk_per_unit
            tp1 = entry_price + (atr * self.config.tp1_atr_mult)
            trail_ref = entry_price
        else:
            stop = entry_price + risk_per_unit
            tp1 = entry_price - (atr * self.config.tp1_atr_mult)
            trail_ref = entry_price

        max_favorable_r = 0.0
        max_adverse_r = 0.0
        ambiguous = False
        final_ts = int(candles[-1]["timestamp"])

        for candle in candles:
            high = float(candle["high"])
            low = float(candle["low"])
            close = float(candle["close"])
            candle_ts = int(candle["timestamp"])
            final_ts = candle_ts

            if direction == "long":
                max_favorable_r = max(max_favorable_r, (high - entry_price) / risk_per_unit)
                max_adverse_r = max(max_adverse_r, (entry_price - low) / risk_per_unit)
            else:
                max_favorable_r = max(max_favorable_r, (entry_price - low) / risk_per_unit)
                max_adverse_r = max(max_adverse_r, (high - entry_price) / risk_per_unit)

            if not tp1_hit:
                if direction == "long":
                    stop_hit = low <= stop
                    tp1_hit_now = high >= tp1
                else:
                    stop_hit = high >= stop
                    tp1_hit_now = low <= tp1

                if stop_hit and tp1_hit_now:
                    ambiguous = True
                    exit_price = stop
                    exit_reason = "AMBIGUOUS_SL_AND_TP1"
                    realized_r = -1.0
                    break

                if stop_hit:
                    exit_price = stop
                    exit_reason = "SL"
                    realized_r = -1.0
                    break

                if tp1_hit_now:
                    tp1_hit = True
                    tp1_hit_ts = candle_ts
                    trail_active = True
                    breakeven_applied = bool(self.config.breakeven_after_tp1)
                    tp1_r = abs(tp1 - entry_price) / risk_per_unit
                    realized_r += tp1_portion * tp1_r
                    remaining_portion = 1.0 - tp1_portion
                    trail_ref = high if direction == "long" else low

            if trail_active and remaining_portion > 0 and candle_ts != tp1_hit_ts:
                if direction == "long":
                    trail_ref = max(trail_ref, high)
                    trailing_stop = trail_ref - (atr * self.config.trailing_atr_mult)
                    if breakeven_applied:
                        trailing_stop = max(trailing_stop, entry_price)
                    if low <= trailing_stop:
                        exit_price = trailing_stop
                        exit_reason = "TRAILING_STOP"
                        realized_r += remaining_portion * ((trailing_stop - entry_price) / risk_per_unit)
                        break
                else:
                    trail_ref = min(trail_ref, low)
                    trailing_stop = trail_ref + (atr * self.config.trailing_atr_mult)
                    if breakeven_applied:
                        trailing_stop = min(trailing_stop, entry_price)
                    if high >= trailing_stop:
                        exit_price = trailing_stop
                        exit_reason = "TRAILING_STOP"
                        realized_r += remaining_portion * ((entry_price - trailing_stop) / risk_per_unit)
                        break

            final_close = close
            final_ts = candle_ts
        else:
            final_close = float(candles[-1]["close"])
            final_ts = int(candles[-1]["timestamp"])

        if exit_reason is None:
            exit_price = final_close
            exit_reason = "OPEN_END"
            if direction == "long":
                unrealized_r = (final_close - entry_price) / risk_per_unit
            else:
                unrealized_r = (entry_price - final_close) / risk_per_unit
            realized_r += remaining_portion * unrealized_r

        label = self._counterfactual_label(realized_r, tp1_hit, exit_reason, ambiguous)
        return {
            "status": "ambiguous_intrabar" if ambiguous else "simulated",
            "label": label,
            "direction": direction,
            "entry_price": round(entry_price, 10),
            "atr": round(atr, 10),
            "risk_per_unit": round(risk_per_unit, 10),
            "sl_atr_mult": self.config.sl_atr_mult,
            "tp1_atr_mult": self.config.tp1_atr_mult,
            "tp1_portion_pct": self.config.tp1_portion_pct,
            "trailing_atr_mult": self.config.trailing_atr_mult,
            "breakeven_after_tp1": self.config.breakeven_after_tp1,
            "stop_price": round(stop, 10),
            "tp1_price": round(tp1, 10),
            "tp1_hit": tp1_hit,
            "exit_reason": exit_reason,
            "exit_price": round(float(exit_price), 10) if exit_price is not None else None,
            "r_multiple": round(realized_r, 4),
            "max_favorable_r": round(max_favorable_r, 4),
            "max_adverse_r": round(max_adverse_r, 4),
            "bars": len(candles),
            "exit_ts": final_ts,
        }

    def _counterfactual_label(self, r_multiple: float, tp1_hit: bool, exit_reason: str, ambiguous: bool) -> str:
        if ambiguous:
            return "cf_ambiguous_intrabar"
        if exit_reason == "SL":
            return "cf_loss"
        if tp1_hit and r_multiple > 0:
            return "cf_tp1_then_positive"
        if r_multiple >= 1.0:
            return "cf_win"
        if r_multiple > 0:
            return "cf_small_win"
        if r_multiple == 0:
            return "cf_breakeven"
        return "cf_loss"

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
    parser.add_argument("--relabel", action="store_true", help="Rebuild existing labeled outcomes too.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    labeler = StrategyEventOutcomeLabeler()
    stats = labeler.label_pending_events(apply=args.apply, limit=args.limit, relabel=args.relabel)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
