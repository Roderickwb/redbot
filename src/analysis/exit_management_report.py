"""Read-only exit management report.

This report summarizes how positions are being closed from the existing trades
table. It does not change live trading behavior. The current trades schema does
not persist an explicit close reason, so TP1/close paths are inferred from child
trade rows and the report surfaces that as a data-quality limitation.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.config.config import DB_FILE


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "exits")
DEFAULT_LATEST_FILE = "latest_exit_management_report.json"
DEFAULT_STRATEGY_NAME = "trend_4h"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _ts_to_hours(start_ms: Any, end_ms: Any) -> Optional[float]:
    start = _safe_int(start_ms)
    end = _safe_int(end_ms)
    if not start or not end or end < start:
        return None
    return round((end - start) / 3_600_000, 3)


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class ExitManagementReport:
    def __init__(self, db_path: str = DB_FILE, strategy_name: str = DEFAULT_STRATEGY_NAME):
        self.db_path = db_path
        self.strategy_name = strategy_name

    def build_report(self, limit: Optional[int] = None) -> dict:
        masters, children, data_quality = self._load_trades(limit=limit)
        children_by_position: dict[str, list[dict]] = defaultdict(list)
        for child in children:
            position_id = str(child.get("position_id") or "")
            if position_id:
                children_by_position[position_id].append(child)

        positions = []
        for master in masters:
            position_id = str(master.get("position_id") or "")
            child_rows = sorted(children_by_position.get(position_id, []), key=lambda row: (_safe_int(row.get("timestamp")), _safe_int(row.get("id"))))
            positions.append(self._position_summary(master, child_rows))

        summary = self._summary(positions, data_quality)
        return {
            "created_utc": _utc_now(),
            "status": "OK" if positions else "NO_DATA",
            "meta": {
                "db_path": self.db_path,
                "strategy_name": self.strategy_name,
                "limit": limit,
                "read_only": True,
                "live_effect": False,
            },
            "summary": summary,
            "positions": positions[:200],
            "data_quality": data_quality,
        }

    def _load_trades(self, limit: Optional[int]) -> tuple[list[dict], list[dict], dict]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            cols = self._columns(con)
            required = {"id", "timestamp", "symbol", "side", "price", "amount", "position_id", "status", "pnl_eur", "fees", "strategy_name", "is_master"}
            missing = sorted(required - set(cols))
            if missing:
                return [], [], {
                    "reason_available": False,
                    "missing_columns": missing,
                    "child_rows_without_position_id": 0,
                    "masters_without_position_id": 0,
                    "close_reason_missing": 0,
                }

            master_sql = """
                SELECT id, timestamp, datetime_utc, symbol, side, price, amount,
                       position_id, position_type, status, pnl_eur, fees,
                       trade_cost, exchange, strategy_name, is_master
                  FROM trades
                 WHERE strategy_name=?
                   AND is_master=1
                 ORDER BY timestamp DESC, id DESC
            """
            params: list[Any] = [self.strategy_name]
            if limit:
                master_sql += " LIMIT ?"
                params.append(int(limit))
            masters = [dict(row) for row in con.execute(master_sql, params).fetchall()]

            child_sql = """
                SELECT id, timestamp, datetime_utc, symbol, side, price, amount,
                       position_id, position_type, status, pnl_eur, fees,
                       trade_cost, exchange, strategy_name, is_master
                  FROM trades
                 WHERE strategy_name=?
                   AND is_master=0
                 ORDER BY timestamp ASC, id ASC
            """
            children = [dict(row) for row in con.execute(child_sql, (self.strategy_name,)).fetchall()]

            data_quality = {
                "reason_available": "exit_reason" in cols or "reason" in cols,
                "missing_columns": [],
                "child_rows_without_position_id": sum(1 for row in children if not row.get("position_id")),
                "masters_without_position_id": sum(1 for row in masters if not row.get("position_id")),
                "close_reason_missing": sum(1 for row in children if str(row.get("status") or "").lower() in {"partial", "closed"}),
            }
            return masters, children, data_quality
        finally:
            con.close()

    @staticmethod
    def _columns(con: sqlite3.Connection) -> set[str]:
        rows = con.execute("PRAGMA table_info(trades)").fetchall()
        return {str(row[1]) for row in rows}

    def _position_summary(self, master: dict, children: list[dict]) -> dict:
        child_statuses = [str(row.get("status") or "").lower() for row in children]
        partials = [row for row in children if str(row.get("status") or "").lower() == "partial"]
        closes = [row for row in children if str(row.get("status") or "").lower() == "closed"]
        total_child_amount = sum(_safe_float(row.get("amount")) for row in children)
        master_amount = _safe_float(master.get("amount"))
        initial_amount_estimate = max(master_amount, total_child_amount)
        entry_price = _safe_float(master.get("price"))
        exposure = entry_price * initial_amount_estimate
        pnl = sum(_safe_float(row.get("pnl_eur")) for row in children)
        fees = sum(_safe_float(row.get("fees")) for row in children)
        last_child = children[-1] if children else None
        closed = bool(closes) or str(master.get("status") or "").lower() == "closed"
        path = self._exit_path(child_statuses, closed)
        hold_hours = _ts_to_hours(master.get("timestamp"), (last_child or {}).get("timestamp")) if last_child else None
        return {
            "position_id": master.get("position_id"),
            "master_trade_id": master.get("id"),
            "symbol": master.get("symbol"),
            "side": master.get("side"),
            "entry_ts": master.get("timestamp"),
            "entry_datetime_utc": master.get("datetime_utc"),
            "entry_price": _round(entry_price),
            "initial_amount_estimate": _round(initial_amount_estimate),
            "current_master_amount": _round(master_amount),
            "status": master.get("status"),
            "closed": closed,
            "exit_path": path,
            "exit_steps": len(children),
            "partial_exits": len(partials),
            "close_events": len(closes),
            "realized_pnl_eur": _round(pnl),
            "fees_eur": _round(fees),
            "roi_pct": _round((pnl / exposure) * 100.0, 4) if exposure else 0.0,
            "hold_hours": hold_hours,
            "first_exit_ts": children[0].get("timestamp") if children else None,
            "last_exit_ts": last_child.get("timestamp") if last_child else None,
            "last_exit_datetime_utc": last_child.get("datetime_utc") if last_child else None,
        }

    @staticmethod
    def _exit_path(child_statuses: list[str], closed: bool) -> str:
        partials = child_statuses.count("partial")
        closes = child_statuses.count("closed")
        if partials and closes:
            return "tp1_then_close_proxy"
        if partials and not closes and not closed:
            return "tp1_still_open_proxy"
        if partials:
            return "partial_only_proxy"
        if closes:
            return "direct_close_proxy"
        return "no_child_events"

    def _summary(self, positions: list[dict], data_quality: dict) -> dict:
        closed = [pos for pos in positions if pos.get("closed")]
        with_tp1 = [pos for pos in positions if _safe_int(pos.get("partial_exits")) > 0]
        total_pnl = sum(_safe_float(pos.get("realized_pnl_eur")) for pos in positions)
        wins = [pos for pos in closed if _safe_float(pos.get("realized_pnl_eur")) > 0]
        losses = [pos for pos in closed if _safe_float(pos.get("realized_pnl_eur")) < 0]
        hold_values = [_safe_float(pos.get("hold_hours")) for pos in closed if pos.get("hold_hours") is not None]
        by_path = Counter(str(pos.get("exit_path") or "unknown") for pos in positions)

        by_symbol = {}
        symbol_groups: dict[str, list[dict]] = defaultdict(list)
        for pos in positions:
            symbol_groups[str(pos.get("symbol") or "UNKNOWN")].append(pos)
        for symbol, rows in sorted(symbol_groups.items()):
            closed_rows = [row for row in rows if row.get("closed")]
            by_symbol[symbol] = {
                "positions": len(rows),
                "closed": len(closed_rows),
                "tp1_proxy": sum(1 for row in rows if _safe_int(row.get("partial_exits")) > 0),
                "realized_pnl_eur": _round(sum(_safe_float(row.get("realized_pnl_eur")) for row in rows)),
                "win_rate_pct": _round((sum(1 for row in closed_rows if _safe_float(row.get("realized_pnl_eur")) > 0) / len(closed_rows)) * 100.0, 2) if closed_rows else 0.0,
            }

        verdict = "collect_more_exit_data"
        if closed and not data_quality.get("reason_available"):
            verdict = "exit_logging_needs_reason_field"
        if len(closed) >= 20 and data_quality.get("reason_available"):
            verdict = "exit_data_ready_for_review"

        ranked = sorted(positions, key=lambda row: _safe_float(row.get("realized_pnl_eur")), reverse=True)
        return {
            "positions_loaded": len(positions),
            "closed_positions": len(closed),
            "open_or_partial_positions": len(positions) - len(closed),
            "positions_with_tp1_proxy": len(with_tp1),
            "tp1_proxy_rate_pct": _round((len(with_tp1) / len(positions)) * 100.0, 2) if positions else 0.0,
            "total_realized_pnl_eur": _round(total_pnl),
            "avg_realized_pnl_eur": _round(total_pnl / len(closed)) if closed else 0.0,
            "winning_closed_positions": len(wins),
            "losing_closed_positions": len(losses),
            "win_rate_pct": _round((len(wins) / len(closed)) * 100.0, 2) if closed else 0.0,
            "avg_hold_hours_closed": _round(sum(hold_values) / len(hold_values), 3) if hold_values else 0.0,
            "by_exit_path": dict(by_path),
            "by_symbol": by_symbol,
            "top_winners": ranked[:5],
            "top_losers": list(reversed(ranked[-5:])) if ranked else [],
            "verdict": verdict,
            "reason_available": bool(data_quality.get("reason_available")),
        }


def run_exit_management_report(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    db_path: str = DB_FILE,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
    limit: Optional[int] = None,
) -> dict:
    report = ExitManagementReport(db_path=db_path, strategy_name=strategy_name).build_report(limit=limit)
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    _write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build read-only exit management report.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db-path", type=str, default=DB_FILE)
    parser.add_argument("--strategy-name", type=str, default=DEFAULT_STRATEGY_NAME)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_exit_management_report(
        output_dir=args.output_dir,
        db_path=args.db_path,
        strategy_name=args.strategy_name,
        limit=args.limit,
    )
    print(json.dumps({
        "status": report.get("status"),
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
