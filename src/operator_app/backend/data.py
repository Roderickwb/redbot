from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from src.config.config import DB_FILE


REPORTS = {
    "snapshot": os.path.join("analysis", "operator_app", "latest_operator_app_snapshot.json"),
    "cockpit": os.path.join("analysis", "operator_cockpit", "latest_operator_cockpit.json"),
    "daily_control": os.path.join("analysis", "daily_control", "latest_daily_control_report.json"),
    "recommendations": os.path.join("analysis", "recommendations", "latest_recommendation_aggregator.json"),
    "recommendation_quality": os.path.join("analysis", "recommendations", "latest_recommendation_quality_report.json"),
    "operator_decisions": os.path.join("analysis", "operator_decisions", "latest_operator_decisions.json"),
    "safety": os.path.join("analysis", "safety", "latest_safety_control_report.json"),
    "positions": os.path.join("analysis", "positions", "latest_position_lifecycle_report.json"),
    "exits": os.path.join("analysis", "exits", "latest_exit_management_report.json"),
}


def load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default if default is not None else {"_missing": True, "_path": path}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e), "_path": path}


def report(name: str) -> dict:
    return load_json(REPORTS[name], {})


def mobile_bundle(trade_limit: int = 40) -> dict:
    trades = recent_trades(limit=trade_limit)
    return {
        "status": "OK",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "snapshot": report("snapshot"),
        "cockpit": report("cockpit"),
        "recommendations": report("recommendations"),
        "recommendation_quality": report("recommendation_quality"),
        "operator_decisions": report("operator_decisions"),
        "safety": report("safety"),
        "positions": report("positions"),
        "exits": report("exits"),
        "trades": trades,
        "live_effect": False,
    }


def recent_trades(limit: int = 100, symbol: Optional[str] = None, strategy_name: Optional[str] = None) -> dict:
    limit = max(1, min(int(limit or 100), 500))
    where: list[str] = []
    params: list[Any] = []
    if strategy_name:
        where.append("strategy_name=?")
        params.append(strategy_name)
    if symbol:
        where.append("symbol=?")
        params.append(symbol)
    params.append(limit)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    sql = f"""
        SELECT id, timestamp, datetime_utc, symbol, side, price, amount,
               position_id, position_type, status, pnl_eur, fees, trade_cost,
               exchange, strategy_name, is_master, exit_reason, exit_event_type
          FROM trades
         {where_sql}
         ORDER BY timestamp DESC, id DESC
         LIMIT ?
    """
    fallback_sql = f"""
        SELECT id, timestamp, datetime_utc, symbol, side, price, amount,
               position_id, position_type, status, pnl_eur, fees, trade_cost,
               exchange, strategy_name, is_master
          FROM trades
         {where_sql}
         ORDER BY timestamp DESC, id DESC
         LIMIT ?
    """
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    error = ""
    try:
        rows = [dict(row) for row in con.execute(sql, params).fetchall()]
    except sqlite3.OperationalError as e:
        error = str(e)
        try:
            rows = [dict(row) for row in con.execute(fallback_sql, params).fetchall()]
            for row in rows:
                row.setdefault("exit_reason", "")
                row.setdefault("exit_event_type", "")
        except sqlite3.OperationalError as e2:
            error = f"{error}; fallback: {e2}"
            rows = []
    finally:
        con.close()
    return {
        "status": "OK",
        "limit": limit,
        "symbol": symbol,
        "strategy_name": strategy_name,
        "row_count": len(rows),
        "rows": rows,
        "warning": error,
        "live_effect": False,
    }
