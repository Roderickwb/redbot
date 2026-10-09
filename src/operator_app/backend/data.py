from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from src.config.config import DB_FILE, yaml_config
from src.analysis.trade_accounting import position_accounting


REPORTS = {
    "snapshot": os.path.join("analysis", "operator_app", "latest_operator_app_snapshot.json"),
    "cockpit": os.path.join("analysis", "operator_cockpit", "latest_operator_cockpit.json"),
    "daily_control": os.path.join("analysis", "daily_control", "latest_daily_control_report.json"),
    "recommendations": os.path.join("analysis", "recommendations", "latest_recommendation_aggregator.json"),
    "adaptive_restrictions": os.path.join("analysis", "adaptive_restrictions", "latest_adaptive_restrictions.json"),
    "adaptive_restriction_outcomes": os.path.join("analysis", "adaptive_restrictions", "latest_adaptive_restriction_outcomes.json"),
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
    reports = {key: report(key) for key in REPORTS}
    recs = reports['recommendations']
    adaptive = reports['adaptive_restrictions']
    outcomes = reports['adaptive_restriction_outcomes']
    positions = recent_positions(limit=trade_limit)
    def summary(key):
        data = reports[key]
        return {'status': data.get('status'), 'created_utc': data.get('created_utc'),
                'summary': {k: v for k, v in (data.get('summary') or {}).items()
                            if not isinstance(v, (dict, list))}}
    item_fields = {'id', 'title', 'operator_title', 'status', 'candidate_type', 'effect_level',
                   'operator_question', 'operator_summary', 'operator_consequence',
                   'proposed_change', 'test_plan', 'learning_question', 'operator_evidence',
                   'operator_actions', 'allowed_actions_v1', 'source_path', 'recommended_action',
                   'current_phase', 'target_phase', 'returns_as', 'scope'}
    warnings = [key for key, data in reports.items() if not data or data.get('_error') or data.get('_missing')]
    return {
        "status": "DEGRADED" if warnings else "OK",
        "warnings": warnings,
        "trading_mode": (yaml_config.get('trend_strategy_4h') or {}).get('trading_mode', 'unknown'),
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "snapshot": summary('snapshot'),
        "cockpit": {k: v for k, v in reports['cockpit'].items() if k in {'status', 'created_utc', 'daily_decision', 'bot_health', 'learning', 'risk', 'safety', 'live_readiness', 'action_needed'}},
        "recommendations": {**summary('recommendations'), 'items': [{k: v for k, v in item.items() if k in item_fields} for item in recs.get('items', [])]},
        "adaptive_restrictions": {**summary('adaptive_restrictions'), 'active_source_ids': adaptive.get('active_source_ids', []),
                                  'restrictions': [{k: v for k, v in row.items() if k in {'restriction_id', 'source_item_id', 'scope', 'symbol', 'state', 'rule_id', 'risk_multiplier', 'reopen_criteria', 'review_after'}} for row in adaptive.get('restrictions', [])]},
        "adaptive_restriction_outcomes": {**summary('adaptive_restriction_outcomes'), 'restrictions': outcomes.get('restrictions', [])},
        "recommendation_quality": summary('recommendation_quality'),
        "operator_decisions": {**summary('operator_decisions'), 'recent': [{k: v for k, v in row.items() if k in {'action', 'source_id', 'created_utc', 'reason'}} for row in reports['operator_decisions'].get('recent', [])[:10]]},
        "safety": {k: v for k, v in reports['safety'].items() if k in {'status', 'reason', 'updated_utc', 'kill_switch_active', 'meltdown', 'live_entry_orders_allowed', 'live_enforcement_allowed'}},
        "positions": positions,
        "exits": summary('exits'),
        "trades": {'rows': [], 'warning': 'Use /api/trades for raw executions'},
        "live_effect": False,
    }


def recent_positions(limit: int = 40) -> dict:
    limit = max(1, min(int(limit), 200))
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    try:
        masters = con.execute("""SELECT m.* FROM trades m WHERE m.is_master=1 AND m.strategy_name='trend_4h'
                              ORDER BY (m.status != 'closed') DESC,
                              COALESCE((SELECT MAX(c.timestamp) FROM trades c WHERE c.position_id=m.position_id AND c.is_master=0), m.timestamp) DESC LIMIT ?""", (limit,)).fetchall()
        rows = []
        for raw in masters:
            master = dict(raw)
            children = [dict(row) for row in con.execute('SELECT * FROM trades WHERE position_id=? AND is_master=0 ORDER BY timestamp,id', (master['position_id'],))]
            accounting = position_accounting(master, children)
            last = children[-1] if children else {}
            rows.append({k: master.get(k) for k in ('id', 'position_id', 'symbol', 'position_type', 'status', 'timestamp', 'price', 'amount')} | {
                'pnl_eur': round(accounting['pnl_eur'], 6), 'fees_eur': round(accounting['fees_eur'], 6),
                'exit_reason': last.get('exit_reason'), 'close_ts': last.get('timestamp') if master.get('status') == 'closed' else None})
        return {'rows': rows, 'status': 'OK', 'warning': ''}
    except (sqlite3.Error, ValueError) as error:
        return {'rows': [], 'status': 'ERROR', 'warning': str(error)}
    finally:
        con.close()


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
