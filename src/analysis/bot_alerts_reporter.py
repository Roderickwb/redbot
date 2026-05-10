# ============================================================
# src/analysis/bot_alerts_reporter.py
# ============================================================
"""
Daily bot health and alert digest.

This module is intentionally read-only for trading state. It summarizes errors
and quality signals that should not be checked manually every day.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Iterable, Optional

from dotenv import load_dotenv

from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager
from src.notifier.telegram_notifier import TelegramNotifier


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "bot_alerts")
DEFAULT_LATEST_FILE = "latest_bot_alerts_report.json"
DAILY_ALERT_SENT_FILE = ".daily_bot_alert_sent"


def _pct(part: int, total: int) -> float:
    return round((part / total * 100.0), 2) if total else 0.0


def _row_value(rows: list[tuple] | None, default: Any = None) -> Any:
    if not rows or not rows[0]:
        return default
    return rows[0][0]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


class BotAlertsReporter:
    def __init__(self, db: Optional[DatabaseManager] = None, db_path: str = DB_FILE):
        self.db_path = db_path
        self.db = db or DatabaseManager(db_path=db_path)

    def build_report(self, hours: int = 24) -> dict:
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - int(hours * 3600 * 1000)

        report = {
            "created_ts": now_ms,
            "created_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "window_hours": hours,
            "status": "OK",
            "alerts": [],
            "gpt": self._gpt_health(cutoff_ms),
            "data": self._data_health(now_ms),
            "learning": self._learning_health(),
            "database": self._database_health(),
        }

        self._add_alerts(report)
        if any(a["level"] == "ALERT" for a in report["alerts"]):
            report["status"] = "ALERT"
        elif any(a["level"] == "WARN" for a in report["alerts"]):
            report["status"] = "WARN"
        return report

    def _gpt_health(self, cutoff_ms: int) -> dict:
        rows = self.db.execute_query(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN gpt_confidence = 0 THEN 1 ELSE 0 END) AS zero_conf,
              SUM(CASE WHEN json_extract(features_json,'$.primary_veto') = 'gpt_error' THEN 1 ELSE 0 END) AS gpt_errors,
              SUM(CASE WHEN json_extract(features_json,'$.primary_veto') = 'parse_error' THEN 1 ELSE 0 END) AS parse_errors,
              SUM(CASE WHEN json_extract(features_json,'$.scores.entry') IS NULL THEN 1 ELSE 0 END) AS missing_entry_score,
              SUM(CASE WHEN COALESCE(json_extract(features_json,'$.rationale'), '') LIKE '%Request timed out%' THEN 1 ELSE 0 END) AS timeouts
            FROM strategy_events
            WHERE event_type='gpt_decision'
              AND timestamp >= ?
            """,
            (cutoff_ms,),
        )
        row = rows[0] if rows else (0, 0, 0, 0, 0, 0)
        total = _safe_int(row[0])
        zero_conf = _safe_int(row[1])
        gpt_errors = _safe_int(row[2])
        parse_errors = _safe_int(row[3])
        missing_entry_score = _safe_int(row[4])
        timeouts = _safe_int(row[5])

        top_timeout_rows = self.db.execute_query(
            """
            SELECT symbol, COUNT(*) AS n
            FROM strategy_events
            WHERE event_type='gpt_decision'
              AND timestamp >= ?
              AND COALESCE(json_extract(features_json,'$.rationale'), '') LIKE '%Request timed out%'
            GROUP BY symbol
            ORDER BY n DESC, symbol ASC
            LIMIT 5
            """,
            (cutoff_ms,),
        )
        return {
            "total": total,
            "zero_conf": zero_conf,
            "zero_conf_pct": _pct(zero_conf, total),
            "gpt_errors": gpt_errors,
            "parse_errors": parse_errors,
            "missing_entry_score": missing_entry_score,
            "missing_entry_score_pct": _pct(missing_entry_score, total),
            "timeouts": timeouts,
            "timeout_symbols": [{"symbol": r[0], "count": r[1]} for r in (top_timeout_rows or [])],
        }

    def _data_health(self, now_ms: int) -> dict:
        rows = self.db.execute_query(
            """
            SELECT interval, MAX(timestamp), COUNT(*)
            FROM candles_kraken
            GROUP BY interval
            ORDER BY interval
            """
        )
        candles = {}
        for interval, latest_ts, count in rows or []:
            latest_ts = _safe_int(latest_ts)
            candles[str(interval)] = {
                "latest_ts": latest_ts,
                "latest_utc": datetime.utcfromtimestamp(latest_ts / 1000).strftime("%Y-%m-%d %H:%M:%S") if latest_ts else None,
                "age_min": round((now_ms - latest_ts) / 60000.0, 1) if latest_ts else None,
                "count": _safe_int(count),
            }

        event_rows = self.db.execute_query(
            """
            SELECT MAX(timestamp), COUNT(*)
            FROM strategy_events
            """
        )
        latest_event_ts = _safe_int(event_rows[0][0]) if event_rows else 0
        return {
            "candles_kraken": candles,
            "latest_strategy_event_ts": latest_event_ts,
            "latest_strategy_event_utc": datetime.utcfromtimestamp(latest_event_ts / 1000).strftime("%Y-%m-%d %H:%M:%S") if latest_event_ts else None,
            "latest_strategy_event_age_min": round((now_ms - latest_event_ts) / 60000.0, 1) if latest_event_ts else None,
        }

    def _learning_health(self) -> dict:
        rows = self.db.execute_query(
            """
            SELECT outcome_status, COUNT(*)
            FROM strategy_events
            GROUP BY outcome_status
            """
        )
        by_status = {str(status or "pending"): _safe_int(count) for status, count in (rows or [])}

        profile_rows = self.db.execute_query(
            """
            SELECT COUNT(*), MAX(updated_ts)
            FROM coin_profiles
            WHERE source='strategy_events_learning'
            """
        )
        profile_count = _safe_int(profile_rows[0][0]) if profile_rows else 0
        latest_profile_ts = _safe_int(profile_rows[0][1]) if profile_rows else 0
        return {
            "strategy_events_by_status": by_status,
            "learning_profile_count": profile_count,
            "latest_profile_ts": latest_profile_ts,
            "latest_profile_utc": datetime.utcfromtimestamp(latest_profile_ts / 1000).strftime("%Y-%m-%d %H:%M:%S") if latest_profile_ts else None,
        }

    def _database_health(self) -> dict:
        db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0.0
        wal_path = f"{self.db_path}-wal"
        wal_size_mb = os.path.getsize(wal_path) / (1024 * 1024) if os.path.exists(wal_path) else 0.0
        free_gb = None
        try:
            stat = os.statvfs(os.path.dirname(self.db_path) or ".")
            free_gb = stat.f_bavail * stat.f_frsize / (1024 ** 3)
        except Exception:
            pass

        return {
            "db_size_mb": round(db_size_mb, 1),
            "wal_size_mb": round(wal_size_mb, 1),
            "free_gb": round(free_gb, 2) if free_gb is not None else None,
        }

    def _add_alerts(self, report: dict) -> None:
        gpt = report["gpt"]
        data = report["data"]
        db = report["database"]
        learning = report["learning"]

        zero_pct = _safe_float(gpt.get("zero_conf_pct"))
        missing_pct = _safe_float(gpt.get("missing_entry_score_pct"))
        if zero_pct >= 10.0:
            report["alerts"].append({"level": "ALERT", "code": "GPT_ZERO_CONF_HIGH", "message": f"GPT zero-confidence rate is {zero_pct}%."})
        elif zero_pct >= 5.0:
            report["alerts"].append({"level": "WARN", "code": "GPT_ZERO_CONF_WARN", "message": f"GPT zero-confidence rate is {zero_pct}%."})

        if _safe_int(gpt.get("timeouts")) >= 3:
            report["alerts"].append({"level": "WARN", "code": "GPT_TIMEOUTS", "message": f"GPT timeouts in window: {gpt.get('timeouts')}."})

        if missing_pct >= 10.0:
            report["alerts"].append({"level": "WARN", "code": "GPT_STRUCTURED_OUTPUT_MISSING", "message": f"Missing entry scores: {missing_pct}%."})

        five_min = data.get("candles_kraken", {}).get("5m", {})
        if five_min.get("age_min") is not None and five_min["age_min"] > 30:
            report["alerts"].append({"level": "ALERT", "code": "CANDLES_STALE", "message": f"Latest 5m candle age is {five_min['age_min']} min."})

        if db.get("free_gb") is not None and db["free_gb"] < 5.0:
            report["alerts"].append({"level": "ALERT", "code": "DISK_LOW", "message": f"Disk free is {db['free_gb']} GB."})

        if db.get("wal_size_mb", 0.0) > 50.0:
            report["alerts"].append({"level": "WARN", "code": "WAL_EXISTS", "message": f"WAL size is {db['wal_size_mb']} MB."})

        if _safe_int(learning.get("learning_profile_count")) < 10:
            report["alerts"].append({"level": "WARN", "code": "LEARNING_PROFILES_LOW", "message": f"Learning profiles count is {learning.get('learning_profile_count')}."})


def format_alert_message(report: dict) -> str:
    gpt = report.get("gpt", {})
    data = report.get("data", {})
    db = report.get("database", {})
    learning = report.get("learning", {})
    alerts = report.get("alerts", [])
    candles_5m = (data.get("candles_kraken") or {}).get("5m", {})
    pending = (learning.get("strategy_events_by_status") or {}).get("pending", 0)
    labeled = (learning.get("strategy_events_by_status") or {}).get("labeled", 0)

    alert_lines = "\n".join(
        f"- {a.get('level')} {a.get('code')}: {a.get('message')}"
        for a in alerts[:8]
    ) or "- none"

    timeout_symbols = ", ".join(
        f"{row.get('symbol')}:{row.get('count')}"
        for row in (gpt.get("timeout_symbols") or [])[:5]
    ) or "-"

    return (
        f"Bot Health Digest [{report.get('status')}]\n"
        f"Window: {report.get('window_hours')}h | created={report.get('created_local')}\n"
        f"GPT: total={gpt.get('total', 0)} | zero={gpt.get('zero_conf', 0)} ({gpt.get('zero_conf_pct', 0)}%) | timeouts={gpt.get('timeouts', 0)} | missing_scores={gpt.get('missing_entry_score_pct', 0)}%\n"
        f"Timeout symbols: {timeout_symbols}\n"
        f"Data: latest 5m candle age={candles_5m.get('age_min')} min | strategy_event_age={data.get('latest_strategy_event_age_min')} min\n"
        f"Learning: labeled={labeled} | pending={pending} | profiles={learning.get('learning_profile_count', 0)}\n"
        f"DB: size={db.get('db_size_mb')}MB | wal={db.get('wal_size_mb')}MB | free={db.get('free_gb')}GB\n"
        f"Alerts:\n{alert_lines}"
    )


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def send_telegram_once_per_day(report: dict, output_dir: str, force: bool = False) -> bool:
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(output_dir, exist_ok=True)
    sent_path = os.path.join(output_dir, DAILY_ALERT_SENT_FILE)

    if not force and os.path.exists(sent_path):
        try:
            with open(sent_path, "r", encoding="utf-8") as f:
                if f.read().strip() == today:
                    return False
        except Exception:
            pass

    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False

    TelegramNotifier(token, chat_id).safe_send(format_alert_message(report))
    with open(sent_path, "w", encoding="utf-8") as f:
        f.write(today)
    return True


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build daily bot health and alerts report.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--send", action="store_true", help="Send Telegram digest once per day.")
    parser.add_argument("--force-send", action="store_true", help="Ignore daily sent marker.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    db = DatabaseManager(db_path=DB_FILE)
    try:
        reporter = BotAlertsReporter(db=db)
        report = reporter.build_report(hours=args.hours)
    finally:
        db.close_connection()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, DEFAULT_LATEST_FILE)
    write_json(output_path, report)
    sent = False
    if args.send or args.force_send:
        sent = send_telegram_once_per_day(report, args.output_dir, force=args.force_send)

    result = {
        "status": report.get("status"),
        "alerts": len(report.get("alerts", [])),
        "output_path": output_path,
        "telegram_sent": sent,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
