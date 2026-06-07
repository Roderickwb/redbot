"""Small runtime health check used by the Pi watchdog.

The check is intentionally independent from the bot process. It reads the
latest Kraken 5m candle directly from SQLite so it can detect a running but
stalled runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

from src.config.config import DB_FILE


EXIT_HEALTHY = 0
EXIT_STALE = 2
EXIT_UNAVAILABLE = 3


def _utc_ms(timestamp_ms: int) -> Optional[str]:
    if not timestamp_ms:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def check_runtime_health(
    db_path: str = DB_FILE,
    interval: str = "5m",
    max_age_min: float = 30.0,
) -> dict:
    now_ms = int(time.time() * 1000)
    result = {
        "status": "UNAVAILABLE",
        "interval": interval,
        "max_age_min": float(max_age_min),
        "latest_ts": 0,
        "latest_utc": None,
        "age_min": None,
        "db_path": db_path,
        "reason": "",
    }

    if not os.path.exists(db_path):
        result["reason"] = "database_not_found"
        return result

    try:
        uri = f"file:{os.path.abspath(db_path)}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=5) as con:
            row = con.execute(
                "SELECT MAX(timestamp), COUNT(*) FROM candles_kraken WHERE interval=?",
                (interval,),
            ).fetchone()
    except Exception as exc:
        result["reason"] = f"database_error:{exc}"
        return result

    latest_ts = int((row or (0, 0))[0] or 0)
    count = int((row or (0, 0))[1] or 0)
    if latest_ts and latest_ts < 10_000_000_000:
        latest_ts *= 1000

    result["latest_ts"] = latest_ts
    result["latest_utc"] = _utc_ms(latest_ts)
    result["count"] = count
    if not latest_ts:
        result["reason"] = "no_candles"
        return result

    age_min = round((now_ms - latest_ts) / 60_000.0, 2)
    result["age_min"] = age_min
    if age_min > float(max_age_min):
        result["status"] = "STALE"
        result["reason"] = "latest_candle_too_old"
    else:
        result["status"] = "HEALTHY"
        result["reason"] = "latest_candle_fresh"
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check Red Bot candle freshness.")
    parser.add_argument("--db-path", default=DB_FILE)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--max-age-min", type=float, default=30.0)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = check_runtime_health(
        db_path=args.db_path,
        interval=args.interval,
        max_age_min=args.max_age_min,
    )
    if args.compact:
        age = result.get("age_min")
        age_text = f"{age:.2f} min" if isinstance(age, (int, float)) else "unknown"
        print(
            f"Candles: {result.get('status')} | {result.get('interval')} age={age_text} | "
            f"latest={result.get('latest_utc')} | rows={result.get('count', 0)}"
        )
    else:
        print(json.dumps(result, ensure_ascii=False))
    if result["status"] == "HEALTHY":
        return EXIT_HEALTHY
    if result["status"] == "STALE":
        return EXIT_STALE
    return EXIT_UNAVAILABLE


if __name__ == "__main__":
    raise SystemExit(main())
