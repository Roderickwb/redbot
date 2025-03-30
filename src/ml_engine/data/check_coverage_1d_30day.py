import pymysql
import math
from datetime import datetime, timezone, timedelta

DB_CONFIG = {
    "host": "localhost",
    "user": "botuser",
    "password": "MySQL194860!",
    "database": "tradebot"
}

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def get_minmax_ts_1d(symbol_id: int):
    """
    Haal MIN en MAX timestamp_ms op uit market_data
    waar interval='1d' (i.p.v. '1h').
    """
    conn = get_db_connection()
    cur = conn.cursor()
    sql = """
    SELECT MIN(timestamp_ms), MAX(timestamp_ms)
    FROM market_data
    WHERE symbol_id=%s
      AND `interval`='1d'
    """
    cur.execute(sql, (symbol_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row or row[0] is None or row[1] is None:
        return None, None
    return row[0], row[1]

def count_1d_candles_in_range(symbol_id: int, start_ts: int, end_ts: int):
    """
    Tel hoeveel 1d-candles er in market_data interval='1d'
    tussen [start_ts..end_ts).
    """
    conn = get_db_connection()
    cur = conn.cursor()
    sql = """
    SELECT COUNT(*)
    FROM market_data
    WHERE symbol_id=%s
      AND `interval`='1d'
      AND timestamp_ms >= %s
      AND timestamp_ms < %s
    """
    cur.execute(sql, (symbol_id, start_ts, end_ts))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else 0

def check_coverage_1d_30days(symbol_id: int):
    """
    Net als 'check_coverage_30days', maar voor '1d'-candles.
    We maken blokken van 30 dagen (in ms).
    Per blok => actual / expected
      - actual = # of 1d-candles in [block_start, block_end)
      - expected = floor((block_end - block_start)/86400000)
    """
    min_ts, max_ts = get_minmax_ts_1d(symbol_id)
    if not min_ts or not max_ts:
        print(f"[symbol_id={symbol_id}] => No 1d data.")
        return

    # 30 dagen in ms
    THIRTY_DAYS_MS = 30*24*3600*1000

    block_start = min_ts
    block_num = 0

    while block_start < max_ts:
        block_end = block_start + THIRTY_DAYS_MS
        if block_end > max_ts:
            block_end = max_ts + 1  # tot inclusieve max

        # Tel feitelijke 1d-candles
        actual = count_1d_candles_in_range(symbol_id, block_start, block_end)

        # Verwacht: (block_end - block_start) / 86400000
        block_size_ms = block_end - block_start
        expected = int(block_size_ms // 86400000)

        coverage = (actual / expected) if expected>0 else 1.0

        dt_start = datetime.utcfromtimestamp(block_start/1000)
        dt_end   = datetime.utcfromtimestamp(block_end/1000)

        print(f"[symbol_id={symbol_id}] block#{block_num} => "
              f"{dt_start.strftime('%Y-%m-%d')}..{dt_end.strftime('%Y-%m-%d')} => "
              f"coverage={coverage*100:.2f}% (actual={actual}, exp={expected})")

        block_num += 1
        block_start = block_end

def main():
    symbol_id = 22  # coin ID dat je wilt checken op 1d
    check_coverage_1d_30days(symbol_id)

if __name__=="__main__":
    main()
