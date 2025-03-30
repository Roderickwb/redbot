import pymysql

DB_CONFIG = {
    "host": "localhost",
    "user": "botuser",
    "password": "MySQL194860!",
    "database": "tradebot"
}

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def fetch_symbol_ids_with_4h():
    """
    Haal alle symbol_id's op die 4h-data hebben.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    sql = """
    SELECT DISTINCT symbol_id
    FROM market_data
    WHERE `interval`='4h'
    """
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [r[0] for r in rows]

def check_coverage_4h(symbol_id):
    """
    1) Haal min(timestamp_ms), max(timestamp_ms), count(*)
    2) Bepaal expected # of 4h-candles => (diff_ms / 14,400,000) + 1
    3) coverage = actual / expected
    """
    conn = get_db_connection()
    cur = conn.cursor()
    sql = """
    SELECT MIN(timestamp_ms), MAX(timestamp_ms), COUNT(*)
    FROM market_data
    WHERE symbol_id=%s
      AND `interval`='4h'
    """
    cur.execute(sql, (symbol_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row or not row[0] or not row[1]:
        print(f"symbol_id={symbol_id} => no 4h data.")
        return

    min_ts, max_ts, actual_count = row

    diff_ms = max_ts - min_ts
    if diff_ms <= 0:
        # Als min_ts >= max_ts of empty?
        coverage = 1.0 if actual_count > 0 else 0.0
    else:
        # 4 uur = 4*3600000 = 14,400,000 ms
        hours_4_ms = 4*3600000
        expected = int(diff_ms // hours_4_ms) + 1
        coverage = actual_count / expected

    print(f"symbol_id={symbol_id}, 4h coverage={coverage*100:.2f}% "
          f"(actual={actual_count}, expected~{expected})")

def main():
    syms = fetch_symbol_ids_with_4h()
    if not syms:
        print("No symbol_ids with 4h data found.")
        return

    for sid in syms:
        check_coverage_4h(sid)

if __name__=="__main__":
    main()
