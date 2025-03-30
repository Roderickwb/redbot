import pymysql

DB_CONFIG = {
    "host": "localhost",
    "user": "botuser",
    "password": "MySQL194860!",
    "database": "tradebot"
}

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def fetch_symbol_ids_with_1d():
    """
    Haal alle symbol_id's op die '1d'-data hebben in market_data.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    sql = """
    SELECT DISTINCT symbol_id
    FROM market_data
    WHERE `interval`='1d'
    """
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [r[0] for r in rows]

def check_coverage_1d(symbol_id):
    """
    1) Haal min(timestamp_ms), max(timestamp_ms), count(*)
    2) Bepaal expected # of days => (max_ts - min_ts)/86400000 + 1
    3) coverage = actual_count / expected
    """
    conn = get_db_connection()
    cur = conn.cursor()
    sql = """
    SELECT MIN(timestamp_ms), MAX(timestamp_ms), COUNT(*)
    FROM market_data
    WHERE symbol_id=%s
      AND `interval`='1d'
    """
    cur.execute(sql, (symbol_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row or not row[0] or not row[1]:
        print(f"symbol_id={symbol_id} => no 1d data.")
        return

    min_ts, max_ts, actual_count = row

    diff_ms = max_ts - min_ts
    if diff_ms <= 0:
        coverage = 1.0 if actual_count>0 else 0.0
    else:
        # 1 dag = 24 * 3600000 = 86400000 ms
        day_ms = 86400000
        expected = int(diff_ms // day_ms) + 1
        coverage = actual_count / expected

    print(f"symbol_id={symbol_id}, 1d coverage={coverage*100:.2f}% "
          f"(actual={actual_count}, expected~{expected})")

def main():
    syms = fetch_symbol_ids_with_1d()
    if not syms:
        print("No symbol_ids with '1d' data found.")
        return

    for sid in syms:
        check_coverage_1d(sid)

if __name__=="__main__":
    main()
