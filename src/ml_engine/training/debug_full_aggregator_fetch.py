import mysql.connector
import pandas as pd

DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}


def debug_full_aggregator_fetch(symbol_id=1,
                                interval='1h',
                                start_date='2020-01-01',
                                end_date='2020-01-31 23:59:00'):
    """
    Imiteert aggregator:
    1) SELECT alle rows (geen WHERE op timestamp_ms)
    2) pd.DataFrame(rows)
    3) to_numeric
    4) df['datetime'] = pd.to_datetime(..., unit='ms')
    5) df.set_index('datetime')
    6) df = df.loc[start_date:end_date]
    7) print(len(df)), done
    """

    print("[DEBUG] Start debug_full_aggregator_fetch...")
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT timestamp_ms, open, high, low, close, volume
        FROM market_data
        WHERE symbol_id=%s
          AND `interval`=%s
        ORDER BY timestamp_ms ASC
    """
    cursor.execute(query, (symbol_id, interval))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    print(f"[DEBUG] Total rows from DB (no date filter) = {len(rows)}")

    # exact aggregator approach
    df = pd.DataFrame(rows)
    if df.empty:
        print("[DEBUG] df empty => no data.")
        return

    # to_numeric
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open','high','low','close','volume'], inplace=True)

    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    # Python-slice
    df = df.loc[start_date:end_date]
    print(f"[DEBUG] After slicing {start_date}~{end_date}, rows={len(df)}")
    print(df.head(5))
    print(df.tail(5))


if __name__ == "__main__":
    debug_full_aggregator_fetch(
        symbol_id=1,
        interval='1h',
        start_date='2020-01-01',
        end_date='2020-01-31 23:59:00'
    )
    print("[DEBUG] Done.")
