import mysql.connector
import sys

# PAS AAN naar jouw DB-config:
DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}

def debug_fetch_1h_data(start_timestamp_ms, end_timestamp_ms):
    """
    Laadt alle rows uit 'market_data' (1h) met timestamp_ms in [start_timestamp_ms, end_timestamp_ms).
    Fetcht row-by-row en telt hoeveel rijen we ophalen.

    start_timestamp_ms, end_timestamp_ms: integers (epoch in milliseconden).
    """
    print(f"\n[DEBUG] Start fetch: from {start_timestamp_ms} to {end_timestamp_ms}")
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    query = f"""
        SELECT timestamp_ms, open, high, low, close, volume
        FROM market_data
        WHERE symbol_id=1
          AND `interval`='1h'
          AND timestamp_ms >= {start_timestamp_ms}
          AND timestamp_ms < {end_timestamp_ms}
        ORDER BY timestamp_ms ASC
    """
    cursor.execute(query)
    count = 0

    try:
        while True:
            row = cursor.fetchone()
            if not row:
                break
            count += 1
            # Desgewenst kun je debug prints doen, maar laten we dat pas
            # doen als we kleinere ranges onderzoeken.
            # if count < 5:
            #     print(row)
    except Exception as e:
        print(f"[ERROR] Exception while fetching row: {e}")
        # als we hier komen is het een Python exception, niet de 0xC0000005 crash
    finally:
        cursor.close()
        conn.close()

    print(f"[DEBUG] Fetched {count} rows successfully from {start_timestamp_ms}..{end_timestamp_ms}")


def timestamp_ms_of(date_str):
    """
    date_str in 'YYYY-MM-DD' of 'YYYY-MM-DD HH:MM:SS'
    Geeft epoch in milliseconden terug.
    """
    import datetime
    import time

    # parse string => datetime obj
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    # of: if je HH:MM:SS wilt, gebruik: dt = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    # convert to unix epoch in seconds
    epoch_s = int(dt.timestamp())
    # ms:
    return epoch_s * 1000

if __name__ == "__main__":
    # 1) Probeer heel januari 2020: [2020-01-01, 2020-02-01)
    #    2020-01-01 => 1577836800000 ms
    #    2020-02-01 => 1580515200000 ms
    start_ms = timestamp_ms_of("2020-01-01")
    end_ms   = timestamp_ms_of("2020-02-01")

    debug_fetch_1h_data(start_ms, end_ms)
    print("\n[DEBUG] Done with debug_fetch_1h_data.")
