import os
import numpy as np
import pandas as pd
import pymysql
from pymysql import OperationalError

DB_CONFIG = {
    "host": "localhost",
    "user": "botuser",
    "password": "MySQL194860!",
    "database": "tradebot"
}

CSV_DIR = r"C:\Users\My ACER\Downloads\Kraken_OHLCVT"
INTERVAL_STR = "1h"
MINUTES_EXPECTED = 60

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def get_symbol_id(conn, symbol_str: str):
    sql = "SELECT symbol_id FROM symbols WHERE symbol=%s LIMIT 1"
    with conn.cursor() as c:
        c.execute(sql, (symbol_str,))
        row = c.fetchone()
        return row[0] if row else None

def insert_candle_1h(conn,
                     symbol_id: int,
                     timestamp_ms: int,
                     dt_utc,
                     open_: float,
                     high_: float,
                     low_: float,
                     close_: float,
                     volume_: float,
                     trades_: float):
    sql = """
    INSERT INTO market_data
    (symbol_id, `interval`, timestamp_ms, datetime_utc,
     `open`, high, low, `close`, volume, trades)
    VALUES
    (%s, %s, %s, %s,
     %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      `open`=VALUES(`open`),
      high=VALUES(high),
      low=VALUES(low),
      `close`=VALUES(`close`),
      volume=VALUES(volume),
      trades=VALUES(trades),
      datetime_utc=VALUES(datetime_utc)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            symbol_id,
            INTERVAL_STR,
            timestamp_ms,
            dt_utc,
            open_, high_, low_, close_, volume_, trades_
        ))

def process_csv_file(filepath: str):
    filename = os.path.basename(filepath)
    base, _ = os.path.splitext(filename)

    parts = base.split("_")
    if len(parts) != 2:
        print(f"[SKIP] bad filename => {filename}")
        return

    coin_plus_eur, minstr = parts
    if not minstr.isdigit():
        print(f"[SKIP] minutes not digit => {filename}")
        return

    minutes_val = int(minstr)
    if minutes_val != MINUTES_EXPECTED:
        print(f"[SKIP] {filename} => {minutes_val} != {MINUTES_EXPECTED}")
        return

    if not coin_plus_eur.endswith("EUR"):
        print(f"[SKIP] {filename}, does not end with 'EUR'")
        return

    coin = coin_plus_eur[:-3]
    symbol_str = f"{coin}/EUR"

    try:
        df = pd.read_csv(
            filepath,
            header=None,
            names=["ts_s","open","high","low","close","volume","trades"]
        )
    except Exception as e:
        print(f"[SKIP] {filename} => error reading CSV => {e}")
        return

    if df.empty:
        print(f"[SKIP] {filename} => empty CSV.")
        return

    # GEBRUIK int64 i.p.v. 32-bit int
    df["ts_s"] = df["ts_s"].astype(float)
    df["timestamp_ms"] = (df["ts_s"] * 1000).astype(np.int64)  # <--- HIER int64
    df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    conn = get_db_connection()
    sym_id = get_symbol_id(conn, symbol_str)
    if sym_id is None:
        print(f"[SKIP] {filename} => symbol not found => {symbol_str}")
        conn.close()
        return

    inserted_count = 0
    skipped_count = 0

    for i, row_ in df.iterrows():
        ts_ms = int(row_["timestamp_ms"])
        dt_utc= row_["datetime_utc"]
        op    = float(row_["open"])
        hi    = float(row_["high"])
        lo    = float(row_["low"])
        cl    = float(row_["close"])
        vol   = float(row_["volume"])
        trd   = float(row_["trades"])

        try:
            insert_candle_1h(
                conn,
                sym_id,
                ts_ms,
                dt_utc,
                op, hi, lo, cl,
                vol,
                trd
            )
            inserted_count += 1
            if inserted_count % 2000 == 0:
                conn.commit()
        except pymysql.err.OperationalError as e:
            skipped_count += 1
            print(f"[SKIP row {i}] => {filename}, ts_s={row_['ts_s']}, dt_utc={dt_utc}, error={e}")
            continue

    conn.commit()
    conn.close()
    print(f"[OK] {filename} => inserted={inserted_count}, skipped={skipped_count} => {INTERVAL_STR}")

def main():
    files = os.listdir(CSV_DIR)
    csv_count = 0
    for f in files:
        if not f.endswith(".csv"):
            continue
        fullpath = os.path.join(CSV_DIR, f)
        process_csv_file(fullpath)
        csv_count += 1

    print(f"[DONE] => processed {csv_count} CSV files => {INTERVAL_STR}")

if __name__=="__main__":
    main()
