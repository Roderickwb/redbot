import os
import math
import pandas as pd
import pymysql
import numpy as np

###############################################################################
# DB CONFIG
###############################################################################
DB_CONFIG = {
    "host": "localhost",
    "user": "botuser",
    "password": "MySQL194860!",
    "database": "tradebot"
}

###############################################################################
# CSV DIRECTORY
###############################################################################
# Gebruik je echte map met ALL CSV's. We filteren straks op *_1440.csv
CSV_DIR = r"C:\Users\My ACER\Downloads\Kraken_OHLCVT"

###############################################################################
def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def get_symbol_id(conn, symbol_str):
    """
    Leest symbol_id uit 'symbols' waar symbol=...
    Return None als niet gevonden.
    """
    sql = "SELECT symbol_id FROM symbols WHERE symbol=%s LIMIT 1"
    with conn.cursor() as cur:
        cur.execute(sql, (symbol_str,))
        row = cur.fetchone()
        return row[0] if row else None

def insert_candle_1d(conn,
                     symbol_id: int,
                     ts_ms: int,
                     dt_utc,
                     op: float,
                     hi: float,
                     lo: float,
                     cl: float,
                     vol: float,
                     trd: float):
    """
    Schrijf 1d-candle in market_data (interval='1d') ON DUPLICATE KEY.
    """
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
            "1d",
            ts_ms,
            dt_utc,
            op, hi, lo, cl,
            vol, trd
        ))

def process_csv_1d(filepath: str):
    """
    1) We check filename => must end with _1440.csv
    2) Parse coin name => e.g. "ADAEUR_1440.csv" => coin='ADA', symbol='ADA/EUR'
    3) Read CSV => [timestamp_s, open, high, low, close, volume, trades]
    4) Convert timestamp => ms => dt_utc
    5) Insert row-by-row with interval='1d'
    """
    fn = os.path.basename(filepath)
    base, _ = os.path.splitext(fn)  # e.g. "ADAEUR_1440"
    parts = base.split("_")
    if len(parts) != 2:
        print(f"[SKIP] filename pattern not recognized => {fn}")
        return

    coin_plus_eur, minutes_str = parts
    if not minutes_str.isdigit():
        print(f"[SKIP] {fn}, minutes part not digit => {minutes_str}")
        return

    m_val = int(minutes_str)
    if m_val != 1440:
        print(f"[SKIP] {fn}, we only handle 1440 => skip.")
        return

    # Must end with EUR
    if not coin_plus_eur.endswith("EUR"):
        print(f"[SKIP] {fn}, does not end with 'EUR'")
        return

    coin = coin_plus_eur[:-3]  # remove 'EUR' suffix => e.g. "ADA"
    symbol_str = f"{coin}/EUR" # => "ADA/EUR"

    # now read CSV
    try:
        df = pd.read_csv(
            filepath,
            header=None,
            names=["ts_s","open","high","low","close","volume","trades"]
        )
    except Exception as e:
        print(f"[SKIP] error reading {fn} => {e}")
        return

    if df.empty:
        print(f"[SKIP] {fn}, empty CSV.")
        return

    # if CSV in seconds => multiply by 1000 => int64
    df["ts_s"] = df["ts_s"].astype(float)
    df["timestamp_ms"] = (df["ts_s"] * 1000).astype(np.int64)
    df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    conn = get_db_connection()
    sym_id = get_symbol_id(conn, symbol_str)
    if sym_id is None:
        print(f"[SKIP] {fn}, symbol not found => {symbol_str}")
        conn.close()
        return

    count_inserted = 0
    for i, row_ in df.iterrows():
        ts_ms = int(row_["timestamp_ms"])
        dt_utc= row_["datetime_utc"]
        op    = float(row_["open"])
        hi    = float(row_["high"])
        lo    = float(row_["low"])
        cl    = float(row_["close"])
        vol   = float(row_["volume"])
        trd   = float(row_["trades"])

        insert_candle_1d(
            conn,
            sym_id,
            ts_ms,
            dt_utc,
            op, hi, lo, cl,
            vol, trd
        )
        count_inserted += 1
        if count_inserted%2000 == 0:
            conn.commit()

    conn.commit()
    conn.close()
    print(f"[OK] {fn} => inserted/updated {count_inserted} rows => '1d'")

def main():
    files = os.listdir(CSV_DIR)
    csv_count = 0
    for f in files:
        # We only process .csv
        if not f.endswith(".csv"):
            continue
        fullpath = os.path.join(CSV_DIR, f)
        process_csv_1d(fullpath)
        csv_count += 1

    print(f"[DONE] => processed {csv_count} CSV files from {CSV_DIR}")

if __name__=="__main__":
    main()
