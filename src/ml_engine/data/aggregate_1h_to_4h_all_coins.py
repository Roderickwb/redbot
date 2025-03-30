import os
import math
import pandas as pd
import pymysql
from sqlalchemy import create_engine

DB_CONFIG = {
    "host": "localhost",
    "user": "botuser",
    "password": "MySQL194860!",
    "database": "tradebot"
}

def get_engine():
    """
    Maakt een SQLAlchemy-engine aan zodat we Pandas read_sql kunnen gebruiken.
    """
    from urllib.parse import quote_plus
    user = DB_CONFIG['user']
    pw   = quote_plus(DB_CONFIG['password'])
    host = DB_CONFIG['host']
    db   = DB_CONFIG['database']
    conn_str = f"mysql+pymysql://{user}:{pw}@{host}/{db}?charset=utf8mb4"
    engine = create_engine(conn_str)
    return engine

def get_all_symbol_ids():
    """
    Haalt alle symbol_id's op die 1h-data hebben in market_data.
    Of pas aan als je alleen 'EUR' wilt of 'active=1' etc.
    """
    engine = get_engine()
    query = """
    SELECT DISTINCT symbol_id
    FROM market_data
    WHERE `interval`='1h'
    """
    df_ids = pd.read_sql(query, engine)
    if df_ids.empty:
        return []
    return df_ids["symbol_id"].tolist()

def fetch_1h_data(symbol_id: int) -> pd.DataFrame:
    """
    Haal ALLE 1h-candles (zonder datumbereik-filter) voor de hele historie in market_data.
    Retouneert DataFrame met kolommen [timestamp_ms, open, high, low, close, volume, trades].
    Geordend op timestamp_ms.
    """
    engine = get_engine()
    query = f"""
    SELECT timestamp_ms, `open`, high, low, `close`, volume, trades
    FROM market_data
    WHERE symbol_id={symbol_id}
      AND `interval`='1h'
    ORDER BY timestamp_ms ASC
    """
    df = pd.read_sql(query, engine)
    return df

def resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas resample van 1h -> 4h:
      - 'open' = eerste open
      - 'high' = max
      - 'low'  = min
      - 'close'= laatste close
      - 'volume' = sum
      - 'trades' = sum
    Drop rijen waar open/close NaN is.
    """
    if df_1h.empty:
        return df_1h

    # Zet timestamp_ms -> datetime index
    df = df_1h.copy()
    df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df.set_index("datetime_utc", inplace=True)
    df.sort_index(inplace=True)

    # Resample("4H")
    df_4h = df.resample("4H").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
        "trades": "sum"
    })
    # Drop onvolledige blokken (NaN in open/close => aggregator had geen data in die 4h)
    df_4h.dropna(subset=["open","close"], inplace=True)
    return df_4h

def insert_4h_data(symbol_id: int, df_4h: pd.DataFrame):
    """
    Schrijf de df_4h in market_data met interval='4h'.
    Gebruikt (symbol_id, '4h', timestamp_ms) als unieke key.
    timestamp_ms = begin van de 4h-blok in ms.
    """
    if df_4h.empty:
        return

    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()

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

    count_inserted = 0
    for idx, row in df_4h.iterrows():
        dt_utc = idx  # 4h-blok start in datetime
        # Omgerekend naar ms
        ts_ms = int(dt_utc.timestamp()*1000)

        op = row["open"]
        hi = row["high"]
        lo = row["low"]
        cl = row["close"]
        vol= row["volume"]
        trd= row["trades"]

        cur.execute(sql, (
            symbol_id, "4h",
            ts_ms,
            dt_utc,
            op, hi, lo, cl,
            vol, trd
        ))
        count_inserted += 1
        if count_inserted % 2000 == 0:
            conn.commit()

    conn.commit()
    cur.close()
    conn.close()
    print(f"[OK] symbol_id={symbol_id} => inserted/updated {count_inserted} rows => '4h'")

def aggregate_1h_to_4h(symbol_id: int):
    """
    - fetch_1h_data
    - resample_4h
    - insert_4h_data
    """
    df_1h = fetch_1h_data(symbol_id)
    if df_1h.empty:
        print(f"[SKIP] symbol_id={symbol_id}, no 1h data found at all.")
        return

    df_4h = resample_4h(df_1h)
    if df_4h.empty:
        print(f"[SKIP] symbol_id={symbol_id}, after resample => 4h is empty.")
        return

    insert_4h_data(symbol_id, df_4h)

def main():
    # Haal alle symbol_id's die 1h-data hebben
    syms = get_all_symbol_ids()
    if not syms:
        print("[NO] no symbol_ids found with '1h' data.")
        return

    print(f"[INFO] aggregator => found {len(syms)} symbol_ids with 1h => now 1h->4h")
    for sid in syms:
        aggregate_1h_to_4h(sid)

    print("[DONE] aggregator => 1h->4h for all coins complete.")

if __name__=="__main__":
    main()
