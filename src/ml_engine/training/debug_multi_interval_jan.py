"""
debug_multi_interval_jan.py

1) Laadt 1h, 4h, 1d data voor jan 2020
2) Berekent RSI, MACD, ADX, Stoch, Bollinger
3) Merge-asof (1h <- 4h, dan merge <- 1d)
4) Check of je wel/geen crash krijgt.

Voor stap-voor-stap debuggen in 1 script.
"""

import mysql.connector
import pandas as pd
import ta  # pip install ta


DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}

def load_interval_entire_then_slice(interval='1h',
                                    start_date='2020-01-01',
                                    end_date='2020-01-31 23:59:00',
                                    symbol_id=1):
    """
    1) Haalt ALLE data van 'interval', symbol_id=1, sorted by timestamp_ms.
    2) Zet in DataFrame, to_numeric, datetime-index
    3) Slice op [start_date, end_date]
    4) Return df
    """
    print(f"\n[LOAD] interval={interval}, {start_date} -> {end_date}")
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

    print(f"[INFO] total {interval} rows from DB = {len(rows)}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] no data loaded for interval=", interval)
        return df

    # Convert
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open','high','low','close','volume'], inplace=True)

    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    df = df.loc[start_date:end_date]
    print(f"[INFO] after slice => rows={len(df)}")

    return df

def calc_indicators(df, interval='1h'):
    """
    Berekent RSI(14), MACD(12,26,9), ADX(14), Stoch(14,3,3), Bollinger(20,2)
    en dropna. Print debug info.
    """
    if df.empty:
        print(f"[WARN] df empty => skip indicator calc for {interval}")
        return df

    print(f"[STEP] calc_indicators on {interval} start. rows={len(df)}")

    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    # MACD
    df['macd'] = ta.trend.macd(df['close'], window_slow=26, window_fast=12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    # ADX
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    # Stoch
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], 14, 3)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], 14, 3)
    # Bollinger
    df['bb_mavg'] = ta.volatility.bollinger_mavg(df['close'], window=20)
    df['bb_hband'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
    df['bb_lband'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)

    df.dropna(inplace=True)
    print(f"[STEP] calc_indicators on {interval} done. rows={len(df)}")
    return df

if __name__=="__main__":
    # Periode = jan 2020
    start_d = "2014-01-01"
    end_d   = "2024-02-28 23:59:00"

    print("[DEBUG] Start loading 1h january")
    df_1h = load_interval_entire_then_slice('1h', start_d, end_d)
    df_1h = calc_indicators(df_1h, interval='1h')
    print("[DEBUG] 1h final rows:", len(df_1h))

    print("\n[DEBUG] Start loading 4h january")
    df_4h = load_interval_entire_then_slice('4h', start_d, end_d)
    df_4h = calc_indicators(df_4h, interval='4h')
    print("[DEBUG] 4h final rows:", len(df_4h))

    print("\n[DEBUG] Start loading 1d january")
    df_1d = load_interval_entire_then_slice('1d', start_d, end_d)
    df_1d = calc_indicators(df_1d, interval='1d')
    print("[DEBUG] 1d final rows:", len(df_1d))

    # Als alles nog leeft, doen we merge_asof: 1h <- 4h, dan merge <- 1d
    if df_1h.empty or df_4h.empty or df_1d.empty:
        print("[WARN] One of the dataframes is empty => skip merge.")
    else:
        print("\n[DEBUG] Start merges (1h<-4h, then merge<-1d).")
        df_1h_srt = df_1h.sort_index()
        df_4h_srt = df_4h.sort_index()
        df_1d_srt = df_1d.sort_index()

        import pandas as pd

        print("[DEBUG] merge_asof 1h<-4h ...")
        df_merge = pd.merge_asof(
            df_1h_srt, df_4h_srt,
            left_index=True, right_index=True,
            direction='backward'
        )
        print("[DEBUG] after merge(1h<-4h): rows=", len(df_merge))

        print("[DEBUG] merge_asof (merge<-1d) ...")
        df_merge = pd.merge_asof(
            df_merge, df_1d_srt,
            left_index=True, right_index=True,
            direction='backward'
        )
        print("[DEBUG] after merge(...<-1d): rows=", len(df_merge))

        # final dropna
        df_merge.dropna(inplace=True)
        print("[DEBUG] final merged dropna => rows=", len(df_merge))

    print("\nDone - no crash!")
