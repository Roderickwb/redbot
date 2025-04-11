import mysql.connector
import pandas as pd
import ta  # pip install ta

DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}

def load_1h_entire_then_slice(symbol_id=1, start_date='2020-01-01', end_date='2020-01-31 23:59:00'):
    """
    1) Haal alle 1h-data (symbol_id=1)
    2) pd.DataFrame(...) + to_numeric
    3) datetime index
    4) slice op [start_date, end_date]
    Return df
    """
    print("[STEP] Load entire DB for 1h, then slice in Python.")
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT timestamp_ms, open, high, low, close, volume
        FROM market_data
        WHERE symbol_id=%s
          AND `interval`='1h'
        ORDER BY timestamp_ms ASC
    """
    cursor.execute(query, (symbol_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    print(f"[INFO] total rows from DB = {len(rows)}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] no data loaded.")
        return df

    # Convert to numeric
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open','high','low','close','volume'], inplace=True)

    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    df = df.loc[start_date:end_date]
    print(f"[INFO] After slice {start_date}~{end_date}: rows={len(df)}")
    return df


def step_rsi(df):
    """
    1) Bereken RSI(14)
    2) dropna
    """
    print("[STEP] step_rsi started.")
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df.dropna(inplace=True)
    print(f"[STEP] step_rsi done. rows={len(df)}")


def step_macd(df):
    """
    1) MACD(12,26)
    2) macd_signal(12,26,9)
    3) dropna
    """
    print("[STEP] step_macd started.")
    df['macd'] = ta.trend.macd(df['close'], window_slow=26, window_fast=12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    df.dropna(inplace=True)
    print(f"[STEP] step_macd done. rows={len(df)}")


def step_adx(df):
    print("[STEP] step_adx started.")
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df.dropna(inplace=True)
    print(f"[STEP] step_adx done. rows={len(df)}")


def step_stoch(df):
    print("[STEP] step_stoch started.")
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], 14, 3)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], 14, 3)
    df.dropna(inplace=True)
    print(f"[STEP] step_stoch done. rows={len(df)}")


def step_bollinger(df):
    print("[STEP] step_bollinger started.")
    bb_mavg = ta.volatility.bollinger_mavg(df['close'], window=20)
    bb_hband = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
    bb_lband = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
    df['bb_mavg'] = bb_mavg
    df['bb_hband'] = bb_hband
    df['bb_lband'] = bb_lband
    df.dropna(inplace=True)
    print(f"[STEP] step_bollinger done. rows={len(df)}")


if __name__ == "__main__":
    # 1) Load january 2020
    df_jan = load_1h_entire_then_slice(symbol_id=1,
                                       start_date='2020-01-01',
                                       end_date='2020-01-31 23:59:00')
    if df_jan.empty:
        print("[ERROR] No data, nothing to do.")
        exit(0)

    # 2) We voeren STAPSGEWIJS indicatoren uit:
    #    run script => check of crash?
    #    - if OK, uncomment next step.

    step_rsi(df_jan)
    # PAS OP: Als dit OK is, uncomment de volgende lines:
    step_macd(df_jan)
    step_adx(df_jan)
    step_stoch(df_jan)
    step_bollinger(df_jan)

    # etc. (ATR, EMA, ...)

    print("[DEBUG] Final df columns:", df_jan.columns)
    print("[DEBUG] final rows:", len(df_jan))
    print(df_jan.head(5))
    print("\nDone - no crash!")
