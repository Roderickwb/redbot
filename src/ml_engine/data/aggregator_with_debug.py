import mysql.connector
import pandas as pd
import numpy as np
import ta  # pip install ta

DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}


def load_ohlcv_from_db(symbol_id, interval='1h', start_date=None, end_date=None):
    print(f"[DEBUG] load_ohlcv_from_db: symbol={symbol_id}, interval={interval}, "
          f"start_date={start_date}, end_date={end_date}")

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT 
            timestamp_ms,
            open,
            high,
            low,
            close,
            volume
        FROM market_data
        WHERE symbol_id = %s
          AND `interval` = %s
        ORDER BY timestamp_ms ASC
    """
    cursor.execute(query, (symbol_id, interval))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    df = pd.DataFrame(rows)
    print(f"[DEBUG] load_ohlcv_from_db: raw rows={len(df)}")
    if df.empty:
        return df

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    if start_date:
        df = df.loc[start_date:]
    if end_date:
        df = df.loc[:end_date]

    print(f"[DEBUG] load_ohlcv_from_db: after filter => {len(df)} rows, "
          f"range: {df.index.min()} -> {df.index.max()}")
    return df


def add_indicators_1h(df):
    print(f"[DEBUG] add_indicators_1h: pre rows={len(df)}")
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'], 26, 12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], 14, 3)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], 14, 3)
    df['bb_mavg'] = ta.volatility.bollinger_mavg(df['close'], window=20)
    df['bb_hband'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
    df['bb_lband'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()
    df['atr14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    df.dropna(inplace=True)
    print(f"[DEBUG] add_indicators_1h: post rows={len(df)}")
    return df


def add_indicators_4h(df):
    print(f"[DEBUG] add_indicators_4h: pre rows={len(df)}")
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'], 26, 12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    df['atr14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()
    df.dropna(inplace=True)
    print(f"[DEBUG] add_indicators_4h: post rows={len(df)}")
    return df


def add_indicators_1d(df):
    print(f"[DEBUG] add_indicators_1d: pre rows={len(df)}")
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'], 26, 12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    df['atr14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['ema_200'] = df['close'].ewm(span=200).mean()
    df.dropna(inplace=True)
    print(f"[DEBUG] add_indicators_1d: post rows={len(df)}")
    return df


def add_target_return(df, shift=5):
    print(f"[DEBUG] add_target_return: pre rows={len(df)}")
    df['future_close'] = df['close'].shift(-shift)
    df['target_return'] = (df['future_close'] - df['close']) / df['close'] * 100.0
    df.dropna(inplace=True)
    df.drop(columns=['future_close'], inplace=True)
    print(f"[DEBUG] add_target_return: post rows={len(df)}")
    return df


def build_dataset(symbol_id, interval='1h', shift=5, start_date=None, end_date=None):
    print(f"[DEBUG] build_dataset: interval={interval}, shift={shift}, "
          f"start={start_date}, end={end_date}")
    df = load_ohlcv_from_db(symbol_id, interval, start_date, end_date)
    if df.empty:
        return df

    # Indicatoren
    if interval == '1h':
        df = add_indicators_1h(df)
    elif interval == '4h':
        df = add_indicators_4h(df)
    elif interval == '1d':
        df = add_indicators_1d(df)
    else:
        print(f"[WARN] Unknown interval={interval}, no indicators added.")
        return df

    if df.empty:
        print(f"[WARN] build_dataset: empty after indicator step. Return.")
        return df

    df = add_target_return(df, shift)
    return df


def build_multi_timeframe_dataset(symbol_id=1, shift=5, start_date=None, end_date=None):
    print(">> [INFO] build_multi_timeframe_dataset start...")
    print(f"   symbol_id={symbol_id}, shift={shift}, start={start_date}, end={end_date}")

    print("[DEBUG] build_dataset for 1h ...")
    df_1h = build_dataset(symbol_id, '1h', shift, start_date, end_date)
    print("[DEBUG] build_dataset(1h) rows =", len(df_1h))

    print("[DEBUG] build_dataset for 4h ...")
    df_4h = build_dataset(symbol_id, '4h', shift, start_date, end_date)
    print("[DEBUG] build_dataset(4h) rows =", len(df_4h))

    print("[DEBUG] build_dataset for 1d ...")
    df_1d = build_dataset(symbol_id, '1d', shift, start_date, end_date)
    print("[DEBUG] build_dataset(1d) rows =", len(df_1d))

    if df_1h.empty or df_4h.empty or df_1d.empty:
        print("[ERROR] One of the dataframes is empty => return empty df.")
        return pd.DataFrame()

    # Renames
    df_1h = df_1h.rename(columns={
        'open': 'open_1h',
        'high': 'high_1h',
        'low': 'low_1h',
        'close': 'close_1h',
        'volume': 'volume_1h',
        'target_return': 'target_return_1h'
    })
    df_4h = df_4h.rename(columns={
        'open': 'open_4h',
        'high': 'high_4h',
        'low': 'low_4h',
        'close': 'close_4h',
        'volume': 'volume_4h',
        'target_return': 'target_return_4h'
    })
    df_1d = df_1d.rename(columns={
        'open': 'open_1d',
        'high': 'high_1d',
        'low': 'low_1d',
        'close': 'close_1d',
        'volume': 'volume_1d',
        'target_return': 'target_return_1d'
    })

    # Suffix rename
    for col in list(df_1h.columns):
        if col not in ['timestamp_ms','open_1h','high_1h','low_1h','close_1h','volume_1h','target_return_1h']:
            df_1h.rename(columns={col: col+"_1h"}, inplace=True)

    for col in list(df_4h.columns):
        if col not in ['timestamp_ms','open_4h','high_4h','low_4h','close_4h','volume_4h','target_return_4h']:
            df_4h.rename(columns={col: col+"_4h"}, inplace=True)

    for col in list(df_1d.columns):
        if col not in ['timestamp_ms','open_1d','high_1d','low_1d','close_1d','volume_1d','target_return_1d']:
            df_1d.rename(columns={col: col+"_1d"}, inplace=True)

    df_1h_sorted = df_1h.sort_index()
    df_4h_sorted = df_4h.sort_index()
    df_1d_sorted = df_1d.sort_index()

    print("[DEBUG] merging 1h <- 4h ...")
    df_merge = pd.merge_asof(
        df_1h_sorted, df_4h_sorted,
        left_index=True, right_index=True,
        direction='backward'
    )
    print("[DEBUG] after merge(1h<-4h): rows=", len(df_merge))

    print("[DEBUG] merging (merge) <- 1d ...")
    df_merge = pd.merge_asof(
        df_merge, df_1d_sorted,
        left_index=True, right_index=True,
        direction='backward'
    )
    print("[DEBUG] after merge(...<-1d): rows=", len(df_merge))

    df_merge.dropna(inplace=True)
    print(">> [INFO] build_multi_timeframe_dataset DONE. Rows =", len(df_merge))
    return df_merge


if __name__ == "__main__":
    # Test single aggregator call
    symbol_id = 1
    SHIFT = 5
    start_d = "2020-01-01"
    end_d   = "2020-01-31"

    print(f"[DEBUG] aggregator_with_debug main: {start_d} -> {end_d}")
    df_test = build_multi_timeframe_dataset(symbol_id, SHIFT, start_d, end_d)
    print("[DEBUG] final df rows =", len(df_test))
    print(df_test.head())
