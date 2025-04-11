import mysql.connector
import pandas as pd
import ta  # pip install ta

# =========================
# [A] DATABASECONFIG (PAS AAN)
# =========================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}

# =========================
# [B] LAADFUNCTIE
# =========================
def load_df(symbol_id, interval, start_date, end_date):
    """
    Haalt ALLE data (symbol_id, interval) uit 'market_data'.
    Filtert dan in Python op [start_date, end_date].
    Returned DataFrame (index=datetime).
    """
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

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[DEBUG] load_df: symbol={symbol_id}, interval={interval},"
              f" => 0 rows (na fetchall)!")
        return df

    # Convert kolommen naar float, drop NaN
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open','high','low','close','volume'], inplace=True)

    # datetime index + slicen
    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    df = df.loc[start_date:end_date]
    print(f"[DEBUG] load_df: symbol={symbol_id}, interval={interval}, after slice => {len(df)} rows.")
    return df

# =========================
# [C] INDICATORFUNCTIE
# =========================
def calc_indicators(df, interval='1h'):
    """
    Berekent RSI(14), MACD(12,26,9), ADX(14), Stoch(14,3,3), Bollinger(20,2).
    Dropna na berekening.
    """
    if df.empty:
        print(f"[DEBUG] calc_indicators: df empty => skip.")
        return df

    print(f"[DEBUG] calc_indicators({interval}): pre rows={len(df)}")

    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    # MACD
    df['macd'] = ta.trend.macd(df['close'], window_slow=26, window_fast=12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    # ADX
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
    # Stoch
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], 14, 3)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], 14, 3)
    # Bollinger(20)
    df['bb_mavg'] = ta.volatility.bollinger_mavg(df['close'], window=20)
    df['bb_hband'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
    df['bb_lband'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)

    df.dropna(inplace=True)
    print(f"[DEBUG] calc_indicators({interval}): post rows={len(df)}")
    return df

# =========================
# [D] SHIFT-target
# =========================
def add_shift_target(df, shift=5):
    """
    SHIFT-target op basis van close(t+shift).
    df['target_return'] = ...
    """
    if df.empty:
        print("[DEBUG] add_shift_target: df empty => skip")
        return df

    print(f"[DEBUG] add_shift_target: pre rows={len(df)}")
    df['future_close'] = df['close'].shift(-shift)
    df['target_return'] = (df['future_close'] - df['close']) / df['close'] * 100.0
    df.dropna(inplace=True)
    df.drop(columns=['future_close'], inplace=True)
    print(f"[DEBUG] add_shift_target: post rows={len(df)}")
    return df

# =========================
# [E] MAIN AGGREGATOR
# =========================
def build_multi_tf_df(symbol_id, start_date, end_date, shift=5):
    """
    1) 1h => load_df => indicators
    2) 4h => load_df => indicators
    3) 1d => load_df => indicators
    4) merges => dropna
    5) SHIFT-target
    Return final DF
    """
    print(f"\n>> build_multi_tf_df(symbol={symbol_id}, start={start_date}, end={end_date})")

    # --- 1h
    df_1h = load_df(symbol_id, '1h', start_date, end_date)
    df_1h = calc_indicators(df_1h, '1h')
    if not df_1h.empty:
        print(f"[DEBUG] df_1h final post-indicators => {len(df_1h)}")
        df_1h.rename(columns={
            'open':'open_1h','high':'high_1h','low':'low_1h','close':'close_1h','volume':'volume_1h'
        }, inplace=True)
    else:
        print("[DEBUG] df_1h is empty, aggregator ends.")
        return pd.DataFrame()

    # --- 4h
    df_4h = load_df(symbol_id, '4h', start_date, end_date)
    df_4h = calc_indicators(df_4h, '4h')
    if not df_4h.empty:
        print(f"[DEBUG] df_4h final post-indicators => {len(df_4h)}")
        df_4h.rename(columns={
            'open':'open_4h','high':'high_4h','low':'low_4h','close':'close_4h','volume':'volume_4h'
        }, inplace=True)
    else:
        print("[DEBUG] df_4h is empty, aggregator ends.")
        return pd.DataFrame()

    # --- 1d
    df_1d = load_df(symbol_id, '1d', start_date, end_date)
    df_1d = calc_indicators(df_1d, '1d')
    if not df_1d.empty:
        print(f"[DEBUG] df_1d final post-indicators => {len(df_1d)}")
        df_1d.rename(columns={
            'open':'open_1d','high':'high_1d','low':'low_1d','close':'close_1d','volume':'volume_1d'
        }, inplace=True)
    else:
        print("[DEBUG] df_1d is empty, aggregator ends.")
        return pd.DataFrame()

    # --- Merges
    df_1h_srt = df_1h.sort_index()
    df_4h_srt = df_4h.sort_index()
    df_1d_srt = df_1d.sort_index()

    print("[DEBUG] merging 1h <- 4h ...")
    df_merge = pd.merge_asof(df_1h_srt, df_4h_srt, left_index=True, right_index=True, direction='backward')
    print("[DEBUG] after 1h<-4h merge =>", len(df_merge))

    print("[DEBUG] merging (merge) <- 1d ...")
    df_merge = pd.merge_asof(df_merge.sort_index(), df_1d_srt, left_index=True, right_index=True, direction='backward')
    print("[DEBUG] after merge(...<-1d) =>", len(df_merge))

    df_merge.dropna(inplace=True)
    print("[DEBUG] after final dropna =>", len(df_merge))

    # --- SHIFT
    if df_merge.empty:
        print("[DEBUG] aggregator: merges => 0 rows => returning empty DF.")
        return df_merge

    # SHIFT-target
    df_merge['future_close_1h'] = df_merge['close_1h'].shift(-shift)
    df_merge['target_return_1h'] = (df_merge['future_close_1h'] - df_merge['close_1h']) / df_merge['close_1h']*100.0
    df_merge.dropna(inplace=True)
    df_merge.drop(columns=['future_close_1h'], inplace=True)
    print("[DEBUG] final SHIFT =>", len(df_merge))

    return df_merge

# =========================
# [F] TEST/MAIN
# =========================
if __name__ == "__main__":
    # Pas deze periode aan (bijv. jan/feb 2018) om te debuggen.
    train_start = "2016-01-01"
    train_end   = "2025-01-31 23:59:59"
    # of test
    # train_start = "2017-12-01"
    # train_end   = "2018-01-31 23:59:59"

    df_debug = build_multi_tf_df(
        symbol_id=1,
        start_date=train_start,
        end_date=train_end,
        shift=5
    )
    print("\n[DEBUG] final aggregator DF =>", len(df_debug), "rows.")
    if not df_debug.empty:
        print(df_debug.head(5))
