import mysql.connector
import pandas as pd
import ta  # pip install ta

# === [A] Databaseconfig ===
DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}


def load_ohlcv_from_db(symbol_id, interval='1h'):
    """
    Haalt ruwe OHLCV-data uit `market_data` (MySQL),
    gesorteerd op timestamp_ms ASC.
    """
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
    return df


# === [B] Indicator-functies per timeframe ===
def add_indicators_1h(df):
    """
    Voor 1h: RSI(14), MACD(12,26,9), ADX(14), Stoch(14,3,3),
    Bollinger(20,2), EMA(20), EMA(50), EMA(200), ATR(14)
    """
    # 1) RSI(14)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    # 2) MACD(12,26,9)
    df['macd'] = ta.trend.macd(df['close'], window_slow=26, window_fast=12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)

    # 3) ADX(14)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

    # 4) Stoch(14,3,3)
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'],
                                      window=14, smooth_window=3)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'],
                                             window=14, smooth_window=3)

    # 5) Bollinger(20,2)
    df['bb_mavg'] = ta.volatility.bollinger_mavg(df['close'], window=20)
    df['bb_hband'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
    df['bb_lband'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)

    # 6) EMA(20), EMA(50), EMA(200)
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()

    # 7) ATR(14)
    df['atr14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    # Verwijder NaN-rijen door rolling
    df.dropna(inplace=True)

    return df


def add_indicators_4h(df):
    """
    Voor 4h: RSI(14), MACD(12,26,9), ATR(14),
    EMA(50), EMA(200) (zoals in je overzicht).
    """
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    df['macd'] = ta.trend.macd(df['close'], window_slow=26, window_fast=12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)

    df['atr14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()

    df.dropna(inplace=True)
    return df


def add_indicators_1d(df):
    """
    Voor 1d: RSI(14), MACD(12,26,9), ATR(14), EMA(200)
    (eventueel kun je meer kolommen toevoegen)
    """
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    df['macd'] = ta.trend.macd(df['close'], window_slow=26, window_fast=12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)

    df['atr14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    df['ema_200'] = df['close'].ewm(span=200).mean()

    df.dropna(inplace=True)
    return df


# === [C] Target-functie (SHIFT) ===
def add_target_return(df, shift=5):
    """
    Maakt kolom 'target_return' = %verandering over SHIFT candles.
    """
    df['future_close'] = df['close'].shift(-shift)
    df['target_return'] = (df['future_close'] - df['close']) / df['close'] * 100.0

    # Drop de laatste SHIFT rijen (geen future)
    df.dropna(inplace=True)
    df.drop(columns=['future_close'], inplace=True)
    return df


# === [D] build_dataset() - alles samen ===
def build_dataset(symbol_id, interval='1h', shift=5):
    """
    1) Laadt ruwe OHLCV uit DB
    2) Voegt de juiste indicatoren toe (afh. van interval)
    3) Voegt target_return toe (shift=5 default)
    4) Returnt DataFrame klaar voor ML
    """
    # 1) OHLCV laden
    df = load_ohlcv_from_db(symbol_id, interval)

    if df.empty:
        print(f"[FOUT] Geen data voor symbol_id={symbol_id} interval={interval}")
        return df  # leeg

    # 2) Indicators afh. van timeframe
    if interval == '1h':
        df = add_indicators_1h(df)
    elif interval == '4h':
        df = add_indicators_4h(df)
    elif interval == '1d':
        df = add_indicators_1d(df)
    else:
        print(f"[WAARSCHUWING] Onbekend interval={interval}, geen indicatoren toegevoegd!")

    if df.empty:
        print(f"[FOUT] Na indicator-berekening geen rows over (NaN-drop).")
        return df

    # 3) SHIFT / target_return
    df = add_target_return(df, shift=shift)

    return df


# === [E] Test / Main ===
if __name__ == '__main__':
    # Voorbeeld: We pakken symbol_id=1 (bijv. BTC/EUR), interval='1h', SHIFT=5
    dataset_1h = build_dataset(symbol_id=1, interval='1h', shift=5)

    print(">>> 1h dataset kolommen:", dataset_1h.columns)
    print("Aantal rijen:", len(dataset_1h))
    print(dataset_1h.head(5))
    print(dataset_1h.tail(5))

    # Voorbeeld: 4h dataset, SHIFT=3
    dataset_4h = build_dataset(symbol_id=1, interval='4h', shift=3)
    print("\n>>> 4h dataset kolommen:", dataset_4h.columns)
    print("Aantal rijen:", len(dataset_4h))

    # Voorbeeld: 1d dataset, SHIFT=1
    dataset_1d = build_dataset(symbol_id=1, interval='1d', shift=1)
    print("\n>>> 1d dataset kolommen:", dataset_1d.columns)
    print("Aantal rijen:", len(dataset_1d))
