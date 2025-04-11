"""
my_full_script.py

Dit script:
1) Haalt 1h, 4h, 1d-data uit MySQL in een beperkte periode (om geheugengebruik te beperken).
2) Berekent alle indicatoren en de SHIFT-target per timeframe.
3) Merged ze tot één 'df_multi' DataFrame (multi-timeframe aggregator).
4) Doet een eenvoudige train/test-split en GridSearch met XGBoost.

Installeer nodig:
    pip install mysql-connector-python
    pip install ta
    pip install xgboost
    pip install scikit-learn
    pip install pandas
    pip install numpy
"""

import mysql.connector
import pandas as pd
import numpy as np
import ta  # van pip install ta
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# === [A] Databaseconfig (pas aan naar jouw DB-credentials) ===
DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}


# ---------------------------------------------------------------------
# [1] FUNCTIES VOOR DATA LADEN (MET PERIODE-FILTER)
# ---------------------------------------------------------------------
def load_ohlcv_from_db(symbol_id, interval='1h', start_date=None, end_date=None):
    """
    Haalt ruwe OHLCV-data uit 'market_data' (MySQL) alléén in [start_date, end_date].
    start_date/end_date verwacht strings in 'YYYY-MM-DD' (of 'YYYY-MM-DD HH:MM:SS') formaat.
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    # Bouw query:
    # Timestamps in DB zijn in milliseconden => we filteren op 'timestamp_ms' door te converteren
    # KISS: We laden alles op, en filtern in Python, OF we filteren direct in SQL.
    # Hier laten we direct in Python filteren (simpel, maar eventueel minder efficiënt).
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

    if df.empty:
        return df

    # Convert kolommen naar float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

    # Maak datetime-index
    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    # Filter in Python op start_date / end_date
    if start_date:
        df = df.loc[start_date:]
    if end_date:
        df = df.loc[:end_date]

    return df


def add_indicators_1h(df):
    """
    Voor 1h: RSI(14), MACD(12,26,9), ADX(14), Stoch(14,3,3),
    Bollinger(20,2), EMA(20), EMA(50), EMA(200), ATR(14)
    """
    if df.empty:
        return df

    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'], window_slow=26, window_fast=12)
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
    return df


def add_indicators_4h(df):
    """
    Voor 4h: RSI(14), MACD(12,26,9), ATR(14), EMA(50), EMA(200)
    """
    if df.empty:
        return df

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
    """
    if df.empty:
        return df

    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'], window_slow=26, window_fast=12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    df['atr14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['ema_200'] = df['close'].ewm(span=200).mean()

    df.dropna(inplace=True)
    return df


def add_target_return(df, shift=5):
    """
    target_return = ((close(t+shift) - close(t)) / close(t)) * 100
    """
    df['future_close'] = df['close'].shift(-shift)
    df['target_return'] = (df['future_close'] - df['close']) / df['close'] * 100.0
    df.dropna(inplace=True)
    df.drop(columns=['future_close'], inplace=True)
    return df


def build_dataset(symbol_id, interval='1h', shift=5, start_date=None, end_date=None):
    """
    Laad DB -> filter op periode -> bereken indicatoren -> target
    """
    df = load_ohlcv_from_db(symbol_id, interval, start_date, end_date)
    if df.empty:
        print(f"[WARN] No data after filtering for {interval}")
        return df

    # Indicatoren
    if interval == '1h':
        df = add_indicators_1h(df)
    elif interval == '4h':
        df = add_indicators_4h(df)
    elif interval == '1d':
        df = add_indicators_1d(df)

    if df.empty:
        return df

    # SHIFT
    df = add_target_return(df, shift)
    return df


# ---------------------------------------------------------------------
# [2] MULTI-TIMEFRAME AGGREGATOR
# ---------------------------------------------------------------------
def build_multi_timeframe_dataset(symbol_id=1, shift=5, start_date=None, end_date=None):
    """
    Haalt 1h, 4h, 1d data (beperkte periode) en merge_asof.
    target_return komt uit de 1h-data (kolom target_return_1h),
    maar we houden ook target_return_4h en _1d als je die later wilt testen.
    """
    print(">> [INFO] build_multi_timeframe_dataset start...")

    df_1h = build_dataset(symbol_id, '1h', shift, start_date, end_date)
    df_4h = build_dataset(symbol_id, '4h', shift, start_date, end_date)
    df_1d = build_dataset(symbol_id, '1d', shift, start_date, end_date)

    if df_1h.empty or df_4h.empty or df_1d.empty:
        print("[ERROR] One of the dataframes is empty (1h/4h/1d). Returning empty.")
        return pd.DataFrame()

    # Hernoemen om suffix _1h, _4h, _1d
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

    # Andere indicator-kolommen renamen
    for col in list(df_1h.columns):
        if col not in ['timestamp_ms', 'open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h', 'target_return_1h']:
            df_1h.rename(columns={col: col + "_1h"}, inplace=True)

    for col in list(df_4h.columns):
        if col not in ['timestamp_ms', 'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h', 'target_return_4h']:
            df_4h.rename(columns={col: col + "_4h"}, inplace=True)

    for col in list(df_1d.columns):
        if col not in ['timestamp_ms', 'open_1d', 'high_1d', 'low_1d', 'close_1d', 'volume_1d', 'target_return_1d']:
            df_1d.rename(columns={col: col + "_1d"}, inplace=True)

    # Merge
    df_1h_sorted = df_1h.sort_index()
    df_4h_sorted = df_4h.sort_index()
    df_1d_sorted = df_1d.sort_index()

    df_merge = pd.merge_asof(
        df_1h_sorted, df_4h_sorted,
        left_index=True, right_index=True,
        direction='backward'
    )
    df_merge = pd.merge_asof(
        df_merge, df_1d_sorted,
        left_index=True, right_index=True,
        direction='backward'
    )

    df_merge.dropna(inplace=True)
    print(">> [INFO] build_multi_timeframe_dataset DONE. Rows =", len(df_merge))
    return df_merge


# ---------------------------------------------------------------------
# [3] EENVOUDIGE PARAM SEARCH MET XGBOOST
# ---------------------------------------------------------------------
def simple_param_search(symbol_id=1, start_date='2020-01-01', end_date='2020-03-01'):
    """
    1) Bouw de multi-timeframe dataset (alleen 2 maanden!)
    2) Splits in train/test (80/20)
    3) Doe GridSearchCV op XGBoost
    4) Print beste params, MSE op test.
    """
    print(f"\n>> [INFO] building multi-timeframe dataset for {start_date} - {end_date} ...")
    df = build_multi_timeframe_dataset(symbol_id=symbol_id, shift=5,
                                       start_date=start_date, end_date=end_date)
    if df.empty:
        print("[FOUT] Geen data, kan niet verder.")
        return

    print("Data in periode:", len(df), "rows")

    # TARGET = target_return_1h
    # De andere 2 (4h, 1d) laten we weg
    drop_targets = ['target_return_1h', 'target_return_4h', 'target_return_1d']
    target_col = 'target_return_1h'

    X = df.drop(columns=drop_targets)
    y = df[target_col]

    # Eenvoudige train/test-split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print("Train rows:", len(X_train), " | Test rows:", len(X_test))

    # Kleine hyperparam-grid
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
    }

    from sklearn.model_selection import GridSearchCV
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=2,  # minder folds => sneller
        n_jobs=-1
    )
    print("\n[INFO] Start GridSearchCV ...")
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    best_score = -grid.best_score_  # neg MSE => draai om
    print(f"[RESULT] best_params={best_params}, best_cv_MSE={best_score:.4f}")

    # Test-set MSE
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    mse_test = np.mean((y_test - y_pred) ** 2)
    print(f"[RESULT] test-set MSE={mse_test:.4f}")


# ---------------------------------------------------------------------
# [4] MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # We doen een test op 2 maanden (1 jan 2020 - 1 mrt 2020).
    # Pas dit gerust aan om periode te vergroten.
    simple_param_search(
        symbol_id=1,
        start_date='2020-01-01',
        end_date='2020-03-01'
    )

    print("\nDone.")
