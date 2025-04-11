import mysql.connector
import pandas as pd
import ta  # pip install ta
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# === [A] Databaseconfig (PAS AAN als nodig) ===
DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}

# === [B] load_df: laadt alle rows (symbol_id, interval) => Pandas => filter
def load_df(symbol_id, interval, start_date, end_date):
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
        return df

    # Naar float
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open','high','low','close','volume'], inplace=True)

    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    # Slicen
    df = df.loc[start_date:end_date]
    return df

def calc_indicators(df, interval='1h'):
    """
    RSI, MACD, ADX, Stoch, Bollinger => dropna
    """
    if df.empty:
        return df

    df['rsi'] = ta.momentum.rsi(df['close'], 14)
    df['macd'] = ta.trend.macd(df['close'], window_slow=26, window_fast=12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], 14, 3)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], 14, 3)
    df['bb_mavg'] = ta.volatility.bollinger_mavg(df['close'], 20)
    df['bb_hband'] = ta.volatility.bollinger_hband(df['close'], 20, 2)
    df['bb_lband'] = ta.volatility.bollinger_lband(df['close'], 20, 2)

    df.dropna(inplace=True)
    return df


def build_multi_tf_df(symbol_id, start_date, end_date):
    """
    1h => df_1h
    4h => df_4h
    1d => df_1d
    indicators => rename => merge_asof => dropna => return
    """
    # 1h
    df_1h = load_df(symbol_id, '1h', start_date, end_date)
    df_1h = calc_indicators(df_1h, '1h')
    if not df_1h.empty:
        df_1h.rename(columns={
            'open': 'open_1h',
            'high': 'high_1h',
            'low': 'low_1h',
            'close': 'close_1h',
            'volume': 'volume_1h'
        }, inplace=True)

    # 4h
    df_4h = load_df(symbol_id, '4h', start_date, end_date)
    df_4h = calc_indicators(df_4h, '4h')
    if not df_4h.empty:
        df_4h.rename(columns={
            'open': 'open_4h',
            'high': 'high_4h',
            'low': 'low_4h',
            'close': 'close_4h',
            'volume': 'volume_4h'
        }, inplace=True)

    # 1d
    df_1d = load_df(symbol_id, '1d', start_date, end_date)
    df_1d = calc_indicators(df_1d, '1d')
    if not df_1d.empty:
        df_1d.rename(columns={
            'open': 'open_1d',
            'high': 'high_1d',
            'low': 'low_1d',
            'close': 'close_1d',
            'volume': 'volume_1d'
        }, inplace=True)

    # Check empties
    if df_1h.empty or df_4h.empty or df_1d.empty:
        # geen data => return lege DF
        return pd.DataFrame()

    # merges
    df_1h_srt = df_1h.sort_index()
    df_4h_srt = df_4h.sort_index()
    df_1d_srt = df_1d.sort_index()

    df_merge = pd.merge_asof(
        df_1h_srt, df_4h_srt,
        left_index=True, right_index=True,
        direction='backward'
    )
    df_merge = pd.merge_asof(
        df_merge, df_1d_srt,
        left_index=True, right_index=True,
        direction='backward'
    )

    df_merge.dropna(inplace=True)
    return df_merge


def add_shift_target(df, shift=5):
    """
    df['future_close_1h'] => shift -5
    df['target_return_1h'] => (fut - now)/now*100
    dropna => drop col
    """
    if df.empty:
        return df
    df['future_close_1h'] = df['close_1h'].shift(-shift)
    df['target_return_1h'] = (df['future_close_1h'] - df['close_1h']) / df['close_1h'] * 100.0
    df.dropna(inplace=True)
    df.drop(columns=['future_close_1h'], inplace=True)
    return df


def single_iteration_small():
    """
    1 iteration:
    Train= 2018-01-01..2018-01-31
    Test=  2018-02-01..2018-02-28
    SHIFT=5
    Minimale param search => 1 combo => XGB => MSE test
    """
    train_start='2017-01-01'
    train_end='2018-01-31 23:59:59'
    test_start='2018-02-01'
    test_end='2019-02-28 23:59:59'
    SHIFT=5

    print("\n[DEBUG] Building TRAIN data ...")
    df_train = build_multi_tf_df(1, train_start, train_end)
    df_train = add_shift_target(df_train, SHIFT)
    print("Train data rows =", len(df_train))

    print("\n[DEBUG] Building TEST data ...")
    df_test = build_multi_tf_df(1, test_start, test_end)
    df_test = add_shift_target(df_test, SHIFT)
    print("Test data rows =", len(df_test))

    if len(df_train)<50 or len(df_test)<10:
        print("[WARN] Not enough data => skip.")
        return

    # X,y
    drop_cols = ['target_return_1h']
    feats = [c for c in df_train.columns if c not in drop_cols]
    X_train = df_train[feats]
    y_train = df_train['target_return_1h']

    X_test  = df_test[feats]
    y_test  = df_test['target_return_1h']

    # Minimale param search
    param_grid = [
        {'learning_rate':[0.1], 'max_depth':[3]}  # 1 combo
    ]
    xgb = XGBRegressor(n_estimators=50, random_state=42)

    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=1,  # geen cross-val
        n_jobs=-1
    )
    print("\n[DEBUG] Start minimal param search (only 1 combo, cv=1) ...")
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    print("[DEBUG] best_params =", best_params)
    final_model = XGBRegressor(n_estimators=50, random_state=42, **best_params)
    final_model.fit(X_train, y_train)

    preds = final_model.predict(X_test)
    mse_test = mean_squared_error(y_test, preds)
    print(f"[RESULT] Single iteration MSE={mse_test:.4f}, best_params={best_params}")


if __name__=="__main__":
    single_iteration_small()
    print("\nDone.")
