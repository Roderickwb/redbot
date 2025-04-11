"""
walk_forward_full.py

Dit script toont een volledig voorbeeld van:
 - Multi-timeframe aggregator (1h,4h,1d) + indicatoren
 - Maandelijkse walk-forward (van 2018-01 tot 2020-12, b.v.)
 - Param search (mini) met XGBoost
 - MSE op de test-maand loggen

WAARSCHUWING: Dit kan lang duren & veel RAM vereisen,
dus begin met korte periode en kleine param_grid.

Pas de aggregator code aan / importeer indien nodig.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import ta  # we gebruiken ta in aggregator-functies

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import mysql.connector

# === [A] DB CONFIG (PAS AAN) ===
DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}

# === [B] aggregator-achtige functies (1h,4h,1d) + indicators + merge ===
def load_df(symbol_id, interval, start_date, end_date):
    """
    Haalt ALLE rows (symbol_id, interval),
    zet in DF, to_numeric, to_datetime,
    slice [start_date, end_date].
    Ret: Pandas DataFrame
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
        return df

    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open','high','low','close','volume'], inplace=True)

    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    df = df.loc[start_date:end_date]
    return df

def calc_indicators(df, interval='1h'):
    """ RSI, MACD, ADX, Stoch, Bollinger. Dropna. """
    if df.empty:
        return df

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
    # Boll
    df['bb_mavg'] = ta.volatility.bollinger_mavg(df['close'], window=20)
    df['bb_hband'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
    df['bb_lband'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)

    df.dropna(inplace=True)
    return df

def build_multi_tf_df(symbol_id, start_date, end_date):
    """
    1) Load 1h,4h,1d in [start_date, end_date]
    2) calc_indicators
    3) merge_asof
    4) dropna
    5) return
    """
    df_1h = load_df(symbol_id, '1h', start_date, end_date)
    df_1h = calc_indicators(df_1h, '1h')
    if not df_1h.empty:
        # rename
        df_1h.rename(columns={
            'open':'open_1h','high':'high_1h','low':'low_1h','close':'close_1h','volume':'volume_1h'
        }, inplace=True)

    df_4h = load_df(symbol_id, '4h', start_date, end_date)
    df_4h = calc_indicators(df_4h, '4h')
    if not df_4h.empty:
        df_4h.rename(columns={
            'open':'open_4h','high':'high_4h','low':'low_4h','close':'close_4h','volume':'volume_4h'
        }, inplace=True)

    df_1d = load_df(symbol_id, '1d', start_date, end_date)
    df_1d = calc_indicators(df_1d, '1d')
    if not df_1d.empty:
        df_1d.rename(columns={
            'open':'open_1d','high':'high_1d','low':'low_1d','close':'close_1d','volume':'volume_1d'
        }, inplace=True)

    if df_1h.empty or df_4h.empty or df_1d.empty:
        # return empty => no merges
        return pd.DataFrame()

    # merges
    df_1h_srt = df_1h.sort_index()
    df_4h_srt = df_4h.sort_index()
    df_1d_srt = df_1d.sort_index()

    df_merge = pd.merge_asof(df_1h_srt, df_4h_srt, left_index=True, right_index=True, direction='backward')
    df_merge = pd.merge_asof(df_merge, df_1d_srt, left_index=True, right_index=True, direction='backward')

    df_merge.dropna(inplace=True)
    return df_merge

# === [C] SHIFT-Target (of ga je SHIFT=5 vast gebruiken?) ===
def add_shift_target(df, shift=5):
    """ target_return_1h = (close_1h(t+shift) - close_1h(t))/close_1h(t)*100 """
    if df.empty:
        return df
    df['future_close_1h'] = df['close_1h'].shift(-shift)
    df['target_return_1h'] = (df['future_close_1h'] - df['close_1h']) / df['close_1h'] * 100
    df.dropna(inplace=True)
    df.drop(columns=['future_close_1h'], inplace=True)
    return df

# === [D] De maandelijkse walk-forward routine ===
def monthly_walk_forward(symbol_id=1,
                         wf_start='2018-01-01',
                         wf_end='2020-12-31',
                         shift=5):
    """
    1) Maakt lijst van maand-begindata (bv. 2018-01, 2018-02, ...)
    2) For each testmaand:
       - train = [wf_start, test_start)
       - test  = [test_start, test_end)
       - build_multi_tf_df() op train, add_shift_target
         => X_train, y_train
       - (mini) param search (XGBoost)
       - retrain best model => predict test => MSE
       - log
    """
    dt_start = datetime.strptime(wf_start, "%Y-%m-%d")
    dt_end   = datetime.strptime(wf_end, "%Y-%m-%d")

    # Lijst maand-begindata
    dt_list = []
    current = dt_start
    while current <= dt_end:
        dt_list.append(current)
        current = current + relativedelta(months=1)

    results = []
    for i in range(1, len(dt_list)):
        test_start = dt_list[i]
        test_end   = test_start + relativedelta(months=1)
        if test_end > dt_end:
            test_end = dt_end

        train_start_str = dt_list[0].strftime("%Y-%m-%d")
        train_end_str   = (test_start - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        test_start_str  = test_start.strftime("%Y-%m-%d")
        test_end_str    = test_end.strftime("%Y-%m-%d")

        print(f"\n[WALK-FWD] Test-maand: {test_start_str} -> {test_end_str}")
        print(f"  Train-periode: {train_start_str} -> {train_end_str}")

        # Build train
        df_train = build_multi_tf_df(symbol_id, train_start_str, train_end_str)
        df_train = add_shift_target(df_train, shift=shift)
        if len(df_train) < 50:
            print("  [WARN] Te weinig train-data => skip")
            continue

        # Build test
        df_test = build_multi_tf_df(symbol_id, test_start_str, test_end_str)
        df_test = add_shift_target(df_test, shift=shift)
        if len(df_test) < 10:
            print("  [WARN] Te weinig test-data => skip")
            continue

        # X,y
        # We hebben 'target_return_1h' als y
        # en we droppen 'target_return_1h' + andere target col uit X
        drop_cols = ['target_return_1h']
        features = [c for c in df_train.columns if c not in drop_cols]

        X_train = df_train[features]
        y_train = df_train['target_return_1h']

        X_test = df_test[features]
        y_test = df_test['target_return_1h']

        # Param search
        param_grid = {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3,5],
            'subsample': [0.8,1.0]
        }
        xgb = XGBRegressor(n_estimators=50, random_state=42)
        grid = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=2,
            n_jobs=-1
        )
        print("  [INFO] Start param-search ...")
        grid.fit(X_train, y_train)

        best_params = grid.best_params_
        best_cv_mse = -grid.best_score_

        # Retrain best model
        final_model = XGBRegressor(n_estimators=50, random_state=42, **best_params)
        final_model.fit(X_train, y_train)

        # Predict test => MSE
        preds = final_model.predict(X_test)
        mse_test = mean_squared_error(y_test, preds)

        print(f"  [RESULT] MSE={mse_test:.4f}, best_params={best_params}")

        results.append({
            'test_month_start': test_start_str,
            'test_month_end': test_end_str,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'best_params': best_params,
            'cv_mse': best_cv_mse,
            'test_mse': mse_test
        })

    df_res = pd.DataFrame(results)
    return df_res


if __name__=="__main__":
    # Voorbeeld: monthly walk-forward van 2018-01-01 t/m 2020-12-31
    # SHIFT=5. Pas aan naar eigen periode.
    df_out = monthly_walk_forward(symbol_id=1,
                                  wf_start='2018-01-01',
                                  wf_end='2020-12-31',
                                  shift=5)
    print("\n=== WALK-FWD RESULTS ===")
    print(df_out)
    print("\nDone.")
