#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mysql.connector
import pandas as pd
import ta
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}


def load_ohlcv_from_db(symbol_id, interval='1h'):
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
    if df.empty:
        return df

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float, errors='ignore')
    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

    return df


def add_indicators_1h(df):
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'], 26, 12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)

    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], 14, 3)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], 14, 3)

    df['bb_mavg'] = ta.volatility.bollinger_mavg(df['close'], 20)
    df['bb_hband'] = ta.volatility.bollinger_hband(df['close'], 20, 2)
    df['bb_lband'] = ta.volatility.bollinger_lband(df['close'], 20, 2)

    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()

    df['atr14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)

    df.dropna(inplace=True)
    return df


def add_target_return(df, shift=5):
    df['future_close'] = df['close'].shift(-shift)
    df['target_return'] = (df['future_close'] - df['close']) / df['close'] * 100.0
    df.dropna(inplace=True)
    df.drop(columns=['future_close'], inplace=True)
    return df


def build_dataset(symbol_id, interval='1h', shift=5):
    df = load_ohlcv_from_db(symbol_id, interval)
    if df.empty:
        print(f"[FOUT] Geen data voor coin {symbol_id}")
        return df

    df = add_indicators_1h(df)
    if df.empty:
        print(f"[FOUT] Na indicator-berekening geen rows over voor coin {symbol_id}")
        return df

    df = add_target_return(df, shift)
    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df.sort_values('datetime', inplace=True)
    return df


def walk_forward_monthly_with_param_search(df):
    df['year_month'] = df['datetime'].dt.to_period('M')
    unique_months = df['year_month'].unique().tolist()

    # Uitgebreider param-grid
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 6, 8],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.7, 1.0],
        'reg_lambda': [1.0, 5.0]
    }

    results = []
    from sklearn.model_selection import GridSearchCV

    for i in range(1, len(unique_months)):
        train_months = unique_months[:i]
        test_month = unique_months[i]

        train_df = df[df['year_month'].isin(train_months)]
        test_df = df[df['year_month'] == test_month]

        if len(train_df) < 50 or len(test_df) < 10:
            continue

        exclude_cols = ['timestamp_ms', 'datetime', 'year_month', 'target_return']
        features = [c for c in df.columns if c not in exclude_cols]

        X_train = train_df[features]
        y_train = train_df['target_return']
        X_test = test_df[features]
        y_test = test_df['target_return']

        xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=42)
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=3,
            verbose=0,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        preds = best_model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        results.append({
            'test_month': str(test_month),
            'mse': mse,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'best_params': best_params
        })

    return results


if __name__ == '__main__':
    coin_list = [1, 2]  # BTC, ETH
    interval = '1h'
    SHIFT = 5
    START_DATE = '2021-01-01'
    END_DATE = '2023-01-01'

    for sym_id in coin_list:
        print(f"\n=== [Coin={sym_id}] ===")
        df = build_dataset(symbol_id=sym_id, interval=interval, shift=SHIFT)
        if df.empty:
            print(f"[STOP] Geen dataset voor sym={sym_id}")
            continue

        # Korte periode
        df = df[(df['datetime'] >= START_DATE) & (df['datetime'] < END_DATE)]
        df.sort_values('datetime', inplace=True)

        if len(df) < 1000:
            print("[WAARSCHUWING] Weinig data na filtering.")

        wf_results = walk_forward_monthly_with_param_search(df)

        print(f"=== MAANDRESULTATEN (Coin={sym_id}) ===")
        for r in wf_results:
            print(f"  maand={r['test_month']}, mse={r['mse']:.3f}, "
                  f"train={r['train_size']}, test={r['test_size']}, params={r['best_params']}")
        print(f"Totaal testmaanden: {len(wf_results)}")
