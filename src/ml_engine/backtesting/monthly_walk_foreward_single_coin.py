#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mysql.connector
import pandas as pd
import ta
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# ===================== DB-CONFIG =====================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}

# ===================== 1) DATASET-BUILDER =====================
def load_ohlcv_from_db(symbol_id=1, interval='1h'):
    """
    Laadt ruwe OHLCV-data (open, high, low, close, volume) uit DB,
    sorteert op timestamp_ms. Converteert kolommen naar float en dropt NaN.
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
    if df.empty:
        return df

    # Convert DECIMAL/None -> float, drop NaN
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float, errors='ignore')
    df.dropna(subset=['open','high','low','close','volume'], inplace=True)

    return df

def add_indicators_1h(df):
    """
    Voor 1h: RSI, MACD, ADX, Stoch, Bollinger, EMA20/50/200, ATR14
    """
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'], 26, 12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

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
    """
    Maakt kolom 'target_return' = %verandering (close[t+shift]/close[t] -1)*100
    Dropt de laatste SHIFT rijen (geen future).
    """
    df['future_close'] = df['close'].shift(-shift)
    df['target_return'] = (df['future_close'] - df['close']) / df['close'] * 100.0
    df.dropna(inplace=True)
    df.drop(columns=['future_close'], inplace=True)
    return df

def build_dataset(symbol_id=1, interval='1h', shift=5):
    df = load_ohlcv_from_db(symbol_id, interval)
    if df.empty:
        print(f"[FOUT] Geen data voor symbol_id={symbol_id}, interval={interval}")
        return df

    df = add_indicators_1h(df)
    if df.empty:
        print("[FOUT] Geen rows over na indicator-berekening.")
        return df

    df = add_target_return(df, shift=shift)
    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df.sort_values('datetime', inplace=True)
    return df


# ===================== 2) WALK-FORWARD MET PARAM SEARCH =====================
def walk_forward_monthly_with_param_search(df):
    """
    Voor elke maand in df:
      - train_df = alle voorgaande maanden
      - test_df  = de 'huidige' maand
    Voert per train_df een mini gridsearch uit over XGBoost hyperparameters,
    traint het beste model, test op test_df -> MSE.
    Logt MSE en beste params.
    """
    df['year_month'] = df['datetime'].dt.to_period('M')
    unique_months = df['year_month'].unique().tolist()

    # Een kleine param-grid (je kunt deze uitbreiden)
    param_grid = {
        'learning_rate': [0.01, 0.05],
        'max_depth': [4, 6],
        'subsample': [0.8, 1.0]
    }

    results = []
    for i in range(1, len(unique_months)):
        train_months = unique_months[:i]
        test_month = unique_months[i]

        train_df = df[df['year_month'].isin(train_months)]
        test_df = df[df['year_month'] == test_month]

        if len(train_df) < 50 or len(test_df) < 10:
            # Te weinig data
            continue

        exclude_cols = ['timestamp_ms','datetime','year_month','target_return']
        features = [c for c in df.columns if c not in exclude_cols]

        X_train = train_df[features]
        y_train = train_df['target_return']
        X_test = test_df[features]
        y_test = test_df['target_return']

        # ========== PARAM SEARCH OP train_df ==========
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',  # want we willen MSE minimaliseren
            cv=3,              # 3-fold cross validation
            verbose=0,         # zet naar 1 als je meer logging wilt
            n_jobs=-1          # gebruik multi-core
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_  # model met beste params
        best_params = grid_search.best_params_

        # ========== Test op test_df ==========
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


# ===================== 3) MAIN: DE-FACTO SCRIPT =====================
if __name__ == '__main__':
    SYMBOL_ID = 1
    INTERVAL = '1h'
    SHIFT = 5

    df = build_dataset(symbol_id=SYMBOL_ID, interval=INTERVAL, shift=SHIFT)
    # Filter periode 2021-01-01 t/m 2022-12-31
    df = df[(df['datetime'] >= '2021-01-01') & (df['datetime'] < '2023-01-01')]
    # Als je wilt, sorteer nog even
    df.sort_values('datetime', inplace=True)
    if df.empty:
        print("[STOP] build_dataset gaf geen data terug.")
        exit(0)

    wf_results = walk_forward_monthly_with_param_search(df)

    print("\n=== MAANDELIJKSE WALK-FORWARD met Param Search ===")
    for r in wf_results:
        print(f"Test maand={r['test_month']}, "
              f"MSE={r['mse']:.4f}, "
              f"TrainSize={r['train_size']}, "
              f"TestSize={r['test_size']}, "
              f"best_params={r['best_params']}")
    print(f"\nTotaal geteste maanden: {len(wf_results)}")
