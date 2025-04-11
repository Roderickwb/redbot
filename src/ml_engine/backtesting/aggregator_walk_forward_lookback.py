import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import ta  # pip install ta
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ======================
# [A] DATABASECONFIG - PAS AAN
# ======================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}

# ======================
# [B] AGGREGATOR-FUNCTIES
# ======================
def load_ohlcv(symbol_id, interval, start_date, end_date):
    """
    Haalt alle OHLCV van 'market_data' in [start_date, end_date], gesorteerd op timestamp_ms ASC.
    Returned DataFrame(index=datetime).
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

    # Filter in python
    df = df.loc[start_date:end_date]
    return df

def calc_indicators(df, interval='1h'):
    """
    Berekent RSI(14), MACD(12,26,9), ADX(14), Stoch(14,3,3), Boll(20,2).
    Dropna() na berekening.
    """
    if df.empty:
        return df

    # RSI(14)
    df['rsi'] = ta.momentum.rsi(df['close'], 14)
    # MACD(12,26,9)
    df['macd'] = ta.trend.macd(df['close'], 26, 12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    # ADX(14)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
    # Stoch(14,3,3)
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], 14, 3)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], 14, 3)
    # Boll(20,2)
    df['bb_mavg'] = ta.volatility.bollinger_mavg(df['close'], 20)
    df['bb_hband'] = ta.volatility.bollinger_hband(df['close'], 20, 2)
    df['bb_lband'] = ta.volatility.bollinger_lband(df['close'], 20, 2)

    df.dropna(inplace=True)
    return df

def build_multi_tf_df(symbol_id, start_date, end_date, shift=5):
    """
    1) load_ohlcv voor 1h, 4h, 1d
    2) calc_indicators => dropna
    3) rename
    4) merges => dropna
    5) SHIFT => target_return_1h
    Returned DF met features + target_return_1h
    """
    # --- 1h
    df_1h = load_ohlcv(symbol_id, '1h', start_date, end_date)
    df_1h = calc_indicators(df_1h, '1h')
    if df_1h.empty: return pd.DataFrame()
    df_1h.rename(columns={
        'open': 'open_1h','high': 'high_1h','low': 'low_1h','close': 'close_1h','volume': 'volume_1h'
    }, inplace=True)

    # --- 4h
    df_4h = load_ohlcv(symbol_id, '4h', start_date, end_date)
    df_4h = calc_indicators(df_4h, '4h')
    if df_4h.empty: return pd.DataFrame()
    df_4h.rename(columns={
        'open': 'open_4h','high': 'high_4h','low': 'low_4h','close': 'close_4h','volume': 'volume_4h'
    }, inplace=True)

    # --- 1d
    df_1d = load_ohlcv(symbol_id, '1d', start_date, end_date)
    df_1d = calc_indicators(df_1d, '1d')
    if df_1d.empty: return pd.DataFrame()
    df_1d.rename(columns={
        'open': 'open_1d','high': 'high_1d','low': 'low_1d','close': 'close_1d','volume': 'volume_1d'
    }, inplace=True)

    # merges
    df_1h_srt = df_1h.sort_index()
    df_4h_srt = df_4h.sort_index()
    df_1d_srt = df_1d.sort_index()

    df_merge = pd.merge_asof(df_1h_srt, df_4h_srt, left_index=True, right_index=True, direction='backward')
    df_merge = pd.merge_asof(df_merge.sort_index(), df_1d_srt, left_index=True, right_index=True, direction='backward')
    df_merge.dropna(inplace=True)
    if df_merge.empty: return df_merge

    # SHIFT => target_return_1h
    df_merge['future_close_1h'] = df_merge['close_1h'].shift(-shift)
    df_merge['target_return_1h'] = (df_merge['future_close_1h'] - df_merge['close_1h'])/df_merge['close_1h']*100.0
    df_merge.dropna(inplace=True)
    df_merge.drop(columns=['future_close_1h'], inplace=True)

    return df_merge

# ======================
# [C] LOOKBACK-FUNCTIE VOOR TEST DATA
# ======================
def build_test_data_with_lookback(symbol_id, test_start_dt, test_end_dt, shift=5, lookback_months=2):
    """
    Laadt aggregator met extra 'lookback_months' voor indicatorwarmup.
    Returnt alleen de test-slice [test_start_dt, test_end_dt] van die aggregator DF.
    """
    # Bepaal lookback start => bv. test_start_dt - 2 maanden
    lookback_start = test_start_dt - relativedelta(months=lookback_months)
    lookback_start_str = lookback_start.strftime("%Y-%m-%d")
    test_end_str = test_end_dt.strftime("%Y-%m-%d %H:%M:%S")

    # Aggregator
    df_big = build_multi_tf_df(symbol_id, lookback_start_str, test_end_str, shift=shift)
    if df_big.empty:
        return df_big

    # Filter => alleen de testperiode
    df_test = df_big.loc[test_start_dt : test_end_dt]
    return df_test

# ======================
# [D] MONTHLY WALK-FORWARD
# ======================
def monthly_walk_forward_with_lookback(
    symbol_id=1,
    wf_start='2019-01-01',
    wf_end='2020-12-31',
    shift=5
):
    """
    1) Lijst van maand-starts van wf_start..wf_end
    2) i in [1..n]:
       - test_start = month_list[i]
       - test_end = test_start + 1 maand
       - train = [wf_start..test_start)
       - test = [test_start.. test_end)
         * let op => test aggregator krijgt lookback!
       - random search param => best model => predict => metrics
    Return DataFrame met testmaand + metrics
    """
    dt_start = datetime.strptime(wf_start, "%Y-%m-%d")
    dt_end   = datetime.strptime(wf_end, "%Y-%m-%d")

    month_list = []
    current = dt_start.replace(day=1)
    while current <= dt_end:
        month_list.append(current)
        current = current + relativedelta(months=1)

    results = []
    for i in range(1, len(month_list)):
        test_start_dt = month_list[i]
        test_end_dt   = test_start_dt + relativedelta(months=1)
        if test_end_dt > dt_end:
            test_end_dt = dt_end

        # train-range = [wf_start.. test_start_dt)
        train_start_str = wf_start
        train_end_str = (test_start_dt - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")

        test_start_str = test_start_dt.strftime("%Y-%m-%d")
        test_end_str   = test_end_dt.strftime("%Y-%m-%d")

        print(f"\n[WALK-FWD] Test = {test_start_str}..{test_end_str}, Train = {train_start_str}..{train_end_str}")

        # Build train
        df_train = build_multi_tf_df(symbol_id, train_start_str, train_end_str, shift=shift)
        if df_train.empty or len(df_train)<50:
            print(f"[WARN] Te weinig train-data => skip {test_start_str}")
            continue

        # Build test => let op lookback
        df_test = build_test_data_with_lookback(symbol_id, test_start_dt, test_end_dt, shift=shift, lookback_months=2)
        if df_test.empty or len(df_test)<10:
            print(f"[WARN] test DF<10 rows => skip {test_start_str}")
            continue

        # Features
        drop_cols = [c for c in df_train.columns if c.startswith('target_return_')]
        feature_cols = [c for c in df_train.columns if c not in drop_cols]

        y_train = df_train['target_return_1h']
        X_train = df_train[feature_cols]
        y_test  = df_test['target_return_1h']
        X_test  = df_test[feature_cols]

        # Hyperparam random search
        param_dist = {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3,5],
            'subsample': [0.8, 1.0],
        }
        xgb_model = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')

        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_dist,
            n_iter=4,  # test 4 combos
            scoring='neg_mean_squared_error',
            cv=2,  # cv=2
            n_jobs=-1,
            random_state=42
        )
        print("   [INFO] RandomSearch start ...")
        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_
        best_score_cv = -random_search.best_score_

        final_model = XGBRegressor(
            n_estimators=100,
            random_state=42,
            objective='reg:squarederror',
            **best_params
        )
        final_model.fit(X_train, y_train)

        preds = final_model.predict(X_test)
        mse_test = mean_squared_error(y_test, preds)
        mae_test = mean_absolute_error(y_test, preds)
        r2_test  = r2_score(y_test, preds)

        print(f"   [RESULT] MSE={mse_test:.4f}, MAE={mae_test:.4f}, R2={r2_test:.3f}, best={best_params}")

        results.append({
            'test_month': test_start_str,
            'train_size': len(df_train),
            'test_size': len(df_test),
            'best_params': best_params,
            'cv_mse': best_score_cv,
            'mse_test': mse_test,
            'mae_test': mae_test,
            'r2_test': r2_test
        })

    df_res = pd.DataFrame(results)
    return df_res

# ======================
# [E] MAIN
# ======================
if __name__=="__main__":
    # Voorbeeld: walk-forward 2019-01-01..2020-12-31, SHIFT=5
    df_out = monthly_walk_forward_with_lookback(
        symbol_id=1,
        wf_start='2018-01-01',
        wf_end='2019-12-31',
        shift=5
    )
    print("\n=== WALK-FWD RESULTS ===")
    print(df_out)
    print("Done.")
