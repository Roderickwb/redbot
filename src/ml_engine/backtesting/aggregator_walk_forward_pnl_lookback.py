import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import ta
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# ======================
# [A] DATABASECONFIG (PAS AAN)
# ======================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}

# ======================
# [B] LOADDATA + INDICATORS
# ======================
def load_ohlcv(symbol_id, interval, start_date, end_date):
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

    df = df.loc[start_date:end_date]
    return df

def calc_indicators(df):
    if df.empty: return df
    # RSI(14)
    df['rsi'] = ta.momentum.rsi(df['close'], 14)
    # MACD(26,12,9)
    df['macd'] = ta.trend.macd(df['close'], 26, 12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    # ADX(14)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
    # Stoch(14,3)
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
    1h,4h,1d => merges => SHIFT => target_return_1h
    """
    # 1h
    df_1h = load_ohlcv(symbol_id, '1h', start_date, end_date)
    df_1h = calc_indicators(df_1h)
    if df_1h.empty: return df_1h
    df_1h.rename(columns={
        'open':'open_1h','high':'high_1h','low':'low_1h','close':'close_1h','volume':'volume_1h'
    }, inplace=True)

    # 4h
    df_4h = load_ohlcv(symbol_id, '4h', start_date, end_date)
    df_4h = calc_indicators(df_4h)
    if df_4h.empty: return df_4h
    df_4h.rename(columns={
        'open':'open_4h','high':'high_4h','low':'low_4h','close':'close_4h','volume':'volume_4h'
    }, inplace=True)

    # 1d
    df_1d = load_ohlcv(symbol_id, '1d', start_date, end_date)
    df_1d = calc_indicators(df_1d)
    if df_1d.empty: return df_1d
    df_1d.rename(columns={
        'open':'open_1d','high':'high_1d','low':'low_1d','close':'close_1d','volume':'volume_1d'
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

# =============
# [C] LOOKBACK-FUNCTIE VOOR TEST
# =============
def build_test_data_with_lookback(symbol_id, test_start_dt, test_end_dt, shift=5, lookback_months=2):
    """
    Laadt aggregator met extra 'lookback_months' als start,
    return alleen [test_start_dt..test_end_dt].
    """
    lookback_start = test_start_dt - relativedelta(months=lookback_months)
    lookback_start_str = lookback_start.strftime("%Y-%m-%d")
    test_end_str = test_end_dt.strftime("%Y-%m-%d %H:%M:%S")

    df_big = build_multi_tf_df(symbol_id, lookback_start_str, test_end_str, shift=shift)
    if df_big.empty:
        return df_big

    # Filter op echte test-slice
    df_test = df_big.loc[test_start_dt : test_end_dt]
    return df_test

# =================
# [D] Simuleer PnL
# =================
def simulate_pnl(df, shift=5, threshold=1.0, fee=0.004):
    """
    'pred' kolom in df => als > +1% => long => (close[i+shift]/close[i]-1) - fee
                        als < -1% => short => (1 - close[i+shift]/close[i]) - fee
                        anders 0
    sum => total_pnl
    """
    if len(df) < shift: return 0.0
    pnl_list = []
    c_close = df['close_1h'].values
    preds = df['pred'].values

    for i in range(len(df)-shift):
        pred = preds[i]
        c_i = c_close[i]
        c_future = c_close[i+shift]
        if pred > threshold:
            ret = (c_future/c_i)-1.0
            net_ret = ret - fee
            pnl_list.append(net_ret)
        elif pred < -threshold:
            ret = (1.0 - c_future/c_i)
            net_ret = ret - fee
            pnl_list.append(net_ret)
        else:
            pnl_list.append(0.0)
    total_pnl = np.sum(pnl_list)
    return total_pnl


# =================
# [E] MONTHLY WF
# =================
def monthly_walk_forward_pnl_lookback(
    symbol_id=1,
    wf_start='2019-01-01',
    wf_end='2020-12-31',
    shift=5
):
    dt_start = datetime.strptime(wf_start, "%Y-%m-%d")
    dt_end   = datetime.strptime(wf_end, "%Y-%m-%d")

    # Genereer maand-starts
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

        train_start_str = wf_start
        train_end_str = (test_start_dt - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        test_start_str = test_start_dt.strftime("%Y-%m-%d")
        test_end_str   = test_end_dt.strftime("%Y-%m-%d")

        print(f"\n[WALK-FWD] Test={test_start_str}..{test_end_str}, Train={train_start_str}..{train_end_str}")

        # BUILD TRAIN
        df_train = build_multi_tf_df(symbol_id, train_start_str, train_end_str, shift=shift)
        if df_train.empty or len(df_train)<50:
            print("[WARN] train empty => skip")
            continue

        # BUILD TEST (met lookback)
        df_test = build_test_data_with_lookback(symbol_id, test_start_dt, test_end_dt, shift=shift, lookback_months=2)
        if df_test.empty or len(df_test)<10:
            print("[WARN] test empty => skip")
            continue

        # X,y
        drop_train = [c for c in df_train.columns if c.startswith('target_return_')]
        feats_train = [c for c in df_train.columns if c not in drop_train]

        y_train = df_train['target_return_1h']
        X_train = df_train[feats_train]

        # Param-dist
        from sklearn.model_selection import RandomizedSearchCV
        param_dist = {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3,5],
            'subsample': [0.8, 1.0],
        }
        xgb_model = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')

        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_dist,
            n_iter=3,
            scoring='neg_mean_squared_error',
            cv=2,
            n_jobs=-1,
            random_state=42
        )
        print("  [INFO] random search param ...")
        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_
        best_model = XGBRegressor(
            n_estimators=100, random_state=42, objective='reg:squarederror', **best_params
        )
        best_model.fit(X_train, y_train)

        # PRED TEST
        drop_test = [c for c in df_test.columns if c.startswith('target_return_')]
        feats_test = [c for c in df_test.columns if c not in drop_test]

        X_test = df_test[feats_test]
        preds = best_model.predict(X_test)

        df_test['pred'] = preds
        # Simulate PnL
        total_pnl = simulate_pnl(df_test, shift=shift, threshold=1.0, fee=0.004)

        print(f"  [RESULT] total_pnl={total_pnl:.4f}, best_params={best_params}")

        results.append({
            'test_month': test_start_str,
            'train_size': len(df_train),
            'test_size': len(df_test),
            'best_params': best_params,
            'total_pnl': total_pnl
        })

    df_res = pd.DataFrame(results)
    return df_res


if __name__=="__main__":
    df_out = monthly_walk_forward_pnl_lookback(
        symbol_id=1,
        wf_start='2019-01-01',
        wf_end='2020-12-31',
        shift=5
    )
    print("\n=== WALK-FWD RESULTS (PnL + LOOKBACK) ===")
    print(df_out)
    print("Done.")
