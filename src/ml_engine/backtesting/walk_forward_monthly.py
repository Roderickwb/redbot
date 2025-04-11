import mysql.connector
import pandas as pd
import numpy as np
import ta  # pip install ta
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import datetime
from dateutil.relativedelta import relativedelta

# [A] DATABASECONFIG (PAS AAN)
DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}

# =========================
# [1] LAADFUNCTIE (1h/4h/1d)
# =========================
def load_df(symbol_id, interval, start_date, end_date):
    """
    Haalt ALLE data (symbol_id, interval) uit 'market_data'.
    Filtert in Python op [start_date, end_date].
    Returnt DataFrame (index=datetime).
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

# =========================
# [2] INDICATORS
# =========================
def calc_indicators(df, interval='1h'):
    """
    Voorbeeld: RSI(14), MACD(12,26,9), ADX(14), Stoch(14,3,3), Boll(20,2)
    """
    if df.empty:
        return df

    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], 14)
    # MACD
    df['macd'] = ta.trend.macd(df['close'], 26, 12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    # ADX
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
    # Stoch
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], 14, 3)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], 14, 3)
    # Boll
    df['bb_mavg'] = ta.volatility.bollinger_mavg(df['close'], 20)
    df['bb_hband'] = ta.volatility.bollinger_hband(df['close'], 20, 2)
    df['bb_lband'] = ta.volatility.bollinger_lband(df['close'], 20, 2)

    df.dropna(inplace=True)
    return df

# =========================
# [3] build_multi_tf_df
# =========================
def build_multi_tf_df(symbol_id, start_date, end_date):
    """
    1h,4h,1d => load => indicators => merges => dropna
    Return DF met kolommen close_1h,close_4h,close_1d, rsi_1h, etc.
    ZONDER SHIFT-target (die voegen we hieronder pas toe).
    """
    df_1h = load_df(symbol_id, '1h', start_date, end_date)
    df_1h = calc_indicators(df_1h, '1h')
    if df_1h.empty:
        return pd.DataFrame()
    df_1h.rename(columns={
        'open':'open_1h','high':'high_1h','low':'low_1h','close':'close_1h','volume':'volume_1h'
    }, inplace=True)

    df_4h = load_df(symbol_id, '4h', start_date, end_date)
    df_4h = calc_indicators(df_4h, '4h')
    if df_4h.empty:
        return pd.DataFrame()
    df_4h.rename(columns={
        'open':'open_4h','high':'high_4h','low':'low_4h','close':'close_4h','volume':'volume_4h'
    }, inplace=True)

    df_1d = load_df(symbol_id, '1d', start_date, end_date)
    df_1d = calc_indicators(df_1d, '1d')
    if df_1d.empty:
        return pd.DataFrame()
    df_1d.rename(columns={
        'open':'open_1d','high':'high_1d','low':'low_1d','close':'close_1d','volume':'volume_1d'
    }, inplace=True)

    # merges
    df_1h_srt = df_1h.sort_index()
    df_4h_srt = df_4h.sort_index()
    df_1d_srt = df_1d.sort_index()

    df_merge = pd.merge_asof(df_1h_srt, df_4h_srt, left_index=True, right_index=True,
                             direction='backward')
    df_merge = pd.merge_asof(df_merge.sort_index(), df_1d_srt,
                             left_index=True, right_index=True,
                             direction='backward')
    df_merge.dropna(inplace=True)

    return df_merge

def add_shifted_target(df, shift=5):
    """
    df['target_return_1h'] = %verandering over SHIFT candles
    """
    if df.empty:
        return df
    df['future_close_1h'] = df['close_1h'].shift(-shift)
    df['target_return_1h'] = (df['future_close_1h'] - df['close_1h']) / df['close_1h'] * 100.0
    df.dropna(inplace=True)
    df.drop(columns=['future_close_1h'], inplace=True)
    return df

# =========================
# [4] MONTHLY WALK-FORWARD + PARAM SEARCH
# =========================
def monthly_walk_forward_param_search(symbol_id=1, shift=5,
                                      start_date='2018-01-01',
                                      end_date='2019-12-31'):
    """
    1) Genereer lijst van maand-begindata tussen start_date en end_date.
    2) For each month i=1..N:
       - train_period = [start_date, test_month_start)
       - test_period  = die maand
       - build_multi_tf_df => SHIFT => X,y
       - Param search => best params => train model => predict test => MSE
       - Log
    Return results DataFrame
    """
    # parse
    dt_start = pd.to_datetime(start_date)
    dt_end   = pd.to_datetime(end_date)

    # lijst van maand-starts
    current = dt_start
    months = []
    while current <= dt_end:
        months.append(current)
        current = current + relativedelta(months=1)

    results = []
    for i in range(1, len(months)):
        test_start = months[i]
        test_end   = test_start + relativedelta(months=1)
        if test_end > dt_end:
            test_end = dt_end

        # definieer train-end = test_start - 1 candle
        # of je kunt een 'expanding window' doen: train = [start_date .. test_start)
        train_start_str = dt_start.strftime("%Y-%m-%d")
        train_end_str   = (test_start - datetime.timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        test_start_str  = test_start.strftime("%Y-%m-%d")
        test_end_str    = test_end.strftime("%Y-%m-%d")

        # Stop als test_start == dt_start of test_end <= test_start
        if test_end <= test_start:
            break

        # build train DF
        df_train = build_multi_tf_df(symbol_id, train_start_str, train_end_str)
        df_train = add_shifted_target(df_train, shift)
        if len(df_train)<50:
            print(f"[WARN] train DF<50 rows => skip {test_start_str}")
            continue

        # build test DF
        df_test = build_multi_tf_df(symbol_id, test_start_str, test_end_str)
        df_test = add_shifted_target(df_test, shift)
        if len(df_test)<10:
            print(f"[WARN] test DF<10 rows => skip {test_start_str}")
            continue

        X_train = df_train.drop(columns=[c for c in df_train.columns
                                         if c.startswith('target_return_')])
        y_train = df_train['target_return_1h']
        X_test = df_test.drop(columns=[c for c in df_test.columns
                                       if c.startswith('target_return_')])
        y_test = df_test['target_return_1h']

        print(f"\n[WALK-FORWARD] Test month: {test_start_str[:7]}")
        print(f"  Train: {train_start_str} -> {train_end_str} => {len(X_train)} rows")
        print(f"  Test : {test_start_str} -> {test_end_str} => {len(X_test)} rows")

        # Klein param-grid (uitbreid naar wens):
        param_grid = {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
        }
        from sklearn.model_selection import GridSearchCV
        xgb_model = XGBRegressor(n_estimators=50, random_state=42)
        grid = GridSearchCV(
            xgb_model,
            param_grid,
            scoring='neg_mean_squared_error',
            cv=2,  # 2-fold om tijd te sparen
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_params = grid.best_params_
        best_score = -grid.best_score_

        # Retrain final
        final_model = XGBRegressor(n_estimators=50, random_state=42, **best_params)
        final_model.fit(X_train, y_train)

        preds = final_model.predict(X_test)
        mse_test = mean_squared_error(y_test, preds)

        print(f"  [RESULT] best_params={best_params}, best_cv_mse={best_score:.4f}, test_mse={mse_test:.4f}")

        results.append({
            'test_month': test_start_str[:7],
            'train_size': len(X_train),
            'test_size': len(X_test),
            'best_params': best_params,
            'cv_mse': best_score,
            'mse': mse_test
        })

    results_df = pd.DataFrame(results)
    return results_df


if __name__=="__main__":
    # Voorbeeld: we gaan van 2017-01-01 tot 2020-12-31
    # SHIFT=5
    WF_START = '2017-01-01'
    WF_END   = '2020-12-31'
    SHIFT    = 5

    df_res = monthly_walk_forward_param_search(
        symbol_id=1,
        shift=SHIFT,
        start_date=WF_START,
        end_date=WF_END
    )
    print("\n===== WALK-FORWARD RESULTS =====")
    print(df_res)
    print("\nDone.")
