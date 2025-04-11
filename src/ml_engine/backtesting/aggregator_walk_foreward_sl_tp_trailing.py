import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import ta
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# ======================
# [A] DATABASECONFIG
# ======================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}

# ============ [B] LOAD + INDICATORS ==============
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
    if df.empty:
        return df
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
    Aggregator: 1h,4h,1d => merges => SHIFT => target_return_1h
    (SHIFT is nog steeds je label-horizon)
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
    if df_merge.empty:
        return df_merge

    # SHIFT => target_return_1h
    df_merge['future_close_1h'] = df_merge['close_1h'].shift(-shift)
    df_merge['target_return_1h'] = (df_merge['future_close_1h'] - df_merge['close_1h']) / df_merge['close_1h'] * 100.0
    df_merge.dropna(inplace=True)
    df_merge.drop(columns=['future_close_1h'], inplace=True)
    return df_merge

# ============ [C] LOOKBACK FOR TEST ============
def build_test_data_with_lookback(symbol_id, test_start_dt, test_end_dt, shift=5, lookback_months=2):
    lookback_start = test_start_dt - relativedelta(months=lookback_months)
    lookback_start_str = lookback_start.strftime("%Y-%m-%d")
    test_end_str = test_end_dt.strftime("%Y-%m-%d %H:%M:%S")

    df_big = build_multi_tf_df(symbol_id, lookback_start_str, test_end_str, shift=shift)
    if df_big.empty:
        return df_big

    df_test = df_big.loc[test_start_dt : test_end_dt]
    return df_test

# ============ [D] SIMULATE TRADES (SL/TP/TRAILING) ============
def simulate_trades_sl_tp_trail(
    df,
    threshold=0.8,   # +/- 0.8% -> open position
    fee=0.004,       # 0.4% in +0.4% out = 0.8% total
    sl_pct=1.0,      # 1.0% stop-loss
    tp_pct=2.0,      # 2.0% take-profit
    trailing_pct=1.0 # 1.0% trailing stop
):
    """
    - We loop candle-voor-candle
    - Als we GEEN positie hebben en pred>+threshold => open LONG
      * EntryPrice = close_1h
      * SL = entry * (1 - sl_pct/100)   # bv 1.0% => 0.99 * entry
      * TP = entry * (1 + tp_pct/100)
      * trailingStop = SL (start)
    - Elke volgende candle:
      * update trailingStop als (close-highestSinceEntry)*...
      * check of high >= TP => exit => PN = (TP/entry -1) - fee
      * check of low <= SL => exit => PN = (SL/entry -1) - fee
    - idem SHORT als pred<-threshold
    - we laten pos open tot SL/TP/trailingHit? => exit => neem PnL => fee
    * trailing => als close> entry => trailingStop = max(trailingStop, close*(1 - trailing_pct/100))
    * let op dat je high/low candle-wise checkt
    """
    if df.empty:
        return 0.0

    # columns we need:
    # df['pred'], df['open_1h'], df['high_1h'], df['low_1h'], df['close_1h']
    c_close = df['close_1h'].values
    c_high  = df['high_1h'].values
    c_low   = df['low_1h'].values
    preds   = df['pred'].values

    pos_active = False
    pos_side = None  # 'long' or 'short'
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    trailing_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry  = 0.0

    total_pnl = 0.0

    for i in range(len(df)):
        pred = preds[i]
        o = c_close[i]  # 'close' as ref for open position price

        hi = c_high[i]
        lo = c_low[i]

        if not pos_active:
            # check if we open pos
            if pred > threshold:
                # open LONG
                pos_active = True
                pos_side = 'long'
                entry_price = o
                # define SL, TP
                sl_price = entry_price*(1.0 - sl_pct/100.0)
                tp_price = entry_price*(1.0 + tp_pct/100.0)
                # trailing stop = same as SL start
                trailing_stop = sl_price
                highest_since_entry = entry_price
                lowest_since_entry  = entry_price
            elif pred < -threshold:
                # open SHORT
                pos_active = True
                pos_side = 'short'
                entry_price = o
                sl_price = entry_price*(1.0 + sl_pct/100.0)
                tp_price = entry_price*(1.0 - tp_pct/100.0)
                trailing_stop = sl_price
                highest_since_entry = entry_price
                lowest_since_entry  = entry_price
            else:
                # do nothing
                pass
        else:
            # pos is active => check SL/TP/trailing
            if pos_side=='long':
                # update highest_since_entry
                if hi>highest_since_entry:
                    highest_since_entry = hi
                    # update trailing
                    new_trailing = highest_since_entry*(1.0 - trailing_pct/100.0)
                    if new_trailing>trailing_stop:
                        trailing_stop = new_trailing

                # check if we are triggered by TP or SL
                # candle's range is [low..high], so if hi >= tp_price => we exit at tp_price
                if lo<= trailing_stop:
                    # triggered trailing/SL
                    exit_price = trailing_stop
                    # PnL
                    ret = (exit_price/entry_price)-1.0
                    net_ret = ret - fee
                    total_pnl += net_ret
                    pos_active=False
                elif hi>= tp_price:
                    exit_price = tp_price
                    ret = (exit_price/entry_price)-1.0
                    net_ret = ret - fee
                    total_pnl += net_ret
                    pos_active=False
                else:
                    # still hold
                    pass

            elif pos_side=='short':
                # update lowest_since_entry
                if lo<lowest_since_entry:
                    lowest_since_entry=lo
                    new_trailing = lowest_since_entry*(1.0 + trailing_pct/100.0)
                    if new_trailing< trailing_stop:
                        trailing_stop = new_trailing

                # check SL/TP
                # short => if hi> sl => we exit at sl
                # if lo< tp => we exit at tp
                if hi>= trailing_stop:
                    # triggered trailing or SL
                    exit_price = trailing_stop
                    ret = 1.0 - (exit_price/entry_price)
                    net_ret = ret - fee
                    total_pnl += net_ret
                    pos_active=False
                elif lo<= tp_price:
                    exit_price = tp_price
                    ret = 1.0 - (exit_price/entry_price)
                    net_ret = ret - fee
                    total_pnl += net_ret
                    pos_active=False
                else:
                    pass
        # end for pos_active check
    # end for loop

    return total_pnl

# ================= [E] MONTHLY WF + ParamSearch + SL/TP/Trail ==============
def monthly_walk_forward_sl_tp_trail(
    symbol_id=1,
    wf_start='2019-01-01',
    wf_end='2020-12-31',
    shift=5
):
    dt_start = datetime.strptime(wf_start, "%Y-%m-%d")
    dt_end   = datetime.strptime(wf_end, "%Y-%m-%d")

    month_list = []
    current = dt_start.replace(day=1)
    while current <= dt_end:
        month_list.append(current)
        current = current + relativedelta(months=1)

    results=[]
    for i in range(1, len(month_list)):
        test_start_dt = month_list[i]
        test_end_dt   = test_start_dt+ relativedelta(months=1)
        if test_end_dt> dt_end:
            test_end_dt= dt_end

        train_start_str= wf_start
        train_end_str= (test_start_dt - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")

        test_start_str= test_start_dt.strftime("%Y-%m-%d")
        test_end_str= test_end_dt.strftime("%Y-%m-%d")

        print(f"\n[WALK-FWD] Test={test_start_str}..{test_end_str}, Train={train_start_str}..{train_end_str}")

        # BUILD TRAIN
        df_train= build_multi_tf_df(symbol_id, train_start_str, train_end_str, shift=shift)
        if df_train.empty or len(df_train)<50:
            print("[WARN] train empty => skip")
            continue

        # BUILD TEST with lookback
        df_test= build_test_data_with_lookback(symbol_id, test_start_dt, test_end_dt, shift=shift, lookback_months=2)
        if df_test.empty or len(df_test)<10:
            print("[WARN] test empty => skip")
            continue

        # X,y
        dropcols= [c for c in df_train.columns if c.startswith('target_return_')]
        feats_train= [c for c in df_train.columns if c not in dropcols]

        y_train= df_train['target_return_1h']
        X_train= df_train[feats_train]

        # Param-dist
        param_dist= {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3,5],
            'subsample': [0.8,1.0]
        }
        model= XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')

        random_search= RandomizedSearchCV(
            estimator= model,
            param_distributions= param_dist,
            n_iter=3,
            scoring='neg_mean_squared_error',
            cv=2,
            n_jobs=-1,
            random_state=42
        )
        print("  [INFO] random search param...")
        random_search.fit(X_train, y_train)
        best_params= random_search.best_params_

        best_model= XGBRegressor(
            n_estimators=100,
            random_state=42,
            objective='reg:squarederror',
            **best_params
        )
        best_model.fit(X_train, y_train)

        drop_test= [c for c in df_test.columns if c.startswith('target_return_')]
        feats_test= [c for c in df_test.columns if c not in drop_test]

        X_test= df_test[feats_test]
        preds= best_model.predict(X_test)
        df_test['pred']= preds

        # SIMULATE PNL (SL=1.0%,TP=2.0%,trail=1.0%)
        total_pnl= simulate_trades_sl_tp_trail(
            df_test,
            threshold=0.8,  # 0.8% => open pos
            fee=0.004,
            sl_pct=1.0,
            tp_pct=2.0,
            trailing_pct=1.0
        )
        print(f"  [RESULT] total_pnl={total_pnl:.4f}, best_params={best_params}")

        results.append({
            'test_month': test_start_str,
            'train_size': len(df_train),
            'test_size': len(df_test),
            'best_params': best_params,
            'total_pnl': total_pnl
        })

    df_res= pd.DataFrame(results)
    return df_res

# ================== MAIN ===================
if __name__=="__main__":
    df_out= monthly_walk_forward_sl_tp_trail(
        symbol_id=1,
        wf_start='2019-01-01',
        wf_end='2020-12-31',
        shift=5
    )
    print("\n=== WALK-FWD RESULTS (PnL with SL/TP/Trail) ===")
    print(df_out)
    print("Done.")
