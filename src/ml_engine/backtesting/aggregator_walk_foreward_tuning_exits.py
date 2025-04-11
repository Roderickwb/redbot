import mysql.connector
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta

import ta  # pip install ta
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterSampler
from sklearn.base import clone

### =============== [A] DB CONFIG: PAS EVENTUEEL AAN ===============
DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}

def parse_date_to_ms(datestr):
    """ 'YYYY-MM-DD' -> ms of 'YYYY-MM-DD HH:MM:SS' -> ms """
    if len(datestr)<=10:
        datestr += " 23:59:59"
    dt_obj = dt.datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S")
    return int(dt_obj.timestamp()*1000)

def load_ohlcv(symbol_id, interval, start_date, end_date):
    """
    Haal data direct uit MySQL in [start_ms..end_ms], i.p.v. alles en in Python filteren.
    """
    start_ms = parse_date_to_ms(start_date)
    end_ms   = parse_date_to_ms(end_date)

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT timestamp_ms, open, high, low, close, volume
        FROM market_data
        WHERE symbol_id=%s
          AND `interval`=%s
          AND timestamp_ms BETWEEN %s AND %s
        ORDER BY timestamp_ms ASC
    """
    cursor.execute(query, (symbol_id, interval, start_ms, end_ms))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    df = pd.DataFrame(rows)
    if df.empty: return df

    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open','high','low','close','volume'], inplace=True)

    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    return df

### =============== [B] INDICATORS ===============
def calc_indicators_1h(df):
    """
    Voor 1h: RSI(14), MACD(26,12,9), ADX(14), Stoch(14,3), Boll(20,2), ATR(14).
    """
    if df.empty: return df
    df['rsi'] = ta.momentum.rsi(df['close'], 14)
    df['macd'] = ta.trend.macd(df['close'], 26, 12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], 14, 3)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], 14, 3)
    df['bb_mavg'] = ta.volatility.bollinger_mavg(df['close'], 20)
    df['bb_hband'] = ta.volatility.bollinger_hband(df['close'], 20, 2)
    df['bb_lband'] = ta.volatility.bollinger_lband(df['close'], 20, 2)
    df['atr14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)

    df.dropna(inplace=True)
    return df

def calc_indicators_generic(df):
    """
    Voor 4h/1d: RSI(14), MACD(26,12,9), ADX(14) (basic).
    """
    if df.empty: return df
    df['rsi'] = ta.momentum.rsi(df['close'], 14)
    df['macd'] = ta.trend.macd(df['close'], 26, 12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], 26, 12, 9)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
    df.dropna(inplace=True)
    return df

### =============== [C] BUILD SHIFT-9 => SHIFT=3,6,9 LABELS ===============
def build_multi_shift_df(symbol_id, start_date, end_date):
    """
    1h => calc_indicators_1h (incl ATR14).
    4h => calc_indicators_generic
    1d => calc_indicators_generic
    merges => SHIFT=9 => we create 'target_return_3','target_return_6','target_return_9'
    """
    # 1h
    df_1h = load_ohlcv(symbol_id, '1h', start_date, end_date)
    df_1h = calc_indicators_1h(df_1h)
    if df_1h.empty: return df_1h
    df_1h.rename(columns={
        'open':'open_1h','high':'high_1h','low':'low_1h','close':'close_1h','volume':'volume_1h'
    }, inplace=True)

    # 4h
    df_4h = load_ohlcv(symbol_id, '4h', start_date, end_date)
    df_4h = calc_indicators_generic(df_4h)
    if df_4h.empty: return df_4h
    df_4h.rename(columns={
        'open':'open_4h','high':'high_4h','low':'low_4h','close':'close_4h','volume':'volume_4h',
        'rsi':'rsi_4h','macd':'macd_4h','macd_signal':'macd_signal_4h','adx':'adx_4h'
    }, inplace=True)

    # 1d
    df_1d = load_ohlcv(symbol_id, '1d', start_date, end_date)
    df_1d = calc_indicators_generic(df_1d)
    if df_1d.empty: return df_1d
    df_1d.rename(columns={
        'open':'open_1d','high':'high_1d','low':'low_1d','close':'close_1d','volume':'volume_1d',
        'rsi':'rsi_1d','macd':'macd_1d','macd_signal':'macd_signal_1d','adx':'adx_1d'
    }, inplace=True)

    # merges
    df_1h_srt = df_1h.sort_index()
    df_4h_srt = df_4h.sort_index()
    df_1d_srt = df_1d.sort_index()

    df_merge = pd.merge_asof(df_1h_srt, df_4h_srt, left_index=True, right_index=True, direction='backward')
    df_merge = pd.merge_asof(df_merge.sort_index(), df_1d_srt, left_index=True, right_index=True, direction='backward')
    df_merge.dropna(inplace=True)
    if df_merge.empty: return df_merge

    # SHIFT=9 => columns => SHIFT=3,6,9
    df_merge['fclose_s3'] = df_merge['close_1h'].shift(-3)
    df_merge['fclose_s6'] = df_merge['close_1h'].shift(-6)
    df_merge['fclose_s9'] = df_merge['close_1h'].shift(-9)

    df_merge['target_return_3'] = (df_merge['fclose_s3'] - df_merge['close_1h']) / df_merge['close_1h']*100.0
    df_merge['target_return_6'] = (df_merge['fclose_s6'] - df_merge['close_1h']) / df_merge['close_1h']*100.0
    df_merge['target_return_9'] = (df_merge['fclose_s9'] - df_merge['close_1h']) / df_merge['close_1h']*100.0

    df_merge.dropna(inplace=True)
    return df_merge

def build_test_data_with_lookback(symbol_id, test_start_dt, test_end_dt, lookback_months=2):
    lookback_start = test_start_dt - relativedelta(months=lookback_months)
    lookback_start_str = lookback_start.strftime("%Y-%m-%d")
    test_end_str = test_end_dt.strftime("%Y-%m-%d")

    df_big = build_multi_shift_df(symbol_id, lookback_start_str, test_end_str)
    if df_big.empty:
        return df_big
    df_test = df_big.loc[test_start_dt : test_end_dt]
    return df_test

### =============== ATR-based Exits SIM ===============
def simulate_trades_atr_exits(df, threshold=0.8,
                              fee=0.004,
                              sl_atr=1.0,
                              tp_atr=2.0,
                              trail_atr=1.0):
    """
    Bij entry:
      sl_price = entry +/- sl_atr * atr14
      tp_price = entry +/- tp_atr * atr14
      trailing_stop init = sl_price
    trailing update => 'new_trail = highest_since_entry - trail_atr*atr14' (long)
    """
    if df.empty:
        return 0.0
    needed= ['pred','close_1h','high_1h','low_1h','atr14']
    for c in needed:
        if c not in df.columns:
            return 0.0

    c_close= df['close_1h'].values
    c_high= df['high_1h'].values
    c_low= df['low_1h'].values
    c_atr= df['atr14'].values
    preds= df['pred'].values

    pos_active=False
    pos_side=None
    entry_price= 0.0
    atr_entry= 0.0
    sl_price= 0.0
    tp_price= 0.0
    trailing_stop=0.0
    highest_since_entry=0.0
    lowest_since_entry=0.0
    total_pnl=0.0

    for i in range(len(df)):
        pr= preds[i]
        hi= c_high[i]
        lo= c_low[i]
        cl= c_close[i]
        atr_i= c_atr[i]

        if not pos_active:
            if pr> threshold:
                pos_active=True
                pos_side='long'
                entry_price= cl
                atr_entry= atr_i
                sl_price= entry_price - sl_atr*atr_entry
                tp_price= entry_price + tp_atr*atr_entry
                trailing_stop= sl_price
                highest_since_entry= entry_price
            elif pr< -threshold:
                pos_active=True
                pos_side='short'
                entry_price= cl
                atr_entry= atr_i
                sl_price= entry_price + sl_atr*atr_entry
                tp_price= entry_price - tp_atr*atr_entry
                trailing_stop= sl_price
                lowest_since_entry= entry_price
            else:
                pass
        else:
            if pos_side=='long':
                if hi> highest_since_entry:
                    highest_since_entry= hi
                    new_trail= highest_since_entry - trail_atr*atr_entry
                    if new_trail> trailing_stop:
                        trailing_stop= new_trail

                if lo<= trailing_stop:
                    exitp= trailing_stop
                    ret= (exitp/entry_price)-1.0
                    total_pnl+= (ret - fee)
                    pos_active=False
                elif hi>= tp_price:
                    exitp= tp_price
                    ret= (exitp/entry_price)-1.0
                    total_pnl+= (ret - fee)
                    pos_active=False
                else:
                    pass
            else:
                # short
                if lo< lowest_since_entry:
                    lowest_since_entry= lo
                    new_trail= lowest_since_entry + trail_atr*atr_entry
                    if new_trail< trailing_stop:
                        trailing_stop= new_trail

                if hi>= trailing_stop:
                    exitp= trailing_stop
                    ret= 1.0 - (exitp/entry_price)
                    total_pnl+= (ret - fee)
                    pos_active=False
                elif lo<= tp_price:
                    exitp= tp_price
                    ret= 1.0 - (exitp/entry_price)
                    total_pnl+= (ret - fee)
                    pos_active=False
                else:
                    pass

    return total_pnl

### =============== XGB WRAPPER ( SHIFT in {3,6,9}, sl_atr, etc.) ===============
class XGBRegWithExits(XGBRegressor):
    def __init__(self,
                 SHIFT=3,  # which label we use => 'target_return_3/6/9'
                 sl_atr=1.0, tp_atr=2.0, trail_atr=1.0,
                 threshold=0.8,
                 learning_rate=0.1, max_depth=3, subsample=1.0,
                 n_estimators=100, random_state=42, **kwargs):
        super().__init__(
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )
        self.SHIFT= SHIFT
        self.sl_atr= sl_atr
        self.tp_atr= tp_atr
        self.trail_atr= trail_atr
        self.threshold= threshold

### =============== CUSTOM ParamSearch op 1 fold (heel train) ===============
def train_pnl_scorer(estimator, X, SHIFT_val):
    """
    We do 'predict' => df['pred'] => 'simulate_trades_atr_exits'
    SHIFT_val used for partial debugging if needed, but no direct usage here.
    """
    preds= estimator.predict(X)
    df_temp= X.copy()
    df_temp['pred']= preds

    total_pnl= simulate_trades_atr_exits(
        df_temp,
        threshold= estimator.threshold,
        fee=0.004,
        sl_atr= estimator.sl_atr,
        tp_atr= estimator.tp_atr,
        trail_atr= estimator.trail_atr
    )
    return total_pnl

def param_search_single_cv(model, param_dist, n_iter, df_train):
    """
    We pick SHIFT from param-dist => build X_train,y_train => fit => measure train-PnL => best
    """
    from sklearn.model_selection import ParameterSampler
    sampler= list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))

    best_score= -999999
    best_params= None
    best_estimator= None

    for combo in sampler:
        SHIFT_val= combo.get('SHIFT', 3)
        col_label= f"target_return_{SHIFT_val}"
        if col_label not in df_train.columns:
            continue

        # Features
        dropcols= [c for c in df_train.columns if c.startswith('target_return_')]
        feat_cols= [c for c in df_train.columns if c not in dropcols]
        X_train= df_train[feat_cols]
        y_train= df_train[col_label]

        est= clone(model)
        # set param
        for k,v in combo.items():
            setattr(est, k, v)
        est.fit(X_train, y_train)

        score= train_pnl_scorer(est, X_train, SHIFT_val)
        if score> best_score:
            best_score= score
            best_params= combo
            best_estimator= est

    return best_params, best_estimator, best_score

### =============== MAIN WF ===============
def monthly_walk_forward_tuning_exits(
    symbol_id=1,
    wf_start='2019-01-01',
    wf_end='2020-12-31'
):
    dt_start= dt.datetime.strptime(wf_start, "%Y-%m-%d")
    dt_end= dt.datetime.strptime(wf_end, "%Y-%m-%d")

    month_list=[]
    current= dt_start.replace(day=1)
    while current<= dt_end:
        month_list.append(current)
        current= current+ relativedelta(months=1)

    results=[]
    for i in range(1, len(month_list)):
        test_start_dt= month_list[i]
        test_end_dt= test_start_dt+ relativedelta(months=1)
        if test_end_dt> dt_end:
            test_end_dt= dt_end

        train_start_str= wf_start
        train_end_str= (test_start_dt - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        test_start_str= test_start_dt.strftime("%Y-%m-%d")
        test_end_str= test_end_dt.strftime("%Y-%m-%d")

        print(f"\n[WALK-FWD SHIFT+ATR] Test={test_start_str}..{test_end_str}, Train={train_start_str}..{train_end_str}")

        # Build train => SHIFT=9 aggregator
        df_train= build_multi_shift_df(symbol_id, train_start_str, train_end_str)
        if df_train.empty or len(df_train)<50:
            print("[WARN] train empty => skip")
            continue

        df_test= build_test_data_with_lookback(symbol_id, test_start_dt, test_end_dt, lookback_months=2)
        if df_test.empty or len(df_test)<10:
            print("[WARN] test empty => skip")
            continue

        # Param-dist
        param_dist= {
            'SHIFT':[3,6,9],
            'sl_atr':[1.0,1.5,2.0],
            'tp_atr':[1.0,2.0,3.0],
            'trail_atr':[0.5,1.0],
            'threshold':[0.5,0.8,1.0],
            'learning_rate':[0.01,0.1],
            'max_depth':[3,5],
            'subsample':[0.8,1.0]
        }
        model= XGBRegWithExits()

        best_params, best_estimator, best_score= param_search_single_cv(
            model, param_dist, n_iter=6, df_train=df_train
        )
        print(f"   [INFO] best_train_PnL={best_score:.4f}, best_params={best_params}")
        if not best_estimator:
            continue

        SHIFT_val= best_estimator.SHIFT
        col_label_test= f"target_return_{SHIFT_val}"
        if col_label_test not in df_test.columns:
            print(f"[WARN] test missing {col_label_test}, skip.")
            continue

        dropcols_test= [c for c in df_test.columns if c.startswith('target_return_')]
        featcols_test= [c for c in df_test.columns if c not in dropcols_test]
        X_test= df_test[featcols_test]
        preds= best_estimator.predict(X_test)
        df_test['pred']= preds

        test_pnl= simulate_trades_atr_exits(
            df_test,
            threshold= best_estimator.threshold,
            fee=0.004,
            sl_atr= best_estimator.sl_atr,
            tp_atr= best_estimator.tp_atr,
            trail_atr= best_estimator.trail_atr
        )
        print(f"   [RESULT] test_pnl={test_pnl:.4f}")

        results.append({
            'test_month': test_start_str,
            'train_size': len(df_train),
            'test_size': len(df_test),
            'best_params': best_params,
            'train_pnl': best_score,
            'test_pnl': test_pnl
        })

    df_res= pd.DataFrame(results)
    return df_res

if __name__=="__main__":
    # PAS AAN: symbol_id, periode
    df_out = monthly_walk_forward_tuning_exits(
        symbol_id=1,
        wf_start='2019-01-01',
        wf_end='2020-12-31'
    )
    print("\n=== SHIFT(3,6,9)+ATR-based Exits param-search => final results ===")
    print(df_out)
    print("Done.")
