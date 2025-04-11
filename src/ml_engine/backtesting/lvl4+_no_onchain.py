import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import ta
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, ParameterSampler
from sklearn.base import clone

###################### [A] DB CONFIG ######################
DB_CONFIG = {
    'host': 'localhost',
    'user': 'botuser',
    'password': 'MySQL194860!',
    'database': 'tradebot'
}

def parse_date_to_ms(datestr):
    # 'YYYY-MM-DD' -> epoch ms
    if len(datestr)<=10:
        datestr += " 23:59:59"
    dt_obj= datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S")
    return int(dt_obj.timestamp()*1000)

def load_ohlcv(symbol_id, interval, start_date, end_date):
    start_ms= parse_date_to_ms(start_date)
    end_ms= parse_date_to_ms(end_date)
    conn= mysql.connector.connect(**DB_CONFIG)
    cursor= conn.cursor(dictionary=True)
    query= """
        SELECT timestamp_ms, open, high, low, close, volume
        FROM market_data
        WHERE symbol_id=%s
          AND `interval`=%s
          AND timestamp_ms BETWEEN %s AND %s
        ORDER BY timestamp_ms ASC
    """
    cursor.execute(query, (symbol_id, interval, start_ms, end_ms))
    rows= cursor.fetchall()
    cursor.close()
    conn.close()

    df= pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ['open','high','low','close','volume']:
        df[col]= pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open','high','low','close','volume'], inplace=True)
    df['datetime']= pd.to_datetime(df['timestamp_ms'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    return df

###################### [B] INDICATORS + SHIFT=1,6,24 ######################
def calc_indicators_1h(df):
    if df.empty:
        return df
    df['rsi']= ta.momentum.rsi(df['close'],14)
    df['macd']= ta.trend.macd(df['close'],26,12)
    df['macd_signal']= ta.trend.macd_signal(df['close'],26,12,9)
    df['adx']= ta.trend.adx(df['high'], df['low'], df['close'],14)
    df['atr14']= ta.volatility.average_true_range(df['high'], df['low'], df['close'],14)
    # BearBull regime => sma200
    df['sma200']= df['close'].rolling(200).mean()
    df['regime']= (df['close']> df['sma200']).astype(int)

    df.dropna(inplace=True)
    return df

def build_multi_shift_df(symbol_id, start_date, end_date):
    df_1h= load_ohlcv(symbol_id, '1h', start_date, end_date)
    if df_1h.empty:
        return df_1h

    df_1h= calc_indicators_1h(df_1h)
    if df_1h.empty:
        return df_1h

    df_1h.rename(columns={
        'open':'open_1h','high':'high_1h','low':'low_1h',
        'close':'close_1h','volume':'volume_1h'
    }, inplace=True)

    # SHIFT => 1,6,24
    df_1h['fclose_s1']= df_1h['close_1h'].shift(-1)
    df_1h['fclose_s6']= df_1h['close_1h'].shift(-6)
    df_1h['fclose_s24']= df_1h['close_1h'].shift(-24)

    df_1h['target_return_1']= (df_1h['fclose_s1']- df_1h['close_1h'])/df_1h['close_1h']*100.0
    df_1h['target_return_6']= (df_1h['fclose_s6']- df_1h['close_1h'])*100.0/ df_1h['close_1h']
    df_1h['target_return_24']= (df_1h['fclose_s24']- df_1h['close_1h'])*100.0/ df_1h['close_1h']

    df_1h.dropna(inplace=True)
    return df_1h

###################### [C] PARTIAL EXIT(50%) + TIME-LIMIT(24) + SL + START_BAL=1000 ######################
def simulate_trades_partial50_time_limit(df,
                                         threshold=0.8,
                                         fee=0.004,
                                         sl_atr=1.0,
                                         tp_atr=1.0,
                                         time_limit=24,
                                         start_balance=1000.0):
    """
    - open if pred> threshold => long
      partial exit => +tp_atr* ATR => 50%
      time-limit => close after 24 candles
      SL => -sl_atr*ATR
    - short sym
    - risk=1% per trade => invest= balance*0.01
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

    balance= start_balance
    pos_active= False
    pos_side= None
    entry_idx= 0
    entry_price= 0.0
    sl_price= 0.0
    partial_price= 0.0
    partial_closed= False
    units= 0.0

    for i in range(len(df)):
        pr= preds[i]
        cl= c_close[i]
        atr_i= c_atr[i]

        if not pos_active:
            if pr> threshold:
                pos_active= True
                pos_side= 'long'
                entry_price= cl
                sl_price= entry_price - sl_atr* atr_i
                partial_price= entry_price + tp_atr* atr_i
                invest= balance*0.05
                if invest<=0: continue
                units= invest/ entry_price
                entry_idx= i
                partial_closed= False
            elif pr< -threshold:
                pos_active= True
                pos_side= 'short'
                entry_price= cl
                sl_price= entry_price + sl_atr* atr_i
                partial_price= entry_price - tp_atr* atr_i
                invest= balance*0.01
                if invest<=0: continue
                units= invest/ entry_price
                entry_idx= i
                partial_closed= False
            else:
                pass
        else:
            # time-limit
            if i- entry_idx >= time_limit:
                exitp= cl
                if pos_side=='long':
                    ret= (exitp/ entry_price)-1.0
                else:
                    ret= 1.0-(exitp/ entry_price)
                trade_pnl= ret*(units* entry_price)
                trade_fee= fee*(units* entry_price)
                balance+= (trade_pnl- trade_fee)
                pos_active= False
            else:
                # partial
                if pos_side=='long':
                    if (not partial_closed) and (c_high[i]>= partial_price):
                        exitp= partial_price
                        ret= (exitp/ entry_price)-1.0
                        trade_pnl= ret*(0.5*units* entry_price)
                        trade_fee= fee*(0.5*units* entry_price)
                        balance+= (trade_pnl- trade_fee)
                        partial_closed= True

                    # SL
                    if c_low[i]<= sl_price:
                        exitp= sl_price
                        leftover_frac= (1.0 if not partial_closed else 0.5)
                        ret= (exitp/ entry_price)-1.0
                        trade_pnl= ret*(leftover_frac*units* entry_price)
                        trade_fee= fee*(leftover_frac*units* entry_price)
                        balance+= (trade_pnl- trade_fee)
                        pos_active= False
                else:
                    # short partial
                    if (not partial_closed) and (c_low[i]<= partial_price):
                        exitp= partial_price
                        ret= 1.0-(exitp/ entry_price)
                        trade_pnl= ret*(0.5*units* entry_price)
                        trade_fee= fee*(0.5*units* entry_price)
                        balance+= (trade_pnl- trade_fee)
                        partial_closed= True

                    if c_high[i]>= sl_price:
                        exitp= sl_price
                        leftover_frac= (1.0 if not partial_closed else 0.5)
                        ret= 1.0-(exitp/ entry_price)
                        trade_pnl= ret*(leftover_frac*units* entry_price)
                        trade_fee= fee*(leftover_frac*units* entry_price)
                        balance+= (trade_pnl- trade_fee)
                        pos_active= False

    return balance- start_balance

###################### [D] MODEL WRAPPER ######################
class XGBPartialOne(XGBRegressor):
    def __init__(self,
                 SHIFT=1,
                 sl_atr=1.0,
                 tp_atr=1.0,
                 threshold=0.8,
                 gamma=0.0,
                 reg_alpha=0.0,
                 learning_rate=0.1,
                 max_depth=3,
                 subsample=1.0,
                 random_state=42,
                 n_estimators=100,
                 **kwargs):
        super().__init__(
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            n_estimators=n_estimators,
            gamma=gamma,
            reg_alpha=reg_alpha,
            random_state=random_state,
            **kwargs
        )
        self.SHIFT= SHIFT
        self.sl_atr= sl_atr
        self.tp_atr= tp_atr
        self.threshold= threshold

def crossval_pnl_scorer(estimator, X, y, tscv):
    folds_scores= []
    for train_idx, valid_idx in tscv.split(X):
        X_train_fold= X.iloc[train_idx]
        y_train_fold= y.iloc[train_idx]
        X_valid_fold= X.iloc[valid_idx]

        c_est= clone(estimator)
        c_est.fit(X_train_fold, y_train_fold)

        preds= c_est.predict(X_valid_fold)
        df_temp= X_valid_fold.copy()
        df_temp['pred']= preds

        fold_pnl= simulate_trades_partial50_time_limit(
            df_temp,
            threshold= c_est.threshold,
            sl_atr= c_est.sl_atr,
            tp_atr= c_est.tp_atr,
            time_limit=24,
            fee=0.004,
            start_balance=1000.0
        )
        folds_scores.append(fold_pnl)
    return np.mean(folds_scores)

def param_search_timeseries_cv(model, param_dist, n_iter, df_train, n_splits=5):
    from sklearn.model_selection import ParameterSampler
    sampler= list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))

    best_score= -999999
    best_params= None
    best_estimator= None

    tscv= TimeSeriesSplit(n_splits=n_splits)

    for combo in sampler:
        SHIFT_val= combo.get('SHIFT',1)
        col_label= f"target_return_{SHIFT_val}"
        if col_label not in df_train.columns:
            continue

        dropcols= [c for c in df_train.columns if c.startswith('target_return_')]
        feat_cols= [c for c in df_train.columns if c not in dropcols]
        X_full= df_train[feat_cols]
        y_full= df_train[col_label]

        # set param
        est= clone(model)
        for k,v in combo.items():
            setattr(est, k, v)

        cv_score= crossval_pnl_scorer(est, X_full, y_full, tscv)
        if cv_score> best_score:
            best_score= cv_score
            best_params= combo
            best_estimator= clone(est)

    if best_estimator:
        SHIFT_val= best_params['SHIFT']
        col_label= f"target_return_{SHIFT_val}"
        dropcols= [c for c in df_train.columns if c.startswith('target_return_')]
        feat_cols= [c for c in df_train.columns if c not in dropcols]
        X_final= df_train[feat_cols]
        y_final= df_train[col_label]
        for k,v in best_params.items():
            setattr(best_estimator, k, v)
        best_estimator.fit(X_final, y_final)

    return best_params, best_estimator, best_score

###################### [E] MONTHLY WF SHIFT={1,6,24} + partial exit + TSCV(5) + n_iter=30 + cumPnl ######################
def monthly_walk_forward(
    symbol_id=1,
    wf_start='2019-01-01',
    wf_end='2020-12-31'
):
    dt_start= datetime.strptime(wf_start, "%Y-%m-%d")
    dt_end= datetime.strptime(wf_end, "%Y-%m-%d")

    months=[]
    current= dt_start.replace(day=1)
    while current<= dt_end:
        months.append(current)
        current= current+ relativedelta(months=1)

    cumulative_pnl= 0.0
    results= []

    for i in range(1, len(months)):
        test_start_dt= months[i]
        test_end_dt= test_start_dt+ relativedelta(months=1)
        if test_end_dt> dt_end:
            test_end_dt= dt_end

        train_start_str= wf_start
        train_end_str= (test_start_dt - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        test_start_str= test_start_dt.strftime("%Y-%m-%d")
        test_end_str= test_end_dt.strftime("%Y-%m-%d")

        print(f"\n[WF SHIFT(1,6,24) partial50, TSCV(5), time-limit(24), n_iter=30]   Test={test_start_str}..{test_end_str}, Train={train_start_str}..{train_end_str}")

        df_train= build_multi_shift_df(symbol_id, train_start_str, train_end_str)
        if df_train.empty or len(df_train)<50:
            print("[WARN] train empty => skip")
            continue

        df_test= build_multi_shift_df(symbol_id, test_start_str, test_end_str)
        if df_test.empty or len(df_test)<10:
            print("[WARN] test empty => skip")
            continue

        # param-dist
        param_dist= {
            'SHIFT':[1,6,24],
            'sl_atr':[1.0,1.5],
            'tp_atr':[1.0,1.5,2.0],
            'threshold':[0.5,0.8],
            'gamma':[0,1],
            'reg_alpha':[0,0.1],
            'learning_rate':[0.01,0.1],
            'max_depth':[3,5],
            'subsample':[0.8,1.0]
        }
        model= XGBPartialOne()

        best_params, best_estimator, best_score= param_search_timeseries_cv(
            model, param_dist, n_iter=30, df_train=df_train, n_splits=5
        )
        print(f"   [INFO] best_cv_score(train)={best_score:.3f}, best_params={best_params}")
        if not best_estimator:
            continue

        SHIFT_val= best_params['SHIFT']
        col_label_test= f"target_return_{SHIFT_val}"
        if col_label_test not in df_test.columns:
            print(f"[WARN] missing {col_label_test} => skip.")
            continue

        dropcols= [c for c in df_test.columns if c.startswith('target_return_')]
        feats_test= [c for c in df_test.columns if c not in dropcols]
        X_test= df_test[feats_test]
        preds= best_estimator.predict(X_test)
        df_test['pred']= preds

        test_pnl= simulate_trades_partial50_time_limit(
            df_test,
            threshold= best_estimator.threshold,
            sl_atr= best_estimator.sl_atr,
            tp_atr= best_estimator.tp_atr,
            time_limit=24,
            fee=0.004,
            start_balance=1000.0
        )
        cumulative_pnl+= test_pnl
        print(f"   [RESULT] test_pnl={test_pnl:.3f},   cumulative_pnl={cumulative_pnl:.3f}")

        results.append({
            'test_month': test_start_str,
            'train_size': len(df_train),
            'test_size': len(df_test),
            'best_params': best_params,
            'train_cv_pnl': best_score,
            'test_pnl': test_pnl,
            'cumulative_pnl': cumulative_pnl
        })

    df_res= pd.DataFrame(results)
    return df_res

if __name__=="__main__":
    df_out= monthly_walk_forward(
        symbol_id=1,
        wf_start='2019-01-01',
        wf_end='2020-12-31'
    )
    print("\n=== SHIFT(1,6,24)+Partial(50%)+TimeLimit(24candles)+TSCV(5)+n_iter=30 => final results ===")
    print(df_out)
    print("Done.")
