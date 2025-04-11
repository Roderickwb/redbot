"""
walk_forward_example.py

Voorbeeld van maandelijkse walk-forward training + mini param search met XGBoost.
Maanden: 2020-02, 2020-03, 2020-04 als testperiodes.
Train = alle data tot net voor testmaand, Test = die testmaand.

LET OP:
- Zorg dat je aggregator.py (in data/) up-to-date is en importeer correct.
- Gebruik kleine periodes om geheugenproblemen te voorkomen.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# PAS DEZE IMPORTS AAN op jouw projectstructuur
from src.ml_engine.data.aggregator import build_multi_timeframe_dataset


def monthly_walk_forward(symbol_id=1, shift=5,
                         wf_start='2020-01-01', wf_end='2020-04-01'):
    """
    We genereren maand-begindata: 2020-01-01, 2020-02-01, 2020-03-01, 2020-04-01
    Dan lopen we in steps:
      i=1 => test_start=2020-02-01, train=[2020-01-01, 2020-01-31 23:59]
      i=2 => test_start=2020-03-01, train=[2020-01-01, 2020-02-29 23:59]
      i=3 => test_start=2020-04-01, train=[2020-01-01, 2020-03-31 23:59]
    """

    # Maak een lijst van maand-start datums
    dt_start = datetime.strptime(wf_start, "%Y-%m-%d")
    dt_end   = datetime.strptime(wf_end, "%Y-%m-%d")

    dt_list = []
    current = dt_start
    while current <= dt_end:
        dt_list.append(current)
        current = current + relativedelta(months=1)

    results = []

    for i in range(1, len(dt_list)):
        test_start = dt_list[i]
        test_end = test_start + relativedelta(months=1)  # 1 maand later
        if test_end > dt_end:
            test_end = dt_end  # niet buiten wf_end

        train_start = dt_list[0]  # we trainen vanaf het begin
        train_end   = test_start - timedelta(minutes=1)  # tot net voor test_start

        # Converteer naar strings
        train_start_str = train_start.strftime("%Y-%m-%d")
        train_end_str   = train_end.strftime("%Y-%m-%d %H:%M:%S")
        test_start_str  = test_start.strftime("%Y-%m-%d")
        test_end_str    = test_end.strftime("%Y-%m-%d")

        print(f"\n[WALK-FWD] Test-maand: {test_start_str} -> {test_end_str}")
        print(f"  Train: {train_start_str} -> {train_end_str}")

        # ---- BUILD TRAIN DATA ----
        df_train = build_multi_timeframe_dataset(
            symbol_id=symbol_id,
            shift=shift,
            start_date=train_start_str,
            end_date=train_end_str
        )
        if df_train.empty or len(df_train) < 50:
            print(" [WARN] Te weinig train-data, skip deze maand.")
            continue

        # ---- BUILD TEST DATA ----
        df_test = build_multi_timeframe_dataset(
            symbol_id=symbol_id,
            shift=shift,
            start_date=test_start_str,
            end_date=test_end_str
        )
        if df_test.empty or len(df_test) < 10:
            print(" [WARN] Te weinig test-data, skip deze maand.")
            continue

        # ---- Features & Target ----
        drop_targets = ['target_return_1h','target_return_4h','target_return_1d']
        X_train = df_train.drop(columns=drop_targets)
        y_train = df_train['target_return_1h']

        X_test  = df_test.drop(columns=drop_targets)
        y_test  = df_test['target_return_1h']

        # ---- Mini Param-Search ----
        #   We doen 2x2=4 combinaties, cv=1 => heel weinig rekenwerk
        param_grid = {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            # 'subsample': [0.8, 1.0],  # kun je erbij doen, maar laten we 'm weg
        }
        xgb = XGBRegressor(n_estimators=50, random_state=42)

        # cv=1 => er is eigenlijk geen cross-val, hij traint 1x per param.
        grid = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=1,
            n_jobs=-1
        )
        print("  [INFO] Start mini param-search ...")
        grid.fit(X_train, y_train)

        best_params = grid.best_params_
        best_score_cv = -grid.best_score_

        # Retrain met best_params op volledige train-set
        final_model = XGBRegressor(n_estimators=50, random_state=42, **best_params)
        final_model.fit(X_train, y_train)

        # Evaluate op test-set
        preds = final_model.predict(X_test)
        mse_test = mean_squared_error(y_test, preds)

        print(f"  [RESULT] best_params={best_params}, best_cv_mse={best_score_cv:.4f}, test_mse={mse_test:.4f}")

        results.append({
            'test_month_start': test_start_str,
            'test_month_end': test_end_str,
            'train_size': len(df_train),
            'test_size': len(df_test),
            'best_params': best_params,
            'cv_mse': best_score_cv,
            'test_mse': mse_test
        })

    # Einde for-lus
    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    # Probeer SHIFT=5, walk-forward van 2020-01-01 tot 2020-04-01
    # (3 testmaanden: feb, mar, apr)
    df_res = monthly_walk_forward(
        symbol_id=1,
        shift=5,
        wf_start='2020-01-01',
        wf_end='2020-04-01'
    )
    print("\n===== WALK-FORWARD RESULTS =====")
    print(df_res)
    print("Done.")
