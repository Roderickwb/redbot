import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def single_iteration_small():
    train_start='2018-01-01'
    train_end='2018-01-31 23:59:59'
    test_start='2018-02-01'
    test_end='2018-02-28 23:59:59'
    SHIFT=5

    df_train = build_multi_tf_df(1, train_start, train_end)
    df_train = add_shift_target(df_train, SHIFT)
    print("Train data rows=", len(df_train))

    df_test = build_multi_tf_df(1, test_start, test_end)
    df_test = add_shift_target(df_test, SHIFT)
    print("Test data rows=", len(df_test))

    if len(df_train)<50 or len(df_test)<10:
        print("[WARN] Not enough data.")
        return

    # Features
    features = [c for c in df_train.columns if c!='target_return_1h']
    X_train = df_train[features]
    y_train = df_train['target_return_1h']
    X_test = df_test[features]
    y_test = df_test['target_return_1h']

    # Minimal param search
    param_candidates = [
        {'learning_rate':[0.1], 'max_depth':[3]}
    ]
    xgb = XGBRegressor(n_estimators=50, random_state=42)
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(xgb, param_candidates, scoring='neg_mean_squared_error', cv=1)
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    final_model = XGBRegressor(n_estimators=50, random_state=42, **best_params)
    final_model.fit(X_train, y_train)

    preds = final_model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print("Single iteration MSE=", mse, "best_params=", best_params)

if __name__=="__main__":
    single_iteration_small()
    print("Done.")
