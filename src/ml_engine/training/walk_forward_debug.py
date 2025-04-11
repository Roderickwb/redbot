"""
walk_forward_debug.py
Geen XGBoost, geen model. Alleen aggregator calls met debug-prints.
"""

import datetime
import pandas as pd

# PAS DIT AAN op jouw projectstructuur
# Zorg dat aggregator_with_debug.py in 'src/ml_engine/data/' staat:
from src.ml_engine.data.aggregator_with_debug import build_multi_timeframe_dataset

def single_month_walk_forward_debug():
    """
    Train-periode: 2020-01-01 t/m 2020-01-31
    Test-periode: 2020-02-01 t/m 2020-02-29
    SHIFT=5
    """
    train_start = '2020-01-01'
    train_end   = '2020-01-31 23:59:00'
    test_start  = '2020-02-01'
    test_end    = '2020-02-29'

    symbol_id=1
    SHIFT=5

    print("\n[DEBUG WF] Build TRAIN dataset ...")
    df_train = build_multi_timeframe_dataset(
        symbol_id=symbol_id,
        shift=SHIFT,
        start_date=train_start,
        end_date=train_end
    )
    print("[DEBUG WF] df_train rows=", len(df_train))
    if len(df_train) > 0:
        print(df_train.head(3))

    print("\n[DEBUG WF] Build TEST dataset ...")
    df_test = build_multi_timeframe_dataset(
        symbol_id=symbol_id,
        shift=SHIFT,
        start_date=test_start,
        end_date=test_end
    )
    print("[DEBUG WF] df_test rows=", len(df_test))
    if len(df_test) > 0:
        print(df_test.head(3))

    print("\n[DEBUG WF] Done. No model, no param search.")


if __name__ == "__main__":
    single_month_walk_forward_debug()
