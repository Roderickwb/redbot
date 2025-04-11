from aggregator import build_multi_timeframe_dataset

if __name__ == "__main__":
    print("[DEBUG] Start aggregator test for 2020-01-01 to 2020-02-15")
    df_test = build_multi_timeframe_dataset(
        symbol_id=1,
        shift=5,
        start_date='2020-01-01',
        end_date='2020-02-15'
    )
    print("[DEBUG] aggregator returned df with rows =", len(df_test))
    print(df_test.head())
