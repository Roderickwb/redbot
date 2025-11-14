from gpt_trend_decider import ask_gpt_trend_decider, get_gpt_decision
import json


def run_simple_test():
    print("== Test 1: simpele GPT test ==\n")
    result = ask_gpt_trend_decider("Geef een korte testzin.")
    print("GPT RESPONSE:")
    print(result)


def run_decision_test():
    print("\n== Test 2: get_gpt_decision met dummy data ==\n")

    symbol = "BTC-EUR"
    algo_signal = "long_candidate"
    trend_1h = "bull"
    trend_4h = "bull"
    structure_1h = "higher_lows"
    structure_4h = "higher_lows"

    ema_1h = {
        "20": 1.23,
        "50": 1.20,
        "relation": "20>50",
        "slope_20": "up"
    }

    ema_4h = {
        "20": 1.22,
        "50": 1.18,
        "relation": "20>50",
        "slope_20": "up"
    }

    rsi_1h = 58.0
    rsi_slope_1h = 3.5
    macd_1h = 0.004

    rsi_4h = 62.0
    rsi_slope_4h = 2.1
    macd_4h = 0.006

    levels_1h = {
        "support": 1.20,
        "resistance": 1.28
    }

    levels_4h = {
        "support": 1.15,
        "resistance": 1.30
    }

    # paar dummy candles (normaal worden dit er 20)
    candles_1h = [
        {
            "o": 1.21, "h": 1.23, "l": 1.20, "c": 1.22,
            "ema20": 1.21, "ema50": 1.20,
            "rsi": 55, "macd_hist": 0.001,
            "vol": 10000,
            "top_wick_pct": 5.0, "bot_wick_pct": 3.0
        },
        {
            "o": 1.22, "h": 1.24, "l": 1.21, "c": 1.23,
            "ema20": 1.215, "ema50": 1.205,
            "rsi": 57, "macd_hist": 0.002,
            "vol": 12000,
            "top_wick_pct": 4.0, "bot_wick_pct": 2.0
        }
    ]

    candles_4h = [
        {
            "o": 1.18, "h": 1.22, "l": 1.17, "c": 1.21,
            "ema20": 1.19, "ema50": 1.17,
            "rsi": 60, "macd_hist": 0.003,
            "vol": 50000,
            "top_wick_pct": 6.0, "bot_wick_pct": 4.5
        },
        {
            "o": 1.21, "h": 1.25, "l": 1.20, "c": 1.24,
            "ema20": 1.20, "ema50": 1.18,
            "rsi": 63, "macd_hist": 0.004,
            "vol": 55000,
            "top_wick_pct": 7.0, "bot_wick_pct": 3.0
        }
    ]

    decision = get_gpt_decision(
        symbol=symbol,
        algo_signal=algo_signal,
        trend_1h=trend_1h,
        trend_4h=trend_4h,
        structure_1h=structure_1h,
        structure_4h=structure_4h,
        ema_1h=ema_1h,
        ema_4h=ema_4h,
        rsi_1h=rsi_1h,
        rsi_slope_1h=rsi_slope_1h,
        macd_1h=macd_1h,
        rsi_4h=rsi_4h,
        rsi_slope_4h=rsi_slope_4h,
        macd_4h=macd_4h,
        levels_1h=levels_1h,
        levels_4h=levels_4h,
        candles_1h=candles_1h,
        candles_4h=candles_4h,
    )

    print("GPT DECISION (raw):")
    print(decision)
    print("\nNetjes geformatteerd:")
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_simple_test()
    run_decision_test()
