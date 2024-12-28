# test_save_candles.py

from database_manager import save_candles
import time

def test_save_candles():
    current_timestamp = int(time.time() * 1000)  # huidige tijd in milliseconden
    test_data = [
        (current_timestamp, "BTC-USD", "1m", 50000.0, 50100.0, 49900.0, 50050.0, 10.5),
        (current_timestamp + 60000, "BTC-USD", "1m", 50050.0, 50200.0, 50000.0, 50150.0, 12.3),
    ]
    save_candles(test_data)
    print("✅ Test candle data opgeslagen.")

if __name__ == "__main__":
    test_save_candles()
