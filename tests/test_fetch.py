# test_fetch.py

import logging
from src.utils.fetch_historical import fetch_historical_candles

# Configureer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    market = "BTC-EUR"
    interval = "1m"
    candles = fetch_historical_candles(market, interval, limit=5)
    logging.info(f"Opgehaalde candles voor {market} - {interval}: {candles}")
