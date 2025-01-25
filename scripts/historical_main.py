#############################################
# scripts/historical_main.py
#############################################

import logging

from src.database_manager.database_manager import DatabaseManager
from scripts.historical_data import (
    fetch_historical_candles,
    transform_candle_data,
    create_10m_candles
)

logger = logging.getLogger(__name__)


def main():
    # 1) Maak DatabaseManager aan, zodat je DB-tabellen hebt.
    db_manager = DatabaseManager()

    markets = ["XRP-EUR", "BTC-EUR", "ETH-EUR", "DOGE-EUR", "SOL-EUR"]  # <-- ADD
    # Voor demo: 1m, 5m, 15m
    intervals = ["1m", "5m", "15m"]
    limit = 200  # Haal 200 candles op (pas aan naar wens)

    # 2) Voor elke market + interval => haal data op + DB opslaan
    for market in markets:  # <-- CHANGED: was 1 market, nu loop
        for interval in intervals:
             logger.info(f"=== Historische data ophalen voor {market} {interval} ===")
             # A) haal de raw data op
             raw_data = fetch_historical_candles(market=market, interval=interval, limit=limit)
             # B) transformeer naar jouw tuple
             candle_tuples = transform_candle_data(raw_data, market, interval)
             # C) sla op in DB
             db_manager.save_candles(candle_tuples)
             logger.info(f"{len(candle_tuples)} {interval}-candles opgeslagen in DB voor {market}.")

    # 3) Creëer 10m-candles uit 5m-candles.
    #    We hebben net 5m-candles in de DB, maar we kunnen ze direct in memory gebruiken
    #    of uit DB fetchen. Hier laten we zien hoe je ze uit memory gebruikt:

    for market in markets:  # <-- ADD: loop over alle markten
        logger.info(f"--- 10m-candles genereren voor {market} op basis van 5m ---")
        raw_5m_for_10m = fetch_historical_candles(market=market, interval="5m", limit=limit)
        c5m = transform_candle_data(raw_5m_for_10m, market, "5m")

        c10m = create_10m_candles(c5m)
        if c10m:
            logger.info(f"Ga {len(c10m)} stuks '10m' candles opslaan voor {market}.")
            db_manager.save_candles(c10m)
            logger.info("10m-candles succesvol opgeslagen in DB.")
        else:
            logger.warning(f"Geen 10m-candles gegenereerd voor {market}; mogelijk te weinig 5m-data?")

    logger.info("Klaar met het ophalen van alle historische data en opslaan in DB.")


if __name__ == "__main__":
    # Optioneel logging instellen
    logging.basicConfig(level=logging.INFO)
    main()
