# test_strategy.py

import logging
from src.strategy.pullback_accumulate_strategy import DynamicStrategy  # Zorg ervoor dat de juiste strategy-module wordt geïmporteerd
from src.database.database_manager import DatabaseManager  # Voor het ophalen van historische data


# Functie om de historische data op te halen voor 1m, 5m en 10m candles
def get_historical_data_for_intervals():
    db_manager = DatabaseManager()  # Maak een instantie van DatabaseManager
    historical_data = {}

    # Verkrijg de historische data voor verschillende intervallen
    for interval in ['1m', '5m', '15m']:
        candles = db_manager.fetch_data('candles', limit=100, interval=interval)
        historical_data[interval] = candles
        logging.info(f"Gegevens voor {interval}m interval opgehaald.")

    return historical_data


# Functie om de strategie toe te passen op de historische data
def test_strategy_on_historical_data(strategy, symbol, db_manager):
    try:
        # Haal historische data op (bijvoorbeeld 300 candles)
        candles = db_manager.fetch_data("candles", market=symbol, limit=300)
        if candles.empty:
            logging.warning(f"Geen historische data beschikbaar voor {symbol}.")
            return

        # Zet data om naar DataFrame
        df = pd.DataFrame(candles)
        df.columns = ['timestamp', 'market', 'interval', 'open', 'high', 'low', 'close', 'volume']

        # Log de eerste paar rijen van de data om te zien of het goed opgehaald is
        logging.info(f"Laatste 5 rijen van de data:\n{df.tail()}")

        # Test de strategie op historische data (belangrijk hier is het aanroepen van execute_strategy)
        strategy.execute_strategy(symbol)  # Dit is waar de strategie wordt uitgevoerd

    except Exception as e:
        logging.error(f"Fout bij testen van de strategie: {e}")


