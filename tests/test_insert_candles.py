import logging
from src.database_manager.database_manager import DatabaseManager

# Configureer logging om DEBUG berichten te tonen
logging.basicConfig(level=logging.DEBUG)


def test_insert_candles():
    db_manager = DatabaseManager(db_path='market_data.db')

    # Voorbeeld candle data
    candle_data = [
        (1735220873436, "BTC-USD", "1m", 50000.0, 50100.0, 49900.0, 50050.0, 10.5),
        (1735220873436, "BTC-USD", "1m", 50050.0, 50200.0, 50000.0, 50150.0, 12.3)
        # Voeg meer records toe indien nodig
    ]

    # Sla de candles op
    db_manager.save_candles(candle_data)

    # Haal de ingevoegde data op om te controleren
    fetched_candles = db_manager.fetch_data("candles", limit=10)
    print(fetched_candles)


if __name__ == "__main__":
    test_insert_candles()
