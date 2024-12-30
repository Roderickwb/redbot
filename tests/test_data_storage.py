from src.database_manager.database_manager import DatabaseManager  # Zorg ervoor dat je de juiste import hebt


def test_data_storage():
    # Initialiseer de DatabaseManager
    db_manager = DatabaseManager()

    # Stap 1: Verwijder de bestaande orderbook-tabel (als deze bestaat)
    db_manager.drop_orderbook_table()

    # Stap 2: Maak de benodigde tabellen aan
    db_manager.create_ticker_table()
    db_manager.create_candles_table()
    db_manager.create_orderbook_table()  # Dit maakt de nieuwe, lege tabel aan


    data_ticker = {
        'timestamp': 1615455600000,
        'market': 'XRP-EUR',
        'bestBid': 1.0,
        'bestAsk': 1.05
    }

    data_orderbook = {
        'market': 'XRP-EUR',
        'bids': [
            [1.0, 100],
            [1.05, 200]
        ],
        'asks': [
            [1.1, 50],
            [1.15, 150]
        ]
    }

    # Gegevens opslaan
    db_manager.save_candles(data_candles)
    db_manager.save_ticker(data_ticker)
    db_manager.save_orderbook(data_orderbook)

    # Haal gegevens op
    df_candles = db_manager.fetch_data('candles')
    print(df_candles)


if __name__ == "__main__":
    test_data_storage()

