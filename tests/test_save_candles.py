from src.database_manager.database_manager import DatabaseManager

# Maak een instantie van DatabaseManager
db_manager = DatabaseManager(db_path="../data/market_data.db")

# Testdata voor candles
test_data = [
    [1735340880000, "XRP-EUR", "1m", 2.0585, 2.0592, 2.0585, 2.0591, 400.78058],
    [1735340700000, "XRP-EUR", "5m", 2.0559, 2.0592, 2.0551, 2.0591, 26999.737608],
]

# Test de save_candles-methode
db_manager.save_candles(test_data)

print("Candles succesvol opgeslagen!")


