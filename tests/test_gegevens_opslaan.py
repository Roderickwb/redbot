# test_gegevens_opslaan.py

from datetime import datetime
from src.database_manager import DatabaseManager  # Voeg deze import toe

# Test data (dummy data voor candles)
test_data = [
    (int(datetime.now().timestamp() * 1000), 'XRP-EUR', '5m', 2.00, 2.05, 1.95, 2.03, 10000),
    (int((datetime.now().timestamp() - 60) * 1000), 'XRP-EUR', '5m', 2.02, 2.06, 1.98, 2.04, 12000)
]

# Instantieer de DatabaseManager
db_manager = DatabaseManager()

# Sla testdata op
db_manager.save_candles(test_data)

# Haal de data op uit de database om te controleren of het goed is opgeslagen
fetched_data = db_manager.fetch_data('candles', limit=10)
print(fetched_data)
