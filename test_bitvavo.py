# bulk_import_candles.py

import os
import logging
from python_bitvavo_api.bitvavo import Bitvavo
from dotenv import load_dotenv

# importeer je DatabaseManager etc.
from src.database_manager.database_manager import DatabaseManager
from src.config.config import DB_FILE

load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Zet logging zo nodig
logging.basicConfig(level=logging.INFO)

db_manager = DatabaseManager(db_path=DB_FILE)
bitvavo = Bitvavo({
    "APIKEY": API_KEY,
    "APISECRET": API_SECRET
})

markets = ["BTC-EUR", "ETH-EUR", "XRP-EUR", "DOGE-EUR", "SOL-EUR"]
intervals = ["4h", "1d"]
limit = 30  # hoeveel candles je wilt ophalen

for market in markets:
    for interval in intervals:
        logging.info(f"== Haal {limit} {interval} candles op voor {market} ==")

        # Ophalen via REST
        candles = bitvavo.candles(market, interval, {"limit": limit})
        if not candles:
            logging.warning(f"Geen candles ontvangen voor {market} ({interval}).")
            continue

        # Omzetten naar records voor db_manager
        records = []
        for c in candles:
            # c = [timestamp, open, high, low, close, volume]
            ts_ms = c[0]
            open_ = float(c[1])
            high_ = float(c[2])
            low_ = float(c[3])
            close_ = float(c[4])
            volume = float(c[5])
            # Schema: (timestamp, market, interval, open, high, low, close, volume)
            records.append((ts_ms, market, interval, open_, high_, low_, close_, volume))

        logging.info(f"Invoeren {len(records)} records in DB voor {market} {interval} ...")
        db_manager.save_candles(records)
        # desgewenst expliciet flush:
        db_manager.flush_candles()

logging.info("Bulk import klaar. Check je DB op 4h en 1d candles!")
