import time
from src.logger.logger import setup_logger
from src.database_manager.database_manager import DatabaseManager
from src.config.config import DB_FILE
from python_bitvavo_api.bitvavo import Bitvavo

logger = setup_logger("historical_fetcher", "logs/historical_fetcher.log")

# 1) DatabaseManager
db_manager = DatabaseManager(db_path=DB_FILE)

# 2) Een Bitvavo REST-instance:
bitvavo = Bitvavo({
    "APIKEY": "jouw_key",
    "APISECRET": "jouw_secret",
    "RESTURL": "https://api.bitvavo.com/v2",
    # evt. rate-limiting etc.
})


# 3) Functie om candles op te halen
def fetch_historical_candles(market, interval, limit=200):
    """
    Roept de Bitvavo-REST endpoint /candles aan, om X candles te ontvangen.
    """
    params = {
        "market": market,
        "interval": interval,
        "limit": limit,
    }
    # /candles/markt/interval?limit=...
    candles = bitvavo.candles(market, interval, {"limit": limit})
    return candles


def store_candles_in_db(market, interval, candles, db_manager):
    """
    Zet de ontvangen candles in het DB-format en sla ze op in 'candles'-tabel.
    """
    # Verwacht: candles is een list van lists [ [timestamp, open, high, low, close, volume], ... ]
    # (Bitvavo-stijl)
    if not candles:
        logger.warning(f"Geen candles voor {market}-{interval}.")
        return

    batch_of_candles = []
    for c in candles:
        # c = [timestamp, open, high, low, close, volume]
        # timestamp is in ms
        ts, op, hi, lo, cl, vol = c
        # In je DB-tabel is de kolomvolgorde (timestamp, market, interval, open, high, low, close, volume)
        formatted = (int(ts),
                     market,
                     interval,
                     float(op),
                     float(hi),
                     float(lo),
                     float(cl),
                     float(vol))
        batch_of_candles.append(formatted)

    db_manager.save_candles(batch_of_candles)
    logger.info(f"Opslaan gelukt: {len(batch_of_candles)} candles voor {market}-{interval}.")


if __name__ == "__main__":
    # 4) Definitie van de "nieuwe" coins
    new_coins = [
        "ADA-EUR", "TRX-EUR", "USDT-EUR",
        "SPX-EUR", "BNB-EUR", "XVG-EUR",
        "LTC-EUR", "TRX-EUR", "XLM-EUR",
        "PEPE-EUR", "LINK-EUR", "UNI-EUR",
        "ATOM-EUR"
    ]

    intervals = ["1h", "4h", "1d"]  # De twee timeframes die je wilt

    for coin in new_coins:
        for interval in intervals:
            # Bijv. 14 candles is vrij weinig – wellicht wil je meer, vb. 100 of 200?
            limit = 20
            logger.info(f"Ophalen {limit} candles voor {coin}, interval={interval}.")
            cdata = fetch_historical_candles(coin, interval, limit=limit)
            store_candles_in_db(coin, interval, cdata, db_manager)
            time.sleep(0.5)  # eventjes rust tussen de calls, ivm rate-limit
