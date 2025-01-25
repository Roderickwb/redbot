# historical_loader.py

import time
from python_bitvavo_api.bitvavo import Bitvavo
from src.database_manager.database_manager import DatabaseManager

############################################
# 1) CONFIGUREER HIER
############################################
API_KEY = "JOUW_API_KEY"
API_SECRET = "JOUW_API_SECRET"

DB_PATH = "C:/Users/My ACER/PycharmProjects/PythonProject4/data/market_data.db"
# Of waar je database ook staat

MARKETS = ["BTC-EUR", "ETH-EUR", "XRP-EUR", "DOGE-EUR", "SOL-EUR"]
INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]

DAYS_TO_LOAD = 20  # 20 dagen


############################################
# 2) DEFINIEER DE FUNCTIE DIE HISTORISCHE CANDLES OPHAALT
############################################
def load_candles_for_symbol_interval(bitvavo, db_manager, symbol, interval, days):
    """
    Haal historische candles op voor 'symbol' en 'interval',
    vanaf (nu - days) tot nu.
    """

    # 2.1) Bereken start-timestamp (ms)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000  # 20 dagen geleden in ms

    # 2.2) Gebruik bitvavo.candles() met start param
    # Let op: Bitvavo doc zegt dat je:
    #  - limit=1000 (max)
    #  - "start": start_ms
    #  - "end": optioneel
    #  We doen het in 1 call. Als 1 call niet genoeg data geeft (meer dan 1000 candles),
    #  kun je in een loop meerdere calls doen, maar 20 dagen * 1m = 28800 candles, dat is
    #  meer dan 1000, dus we doen het in stukjes.

    # Laten we dit in een while-lus gieten, tot we "klaar" zijn:
    all_rows = []
    batch_limit = 1000
    current_start = start_ms

    while True:
        print(f"Haalt candles op voor {symbol}-{interval} met start={current_start} (ms) ...")
        result = bitvavo.candles(symbol, interval, {
            "limit": batch_limit,
            "start": current_start
            # "end": ...  # we laten 'end' weg, dan neemt hij "nu"
        })

        if not isinstance(result, list):
            # Als er een error is, krijg je een dict met 'error' etc.
            print(f"Fout bij ophalen candles: {result}")
            break

        if len(result) == 0:
            print(f"Geen data meer voor {symbol}-{interval} (len=0). Stop.")
            break

        # 2.3) Format result
        rows = []
        for c in result:
            # c = [timestamp, open, high, low, close, volume]
            ts = int(c[0])
            open_ = float(c[1])
            high_ = float(c[2])
            low_ = float(c[3])
            close_ = float(c[4])
            volume = float(c[5])
            # Let op: DB vult "market", "interval"
            rows.append((ts, symbol, interval, open_, high_, low_, close_, volume))

        # 2.4) Sla op in DB
        db_manager.save_candles(rows)

        # 2.5) flush
        db_manager.flush_candles()

        all_rows.extend(rows)
        print(f"{len(rows)} candles binnengehaald in deze batch. Totaal={len(all_rows)}")

        # 2.6) Bepaal de timestamp van de laatste candle
        #      Meestal is result gesorteerd van oud->nieuw
        last_ts = rows[-1][0]
        # We verhogen current_start = last_ts + 1 ms
        # Zodat de volgende call data ophaalt NA de laatst bekende candle
        current_start = last_ts + 1

        # Break als we "nu" bereikt hebben of we wel genoeg data hebben
        if last_ts >= now_ms:
            print("We hebben 'nu' bereikt, stop.")
            break

        if len(result) < batch_limit:
            print("Minder dan 1000 candles in de batch, dus we zijn waarschijnlijk klaar.")
            break

    print(f"==> Klaar met {symbol}-{interval}, totaal {len(all_rows)} candles opgehaald.\n")


############################################
# 3) MAIN-FUNCTIE
############################################
def main():
    # 3.1) Maak de bitvavo–object
    bitvavo = Bitvavo({
        'APIKEY': API_KEY,
        'APISECRET': API_SECRET,
        'ACCESSWINDOW': 10000
    })

    # 3.2) Maak DatabaseManager
    db_manager = DatabaseManager(DB_PATH)
    print(f"DB Manager geinit op {DB_PATH}")

    # 3.3) Voor elk symbool en interval => load historical
    for sym in MARKETS:
        for interval in INTERVALS:
            load_candles_for_symbol_interval(bitvavo, db_manager, sym, interval, DAYS_TO_LOAD)

    print("Alle historische candles geladen.")


if __name__ == "__main__":
    main()
