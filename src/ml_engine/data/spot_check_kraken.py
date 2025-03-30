import time
import random
import mysql.connector
import ccxt

# === [1] Databaseconfig & instellingen ===
DB_CONFIG = {
    "host": "localhost",
    "user": "botuser",
    "password": "MySQL194860!",
    "database": "tradebot"
}

# Symbol-mapping: jouw DB 'symbol_id' -> Kraken trading pair
DB_SYMBOL_TO_KRAKEN_PAIR = {
    1: 'XXBT/EUR',  # BTC/EUR op Kraken = XXBT/EUR (als 'XBT/EUR' niet werkt)
    2: 'ETH/EUR',
    8: 'AAVE/EUR',
    12: 'ADA/EUR',
    13: 'ALGO/EUR',
    14: 'XRP/EUR',
    15: 'XDG/EUR',  # Doge -> XDG/EUR
    16: 'SOL/EUR',
    17: 'DOT/EUR',
    18: 'MATIC/EUR',
    19: 'TRX/EUR',
    20: 'LTC/EUR',
    21: 'LINK/EUR',
    22: 'XLM/EUR',
    23: 'UNI/EUR',
    24: 'ATOM/EUR',
    25: 'ETC/EUR',
    26: 'SAND/EUR',
    27: 'AVAX/EUR',
    28: 'BCH/EUR'
}

# Mapping: DB-interval -> ccxt-timeframe
INTERVAL_MAP = {
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
}


# === [2] Functie: candles ophalen bij Kraken met ccxt ===
def fetch_kraken_candles(pair, timeframe, since_ms, limit=50000):
    """
    Haalt candles op bij Kraken via ccxt, vanaf 'since_ms' (ms sinds epoch)
    en met 'limit' candles (max).

    Return: [[ts_ms, open, high, low, close, volume], ...]
    """
    kraken = ccxt.kraken({'enableRateLimit': True})
    data = kraken.fetch_ohlcv(
        symbol=pair,
        timeframe=timeframe,
        since=since_ms,
        limit=limit
    )
    return data


# === [3] Functie: selecteer 'limit' random candles uit de DB, van de afgelopen X dagen ===
def get_db_candles_recent(symbol_id, interval, limit=5, lookback_days=365):
    """
    Haalt willekeurig 'limit' candles uit de DB, maar alleen
    candles van de afgelopen 'lookback_days'.

    - symbol_id: int (bijv. 2 = ETH/EUR)
    - interval: '1h', '4h', '1d'
    - limit: aantal random candles opvragen
    - lookback_days: aantal dagen terug vanaf 'nu' om te filteren
    """
    # Bereken cutoff in ms (nu - lookback_days)
    now_s = time.time()
    lookback_s = lookback_days * 24 * 3600
    cutoff_s = now_s - lookback_s
    cutoff_ms = int(cutoff_s * 1000)

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    query = f"""
        SELECT timestamp_ms, open, high, low, close, volume
        FROM market_data
        WHERE symbol_id = %s
          AND `interval` = %s
          AND timestamp_ms >= %s
        ORDER BY RAND()
        LIMIT {limit}
    """
    cursor.execute(query, (symbol_id, interval, cutoff_ms))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return rows


# === [4] Functie: Spot-check DB-candles vs. Kraken ===
def spot_check_kraken(symbol_id, interval, limit=5, lookback_days=365):
    """
    1) Selecteert 'limit' candles uit DB (symbol_id, interval),
       maar alleen de laatste 'lookback_days' dagen.
    2) Haalt data bij Kraken op rond die timestamps.
    3) Vergelijkt OHLC en volume.
    """
    if symbol_id not in DB_SYMBOL_TO_KRAKEN_PAIR:
        print(f"[FOUT] Onbekend kraken-pair voor symbol_id={symbol_id}. "
              f"Check DB_SYMBOL_TO_KRAKEN_PAIR.")
        return

    kraken_pair = DB_SYMBOL_TO_KRAKEN_PAIR[symbol_id]

    if interval not in INTERVAL_MAP:
        print(f"[FOUT] Interval '{interval}' staat niet in INTERVAL_MAP.")
        return

    kraken_timeframe = INTERVAL_MAP[interval]

    # 1) Pak candles uit DB
    db_candles = get_db_candles_recent(
        symbol_id=symbol_id,
        interval=interval,
        limit=limit,
        lookback_days=lookback_days
    )
    if not db_candles:
        print(f"[FOUT] Geen candles gevonden in DB voor symbol_id={symbol_id}, interval={interval}, "
              f"in de laatste {lookback_days} dagen.")
        return

    print(f"\n=== Spot-check symbol_id={symbol_id}, interval={interval}, kraken_pair={kraken_pair} ===")
    print(f"   (Random {limit} candles uit DB, max {lookback_days} dagen oud)\n")

    # Mapping naar ms
    interval_ms_map = {
        '1h': 3600000,
        '4h': 14400000,
        '1d': 86400000
    }
    nominal_ms = interval_ms_map.get(interval, 3600000)  # fallback = 1h

    # 2) Voor elke DB-candle => Kraken-data opvragen
    for candle in db_candles:
        ts_db = candle['timestamp_ms']
        db_open = float(candle['open'])
        db_high = float(candle['high'])
        db_low = float(candle['low'])
        db_close = float(candle['close'])
        db_vol = float(candle['volume'])

        # since_ms = ts_db - 2 * nominal_ms (marge)
        since_ms = ts_db - (2 * nominal_ms)

        # Kraken-data ophalen
        try:
            kraken_data = fetch_kraken_candles(
                pair=kraken_pair,
                timeframe=kraken_timeframe,
                since_ms=since_ms,
                limit=10
            )
        except Exception as e:
            print(f"[FOUT] Kan geen candles ophalen bij Kraken: {e}")
            continue

        if not kraken_data:
            print(f"Geen candles ontvangen van Kraken rond ts={ts_db}.")
            continue

        # Zoek candle die het dichtst bij ts_db ligt
        best_diff = None
        best_candle = None
        for c in kraken_data:
            c_ts, c_open, c_high, c_low, c_close, c_vol = c
            diff = abs(c_ts - ts_db)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_candle = c

        if best_candle:
            c_ts, c_open, c_high, c_low, c_close, c_vol = best_candle
            print(f"DB:     ts={ts_db}, "
                  f"O/H/L/C=({db_open:.5f}, {db_high:.5f}, {db_low:.5f}, {db_close:.5f}), vol={db_vol:.4f}")
            print(f"Kraken: ts={c_ts}, diff={best_diff}ms, "
                  f"O/H/L/C=({c_open:.5f}, {c_high:.5f}, {c_low:.5f}, {c_close:.5f}), vol={c_vol:.4f}\n")
        else:
            print(f"Geen 'closest' candle in kraken_data voor ts={ts_db}.")


# === [MAIN] ===
if __name__ == '__main__':
    # Voorbeeld: check de laatste 6 maanden (180 dagen) voor symbol_id=2 (ETH), interval='1h'
    spot_check_kraken(symbol_id=2, interval='1h', limit=5, lookback_days=180)

    # Enkele extra voorbeelden, pas naar wens aan:
    spot_check_kraken(symbol_id=8, interval='1h', limit=5, lookback_days=180)
    spot_check_kraken(symbol_id=16, interval='4h', limit=5, lookback_days=180)
    spot_check_kraken(symbol_id=19, interval='4h', limit=5, lookback_days=180)
    spot_check_kraken(symbol_id=22, interval='1d', limit=5, lookback_days=180)
    spot_check_kraken(symbol_id=28, interval='1d', limit=5, lookback_days=180)

    # Als je wilt:
    # spot_check_kraken(symbol_id=20, interval='1h', limit=5, lookback_days=365)
    # spot_check_kraken(symbol_id=2, interval='1d', limit=3, lookback_days=180)
