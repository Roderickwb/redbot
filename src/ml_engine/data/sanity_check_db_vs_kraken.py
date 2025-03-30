import time
import random
import mysql.connector
import ccxt

# === [A] Databaseconfig & ccxt-instellingen ===
DB_CONFIG = {
    "host": "localhost",
    "user": "botuser",
    "password": "MySQL194860!",
    "database": "tradebot"
}

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


INTERVAL_MAP = {
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
}

# === [B] Functie: pak recentste X maanden uit DB ===
def get_recent_db_candles(symbol_id, interval, limit=5, lookback_days=180):
    """
    Haalt 'limit' willekeurige candles uit je DB (symbol_id, interval),
    die nieuwer zijn dan 'nu - lookback_days'.

    Standaard: 180 dagen (~6 maanden).
    """
    now_s = time.time()  # seconds since epoch
    cutoff_s = now_s - (lookback_days * 24 * 3600)
    cutoff_ms = int(cutoff_s * 1000)

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT timestamp_ms, open, high, low, close, volume
        FROM market_data
        WHERE symbol_id = %s
          AND `interval` = %s
          AND timestamp_ms >= %s
        ORDER BY RAND()
        LIMIT %s
    """
    cursor.execute(query, (symbol_id, interval, cutoff_ms, limit))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return rows

# === [C] Functie: CCXT call naar Kraken, 1 candle set ===
def fetch_kraken_candles(pair, timeframe, since_ms, limit=10):
    """
    Vraagt max 'limit' candles op bij Kraken, vanaf 'since_ms'.
    Returnt list van lists: [[ts_ms, open, high, low, close, volume], ...]
    """
    kraken = ccxt.kraken({'enableRateLimit': True})
    return kraken.fetch_ohlcv(
        symbol=pair,
        timeframe=timeframe,
        since=since_ms,
        limit=limit
    )

# === [D] Functie: spot-check DB vs. Kraken (sanity check) ===
def spot_check_kraken(symbol_id, interval, limit=5, lookback_days=180):
    """
    1) Pak 'limit' candles uit de DB (laatste X dagen).
    2) Voor elke candle: haal ~10 candles bij Kraken rond die timestamp.
    3) Zoek de dichtstbijzijnde candle -> Vergelijk O/H/L/C/volume.
    """
    if symbol_id not in DB_SYMBOL_TO_KRAKEN_PAIR:
        print(f"[FOUT] symbol_id={symbol_id} niet in DB_SYMBOL_TO_KRAKEN_PAIR.")
        return

    if interval not in INTERVAL_MAP:
        print(f"[FOUT] interval={interval} niet in INTERVAL_MAP.")
        return

    kraken_pair = DB_SYMBOL_TO_KRAKEN_PAIR[symbol_id]
    kraken_timeframe = INTERVAL_MAP[interval]

    # 1) DB-candles
    db_candles = get_recent_db_candles(
        symbol_id=symbol_id,
        interval=interval,
        limit=limit,
        lookback_days=lookback_days
    )
    if not db_candles:
        print(f"[FOUT] Geen candles in DB voor symbol_id={symbol_id}, interval={interval}, "
              f"laatste {lookback_days} dagen.")
        return

    print(f"\n=== Sanity-check {symbol_id} / {interval}, {lookback_days}d ===")
    print(f"   (random {limit} candles uit DB vs. Kraken)\n")

    # Interval in ms (voor marge)
    interval_ms_map = {'1h': 3600000, '4h': 14400000, '1d': 86400000}
    nominal_ms = interval_ms_map.get(interval, 3600000)

    # 2) Vergelijk
    for row in db_candles:
        ts_db = row['timestamp_ms']
        db_o = float(row['open'])
        db_h = float(row['high'])
        db_l = float(row['low'])
        db_c = float(row['close'])
        db_v = float(row['volume'])

        # Ruime marge "since_ms" om candles eromheen te krijgen
        since_ms = ts_db - (2 * nominal_ms)

        try:
            kraken_candles = fetch_kraken_candles(
                pair=kraken_pair,
                timeframe=kraken_timeframe,
                since_ms=since_ms,
                limit=10
            )
        except Exception as e:
            print(f"[FOUT] Oproep naar Kraken mislukt: {e}")
            continue

        if not kraken_candles:
            print(f"[WARN] Geen kraken-candles rond ts={ts_db}.")
            continue

        # Zoek "closest" candle in kraken_candles
        best_diff = None
        best_candle = None
        for c in kraken_candles:
            c_ts, c_open, c_high, c_low, c_close, c_vol = c
            diff = abs(c_ts - ts_db)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_candle = c

        if best_candle:
            c_ts, c_o, c_h, c_l, c_cl, c_vol = best_candle
            print(f"DB:     ts={ts_db},   O/H/L/C=({db_o:.5f}, {db_h:.5f}, {db_l:.5f}, {db_c:.5f}), vol={db_v:.2f}")
            print(f"Kraken: ts={c_ts}, diff={best_diff}ms, O/H/L/C=({c_o:.5f}, {c_h:.5f}, {c_l:.5f}, {c_cl:.5f}), vol={c_vol:.2f}\n")
        else:
            print(f"[WARN] Geen 'closest' candle gevonden rond ts={ts_db}.")


# === [E] MAIN - voorbeeldaanroepen ===
if __name__ == '__main__':
    # Voorbeeld: symbol_id=2 (ETH/EUR), interval='1h', 5 candles, laatste 180 dagen (~6 maanden)
    spot_check_kraken(symbol_id=2, interval='1h', limit=5, lookback_days=180)
    spot_check_kraken(symbol_id=8, interval='1h', limit=5, lookback_days=180)
    spot_check_kraken(symbol_id=12, interval='4h', limit=5, lookback_days=180)
    spot_check_kraken(symbol_id=14, interval='4h', limit=5, lookback_days=180)
    spot_check_kraken(symbol_id=20, interval='1d', limit=5, lookback_days=180)
    spot_check_kraken(symbol_id=22, interval='1d', limit=5, lookback_days=180)

    # Nog wat andere voorbeelden (pas aan naar wens):
    # spot_check_kraken(symbol_id=1, interval='1h', limit=3, lookback_days=90)
    # spot_check_kraken(symbol_id=20, interval='1h', limit=3, lookback_days=120)
    # etc.
