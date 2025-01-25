#############################################
# scripts/historical_data.py
#############################################

import requests
import logging

logger = logging.getLogger(__name__)

def fetch_historical_candles(market="XRP-EUR", interval="1m", limit=500):
    """
    Haal historische candles op van Bitvavo via hun REST endpoint.
    Dit geeft een lijst van lijsten terug:
    [
      [t_ms, open, high, low, close, volume],
      [t_ms, open, high, low, close, volume],
      ...
    ]
    """
    try:
        url = f"https://api.bitvavo.com/v2/{market}/candles"
        params = {
            "interval": interval,
            "limit": limit
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()  # Ruwe candle-data direct van de API
        logger.info(f"Haalde {len(data)} {interval}-candles op voor {market}.")
        return data
    except Exception as e:
        logger.error(f"Fout bij ophalen van {interval}-candles voor {market}: {e}")
        return []

def transform_candle_data(raw_data, market, interval):
    """
    Zet de ruwe candle-lijst om naar jouw tuple-formaat:
      (timestamp, market, interval, open, high, low, close, volume).
    Let op de volgorde: raw_data[i] = [t_ms, open, high, low, close, volume]
    """
    transformed = []
    for row in raw_data:
        # row = [timestamp_ms, open, high, low, close, volume]
        t_ms = int(row[0])
        open_ = float(row[1])
        high = float(row[2])
        low = float(row[3])
        close = float(row[4])
        volume = float(row[5])

        record = (t_ms, market, interval, open_, high, low, close, volume)
        transformed.append(record)
    return transformed

def create_10m_candles(candles_5m):
    """
    Maak 10m-candles door telkens 2 stuks 5m-candles samen te voegen.
    candles_5m is een lijst van tuples (timestamp, market, '5m', open, high, low, close, volume).
    Let op: dit is een voorbeeld. Zorg dat de 5m-candles op volgorde staan en direct op elkaar aansluiten.
    """
    ten_min_candles = []
    i = 0
    while i < len(candles_5m) - 1:
        c1 = candles_5m[i]
        c2 = candles_5m[i+1]

        # c1: (t1, market, interval, open1, high1, low1, close1, vol1)
        t1, mkt, _, open1, high1, low1, close1, vol1 = c1
        t2, _, _, open2, high2, low2, close2, vol2 = c2

        # We nemen aan dat c2 direct na c1 komt.
        # Voor 10m-candle:
        # - timestamp is t1 (de start),
        # - open = open1,
        # - high = max(high1, high2),
        # - low = min(low1, low2),
        # - close = close2,
        # - volume = vol1 + vol2,
        # - interval = "10m"
        ten_ts = t1
        open_ = open1
        high = max(high1, high2)
        low = min(low1, low2)
        close = close2
        vol = vol1 + vol2
        interval_10 = "10m"

        record_10m = (ten_ts, mkt, interval_10, open_, high, low, close, vol)
        ten_min_candles.append(record_10m)

        i += 2  # ga 2 candles verder
    return ten_min_candles

