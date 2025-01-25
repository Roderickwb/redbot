# src/utils/fetch_historical.py

import requests
import time
import logging

def fetch_historical_candles(market, interval="1h", limit=1000, start_ms=None, end_ms=None):
    """
    Haal historische candles op bij Bitvavo via de REST API.

    :param market: Marktnaam, bijvoorbeeld "BTC-EUR"
    :param interval: Interval, bijvoorbeeld "1m", "5m", "15m", "1h", "4h", "1d"
    :param limit: Aantal candles per request (max 1000)
    :param start_ms: Starttijd in milliseconden (optional)
    :param end_ms: Eindtijd in milliseconden (optional)
    :return: Lijst van candles
    """
    base_url = "https://api.bitvavo.com/v2"
    headers = {
        "Content-Type": "application/json"
    }

    params = {
        "limit": limit
    }
    if start_ms:
        params["start"] = start_ms
    if end_ms:
        params["end"] = end_ms

    url = f"{base_url}/candles/{market}/{interval}"
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()  # List of lists: [timestamp, open, high, low, close, volume]
        logging.info(f"Gegevens opgehaald voor {market} - {interval}: {len(data)} candles.")
        return data
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP-fout bij ophalen candles voor {market} - {interval}: {http_err}")
    except Exception as err:
        logging.error(f"Onverwachte fout bij ophalen candles voor {market} - {interval}: {err}")
    return []
