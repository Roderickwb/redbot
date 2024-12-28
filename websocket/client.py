import logging
import my_websocket
import json
import hmac
import hashlib
import time
import os
import sqlite3
from dotenv import load_dotenv

# Laad de configuratie vanuit .env
load_dotenv()

# Logging instellen
logging.basicConfig(
    filename='../websocket.log',
    level=logging.INFO,  # Veranderd naar INFO voor minder gedetailleerde logs in productie
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Jouw API-gegevens
API_KEY = os.getenv("API_KEY")  # Haal de API-sleutel uit de omgevingsvariabelen
API_SECRET = os.getenv("API_SECRET")  # Haal het geheime sleutel uit de omgevingsvariabelen
WS_URL = os.getenv("WS_URL", "wss://ws.bitvavo.com/v2/")  # Gebruik standaardwaarde als WS_URL niet is ingesteld

# Functie om de handtekening te genereren
def generate_signature(secret, timestamp, method, endpoint):
    message = f"{timestamp}{method}{endpoint}"
    signature = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    return signature

# Callback voor ontvangen berichten
def on_message(ws, message):
    data = json.loads(message)
    logging.info(f"Ontvangen bericht: {data}")  # Log het volledige bericht

    # Verwerk candle-data
    if data.get("event") == "candle":
        market = data.get("market")
        interval = data.get("interval")
        candle = data.get("candle", [])

        logging.info(f"Ontvangen candle-data: {candle}")  # Log de ontvangen candle-data

        if isinstance(candle, list) and len(candle) >= 6:  # Controleer dat het een enkele candle is
            try:
                formatted_candle = (
                    int(candle[0]),  # timestamp
                    market,
                    interval,
                    float(candle[1]),  # open
                    float(candle[2]),  # high
                    float(candle[3]),  # low
                    float(candle[4]),  # close
                    float(candle[5])  # volume
                )
                logging.info(f"Geformatteerde candle-data: {formatted_candle}")  # Log de geformatteerde candle
                save_candle(formatted_candle)  # Sla de candle op in de database
                logging.info(f"Candle opgeslagen ({market}, {interval}): {formatted_candle}")
            except (ValueError, IndexError) as e:
                logging.error(f"Ongeldige candle-data ontvangen: {candle}. Fout: {e}")

    # Verwerk ticker-data voor spread
    elif data.get("event") == "ticker":
        try:
            best_bid = float(data.get("bestBid", 0))
            best_ask = float(data.get("bestAsk", 0))
            spread = best_ask - best_bid
            ticker_data = {
                'market': data.get("market"),
                'bestBid': best_bid,
                'bestAsk': best_ask,
                'spread': spread
            }
            logging.info(f"Ontvangen ticker-data: {ticker_data}")  # Log de ontvangen ticker-data
            save_ticker(ticker_data)  # Sla de ticker op in de database
            logging.info(f"Ticker ontvangen: Spread: {spread}")
        except ValueError:
            logging.error("Fout bij het verwerken van ticker-waarden.")

    # Verwerk orderboekgegevens
    elif data.get("event") == "book":
        orderbook_data = {
            'market': data.get("market"),
            'bids': data.get("bids", []),
            'asks': data.get("asks", []),
        }
        logging.info(f"Ontvangen orderboek-data: {orderbook_data}")  # Log de ontvangen orderboek-data
        save_orderbook(orderbook_data)  # Sla het orderboek op in de database
        logging.info(f"Orderboek ontvangen: {orderbook_data}")


# Callback voor fouten
def on_error(ws, error):
    logging.error(f"Fout: {error}")


# Callback voor gesloten verbinding
def on_close(ws, close_status_code, close_msg):
    logging.warning("WebSocket-verbinding gesloten.")


# Callback voor open verbinding
def on_open(ws):
    logging.info("WebSocket-verbinding geopend.")  # Toegevoegde logregel

    # Stap 1: Authenticatie
    timestamp = int(time.time() * 1000)
    message = f"{timestamp}GET/v2/websocket"
    signature = hmac.new(API_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
    auth_payload = {
        "action": "authenticate",
        "key": API_KEY,
        "signature": signature,
        "timestamp": timestamp
    }
    logging.debug(f"Auth-payload: {json.dumps(auth_payload, indent=2)}")
    ws.send(json.dumps(auth_payload))
    logging.info("Authenticatie verzonden.")

    # Stap 2: Abonneren op kanalen
    subscribe_payload = {
        "action": "subscribe",
        "channels": [
            {
                "name": "candles",
                "interval": ["1m", "5m", "15m"],
                "markets": ["XRP-EUR"]
            },
            {
                "name": "ticker",
                "markets": ["XRP-EUR"]  # De markten moeten expliciet worden vermeld voor ticker
            },
            {
                "name": "book",
                "markets": ["XRP-EUR"]  # De markten moeten expliciet worden vermeld voor orderboek
            }
        ]
    }


    logging.debug(f"Subscribe-payload: {json.dumps(subscribe_payload, indent=2)}")
    ws.send(json.dumps(subscribe_payload))
    logging.info("Abonnementen verzonden.")


def save_candle(data):
    conn = sqlite3.connect("../market_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            timestamp INTEGER PRIMARY KEY,
            market TEXT,
            interval TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    """)
    cursor.execute("""
        INSERT OR REPLACE INTO candles (timestamp, market, interval, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()
    conn.close()


def save_orderbook(data):
    conn = sqlite3.connect("../market_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orderbook (
            timestamp INTEGER PRIMARY KEY,
            market TEXT,
            bid_price REAL,
            bid_quantity REAL,
            ask_price REAL,
            ask_quantity REAL
        )
    """)

    # Sla bids op
    for bid in data['bids']:
        cursor.execute("""
            INSERT OR REPLACE INTO orderbook (timestamp, market, bid_price, bid_quantity)
            VALUES (?, ?, ?, ?)
        """, (int(time.time() * 1000), data['market'], float(bid[0]), float(bid[1])))

    # Sla asks op
    for ask in data['asks']:
        cursor.execute("""
            INSERT OR REPLACE INTO orderbook (timestamp, market, ask_price, ask_quantity)
            VALUES (?, ?, ?, ?)
        """, (int(time.time() * 1000), data['market'], float(ask[0]), float(ask[1])))

    conn.commit()
    conn.close()


def save_ticker(data):
    conn = sqlite3.connect("../market_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticker (
            timestamp INTEGER PRIMARY KEY,
            market TEXT,
            best_bid REAL,
            best_ask REAL,
            spread REAL
        )
    """)

    mapped_data = {
        "market": data['market'],
        "best_bid": float(data['bestBid']),
        "best_ask": float(data['bestAsk']),
        "spread": float(data['spread']),
    }

    cursor.execute("""
        INSERT OR REPLACE INTO ticker (timestamp, market, best_bid, best_ask, spread)
        VALUES (?, ?, ?, ?, ?)
    """, (
        int(time.time() * 1000),
        mapped_data['market'],
        mapped_data['best_bid'],
        mapped_data['best_ask'],
        mapped_data['spread']
    ))
    conn.commit()
    conn.close()


# WebSocket-verbinding opzetten
def start_websocket():
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    try:
        logging.info("WebSocket-client draait. Druk op Ctrl+C om te stoppen.")
        ws.run_forever()
    except KeyboardInterrupt:
        logging.info("WebSocket-client gestopt.")
        ws.close()


# Start de WebSocket
if __name__ == "__main__":
    start_websocket()