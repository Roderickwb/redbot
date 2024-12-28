import sqlite3
import pandas as pd
import time

DB_FILE = "market_data.db"


# Functie om een databaseverbinding te maken
def get_connection():
    try:
        conn = sqlite3.connect(DB_FILE)
        return conn
    except sqlite3.Error as e:
        print(f"Databasefout: {e}")
        return None


# Functie om de ticker-tabel aan te maken of bij te werken
def create_ticker_table():
    conn = get_connection()
    if conn is None:
        return

    cursor = conn.cursor()

    # Controleer of de tabel 'ticker' bestaat en de juiste kolommen heeft
    cursor.execute("PRAGMA table_info(ticker)")
    columns = [column[1] for column in cursor.fetchall()]
    print("Aanwezige kolommen in ticker-tabel:", columns)

    # Als de kolommen 'best_bid', 'best_ask', 'spread' ontbreken, voer dan een ALTER TABLE uit
    if 'best_bid' not in columns:
        cursor.execute("ALTER TABLE ticker ADD COLUMN best_bid REAL")
    if 'best_ask' not in columns:
        cursor.execute("ALTER TABLE ticker ADD COLUMN best_ask REAL")
    if 'spread' not in columns:
        cursor.execute("ALTER TABLE ticker ADD COLUMN spread REAL")

    # Maak een index op de timestamp-kolom voor snellere zoekopdrachten
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker_timestamp ON ticker (timestamp)
    """)

    conn.commit()
    conn.close()


# Functie om ticker-gegevens op te slaan
def save_ticker(data):
    conn = get_connection()
    if conn is None:
        return

    cursor = conn.cursor()

    # Zorg ervoor dat we de juiste namen gebruiken voor best_bid en best_ask
    best_bid = data.get('bestBid', 0.0)  # Als 'bestBid' niet aanwezig is, zet op 0.0
    best_ask = data.get('bestAsk', 0.0)  # Als 'bestAsk' niet aanwezig is, zet op 0.0
    spread = best_ask - best_bid if best_bid and best_ask else 0.0

    try:
        cursor.execute("""
            INSERT OR REPLACE INTO ticker (timestamp, market, best_bid, best_ask, spread)
            VALUES (?, ?, ?, ?, ?)
        """, (data['timestamp'], data['market'], best_bid, best_ask, spread))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Fout bij het opslaan van ticker-gegevens: {e}")
    finally:
        conn.close()


# Functie om candle-gegevens op te slaan
def save_candles(data):
    conn = get_connection()
    if conn is None:
        return

    cursor = conn.cursor()

    # Maak de candles-tabel indien nog niet aanwezig
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

    # Maak een index op de timestamp-kolom voor snellere zoekopdrachten
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles (timestamp)
    """)

    try:
        cursor.executemany("""
            INSERT OR REPLACE INTO candles (timestamp, market, interval, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, data)  # Gebruik executemany voor batch-inserties
        conn.commit()
    except sqlite3.Error as e:
        print(f"Fout bij het opslaan van candle-gegevens: {e}")
    finally:
        conn.close()


# Functie om de laatste candles op te halen
def get_candles():
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()

    query = "SELECT * FROM candles ORDER BY timestamp DESC LIMIT 10"  # Halen van de laatste 10 candles
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# Functie om orderboek-gegevens op te slaan
def save_orderbook(data):
    conn = get_connection()
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


# Functie om gegevens op te halen uit een opgegeven tabel
def fetch_data(table_name, limit=50):
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()  # Retourneer een lege DataFrame bij fout

    query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT {limit}"
    try:
        df = pd.read_sql_query(query, conn)
    except sqlite3.Error as e:
        print(f"Fout bij het ophalen van gegevens uit {table_name}: {e}")
        return pd.DataFrame()  # Retourneer een lege DataFrame bij fout
    finally:
        conn.close()

    return df










