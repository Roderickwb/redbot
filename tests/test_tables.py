import sqlite3

# Verbind met de database
db_path = "test_market_data.db"
connection = sqlite3.connect(db_path)
cursor = connection.cursor()

# Helperfunctie om records toe te voegen als ze niet bestaan
def insert_if_not_exists(cursor, query, check_query, check_params, insert_params):
    cursor.execute(check_query, check_params)
    if cursor.fetchone()[0] == 0:
        cursor.execute(query, insert_params)

# Voeg testdata toe aan candles
insert_if_not_exists(
    cursor,
    """
    INSERT INTO candles (timestamp, market, interval, open, high, low, close, volume)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
    "SELECT COUNT(*) FROM candles WHERE timestamp=? AND market=? AND interval=?",
    (1735464990, 'XRP-EUR', '1m'),
    (1735464990, 'XRP-EUR', '1m', 0.5, 0.6, 0.4, 0.55, 1000.0)
)

# Voeg testdata toe aan ticker
insert_if_not_exists(
    cursor,
    """
    INSERT INTO ticker (timestamp, market, best_bid, best_ask, spread)
    VALUES (?, ?, ?, ?, ?)
    """,
    "SELECT COUNT(*) FROM ticker WHERE timestamp=? AND market=?",
    (1735464991, 'XRP-EUR'),
    (1735464991, 'XRP-EUR', 0.57, 0.59, 0.02)
)

# Voeg testdata toe aan orderbook_bids
insert_if_not_exists(
    cursor,
    """
    INSERT INTO orderbook_bids (timestamp, market, bid_p, bid_q)
    VALUES (?, ?, ?, ?)
    """,
    "SELECT COUNT(*) FROM orderbook_bids WHERE timestamp=? AND market=? AND bid_p=? AND bid_q=?",
    (1735464992, 'XRP-EUR', 0.56, 150.0),
    (1735464992, 'XRP-EUR', 0.56, 150.0)
)

# Voeg testdata toe aan orderbook_asks
insert_if_not_exists(
    cursor,
    """
    INSERT INTO orderbook_asks (timestamp, market, ask_p, ask_q)
    VALUES (?, ?, ?, ?)
    """,
    "SELECT COUNT(*) FROM orderbook_asks WHERE timestamp=? AND market=? AND ask_p=? AND ask_q=?",
    (1735464993, 'XRP-EUR', 0.6, 200.0),
    (1735464993, 'XRP-EUR', 0.6, 200.0)
)

connection.commit()
connection.close()

print("Testdata succesvol toegevoegd zonder duplicaten.")

import sqlite3

db_path = "test_market_data.db"
connection = sqlite3.connect(db_path)
cursor = connection.cursor()

# Pas candles aan (al correct)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS candles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        market TEXT NOT NULL,
        interval TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        UNIQUE(timestamp, market, interval)
    );
""")

# Pas ticker aan
cursor.execute("""
    CREATE TABLE IF NOT EXISTS ticker (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        market TEXT NOT NULL,
        best_bid REAL,
        best_ask REAL,
        spread REAL,
        UNIQUE(timestamp, market)
    );
""")

# Pas orderbook_bids aan
cursor.execute("""
    CREATE TABLE IF NOT EXISTS orderbook_bids (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        market TEXT NOT NULL,
        bid_p REAL,
        bid_q REAL,
        UNIQUE(timestamp, market, bid_p, bid_q)
    );
""")

# Pas orderbook_asks aan
cursor.execute("""
    CREATE TABLE IF NOT EXISTS orderbook_asks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        market TEXT NOT NULL,
        ask_p REAL,
        ask_q REAL,
        UNIQUE(timestamp, market, ask_p, ask_q)
    );
""")

connection.commit()
connection.close()

print("Tabellen succesvol bijgewerkt met UNIQUE constraints.")




connection = sqlite3.connect("test_market_data.db")
cursor = connection.cursor()

tables = ["candles", "ticker", "orderbook_bids", "orderbook_asks"]
for table in tables:
    cursor.execute(f"SELECT * FROM {table}")
    rows = cursor.fetchall()
    print(f"Records in {table}:", rows)

connection.close()


