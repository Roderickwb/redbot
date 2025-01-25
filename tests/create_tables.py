import sqlite3

def create_tables(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Tabel voor candles
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

    # Tabel voor ticker
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticker (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            market TEXT NOT NULL,
            best_bid REAL,
            best_ask REAL,
            spread REAL
        );
    """)

    # Tabellen voor orderbook
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orderbook_bids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            market TEXT NOT NULL,
            bid_p REAL,
            bid_q REAL
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orderbook_asks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            market TEXT NOT NULL,
            ask_p REAL,
            ask_q REAL
        );
    """)

    connection.commit()
    connection.close()
    print("Tabellen zijn aangemaakt.")

# Voer dit uit
create_tables("test_market_data.db")
