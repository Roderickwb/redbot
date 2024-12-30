from datetime import datetime, timezone
import pytz

import sqlite3
import pandas as pd
import time
import logging
import os
import json  # Zorg ervoor dat json is geïmporteerd voor het opslaan van bids en asks
from datetime import datetime, timezone, timedelta

def get_current_local_timestamp():
    """Geeft de huidige tijd in CET in milliseconden."""
    cet = timezone(timedelta(hours=1))  # CET is UTC+1
    return int(datetime.now(cet).timestamp() * 1000)  # Tijd in milliseconden


# Configureer logging met tijdstempels en logniveau
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Bepaal het absolute pad naar market_data.db in de hoofdmap
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
DB_FILE = os.path.join(project_root, 'market_data.db')

class DatabaseManager:
    def __init__(self, db_path=DB_FILE):
        self.db_path = db_path
        logging.info(f"Database pad: {self.db_path}")
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.connection.cursor()
            logging.info(f"Verbonden met database: {self.db_path}")
            self.create_tables()
        except sqlite3.Error as e:
            logging.error(f"Fout bij verbinden met database: {e}")
            raise RuntimeError("Kan geen verbinding maken met de database.")
            self.connection = None
            self.cursor = None

    def __del__(self):
        """Sluit de databaseverbinding netjes af wanneer de instantie wordt verwijderd."""
        if self.connection:
            try:
                self.connection.commit()  # Zorg ervoor dat openstaande wijzigingen worden opgeslagen
                self.connection.close()  # Sluit de verbinding
                logging.info("Databaseverbinding netjes afgesloten.")
            except sqlite3.Error as e:
                logging.error(f"Fout bij het sluiten van de databaseverbinding: {e}")

    def execute_query(self, query, params=(), retries=5, delay=0.1):
        """Voer een SQL-query uit met retry logica bij vergrendeling."""
        for attempt in range(retries):
            try:
                self.cursor.execute(query, params)
                self.connection.commit()
                return
            except sqlite3.OperationalError as e:
                if 'locked' in str(e).lower():
                    logging.warning(f"Database is vergrendeld. Retry {attempt + 1}/{retries}...")
                    time.sleep(delay)
                else:
                    logging.error(f"OperationalError: {e}")
                    raise
        logging.error("Maximale aantal retries bereikt. Query niet uitgevoerd.")
        raise sqlite3.OperationalError("Database is vergrendeld na meerdere retries.")

    def get_table_count(self, table_name):
        """Haal het aantal records op uit een tabel."""
        try:
            self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = self.cursor.fetchone()[0]
            return count
        except Exception as e:
            logging.error(f"Error getting count from {table_name}: {e}")
            return 0


    def create_ticker_table(self):
        """Maakt de ticker-tabel aan of werkt deze bij."""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS ticker (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time INTEGER NOT NULL,
                    market TEXT NOT NULL,
                    "best bid" REAL,
                    "best ask" REAL,
                    spread REAL
                )
            """)
            self.cursor.execute("PRAGMA table_info(ticker)")
            columns = [column[1] for column in self.cursor.fetchall()]
            logging.info(f"Aanwezige kolommen in ticker-tabel: {columns}")

            # Voeg kolommen toe indien ze ontbreken
            if 'best_bid' not in columns:
                self.cursor.execute("ALTER TABLE ticker ADD COLUMN best_bid REAL")
            if 'best_ask' not in columns:
                self.cursor.execute("ALTER TABLE ticker ADD COLUMN best_ask REAL")
            if 'spread' not in columns:
                self.cursor.execute("ALTER TABLE ticker ADD COLUMN spread REAL")

            # Maak index aan voor snellere queries op timestamp
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_timestamp ON ticker (timestamp)")
            self.connection.commit()
            logging.info("Ticker table is klaar.")
        except sqlite3.Error as e:
            logging.error(f"Error creating/updating ticker table: {e}")

    def create_candles_table(self):
        """Maakt de candles-tabel aan of werkt deze bij."""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    timestamp INTEGER PRIMARY KEY,
                    market TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL
                );
            """)
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles (timestamp)")
            self.connection.commit()
            logging.info("Candles table is klaar.")
        except Exception as e:
            logging.error(f"Error creating candles table: {e}")

    def create_orderbook_tables(self):
        """Maakt de orderbook_bids en orderbook_asks tabellen aan indien deze nog niet bestaan."""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_bids (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    market TEXT NOT NULL,
                    bid_p REAL NOT NULL,
                    bid_q REAL NOT NULL
                )
            """)
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_orderbook_bids_timestamp ON orderbook_bids (timestamp)")

            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_asks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    market TEXT NOT NULL,
                    ask_p REAL NOT NULL,
                    ask_q REAL NOT NULL
                )
            """)

            self.connection.commit()
            logging.info("Orderbook_bids en orderbook_asks tabellen zijn klaar.")
        except sqlite3.Error as e:
            logging.error(f"Error creating orderbook tables: {e}")

    def create_tables(self):
        """Creëer de benodigde tabellen indien deze nog niet bestaan."""
        try:
            self.create_candles_table()
            self.create_ticker_table()
            self.create_orderbook_tables()
            logging.info("Alle tabellen zijn succesvol aangemaakt of bijgewerkt.")
        except Exception as e:
            logging.error(f"Error creating tables: {e}")

    def save_candles(self, data):
        """Slaat de candle-data op in de database, met validatie en CET-timestamp."""
        try:
            valid_data = []
            for record in data:
                # Controleer of het record de juiste lengte heeft
                if len(record) != 8:
                    logging.warning(f"Ongeldig record genegeerd: {record}")
                    continue

                # Extracteer velden uit het record
                timestamp, market, interval, open_, high, low, close, volume = record

                # Controleer of alle velden geldig zijn
                if (isinstance(timestamp, int) and
                        isinstance(market, str) and
                        isinstance(interval, str) and
                        isinstance(open_, (int, float)) and
                        isinstance(high, (int, float)) and
                        isinstance(low, (int, float)) and
                        isinstance(close, (int, float)) and
                        isinstance(volume, (int, float))):

                    # Gebruik CET-timestamp indien de ontvangen timestamp ongeldig is
                    if not (isinstance(timestamp, int) and timestamp > 0):
                        timestamp = get_current_local_timestamp()

                    # Voeg record toe aan geldige data
                    valid_data.append((timestamp, market, interval, open_, high, low, close, volume))
                else:
                    logging.warning(f"Ongeldig record genegeerd: {record}")

            # Sla geldige records op in de database
            if valid_data:
                batch_size = 100  # Batchverwerking voor efficiëntie
                for i in range(0, len(valid_data), batch_size):
                    batch = valid_data[i:i + batch_size]
                    self.cursor.executemany("""
                        INSERT INTO candles (timestamp, market, interval, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, batch)
                    self.connection.commit()
                logging.info(f"{len(valid_data)} candle records succesvol opgeslagen.")
            else:
                logging.warning("Geen geldige candle data om op te slaan.")
        except Exception as e:
            logging.error(f"Error bij opslaan van candle data: {e}")


    def save_ticker(self, data):
        """Slaat de ticker-data op in de database."""
        try:
            timestamp = get_current_local_timestamp()  # Gebruik CET-tijd
            best_bid = data.get('bestBid', 0.0)
            best_ask = data.get('bestAsk', 0.0)
            spread = best_ask - best_bid

            # Voeg deze regel toe om te loggen welke data wordt verwerkt
            logging.info(f"Ticker data ontvangen voor opslag: {data}")

            self.cursor.execute("""
                INSERT INTO ticker (timestamp, market, best_bid, best_ask, spread)
                VALUES (?, ?, ?, ?, ?)
            """, (
                timestamp,
                data['market'],
                best_bid,
                best_ask,
                spread
            ))
            self.connection.commit()
            logging.info(f"Ticker data succesvol opgeslagen: {data}")
        except Exception as e:
            logging.error(f"Error saving ticker data to the database: {e}")


    def save_orderbook(self, data):
        """Slaat orderbook data op in de database."""
        try:
            timestamp = get_current_local_timestamp()  # Gebruik CET-tijd
            market = data['market']
            bids = data.get('bids', [])
            asks = data.get('asks', [])

            # Insert bids
            for bid in bids:
                self.cursor.execute("""
                    INSERT INTO orderbook_bids (timestamp, market, bid_p, bid_q)
                    VALUES (?, ?, ?, ?)
                """, (
                    timestamp,
                    market,
                    float(bid[0]),
                    float(bid[1])
                ))

            # Insert asks
            for ask in asks:
                self.cursor.execute("""
                    INSERT INTO orderbook_asks (timestamp, market, ask_p, ask_q)
                    VALUES (?, ?, ?, ?)
                """, (
                    timestamp,
                    market,
                    float(ask[0]),
                    float(ask[1])
                ))

            self.connection.commit()
            logging.info("Orderbook data succesvol opgeslagen.")
        except Exception as e:
            logging.error(f"Error saving orderbook data to the database: {e}")

    def fetch_data(self, table_name, limit=100, market=None, interval=None):
        """Haal data op uit een opgegeven tabel."""
        try:
            base_query = f"SELECT * FROM {table_name}"
            params = []

            conditions = []
            if table_name == "candles":
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if interval:
                    conditions.append("interval = ?")
                    params.append(interval)
            elif table_name == "ticker" and market:
                conditions.append("market = ?")
                params.append(market)
            elif table_name in ["orderbook_bids", "orderbook_asks"] and market:
                conditions.append("market = ?")
                params.append(market)

            if conditions:
                base_query += " WHERE " + " AND ".join(conditions)

            base_query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            df = pd.read_sql_query(base_query, self.connection, params=tuple(params))
            logging.info(f"Fetched {len(df)} records from {table_name}.")
            return df
        except Exception as e:
            logging.error(f"Error fetching data from {table_name}: {e}")
            return pd.DataFrame()

    def drop_orderbook_tables(self):
        """Verwijder de orderbook_bids en orderbook_asks tabellen als deze bestaan (om ze opnieuw te creëren)."""
        try:
            self.cursor.execute("DROP TABLE IF EXISTS orderbook_bids")
            self.cursor.execute("DROP TABLE IF EXISTS orderbook_asks")
            self.connection.commit()
            logging.info("Orderbook_bids en orderbook_asks tabellen verwijderd.")
        except Exception as e:
            logging.error(f"Error dropping orderbook tables: {e}")

    def reset_orderbook_tables(self):
        """Verwijder en hercreëer de orderbook_bids en orderbook_asks tabellen."""
        self.drop_orderbook_tables()
        self.create_orderbook_tables()

    def close_connection(self):
        """Sluit de database verbinding."""
        if self.connection:
            self.connection.close()
            logging.info("Database verbinding gesloten.")
