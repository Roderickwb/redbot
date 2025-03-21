import unittest
import os
import sys
import time
import logging  # Voeg deze regel toe om logging te importeren
import gc

# Voeg de hoofdmap toe aan sys.path om toegang te krijgen tot database_manager.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database_manager.database_manager import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Gebruik een aparte test database
        cls.test_db_path = 'test_market_data.db'
        # Verwijder de test database als deze al bestaat
        if os.path.exists(cls.test_db_path):
            os.remove(cls.test_db_path)
        cls.db_manager = DatabaseManager(db_path=cls.test_db_path)
        cls.db_manager.create_candles_table()
        cls.db_manager.create_ticker_table()
        cls.db_manager.create_orderbook_tables()


    @classmethod
    def tearDownClass(cls):
        # Optimaliseer en maak SQLite caches vrij
        cls.db_manager.connection.execute("PRAGMA optimize;")

        # Sluit de databaseverbinding als deze nog open is
        if cls.db_manager.connection:
            cls.db_manager.cursor.close()
            cls.db_manager.connection.close()
            logging.info("Databaseverbinding gesloten.")

        # Forceer garbage collection
        gc.collect()

        # Wacht even om ervoor te zorgen dat SQLite het bestand vrijgeeft
        time.sleep(0.1)

        # Controleer of het bestand bestaat en verwijder het
        if os.path.exists(cls.test_db_path):
            try:
                os.remove(cls.test_db_path)
                logging.info(f"Test database verwijderd: {cls.test_db_path}")
            except PermissionError as e:
                logging.error(f"Kan test database niet verwijderen: {e}")

    class TestDatabaseManager(unittest.TestCase):
        def setUp(self):
            self.db_manager = DatabaseManager("test_market_data.db")
            self.db_manager.create_all_tables()

        def tearDown(self):
            self.db_manager.close()
            os.remove("test_market_data.db")

    def test_create_tables(self):
        # Controleer of de tabellen zijn aangemaakt
        tables = ["candles", "ticker", "orderbook"]
        for table in tables:
            self.db_manager.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';")
            result = self.db_manager.cursor.fetchone()
            self.assertIsNotNone(result, f"Table {table} should exist.")

    def test_save_and_fetch_candles(self):
        # Voeg test data toe
        test_data = [
            (int(time.time()), 'XRP-EUR', '1m', 0.5, 0.6, 0.4, 0.55, 1000.0),
            (int(time.time()), 'XRP-EUR', '1m', 0.55, 0.65, 0.45, 0.6, 1100.0)
        ]
        self.db_manager.save_candles(test_data)
        df = self.db_manager.fetch_data("candles", limit=10, market="XRP-EUR", interval="1m")
        self.assertFalse(df.empty, "Candles data should not be empty.")
        self.assertEqual(len(df), 2, "Should fetch 2 candle records.")
        self.assertIn('timestamp', df.columns, "DataFrame should contain 'timestamp' column.")

    def test_save_and_fetch_ticker(self):
        # Voeg test ticker data toe
        test_ticker = {
            'timestamp': int(time.time()),
            'market': 'XRP-EUR',
            'market': 0.58,
            'bestBid': 0.57,
            'bestAsk': 0.59,
            'spread': 0.002
        }
        self.db_manager.save_ticker(test_ticker)
        df = self.db_manager.fetch_data("ticker", limit=10, market="XRP-EUR")
        self.assertFalse(df.empty, "Ticker data should not be empty.")
        self.assertEqual(len(df), 1, "Should fetch 1 ticker record.")
        self.assertIn('price', df.columns, "DataFrame should contain 'price' column.")
        self.assertIn('spread', df.columns, "DataFrame should contain 'spread' column.")
        self.assertAlmostEqual(df.iloc[0]['spread'], 0.02, places=6, msg="Spread should be correctly calculated.")

    def test_save_and_fetch_orderbook(self):
        # Voeg test orderbook data toe
        test_orderbook = {
            'market': 'XRP-EUR',
            'bids': [
                [0.57, 100.0],
                [0.56, 150.0]
            ],
            'asks': [
                [0.59, 100.0],
                [0.60, 200.0]
            ]
        }
        self.db_manager.save_orderbook(test_orderbook)
        df = self.db_manager.fetch_data("orderbook", limit=10, market="XRP-EUR")
        self.assertFalse(df.empty, "Orderbook data should not be empty.")
        self.assertEqual(len(df), 4, "Should fetch 4 orderbook records (2 bids + 2 asks).")
        self.assertIn('bid_price', df.columns, "DataFrame should contain 'bid_price' column.")
        self.assertIn('ask_price', df.columns, "DataFrame should contain 'ask_price' column.")


if __name__ == '__main__':
    unittest.main()

import unittest
from src.database_manager import DatabaseManager

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        """Setup voor testcases."""
        self.db_manager = DatabaseManager()  # Gebruik standaard DB_FILE vanuit config
        self.connection = self.db_manager.connect()

    def tearDown(self):
        """Ruim de verbinding op na elke test."""
        self.connection.close()

    def test_connection(self):
        """Test of de verbinding correct werkt."""
        self.assertIsNotNone(self.connection, "Databaseverbinding moet succesvol zijn.")

    def test_create_tables(self):
        """Controleer of tabellen worden aangemaakt."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = {"candles", "ticker", "orderbook_bids", "orderbook_asks"}
        self.assertTrue(expected_tables.issubset(set(tables)), "Alle benodigde tabellen moeten aanwezig zijn.")

    def test_insert_and_fetch_data(self):
        """Test of gegevens correct kunnen worden ingevoegd en opgehaald."""
        cursor = self.connection.cursor()

        # Voeg een testrecord in
        cursor.execute("""
            INSERT INTO ticker (timestamp, market, best_bid, best_ask, spread)
            VALUES (?, ?, ?, ?, ?)
        """, (1735464991, 'XRP-EUR', 0.57, 0.59, 0.02))
        self.connection.commit()

        # Haal het record op
        cursor.execute("SELECT * FROM ticker WHERE market = 'XRP-EUR'")
        rows = cursor.fetchall()

        self.assertEqual(len(rows), 1, "Er moet precies één record worden opgehaald.")
        self.assertEqual(rows[0][1], 1735464991, "De timestamp van het record moet correct zijn.")

if __name__ == "__main__":
    unittest.main()
