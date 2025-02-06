# ============================================================
# src/database_manager/database_manager.py
# ============================================================

import sqlite3
import pandas as pd
import time
import logging
from datetime import datetime, timezone
import threading
import os

from src.config.config import DB_FILE

logger = logging.getLogger("database_manager")

def get_current_utc_timestamp_ms():
    """Geeft de huidige tijd in UTC in milliseconden (integer)."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)

class DatabaseManager:
    def __init__(self, db_path=DB_FILE):
        """
        Maakt verbinding met de SQLite-database en initialiseert de attributen.
        Roep zelf create_tables() aan als je alle tabellen wilt aanmaken.
        """
        self.db_path = db_path
        logger.debug(f"[DatabaseManager] DB_FILE: {self.db_path}")
        logger.info(f"[DatabaseManager] Database pad: {self.db_path}")

        try:
            # Verbind met de database en zet deze in WAL-modus
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
            self.connection.execute("PRAGMA journal_mode=WAL;")
            self.cursor = self.connection.cursor()

            # Lock voor concurrency (als er meerdere threads werken)
            self._db_lock = threading.Lock()

            logger.info(f"[DatabaseManager] Verbonden met database: {self.db_path}")

            # ============ Oude buffers voor 'candles' (enkelvoudige tabel) ============
            self.candle_buffer = []
            self.batch_size = 100

            # ============ NIEUW: aparte buffers voor Bitvavo vs Kraken-candles ========
            self.candle_buffer_bitvavo = []
            self.candle_buffer_kraken = []

            # (Eventueel kun je ook separate batch_size willen, of dezelfde delen.)
            self.batch_size_bitvavo = 100
            self.batch_size_kraken = 100

        except sqlite3.Error as e:
            logger.error(f"[DatabaseManager] Fout bij verbinden met database: {e}")
            raise RuntimeError("Kan geen verbinding maken met de database.")

    def init_db(self):
        """
        (optioneel) Maak alle tabellen aan als ze nog niet bestaan.
        """
        self.create_tables()

    def connect(self):
        """Geeft de bestaande verbinding terug of een error als er geen is."""
        if self.connection:
            return self.connection
        else:
            raise RuntimeError("Geen actieve databaseverbinding.")

    def __del__(self):
        """Sluit de databaseverbinding netjes af als deze instantie wordt verwijderd."""
        if hasattr(self, 'connection') and self.connection:
            try:
                logger.warning("[DatabaseManager] Destructor aangeroepen. Controleer waarom dit gebeurt!")
                if self.connection:
                    logger.info("Destructor: Sluiten van databaseverbinding.")
                    # Flushen van 'oude' candlebuffer
                    self.flush_candles()

                    # Flushen van nieuwe candlebuffers
                    self.flush_candles_bitvavo()
                    self.flush_candles_kraken()

                    self.connection.commit()
                    self.connection.close()
                    logger.info("Databaseverbinding netjes afgesloten.")
            except sqlite3.ProgrammingError:
                logger.warning("[DatabaseManager] Verbinding was al gesloten.")
            except sqlite3.Error as e:
                logger.error(f"[DatabaseManager] Fout bij sluiten van de db-verbinding: {e}")

    def execute_query(self, query, params=(), retries=10, delay=0.2):
        """
        Universele query-executor met retries voor 'locked'-errors.
        Geeft:
         - rows (list of tuples) als het een SELECT of PRAGMA is.
         - None als het een UPDATE/INSERT/DELETE is.
        """
        logger.debug(f"[execute_query] Query: {repr(query)} | Params: {params}")
        with self._db_lock:
            for attempt in range(retries):
                try:
                    self.cursor.execute(query, params)
                    self.connection.commit()

                    lower_query = query.strip().lower()
                    if lower_query.startswith("select") or lower_query.startswith("pragma"):
                        rows = self.cursor.fetchall()
                        logger.debug(f"[execute_query] SELECT => fetched {len(rows)} rows.")
                        return rows
                    else:
                        return None

                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower():
                        logger.warning(f"[execute_query] DB locked. Retry {attempt + 1}/{retries}...")
                        time.sleep(delay)
                    else:
                        logger.error(f"[execute_query] OperationalError: {e}")
                        raise
                except Exception as e:
                    logger.error(f"[execute_query] Onverwachte fout: {e}")
                    raise

            logger.error("[execute_query] Max retries bereikt => Database is vergrendeld.")
            raise sqlite3.OperationalError("Database is vergrendeld na meerdere retries.")

    def get_table_count(self, table_name):
        """Haal het aantal records op uit een tabel met een eenvoudige COUNT(*)."""
        try:
            rows = self.execute_query(f"SELECT COUNT(*) FROM {table_name}")
            if rows is None:
                return 0
            count = rows[0][0]
            logger.info(f"[get_table_count] Tabel '{table_name}' => {count} records.")
            return count
        except Exception as e:
            logger.error(f"[get_table_count] Error: {e}")
            return 0

    # --------------------------------------------------------------------------
    # Creëren/updaten tabellen
    # --------------------------------------------------------------------------
    def create_tables(self):
        """
        Creëer (of update) de bestaande tabellen EN de nieuwe per-exchange-tabellen.
        """
        try:
            # ============ OUD/BESTAAND ============
            self.create_candles_table()
            self.create_ticker_table()
            self.create_orderbook_tables()
            self.create_indicators_table()
            self.alter_indicators_table()
            self.create_trades_table()
            self.create_fills_table()
            self._ensure_exchange_in_all_tables()  # Zorgt dat 'exchange'-kolom in de oude tabellen staat

            # ============ NIEUW: exchange-specifieke tabellen ============
            self.create_candles_bitvavo_table()
            self.create_candles_kraken_table()

            self.create_ticker_bitvavo_table()
            self.create_ticker_kraken_table()

            self.create_orderbook_bitvavo_tables()
            self.create_orderbook_kraken_tables()

            self.create_indicators_bitvavo_table()
            self.create_indicators_kraken_table()

            logger.info("[create_tables] Alle (oude en nieuwe) tabellen klaar of bijgewerkt.")
        except Exception as e:
            logger.error(f"[create_tables] Error: {e}")

    def _ensure_exchange_in_all_tables(self):
        """
        Zorgt dat de 'exchange' kolom in de *oude* tabellen (ticker, orderbook, indicators, trades, fills) bestaat.
        """
        tables = ["ticker", "orderbook_bids", "orderbook_asks", "indicators", "trades", "fills"]
        for t in tables:
            try:
                rows = self.execute_query(f"PRAGMA table_info({t})")
                existing_cols = [r[1] for r in rows]
                if 'exchange' not in existing_cols:
                    self.execute_query(f"ALTER TABLE {t} ADD COLUMN exchange TEXT")
                    logger.info(f"Kolom 'exchange' toegevoegd aan {t}.")
            except Exception as e:
                logger.error(f"Fout bij _ensure_exchange_in_all_tables voor {t}: {e}")

    # =========================
    # OUD: candles-table
    # =========================
    def create_candles_table(self):
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,
                    market TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    exchange TEXT,
                    PRIMARY KEY (market, interval, timestamp)
                );
            """)
            # Controleren of 'datetime_utc' al bestaat
            self.cursor.execute("PRAGMA table_info(candles)")
            columns = [column[1] for column in self.cursor.fetchall()]
            if 'datetime_utc' not in columns:
                self.cursor.execute("ALTER TABLE candles ADD COLUMN datetime_utc TEXT")

            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles (timestamp)")
            self.connection.commit()
            logger.info("[create_candles_table] Candles tabel klaar.")
        except Exception as e:
            logger.error(f"[create_candles_table] Error: {e}")

    # =========================
    # NIEUW: candles_bitvavo en candles_kraken
    # =========================
    def create_candles_bitvavo_table(self):
        """
        Zelfde structuur als 'candles', maar zonder 'exchange'-kolom,
        omdat we dit *exclusief* voor Bitvavo gebruiken.
        """
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles_bitvavo (
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,
                    market TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (market, interval, timestamp)
                );
            """)
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_candles_bitvavo_ts
                ON candles_bitvavo (timestamp)
            """)
            self.connection.commit()
            logger.info("[create_candles_bitvavo_table] Aparte tabel voor Bitvavo-candles klaar.")
        except Exception as e:
            logger.error(f"[create_candles_bitvavo_table] Error: {e}")

    def create_candles_kraken_table(self):
        """
        Zelfde structuur als 'candles', maar zonder 'exchange'-kolom,
        exclusief voor Kraken.
        """
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles_kraken (
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,
                    market TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (market, interval, timestamp)
                );
            """)
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_candles_kraken_ts
                ON candles_kraken (timestamp)
            """)
            self.connection.commit()
            logger.info("[create_candles_kraken_table] Aparte tabel voor Kraken-candles klaar.")
        except Exception as e:
            logger.error(f"[create_candles_kraken_table] Error: {e}")

    def drop_candles_table(self):
        """(Oude) Voor debug/doeleinden - verwijder de hele 'candles'-tabel."""
        try:
            self.cursor.execute("DROP TABLE IF EXISTS candles")
            self.connection.commit()
            logger.info("[drop_candles_table] Tabel 'candles' is verwijderd.")
        except Exception as e:
            logger.error(f"[drop_candles_table] Error: {e}")

    # =========================
    # Ticker (oud + nieuw)
    # =========================
    def create_ticker_table(self):
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS ticker (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,
                    market TEXT NOT NULL,
                    best_bid REAL,
                    best_ask REAL,
                    spread REAL,
                    exchange TEXT
                )
            """)
            self.cursor.execute("PRAGMA table_info(ticker)")
            columns = [column[1] for column in self.cursor.fetchall()]
            if 'datetime_utc' not in columns:
                self.cursor.execute("ALTER TABLE ticker ADD COLUMN datetime_utc TEXT")
            if 'best_bid' not in columns:
                self.cursor.execute("ALTER TABLE ticker ADD COLUMN best_bid REAL")
            if 'best_ask' not in columns:
                self.cursor.execute("ALTER TABLE ticker ADD COLUMN best_ask REAL")
            if 'spread' not in columns:
                self.cursor.execute("ALTER TABLE ticker ADD COLUMN spread REAL")

            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_timestamp ON ticker (timestamp)")
            self.connection.commit()
            logger.info("[create_ticker_table] Ticker table is klaar.")
        except sqlite3.Error as e:
            logger.error(f"[create_ticker_table] Error: {e}")

    def create_ticker_bitvavo_table(self):
        """Zelfde als 'ticker' maar exclusief voor Bitvavo (geen 'exchange'-kolom)."""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS ticker_bitvavo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,
                    market TEXT NOT NULL,
                    best_bid REAL,
                    best_ask REAL,
                    spread REAL
                )
            """)
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_bitvavo_ts ON ticker_bitvavo (timestamp)")
            self.connection.commit()
            logger.info("[create_ticker_bitvavo_table] Klaar.")
        except Exception as e:
            logger.error(f"[create_ticker_bitvavo_table] Error: {e}")

    def create_ticker_kraken_table(self):
        """Zelfde als 'ticker' maar exclusief voor Kraken."""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS ticker_kraken (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,
                    market TEXT NOT NULL,
                    best_bid REAL,
                    best_ask REAL,
                    spread REAL
                )
            """)
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_kraken_ts ON ticker_kraken (timestamp)")
            self.connection.commit()
            logger.info("[create_ticker_kraken_table] Klaar.")
        except Exception as e:
            logger.error(f"[create_ticker_kraken_table] Error: {e}")

    # =========================
    # Orderbook (oud + nieuw)
    # =========================
    def create_orderbook_tables(self):
        try:
            # Bids
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_bids (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,
                    market TEXT NOT NULL,
                    bid_p REAL NOT NULL,
                    bid_q REAL NOT NULL,
                    exchange TEXT
                )
            """)
            self.cursor.execute("PRAGMA table_info(orderbook_bids)")
            columns = [column[1] for column in self.cursor.fetchall()]
            if 'datetime_utc' not in columns:
                self.cursor.execute("ALTER TABLE orderbook_bids ADD COLUMN datetime_utc TEXT")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_orderbook_bids_ts ON orderbook_bids (timestamp)")

            # Asks
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_asks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,
                    market TEXT NOT NULL,
                    ask_p REAL NOT NULL,
                    ask_q REAL NOT NULL,
                    exchange TEXT
                )
            """)
            self.cursor.execute("PRAGMA table_info(orderbook_asks)")
            columns = [column[1] for column in self.cursor.fetchall()]
            if 'datetime_utc' not in columns:
                self.cursor.execute("ALTER TABLE orderbook_asks ADD COLUMN datetime_utc TEXT")

            # Extra index op candles (oud)
            self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_candles_mkt_int_ts
                ON candles (market, interval, timestamp)
            """)
            self.connection.commit()
            logger.info("[create_orderbook_tables] orderbook_bids/asks zijn klaar.")
        except sqlite3.Error as e:
            logger.error(f"[create_orderbook_tables] Error: {e}")

    def create_orderbook_bitvavo_tables(self):
        """Aparte tabellen voor Bitvavo: orderbook_bids_bitvavo, orderbook_asks_bitvavo"""
        try:
            # Bids
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_bids_bitvavo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,
                    market TEXT NOT NULL,
                    bid_p REAL NOT NULL,
                    bid_q REAL NOT NULL
                )
            """)
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ob_bids_bitvavo_ts
                ON orderbook_bids_bitvavo (timestamp)
            """)

            # Asks
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_asks_bitvavo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,
                    market TEXT NOT NULL,
                    ask_p REAL NOT NULL,
                    ask_q REAL NOT NULL
                )
            """)
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ob_asks_bitvavo_ts
                ON orderbook_asks_bitvavo (timestamp)
            """)
            self.connection.commit()
            logger.info("[create_orderbook_bitvavo_tables] Klaar.")
        except sqlite3.Error as e:
            logger.error(f"[create_orderbook_bitvavo_tables] Error: {e}")

    def create_orderbook_kraken_tables(self):
        """Aparte tabellen voor Kraken: orderbook_bids_kraken, orderbook_asks_kraken"""
        try:
            # Bids
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_bids_kraken (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,
                    market TEXT NOT NULL,
                    bid_p REAL NOT NULL,
                    bid_q REAL NOT NULL
                )
            """)
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ob_bids_kraken_ts
                ON orderbook_bids_kraken (timestamp)
            """)

            # Asks
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_asks_kraken (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,
                    market TEXT NOT NULL,
                    ask_p REAL NOT NULL,
                    ask_q REAL NOT NULL
                )
            """)
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ob_asks_kraken_ts
                ON orderbook_asks_kraken (timestamp)
            """)
            self.connection.commit()
            logger.info("[create_orderbook_kraken_tables] Klaar.")
        except sqlite3.Error as e:
            logger.error(f"[create_orderbook_kraken_tables] Error: {e}")

    def drop_orderbook_tables(self):
        """Verwijder de oude orderbook-bids en orderbook-asks tabellen."""
        try:
            self.cursor.execute("DROP TABLE IF EXISTS orderbook_bids")
            self.cursor.execute("DROP TABLE IF EXISTS orderbook_asks")
            self.connection.commit()
            logger.info("[drop_orderbook_tables] Tabel 'orderbook_bids' & 'orderbook_asks' verwijderd.")
        except Exception as e:
            logger.error(f"[drop_orderbook_tables] Error: {e}")

    def reset_orderbook_tables(self):
        """Drop en hercreëer de oude orderbook-bids en asks tabellen."""
        self.drop_orderbook_tables()
        self.create_orderbook_tables()

    # =========================
    # Indicators (oud + nieuw)
    # =========================
    def create_indicators_table(self):
        try:
            create_indicators_table_sql = """
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                datetime_utc TEXT,
                market TEXT,
                interval TEXT,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                bollinger_upper REAL,
                bollinger_lower REAL,
                moving_average REAL,
                ema_9 REAL,
                ema_21 REAL,
                atr14 REAL,
                exchange TEXT
            );
            """
            self.connect().execute(create_indicators_table_sql)
            self.connection.commit()

            self.cursor.execute("PRAGMA table_info(indicators)")
            columns = [column[1] for column in self.cursor.fetchall()]
            if 'datetime_utc' not in columns:
                self.cursor.execute("ALTER TABLE indicators ADD COLUMN datetime_utc TEXT")

            logger.info("[create_indicators_table] Tabel 'indicators' is aangemaakt/bestond al.")
        except Exception as e:
            logger.error(f"[create_indicators_table] Error: {e}")

    def alter_indicators_table(self):
        alter_statements = [
            "ALTER TABLE indicators ADD COLUMN ema_9 REAL",
            "ALTER TABLE indicators ADD COLUMN ema_21 REAL",
            "ALTER TABLE indicators ADD COLUMN atr14 REAL"
        ]
        for sql in alter_statements:
            try:
                self.cursor.execute(sql)
                self.connection.commit()
                logger.info(f"[alter_indicators_table] Uitgevoerd: {sql}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    logger.info(f"[alter_indicators_table] Kolom bestaat al, skip: {sql}")
                else:
                    logger.error(f"[alter_indicators_table] Fout bij {sql}: {e}")
            except Exception as e:
                logger.error(f"[alter_indicators_table] Onverwacht: {sql} => {e}")

    def create_indicators_bitvavo_table(self):
        """Indicators exclusief voor Bitvavo, geen exchange-kolom."""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS indicators_bitvavo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    datetime_utc TEXT,
                    market TEXT,
                    interval TEXT,
                    rsi REAL,
                    macd REAL,
                    macd_signal REAL,
                    bollinger_upper REAL,
                    bollinger_lower REAL,
                    moving_average REAL,
                    ema_9 REAL,
                    ema_21 REAL,
                    atr14 REAL
                );
            """)
            self.connection.commit()
            logger.info("[create_indicators_bitvavo_table] Klaar.")
        except Exception as e:
            logger.error(f"[create_indicators_bitvavo_table] Error: {e}")

    def create_indicators_kraken_table(self):
        """Indicators exclusief voor Kraken."""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS indicators_kraken (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    datetime_utc TEXT,
                    market TEXT,
                    interval TEXT,
                    rsi REAL,
                    macd REAL,
                    macd_signal REAL,
                    bollinger_upper REAL,
                    bollinger_lower REAL,
                    moving_average REAL,
                    ema_9 REAL,
                    ema_21 REAL,
                    atr14 REAL
                );
            """)
            self.connection.commit()
            logger.info("[create_indicators_kraken_table] Klaar.")
        except Exception as e:
            logger.error(f"[create_indicators_kraken_table] Error: {e}")

    # =========================
    # Trades & fills
    # (gecombineerd, ongewijzigd)
    # =========================
    def create_trades_table(self):
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    amount REAL,
                    position_id TEXT,
                    position_type TEXT,
                    status TEXT,
                    pnl_eur REAL,
                    fees REAL,
                    trade_cost REAL,
                    exchange TEXT
                )
            """)
            self.connection.commit()
            logger.info("[create_trades_table] Trades tabel is klaar.")

            self.cursor.execute("PRAGMA table_info(trades)")
            columns = [col[1] for col in self.cursor.fetchall()]
            logger.debug(f"[create_trades_table] Kolommen in 'trades': {columns}")

            # Toevoeging: Kolom strategy_name, zodat we per strategie kunnen filteren.
            maybe_add = {
                'datetime_utc': 'TEXT',
                'position_id': 'TEXT',
                'position_type': 'TEXT',
                'status': 'TEXT',
                'pnl_eur': 'REAL',
                'fees': 'REAL',
                'trade_cost': 'REAL',
                'strategy_name': 'TEXT'  # <- nieuw
            }
            for col_name, col_type in maybe_add.items():
                if col_name not in columns:
                    try:
                        self.cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
                        self.connection.commit()
                        logger.info(f"[create_trades_table] Kolom '{col_name}' toegevoegd.")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" in str(e).lower():
                            logger.info(f"[create_trades_table] Kolom '{col_name}' bestaat al, skip.")
                        else:
                            logger.error(f"[create_trades_table] Fout bij kolom {col_name}: {e}")

            logger.info("[create_trades_table] Tabel 'trades' is nu volledig opgebouwd.")
        except Exception as e:
            logger.error(f"[create_trades_table] Error: {e}")

    def create_fills_table(self):
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT,
                    market TEXT,
                    side TEXT,
                    fill_amount REAL,
                    fill_price REAL,
                    fee_amount REAL,
                    timestamp INTEGER,
                    datetime_utc TEXT,
                    exchange TEXT
                )
            """)
            self.connection.commit()
            logger.info("[create_fills_table] Fills tabel is aangemaakt/bestond al.")
        except Exception as e:
            logger.error(f"[create_fills_table] Error: {e}")

    def save_fill(self, fill_data: dict):
        try:
            q = """
                INSERT INTO fills
                (order_id, market, side, fill_amount, fill_price, fee_amount,
                 timestamp, datetime_utc, exchange)
                VALUES (
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  datetime(?/1000, 'unixepoch'),
                  ?
                )
            """
            order_id = fill_data.get("order_id", "")
            market = fill_data.get("market", "")
            side = fill_data.get("side", "")
            f_amt = fill_data.get("fill_amount", 0.0)
            f_price = fill_data.get("fill_price", 0.0)
            fee_amt = fill_data.get("fee_amount", 0.0)
            ts = fill_data.get("timestamp", get_current_utc_timestamp_ms())
            # Oorspronkelijke default "Bitvavo" => vervangen door "Kraken"
            # exch = fill_data.get("exchange", "Bitvavo")
            exch = fill_data.get("exchange", "Kraken")

            params = (order_id, market, side, f_amt, f_price, fee_amt, ts, ts, exch)
            self.execute_query(q, params)
            logger.info(f"[save_fill] Fill opgeslagen: {fill_data}")
        except Exception as e:
            logger.error(f"[save_fill] Fout: {e}")

    # --------------------------------------------------------------------------
    # Candle buffer / flush (OUD)
    # --------------------------------------------------------------------------
    def _validate_and_buffer_candles(self, data):
        """
        OUD: data => list van (timestamp, market, interval, open, high, low, close, volume, exchange?)
        """
        try:
            valid_data = []
            for record in data:
                if len(record) == 8:
                    # i.p.v. default "Bitvavo" => "Kraken"
                    record = record + ("Kraken",)
                if len(record) != 9:
                    logger.warning(f"[validate_candles] Ongeldig record: {record}")
                    continue
                timestamp, market, interval, open_, high, low, close, volume, exch = record
                if (isinstance(timestamp, int)
                        and isinstance(market, str)
                        and isinstance(interval, str)
                        and isinstance(open_, (int, float))
                        and isinstance(high, (int, float))
                        and isinstance(low, (int, float))
                        and isinstance(close, (int, float))
                        and isinstance(volume, (int, float))
                        and isinstance(exch, str)):
                    if timestamp <= 0:
                        timestamp = get_current_utc_timestamp_ms()
                    valid_data.append((timestamp, market, interval, open_, high, low, close, volume, exch))
                else:
                    logger.warning(f"[validate_candles] Ongeldig record skip: {record}")

            if not valid_data:
                logger.warning("[validate_candles] Geen geldige candle data om te bufferen.")
                return 0

            self.candle_buffer.extend(valid_data)
            if len(self.candle_buffer) >= self.batch_size:
                self._flush_candle_buffer()

            return len(valid_data)

        except sqlite3.OperationalError as e:
            logger.error(f"[validate_candles] Databasefout: {e}")
            return 0
        except Exception as e:
            logger.error(f"[validate_candles] Onverwachte fout: {e}")
            return 0

    def _flush_candle_buffer(self):
        """In 1 batch wegschrijven in de OUD 'candles'-tabel."""
        if not self.candle_buffer:
            return
        try:
            final_list = []
            for row in self.candle_buffer:
                ts, market, interval, o, h, l, c, vol, exch = row
                dt_utc = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                final_list.append((ts, dt_utc, market, interval, o, h, l, c, vol, exch))

            with self.connection:
                self.connection.executemany("""
                    INSERT OR REPLACE INTO candles
                    (timestamp, datetime_utc, market, interval, open, high, low, close, volume, exchange)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, final_list)

            logger.info(f"[flush_candle_buffer] {len(self.candle_buffer)} candle records weggeschreven (oud).")
            self.candle_buffer.clear()
        except Exception as e:
            logger.error(f"[flush_candle_buffer] Fout: {e}")

    def flush_candles(self):
        self._flush_candle_buffer()

    def start_flush_timer(self, interval_seconds=5):
        """
        Start een achtergrondthread die elke 'interval_seconds' de buffers flushes
        (zowel de OUD 'candles' buffer als de nieuwe bitvavo/kraken buffers).
        """
        def flush_loop():
            while True:
                time.sleep(interval_seconds)
                try:
                    self.flush_candles()           # oude buffer
                    self.flush_candles_bitvavo()   # nieuwe
                    self.flush_candles_kraken()    # nieuwe
                    logger.debug("[flush_timer] Alle candle-buffers geflusht.")
                except Exception as e:
                    logger.error(f"[flush_timer] Fout bij flushen: {e}")

        timer_thread = threading.Thread(target=flush_loop, daemon=True)
        timer_thread.start()

    def save_candles(self, data):
        """
        OUD: public method om *enkelvoudige* candles-tabel te saven.
        """
        logger.info(f"[save_candles] Aangeroepen met {len(data)} records (oud).")
        count = self._validate_and_buffer_candles(data)
        if count > 0:
            logger.info(f"[save_candles] {count} candle records gebufferd. (Buffer size={len(self.candle_buffer)})")
        else:
            logger.info("[save_candles] Geen geldige candle-records gebufferd.")

    # --------------------------------------------------------------------------
    # NIEUW: buffers en saves voor candles_bitvavo en candles_kraken
    # --------------------------------------------------------------------------
    def _validate_and_buffer_candles_bitvavo(self, data):
        """
        data => list van tuples: (timestamp, market, interval, open, high, low, close, volume)
        GEEN exchange nodig, want dit is specifiek voor Bitvavo.
        """
        valid_data = []
        for record in data:
            if len(record) != 8:
                logger.warning(f"[validate_candles_bitvavo] Ongeldig record: {record}")
                continue
            timestamp, market, interval, open_, high, low, close, volume = record
            if (isinstance(timestamp, int)
                and isinstance(market, str)
                and isinstance(interval, str)
                and isinstance(open_, (int, float))
                and isinstance(high, (int, float))
                and isinstance(low, (int, float))
                and isinstance(close, (int, float))
                and isinstance(volume, (int, float))):
                if timestamp <= 0:
                    timestamp = get_current_utc_timestamp_ms()
                valid_data.append((timestamp, market, interval, open_, high, low, close, volume))
            else:
                logger.warning(f"[validate_candles_bitvavo] Ongeldig record skip: {record}")

        if not valid_data:
            return 0

        self.candle_buffer_bitvavo.extend(valid_data)
        if len(self.candle_buffer_bitvavo) >= self.batch_size_bitvavo:
            self._flush_candle_buffer_bitvavo()
        return len(valid_data)

    def _flush_candle_buffer_bitvavo(self):
        """In 1 batch wegschrijven naar 'candles_bitvavo'."""
        if not self.candle_buffer_bitvavo:
            return
        try:
            final_list = []
            for row in self.candle_buffer_bitvavo:
                ts, market, interval, o, h, l, c, vol = row
                dt_utc = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                final_list.append((ts, dt_utc, market, interval, o, h, l, c, vol))

            with self.connection:
                self.connection.executemany("""
                    INSERT OR REPLACE INTO candles_bitvavo
                    (timestamp, datetime_utc, market, interval, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, final_list)

            logger.info(f"[flush_candle_buffer_bitvavo] {len(self.candle_buffer_bitvavo)} records -> candles_bitvavo.")
            self.candle_buffer_bitvavo.clear()
        except Exception as e:
            logger.error(f"[flush_candle_buffer_bitvavo] Fout: {e}")

    def flush_candles_bitvavo(self):
        self._flush_candle_buffer_bitvavo()

    def save_candles_bitvavo(self, data):
        """
        Public method om *Bitvavo-candles* in bulk te saven:
         - data: list van (ts, market, interval, open, high, low, close, volume)
        """
        logger.info(f"[save_candles_bitvavo] {len(data)} records.")
        count = self._validate_and_buffer_candles_bitvavo(data)
        if count > 0:
            logger.info(f"[save_candles_bitvavo] {count} records gebufferd. (Buffer={len(self.candle_buffer_bitvavo)})")

    def _validate_and_buffer_candles_kraken(self, data):
        """
        Zelfde idee als bitvavo, maar voor 'candles_kraken'.
        """
        valid_data = []
        for record in data:
            if len(record) != 8:
                logger.warning(f"[validate_candles_kraken] Ongeldig record: {record}")
                continue
            timestamp, market, interval, open_, high, low, close, volume = record
            if (isinstance(timestamp, int)
                and isinstance(market, str)
                and isinstance(interval, str)
                and isinstance(open_, (int, float))
                and isinstance(high, (int, float))
                and isinstance(low, (int, float))
                and isinstance(close, (int, float))
                and isinstance(volume, (int, float))):
                if timestamp <= 0:
                    timestamp = get_current_utc_timestamp_ms()
                valid_data.append((timestamp, market, interval, open_, high, low, close, volume))
            else:
                logger.warning(f"[validate_candles_kraken] Ongeldig record skip: {record}")

        if not valid_data:
            return 0

        self.candle_buffer_kraken.extend(valid_data)
        if len(self.candle_buffer_kraken) >= self.batch_size_kraken:
            self._flush_candle_buffer_kraken()
        return len(valid_data)

    def _flush_candle_buffer_kraken(self):
        """In 1 batch wegschrijven naar 'candles_kraken'."""
        if not self.candle_buffer_kraken:
            return
        try:
            final_list = []
            for row in self.candle_buffer_kraken:
                ts, market, interval, o, h, l, c, vol = row
                dt_utc = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                final_list.append((ts, dt_utc, market, interval, o, h, l, c, vol))

            with self.connection:
                self.connection.executemany("""
                    INSERT OR REPLACE INTO candles_kraken
                    (timestamp, datetime_utc, market, interval, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, final_list)

            logger.info(f"[flush_candle_buffer_kraken] {len(self.candle_buffer_kraken)} -> candles_kraken.")
            self.candle_buffer_kraken.clear()
        except Exception as e:
            logger.error(f"[flush_candle_buffer_kraken] Fout: {e}")

    def flush_candles_kraken(self):
        self._flush_candle_buffer_kraken()

    def save_candles_kraken(self, data):
        """
        Public method om *Kraken-candles* in bulk te saven.
        """
        logger.info(f"[save_candles_kraken] {len(data)} records.")
        count = self._validate_and_buffer_candles_kraken(data)
        if count > 0:
            logger.info(f"[save_candles_kraken] {count} gebufferd. (Buffer={len(self.candle_buffer_kraken)})")

    # --------------------------------------------------------------------------
    # save_ticker (oud) + aparte methods voor bitvavo/kraken
    # --------------------------------------------------------------------------
    def save_ticker(self, data):
        """
        OUD: sla op in de 'ticker' table met exchange-kolom.
        """
        try:
            timestamp = get_current_utc_timestamp_ms()
            market = data['market']
            best_bid = data.get('bestBid', 0.0)
            best_ask = data.get('bestAsk', 0.0)
            spread = best_ask - best_bid
            # Oorspronkelijk default "Bitvavo"
            # exchange = data.get('exchange', 'Bitvavo')
            exchange = data.get('exchange', 'Kraken')  # <--- AANPASSING

            q = """
                INSERT INTO ticker
                (timestamp, datetime_utc, market, best_bid, best_ask, spread, exchange)
                VALUES (
                  ?,
                  datetime(?/1000, 'unixepoch'),
                  ?,
                  ?,
                  ?,
                  ?,
                  ?
                )
            """
            p = (timestamp, timestamp, market, best_bid, best_ask, spread, exchange)
            self.execute_query(q, p)
            logger.info(f"[save_ticker] Ticker data (oud) opgeslagen: {data}")
        except Exception as e:
            logger.error(f"[save_ticker] Fout: {e}")

    def save_ticker_bitvavo(self, data):
        """
        Nieuw: sla op in 'ticker_bitvavo' table (geen exchange-kolom).
        data = {
          'market': 'BTC-EUR',
          'bestBid': 12345,
          'bestAsk': 12346,
          ...
        }
        """
        try:
            timestamp = get_current_utc_timestamp_ms()
            market = data['market']
            best_bid = data.get('bestBid', 0.0)
            best_ask = data.get('bestAsk', 0.0)
            spread = best_ask - best_bid
            q = """
                INSERT INTO ticker_bitvavo
                (timestamp, datetime_utc, market, best_bid, best_ask, spread)
                VALUES (
                  ?,
                  datetime(?/1000, 'unixepoch'),
                  ?,
                  ?,
                  ?,
                  ?
                )
            """
            params = (timestamp, timestamp, market, best_bid, best_ask, spread)
            self.execute_query(q, params)
            logger.info(f"[save_ticker_bitvavo] Opgeslagen: {data}")
        except Exception as e:
            logger.error(f"[save_ticker_bitvavo] Fout: {e}")

    def save_ticker_kraken(self, data):
        """
        Nieuw: sla op in 'ticker_kraken' (geen exchange-kolom).
        """
        try:
            timestamp = get_current_utc_timestamp_ms()
            market = data['market']
            best_bid = data.get('bestBid', 0.0)
            best_ask = data.get('bestAsk', 0.0)
            spread = best_ask - best_bid
            q = """
                INSERT INTO ticker_kraken
                (timestamp, datetime_utc, market, best_bid, best_ask, spread)
                VALUES (
                  ?,
                  datetime(?/1000, 'unixepoch'),
                  ?,
                  ?,
                  ?,
                  ?
                )
            """
            params = (timestamp, timestamp, market, best_bid, best_ask, spread)
            self.execute_query(q, params)
            logger.info(f"[save_ticker_kraken] Opgeslagen: {data}")
        except Exception as e:
            logger.error(f"[save_ticker_kraken] Fout: {e}")

    # --------------------------------------------------------------------------
    # save_orderbook (oud) + aparte methodes
    # --------------------------------------------------------------------------
    def save_orderbook(self, data):
        """
        OUD: sla op in de 'orderbook_bids' en 'orderbook_asks' tabellen met exchange-kolom.
        data = {
           'market': 'BTC-EUR',
           'bids': [[price,qty], ...],
           'asks': [[price,qty], ...],
           'exchange': ...
        }
        """
        try:
            timestamp = get_current_utc_timestamp_ms()
            market = data['market']
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            # exchange = data.get('exchange', 'Bitvavo')
            exchange = data.get('exchange', 'Kraken')  # <--- AANPASSING

            for bid in bids:
                q = """
                    INSERT INTO orderbook_bids
                    (timestamp, datetime_utc, market, bid_p, bid_q, exchange)
                    VALUES (
                      ?,
                      datetime(?/1000,'unixepoch'),
                      ?,
                      ?,
                      ?,
                      ?
                    )
                """
                p = (timestamp, timestamp, market, float(bid[0]), float(bid[1]), exchange)
                self.execute_query(q, p)

            for ask in asks:
                q = """
                    INSERT INTO orderbook_asks
                    (timestamp, datetime_utc, market, ask_p, ask_q, exchange)
                    VALUES (
                      ?,
                      datetime(?/1000,'unixepoch'),
                      ?,
                      ?,
                      ?,
                      ?
                    )
                """
                p = (timestamp, timestamp, market, float(ask[0]), float(ask[1]), exchange)
                self.execute_query(q, p)

            logger.info("[save_orderbook] (oud) data opgeslagen.")
        except Exception as e:
            logger.error(f"[save_orderbook] Fout: {e}")

    def save_orderbook_bitvavo(self, data):
        """
        Sla bids/asks op in orderbook_bids_bitvavo / orderbook_asks_bitvavo (geen exchange-kolom).
        data = {
          'market': 'BTC-EUR',
          'bids': [[p,q], ...],
          'asks': [[p,q], ...]
        }
        """
        try:
            timestamp = get_current_utc_timestamp_ms()
            market = data['market']
            bids = data.get('bids', [])
            asks = data.get('asks', [])

            for bid in bids:
                q = """
                    INSERT INTO orderbook_bids_bitvavo
                    (timestamp, datetime_utc, market, bid_p, bid_q)
                    VALUES (
                      ?,
                      datetime(?/1000,'unixepoch'),
                      ?,
                      ?,
                      ?
                    )
                """
                p = (timestamp, timestamp, market, float(bid[0]), float(bid[1]))
                self.execute_query(q, p)

            for ask in asks:
                q = """
                    INSERT INTO orderbook_asks_bitvavo
                    (timestamp, datetime_utc, market, ask_p, ask_q)
                    VALUES (
                      ?,
                      datetime(?/1000,'unixepoch'),
                      ?,
                      ?,
                      ?
                    )
                """
                p = (timestamp, timestamp, market, float(ask[0]), float(ask[1]))
                self.execute_query(q, p)

            logger.info("[save_orderbook_bitvavo] data opgeslagen.")
        except Exception as e:
            logger.error(f"[save_orderbook_bitvavo] Fout: {e}")

    def save_orderbook_kraken(self, data):
        """
        Sla bids/asks op in orderbook_bids_kraken / orderbook_asks_kraken (geen exchange).
        """
        try:
            timestamp = get_current_utc_timestamp_ms()
            market = data['market']
            bids = data.get('bids', [])
            asks = data.get('asks', [])

            for bid in bids:
                q = """
                    INSERT INTO orderbook_bids_kraken
                    (timestamp, datetime_utc, market, bid_p, bid_q)
                    VALUES (
                      ?,
                      datetime(?/1000,'unixepoch'),
                      ?,
                      ?,
                      ?
                    )
                """
                p = (timestamp, timestamp, market, float(bid[0]), float(bid[1]))
                self.execute_query(q, p)

            for ask in asks:
                q = """
                    INSERT INTO orderbook_asks_kraken
                    (timestamp, datetime_utc, market, ask_p, ask_q)
                    VALUES (
                      ?,
                      datetime(?/1000,'unixepoch'),
                      ?,
                      ?,
                      ?
                    )
                """
                p = (timestamp, timestamp, market, float(ask[0]), float(ask[1]))
                self.execute_query(q, p)

            logger.info("[save_orderbook_kraken] data opgeslagen.")
        except Exception as e:
            logger.error(f"[save_orderbook_kraken] Fout: {e}")

    # --------------------------------------------------------------------------
    # save_indicators (oud) + nieuwe varianten
    # --------------------------------------------------------------------------
    def save_indicators(self, df_with_indicators: pd.DataFrame):
        """
        OUD: sla op in de 'indicators' tabel met exchange-kolom.
        """
        insert_q = """
            INSERT OR REPLACE INTO indicators
            (timestamp, datetime_utc, market, interval,
             rsi, macd, macd_signal,
             bollinger_upper, bollinger_lower, moving_average,
             ema_9, ema_21, atr14, exchange)
            VALUES (
              ?,
              datetime(?/1000, 'unixepoch'),
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?
            )
        """
        rows = []
        for _, row in df_with_indicators.iterrows():
            ts_val = row.get("timestamp", 0)
            if isinstance(ts_val, pd.Timestamp):
                ts_val = int(ts_val.timestamp() * 1000)

            market_val = row.get("market", "UNKNOWN")
            interval_val = row.get("interval", "1m")
            rsi_val = row.get("rsi", None)
            macd_val = row.get("macd", None)
            macd_sig = row.get("macd_signal", None)
            boll_up = row.get("bollinger_upper", None)
            boll_low = row.get("bollinger_lower", None)
            mov_avg = row.get("moving_average", None)
            ema_9 = row.get("ema_9", None)
            ema_21 = row.get("ema_21", None)
            atr14 = row.get("atr14", None)
            # Oorspronkelijk default "Bitvavo", nu "Kraken"
            # exchange_val = row.get("exchange", "Bitvavo")
            exchange_val = row.get("exchange", "Kraken")

            rows.append((
                ts_val,
                ts_val,
                market_val,
                interval_val,
                rsi_val,
                macd_val,
                macd_sig,
                boll_up,
                boll_low,
                mov_avg,
                ema_9,
                ema_21,
                atr14,
                exchange_val
            ))
        if not rows:
            logger.info("[save_indicators] Geen indicator-rows om op te slaan (oud).")
            return

        try:
            with self.connect() as conn:
                conn.executemany(insert_q, rows)
            logger.info(f"[save_indicators] {len(rows)} rows inserted/updated in 'indicators' (oud).")
        except Exception as e:
            logger.error(f"[save_indicators] Fout: {e}")

    def save_indicators_bitvavo(self, df_with_indicators: pd.DataFrame):
        """
        Nieuw: sla op in 'indicators_bitvavo' (geen exchange-kolom).
        Verwacht columns: timestamp, market, interval, rsi, macd, macd_signal, ...
        """
        insert_q = """
            INSERT OR REPLACE INTO indicators_bitvavo
            (timestamp, datetime_utc, market, interval,
             rsi, macd, macd_signal,
             bollinger_upper, bollinger_lower, moving_average,
             ema_9, ema_21, atr14)
            VALUES (
              ?,
              datetime(?/1000, 'unixepoch'),
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?
            )
        """
        self._save_indicators_generic(df_with_indicators, insert_q, "[save_indicators_bitvavo]")

    def save_indicators_kraken(self, df_with_indicators: pd.DataFrame):
        """
        Nieuw: sla op in 'indicators_kraken'.
        """
        insert_q = """
            INSERT OR REPLACE INTO indicators_kraken
            (timestamp, datetime_utc, market, interval,
             rsi, macd, macd_signal,
             bollinger_upper, bollinger_lower, moving_average,
             ema_9, ema_21, atr14)
            VALUES (
              ?,
              datetime(?/1000, 'unixepoch'),
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              ?
            )
        """
        self._save_indicators_generic(df_with_indicators, insert_q, "[save_indicators_kraken]")

    def _save_indicators_generic(self, df: pd.DataFrame, insert_query: str, log_prefix: str):
        """
        Hulpmethode om duplicatie te verminderen in save_indicators_xxx.
        """
        rows = []
        for _, row in df.iterrows():
            ts_val = row.get("timestamp", 0)
            if isinstance(ts_val, pd.Timestamp):
                ts_val = int(ts_val.timestamp() * 1000)

            market_val = row.get("market", "UNKNOWN")
            interval_val = row.get("interval", "1m")
            rsi_val = row.get("rsi", None)
            macd_val = row.get("macd", None)
            macd_sig = row.get("macd_signal", None)
            boll_up = row.get("bollinger_upper", None)
            boll_low = row.get("bollinger_lower", None)
            mov_avg = row.get("moving_average", None)
            ema_9 = row.get("ema_9", None)
            ema_21 = row.get("ema_21", None)
            atr14 = row.get("atr14", None)

            rows.append((
                ts_val,               # timestamp
                ts_val,               # datetime(?/1000)
                market_val,
                interval_val,
                rsi_val,
                macd_val,
                macd_sig,
                boll_up,
                boll_low,
                mov_avg,
                ema_9,
                ema_21,
                atr14
            ))
        if not rows:
            logger.info(f"{log_prefix} Geen rows om op te slaan.")
            return

        try:
            with self.connect() as conn:
                conn.executemany(insert_query, rows)
            logger.info(f"{log_prefix} {len(rows)} rows inserted/updated.")
        except Exception as e:
            logger.error(f"{log_prefix} Fout: {e}")

    # --------------------------------------------------------------------------
    # save_trade en update_trade
    # --------------------------------------------------------------------------
    def save_trade(self, trade_data: dict):
        """
        OUD: Alles in 'trades' met exchange-kolom.
        AANPASSING: default exchange="Kraken", en optioneel 'strategy_name'.
        """
        try:
            # We breiden de kolommen uit met strategy_name, dus extra veld in INSERT.
            query = """
                INSERT INTO trades
                (timestamp, datetime_utc, symbol, side, price, amount,
                 position_id, position_type, status, pnl_eur, fees, trade_cost, exchange, strategy_name)
                VALUES (
                  ?,
                  datetime(?/1000, 'unixepoch'),
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?
                )
            """
            tstamp = trade_data['timestamp']
            # exchange default "Kraken"
            # exchange_val = trade_data.get('exchange', 'Bitvavo')
            exchange_val = trade_data.get('exchange', 'Kraken')
            # strategy_name (optioneel)
            strategy_val = trade_data.get('strategy_name', None)

            params = (
                tstamp,
                tstamp,
                trade_data['symbol'],
                trade_data['side'],
                trade_data['price'],
                trade_data.get('amount', 0.0),
                trade_data.get('position_id', None),
                trade_data.get('position_type', None),
                trade_data.get('status', None),
                trade_data.get('pnl_eur', 0.0),
                trade_data.get('fees', 0.0),
                trade_data.get('trade_cost', 0.0),
                exchange_val,
                strategy_val
            )
            self.execute_query(query, params)
            logger.info(f"[save_trade] Trade data opgeslagen: {trade_data}")
        except Exception as e:
            logger.error(f"[save_trade] Fout: {e}")

    def update_trade(self, trade_id: int, updates: dict):
        """
        updates => bijv. {"status": "closed", "pnl_eur": 12.34}
        """
        try:
            set_clauses = []
            params = []
            for col, val in updates.items():
                set_clauses.append(f"{col} = ?")
                params.append(val)
            set_clause_str = ", ".join(set_clauses)

            q = f"UPDATE trades SET {set_clause_str} WHERE id = ?"
            params.append(trade_id)

            self.execute_query(q, tuple(params))
            logger.info(f"[update_trade] Trade {trade_id} geüpdatet => {updates}")
        except Exception as e:
            logger.error(f"[update_trade] Fout: {e}")

    #
    # Extra: de fills-tabel is NIET hetzelfde als trades. Je kunt in je client-code
    # `_handle_fill_update(...)` (Bitvavo) of `_handle_own_trade(...)` (Kraken) aanroepen
    # en direct `save_fill(...)` gebruiken.
    #

    # --------------------------------------------------------------------------
    # Prune, fetch_data, get_xxx, etc.
    # (Oude universele methodes blijven)
    # --------------------------------------------------------------------------
    def prune_old_candles(self, days=30, interval=None):
        """
        Verwijder candles ouder dan X dagen (opt. alleen van een bepaald interval)
        uit de OUD 'candles' table.
        """
        cutoff_ms = int(time.time() * 1000) - days * 24 * 60 * 60 * 1000
        if interval is not None:
            query = """
                DELETE FROM candles
                WHERE timestamp < ?
                  AND interval = ?
            """
            params = (cutoff_ms, interval)
            extra_info = f"interval={interval}"
        else:
            query = "DELETE FROM candles WHERE timestamp < ?"
            params = (cutoff_ms,)
            extra_info = "alle intervals"

        try:
            self.execute_query(query, params)
            logger.info(f"[prune_old_candles] Verwijderd: candles ouder dan {days} dagen ({extra_info}).")
        except Exception as e:
            logger.error(f"[prune_old_candles] Fout: {e}")

    def fetch_data(self, table_name, limit=100, market=None, interval=None, exchange=None):
        """
        Universele fetch-functie voor zowel de oude tabellen (candles/ticker/orderbook/...),
        als de nieuwe bitvavo/kraken-tabellen (candles_bitvavo, ticker_kraken, etc.).
        Je kunt dus bv. self.fetch_data("candles_bitvavo", market="BTC-EUR", interval="1m").
        """
        try:
            params = []
            conditions = []

            # ============= OUD: candles (enkel) =============
            if table_name == "candles":
                base_query = """
                    SELECT DISTINCT
                        timestamp,
                        datetime_utc,
                        market,
                        interval,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        exchange
                    FROM candles
                """
                # filter
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if interval:
                    conditions.append("interval = ?")
                    params.append(interval)
                if exchange:
                    conditions.append("exchange = ?")
                    params.append(exchange)

            # ============= NIEUW: candles_bitvavo =============
            elif table_name == "candles_bitvavo":
                base_query = """
                    SELECT
                        timestamp,
                        datetime_utc,
                        market,
                        interval,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM candles_bitvavo
                """
                # geen exchange-kolom
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if interval:
                    conditions.append("interval = ?")
                    params.append(interval)

            # ============= NIEUW: candles_kraken =============
            elif table_name == "candles_kraken":
                base_query = """
                    SELECT
                        timestamp,
                        datetime_utc,
                        market,
                        interval,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM candles_kraken
                """
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if interval:
                    conditions.append("interval = ?")
                    params.append(interval)

            # ============= OUD: ticker (enkel) =============
            elif table_name == "ticker":
                base_query = """
                    SELECT
                        id,
                        timestamp,
                        datetime_utc,
                        market,
                        best_bid,
                        best_ask,
                        spread,
                        exchange
                    FROM ticker
                """
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if exchange:
                    conditions.append("exchange = ?")
                    params.append(exchange)

            # ============= NIEUW: ticker_bitvavo =============
            elif table_name == "ticker_bitvavo":
                base_query = """
                    SELECT
                        id,
                        timestamp,
                        datetime_utc,
                        market,
                        best_bid,
                        best_ask,
                        spread
                    FROM ticker_bitvavo
                """
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                # geen exchange-filter hier

            # ============= NIEUW: ticker_kraken =============
            elif table_name == "ticker_kraken":
                base_query = """
                    SELECT
                        id,
                        timestamp,
                        datetime_utc,
                        market,
                        best_bid,
                        best_ask,
                        spread
                    FROM ticker_kraken
                """
                if market:
                    conditions.append("market = ?")
                    params.append(market)

            # ============= OUD: orderbook_bids / orderbook_asks (enkel) =============
            elif table_name in ["orderbook_bids", "orderbook_asks"]:
                col_p = "bid_p, bid_q" if table_name == "orderbook_bids" else "ask_p, ask_q"
                base_query = f"""
                    SELECT
                        id,
                        timestamp,
                        datetime_utc,
                        market,
                        {col_p},
                        exchange
                    FROM {table_name}
                """
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if exchange:
                    conditions.append("exchange = ?")
                    params.append(exchange)

            # ============= NIEUW: orderbook_bids_bitvavo / orderbook_asks_bitvavo =============
            elif table_name in ["orderbook_bids_bitvavo", "orderbook_asks_bitvavo"]:
                if table_name == "orderbook_bids_bitvavo":
                    base_query = """
                        SELECT
                            id,
                            timestamp,
                            datetime_utc,
                            market,
                            bid_p,
                            bid_q
                        FROM orderbook_bids_bitvavo
                    """
                else:
                    base_query = """
                        SELECT
                            id,
                            timestamp,
                            datetime_utc,
                            market,
                            ask_p,
                            ask_q
                        FROM orderbook_asks_bitvavo
                    """
                # geen exchange-kolom
                if market:
                    conditions.append("market = ?")
                    params.append(market)

            # ============= NIEUW: orderbook_bids_kraken / orderbook_asks_kraken =============
            elif table_name in ["orderbook_bids_kraken", "orderbook_asks_kraken"]:
                if table_name == "orderbook_bids_kraken":
                    base_query = """
                        SELECT
                            id,
                            timestamp,
                            datetime_utc,
                            market,
                            bid_p,
                            bid_q
                        FROM orderbook_bids_kraken
                    """
                else:
                    base_query = """
                        SELECT
                            id,
                            timestamp,
                            datetime_utc,
                            market,
                            ask_p,
                            ask_q
                        FROM orderbook_asks_kraken
                    """
                if market:
                    conditions.append("market = ?")
                    params.append(market)

            # ============= OUD: indicators (enkel) =============
            elif table_name == "indicators":
                base_query = """
                    SELECT
                        id,
                        timestamp,
                        datetime_utc,
                        market,
                        interval,
                        rsi,
                        macd,
                        macd_signal,
                        bollinger_upper,
                        bollinger_lower,
                        moving_average,
                        ema_9,
                        ema_21,
                        atr14,
                        exchange
                    FROM indicators
                """
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if interval:
                    conditions.append("interval = ?")
                    params.append(interval)
                if exchange:
                    conditions.append("exchange = ?")
                    params.append(exchange)

            # ============= NIEUW: indicators_bitvavo =============
            elif table_name == "indicators_bitvavo":
                base_query = """
                    SELECT
                        id,
                        timestamp,
                        datetime_utc,
                        market,
                        interval,
                        rsi,
                        macd,
                        macd_signal,
                        bollinger_upper,
                        bollinger_lower,
                        moving_average,
                        ema_9,
                        ema_21,
                        atr14
                    FROM indicators_bitvavo
                """
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if interval:
                    conditions.append("interval = ?")
                    params.append(interval)

            # ============= NIEUW: indicators_kraken =============
            elif table_name == "indicators_kraken":
                base_query = """
                    SELECT
                        id,
                        timestamp,
                        datetime_utc,
                        market,
                        interval,
                        rsi,
                        macd,
                        macd_signal,
                        bollinger_upper,
                        bollinger_lower,
                        moving_average,
                        ema_9,
                        ema_21,
                        atr14
                    FROM indicators_kraken
                """
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if interval:
                    conditions.append("interval = ?")
                    params.append(interval)

            # ============= OUD: trades (enkel) =============
            elif table_name == "trades":
                base_query = """
                    SELECT
                        id,
                        timestamp,
                        datetime_utc,
                        symbol,
                        side,
                        price,
                        amount,
                        position_id,
                        position_type,
                        status,
                        pnl_eur,
                        fees,
                        trade_cost,
                        exchange,
                        strategy_name  -- Laat strategy_name ook zien, als aanwezig.
                    FROM trades
                """
                if market:
                    conditions.append("symbol = ?")
                    params.append(market)
                if exchange:
                    conditions.append("exchange = ?")
                    params.append(exchange)

            # ============= OUD: fills (enkel) =============
            elif table_name == "fills":
                base_query = """
                    SELECT
                        id,
                        order_id,
                        market,
                        side,
                        fill_amount,
                        fill_price,
                        fee_amount,
                        timestamp,
                        datetime_utc,
                        exchange
                    FROM fills
                """
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if exchange:
                    conditions.append("exchange = ?")
                    params.append(exchange)

            else:
                # Fallback => "SELECT * FROM <table_name>"
                base_query = f"SELECT * FROM {table_name}"

            # ===== conditions (WHERE ...) =====
            if conditions:
                base_query += " WHERE " + " AND ".join(conditions)
            base_query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            logger.info(f"[fetch_data] => {base_query}, params={params}")
            df = pd.read_sql_query(base_query, self.connection, params=params)
            logger.info(f"[fetch_data] {len(df)} records opgehaald uit {table_name}.")
            return df

        except Exception as e:
            logger.error(f"[fetch_data] Error from {table_name}: {e}")
            return pd.DataFrame()

    def get_candlesticks(self, market, interval="1m", limit=100, exchange=None):
        """
        Haal candles op uit de OUD 'candles'-table (optioneel filter op exchange).
        """
        try:
            df = self.fetch_data("candles", limit=limit, market=market, interval=interval, exchange=exchange)
            logger.debug(f"[get_candlesticks] {df.shape[0]} rows (oud).")
            return df
        except Exception as e:
            logger.error(f"[get_candlesticks] Fout: {e}")
            return pd.DataFrame()

    def get_ticker(self, market: str, exchange=None):
        """
        Haal 1 row op uit de OUD 'ticker' table.
        """
        df = self.fetch_data("ticker", market=market, limit=1, exchange=exchange)
        if not df.empty:
            return df.iloc[0].to_dict()
        return {}

    def fetch_open_trades(self):
        """
        Zoekt trades met status='open' in de OUD 'trades' table.
        Return is list van dicts.
        """
        query = """
            SELECT
                symbol, side, amount, price,
                position_id, position_type, status, exchange
            FROM trades
            WHERE status = 'open'
        """
        rows = self.execute_query(query)
        if not rows:
            return []
        columns = ["symbol", "side", "amount", "price", "position_id", "position_type", "status", "exchange"]
        results = [dict(zip(columns, row)) for row in rows]
        return results

    def get_orderbook_snapshot(self, market: str, exchange=None):
        """
        Haalt tot 50 bids en 50 asks uit de OUD 'orderbook_bids'/'orderbook_asks'.
        Return => {"bids":[[p,q],...], "asks":[[p,q],...]}
        """
        df_bids = self.fetch_data("orderbook_bids", market=market, limit=50, exchange=exchange)
        df_asks = self.fetch_data("orderbook_asks", market=market, limit=50, exchange=exchange)

        bids_list = df_bids[['bid_p', 'bid_q']].values.tolist() if not df_bids.empty else []
        asks_list = df_asks[['ask_p', 'ask_q']].values.tolist() if not df_asks.empty else []
        return {"bids": bids_list, "asks": asks_list}

    def close_connection(self):
        """Manuele afsluitmethode. Sluit DB-verbinding en flush buffers."""
        if self.connection:
            self.flush_candles()
            self.flush_candles_bitvavo()
            self.flush_candles_kraken()
            self.connection.close()
            logger.info("[close_connection] DB-verbinding is gesloten.")

    def save_order(self, order_data: dict):
        """
        Vangt de aanroep 'save_order(order_row)' uit client.py op,
        maar slaat het (voorlopig) op in je bestaande 'trades' tabel
        via 'save_trade(...)'.
        """

        # Timestamp uit order_data of huidige tijd
        tstamp = order_data.get("timestamp", get_current_utc_timestamp_ms())

        # In je client.py heet de coin/markt "market", terwijl 'save_trade' expects "symbol".
        # We mappen dat dus even om:
        trade_data = {
            "timestamp": tstamp,
            "symbol": order_data.get("market", "UNKNOWN"),  # mapped
            "side": order_data.get("side", "UNKNOWN"),
            "price": order_data.get("price", 0.0),
            "amount": order_data.get("amount", 0.0),
            # Oorspronkelijk "Bitvavo", nu "Kraken"
            # "exchange": order_data.get("exchange", "Bitvavo"),
            "exchange": order_data.get("exchange", "Kraken"),
            "status": order_data.get("status", "open"),
            "pnl_eur": 0.0,
            "fees": 0.0,
            "trade_cost": 0.0,

            # Position/logische kolommen:
            "position_id": order_data.get("order_id", None),
            "position_type": None,

            # AANPASSING: Als je 'strategy_name' in order_data meegeeft, nemen we die over.
            "strategy_name": order_data.get("strategy_name", None)
        }

        # Re-use je bestaande trades-logica:
        self.save_trade(trade_data)
        logger.info(f"[save_order] order_data gemapt -> save_trade: {order_data}")
