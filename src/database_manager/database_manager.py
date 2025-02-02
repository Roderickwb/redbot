# ============================================================
# src/database_manager/database_manager.py
# ============================================================

import sqlite3
import pandas as pd
import time
import logging
from datetime import datetime, timezone, timedelta
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

            # Buffer + batch_size voor candles
            self.candle_buffer = []
            self.batch_size = 100

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
                    self.flush_candles()  # Flush indien nodig
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

                    # Bepaal of we fetchall() moeten doen
                    lower_query = query.strip().lower()
                    if lower_query.startswith("select") or lower_query.startswith("pragma"):
                        rows = self.cursor.fetchall()
                        logger.debug(f"[execute_query] SELECT => fetched {len(rows)} rows.")
                        return rows
                    else:
                        # Bij INSERT, UPDATE, DELETE => None terug
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
        Creëer alle benodigde tabellen (candles, ticker, orderbook, indicators, trades, fills)
        en voer alter-statements uit (inclusief 'exchange' kolommen).
        """
        try:
            self.create_candles_table()
            self.create_ticker_table()
            self.create_orderbook_tables()
            self.create_indicators_table()
            self.alter_indicators_table()
            self.create_trades_table()
            self.create_fills_table()  # Nieuw

            # Zorg dat in diverse tabellen de kolom 'exchange' bestaat.
            self._ensure_exchange_in_all_tables()

            logger.info("[create_tables] Alle tabellen klaar of bijgewerkt.")
        except Exception as e:
            logger.error(f"[create_tables] Error: {e}")

    def _ensure_exchange_in_all_tables(self):
        """
        Zorgt dat de 'exchange' kolom in ticker, orderbook, indicators, trades, fills etc. bestaat.
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
            # Controleer of 'datetime_utc' al bestaat (niet strikt nodig als de CREATE statement klopt)
            self.cursor.execute("PRAGMA table_info(candles)")
            columns = [column[1] for column in self.cursor.fetchall()]
            if 'datetime_utc' not in columns:
                self.cursor.execute("ALTER TABLE candles ADD COLUMN datetime_utc TEXT")

            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles (timestamp)")
            self.connection.commit()
            logger.info("[create_candles_table] Candles tabel klaar.")
        except Exception as e:
            logger.error(f"[create_candles_table] Error: {e}")

    def drop_candles_table(self):
        """Voor debug/doeleinden - verwijder de hele 'candles'-tabel."""
        try:
            self.cursor.execute("DROP TABLE IF EXISTS candles")
            self.connection.commit()
            logger.info("[drop_candles_table] Tabel 'candles' is verwijderd.")
        except Exception as e:
            logger.error(f"[drop_candles_table] Error: {e}")

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

            # Extra index op candles
            self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_candles_mkt_int_ts
                ON candles (market, interval, timestamp)
            """)
            self.connection.commit()
            logger.info("[create_orderbook_tables] orderbook_bids/asks zijn klaar.")
        except sqlite3.Error as e:
            logger.error(f"[create_orderbook_tables] Error: {e}")

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

            # Check kolommen
            maybe_add = {
                'datetime_utc': 'TEXT',
                'position_id': 'TEXT',
                'position_type': 'TEXT',
                'status': 'TEXT',
                'pnl_eur': 'REAL',
                'fees': 'REAL',
                'trade_cost': 'REAL'
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

    def alter_trades_table(self):
        """
        Bestaande kolommen updaten voor 'trades' (als ze niet bestaan).
        """
        try:
            self.cursor.execute("PRAGMA table_info(trades)")
            existing_cols = [row[1] for row in self.cursor.fetchall()]

            new_columns = [
                ("position_id", "TEXT"),
                ("position_type", "TEXT"),
                ("status", "TEXT"),
                ("pnl_eur", "REAL"),
                ("fees", "REAL")
            ]
            for col_name, col_type in new_columns:
                if col_name not in existing_cols:
                    try:
                        self.cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
                        self.connection.commit()
                        logger.info(f"[alter_trades_table] Kolom '{col_name}' toegevoegd.")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" in str(e).lower():
                            logger.info(f"[alter_trades_table] Kolom {col_name} bestaat al => skip.")
                        else:
                            logger.error(f"[alter_trades_table] Fout bij kolom {col_name}: {e}")
        except Exception as e:
            logger.error(f"[alter_trades_table] Error: {e}")

    #
    # === NIEUW: fill-table & fill-methode
    #
    def create_fills_table(self):
        """
        Voor partial fills. Slaat elk fill-event op, bijvoorbeeld:
         - order_id / txid
         - market
         - side
         - fill_amount / fill_price / fee
         - timestamp
         - ...
        """
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
        """
        fill_data kan er bijvoorbeeld zo uitzien:
        {
           "order_id": "abcdef-12345",   # ordertxid / fill txid
           "market": "BTC-EUR",
           "side": "buy",               # of "sell"
           "fill_amount": 0.05,
           "fill_price": 30000.0,
           "fee_amount": 1.2,
           "timestamp": 1690023001234,
           "exchange": "Kraken" (of "Bitvavo", etc.)
        }
        """
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
            exch = fill_data.get("exchange", "Bitvavo")

            params = (order_id, market, side, f_amt, f_price, fee_amt,
                      ts, ts, exch)
            self.execute_query(q, params)
            logger.info(f"[save_fill] Fill opgeslagen: {fill_data}")
        except Exception as e:
            logger.error(f"[save_fill] Fout: {e}")

    # --------------------------------------------------------------------------
    # Candle buffer / flush
    # --------------------------------------------------------------------------
    def _validate_and_buffer_candles(self, data):
        """
        data => list van records: (timestamp, market, interval, open, high, low, close, volume).
        Als record 8 elementen bevat, voeg dan een default exchange toe ("Kraken").
        """
        try:
            valid_data = []
            for record in data:
                if len(record) == 8:
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
        """In 1 batch wegschrijven incl. datetime_utc en exchange."""
        if not self.candle_buffer:
            return
        try:
            final_list = []
            for row in self.candle_buffer:
                # Verwacht: (timestamp, market, interval, open, high, low, close, volume, exchange)
                ts, market, interval, o, h, l, c, vol, exch = row
                dt_utc = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                final_list.append((ts, dt_utc, market, interval, o, h, l, c, vol, exch))

            with self.connection:
                self.connection.executemany("""
                    INSERT OR REPLACE INTO candles
                    (timestamp, datetime_utc, market, interval, open, high, low, close, volume, exchange)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, final_list)

            logger.info(f"[flush_candle_buffer] {len(self.candle_buffer)} candle records weggeschreven.")
            self.candle_buffer.clear()
        except Exception as e:
            logger.error(f"[flush_candle_buffer] Fout: {e}")

    def flush_candles(self):
        """Extern aanroepbare methode om de buffer te flushen."""
        self._flush_candle_buffer()

    def start_flush_timer(self, interval_seconds=5):
        """
        Start een achtergrondthread die elke 'interval_seconds' de buffer flushes.
        Dit zorgt ervoor dat de candledata periodiek naar de database wordt geschreven.
        """
        def flush_loop():
            while True:
                time.sleep(interval_seconds)
                try:
                    self.flush_candles()
                    logger.debug("[flush_timer] Buffer geflusht.")
                except Exception as e:
                    logger.error(f"[flush_timer] Fout bij flushen: {e}")
        timer_thread = threading.Thread(target=flush_loop, daemon=True)
        timer_thread.start()


    def save_candles(self, data):
        """
        Public method om candles in bulk te saven:
         - data is list van tuples: (ts, market, interval, open, high, low, close, volume)
         - Als het record 8 elementen bevat, wordt default exchange toegevoegd.
         - Bufferen en flushen op batch_size.
        """
        logger.info(f"[save_candles] Aangeroepen met {len(data)} records.")
        count = self._validate_and_buffer_candles(data)
        if count > 0:
            logger.info(f"[save_candles] {count} candle records gebufferd. (Buffer size={len(self.candle_buffer)})")
        else:
            logger.info("[save_candles] Geen geldige candle-records gebufferd.")

    # --------------------------------------------------------------------------
    # save_ticker
    # --------------------------------------------------------------------------
    def save_ticker(self, data):
        """
        Verwacht data: {
          'market': 'BTC-EUR',
          'bestBid': ...,
          'bestAsk': ...,
          'exchange': 'Bitvavo' of 'Kraken' (optioneel, default 'Bitvavo'),
          ...
        }
        """
        try:
            timestamp = get_current_utc_timestamp_ms()
            market = data['market']
            best_bid = data.get('bestBid', 0.0)
            best_ask = data.get('bestAsk', 0.0)
            spread = best_ask - best_bid
            exchange = data.get('exchange', 'Bitvavo')
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
            logger.info(f"[save_ticker] Ticker data opgeslagen: {data}")
        except Exception as e:
            logger.error(f"[save_ticker] Fout: {e}")

    # --------------------------------------------------------------------------
    # save_orderbook
    # --------------------------------------------------------------------------
    def save_orderbook(self, data):
        """
        data = {
           'market': 'BTC-EUR',
           'bids': [...],
           'asks': [...],
           'exchange': 'Kraken' / 'Bitvavo' (optioneel, default 'Bitvavo')
        }
        """
        try:
            timestamp = get_current_utc_timestamp_ms()
            market = data['market']
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            exchange = data.get('exchange', 'Bitvavo')

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

            logger.info("[save_orderbook] orderbook data opgeslagen.")
        except Exception as e:
            logger.error(f"[save_orderbook] Fout: {e}")

    # --------------------------------------------------------------------------
    # save_indicators
    # --------------------------------------------------------------------------
    def save_indicators(self, df_with_indicators: pd.DataFrame):
        """
        Verwacht in df_with_indicators (pandas) de kolommen:
          timestamp, market, interval, rsi, macd, macd_signal,
          bollinger_upper, bollinger_lower, moving_average,
          ema_9, ema_21, atr14, exchange(?)
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
            exchange_val = row.get("exchange", "Bitvavo")

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
            logger.info("[save_indicators] Geen indicator-rows om op te slaan.")
            return

        try:
            with self.connect() as conn:
                conn.executemany(insert_q, rows)
            logger.info(f"[save_indicators] {len(rows)} rows inserted/updated in 'indicators'.")
        except Exception as e:
            logger.error(f"[save_indicators] Fout: {e}")

    # --------------------------------------------------------------------------
    # save_trade
    # --------------------------------------------------------------------------
    def save_trade(self, trade_data: dict):
        """
        Opslaan in trades:
          trade_data = {
            'timestamp': <ms>,
            'symbol': 'BTC-EUR',
            'side': 'buy'/'sell',
            'price': float,
            'amount': float,
            'position_id': str,
            'position_type': str,
            'status': 'open'/etc,
            'pnl_eur': float,
            'fees': float,
            'trade_cost': float,
            'exchange': 'Kraken' / 'Bitvavo' (optioneel, default 'Bitvavo')
          }
        """
        try:
            query = """
                INSERT INTO trades
                (timestamp, datetime_utc, symbol, side, price, amount,
                 position_id, position_type, status, pnl_eur, fees, trade_cost, exchange)
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
            tstamp = trade_data['timestamp']
            exchange_val = trade_data.get('exchange', 'Bitvavo')

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
                exchange_val
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
    # Misc: prune, fetch_data, get_candlesticks, etc.
    # --------------------------------------------------------------------------
    def prune_old_candles(self, days=30, interval=None):
        """
        Verwijder candles ouder dan X dagen (opt. alleen van een bepaald interval).
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
        Universele fetch-functie voor candles, ticker, orderbook, trades, indicators, etc.
        Met opt. filter op market, interval, exchange en limit.
        """
        try:
            params = []
            conditions = []

            # Bepaal de kolommen en base_query per tabel
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
                        exchange
                    FROM trades
                """
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
            else:
                base_query = f"SELECT * FROM {table_name}"

            # conditions
            if table_name == "candles":
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if interval:
                    conditions.append("interval = ?")
                    params.append(interval)
                if exchange:
                    conditions.append("exchange = ?")
                    params.append(exchange)

            elif table_name == "ticker":
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if exchange:
                    conditions.append("exchange = ?")
                    params.append(exchange)

            elif table_name in ["orderbook_bids", "orderbook_asks"]:
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if exchange:
                    conditions.append("exchange = ?")
                    params.append(exchange)

            elif table_name == "trades":
                if market:
                    conditions.append("symbol = ?")
                    params.append(market)
                if exchange:
                    conditions.append("exchange = ?")
                    params.append(exchange)

            elif table_name == "indicators":
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if interval:
                    conditions.append("interval = ?")
                    params.append(interval)
                if exchange:
                    conditions.append("exchange = ?")
                    params.append(exchange)

            elif table_name == "fills":
                if market:
                    conditions.append("market = ?")
                    params.append(market)
                if exchange:
                    conditions.append("exchange = ?")
                    params.append(exchange)

            # Als er conditions zijn, append "WHERE cond1 AND cond2..."
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

    def drop_orderbook_tables(self):
        """Verwijder de orderbook-bids en orderbook-asks tabellen (bijv. voor herinitialisatie)."""
        try:
            self.cursor.execute("DROP TABLE IF EXISTS orderbook_bids")
            self.cursor.execute("DROP TABLE IF EXISTS orderbook_asks")
            self.connection.commit()
            logger.info("[drop_orderbook_tables] Tabel 'orderbook_bids' & 'orderbook_asks' verwijderd.")
        except Exception as e:
            logger.error(f"[drop_orderbook_tables] Error: {e}")

    def reset_orderbook_tables(self):
        """Drop en hercreëer de orderbook-bids en asks tabellen."""
        self.drop_orderbook_tables()
        self.create_orderbook_tables()

    def get_candlesticks(self, market, interval="1m", limit=100, exchange=None):
        """
        Haal candles op met optioneel filter op exchange. Return is een pd.DataFrame.
        """
        try:
            df = self.fetch_data("candles", limit=limit, market=market, interval=interval, exchange=exchange)
            logger.debug(f"[get_candlesticks] Opgehaalde {df.shape[0]} rows voor {market} ({interval}).")
            return df
        except Exception as e:
            logger.error(f"[get_candlesticks] Fout: {e}")
            return pd.DataFrame()

    def get_ticker(self, market: str, exchange=None):
        """
        Haal 1 row op uit ticker (DESC-limiter=1), return dict.
        """
        df = self.fetch_data("ticker", market=market, limit=1, exchange=exchange)
        if not df.empty:
            return df.iloc[0].to_dict()
        return {}

    def fetch_open_trades(self):
        """
        Zoekt trades met status='open' (vb. om posities te herstellen).
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
        Haalt tot 50 bids en 50 asks op (DESC-limit=50).
        Return => {"bids":[[p,q],...], "asks":[[p,q],...]}
        """
        df_bids = self.fetch_data("orderbook_bids", market=market, limit=50, exchange=exchange)
        df_asks = self.fetch_data("orderbook_asks", market=market, limit=50, exchange=exchange)

        bids_list = df_bids[['bid_p', 'bid_q']].values.tolist() if not df_bids.empty else []
        asks_list = df_asks[['ask_p', 'ask_q']].values.tolist() if not df_asks.empty else []
        return {"bids": bids_list, "asks": asks_list}

    def close_connection(self):
        """Manuele afsluitmethode. Sluit DB-verbinding en flush buffer."""
        if self.connection:
            self.flush_candles()
            self.connection.close()
            logger.info("[close_connection] DB-verbinding is gesloten.")



