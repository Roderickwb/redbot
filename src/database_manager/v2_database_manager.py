# database_manager.py

import sqlite3
import pandas as pd
import time
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import threading

from src.config.config import DB_FILE

logger = logging.getLogger("database_manager")

def get_current_utc_timestamp_ms():
    """Geeft de huidige tijd in UTC in milliseconden."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)

class DatabaseManager:
    def __init__(self, db_path=DB_FILE):
        """
        Maakt verbinding met de SQLite–database en initialiseert de attributen.
        Roep zelf init_db() aan als je alle tabellen wilt maken.
        """
        self.db_path = db_path
        logger.debug(f"DB_FILE in DatabaseManager: {self.db_path}")
        logger.info(f"Database pad: {self.db_path}")

        try:
            # Verbind met de database en zet deze in WAL-modus
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
            self.connection.execute("PRAGMA journal_mode=WAL;")
            self.cursor = self.connection.cursor()
            self._db_lock = threading.Lock()  # Lock voor concurrency

            logger.info(f"Verbonden met database: {self.db_path}")

            # Niet automatisch self.create_tables() aanroepen, want we willen controle
            self.candle_buffer = []
            self.batch_size = 100

        except sqlite3.Error as e:
            logger.error(f"Fout bij verbinden met database: {e}")
            raise RuntimeError("Kan geen verbinding maken met de database.")

    def init_db(self):
        """
        (optioneel) Maak alle tabellen aan als ze nog niet bestaan,
        in dezelfde stijl als voorheen, maar nu handmatig oproepbaar.
        """
        self.create_tables()

    def connect(self):
        """Geeft de bestaande verbinding terug."""
        if self.connection:
            return self.connection
        else:
            raise RuntimeError("Geen actieve databaseverbinding.")

    def __del__(self):
        """Sluit de databaseverbinding netjes af wanneer de instantie wordt verwijderd."""
        if hasattr(self, 'connection') and self.connection:
            try:
                logger.warning("[DatabaseManager] Destructor aangeroepen. Controleer waarom dit gebeurt!")
                if self.connection:
                    logger.info("Destructor aangeroepen: Sluiten van databaseverbinding.")
                    self.flush_candles()  # Flush indien nodig
                    self.connection.commit()
                    self.connection.close()
                    logger.info("Databaseverbinding netjes afgesloten.")
            except sqlite3.ProgrammingError:
                logger.warning("Databaseverbinding was al gesloten.")
            except sqlite3.Error as e:
                logger.error(f"Fout bij het sluiten van de databaseverbinding: {e}")

    def execute_query(self, query, params=(), retries=10, delay=0.2):
        print("DEBUG: execute_query() is called with:", repr(query))
        with self._db_lock:
            for attempt in range(retries):
                try:
                    self.cursor.execute(query, params)
                    self.connection.commit()
                    print(f"DEBUG: query executed OK, now check if we do fetchall or None")
                    logger.info(f"Query succesvol uitgevoerd: {query} | Params: {params}")

                    lower_query = query.strip().lower()
                    print("DEBUG: lower_query =>", repr(lower_query))

                    if lower_query.startswith("select") or lower_query.startswith("pragma"):
                        print("DEBUG: We do a fetchall on cursor (for SELECT or PRAGMA).")
                        rows = self.cursor.fetchall()
                        print("DEBUG: fetchall() returned =>", rows)
                        return rows
                    else:
                        print("DEBUG: We return None (not SELECT or PRAGMA).")
                        return None

                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower():
                        logger.warning(f"Database is vergrendeld. Retry {attempt + 1}/{retries}...")
                        time.sleep(delay)
                    else:
                        logger.error(f"OperationalError bij query: {query} | Fout: {e}")
                        raise
                except Exception as e:
                    logger.error(f"Onverwachte fout bij uitvoeren query: {query} | Fout: {e}")
                    raise

            logger.error("Maximale aantal retries bereikt. Query niet uitgevoerd.")
            raise sqlite3.OperationalError("Database is vergrendeld na meerdere retries.")

    def get_table_count(self, table_name):
        """Haal het aantal records op uit een tabel."""
        try:
            rows = self.execute_query(f"SELECT COUNT(*) FROM {table_name}")
            if rows is None:
                return 0
            count = rows[0][0]
            logger.info(f"Tabel '{table_name}' bevat {count} records.")
            return count
        except Exception as e:
            logger.error(f"Error getting count from {table_name}: {e}")
            return 0

    # --------------------------------------------------------------------------
    # Creëren van tabellen, incl. extra kolom "datetime_utc"
    # --------------------------------------------------------------------------
    def create_ticker_table(self):
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS ticker (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,       -- extra kolom voor leesbare UTC
                    market TEXT NOT NULL,
                    best_bid REAL,
                    best_ask REAL,
                    spread REAL
                )
            """)
            self.cursor.execute("PRAGMA table_info(ticker)")
            columns = [column[1] for column in self.cursor.fetchall()]
            logger.info(f"Aanwezige kolommen in ticker-tabel: {columns}")

            # Eventueel alter statements:
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
            logger.info("Ticker table is klaar (incl. datetime_utc).")
        except sqlite3.Error as e:
            logger.error(f"Error creating/updating ticker table: {e}")

    def create_candles_table(self):
        """
        Maakt de candles-tabel aan of werkt deze bij.
        Nu met PRIMARY KEY (market, interval, timestamp), plus datetime_utc.
        """
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,   -- extra kolom
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
            self.cursor.execute("PRAGMA table_info(candles)")
            columns = [column[1] for column in self.cursor.fetchall()]
            if 'datetime_utc' not in columns:
                self.cursor.execute("ALTER TABLE candles ADD COLUMN datetime_utc TEXT")

            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles (timestamp)")
            self.connection.commit()
            logger.info("Candles table is klaar (incl. datetime_utc).")
        except Exception as e:
            logger.error(f"Error creating candles table: {e}")

    def drop_candles_table(self):
        try:
            self.cursor.execute("DROP TABLE IF EXISTS candles")
            self.connection.commit()
            logger.info("Tabel 'candles' is gedropt (verwijderd).")
        except Exception as e:
            logger.error(f"Error dropping candles table: {e}")

    def create_orderbook_tables(self):
        try:
            # bids
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_bids (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,   -- extra kolom
                    market TEXT NOT NULL,
                    bid_p REAL NOT NULL,
                    bid_q REAL NOT NULL
                )
            """)
            self.cursor.execute("PRAGMA table_info(orderbook_bids)")
            columns = [column[1] for column in self.cursor.fetchall()]
            if 'datetime_utc' not in columns:
                self.cursor.execute("ALTER TABLE orderbook_bids ADD COLUMN datetime_utc TEXT")

            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_orderbook_bids_timestamp ON orderbook_bids (timestamp)")

            # asks
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_asks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    datetime_utc TEXT,   -- extra kolom
                    market TEXT NOT NULL,
                    ask_p REAL NOT NULL,
                    ask_q REAL NOT NULL
                )
            """)
            self.cursor.execute("PRAGMA table_info(orderbook_asks)")
            columns = [column[1] for column in self.cursor.fetchall()]
            if 'datetime_utc' not in columns:
                self.cursor.execute("ALTER TABLE orderbook_asks ADD COLUMN datetime_utc TEXT")

            self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_candles_mkt_int_ts
                ON candles (market, interval, timestamp)
            """)
            self.connection.commit()
            logger.info("Orderbook_bids en orderbook_asks tabellen zijn klaar (incl. datetime_utc).")
        except sqlite3.Error as e:
            logger.error(f"Error creating orderbook tables: {e}")

    def create_indicators_table(self):
        try:
            create_indicators_table_sql = """
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                datetime_utc TEXT,   -- extra kolom
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
            """
            self.connect().execute(create_indicators_table_sql)
            self.connection.commit()

            self.cursor.execute("PRAGMA table_info(indicators)")
            columns = [column[1] for column in self.cursor.fetchall()]
            if 'datetime_utc' not in columns:
                self.cursor.execute("ALTER TABLE indicators ADD COLUMN datetime_utc TEXT")

            logger.info("Tabel 'indicators' is succesvol aangemaakt of bestond al.")
        except Exception as e:
            logger.error(f"Fout bij aanmaken 'indicators' tabel: {e}")

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
                logger.info(f"Uitgevoerd: {sql}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    logger.info(f"Kolom bestaat al, skip: {sql}")
                else:
                    logger.error(f"Fout bij alter_indicators_table ({sql}): {e}")
            except Exception as e:
                logger.error(f"Onverwachte fout bij alter_indicators_table ({sql}): {e}")

    def create_trades_table(self):
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,   -- UTC in ms
                    datetime_utc TEXT,           -- extra kolom
                    symbol TEXT NOT NULL,         -- bijv. 'BTC-EUR'
                    side TEXT NOT NULL,           -- 'BUY' of 'SELL'
                    price REAL NOT NULL,
                    amount REAL,
                    position_id TEXT,
                    position_type TEXT,
                    status TEXT,
                    pnl_eur REAL,
                    fees REAL,
                    trade_cost REAL
                )
            """)
            self.connection.commit()
            logger.info("Trades table is klaar.")

            self.cursor.execute("PRAGMA table_info(trades)")
            columns = [col[1] for col in self.cursor.fetchall()]
            logger.info(f"Kolommen in 'trades': {columns}")

            if 'datetime_utc' not in columns:
                self.cursor.execute("ALTER TABLE trades ADD COLUMN datetime_utc TEXT")
            if 'position_id' not in columns:
                self.cursor.execute("ALTER TABLE trades ADD COLUMN position_id TEXT")
            if 'position_type' not in columns:
                self.cursor.execute("ALTER TABLE trades ADD COLUMN position_type TEXT")
            if 'status' not in columns:
                self.cursor.execute("ALTER TABLE trades ADD COLUMN status TEXT")
            if 'pnl_eur' not in columns:
                self.cursor.execute("ALTER TABLE trades ADD COLUMN pnl_eur REAL")
            if 'fees' not in columns:
                self.cursor.execute("ALTER TABLE trades ADD COLUMN fees REAL")
            if 'trade_cost' not in columns:
                self.cursor.execute("ALTER TABLE trades ADD COLUMN trade_cost REAL")

            self.connection.commit()
            logger.info("Trades table is klaar (inclusief extra kolommen).")

        except Exception as e:
            logger.error(f"Fout bij create_trades_table: {e}")

    def create_tables(self):
        """
        Creëer de benodigde tabellen en voer alter-statements uit.
        """
        try:
            self.create_candles_table()
            self.create_ticker_table()
            self.create_orderbook_tables()
            self.create_indicators_table()
            self.alter_indicators_table()
            self.create_trades_table()
            logger.info("Alle tabellen zijn succesvol aangemaakt of bijgewerkt.")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")

    def alter_trades_table(self):
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
                        logger.info(f"Kolom '{col_name}' toegevoegd aan trades-tabel.")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" in str(e).lower():
                            logger.info(f"Kolom {col_name} bestaat al, skip.")
                        else:
                            logger.error(f"Fout bij toevoegen van kolom {col_name}: {e}")
                    except Exception as e:
                        logger.error(f"Onverwachte fout bij alter_trades_table ({col_name}): {e}")

        except Exception as e:
            logger.error(f"Fout bij alter_trades_table: {e}")

    # --------------------------------------------------------------------------
    # Candle buffer / flush
    # --------------------------------------------------------------------------
    def _validate_and_buffer_candles(self, data):
        """
        Elke record is: (timestamp, market, interval, open, high, low, close, volume).
        timestamp is ms.
        """
        try:
            valid_data = []
            for record in data:
                if len(record) != 8:
                    logger.warning(f"Ongeldig record genegeerd: {record}")
                    continue

                timestamp, market, interval, open_, high, low, close, volume = record
                # Houd alle isinstance-checks die je al had:
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
                    logger.warning(f"Ongeldig record genegeerd: {record}")

            if not valid_data:
                logger.warning("Geen geldige candle data om toe te voegen aan buffer.")
                return 0

            self.candle_buffer.extend(valid_data)
            if len(self.candle_buffer) >= self.batch_size:
                self._flush_candle_buffer()

            return len(valid_data)

        except sqlite3.OperationalError as e:
            logger.error(f"Databasefout bij buffering van candle data: {e}")
            return 0
        except Exception as e:
            logger.error(f"Onverwachte fout bij buffering van candle data: {e}")
            return 0

    def _flush_candle_buffer(self):
        """In 1 batch wegschrijven incl. datetime_utc."""
        if not self.candle_buffer:
            return
        try:
            # We bouwen een list of tuples, nu met extra param => datetime_utc = datetime(?/1000,'unixepoch')
            # We vervormen elk tuple (ts, market, interval, open, high, low, close, volume)
            final_list = []
            for row in self.candle_buffer:
                (ts, market, interval, open_, high, low, close, volume) = row
                final_list.append((
                    ts,
                    ts,  # voor datetime(?/1000,'unixepoch')
                    market,
                    interval,
                    open_,
                    high,
                    low,
                    close,
                    volume
                ))

            with self.connection:
                self.connection.executemany("""
                    INSERT OR REPLACE INTO candles
                    (timestamp, datetime_utc, market, interval, open, high, low, close, volume)
                    VALUES (
                      ?,
                      datetime(?/1000, 'unixepoch'),
                      ?,
                      ?,
                      ?,
                      ?,
                      ?,
                      ?,
                      ?
                    )
                """, final_list)

            logger.info(f"{len(self.candle_buffer)} candle records in één keer weggeschreven (incl. datetime_utc).")
            self.candle_buffer.clear()
        except Exception as e:
            logger.error(f"Fout bij flushen van candle_buffer: {e}")

    def flush_candles(self):
        self._flush_candle_buffer()

    def save_candles(self, data):
        logger.info(f"save_candles() aangeroepen met {len(data)} records.")
        count = self._validate_and_buffer_candles(data)
        if count > 0:
            logger.info(f"{count} candle records gevalideerd en gebufferd. (Buffer size = {len(self.candle_buffer)})")
        else:
            logger.info("Geen geldige candle-records gebufferd.")

    # --------------------------------------------------------------------------
    # save_ticker
    # --------------------------------------------------------------------------
    def save_ticker(self, data):
        """Slaat de ticker-data op in de database met retry-mechanisme + datetime_utc."""
        try:
            timestamp = get_current_utc_timestamp_ms()
            market = data['market']
            best_bid = data.get('bestBid', 0.0)
            best_ask = data.get('bestAsk', 0.0)
            spread = best_ask - best_bid
            logger.info(f"Ticker data ontvangen voor opslag: {data}")

            query = """
                INSERT INTO ticker
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
            self.execute_query(query, params)
            logger.info(f"Ticker data succesvol opgeslagen: {data}")
        except sqlite3.OperationalError as e:
            logger.error(f"Databasefout bij opslaan van ticker data: {e}")
        except Exception as e:
            logger.error(f"Fout bij opslaan van ticker data: {e}")

    # --------------------------------------------------------------------------
    # save_orderbook
    # --------------------------------------------------------------------------
    def save_orderbook(self, data):
        """Slaat orderbook data op (bids & asks) in DB, incl. datetime_utc."""
        try:
            timestamp = get_current_utc_timestamp_ms()
            market = data['market']
            bids = data.get('bids', [])
            asks = data.get('asks', [])

            for bid in bids:
                query = """
                    INSERT INTO orderbook_bids
                    (timestamp, datetime_utc, market, bid_p, bid_q)
                    VALUES (
                      ?,
                      datetime(?/1000, 'unixepoch'),
                      ?,
                      ?,
                      ?
                    )
                """
                params = (timestamp, timestamp, market, float(bid[0]), float(bid[1]))
                self.execute_query(query, params)

            for ask in asks:
                query = """
                    INSERT INTO orderbook_asks
                    (timestamp, datetime_utc, market, ask_p, ask_q)
                    VALUES (
                      ?,
                      datetime(?/1000, 'unixepoch'),
                      ?,
                      ?,
                      ?
                    )
                """
                params = (timestamp, timestamp, market, float(ask[0]), float(ask[1]))
                self.execute_query(query, params)

            logger.info("Orderbook data succesvol opgeslagen.")
        except sqlite3.OperationalError as e:
            logger.error(f"Databasefout bij opslaan van orderbook data: {e}")
        except Exception as e:
            logger.error(f"Fout bij opslaan van orderbook data: {e}")

    # --------------------------------------------------------------------------
    # save_indicators
    # --------------------------------------------------------------------------
    def save_indicators(self, df_with_indicators: pd.DataFrame):
        """
        Sla de indicatoren uit df_with_indicators op in de tabel 'indicators',
        incl. datetime_utc = datetime(timestamp/1000,'unixepoch').
        """
        insert_query = """
            INSERT OR REPLACE INTO indicators
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
        rows_to_insert = []
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

            rows_to_insert.append((
                ts_val,
                ts_val,  # voor datetime(?/1000,'unixepoch')
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
        if not rows_to_insert:
            logger.info("Geen indicator-rows om op te slaan.")
            return

        try:
            with self.connect() as conn:
                conn.executemany(insert_query, rows_to_insert)
            logger.info(f"{len(rows_to_insert)} indicator-rows inserted/updated in 'indicators'.")
        except Exception as e:
            logger.error(f"Fout bij opslaan van indicators: {e}")

    # --------------------------------------------------------------------------
    # save_trade
    # --------------------------------------------------------------------------
    def save_trade(self, trade_data: dict):
        """
        Voorbeeld trade_data = {
          'timestamp': <ms>,
          'symbol': ...,
          'side': ...,
          ...
        }
        We slaan ook datetime_utc = datetime(timestamp/1000,'unixepoch').
        """
        try:
            query = """
                INSERT INTO trades
                (timestamp, datetime_utc, symbol, side, price, amount,
                 position_id, position_type, status, pnl_eur, fees, trade_cost)
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
                  ?
                )
            """
            position_id = trade_data.get('position_id', None)
            position_type = trade_data.get('position_type', None)
            status = trade_data.get('status', None)
            pnl_eur = trade_data.get('pnl_eur', 0.0)
            fees = trade_data.get('fees', 0.0)
            trade_cost = trade_data.get('trade_cost', 0.0)

            params = (
                trade_data['timestamp'],
                trade_data['timestamp'],
                trade_data['symbol'],
                trade_data['side'],
                trade_data['price'],
                trade_data['amount'],
                position_id,
                position_type,
                status,
                pnl_eur,
                fees,
                trade_cost
            )
            self.execute_query(query, params)
            logger.info(f"Trade data succesvol opgeslagen: {trade_data}")

        except Exception as e:
            logger.error(f"Onverwachte fout bij opslaan van trade data: {e}")

    def update_trade(self, trade_id: int, updates: dict):
        """
        Wijzig kolommen in trades na partial close of exit.
        vb. updates = {"status": "closed", "pnl_eur": 12.34}
        """
        try:
            set_clauses = []
            params = []
            for col, val in updates.items():
                set_clauses.append(f"{col} = ?")
                params.append(val)

            set_clause_str = ", ".join(set_clauses)
            query = f"UPDATE trades SET {set_clause_str} WHERE id = ?"
            params.append(trade_id)

            self.execute_query(query, tuple(params))
            logger.info(f"Trade {trade_id} geüpdatet met {updates}")

        except Exception as e:
            logger.error(f"Fout bij update_trade({trade_id}): {e}")

    # --------------------------------------------------------------------------
    # Misc
    # --------------------------------------------------------------------------
    def prune_old_candles(self, days=30, interval=None):
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
            query = """
                DELETE FROM candles
                WHERE timestamp < ?
            """
            params = (cutoff_ms,)
            extra_info = "alle intervals"

        try:
            self.execute_query(query, params)
            logger.info(f"Verwijderd: candles ouder dan {days} dagen ({extra_info}).")
        except Exception as e:
            logger.error(f"Fout bij prunen van oude candles: {e}")

    def fetch_data(self, table_name, limit=100, market=None, interval=None):
        """
        Haal data op uit de DB (candles, ticker, orderbook, trades, enz.).
        """
        try:
            params = []
            conditions = []

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
                        volume
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
                        spread
                    FROM ticker
                """
            elif table_name in ["orderbook_bids", "orderbook_asks"]:
                base_query = f"""
                    SELECT
                        id,
                        timestamp,
                        datetime_utc,
                        market,
                        {'bid_p, bid_q' if table_name=='orderbook_bids' else 'ask_p, ask_q'}
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
                        atr14
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
                        trade_cost
                    FROM trades
                """
            else:
                # fallback
                base_query = f"SELECT * FROM {table_name}"

            # conditions
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
            elif table_name == "trades" and market:
                conditions.append("symbol = ?")
                params.append(market)
            # indicators: (optioneel) filteren op market, interval, etc.

            if conditions:
                base_query += " WHERE " + " AND ".join(conditions)

            base_query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            logger.info(f"fetch_data => {base_query} | params={params}")
            df = pd.read_sql_query(base_query, self.connection, params=params)
            logger.info(f"Fetched {len(df)} records from {table_name}.")
            return df

        except Exception as e:
            logger.error(f"Error fetching data from {table_name}: {e}")
            return pd.DataFrame()

    def drop_orderbook_tables(self):
        try:
            self.cursor.execute("DROP TABLE IF EXISTS orderbook_bids")
            self.cursor.execute("DROP TABLE IF EXISTS orderbook_asks")
            self.connection.commit()
            logger.info("Orderbook_bids en orderbook_asks tabellen verwijderd.")
        except Exception as e:
            logger.error(f"Error dropping orderbook tables: {e}")

    def reset_orderbook_tables(self):
        self.drop_orderbook_tables()
        self.create_orderbook_tables()

    def close_connection(self):
        if self.connection:
            self.flush_candles()
            self.connection.close()
            logger.info("Database verbinding gesloten.")

    def get_candlesticks(self, market, interval="1m", limit=100):
        df = self.fetch_data("candles", limit=limit, market=market, interval=interval)
        logger.debug(f"Ontvangen candlestick-data:\n{df.head()}")
        return df

    def get_ticker(self, market: str):
        df = self.fetch_data("ticker", market=market, limit=1)
        if not df.empty:
            return df.iloc[0].to_dict()
        return {}

    def fetch_open_trades(self):
        query = """
            SELECT
                symbol, side, amount, price,
                position_id, position_type, status
            FROM trades
            WHERE status = 'open'
        """
        rows = self.execute_query(query)
        if not rows:
            return []

        columns = ["symbol", "side", "amount", "price", "position_id", "position_type", "status"]
        results = []
        for row in rows:
            rowdict = dict(zip(columns, row))
            results.append(rowdict)
        return results

    def get_orderbook_snapshot(self, market: str):
        df_bids = self.fetch_data("orderbook_bids", market=market, limit=50)
        df_asks = self.fetch_data("orderbook_asks", market=market, limit=50)

        bids_list = df_bids[['bid_p', 'bid_q']].values.tolist() if not df_bids.empty else []
        asks_list = df_asks[['ask_p', 'ask_q']].values.tolist() if not df_asks.empty else []
        return {"bids": bids_list, "asks": asks_list}
