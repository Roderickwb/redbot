# main.py

import logging
import os
from python_bitvavo_api.bitvavo import Bitvavo
from dotenv import load_dotenv

# Lokale imports
from src.config.config import DB_FILE, MAIN_LOG_FILE
from src.logger.logger import setup_logger, setup_database_logger
from src.database_manager.database_manager import DatabaseManager
from src.trading_engine.executor import Executor

logger = setup_logger(name="main", log_file=MAIN_LOG_FILE, level=logging.DEBUG)

# Globale DatabaseManager, zodat bijv. dashboard.py 'm kan importeren
db_manager = DatabaseManager(db_path=DB_FILE)

def main():
    """
    Hoofdscript dat:
      - .env leest (ENVIRONMENT, API_KEY, API_SECRET, KRAKEN_ENV)
      - USE_WEBSOCKET en PAPER_TRADING bepaalt o.b.v. ENVIRONMENT
      - USE_KRAKEN en KRAKEN_PAPER bepaalt o.b.v. KRAKEN_ENV
      - Tabellen aanmaakt
      - Executor start
    """
    # 1) Lees .env voor ENVIRONMENT, API_KEY, API_SECRET, en evt. KRAKEN_ENV
    load_dotenv()
    # Debug-print direct daarna:
    print("[DEBUG] In main, BITVAVO_API_KEY =", os.getenv("BITVAVO_API_KEY"))
    print("[DEBUG] In main, BITVAVO_API_SECRET =", os.getenv("BITVAVO_API_SECRET"))

    # ### (NIEUW/GEWIJZIGD) Bepaal environment-mode (Bitvavo)
    ENVIRONMENT = os.getenv("ENVIRONMENT", "paper")
    if ENVIRONMENT == "production":
        USE_WEBSOCKET = True
        PAPER_TRADING = False
    elif ENVIRONMENT == "paper":
        USE_WEBSOCKET = True
        PAPER_TRADING = True
    else:  # development
        USE_WEBSOCKET = False
        PAPER_TRADING = True

    # ### (NIEUW/GEWIJZIGD) KRAKEN_ENV lezen
    KRAKEN_ENV = os.getenv("KRAKEN_ENV", "off")  # default => geen kraken
    if KRAKEN_ENV.lower() == "paper":
        USE_KRAKEN = True
        KRAKEN_PAPER = True
    elif KRAKEN_ENV.lower() == "real":
        USE_KRAKEN = True
        KRAKEN_PAPER = False
    else:
        USE_KRAKEN = False
        KRAKEN_PAPER = False

    # Haal API-key en secret uit .env (Bitvavo)
    API_KEY = os.getenv("BITVAVO_API_KEY")
    API_SECRET = os.getenv("BITVAVO_API_SECRET")
    if not API_KEY or not API_SECRET:
        logger.warning("API_KEY/API_SECRET niet gevonden in .env (of leeg).")

    # 2) Setup Database logger
    database_logger = setup_database_logger(
        logfile="logs/database_manager.log",
        level=logging.DEBUG
    )

    # [CHANGED] - create_tables() zorgt dat in alle tabellen de 'exchange' kolom nu bestaat.
    db_manager.create_tables()
    logger.info("Database-tables ensured/created.")

    # 4) Log info over modus
    logger.info(f"ENVIRONMENT={ENVIRONMENT}, USE_WEBSOCKET={USE_WEBSOCKET}, PAPER_TRADING={PAPER_TRADING}")
    logger.info(f"KRAKEN_ENV={KRAKEN_ENV}, USE_KRAKEN={USE_KRAKEN}, KRAKEN_PAPER={KRAKEN_PAPER}")

    if USE_WEBSOCKET:
        logger.info("LIVE WEBSOCKET mode ingeschakeld (voor Bitvavo).")
    else:
        logger.info("WEBSOCKET uitgeschakeld (ontwikkeling/paper) voor Bitvavo.")

    if PAPER_TRADING:
        logger.info("PAPER TRADING mode (fake orders) voor Bitvavo.")
    else:
        logger.info("REAL TRADING mode (echte orders) voor Bitvavo.")

    if USE_KRAKEN:
        if KRAKEN_PAPER:
            logger.info("Kraken => PAPER mode (fake orders), wel live data (OHLC).")
        else:
            logger.info("Kraken => REAL trading (nog niet geïmplementeerd?), wel live data.")
    else:
        logger.info("Kraken uitgeschakeld (KRAKEN_ENV=off).")

    # 5) (optioneel) Bitvavo REST
    bitvavo = Bitvavo({
        'APIKEY': API_KEY,
        'APISECRET': API_SECRET
    })

    # 6) Maak een Executor aan en start
    executor = Executor(
        db_manager=db_manager,
        use_websocket=USE_WEBSOCKET,
        paper_trading=PAPER_TRADING,
        api_key=API_KEY,
        api_secret=API_SECRET,
        ### (NIEUW/GEWIJZIGD) => geef hier Kraken-flags door
        use_kraken=USE_KRAKEN,
        kraken_paper=KRAKEN_PAPER
    )

    try:
        # Eventueel daily tasks (ML training, etc.)
        executor.run_daily_tasks()

        # hoofd-loop
        executor.run()

    except KeyboardInterrupt:
        logger.info("Bot handmatig gestopt (CTRL+C).")
    except Exception as e:
        logger.exception(f"Fout in executor.run(): {e}")
    finally:
        logger.info("Bot wordt afgesloten (main).")

        # DB-check: zie hoevel data erin zit
        try:
            candles_count = db_manager.get_table_count("candles")
            ticker_count = db_manager.get_table_count("ticker")
            bids_count = db_manager.get_table_count("orderbook_bids")
            asks_count = db_manager.get_table_count("orderbook_asks")
            logger.info(
                f"Data in DB -> Candles: {candles_count}, "
                f"Ticker: {ticker_count}, Bids: {bids_count}, Asks: {asks_count}"
            )
        except Exception as e:
            logger.error(f"Fout bij controleren van data in DB: {e}")

if __name__ == "__main__":
    logger.info(f"Start main script, PID={os.getpid()}")
    main()
