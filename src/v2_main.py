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
      - .env leest (ENVIRONMENT, API_KEY, API_SECRET)
      - USE_WEBSOCKET en PAPER_TRADING bepaalt o.b.v. ENVIRONMENT
      - Tabellen aanmaakt
      - Executor start
    """
    # 1) Lees .env voor ENVIRONMENT, API_KEY, API_SECRET
    load_dotenv()

    # Bepaal environment-mode
    ENVIRONMENT = os.getenv("ENVIRONMENT", "paper")  # default: 'paper'
    if ENVIRONMENT == "production":
        USE_WEBSOCKET = True
        PAPER_TRADING = False
    elif ENVIRONMENT == "paper":
        USE_WEBSOCKET = True
        PAPER_TRADING = True
    else:
        # Bijv. "development"
        USE_WEBSOCKET = False
        PAPER_TRADING = True

    # Haal API-key en secret uit .env
    API_KEY = os.getenv("API_KEY")
    API_SECRET = os.getenv("API_SECRET")
    if not API_KEY or not API_SECRET:
        logger.warning("API_KEY/API_SECRET niet gevonden in .env (of leeg). Controleer je .env!")

    # 2) Setup Database logger
    database_logger = setup_database_logger(
        logfile="logs/database_manager.log",
        level=logging.DEBUG
    )

    # 3) Maak tabellen aan (indien nog niet bestaan)
    db_manager.create_tables()
    logger.info("Database-tables ensured/created.")

    # 4) Log info over modus
    logger.info(f"ENVIRONMENT={ENVIRONMENT}, USE_WEBSOCKET={USE_WEBSOCKET}, PAPER_TRADING={PAPER_TRADING}")

    if USE_WEBSOCKET:
        logger.info("LIVE WEBSOCKET mode ingeschakeld.")
    else:
        logger.info("WEBSOCKET uitgeschakeld (ontwikkeling/paper).")

    if PAPER_TRADING:
        logger.info("PAPER TRADING mode (fake orders).")
    else:
        logger.info("REAL TRADING mode (echte orders).")

    # 5) Eventueel Bitvavo REST (optioneel)
    #    Dit is handig als je naast WebSocket ook REST-calls wilt doen
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
        api_secret=API_SECRET
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

        # Als je in Executor wél self.ws_client hebt en je wilt expliciet stop_websocket aanroepen:
        # if executor.ws_client:
        #     executor.ws_client.stop_websocket()

        # Eventueel DB-check
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
