# main.py

import logging
import os
from python_bitvavo_api.bitvavo import Bitvavo
from dotenv import load_dotenv

from src.config.config import DB_FILE, MAIN_LOG_FILE
from src.logger.logger import setup_logger, setup_database_logger
from src.database_manager.database_manager import DatabaseManager
from src.trading_engine.executor import Executor

# ------------------------------------------
# Zet hier de booleans:
USE_WEBSOCKET = True     # True => LIVE data via WebSocket
PAPER_TRADING = True     # True => Fake (paper) orders, False => Echte orders
# ------------------------------------------

logger = setup_logger(name="main", log_file=MAIN_LOG_FILE, level=logging.DEBUG) # tijdelijk op debug dadelijk weer op warning of info

def main():
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    API_SECRET = os.getenv("API_SECRET")

    # Voorbeeld: je kunt de Bitvavo REST-client nog gebruiken voor extra calls
    bitvavo = Bitvavo({
        'APIKEY': API_KEY,
        'APISECRET': API_SECRET
    })

    # 1) Setup Database-logger VOOR de DatabaseManager
    database_logger = setup_database_logger(
        "logs/database_manager.log",
        level=logging.DEBUG
    )

    # 2) Maak de DB-manager aan
    db_manager = DatabaseManager(db_path=DB_FILE)
    db_manager.create_tables()
    logger.info("Database tables ensured/created.")

    # 3) Start de Executor
    if USE_WEBSOCKET:
        logger.info("LIVE WEBSOCKET mode ingeschakeld.")
    else:
        logger.info("WEBSOCKET uitgeschakeld (mogelijke PAPER-only mode).")

    if PAPER_TRADING:
        logger.info("PAPER TRADING mode (fake orders).")
    else:
        logger.info("REAL TRADING mode (echte orders).")

    executor = Executor(
        db_manager=db_manager,
        use_websocket=USE_WEBSOCKET,
        paper_trading=PAPER_TRADING  # <-- nieuw
    )

    try:
        # Eventueel extra daily tasks (ML-training e.d.):
        executor.run_daily_tasks()


        # 4) Draai de hoofd-loop
        executor.run()
    except KeyboardInterrupt:
        logger.info("Bot handmatig gestopt (main).")
    except Exception as e:
        logger.exception(f"Fout in executor.run(): {e}")
    finally:
        logger.info("Bot is afgesloten (main).")

    # === Stap 2: Stop de WebSocket expliciet vóór we DB-checks doen ===

    #if executor.ws_client:
    #    logger.info("Stop websocket expliciet vóór DB-check.")
    #    executor.ws_client.stop_websocket()
      # Eventueel mini-pauze als je zeker wilt zijn dat de thread is vrijgegeven:
    #    import time; time.sleep(1)

    # 5) Voorbeeld: check data in DB
    try:
        candles_count = db_manager.get_table_count("candles")
        ticker_count = db_manager.get_table_count("ticker")
        bids_count = db_manager.get_table_count("orderbook_bids")
        asks_count = db_manager.get_table_count("orderbook_asks")

        logger.info(
            f"Data beschikbaar - Candles: {candles_count}, "
            f"Ticker: {ticker_count}, Bids: {bids_count}, Asks: {asks_count}"
        )
    except Exception as e:
        logger.error(f"Fout bij controleren van data beschikbaarheid: {e}")


if __name__ == "__main__":
    logger.info(f"Start main script, PID={os.getpid()}")
    main()
