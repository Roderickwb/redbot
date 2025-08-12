# ============================================================
# src/main.py
# ============================================================

import logging
import os
from python_bitvavo_api.bitvavo import Bitvavo
from dotenv import load_dotenv
import yaml
from datetime import datetime, timedelta, timezone


# Lokale imports
# Let op: we importeren direct config_logger niet - we vertrouwen op de logger uit config of elders
from src.config.config import DB_FILE, MAIN_LOG_FILE, yaml_config
from src.logger.logger import setup_logger, setup_database_logger
from src.database_manager.database_manager import DatabaseManager
from src.trading_engine.executor import Executor
from src.utils.notifier import Notifier
from src.notifier.telegram_notifier import TelegramNotifier
from src.notifier.bus import set_notifier

# Stel de "main" logger in (RotatingFileHandler via setup_logger)
logger = setup_logger(name="main", log_file=MAIN_LOG_FILE, level=logging.INFO)
logger.info("Main logger geconfigureerd.")

# Globale DatabaseManager
db_manager = DatabaseManager(db_path=DB_FILE)

# Start de flush-timer: dit zorgt ervoor dat de candle-buffer elke 5 seconden wordt geleegd
db_manager.start_flush_timer(5)  # Flush elke 5 seconden

def main():
    """
    Hoofdscript dat:
      1) .env leest
      2) Bepaalt environment (paper vs production)
      3) Bepaalt kraken-env (off/paper/real)
      4) Leest config.yaml
      5) Maakt DB-tabellen
      6) Init Executor en start loop
    """
    # === Stap 1) .env inlezen ===
    load_dotenv()

    # Debugprint om te checken (console)
    #print("[DEBUG] BITVAVO_API_KEY =", os.getenv("BITVAVO_API_KEY"))
    #print("[DEBUG] BITVAVO_API_SECRET =", os.getenv("BITVAVO_API_SECRET"))
    #print("[DEBUG] KRAKEN_API_KEY =", os.getenv("KRAKEN_API_KEY"))
    #print("[DEBUG] KRAKEN_API_SECRET =", os.getenv("KRAKEN_API_SECRET"))

    # === Bepaal environment (Bitvavo) ===
    ENVIRONMENT = os.getenv("ENVIRONMENT", "paper").lower()
    if ENVIRONMENT == "production":
        USE_WEBSOCKET = True
        PAPER_TRADING = False
    elif ENVIRONMENT == "paper":
        USE_WEBSOCKET = True
        PAPER_TRADING = True
    else:  # development
        USE_WEBSOCKET = False
        PAPER_TRADING = True  # tijdelijk => Bitvavo even uitgeschakeld

    # === Bepaal KRAKEN_ENV (paper/real/off) ===
    KRAKEN_ENV = os.getenv("KRAKEN_ENV", "off").lower()
    if KRAKEN_ENV == "paper":
        USE_KRAKEN = True
        KRAKEN_PAPER = True
    elif KRAKEN_ENV == "real":
        USE_KRAKEN = True
        KRAKEN_PAPER = False
    else:
        USE_KRAKEN = False
        KRAKEN_PAPER = False

    # === Stap 2) config.yaml inlezen ===
    # je hebt config.yaml al ingelezen als yaml_config in config.py
    logger.debug("config.yaml is al ingelezen via config.py.")
    logger.debug(f"Inhoud van yaml_config: {yaml_config}")

    # === Stap 3) Haal API-keys uit .env (zowel Bitvavo als Kraken) ===
    BITVAVO_API_KEY = os.getenv("BITVAVO_API_KEY", "")
    BITVAVO_API_SECRET = os.getenv("BITVAVO_API_SECRET", "")

    # Kraken keys
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

    if not BITVAVO_API_KEY or not BITVAVO_API_SECRET:
        logger.warning("BITVAVO_API_KEY/SECRET niet gevonden of leeg.")

    database_logger = setup_database_logger(logfile="logs/database_manager.log", level=logging.INFO)

    # === Stap 4) Maak tabellen aan ===
    db_manager.create_tables()
    logger.info("Database-tables ensured/created.")

    # === Samenvattende log ===
    logger.info(f"ENVIRONMENT={ENVIRONMENT}, USE_WEBSOCKET={USE_WEBSOCKET}, PAPER_TRADING={PAPER_TRADING}")
    logger.info(f"KRAKEN_ENV={KRAKEN_ENV}, USE_KRAKEN={USE_KRAKEN}, KRAKEN_PAPER={KRAKEN_PAPER}")

    if USE_WEBSOCKET:
        logger.info("LIVE WEBSOCKET mode (Bitvavo).")
    else:
        logger.info("WS uitgeschakeld (Bitvavo).")

    if PAPER_TRADING:
        logger.info("Paper Trading (Bitvavo).")
    else:
        logger.info("REAL Trading (Bitvavo).")

    if USE_KRAKEN:
        if KRAKEN_PAPER:
            logger.info("Kraken => PAPER mode.")
        else:
            logger.info("Kraken => REAL mode.")
    else:
        logger.info("Kraken uitgeschakeld.")

    ### BITVAVO OFF ###
    # bitvavo = Bitvavo({
    #     'APIKEY': BITVAVO_API_KEY,
    #     'APISECRET': BITVAVO_API_SECRET
    # })
    #
    # logger.debug("[main] Aangemaakte Bitvavo-instance (REST + WS). (UIT, want Bitvavo = OFF)")
    ### END BITVAVO OFF ###

    # zet in yaml_config de kraken keys
    if "kraken" not in yaml_config:
        yaml_config["kraken"] = {}
    yaml_config["kraken"]["apiKey"] = KRAKEN_API_KEY
    yaml_config["kraken"]["apiSecret"] = KRAKEN_API_SECRET

    logger.debug(f"Kraken-configuratie in main.py: {yaml_config.get('kraken', {})}")

    # --- NOTIFIER (Telegram) ---
    n_cfg = yaml_config.get("notifier", {})
    notifier = Notifier(
        enabled=bool(n_cfg.get("enabled", False)),
        chat_id=str(n_cfg.get("chat_id", "")),
        token_env=str(n_cfg.get("token_env", "TELEGRAM_BOT_TOKEN")),
    )
    # --- END NOTIFIER ---

    # --- Telegram notifier (optional) ---
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        set_notifier(TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID))
        logger.info("Telegram notifier enabled.")
    else:
        logger.info("Telegram notifier not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID).")

    # === Stap 5) Maak Executor aan ===
    executor = Executor(
        db_manager=db_manager,
        use_websocket=USE_WEBSOCKET,
        paper_trading=PAPER_TRADING,
        api_key=BITVAVO_API_KEY,
        api_secret=BITVAVO_API_SECRET,
        use_kraken=USE_KRAKEN,
        kraken_paper=KRAKEN_PAPER,
        yaml_config=yaml_config
    )

    # === Stap 6) run daily tasks + hoofd-loop ===
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
        # DB-check
        try:
            candles_count = db_manager.get_table_count("candles")
            ticker_count = db_manager.get_table_count("ticker")
            bids_count = db_manager.get_table_count("orderbook_bids")
            asks_count = db_manager.get_table_count("orderbook_asks")
            logger.info(
                f"Data => Candles={candles_count}, Ticker={ticker_count}, Bids={bids_count}, Asks={asks_count}"
            )
        except Exception as ex:
            logger.error(f"Fout bij controleren DB: {ex}")


if __name__ == "__main__":
    logger.info(f"Start main script, PID={os.getpid()}")
    main()
