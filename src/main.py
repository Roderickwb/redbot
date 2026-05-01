# ============================================================
# src/main.py
# ============================================================

import logging
import os
from dotenv import load_dotenv

# Lokale imports
# Let op: we importeren direct config_logger niet - we vertrouwen op de logger uit config of elders
from src.config.config import DB_FILE, MAIN_LOG_FILE, yaml_config
from src.logger.logger import setup_logger, setup_database_logger
from src.database_manager.database_manager import DatabaseManager
from src.trading_engine.executor import Executor
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
    Hoofdscript voor de Kraken trend-bot:
      1) .env lezen
      2) Kraken runtime bepalen (off/paper/real)
      3) config aanvullen met API-keys
      4) DB-tabellen klaarzetten
      5) Executor starten
    """
    # === Stap 1) .env inlezen ===
    load_dotenv()

    # Runtime label only; active trading venue is controlled by KRAKEN_ENV.
    ENVIRONMENT = os.getenv("ENVIRONMENT", "paper").lower()

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

    # config.yaml is al ingelezen via config.py
    logger.debug("config.yaml is al ingelezen via config.py.")

    # === Stap 3) Haal Kraken API-keys uit .env ===
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

    if USE_KRAKEN and (not KRAKEN_API_KEY or not KRAKEN_API_SECRET):
        logger.warning("KRAKEN_API_KEY/SECRET niet gevonden of leeg.")

    database_logger = setup_database_logger(logfile="logs/database_manager.log", level=logging.INFO)

    # === Stap 4) Maak tabellen aan ===
    db_manager.create_tables()
    db_manager.assert_writeable()
    logger.info("Database-tables ensured/created.")

    # === Samenvattende log ===
    logger.info(f"ENVIRONMENT={ENVIRONMENT}")
    logger.info(f"KRAKEN_ENV={KRAKEN_ENV}, USE_KRAKEN={USE_KRAKEN}, KRAKEN_PAPER={KRAKEN_PAPER}")

    if USE_KRAKEN:
        if KRAKEN_PAPER:
            logger.info("Kraken => PAPER mode.")
        else:
            logger.info("Kraken => REAL mode.")
    else:
        logger.info("Kraken uitgeschakeld.")

    # Inject Kraken API keys from .env into runtime config
    if "kraken" not in yaml_config:
        yaml_config["kraken"] = {}
    yaml_config["kraken"]["apiKey"] = KRAKEN_API_KEY
    yaml_config["kraken"]["apiSecret"] = KRAKEN_API_SECRET

    # Active Telegram notifier for strategy messages
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
            candles_kraken_count = db_manager.get_table_count("candles_kraken")
            trades_count = db_manager.get_table_count("trades")
            trade_signals_count = db_manager.get_table_count("trade_signals")
            logger.info(
                f"Data => candles_kraken={candles_kraken_count}, trades={trades_count}, "
                f"trade_signals={trade_signals_count}"
            )
        except Exception as ex:
            logger.error(f"Fout bij controleren DB: {ex}")

if __name__ == "__main__":
    logger.info(f"Start main script, PID={os.getpid()}")
    main()
