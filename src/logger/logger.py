import os
import json
import logging
from logging.handlers import RotatingFileHandler


## In logger/logger.py
def setup_database_logger(logfile="database_manager.log", level=logging.DEBUG):
    logger = logging.getLogger("database_manager")
    logger.setLevel(level)

    # FileHandler
    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    fh.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # === Toevoegen van console output op DEBUG ===
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def setup_logger(
    name,
    log_file,
    level=logging.DEBUG,  # default op DEBUG, INFO,  WARNING OR ERROR
    max_bytes=5_000_000,
    backup_count=5,
    use_json=False
):
    """
    Setup logger with rotating file handler and optional JSON formatting.

    :param name: Logger name.
    :param log_file: Path to the log file.
    :param level: Logging level (default: DEBUG).
    :param max_bytes: Max size of the log file in bytes before rotation (default: 5MB).
    :param backup_count: Number of backup log files to keep (default: 5).
    :param use_json: Whether to use JSON formatting for logs (default: False).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Controleer of er al handlers bestaan. Zo niet, voeg ze toe.
    if not logger.handlers:
        # Zorg dat de log-directory bestaat
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Formatter: JSON of tekst
        if use_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

        # === BEGIN CHANGE 1: RotatingFileHandler met delay=True ===
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            delay=True
        )
        # === END CHANGE 1

        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Eventueel ook console output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "name": record.name
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


# Hieronder de helperfuncties voor specifiekere logberichten.

def log_trade(logger, action, market, price, quantity, success=True):
    if success:
        logger.info(f"{action.capitalize()} uitgevoerd op {market} voor prijs {price} met hoeveelheid {quantity}.")
    else:
        logger.error(f"Fout bij {action} op {market} voor prijs {price} met hoeveelheid {quantity}.")

def log_error(logger, error_message):
    logger.error(f"Fout opgetreden: {error_message}")

def log_exception(logger, exception):
    logger.exception(f"Onverwachte uitzondering: {exception}")

def log_strategy_decision(logger, decision, reason):
    logger.info(f"Strategiebeslissing: {decision}. Reden: {reason}")

def log_account_balance(logger, balance):
    logger.info(f"Huidige accountbalans: {balance}")

def log_startup(logger):
    logger.info("Trading bot is gestart.")

def log_shutdown(logger):
    logger.info("Trading bot is gestopt.")

def log_config(logger, config):
    logger.info(f"Configuratie-instellingen: {config}")

def log_api_request(logger, api_endpoint, params):
    logger.info(f"API-aanroep naar {api_endpoint} met parameters: {params}")


# Testen van de functionaliteit (alleen als je dit script direct aanroept)
if __name__ == "__main__":
    # Stel hieronder even een test-logpad in, bijvoorbeeld:
    test_log_file = "logs/test_logger.log"

    test_logger = setup_logger("test_logger", test_log_file, level=logging.DEBUG)
    test_logger.info("Dit is een INFO bericht.")
    test_logger.debug("Dit is een DEBUG bericht.")
    test_logger.error("Dit is een ERROR bericht.")
    test_logger.warning("Dit is een WARNING bericht.")

    test_logger.debug(f"Test logger setup succesvol. Logbestand: {test_log_file}")
    test_logger.debug("Logger is klaar voor gebruik.")

