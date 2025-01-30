import os
import json
import logging
from logging.handlers import RotatingFileHandler

################################################################################
# Database logger
################################################################################
def setup_database_logger(logfile="database_manager.log", level=logging.DEBUG):
    """
    Deze logger is bedoeld voor database_manager.
    Schrijft naar een file én de console, op DEBUG-level.
    """
    logger = logging.getLogger("database_manager")
    logger.setLevel(level)

    # Zorg dat de directory bestaat
    log_dir = os.path.dirname(logfile)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    fh.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console output
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


################################################################################
# Hoofd setup_logger (RotatingFileHandler, JSON-optie)
################################################################################
def setup_logger(
    name,
    log_file,
    level=logging.DEBUG,  # default op DEBUG
    max_bytes=5_000_000,
    backup_count=5,
    use_json=False
):
    """
    Setup logger with rotating file handler and optional JSON formatting.

    :param name: Logger name.
    :param log_file: Path to the log file.
    :param level: Logging level (default: DEBUG).
    :param max_bytes: Max size in bytes before rotation (default: 5MB).
    :param backup_count: # backups
    :param use_json: True => JSONFormatter, else plain text
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if the logger has handlers; only add if none
    if not logger.handlers:
        # Ensure directory
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if use_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            delay=True
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


################################################################################
# (NIEUW) setup_kraken_logger
################################################################################
def setup_kraken_logger(
    logfile="logs/kraken_client.log",
    level=logging.DEBUG,
    max_bytes=5_000_000,
    backup_count=5,
    use_json=False
):
    """
    Shortcut om snel een 'kraken_client' logger te maken,
    als je in kraken.py of kraken_mixed_client.py een dedicated log wil.
    """
    return setup_logger(
        name="kraken_client",
        log_file=logfile,
        level=level,
        max_bytes=max_bytes,
        backup_count=backup_count,
        use_json=use_json
    )


################################################################################
# JSONFormatter
################################################################################
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


################################################################################
# Helpers for specific log messages
################################################################################
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


################################################################################
# Test if called directly
################################################################################
if __name__ == "__main__":
    test_log_file = "logs/test_logger.log"
    test_logger = setup_logger("test_logger", test_log_file, level=logging.DEBUG)
    test_logger.info("Dit is een INFO bericht.")
    test_logger.debug("Dit is een DEBUG bericht.")
    test_logger.error("Dit is een ERROR bericht.")
    test_logger.warning("Dit is een WARNING bericht.")
    test_logger.debug(f"Test logger setup succesvol. Logbestand: {test_log_file}")
    test_logger.debug("Logger is klaar voor gebruik.")
