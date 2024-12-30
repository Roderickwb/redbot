# logger/logger.py

import logging
import os
import json
from logging.handlers import RotatingFileHandler


def setup_logger(name, log_file, level=logging.INFO, max_bytes=5_000_000, backup_count=5, use_json=False):
    """
    Setup logger with rotating file handler and optional JSON formatting.

    :param name: Logger name.
    :param log_file: Path to the log file.
    :param level: Logging level (default: INFO).
    :param max_bytes: Max size of the log file in bytes before rotation (default: 5MB).
    :param backup_count: Number of backup log files to keep (default: 5).
    :param use_json: Whether to use JSON formatting for logs (default: False).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Formatter: JSON or default text
        if use_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Rotating File Handler
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console Handler
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


# Helper functions for specific logging needs
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
