import os
import json
import logging
from logging.handlers import RotatingFileHandler  # TimedRotatingFileHandler laten we erin, eventueel nog gebruikt?
# from logging.handlers import TimedRotatingFileHandler # <--- Niet meer nodig voor websocket-logger

################################################################################
# JSONFormatter
################################################################################
class JSONFormatter(logging.Formatter):
    """Custom JSON formatter voor gestructureerde logging."""

    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage()
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


################################################################################
# Hoofd setup_logger (RotatingFileHandler, optioneel JSON-format)
################################################################################
def setup_logger(name,
                 log_file,
                 level=logging.INFO,  # Default: DEBUG Kan altijd weer
                 max_bytes=5_000_000,  # 5 MB standaard (pas dit aan indien nodig)
                 backup_count=5,
                 use_json=False):
    """
    Stel een logger in met een RotatingFileHandler en optioneel JSON-formatting.

    :param name: Naam van de logger.
    :param log_file: Pad naar het logbestand.
    :param level: Logging level (default: DEBUG).
    :param max_bytes: Maximum aantal bytes voor rotatie (default: 5 MB).
    :param backup_count: Aantal backup-bestanden.
    :param use_json: Indien True, wordt JSONFormatter gebruikt; anders gewone tekst.
    :return: De geconfigureerde logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Voeg handlers alleen toe als deze nog niet bestaan
    if not logger.handlers:
        # Zorg dat de directory voor het logbestand bestaat
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Kies de formatter: JSON of plain text
        if use_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

        # Standaard: RotatingFileHandler op grootte
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            delay=True,  # Bestandsopening uitgesteld tot eerste log
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Voorkom dubbele logs via root logger
        logger.propagate = False

    return logger


################################################################################
# WebSocket-logger ZONDER TimedRotatingFileHandler
################################################################################
def setup_websocket_logger(log_file="logs/websocket_client.log",
                           level=logging.DEBUG,
                           # De volgende argumenten houden we voor compatibiliteit,
                           # maar we negeren ze, zodat we geen rename-problemen krijgen.
                           when="midnight",
                           interval=1,
                           backup_count=5,
                           use_json=False):
    """
    Logger voor WebSocket. Oorspronkelijk met TimedRotatingFileHandler,
    maar nu vervangen door een simpele FileHandler om WinError 32 te voorkomen.
    Er vindt dus geen automatische rotatie meer plaats.
    """
    name = "websocket_client"
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Alleen filehandler + console, zonder tijd-gebaseerde rotatie:
    if not logger.handlers:
        # Zorg dat de directory bestaat
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if use_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

        # Gewone FileHandler i.p.v. TimedRotatingFileHandler
        file_handler = logging.FileHandler(
            filename=log_file,
            mode='a',  # Append-mode
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Geen propagation naar root, zodat we geen dubbele logs krijgen
        logger.propagate = False

    return logger


################################################################################
# Specifieke logger voor de database
################################################################################
def setup_database_logger(logfile="database_manager.log", level=logging.DEBUG):
    """
    Deze logger is bedoeld voor de database_manager.
    (RotatingFileHandler gebaseerd op 10 MB).
    """
    # Zorg dat de directory bestaat
    log_dir = os.path.dirname(logfile)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger("database_manager")
    logger.setLevel(level)

    # Check of er al handlers bestaan
    if not logger.handlers:
        # RotatingFileHandler: 10 MB limiet, 5 backups
        fh = RotatingFileHandler(
            logfile,
            mode='a',
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8',
            delay=True
        )
        fh.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console output toevoegen
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        logger.propagate = False

    return logger


################################################################################
# Specifieke logger voor Kraken
################################################################################
def setup_kraken_logger(logfile="logs/kraken_client.log",
                        level=logging.DEBUG,
                        max_bytes=5_000_000,
                        backup_count=5,
                        use_json=False):
    """
    Snel een dedicated 'kraken_client' logger maken.
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
# Overige hulpfuncties (ongewijzigd)
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
# Test als direct uitgevoerd
################################################################################
if __name__ == "__main__":
    # Test: WebSocket-logger zonder rotatie (en dus geen WinError 32).
    ws_logger = setup_websocket_logger(
        log_file="logs/websocket_client.log",
        level=logging.DEBUG,
        when="midnight",  # Wordt nu genegeerd
        interval=1,       # Genegeerd
        backup_count=5,   # Genegeerd
        use_json=False
    )
    ws_logger.info("Dit is een test-INFO voor de websocket-logger (ZONDER TimedRotatingFileHandler).")

    # Test: hoofd-logger (met rotating op 5 MB)
    main_log = setup_logger(
        name="main",
        log_file="logs/main.log",
        level=logging.DEBUG,
        max_bytes=5_000_000,
        backup_count=5,
        use_json=False
    )
    main_log.info("Dit is een test-INFO voor de hoofd-logger (met size-based rotatie).")
