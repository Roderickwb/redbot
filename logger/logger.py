import logging

def setup_logger(log_file="app.log"):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()

# logging van handelsacties
def log_trade(action, market, price, quantity, success=True):
    if success:
        logger.info(f"{action.capitalize()} uitgevoerd op {market} voor prijs {price} met hoeveelheid {quantity}.")
    else:
        logger.error(f"Fout bij {action} op {market} voor prijs {price} met hoeveelheid {quantity}.")

# logging van fouten en uitzonderingen in je strategie
def log_error(error_message):
    logger.error(f"Fout opgetreden: {error_message}")

# logging van strategie beslissingen
def log_strategy_decision(decision, reason):
    logger.info(f"Strategiebeslissing: {decision}. Reden: {reason}")

# logging accountbalans
def log_account_balance(balance):
    logger.info(f"Huidige accountbalans: {balance}")


# loggen van het starten en stoppen van de bot
def log_startup():
    logger.info("Trading bot is gestart.")

def log_shutdown():
    logger.info("Trading bot is gestopt.")

# logging van configuratie instellingen
def log_config(config):
    logger.info(f"Configuratie-instellingen: {config}")

# logging van API-aanroepen
def log_api_request(api_endpoint, params):
    logger.info(f"API-aanroep naar {api_endpoint} met parameters: {params}")
