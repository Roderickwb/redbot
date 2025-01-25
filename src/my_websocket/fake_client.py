# src/my_websocket/fake_client.py

import logging
from decimal import Decimal
from src.config.config import PULLBACK_CONFIG  # Correcte import


# in fake_client.py
class FakeClient:
    def __init__(self, pairs=None):
        self.logger = logging.getLogger(__name__)
        self.pairs = pairs
        # wat je ook maar wilt doen met pairs
        ...

    def get_balance(self):
        self._check_rate_limit()
        self._increment_call()

        # Haal budget uit de PULLBACK_CONFIG
        yaml_budget = PULLBACK_CONFIG.get("initial_capital", 100)
        logging.debug(f"YAML Budget opgehaald: {yaml_budget}")

        # Casten naar Decimal
        budget_decimal = Decimal(str(yaml_budget))
        logging.debug(f"Budget Decimal: {budget_decimal}")

        return {"EUR": budget_decimal}

    def _check_rate_limit(self):
        # Implementatie van rate limit check
        pass

    def _increment_call(self):
        # Implementatie van call increment
        pass

    def place_order(self, side, symbol, amount, order_type=None):
        # paper trading => niets doen, alleen loggen
        self.logger.info(f"[Paper Trading] place_order({side}, {symbol}, amt={amount}, order_type={order_type})")
        # eventueel returnen van een fake response
        return {"status": "ok", "orderId": "paper123"}


# Gebruik van FakeClient (optioneel, voor testdoeleinden)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    client = FakeClient()
    balance = client.get_balance()
    print(f"Balans: €{balance}")
