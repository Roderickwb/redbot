import logging
import json
from decimal import Decimal
import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import RSIIndicator

class StrategyManager:
    def __init__(self, client, db_manager, config):
        """
        Beheert de strategie-logica voor het handelen.

        :param client: Een instantie van WebSocketClient voor interactie met de API.
        :param db_manager: Een instantie van DatabaseManager voor opslag en gegevensbeheer.
        :param config: Configuratieparameters voor de strategie.
        """
        self.client = client
        self.db_manager = db_manager
        self.config = config

        # Basisconfiguratie
        self.pairs = config["pairs"]
        self.partial_sell_threshold = config.get("partial_sell_threshold", 0.01)
        self.dip_rebuy_threshold = config.get("dip_rebuy_threshold", 0.01)
        self.core_ratio = config.get("core_ratio", 0.50)
        self.fallback_allocation_ratio = Decimal(str(config.get("fallback_allocation_ratio", 0.25)))
        self.first_profit_threshold = config.get("first_profit_threshold", 1.02)
        self.second_profit_threshold = config.get("second_profit_threshold", 1.05)

    def execute_strategy(self, symbol):
        """
        Voert de strategie uit voor een specifieke handelsmarkt.

        :param symbol: De markt (bijv. 'XRP-EUR').
        """
        try:
            # Haal actuele marktgegevens op
            candles = self.db_manager.fetch_data("candles", market=symbol, limit=300)
            if candles.empty:
                logging.warning(f"Geen candles beschikbaar voor {symbol}. Strategie overgeslagen.")
                return

            # Converteer naar DataFrame en bereken indicatoren
            df = pd.DataFrame(candles)
            df.columns = ['timestamp', 'market', 'interval', 'open', 'high', 'low', 'close', 'volume']
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
            macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()

            # Haal actuele balans op
            balance = self.client.get_balance()
            eur_balance = Decimal(balance.get('EUR', 0))
            coin_balance = Decimal(balance.get(symbol.split('-')[0], 0))

            # Fallback buy
            if coin_balance == 0:
                self._attempt_fallback_buy(symbol, eur_balance, df.iloc[-1]['close'])

            # Dip re-buy
            local_min = self._get_local_min(symbol)
            current_price = df.iloc[-1]['close']
            if local_min is None or current_price < local_min:
                self._save_local_min(symbol, current_price)
            else:
                ratio_down = (local_min - current_price) / local_min
                if ratio_down > self.dip_rebuy_threshold and eur_balance > 0:
                    self._attempt_dip_buy(symbol, eur_balance, current_price, ratio_down)

            # Partial sell
            if coin_balance > 0:
                self._attempt_partial_sell(symbol, coin_balance, current_price, local_min)

            # Log strategiebeslissingen naar dashboard
            decision = self._generate_strategy_decision(df, current_price, eur_balance, coin_balance)
            self._log_to_dashboard(symbol, decision)

        except Exception as e:
            logging.error(f"Fout bij uitvoeren van strategie voor {symbol}: {e}")

    def _attempt_fallback_buy(self, symbol, eur_balance, current_price):
        """Als de balans nul is, koop een fallback hoeveelheid."""
        fallback_eur = self.fallback_allocation_ratio * eur_balance
        if fallback_eur > 0:
            amount = fallback_eur / Decimal(current_price)
            self.client.place_order("buy", symbol, float(amount))
            logging.info(f"Fallback buy uitgevoerd voor {symbol} met EUR {fallback_eur}.")

    def _attempt_dip_buy(self, symbol, eur_balance, current_price, ratio_down):
        """Koop extra bij een significante prijsdaling."""
        buy_amount = eur_balance * Decimal("0.1")  # 10% als voorbeeld
        self.client.place_order("buy", symbol, float(buy_amount / current_price))
        logging.info(f"Dip-buy uitgevoerd voor {symbol} met {buy_amount} EUR bij ratio {ratio_down}.")

    def _attempt_partial_sell(self, symbol, coin_balance, current_price, local_min):
        """Verkoop gedeeltelijk als de prijs boven een drempel komt."""
        if local_min is not None:
            ratio_up = current_price / local_min
            if ratio_up > (1 + self.partial_sell_threshold):
                sell_amount = coin_balance * Decimal("0.15")  # Verkoop 15%
                self.client.place_order("sell", symbol, float(sell_amount))
                logging.info(f"Partial sell uitgevoerd voor {symbol}: {sell_amount} bij prijs {current_price}.")

    def _get_local_min(self, symbol):
        """Haal het lokale minimum op uit de database."""
        param = self.db_manager.fetch_data("parameters", market=symbol, limit=1)
        if not param.empty:
            return Decimal(param.iloc[0]['value'])
        return None

    def _save_local_min(self, symbol, value):
        """Sla een nieuw lokaal minimum op in de database."""
        self.db_manager.save_data("parameters", {
            'market': symbol,
            'param_name': 'local_min',
            'value': str(value)
        })
        logging.info(f"Lokaal minimum bijgewerkt voor {symbol}: {value}")

    def _generate_strategy_decision(self, df, current_price, eur_balance, coin_balance):
        """Genereer een strategieadvies gebaseerd op RSI en MACD."""
        last_row = df.iloc[-1]
        rsi = last_row['rsi']
        macd = last_row['macd']
        macd_signal = last_row['macd_signal']

        if rsi < 30 and macd > macd_signal:
            return "Koop"
        elif rsi > 70 and macd < macd_signal:
            return "Verkoop"
        else:
            return "Neutraal"

    def _log_to_dashboard(self, symbol, decision):
        """Log strategiebeslissingen naar het dashboard."""
        logging.info(f"Strategiebeslissing voor {symbol}: {decision}")
        # Hier kun je de beslissing uitbreiden naar een API of dashboard log


