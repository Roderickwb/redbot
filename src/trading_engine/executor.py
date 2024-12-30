
import os
import time
import logging
import schedule
import traceback
from decimal import Decimal
import yaml

from client import WebSocketClient
from database_manager import DatabaseManager
from strategy_manager import StrategyManager
from ml_engine import MLEngine

class Executor:
    def __init__(self, config_path="config/config.yml"):
        self.config = self._load_config(config_path)

        # Instantieer DatabaseManager
        self.db_manager = DatabaseManager()

        # Instantieer WebSocketClient
        self.ws_client = WebSocketClient(
            ws_url=self.config['websocket']['url'],
            db_manager=self.db_manager,
            api_key=os.getenv("API_KEY"),
            api_secret=os.getenv("API_SECRET")
        )

        # Instantieer StrategyManager
        self.strategy_mgr = StrategyManager(
            client=self.ws_client,
            db_manager=self.db_manager,
            config=self.config
        )

        # Instantieer MLEngine
        self.ml_engine = MLEngine(self.db_manager)

        # Queue voor orderupdates
        self.order_updates_queue = self.ws_client

    def _load_config(self, path):
        """
        Laadt de configuratie vanuit een YAML-bestand.
        """
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def run(self):
        """
        Start de planning en uitvoer van strategieën.
        """
        pairs = self.config.get("pairs", {})

        # Plan de strategie-uitvoering voor elk paar
        for symbol in pairs.keys():
            schedule.every(5).minutes.do(
                lambda s=symbol: self.strategy_mgr.execute_strategy(s)
            )

        # Plan dagelijkse trainingstaken
        schedule.every().day.at("06:00").do(self.ml_engine.train_model)

        logging.info("[Executor] Main loop started.")
        try:
            while True:
                schedule.run_pending()
                self._process_ws_order_updates()
                time.sleep(0.2)
        except KeyboardInterrupt:
            logging.info("[Executor] Stopped by user.")
        except Exception as e:
            logging.error(f"[Executor] Unexpected error: {e}", exc_info=True)

    def _process_ws_order_updates(self):
        """
        Verwerkt orderupdates vanuit de WebSocket.
        """
        while not self.order_updates_queue.empty():
            trades_info = self.order_updates_queue.get()
            self.strategy_mgr.handle_order_updates(trades_info)

# Start de executor als dit script direct wordt uitgevoerd
if __name__ == "__main__":
    executor = Executor(config_path="config/config.yml")
    executor.run()