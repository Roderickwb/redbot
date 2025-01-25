# src/trading_engine/executor.py

import os
import time
import logging
from threading import Thread


from src.logger.logger import setup_logger
from src.my_websocket.client import WebSocketClient
from src.my_websocket.fake_client import FakeClient
from src.strategy.pullback_accumulate_strategy import PullbackAccumulateStrategy
from src.ml_engine.ml_engine import MLEngine
from src.indicator_analysis.indicators import process_indicators
from src.config.config import EXECUTOR_LOG_FILE, PAIRS_CONFIG


class Executor:
    def __init__(self, db_manager, use_websocket=True, paper_trading=True):
        """
        :param db_manager: instance van DatabaseManager
        :param use_websocket: True => Live marktdata via WebSocket
        :param paper_trading: True => fake (paper) orders, False => real orders
        """
        self.logger = setup_logger("executor", EXECUTOR_LOG_FILE, logging.DEBUG)
        self.logger.info("Executor init started.")

        self.db_manager = db_manager
        self.use_websocket = use_websocket
        self.paper_trading = paper_trading

        # 1) Optioneel: maak WebSocketClient voor LIVE data
        self.ws_client = None
        if self.use_websocket:
            ws_url = os.getenv("WS_URL", "wss://ws.bitvavo.com/v2/")
            api_key = os.getenv("API_KEY")
            api_secret = os.getenv("API_SECRET")
            self.ws_client = WebSocketClient(
                ws_url=ws_url,
                db_manager=self.db_manager,
                api_key=api_key,
                api_secret=api_secret
            )
            # We hebben voor "live koersen" ALTIJD de ws_client
            self.data_client = self.ws_client
            self.logger.info("WebSocketClient aangemaakt voor LIVE data.")

        # 2) Kies of we fake of real orders doen
        if self.paper_trading:
            self.logger.info("Paper Trading actief => FakeClient voor orders.")
            self.order_client = FakeClient(
                pairs=PAIRS_CONFIG
            )
        else:
            # Echte orders, en live data
            self.logger.info("Real Trading actief => WebSocketClient voor orders.")
            self.order_client = self.ws_client

        # 3) Interne config
        self.config = {
            "pairs": PAIRS_CONFIG,  # ← Vervanging
            # Overige keys
            "partial_sell_threshold": 0.02,
            "dip_rebuy_threshold": 0.01,
            "core_ratio": 0.50,
            "fallback_allocation_ratio": 0.25,
            "first_profit_threshold": 1.02,
            "second_profit_threshold": 1.05
        }

        # 4) ML-engine

        self.ml_engine = MLEngine(self.db_manager, model_path="models/pullback_model.pkl")
        self.pullback_strategy = PullbackAccumulateStrategy(
            data_client=self.ws_client,  # voor koersen
            order_client=self.order_client,  # FakeClient of WebSocketClient
            db_manager=self.db_manager,
            config_path="src/config/config.yaml"
        )
        # Strategy kan ook ML-engine gebruiken om ml_signal=...
        self.pullback_strategy.set_ml_engine(self.ml_engine)

        self.logger.info("Executor init completed.")

    def run(self):
        self.logger.info("Executor.run() gestart.")

        # Start evt. websocket
        if self.use_websocket and self.ws_client:
            self.ws_client.start_websocket()
            self.logger.info("WebSocket client thread gestart (LIVE MODE).")

        try:
            loop_count = 0

            while True:
                loop_count += 1

                # === A) Verwerk alle events uit de queue
                self._process_ws_events()

                # === B) Strategie op elk symbool ===
                for symbol in self.config["pairs"]:
                    self.pullback_strategy.execute_strategy(symbol)

                # === C) DB-checks 1× per uur (720 * 5s = 3600s) ===
                if loop_count % 720 == 0:
                    self._hourly_db_checks()  # zie verderop

                # === D) Slapen 5s, dan volgende loop ===
                time.sleep(5)

        except KeyboardInterrupt:
            self.logger.info("Bot is handmatig gestopt via Ctrl+C (Executor).")
        except Exception as e:
            self.logger.exception(f"Fout in Executor run-lus: {e}")
        finally:
            self.logger.info("Executor shut down.")
            if self.ws_client:
                self.logger.info("Stop WebSocket in finally (Executor).")
                self.ws_client.stop_websocket()

    def _indicator_analysis_thread(self):
        """
        (Optioneel) Elke minuut indicatoren berekenen in aparte thread.
        """
        while True:
            process_indicators(self.db_manager)
            time.sleep(60)

    def run_daily_tasks(self):
        """
        1) ML-training
        2) Model laden
        3) Scenario-tests
        4) YAML overschrijven
        """
        self.logger.info("run_daily_tasks() gestart.")

        # 1) train model
        ok_train = self.ml_engine.train_model()
        if ok_train:
            self.logger.info("ML-model training geslaagd.")

        # 2) load model
        ok_load = self.ml_engine.load_model_from_db()
        if ok_load:
            self.logger.info("ML-model geladen => pullback_strategy kan ml_signal gebruiken")

        # 3) scenario-tests
        result = self.ml_engine.run_scenario_tests()
        best_paramset = result.get("best_paramset")
        best_score = result.get("best_score", 0)
        self.logger.info(f"[Executor] best_paramset={best_paramset}, best_score={best_score:.2f}")

        # 4) overschrijven in config.yaml
        if best_paramset:
            self.ml_engine.write_best_params_to_yaml(best_paramset)

        self.logger.info("run_daily_tasks() voltooid.")

    def _hourly_db_checks(self):
        """
        Draait 1x per ~uur in de run()-loop (loop_count % 720 == 0).
        Hier kun je database-tellingen doen, opschonen, vacuum, etc.
        """
        self.logger.info("[Hourly DB Checks] Starten van DB-checks...")

        try:
            candles_count = self.db_manager.get_table_count("candles")
            ticker_count = self.db_manager.get_table_count("ticker")
            bids_count = self.db_manager.get_table_count("orderbook_bids")
            asks_count = self.db_manager.get_table_count("orderbook_asks")

            self.logger.info(
                f"[Hourly DB Checks] candles={candles_count}, ticker={ticker_count}, "
                f"bids={bids_count}, asks={asks_count}"
            )

            # Voorbeeld: Prunen
            # self.db_manager.prune_old_candles(days=30)
            # self.logger.info("[Hourly DB Checks] Oude candles > 30d verwijderd.")

            # Voorbeeld: vacuum
            # self.db_manager.connection.execute("VACUUM")
            # self.logger.info("[Hourly DB Checks] VACUUM uitgevoerd.")

        except Exception as e:
            self.logger.error(f"[Hourly DB Checks] Fout bij DB-check: {e}")

        self.logger.info("[Hourly DB Checks] Einde.")

    def _process_ws_events(self):
        import queue
        while True:
            try:
                event_data = self.ws_client.order_updates_queue.get(block=False)
            except queue.Empty:
                break

            event_type = event_data.get("event")
            if event_type == "order":
                self.ws_client.handle_order_update(event_data)
            elif event_type == "fill":
                self.ws_client.handle_fill_update(event_data)
                # === HIER roep je partial fill update in de strategy:
                self.pullback_strategy.update_position_with_fill(event_data)
            else:
                self.logger.warning(f"Ongeldig event in queue: {event_type}")


