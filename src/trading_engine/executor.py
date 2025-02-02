# ============================================================
# src/trading_engine/executor.py
# ============================================================

import os
import time
import logging
from threading import Thread
from datetime import datetime, timedelta, timezone

from src.logger.logger import setup_logger
from src.my_websocket.client import WebSocketClient
from src.my_websocket.fake_client import FakeClient

# Strategy modules
from src.strategy.pullback_accumulate_strategy import PullbackAccumulateStrategy
from src.strategy.breakout_strategy import BreakoutStrategy

from src.ml_engine.ml_engine import MLEngine
from src.indicator_analysis.indicators import process_indicators
from src.config.config import EXECUTOR_LOG_FILE, PAIRS_CONFIG
from src.exchange.kraken.kraken_mixed_client import KrakenMixedClient

import requests  # <-- voor de except-block

class Executor:
    def __init__(
        self,
        db_manager,
        use_websocket=True,
        paper_trading=True,
        api_key=None,
        api_secret=None,
        use_kraken=True,
        kraken_paper=True,
        yaml_config=None
    ):
        """
        :param db_manager:      DatabaseManager
        :param use_websocket:   True => LIVE marktdata (Bitvavo) via WebSocket
        :param paper_trading:   True => fake (paper) orders op Bitvavo
        :param api_key:         Bitvavo API-key (optioneel)
        :param api_secret:      Bitvavo secret
        :param use_kraken:      True => ook Kraken-data
        :param kraken_paper:    True => fake (paper) orders op Kraken
        :param yaml_config:     dict uit config.yaml (of leeg)
        """
        self.logger = setup_logger("executor", EXECUTOR_LOG_FILE, logging.DEBUG)
        self.logger.info("Executor init started.")

        # ------------------------------
        # A) Bewaar constructor-params
        # ------------------------------
        self.db_manager = db_manager
        self.use_websocket = use_websocket
        self.paper_trading = paper_trading
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_kraken = use_kraken
        self.kraken_paper = kraken_paper

        # Lees YAML
        self.yaml_config = yaml_config or {}
        bitvavo_cfg = self.yaml_config.get("bitvavo", {})
        kraken_cfg  = self.yaml_config.get("kraken", {})

        # ------------------------------
        # B) Bitvavo – data + orders
        # ------------------------------
        # 1) Bitvavo pairs uit YAML of fallback
        fallback_bitvavo_pairs = PAIRS_CONFIG
        self.bitvavo_pairs = bitvavo_cfg.get("pairs", fallback_bitvavo_pairs)
        self.logger.info(f"[Executor] bitvavo_pairs={self.bitvavo_pairs}")

        self.ws_client = None
        if self.use_websocket:
            ws_url = bitvavo_cfg.get("websocket_url", os.getenv("WS_URL", "wss://ws.bitvavo.com/v2/"))
            self.ws_client = WebSocketClient(
                ws_url=ws_url,
                db_manager=self.db_manager,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            self.logger.info(f"[Executor] Bitvavo WebSocket aangemaakt op {ws_url}")
            self.data_client = self.ws_client
        else:
            self.data_client = None

        # 2) Bitvavo orders
        if self.paper_trading:
            self.logger.info("[Executor] Paper Trading actief => FakeClient (Bitvavo).")
            self.order_client = FakeClient(pairs=self.bitvavo_pairs)
        else:
            self.logger.info("[Executor] Real Trading actief => WebSocketClient (Bitvavo).")
            self.order_client = self.ws_client

        # ------------------------------
        # C) Kraken – data + orders
        # ------------------------------
        self.kraken_data_client = None
        self.kraken_order_client = None
        if self.use_kraken:
            self.logger.info("[Executor] Kraken => init from kraken_cfg.")

            # 1) Bepaal of we private WS willen (als niet paper)
            #    Je kunt zelf beslissen. Hier doen we: als kraken_paper=False => use_private_ws=True
            #    Of je maakt er gewoon "use_private_ws=True" van, net wat je wilt
            use_private_ws = (not self.kraken_paper)  # = True als je ECHT wil traden

            self.kraken_data_client = KrakenMixedClient(
                db_manager=self.db_manager,
                kraken_cfg=kraken_cfg,
                use_private_ws=use_private_ws
            )

            # Orders op Kraken => paper of real?
            if self.kraken_paper:
                self.logger.info("[Executor] Kraken PAPER => FakeClient.")
                kraken_pairs = kraken_cfg.get("pairs", [])
                self.kraken_order_client = FakeClient(pairs=kraken_pairs)
            else:
                self.logger.info("[Executor] Kraken REAL => hier zou je private client gebruiken of direct ccxt.")
                # self.kraken_order_client = KrakenPrivateClient(...)

        # ------------------------------
        # D) Overige executor-config
        # ------------------------------
        default_executor_cfg = {
            "pairs": self.bitvavo_pairs,
            "partial_sell_threshold": 0.02,
            "dip_rebuy_threshold": 0.01,
            "core_ratio": 0.50,
            "fallback_allocation_ratio": 0.25,
            "first_profit_threshold": 1.02,
            "second_profit_threshold": 1.05
        }
        self.config = self.yaml_config.get("executor_config", default_executor_cfg)

        # ------------------------------
        # E) ML-engine + strategieën
        # ------------------------------
        self.ml_engine = MLEngine(
            db_manager=self.db_manager,
            model_path="models/pullback_model.pkl"
        )

        # Pullback voor Bitvavo
        self.pullback_strategy = PullbackAccumulateStrategy(
            data_client=self.data_client,
            order_client=self.order_client,
            db_manager=self.db_manager,
            config_path="src/config/config.yaml"
        )
        self.pullback_strategy.set_ml_engine(self.ml_engine)

        # Breakout voor Bitvavo (MAAR we gaan 'm hier UITZETTEN)
        # [CHANGED] - We skip the call in run(), so this strategy is effectively disabled on Bitvavo
        self.breakout_strategy = BreakoutStrategy(
            client=self.order_client,
            db_manager=self.db_manager,
            config_path="src/config/config.yaml"
        )
        # self.breakout_strategy.set_ml_engine(self.ml_engine)

        # BreakoutStrategy voor Kraken
        self.breakout_strategy_kraken = None
        if self.kraken_order_client:
            self.breakout_strategy_kraken = BreakoutStrategy(
                client=self.kraken_order_client,    # Fake of real
                db_manager=self.db_manager,
                config_path="src/config/config.yaml"
            )

        self.logger.info("Executor init completed.")
        self.logger.info(
            f"[Executor] use_websocket={self.use_websocket}, paper_trading={self.paper_trading}, "
            f"use_kraken={self.use_kraken}, kraken_paper={self.kraken_paper}"
        )

    def run(self):
        """
        De hoofd-loop van de Executor.
        - start Bitvavo-websocket (als nodig)
        - start KrakenMixedClient (als nodig)
        - elke 5s: run strategieën, verwerk events, etc.
        """
        self.logger.info("[Executor] run() gestart.")

        # A) Start Bitvavo WS
        if self.use_websocket and self.ws_client:
            self.ws_client.start_websocket()
            self.logger.info("[Executor] Bitvavo WS client thread gestart.")

        # B) Start Kraken (WS + poll)
        if self.use_kraken and self.kraken_data_client:
            self.logger.info("[Executor] KrakenMixedClient => start()")
            self.kraken_data_client.start()

        try:
            loop_count = 0
            while True:
                loop_count += 1

                # A) verwerk Bitvavo-WS events
                self._process_ws_events()

                # B) Pullback => Bitvavo
                for symbol in self.bitvavo_pairs:
                    self.pullback_strategy.execute_strategy(symbol)

                # C) Breakout => Bitvavo (uitgezet)
                # for symbol in self.bitvavo_pairs:
                #     self.breakout_strategy.execute_strategy(symbol)

                # C) Breakout => Kraken
                if self.breakout_strategy_kraken and self.kraken_data_client:
                    for symbol in self.kraken_data_client.pairs:
                        self.breakout_strategy_kraken.execute_strategy(symbol)

                # D) DB-check 1× per uur (120 * 5s = 600s = 10 min, pas aan naar wens)
                if loop_count % 120 == 0:
                    self._hourly_db_checks()

                time.sleep(5)

        except KeyboardInterrupt:
            self.logger.info("[Executor] Bot handmatig gestopt (Ctrl+C).")

        except requests.exceptions.ConnectionError as ce:
            # Hier vangen we specifically ConnectionErrors op en loggen, en we kunnen beslissen door te gaan:
            self.logger.error(f"[Executor] Netwerkfout => {ce}, wachten en doorgaan")
            time.sleep(5)
            # Optional: je kunt hier `run()` herstarten of “continue”:
            # continue
            # of we breken toch:
            self.logger.error("[Executor] We breken nu run-lus.")
        except Exception as e:
            self.logger.exception(f"[Executor] Fout in run-lus: {e}")

        finally:
            self.logger.info("[Executor] shutting down.")

            # 1) Stop Bitvavo
            if self.ws_client:
                self.logger.info("[Executor] Stop Bitvavo websocket.")
                self.ws_client.stop_websocket()

            # 2) Stop Kraken
            if self.kraken_data_client:
                self.logger.info("[Executor] Stop KrakenMixedClient.")
                self.kraken_data_client.stop()

            self.logger.info("[Executor] alles gestopt.")

    def run_daily_tasks(self):
        self.logger.info("[Executor] run_daily_tasks() gestart.")

        # 1) Train model
        ok_train = self.ml_engine.train_model()
        if ok_train:
            self.logger.info("[Executor] ML-model training geslaagd.")

        # 2) Load model
        ok_load = self.ml_engine.load_model_from_db()
        if ok_load:
            self.logger.info("[Executor] ML-model geladen => pullback_strategy kan ml_signal gebruiken")

        # 3) Scenario-tests
        result = self.ml_engine.run_scenario_tests()
        best_paramset = result.get("best_paramset")
        best_score = result.get("best_score", 0)
        self.logger.info(f"[Executor] best_paramset={best_paramset}, best_score={best_score:.2f}")

        # 4) Schrijf beste params naar config.yaml
        if best_paramset:
            self.ml_engine.write_best_params_to_yaml(best_paramset)

        self.logger.info("[Executor] run_daily_tasks() voltooid.")

    def _hourly_db_checks(self):
        self.logger.info("[Executor] _hourly_db_checks => start.")
        try:
            c = self.db_manager.get_table_count("candles")
            t = self.db_manager.get_table_count("ticker")
            b = self.db_manager.get_table_count("orderbook_bids")
            a = self.db_manager.get_table_count("orderbook_asks")
            self.logger.info(f"[Executor] candles={c}, ticker={t}, bids={b}, asks={a}")
        except Exception as e:
            self.logger.error(f"[Executor] fout in _hourly_db_checks: {e}")
        self.logger.info("[Executor] _hourly_db_checks => done.")

    def _process_ws_events(self):
        if not self.ws_client:
            return
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
                self.pullback_strategy.update_position_with_fill(event_data)
            else:
                self.logger.warning(f"[Executor] Onbekend event in queue: {event_type}")

    def _indicator_analysis_thread(self):
        while True:
            process_indicators(self.db_manager)
            time.sleep(60)
