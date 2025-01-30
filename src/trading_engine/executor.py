# src/trading_engine/executor.py

import os
import time
import logging
from threading import Thread

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


class Executor:
    def __init__(
        self,
        db_manager,
        use_websocket=True,
        paper_trading=True,
        api_key=None,
        api_secret=None,
        ### (NIEUW) extra params:
        use_kraken=True,
        kraken_paper=True,
        yaml_config=None
    ):
        """
        :param db_manager: instance van DatabaseManager
        :param use_websocket: True => LIVE marktdata via WebSocket (Bitvavo)
        :param paper_trading: True => fake (paper) orders, False => echte orders (Bitvavo)
        :param api_key: voor Bitvavo
        :param api_secret: voor Bitvavo
        :param use_kraken: of je óók Kraken-data wilt
        :param kraken_paper: of je Kraken in paper-mode (fake orders) wilt
        :param yaml_config: dict gelezen uit config.yaml (optioneel)
        """
        self.logger = setup_logger("executor", EXECUTOR_LOG_FILE, logging.DEBUG)
        self.logger.info("Executor init started.")

        self.db_manager = db_manager
        self.use_websocket = use_websocket
        self.paper_trading = paper_trading
        self.api_key = api_key
        self.api_secret = api_secret

        ### (NIEUW) Opslaan van use_kraken en kraken_paper
        self.use_kraken = use_kraken
        self.kraken_paper = kraken_paper

        ### (NIEUW) Lees YAML-sectie “kraken” (pairs, intervals) als beschikbaar
        self.yaml_config = yaml_config or {}
        kraken_cfg = self.yaml_config.get("kraken", {})
        # Voorbeeld:
        # kraken_cfg = {
        #   "pairs": ["BTC-EUR","ETH-EUR",...],
        #   "intervals_realtime": [15],
        #   "intervals_poll": [60,240,1440],
        #   "poll_interval_seconds": 300
        # }

        # =============== 1) Bitvavo data client ===============
        self.ws_client = None
        if self.use_websocket:
            ws_url = os.getenv("WS_URL", "wss://ws.bitvavo.com/v2/")
            self.ws_client = WebSocketClient(
                ws_url=ws_url,
                db_manager=self.db_manager,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            self.data_client = self.ws_client
            self.logger.info("WebSocketClient (Bitvavo) aangemaakt voor LIVE data.")
        else:
            self.data_client = None

        # =============== 2) Bitvavo orders (paper of real) ===============
        if self.paper_trading:
            self.logger.info("Paper Trading actief => FakeClient voor orders (Bitvavo).")
            self.order_client = FakeClient(pairs=PAIRS_CONFIG)
        else:
            self.logger.info("Real Trading actief => WebSocketClient (Bitvavo) voor orders.")
            self.order_client = self.ws_client

        # =============== 3) Kraken data client ===============
        self.kraken_data_client = None
        self.kraken_order_client = None

        if self.use_kraken:
            # Lees pairs & intervals uit config
            kraken_pairs = kraken_cfg.get("pairs", ["BTC-EUR","ETH-EUR"])
            intervals_realtime = kraken_cfg.get("intervals_realtime",[15])
            intervals_poll = kraken_cfg.get("intervals_poll",[60,240,1440])
            poll_interval_s = kraken_cfg.get("poll_interval_seconds",300)

            self.logger.info("Kraken MixedClient init => pairs=%s realtime=%s poll=%s poll_int=%ds",
                             kraken_pairs, intervals_realtime, intervals_poll, poll_interval_s)

            # Maak KrakenMixedClient
            self.kraken_data_client = KrakenMixedClient(
                db_manager=self.db_manager,
                pairs=kraken_pairs,
                intervals_realtime=intervals_realtime,
                intervals_poll=intervals_poll,
                poll_interval_seconds=poll_interval_s
            )

            # (NIEUW) orders op Kraken => paper of real?
            if self.kraken_paper:
                self.logger.info("Kraken in paper-mode => FakeClient (no real orders).")
                self.kraken_order_client = FakeClient(pairs=kraken_pairs)
            else:
                self.logger.info("Kraken real trading => hier zou je KrakenPrivateClient doen.")
                # self.kraken_order_client = KrakenPrivateClient(...)

        # =============== 4) Interne config ===============
        self.config = {
            "pairs": PAIRS_CONFIG,
            "partial_sell_threshold": 0.02,
            "dip_rebuy_threshold": 0.01,
            "core_ratio": 0.50,
            "fallback_allocation_ratio": 0.25,
            "first_profit_threshold": 1.02,
            "second_profit_threshold": 1.05
        }

        # =============== 5) ML-engine en strategieën ===============
        self.ml_engine = MLEngine(
            db_manager=self.db_manager,
            model_path="models/pullback_model.pkl"
        )
        # PullbackAccumulateStrategy (Bitvavo)
        self.pullback_strategy = PullbackAccumulateStrategy(
            data_client=self.data_client,   # (Bitvavo-livestream)
            order_client=self.order_client, # (Bitvavo) paper/real
            db_manager=self.db_manager,
            config_path="src/config/config.yaml"
        )
        # BreakoutStrategy (Bitvavo) - als je hem óók op Bitvavo wil laten lopen
        self.breakout_strategy = BreakoutStrategy(
            client=self.order_client,
            db_manager=self.db_manager,
            config_path="src/config/config.yaml"
        )
        self.pullback_strategy.set_ml_engine(self.ml_engine)

        # [CHANGED] - Aparte BreakoutStrategy voor Kraken:
        self.breakout_strategy_kraken = None
        if self.kraken_order_client:
            self.breakout_strategy_kraken = BreakoutStrategy(
                client=self.kraken_order_client,    # Fake of real
                db_manager=self.db_manager,
                config_path="src/config/config.yaml"
            )

        self.logger.info("Executor init completed.")
        self.logger.info(
            f"Executor => use_websocket={use_websocket}, paper_trading={paper_trading}, "
            f"use_kraken={use_kraken}, kraken_paper={kraken_paper}"
        )

    def run(self):
        """
        De hoofd-loop van de Executor: start evt. websocket (Bitvavo)
        + start evt. KrakenMixedClient, en loop elke 5s.
        """
        self.logger.info("Executor.run() gestart.")

        # Bitvavo start
        if self.use_websocket and self.ws_client:
            self.ws_client.start_websocket()
            self.logger.info("Bitvavo WebSocket client thread gestart (LIVE MODE).")

        # Kraken start
        if self.use_kraken and self.kraken_data_client:
            self.logger.info("KrakenMixedClient start => WS(15m) + poll(1h,4h,1d?).")
            self.kraken_data_client.start()

        try:
            loop_count = 0
            while True:
                loop_count += 1

                # A) Verwerk events uit queue (Bitvavo orders/fills)
                self._process_ws_events()

                # B) Pullback-strategy op Bitvavo:
                for symbol in self.config["pairs"]:
                    self.pullback_strategy.execute_strategy(symbol)

                # C) Breakout-strategy op Bitvavo (optioneel):
                for symbol in self.config["pairs"]:
                    self.breakout_strategy.execute_strategy(symbol)

                # [CHANGED] D) Breakout-strategy op Kraken-data (optioneel):
                if self.breakout_strategy_kraken:
                    # Voor nu even dezelfde symbol-lijst, of wat je wilt
                    for symbol in ["BTC-EUR", "ETH-EUR"]:
                        # Je moet in de BreakoutStrategy zélf zorgen dat hij
                        # db_manager.get_candlesticks(..., exchange="Kraken") gebruikt
                        self.breakout_strategy_kraken.execute_strategy(symbol)

                # E) DB-checks 1× per uur (720 * 5s = 3600s)
                if loop_count % 720 == 0:
                    self._hourly_db_checks()

                time.sleep(5)

        except KeyboardInterrupt:
            self.logger.info("Bot is handmatig gestopt via Ctrl+C (Executor).")
        except Exception as e:
            self.logger.exception(f"Fout in Executor run-lus: {e}")
        finally:
            self.logger.info("Executor shut down.")
            if self.ws_client:
                self.logger.info("Stop WebSocket (Bitvavo) in finally (Executor).")
                self.ws_client.stop_websocket()

            if self.kraken_data_client:
                self.logger.info("Stop KrakenMixedClient in finally => WS + poll.")
                self.kraken_data_client.stop()

    def _indicator_analysis_thread(self):
        while True:
            process_indicators(self.db_manager)
            time.sleep(60)

    def run_daily_tasks(self):
        self.logger.info("run_daily_tasks() gestart.")

        # 1) Train model
        ok_train = self.ml_engine.train_model()
        if ok_train:
            self.logger.info("ML-model training geslaagd.")

        # 2) Load model
        ok_load = self.ml_engine.load_model_from_db()
        if ok_load:
            self.logger.info("ML-model geladen => pullback_strategy kan ml_signal gebruiken")

        # 3) Scenario-tests
        result = self.ml_engine.run_scenario_tests()
        best_paramset = result.get("best_paramset")
        best_score = result.get("best_score", 0)
        self.logger.info(f"[Executor] best_paramset={best_paramset}, best_score={best_score:.2f}")

        # 4) Schrijf beste params naar config.yaml
        if best_paramset:
            self.ml_engine.write_best_params_to_yaml(best_paramset)

        self.logger.info("run_daily_tasks() voltooid.")

    def _hourly_db_checks(self):
        self.logger.info("[Hourly DB Checks] Starten van DB-checks...")

        try:
            candles_count = self.db_manager.get_table_count("candles")
            ticker_count = self.db_manager.get_table_count("ticker")
            bids_count = self.db_manager.get_table_count("orderbook_bids")
            asks_count = self.db_manager.get_table_count("orderbook_asks")

            self.logger.info(
                f"[Hourly DB Checks] candles={candles_count}, "
                f"ticker={ticker_count}, bids={bids_count}, asks={asks_count}"
            )
        except Exception as e:
            self.logger.error(f"[Hourly DB Checks] Fout bij DB-check: {e}")

        self.logger.info("[Hourly DB Checks] Einde.")

    def _process_ws_events(self):
        """
        Leeg de order_updates_queue van self.ws_client (Bitvavo) en verwerk
        order/fill events. Bij fill => update strategy.
        """
        if not self.ws_client:
            # Als self.ws_client=None => geen live data => skip
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
                self.logger.warning(f"Ongeldig event in queue: {event_type}")
