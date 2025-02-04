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

# [# AltcoinScanner ADDED]
from src.strategy.KrakenAltcoinScannerStrategy import KrakenAltcoinScannerStrategy

# Kraken client
from src.exchange.kraken.kraken_mixed_client import KrakenMixedClient

import requests  # <-- voor de except-block


# [# NEW] -> import of je eigen helper (als je is_candle_closed in indicators hebt)
# from src.indicator_analysis.indicators import is_candle_closed
def is_candle_closed(candle_timestamp_ms: int, interval_str: str) -> bool:
    """
    Hulpmethode om te checken of een candle (start=candle_timestamp_ms)
    definitief is afgesloten. Eenvoudige parse van interval_str ('5m','15m','4h', etc.).
    """
    now_ms = int(time.time() * 1000)
    unit = interval_str[-1]  # 'm','h','d' ...
    val = int(interval_str[:-1])
    if unit == 'm':
        duration_ms = val * 60_000
    elif unit == 'h':
        duration_ms = val * 60 * 60_000
    elif unit == 'd':
        duration_ms = val * 24 * 60 * 60_000
    else:
        duration_ms = 0
    candle_end = candle_timestamp_ms + duration_ms
    return now_ms >= candle_end


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
        :param api_key:         Bitvavo API-key
        :param api_secret:      Bitvavo secret
        :param use_kraken:      True => ook Kraken-data
        :param kraken_paper:    True => fake (paper) orders op Kraken
        :param yaml_config:     dict uit config.yaml
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
            self.logger.info("[Executor] Real Trading => WSClient (Bitvavo).")
            self.order_client = self.ws_client

        # -----------------------------
        # KRAKEN: data + orders
        # -----------------------------
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
                self.logger.info("[Executor] Kraken REAL => hier evt. echte private client.")
                # self.kraken_order_client = KrakenPrivateClient(...)

        # -----------------------------
        # EXECUTOR-config
        # -----------------------------
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

        # -----------------------------
        # ML-engine + strategieën
        # -----------------------------
        self.ml_engine = MLEngine(
            db_manager=self.db_manager,
            model_path="models/pullback_model.pkl"
        )

        # A) Pullback (Bitvavo)
        self.pullback_strategy = PullbackAccumulateStrategy(
            data_client=self.data_client,
            order_client=self.order_client,
            db_manager=self.db_manager,
            config_path="src/config/config.yaml"
        )
        self.pullback_strategy.set_ml_engine(self.ml_engine)

        # B) Breakout (Bitvavo) - default staat 'aan' in code
        self.breakout_strategy = BreakoutStrategy(
            client=self.order_client,
            db_manager=self.db_manager,
            config_path="src/config/config.yaml"
        )
        # self.breakout_strategy.set_ml_engine(self.ml_engine)

        # C) Breakout (Kraken)
        self.breakout_strategy_kraken = None
        if self.kraken_order_client:
            self.breakout_strategy_kraken = BreakoutStrategy(
                client=self.kraken_order_client,    # Fake of real
                db_manager=self.db_manager,
                config_path="src/config/config.yaml"
            )

        # [# AltcoinScanner ADDED] --------------------------------
        self.altcoin_scanner_kraken = None
        if self.use_kraken and self.kraken_data_client:
            # We lezen 'altcoin_scanner_strategy' uit je config
            alt_cfg = self.yaml_config.get("altcoin_scanner_strategy", {})
            self.logger.info("[Executor] init KrakenAltcoinScannerStrategy => alt_cfg=%s", alt_cfg)

            self.altcoin_scanner_kraken = KrakenAltcoinScannerStrategy(
                kraken_client=self.kraken_data_client,  # Jouw data client
                db_manager=self.db_manager,
                config=alt_cfg,
                logger=None  # of self.logger als je wilt
            )
        # ---------------------------------------------------------

        self.logger.info("Executor init completed.")
        self.logger.info(
            f"[Executor] use_websocket={self.use_websocket}, paper_trading={self.paper_trading}, "
            f"use_kraken={self.use_kraken}, kraken_paper={self.kraken_paper}"
        )

        # [# NEW] Dictionary om per (table, symbol, interval) de laatst verwerkte candle-ts te onthouden
        self.last_closed_ts = {}

    def run(self):
        """
        Hoofd-loop:
          - Start Bitvavo-WS
          - Start KrakenMixedClient
          - Periodiek strategies callen (Bitvavo + Kraken)
          - Event-loops
        """
        self.logger.info("[Executor] run() gestart.")

        # A) Bitvavo-WS
        if self.use_websocket and self.ws_client:
            self.ws_client.start_websocket()
            self.logger.info("[Executor] Bitvavo WS client thread gestart.")

        # B) Start Kraken (WS + poll)
        if self.use_kraken and self.kraken_data_client:
            self.logger.info("[Executor] KrakenMixedClient => start()")
            self.kraken_data_client.start()

        loop_count = 0
        try:
            while True:
                loop_count += 1

                # 1) Verwerk Bitvavo WS events
                self._process_ws_events()

                # [# CHANGED] i.p.v. direct de strategies, check of er nieuwe candles "closed" zijn
                for symbol in self.bitvavo_pairs:
                    # Pullback => bijv. "candles_bitvavo", interval="5m"
                    if self._has_new_closed_candle("candles_bitvavo", symbol, "5m"):
                        self.pullback_strategy.execute_strategy(symbol)

                    # Breakout => "candles_bitvavo", interval="15m"
                    if self._has_new_closed_candle("candles_bitvavo", symbol, "15m"):
                        self.breakout_strategy.execute_strategy(symbol)

                # 2) Kraken strategies
                if self.kraken_data_client:
                    # Breakout => 15m in "candles_kraken"
                    if self.breakout_strategy_kraken:
                        for symbol in self.kraken_data_client.pairs:
                            if self._has_new_closed_candle("candles_kraken", symbol, "15m"):
                                self.breakout_strategy_kraken.execute_strategy(symbol)

                    # AltcoinScanner => ook 15m in "candles_kraken"
                    if self.altcoin_scanner_kraken:
                        # Let op: je kunt of per symbol checken, of 1 check en altcoin_scanner_kraken
                        # scant intern. Hier is de minimal approach: if any symbol has new candle => run once:
                        for symbol in self.kraken_data_client.pairs:
                            if self._has_new_closed_candle("candles_kraken", symbol, "15m"):
                                self.altcoin_scanner_kraken.execute_strategy()
                                break

                # 4) DB-check 1× per 10 min
                if loop_count % 120 == 0:
                    self._hourly_db_checks()

                time.sleep(5)

        except KeyboardInterrupt:
            self.logger.info("[Executor] Bot handmatig gestopt (Ctrl+C).")

        except requests.exceptions.ConnectionError as ce:
            # Hier vangen we specifically ConnectionErrors op en loggen, en we kunnen beslissen door te gaan:
            self.logger.error(f"[Executor] Netwerkfout => {ce}, wachten en doorgaan")
            time.sleep(5)
        except Exception as e:
            self.logger.exception(f"[Executor] Fout in run-lus: {e}")

        finally:
            self.logger.info("[Executor] shutting down.")

            # 1) Stop Bitvavo
            if self.ws_client:
                self.logger.info("[Executor] Stop Bitvavo WS.")
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

    # [# NEW] Hulpmethode die checkt of er een nieuwe candle in table_name staat voor (symbol, interval)
    #         én of die candle closed is. Return True als strategy nog niet heeft verwerkt.
    def _has_new_closed_candle(self, table_name, symbol, interval) -> bool:
        # Haal 1 nieuwste candle op
        df = self.db_manager.fetch_data(table_name, limit=1, market=symbol, interval=interval)
        if df.empty:
            return False
        newest_ts = df["timestamp"].iloc[0]

        # Als candle niet closed => skip
        if not is_candle_closed(newest_ts, interval):
            return False

        # Key in self.last_closed_ts
        key = (table_name, symbol, interval)
        last_seen_ts = self.last_closed_ts.get(key, 0)

        if newest_ts > last_seen_ts:
            self.last_closed_ts[key] = newest_ts
            return True
        else:
            return False
