# ============================================================
# src/trading_engine/executor.py
# ============================================================

import os
import threading
import time
import logging
from datetime import datetime

from src.logger.logger import setup_logger
from src.my_websocket.client import WebSocketClient
from src.my_websocket.fake_client import FakeClient

# Strategy modules
from src.strategy.pullback_accumulate_strategy import PullbackAccumulateStrategy
from src.strategy.trend_strategy_4h import TrendStrategy4H
from src.strategy.breakout_strategy import BreakoutStrategy

from src.ml_engine.ml_engine import MLEngine
from src.indicator_analysis.indicators import process_indicators
from src.config.config import EXECUTOR_LOG_FILE, PAIRS_CONFIG

# [# AltcoinScanner ADDED]
from src.strategy.KrakenAltcoinScannerStrategy import KrakenAltcoinScannerStrategy

# Kraken client
from src.exchange.kraken.kraken_mixed_client import KrakenMixedClient

import requests  # <-- voor de except-block


def is_candle_closed(candle_timestamp_ms: int, interval_str: str) -> bool:
    now_ms = int(time.time() * 1000)
    return now_ms >= candle_timestamp_ms

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

        self.keep_running = True  # flag om threads te laten stoppen bij shutdown

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
        kraken_cfg = self.yaml_config.get("kraken", {})

        # ------------------------------
        # B) Bitvavo – data + orders
        # ------------------------------
        # 1) Bitvavo pairs uit YAML of fallback
        fallback_bitvavo_pairs = PAIRS_CONFIG
        self.bitvavo_pairs = bitvavo_cfg.get("pairs", fallback_bitvavo_pairs)
        self.logger.info(f"[Executor] bitvavo_pairs={self.bitvavo_pairs}")

        # == BITVAVO init ==
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

            # Als kraken_paper=False => use_private_ws=True
            use_private_ws = (not self.kraken_paper)

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
                self.kraken_order_client = self.kraken_data_client

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

        # ML-engine
        self.ml_engine = MLEngine(
            db_manager=self.db_manager,
            model_path="models/pullback_model.pkl"
        )

        # == PULLBACK (Bitvavo) ==
        # --------------------------------------------------------
        # UITGEKOMMENTEERD: We laten Pullback NIET op Bitvavo lopen
        """
        self.pullback_strategy_bitvavo = PullbackAccumulateStrategy(
            data_client=self.data_client,
            order_client=self.order_client,
            db_manager=self.db_manager,
            config_path="src/config/config.yaml"
        )
        self.pullback_strategy_bitvavo.set_ml_engine(self.ml_engine)
        """

        # == BREAKOUT (Bitvavo) ==
        # --------------------------------------------------------
        # UITGEKOMMENTEERD: We laten Breakout NIET op Bitvavo lopen
        """
        self.breakout_strategy_bitvavo = BreakoutStrategy(
            client=self.order_client,
            db_manager=self.db_manager,
            config_path="src/config/config.yaml"
        )
        """

        # BREAKOUT (Kraken)
        self.breakout_strategy_kraken = None
        #if self.kraken_order_client:
        #    self.breakout_strategy_kraken = BreakoutStrategy(
        #        client=self.kraken_order_client,
        #        db_manager=self.db_manager,
        #        config_path="src/config/config.yaml"
        #    )

        # ALTCOIN SCANNER (Kraken)
        self.altcoin_scanner_kraken = None
        if self.use_kraken and self.kraken_data_client:
            # We lezen 'altcoin_scanner_strategy' uit je config
            alt_cfg = self.yaml_config.get("altcoin_scanner_strategy", {})
            self.logger.info("[Executor] init KrakenAltcoinScannerStrategy => alt_cfg=%s", alt_cfg)
            self.altcoin_scanner_kraken = KrakenAltcoinScannerStrategy(
                kraken_client=self.kraken_data_client,
                db_manager=self.db_manager,
                config=alt_cfg,
                logger=None
            )

        # PULLBACK (Kraken) (15m by default)
        self.pullback_strategy_kraken = None
        if self.kraken_order_client:
            self.pullback_strategy_kraken = PullbackAccumulateStrategy(
                data_client=self.kraken_data_client,
                order_client=self.kraken_order_client,
                db_manager=self.db_manager,
                config_path="src/config/config.yaml"
            )
            self.pullback_strategy_kraken.set_ml_engine(self.ml_engine)

        # TREND (Kraken) — watch-only skeleton
        self.trend_strategy_kraken = None
        if self.kraken_order_client:
            self.trend_strategy_kraken = TrendStrategy4H(
                data_client=self.kraken_data_client,
                order_client=self.kraken_order_client,
                db_manager=self.db_manager,
                config_path="src/config/config.yaml"
            )

        self.logger.info("Executor init completed.")
        self.logger.info(
            f"[Executor] use_websocket={self.use_websocket}, paper_trading={self.paper_trading}, "
            f"use_kraken={self.use_kraken}, kraken_paper={self.kraken_paper}"
        )

        # Dictionary om per (table, symbol, interval) de laatst verwerkte candle-ts te onthouden
        self.last_closed_ts = {}

    # =============================================================================
    # ORIGINELE run() => commentaarblok, zodat ALLE code + commentaar bewaard blijft
    # =============================================================================
    """
    def run(self):
        \"\"\"
        Hoofd-loop:
          - Start Bitvavo-WS
          - Start KrakenMixedClient
          - Periodiek strategies callen (Bitvavo + Kraken)
          - Event-loops
        \"\"\"
        self.logger.info("[Executor] run() gestart.")

        # A) Bitvavo-WS
        if self.use_websocket and self.ws_client:
            self.ws_client.start_websocket()
            self.logger.info("[Executor] Bitvavo WS client thread gestart.")

        # B) Start Kraken (WS + poll)
        if self.use_kraken and self.kraken_data_client:
            self.logger.info("[Executor] KrakenMixedClient => start()")
            self.kraken_data_client.start()

        self.logger.debug("[Executor] run() => about to enter main loop")

        loop_count = 0
        try:
            while True:
                loop_count += 1

                # 1) Verwerk evt. Bitvavo WS events
                self._process_ws_events()

                # 2) Check of er een "new closed 15m candle" is voor minimaal 1 coin => run pullback
                new_15m_found = False
                if self.kraken_data_client:
                    for symbol in self.kraken_data_client.pairs:
                        if self._has_new_closed_candle("candles_kraken", symbol, "15m"):
                            new_15m_found = True
                    if new_15m_found:
                        self.logger.info("[Executor] Found new closed 15m => run 2-pass Pullback now.")
                        self.run_once_pullback_15m()

                # 3) Breakout => old _has_new_closed_candle approach
                if self.breakout_strategy_kraken and self.kraken_data_client:
                    for symbol in self.kraken_data_client.pairs:
                        if self._has_new_closed_candle("candles_kraken", symbol, "15m"):
                            self.logger.debug(f"[Executor] NEW CLOSED 15m => breakout_kraken({symbol}).")
                            self.breakout_strategy_kraken.execute_strategy(symbol)

                # 4) AltcoinScanner => unchanged
                if self.altcoin_scanner_kraken and self.kraken_data_client:
                    any_new_candle = False
                    for symbol in self.kraken_data_client.pairs:
                        if self._has_new_closed_candle("candles_kraken", symbol, "15m"):
                            self.logger.info(f"[Executor] FOUND new closed 15m candle for {symbol} => alt_scanner_kraken!")
                            any_new_candle = True
                            break
                    if any_new_candle:
                        self.logger.info("[Executor] Call altcoin_scanner_kraken.execute_strategy()")
                        self.altcoin_scanner_kraken.execute_strategy()
                    else:
                        self.logger.debug("[Executor] No new closed 15m candle => skip altcoin_scanner_kraken")

                # 5) Intra-candle checks => HIER STAAN JE LOGGERS NOG STEEDS
                if self.altcoin_scanner_kraken:
                    self.logger.debug("[Executor] altcoin_scanner_kraken.manage_intra_candle_exits() called.")
                    self.altcoin_scanner_kraken.manage_intra_candle_exits()

                if self.breakout_strategy_kraken:
                    self.logger.debug("[Executor] breakout_strategy_kraken.manage_intra_candle_exits() called.")
                    self.breakout_strategy_kraken.manage_intra_candle_exits()

                if self.pullback_strategy_kraken:
                    self.logger.info("[Executor] pullback_strategy_kraken.manage_intra_candle_exits() called.")
                    self.pullback_strategy_kraken.manage_intra_candle_exits()

                # 6) Elk uur DB-check
                if loop_count % 120 == 0:
                    self._hourly_db_checks()

                time.sleep(5)

        except KeyboardInterrupt:
            self.logger.info("[Executor] Bot handmatig gestopt (Ctrl+C).")
        except requests.exceptions.ConnectionError as ce:
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
    """

    def _intra_candle_exits_loop(self):
        """
        Deze functie draait in een aparte thread en roept elke 5 seconden
        de 'manage_intra_candle_exits()' van je strategies aan.
        """
        while self.keep_running:
            try:
                # 1) Pullback exit-check
                if self.pullback_strategy_kraken:
                    self.pullback_strategy_kraken.manage_intra_candle_exits()

                # 2) Breakout exit-check
                if self.breakout_strategy_kraken:
                    self.breakout_strategy_kraken.manage_intra_candle_exits()

                # 3) AltcoinScanner exit-check (optioneel)
                if self.altcoin_scanner_kraken:
                    self.altcoin_scanner_kraken.manage_intra_candle_exits()

                # hier kun je evt. meltdown_manager checken, als dat in de strategy is
                # of in meltdown_manager zelf.

            except Exception as e:
                self.logger.error(f"[IntraCandleThread] Fout in exit-check: {e}", exc_info=True)

            # 5 seconden rust
            time.sleep(30)

    # =========================================================================
    # TIMED run() elke 15m, met 3 passes Pullback en Breakout/AltScanner
    # =========================================================================
    def run(self):
        """
        1) Start WS's
        2) Start 2e thread voor intra-candle exits elke 5s
        3) Hoofdloop: elke 15m poll + strategy
        """

        self.logger.info("[Executor] run() gestart => TIMED 15m + separate 5s exit-check thread.")

        # A) Start Bitvavo-WS
        if self.use_websocket and self.ws_client:
            self.ws_client.start_websocket()
            self.logger.info("[Executor] Bitvavo WS client thread gestart.")

        # B) Start Kraken (WS + poll)
        if self.use_kraken and self.kraken_data_client:
            self.logger.info("[Executor] KrakenMixedClient => start()")
            self.kraken_data_client.start()

        # C) Start extra thread voor intra-candle (5s) exit-checks
        self.keep_running = True
        exit_thread = threading.Thread(target=self._intra_candle_exits_loop, daemon=True)
        exit_thread.start()
        self.logger.info("[Executor] Intra-candle exit-check thread gestart (5s interval).")

        self.logger.debug("[Executor] TIMED run() => enter main 15m loop")

        loop_count = 0
        try:
            while True:
                # 1) Wacht tot volgende kwartier
                self._sleep_until_next_quarter_hour()

                # 2) 30s marge
                self.logger.info("[Executor] Slaap 30s marge ná kwartiergrens.")
                time.sleep(25)

                loop_count += 1

                # 3) Poll 15m + run pullback
                if self.kraken_data_client:
                    self.logger.info("[Executor] poll_15m_only()")
                    self.kraken_data_client._poll_15m_only()

                self.logger.info("[Executor] run_once_pullback_15m()")
                self.run_once_pullback_15m()

                # 4h-trendstrategie (watch-only): draai bij nieuwe 1h of 4h candle
                if self.trend_strategy_kraken and self.kraken_data_client:
                    for symbol in self.kraken_data_client.pairs:
                        if (self._has_new_closed_candle("candles_kraken", symbol, "1h")
                                or self._has_new_closed_candle("candles_kraken", symbol, "4h")):
                            try:
                                self.trend_strategy_kraken.execute_strategy(symbol)
                            except Exception as e:
                                self.logger.warning("[Executor] TrendStrategy4H error for %s: %s", symbol, e)

                # 4) Breakout => ...
                if self.breakout_strategy_kraken and self.kraken_data_client:
                    for symbol in self.kraken_data_client.pairs:
                        if self._has_new_closed_candle("candles_kraken", symbol, "15m"):
                            self.logger.debug(f"[Executor] => breakout_kraken({symbol})")
                            self.breakout_strategy_kraken.execute_strategy(symbol)

                # 5) AltcoinScanner => ...
                if self.altcoin_scanner_kraken and self.kraken_data_client:
                    any_new_candle = False
                    for symbol in self.kraken_data_client.pairs:
                        if self._has_new_closed_candle("candles_kraken", symbol, "15m"):
                            self.logger.info(f"[Executor] => alt_scanner_kraken for {symbol}")
                            any_new_candle = True
                            break
                    if any_new_candle:
                        self.altcoin_scanner_kraken.execute_strategy()
                    else:
                        self.logger.debug("[Executor] No new 15m candle => skip alt_scanner_kraken")

                # 6) Evt. DB-check elk uur
                if loop_count % 4 == 0:
                    self._hourly_db_checks()

                # 7) Verwerk evt. Bitvavo WS events
                self._process_ws_events()

        except KeyboardInterrupt:
            self.logger.info("[Executor] Bot handmatig gestopt (Ctrl+C).")
        except requests.exceptions.ConnectionError as ce:
            self.logger.error(f"[Executor] Netwerkfout => {ce}, wachten en doorgaan")
            time.sleep(5)
        except Exception as e:
            self.logger.exception(f"[Executor] Fout in run-lus: {e}")
        finally:
            self.logger.info("[Executor] shutting down.")
            # stop de 2e thread
            self.keep_running = False
            if exit_thread.is_alive():
                exit_thread.join(timeout=5)

            # 1) Stop Bitvavo
            if self.ws_client:
                self.logger.info("[Executor] Stop Bitvavo WS.")
                self.ws_client.stop_websocket()
            # 2) Stop Kraken
            if self.kraken_data_client:
                self.logger.info("[Executor] Stop KrakenMixedClient.")
                self.kraken_data_client.stop()
            self.logger.info("[Executor] alles gestopt.")

    def run_once_pullback_15m(self):
        """
        Nieuw: voer de PullbackAccumulateStrategy uit in twee passes (15m).
         - Pass #1: check all pairs => skip_not_closed => coins_skipped
         - time.sleep(20)
         - Pass #2: check only coins_skipped
        (Deze methode is ongewijzigd uit je oorspronkelijke code.)
        """
        if not self.pullback_strategy_kraken or not self.kraken_data_client:
            return

        coins = self.kraken_data_client.pairs
        coins_skipped = []

        self.logger.info("[Executor] PASS #1 => Pullback (15m) for all coins.")
        for symbol in coins:
            result = self.pullback_strategy_kraken.execute_strategy(symbol)
            if result == "skip_not_closed":
                coins_skipped.append(symbol)

        if coins_skipped:
            self.logger.info(f"[Executor] PASS #1 skip_not_closed={coins_skipped} => second pass in 20s.")
            time.sleep(20)
            self.logger.info("[Executor] PASS #2 => re-check the coins_skipped.")
            for symbol in coins_skipped:
                self.pullback_strategy_kraken.execute_strategy(symbol)
        else:
            self.logger.info("[Executor] No coins skipped => no second pass needed.")

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
        """
        Verwerkt de Bitvavo 'order_updates_queue'.
        We laten dit staan voor later re-activatie,
        ook al is Bitvavo nu feitelijk uitgeschakeld.
        """
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
                # meltdown/fill updates?
            elif event_type == "fill":
                self.ws_client.handle_fill_update(event_data)
                # if self.pullback_strategy_bitvavo:
                #     self.pullback_strategy_bitvavo.update_position_with_fill(event_data)
            else:
                self.logger.warning(f"[Executor] Onbekend event in queue: {event_type}")

    def _indicator_analysis_thread(self):
        """
        Eventuele periodieke indicator analysis,
        we laten het hier volledig intact.
        """
        while True:
            process_indicators(self.db_manager)
            time.sleep(60)

    # [NEW] Hulpmethode die checkt of er een nieuwe candle in table_name staat voor (symbol, interval)
    #       én of die candle closed is. Return True als strategy nog niet heeft verwerkt.
    def _has_new_closed_candle(self, table_name, symbol, interval) -> bool:
        """
        Ongewijzigd: wordt nog gebruikt door breakout en alt-scanner.
        Voor pullback (15m) niet meer gebruikt, want we doen 2-pass skip-not-closed.
        """
        df = self.db_manager.fetch_data(table_name, limit=1, market=symbol, interval=interval)

        if df.empty:
            self.logger.debug(
                f"[_has_new_closed_candle] table={table_name}, sym={symbol}, interval={interval} => DF is EMPTY."
            )
            return False

        newest_ts = df["timestamp"].iloc[0]
        self.logger.debug(
            f"[_has_new_closed_candle] table={table_name}, sym={symbol}, interval={interval}, newest_ts={newest_ts}"
        )

        # Check of candle closed
        closed = is_candle_closed(newest_ts, interval)
        if not closed:
            self.logger.debug(
                f"[_has_new_closed_candle] Candle ts={newest_ts} is NOT closed for {symbol}-{interval}"
            )
            return False

        # Key in self.last_closed_ts
        key = (table_name, symbol, interval)
        last_seen_ts = self.last_closed_ts.get(key, 0)
        self.logger.debug(
            f"[_has_new_closed_candle] last_seen_ts={last_seen_ts} for key={key}"
        )

        if newest_ts > last_seen_ts:
            self.last_closed_ts[key] = newest_ts
            self.logger.debug(
                f"[_has_new_closed_candle] Found NEW candle => store {newest_ts} as last_seen."
            )
            return True
        else:
            self.logger.debug(
                f"[_has_new_closed_candle] newest_ts={newest_ts} <= last_seen_ts={last_seen_ts}, skip."
            )
            return False

    # hulpfunctie om naar het eerstvolgende kwartiermoment te wachten
    def _sleep_until_next_quarter_hour(self):
        now = datetime.now()
        minute = now.minute
        second = now.second
        microsecond = now.microsecond

        # Bepaal de eerstvolgende "kwartier": 0, 15, 30 of 45
        if minute < 15:
            target_min = 15
        elif minute < 30:
            target_min = 30
        elif minute < 45:
            target_min = 45
        else:
            target_min = 60  # => nieuw uur

        if target_min == 60:
            next_hour = now.hour + 1
            next_dt = now.replace(hour=(next_hour % 24), minute=0, second=0, microsecond=0)
        else:
            next_dt = now.replace(minute=target_min, second=0, microsecond=0)

        delta = (next_dt - now).total_seconds()
        if delta < 0:
            delta += 900  # 15 * 60

        # Extra clamp om veilig te zijn
        if delta < 0:
            delta = 0

        self.logger.info(f"[Executor] Sleep {delta:.1f}s until next quarter-hour => {next_dt}")
        time.sleep(delta)
