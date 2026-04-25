# ============================================================
# src/trading_engine/executor.py
# ============================================================

import threading
import time
import logging
from datetime import datetime, timezone

from src.logger.logger import setup_logger
from src.my_websocket.fake_client import FakeClient

# Strategy modules
from src.strategy.trend_strategy_4h import TrendStrategy4H

from src.config.config import EXECUTOR_LOG_FILE

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
        yaml_config=None,
        notifier=None,
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
        self.use_kraken = use_kraken
        self.kraken_paper = kraken_paper

        # Lees YAML
        self.yaml_config = yaml_config or {}
        kraken_cfg = self.yaml_config.get("kraken", {})

        # ------------------------------
        # B) Bitvavo disabled
        # ------------------------------
        self.ws_client = None
        self.data_client = None
        self.order_client = None

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
        self.config = self.yaml_config.get("executor_config", {})

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

    def _intra_candle_exits_loop(self):
        """
        Deze functie draait in een aparte thread en roept periodiek
        trend_4h.manage_intra_candle_exits() aan.
        """
        while self.keep_running:
            try:
                # Trend exit-check
                if self.trend_strategy_kraken:
                    self.trend_strategy_kraken.manage_intra_candle_exits()

            except Exception as e:
                self.logger.error(f"[IntraCandleThread] Fout in exit-check: {e}", exc_info=True)

            # 5 seconden rust
            time.sleep(30)

    # =========================================================================
    # TIMED run() elke 15m: poll data, run trend_4h, manage exits in separate thread
    # =========================================================================
    def run(self):
        """
        1) Start WS's
        2) Start 2e thread voor intra-candle exits elke 5s
        3) Hoofdloop: elke 15m poll + trend checks
        """

        self.logger.info("[Executor] run() gestart => TIMED 15m + separate 5s exit-check thread.")

        # Start Kraken (WS + poll)
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

                # 3) Poll 15m data
                if self.kraken_data_client:
                    self.logger.info("[Executor] poll_15m_only()")
                    self.kraken_data_client._poll_15m_only()

                # 4h-trendstrategie (watch/dryrun/auto): draai bij nieuwe 1h of 4h candle
                if self.trend_strategy_kraken and self.kraken_data_client:
                    total_pairs = len(self.kraken_data_client.pairs)
                    trend_triggered = 0

                    for symbol in self.kraken_data_client.pairs:
                        has_new_1h = self._has_new_closed_candle("candles_kraken", symbol, "1h")
                        has_new_4h = self._has_new_closed_candle("candles_kraken", symbol, "4h")

                        if has_new_1h or has_new_4h:
                            trend_triggered += 1
                            try:
                                self.logger.debug(
                                    "[Executor] Trend trigger for %s (new %s%s candle)",
                                    symbol,
                                    "1h" if has_new_1h else "",
                                    "/4h" if has_new_4h else ""
                                )
                                self.trend_strategy_kraken.execute_strategy(symbol)
                            except Exception as e:
                                self.logger.warning("[Executor] TrendStrategy4H error for %s: %s", symbol, e)

                    # INFO-regel: hoeveel coins zijn echt door de strategy gegaan
                    self.logger.info(
                        "[Executor][TREND] execute_strategy aangeroepen voor %d/%d pairs in deze uur-check.",
                        trend_triggered,
                        total_pairs
                    )

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

            # Stop Kraken
            if self.kraken_data_client:
                self.logger.info("[Executor] Stop KrakenMixedClient.")
                self.kraken_data_client.stop()
            self.logger.info("[Executor] alles gestopt.")

    def run_daily_tasks(self):
        self.logger.info("[Executor] run_daily_tasks() skipped (trend-only runtime).")

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

        # DEBUG: if trend seems silent, show what we actually see in DB for 1h/4h
        if interval in ("1h", "4h"):
            if df.empty:
                self.logger.info(f"[_has_new_closed_candle][DEBUG] {symbol}-{interval} => DB empty")
            else:
                newest_ts = int(df['timestamp'].iloc[0])
                now_ms = int(time.time() * 1000)
                self.logger.info(
                    f"[_has_new_closed_candle][DEBUG] {symbol}-{interval} newest_ts={newest_ts} "
                    f"now_ms={now_ms} closed={now_ms >= newest_ts}"
                )

        if df.empty:
            self.logger.debug(
                f"[_has_new_closed_candle] table={table_name}, sym={symbol}, interval={interval} => DF is EMPTY."
            )
            return False

        newest_ts = df["timestamp"].iloc[0]
        self.logger.debug(
            f"[_has_new_closed_candle] table={table_name}, sym={symbol}, interval={interval}, newest_ts={newest_ts}"
        )

        # Hard guard: voorkom “tussenstanden” voor 1h/4h
        ts = datetime.utcfromtimestamp(newest_ts / 1000.0)  # UTC
        if interval == "1h":
            if ts.minute != 0:
                self.logger.debug(f"[_has_new_closed_candle] guard 1h: minute={ts.minute} != 0 => skip.")
                return False
        elif interval == "4h":
            if ts.minute != 0 or (ts.hour % 4) != 0:
                self.logger.debug(
                    f"[_has_new_closed_candle] guard 4h: {ts.hour:02d}:{ts.minute:02d} not a 4h close => skip.")
                return False

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
