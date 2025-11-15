# ============================================================
# src/strategy/trend_strategy_4h.py
# ============================================================
# 4h trend-volgende strategie met 1h entry-filtering.
# - Modes:
#     "watch"  : alleen signalen loggen + optional save_trade_signals()
#     "dryrun" : trades simuleren in DB (zoals pullback), geen echte orders
#     "auto"   : echte orders via order_client + DB
#
# - Risk/TP/Trailing zoals bij pullback (ATR-based).
# - Strategy-naam in DB: "trend_4h"  (zodat je alles in 1 trades-tabel kunt houden)
#
# Afhankelijkheden:
#  - ta (pip install ta)
#  - jouw IndicatorAnalysis voor RSI (en evt. BB)
#  - jouw DatabaseManager interface: save_trade, update_trade, save_trade_signals, fetch_data
#  - jouw order_client: .place_order(side, symbol, amount, ordertype="market") (in auto mode)
#
# Let op:
#  - Supertrend is optioneel; als je IndicatorAnalysis geen supertrend heeft, zetten we 'm uit via config.
#  - We halen candles uit de table "candles_kraken" (zoals in jouw repo).
#  - We gebruiken 4h voor trend, 1h voor entry + ATR/risk.
# ============================================================

import time
import logging
from decimal import Decimal
from typing import Optional, Dict

import pandas as pd
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator, MACD

from src.logger.logger import setup_logger
from src.config.config import yaml_config  # al door main geladen
from src.indicator_analysis.indicators import IndicatorAnalysis
from src.notifier.telegram_notifier import TelegramNotifier
from src.notifier.bus import send as bus_send
from src.ai.gpt_trend_decider import get_gpt_action

def _to_decimal(x) -> Decimal:
    try:
        return Decimal(str(x))
    except Exception:
        return Decimal("0")


class TrendStrategy4H:
    STRATEGY_NAME = "trend_4h"
    STRATEGY_VERSION = "2025-11-15-gpt-v1"  # <<< nieuw

    def __init__(self, data_client, order_client, db_manager, notifier: Optional[TelegramNotifier] = None,
                     config_path=None):

        """
        :param data_client:    KrakenMixedClient (data)
        :param order_client:   FakeClient (paper) of Kraken client (real)
        :param db_manager:     DatabaseManager
        :param config_path:    niet gebruikt; we lezen rechtstreeks uit yaml_config
        """
        cfg = yaml_config.get("trend_strategy_4h", {})
        log_file = cfg.get("log_file", "logs/trend_strategy_4h.log")
        self.logger = setup_logger("trend_strategy_4h", log_file, logging.INFO)

        self.data_client = data_client
        self.order_client = order_client
        self.db_manager = db_manager
        self.notifier = notifier  # <— DIT IS NIEUW

        # === YAML settings ===
        self.enabled = bool(cfg.get("enabled", False))

        # Timeframes
        self.trend_tf = cfg.get("trend_timeframe", "4h")
        self.entry_tf = cfg.get("entry_timeframe", "1h")

        # Trading mode: watch | dryrun | auto
        self.trading_mode = cfg.get("trading_mode", "watch").lower()
        if self.trading_mode not in ("watch", "dryrun", "auto"):
            self.trading_mode = "watch"

        # Trend/momentum filters
        self.use_adx_filter = bool(cfg.get("use_adx_filter", True))
        self.adx_entry_tf_threshold = float(cfg.get("adx_entry_tf_threshold", 20.0))
        self.use_adx_multitimeframe = bool(cfg.get("use_adx_multitimeframe", True))
        self.adx_high_tf_threshold = float(cfg.get("adx_high_tf_threshold", 20.0))
        self.use_adx_directional_filter = bool(cfg.get("use_adx_directional_filter", True))
        self.adx_window = int(cfg.get("adx_window", 14))  # NEW

        # EMA-structuur
        self.ema_fast = int(cfg.get("ema_fast", 20))
        self.ema_slow = int(cfg.get("ema_slow", 50))
        self.require_trend_stack = bool(cfg.get("require_trend_stack", True))

        # Supertrend (optioneel)
        self.use_supertrend = bool(cfg.get("use_supertrend", False))
        self.supertrend_period = int(cfg.get("supertrend_period", 10))
        self.supertrend_multiplier = float(cfg.get("supertrend_multiplier", 3.0))

        # Risk / positionering
        self.atr_window = int(cfg.get("atr_window", 14))
        self.sl_atr_mult = _to_decimal(cfg.get("sl_atr_mult", "1.5"))
        self.tp1_atr_mult = _to_decimal(cfg.get("tp1_atr_mult", "1.5"))
        self.tp1_portion_pct = _to_decimal(cfg.get("tp1_portion_pct", "0.50"))
        self.trailing_atr_mult = _to_decimal(cfg.get("trailing_atr_mult", "1.0"))

        # NEW exit/guard flags
        self.breakeven_after_tp1 = bool(cfg.get("breakeven_after_tp1", True))
        self.max_hold_hours = int(cfg.get("max_hold_hours", 0))  # 0 = disabled
        self.time_stop_action = str(cfg.get("time_stop_action", "breakeven")).lower()

        # Limieten
        self.min_lot_multiplier = _to_decimal(cfg.get("min_lot_multiplier", "2.1"))
        self.max_position_pct = _to_decimal(cfg.get("max_position_pct", "0.05"))
        self.max_position_eur = _to_decimal(cfg.get("max_position_eur", "15"))
        self.fee_rate = _to_decimal(cfg.get("fee_rate", "0.0035"))  # NEW

        # Interne state
        self.open_positions: Dict[str, dict] = {}   # per symbol
        self.last_processed_candle_ts: Dict[str, int] = {}
        # << NIEUW >>
        self._load_open_positions_from_db()

        self.intra_log_verbose = bool(cfg.get("intra_log_verbose", True))

        self.logger.info("[TrendStrategy4H] initialised (enabled=%s, mode=%s)", self.enabled, self.trading_mode)

    def _notify(self, text: str):
        """Stuur een Telegram-bericht via de globale notifier-bus (als die er is)."""
        try:
            bus_send(text)
        except Exception:
            # notificatie mag nooit de strategie slopen
            pass

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def execute_strategy(self, symbol: str):
        """Wordt aangeroepen bij nieuwe 1h of 4h candle (via executor)."""
        if not self.enabled:
            return

        # 1) Trend op 4h
        df_4h = self._fetch_df(symbol, self.trend_tf, limit=200)
        if df_4h.empty:
            self.logger.debug("[%s] geen %s data.", symbol, self.trend_tf)
            return

        # Skip als laatste candle nog open (failsafe)
        if not self._last_candle_closed(df_4h):
            return

        ema_fast_4h = df_4h["close"].ewm(span=self.ema_fast).mean()
        ema_slow_4h = df_4h["close"].ewm(span=self.ema_slow).mean()
        df_4h[f"ema_{self.ema_fast}"] = ema_fast_4h
        df_4h[f"ema_{self.ema_slow}"] = ema_slow_4h

        adx_4h, _, _ = self._compute_adx_di(df_4h)
        # Voeg ook RSI/MACD toe op 4h zodat GPT die kan gebruiken
        df_4h = self._add_rsi_macd(df_4h)


        # Trendrichting
        last_close_4h = df_4h["close"].iloc[-1]
        last_ema_fast_4h = df_4h[f"ema_{self.ema_fast}"].iloc[-1]
        last_ema_slow_4h = df_4h[f"ema_{self.ema_slow}"].iloc[-1]

        trend_dir = "range"
        if last_ema_fast_4h > last_ema_slow_4h and last_close_4h > last_ema_fast_4h:
            trend_dir = "bull"
        elif last_ema_fast_4h < last_ema_slow_4h and last_close_4h < last_ema_fast_4h:
            trend_dir = "bear"

        if self.use_adx_multitimeframe and adx_4h is not None and adx_4h < self.adx_high_tf_threshold:
            self.logger.info("[%s] 4h adx=%.2f<th=%.1f => skip context", symbol, adx_4h, self.adx_high_tf_threshold)
            return

        if self.require_trend_stack and trend_dir == "range":
            self.logger.info("[%s] trend=range (ema%u vs ema%u) => skip", symbol, self.ema_fast, self.ema_slow)
            return

        # 2) Entry-context op 1h
        df_1h = self._fetch_df(symbol, self.entry_tf, limit=200)
        if df_1h.empty or not self._last_candle_closed(df_1h):
            return

        # RSI + MACD op 1h
        df_1h = self._add_rsi_macd(df_1h)

        # ADX + DI op 1h
        adx_1h, di_pos_1h, di_neg_1h = self._compute_adx_di(df_1h)

        # Supertrend gate (optioneel, guarded)
        if self.use_supertrend:
            if hasattr(IndicatorAnalysis, "calculate_supertrend"):
                try:
                    st = IndicatorAnalysis.calculate_supertrend(
                        df_1h,
                        period=self.supertrend_period,
                        multiplier=self.supertrend_multiplier
                    )
                    df_1h["supertrend"] = st["supertrend"]
                except Exception as e:
                    self.logger.warning("[supertrend] kon niet berekenen (%s) => gate overslaan", e)

            else:
                self.logger.warning("[supertrend] niet beschikbaar in IndicatorAnalysis => gate overslaan")
                # je kunt hier desgewenst self.use_supertrend=False zetten

        # --- Supertrend as a hard gate (only if computed) ---
        if self.use_supertrend and "supertrend" in df_1h.columns:
            st_val = float(df_1h["supertrend"].iloc[-1])
            close_1h = float(df_1h["close"].iloc[-1])

            if trend_dir == "bull" and close_1h <= st_val:
                self.logger.info("[%s] Supertrend gate: long blocked (close %.4f <= ST %.4f)",
                                 symbol, close_1h, st_val)
                return

            if trend_dir == "bear" and close_1h >= st_val:
                self.logger.info("[%s] Supertrend gate: short blocked (close %.4f >= ST %.4f)",
                                 symbol, close_1h, st_val)
                return
        # --- end Supertrend gate ---

        # Logging van setup
        self.logger.info(
            "[SETUP][%s] trend=%s | 4h(adx=%.2f, ema%d=%.2f, ema%d=%.2f) | 1h(adx=%.2f, +DI=%.2f, -DI=%.2f, rsi=%.1f, macd=%.3f)",
            symbol, trend_dir,
            adx_4h if adx_4h is not None else -1.0,
            self.ema_fast, last_ema_fast_4h,
            self.ema_slow, last_ema_slow_4h,
            adx_1h if adx_1h is not None else -1.0,
            di_pos_1h if di_pos_1h is not None else -1.0,
            di_neg_1h if di_neg_1h is not None else -1.0,
            df_1h["rsi"].iloc[-1] if "rsi" in df_1h.columns else -1.0,
            df_1h["macd"].iloc[-1] if "macd" in df_1h.columns else 0.0
        )

        # 3) Entry filters (lean)
        if self.use_adx_filter:
            if adx_1h is None or adx_1h < self.adx_entry_tf_threshold:
                self.logger.info("[%s] entry ADX=%.2f<th=%.1f => skip",
                                 symbol, adx_1h if adx_1h is not None else -1.0, self.adx_entry_tf_threshold)
                return

        if self.use_adx_directional_filter and di_pos_1h is not None and di_neg_1h is not None:
            if trend_dir == "bull" and not (di_pos_1h > di_neg_1h):
                self.logger.info("[%s] bull maar +DI<=-DI (%.2f<=%.2f) => skip", symbol, di_pos_1h, di_neg_1h)
                return
            if trend_dir == "bear" and not (di_neg_1h > di_pos_1h):
                self.logger.info("[%s] bear maar -DI<=+DI (%.2f<=%.2f) => skip", symbol, di_neg_1h, di_pos_1h)
                return

        # 4) ATR (1h) voor risk
        atr_1h = self._compute_atr(df_1h, self.atr_window)
        if atr_1h is None or atr_1h <= 0:
            self.logger.info("[%s] geen ATR => skip", symbol)
            return

        # 5) Mode-actie
        #    - watch: alleen signals opslaan
        #    - dryrun/auto: positioneren (als geen open positie)
        current_price = _to_decimal(df_1h["close"].iloc[-1])

        # eventueel signal logging naar DB
        try:
            self._save_signal_snapshot(symbol, trend_dir, adx_4h, adx_1h, di_pos_1h, di_neg_1h, atr_1h)
        except Exception:
            pass

        has_pos = (symbol in self.open_positions)

        # In watch-mode nog geen GPT-calls / entries forceren
        if self.trading_mode == "watch":
            return

        # Alleen een nieuwe setup als er nog geen positie open is
        if not has_pos and trend_dir in ("bull", "bear"):
            # 1) Algo-signal bepalen
            algo_signal = "long_candidate" if trend_dir == "bull" else "short_candidate"

            # 2) Extra trend-info 1h op basis van EMA
            df_1h[f"ema_{self.ema_fast}"] = df_1h["close"].ewm(span=self.ema_fast).mean()
            df_1h[f"ema_{self.ema_slow}"] = df_1h["close"].ewm(span=self.ema_slow).mean()
            last_close_1h = float(df_1h["close"].iloc[-1])
            last_ema_fast_1h = float(df_1h[f"ema_{self.ema_fast}"].iloc[-1])
            last_ema_slow_1h = float(df_1h[f"ema_{self.ema_slow}"].iloc[-1])

            trend_1h = "range"
            if last_ema_fast_1h > last_ema_slow_1h and last_close_1h > last_ema_fast_1h:
                trend_1h = "bull"
            elif last_ema_fast_1h < last_ema_slow_1h and last_close_1h < last_ema_fast_1h:
                trend_1h = "bear"

            trend_4h = trend_dir  # hergebruik je bestaande 4h-trend

            # 3) EMA-dicts
            ema_1h = {
                "20": last_ema_fast_1h,
                "50": last_ema_slow_1h,
                "relation": "20>50" if last_ema_fast_1h > last_ema_slow_1h else "20<50",
                "slope_20": "up" if df_1h[f"ema_{self.ema_fast}"].iloc[-1] > df_1h[f"ema_{self.ema_fast}"].iloc[-2] else "down"
            }

            ema_4h = {
                "20": float(last_ema_fast_4h),
                "50": float(last_ema_slow_4h),
                "relation": "20>50" if last_ema_fast_4h > last_ema_slow_4h else "20<50",
                "slope_20": "up" if df_4h[f"ema_{self.ema_fast}"].iloc[-1] > df_4h[f"ema_{self.ema_fast}"].iloc[-2] else "down"
            }

            # 4) RSI / MACD voor GPT
            rsi_1h = float(df_1h["rsi"].iloc[-1]) if "rsi" in df_1h.columns else 50.0
            rsi_slope_1h = rsi_1h - (float(df_1h["rsi"].iloc[-2]) if "rsi" in df_1h.columns else rsi_1h)
            macd_1h = float(df_1h["macd"].iloc[-1]) if "macd" in df_1h.columns else 0.0

            rsi_4h = float(df_4h["rsi"].iloc[-1]) if "rsi" in df_4h.columns else 50.0
            rsi_slope_4h = rsi_4h - (float(df_4h["rsi"].iloc[-2]) if "rsi" in df_4h.columns else rsi_4h)
            macd_4h = float(df_4h["macd"].iloc[-1]) if "macd" in df_4h.columns else 0.0

            # 5) Support/resistance
            levels_1h = self._compute_sr_levels(df_1h, window=20)
            levels_4h = self._compute_sr_levels(df_4h, window=20)

            # 6) Compacte candles bouwen
            candles_1h = self._build_compact_candles(
                df_1h,
                ema_fast_col=f"ema_{self.ema_fast}",
                ema_slow_col=f"ema_{self.ema_slow}",
                limit=20
            )
            candles_4h = self._build_compact_candles(
                df_4h,
                ema_fast_col=f"ema_{self.ema_fast}",
                ema_slow_col=f"ema_{self.ema_slow}",
                limit=20
            )

            # 7) GPT om een besluit vragen
            try:
                action, decision = get_gpt_action(
                    symbol=symbol,
                    algo_signal=algo_signal,
                    trend_1h=trend_1h,
                    trend_4h=trend_4h,
                    structure_1h="unknown",   # later kun je HL/LH logica toevoegen
                    structure_4h="unknown",
                    ema_1h=ema_1h,
                    ema_4h=ema_4h,
                    rsi_1h=rsi_1h,
                    rsi_slope_1h=rsi_slope_1h,
                    macd_1h=macd_1h,
                    rsi_4h=rsi_4h,
                    rsi_slope_4h=rsi_slope_4h,
                    macd_4h=macd_4h,
                    levels_1h=levels_1h,
                    levels_4h=levels_4h,
                    candles_1h=candles_1h,
                    candles_4h=candles_4h,
                )
            except Exception as e:
                self.logger.warning("[%s] GPT decision failed (%s) => fallback HOLD", symbol, e)
                action = "HOLD"
                decision = {
                    "action": "HOLD",
                    "confidence": 0,
                    "rationale": "GPT error",
                    "journal_tags": []
                }

            self.logger.info(
                "[GPT][%s] algo_signal=%s => action=%s, conf=%s, rationale=%s, tags=%s",
                symbol,
                algo_signal,
                action,
                decision.get("confidence"),
                decision.get("rationale"),
                decision.get("journal_tags"),
            )

            # 8) Actie van GPT vertalen naar side/open
            if action == "OPEN_LONG":
                side = "buy"
            elif action == "OPEN_SHORT":
                side = "sell"
            else:
                # HOLD => geen positie openen
                self.logger.info("[%s] GPT -> HOLD => geen trade geopend.", symbol)
                # Eventueel Telegram bij HOLD (optioneel)
                self._notify(
                    f"[GPT-HOLD][{symbol}] "
                    f"signal={algo_signal} | conf={decision.get('confidence', 0)}\n"
                    f"reason: {decision.get('rationale', '')}"
                )
                return

            # 9) Als GPT een trade wil en mode is dryrun/auto => openen
            self._open_position(symbol, side=side, entry_price=current_price, atr_value=_to_decimal(atr_1h))

            # Extra Telegram met GPT-beslissing
            try:
                tags = decision.get("journal_tags", [])
                tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
                self._notify(
                    f"[GPT-OPEN][{symbol}] action={action} | signal={algo_signal}\n"
                    f"conf={decision.get('confidence', 0)} | tags=[{tags_str}]\n"
                    f"reason: {decision.get('rationale', '')}"
                )
            except Exception:
                pass


    def manage_intra_candle_exits(self):
        """Aangeroepen door executor (aparte thread)."""
        if not self.enabled or self.trading_mode == "watch":
            return
        for sym, pos in list(self.open_positions.items()):
            px = self._latest_price(sym)
            if px <= 0:
                continue
            # --- TIME-STOP (if enabled) ---
            try:
                if self.max_hold_hours and self.max_hold_hours > 0:
                    opened_ts = pos.get("opened_ts", int(time.time()))
                    hours_open = (time.time() - opened_ts) / 3600.0
                    if hours_open >= float(self.max_hold_hours):
                        side = pos["side"]
                        entry = pos["entry_price"]
                        cur = _to_decimal(px)

                        if self.time_stop_action == "close":
                            self.logger.info("[TIME-STOP][%s] max_hold_hours=%s reached => CLOSE NOW",
                                             sym, self.max_hold_hours)
                            self._close_all(sym, reason="TimeStop", exec_price=cur)
                            continue
                        else:
                            # breakeven action: close only if we can do so at or better than entry
                            closeable = ((side == "buy" and cur >= entry) or
                                         (side == "sell" and cur <= entry))
                            if closeable:
                                self.logger.info("[TIME-STOP][%s] breakeven close allowed at %.4f (entry %.4f)",
                                                 sym, float(cur), float(entry))
                                self._close_all(sym, reason="TimeStopBreakeven", exec_price=cur)
                                continue
                            else:
                                self.logger.info(
                                    "[TIME-STOP][%s] breakeven not possible yet (px=%.4f, entry=%.4f) => hold",
                                    sym, float(cur), float(entry))
            except Exception as e:
                self.logger.debug("[TIME-STOP][%s] error: %s", sym, e)
            # --- END TIME-STOP ---

            if self.intra_log_verbose:
                pos = self.open_positions[sym]
                side = pos["side"]
                entry = pos["entry_price"]
                one_r = pos["atr"]
                tp1_done = pos.get("tp1_done", False)
                trail_active = pos.get("trail_active", False)
                trail_high = pos.get("trail_high", entry)

                if side == "buy":
                    sl = entry - (one_r * self.sl_atr_mult)
                    tp1 = entry + (one_r * self.tp1_atr_mult)
                    trailing = (trail_high - (one_r * self.trailing_atr_mult)) if trail_active else None
                else:
                    sl = entry + (one_r * self.sl_atr_mult)
                    tp1 = entry - (one_r * self.tp1_atr_mult)
                    trailing = (trail_high + (one_r * self.trailing_atr_mult)) if trail_active else None

                if trailing is not None:
                    self.logger.info(
                        "[INTRA][%s] %s price=%.4f | SL=%.4f | TP1=%.4f | trail_stop=%.4f | tp1_done=%s | trail_active=%s",
                        sym, ("LONG" if side == "buy" else "SHORT"),
                        float(px), float(sl), float(tp1), float(trailing),
                        tp1_done, trail_active
                    )
                else:
                    self.logger.info(
                        "[INTRA][%s] %s price=%.4f | SL=%.4f | TP1=%.4f | trail_stop=-- | tp1_done=%s | trail_active=%s",
                        sym, ("LONG" if side == "buy" else "SHORT"),
                        float(px), float(sl), float(tp1),
                        tp1_done, trail_active
                    )

            self._manage_position(sym, current_price=_to_decimal(px), atr_value=pos["atr"])

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _fetch_df(self, symbol: str, interval: str, limit=200) -> pd.DataFrame:
        df = self.db_manager.fetch_data(
            table_name="candles_kraken",
            limit=limit,
            market=symbol,
            interval=interval
        )
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        # Normaliseer kolommen (zoals je pullback doet)
        cols = list(df.columns)
        # verwacht: [timestamp, market, interval, open, high, low, close, volume, ...]
        rename_map = {}
        if "timestamp" in cols:
            rename_map["timestamp"] = "timestamp_ms"
        df = df.rename(columns=rename_map)

        # force numeriek
        for c in ("open", "high", "low", "close", "volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Index als datetime (UTC)
        if "timestamp_ms" in df.columns:
            df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df = df.set_index("datetime_utc", drop=False)
            df.sort_index(inplace=True)

        return df

    def _last_candle_closed(self, df: pd.DataFrame) -> bool:
        # we gaan ervan uit dat "timestamp_ms" = eindtijd van de candle
        try:
            last_ms = int(df["timestamp_ms"].iloc[-1])
            now_ms = int(time.time() * 1000)
            return now_ms >= last_ms
        except Exception:
            return True

    def _add_rsi_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = IndicatorAnalysis.calculate_indicators(df, rsi_window=14)
        except Exception:
            pass
        try:
            macd_ind = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
            df["macd"] = macd_ind.macd()
            df["macd_signal"] = macd_ind.macd_signal()
        except Exception:
            df["macd"] = 0.0
            df["macd_signal"] = 0.0
        return df

    def _compute_adx_di(self, df: pd.DataFrame):
        try:
            adx_obj = ADXIndicator(
                high=df["high"], low=df["low"], close=df["close"], window=self.adx_window
            )
            adx = float(adx_obj.adx().iloc[-1])
            di_pos = float(adx_obj.adx_pos().iloc[-1])
            di_neg = float(adx_obj.adx_neg().iloc[-1])
            return adx, di_pos, di_neg
        except Exception:
            return None, None, None

    def _compute_atr(self, df: pd.DataFrame, window=14) -> Optional[float]:
        try:
            atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=window)
            val = atr.average_true_range().iloc[-1]
            return float(val) if pd.notna(val) else None
        except Exception:
            return None

    def _compute_sr_levels(self, df: pd.DataFrame, window: int = 20) -> dict:
        """
        Simpele support/resistance op basis van laatste N candles.
        """
        if df.empty or "high" not in df.columns or "low" not in df.columns:
            return {"support": None, "resistance": None}

        recent = df.tail(window)
        support = float(recent["low"].min())
        resistance = float(recent["high"].max())
        return {"support": support, "resistance": resistance}

    def _build_compact_candles(self, df: pd.DataFrame, ema_fast_col: str, ema_slow_col: str, limit: int = 20) -> list:
        """
        Bouw compacte candle-structuur (B) met OHLC, EMA, RSI, MACD, volume, wicks.
        """
        if df.empty:
            return []

        subset = df.tail(limit).copy()
        candles = []

        for _, row in subset.iterrows():
            try:
                o = float(row["open"])
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])
                vol = float(row["volume"]) if "volume" in row else 0.0

                ema_fast = float(row[ema_fast_col]) if ema_fast_col in row else c
                ema_slow = float(row[ema_slow_col]) if ema_slow_col in row else c

                rsi = float(row["rsi"]) if "rsi" in row and pd.notna(row["rsi"]) else 50.0

                if "macd" in row and "macd_signal" in row and pd.notna(row["macd"]) and pd.notna(row["macd_signal"]):
                    macd_hist = float(row["macd"] - row["macd_signal"])
                else:
                    macd_hist = 0.0

                total_range = max(h - l, 1e-9)
                top_wick = max(h - c, h - o)
                bot_wick = max(c - l, o - l)
                top_wick_pct = float((top_wick / total_range) * 100.0)
                bot_wick_pct = float((bot_wick / total_range) * 100.0)

                candles.append({
                    "o": o,
                    "h": h,
                    "l": l,
                    "c": c,
                    "ema20": ema_fast,
                    "ema50": ema_slow,
                    "rsi": rsi,
                    "macd_hist": macd_hist,
                    "vol": vol,
                    "top_wick_pct": top_wick_pct,
                    "bot_wick_pct": bot_wick_pct,
                })
            except Exception:
                # één kapotte candle mag de boel niet slopen
                continue

        return candles

    def _latest_price(self, symbol: str) -> Decimal:
        # probeeer WS prijs via data_client
        try:
            px = self.data_client.get_latest_ws_price(symbol)
            if px and px > 0:
                return _to_decimal(px)
        except Exception:
            pass
        # backfill via DB 1m
        df = self.db_manager.fetch_data(
            table_name="candles_kraken",
            limit=1,
            market=symbol,
            interval="1m"
        )
        if isinstance(df, pd.DataFrame) and not df.empty and "close" in df.columns:
            return _to_decimal(df["close"].iloc[0])
        return Decimal("0")

    def _min_lot(self, symbol: str) -> Decimal:
        try:
            return _to_decimal(self.data_client.get_min_lot(symbol))
        except Exception:
            return Decimal("0.0001")

    def _equity_estimate(self) -> Decimal:
        try:
            bal = self.order_client.get_balance()
        except Exception:
            bal = {}

        total = Decimal("0")
        for asset, amt in (bal or {}).items():
            amt = _to_decimal(amt)
            if asset.upper() == "EUR":
                total += amt
            else:
                sym = f"{asset.upper()}-EUR"
                px = self._latest_price(sym)
                if px > 0:
                    total += (amt * px)

        # Fallback so paper sizing works
        if total <= 0 and self.trading_mode == "dryrun":
            total = _to_decimal(yaml_config.get("paper_equity_eur", 1000))

        return total

    # ---------------------------------------------------------
    # Trading (dryrun/auto) - in lijn met pullback
    # ---------------------------------------------------------
    def _open_position(self, symbol: str, side: str, entry_price: Decimal, atr_value: Decimal):
        if symbol in self.open_positions:
            return

        # sizing: 5% equity capped by max_position_eur, minimaal min_lot*multiplier
        equity = self._equity_estimate()
        allowed_eur_pct = equity * self.max_position_pct
        allowed_eur = allowed_eur_pct if allowed_eur_pct < self.max_position_eur else self.max_position_eur

        min_lot = self._min_lot(symbol)
        needed_eur_for_min = min_lot * self.min_lot_multiplier * entry_price

        if needed_eur_for_min > allowed_eur:
            self.logger.info("[%s] allowed %.2f < minTrade %.2f => skip open",
                             symbol, float(allowed_eur), float(needed_eur_for_min))
            return

        buy_eur = needed_eur_for_min
        amount = buy_eur / entry_price

        # === DB: master trade (in dryrun/auto). In watch doen we niets. ===
        master_id = None
        pos_id = f"{symbol}-{int(time.time())}"  # << NEW: one id used for DB and RAM
        if self.trading_mode in ("dryrun", "auto"):
            # opening exposure and opening fee (like pullback)
            open_trade_cost = float(entry_price * _to_decimal(amount))  # actual notional at entry
            open_fee = open_trade_cost * float(self.fee_rate)

            trade_data = {
                "symbol": symbol,
                "side": side,
                "amount": float(amount),
                "price": float(entry_price),
                "timestamp": int(time.time() * 1000),
                "position_id": pos_id,
                "position_type": "long" if side == "buy" else "short",
                "status": "open",
                "pnl_eur": 0.0,
                "fees": float(open_fee),  # << record opening fee
                "trade_cost": float(open_trade_cost),  # << record exposure at entry
                "strategy_name": self.STRATEGY_NAME,
                "is_master": 1
            }
            self.db_manager.save_trade(trade_data)
            master_id = self.db_manager.cursor.lastrowid

        # echte order alleen in auto
        if self.trading_mode == "auto":
            try:
                self.order_client.place_order(side, symbol, float(amount), ordertype="market")
            except Exception as e:
                self.logger.warning("[order] kon niet plaatsen: %s", e)

        self.open_positions[symbol] = {
            "side": side,
            "entry_price": entry_price,
            "amount": _to_decimal(amount),
            "atr": atr_value,
            "tp1_done": False,
            "trail_active": False,
            "trail_high": entry_price,  # works for both long/short
            "position_id": pos_id,  # << uses the same id as DB
            "master_id": master_id,
            "opened_ts": int(time.time()),  # << NEW
            "breakeven_applied": False,  # << NEW
            # --- NIEUW: voor Telegram / PnL bij full close ---
            "initial_amount": _to_decimal(amount),
            "realized_pnl": Decimal("0"),
            "total_fees": Decimal("0")
        }

        self.logger.info("[OPEN][%s] %s @ %.4f (ATR=%.4f, mode=%s)",
                         symbol, "LONG" if side == "buy" else "SHORT",
                         float(entry_price), float(atr_value), self.trading_mode)

        # log signals bij open
        try:
            self._save_trade_signals(master_id, "open", symbol, atr_value)
        except Exception:
            pass

        # Telegram notificatie bij open
        # AFTER logger.info("[OPEN]...") and _save_trade_signals

        sl = float(entry_price - (atr_value * self.sl_atr_mult)) if side == "buy" else float(
            entry_price + (atr_value * self.sl_atr_mult))
        tp1 = float(entry_price + (atr_value * self.tp1_atr_mult)) if side == "buy" else float(
            entry_price - (atr_value * self.tp1_atr_mult))

        eur_size = float(amount * entry_price)

        self._notify(
            f"[TREND-OPEN][{symbol}] {'LONG' if side == 'buy' else 'SHORT'} @ {float(entry_price):.4f}\n"
            f"size: €{eur_size:.2f} ({float(amount):.6f})\n"
            f"SL={sl:.4f} | TP1={tp1:.4f} | ATR={float(atr_value):.4f}\n"
            f"GPT: pending\n"  # wordt straks live ingevuld
            f"reason: pending"
        )

    def _manage_position(self, symbol: str, current_price: Decimal, atr_value: Decimal):
        if symbol not in self.open_positions:
            return
        pos = self.open_positions[symbol]
        side = pos["side"]
        entry = pos["entry_price"]
        amount = pos["amount"]
        master_id = pos.get("master_id")
        one_r = atr_value

        if side == "buy":
            stop = entry - (one_r * self.sl_atr_mult)
            if current_price <= stop:
                self._close_all(symbol, reason="StopLoss", exec_price=current_price)
                return

            tp1 = entry + (one_r * self.tp1_atr_mult)
            if not pos["tp1_done"] and current_price >= tp1:
                self._partial_close(symbol, portion=self.tp1_portion_pct, reason="TP1", exec_price=current_price)
                pos["tp1_done"] = True
                pos["trail_active"] = True
                pos["trail_high"] = max(pos["trail_high"], current_price)

            # --- BREAKEVEN AFTER TP1 (LONG) ---
            if pos.get("tp1_done") and self.breakeven_after_tp1 and not pos.get("breakeven_applied", False):
                pos["breakeven_applied"] = True  # mark once
                # No immediate order; we clamp the trailing stop to never go below entry.
                self.logger.info("[BREAKEVEN][%s] LONG enforced at entry %.4f after TP1", symbol, float(entry))
            # --- END BREAKEVEN AFTER TP1 ---

            if pos["trail_active"]:
                if current_price > pos["trail_high"]:
                    pos["trail_high"] = current_price
                trailing_stop = pos["trail_high"] - (one_r * self.trailing_atr_mult)
                # clamp to breakeven if enabled/applied
                if pos.get("breakeven_applied", False):
                    trailing_stop = max(trailing_stop, entry)
                if current_price <= trailing_stop:
                    self._close_all(symbol, reason="TrailingStop", exec_price=current_price)
                    return

        else:  # short
            stop = entry + (one_r * self.sl_atr_mult)
            if current_price >= stop:
                self._close_all(symbol, reason="StopLoss", exec_price=current_price)
                return

            tp1 = entry - (one_r * self.tp1_atr_mult)
            if not pos["tp1_done"] and current_price <= tp1:
                self._partial_close(symbol, portion=self.tp1_portion_pct, reason="TP1", exec_price=current_price)
                pos["tp1_done"] = True
                pos["trail_active"] = True
                pos["trail_high"] = current_price  # voor short gebruiken we 'low' als 'trail_high' holder

            # --- BREAKEVEN AFTER TP1 (SHORT) ---
            if pos.get("tp1_done") and self.breakeven_after_tp1 and not pos.get("breakeven_applied", False):
                pos["breakeven_applied"] = True
                self.logger.info("[BREAKEVEN][%s] SHORT enforced at entry %.4f after TP1", symbol, float(entry))
            # --- END BREAKEVEN AFTER TP1 ---

            if pos["trail_active"]:
                if current_price < pos["trail_high"]:
                    pos["trail_high"] = current_price
                trailing_stop = pos["trail_high"] + (one_r * self.trailing_atr_mult)
                # clamp to breakeven if enabled/applied
                if pos.get("breakeven_applied", False):
                    trailing_stop = min(trailing_stop, entry)
                if current_price >= trailing_stop:
                    self._close_all(symbol, reason="TrailingStop", exec_price=current_price)
                    return

        # leftover check (klein restant sluiten)
        if pos["amount"] <= 0 or pos["amount"] < self._min_lot(symbol):
            if master_id:
                self.db_manager.update_trade(master_id, {"status": "closed"})
            del self.open_positions[symbol]

    def _partial_close(self, symbol: str, portion: Decimal, reason: str, exec_price: Optional[Decimal] = None):
        if symbol not in self.open_positions:
            return
        pos = self.open_positions[symbol]
        side = pos["side"]
        amount = pos["amount"]
        min_lot = self._min_lot(symbol)

        amt_to_close = amount * portion
        leftover_after = amount - amt_to_close
        if leftover_after > 0 and leftover_after < min_lot:
            amt_to_close = amount
            portion = Decimal("1.0")

        px = exec_price if exec_price is not None else self._latest_price(symbol)
        if px <= 0:
            return

        trade_side = "sell" if side == "buy" else "buy"  # long -> verkopen; short -> terugkopen

        # auto: echte order
        if self.trading_mode == "auto":
            try:
                self.order_client.place_order(trade_side, symbol, float(amt_to_close), ordertype="market")
            except Exception as e:
                self.logger.warning("[order child] %s", e)

        # DB child in dryrun/auto
        if self.trading_mode in ("dryrun", "auto"):
            pnl_raw = (px - pos["entry_price"]) * amt_to_close if side == "buy" else (pos["entry_price"] - px) * amt_to_close
            trade_cost = px * amt_to_close
            fees = float(trade_cost * self.fee_rate)
            realized_pnl = float(pnl_raw) - fees
            # --- NIEUW: totals in RAM bijhouden voor CLOSE-bericht ---
            pos["realized_pnl"] = pos.get("realized_pnl", Decimal("0")) + _to_decimal(realized_pnl)
            pos["total_fees"] = pos.get("total_fees", Decimal("0")) + _to_decimal(fees)


            child = {
                "symbol": symbol,
                "side": trade_side,
                "amount": float(amt_to_close),
                "price": float(px),
                "timestamp": int(time.time() * 1000),
                "position_id": pos.get("position_id"),
                "position_type": "long" if side == "buy" else "short",
                "status": "partial" if portion < 1 else "closed",
                "pnl_eur": realized_pnl,
                "fees": fees,
                "trade_cost": float(trade_cost),
                "strategy_name": self.STRATEGY_NAME,
                "is_master": 0
            }
            self.db_manager.save_trade(child)

            # update leftover
            pos["amount"] = amount - amt_to_close
            # update master pnl/fees (best-effort)
            # update master pnl/fees (best-effort)
            master_id = pos.get("master_id")
            if master_id:
                try:
                    old = self.db_manager.execute_query(
                        "SELECT fees, pnl_eur FROM trades WHERE id=? LIMIT 1",
                        (master_id,)
                    )
                    if old:
                        old_fees, old_pnl = old[0]
                        old_fees = float(old_fees or 0.0)
                        old_pnl = float(old_pnl or 0.0)
                        self.db_manager.update_trade(master_id, {
                            "status": "partial" if pos["amount"] > 0 else "closed",
                            "fees": old_fees + float(fees),
                            "pnl_eur": old_pnl + float(realized_pnl),
                            "amount": float(pos["amount"])
                        })
                except Exception:
                    pass

        self.logger.info("[PARTIAL CLOSE][%s] portion=%.2f, px=%.4f, reason=%s",
                         symbol, float(portion), float(px), reason)

        # Telegram partial
        side_label = "LONG" if side == "buy" else "SHORT"
        entry = pos["entry_price"]

        exposure = float(entry * amt_to_close)
        pnl_raw = (float(px) - float(entry)) * float(amt_to_close) if side == "buy" else (float(entry) - float(
            px)) * float(amt_to_close)
        pnl = pnl_raw - float(fees)
        roi = (pnl_raw / exposure) * 100 if exposure > 0 else 0.0
        r_value = pnl_raw / (
                    float(pos["atr"]) * float(self.sl_atr_mult) * float(amt_to_close)) if amt_to_close > 0 else 0.0

        self._notify(
            f"[TREND-PARTIAL][{symbol}] {side_label}\n"
            f"{float(portion):.2f} closed @ {float(px):.4f} ({reason})\n"
            f"realized PnL: {pnl:+.2f} EUR | {r_value:+.2f}R | ROI: {roi:+.1f}%\n"
            f"fees: {fees:.4f}\n"
            f"remaining: €{float((amount - amt_to_close) * entry):.2f} ({float(amount - amt_to_close):.6f})"
        )

        # log signals bij partial
        try:
            self._save_trade_signals(pos.get("master_id"), "partial", symbol, pos["atr"])
        except Exception:
            pass

        if pos["amount"] <= 0:
            # volledig dicht -> master op closed
            mid = pos.get("master_id")
            if mid:
                try:
                    self.db_manager.update_trade(mid, {"status": "closed"})
                except Exception:
                    pass

            # --- NIEUW: totaal PnL / ROI / R / fees / hold time ---
            total_pnl = float(pos.get("realized_pnl", Decimal("0")))
            total_fees = float(pos.get("total_fees", Decimal("0")))
            entry = pos["entry_price"]
            initial_amount = pos.get("initial_amount", amount)

            exposure = float(entry * initial_amount)
            roi_total = (total_pnl / exposure) * 100 if exposure > 0 else 0.0
            r_total = (
                total_pnl /
                (float(pos["atr"]) * float(self.sl_atr_mult) * float(initial_amount))
                if initial_amount > 0 else 0.0
            )
            hold_hours = (time.time() - pos.get("opened_ts", int(time.time()))) / 3600.0

            side_label = "LONG" if side == "buy" else "SHORT"

            self._notify(
                f"[TREND-CLOSE][{symbol}] {side_label} ({reason})\n"
                f"exit @ {float(px):.4f}\n"
                f"total PnL: {total_pnl:+.2f} EUR | {r_total:+.2f}R | ROI: {roi_total:+.1f}%\n"
                f"fees total: {total_fees:.4f}\n"
                f"hold time: {hold_hours:.1f} uur"
            )

            # tenslotte uit RAM halen
            del self.open_positions[symbol]

    def _close_all(self, symbol: str, reason: str, exec_price: Optional[Decimal] = None):
        # convenience: sluit volledige positie (dryrun/auto) + DB update
        self._partial_close(symbol, portion=Decimal("1.0"), reason=reason, exec_price=exec_price)

    # ---------------------------------------------------------
    # Signals opslag (optioneel, zelfde stijl als pullback)
    # ---------------------------------------------------------
    def _save_signal_snapshot(self, symbol: str, trend_dir: str,
                              adx_4h: Optional[float], adx_1h: Optional[float],
                              di_pos_1h: Optional[float], di_neg_1h: Optional[float],
                              atr_1h: float):
        # schrijf een klein snapshotje in signals-tabel
        data = {
            "trade_id": None,  # los signaal
            "event_type": "setup",
            "symbol": symbol,
            "strategy_name": self.STRATEGY_NAME,
            "rsi_15m": 0.0,           # niet gebruikt hier
            "macd_val": 0.0,
            "macd_signal": 0.0,
            "atr_value": float(atr_1h),
            "depth_score": 0.0,
            "ml_signal": 0.0,
            "rsi_h4": 0.0,
            "timestamp": int(time.time() * 1000)
        }
        try:
            self.db_manager.save_trade_signals(data)
        except Exception:
            pass

    def _save_trade_signals(self, trade_id: Optional[int], event_type: str, symbol: str, atr_value: Decimal):
        if not trade_id:
            return
        try:
            df_1h = self._fetch_df(symbol, self.entry_tf, limit=40)
            df_1h = self._add_rsi_macd(df_1h)
            macd_signal_1h = float(df_1h["macd_signal"].iloc[-1]) if "macd_signal" in df_1h.columns else 0.0
            rsi_1h = float(df_1h["rsi"].iloc[-1]) if "rsi" in df_1h.columns else 50.0

            data = {
                "trade_id": trade_id,
                "event_type": event_type,
                "symbol": symbol,
                "strategy_name": self.STRATEGY_NAME,
                "rsi_15m": rsi_1h,              # we gebruiken hier 1h rsi
                "macd_val": float(df_1h["macd"].iloc[-1]) if "macd" in df_1h.columns else 0.0,
                "macd_signal": macd_signal_1h,
                "atr_value": float(atr_value),
                "depth_score": 0.0,
                "ml_signal": 0.0,
                "rsi_h4": 0.0,
                "timestamp": int(time.time() * 1000)
            }
            self.db_manager.save_trade_signals(data)
        except Exception as e:
            self.logger.debug("[signals] kon niet opslaan: %s", e)

    def _load_open_positions_from_db(self):
        """
        Load open/partial MASTER trades (strategy_name='trend_4h') from DB
        and rebuild self.open_positions with a fresh ATR (1h).
        """
        try:
            rows = self.db_manager.execute_query(
                """
                SELECT id, symbol, side, amount, price, position_id, position_type, status
                  FROM trades
                 WHERE is_master=1
                   AND status IN ('open','partial')
                   AND strategy_name=?
                """,
                (self.STRATEGY_NAME,)
            )
        except Exception as e:
            self.logger.warning("[reload] DB read failed: %s", e)
            return

        if not rows:
            self.logger.info("[reload] no open/partial trend_4h masters found.")
            return

        for (db_id, symbol, side, amount, entry_price, position_id, position_type, status) in rows:
            # Recompute ATR from 1h
            df_1h = self._fetch_df(symbol, self.entry_tf, limit=200)
            # NEW: tiny retry for late-writing candles (up to ~10s total)
            if (df_1h.empty or not self._last_candle_closed(df_1h)):
                for _ in range(5):  # 5 retries x 2s = ~10s
                    time.sleep(2)
                    df_1h = self._fetch_df(symbol, self.entry_tf, limit=200)
                    if not df_1h.empty and self._last_candle_closed(df_1h):
                        break

            if df_1h.empty or not self._last_candle_closed(df_1h):
                return

            atr_val = self._compute_atr(df_1h, self.atr_window)
            if not atr_val:
                self.logger.info("[reload][%s] no ATR available => skip restore.", symbol)
                continue

            self.open_positions[symbol] = {
                "side": side,  # "buy" or "sell"
                "entry_price": _to_decimal(entry_price),
                "amount": _to_decimal(amount),
                "atr": _to_decimal(atr_val),
                "tp1_done": False,
                "trail_active": False,
                "trail_high": _to_decimal(entry_price),
                "position_id": position_id,
                "position_type": position_type,
                "master_id": db_id,
                "opened_ts": int(time.time()),  # default since DB has no open timestamp
                "breakeven_applied": False
            }

            self.logger.info(
                "[reload][%s] restored: side=%s, amt=%s @ %s, ATR=%.4f, master_id=%s",
                symbol, side, str(amount), str(entry_price), float(atr_val), db_id
            )

    def reload_open_positions(self):
        """Manual/periodic refresh from DB into RAM using the unified loader."""
        try:
            self.logger.info("[reload] manual reload requested for trend_4h.")
            self.open_positions = {}  # clear RAM view
            self._load_open_positions_from_db()  # single source of truth
        except Exception as e:
            self.logger.warning("[reload] failed: %s", e)

    def _load_atr_from_signals(self, trade_id: int):
        """Pak laatste atr_value uit signals voor deze trade_id."""
        try:
            row = self.db_manager.execute_query(
                """
                SELECT atr_value
                FROM signals
                WHERE trade_id=?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (trade_id,)
            )
            if row and row[0] and row[0][0] is not None:
                return float(row[0][0])
        except Exception:
            pass
        return None


