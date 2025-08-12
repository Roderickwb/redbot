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
from src.meltdown_manager.meltdown_manager import MeltdownManager
from src.notifier.bus import send as notify

def _to_decimal(x) -> Decimal:
    try:
        return Decimal(str(x))
    except Exception:
        return Decimal("0")


class TrendStrategy4H:
    STRATEGY_NAME = "trend_4h"

    def __init__(self, data_client, order_client, db_manager, config_path=None):
        """
        :param data_client:    KrakenMixedClient (data)
        :param order_client:   FakeClient (paper) of Kraken client (real)
        :param db_manager:     DatabaseManager
        :param config_path:    niet gebruikt; we lezen rechtstreeks uit yaml_config
        """
        cfg = yaml_config.get("trend_strategy_4h", {})
        log_file = cfg.get("log_file", "logs/trend_strategy_4h.log")
        self.strategy_config = cfg
        self.logger = setup_logger("trend_strategy_4h", log_file, logging.INFO)

        # --- equity fallback like pullback ---
        self.initial_capital = _to_decimal(self.strategy_config.get("initial_capital", "100"))
        # Optional alias: allow YAML to use 'equity_eur' instead of 'initial_capital'
        if "equity_eur" in self.strategy_config and "initial_capital" not in self.strategy_config:
            self.initial_capital = _to_decimal(self.strategy_config.get("equity_eur"))

        # allow overriding the DB tag via YAML, but keep default
        self.STRATEGY_NAME = cfg.get("strategy_name", self.STRATEGY_NAME)
        self.logger.info("[TrendStrategy4H] strategy tag=%s", self.STRATEGY_NAME)

        self.data_client = data_client
        self.order_client = order_client
        self.db_manager = db_manager

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

        # EMA-structuur
        self.ema_fast = int(cfg.get("ema_fast", 20))
        self.ema_slow = int(cfg.get("ema_slow", 50))
        self.require_trend_stack = bool(cfg.get("require_trend_stack", True))

        # Supertrend (optioneel)
        self.use_supertrend = bool(cfg.get("use_supertrend", False))
        self.supertrend_period = int(cfg.get("supertrend_period", 10))
        self.supertrend_multiplier = float(cfg.get("supertrend_multiplier", 3.0))

        # EMA-200 trend gate (optional)
        self.use_ema_trend_200 = bool(cfg.get("use_ema_trend_200", False))
        self.ema_trend_period = int(cfg.get("ema_trend_period", 200))

        # LLM guard (optional)
        self.use_llm_guard = bool(cfg.get("use_llm_guard", False))
        self.llm_model = cfg.get("llm_model", "gpt-4.1-mini")
        self.llm_timeout_sec = int(cfg.get("llm_timeout_sec", 4))
        self.llm_client = None  # injectable

        # Risk / positionering
        self.atr_window = int(cfg.get("atr_window", 14))
        self.sl_atr_mult = _to_decimal(cfg.get("sl_atr_mult", "1.5"))
        self.tp1_atr_mult = _to_decimal(cfg.get("tp1_atr_mult", "1.5"))
        self.tp1_portion_pct = _to_decimal(cfg.get("tp1_portion_pct", "0.50"))
        self.trailing_atr_mult = _to_decimal(cfg.get("trailing_atr_mult", "1.0"))

        # Limieten
        self.min_lot_multiplier = _to_decimal(cfg.get("min_lot_multiplier", "2.1"))
        self.max_position_pct = _to_decimal(cfg.get("max_position_pct", "0.05"))
        self.max_position_eur = _to_decimal(cfg.get("max_position_eur", "15"))

        # Fees (maker/taker baseline). Change per exchange in YAML.
        self.fee_rate = _to_decimal(cfg.get("fee_rate", "0.0035"))

        # --- Global risk gate (same as pullback) ---
        meltdown_cfg = yaml_config.get("meltdown_manager", {})
        self.meltdown_manager = MeltdownManager(
            meltdown_cfg,
            db_manager=self.db_manager,
            logger=setup_logger("meltdown_manager_trend", "logs/meltdown_manager_trend.log", logging.DEBUG)
        )

        # Interne state
        self.open_positions: Dict[str, dict] = {}  # per symbol
        self.last_processed_candle_ts: Dict[str, int] = {}

        # ===== NIEUW: definieer extra guards/caches =====
        self.cold_start_until = time.time() + 180  # 3 min warm‑up
        self.max_price_age_ms = 180_000  # 3 min max leeftijd voor 1m fallback
        self.last_good_px: Dict[str, Decimal] = {}  # cache laatste geldige prijs per symbool
        # ===============================================

        # herlaad open posities en seed de cache
        self.reload_open_positions()
        for sym, pos in self.open_positions.items():
            self.last_good_px[sym] = pos["entry_price"]

        self.intra_log_verbose = bool(cfg.get("intra_log_verbose", True))

        self.logger.info("[TrendStrategy4H] initialised (enabled=%s, mode=%s)", self.enabled, self.trading_mode)

    # ---------------------------------------------------------
    # Helper: check if an open/partial master exists in DB (per strategy)
    # ---------------------------------------------------------
    def _has_open_master_in_db(self, symbol: str) -> bool:
        try:
            row = self.db_manager.execute_query(
                """
                SELECT 1
                FROM trades
                WHERE symbol=? AND is_master=1
                    AND status IN ('open','partial')
                    AND strategy_name=?
                LIMIT 1
                """,
                (symbol, self.STRATEGY_NAME)
            )
            return bool(row)
        except Exception:
            return False

    def _get_equity_estimate(self) -> Decimal:
        """
        SAME behavior as PullbackAccumulateStrategy:
        - Try full wallet valuation in EUR (EUR + coins*EUR price)
        - Use live/paper get_balance() if available
        - Fall back to initial_capital from this strategy's config
        Never crash — always return a Decimal.
        """
        # If we have no order_client, fall back to configured initial_capital
        if not getattr(self, "order_client", None):
            return getattr(self, "initial_capital", Decimal("100"))

        try:
            bal = self.order_client.get_balance() or {}
        except Exception:
            # balance lookup failed => fallback to configured initial_capital
            return getattr(self, "initial_capital", Decimal("100"))

        total_wallet_eur = Decimal("0")
        for asset, amount_str in bal.items():
            amt = Decimal(str(amount_str))
            if asset.upper() == "EUR":
                total_wallet_eur += amt
            else:
                # Convert asset -> EUR using latest price
                symbol = f"{asset.upper()}-EUR"
                price = self._get_latest_price(symbol)
                if price > 0:
                    total_wallet_eur += (amt * price)

        return total_wallet_eur

    def _get_latest_price(self, symbol: str) -> Decimal:
        """
        Mirror of pullback’s price helper:
        - Prefer live ws price (via data_client)
        - Fallback to last 1m close in DB
        """
        # Try ticker (live WS mid price)
        if getattr(self, "data_client", None):
            try:
                px_float = self.data_client.get_latest_ws_price(symbol)
                if px_float and px_float > 0.0:
                    return Decimal(str(px_float))
            except Exception:
                pass

        # Fallback: last 1m close from DB
        try:
            df_1m = self.db_manager.fetch_data(
                table_name="candles_kraken",
                limit=1,
                market=symbol,
                interval="1m"
            )
            if isinstance(df_1m, pd.DataFrame) and not df_1m.empty and "close" in df_1m.columns:
                last_close = df_1m["close"].iloc[0]
                return Decimal(str(last_close))
        except Exception:
            pass

        return Decimal("0")

    def _get_ws_price(self, symbol: str) -> Decimal:
        """
        Convenience (if you happen to use it in Trend too).
        """
        if not getattr(self, "data_client", None):
            return Decimal("0")
        try:
            px_float = self.data_client.get_latest_ws_price(symbol)
            if px_float and px_float > 0.0:
                return Decimal(str(px_float))
        except Exception:
            pass
        return Decimal("0")

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def execute_strategy(self, symbol: str):
        """Wordt aangeroepen bij nieuwe 1h of 4h candle (via executor)."""
        if not self.enabled:
            return

        # Global meltdown gate: no new entries, still manage open positions
        meltdown_active = self.meltdown_manager.update_meltdown_state(strategy=self, symbol=symbol)
        if meltdown_active:
            if symbol in self.open_positions:
                px = self._latest_price(symbol)
                if px > 0:
                    self._manage_position(symbol, current_price=_to_decimal(px), atr_value=self.open_positions[symbol]["atr"])
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

        # Trendrichting
        last_close_4h = df_4h["close"].iloc[-1]
        last_ema_fast_4h = df_4h[f"ema_{self.ema_fast}"].iloc[-1]
        last_ema_slow_4h = df_4h[f"ema_{self.ema_slow}"].iloc[-1]

        trend_dir = "range"
        if last_ema_fast_4h > last_ema_slow_4h and last_close_4h > last_ema_fast_4h:
            trend_dir = "bull"
        elif last_ema_fast_4h < last_ema_slow_4h and last_close_4h < last_ema_fast_4h:
            trend_dir = "bear"

        # Optional: 4h EMA-200 gate
        if self.use_ema_trend_200:
            ema200_4h = df_4h["close"].ewm(span=self.ema_trend_period).mean().iloc[-1]
            if trend_dir == "bull" and not (last_close_4h > ema200_4h):
                self.logger.info("[%s] EMA200 gate: long blocked (close %.4f <= ema200 %.4f)", symbol, float(last_close_4h), float(ema200_4h))
                return
            if trend_dir == "bear" and not (last_close_4h < ema200_4h):
                self.logger.info("[%s] EMA200 gate: short blocked (close %.4f >= ema200 %.4f)", symbol, float(last_close_4h), float(ema200_4h))
                return

        if self.use_adx_multitimeframe and adx_4h is not None and adx_4h < self.adx_high_tf_threshold:
            self.logger.info("[%s] 4h adx=%.2f<th=%.1f => skip context", symbol, adx_4h, self.adx_high_tf_threshold)
            return

        if self.require_trend_stack and trend_dir == "range":
            self.logger.info("[%s] trend=range (ema%u vs ema%u) => skip", symbol, self.ema_fast, self.ema_slow)
            return

        # [A] EMA-slope check (4h): som van de laatste 3 fast-EMA deltas moet richting trend zijn
        try:
            ema_fast_series_4h = df_4h[f"ema_{self.ema_fast}"]
            ema_fast_slope_4h = float(ema_fast_series_4h.diff().iloc[-3:].sum())
            if trend_dir == "bull" and ema_fast_slope_4h <= 0:
                self.logger.info("[%s] bull maar EMA-fast slope<=0 (%.6f) => skip", symbol, ema_fast_slope_4h)
                return
            if trend_dir == "bear" and ema_fast_slope_4h >= 0:
                self.logger.info("[%s] bear maar EMA-fast slope>=0 (%.6f) => skip", symbol, ema_fast_slope_4h)
                return
        except Exception:
            # Geen harde stop als slope niet te berekenen is
            pass

        # 2) Entry-context op 1h
        df_1h = self._fetch_df(symbol, self.entry_tf, limit=200)
        if df_1h.empty or not self._last_candle_closed(df_1h):
            return

        # --- Candle gate: verwerk elke 1h-candle maar 1x per symbol ---
        try:
            last_ms = int(df_1h["timestamp_ms"].iloc[-1])
        except Exception:
            # Failsafe: als timestamp ontbreekt, niet gate-en
            last_ms = None

        if last_ms is not None:
            prev_ms = self.last_processed_candle_ts.get(symbol)
            if prev_ms == last_ms:
                # deze 1h-candle is al verwerkt → klaar
                return
            # markeer dat we deze candle nu gaan verwerken
            self.last_processed_candle_ts[symbol] = last_ms
        # ---------------------------------------------------------------

        # RSI + MACD op 1h
        df_1h = self._add_rsi_macd(df_1h)

        # [B] MACD-histogram bevestigt de trendrichting
        macd_hist_1h = 0.0  # default for LLM guard / logs
        try:
            macd_val = float(df_1h["macd"].iloc[-1])
            macd_sig = float(df_1h["macd_signal"].iloc[-1])
            macd_hist_1h = macd_val - macd_sig
            if trend_dir == "bull" and macd_hist_1h <= 0:
                self.logger.info("[%s] bull maar MACD-hist<=0 (%.6f) => skip", symbol, macd_hist_1h)
                return
            if trend_dir == "bear" and macd_hist_1h >= 0:
                self.logger.info("[%s] bear maar MACD-hist>=0 (%.6f) => skip", symbol, macd_hist_1h)
                return
        except Exception:
            # Als MACD niet beschikbaar is, niet hard blokken
            pass

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

        # >>> ADD THIS RIGHT BELOW THE BLOCK ABOVE <<<
        # Only act as a gate if ST is enabled AND was computed
        if self.use_supertrend and "supertrend" in df_1h.columns:
            st_val = float(df_1h["supertrend"].iloc[-1])
            close_1h = float(df_1h["close"].iloc[-1])

            if trend_dir == "bull" and close_1h <= st_val:
                self.logger.info("[%s] Supertrend gate: long blocked (close %.4f <= ST %.4f)", symbol, close_1h,
                                         st_val)
                return

            if trend_dir == "bear" and close_1h >= st_val:
                self.logger.info("[%s] Supertrend gate: short blocked (close %.4f >= ST %.4f)", symbol,
                                    close_1h, st_val)
                return

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

        live_px = self._latest_price(symbol)
        if live_px <= 0:
            self.logger.info("[%s] live price unavailable => skip open this candle", symbol)
            return
        current_price = live_px

        try:
            self._save_signal_snapshot(symbol, trend_dir, adx_4h, adx_1h, di_pos_1h, di_neg_1h, atr_1h)
        except Exception:
            pass

        # Optional LLM veto (after all technical filters; before open)
        if self.use_llm_guard:
            rsi_1h_val = float(df_1h["rsi"].iloc[-1]) if "rsi" in df_1h.columns else 50.0
            ema_fast_slope_val = 0.0
            try:
                ema_fast_slope_val = float(df_4h[f"ema_{self.ema_fast}"].diff().iloc[-3:].sum())
            except Exception:
                pass
            if not self._llm_guard(symbol, trend_dir, adx_4h, adx_1h, di_pos_1h, di_neg_1h,
                                   rsi_1h_val, macd_hist_1h, ema_fast_slope_val):
                return

        has_pos = (symbol in self.open_positions)

        if self.trading_mode == "watch":
            # niets openen, alleen kijken
            return

        if (not has_pos) and (not self._has_open_master_in_db(symbol)) and trend_dir in ("bull", "bear"):
            side = "buy" if trend_dir == "bull" else "sell"
            self._open_position(symbol, side=side, entry_price=current_price, atr_value=_to_decimal(atr_1h))

        # als al positie, wordt trailing/sl/tp in manage_intra_candle_exits() gedaan

    def manage_intra_candle_exits(self):
        """Aangeroepen door executor (aparte thread)."""
        if not self.enabled or self.trading_mode == "watch":
            return

        # COLD-START GUARD: skip de eerste seconden na start
        if hasattr(self, "cold_start_until") and time.time() < self.cold_start_until:
            return

        # Extra failsafe: als er geen open posities zijn, stop
        if not self.open_positions:
            return

        for sym, pos in list(self.open_positions.items()):
            px = self._latest_price(sym)
            if px <= 0:
                if not getattr(self, "_warned_price_zero", False):
                    self.logger.info("[INTRA][%s] price unavailable (WS down of 1m too old) => skip manage", sym)
                    self._warned_price_zero = True
                continue

            # prijs is OK → reset ‘eenmalige waarschuwing’
            if getattr(self, "_warned_price_zero", False):
                self._warned_price_zero = False

            # extra vangnet: na herstart geen >30% gap vs entry toestaan
            try:
                if time.time() < (self.cold_start_until + 120):
                    entry = pos["entry_price"]
                    if entry > 0:
                        dev = abs((px - entry) / entry)
                        if dev > Decimal("0.30"):
                            self.logger.warning(
                                "[INTRA][%s] prijs outlier %.4f vs entry %.4f (dev=%.1f%%) => skip tick",
                                sym, float(px), float(entry), float(dev * 100))
                            continue
            except Exception:
                pass

            if self.intra_log_verbose:
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
    def set_llm_client(self, client):
        """Inject an LLM client with .judge(symbol, payload, model, timeout) -> 'allow'/'block'."""
        self.llm_client = client

    def _llm_guard(self, symbol: str, trend_dir: str,
                   adx_4h, adx_1h, di_pos_1h, di_neg_1h,
                   rsi_1h, macd_hist_1h, ema_fast_slope_4h) -> bool:
        """Return True to allow, False to block. Fail-open (allow) on any error."""
        if not self.llm_client:
            return True
        try:
            payload = {
                "symbol": symbol,
                "trend_dir": trend_dir,
                "adx_4h": adx_4h, "adx_1h": adx_1h,
                "di_pos_1h": di_pos_1h, "di_neg_1h": di_neg_1h,
                "rsi_1h": rsi_1h, "macd_hist_1h": macd_hist_1h,
                "ema_fast_slope_4h": ema_fast_slope_4h,
            }
            verdict = self.llm_client.judge(
                symbol, payload, model=self.llm_model, timeout=self.llm_timeout_sec
            )
            allow = str(verdict).strip().lower().startswith("allow")
            if not allow:
                self.logger.info("[LLM GUARD][%s] blocked by LLM verdict=%s payload=%s",
                                 symbol, verdict, payload)
            return allow
        except Exception as e:
            self.logger.warning("[LLM GUARD][%s] error: %s -> default ALLOW", symbol, e)
            return True

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
            adx_obj = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
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

    def _latest_price(self, symbol: str) -> Decimal:
        # 1) WebSocket prijs eerst
        try:
            px = self.data_client.get_latest_ws_price(symbol)
            if px and px > 0:
                d = _to_decimal(px)
                self.last_good_px[symbol] = d
                return d
        except Exception:
            pass

        # 2) Ticker mid-price fallback (best_bid/best_ask)
        try:
            df_ticker = self.db_manager.fetch_data(
                table_name="ticker_kraken", limit=1, market=symbol
            )
            if isinstance(df_ticker, pd.DataFrame) and not df_ticker.empty:
                bid = float(df_ticker["best_bid"].iloc[0]) if "best_bid" in df_ticker.columns else 0.0
                ask = float(df_ticker["best_ask"].iloc[0]) if "best_ask" in df_ticker.columns else 0.0
                if bid > 0 and ask > 0:
                    d = _to_decimal((bid + ask) / 2)
                    self.last_good_px[symbol] = d
                    return d
        except Exception:
            pass

        # 3) Candles 1m fallback: sorteer + verouderingscheck
        df = self.db_manager.fetch_data(
            table_name="candles_kraken", limit=5, market=symbol, interval="1m"
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            # bepaal timestamp-kolom
            ts_col = "timestamp_ms" if "timestamp_ms" in df.columns else (
                "timestamp" if "timestamp" in df.columns else None)
            try:
                if ts_col:
                    df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
                    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
                    last_ts = int(df[ts_col].iloc[-1])
                else:
                    last_ts = None
                last_close = df["close"].iloc[-1] if "close" in df.columns else None
            except Exception:
                last_ts, last_close = None, None

            if last_close is not None:
                if last_ts is not None:
                    now_ms = int(time.time() * 1000)
                    if now_ms - last_ts > self.max_price_age_ms:
                        self.logger.debug("[price stale][%s] 1m close too old: age=%dms > %dms",
                                          symbol, now_ms - last_ts, self.max_price_age_ms)
                        return Decimal("0")
                d = _to_decimal(last_close)
                self.last_good_px[symbol] = d
                return d

        # 4) Laatste bekende goede prijs als redmiddel
        if symbol in self.last_good_px:
            return self.last_good_px[symbol]

        return Decimal("0")

    def _min_lot(self, symbol: str) -> Decimal:
        try:
            return _to_decimal(self.data_client.get_min_lot(symbol))
        except Exception:
            return Decimal("0.0001")

    # ---------------------------------------------------------
    # Trading (dryrun/auto) - in lijn met pullback
    # ---------------------------------------------------------
    def _open_position(self, symbol: str, side: str, entry_price: Decimal, atr_value: Decimal):
        if symbol in self.open_positions:
            return

        # sizing: 5% equity capped by max_position_eur, minimaal min_lot*multiplier
        equity = self._get_equity_estimate()
        allowed_eur_pct = equity * self.max_position_pct
        allowed_eur = min(allowed_eur_pct, self.max_position_eur)

        min_lot = self._min_lot(symbol)
        needed_eur_for_min = min_lot * self.min_lot_multiplier * entry_price

        if needed_eur_for_min > allowed_eur:
            self.logger.info("[%s] allowed %.2f < minTrade %.2f => skip open",
                             symbol, float(allowed_eur), float(needed_eur_for_min))
            return

        buy_eur = needed_eur_for_min
        amount = buy_eur / entry_price

        if side == "buy" and self.trading_mode == "auto":
            try:
                bal = self.order_client.get_balance()
                eur = _to_decimal(bal.get("EUR", "0"))
                if eur < buy_eur:
                    self.logger.info("[%s] long skipped: EUR %.2f < needed %.2f", symbol, float(eur), float(buy_eur))
                    return
            except Exception as e:
                self.logger.warning("[%s] EUR balance check failed (%s) => skip", symbol, e)
                return

        # === SPOT SHORT SAFETY (same behavior as pullback) ===
        if side == "sell":
            try:
                bal = self.order_client.get_balance() if self.order_client else {}
            except Exception:
                bal = {}

            base = symbol.split("-")[0].upper()
            have = _to_decimal(bal.get(base, "0"))
            needed_coins = self._min_lot(symbol) * self.min_lot_multiplier

            # 1) must own the base coin
            if have <= 0:
                self.logger.warning("[OPEN][%s] skip SHORT: no %s in wallet.", symbol, base)
                return

            # 2) must have at least min_lot * multiplier (same rule as pullback)
            if have < needed_coins:
                self.logger.warning(
                    "[OPEN][%s] skip SHORT: have %.8f %s < needed %.8f (min_lot*multiplier).",
                    symbol, float(have), base, float(needed_coins)
                )
                return

            # ensure we never try to sell more than we own
            max_sell_amt = have
            if amount > max_sell_amt:
                amount = max_sell_amt

            # also cap by risk budget (same allowed_eur you computed above)
            max_by_risk = allowed_eur / entry_price
            if amount > max_by_risk:
                amount = max_by_risk

        # === END SPOT SHORT SAFETY ===

        # === DB: master trade (in dryrun/auto). In watch doen we niets. ===
        master_id = None
        pos_id = f"{symbol}-{int(time.time())}"  # ← nieuw, altijd vóór gebruik

        if self.trading_mode in ("dryrun", "auto"):
            trade_data = {
                "symbol": symbol,
                "side": side,
                "amount": float(amount),
                "price": float(entry_price),
                "timestamp": int(time.time() * 1000),
                "position_id": pos_id,  # ← nu geen error
                "position_type": "long" if side == "buy" else "short",
                "status": "open",
                "pnl_eur": 0.0,
                "fees": 0.0,
                "trade_cost": float(entry_price * amount),
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
            "amount": amount,
            "atr": atr_value,
            "tp1_done": False,
            "trail_active": False,
            "trail_high": entry_price,
            "position_id": pos_id,  # ← zelfde waarde als in DB
            "master_id": master_id,
        }

        self.logger.info("[OPEN][%s] %s @ %.4f (ATR=%.4f, mode=%s)",
                         symbol, "LONG" if side == "buy" else "SHORT",
                         float(entry_price), float(atr_value), self.trading_mode)

        notify(f"[OPEN] {symbol} {'LONG' if side == 'buy' else 'SHORT'} @ {float(entry_price):.4f} "
               f"(ATR={float(atr_value):.4f}, mode={self.trading_mode})")

        # log signals bij open
        try:
            self._save_trade_signals(master_id, "open", symbol, atr_value)
        except Exception:
            pass

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

            if pos["trail_active"]:
                if current_price > pos["trail_high"]:
                    pos["trail_high"] = current_price
                trailing_stop = pos["trail_high"] - (one_r * self.trailing_atr_mult)
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

            if pos["trail_active"]:
                if current_price < pos["trail_high"]:
                    pos["trail_high"] = current_price
                trailing_stop = pos["trail_high"] + (one_r * self.trailing_atr_mult)
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

            child = {
                "symbol": symbol,
                "side": trade_side,
                "amount": float(amt_to_close),
                "price": float(px),
                "timestamp": int(time.time() * 1000),
                "position_id": pos["position_id"],
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

            master_id = pos.get("master_id")
            if master_id:
                try:
                    old = self.db_manager.execute_query(
                        "SELECT fees, pnl_eur FROM trades WHERE id=? LIMIT 1",
                        (master_id,)
                    )
                    if old:
                        old_fees, old_pnl = old[0]
                        if pos["amount"] > 0:
                            # PARTIAL: update ook leftover amount
                            self.db_manager.update_trade(master_id, {
                                "status": "partial",
                                "fees": old_fees + fees,
                                "pnl_eur": old_pnl + realized_pnl,
                                "amount": float(pos["amount"])
                            })
                        else:
                            # CLOSED: géén 'amount' meer wegschrijven (laat laatste leftover staan)
                            self.db_manager.update_trade(master_id, {
                                "status": "closed",
                                "fees": old_fees + fees,
                                "pnl_eur": old_pnl + realized_pnl
                            })
                except Exception:
                    pass

        self.logger.info("[PARTIAL CLOSE][%s] portion=%.2f, px=%.4f, reason=%s",
                         symbol, float(portion), float(px), reason)

        notify(f"[{reason}] {symbol} portion={float(portion):.2f} px={float(px):.4f}")

        # log signals bij partial
        # log signals with correct event
        try:
            ev = "close" if portion == Decimal("1.0") else "partial"
            self._save_trade_signals(pos.get("master_id"), ev, symbol, pos["atr"])
        except Exception:
            pass

        if pos["amount"] <= 0:
            # volledig dicht
            mid = pos.get("master_id")
            if mid:
                try:
                    self.db_manager.update_trade(mid, {"status": "closed"})
                except Exception:
                    pass
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

    def _sum_child_trade_amounts(self, position_id: str) -> Optional[Decimal]:
        """Sommeer amount van alle child-trades (is_master=0) met dezelfde position_id."""
        try:
            row = self.db_manager.execute_query(
                """
                SELECT COALESCE(SUM(amount), 0)
                FROM trades
                WHERE is_master=0
                  AND position_id=?
                  AND strategy_name=?
                """,
                (position_id, self.STRATEGY_NAME)
            )
            if row and row[0] and row[0][0] is not None:
                return _to_decimal(row[0][0])
        except Exception as e:
            self.logger.debug("[reload] child-sum query faalde voor %s: %s", position_id, e)
        return None

    def reload_open_positions(self):
        """Lees open/partial master trades (trend_4h) uit DB en herstel self.open_positions."""
        try:
            rows = self.db_manager.execute_query(
                """
                SELECT id, symbol, side, price, amount, position_id
                FROM trades
                WHERE is_master=1
                  AND status IN ('open','partial')
                  AND strategy_name=?
                """,
                (self.STRATEGY_NAME,)
            )

        except Exception as e:
            self.logger.warning("[reload] DB-query faalde: %s", e)
            rows = []

        restored = 0
        self.open_positions = {}  # reset

        for (trade_id, symbol, side, price, amount, position_id) in rows or []:
            # 1) ATR proberen te halen uit laatste signals van deze trade
            atr_val = self._load_atr_from_signals(trade_id)
            if atr_val is None:
                # fallback: bereken ATR op 1h
                df_1h = self._fetch_df(symbol, self.entry_tf, limit=100)
                atr_val = self._compute_atr(df_1h, self.atr_window) if not df_1h.empty else None
            if atr_val is None:
                self.logger.info("[reload] %s trade %s zonder ATR => skip", symbol, trade_id)
                continue

            entry_price = _to_decimal(price)
            amt = _to_decimal(amount)

            # Min-lot check (gebruik de bestaande helper)
            min_lot = self._min_lot(symbol)
            if amt <= 0:
                self.logger.info("[reload] %s trade %s amount<=0 => skip", symbol, trade_id)
                continue

            if (min_lot > 0 and amt < min_lot):
                if self.trading_mode == "auto":
                    self.logger.info("[reload] %s trade %s onder min lot => skip (auto)", symbol, trade_id)
                    continue
                else:
                    self.logger.info("[reload] %s trade %s onder min lot => HERSTEL (dryrun)", symbol, trade_id)
                    # Niet 'continue' in dryrun

            # Child-som check: als children de master al (bijna) leeg hebben, skip
            total_child_amount = self._sum_child_trade_amounts(position_id)
            if total_child_amount is not None and amt <= total_child_amount:
                self.logger.info("[reload] %s trade %s reeds afgebouwd door children => skip", symbol, trade_id)
                continue

            self.open_positions[symbol] = {
                "side": side,  # "buy" of "sell"
                "entry_price": entry_price,
                "amount": amt,
                "atr": _to_decimal(atr_val),
                "tp1_done": False,  # kan je later afleiden uit child-trades
                "trail_active": False,
                "trail_high": entry_price,  # wordt tijdens runtime bijgewerkt
                "position_id": position_id,
                "master_id": trade_id,
            }
            restored += 1
            self.logger.info("[reload] hersteld: %s (%s) entry=%.4f amount=%.8f ATR=%.5f",
                             symbol, "LONG" if side == "buy" else "SHORT",
                             float(entry_price), float(amt), float(atr_val))

        if restored == 0:
            self.logger.info("[reload] geen open/partial trend_4h master trades gevonden.")
        else:
            self.logger.info("[reload] %d open positie(s) hersteld voor trend_4h.", restored)

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


