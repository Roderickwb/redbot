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
        self.logger = setup_logger("trend_strategy_4h", log_file, logging.INFO)

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

        # Interne state
        self.open_positions: Dict[str, dict] = {}   # per symbol
        self.last_processed_candle_ts: Dict[str, int] = {}
        # << NIEUW >>
        self._load_open_positions_from_db()

        self.intra_log_verbose = bool(cfg.get("intra_log_verbose", True))

        self.logger.info("[TrendStrategy4H] initialised (enabled=%s, mode=%s)", self.enabled, self.trading_mode)

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

        if self.trading_mode == "watch":
            # niets openen, alleen kijken
            return

        if not has_pos and trend_dir in ("bull", "bear"):
            side = "buy" if trend_dir == "bull" else "sell"
            self._open_position(symbol, side=side, entry_price=current_price, atr_value=_to_decimal(atr_1h))

        # als al positie, wordt trailing/sl/tp in manage_intra_candle_exits() gedaan

    def manage_intra_candle_exits(self):
        """Aangeroepen door executor (aparte thread)."""
        if not self.enabled or self.trading_mode == "watch":
            return
        for sym, pos in list(self.open_positions.items()):
            px = self._latest_price(sym)
            if px <= 0:
                continue

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
        # zelfde aanpak als pullback: som wallet (EUR + assets in EUR)
        try:
            bal = self.order_client.get_balance()
        except Exception:
            return Decimal("0")
        total = Decimal("0")
        for asset, amt in bal.items():
            amt = _to_decimal(amt)
            if asset.upper() == "EUR":
                total += amt
            else:
                sym = f"{asset.upper()}-EUR"
                px = self._latest_price(sym)
                if px > 0:
                    total += (amt * px)
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
        if self.trading_mode in ("dryrun", "auto"):
            trade_data = {
                "symbol": symbol,
                "side": side,
                "amount": float(amount),
                "price": float(entry_price),
                "timestamp": int(time.time() * 1000),
                "position_id": f"{symbol}-{int(time.time())}",
                "position_type": "long" if side == "buy" else "short",
                "status": "open",
                "pnl_eur": 0.0,
                "fees": 0.0,
                "trade_cost": float(buy_eur),
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
            "trail_high": entry_price if side == "buy" else entry_price,  # reuse field for both
            "master_id": master_id,
        }

        self.logger.info("[OPEN][%s] %s @ %.4f (ATR=%.4f, mode=%s)",
                         symbol, "LONG" if side == "buy" else "SHORT",
                         float(entry_price), float(atr_value), self.trading_mode)

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
            fees = float(trade_cost * Decimal("0.0035"))
            realized_pnl = float(pnl_raw) - fees

            child = {
                "symbol": symbol,
                "side": trade_side,
                "amount": float(amt_to_close),
                "price": float(px),
                "timestamp": int(time.time() * 1000),
                "position_id": f"{symbol}-{int(time.time())}",
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
            master_id = pos.get("master_id")
            if master_id:
                try:
                    old = self.db_manager.execute_query("SELECT fees, pnl_eur FROM trades WHERE id=? LIMIT 1",
                                                        (master_id,))
                    if old:
                        old_fees, old_pnl = old[0]
                        self.db_manager.update_trade(master_id, {
                            "status": "partial" if pos["amount"] > 0 else "closed",
                            "fees": old_fees + fees,
                            "pnl_eur": old_pnl + realized_pnl,
                            "amount": float(pos["amount"])
                        })
                except Exception:
                    pass

        self.logger.info("[PARTIAL CLOSE][%s] portion=%.2f, px=%.4f, reason=%s",
                         symbol, float(portion), float(px), reason)

        # log signals bij partial
        try:
            self._save_trade_signals(pos.get("master_id"), "partial", symbol, pos["atr"])
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

    def _load_open_positions_from_db(self):
        """
        Herlaadt open/partial MASTER trades (is_master=1) met strategy_name='trend_4h'
        en zet ze terug in self.open_positions, incl. ATR (1h) voor risk-management.
        """
        try:
            rows = self.db_manager.execute_query("""
                SELECT id, symbol, side, amount, price, position_id, position_type, status
                  FROM trades
                 WHERE is_master=1
                   AND status IN ('open','partial')
                   AND strategy_name=?
            """, (self.STRATEGY_NAME,))
        except Exception as e:
            self.logger.warning("[reload] kon DB niet lezen: %s", e)
            return

        if not rows:
            self.logger.info("[reload] geen open/partial trend_4h master trades gevonden.")
            return

        for (db_id, symbol, side, amount, entry_price, position_id, position_type, status) in rows:
            # ATR opnieuw uit 1h voor actuele R
            df_1h = self._fetch_df(symbol, self.entry_tf, limit=200)
            atr_val = self._compute_atr(df_1h, self.atr_window)
            if not atr_val:
                self.logger.info("[reload][%s] geen ATR beschikbaar => skip herstel.", symbol)
                continue

            self.open_positions[symbol] = {
                "side": side,
                "entry_price": _to_decimal(entry_price),
                "amount": _to_decimal(amount),
                "atr": _to_decimal(atr_val),
                "tp1_done": False,  # we houden het lean; eventueel kun je dit later uit signals afleiden
                "trail_active": False,
                "trail_high": _to_decimal(entry_price),
                "master_id": db_id,
            }
            self.logger.info("[reload][%s] hersteld: side=%s, amt=%s @ %s, ATR=%.4f, master_id=%s",
                             symbol, side, str(amount), str(entry_price), float(atr_val), db_id)

