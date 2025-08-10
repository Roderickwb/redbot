# ============================================================
# src/strategy/trend_strategy_4h.py
# ============================================================
import logging
import time
from decimal import Decimal
from typing import Optional

import pandas as pd
from ta.trend import ADXIndicator
from ta.trend import EMAIndicator
from ta.trend import MACD  # alleen voor logging/diagnose
from ta.volatility import AverageTrueRange

from src.logger.logger import setup_logger
from src.config.config import load_config_file
from src.indicator_analysis.indicators import IndicatorAnalysis


def _is_candle_closed(ts_ms: int) -> bool:
    return int(time.time() * 1000) >= ts_ms


class TrendStrategy4H:
    """
    Trend-volgende strategy:
      - Trend-TF: 4h (richtingsbepaling)
      - Entry-TF: 1h (timing + risk ATR)
      - Risk: ATR(atr_timeframe_for_risk), SL/TP/trailing zoals pullback
      - Trading modes: watch | dryrun | auto
    """

    def __init__(self, data_client, order_client, db_manager, config_path: Optional[str] = None):
        # IO
        self.data_client = data_client
        self.order_client = order_client
        self.db_manager = db_manager

        # Config
        full_cfg = load_config_file(config_path) if config_path else {}
        self.cfg = full_cfg.get("trend_strategy_4h", {})

        # Flags & params
        self.enabled = bool(self.cfg.get("enabled", False))
        self.trading_mode = self.cfg.get("trading_mode", "watch")  # watch|dryrun|auto
        self.strategy_name = "trend"

        # Timeframes
        self.trend_tf = self.cfg.get("trend_timeframe", "4h")
        self.entry_tf = self.cfg.get("entry_timeframe", "1h")
        self.atr_tf = self.cfg.get("atr_timeframe_for_risk", self.entry_tf)

        # Indicators / filters
        self.use_adx_filter = bool(self.cfg.get("use_adx_filter", True))
        self.use_adx_directional_filter = bool(self.cfg.get("use_adx_directional_filter", True))
        self.use_adx_multitimeframe = bool(self.cfg.get("use_adx_multitimeframe", True))

        self.adx_window = int(self.cfg.get("adx_window", 14))
        self.adx_entry_tf_threshold = float(self.cfg.get("adx_entry_tf_threshold", 20.0))
        self.adx_high_tf_threshold = float(self.cfg.get("adx_high_tf_threshold", 20.0))

        self.ema_fast = int(self.cfg.get("ema_fast", 20))
        self.ema_slow = int(self.cfg.get("ema_slow", 50))
        self.require_trend_stack = bool(self.cfg.get("require_trend_stack", True))

        # Risk (identiek aan pullback flow)
        self.atr_window = int(self.cfg.get("atr_window", 14))
        self.sl_atr_mult = Decimal(str(self.cfg.get("sl_atr_mult", 1.5)))
        self.tp1_atr_mult = Decimal(str(self.cfg.get("tp1_atr_mult", 1.5)))
        self.tp1_portion_pct = Decimal(str(self.cfg.get("tp1_portion_pct", 0.50)))
        self.trailing_atr_mult = Decimal(str(self.cfg.get("trailing_atr_mult", 1.0)))

        self.min_lot_multiplier = Decimal(str(self.cfg.get("min_lot_multiplier", 2.1)))
        self.max_position_pct = Decimal(str(self.cfg.get("max_position_pct", 0.05)))
        self.max_position_eur = Decimal(str(self.cfg.get("max_position_eur", 15)))
        self.initial_capital = Decimal(str(self.cfg.get("initial_capital", 350)))

        # Logging
        self.log_file = self.cfg.get("log_file", "logs/trend_strategy_4h.log")
        self.logger = setup_logger("trend_strategy_4h", self.log_file, logging.INFO)
        self.logger.info("[TrendStrategy4H] initialised (enabled=%s)", self.enabled)

        # State
        self.open_positions = {}  # {symbol: {...}}
        self.last_processed_candle_ts = {}  # per symbol (entry TF)

    # --------------- public API ---------------

    def execute_strategy(self, symbol: str):
        if not self.enabled:
            return

        # 1) Data ophalen
        df_trend = self._fetch_and_indicators(symbol, self.trend_tf, limit=200)
        if df_trend.empty:
            self.logger.warning(f"[{symbol}] Geen {self.trend_tf} data.")
            return

        df_entry = self._fetch_and_indicators(symbol, self.entry_tf, limit=200)
        if df_entry.empty:
            self.logger.warning(f"[{symbol}] Geen {self.entry_tf} data.")
            return

        # 2) Candle close guard (entry TF)
        last_ts = int(df_entry["timestamp_ms"].iloc[-1])
        if not _is_candle_closed(last_ts):
            self.logger.debug(f"[{symbol}] {self.entry_tf} candle nog open => skip nu.")
            return

        prev_ts = self.last_processed_candle_ts.get(symbol)
        if prev_ts == last_ts:
            self.logger.debug(f"[{symbol}] {self.entry_tf} candle {last_ts} al verwerkt => skip.")
            return
        self.last_processed_candle_ts[symbol] = last_ts

        # 3) Trend & setup bepalen
        setup = self._build_setup(symbol, df_trend, df_entry)
        if not setup["ok"]:
            self.logger.info(f"[SKIP][{symbol}] {setup['reason']}")
            return

        side = setup["side"]  # "buy" of "sell"

        # 4) Risk: ATR op gewenste TF (default entry TF)
        df_risk = df_entry if (self.atr_tf == self.entry_tf) else self._fetch_and_indicators(symbol, self.atr_tf, 200)
        atr_value = self._calc_atr(df_risk, self.atr_window)
        if atr_value is None or atr_value <= 0:
            self.logger.warning(f"[{symbol}] ATR({self.atr_tf}) niet beschikbaar => skip.")
            return

        # 5) Huidige prijs
        current_price = self._get_price(symbol)
        if current_price <= 0:
            self.logger.warning(f"[{symbol}] current_price=0 => skip.")
            return

        # 6) Mode: watch/dryrun/auto
        if self.trading_mode == "watch":
            self._log_setup(symbol, setup, atr_value, current_price)
            return

        # 7) Order plaatsen (dryrun=DB only, auto=ook exchange)
        self._open_position(symbol, side, Decimal(str(current_price)), Decimal(str(atr_value)))

    def manage_intra_candle_exits(self):
        """
        Elke ~exit_check_intra_seconds prijs checken vs SL/TP/trailing, exact als pullback.
        (Executor roept deze periodiek aan.)
        """
        for sym, pos in list(self.open_positions.items()):
            current = Decimal(str(self._get_price(sym)))
            if current <= 0:
                continue
            self._manage_open_position(sym, current, pos["atr"])

    # --------------- intern ---------------

    def _build_setup(self, symbol: str, df_trend: pd.DataFrame, df_entry: pd.DataFrame) -> dict:
        """
        Bepaalt LONG/SHORT setup o.b.v. EMA-stack + ADX filters.
        """
        # Trend-TF (4h)
        ema_fast_tr = df_trend["ema_fast"].iloc[-1]
        ema_slow_tr = df_trend["ema_slow"].iloc[-1]
        adx_tr = df_trend["adx"].iloc[-1] if "adx" in df_trend.columns else None

        # Entry-TF (1h)
        ema_fast_en = df_entry["ema_fast"].iloc[-1]
        ema_slow_en = df_entry["ema_slow"].iloc[-1]
        adx_en = df_entry["adx"].iloc[-1] if "adx" in df_entry.columns else None
        di_pos = df_entry["di_pos"].iloc[-1] if "di_pos" in df_entry.columns else None
        di_neg = df_entry["di_neg"].iloc[-1] if "di_neg" in df_entry.columns else None

        # Richting
        bull_stack_tr = ema_fast_tr > ema_slow_tr
        bear_stack_tr = ema_fast_tr < ema_slow_tr
        bull_stack_en = ema_fast_en > ema_slow_en
        bear_stack_en = ema_fast_en < ema_slow_en

        # ADX MTFA
        if self.use_adx_multitimeframe and adx_tr is not None and adx_tr < self.adx_high_tf_threshold:
            return {"ok": False, "reason": f"4h ADX {adx_tr:.1f} < {self.adx_high_tf_threshold}"}

        # Entry ADX
        if self.use_adx_filter and adx_en is not None and adx_en < self.adx_entry_tf_threshold:
            return {"ok": False, "reason": f"1h ADX {adx_en:.1f} < {self.adx_entry_tf_threshold}"}

        # Directional filter
        if self.use_adx_directional_filter and (di_pos is not None) and (di_neg is not None):
            di_ok_long = di_pos > di_neg
            di_ok_short = di_neg > di_pos
        else:
            di_ok_long = di_ok_short = True

        # Stack‑eis
        if self.require_trend_stack:
            long_ok = bull_stack_tr and bull_stack_en and di_ok_long
            short_ok = bear_stack_tr and bear_stack_en and di_ok_short
        else:
            long_ok = di_ok_long
            short_ok = di_ok_short

        if long_ok:
            return {"ok": True, "side": "buy", "reason": "bull stack + ADX ok",
                    "diag": {"adx4h": adx_tr, "adx1h": adx_en, "di+": di_pos, "di-": di_neg,
                             "ema_tr": (ema_fast_tr, ema_slow_tr), "ema_en": (ema_fast_en, ema_slow_en)}}
        if short_ok:
            return {"ok": True, "side": "sell", "reason": "bear stack + ADX ok",
                    "diag": {"adx4h": adx_tr, "adx1h": adx_en, "di+": di_pos, "di-": di_neg,
                             "ema_tr": (ema_fast_tr, ema_slow_tr), "ema_en": (ema_fast_en, ema_slow_en)}}

        return {"ok": False, "reason": "geen stack +/‑ of DI filter faalt"}

    # ------- position mgmt (in lijn met pullback) -------

    def _open_position(self, symbol: str, side: str, current_price: Decimal, atr_value: Decimal):
        # Position sizing (zelfde regels als pullback)
        equity_now = self._get_equity_estimate()
        allowed_eur_pct = equity_now * self.max_position_pct
        allowed_eur = min(allowed_eur_pct, self.max_position_eur)

        min_lot = self._get_min_lot(symbol) * self.min_lot_multiplier
        needed_eur_for_min = min_lot * current_price
        if needed_eur_for_min > allowed_eur:
            self.logger.info(f"[{symbol}] needed={needed_eur_for_min:.2f} > allowed={allowed_eur:.2f} => skip.")
            return

        buy_eur = needed_eur_for_min
        amount = buy_eur / current_price

        # Exchange order (dryrun => niet sturen)
        if self.trading_mode == "auto" and self.order_client:
            try:
                self.order_client.place_order(side, symbol, float(amount))
            except Exception as e:
                self.logger.warning(f"[{symbol}] order failed: {e}")
                return

        # DB: master trade
        position_id = f"{symbol}-{int(time.time())}"
        trade_data = {
            "symbol": symbol,
            "side": side,
            "amount": float(amount),
            "price": float(current_price),
            "timestamp": int(time.time() * 1000),
            "position_id": position_id,
            "position_type": "long" if side == "buy" else "short",
            "status": "open",
            "pnl_eur": 0.0,
            "fees": 0.0,
            "trade_cost": float(buy_eur),
            "strategy_name": self.strategy_name,
            "is_master": 1
        }
        self.db_manager.save_trade(trade_data)
        master_id = self.db_manager.cursor.lastrowid

        # Pos state
        self.open_positions[symbol] = {
            "side": side,
            "entry_price": current_price,
            "amount": Decimal(str(amount)),
            "atr": atr_value,
            "tp1_done": False,
            "trail_active": False,
            "trail_high": current_price,
            "position_id": position_id,
            "position_type": trade_data["position_type"],
            "master_id": master_id
        }

        self.logger.info(f"[OPEN][{symbol}] {side} @ {current_price} | ATR({self.atr_tf})={atr_value}")

    def _manage_open_position(self, symbol: str, current_price: Decimal, atr_value: Decimal):
        if symbol not in self.open_positions:
            return
        pos = self.open_positions[symbol]
        side = pos["side"]
        entry = pos["entry_price"]
        amount = pos["amount"]
        one_r = atr_value

        if side == "buy":
            sl = entry - (one_r * self.sl_atr_mult)
            tp1 = entry + (one_r * self.tp1_atr_mult)
            # SL
            if current_price <= sl:
                self._close_all(symbol, reason="StopLoss", exec_price=current_price)
                return
            # TP1
            if (not pos["tp1_done"]) and (current_price >= tp1):
                self._partial_close(symbol, portion=self.tp1_portion_pct, reason="TP1", exec_price=current_price)
                pos["tp1_done"] = True
                pos["trail_active"] = True
                pos["trail_high"] = max(pos["trail_high"], current_price)
            # Trailing
            if pos["trail_active"]:
                if current_price > pos["trail_high"]:
                    pos["trail_high"] = current_price
                tstop = pos["trail_high"] - (one_r * self.trailing_atr_mult)
                if current_price <= tstop:
                    self._close_all(symbol, reason="TrailingStop", exec_price=current_price)
                    return

        else:  # SHORT
            sl = entry + (one_r * self.sl_atr_mult)
            tp1 = entry - (one_r * self.tp1_atr_mult)
            if current_price >= sl:
                self._close_all(symbol, reason="StopLoss", exec_price=current_price)
                return
            if (not pos["tp1_done"]) and (current_price <= tp1):
                self._partial_close(symbol, portion=self.tp1_portion_pct, reason="TP1", exec_price=current_price)
                pos["tp1_done"] = True
                pos["trail_active"] = True
                pos["trail_high"] = current_price
            if pos["trail_active"]:
                if current_price < pos["trail_high"]:
                    pos["trail_high"] = current_price
                tstop = pos["trail_high"] + (one_r * self.trailing_atr_mult)
                if current_price >= tstop:
                    self._close_all(symbol, reason="TrailingStop", exec_price=current_price)
                    return

        # leftover housekeeping
        if pos["amount"] <= 0 or pos["amount"] < self._get_min_lot(symbol):
            self.db_manager.update_trade(pos["master_id"], {"status": "closed"})
            del self.open_positions[symbol]

    def _partial_close(self, symbol: str, portion: Decimal, reason: str, exec_price: Decimal):
        pos = self.open_positions[symbol]
        amt = pos["amount"] * portion
        if amt <= 0:
            return
        # Exchange order
        if self.trading_mode == "auto" and self.order_client:
            side = "sell" if pos["side"] == "buy" else "buy"
            self.order_client.place_order(side, symbol, float(amt), ordertype="market")

        # Child trade
        child = {
            "symbol": symbol,
            "side": "sell" if pos["side"] == "buy" else "buy",
            "amount": float(amt),
            "price": float(exec_price),
            "timestamp": int(time.time() * 1000),
            "position_id": pos["position_id"],
            "position_type": pos["position_type"],
            "status": "partial",
            "pnl_eur": 0.0,
            "fees": 0.0,
            "trade_cost": float(exec_price * amt),
            "strategy_name": self.strategy_name,
            "is_master": 0
        }
        self.db_manager.save_trade(child)
        pos["amount"] -= amt

    def _close_all(self, symbol: str, reason: str, exec_price: Decimal):
        pos = self.open_positions.get(symbol)
        if not pos:
            return
        amt = pos["amount"]
        if amt > 0 and self.trading_mode == "auto" and self.order_client:
            side = "sell" if pos["side"] == "buy" else "buy"
            self.order_client.place_order(side, symbol, float(amt), ordertype="market")

        # child trade
        child = {
            "symbol": symbol,
            "side": "sell" if pos["side"] == "buy" else "buy",
            "amount": float(amt),
            "price": float(exec_price),
            "timestamp": int(time.time() * 1000),
            "position_id": pos["position_id"],
            "position_type": pos["position_type"],
            "status": "closed",
            "pnl_eur": 0.0,
            "fees": 0.0,
            "trade_cost": float(exec_price * amt),
            "strategy_name": self.strategy_name,
            "is_master": 0
        }
        self.db_manager.save_trade(child)
        # master afsluiten
        self.db_manager.update_trade(pos["master_id"], {"status": "closed"})
        del self.open_positions[symbol]
        self.logger.info(f"[CLOSE][{symbol}] {reason}")

    # ------- data/helpers -------

    def _fetch_and_indicators(self, symbol: str, interval: str, limit=200) -> pd.DataFrame:
        df = self.db_manager.fetch_data(
            table_name="candles_kraken",
            limit=limit,
            market=symbol,
            interval=interval
        )
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        # normaliseer kolommen (zelfde patroon als bij pullback)
        for col in ['datetime_utc', 'exchange']:
            if col in df.columns:
                df.drop(columns=col, inplace=True, errors='ignore')
        df.columns = ["timestamp_ms", "market", "interval", "open", "high", "low", "close", "volume"]
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float, errors="raise")
        df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df.set_index("datetime_utc", inplace=True, drop=False)
        df.sort_index(inplace=True)

        # basis indicators (RSI, BB) voor eventuele logging/diagnose
        df = IndicatorAnalysis.calculate_indicators(df, rsi_window=14)

        # EMA stack
        try:
            df["ema_fast"] = EMAIndicator(df["close"], window=self.ema_fast).ema_indicator()
            df["ema_slow"] = EMAIndicator(df["close"], window=self.ema_slow).ema_indicator()
        except Exception:
            df["ema_fast"] = df["close"]
            df["ema_slow"] = df["close"]

        # ADX & DI
        try:
            adx_obj = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=self.adx_window)
            df["adx"] = adx_obj.adx()
            df["di_pos"] = adx_obj.adx_pos()
            df["di_neg"] = adx_obj.adx_neg()
        except Exception:
            pass

        # MACD alleen voor logging (optioneel)
        try:
            macd = MACD(close=df["close"])
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
        except Exception:
            pass

        return df

    def _calc_atr(self, df: pd.DataFrame, window=14) -> Optional[Decimal]:
        if df.empty or len(df) < window:
            return None
        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=window).average_true_range().iloc[-1]
        if pd.isna(atr):
            return None
        return Decimal(str(atr))

    def _get_price(self, symbol: str) -> Decimal:
        if not self.data_client:
            return Decimal("0")
        px = self.data_client.get_latest_ws_price(symbol)
        if px and px > 0:
            return Decimal(str(px))
        # fallback (laatste 1m close uit DB)
        df_1m = self.db_manager.fetch_data("candles_kraken", limit=1, market=symbol, interval="1m")
        if not df_1m.empty and "close" in df_1m.columns:
            return Decimal(str(df_1m["close"].iloc[0]))
        return Decimal("0")

    def _get_min_lot(self, symbol: str) -> Decimal:
        if not self.data_client:
            return Decimal("1.0")
        return Decimal(str(self.data_client.get_min_lot(symbol)))

    def _get_equity_estimate(self) -> Decimal:
        if not self.order_client:
            return self.initial_capital
        bal = self.order_client.get_balance()
        total = Decimal("0")
        for asset, amt in bal.items():
            amt = Decimal(str(amt))
            if asset.upper() == "EUR":
                total += amt
            else:
                sym = f"{asset.upper()}-EUR"
                px = self._get_price(sym)
                if px > 0:
                    total += (amt * px)
        return total

    def _log_setup(self, symbol: str, setup: dict, atr: Decimal, price: Decimal):
        d = setup.get("diag", {})
        self.logger.info(
            f"[SETUP][{symbol}] side={setup.get('side','-')} | reason={setup.get('reason','')} | "
            f"4h(adx={d.get('adx4h')}, ema={d.get('ema_tr')}) | 1h(adx={d.get('adx1h')}, di+={d.get('di+')}, di-={d.get('di-')}, ema={d.get('ema_en')}) | "
            f"ATR({self.atr_tf})={atr} | px={price}"
        )
