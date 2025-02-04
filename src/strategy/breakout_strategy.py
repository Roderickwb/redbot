import logging
import pandas as pd
import time
from decimal import Decimal, InvalidOperation
from typing import Optional
from datetime import datetime, timedelta, timezone

# TA-bibliotheken
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator

# Lokale imports
from src.logger.logger import setup_logger
# We halen NIET 'Market' binnen, want je hebt geen Market-class
from src.indicator_analysis.indicators import IndicatorAnalysis
from src.meltdown_manager.meltdown_manager import MeltdownManager

def is_candle_closed(candle_timestamp_ms: int, interval_hours: float) -> bool:
    """
    Bepaalt of een candle (gebaseerd op de starttijd in ms) volledig is afgesloten.
    De candle wordt geacht afgesloten als het huidige UTC-tijdstip ≥ candle start + interval.
    """
    candle_end = datetime.fromtimestamp(candle_timestamp_ms / 1000, tz=timezone.utc) + timedelta(hours=interval_hours)
    return datetime.now(timezone.utc) >= candle_end

class BreakoutStrategy:
    """
    Breakout Strategy (4h)
    ---------------------------------------------------------
    - Gebruikt Daily RSI(14) als trendfilter (rsi_bull_level / rsi_bear_level)
    - Op 4h:
      1) Bepaal 'range': highest high / lowest low (laatste X candles) + Bollinger-limieten
      2) Candle sluit >= 0.5% boven/onder die limiet (buffer)
      3) Volume-check (moet > 'volume_threshold_factor' x gemiddeld volume)
      4) Open positie => Stoploss = 1xATR, partial TP (25%) bij 1R, rest trailing
    - Max 90% van je kapitaal in posities, etc.
    """

    def __init__(self, client, db_manager, config_path=None):

        self.client = client
        self.db_manager = db_manager

        # 1) Config inladen
        if config_path:
            full_config = self._load_config_file(config_path)
            self.strategy_config = full_config.get("breakout_strategy", {})
        else:
            self.strategy_config = {}

        self.logger = setup_logger("breakout_strategy", "logs/breakout_strategy.log", logging.DEBUG)
        meltdown_cfg = self.strategy_config.get("meltdown_manager", {})
        self.meltdown_manager = MeltdownManager(meltdown_cfg, db_manager=self.db_manager, logger=self.logger)

        # Defaults (gebruik de waarden uit de config, of stel een standaard in)
        self.log_file = self.strategy_config.get("log_file", "logs/breakout_strategy.log")
        self.daily_timeframe = self.strategy_config.get("daily_timeframe", "1d")
        self.main_timeframe = self.strategy_config.get("main_timeframe", "4h")

        self.rsi_bull_level = float(self.strategy_config.get("rsi_bull_level", 55))
        self.rsi_bear_level = float(self.strategy_config.get("rsi_bear_level", 45))

        self.lookback_candles = int(self.strategy_config.get("lookback_candles", 20))
        self.breakout_buffer_pct = Decimal(str(self.strategy_config.get("breakout_buffer_pct", "0.5")))
        self.volume_threshold_factor = float(self.strategy_config.get("volume_threshold_factor", 1.2))

        self.atr_window = int(self.strategy_config.get("atr_window", 14))
        self.sl_atr_mult = Decimal(str(self.strategy_config.get("sl_atr_mult", "1.0")))
        self.trail_atr_mult = Decimal(str(self.strategy_config.get("trail_atr_mult", "1.0")))

        self.partial_tp_r = Decimal(str(self.strategy_config.get("partial_tp_r", "1.0")))
        self.partial_tp_pct = Decimal(str(self.strategy_config.get("partial_tp_pct", "0.25")))
        self.position_size_pct = Decimal(str(self.strategy_config.get("position_size_pct", "0.10")))
        self.max_positions_equity_pct = Decimal(str(self.strategy_config.get("max_positions_equity_pct", "0.90")))
        self.initial_capital = Decimal(str(self.strategy_config.get("initial_capital", "125")))

        self.logger = setup_logger("breakout_strategy", self.log_file, logging.DEBUG)
        if config_path:
            self.logger.info("[BreakoutStrategy] init with config_path=%s", config_path)
        else:
            self.logger.info("[BreakoutStrategy] init (no config_path)")

        self.open_positions = {}

    def _load_config_file(self, path: str) -> dict:
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return data

    def execute_strategy(self, symbol: str):
        meltdown_active = self.meltdown_manager.update_meltdown_state(strategy=self, symbol=symbol)
        if meltdown_active:
            self.logger.info("[Breakout] meltdown => skip trades & close pos.")
            return
        """
        1) Check daily RSI-filter
        2) Check breakout op 4h
        3) Open/Manage posities
        """
        self.logger.info(f"[BreakoutStrategy] Start for {symbol}")

        meltdown_active = self.meltdown_manager.update_meltdown_state(strategy=self, symbol=symbol)
        if meltdown_active:
            self.logger.info("[Breakout] meltdown => skip trades.")
            return

        # 1) Trend via daily RSI
        trend = self._check_daily_rsi(symbol)
        if trend == "neutral":
            self.logger.info(f"[Breakout] Trend = neutral => geen trades ({symbol})")
            return

        # 2) Detect breakout (gebaseerd op de laatst volledig afgesloten 4h candle)
        breakout_signal = self._detect_breakout(symbol, trend)
        if breakout_signal["breakout_detected"]:
            self.logger.info(
                f"[Breakout] {symbol} => breakout => side={breakout_signal['side']}, "
                f"price (last closed 4h candle)={breakout_signal['price']}, ATR={breakout_signal['atr']}, "
                f"limit_price={breakout_signal['limit_price']}"
            )
            if symbol not in self.open_positions:
                self._open_position(symbol, breakout_signal)
        else:
            # [CHANGED] - Log waarom geen breakout
            self.logger.info(f"[Breakout] {symbol} => geen breakout detected => skip")

        # 3) Manage open posities
        if symbol in self.open_positions:
            self._manage_position(symbol)

    def _check_daily_rsi(self, symbol: str) -> str:
        df_daily = self._fetch_and_indicator(symbol, self.daily_timeframe, limit=60)
        if df_daily.empty:
            self.logger.debug(f"[Breakout] {symbol} => geen daily data => neutral")
            return "neutral"

        latest_rsi = df_daily["rsi"].iloc[-1]
        # [CHANGED] - Log de RSI en de thresholds
        self.logger.info(f"[Breakout] {symbol} daily RSI={latest_rsi:.2f} (bull={self.rsi_bull_level}, bear={self.rsi_bear_level})")
        if latest_rsi >= self.rsi_bull_level:
            self.logger.debug(f"[Breakout] {symbol} => RSI >= bull_level => bull trend")
            return "bull"
        elif latest_rsi <= self.rsi_bear_level:
            self.logger.debug(f"[Breakout] {symbol} => RSI <= bear_level => bear trend")
            return "bear"
        else:
            return "neutral"

    def _detect_breakout(self, symbol: str, trend: str) -> dict:
        df_4h = self._fetch_and_indicator(symbol, self.main_timeframe, limit=200)
        # Zorg ervoor dat de data in oplopende volgorde staat (oudste eerst)
        df_4h.sort_values("timestamp", inplace=True)
        if len(df_4h) < self.lookback_candles:
            self.logger.debug(f"[Breakout] {symbol} => {len(df_4h)} candles < lookback={self.lookback_candles} => skip breakout check")
            return {"breakout_detected": False}

        # Bepaal het interval in uren (bijv. "4h" → 4)
        if self.main_timeframe.endswith("h"):
            interval_hours = int(self.main_timeframe[:-1])
        elif self.main_timeframe.endswith("d"):
            interval_hours = int(self.main_timeframe[:-1]) * 24
        elif self.main_timeframe.endswith("m"):
            interval_hours = int(self.main_timeframe[:-1]) / 60
        else:
            interval_hours = 4  # standaardwaarde

        recent_slice = df_4h.iloc[-self.lookback_candles:]
        hh = Decimal(str(recent_slice["high"].max()))
        ll = Decimal(str(recent_slice["low"].min()))
        upper_bb = Decimal(str(df_4h["bb_upper"].iloc[-1]))
        lower_bb = Decimal(str(df_4h["bb_lower"].iloc[-1]))

        if trend == "bull":
            breakout_limit = hh if hh < upper_bb else upper_bb
        else:  # bear
            breakout_limit = ll if ll > lower_bb else lower_bb

        # Gebruik de laatst beschikbare candle, maar controleer eerst of deze volledig is afgesloten
        last_candle_ts = int(df_4h["timestamp"].iloc[-1])
        if not is_candle_closed(last_candle_ts, interval_hours):
            self.logger.debug(f"[Breakout] {symbol} => Laatste candle (ts={last_candle_ts}) niet volledig afgesloten; gebruik de penultimate candle.")
            if len(df_4h) >= 2:
                last_4h_close = Decimal(str(df_4h["close"].iloc[-2]))
            else:
                last_4h_close = Decimal(str(df_4h["close"].iloc[-1]))
        else:
            last_4h_close = Decimal(str(df_4h["close"].iloc[-1]))

        # Haal de actuele prijs op (alleen voor logging; de strategie gebruikt de candle-close)
        current_price = self._get_latest_price(symbol)
        price_diff = current_price - last_4h_close
        self.logger.debug(f"[Breakout] {symbol} => last_4h_close={last_4h_close}, current_price={current_price} (diff={price_diff})")

        last_volume = Decimal(str(df_4h["volume"].iloc[-1]))
        avg_volume = Decimal(str(df_4h["volume"].tail(20).mean()))
        self.logger.debug(f"[Breakout] {symbol} => last_volume={last_volume}, avg_volume={avg_volume}")

        buffer_decimal = self.breakout_buffer_pct / Decimal("100")
        atr_val = self._calculate_atr(df_4h, self.atr_window)
        if not atr_val or atr_val <= 0:
            self.logger.debug(f"[Breakout] {symbol} => ATR=0 => skip breakout")
            return {"breakout_detected": False}

        if trend == "bull":
            required_price = breakout_limit * (Decimal("1") + buffer_decimal)
            self.logger.debug(f"[Breakout] {symbol} (bull) => breakout_limit={breakout_limit}, required_price={required_price}, using last_4h_close={last_4h_close}")
            if last_4h_close > required_price:
                if last_volume > (avg_volume * Decimal(str(self.volume_threshold_factor))):
                    return {"breakout_detected": True, "side": "buy", "price": last_4h_close, "atr": atr_val, "limit_price": breakout_limit}
                else:
                    self.logger.debug(f"[Breakout] {symbol} => volume check fail: {last_volume} <= {avg_volume} * factor")
        else:  # bear
            required_price = breakout_limit * (Decimal("1") - buffer_decimal)
            self.logger.debug(f"[Breakout] {symbol} (bear) => breakout_limit={breakout_limit}, required_price={required_price}, using last_4h_close={last_4h_close}")
            if last_4h_close < required_price:
                if last_volume > (avg_volume * Decimal(str(self.volume_threshold_factor))):
                    return {"breakout_detected": True, "side": "sell", "price": last_4h_close, "atr": atr_val, "limit_price": breakout_limit}
                else:
                    self.logger.debug(f"[Breakout] {symbol} => volume check fail: {last_volume} <= {avg_volume} * factor")
        return {"breakout_detected": False}

    def _open_position(self, symbol: str, breakout_signal: dict):
        side = breakout_signal["side"]
        price = breakout_signal["price"]
        atr_val = breakout_signal["atr"]

        if not self._can_open_new_position():
            self.logger.warning(f"[Breakout] {symbol} => Max belegd => skip")
            return

        eur_balance = self._get_eur_balance()
        trade_capital = eur_balance * self.position_size_pct
        if trade_capital < 5:
            self.logger.warning(f"[Breakout] {symbol} => te weinig capital => skip")
            return

        amount = trade_capital / price if price > 0 else Decimal("0")
        if amount <= 0:
            self.logger.warning(f"[Breakout] {symbol} => amount <= 0 => skip")
            return

        self.logger.info(
            f"[Breakout] _open_position => side={side}, price={price}, invest={trade_capital:.2f} EUR => amount={amount:.4f}"
        )

        if self.client:
            self.client.place_order(side, symbol, float(amount), order_type="market")
            self.logger.info(f"[LIVE] {side.upper()} {symbol}, amt={amount:.4f}, price={price:.2f}")
        else:
            self.logger.info(f"[Paper] {side.upper()} {symbol}, amt={amount:.4f}, price={price:.2f}")

        sl_dist = atr_val * self.sl_atr_mult
        if side == "buy":
            sl_price = price - sl_dist
        else:
            sl_price = price + sl_dist

        pos_info = {
            "side": side,
            "entry_price": price,
            "amount": amount,
            "stop_loss": sl_price,
            "atr": atr_val,
            "tp1_done": False,
            "partial_qty": amount * Decimal(str(self.partial_tp_pct)),
            "highest_price": price if side == "buy" else Decimal("0"),
            "lowest_price": price if side == "sell" else Decimal("999999")
        }
        self.open_positions[symbol] = pos_info
        self.logger.info(f"[Breakout] OPEN {side.upper()} {symbol} @ {price:.2f}, SL={sl_price:.2f}")

    def _manage_position(self, symbol: str):
        pos = self.open_positions.get(symbol)
        if not pos:
            return

        side = pos["side"]
        entry = pos["entry_price"]
        amount = pos["amount"]
        sl_price = pos["stop_loss"]
        current_price = self._get_latest_price(symbol)
        if current_price <= 0:
            return

        # [CHANGED] - Extra debug
        self.logger.debug(f"[Breakout-manage] symbol={symbol}, side={side}, entry={entry}, curr={current_price}, SL={sl_price}")

        # Stop-loss check
        if side == "buy":
            if current_price <= sl_price:
                self.logger.info(f"[Breakout] SL geraakt => close LONG {symbol}")
                self._close_position(symbol)
                return
        else:  # sell
            if current_price >= sl_price:
                self.logger.info(f"[Breakout] SL geraakt => close SHORT {symbol}")
                self._close_position(symbol)
                return

        # Partial TP bij 1R
        if not pos["tp1_done"]:
            if side == "buy":
                risk = entry - sl_price
                tp1 = entry + risk
                self.logger.debug(f"[Breakout-manage] {symbol} => LONG risk={risk:.4f}, tp1={tp1:.4f}")
                if current_price >= tp1:
                    self.logger.info(f"[Breakout] {symbol} => 1R target => take partial LONG => {pos['partial_qty']:.4f}")
                    self._take_partial_profit(symbol, pos["partial_qty"], "sell")
                    pos["tp1_done"] = True
            else:
                risk = sl_price - entry
                tp1 = entry - risk
                self.logger.debug(f"[Breakout-manage] {symbol} => SHORT risk={risk:.4f}, tp1={tp1:.4f}")
                if current_price <= tp1:
                    self.logger.info(f"[Breakout] {symbol} => 1R target => take partial SHORT => {pos['partial_qty']:.4f}")
                    self._take_partial_profit(symbol, pos["partial_qty"], "buy")
                    pos["tp1_done"] = True

        # Trailing stop
        atr_val = pos["atr"]
        trail_dist = atr_val * self.trail_atr_mult
        if side == "buy":
            if current_price > pos["highest_price"]:
                pos["highest_price"] = current_price
            new_sl = pos["highest_price"] - trail_dist
            if new_sl > pos["stop_loss"]:
                old_sl = pos["stop_loss"]
                pos["stop_loss"] = new_sl
                self.logger.info(f"[Breakout] {symbol} => update trailing SL => old={old_sl:.2f}, new={new_sl:.2f}")
        else:
            if current_price < pos["lowest_price"]:
                pos["lowest_price"] = current_price
            new_sl = pos["lowest_price"] + trail_dist
            if new_sl < pos["stop_loss"]:
                old_sl = pos["stop_loss"]
                pos["stop_loss"] = new_sl
                self.logger.info(f"[Breakout] {symbol} => update trailing SL => old={old_sl:.2f}, new={new_sl:.2f}")

    def _close_position(self, symbol: str):
        pos = self.open_positions.get(symbol)
        if not pos:
            return
        side = pos["side"]
        amt = pos["amount"]

        if side == "buy":
            if self.client:
                self.client.place_order("sell", symbol, float(amt), order_type="market")
                self.logger.info(f"[LIVE] CLOSE LONG => SELL {symbol} amt={amt:.4f}")
            else:
                self.logger.info(f"[Paper] CLOSE LONG => SELL {symbol} amt={amt:.4f}")
        else:
            if self.client:
                self.client.place_order("buy", symbol, float(amt), order_type="market")
                self.logger.info(f"[LIVE] CLOSE SHORT => BUY {symbol} amt={amt:.4f}")
            else:
                self.logger.info(f"[Paper] CLOSE SHORT => BUY {symbol} amt={amt:.4f}")

        del self.open_positions[symbol]
        self.logger.info(f"[Breakout] Positie {symbol} volledig gesloten.")

    def _take_partial_profit(self, symbol: str, qty: Decimal, exit_side: str):
        """
        exit_side="sell" => partial exit voor een LONG
        exit_side="buy"  => partial exit voor een SHORT
        """
        if symbol not in self.open_positions:
            return
        pos = self.open_positions[symbol]
        current_price = self._get_latest_price(symbol)

        if self.client:
            self.client.place_order(exit_side, symbol, float(qty), order_type="market")
            self.logger.info(f"[LIVE] PARTIAL EXIT => {exit_side.upper()} {qty:.4f} {symbol} @ ~{current_price:.2f}")
        else:
            self.logger.info(f"[Paper] PARTIAL EXIT => {exit_side.upper()} {qty:.4f} {symbol} @ ~{current_price:.2f}")

        pos["amount"] -= qty
        if pos["amount"] <= Decimal("0"):
            self.logger.info(f"[Breakout] Positie uitgeput => {symbol} closed")
            del self.open_positions[symbol]

    def _fetch_and_indicator(self, symbol: str, interval: str, limit=200) -> pd.DataFrame:
        """
        Haalt de candles uit 'candles_kraken' via db_manager.fetch_data("candles_kraken", ...).
        Daarna berekent het RSI en Bollinger, net als je oorspronkelijke code.
        Dit is i.p.v. de (niet-bestaande) Market-class.
        """
        # 1) Query de 'candles_kraken' tabel
        df = self.db_manager.fetch_data(
            table_name="candles_kraken",
            limit=limit,
            market=symbol,
            interval=interval,
            exchange=None  # Of je laat 'exchange=Kraken' weg, want we filteren al op 'candles_kraken'
        )

        if df.empty:
            return pd.DataFrame()

        # We verwachten kolommen:
        # timestamp, datetime_utc, market, interval, open, high, low, close, volume, exchange(?)
        # In 'candles_kraken' heb je misschien wel/niet 'exchange' kolom, maar we laten het zoals is:
        # Hernoem en sorteer net als je deed in je code
        # (let op, fetch_data() geeft misschien al columns=[...], check wat je db_manager.fetch_data() doet)

        # Voor de zekerheid even checken of df de kolommen heeft die we nodig hebben:
        needed_cols = ["timestamp","market","interval","open","high","low","close","volume"]
        existing_cols = df.columns.tolist()
        for col in needed_cols:
            if col not in existing_cols:
                self.logger.debug(f"[Breakout] {symbol} => kolom {col} niet in df.columns => might cause issues")

        # Sorteer oplopend
        df.sort_values(by="timestamp", inplace=True)

        # Cast numeric
        for col in ["open","high","low","close","volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float, errors="ignore")

        # Indicator berekening
        df = IndicatorAnalysis.calculate_indicators(df, rsi_window=14)
        bb = IndicatorAnalysis.calculate_bollinger_bands(df, window=20, num_std_dev=2)
        df["bb_upper"] = bb["bb_upper"]
        df["bb_lower"] = bb["bb_lower"]

        return df

    def _get_latest_price(self, symbol: str) -> Decimal:
        """
        Haalt de recentste prijs uit 'ticker_kraken' d.m.v. db_manager.fetch_data("ticker_kraken", ...).
        Fallback => 1m-candle in 'candles_kraken'.
        """
        df_ticker = self.db_manager.fetch_data(
            table_name="ticker_kraken",
            limit=1,
            market=symbol
            # exchange=... (kun je doen of laten, want we filteren op de 'ticker_kraken' tabel)
        )
        if not df_ticker.empty:
            best_bid = df_ticker["best_bid"].iloc[0] if "best_bid" in df_ticker.columns else 0
            best_ask = df_ticker["best_ask"].iloc[0] if "best_ask" in df_ticker.columns else 0
            if best_bid>0 and best_ask>0:
                return Decimal(str((best_bid+best_ask)/2))

        # Fallback => 1m in 'candles_kraken'
        df_1m = self.db_manager.fetch_data(
            table_name="candles_kraken",
            limit=1,
            market=symbol,
            interval="1m"
        )
        if not df_1m.empty and "close" in df_1m.columns:
            last_close = df_1m["close"].iloc[0]
            return Decimal(str(last_close))

        return Decimal("0")

    def _calculate_atr(self, df: pd.DataFrame, window: int) -> Optional[Decimal]:
        if len(df) < window:
            return None
        atr_obj = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=window
        )
        val = atr_obj.average_true_range().iloc[-1]
        if pd.isna(val):
            return None
        return Decimal(str(val))

    def _get_equity_estimate(self) -> Decimal:
        if not self.client:
            return self.initial_capital

        bal = self.client.get_balance()
        eur_balance = Decimal(str(bal.get("EUR", "0")))

        total_positions = Decimal("0")
        for sym, pos in self.open_positions.items():
            amt = pos["amount"]
            px = self._get_latest_price(sym)
            total_positions += amt * px
        return eur_balance + total_positions

    def _get_eur_balance(self) -> Decimal:
        if not self.client:
            return self.initial_capital
        bal = self.client.get_balance()
        return Decimal(str(bal.get("EUR", "0")))

    def _can_open_new_position(self) -> bool:
        total_equity = self._get_equity_estimate()
        eur_balance = self._get_eur_balance()
        invested = total_equity - eur_balance
        ratio = invested / total_equity if total_equity > 0 else Decimal("0")
        return ratio < self.max_positions_equity_pct
