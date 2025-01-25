import logging
import pandas as pd
import yaml
import time
from typing import Optional
from decimal import Decimal, InvalidOperation
from collections import deque

# TA-bibliotheken
from ta.volatility import AverageTrueRange
from ta.trend import MACD

# Lokale imports
from src.config.config import PULLBACK_STRATEGY_LOG_FILE, PULLBACK_CONFIG, load_config_file
from src.logger.logger import setup_logger
from src.indicator_analysis.indicators import Market, IndicatorAnalysis

try:
    import joblib
except ImportError:
    joblib = None


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    print("[DEBUG] In load_config =>", data)
    return data


class PullbackAccumulateStrategy:
    """
    Pullback & Accumulate Strategy
    ---------------------------------------------------------
     - self.data_client => om de laatste koersen te zien (bv. .latest_prices)
     - self.order_client => om orders te plaatsen en get_balance() te doen
    """

    # CHANGED: constructor heeft data_client en order_client
    def __init__(self, data_client, order_client, db_manager, config_path=None):
        self.data_client = data_client
        self.order_client = order_client
        self.db_manager = db_manager

        # 1) Config inladen
        if config_path:
            full_config = load_config_file(config_path)
            self.strategy_config = full_config.get("pullback_accumulate_strategy", {})
        else:
            self.strategy_config = PULLBACK_CONFIG

        # Logger
        self.logger = setup_logger("pullback_strategy", PULLBACK_STRATEGY_LOG_FILE, logging.DEBUG)
        if config_path:
            self.logger.info("[PullbackAccumulateStrategy] init with config_path=%s", config_path)
        else:
            self.logger.info("[PullbackAccumulateStrategy] init (no config_path)")

        # -----------------------------------
        # Hier de rest van je config uit YAML
        # -----------------------------------
        self.pullback_rolling_window = int(self.strategy_config.get("pullback_rolling_window", 20))
        self.daily_timeframe = self.strategy_config.get("daily_timeframe", "1d")
        self.trend_timeframe = self.strategy_config.get("trend_timeframe", "4h")
        self.main_timeframe = self.strategy_config.get("main_timeframe", "1h")
        # Entry-timeframe: 15m
        self.entry_timeframe = self.strategy_config.get("entry_timeframe", "15m")
        self.flash_crash_tf = self.strategy_config.get("flash_crash_timeframe", "5m")

        self.pullback_threshold_pct = Decimal(str(self.strategy_config.get("pullback_threshold_pct", "0.5")))

        # Overige settings
        self.accumulate_threshold = Decimal(str(self.strategy_config.get("accumulate_threshold", "1.25")))
        self.position_size_pct = Decimal(str(self.strategy_config.get("position_size_pct", "0.05")))
        self.tp1_atr_mult = Decimal(str(self.strategy_config.get("tp1_atr_mult", "1.0")))
        self.tp2_atr_mult = Decimal(str(self.strategy_config.get("tp2_atr_mult", "2.0")))
        self.trail_atr_mult = Decimal(str(self.strategy_config.get("trail_atr_mult", "1.0")))
        self.atr_window = int(self.strategy_config.get("atr_window", 14))
        self.initial_capital = Decimal(str(self.strategy_config.get("initial_capital", "100")))
        self.log_file = self.strategy_config.get("log_file", PULLBACK_STRATEGY_LOG_FILE)

        # Indicator-drempels (voor RSI & MACD) uit config
        self.rsi_bull_threshold = float(self.strategy_config.get("rsi_bull_threshold", 55))
        self.rsi_bear_threshold = float(self.strategy_config.get("rsi_bear_threshold", 45))
        self.macd_bull_threshold = float(self.strategy_config.get("macd_bull_threshold", 0))
        self.macd_bear_threshold = float(self.strategy_config.get("macd_bear_threshold", 0))
        self.depth_threshold_bull = float(self.strategy_config.get("depth_threshold_bull", 0.0))
        self.depth_threshold_bear = float(self.strategy_config.get("depth_threshold_bear", 0.0))
        self.daily_bull_rsi = float(self.strategy_config.get("daily_bull_rsi", 60))
        self.daily_bear_rsi = float(self.strategy_config.get("daily_bear_rsi", 40))
        self.h4_bull_rsi = float(self.strategy_config.get("h4_bull_rsi", 50))
        self.h4_bear_rsi = float(self.strategy_config.get("h4_bear_rsi", 50))

        # Fail-safes
        self.max_daily_loss_pct = Decimal(str(self.strategy_config.get("max_daily_loss_pct", 5.0)))
        self.flash_crash_drop_pct = Decimal(str(self.strategy_config.get("flash_crash_drop_pct", 10.0)))
        self.use_depth_trend = bool(self.strategy_config.get("use_depth_trend", True))

        # RSI/MACD config
        self.rsi_window = int(self.strategy_config.get("rsi_window", 14))
        self.macd_fast = int(self.strategy_config.get("macd_fast", 12))
        self.macd_slow = int(self.strategy_config.get("macd_slow", 26))
        self.macd_signal = int(self.strategy_config.get("macd_signal", 9))
        self.pivot_points_window = int(self.strategy_config.get("pivot_points_window", 20))

        # ML
        self.ml_model_enabled = bool(self.strategy_config.get("ml_model_enabled", False))
        self.ml_model_path = self.strategy_config.get("ml_model_path", "models/pullback_model.pkl")
        self.ml_engine = None

        # Logger
        self.logger = setup_logger("pullback_strategy", self.log_file, logging.DEBUG)
        if config_path:
            self.logger.info("[PullbackAccumulateStrategy] init with config_path=%s", config_path)
        else:
            self.logger.info("[PullbackAccumulateStrategy] init (no config_path)")

        # Posities & vlag
        self.open_positions = {}
        self.invested_extra = False

        # DepthTrend rolling average
        self.depth_trend_history = deque(maxlen=5)

    # ----------------------------------------------------------------
    # Fees & PnL
    # ----------------------------------------------------------------
    def _calculate_fees_and_pnl(self, side: str, amount: float, price: float, reason: str) -> (float, float):
        trade_cost = amount * price
        fees = 0.0025 * trade_cost
        # Simpele placeholder:
        if reason.startswith("TP") or reason == "TrailingStop":
            realized_pnl = trade_cost - fees
        elif side.lower() == "sell":
            realized_pnl = trade_cost - fees
        elif side.lower() == "buy" and reason in ("TP1", "TP2", "TrailingStop"):
            realized_pnl = trade_cost - fees
        else:
            realized_pnl = 0.0
        return fees, realized_pnl

    # ----------------------------------------------------------------
    # Hoofdstrategie
    # ----------------------------------------------------------------
    def execute_strategy(self, symbol: str):
        """
         1) Fail-safes
         2) Trend (daily + h4)
         3) ATR (h1)
         4) Pullback (15m)
         5) Depth Trend
         6) Manage / open pos
        """
        self.logger.info(f"[PullbackStrategy] Start for {symbol}")

        # (1) fail-safes
        if self._check_fail_safes(symbol):
            self.logger.warning(f"[PullbackStrategy] Fail-safe => skip trading {symbol}")
            return

        # (2) Trend => daily + H4
        df_daily = self._fetch_and_indicator(symbol, self.daily_timeframe, limit=60)
        df_h4 = self._fetch_and_indicator(symbol, self.trend_timeframe, limit=200)
        if df_daily.empty or df_h4.empty:
            self.logger.warning(f"[PullbackStrategy] No daily/H4 data => skip {symbol}")
            return

        rsi_daily = df_daily["rsi"].iloc[-1]
        rsi_h4 = df_h4["rsi"].iloc[-1]
        self.logger.info(f"[Daily/H4 RSI] symbol={symbol}, rsi_daily={rsi_daily:.2f}, rsi_h4={rsi_h4:.2f}")
        direction = self._check_trend_direction(df_daily, df_h4)
        self.logger.info(f"[PullbackStrategy] Direction={direction} for {symbol}")

        # AANPASSING #1: extra debug pivot
        pivot_levels = self._calculate_pivot_points(df_h4)
        self.logger.debug(f"[Debug-Pivot] {symbol} => {pivot_levels}")

        # (3) H1 => ATR
        df_main = self._fetch_and_indicator(symbol, self.main_timeframe, limit=200)
        if df_main.empty:
            self.logger.warning(f"[PullbackStrategy] No H1 data => skip {symbol}")
            return
        atr_value = self._calculate_atr(df_main, self.atr_window)
        if atr_value is None:
            self.logger.warning(f"[PullbackStrategy] Not enough H1 data => skip {symbol}")
            return
        else:
            # AANPASSING #2: debug
            self.logger.debug(f"[Debug-ATR] {symbol}: ATR({self.atr_window}) = {atr_value}")

        # (4) Pullback => 15m
        current_ws_price = Decimal("0")
        df_entry = self._fetch_and_indicator(symbol, self.entry_timeframe, limit=100)
        if not df_entry.empty:
            rsi_val = df_entry["rsi"].iloc[-1]
            macd_signal_score = self._check_macd(df_entry)

            # CHANGED/ADDED: Candle-close als backup
            candle_close_price = Decimal(str(df_entry["close"].iloc[-1]))

            # CHANGED/ADDED: Live WebSocket-prijs
            ws_price = self._get_ws_price(symbol)

            # Log beide
            self.logger.debug(
                f"[Debug-15m] symbol={symbol}, last_close(15m)={candle_close_price}, ws_price={ws_price}"
            )

            # CHANGED: Gebruik primair ws_price, fallback = candle_close
            if ws_price > 0:
                current_price = ws_price
            else:
                current_price = candle_close_price

            pullback_detected = self._detect_pullback(df_entry, current_price, direction)
        else:
            rsi_val = 50
            macd_signal_score = 0
            current_price = Decimal("0")
            pullback_detected = False

        # (5) Depth + ML
        ml_signal = self._ml_predict_signal(df_daily)
        depth_score = 0.0
        if self.use_depth_trend:
            depth_score_instant = self._analyze_depth_trend_instant(symbol)
            # Rolling average
            self.depth_trend_history.append(depth_score_instant)
            depth_score = sum(self.depth_trend_history) / len(self.depth_trend_history)
            self.logger.info(f"[DepthTrend] instant={depth_score_instant:.2f}, rolling_avg={depth_score:.2f}")

        # (6) Manage / open
        has_position = (symbol in self.open_positions)
        total_equity = self._get_equity_estimate()
        invest_extra_flag = False
        if (total_equity >= self.initial_capital * self.accumulate_threshold) and not self.invested_extra:
            invest_extra_flag = True
            self.logger.info("[PullbackStrategy] +25%% => next pullback => invest extra in %s", symbol)

        self.logger.info(
            f"[Decision Info] symbol={symbol}, direction={direction}, pullback={pullback_detected}, "
            f"rsi_val={rsi_val:.2f}, macd_signal_score={macd_signal_score}, ml_signal={ml_signal}, depth_score={depth_score:.2f}"
        )

        # CHANGED: Hier is current_price nu (live) = ws_price of fallback
        if pullback_detected and not has_position:
            if direction == "bull":
                self.logger.debug(
                    f"[DEBUG-bull] symbol={symbol} | "
                    f"rsi_val={rsi_val:.2f} >= {self.rsi_bull_threshold}? => {rsi_val >= self.rsi_bull_threshold}, "
                    f"macd_signal_score={macd_signal_score} >= {self.macd_bull_threshold}? => {macd_signal_score >= self.macd_bull_threshold}, "
                    f"ml_signal={ml_signal} >= 0? => {ml_signal >= 0}, "
                    f"depth_score={depth_score:.2f} >= {self.depth_threshold_bull}? => {depth_score >= self.depth_threshold_bull}"
                )
                if (rsi_val >= self.rsi_bull_threshold
                    and macd_signal_score >= self.macd_bull_threshold
                    and ml_signal >= 0
                    and depth_score >= self.depth_threshold_bull):
                    self._open_position(symbol, side="buy", current_price=current_price,
                                        atr_value=atr_value, extra_invest=invest_extra_flag)
                    if invest_extra_flag:
                        self.invested_extra = True

            elif direction == "bear":
                self.logger.debug(
                    f"[DEBUG-bear] {symbol}: rsi_val={rsi_val:.2f} <= {self.rsi_bear_threshold}? , "
                    f"macd_signal={macd_signal_score} <= {self.macd_bear_threshold}? , ml={ml_signal} <= 0? , "
                    f"depth={depth_score} <= {self.depth_threshold_bear}? "
                )
                if (rsi_val <= self.rsi_bear_threshold
                    and macd_signal_score <= self.macd_bear_threshold
                    and ml_signal <= 0
                    and depth_score <= self.depth_threshold_bear):
                    self._open_position(symbol, side="sell", current_price=current_price,
                                        atr_value=atr_value, extra_invest=invest_extra_flag)
                    if invest_extra_flag:
                        self.invested_extra = True

        elif has_position:
            # CHANGED: Geef nu dezelfde current_price door
            self._manage_open_position(symbol, current_price, atr_value)

    # ------------------------------------------------
    #   TREND CHECK
    # ------------------------------------------------
    def _check_trend_direction(self, df_daily: pd.DataFrame, df_h4: pd.DataFrame) -> str:
        if df_daily.empty or df_h4.empty:
            return "range"

        rsi_daily = df_daily["rsi"].iloc[-1]
        rsi_h4 = df_h4["rsi"].iloc[-1]
        if rsi_daily > self.daily_bull_rsi and rsi_h4 > self.daily_bull_rsi:
            return "bull"
        elif rsi_daily < self.daily_bear_rsi and rsi_h4 < self.daily_bear_rsi:
            return "bear"
        else:
            return "range"

    # ------------------------------------------------
    #   FAIL-SAFES
    # ------------------------------------------------
    def _check_fail_safes(self, symbol: str) -> bool:
        if self._daily_loss_exceeded():
            return True
        if self._flash_crash_detected(symbol):
            return True
        return False

    def _daily_loss_exceeded(self) -> bool:
        if not self.order_client:
            return False

        bal = self.order_client.get_balance()
        eur_balance = Decimal(str(bal.get("EUR", "100")))
        drop_pct = (self.initial_capital - eur_balance) / self.initial_capital * Decimal("100")

        if drop_pct >= self.max_daily_loss_pct:
            self.logger.warning(f"[FailSafe] daily loss {drop_pct:.2f}% >= {self.max_daily_loss_pct}% => STOP.")
            return True
        return False

    def _flash_crash_detected(self, symbol: str) -> bool:
        df_fc = self._fetch_and_indicator(symbol, self.flash_crash_tf, limit=3)
        if df_fc.empty or len(df_fc) < 3:
            return False

        first_close_val = df_fc["close"].iloc[0]
        last_close_val = df_fc["close"].iloc[-1]
        if pd.isna(first_close_val) or pd.isna(last_close_val):
            self.logger.warning("[FailSafe] flash_crash: close-waarde is NaN => skip.")
            return False

        try:
            first_dec = Decimal(str(first_close_val))
            last_dec = Decimal(str(last_close_val))
        except InvalidOperation:
            return False

        if first_dec == 0:
            return False

        drop_pct = (first_dec - last_dec) / first_dec * Decimal("100")
        if drop_pct >= self.flash_crash_drop_pct:
            self.logger.warning(f"[FailSafe] Flash crash => drop {drop_pct:.2f}% on {self.flash_crash_tf}")
            return True
        return False

    # ------------------------------------------------
    #   DATA & INDICATORS
    # ------------------------------------------------
    def _fetch_and_indicator(self, symbol: str, interval: str, limit=200) -> pd.DataFrame:
        market_obj = Market(symbol, self.db_manager)
        df = market_obj.fetch_candles(interval=interval, limit=limit)
        if df.empty:
            return pd.DataFrame()

        df.columns = ['timestamp', 'market', 'interval', 'open', 'high', 'low', 'close', 'volume']

        df = IndicatorAnalysis.calculate_indicators(df, rsi_window=self.rsi_window)

        macd_ind = MACD(
            close=df['close'],
            window_slow=self.macd_slow,
            window_fast=self.macd_fast,
            window_sign=self.macd_signal
        )
        df['macd'] = macd_ind.macd()
        df['macd_signal'] = macd_ind.macd_signal()
        return df

    def _calculate_atr(self, df: pd.DataFrame, window=14) -> Optional[Decimal]:
        if len(df) < window:
            return None
        atr_obj = AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'],
            window=window
        )
        series_atr = atr_obj.average_true_range()
        last_atr = series_atr.iloc[-1]
        return Decimal(str(last_atr))

    def _calculate_pivot_points(self, df_h4: pd.DataFrame) -> dict:
        if len(df_h4) < self.pivot_points_window:
            return {}
        subset = df_h4.iloc[-self.pivot_points_window:]
        hi = Decimal(str(subset["high"].max()))
        lo = Decimal(str(subset["low"].min()))
        cls = Decimal(str(subset["close"].iloc[-1]))
        pivot = (hi + lo + cls) / Decimal("3")
        r1 = (2 * pivot) - lo
        s1 = (2 * pivot) - hi
        return {"pivot": pivot, "R1": r1, "S1": s1}

    def _detect_pullback(self, df: pd.DataFrame, current_price: Decimal, direction: str) -> bool:
        if len(df) < self.pullback_rolling_window:
            self.logger.debug(f"[Pullback] <{self.pullback_rolling_window} candles => skip.")
            return False

        # 2) BULL scenario
        if direction == "bull":
            # Bepaal recent high over X candles (pullback_rolling_window)
            recent_high = df["high"].rolling(self.pullback_rolling_window).max().iloc[-1]
            if recent_high <= 0:
                return False

            # drop_pct = percentage verschil tussen recent_high en current_price
            drop_pct = (Decimal(str(recent_high)) - current_price) / Decimal(str(recent_high)) * Decimal("100")
            if drop_pct >= self.pullback_threshold_pct:
                self.logger.info(
                    f"[Pullback-bull] {drop_pct:.2f}% below recent high => pullback "
                    f"(threshold={self.pullback_threshold_pct}%, window={self.pullback_rolling_window})."
                )
                return True
            return False

        # 3) BEAR scenario
        elif direction == "bear":
            # Bepaal recent low over X candles (pullback_rolling_window)
            recent_low = df["low"].rolling(self.pullback_rolling_window).min().iloc[-1]
            if recent_low <= 0:
                return False

            # rally_pct = percentage verschil tussen current_price en recent_low
            rally_pct = (current_price - Decimal(str(recent_low))) / Decimal(str(recent_low)) * Decimal("100")
            if rally_pct >= self.pullback_threshold_pct:
                self.logger.info(
                    f"[Pullback-bear] {rally_pct:.2f}% above recent low => pullback "
                    f"(threshold={self.pullback_threshold_pct}%, window={self.pullback_rolling_window})."
                )
                return True
            return False

        # 4) Range of onbepaald => geen pullback detectie
        return False

    def _check_macd(self, df: pd.DataFrame) -> float:
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            return 0
        macd_val = df['macd'].iloc[-1]
        macd_sig = df['macd_signal'].iloc[-1]
        gap = macd_val - macd_sig
        if gap > 0:
            return 1
        elif gap < 0:
            return -1
        return 0

    # ------------------------------------------------
    #   DEPTH TREND
    # ------------------------------------------------
    def _analyze_depth_trend_instant(self, symbol: str) -> float:
        orderbook = self.db_manager.get_orderbook_snapshot(symbol)
        if not orderbook:
            return 0.0
        total_bids = sum([float(b[1]) for b in orderbook["bids"]])
        total_asks = sum([float(a[1]) for a in orderbook["asks"]])
        denom = total_bids + total_asks
        if denom == 0:
            return 0.0
        score = (total_bids - total_asks) / denom
        return float(score)

    # ------------------------------------------------
    #   ML
    # ------------------------------------------------
    def set_ml_engine(self, ml_engine):
        """
        Optionele setter om de ml_engine toe te wijzen.
        """
        self.ml_engine = ml_engine
        self.logger.info("[PullbackAccumulateStrategy] ML-engine is succesvol gezet.")

    def _ml_predict_signal(self, df: pd.DataFrame) -> int:
        # 1) Check of ML aanstaat en of ml_engine gezet is
        if not self.ml_model_enabled or self.ml_engine is None:
            return 0
        last_row = df.iloc[-1]
        features = [
            last_row.get("rsi", 50),
            last_row.get("macd", 0),
            last_row.get("macd_signal", 0),
            last_row.get("volume", 0),
        ]
        return self.ml_engine.predict_signal(features)

    # ----------------------------------------------------------------
    #   Open/Manage pos
    # ----------------------------------------------------------------
    def _open_position(self, symbol: str, side: str, current_price: Decimal,
                       atr_value: Decimal, extra_invest=False):
        self.logger.info(f"[PullbackStrategy] OPEN => side={side}, {symbol}@{current_price}, extra_invest={extra_invest}")

        eur_balance = Decimal("100")
        if self.order_client:
            bal = self.order_client.get_balance()
            eur_balance = Decimal(str(bal.get("EUR", "100")))

        # check op current_price=0
        if current_price is None or current_price <= 0:
            self.logger.warning(f"[PullbackStrategy] current_price={current_price} => skip open pos for {symbol}")
            return

        # 3) portion
        pct = self.position_size_pct
        if extra_invest:
            pct = Decimal("0.10")
        buy_eur = eur_balance * pct
        if buy_eur < 5:
            self.logger.warning(f"[PullbackStrategy] buy_eur < 5 => skip {symbol}")
            return

        amount = buy_eur / current_price

        # 2) Bepaal position_id en position_type
        position_id = f"{symbol}-{int(time.time())}"
        if side == "buy":
            position_type = "long"
        else:
            position_type = "short"

        # 5) place order (via order_client)
        if self.order_client:
            self.order_client.place_order(side, symbol, float(amount), order_type="market")
            self.logger.info(
                f"[LIVE/PAPER] {side.upper()} {symbol} => amt={amount:.4f}, price={current_price}, cost={buy_eur}"
            )

        fees = 0.0
        pnl_eur = 0.0
        trade_cost = float(amount * current_price)

        # in DB:
        trade_data = {
            "symbol": symbol,
            "side": side,
            "amount": float(amount),
            "price": float(current_price),
            "timestamp": int(time.time() * 1000),
            "position_id": position_id,
            "position_type": position_type,
            "status": "open",
            "pnl_eur": pnl_eur,
            "fees": fees,
            "trade_cost": trade_cost
        }
        self.db_manager.save_trade(trade_data)

        # 6) open_positions
        self.open_positions[symbol] = {
            "side": side,
            "entry_price": current_price,
            "amount": amount,
            "atr": atr_value,
            "tp1_done": False,
            "tp2_done": False,
            "trail_active": False,
            "trail_high": current_price,
            "position_id": position_id,
            "position_type": position_type
        }

    def _manage_open_position(self, symbol: str, current_price: Decimal, atr_value: Decimal):
        if current_price is None or current_price <= 0:
            self.logger.warning(f"[PullbackStrategy] current_price={current_price} => skip manage pos for {symbol}")
            return

        pos = self.open_positions[symbol]
        side = pos["side"]
        entry = pos["entry_price"]
        amount = pos["amount"]

        if side == "buy":
            # LONG
            tp1_price = entry + pos["atr"] * self.tp1_atr_mult
            tp2_price = entry + pos["atr"] * self.tp2_atr_mult

            self.logger.debug(
                f"[DEBUG-manage-LONG] symbol={symbol}, tp1_done={pos['tp1_done']}, current_price={current_price}, "
                f"tp1_price={tp1_price:.4f}, tp2_done={pos['tp2_done']}, tp2_price={tp2_price:.4f}"
            )

            if (not pos["tp1_done"]) and (current_price >= tp1_price):
                self.logger.info(f"[PullbackStrategy] LONG TP1 => Sell 25% {symbol}")

                # (A) GEBRUIK DEZELFDE current_price:
                self._sell_portion(
                    symbol, amount, portion=Decimal("0.25"), reason="TP1",
                    exec_price=current_price  # doorgeven!
                )
                pos["tp1_done"] = True
                pos["trail_active"] = True
                pos["trail_high"] = max(pos["trail_high"], current_price)

            elif (not pos["tp2_done"]) and (current_price >= tp2_price):
                self.logger.info(f"[PullbackStrategy] LONG TP2 => Sell 25% {symbol}")
                self._sell_portion(
                    symbol, amount, portion=Decimal("0.25"), reason="TP2",
                    exec_price=current_price
                )
                pos["tp2_done"] = True
                pos["trail_active"] = True
                pos["trail_high"] = max(pos["trail_high"], current_price)

            if pos["trail_active"]:
                if current_price > pos["trail_high"]:
                    pos["trail_high"] = current_price
                trailing_stop_price = pos["trail_high"] - (atr_value * self.trail_atr_mult)
                self.logger.debug(
                    f"[DEBUG-trailing-LONG] symbol={symbol}, trail_high={pos['trail_high']}, "
                    f"trailing_stop_price={trailing_stop_price}, current_price={current_price}"
                )

                if current_price <= trailing_stop_price:
                    self.logger.info(f"[PullbackStrategy] LONG TrailingStop => close last 50% {symbol}")
                    self._sell_portion(
                        symbol, amount, portion=Decimal("1.0"), reason="TrailingStop",
                        exec_price=current_price
                    )
                    if symbol in self.open_positions:
                        del self.open_positions[symbol]

        else:
            # SHORT
            tp1_price = entry - pos["atr"] * self.tp1_atr_mult
            tp2_price = entry - pos["atr"] * self.tp2_atr_mult

            self.logger.debug(
                f"[DEBUG-manage-SHORT] symbol={symbol}, tp1_done={pos['tp1_done']}, current_price={current_price}, "
                f"tp1_price={tp1_price:.4f}, tp2_done={pos['tp2_done']}, tp2_price={tp2_price:.4f}"
            )

            if (not pos["tp1_done"]) and (current_price <= tp1_price):
                self.logger.info(f"[PullbackStrategy] SHORT TP1 => Buy-to-Close 25% {symbol}")
                self._buy_portion(
                    symbol, amount, portion=Decimal("0.25"), reason="TP1",
                    exec_price=current_price
                )
                pos["tp1_done"] = True
                pos["trail_active"] = True

            elif (not pos["tp2_done"]) and (current_price <= tp2_price):
                self.logger.info(f"[PullbackStrategy] SHORT TP2 => Buy-to-Close 25% {symbol}")
                self._buy_portion(
                    symbol, amount, portion=Decimal("0.25"), reason="TP2",
                    exec_price=current_price
                )
                pos["tp2_done"] = True
                pos["trail_active"] = True

            if pos["trail_active"]:
                trailing_stop_price = entry + (pos["atr"] * self.trail_atr_mult)
                self.logger.debug(
                    f"[DEBUG-trailing-SHORT] symbol={symbol}, entry={entry}, "
                    f"trail_stop={trailing_stop_price:.4f}, current_price={current_price}"
                )
                # TrailingStop => close last 50% (of entire) etc.
                if current_price >= trailing_stop_price:
                    self.logger.info(f"[PullbackStrategy] SHORT TrailingStop => close last 50% {symbol}")
                    self._buy_portion(
                        symbol, amount, portion=Decimal("0.50"), reason="TrailingStop",
                        exec_price=current_price
                    )
                    if symbol in self.open_positions:
                        del self.open_positions[symbol]

    # (B) Pas _sell_portion aan met exec_price
    def _sell_portion(self, symbol: str, total_amt: Decimal, portion: Decimal, reason: str, exec_price=None):
        """
        Een deel van een LONG-positie verkopen.
        """
        amt_to_sell = total_amt * portion
        pos = self.open_positions[symbol]
        entry_price = pos["entry_price"]
        position_id = pos["position_id"]
        position_type = pos["position_type"]

        # => ALS exec_price NIET None is, gebruik die
        if exec_price is not None:
            current_price = exec_price
        else:
            current_price = self._get_ws_price(symbol)

        if current_price <= 0:
            self.logger.warning(f"[PullbackStrategy] _sell_portion => current_price=0 => skip SELL {symbol}")
            return

        raw_pnl = (current_price - entry_price) * amt_to_sell
        trade_cost = current_price * amt_to_sell
        fees = float(trade_cost * Decimal("0.0025"))
        realized_pnl = float(raw_pnl) - fees

        self.logger.debug(
            f"[DEBUG {reason}] {symbol}: portion={portion}, amt_to_sell={amt_to_sell:.6f}, "
            f"entry={entry_price}, current_price={current_price}, trade_cost={trade_cost}, fees={fees:.2f}"
        )

        if portion < 1:
            trade_status = "partial"
        else:
            trade_status = "closed"

        # => ORDER
        if self.order_client:
            self.order_client.place_order("sell", symbol, float(amt_to_sell), order_type="market")
            self.logger.info(
                f"[LIVE/PAPER] SELL {symbol} => {portion*100:.1f}%, amt={amt_to_sell:.4f}, reason={reason}, "
                f"fees={fees:.2f}, pnl={realized_pnl:.2f}"
            )
            trade_data = {
                "symbol": symbol,
                "side": "sell",
                "amount": float(amt_to_sell),
                "price": float(current_price),
                "timestamp": int(time.time() * 1000),
                "position_id": position_id,
                "position_type": position_type,
                "status": trade_status,
                "pnl_eur": realized_pnl,
                "fees": fees,
                "trade_cost": float(trade_cost)
            }
            self.db_manager.save_trade(trade_data)
        else:
            self.logger.info(
                f"[Paper] SELL {symbol} => {portion*100:.1f}%, amt={amt_to_sell:.4f}, reason={reason}, "
                f"(fees={fees:.2f}, pnl={realized_pnl:.2f})"
            )

        self.open_positions[symbol]["amount"] -= amt_to_sell
        if self.open_positions[symbol]["amount"] <= Decimal("0"):
            self.logger.info(f"[PullbackStrategy] Full position closed => {symbol}")
            del self.open_positions[symbol]

    # (B) Pas _buy_portion aan met exec_price
    def _buy_portion(self, symbol: str, total_amt: Decimal, portion: Decimal, reason: str, exec_price=None):
        """
        Een deel van een SHORT-positie sluiten => buy
        """
        amt_to_buy = total_amt * portion
        pos = self.open_positions[symbol]
        entry_price = pos["entry_price"]
        position_id = pos["position_id"]
        position_type = pos["position_type"]

        # => ALS exec_price NIET None is, gebruik die:
        if exec_price is not None:
            current_price = exec_price
        else:
            current_price = self._get_ws_price(symbol)

        if current_price <= 0:
            self.logger.warning(f"[PullbackStrategy] _buy_portion => current_price=0 => skip BUY {symbol}")
            return

        raw_pnl = (entry_price - current_price) * amt_to_buy
        trade_cost = current_price * amt_to_buy
        fees = float(trade_cost * Decimal("0.0025"))
        realized_pnl = float(raw_pnl) - fees

        self.logger.debug(
            f"[DEBUG {reason}] {symbol}: portion={portion}, amt_to_buy={amt_to_buy:.6f}, "
            f"entry={entry_price}, current_price={current_price}, trade_cost={trade_cost}, fees={fees:.2f}"
        )

        if portion < 1:
            trade_status = "partial"
        else:
            trade_status = "closed"

        if self.order_client:
            self.order_client.place_order("buy", symbol, float(amt_to_buy), order_type="market")
            self.logger.info(
                f"[LIVE/PAPER] BUY {symbol} => {portion*100:.1f}%, amt={amt_to_buy:.4f}, reason={reason}, "
                f"fees={fees:.2f}, pnl={realized_pnl:.2f}"
            )
            trade_data = {
                "symbol": symbol,
                "side": "buy",
                "amount": float(amt_to_buy),
                "price": float(current_price),
                "timestamp": int(time.time() * 1000),
                "position_id": position_id,
                "position_type": position_type,
                "status": trade_status,
                "pnl_eur": realized_pnl,
                "fees": fees,
                "trade_cost": float(trade_cost)
            }
            self.db_manager.save_trade(trade_data)
        else:
            self.logger.info(
                f"[Paper] BUY {symbol} => {portion*100:.1f}%, amt={amt_to_buy:.4f}, reason={reason}, "
                f"(fees={fees:.2f}, pnl={realized_pnl:.2f})"
            )

        # 5) Update open_positions
        self.open_positions[symbol]["amount"] -= amt_to_buy
        if self.open_positions[symbol]["amount"] <= Decimal("0"):
            self.logger.info(f"[PullbackStrategy] Full short position closed => {symbol}")
            del self.open_positions[symbol]

    # ----------------------------------------------------------------
    #   Hulp: equity, ws price
    # ----------------------------------------------------------------
    def _get_equity_estimate(self) -> Decimal:
        if not self.order_client:
            return self.initial_capital

        bal = self.order_client.get_balance()
        eur_balance = Decimal(str(bal.get("EUR", "1000")))
        total_pos_value = Decimal("0")
        for sym, pos_info in self.open_positions.items():
            amt = pos_info["amount"]
            latest_price = self._get_latest_price(sym)
            if latest_price > 0:
                total_pos_value += (amt * latest_price)
        return eur_balance + total_pos_value

    def _get_latest_price(self, symbol: str) -> Decimal:
        """Valt terug op DB (ticker of 1m candle)."""
        ticker_data = self.db_manager.get_ticker(symbol)
        if ticker_data:
            best_bid = Decimal(str(ticker_data.get("bestBid", 0)))
            best_ask = Decimal(str(ticker_data.get("bestAsk", 0)))
            if best_bid > 0 and best_ask > 0:
                return (best_bid + best_ask) / Decimal("2")

        df_1m = self._fetch_and_indicator(symbol, "1m", limit=1)
        if not df_1m.empty:
            last_close = df_1m["close"].iloc[-1]
            return Decimal(str(last_close))

        return Decimal("0")

    def _get_ws_price(self, symbol):
        if not self.data_client:
            self.logger.warning("[PullbackStrategy] data_client=None => return price=0")
            return Decimal("0")

        price = self.data_client.get_price_with_fallback(symbol, max_age=10)
        return price if price > 0 else Decimal("0")
