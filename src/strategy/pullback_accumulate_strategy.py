import logging
from decimal import Decimal
import pandas as pd
from typing import Optional
import os
import yaml


# -----------------------------
# CONFIG LOADER (CHANGED)
# - We plaatsen load_config bovenin, en import yaml hier.
# -----------------------------
def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# TA-bibliotheken
from ta.volatility import AverageTrueRange
from ta.trend import MACD

# Lokale imports
# (CHANGED) Hier verwijzen we naar je eigen modules.
from src.logger.logger import setup_logger
from src.database_manager.database_manager import DatabaseManager
from src.indicator_analysis.indicators import Market, IndicatorAnalysis

# (Optioneel) joblib als ML-model
try:
    import joblib
except ImportError:
    joblib = None


class PullbackAccumulateStrategy:
    """
    Pullback & Accumulate Strategy
    ---------------------------------------------------------
     - Daily + H4 => hoofdrichting (bull, bear, range) en pivot-points op H4  (CHANGED)
     - H1 => intraday trend + ATR
     - 15m => entry/pullbackmoment
     - 5m => flash crash check (fail-safe) (CHANGED)
     - ATR-gestuurde SL, gefaseerde TP
     - +25% herinvesteren
     - Depth Trend & ML (ML op daily) (CHANGED)
    """

    def __init__(
            self,
            client,
            db_manager: DatabaseManager,
            config_path: Optional[str] = None
    ):
        self.client = client
        self.db_manager = db_manager

        # (CHANGED) We verplaatsen load_config + config parsing bovenin
        if config_path and os.path.isfile(config_path):
            global_config = load_config(config_path)
            self.strategy_config = global_config.get("pullback_accumulate_strategy", {})
        else:
            self.strategy_config = {}

        # (CHANGED) Timeframes uit config, defaults:
        self.daily_timeframe = self.strategy_config.get("daily_timeframe", "1d")
        self.trend_timeframe = self.strategy_config.get("trend_timeframe", "4h")
        self.main_timeframe = self.strategy_config.get("main_timeframe", "1h")
        self.entry_timeframe = self.strategy_config.get("entry_timeframe", "15m")
        # (ADDED) flash_crash_tf voor 5m flash crash-detectie
        self.flash_crash_tf = self.strategy_config.get("flash_crash_timeframe", "5m")

        # (UNTOUCHED) Overige strategy-params
        self.atr_window = int(self.strategy_config.get("atr_window", 14))
        self.pullback_threshold_pct = Decimal(str(self.strategy_config.get("pullback_threshold_pct", 1.0)))
        self.accumulate_threshold = Decimal(str(self.strategy_config.get("accumulate_threshold", 1.25)))
        self.position_size_pct = Decimal(str(self.strategy_config.get("position_size_pct", 0.05)))

        self.tp1_atr_mult = Decimal(str(self.strategy_config.get("tp1_atr_mult", "1.0")))
        self.tp2_atr_mult = Decimal(str(self.strategy_config.get("tp2_atr_mult", "2.0")))
        self.trail_atr_mult = Decimal(str(self.strategy_config.get("trail_atr_mult", "1.0")))
        self.initial_capital = Decimal(str(self.strategy_config.get("initial_capital", "1000")))
        self.log_file = self.strategy_config.get("log_file", "pullback_strategy.log")

        # (UNCHANGED) Fail-safes
        self.max_daily_loss_pct = Decimal(str(self.strategy_config.get("max_daily_loss_pct", 5.0)))
        self.flash_crash_drop_pct = Decimal(str(self.strategy_config.get("flash_crash_drop_pct", 10.0)))
        self.use_depth_trend = self.strategy_config.get("use_depth_trend", True)

        # (CHANGED) RSI/MACD/pivot: pivot_points_window is op H4
        self.rsi_window = int(self.strategy_config.get("rsi_window", 14))
        self.macd_fast = int(self.strategy_config.get("macd_fast", 12))
        self.macd_slow = int(self.strategy_config.get("macd_slow", 26))
        self.macd_signal = int(self.strategy_config.get("macd_signal", 9))
        self.pivot_points_window = int(self.strategy_config.get("pivot_points_window", 20))

        # (CHANGED) ML
        self.ml_model_enabled = self.strategy_config.get("ml_model_enabled", False)
        self.ml_model_path = self.strategy_config.get("ml_model_path", "models/pullback_model.pkl")
        self.ml_model = None

        # (UNCHANGED) Logger
        self.logger = setup_logger("pullback_strategy", self.log_file, logging.DEBUG)
        self.logger.info("[PullbackAccumulateStrategy] init from %s", config_path)

        # (CHANGED) ML-engine
        from src.ml_engine.ml_engine import MLEngine
        self.ml_engine = MLEngine(self.db_manager, config_path=config_path)

        # (CHANGED) Als ML geactiveerd is, proberen we joblib-model te laden
        if self.ml_model_enabled and joblib is not None and os.path.exists(self.ml_model_path):
            try:
                self.ml_model = joblib.load(self.ml_model_path)
            except Exception as e:
                self.logger.warning(f"[PullbackAccumulateStrategy] ML model load error: {e}")
                self.ml_model = None

        # (UNCHANGED) Posities en +25% flag
        self.open_positions = {}
        self.invested_extra = False

    # --------------------------------------------------------------------------
    # EXECUTE STRATEGY (KERN-FLOW) (CHANGED: herschreven opzet)
    # --------------------------------------------------------------------------
    def execute_strategy(self, symbol: str):
        """
        1) Fail-safes (flash crash/drawdown)
        2) Daily/H4 => hoofdrichting en pivot (op H4)
        3) H1 => ATR, RSI, MACD
        4) 15m => pullback
        5) Depth Trend + ML
        6) Open/Manage posities
        7) +25% check
        """
        self.logger.info(f"[PullbackStrategy] Start for {symbol}")

        # (1) Fail-safes
        if self._check_fail_safes(symbol):
            self.logger.warning(f"[PullbackStrategy] Fail-safe => skip trading {symbol}")
            return

        # (2) Daily & H4
        # CHANGED: We halen df_daily en df_h4 (was soms door elkaar in je oude code).
        df_daily = self._fetch_and_indicator(symbol, self.daily_timeframe, limit=60)
        df_h4 = self._fetch_and_indicator(symbol, self.trend_timeframe, limit=200)
        if df_daily.empty or df_h4.empty:
            self.logger.warning(f"[PullbackStrategy] No daily/H4 data => skip {symbol}")
            return

        direction = self._check_trend_direction(df_daily, df_h4)
        self.logger.info(f"[PullbackStrategy] Direction={direction} for {symbol}")

        # pivot-points op H4 (CHANGED)
        pivot_levels = self._calculate_pivot_points(df_h4)

        # ML op daily (CHANGED)
        ml_signal = 0
        if self.ml_model_enabled and self.ml_model:
            ml_signal = self._ml_predict_signal(df_daily)
            self.logger.debug(f"[ML] daily-based signal={ml_signal}")

        # (3) H1 => ATR, RSI, MACD (CHANGED: netter + consistent)
        df_main = self._fetch_and_indicator(symbol, self.main_timeframe, limit=200)
        if df_main.empty:
            self.logger.warning(f"[PullbackStrategy] No H1 data => skip {symbol}")
            return

        atr_value = self._calculate_atr(df_main, window=self.atr_window)
        if atr_value is None:
            self.logger.warning(f"[PullbackStrategy] Not enough H1 data for ATR => skip {symbol}")
            return

        rsi_val = df_main["rsi"].iloc[-1]
        macd_signal_score = self._check_macd(df_main)
        current_price = Decimal(str(df_main['close'].iloc[-1]))

        # (4) 15m => pullback
        # CHANGED: Herkomst was onduidelijk, nu consistent
        df_15m = self._fetch_and_indicator(symbol, self.entry_timeframe, limit=100)
        if not df_15m.empty:
            last_close_15m = Decimal(str(df_15m['close'].iloc[-1]))
            pullback_detected = self._detect_pullback(df_15m, last_close_15m)
        else:
            pullback_detected = False

        # (5) Depth Trend
        depth_score = 0.0
        if self.use_depth_trend:
            depth_score = self._analyze_depth_trend(symbol)
            self.logger.info(f"[DepthTrend] {symbol} => {depth_score:.2f}")

        # (6) Open / Manage
        has_position = (symbol in self.open_positions)

        # +25% check
        total_equity = self._get_equity_estimate()
        invest_extra_flag = False
        if (total_equity >= self.initial_capital * self.accumulate_threshold) and not self.invested_extra:
            invest_extra_flag = True
            self.logger.info("[PullbackStrategy] +25%% => next pullback => invest extra in %s", symbol)

        # Beslissing
        # CHANGED: netter if/elif
        if pullback_detected and not has_position:
            # Voorbeeld: direction == bull, RSI <50 => net dip, macd>=0, ml>=0, depth>=0 => koop
            if (direction == "bull") and (rsi_val < 50) and (macd_signal_score >= 0) and (ml_signal >= 0) and (
                    depth_score >= 0):
                self._open_position(symbol, current_price, atr_value, extra_invest=invest_extra_flag)
                if invest_extra_flag:
                    self.invested_extra = True
        elif has_position:
            self._manage_open_position(symbol, current_price, atr_value)

        self.logger.info(f"[PullbackStrategy] Done for {symbol}")

    # --------------------------------------------------------------------------
    # TREND: Daily + H4 => bull, bear, range (CHANGED)
    # --------------------------------------------------------------------------
    def _check_trend_direction(self, df_daily: pd.DataFrame, df_h4: pd.DataFrame) -> str:
        if df_daily.empty or df_h4.empty:
            return "range"

        rsi_daily = df_daily["rsi"].iloc[-1]
        rsi_h4 = df_h4["rsi"].iloc[-1]

        if rsi_daily > 60 and rsi_h4 > 60:
            return "bull"
        elif rsi_daily < 40 and rsi_h4 < 40:
            return "bear"
        else:
            return "range"

    # --------------------------------------------------------------------------
    # FAIL-SAFES: daily drawdown + flash crash (CHANGED)
    # --------------------------------------------------------------------------
    def _check_fail_safes(self, symbol: str) -> bool:
        if self._daily_loss_exceeded():
            return True
        if self._flash_crash_detected(symbol):
            return True
        return False

    def _daily_loss_exceeded(self) -> bool:
        if not self.client:
            return False
        balance_eur = Decimal(str(self.client.get_balance().get("EUR", "1000")))
        drop_pct = (self.initial_capital - balance_eur) / self.initial_capital * Decimal("100")
        if drop_pct >= self.max_daily_loss_pct:
            self.logger.warning(f"[FailSafe] daily loss {drop_pct:.2f}% >= {self.max_daily_loss_pct}% => STOP.")
            return True
        return False

    def _flash_crash_detected(self, symbol: str) -> bool:
        # (CHANGED) We gebruiken nu self.flash_crash_tf ("5m")
        df_fc = self._fetch_and_indicator(symbol, self.flash_crash_tf, limit=3)
        if df_fc.empty or len(df_fc) < 3:
            return False

        first_close = Decimal(str(df_fc['close'].iloc[0]))
        last_close = Decimal(str(df_fc['close'].iloc[-1]))
        drop_pct = (first_close - last_close) / first_close * Decimal("100")

        if drop_pct >= self.flash_crash_drop_pct:
            self.logger.warning(f"[FailSafe] Flash crash => drop {drop_pct:.2f}% on {self.flash_crash_tf}")
            return True
        return False

    # --------------------------------------------------------------------------
    # DATA & INDICATORS (CHANGED: consistent naming df_main/df_daily/df_15m etc.)
    # --------------------------------------------------------------------------
    def _fetch_and_indicator(self, symbol: str, interval: str, limit=200) -> pd.DataFrame:
        market_obj = Market(symbol, self.db_manager)
        df = market_obj.fetch_candles(interval=interval, limit=limit)
        if df.empty:
            return pd.DataFrame()

        df.columns = ['timestamp', 'market', 'interval', 'open', 'high', 'low', 'close', 'volume']

        # RSI e.d.
        df = IndicatorAnalysis.calculate_indicators(df, rsi_window=self.rsi_window)

        # MACD
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
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=window
        )
        atr_series = atr_obj.average_true_range()
        last_atr = atr_series.iloc[-1]
        return Decimal(str(last_atr))

    # (CHANGED) PIVOT: nu op 4H (in code roepen we het aan met df_h4)
    def _calculate_pivot_points(self, df_h4: pd.DataFrame) -> dict:
        if len(df_h4) < self.pivot_points_window:
            return {}

        subset = df_h4.iloc[-self.pivot_points_window:]
        hi = Decimal(str(subset['high'].max()))
        lo = Decimal(str(subset['low'].min()))
        cls = Decimal(str(subset['close'].iloc[-1]))

        pivot = (hi + lo + cls) / Decimal("3")
        r1 = (2 * pivot) - lo
        s1 = (2 * pivot) - hi
        return {
            "pivot": pivot,
            "R1": r1,
            "S1": s1
        }

    # (CHANGED) Pullback-check => op 15m
    def _detect_pullback(self, df: pd.DataFrame, current_price: Decimal) -> bool:
        if len(df) < 20:
            return False
        recent_high = Decimal(str(df['high'].rolling(20).max().iloc[-1]))
        if recent_high == 0:
            return False

        drop_pct = (recent_high - current_price) / recent_high * Decimal("100")
        if drop_pct >= self.pullback_threshold_pct:
            self.logger.info(f"[PullbackStrategy] Pullback => {drop_pct:.2f}% below recent high.")
            return True
        return False

    def _check_macd(self, df: pd.DataFrame) -> float:
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            return 0
        macd_val = df['macd'].iloc[-1]
        macd_signal_val = df['macd_signal'].iloc[-1]
        gap = macd_val - macd_signal_val
        if gap > 0:
            return 1
        elif gap < 0:
            return -1
        return 0

    def _analyze_depth_trend(self, symbol: str) -> float:
        # (UNCHANGED) Eenvoudige implementatie
        orderbook = self.db_manager.get_orderbook_snapshot(symbol)
        if not orderbook:
            return 0.0

        total_bids = sum([float(b[1]) for b in orderbook['bids']])
        total_asks = sum([float(a[1]) for a in orderbook['asks']])
        denom = total_bids + total_asks
        if denom == 0:
            return 0.0
        score = (total_bids - total_asks) / denom
        return float(score)

    # (CHANGED) ML => op daily
    def _ml_predict_signal(self, df: pd.DataFrame) -> int:
        if not self.ml_model_enabled:
            return 0
        if self.ml_engine is None or self.ml_model is None:
            return 0

        last_row = df.iloc[-1]
        features = [
            last_row.get('rsi', 50),
            last_row.get('macd', 0),
            last_row.get('macd_signal', 0),
            last_row.get('volume', 0)
        ]
        signal = self.ml_engine.predict_signal(features)
        return signal

    # --------------------------------------------------------------------------
    # POSITION HANDLING
    # --------------------------------------------------------------------------
    def _open_position(self, symbol: str, current_price: Decimal, atr_value: Decimal, extra_invest=False):
        self.logger.info(f"[PullbackStrategy] OPEN => {symbol} @ {current_price}, extra_invest={extra_invest}")

        eur_balance = Decimal("1000")
        if self.client:
            bal = self.client.get_balance()
            eur_balance = Decimal(str(bal.get("EUR", "1000")))

        pct = self.position_size_pct
        if extra_invest:
            pct = Decimal("0.10")  # (CHANGED) bv. 10% ipv 5%

        buy_eur = eur_balance * pct
        if buy_eur < 10:
            self.logger.warning(f"[PullbackStrategy] buy_eur < 10 => skip {symbol}")
            return

        amount = buy_eur / current_price
        if self.client:
            self.client.place_order("buy", symbol, float(amount), order_type="market")
            self.logger.info(f"[LIVE] BUY {symbol} => amt={amount:.4f}, price={current_price}, cost={buy_eur}")
        else:
            self.logger.info(f"[Paper] BUY {symbol} => amt={amount:.4f}, price={current_price}, cost={buy_eur}")

        self.open_positions[symbol] = {
            "entry_price": current_price,
            "amount": amount,
            "atr": atr_value,
            "tp1_done": False,
            "tp2_done": False,
            "trail_active": False,
            "trail_high": current_price
        }

    def _manage_open_position(self, symbol: str, current_price: Decimal, atr_value: Decimal):
        pos = self.open_positions[symbol]
        entry = pos["entry_price"]
        amount = pos["amount"]

        tp1_price = entry + (pos["atr"] * self.tp1_atr_mult)
        tp2_price = entry + (pos["atr"] * self.tp2_atr_mult)

        # 25% op +1x ATR
        if (not pos["tp1_done"]) and (current_price >= tp1_price):
            self.logger.info(f"[PullbackStrategy] TP1 => Sell 25% {symbol}")
            self._sell_portion(symbol, amount, portion=Decimal("0.25"), reason="TP1")
            pos["tp1_done"] = True
            pos["trail_active"] = True
            pos["trail_high"] = max(pos["trail_high"], current_price)

        # 25% op +2x ATR
        if (not pos["tp2_done"]) and (current_price >= tp2_price):
            self.logger.info(f"[PullbackStrategy] TP2 => Sell 25% {symbol}")
            self._sell_portion(symbol, amount, portion=Decimal("0.25"), reason="TP2")
            pos["tp2_done"] = True
            pos["trail_active"] = True
            pos["trail_high"] = max(pos["trail_high"], current_price)

        # Resterende 50% => trailing stop
        if pos["trail_active"]:
            if current_price > pos["trail_high"]:
                pos["trail_high"] = current_price

            trailing_stop_price = pos["trail_high"] - (atr_value * self.trail_atr_mult)
            if current_price <= trailing_stop_price:
                self.logger.info(f"[PullbackStrategy] TrailingStop => close last 50% {symbol}")
                self._sell_portion(symbol, amount, portion=Decimal("0.50"), reason="TrailingStop")
                if symbol in self.open_positions:
                    del self.open_positions[symbol]

    def _sell_portion(self, symbol: str, total_amt: Decimal, portion: Decimal, reason: str):
        amt_to_sell = total_amt * portion
        if self.client:
            self.client.place_order("sell", symbol, float(amt_to_sell), order_type="market")
            self.logger.info(f"[LIVE] SELL {symbol} => {portion * 100}%, amt={amt_to_sell:.4f}, reason={reason}")
        else:
            self.logger.info(f"[Paper] SELL {symbol} => {portion * 100}%, amt={amt_to_sell:.4f}, reason={reason}")

        if symbol in self.open_positions:
            self.open_positions[symbol]["amount"] -= amt_to_sell
            if self.open_positions[symbol]["amount"] <= Decimal("0"):
                self.logger.info(f"[PullbackStrategy] Full position closed => {symbol}")
                del self.open_positions[symbol]

    # --------------------------------------------------------------------------
    # EQUITY (UNCHANGED behalve naamconsistentie)
    # --------------------------------------------------------------------------
    def _get_equity_estimate(self) -> Decimal:
        if not self.client:
            return self.initial_capital

        bal = self.client.get_balance()
        eur_balance = Decimal(str(bal.get("EUR", "1000")))

        total_positions_value = Decimal("0")
        for sym, pos_info in self.open_positions.items():
            amount_coins = pos_info["amount"]
            latest_price = self._get_latest_price(sym)
            if latest_price > 0:
                pos_value = amount_coins * latest_price
                total_positions_value += pos_value

        return eur_balance + total_positions_value

    def _get_latest_price(self, symbol: str) -> Decimal:
        ticker_data = self.db_manager.get_ticker(symbol)
        if ticker_data:
            best_bid = Decimal(str(ticker_data.get("bestBid", 0)))
            best_ask = Decimal(str(ticker_data.get("bestAsk", 0)))
            if best_bid > 0 and best_ask > 0:
                return (best_bid + best_ask) / Decimal("2")

        # fallback: laatste candle (1m)
        df_1m = self._fetch_and_indicator(symbol, "1m", limit=1)
        if not df_1m.empty:
            last_close = df_1m['close'].iloc[-1]
            return Decimal(str(last_close))

        return Decimal("0")
