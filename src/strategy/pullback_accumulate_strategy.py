import logging
import pandas as pd
import yaml
import time
from typing import Optional
from decimal import Decimal, InvalidOperation
from collections import deque
from datetime import datetime, timedelta, timezone

# TA-bibliotheken
from ta.volatility import AverageTrueRange
from ta.trend import MACD

# Lokale imports
from src.config.config import PULLBACK_STRATEGY_LOG_FILE, PULLBACK_CONFIG, load_config_file
from src.logger.logger import setup_logger
from src.indicator_analysis.indicators import Market, IndicatorAnalysis
from src.meltdown_manager.meltdown_manager import MeltdownManager

try:
    import joblib
except ImportError:
    joblib = None


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    print("[DEBUG] In load_config =>", data)
    return data


# NIEUW: Helper-functie om te controleren of een candle afgesloten is.
def is_candle_closed(last_candle_ms: int, timeframe: str) -> bool:
    """
    Berekent of de candle die eindigt op last_candle_ms al volledig afgesloten is.
    Hierbij gaan we ervan uit dat de timeframe als volgt kan worden opgegeven:
      - '5m'   : 5 minuten
      - '15m'  : 15 minuten
      - '1h'   : 60 minuten, etc.
    """
    # Zet de timeframe om in milliseconden
    unit = timeframe[-1]
    try:
        value = int(timeframe[:-1])
    except ValueError:
        # Als de parsing mislukt, gaan we ervan uit dat de candle niet afgesloten is
        return False

    if unit == "m":
        duration_ms = value * 60 * 1000
    elif unit == "h":
        duration_ms = value * 60 * 60 * 1000
    elif unit == "d":
        duration_ms = value * 24 * 60 * 60 * 1000
    else:
        # Onbekende tijdseenheid
        duration_ms = 0

    current_ms = int(time.time() * 1000)
    # Een candle is afgesloten als de huidige tijd (in ms) groter is dan het eindtijdstip van de candle.
    return current_ms >= (last_candle_ms + duration_ms)


class PullbackAccumulateStrategy:
    """
    Pullback & Accumulate Strategy
    ---------------------------------------------------------
     - self.data_client => om de laatste koersen te zien (bv. .latest_prices)
     - self.order_client => om orders te plaatsen en get_balance() te doen
     - (Nu Bitvavo-only)
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
        self.logger = setup_logger("pullback_strategy", PULLBACK_STRATEGY_LOG_FILE, logging.INFO)  # kan weer naar DEBUG indien nodig
        if config_path:
            self.logger.info("[PullbackAccumulateStrategy] init with config_path=%s", config_path)
        else:
            self.logger.info("[PullbackAccumulateStrategy] init (no config_path)")

        # Laden config, bv. meltdown_cfg = full_config.get("meltdown_manager", {})
        meltdown_cfg = self.strategy_config.get("meltdown_manager", {})
        self.meltdown_manager = MeltdownManager(meltdown_cfg, db_manager=db_manager, logger=self.logger)
        self.initial_capital = Decimal(str(self.strategy_config.get("initial_capital", "100")))

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
        self.stop_loss_pct = Decimal(str(self.strategy_config.get("stop_loss_pct", "0.01")))
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

        # Posities & vlag
        self.open_positions = {}
        self.invested_extra = False

        # DepthTrend rolling average
        self.depth_trend_history = deque(maxlen=5)

        # (NIEUW) Na self.open_positions = {}
        self._load_open_positions_from_db()

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
        """
        1) meltdown-check
        2) normal logic
        """

        meltdown_active = self.meltdown_manager.update_meltdown_state(strategy=self, symbol=symbol)
        if meltdown_active:
            self.logger.warning(f"[Pullback] meltdown => skip new trades for {symbol}.")
            return

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
            self.logger.info(f"[Debug-ATR] {symbol}: ATR({self.atr_window}) = {atr_value}")

        # (4) Pullback => 15m

        # Stel standaardwaarden in voor de pullback-variabelen
        rsi_val = 50.0  # standaard RSI-waarde
        macd_signal_score = 0  # standaard MACD-score
        current_ws_price = Decimal("0")

        df_entry = self._fetch_and_indicator(symbol, self.entry_timeframe, limit=100)
        if not df_entry.empty:
            # Zorg dat de data oplopend gesorteerd is op index
            df_entry.sort_index(inplace=True)
            last_timestamp = df_entry.index[-1]

            # Controleer of last_timestamp een pd.Timestamp is, anders ga je ervan uit dat het al in milliseconden is
            if isinstance(last_timestamp, pd.Timestamp):
                last_candle_ms = int(last_timestamp.timestamp() * 1000)
            else:
                last_candle_ms = int(last_timestamp)

            # Controleer of de laatste candle volledig is afgesloten
            if not is_candle_closed(last_candle_ms, self.entry_timeframe):
                self.logger.debug(
                    f"[Pullback] {symbol}: Laatste {self.entry_timeframe} candle is nog niet afgesloten; gebruik penultimate candle.")
                if len(df_entry) >= 2:
                    used_entry = df_entry.iloc[-2]
                else:
                    used_entry = df_entry.iloc[-1]
            else:
                used_entry = df_entry.iloc[-1]

            # Haal de RSI en MACD-waarden op uit de DataFrame (of eventueel uit 'used_entry' als dat gewenst is)
            rsi_val = df_entry["rsi"].iloc[-1]
            macd_signal_score = self._check_macd(df_entry)

            # Haal de candle-close op als fallback-prijs
            candle_close_price = Decimal(str(used_entry["close"]))

            # Haal de live WS-prijs op
            ws_price = self._get_ws_price(symbol)
            self.logger.info(
                f"[Debug-15m] {symbol}: last_close(15m)={candle_close_price}, ws_price={ws_price}"
            )

            # Gebruik primair de WS-prijs, fallback naar candle_close
            if ws_price > 0:
                current_price = ws_price
            else:
                current_price = candle_close_price

            pullback_detected = self._detect_pullback(df_entry, current_price, direction)
        else:
            # Als df_entry leeg is, blijven de standaardwaarden behouden
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

        # CHANGED: Gebruik current_price (live) = ws_price of fallback
        if pullback_detected and not has_position:
            if direction == "bull":
                self.logger.info(
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
                self.logger.info(
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
            # CHANGED: Geef dezelfde current_price door
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
        """
        BITVAVO-ONLY AANPASSING:
        i.p.v. Market(...).fetch_candles(...) -> we gebruiken direct self.db_manager.fetch_data("candles_bitvavo", ...).
        Verder ongewijzigd.
        """
        try:
            # i.p.v. market_obj=Market(symbol, self.db_manager) => direct DB call
            df = self.db_manager.fetch_data(
                table_name="candles_bitvavo",
                limit=limit,
                market=symbol,
                interval=interval
            )
            if df.empty:
                self.logger.warning(f"[DEBUG] Geen candles opgehaald uit 'candles_bitvavo' voor {symbol} ({interval}).")
                return pd.DataFrame()

            self.logger.info(f"[DEBUG] Opgehaalde candles voor {symbol} ({interval}): {df.shape[0]} rijen")
            self.logger.debug(f"[DEBUG] Eerste rijen:\n{df.head()}")

            # Drop 'datetime_utc' en 'exchange' (als ze bestaan)
            for col in ['datetime_utc', 'exchange']:
                if col in df.columns:
                    df.drop(columns=col, inplace=True, errors='ignore')

            # We verwachten kolommen: timestamp, market, interval, open, high, low, close, volume
            # (als je fetch_data columns=[timestamp, datetime_utc, market, interval, open, high, low, close, volume, exchange], rename net als voorheen):
            # => pas rename toe als je code dat deed
            # We doen wat je oorspronkelijk deed:
            df.columns = ['timestamp', 'market', 'interval', 'open', 'high', 'low', 'close', 'volume']

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float, errors="ignore")

            self.logger.debug(f"[DEBUG] Data na opschonen:\n{df.head()}")
            self.logger.debug(f"[DEBUG] Close-prijzen vóór indicatorberekeningen:\n{df['close'].head(20)}")

            # Indicatoren
            df.sort_values(by="timestamp", inplace=True)
            df = IndicatorAnalysis.calculate_indicators(df, rsi_window=self.rsi_window)

            # MACD
            self.logger.debug("[DEBUG] Start MACD-berekening...")
            macd_ind = MACD(
                close=df['close'],
                window_slow=self.macd_slow,
                window_fast=self.macd_fast,
                window_sign=self.macd_signal
            )
            df['macd'] = macd_ind.macd()
            df['macd_signal'] = macd_ind.macd_signal()

            # Bollinger
            bb = IndicatorAnalysis.calculate_bollinger_bands(df, window=20, num_std_dev=2)
            df["bb_upper"] = bb["bb_upper"]
            df["bb_lower"] = bb["bb_lower"]

            return df

        except Exception as e:
            self.logger.error(f"[ERROR] _fetch_and_indicator faalde: {e}")
            return pd.DataFrame()

    def _calculate_atr(self, df: pd.DataFrame, window=14) -> Optional[Decimal]:
        if len(df) < window:
            return None
        atr_obj = AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'],
            window=window
        )
        series_atr = atr_obj.average_true_range()
        last_atr = series_atr.iloc[-1]
        if pd.isna(last_atr):
            return None
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
            self.logger.info(f"[Pullback] <{self.pullback_rolling_window} candles => skip.")
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
        if not self.ml_model_enabled or self.ml_engine is None or df.empty:
            return 0
        last_row = df.iloc[-1]
        features = [
            last_row.get("rsi", 50),
            last_row.get("macd", 0),
            last_row.get("macd_signal", 0),
            last_row.get("volume", 0),
        ]
        return self.ml_engine.predict_signal(features)

    # ------------------------------------------------
    #   Open/Manage pos
    # ------------------------------------------------
    def _open_position(self, symbol: str, side: str, current_price: Decimal,
                       atr_value: Decimal, extra_invest=False):
        """
        Opent een nieuwe positie (long of short), slaat de trade op in de DB,
        en voegt deze toe aan self.open_positions.
        """
        # [CHANGED] 1) Check if position already exists
        if symbol in self.open_positions:
            self.logger.warning(
                f"[PullbackStrategy] Already have an open position for {symbol}, skip opening a new one."
            )
            return

        self.logger.info(
            f"[PullbackStrategy] OPEN => side={side}, {symbol}@{current_price}, extra_invest={extra_invest}"
        )

        # 1) Bepaal EUR balance uit order_client (paper of real)
        eur_balance = Decimal("100")
        if self.order_client:
            bal = self.order_client.get_balance()
            eur_balance = Decimal(str(bal.get("EUR", "100")))

        # 2) Check op current_price=0 => skip
        if current_price is None or current_price <= 0:
            self.logger.warning(f"[PullbackStrategy] current_price={current_price} => skip open pos for {symbol}")
            return

        # 3) Bepaal hoeveel EUR we willen investeren
        pct = self.position_size_pct
        if extra_invest:
            pct = Decimal("0.10")
        buy_eur = eur_balance * pct
        if buy_eur < 5:
            self.logger.warning(f"[PullbackStrategy] buy_eur < 5 => skip {symbol}")
            return

        amount = buy_eur / current_price

        # 4) Maak position_id en position_type
        position_id = f"{symbol}-{int(time.time())}"
        if side == "buy":
            position_type = "long"
        else:
            position_type = "short"

        # 5) Plaats (paper)order
        if self.order_client:
            self.order_client.place_order(side, symbol, float(amount), order_type="market")
            self.logger.info(
                f"[LIVE/PAPER] {side.upper()} {symbol} => amt={amount:.4f}, price={current_price}, cost={buy_eur}"
            )

        # Reken trade_cost en fees nu nog als 0.0 tot er fills zijn
        fees = 0.0
        pnl_eur = 0.0
        trade_cost = float(amount * current_price)

        # 6) In DB opslaan (status='open')
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

        new_trade_id = self.db_manager.cursor.lastrowid
        self.logger.info(f"[PullbackStrategy] new trade row => trade_id={new_trade_id}")

        # 7) Eerst self.open_positions[symbol] aanmaken ...
        desired_amount = amount
        self.open_positions[symbol] = {
            "side": side,
            "entry_price": current_price,
            "desired_amount": desired_amount,
            "filled_amount": Decimal("0.0"),
            "amount": Decimal("0.0"),
            "atr": atr_value,
            "tp1_done": False,
            "tp2_done": False,
            "trail_active": False,
            "trail_high": current_price,
            "position_id": position_id,
            "position_type": position_type
        }

        # 8) ... dan de db_id toevoegen
        self.open_positions[symbol]["db_id"] = new_trade_id

    def _manage_open_position(self, symbol: str, current_price: Decimal, atr_value: Decimal):
        if current_price is None or current_price <= 0:
            self.logger.warning(f"[PullbackStrategy] current_price={current_price} => skip manage pos for {symbol}")
            return

        pos = self.open_positions[symbol]
        side = pos["side"]
        entry = pos["entry_price"]
        amount = pos["amount"]

        # ---- VASTE STOP-LOSS EERST ----
        if side == "buy":
            # Bij LONG => stoploss als current_price <= entry*(1 - stop_loss_pct)
            stop_loss_price = entry * (Decimal("1.0") - self.stop_loss_pct)
            if current_price <= stop_loss_price:
                self.logger.info(f"[PullbackStrategy] LONG STOPLOSS => close entire position {symbol}")
                self._sell_portion(
                    symbol, amount, portion=Decimal("1.0"), reason="StopLoss",
                    exec_price=current_price
                )
                if symbol in self.open_positions:
                    del self.open_positions[symbol]
                return

            # -- Bestaande partial take-profits --
            tp1_price = entry + pos["atr"] * self.tp1_atr_mult
            tp2_price = entry + pos["atr"] * self.tp2_atr_mult

            self.logger.info(
                f"[DEBUG-manage-LONG] symbol={symbol}, tp1_done={pos['tp1_done']}, current_price={current_price}, "
                f"tp1_price={tp1_price:.4f}, tp2_done={pos['tp2_done']}, tp2_price={tp2_price:.4f}"
            )

            # TP1
            if (not pos["tp1_done"]) and (current_price >= tp1_price):
                self.logger.info(f"[PullbackStrategy] LONG TP1 => Sell 25% {symbol}")
                self._sell_portion(
                    symbol, amount, portion=Decimal("0.25"), reason="TP1",
                    exec_price=current_price
                )
                pos["tp1_done"] = True
                pos["trail_active"] = True
                pos["trail_high"] = max(pos["trail_high"], current_price)

            # TP2
            elif (not pos["tp2_done"]) and (current_price >= tp2_price):
                self.logger.info(f"[PullbackStrategy] LONG TP2 => Sell 25% {symbol}")
                self._sell_portion(
                    symbol, amount, portion=Decimal("0.25"), reason="TP2",
                    exec_price=current_price
                )
                pos["tp2_done"] = True
                pos["trail_active"] = True
                pos["trail_high"] = max(pos["trail_high"], current_price)

            # Trailing stop
            if pos["trail_active"]:
                if current_price > pos["trail_high"]:
                    pos["trail_high"] = current_price
                trailing_stop_price = pos["trail_high"] - (atr_value * self.trail_atr_mult)
                self.logger.info(
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
            stop_loss_price = entry * (Decimal("1.0") + self.stop_loss_pct)
            if current_price >= stop_loss_price:
                self.logger.info(f"[PullbackStrategy] SHORT STOPLOSS => close entire position {symbol}")
                self._buy_portion(
                    symbol, amount, portion=Decimal("1.0"), reason="StopLoss",
                    exec_price=current_price
                )
                if symbol in self.open_positions:
                    del self.open_positions[symbol]
                return

            tp1_price = entry - pos["atr"] * self.tp1_atr_mult
            tp2_price = entry - pos["atr"] * self.tp2_atr_mult

            self.logger.info(
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
                self.logger.info(
                    f"[DEBUG-trailing-SHORT] symbol={symbol}, entry={entry}, "
                    f"trail_stop={trailing_stop_price:.4f}, current_price={current_price}"
                )
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

        self.logger.info(
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

            # (NIEUW) => update DB => status='closed'
            db_id = pos.get("db_id", None)
            if db_id:
                self.db_manager.update_trade(db_id, {"status": "closed"})
                self.logger.info(f"[PullbackStrategy] Trade {db_id} => status=closed in DB")

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

        self.logger.info(
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

            # (NIEUW) => update DB => status='closed'
            db_id = pos.get("db_id", None)
            if db_id:
                self.db_manager.update_trade(db_id, {"status": "closed"})
                self.logger.info(f"[PullbackStrategy] Trade {db_id} => status=closed in DB")

            del self.open_positions[symbol]

    # ------------------------------------------------
    #   Hulp: equity, ws price
    # ------------------------------------------------
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
        """
        AANPASSING: i.p.v. self.db_manager.get_ticker(symbol) => BITVAVO => fetch_data("ticker_bitvavo", ...)
        """
        # i.p.v. 'get_ticker(symbol)', we doen direct bitvavo's 'ticker_bitvavo'.
        df_ticker = self.db_manager.fetch_data(
            table_name="ticker_bitvavo",
            limit=1,
            market=symbol
        )
        if not df_ticker.empty:
            best_bid = df_ticker["best_bid"].iloc[0] if "best_bid" in df_ticker.columns else 0
            best_ask = df_ticker["best_ask"].iloc[0] if "best_ask" in df_ticker.columns else 0
            if best_bid > 0 and best_ask > 0:
                return (Decimal(str(best_bid)) + Decimal(str(best_ask))) / Decimal("2")

        # Fallback => 1m in 'candles_bitvavo'
        df_1m = self.db_manager.fetch_data(
            table_name="candles_bitvavo",
            limit=1,
            market=symbol,
            interval="1m"
        )
        if not df_1m.empty and "close" in df_1m.columns:
            last_close = df_1m["close"].iloc[0]
            return Decimal(str(last_close))

        return Decimal("0")

    def _get_ws_price(self, symbol):
        if not self.data_client:
            self.logger.warning("[PullbackStrategy] data_client=None => return price=0")
            return Decimal("0")

        price = self.data_client.get_price_with_fallback(symbol, max_age=10)
        return price if price > 0 else Decimal("0")

    def update_position_with_fill(self, fill_data: dict):
        """
        Voorbeeld fill_data:
        {
          "event": "fill",
          "orderId": "...",
          "market": "BTC-EUR",
          "side": "buy",    # Dit is de kant van de order
          "amount": "0.03", # fill-amount
          "price": "30000",
          "fee": "0.1",
          ...
        }
        """
        symbol = fill_data.get("market")
        fill_side = fill_data.get("side", "").lower()
        fill_amt = Decimal(str(fill_data.get("amount", "0")))
        fill_price = Decimal(str(fill_data.get("price", "0")))

        if symbol not in self.open_positions:
            self.logger.info(f"[update_position_with_fill] Geen open positie voor {symbol}, skip fill.")
            return

        pos = self.open_positions[symbol]

        # Voorbeeld: als 'side' == 'buy', is dit een LONG-open (of short-close),
        # maar in jouw code is 'pos["side"]' == 'buy' => je opent long.
        # We gaan ervan uit dat 'fill_side' == pos["side"] => partial fill van open.
        # (Zo niet, dan is 'fill_side' = 'buy' om short te sluiten, wat meer logic vergt.)

        # === 1) Herbereken average entry_price als volume-weighted average ===
        old_filled = pos["filled_amount"]
        new_filled = old_filled + fill_amt

        if new_filled > Decimal("0"):
            old_price = pos["entry_price"]
            # volume-weigted:
            # pos["entry_price"] = (old_price * old_filled + fill_price * fill_amt) / new_filled
            # Maar als old_filled=0 => direct new fill => just fill_price
            if old_filled == 0:
                pos["entry_price"] = fill_price
            else:
                pos["entry_price"] = ((old_price * old_filled) + (fill_price * fill_amt)) / new_filled

        # === 2) Update filled_amount & actual 'amount' ===
        pos["filled_amount"] = new_filled
        pos["amount"] = new_filled  # we doen alsof 'amount' = echt open volume

        # === 3) Check of we nu fully filled ===
        desired = pos["desired_amount"]
        if pos["filled_amount"] >= desired:
            # Volledig open (of overshoot)
            pos["amount"] = desired
            pos["filled_amount"] = desired
            self.logger.info(
                f"[update_position_with_fill] {symbol}: order fully filled => {desired} / {desired}"
            )
        else:
            self.logger.info(
                f"[update_position_with_fill] {symbol}: partial fill => {pos['filled_amount']}/{desired} @ last fill price={fill_price}"
            )

    def _load_open_positions_from_db(self):
        """
        Leest alle trades met status='open' uit de DB,
        en zet ze in self.open_positions[symbol].
        """
        open_rows = self.db_manager.fetch_open_trades()
        if not open_rows:
            self.logger.info("[PullbackStrategy] Geen open trades gevonden in DB.")
            return

        for row in open_rows:
            symbol = row["symbol"]
            side = row["side"]
            amount = Decimal(str(row["amount"]))
            entry_price = Decimal(str(row["price"]))
            position_id = row.get("position_id", None)
            position_type = row.get("position_type", None)

            # Bouw net zo'n dict als je _open_position() doet:
            # ATR weten we niet, dus zetten we op 0.0 of later herberekenen
            pos_data = {
                "side": side,
                "entry_price": entry_price,
                "amount": amount,
                "atr": Decimal("0.0"),
                "tp1_done": False,
                "tp2_done": False,
                "trail_active": False,
                "trail_high": entry_price,
                "position_id": position_id,
                "position_type": position_type
            }

            # === Nieuw: bereken de ATR opnieuw voor main_timeframe ===
            df_main = self._fetch_and_indicator(symbol, self.main_timeframe, limit=200)
            atr_value = self._calculate_atr(df_main, self.atr_window)
            if atr_value:
                pos_data["atr"] = Decimal(str(atr_value))
            else:
                # Als nog niet genoeg data, laat 0.0 staan
                pass

            self.open_positions[symbol] = pos_data
            self.logger.info(
                f"[PullbackStrategy] Hersteld open pos in memory => {symbol}, side={side}, amt={amount}, entry={entry_price}"
            )
