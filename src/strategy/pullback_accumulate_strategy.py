import logging
import pandas as pd
import yaml
import time
from datetime import datetime

from typing import Optional
from decimal import Decimal
from collections import deque

# TA-bibliotheken
from ta.volatility import AverageTrueRange
from ta.trend import MACD
from ta.trend import ADXIndicator  # NIEUW

# Lokale imports
from src.config.config import PULLBACK_STRATEGY_LOG_FILE, PULLBACK_CONFIG, load_config_file
from src.logger.logger import setup_logger
from src.indicator_analysis.indicators import IndicatorAnalysis
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

def is_candle_closed(candle_timestamp_ms: int, timeframe: str) -> bool:
    """
    We gaan ervan uit dat candle_timestamp_ms = candle-eindtijd (gesloten).
    Dus: closed als now_ms >= candle_timestamp_ms.
    """
    now_ms = int(time.time() * 1000)
    return now_ms >= candle_timestamp_ms


class PullbackAccumulateStrategy:
    """
    Pullback & Accumulate Strategy
    ---------------------------------------------------------
     - self.data_client => om live data te zien
     - self.order_client => om orders te plaatsen / get_balance()
     - self.db_manager => database afhandeling

    [AANPASSING] Code is opgeschoond om enkel 4h (trend) + 15m (entry) te gebruiken,
                  en de 'daily' is uitgecommentarieerd (zie # [UITGECOMMENTARIEERD]).
                  Ook is 'R' geïntroduceerd als risk-eenheid = (1 × ATR).
    """

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

        self.logger = setup_logger("pullback_strategy", PULLBACK_STRATEGY_LOG_FILE,
                                   logging.INFO)  # kan weer naar INFO indien nodig
        if config_path:
            self.logger.debug("[PullbackAccumulateStrategy] init with config_path=%s", config_path)
        else:
            self.logger.debug("[PullbackAccumulateStrategy] init (no config_path)")

        # -----------------------------------------------------
        # MeltdownManager krijgt eigen logger (ipv self.logger)
        # -----------------------------------------------------
        meltdown_logger = setup_logger(
            name="meltdown_manager",
            log_file="logs/meltdown_manager.log",
            level=logging.DEBUG
        )

        # Laden config, bv. meltdown_cfg = full_config.get("meltdown_manager", {})
        meltdown_cfg = self.strategy_config.get("meltdown_manager", {})
        self.meltdown_manager = MeltdownManager(meltdown_cfg, db_manager=db_manager, logger=meltdown_logger)
        self.initial_capital = Decimal(str(self.strategy_config.get("initial_capital", "100")))

        # -----------------------------------------------------
        # Overige configuraties uit YAML
        # -----------------------------------------------------
        self.pullback_rolling_window = int(self.strategy_config.get("pullback_rolling_window", 20))

        # [UITGECOMMENTARIEERD] daily, tenzij je die later wilt activeren
        # self.daily_timeframe = self.strategy_config.get("daily_timeframe", "1d")

        # 4h timeframe als "hoofdtf" -> trend check
        self.trend_timeframe = self.strategy_config.get("trend_timeframe", "4h")

        # main_timeframe (1h) => we gebruiken 'm om ATR te berekenen
        self.main_timeframe = self.strategy_config.get("main_timeframe", "1h")

        # entry_timeframe (15m)
        self.entry_timeframe = self.strategy_config.get("entry_timeframe", "15m")
        self.flash_crash_tf = self.strategy_config.get("flash_crash_timeframe", "5m")

        # ATR / SL / TP / TR
        self.atr_window = int(self.strategy_config.get("atr_window", 14))
        self.pullback_atr_mult = Decimal(str(self.strategy_config.get("pullback_atr_mult", "1.0")))
        self.tp1_atr_mult = Decimal(str(self.strategy_config.get("tp1_atr_mult", "1.0")))
        self.trail_atr_mult = Decimal(str(self.strategy_config.get("trailing_atr_mult", "1.0")))
        self.sl_atr_mult = Decimal(str(self.strategy_config.get("sl_atr_mult", "1.0")))

        self.log_file = self.strategy_config.get("log_file", PULLBACK_STRATEGY_LOG_FILE)

        # partial TP
        self.tp1_portion_pct = Decimal(str(self.strategy_config.get("tp1_portion_pct", "0.50")))

        # Overige filters / drempels
        self.h4_bull_rsi = float(self.strategy_config.get("h4_bull_rsi", 50))   # bovengrens bull
        self.h4_bear_rsi = float(self.strategy_config.get("h4_bear_rsi", 50))   # ondergrens bear

        self.use_depth_trend = bool(self.strategy_config.get("use_depth_trend", True))
        self.use_ema_pullback_check = bool(self.strategy_config.get("use_ema_pullback_check", False))  # [NIEUW]
        self.pullback_ema_period = int(self.strategy_config.get("pullback_ema_period", 20))
        self.pullback_ema_tolerance_bull = float(self.strategy_config.get("pullback_ema_tolerance_bull", 1.02))
        self.pullback_ema_tolerance_bear = float(self.strategy_config.get("pullback_ema_tolerance_bear", 0.98))

        self.macd_bull_threshold = float(self.strategy_config.get("macd_bull_threshold", 0))
        self.macd_bear_threshold = float(self.strategy_config.get("macd_bear_threshold", 0))

        # RSI/MACD config
        self.rsi_window = int(self.strategy_config.get("rsi_window", 14))
        self.macd_fast = int(self.strategy_config.get("macd_fast", 12))
        self.macd_slow = int(self.strategy_config.get("macd_slow", 26))
        self.macd_signal = int(self.strategy_config.get("macd_signal", 9))

        # === ADX-config (uit YAML) ===
        self.use_adx_filter = bool(self.strategy_config.get("use_adx_filter", False))
        self.use_adx_directional_filter = bool(self.strategy_config.get("use_adx_directional_filter", False))
        self.use_adx_multitimeframe = bool(self.strategy_config.get("use_adx_multitimeframe", False))

        self.adx_window = int(self.strategy_config.get("adx_window", 14))
        self.adx_entry_tf_threshold = float(self.strategy_config.get("adx_entry_tf_threshold", 20.0))  # op entry TF (nu 1h)
        self.adx_high_tf_threshold = float(self.strategy_config.get("adx_high_tf_threshold", 20.0))  # op 4h

        # ML
        self.ml_model_enabled = bool(self.strategy_config.get("ml_model_enabled", False))
        self.ml_model_path = self.strategy_config.get("ml_model_path", "models/pullback_model.pkl")
        self.ml_engine = None

        # Limit ordergroottes
        self.min_lot_multiplier = Decimal(str(self.strategy_config.get("min_lot_multiplier", "2.1")))
        self.max_position_pct = Decimal(str(self.strategy_config.get("max_position_pct", "0.05")))
        self.max_position_eur = Decimal(str(self.strategy_config.get("max_position_eur", "250")))

        # Let op: meltdown / accum
        self.accumulate_threshold = Decimal(str(self.strategy_config.get("accumulate_threshold", "1.25")))

        # open_positions
        self.open_positions = {}
        self.invested_extra = False
        self.depth_trend_history = deque(maxlen=5)

        #Maak variabele aan om surplus op te slaan
        self.surplus_above_100 = Decimal("0")  # <-- NIEUW

        self._load_open_positions_from_db()

        # Na een nieuwe candle maar 1x de strategie uitvoeren
        self.last_processed_candle_ts = {}  # [ADDED] dict: {symbol: last_candle_ms we used}

    def execute_strategy(self, symbol: str):
        meltdown_active = self.meltdown_manager.update_meltdown_state(strategy=self, symbol=symbol)
        """
        Eenvoudige flow:
         1) meltdown-check
         2) bepaal trend op 4h
         3) ATR op 1h
         4) check pullback op 15m
         5) manage pos of open pos
        """
        # Check of er een open positie is voor dit symbool
        has_position = (symbol in self.open_positions)

        if meltdown_active:
            self.logger.warning(f"[Pullback] meltdown => skip opening new trades for {symbol}.")

            # Blijf wel open posities managen
            if has_position:
                current_price = self._get_ws_price(symbol)
                if current_price <= 0:
                    self.logger.warning(f"[Pullback] meltdown: current_price=0 => skip manage pos.")
                    return
                atr_value = self.open_positions[symbol]["atr"]  # of opnieuw berekenen via _calculate_atr(...)
                self._manage_open_position(symbol, current_price, atr_value)
            return

        # -- ER IS GEEN MELTDOWN, GA DOOR --

        # Concurrency-check / check of we al open trades hebben in DB
        existing_db_trades = self.db_manager.execute_query(
            """
            SELECT id
              FROM trades
             WHERE symbol=?
               AND is_master=1
               AND status IN ('open','partial')
               AND strategy_name='pullback'
             LIMIT 1
            """,
            (symbol,)
        )

        if existing_db_trades:
            self.logger.info(
                f"[execute_strategy] Already have open/partial MASTER trade in DB for {symbol} => skip opening.")

            # Wel managen als we die open positie nog in self.open_positions hebben
            if has_position:
                current_price = self._get_ws_price(symbol)
                atr_value = self.open_positions[symbol]["atr"]
                self._manage_open_position(symbol, current_price, atr_value)
            return

        # Check of er >100 EUR vrij is, sla surplus op
        bal_dict = self.order_client.get_balance()
        free_eur = Decimal(bal_dict.get("EUR", "0"))
        if free_eur > 100:
            self.surplus_above_100 = free_eur - 100
            self.logger.info(f"[SurplusCheck] Found EUR {free_eur:.2f}, surplus={self.surplus_above_100:.2f}")
        else:
            self.surplus_above_100 = Decimal("0")

        # Trend => 4h RSI
        df_h4 = self._fetch_and_indicator(symbol, self.trend_timeframe, limit=200)
        if df_h4.empty:
            self.logger.warning(f"[PullbackStrategy] No 4h data => skip {symbol}")
            return

        rsi_h4 = df_h4["rsi"].iloc[-1]

        # Haal MACD; als kolom ontbreekt, zet macd_h4=0
        if "macd" in df_h4.columns:
            macd_h4 = df_h4["macd"].iloc[-1]
        else:
            macd_h4 = 0.0

        direction = self._check_trend_direction_4h(rsi_h4, macd_h4)  # let op: extra arg
        self.logger.info(
            f"[PullbackStrategy] direction={direction}, rsi_h4={rsi_h4:.2f}, macd_h4={macd_h4:.2f} for {symbol}")

        # [UITGECOMMENTARIEERD] daily-check
        # df_daily = self._fetch_and_indicator(symbol, self.daily_timeframe, limit=60)
        # if df_daily.empty:
        #     self.logger.warning(f"[PullbackStrategy] No daily data => skip {symbol}")
        #     return
        # rsi_daily = df_daily["rsi"].iloc[-1]

        # ATR => 1h
        df_main = self._fetch_and_indicator(symbol, self.main_timeframe, limit=200)
        if df_main.empty:
            self.logger.warning(f"[PullbackStrategy] No {self.main_timeframe} data => skip {symbol}")
            return
        atr_value = self._calculate_atr(df_main, self.atr_window)
        if atr_value is None:
            self.logger.warning(f"[PullbackStrategy] Not enough data => skip {symbol}")
            return

        # ============== BEGIN RSI-SLOPE SNIPPET ==============
        # 1) Lees filter-instellingen uit config
        use_slope_filter = bool(self.strategy_config.get("use_rsi_slope_filter", False))
        slope_bull = float(self.strategy_config.get("rsi_slope_min_change_bull", 0.0))
        slope_bear = float(self.strategy_config.get("rsi_slope_min_change_bear", 0.0))

        # 2) Voeg kolom 'rsi_slope' toe (verschil van de laatste 2 candles)
        if "rsi" not in df_main.columns:
                self.logger.warning(f"[RSI-slope] df_main for {symbol} heeft geen RSI-kolom => skip slope check.")
        else:
                df_main["rsi_slope"] = df_main["rsi"].diff()
                rsi_slope_now = df_main["rsi_slope"].iloc[-1]

                if use_slope_filter:
                    # Als direction=bull, maar RSI-slope <= slope_bull => skip LONG.
                    if direction == "bull" and rsi_slope_now <= slope_bull:
                        self.logger.info(
                            f"[RSI-slope] bull-richting, maar slope={rsi_slope_now:.2f} <= {slope_bull} => skip LONG {symbol}"
                        )
                        return

                    # Als direction=bear, maar RSI-slope >= slope_bear => skip SHORT.
                    if direction == "bear" and rsi_slope_now >= slope_bear:
                        self.logger.info(
                            f"[RSI-slope] bear-richting, maar slope={rsi_slope_now:.2f} >= {slope_bear} => skip SHORT {symbol}"
                        )
                        return

        # Pullback => 15m
        df_entry = self._fetch_and_indicator(symbol, self.entry_timeframe, limit=100)
        if df_entry.empty:
            self.logger.warning(f"[PullbackStrategy] No {self.entry_timeframe} data => skip {symbol}")
            return

        df_entry.sort_index(inplace=True)
        last_timestamp = df_entry.index[-1]
        if isinstance(last_timestamp, pd.Timestamp):
            epoch_s = last_timestamp.timestamp()
            last_candle_ms = int(epoch_s * 1000)
            self.logger.debug(f"[DEBUG] {symbol}: last row => pd.Timestamp {last_timestamp} => epoch_s={epoch_s}")
        else:
            last_candle_ms = int(last_timestamp)
            epoch_s = last_candle_ms / 1000
            self.logger.debug(f"[DEBUG] {symbol}: last row => {last_candle_ms} => {datetime.utcfromtimestamp(epoch_s)}")

        # -------------- Skip direct als candle niet gesloten --------------
        if not is_candle_closed(last_candle_ms, self.entry_timeframe):
            self.logger.debug(f"[Pullback] {symbol}: Candle NOT closed => skip => pass #2 later.")
            return "skip_not_closed"
        # -------------------------------------------------------------------------

        prev_candle_ts = self.last_processed_candle_ts.get(symbol, None)
        if prev_candle_ts == last_candle_ms:
            self.logger.debug(f"[Pullback] {symbol}: Candle {last_candle_ms} al verwerkt => return.")
            return

        self.last_processed_candle_ts[symbol] = last_candle_ms

        # ---------------- ADX FILTERS (NIEUW) ----------------
        # 4h ADX (multitimeframe trendsterkte)
        adx_h4 = None
        if "adx" in df_h4.columns:
            adx_h4_series = df_h4["adx"].dropna()
            if not adx_h4_series.empty:
                try:
                    adx_h4 = float(adx_h4_series.iloc[-1])
                except Exception:
                    adx_h4 = None

        if self.use_adx_multitimeframe:
            if adx_h4 is None or adx_h4 < self.adx_high_tf_threshold:
                self.logger.info(f"[ADX-4h] adx_h4={adx_h4} < {self.adx_high_tf_threshold} => skip {symbol}")
                return

        # Entry‑TF ADX (nu 1h, want entry_timeframe is '1h' in je config)
        if "adx" not in df_entry.columns or df_entry["adx"].dropna().empty:
            self.logger.warning(f"[ADX-entry] geen bruikbare ADX op {self.entry_timeframe} => skip {symbol}")
            return

        try:
            adx_entry = float(df_entry["adx"].dropna().iloc[-1])
        except Exception:
            adx_entry = 0.0

        if self.use_adx_filter and adx_entry < self.adx_entry_tf_threshold:
            self.logger.info(f"[ADX-entry] adx={adx_entry:.2f} < {self.adx_entry_tf_threshold} => skip {symbol}")
            return

        # Richtingsfilter met DI+/DI- op entry‑TF
        if "di_pos" not in df_entry.columns or "di_neg" not in df_entry.columns:
            self.logger.warning(f"[ADX-DI] kolommen ontbreken op {self.entry_timeframe} => skip {symbol}")
            return
        di_pos_series = df_entry["di_pos"].dropna()
        di_neg_series = df_entry["di_neg"].dropna()
        if di_pos_series.empty or di_neg_series.empty:
            self.logger.warning(f"[ADX-DI] DI niet bruikbaar (NaN) op {self.entry_timeframe} => skip {symbol}")
            return
        try:
            di_pos = float(di_pos_series.iloc[-1])
            di_neg = float(di_neg_series.iloc[-1])
        except Exception:
            di_pos, di_neg = None, None

        if di_pos is None or di_neg is None:
            self.logger.warning(f"[ADX-DI] DI waarden niet leesbaar => skip {symbol}")
            return

        if direction == "bull" and not (di_pos > di_neg):
            self.logger.info(f"[ADX-DI] bull maar +DI<=-DI ({di_pos:.2f}<= {di_neg:.2f}) => skip LONG {symbol}")
            return

        if direction == "bear" and not (di_neg > di_pos):
            self.logger.info(f"[ADX-DI] bear maar -DI<=+DI ({di_neg:.2f}<= {di_pos:.2f}) => skip SHORT {symbol}")
            return

        # -------------- EINDE ADX FILTERS --------------------

        # Haal current price
        candle_close_price = Decimal(str(df_entry["close"].iloc[-1]))
        ws_price = self._get_ws_price(symbol)
        if ws_price > 0:
            current_price = ws_price
        else:
            current_price = candle_close_price

        # Bepaal of er pullback is, en check evt. 9/20EMA als je dat wilt
        pullback_detected = self._detect_pullback(df_entry, current_price, direction, atr_value)
        if self.use_ema_pullback_check:
            # Extra check: 9EMA + 20EMA
            if not self._check_ema_pullback_15m(df_entry, direction):
                pullback_detected = False
                self.logger.info(f"[PullbackStrategy] 9/20EMA-check => geen valide pullback => skip {symbol}")

        # 2) MACD-check
        if pullback_detected:
            # Pak de laatste MACD‐waarde van de 15m data
            last_macd_15m = df_entry["macd"].iloc[-1]

            # a) Bear-check: skip short als MACD > macd_bear_threshold
            if direction == "bear":
                if last_macd_15m > self.macd_bear_threshold:
                    self.logger.info(
                        f"[MACD-filter-BEAR] macd_15m={last_macd_15m:.2f} > {self.macd_bear_threshold} "
                        f"=> momentum is te bullish => skip short."
                    )
                    pullback_detected = False

            # b) Bull-check: skip long als MACD < macd_bull_threshold
            elif direction == "bull":
                if last_macd_15m < self.macd_bull_threshold:
                    self.logger.info(
                        f"[MACD-filter-BULL] macd_15m={last_macd_15m:.2f} < {self.macd_bull_threshold} "
                        f"=> momentum is te bearish => skip long."
                    )
                    pullback_detected = False

        # Depth + ML
        ml_signal = self._ml_predict_signal(df_entry)
        depth_score = 0.0
        if self.use_depth_trend:
            depth_score_instant = self._analyze_depth_trend_instant(symbol)
            # Rolling average
            self.depth_trend_history.append(depth_score_instant)
            depth_score = sum(self.depth_trend_history) / len(self.depth_trend_history)
            # (4) DepthTrend-regel uitgecommentarieerd:
            # self.logger.info(f"[DepthTrend] instant={depth_score_instant:.2f}, rolling_avg={depth_score:.2f}")

        # Equity check
        total_equity = self._get_equity_estimate()
        invest_extra_flag = False
        #if (total_equity >= self.initial_capital * self.accumulate_threshold) and not self.invested_extra:
        #    invest_extra_flag = True
        #    self.logger.info("[PullbackStrategy] +25%% => next pullback => invest extra in %s", symbol)

        has_position = (symbol in self.open_positions)

        self.logger.info(
            f"[execute_strategy] {symbol} | meltdown={meltdown_active} | "
            f"ATR={atr_value:.3f} | direction={direction} | pullback={pullback_detected} | "
            f"ml={ml_signal} | cprice={current_price:.2f}"
        )

        # Als we geen positie hebben, maar wel pullback + bull => open long
        # no daily rsi check anymore, just h4-based direction + pullback
        if pullback_detected and not has_position:
            if direction == "bull":
                # (Hier kun je nog RSI-checks doen op 15m als je wilt)
                self.logger.info(f"[OPEN LONG] {symbol} at {current_price} | ml_signal={ml_signal}")
                self._open_position(symbol, side="buy",
                                    current_price=current_price,
                                    atr_value=atr_value,
                                    extra_invest=invest_extra_flag)
                if invest_extra_flag:
                    self.invested_extra = True

            elif direction == "bear":
                self.logger.info(f"[OPEN SHORT] {symbol} at {current_price} | ml_signal={ml_signal}")
                self._open_position(symbol, side="sell",
                                    current_price=current_price,
                                    atr_value=atr_value,
                                    extra_invest=invest_extra_flag)
                if invest_extra_flag:
                    self.invested_extra = True

        # Als we al een positie hebben => manage
        elif has_position:
            self._manage_open_position(symbol, current_price, atr_value)

    # Simpele check: rsi_h4 > self.h4_bull_rsi => bull, < self.h4_bear_rsi => bear, anders range
    def _check_trend_direction_4h(self, rsi_h4: float, macd_h4: float) -> str:
        """
        Bepaalt de 'direction' op 4h op basis van RSI óf MACD.
        Als RSI > h4_bull_rsi of macd_h4>0 => bull
        Als RSI < h4_bear_rsi of macd_h4<0 => bear
        Anders => range
        """
        if (rsi_h4 > self.h4_bull_rsi) or (macd_h4 > 0):
            return "bull"
        elif (rsi_h4 < self.h4_bear_rsi) or (macd_h4 < 0):
            return "bear"
        else:
            return "range"

    def _detect_pullback(self,
                         df: pd.DataFrame,
                         current_price: Decimal,
                         direction: str,
                         atr_value: Decimal) -> bool:
        """
        Simpele pullback-check op basis van ATR + 20EMA-check + tolerance factor
        """

        # 1) Check genoeg candles
        if len(df) < self.pullback_rolling_window:
            self.logger.info(f"[Pullback] <{self.pullback_rolling_window} candles => skip.")
            return False

        # 2) Alleen relevant in bull/bear; bij 'range' => return False
        if direction not in ("bull", "bear"):
            return False

        # 1) Zorg dat we een kolom "ema_X" hebben (X = pullback_ema_period)
        ema_col_name = f"ema_{self.pullback_ema_period}"
        if ema_col_name not in df.columns:
            df[ema_col_name] = df["close"].ewm(span=self.pullback_ema_period).mean()

        # Voor straks
        pullback_distance = Decimal("0")
        atr_threshold = Decimal("0")
        ratio = Decimal("0")

        # 3) Bull-richting
        if direction == "bull":
            recent_high = df["high"].rolling(self.pullback_rolling_window).max().iloc[-1]
            if recent_high <= 0:
                return False

            pullback_distance = Decimal(str(recent_high)) - current_price
            atr_threshold = atr_value * self.pullback_atr_mult

            ratio = Decimal("0")
            if atr_threshold != 0:
                ratio = pullback_distance / atr_threshold

            # Log de waarden om 'unused variable' te vermijden
            self.logger.info(f"[Pullback-bull] ratio={ratio:.2f} (>=1 => ok?)")

            # Echte check: >= 1 => “pullback genoeg”
            if ratio >= 1:
                # (1) Hebben we 'genoeg' daling? Ja => check EMA
                ema_val = Decimal(str(df[ema_col_name].iloc[-1]))
                if current_price <= ema_val * Decimal(str(self.pullback_ema_tolerance_bull)):
                    self.logger.info(
                        f"[Pullback-bull] price={current_price} <= {ema_val} × {self.pullback_ema_tolerance_bull} => DETECTED"
                    )
                    return True
                else:
                    self.logger.info(
                        f"[Pullback-bull] ratio OK, maar price={current_price:.2f} is boven {ema_val:.2f} × {self.pullback_ema_tolerance_bull}"
                    )
                    return False
            else:
                return False

        # 4) Bear-richting
        else:  # direction == "bear"
            recent_low = df["low"].rolling(self.pullback_rolling_window).min().iloc[-1]
            if recent_low <= 0:
                return False
            rally_distance = current_price - Decimal(str(recent_low))
            atr_threshold = atr_value * self.pullback_atr_mult

            ratio = Decimal("0")
            if atr_threshold > 0:
                ratio = rally_distance / atr_threshold

            self.logger.info(f"[Pullback-bear] ratio={ratio:.2f} (>=1 => ok?)")

            if ratio >= 1:
                ema_val = Decimal(str(df[ema_col_name].iloc[-1]))
                # => Bij bear-check: current_price >= ema_val * self.pullback_ema_tolerance_bear
                if current_price >= ema_val * Decimal(str(self.pullback_ema_tolerance_bear)):
                    self.logger.info(
                        f"[Pullback-bear] price={current_price} >= {ema_val} × {self.pullback_ema_tolerance_bear} => DETECTED"
                    )
                    return True
                else:
                    self.logger.info(
                        f"[Pullback-bear] ratio OK, maar price={current_price:.2f} is onder {ema_val:.2f} × {self.pullback_ema_tolerance_bear}"
                    )
                    return False
            else:
                return False

    # Eenvoudige check: 9EMA, 20EMA, pullback in uptrend => price zakt door 9ema maar boven 20ema
    def _check_ema_pullback_15m(self, df: pd.DataFrame, direction: str) -> bool:
        """
        Gebruikt de laatste candle(15m) en checkt de 9EMA & 20EMA:
          - bull => prijs > 20ema, maar (candle-close) net onder 9ema => pullback,
            en de candle is bezig te herstellen (sluit niet ver onder 20ema).
          - bear => omgekeerd, etc.

        Simpel, ter illustratie. Pas het naar eigen wens aan.
        """
        # Als er geen columns 'ema_9', 'ema_20' zijn, rekenen we ze even uit.
        if "ema_9" not in df.columns or "ema_20" not in df.columns:
            df["ema_9"] = df["close"].ewm(span=9).mean()
            df["ema_20"] = df["close"].ewm(span=20).mean()

        last_close = df["close"].iloc[-1]
        last_ema9 = df["ema_9"].iloc[-1]
        last_ema20 = df["ema_20"].iloc[-1]

        if direction == "bull":
            # bull: we willen dat de prijs boven 20ema zit (grotere uptrend)
            # en net even onder 9ema of rond 9ema => pullback
            if (last_close > last_ema20) and (last_close <= last_ema9 * 1.01):
                self.logger.info(f"[EMA-check-bull] close={last_close}, ema9={last_ema9}, ema20={last_ema20} => True")
                return True
            else:
                return False

        elif direction == "bear":
            # bear: we willen dat de prijs onder 20ema zit, en net even boven 9ema => pullback
            if (last_close < last_ema20) and (last_close >= last_ema9 * 0.99):
                self.logger.info(f"[EMA-check-bear] close={last_close}, ema9={last_ema9}, ema20={last_ema20} => True")
                return True
            else:
                return False

        else:
            return False

    def _manage_open_position(self, symbol: str, current_price: Decimal, atr_value: Decimal):
        """
        Bij open positie => check SL, TP1, trailing, etc.
        [AANPASSING] R-concept = 1 × ATR
        """
        # 1) Check of 'symbol' überhaupt in open_positions zit
        if symbol not in self.open_positions:
            self.logger.warning(f"[manage_open_position] {symbol} not in open_positions => skip.")
            return

        # 2) Check of current_price > 0
        if current_price is None or current_price <= 0:
            self.logger.warning(f"[PullbackStrategy] current_price={current_price} => skip manage pos for {symbol}")
            return

        # 3) Debug-log BEGIN
        pos = self.open_positions[symbol]
        self.logger.debug(
            f"[_manage_open_position] START => symbol={symbol}, side={pos['side']}, amount={pos['amount']}, "
            f"entry={pos['entry_price']}, leftoverCheck?"
        )

        # 4) Overnemen van je huidige code
        side = pos["side"]
        entry = pos["entry_price"]
        amount = pos["amount"]
        one_r = atr_value

        if side == "buy":
            stop_loss_price = entry - (atr_value * self.sl_atr_mult)
            if current_price <= stop_loss_price:
                self.logger.info(f"[ManagePos] LONG STOPLOSS => close entire {symbol}")

                # (A) Pak master_id direct uit pos en zet master op 'closed'
                master_id = pos["master_id"]
                self.db_manager.update_trade(master_id, {"status": "closed"})
                self.logger.info(f"[StopLoss] Master {master_id} => closed in DB")

                # (B) Nu pas child-trade voor portion=1.0
                self._sell_portion(symbol, amount, portion=Decimal("1.0"), reason="StopLoss", exec_price=current_price)

                if symbol in self.open_positions:
                    del self.open_positions[symbol]
                return

            # TP1
            tp1_price = entry + (one_r * self.tp1_atr_mult)
            self.logger.info(f"[ManagePos-LONG] {symbol} => 1R={one_r}, tp1_price={tp1_price}, current={current_price}")
            if (not pos["tp1_done"]) and (current_price >= tp1_price):
                self.logger.info(f"[ManagePos-LONG] => TP1 => Sell portion={self.tp1_portion_pct}")
                self._sell_portion(symbol, amount, portion=self.tp1_portion_pct, reason="TP1", exec_price=current_price)
                pos["tp1_done"] = True
                pos["trail_active"] = True
                pos["trail_high"] = max(pos["trail_high"], current_price)

            # trailing stop
            if pos["trail_active"]:
                if current_price > pos["trail_high"]:
                    pos["trail_high"] = current_price
                trailing_stop_price = pos["trail_high"] - (one_r * self.trail_atr_mult)
                self.logger.info(
                    f"[Trailing-LONG] {symbol}, trail_high={pos['trail_high']}, trailing_stop={trailing_stop_price}"
                )
                if current_price <= trailing_stop_price:
                    self.logger.info(f"[TrailingStop] => close entire leftover => {symbol}")
                    master_id = pos["master_id"]
                    self.db_manager.update_trade(master_id, {"status": "closed"})
                    self.logger.info(f"[TrailingStop] Master {master_id} => closed in DB")
                    self._sell_portion(symbol, amount, portion=Decimal("1.0"), reason="TrailingStop",
                                       exec_price=current_price)
                    if symbol in self.open_positions:
                        del self.open_positions[symbol]
        else:  # SHORT
            stop_loss_price = entry + (one_r * self.sl_atr_mult)
            if current_price >= stop_loss_price:
                self.logger.info(f"[ManagePos] SHORT STOPLOSS => close entire {symbol}")

                # (1) Master direct closed
                master_id = pos["master_id"]
                self.db_manager.update_trade(master_id, {"status": "closed"})
                self.logger.info(f"[StopLoss] Master {master_id} => closed in DB")

                # (2) Child => buy leftover
                self._buy_portion(symbol, amount, portion=Decimal("1.0"), reason="StopLoss", exec_price=current_price)

                if symbol in self.open_positions:
                    del self.open_positions[symbol]
                return

            # TP1
            tp1_price = entry - (one_r * self.tp1_atr_mult)
            self.logger.info(
                f"[ManagePos-SHORT] {symbol} => 1R={one_r}, tp1_price={tp1_price}, current={current_price}")
            if (not pos["tp1_done"]) and (current_price <= tp1_price):
                self.logger.info(f"[ManagePos-SHORT] => TP1 => Buy portion={self.tp1_portion_pct}")
                self._buy_portion(symbol, amount, portion=self.tp1_portion_pct, reason="TP1", exec_price=current_price)
                pos["tp1_done"] = True
                pos["trail_active"] = True

            # trailing stop
            if pos["trail_active"]:
                if current_price < pos["trail_high"]:
                    pos["trail_high"] = current_price
                trailing_stop_price = pos["trail_high"] + (one_r * self.trail_atr_mult)
                self.logger.info(
                    f"[Trailing-SHORT] {symbol}, trail_high={pos['trail_high']}, trailing_stop={trailing_stop_price}"
                )
                if current_price >= trailing_stop_price:
                    self.logger.info(f"[TrailingStop] => close entire leftover => {symbol}")

                    # (1) Master closed
                    master_id = pos["master_id"]
                    self.db_manager.update_trade(master_id, {"status": "closed"})
                    self.logger.info(f"[TrailingStop] Master {master_id} => closed in DB")

                    # (2) leftover => portion=1.0
                    self._buy_portion(symbol, amount, portion=Decimal("1.0"), reason="TrailingStop",
                                      exec_price=current_price)

                    if symbol in self.open_positions:
                        del self.open_positions[symbol]
                    return

        # ===== leftover=0 check =====
        pos = self.open_positions.get(symbol)
        if pos:
            leftover_amt = pos["amount"]
            if leftover_amt <= 0 or leftover_amt < self._get_min_lot(symbol):
                master_id = pos["master_id"]
                self.logger.info(f"[Leftover=0 check] => close master {master_id} for {symbol}")
                self.db_manager.update_trade(master_id, {"status": "closed"})
                del self.open_positions[symbol]

        # 5) Debug-log EINDE
        self.logger.debug(f"[_manage_open_position] END => symbol={symbol}, final pos={pos}")

    def _close_position(self, symbol: str, reason: str = "ForcedClose"):
        """
        Forceert het sluiten van de volledige positie,
        zet status=closed in de DB, en verwijdert self.open_positions[symbol].
        """
        if symbol not in self.open_positions:
            self.logger.warning(f"[_close_position] {symbol} staat niet in open_positions => skip.")
            return

        pos = self.open_positions[symbol]
        side = pos["side"]
        amount = pos["amount"]  # leftover
        entry_price = pos["entry_price"]
        db_id = pos.get("db_id", None)

        self.logger.info(f"[_close_position] => side={side}, leftover={amount}, reason={reason}")

        # 1) Sluit via buy_portion (voor short) of sell_portion (voor long)
        if side == "buy":
            self._sell_portion(symbol, amount, portion=Decimal("1.0"), reason=reason)
        else:  # short
            self._buy_portion(symbol, amount, portion=Decimal("1.0"), reason=reason)

        # [MASTER_ID FIX] - Gebruik master_id, niet db_id van de child
        master_id = pos.get("master_id", None)
        if master_id:
            self.db_manager.update_trade(master_id, {"status": "closed"})
            self.logger.info(f"[_close_position] Master trade {master_id} => status=closed in DB (forced).")

        self.logger.debug(
            f"[_close_position] about to del => {symbol}, current open_positions keys={list(self.open_positions.keys())}"
        )

        # 3) Verwijder uit self.open_positions
        if symbol in self.open_positions:
            self.logger.debug(
                f"[_close_position] about to del => {symbol}, current open_positions={list(self.open_positions.keys())}")
            del self.open_positions[symbol]
            self.logger.info(f"[_close_position] open_positions => {symbol} verwijderd.")

    # Kleine helper-functie om ATR te berekenen
    def _calculate_atr(self, df, window=14) -> Optional[Decimal]:
        # 1) check of df een DataFrame is:
        if not isinstance(df, pd.DataFrame):
            self.logger.warning("[_calculate_atr] df is geen DataFrame => return None.")
            return None

        if df.empty or len(df) < window:
            return None
        atr_obj = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=window)
        series_atr = atr_obj.average_true_range()
        last_atr = series_atr.iloc[-1]
        if pd.isna(last_atr):
            return None
        return Decimal(str(last_atr))

    # [UITGECOMMENTARIEERD] de daily-check
    # def _check_trend_direction_daily(self, rsi_daily: float) -> str:
    #     if rsi_daily > 60:
    #         return "bull"
    #     elif rsi_daily < 40:
    #         return "bear"
    #     else:
    #         return "range"

    # [OVERIGE CODE HEB IK NIET GEWIJZIGD]
    # ----------------------------------------------------------------
    # Rest: _open_position(), _buy_portion(), _sell_portion(), DB-calls, etc.
    # ----------------------------------------------------------------

    # Dit heb ik 1:1 gekopieerd uit je huidige code, behalve dat ik
    # commentaar heb toegevoegd waar relevant en sommige logs heb geshort.
    # VÓÓR de codezie je "# [AANPASSING]" of "# [UITGECOMMENTARIEERD]" om te markeren wat er is veranderd.

    # LET OP: Bestaande code hieronder (open_position, buy_portion, etc.) is je ongewijzigde versie, dus je verliest niets.


    # ----------------------------------------------------------------
    # Open_position
    # ----------------------------------------------------------------
    def _open_position(self, symbol: str, side: str, current_price: Decimal,
                       atr_value: Decimal, extra_invest=False):
        """
        [ongewijzigd t.o.v. jouw code, we laten het staan]
        """
        if symbol in self.open_positions:
            self.logger.warning(f"[PullbackStrategy] Already have an open position for {symbol}, skip opening.")
            return

        self.logger.info(
            f"[PullbackStrategy] OPEN => side={side}, {symbol}@{current_price}, extra_invest={extra_invest}")

        if side == "sell":
            self.logger.info(f"### EXTRA LOG ### [OPEN SHORT] {symbol} at {current_price}")
        else:
            self.logger.info(f"### EXTRA LOG ### [OPEN LONG] {symbol} at {current_price}")

        eur_balance = Decimal("250")
        if self.order_client:
            bal = self.order_client.get_balance()
            eur_balance = Decimal(str(bal.get("EUR", "100")))

            # Spot‐only check: als side=='sell' => we hebben de coin nodig in de wallet
            if side == "sell":
                coin_name = symbol.split("-")[0]
                coin_balance = Decimal(str(bal.get(coin_name, "0")))
                if coin_balance <= 0:
                    self.logger.warning(
                        f"[PullbackStrategy] No {coin_name} in wallet => skip short {symbol}."
                    )
                    return

                # Check of we wél wat coin hebben, maar te weinig voor min trade
                needed_coins = self._get_min_lot(symbol) * self.min_lot_multiplier
                if coin_balance < needed_coins:
                    self.logger.warning(
                        f"[PullbackStrategy] Not enough {coin_name} to short => have={coin_balance}, need={needed_coins:.2f} => skip."
                    )
                    return

        if current_price is None or current_price <= 0:
            self.logger.warning(f"[PullbackStrategy] current_price={current_price} => skip open pos for {symbol}")
            return

        equity_now = self._get_equity_estimate()

        # 2) allowed = 5% van je totale equity, capped op self.max_position_eur
        allowed_eur_pct = equity_now * self.max_position_pct
        allowed_eur = min(allowed_eur_pct, self.max_position_eur)

        needed_coins = self._get_min_lot(symbol) * self.min_lot_multiplier
        needed_eur_for_min = needed_coins * current_price

        if needed_eur_for_min > allowed_eur:
            self.logger.warning(
                f"[PullbackStrategy] needed={needed_eur_for_min:.2f} EUR > allowed={allowed_eur:.2f} => skip {symbol}."
            )
            return

        buy_eur = needed_eur_for_min

        # STAP 4: Als extra_invest True is, voeg surplus toe aan buy_eur
        if extra_invest and self.surplus_above_100 > 0:
            self.logger.info(f"[PullbackStrategy] Adding surplus {self.surplus_above_100:.2f} EUR to this trade!")
            buy_eur += self.surplus_above_100
            # Zet surplus direct terug naar 0, zodat we 'm niet nog eens gebruiken
            self.surplus_above_100 = Decimal("0")

        if buy_eur > eur_balance:
            self.logger.warning(
                f"[PullbackStrategy] Not enough EUR => need {buy_eur:.2f}, have {eur_balance:.2f}. skip.")
            return

        amount = buy_eur / current_price

        position_id = f"{symbol}-{int(time.time())}"
        position_type = "long" if side == "buy" else "short"

        if self.order_client:
            try:
                # Zonder 'order_type="market"' als jouw KrakenMixedClient dat niet kent
                self.order_client.place_order(side, symbol, float(amount))
            except Exception as e:
                error_msg = str(e)
                # check of er 'insufficient' in de fout staat:
                if "InsufficientFunds" in error_msg or "insufficient" in error_msg.lower():
                    self.logger.warning(
                        f"[PullbackStrategy] skip => insufficient funds voor {symbol}. Error: {error_msg}")
                else:
                    self.logger.warning(f"[PullbackStrategy] skip => place_order error voor {symbol}: {error_msg}")
                return

        fees = 0.0
        pnl_eur = 0.0
        trade_cost = float(buy_eur)

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
            "trade_cost": trade_cost,
            "strategy_name": "pullback",
            "is_master": 1
        }
        self.db_manager.save_trade(trade_data)

        # [MASTER_ID FIX] - Sla 'master_id' expliciet op
        master_id = self.db_manager.cursor.lastrowid
        self.logger.info(f"[PullbackStrategy] new MASTER trade row => trade_id={master_id}")
        self.logger.debug(f"[_open_position] master_id opgeslagen: {master_id}")  # Nieuw

        self.__record_trade_signals(master_id, event_type="open", symbol=symbol, atr_mult=self.pullback_atr_mult)

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
            "position_type": position_type,
            "master_id": master_id  # Nieuw: master_id opslaan
        }
        # [CONCRETE FIX 2-A] => Zet meteen de werkelijke hoeveelheid in 'amount' en 'filled_amount'"
        self.open_positions[symbol]["amount"] = Decimal(str(amount))
        self.open_positions[symbol]["filled_amount"] = Decimal(str(amount))

    def _buy_portion(self, symbol: str, total_amt: Decimal, portion: Decimal, reason: str, exec_price=None):
        self.logger.info(f"### BUY portion => reason={reason}, symbol={symbol}")
        self.logger.debug(
            f"[_buy_portion] START => total_amt={total_amt}, portion={portion}, "
            f"open_positions keys={list(self.open_positions.keys())}"
        )

        # 1) Bestaande logic: bereken amt_to_buy, leftover, min_lot, etc.
        amt_to_buy = total_amt * portion
        leftover_after_buy = total_amt - amt_to_buy
        min_lot = self._get_min_lot(symbol)

        if portion < 1 and leftover_after_buy > 0 and leftover_after_buy < min_lot:
            self.logger.info(f"[buy_portion] leftover {leftover_after_buy} < minLot={min_lot}, "
                             f"force entire close => portion=1.0")
            amt_to_buy = total_amt
            portion = Decimal("1.0")

        if amt_to_buy < min_lot:
            self.logger.info(f"[buy_portion] leftover {leftover_after_buy} < minLot={min_lot}, "
                             f"force entire close => portion=1.0")
            amt_to_buy = total_amt
            portion = Decimal("1.0")

        # 2) Pak het pos-object
        pos = self.open_positions[symbol]
        entry_price = pos["entry_price"]
        position_id = pos["position_id"]
        position_type = pos["position_type"]
        master_id = pos.get("master_id", None)

        # 3) Current price
        if exec_price is not None:
            current_price = exec_price
        else:
            current_price = self._get_ws_price(symbol)
        if current_price <= 0:
            self.logger.warning(f"[PullbackStrategy] _buy_portion => price=0 => skip BUY {symbol}")
            return

        # 4) Bereken fees / realized pnl
        raw_pnl = (entry_price - current_price) * amt_to_buy
        trade_cost = current_price * amt_to_buy
        fees = float(trade_cost * Decimal("0.0035"))
        realized_pnl = float(raw_pnl) - fees

        # 5) Child-status
        if portion < 1:
            child_status = "partial"
        else:
            child_status = "closed"

        self.logger.info(
            f"[INFO {reason}] {symbol}: portion={portion}, amt_to_buy={amt_to_buy:.4f}, "
            f"entry={entry_price}, current_price={current_price}, trade_cost={trade_cost}, "
            f"fees={fees:.2f}, child_status={child_status}"
        )

        # 6) Plaats live/paper order
        if self.order_client:
            self.order_client.place_order("buy", symbol, float(amt_to_buy), ordertype="market")
            self.logger.info(
                f"[LIVE/PAPER] BUY {symbol} => portion={portion}, amt={amt_to_buy:.4f}, reason={reason}, "
                f"fees={fees:.2f}, pnl={realized_pnl:.2f}"
            )

        # 7) Schrijf child-trade in DB
        child_data = {
            "symbol": symbol,
            "side": "buy",
            "amount": float(amt_to_buy),
            "price": float(current_price),
            "timestamp": int(time.time() * 1000),
            "position_id": position_id,
            "position_type": position_type,
            "status": child_status,
            "pnl_eur": realized_pnl,
            "fees": fees,
            "trade_cost": float(trade_cost),
            "strategy_name": "pullback",
            "is_master": 0
        }
        self.db_manager.save_trade(child_data)

        # 8) Update leftover
        pos["amount"] -= amt_to_buy
        leftover_amt = pos["amount"]

        self.logger.debug(f"[BUY leftover-check] leftover_amt before epsilon => {leftover_amt}, min_lot={min_lot}")

        # Epsilon-check
        if leftover_amt < (min_lot * Decimal("1.00001")):
            leftover_amt = Decimal("0")
            pos["amount"] = Decimal("0")
            self.logger.debug("[PortionCheck] leftover_amt is superklein => set to 0")

        # >>> hier extra debug <<<
        self.logger.debug(
            f"[_buy_portion] leftover_amt={pos['amount']}, min_lot={min_lot} (before close-check)."
        )

        # 9) Sluit master of partial-update master
        self.logger.debug(f"[PortionCheck] leftover={leftover_amt}, master_id={master_id}")
        if leftover_amt <= Decimal("0") or leftover_amt < min_lot:
            self.logger.info(f"[PullbackStrategy] Full short position closed => {symbol}")
            if master_id:
                old_row = self.db_manager.execute_query(
                    "SELECT fees, pnl_eur FROM trades WHERE id=? LIMIT 1",
                    (master_id,)
                )
                if old_row:
                    old_fees, old_pnl = old_row[0]
                    self.logger.debug(
                        f"[MASTER CLOSE DEBUG] leftover={leftover_amt}, master_id={master_id}, "
                        f"fees={fees}, realized_pnl={realized_pnl}"
                    )
                    new_fees = old_fees + fees
                    new_pnl = old_pnl + realized_pnl
                    self.db_manager.update_trade(master_id, {
                        "fees": new_fees,
                        "pnl_eur": new_pnl,
                        "amount": float(leftover_amt)
                    })

        else:
            if master_id:
                old_row = self.db_manager.execute_query(
                    "SELECT fees, pnl_eur FROM trades WHERE id=? LIMIT 1",
                    (master_id,)
                )
                if old_row:
                    old_fees, old_pnl = old_row[0]
                    new_fees = old_fees + fees
                    new_pnl = old_pnl + realized_pnl
                    self.db_manager.update_trade(master_id, {
                        "status": "partial",
                        "fees": new_fees,
                        "pnl_eur": new_pnl,
                        "amount": float(leftover_amt)
                    })
                    self.logger.info(
                        f"[PullbackStrategy] updated master trade {master_id} => partial fees={new_fees}, pnl={new_pnl}"
                    )
            self.__record_trade_signals(master_id, event_type="partial", symbol=symbol, atr_mult=self.pullback_atr_mult)

        # *** extra debug eind ***
        self.logger.debug(
            f"[_buy_portion] END => leftover_amt={pos['amount']}, open_positions keys={list(self.open_positions.keys())}"
        )

    def _sell_portion(self, symbol: str, total_amt: Decimal, portion: Decimal, reason: str, exec_price=None):
        self.logger.info(f"### SELL portion => reason={reason}, symbol={symbol}")
        self.logger.debug(
            f"[_sell_portion] START => total_amt={total_amt}, portion={portion}, "
            f"open_positions={list(self.open_positions.keys())}"
        )

        amt_to_sell = total_amt * portion
        leftover_after_sell = total_amt - amt_to_sell
        min_lot = self._get_min_lot(symbol)

        if portion < 1 and leftover_after_sell > 0 and leftover_after_sell < min_lot:
            self.logger.info(
                f"[sell_portion] leftover {leftover_after_sell} < minLot={min_lot}, force entire close => portion=1.0"
            )
            self.logger.debug(
                f"[DEBUG leftover] leftover_after_sell={leftover_after_sell}, min_lot={min_lot}"
            )
            amt_to_sell = total_amt
            portion = Decimal("1.0")

        if amt_to_sell < min_lot:
            self.logger.info(
                f"[sell_portion] leftover {leftover_after_sell} < minLot={min_lot}, force entire close => portion=1.0"
            )
            amt_to_sell = total_amt
            portion = Decimal("1.0")

        pos = self.open_positions[symbol]
        entry_price = pos["entry_price"]
        position_id = pos["position_id"]
        position_type = pos["position_type"]
        db_id = pos.get("db_id", None)

        # [MASTER_ID FIX]
        master_id = pos.get("master_id", None)

        if exec_price is not None:
            current_price = exec_price
        else:
            current_price = self._get_ws_price(symbol)
        if current_price <= 0:
            self.logger.warning(f"[PullbackStrategy] _sell_portion => price=0 => skip SELL {symbol}")
            return

        # realized PnL
        raw_pnl = (current_price - entry_price) * amt_to_sell
        trade_cost = current_price * amt_to_sell
        fees = float(trade_cost * Decimal("0.0035"))
        realized_pnl = float(raw_pnl) - fees

        if portion < 1:
            child_status = "partial"
        else:
            child_status = "closed"

        self.logger.info(
            f"[INFO {reason}] {symbol}: portion={portion}, amt_to_sell={amt_to_sell:.4f}, "
            f"entry={entry_price}, current_price={current_price}, trade_cost={trade_cost:.2f}, fees={fees:.2f}"
        )

        if self.order_client:
            self.order_client.place_order("sell", symbol, float(amt_to_sell), ordertype="market")
            self.logger.info(
                f"[LIVE/PAPER] SELL {symbol} => portion={portion * 100:.1f}%, amt={amt_to_sell:.4f}, "
                f"reason={reason}, fees={fees:.2f}, pnl={realized_pnl:.2f}"
            )

        # Child-trade#
        child_data = {
            "symbol": symbol,
            "side": "sell",
            "amount": float(amt_to_sell),
            "price": float(current_price),
            "timestamp": int(time.time() * 1000),
            "position_id": position_id,
            "position_type": position_type,
            "status": child_status,
            "pnl_eur": realized_pnl,
            "fees": fees,
            "trade_cost": float(trade_cost),
            "strategy_name": "pullback",
            "is_master": 0
        }
        self.db_manager.save_trade(child_data)

        pos["amount"] -= amt_to_sell
        leftover_amt = pos["amount"]
        self.logger.debug(f"[PortionCheck] leftover={leftover_amt}, master_id={master_id}")

        self.logger.debug(
            f"[SELL leftover-check] leftover_amt before epsilon => {leftover_amt}, min_lot={min_lot}"
        )
        self.logger.debug(f"[PortionCheck] leftover_amt before epsilon-check => {leftover_amt}")

        if leftover_amt < (min_lot * Decimal("1.00001")):
            leftover_amt = Decimal("0")
            pos["amount"] = Decimal("0")
            self.logger.debug("[PortionCheck] leftover_amt is superklein => set to 0")

        # << Extra debug vlak na leftover >>
        self.logger.debug(
            f"[_sell_portion] leftover_amt={pos['amount']}, min_lot={min_lot} (before close-check)."
        )

        if leftover_amt <= Decimal("0") or leftover_amt < min_lot:
            self.logger.info(f"[PullbackStrategy] Full position closed => {symbol}")
            if master_id:
                old_row = self.db_manager.execute_query("SELECT fees, pnl_eur FROM trades WHERE id=? LIMIT 1",
                                                        (master_id,))
                if old_row:
                    old_fees, old_pnl = old_row[0]
                    self.logger.debug(
                        f"[MASTER CLOSE DEBUG] leftover={leftover_amt}, master_id={master_id}, "
                        f"fees={fees}, realized_pnl={realized_pnl}"
                    )
                    new_fees = old_fees + fees
                    new_pnl = old_pnl + realized_pnl
                    self.logger.debug(f"[MASTER CLOSE DEBUG] master_id={master_id}, leftover={leftover_amt}")
                    self.db_manager.update_trade(master_id, {
                        "fees": new_fees,
                        "pnl_eur": new_pnl
                    })

        else:
            if master_id:
                old_row = self.db_manager.execute_query("SELECT fees, pnl_eur FROM trades WHERE id=? LIMIT 1",
                                                        (master_id,))
                if old_row:
                    old_fees, old_pnl = old_row[0]
                    new_fees = old_fees + fees
                    new_pnl = old_pnl + realized_pnl
                    self.db_manager.update_trade(master_id, {
                        "status": "partial",
                        "fees": new_fees,
                        "pnl_eur": new_pnl,
                        "amount": float(leftover_amt)
                    })
                    self.logger.info(
                        f"[PullbackStrategy] updated master trade {master_id} => partial => fees={new_fees}, pnl={new_pnl}"
                    )
            self.__record_trade_signals(master_id, event_type="partial", symbol=symbol, atr_mult=self.pullback_atr_mult)

        # onderaan:
        self.logger.debug(
            f"[_sell_portion] END => leftover_amt={pos['amount']}, open_positions keys={list(self.open_positions.keys())}"
        )

    # [Hulp-code]
    @staticmethod
    def _calculate_fees_and_pnl(side: str, amount: float, price: float, reason: str) -> (float, float):
        trade_cost = amount * price
        fees = 0.0035 * trade_cost
        if reason.startswith("TP") or reason == "TrailingStop":
            realized_pnl = trade_cost - fees
        elif side.lower() == "sell":
            realized_pnl = trade_cost - fees
        elif side.lower() == "buy" and reason in ("TP1", "TP2", "TrailingStop"):
            realized_pnl = trade_cost - fees
        else:
            realized_pnl = 0.0
        return fees, realized_pnl

    def update_position_with_fill(self, fill_data: dict):
        """
        Ongewijzigd
        """
        symbol = fill_data.get("market")
        fill_side = fill_data.get("side", "").lower()
        fill_amt = Decimal(str(fill_data.get("amount", "0")))
        fill_price = Decimal(str(fill_data.get("price", "0")))

        if symbol not in self.open_positions:
            self.logger.info(f"[update_position_with_fill] Geen open positie voor {symbol}, skip fill.")
            return

        pos = self.open_positions[symbol]
        old_filled = pos["filled_amount"]
        new_filled = old_filled + fill_amt

        if new_filled > Decimal("0"):
            old_price = pos["entry_price"]
            if old_filled == 0:
                pos["entry_price"] = fill_price
            else:
                pos["entry_price"] = ((old_price * old_filled) + (fill_price * fill_amt)) / new_filled

        pos["filled_amount"] = new_filled
        pos["amount"] = new_filled

        desired = pos["desired_amount"]
        if pos["filled_amount"] >= desired:
            pos["amount"] = desired
            pos["filled_amount"] = desired
            self.logger.info(f"[update_position_with_fill] {symbol}: order fully filled => {desired} / {desired}")
        else:
            self.logger.info(f"[update_position_with_fill] {symbol}: partial fill => {pos['filled_amount']}/{desired} @ {fill_price}")

    def _load_open_positions_from_db(self):
        """
        Laadt uitsluitend master-trades (is_master=1) met status 'open' of 'partial'
        uit de DB en zet ze in self.open_positions.
        Child-trades (is_master=0) negeren we hier.
        """
        query = """
            SELECT
                id,
                symbol,
                side,
                amount,
                price,
                position_id,
                position_type,
                is_master
            FROM trades
            WHERE is_master=1
              AND status IN ('open','partial')
              AND strategy_name='pullback'              
        """
        rows = self.db_manager.execute_query(query)
        if not rows:
            self.logger.info("[PullbackStrategy] Geen open/partial MASTER-trades in DB.")
            return

        for row in rows:
            db_id = row[0]
            symbol = row[1]
            side = row[2]
            amount = Decimal(str(row[3]))
            entry_price = Decimal(str(row[4]))
            position_id = row[5]
            position_type = row[6]
            # is_master = row[7]  # niet nodig, isMaster=1

            # Bepaal leftover = master - som(children) — GEEN DB-updates hier!
            sum_rows = self.db_manager.execute_query(
                """
                SELECT COALESCE(SUM(amount), 0)
                FROM trades
                WHERE position_id=? AND is_master=0 AND strategy_name='pullback'
                """,
                (position_id,)
            )
            child_sum = Decimal(str(sum_rows[0][0])) if sum_rows and sum_rows[0][0] is not None else Decimal("0")

            # amount = master-amount uit de row
            leftover = amount - child_sum
            if leftover <= 0:
                self.logger.info(
                    f"[_load_open_positions_from_db] {symbol} leftover=0 => NIET herstellen (en GEEN DB-wijziging).")
                continue

            # Anders: pos is nog echt open/partial
            pos_data = {
                "side": side,
                "entry_price": entry_price,
                "amount": leftover,
                "atr": Decimal("0.0"),
                "tp1_done": False,
                "tp2_done": False,
                "trail_active": False,
                "trail_high": entry_price,
                "position_id": position_id,
                "position_type": position_type,
                "db_id": db_id,
                "master_id": db_id  # Master = db_id
            }

            # ATR opnieuw berekenen
            df_main = self._fetch_and_indicator(symbol, self.main_timeframe, limit=200)
            atr_value = self._calculate_atr(df_main, self.atr_window)
            if atr_value:
                pos_data["atr"] = Decimal(str(atr_value))

            self.open_positions[symbol] = pos_data
            self.logger.info(
                f"[PullbackStrategy] Hersteld MASTER pos => symbol={symbol}, side={side}, "
                f"amt={amount}, entry={entry_price}, db_id={db_id}"
            )

    def manage_intra_candle_exits(self):
        """
        Check SL/TP/etc in real-time (intra-candle).
        """
        self.logger.debug("[PullbackStrategy] manage_intra_candle_exits => start SL/TP checks.")
        self.logger.debug(f"[manage_intra_candle_exits] open_positions keys => {list(self.open_positions.keys())}")

        for sym in list(self.open_positions.keys()):
            pos = self.open_positions[sym]
            current_price = self._get_ws_price(sym)
            self.logger.info(
                f"[manage_intra_candle_exits] symbol={sym}, side={pos['side']}, "
                f"entry={pos['entry_price']}, current={current_price}"
            )

            if current_price > 0:
                self._manage_open_position(sym, current_price, pos["atr"])

    def _get_min_lot(self, symbol: str) -> Decimal:
        if not self.data_client:
            return Decimal("1.0")
        return self.data_client.get_min_lot(symbol)

    def _get_latest_price(self, symbol: str) -> Decimal:
        df_ticker = self.db_manager.fetch_data(
            table_name="ticker_kraken",
            limit=1,
            market=symbol
        )
        if not df_ticker.empty:
            best_bid = df_ticker["best_bid"].iloc[0] if "best_bid" in df_ticker.columns else 0
            best_ask = df_ticker["best_ask"].iloc[0] if "best_ask" in df_ticker.columns else 0
            if best_bid > 0 and best_ask > 0:
                return (Decimal(str(best_bid)) + Decimal(str(best_ask))) / Decimal("2")

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

    def _get_ws_price(self, symbol: str) -> Decimal:
        if not self.data_client:
            self.logger.warning("[Pullback] data_client=None => return 0")
            return Decimal("0")
        px_float = self.data_client.get_latest_ws_price(symbol)
        if px_float > 0.0:
            return Decimal(str(px_float))

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

    def _get_equity_estimate(self) -> Decimal:
        """
        Tel de volledige walletwaarde in EUR, plus eventueel posities
        als je dat wilt. (Let op double counting!)
        """
        if not self.order_client:
            return self.initial_capital

        bal = self.order_client.get_balance()

        total_wallet_eur = Decimal("0")
        for asset, amount_str in bal.items():
            amt = Decimal(str(amount_str))
            if asset.upper() == "EUR":
                # Gewoon EUR-saldo
                total_wallet_eur += amt
            else:
                # Converteer asset -> EUR
                # Voor "XBT" => symbol="XBT-EUR", "ETH" => "ETH-EUR"
                symbol = f"{asset.upper()}-EUR"
                price = self._get_latest_price(symbol)
                if price > 0:
                    total_wallet_eur += (amt * price)

        # Als je open_positions gebruikt voor trades,
        # en die assets zitten *ook* fysiek in de wallet,
        # dan is de waarde al meegerekend.
        # Dus om double counting te voorkomen, laten we 'm hier weg.
        #
        # Wil je posities *wel* apart meetellen (bijvoorbeeld als je
        # open_positions net andere derivaten zijn), dan kun je:
        #
        # for sym, pos_info in self.open_positions.items():
        #    amt = pos_info["amount"]
        #    px = self._get_latest_price(sym)
        #    total_wallet_eur += (amt * px)

        return total_wallet_eur

    def _fetch_and_indicator(self, symbol: str, interval: str, limit=200) -> pd.DataFrame:
        try:
            df = self.db_manager.fetch_data(
                table_name="candles_kraken",
                limit=limit,
                market=symbol,
                interval=interval
            )

            # ===============================
            # [FIX] Zorg dat we geen int of None krijgen
            # ===============================
            if not isinstance(df, pd.DataFrame):
                self.logger.warning(f"[_fetch_and_indicator] fetch_data gaf {type(df).__name__}, geen DataFrame => return lege DF.")
                return pd.DataFrame()
            # ===============================

            if df.empty:
                # (3) Debug-melding "Geen candles..." uitgecommentarieerd:
                # self.logger.debug(f"[DEBUG] Geen candles uit 'candles_kraken' voor {symbol} ({interval}).")
                return pd.DataFrame()

            for col in ['datetime_utc', 'exchange']:
                if col in df.columns:
                    df.drop(columns=col, inplace=True, errors='ignore')

            # 1) Hernoem kolommen (nu gebruiken we 'timestamp_ms' voor de oorspronkelijke ms-tijd)
            df.columns = ["timestamp_ms", "market", "interval", "open", "high", "low", "close", "volume"]

            # 2) Eventueel onnodige kolommen droppen (optioneel). Voorbeeld:
            # for col in ["datetime_utc", "exchange"]:
            #     if col in df.columns:
            #         df.drop(columns=col, inplace=True, errors="ignore")

            # 3) Zorg dat open, high, low, close, volume numeriek zijn
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float, errors="raise")

            # 4) Maak een datetime‐kolom die je index wordt, maar behoud de kolom
            df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

            # 5) Zet 'datetime_utc' als index, maar behoud die kolom (drop=False)
            df.set_index("datetime_utc", inplace=True, drop=False)

            # 6) Sorteer op de index (nu is 'datetime_utc' de index)
            df.sort_index(inplace=True)
            self.logger.debug(f"[{interval} df] tail:\n{df.tail(3)}")

            # Je DataFrame heeft nu:
            # - Een index = datetime_utc
            # - Kolommen: ["timestamp_ms", "market", "interval", "open", "high", "low", "close", "volume", "datetime_utc"]
            #   waar "timestamp_ms" numeriek is, en "datetime_utc" als kolom + index

            df = IndicatorAnalysis.calculate_indicators(df, rsi_window=self.rsi_window)
            macd_ind = MACD(close=df['close'], window_slow=self.macd_slow,
                            window_fast=self.macd_fast, window_sign=self.macd_signal)
            df['macd'] = macd_ind.macd()
            df['macd_signal'] = macd_ind.macd_signal()

            bb = IndicatorAnalysis.calculate_bollinger_bands(df, window=20, num_std_dev=2)
            df["bb_upper"] = bb["bb_upper"]
            df["bb_lower"] = bb["bb_lower"]

            # === ADX + DI's (trendsterkte) ===
            try:
                # Veiligheidscheck: alleen berekenen bij voldoende candles
                # (kleine marge boven window om NaN/edge-cases te vermijden)
                needed = self.adx_window + 5
                if len(df) >= needed:
                    adx_obj = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=self.adx_window)
                    df["adx"] = adx_obj.adx()
                    df["di_pos"] = adx_obj.adx_pos()  # +DI
                    df["di_neg"] = adx_obj.adx_neg()  # -DI
                else:
                    # Kolommen niet aanmaken (scheelt downstream checks); gewoon overslaan
                    self.logger.debug(f"[ADX] Skip on interval={interval}: len(df)={len(df)} < needed={needed}")
            except Exception as e:
                self.logger.warning(f"[ADX] Kon ADX niet berekenen ({e}) voor interval={interval}.")

            if interval == "4h" and len(df) >= 2:
                df = self._ensure_closed_4h_candle(df)

            return df
        except Exception as e:
            self.logger.error(f"[ERROR] _fetch_and_indicator faalde: {e}")
            return pd.DataFrame()

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

    def set_ml_engine(self, ml_engine):
        self.ml_engine = ml_engine
        self.logger.info("[PullbackAccumulateStrategy] ML-engine is succesvol gezet.")

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

    def __record_trade_signals(self, trade_id: Optional[int], event_type: str, symbol: str, atr_mult: Decimal,
                               macd_15m=None):
        if not trade_id:
            return
        try:
            df_15m = self._fetch_and_indicator(symbol, "15m", limit=30)
            vol_15m = float(df_15m["volume"].iloc[-1]) if (not df_15m.empty) else 0.0
            macd_signal_15m = float(df_15m["macd_signal"].iloc[-1]) if "macd_signal" in df_15m.columns else 0.0

            # 4h
            df_4h = self._fetch_and_indicator(symbol, "4h", limit=30)
            rsi_4h = float(df_4h["rsi"].iloc[-1]) if (not df_4h.empty) else 0.0

            depth_instant = self._analyze_depth_trend_instant(symbol)

            ml_val = 0.0
            if self.ml_model_enabled and (self.ml_engine is not None) and not df_15m.empty:
                ml_val = float(self._ml_predict_signal(df_15m))

            signals_data = {
                "trade_id": trade_id,
                "event_type": event_type,
                "symbol": symbol,
                "strategy_name": "pullback",
                "rsi_15m": float(df_15m["rsi"].iloc[-1]) if (not df_15m.empty) else 0.0,
                "macd_val": macd_15m,
                "macd_signal": macd_signal_15m,  # gebruik de gemeten value
                "atr_value": float(atr_mult),
                "depth_score": depth_instant,
                "ml_signal": ml_val,
                "rsi_h4": rsi_4h,
                "timestamp": int(time.time() * 1000)
            }

            # === LOG-ONLY: ADX/DI (geen DB-schema wijziging nodig) ===
            try:
                adx_15m = float(df_15m["adx"].iloc[-1]) if ("adx" in df_15m.columns and not df_15m.empty) else None
                di_pos_15m = float(df_15m["di_pos"].iloc[-1]) if "di_pos" in df_15m.columns else None
                di_neg_15m = float(df_15m["di_neg"].iloc[-1]) if "di_neg" in df_15m.columns else None
            except Exception:
                adx_15m, di_pos_15m, di_neg_15m = None, None, None

            try:
                adx_4h = float(df_4h["adx"].iloc[-1]) if ("adx" in df_4h.columns and not df_4h.empty) else None
            except Exception:
                adx_4h = None

            self.logger.info(
                f"[Signals][{symbol}] ADX_15m={adx_15m} | +DI_15m={di_pos_15m} | -DI_15m={di_neg_15m} | ADX_4h={adx_4h}"
            )
            # === einde LOG-ONLY ===

            self.db_manager.save_trade_signals(signals_data)
            self.logger.debug(f"[__record_trade_signals] trade_id={trade_id}, event={event_type}, symbol={symbol}")
        except Exception as e:
            self.logger.error(f"[__record_trade_signals] Fout: {e}")

    def _ensure_closed_4h_candle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Controleer of de laatste row in df nog half-open is.
        Als 'timestamp_ms' van de laatste row niet afgesloten is
        (nu < endtijd 4h), dan droppen we die row.
        """

        # Haal de timestamp_ms van de allerlaatste row
        last_ts = df["timestamp_ms"].iloc[-1]

        # Roep jouw bestaande 'is_candle_closed' aan op basis van '4h'
        if not is_candle_closed(int(last_ts), "4h"):
            # Candle is niet gesloten => gooi deze row weg
            # (alleen als er minstens 2 rows zijn, anders hou je niets over)
            self.logger.debug("[_ensure_closed_4h_candle] drop half-open row => using second last row for 4h.")
            return df.iloc[:-1]
        else:
            # Candle is wél dicht => geen wijziging
            return df

