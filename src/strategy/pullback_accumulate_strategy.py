import logging
import pandas as pd
import yaml
import time
from typing import Optional
from decimal import Decimal
from collections import deque

# TA-bibliotheken
from ta.volatility import AverageTrueRange
from ta.trend import MACD

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


# NIEUW: Helper-functie om te controleren of een candle afgesloten is.
def is_candle_closed(last_candle_ms: int, timeframe: str) -> bool:
    """
    Berekent of de candle die eindigt op last_candle_ms al volledig afgesloten is.
    Hierbij gaan we ervan uit dat de timeframe als volgt kan worden opgegeven:
      - '5m'   : 5 minuten
      - '15m'  : 15 minuten
      - '1h'   : 60 minuten
      - '4h'   : 4 uur
      - '1d'   : 1 dag
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
     - self.data_client => om live data te zien
     - self.order_client => om orders te plaatsen / get_balance()
     - self.db_manager => database afhandeling

    [AANPASSING] Code is opgeschoond om enkel 4h (trend) + 15m (entry) te gebruiken,
                  en de 'daily' is uitgecommentarieerd (zie # [UITGECOMMENTARIEERD]).
                  Ook is 'R' geïntroduceerd als risk-eenheid = (1 × ATR).
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
        self.logger = setup_logger("pullback_strategy", PULLBACK_STRATEGY_LOG_FILE,
                                   logging.DEBUG)  # kan weer naar INFO indien nodig
        if config_path:
            self.logger.info("[PullbackAccumulateStrategy] init with config_path=%s", config_path)
        else:
            self.logger.info("[PullbackAccumulateStrategy] init (no config_path)")

        # Laden config, bv. meltdown_cfg = full_config.get("meltdown_manager", {})
        meltdown_cfg = self.strategy_config.get("meltdown_manager", {})
        self.meltdown_manager = MeltdownManager(meltdown_cfg, db_manager=db_manager, logger=self.logger)
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

        # R-concept: we hanteren R = 1 × ATR als basis (zie code in manage_open_position).
        # Let op: Je kunt R = x × ATR doen als je wilt, bv. R = 1.2 × ATR, maar nu laten we 'm op 1.0 × ATR.

        self.log_file = self.strategy_config.get("log_file", PULLBACK_STRATEGY_LOG_FILE)

        # partial TP
        self.tp1_portion_pct = Decimal(str(self.strategy_config.get("tp1_portion_pct", "0.50")))

        # Overige filters / drempels
        # Voor RSI-check op 4h
        self.h4_bull_rsi = float(self.strategy_config.get("h4_bull_rsi", 50))   # bovengrens bull
        self.h4_bear_rsi = float(self.strategy_config.get("h4_bear_rsi", 50))   # ondergrens bear

        self.use_depth_trend = bool(self.strategy_config.get("use_depth_trend", True))
        self.use_ema_pullback_check = bool(self.strategy_config.get("use_ema_pullback_check", False))  # [NIEUW]

        # RSI/MACD config
        self.rsi_window = int(self.strategy_config.get("rsi_window", 14))
        self.macd_fast = int(self.strategy_config.get("macd_fast", 12))
        self.macd_slow = int(self.strategy_config.get("macd_slow", 26))
        self.macd_signal = int(self.strategy_config.get("macd_signal", 9))

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

        # Laad event. open pos. uit DB
        self._load_open_positions_from_db()

        # Nieuw, om na ene nieuwe candle maar 1x de strategie uit te voeren
        self.last_processed_candle_ts = {}  # [ADDED] dict: {symbol: last_candle_ms we used}


    def execute_strategy(self, symbol: str):
        """
        Eenvoudige flow:
         1) meltdown-check
         2) bepaal trend op 4h
         3) ATR op 1h
         4) check pullback op 15m
         5) manage pos of open pos
        """
        meltdown_active = self.meltdown_manager.update_meltdown_state(strategy=self, symbol=symbol)

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
        self.logger.info(f"[PullbackStrategy] Start for {symbol}")

        # Concurrency-check / check of we al open trades hebben in DB
        existing_db_trades = self.db_manager.execute_query(
            """
            SELECT id 
              FROM trades
             WHERE symbol=? 
               AND is_master=1
               AND status IN ('open','partial')
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

        # --- (2) Trend => 4h RSI als simplistische check
        df_h4 = self._fetch_and_indicator(symbol, self.trend_timeframe, limit=200)
        if df_h4.empty:
            self.logger.warning(f"[PullbackStrategy] No 4h data => skip {symbol}")
            return

        rsi_h4 = df_h4["rsi"].iloc[-1]
        direction = self._check_trend_direction_4h(rsi_h4)
        self.logger.info(f"[PullbackStrategy] direction={direction}, rsi_h4={rsi_h4:.2f} for {symbol}")

        # [UITGECOMMENTARIEERD] daily-check
        # df_daily = self._fetch_and_indicator(symbol, self.daily_timeframe, limit=60)
        # if df_daily.empty:
        #     self.logger.warning(f"[PullbackStrategy] No daily data => skip {symbol}")
        #     return
        # rsi_daily = df_daily["rsi"].iloc[-1]

        # (3) ATR => 1h
        df_main = self._fetch_and_indicator(symbol, self.main_timeframe, limit=200)
        if df_main.empty:
            self.logger.warning(f"[PullbackStrategy] No {self.main_timeframe} data => skip {symbol}")
            return
        atr_value = self._calculate_atr(df_main, self.atr_window)
        if atr_value is None:
            self.logger.warning(f"[PullbackStrategy] Not enough data => skip {symbol}")
            return
        self.logger.info(f"[ATR-info] {symbol} => ATR({self.atr_window})={atr_value}")

        # (4) Pullback => 15m
        df_entry = self._fetch_and_indicator(symbol, self.entry_timeframe, limit=100)
        if df_entry.empty:
            self.logger.warning(f"[PullbackStrategy] No {self.entry_timeframe} data => skip {symbol}")
            return

        df_entry.sort_index(inplace=True)
        last_timestamp = df_entry.index[-1]
        if isinstance(last_timestamp, pd.Timestamp):
            last_candle_ms = int(last_timestamp.timestamp() * 1000)
        else:
            last_candle_ms = int(last_timestamp)

        if not is_candle_closed(last_candle_ms, self.entry_timeframe):
            self.logger.debug(f"[Pullback] {symbol}: Laatste candle nog niet afgesloten => return.")
            return

        prev_candle_ts = self.last_processed_candle_ts.get(symbol, None)
        if prev_candle_ts == last_candle_ms:
            self.logger.debug(f"[Pullback] {symbol}: Candle {last_candle_ms} al verwerkt => return.")
            return

        self.last_processed_candle_ts[symbol] = last_candle_ms

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
            # [AANPASSING] Extra check: 9EMA + 20EMA
            if not self._check_ema_pullback_15m(df_entry, direction):
                pullback_detected = False
                self.logger.info(f"[PullbackStrategy] 9/20EMA-check => geen valide pullback => skip {symbol}")

        # Depth + ML
        ml_signal = self._ml_predict_signal(df_entry)  # of df_h4, wat je wilt
        depth_score = 0.0
        if self.use_depth_trend:
            depth_score_instant = self._analyze_depth_trend_instant(symbol)
            # Rolling average
            self.depth_trend_history.append(depth_score_instant)
            depth_score = sum(self.depth_trend_history) / len(self.depth_trend_history)
            self.logger.info(f"[DepthTrend] instant={depth_score_instant:.2f}, rolling_avg={depth_score:.2f}")

        # Equity check
        total_equity = self._get_equity_estimate()
        invest_extra_flag = False
        if (total_equity >= self.initial_capital * self.accumulate_threshold) and not self.invested_extra:
            invest_extra_flag = True
            self.logger.info("[PullbackStrategy] +25%% => next pullback => invest extra in %s", symbol)

        has_position = (symbol in self.open_positions)
        self.logger.info(f"[Decision Info] symbol={symbol}, direction={direction}, pullback={pullback_detected},"
                         f" ml_signal={ml_signal}, depth_score={depth_score:.2f}, current_price={current_price}")

        # Als we geen positie hebben, maar wel pullback + bull => open long
        # [AANPASSING: no daily rsi check anymore, just h4-based direction + pullback]
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

    # [AANPASSING] Simpele check: rsi_h4 > self.h4_bull_rsi => bull, < self.h4_bear_rsi => bear, anders range
    def _check_trend_direction_4h(self, rsi_h4: float) -> str:
        if rsi_h4 > self.h4_bull_rsi:
            return "bull"
        elif rsi_h4 < self.h4_bear_rsi:
            return "bear"
        else:
            return "range"

    def _detect_pullback(self,
                         df: pd.DataFrame,
                         current_price: Decimal,
                         direction: str,
                         atr_value: Decimal) -> bool:
        """
        Simpele pullback-check op basis van ATR.
        Zonder databasecalls, maar met logging van de afstand / drempel,
        zodat er geen 'unused variable' waarschuwingen zijn.
        """

        # 1) Check genoeg candles
        if len(df) < self.pullback_rolling_window:
            self.logger.info(f"[Pullback] <{self.pullback_rolling_window} candles => skip.")
            return False

        # 2) Alleen relevant in bull/bear; bij 'range' => return False
        if direction not in ("bull", "bear"):
            return False

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

            if atr_threshold != 0:
                ratio = pullback_distance / atr_threshold

            # Log de waarden om 'unused variable' te vermijden
            self.logger.info(
                f"[Pullback-bull] distance={pullback_distance:.4f}, "
                f"threshold={atr_threshold:.4f}, ratio={ratio:.2f}"
            )

            # Echte check: >= 1 => “pullback genoeg”
            if ratio >= 1:
                self.logger.info("[Pullback-bull] => DETECTED!")
                return True
            else:
                return False

        # 4) Bear-richting
        else:  # direction == "bear"
            recent_low = df["low"].rolling(self.pullback_rolling_window).min().iloc[-1]
            if recent_low <= 0:
                return False

            rally_distance = current_price - Decimal(str(recent_low))
            atr_threshold = atr_value * self.pullback_atr_mult

            if atr_threshold != 0:
                ratio = rally_distance / atr_threshold

            self.logger.info(
                f"[Pullback-bear] distance={rally_distance:.4f}, "
                f"threshold={atr_threshold:.4f}, ratio={ratio:.2f}"
            )

            if ratio >= 1:
                self.logger.info("[Pullback-bear] => DETECTED!")
                return True
            else:
                return False

    # [NIEUW] Eenvoudige check: 9EMA, 20EMA, pullback in uptrend => price zakt door 9ema maar boven 20ema
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
            if (last_close > last_ema20) and (last_close <= last_ema9 * Decimal("1.01")):
                self.logger.info(f"[EMA-check-bull] close={last_close}, ema9={last_ema9}, ema20={last_ema20} => True")
                return True
            else:
                return False

        elif direction == "bear":
            # bear: we willen dat de prijs onder 20ema zit, en net even boven 9ema => pullback
            if (last_close < last_ema20) and (last_close >= last_ema9 * Decimal("0.99")):
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
        We loggen nu wat '1R' is (hier = atr_value), en laten de code verder intact.
        """
        if current_price is None or current_price <= 0:
            self.logger.warning(f"[PullbackStrategy] current_price={current_price} => skip manage pos for {symbol}")
            return

        pos = self.open_positions[symbol]
        side = pos["side"]
        entry = pos["entry_price"]
        amount = pos["amount"]

        # [NIEUW] R = atr_value (1×ATR)
        one_r = atr_value  # je kunt dit factor > 1 doen als je wilt

        # Vaste (percentage-based) stoploss
        if side == "buy":
            stop_loss_price = entry - (atr_value * self.sl_atr_mult)
            if current_price <= stop_loss_price:
                self.logger.info(f"[ManagePos] LONG STOPLOSS => close entire {symbol}")
                self._sell_portion(symbol, amount, portion=Decimal("1.0"), reason="StopLoss", exec_price=current_price)
                if symbol in self.open_positions:
                    del self.open_positions[symbol]
                return

            # TP1 op 1R ( = entry + 1×ATR ) * self.tp1_atr_mult => bv. 1.0
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
                self.logger.info(f"[Trailing-LONG] {symbol}, trail_high={pos['trail_high']}, trailing_stop={trailing_stop_price}")
                if current_price <= trailing_stop_price:
                    self.logger.info(f"[TrailingStop] => close entire leftover => {symbol}")
                    self._sell_portion(symbol, amount, portion=Decimal("1.0"), reason="TrailingStop", exec_price=current_price)
                    if symbol in self.open_positions:
                        del self.open_positions[symbol]

        else:  # SHORT
            stop_loss_price = entry + (one_r * self.sl_atr_mult)
            if current_price >= stop_loss_price:
                self.logger.info(f"[ManagePos] SHORT STOPLOSS => close entire {symbol}")
                self._buy_portion(symbol, amount, portion=Decimal("1.0"), reason="StopLoss", exec_price=current_price)
                if symbol in self.open_positions:
                    del self.open_positions[symbol]
                return

            # TP1 op entry - 1R => (entry - (one_r * self.tp1_atr_mult))
            tp1_price = entry - (one_r * self.tp1_atr_mult)

            self.logger.info(f"[ManagePos-SHORT] {symbol} => 1R={one_r}, tp1_price={tp1_price}, current={current_price}")
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
                self.logger.info(f"[Trailing-SHORT] {symbol}, trail_high={pos['trail_high']}, trailing_stop={trailing_stop_price}")
                if current_price >= trailing_stop_price:
                    self.logger.info(f"[TrailingStop] => close entire leftover => {symbol}")
                    self._buy_portion(symbol, amount, portion=Decimal("1.0"), reason="TrailingStop", exec_price=current_price)
                    if symbol in self.open_positions:
                        del self.open_positions[symbol]

    # [AANPASSING] Kleine helper-functie om ATR te berekenen
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
    # Voorbeeld: _open_position
    # ----------------------------------------------------------------
    def _open_position(self, symbol: str, side: str, current_price: Decimal,
                       atr_value: Decimal, extra_invest=False):
        """
        [ongewijzigd t.o.v. jouw code, we laten het staan]
        """
        if symbol in self.open_positions:
            self.logger.warning(f"[PullbackStrategy] Already have an open position for {symbol}, skip opening.")
            return

        self.logger.info(f"[PullbackStrategy] OPEN => side={side}, {symbol}@{current_price}, extra_invest={extra_invest}")

        if side == "sell":
            self.logger.info(f"### EXTRA LOG ### [OPEN SHORT] {symbol} at {current_price}")
        else:
            self.logger.info(f"### EXTRA LOG ### [OPEN LONG] {symbol} at {current_price}")

        eur_balance = Decimal("100")
        if self.order_client:
            bal = self.order_client.get_balance()
            eur_balance = Decimal(str(bal.get("EUR", "100")))

        if current_price is None or current_price <= 0:
            self.logger.warning(f"[PullbackStrategy] current_price={current_price} => skip open pos for {symbol}")
            return

        needed_coins = self._get_min_lot(symbol) * self.min_lot_multiplier
        needed_eur_for_min = needed_coins * current_price
        allowed_eur_pct = eur_balance * self.max_position_pct
        allowed_eur = min(allowed_eur_pct, self.max_position_eur)

        if needed_eur_for_min > allowed_eur:
            self.logger.warning(
                f"[PullbackStrategy] needed={needed_eur_for_min:.2f} EUR > allowed={allowed_eur:.2f} => skip {symbol}.")
            return

        buy_eur = needed_eur_for_min
        if buy_eur > eur_balance:
            self.logger.warning(
                f"[PullbackStrategy] Not enough EUR => need {buy_eur:.2f}, have {eur_balance:.2f}. skip.")
            return

        amount = buy_eur / current_price

        position_id = f"{symbol}-{int(time.time())}"
        position_type = "long" if side == "buy" else "short"

        if self.order_client:
            self.order_client.place_order(side, symbol, float(amount), order_type="market")
            self.logger.info(f"[LIVE/PAPER] {side.upper()} {symbol} => amt={amount:.4f}, price={current_price}, cost={buy_eur}")

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
            "strategy_name": "pullback"
        }
        self.db_manager.save_trade(trade_data)

        new_trade_id = self.db_manager.cursor.lastrowid
        self.logger.info(f"[PullbackStrategy] new trade row => trade_id={new_trade_id}")

        # [NIEUW] => SIGNALS LOG
        self.__record_trade_signals(new_trade_id, event_type="open", symbol=symbol, atr_mult=self.pullback_atr_mult)

        desired_amount = amount
        self.open_positions[symbol] = {
            "side": side,
            "entry_price": current_price,
            "desired_amount": desired_amount,
            "filled_amount": Decimal("0.0"),
            "amount": Decimal("0.0"),
            "atr": atr_value,  # later in manage pos
            "tp1_done": False,
            "tp2_done": False,
            "trail_active": False,
            "trail_high": current_price,
            "position_id": position_id,
            "position_type": position_type,
            "db_id": new_trade_id
        }
        # [CONCRETE FIX 2-A] => Zet meteen de werkelijke hoeveelheid in 'amount' en 'filled_amount'
        self.open_positions[symbol]["amount"] = Decimal(str(amount))
        self.open_positions[symbol]["filled_amount"] = Decimal(str(amount))

    def _buy_portion(self, symbol: str, total_amt: Decimal, portion: Decimal, reason: str, exec_price=None):
        """
        [idem: ongewijzigd, behoud van functionaliteit; zie boven in je code]
        """
        self.logger.info(f"### BUY portion => closing SHORT? reason={reason} symbol={symbol}")
        amt_to_buy = total_amt * portion
        leftover_after_buy = total_amt - amt_to_buy
        min_lot = self._get_min_lot(symbol)
        if portion < 1 and leftover_after_buy > 0 and leftover_after_buy < min_lot:
            self.logger.info(f"[buy_portion] leftover {leftover_after_buy} < minLot={min_lot}, force entire close => portion=1.0")
            amt_to_buy = total_amt
            portion = Decimal("1.0")
        if amt_to_buy < min_lot:
            self.logger.info(f"[PullbackStrategy] skip partial BUY => amt_to_buy={amt_to_buy:.4f} < minLot={min_lot:.4f}")
            return
        pos = self.open_positions[symbol]
        entry_price = pos["entry_price"]
        position_id = pos["position_id"]
        position_type = pos["position_type"]

        if exec_price is not None:
            current_price = exec_price
        else:
            current_price = self._get_ws_price(symbol)

        if current_price <= 0:
            self.logger.warning(f"[PullbackStrategy] _buy_portion => price=0 => skip BUY {symbol}")
            return

        raw_pnl = (entry_price - current_price) * amt_to_buy
        trade_cost = current_price * amt_to_buy
        fees = float(trade_cost * Decimal("0.0025"))
        realized_pnl = float(raw_pnl) - fees

        self.logger.info(
            f"[INFO {reason}] {symbol}: portion={portion}, amt_to_buy={amt_to_buy:.6f}, entry={entry_price}, "
            f"current_price={current_price}, trade_cost={trade_cost}, fees={fees:.2f}"
        )
        if portion < 1:
            trade_status = "partial"
        else:
            trade_status = "closed"

        if self.order_client:
            self.order_client.place_order("buy", symbol, float(amt_to_buy), order_type="market")
            self.logger.info(
                f"[LIVE/PAPER] BUY {symbol} => {portion * 100:.1f}%, amt={amt_to_buy:.4f}, reason={reason}, fees={fees:.2f}, pnl={realized_pnl:.2f}"
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
                "trade_cost": float(trade_cost),
                "strategy_name": "pullback"
            }
            self.db_manager.save_trade(trade_data)
        else:
            self.logger.info(
                f"[Paper] BUY {symbol} => {portion * 100:.1f}%, amt={amt_to_buy:.4f}, reason={reason}, (fees={fees:.2f}, pnl={realized_pnl:.2f})"
            )

        self.open_positions[symbol]["amount"] -= amt_to_buy
        if self.open_positions[symbol]["amount"] <= Decimal("0"):
            self.logger.info(f"[PullbackStrategy] Full short position closed => {symbol}")
            db_id = pos.get("db_id", None)
            if db_id:
                self.db_manager.update_trade(db_id, {"status": "closed"})
                self.logger.info(f"[PullbackStrategy] Trade {db_id} => status=closed in DB")
            if db_id:
                self.__record_trade_signals(db_id, event_type="closed", symbol=symbol, atr_mult=self.pullback_atr_mult)
            del self.open_positions[symbol]
        else:
            db_id = pos.get("db_id", None)
            if db_id:
                old_row = self.db_manager.execute_query("SELECT fees, pnl_eur FROM trades WHERE id=? LIMIT 1", (db_id,))
                if old_row:
                    old_fees, old_pnl = old_row[0]
                    new_fees = old_fees + fees
                    new_pnl = old_pnl + realized_pnl
                    self.db_manager.update_trade(db_id, {"status": "partial", "fees": new_fees, "pnl_eur": new_pnl})
                    self.logger.info(f"[PullbackStrategy] updated open trade {db_id} => partial fees={new_fees}, pnl={new_pnl}")
            if db_id:
                self.__record_trade_signals(db_id, event_type="partial", symbol=symbol, atr_mult=self.pullback_atr_mult)

    def _sell_portion(self, symbol: str, total_amt: Decimal, portion: Decimal, reason: str, exec_price=None):
        """
        [idem: ongewijzigd]
        """
        amt_to_sell = total_amt * portion
        leftover_after_sell = total_amt - amt_to_sell
        ml = self._get_min_lot(symbol)

        if portion < 1 and leftover_after_sell > 0 and leftover_after_sell < ml:
            self.logger.info(f"[sell_portion] leftover {leftover_after_sell} < minLot={ml}, force entire close => portion=1.0")
            amt_to_sell = total_amt
            portion = Decimal("1.0")

        if amt_to_sell < ml:
            self.logger.info(f"[sell_portion] skip partial => amt_to_sell={amt_to_sell} < minLot={ml}")
            return

        pos = self.open_positions[symbol]
        entry_price = pos["entry_price"]
        position_id = pos["position_id"]
        position_type = pos["position_type"]

        if exec_price is not None:
            current_price = exec_price
        else:
            current_price = self._get_ws_price(symbol)

        if current_price <= 0:
            self.logger.warning(f"[PullbackStrategy] _sell_portion => price=0 => skip SELL {symbol}")
            return

        raw_pnl = (current_price - entry_price) * amt_to_sell
        trade_cost = current_price * amt_to_sell
        fees = float(trade_cost * Decimal("0.0025"))
        realized_pnl = float(raw_pnl) - fees

        self.logger.info(
            f"[INFO {reason}] {symbol}: portion={portion}, amt_to_sell={amt_to_sell:.6f}, "
            f"entry={entry_price}, current_price={current_price}, trade_cost={trade_cost}, fees={fees:.2f}"
        )
        if portion < 1:
            trade_status = "partial"
        else:
            trade_status = "closed"

        if self.order_client:
            self.order_client.place_order("sell", symbol, float(amt_to_sell), order_type="market")
            self.logger.info(
                f"[LIVE/PAPER] SELL {symbol} => {portion * 100:.1f}%, amt={amt_to_sell:.4f}, reason={reason}, fees={fees:.2f}, pnl={realized_pnl:.2f}"
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
                "trade_cost": float(trade_cost),
                "strategy_name": "pullback"
            }
            self.db_manager.save_trade(trade_data)
        else:
            self.logger.info(
                f"[Paper] SELL {symbol} => {portion * 100:.1f}%, amt={amt_to_sell:.4f}, reason={reason}, (fees={fees:.2f}, pnl={realized_pnl:.2f})"
            )

        self.open_positions[symbol]["amount"] -= amt_to_sell
        if self.open_positions[symbol]["amount"] <= Decimal("0"):
            self.logger.info(f"[PullbackStrategy] Full position closed => {symbol}")
            db_id = pos.get("db_id", None)
            if db_id:
                self.db_manager.update_trade(db_id, {"status": "closed"})
                self.logger.info(f"[PullbackStrategy] Trade {db_id} => status=closed in DB")
            if db_id:
                self.__record_trade_signals(db_id, event_type="closed", symbol=symbol, atr_mult=self.pullback_atr_mult)
            del self.open_positions[symbol]
        else:
            db_id = pos.get("db_id", None)
            if db_id:
                old_row = self.db_manager.execute_query("SELECT fees, pnl_eur FROM trades WHERE id=? LIMIT 1", (db_id,))
                if old_row:
                    old_fees, old_pnl = old_row[0]
                    new_fees = old_fees + fees
                    new_pnl = old_pnl + realized_pnl
                    self.db_manager.update_trade(db_id, {"status": "partial", "fees": new_fees, "pnl_eur": new_pnl})
                    self.logger.info(f"[PullbackStrategy] updated open trade {db_id} => partial fees={new_fees}, pnl={new_pnl}")
            if db_id:
                self.__record_trade_signals(db_id, event_type="partial", symbol=symbol, atr_mult=self.pullback_atr_mult)

    # [Hulp-code, ongewijzigd]
    @staticmethod
    def _calculate_fees_and_pnl(self, side: str, amount: float, price: float, reason: str) -> (float, float):
        trade_cost = amount * price
        fees = 0.0025 * trade_cost
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
        Ongewijzigd: laadt open trades, herberekent ATR
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
                "position_type": position_type,
            }
            df_main = self._fetch_and_indicator(symbol, self.main_timeframe, limit=200)
            atr_value = self._calculate_atr(df_main, self.atr_window)
            if atr_value:
                pos_data["atr"] = Decimal(str(atr_value))

            if "id" in row:
                pos_data["db_id"] = row["id"]

            self.open_positions[symbol] = pos_data
            self.logger.info(f"[PullbackStrategy] Hersteld open pos => {symbol}, side={side}, amt={amount}, entry={entry_price}")

    def manage_intra_candle_exits(self):
        """
        Ongewijzigd
        """
        self.logger.info("[PullbackStrategy] manage_intra_candle_exits => start SL/TP checks.")
        for sym in list(self.open_positions.keys()):
            pos = self.open_positions[sym]
            current_price = self._get_ws_price(sym)
            # [NIEUW] Log de entry_price samen met symbol, side en current_price
            self.logger.info(
                f"[manage_intra_candle_exits] symbol={sym}, side={pos['side']}, entry={pos['entry_price']}, current={current_price}"
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

        equity_now = eur_balance + total_pos_value
        profit_val = equity_now - self.initial_capital
        profit_pct = (profit_val / self.initial_capital * Decimal("100")) if self.initial_capital > 0 else 0
        self.logger.info(f"[EquityCheck] equity_now={equity_now:.2f}, init_cap={self.initial_capital}, "
                         f"profit_val={profit_val:.2f}, profit_pct={profit_pct:.2f}%")
        return equity_now

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
                self.logger.debug(f"[DEBUG] Geen candles uit 'candles_kraken' voor {symbol} ({interval}).")
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
            self.db_manager.save_trade_signals(signals_data)
            self.logger.debug(f"[__record_trade_signals] trade_id={trade_id}, event={event_type}, symbol={symbol}")
        except Exception as e:
            self.logger.error(f"[__record_trade_signals] Fout: {e}")
