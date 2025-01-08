# src/strategy/pullback_accumulate_strategy.py

"""
Pullback & Accumulate Strategy – Uitgebreide versie
--------------------------------------------------
1) Config via YAML (ipv hardcoded).
2) Placeholder voor ml_engine/zelflerend component.
3) Fail-safes (daily max loss, flash crash detectie).
4) Depth Trend en geavanceerde steun/weerstand (pivot-points, RSI, MACD).
5) Alle logica in één module met inline-commentaar waarom en hoe.
"""

import logging
from decimal import Decimal
import pandas as pd
from typing import Tuple, Optional
import os
import yaml

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# TA-bibliotheken
from ta.volatility import AverageTrueRange
from ta.trend import MACD

# Lokale imports (pas paden aan jouw project!)
from src.logger.logger import setup_logger
from src.database_manager.database_manager import DatabaseManager
from src.indicator_analysis.indicators import Market, IndicatorAnalysis
# (of gebruik direct PyYAML als je wilt)


# (Optioneel) placeholder ml_engine-model (apart module?)
try:
    import joblib
except ImportError:
    joblib = None

class PullbackAccumulateStrategy:
    """
    Geavanceerde Pullback & Accumulate Strategy
    ------------------------------------------
    Belangrijkste vernieuwingen:
    - Config-instellingen via YAML (ipv hardcoded).
    - Zelflerende module (ml_engine) als placeholder: kan indicatoren / signaalsterkte herwegen.
    - Fail-safes:
      * daily_max_loss => bij overschrijding => trading stoppen.
      * flash_crash_drop => detecteer extreme daling in x minuten => pauze.
    - Depth Trend & geavanceerde S/R:
      * Eenvoudig pivot-points om (S1, S2, R1, R2) te vinden.
      * RSI & MACD-check voor extra confirmatie.
      * Orderboek-check (Depth Trend).
    """

    def __init__(self, client, db_manager: DatabaseManager, config_path: Optional[str] = None):
        """
        :param client:     Een client (live of paper) om orders te plaatsen.
        :param db_manager: DatabaseManager voor candles, ticker, orderbook, etc.
        :param config_path: pad naar je config.yaml (optioneel).
        """
        self.client = client
        self.db_manager = db_manager
        self.ml_engine = None


        if config_path and os.path.isfile(config_path):
            global_config = load_config(config_path)
            self.strategy_config = global_config.get("pullback_accumulate_strategy", {})
        else:
            # fallback: hardcoded dict of lege dict
            self.strategy_config = {}

        # Lees parameters uit config
        self.main_timeframe = self.strategy_config.get("main_timeframe", "1h")
        self.trend_timeframe = self.strategy_config.get("trend_timeframe", "4h")
        self.entry_timeframe = self.strategy_config.get("entry_timeframe", "15m")
        self.daily_timeframe = self.strategy_config.get("daily_timeframe", "1d")  # of "1D"
        self.atr_window = self.strategy_config.get("atr_window", 14)
        self.pullback_threshold_pct = Decimal(str(self.strategy_config.get("pullback_threshold_pct", 1.0)))
        self.accumulate_threshold = Decimal(str(self.strategy_config.get("accumulate_threshold", 1.25)))
        self.position_size_pct = Decimal(str(self.strategy_config.get("position_size_pct", 0.05)))
        self.tp1_atr_mult = Decimal(str(self.strategy_config.get("tp1_atr_mult", "1.0")))
        self.tp2_atr_mult = Decimal(str(self.strategy_config.get("tp2_atr_mult", "2.0")))
        self.trail_atr_mult = Decimal(str(self.strategy_config.get("trail_atr_mult", "1.0")))
        self.initial_capital = Decimal(str(self.strategy_config.get("initial_capital", "1000")))
        self.log_file = self.strategy_config.get("log_file", "pullback_strategy.log")

        # Fail-safes
        self.max_daily_loss_pct = Decimal(str(self.strategy_config.get("max_daily_loss_pct", 5.0)))
        self.flash_crash_drop_pct = Decimal(str(self.strategy_config.get("flash_crash_drop_pct", 10.0)))
        self.use_depth_trend = self.strategy_config.get("use_depth_trend", True)

        # Geavanceerde S/R & RSI/MACD
        self.pivot_points_window = int(self.strategy_config.get("pivot_points_window", 20))
        self.rsi_window = int(self.strategy_config.get("rsi_window", 14))
        self.macd_fast = int(self.strategy_config.get("macd_fast", 12))
        self.macd_slow = int(self.strategy_config.get("macd_slow", 26))
        self.macd_signal = int(self.strategy_config.get("macd_signal", 9))

        # ml_engine-component
        self.ml_model_enabled = self.strategy_config.get("ml_model_enabled", False)
        self.ml_model_path = self.strategy_config.get("ml_model_path", "models/pullback_model.pkl")
        self.ml_model = None
        if self.ml_model_enabled and joblib is not None:
            if os.path.exists(self.ml_model_path):
                try:
                    self.ml_model = joblib.load(self.ml_model_path)
                except Exception as e:
                    print(f"Fout bij laden ml_engine-model: {e}")
                    self.ml_model = None

        # Logger
        self.logger = setup_logger("pullback_strategy", self.log_file, logging.DEBUG)
        self.logger.info("[PullbackAccumulateStrategy] Initialized with config from %s", config_path)

        from src.ml_engine.ml_engine import MLEngine
        self.ml_engine = MLEngine(self.db_manager, config_path=config_path)

        # Posities opslaan
        self.open_positions = {}

        # Flag om extra te investeren na +25% winst
        self.invested_extra = False


    # --------------------------------------------------------------------------
    # HOOFDLOOP: execute_strategy
    # --------------------------------------------------------------------------
    def execute_strategy(self, symbol: str):
        self.logger.info(f"[PullbackStrategy] Start for {symbol}")

        # 1) Check fail-safes
        #   (flash crash check => 5m => doen we later)
        if self._check_fail_safes(symbol):
            return

        # 2) Daily / H4 / hoofdrichting
        df_daily = self._fetch_and_indicator(symbol, self.daily_timeframe, limit=60)
        df_h4 = self._fetch_and_indicator(symbol, self.trend_timeframe, limit=60)

        if df_daily.empty or df_h4.empty:
            return

        # Analyseer df_daily & df_h4 => hoofdrichting
        direction = self._check_trend_direction(df_daily, df_h4)

        ml_signal = self._ml_predict_signal(df_daily)
        if ml_signal == 1:
            self.logger.info("[ml_engine] Koop signaal!")
        elif ml_signal == -1:
            self.logger.info("[ml_engine] Short signaal!")

        # 3) Haal je main_timeframe (1h) data
        df_main = self._fetch_and_indicator(symbol, self.main_timeframe, limit=200)
        if df_main.empty:
            return

        # Bepaal intraday-trend / ATR op df_main
        atr_value = self._calculate_atr(df_main, window=self.atr_window)
        if not atr_value:
            return

        # 3) Bepaal bull/bear/range
        #    (bv. gem. slope of RSI op H4)
        direction = self._check_trend_direction(df_daily, df_h4)

        # 6) Haal 15m of 5m candles (pullbackdetectie)
        df_15m = self._fetch_and_indicator(symbol, "15m", limit=100)
        # of df_5m
        if not df_15m.empty:
            pullback_detected = self._detect_pullback(df_15m, Decimal(str(df_15m['close'].iloc[-1])))
        else:
            pullback_detected = False

        # 1) Check fail-safes
        if self._check_fail_safes(symbol):
            self.logger.warning(f"[PullbackStrategy] Fail-safe triggered => skip trading {symbol}.")
            return

        # 2) Haal data + indicators
        df_daily = self._fetch_and_indicator(symbol, self.daily_timeframe, limit=200)
        df_main = self._fetch_and_indicator(symbol, self.main_timeframe, limit=200)

        if df_daily.empty or df_main.empty:
            self.logger.warning(f"[PullbackStrategy] No data (daily/main) for {symbol}. Skip.")
            return


        # 3) Bereken ATR & pivot-points
        atr_value = self._calculate_atr(df, window=self.atr_window)
        if atr_value is None:
            self.logger.warning(f"[PullbackStrategy] Not enough data to calc ATR => skip {symbol}.")
            return

        # optioneel: pivot-points als geavanceerde S/R
        pivot_levels = self._calculate_pivot_points(df)

        current_price = Decimal(str(df['close'].iloc[-1]))

        # 4) Depth Trend (orderboek)
        depth_score = 0.0
        if self.use_depth_trend:
            depth_score = self._analyze_depth_trend(symbol)
            self.logger.info(f"[DepthTrend] Score for {symbol} => {depth_score:.2f}")
            # depth_score > 0 => bullish, < 0 => bearish, 0 => neutraal (voorbeeld)

        # 5) Pullback detectie + manage posities
        pullback_detected = self._detect_pullback(df, current_price)
        has_position = (symbol in self.open_positions)

        # 6) RSI + MACD + ml_engine-check
        rsi_val = df["rsi"].iloc[-1]
        macd_signal_score = self._check_macd(df)

        ml_signal = 0
        if self.ml_model_enabled and self.ml_model:
            ml_signal = self._ml_predict_signal(df)
            self.logger.debug(f"[ml_engine] model predicts signal={ml_signal} for {symbol} (1=koop, -1=short, 0=none)")

        # 7) Account-balance check (accumulatie)
        total_equity = self._get_equity_estimate()

        invest_extra_flag = False
        if (total_equity >= self.initial_capital * self.accumulate_threshold) and not self.invested_extra:
            invest_extra_flag = True
            self.logger.info("[PullbackStrategy] +25%% => next pullback => invest extra in %s", symbol)

        # 8) Beslissingslogica: open/close posities
        if pullback_detected and not has_position:
            # Check RSI < 50, MACD bullish, depth_score > 0, ml_signal==1, etc. (afhankelijk van je wensen)
            if (rsi_val < 50) and (macd_signal_score > 0) and (depth_score >= 0) and (ml_signal >= 0):
                self._open_position(symbol, current_price, atr_value, extra_invest=invest_extra_flag)
                if invest_extra_flag:
                    self.invested_extra = True
        elif has_position:
            self._manage_open_position(symbol, current_price, atr_value)

        self.logger.info(f"[PullbackStrategy] Finished execute_strategy for {symbol}")

    def _check_trend_direction(self, df_daily: pd.DataFrame, df_h4: pd.DataFrame) -> str:
        """
        Combineer Daily- en H4-gegevens om 'bull', 'bear' of 'range' te bepalen
        """
        if df_daily.empty or df_h4.empty:
            return "range"

        # Eenvoudig voorbeeld: kijk naar RSI op daily én H4
        rsi_daily = df_daily["rsi"].iloc[-1]
        rsi_h4 = df_h4["rsi"].iloc[-1]

        if rsi_daily > 60 and rsi_h4 > 60:
            return "bull"
        elif rsi_daily < 40 and rsi_h4 < 40:
            return "bear"
        else:
            return "range"

    # --------------------------------------------------------------------------
    # FAIL-SAFES
    # --------------------------------------------------------------------------
    def _check_fail_safes(self, symbol: str) -> bool:
        """
        - daily_max_loss: check of je account > X% gedaald is t.o.v. begin van de dag => return True => skip trading.
        - flash_crash: check of de laatste candle(s) > Y% daling => True => skip trading (evt. 5m checks).
        """
        # 1) daily_max_loss
        if self._daily_loss_exceeded():
            return True

        # 2) flash crash
        if self._flash_crash_detected(symbol):
            return True

        return False

    def _daily_loss_exceeded(self) -> bool:
        """
        Eenvoudig voorbeeld: als je startdagkapitaal (bv. 1000) met >5% gedaald is => skip trading.
        In praktijk zou je dit per dag resetten, of DB-check doen hoeveel equity je had bij start van de dag.
        """
        if not self.client:
            return False

        balance_eur = Decimal(str(self.client.get_balance().get("EUR", "1000")))
        drop_pct = (self.initial_capital - balance_eur) / self.initial_capital * Decimal("100")
        if drop_pct >= self.max_daily_loss_pct:
            self.logger.warning(f"[FailSafe] daily loss {drop_pct:.2f}% >= {self.max_daily_loss_pct}% => STOP.")
            return True
        return False

    def _flash_crash_detected(self, symbol: str) -> bool:
        # haal timeframe uit config (default = 5m)
        fc_timeframe = self.strategy_config.get("flash_crash_timeframe", "5m")

        df_fc = self._fetch_and_indicator(symbol, fc_timeframe, limit=3)
        if df_fc.empty or len(df_fc) < 3:
            return False

        # Zelfde logica voor drop_pct
        first_close = Decimal(str(df_fc['close'].iloc[0]))
        last_close = Decimal(str(df_fc['close'].iloc[-1]))
        drop_pct = (first_close - last_close) / first_close * Decimal("100")

        if drop_pct >= self.flash_crash_drop_pct:
            self.logger.warning(f"[FailSafe] Flash Crash detected => drop {drop_pct:.2f}% on {fc_timeframe}.")
            return True

        return False

        # optioneel check 1m
        #df_1m = self._fetch_and_indicator(symbol, "1m", limit=3)
        #if not df_1m.empty and len(df_1m) >= 3:
            #first_close_1m = Decimal(str(df_1m['close'].iloc[0]))
            #last_close_1m = Decimal(str(df_1m['close'].iloc[-1]))
            #drop_1m = (first_close_1m - last_close_1m) / first_close_1m * Decimal("100")
            #if drop_1m >= (self.flash_crash_drop_pct / 2):  # bijv. 50% streng
                #self.logger.warning(f"[FailSafe] 1m flash crash => {drop_1m:.2f}%.")
                #return True

        #return False

    # --------------------------------------------------------------------------
    # DATA & INDICATOR FUNCTIES
    # --------------------------------------------------------------------------
    def _fetch_and_indicator(self, symbol: str, interval: str, limit=200) -> pd.DataFrame:
        """
        Haal candles op en bereken RSI, MACD, etc.
        Ook save we de indicatoren in DB indien gewenst.
        """
        market_obj = Market(symbol, self.db_manager)
        df = market_obj.fetch_candles(interval=interval, limit=limit)
        if df.empty:
            return pd.DataFrame()

        df.columns = ['timestamp','market','interval','open','high','low','close','volume']
        # Gebruik je IndicatorAnalysis om RSI te berekenen
        df = IndicatorAnalysis.calculate_indicators(df, rsi_window=self.rsi_window)

        # MACD handmatig (of in je IndicatorAnalysis)
        macd_ind = MACD(
            close=df['close'],
            window_slow=self.macd_slow,
            window_fast=self.macd_fast,
            window_sign=self.macd_signal
        )
        df['macd'] = macd_ind.macd()
        df['macd_signal'] = macd_ind.macd_signal()

        # Optioneel: sla indicators in DB
        # self.db_manager.save_indicators(...)  # als je wilt

        return df

    def _calculate_atr(self, df: pd.DataFrame, window=14) -> Optional[Decimal]:
        if len(df) < window:
            return None
        atr_obj = AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=window
        )
        atr_series = atr_obj.average_true_range()
        last_atr = atr_series.iloc[-1]
        return Decimal(str(last_atr))

    def _calculate_pivot_points(self, df: pd.DataFrame) -> dict:
        """
        Eenvoudige pivot-points (Classic) op basis van de laatste X candles (bv. 20).
        Keert terug dict met { 'S1': xx, 'R1': xx, etc. }
        """
        if len(df) < self.pivot_points_window:
            return {}

        window_df = df.iloc[-self.pivot_points_window:]
        high_ = Decimal(str(window_df['high'].max()))
        low_ = Decimal(str(window_df['low'].min()))
        close_ = Decimal(str(window_df['close'].iloc[-1]))

        pivot = (high_ + low_ + close_) / Decimal("3")
        r1 = (2 * pivot) - low_
        s1 = (2 * pivot) - high_

        # Je kunt R2, S2, R3, S3 etc. uitbreiden
        return {
            "pivot": pivot,
            "R1": r1,
            "S1": s1
        }

    def _detect_pullback(self, df: pd.DataFrame, current_price: Decimal) -> bool:
        """
        We nemen de max 'high' van laatste 20 candles,
        checken of daling >= pullback_threshold_pct
        """
        if len(df) < 20:
            return False
        recent_high = Decimal(str(df['high'].rolling(20).max().iloc[-1]))
        if recent_high == 0:
            return False

        drop_pct = (recent_high - current_price) / recent_high * Decimal("100")
        if drop_pct >= self.pullback_threshold_pct:
            self.logger.info(f"[PullbackStrategy] Pullback detect => {drop_pct:.2f}% below recent high.")
            return True
        return False

    # --------------------------------------------------------------------------
    # ADVANCED CHECKS: RSI, MACD, Depth, ml_engine
    # --------------------------------------------------------------------------
    def _check_macd(self, df: pd.DataFrame) -> float:
        """
        Eenvoudige MACD-check: macd - macd_signal > 0 => bullish. Return +1, -1 of 0.
        """
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
        """
        Eenvoudige implementatie: bekijk orderboek BIDS en ASKS.
        Return +score => bullish, -score => bearish, 0 => neutraal.
        """
        # Stel je hebt in je DB tabellen orderbook_bids, orderbook_asks
        # of je hebt een functie get_orderbook(symbol) => {bids: [...], asks: [...]}
        orderbook = self.db_manager.get_orderbook_snapshot(symbol)  # fictieve functie

        if not orderbook:
            return 0.0

        # Tel totale bid-volume en ask-volume
        total_bids = sum([float(b[1]) for b in orderbook['bids']])  # b = [price, amount]
        total_asks = sum([float(a[1]) for a in orderbook['asks']])

        # Depth Trend Score
        # vb: (bids - asks) / (bids + asks)
        # => +1 => volledig bid-dominantie, -1 => volledig ask-dominantie
        denom = (total_bids + total_asks)
        if denom == 0:
            return 0.0
        score = (total_bids - total_asks) / denom
        return float(score)

    def _ml_predict_signal(self, df: pd.DataFrame) -> int:
        """
        Deze methode wordt in de strategy aangeroepen om ml_engine-signaal te bepalen.
        i.p.v. zelf joblib te laden, roepen we nu self.ml_engine aan.
        """
        if not self.ml_model_enabled:
            return 0

        if self.ml_engine is None:
            self.logger.warning("[PullbackStrategy] Geen ml_engine-engine gekoppeld, return 0.")
            return 0

        # 1) Bouw features
        last_row = df.iloc[-1]
        features = [
            last_row['rsi'],
            last_row['macd'],
            last_row['macd_signal'],
            last_row['volume']
        ]

        # 2) Predict
        signal = self.ml_engine.predict_signal(features)
        return signal

    # --------------------------------------------------------------------------
    # POSITION HANDLING
    # --------------------------------------------------------------------------
    def _open_position(self, symbol: str, current_price: Decimal, atr_value: Decimal, extra_invest=False):
        self.logger.info(f"[PullbackStrategy] OPEN position => {symbol} @ {current_price}, extra={extra_invest}")

        eur_balance = Decimal("1000")
        if self.client:
            bal = self.client.get_balance()
            eur_balance = Decimal(str(bal.get("EUR", "1000")))

        pct = self.position_size_pct
        if extra_invest:
            pct = Decimal("0.10")

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
        """
        TP1= +1x ATR => 25% sell
        TP2= +2x ATR => nog 25% sell
        Resterend 50% => trailing stop = trail_high - 1x ATR
        """
        pos = self.open_positions[symbol]
        entry = pos["entry_price"]
        amount = pos["amount"]

        tp1_price = entry + (pos["atr"] * self.tp1_atr_mult)
        tp2_price = entry + (pos["atr"] * self.tp2_atr_mult)

        # TP1
        if (not pos["tp1_done"]) and (current_price >= tp1_price):
            self.logger.info(f"[PullbackStrategy] TP1 => Sell 25% {symbol}")
            self._sell_portion(symbol, amount, portion=Decimal("0.25"), reason="TP1")
            pos["tp1_done"] = True
            pos["trail_active"] = True
            pos["trail_high"] = max(pos["trail_high"], current_price)

        # TP2
        if (not pos["tp2_done"]) and (current_price >= tp2_price):
            self.logger.info(f"[PullbackStrategy] TP2 => Sell 25% {symbol}")
            self._sell_portion(symbol, amount, portion=Decimal("0.25"), reason="TP2")
            pos["tp2_done"] = True
            pos["trail_active"] = True
            pos["trail_high"] = max(pos["trail_high"], current_price)

        # Trailing
        if pos["trail_active"]:
            # Update trail_high
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
            self.logger.info(f"[LIVE] SELL {symbol} => {portion*100}%, amt={amt_to_sell:.4f}, reason={reason}")
        else:
            self.logger.info(f"[Paper] SELL {symbol} => {portion*100}%, amt={amt_to_sell:.4f}, reason={reason}")

        if symbol in self.open_positions:
            self.open_positions[symbol]["amount"] -= amt_to_sell
            if self.open_positions[symbol]["amount"] <= Decimal("0"):
                self.logger.info(f"[PullbackStrategy] Full position closed => {symbol}")
                del self.open_positions[symbol]

    # --------------------------------------------------------------------------
    # HELPER: BALANCE / EQUITY
    # --------------------------------------------------------------------------
    def _get_equity_estimate(self) -> Decimal:
        if not self.client:
            return self.initial_capital

        # 1) Haal EUR-bal op
        bal = self.client.get_balance()
        eur_balance = Decimal(str(bal.get("EUR", "1000")))

        # 2) Tel de waarde open posities op
        total_positions_value = Decimal("0")

        for sym, pos_info in self.open_positions.items():
            amount = pos_info["amount"]  # hoeveelheid coins
            # Haal de huidige prijs
            latest_price = self._get_latest_price(sym)
            if latest_price > 0:
                pos_value = Decimal(str(amount)) * latest_price
                total_positions_value += pos_value

        total_equity = eur_balance + total_positions_value
        return total_equity

    def _get_latest_price(self, symbol: str) -> Decimal:
        """
        Haal de meest recente prijs van dit symbol,
        bijv. uit de 'ticker' DB of uit de laatste candle-close.
        """
        # Voorbeeld: we proberen 'ticker' te pakken
        ticker_data = self.db_manager.get_ticker(symbol)
        if ticker_data:
            # Stel dat 'bestBid' en 'bestAsk' in ticker_data staan:
            best_bid = Decimal(str(ticker_data.get("bestBid", 0)))
            best_ask = Decimal(str(ticker_data.get("bestAsk", 0)))
            if best_bid > 0 and best_ask > 0:
                return (best_bid + best_ask) / Decimal("2")
            else:
                # fallback: 0 => onvolledig
                pass

        # Als we geen ticker of bestBid/bestAsk=0, pak laatste candle:
        df = self._fetch_and_indicator(symbol, "1m", limit=1)
        if not df.empty:
            last_close = df['close'].iloc[-1]
            return Decimal(str(last_close))

        # fallback
        return Decimal("0")

