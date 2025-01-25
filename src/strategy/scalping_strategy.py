# src/strategy/scalping_strategy.py

import logging
from decimal import Decimal
import pandas as pd
from typing import Tuple

from ta.volatility import AverageTrueRange  # Voor ATR (volatiliteit)

# Lokale imports (pas paden aan je projectstructuur aan)
from src.logger.logger import setup_logger
from src.database_manager.database_manager import DatabaseManager
from src.indicator_analysis.indicators import Market, IndicatorAnalysis
from src.config.config import scalping_config, SCALPING_LOG_FILE


class ScalpingStrategy:
    """
    Een aparte strategieklasse voor scalping op korte timeframes (1m, 5m, 10m, 15m).
    We hebben nu:
    - Per timeframe een eigen SL/TP (50% + trailing).
    - RSI-drempels op 40/60.
    - Een mini-ATR-check voor 1m en 5m.
    - Versoepelde MACD-check.
    """

    def __init__(self, client, db_manager: DatabaseManager, config: dict):
        """
        :param client: bv. een WebSocketClient of REST-client om orders te plaatsen
        :param db_manager: DatabaseManager instance voor DB-toegang
        :param config: dict met top-level config, waaronder de subdict "scalping".
        """
        self.client = client
        self.db_manager = db_manager

        # === BEGIN CHANGE #A: lees scalping-subdict en hulpparameters uit config.yaml ===

        # Voorbeeld: rsi_lower & rsi_upper (voorheen hardcoded 40/60)
        self.rsi_lower = scalping_config.get("rsi_lower", 40)
        self.rsi_upper = scalping_config.get("rsi_upper", 60)

        # Volatiliteit & volume
        self.volatility_threshold = scalping_config.get("volatility_threshold", 0.0) # hier verlaagd omdat config niet goed werkt en ik wil weten of de bot dan wel handeld
        self.volume_factor = scalping_config.get("volume_factor", 0.5)

        # Overige
        self.rsi_window = scalping_config.get("rsi_window", 7)
        self.use_trailing_for_last25 = scalping_config.get("use_trailing_for_last25", True)
        # === END CHANGE #A

        # Pairs kun je eventueel nog uit de globale config halen
        self.pairs = config.get("pairs", {})

        # Logger
        self.logger = setup_logger("scalping_strategy", SCALPING_LOG_FILE, logging.DEBUG)
        self.logger.info("[ScalpingStrategy] Initialized.")

        # Optionele opslag open posities
        self.open_positions = {}

    def execute_strategy(self, symbol: str):
        """
        Haalt 1m, 5m, 10m, 15m candles op, bepaalt welke het beste signaal geeft,
        en opent/sluit posities obv scalping-logica (plus extra checks).
        """
        self.logger.info(f"[ScalpingStrategy] Start execute_strategy for {symbol}")

        # Check balance (live of paper)
        if self.client:
            balance = self.client.get_balance()
            self.logger.debug(f"[ScalpingStrategy] Balance: {balance}")
        else:
            balance = {}
            self.logger.debug("[ScalpingStrategy] Paper mode, no real balance from client.")

        try:
            df_1m = self._fetch_and_indicator(symbol, "1m", 300)
            df_5m = self._fetch_and_indicator(symbol, "5m", 300)
            df_10m = self._fetch_and_indicator(symbol, "10m", 300)
            df_15m = self._fetch_and_indicator(symbol, "15m", 300)

            self.logger.debug(
                f"[ScalpingStrategy] Fetched dataframes: "
                f"df_1m={len(df_1m)}, df_5m={len(df_5m)}, "
                f"df_10m={len(df_10m)}, df_15m={len(df_15m)}"
            )

            # Sla de indicatoren van alle DFs meteen op in de DB (optioneel)
            if not df_1m.empty:
                df_1m['market'] = symbol
                df_1m['interval'] = "1m"
                self.db_manager.save_indicators(df_1m)

            if not df_5m.empty:
                df_5m['market'] = symbol
                df_5m['interval'] = "5m"
                self.db_manager.save_indicators(df_5m)

            if not df_10m.empty:
                df_10m['market'] = symbol
                df_10m['interval'] = "10m"
                self.db_manager.save_indicators(df_10m)

            if not df_15m.empty:
                df_15m['market'] = symbol
                df_15m['interval'] = "15m"
                self.db_manager.save_indicators(df_15m)

            # === BEGIN CHANGE #B: Verwijder "als ATR > threshold => skip"
            #    We willen alleen skippen als ATR < threshold (te laag).
            #    Dus we halen de code die ATR > threshold => skip weg.
            # (Hier is dus GEEN code meer dat skipt bij "too high" ATR)
            # === END CHANGE #B

            # Check of we iig wat data hebben
            if df_1m.empty and df_5m.empty and df_10m.empty and df_15m.empty:
                self.logger.warning(f"[ScalpingStrategy] Geen data voor {symbol}. Strategy skipped.")
                return

            # Bepaal beste timeframe
            chosen_interval, chosen_df = self._select_best_timeframe(df_1m, df_5m, df_10m, df_15m)
            if chosen_df.empty:
                self.logger.warning(f"[ScalpingStrategy] Geen geldige candles in chosen_df voor {symbol}.")
                return

            # === BEGIN CHANGE #C: mini-ATR-check gebruikt self.volatility_threshold ===
            if chosen_interval in ("1m", "5m"):
                local_atr = self._calculate_average_volatility(chosen_df)
                # i.p.v. if local_atr < 0.001:
                if local_atr < self.volatility_threshold:
                    self.logger.info(
                        f"[ScalpingStrategy] local ATR={local_atr:.4f} < {self.volatility_threshold} => te weinig volatiliteit => skip."
                    )
                    return
            # === END CHANGE #C

            # Lees RSI/MACD etc.
            latest_close = chosen_df['close'].iloc[-1]
            rsi_val = chosen_df['rsi'].iloc[-1]
            macd_val = chosen_df['macd'].iloc[-1]
            macd_signal_val = chosen_df['macd_signal'].iloc[-1]
            ema9_val = chosen_df['ema_9'].iloc[-1]
            ema21_val = chosen_df['ema_21'].iloc[-1]
            cur_volume = chosen_df['volume'].iloc[-1]
            avg_volume = (
                chosen_df['volume'].rolling(20).mean().iloc[-1]
                if len(chosen_df) >= 20 else 1
            )

            self.logger.debug(
                f"[ScalpingStrategy] For {symbol}: close={latest_close}, "
                f"rsi={rsi_val}, macd={macd_val}, macd_signal={macd_signal_val}, "
                f"ema9={ema9_val}, ema21={ema21_val}, volume={cur_volume}"
            )

            # Volume-filter => skip als 'cur_volume' < X% van 'avg_volume'
            if cur_volume < avg_volume * self.volume_factor:
                self.logger.info(
                    f"[ScalpingStrategy] Volume {cur_volume:.2f} < "
                    f"{self.volume_factor*100}% of avg {avg_volume:.2f}. Skip."
                )
                return

            # Micro R/S detectie (optioneel debug)
            micro_levels = self._detect_micro_res_support(chosen_df)
            self.logger.debug(f"[ScalpingStrategy] micro_levels => {micro_levels}")

            # Bepaal BUY/SELL/HOLD (nu met versimpelde/versoepelde checks)
            action = self._determine_signal(
                chosen_df,
                rsi_val,
                macd_val,
                macd_signal_val,
                ema9_val,
                ema21_val,
                latest_close,
                symbol
            )

            # Open / Manage
            if action == "BUY" and symbol not in self.open_positions:
                self._open_scalp_position(symbol, latest_close, chosen_df, chosen_interval)

            if symbol in self.open_positions:
                self._manage_open_position(symbol, latest_close)

        except Exception as e:
            self.logger.exception(f"[ScalpingStrategy] Fout in execute_strategy({symbol}): {e}")
        finally:
            self.logger.info(f"[ScalpingStrategy] Finished execute_strategy for {symbol}")


    # ----------------------------------------------------------------------
    # HELPER FUNCTIES: DATA & INDICATORS
    # ----------------------------------------------------------------------

    def _fetch_and_indicator(self, symbol: str, interval: str, limit=300) -> pd.DataFrame:
        market_obj = Market(symbol, self.db_manager)
        df = market_obj.fetch_candles(interval=interval, limit=limit)
        if df.empty:
            self.logger.debug("No candles")
            return pd.DataFrame()

        df.columns = ['timestamp', 'market', 'interval', 'open', 'high', 'low', 'close', 'volume']
        df = IndicatorAnalysis.calculate_indicators(df, rsi_window=self.rsi_window)
        col_list = list(df.columns)
        self.logger.debug(f"{symbol} ({interval}) => columns after indicators: {col_list}")
        return df

    def _calculate_average_volatility(self, df: pd.DataFrame) -> float:
        if len(df) < 14:
            return 0.0
        atr_obj = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        atr_series = atr_obj.average_true_range()
        avg_atr = atr_series.mean()
        mean_close = df['close'].mean()
        return float(avg_atr / mean_close) if mean_close != 0 else 0.0

    def _detect_micro_res_support(self, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 10:
            return {"resistance": [], "support": []}

        rolling_high = df['high'].rolling(window=5, center=True).max()
        rolling_low = df['low'].rolling(window=5, center=True).min()

        resist_mask = (df['high'] == rolling_high)
        micro_res = df['high'][resist_mask].tolist()

        support_mask = (df['low'] == rolling_low)
        micro_sup = df['low'][support_mask].tolist()

        return {"resistance": micro_res, "support": micro_sup}

    def _select_best_timeframe(
        self,
        df_1m: pd.DataFrame,
        df_5m: pd.DataFrame,
        df_10m: pd.DataFrame,
        df_15m: pd.DataFrame
    ) -> Tuple[str, pd.DataFrame]:
        scores = {}
        if not df_1m.empty:
            scores["1m"] = self._calculate_signal_strength(df_1m)
        else:
            scores["1m"] = -999.0

        if not df_5m.empty:
            scores["5m"] = self._calculate_signal_strength(df_5m)
        else:
            scores["5m"] = -999.0

        if not df_10m.empty:
            scores["10m"] = self._calculate_signal_strength(df_10m)
        else:
            scores["10m"] = -999.0

        if not df_15m.empty:
            scores["15m"] = self._calculate_signal_strength(df_15m)
        else:
            scores["15m"] = -999.0

        best_tf = max(scores, key=scores.get)
        self.logger.info(f"[ScalpingStrategy] Best timeframe: {best_tf} => score={scores[best_tf]}")

        if best_tf == "1m":
            return "1m", df_1m
        elif best_tf == "5m":
            return "5m", df_5m
        elif best_tf == "10m":
            return "10m", df_10m
        elif best_tf == "15m":
            return "15m", df_15m
        else:
            return "1m", pd.DataFrame()  # fallback

    def _calculate_signal_strength(self, df: pd.DataFrame) -> float:
        """
        Evalueert RSI vs rsi_lower/rsi_upper en MACD vs MACD_signaal
        om een 'score' te geven. Hoe hoger, hoe meer bullish.
        """
        last = df.iloc[-1]
        rsi_val = last['rsi']
        macd_val = last['macd']
        macd_signal_val = last['macd_signal']

        score = 0.0
        # RSI-check
        if rsi_val < self.rsi_lower:
            score += 1
        elif rsi_val > self.rsi_upper:
            score -= 1

        # === BEGIN CHANGE 3: MACD-tolerance versoepelen (lager of hoger),
        # bijvoorbeeld 0.001 i.p.v. 0.0005.
        tolerance = 0.001
        # === END CHANGE 3

        macd_gap = abs(macd_val - macd_signal_val)
        if macd_gap < tolerance:
            # Bijna crossover => licht bullish of bearish
            if macd_val > macd_signal_val:
                score += 0.5
            else:
                score -= 0.5
        else:
            if macd_val > macd_signal_val:
                score += 1
            else:
                score -= 1

        return score

    def _get_timeframe_sl_tp(self, interval: str) -> Tuple[Decimal, Decimal]:
        if interval == "1m":
            return Decimal("0.005"), Decimal("0.0075")   # SL=0.5%, TP=0.75%
        elif interval == "5m":
            return Decimal("0.005"), Decimal("0.01")     # SL=0.5%, TP=1.0%
        elif interval == "10m":
            return Decimal("0.0075"), Decimal("0.0125")  # SL=0.75%, TP=1.25%
        elif interval == "15m":
            return Decimal("0.0075"), Decimal("0.015")   # SL=0.75%, TP=1.5%
        else:
            return Decimal("0.005"), Decimal("0.0075")   # fallback: 1m

    # ----------------------------------------------------------------------
    # BESLISSINGSLOGICA: BUY / SELL / HOLD
    # ----------------------------------------------------------------------

    def _determine_signal(
        self,
        df: pd.DataFrame,
        rsi_val: float,
        macd_val: float,
        macd_signal_val: float,
        ema9: float,
        ema21: float,
        latest_close: float,
        symbol: str
    ) -> str:
        """
        Minder strenge criteria:
        - BUY als (RSI < 40) of (below Boll-lower), en MACD bullish
        - SELL als (RSI > 60) of (above Boll-upper), en MACD bearish
        - Anders HOLD
        """
        boll_lower = df['bollinger_lower'].iloc[-1]
        boll_upper = df['bollinger_upper'].iloc[-1]

        below_boll_lower = (latest_close < boll_lower)
        above_boll_upper = (latest_close > boll_upper)

        tolerance = 0.001
        macd_gap = abs(macd_val - macd_signal_val)
        macd_is_bullish = (macd_val > macd_signal_val) or (macd_gap < tolerance and macd_val >= macd_signal_val)
        macd_is_bearish = (macd_val < macd_signal_val) or (macd_gap < tolerance and macd_val <= macd_signal_val)

        last_timestamp = df['timestamp'].iloc[-1]
        self.logger.debug(f"[ScalpingStrategy] Last candle timestamp for {symbol}: {last_timestamp}")

        # BUY-condities (verlicht)
        if ((rsi_val < 40) or below_boll_lower) and macd_is_bullish:
            return "BUY"

        # SELL-condities (verlicht)
        if ((rsi_val > 60) or above_boll_upper) and macd_is_bearish:
            return "SELL"

        return "HOLD"

    # ----------------------------------------------------------------------
    # TRADE & POSITION MANAGEMENT
    # ----------------------------------------------------------------------

    def _open_scalp_position(self, symbol: str, current_price: float, chosen_df: pd.DataFrame, interval: str):
        """
        Opent een scalp-positie met SL/TP per interval, 50% close op TP,
        en trailing stop voor de resterende 50%.
        """
        self.logger.info(f"[ScalpingStrategy] Opening scalp position for {symbol} at {current_price}, interval={interval}")

        sl_pct, tp_pct = self._get_timeframe_sl_tp(interval)

        if self.client:
            balance = self.client.get_balance()
        else:
            balance = {}
        eur_balance = Decimal(balance.get('EUR', 1000))
        if eur_balance <= 0:
            self.logger.warning(f"[ScalpingStrategy] No EUR for {symbol}. Skip buy.")
            return

        buy_eur = eur_balance * Decimal("0.05")  # 5%
        if buy_eur < 5:
            self.logger.warning(f"[ScalpingStrategy] buy_eur < 5 => too small.")
            return

        amount = buy_eur / Decimal(str(current_price))

        # Plaats kooporder (market)
        if self.client:
            self.client.place_order("buy", symbol, float(amount), order_type="market")
            self.logger.info(
                f"[ScalpingStrategy] LIVE BUY {symbol} => amt={amount}, price={current_price}, used {buy_eur} EUR"
            )
        else:
            self.logger.info(f"(Paper) Would BUY {symbol} => amt={amount} at price={current_price}")

        # Sla de positie op
        self.open_positions[symbol] = {
            "entry_price": Decimal(str(current_price)),
            "amount": amount,
            "stop_loss_pct": sl_pct,
            "take_profit_pct": tp_pct,
            "tp_50_done": False,
            "trailing_active": False,
            "trail_high": None,
        }

        self.logger.info(f"[ScalpingStrategy] Position stored => {symbol}")

    def _manage_open_position(self, symbol: str, current_price: float):
        if symbol not in self.open_positions:
            return

        pos = self.open_positions[symbol]
        entry_price = pos["entry_price"]
        amount = pos["amount"]

        # 1) STOP-LOSS
        sl_pct = pos["stop_loss_pct"]
        stop_loss_price = entry_price * (Decimal("1") - sl_pct)
        if Decimal(str(current_price)) <= stop_loss_price:
            self.logger.info(f"[ScalpingStrategy] STOP-LOSS triggered for {symbol}. Close ALL.")
            self._close_position(symbol, portion=1.0, reason="StopLossHit", price=current_price)
            return

        # 2) TAKE-PROFIT (50%)
        tp_pct = pos["take_profit_pct"]
        tp_price = entry_price * (Decimal("1") + tp_pct)

        if (not pos["tp_50_done"]) and (Decimal(str(current_price)) >= tp_price):
            self.logger.info(f"[ScalpingStrategy] TAKE-PROFIT (50%) triggered for {symbol}")
            self._close_position(symbol, portion=0.5, reason="TP50%", price=current_price)

            pos["tp_50_done"] = True
            pos["trailing_active"] = True
            pos["trail_high"] = Decimal(str(current_price))
            self.logger.info(f"[ScalpingStrategy] Trailing stop geactiveerd voor {symbol}")
            return

        # 3) TRAILING STOP op resterende 50%
        if pos.get("trailing_active", False):
            if Decimal(str(current_price)) > pos["trail_high"]:
                old_trail = pos["trail_high"]
                pos["trail_high"] = Decimal(str(current_price))
                self.logger.debug(f"[ScalpingStrategy] Updating trail_high: {old_trail} => {pos['trail_high']}")

            trailing_dist = Decimal("0.003")  # ~0.3%
            trailing_stop_price = pos["trail_high"] * (Decimal("1") - trailing_dist)

            if Decimal(str(current_price)) <= trailing_stop_price:
                self.logger.info(f"[ScalpingStrategy] Trailing stop triggered for {symbol}. Close remainder.")
                self._close_position(symbol, portion=1.0, reason="TrailingStopHit", price=current_price)

    def _close_position(self, symbol: str, portion: float, reason: str, price: float):
        pos = self.open_positions[symbol]
        amt_to_sell = Decimal(str(pos["amount"])) * Decimal(str(portion))

        if self.client:
            self.client.place_order("sell", symbol, float(amt_to_sell))
            self.logger.info(f"[ScalpingStrategy] LIVE SELL {portion*100}% of {symbol} at {price}, reason={reason}")
        else:
            self.logger.info(f"(Paper) Would SELL {portion*100}% of {symbol} @ {price}, reason={reason}")

        if portion < 1.0:
            pos["amount"] -= amt_to_sell
            if pos["amount"] <= 0:
                self.logger.info(f"[ScalpingStrategy] Position fully closed after partial sells => {symbol}")
                del self.open_positions[symbol]
        else:
            self.logger.info(f"[ScalpingStrategy] Position fully closed => {symbol}, reason={reason}")
            del self.open_positions[symbol]

    def handle_order_updates(self, update_data):
        event_type = update_data.get("event")
        self.logger.debug(f"[ScalpingStrategy] handle_order_updates => {update_data}")

        if event_type == "order":
            order_id = update_data.get("orderId")
            self._handle_order_update(order_id, update_data)
        elif event_type == "fill":
            fill_amount = update_data.get("filledAmount")
            self._handle_fill(fill_amount, update_data)

    def _handle_order_update(self, order_id, update_data):
        self.logger.info(f"[ScalpingStrategy] Order update => {order_id}: {update_data}")

    def _handle_fill(self, fill_amount, update_data):
        self.logger.info(f"[ScalpingStrategy] Fill update => fill_amount={fill_amount}, data={update_data}")
