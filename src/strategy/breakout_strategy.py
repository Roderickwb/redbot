import logging
import pandas as pd
import time
from decimal import Decimal
from typing import Optional
from datetime import datetime, timedelta, timezone

# TA-bibliotheken
from ta.volatility import AverageTrueRange

# Lokale imports
from src.logger.logger import setup_logger
from src.indicator_analysis.indicators import IndicatorAnalysis
from src.meltdown_manager.meltdown_manager import MeltdownManager


def is_candle_closed(candle_timestamp_ms: int, interval_hours: float) -> bool:
    """
    Bepaalt of een candle (gebaseerd op de starttijd in ms) volledig is afgesloten.
    De candle wordt geacht afgesloten als het huidige UTC-tijdstip >= candle_start + interval_hours.
    """
    candle_end = datetime.fromtimestamp(candle_timestamp_ms / 1000, tz=timezone.utc) + timedelta(hours=interval_hours)
    return datetime.now(timezone.utc) >= candle_end


class BreakoutStrategy:
    """
    Breakout Strategy (4h)
    ---------------------------------------------------------
    - Gebruikt Daily RSI(14) als trendfilter (rsi_bull_level / rsi_bear_level)
    - Op 4h:
      1) Bepaal 'range': highest high /lowest low (laatste X candles) + Bollinger-limieten
      2) Candle sluit >= 0.5% boven/onder die limiet (buffer)
      3) Volume-check (moet > 'volume_threshold_factor' x gemiddeld volume)
      4) Open positie =>
         * Stoploss = ATR * sl_atr_mult  (# [CHANGED] ipv sl_pct)
         * TrailingStop = ATR * trailing_atr_mult  (# [CHANGED] ipv trailing_pct)
         * ~~Geen partial-TP in deze versie~~
    - We nemen nu een max_position_pct / max_position_eur in `_open_position()`,
      net als in Pullback, en checken min_lot (met multiplier).
    - Max 90% van je kapitaal in posities, etc.

    [UITGECOMMENTARIEERD partial-TP code laten we staan als commentaar]
    """

    def __init__(self, client, db_manager, config_path=None):

        self.client = client
        self.db_manager = db_manager

        # 1) Config inladen
        if config_path:
            import yaml
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)
            self.strategy_config = full_config.get("breakout_strategy", {})
        else:
            self.strategy_config = {}

        # Logger
        self.logger = setup_logger("breakout_strategy", "logs/breakout_strategy.log", logging.DEBUG)

        # MeltdownManager-config ophalen
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

        # Belangrijk: nu ATR-based SL + trailing
        self.atr_window = int(self.strategy_config.get("atr_window", 14))
        self.sl_atr_mult = Decimal(str(self.strategy_config.get("sl_atr_mult", "1.0")))         # [CHANGED: used]
        self.trail_atr_mult = Decimal(str(self.strategy_config.get("trailing_atr_mult", "1.0"))) # [CHANGED: used]

        # net als pullback
        self.max_position_pct = Decimal(str(self.strategy_config.get("max_position_pct", "0.05")))
        self.max_position_eur = Decimal(str(self.strategy_config.get("max_position_eur", "15")))
        self.min_lot_multiplier = Decimal(str(self.strategy_config.get("min_lot_multiplier", "1.1")))

        # open_positions
        self.open_positions = {}
        self.initial_capital = Decimal(str(self.strategy_config.get("initial_capital", "125")))

        # Eventueel logger opnieuw instellen
        self.logger = setup_logger("breakout_strategy", self.log_file, logging.DEBUG)

        self.logger.info("[BreakoutStrategy] init (config done).")

    # --------------------------------------------------------------------------
    # _get_min_lot(...) => fallback
    # --------------------------------------------------------------------------
    def _get_min_lot(self, symbol: str) -> Decimal:
        """
        Voorbeeld van een lokale dictionary met wat typische Kraken-minima.
        Als je client een get_min_lot() heeft, kun je dat gewoon aanroepen;
        anders fallback naar deze dictionary.
        """
        kraken_minlots = {
            "XBT-EUR": Decimal("0.0002"),
            "ETH-EUR": Decimal("0.001"),
            "XRP-EUR": Decimal("10"),
            "ADA-EUR": Decimal("10"),
            "DOGE-EUR": Decimal("50"),
            "SOL-EUR": Decimal("0.1"),
            "DOT-EUR": Decimal("0.2"),
        }

        # Als de client een methode get_min_lot(symbol) heeft, probeer die:
        if self.client and hasattr(self.client, "get_min_lot"):
            try:
                return self.client.get_min_lot(symbol)
            except:
                pass  # als er een fout is, val terug op de dictionary

        # Fallback: local dict
        return kraken_minlots.get(symbol, Decimal("1.0"))  # Default is 1.0

    def execute_strategy(self, symbol: str):
        """
        1) meltdown-check
        2) concurrency-check in DB
        3) daily RSI => bull/bear/neutral
        4) detect breakout => open
        5) manage open positions
        """
        # 1) meltdown-check (eerste keer)
        meltdown_active = self.meltdown_manager.update_meltdown_state(strategy=self, symbol=symbol)
        if meltdown_active:
            self.logger.info("[Breakout] meltdown => skip trades & manage pos.")
            if symbol in self.open_positions:
                self._manage_position(symbol)
            return

        self.logger.info(f"[BreakoutStrategy] Start for {symbol}")

        # 1b) meltdown-check (tweede keer)
        meltdown_active = self.meltdown_manager.update_meltdown_state(strategy=self, symbol=symbol)
        if meltdown_active:
            self.logger.info("[Breakout] meltdown => skip trades.")
            if symbol in self.open_positions:
                self._manage_position(symbol)
            return

        # 2) CONCURRENCY-CHECK:
        #    Kijk of er al een open/partial master-trade voor dit symbool in de DB staat.
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
            self.logger.info(f"[Breakout] Already have open MASTER in DB => skip new open for {symbol}.")
            # Wel evt. bestaande open posities managen:
            if symbol in self.open_positions:
                self._manage_position(symbol)
            return

        # 3) daily RSI filter => bull/bear/neutral
        trend = self._check_daily_rsi(symbol)
        if trend == "neutral":
            self.logger.info(f"[Breakout] Trend = neutral => geen trades ({symbol})")
            if symbol in self.open_positions:
                self._manage_position(symbol)
            return

        # 4) check breakout
        breakout_signal = self._detect_breakout(symbol, trend)
        if breakout_signal["breakout_detected"]:
            self.logger.info(
                f"[Breakout] {symbol} => breakout => side={breakout_signal['side']}, "
                f"price={breakout_signal['price']}, ATR={breakout_signal['atr']}, "
                f"limit={breakout_signal['limit_price']}"
            )
            if symbol not in self.open_positions:
                self._open_position(symbol, breakout_signal)
        else:
            self.logger.info(f"[Breakout] {symbol} => geen breakout => skip")

        # 5) manage open pos
        if symbol in self.open_positions:
            self._manage_position(symbol)

    def _check_daily_rsi(self, symbol: str) -> str:
        """
        Bepaalt bull/bear/neutral op basis van RSI-drempels (daily)
        """
        df_daily = self._fetch_and_indicator(symbol, self.daily_timeframe, limit=60)
        if df_daily.empty:
            self.logger.debug(f"[Breakout] {symbol} => geen daily data => neutral")
            return "neutral"

        latest_rsi = df_daily["rsi"].iloc[-1]
        # Log RSI
        self.logger.info(f"[Breakout] {symbol} daily RSI={latest_rsi:.2f} "
                         f"(bull>={self.rsi_bull_level}, bear<={self.rsi_bear_level})")

        if latest_rsi >= self.rsi_bull_level:
            self.logger.debug(f"[Breakout] {symbol} => RSI >= bull_level => bull trend")
            return "bull"
        elif latest_rsi <= self.rsi_bear_level:
            self.logger.debug(f"[Breakout] {symbol} => RSI <= bear_level => bear trend")
            return "bear"
        else:
            return "neutral"

    def _detect_breakout(self, symbol: str, trend: str) -> dict:
        """
        4h-lange candles => highest high /lowest low => check of candle-close
        > (limit + buffer)
        """
        df_4h = self._fetch_and_indicator(symbol, self.main_timeframe, limit=200)
        # sort
        df_4h.sort_values("timestamp", inplace=True)
        if len(df_4h) < self.lookback_candles:
            self.logger.debug(
                f"[Breakout] {symbol} => {len(df_4h)} candles < lookback={self.lookback_candles} => skip"
            )
            return {"breakout_detected": False}

        # Bepaal interval_hours
        if self.main_timeframe.endswith("h"):
            interval_hours = int(self.main_timeframe[:-1])
        elif self.main_timeframe.endswith("d"):
            interval_hours = int(self.main_timeframe[:-1]) * 24
        elif self.main_timeframe.endswith("m"):
            interval_hours = int(self.main_timeframe[:-1]) / 60
        else:
            interval_hours = 4

        recent_slice = df_4h.iloc[-self.lookback_candles:]
        hh = Decimal(str(recent_slice["high"].max()))
        ll = Decimal(str(recent_slice["low"].min()))
        upper_bb = Decimal(str(df_4h["bb_upper"].iloc[-1]))
        lower_bb = Decimal(str(df_4h["bb_lower"].iloc[-1]))

        # Houding t.o.v. bollinger
        if trend == "bull":
            breakout_limit = hh if hh < upper_bb else upper_bb
        else:
            breakout_limit = ll if ll > lower_bb else lower_bb

        last_candle_ts = int(df_4h["timestamp"].iloc[-1])

        # ---------------------------------------------------------------
        # (OPTIONEEL) check 15m / 5m fallback: je zou hier in plaats van
        #   is_candle_closed(...) op "4h" logic, kunnen inschakelen
        #   "5m" candle-check.
        #   => if not is_candle_closed(last_candle_ts, 4):
        #          # ga naar _fetch_xxx("5m") of "15m" en check
        # ---------------------------------------------------------------

        if not is_candle_closed(last_candle_ts, interval_hours):
            self.logger.debug(f"[Breakout] {symbol} => laatste candle niet gesloten => penultimate.")
            if len(df_4h) >= 2:
                last_4h_close = Decimal(str(df_4h["close"].iloc[-2]))
            else:
                last_4h_close = Decimal(str(df_4h["close"].iloc[-1]))
        else:
            last_4h_close = Decimal(str(df_4h["close"].iloc[-1]))

        # Haal de actuele prijs op (alleen voor logging; de strategie gebruikt de candle-close)
        current_price = self._get_latest_price(symbol)
        self.logger.debug(
            f"[Breakout] {symbol} => last_4h_close={last_4h_close}, current_price={current_price}, "
            f"diff={current_price - last_4h_close}"
        )

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
            self.logger.debug(
                f"[Breakout] {symbol}(bull) => breakout_limit={breakout_limit}, "
                f"required_price={required_price}, close={last_4h_close}"
            )
            if last_4h_close > required_price:
                if last_volume > (avg_volume * Decimal(str(self.volume_threshold_factor))):
                    return {
                        "breakout_detected": True,
                        "side": "buy",
                        "price": last_4h_close,
                        "atr": atr_val,
                        "limit_price": breakout_limit
                    }
                else:
                    self.logger.debug(f"[Breakout] {symbol} => volume check fail => {last_volume} <= factor * {avg_volume}")
        else:  # bear
            required_price = breakout_limit * (Decimal("1") - buffer_decimal)
            self.logger.debug(
                f"[Breakout] {symbol}(bear) => breakout_limit={breakout_limit}, "
                f"required_price={required_price}, close={last_4h_close}"
            )
            if last_4h_close < required_price:
                if last_volume > (avg_volume * Decimal(str(self.volume_threshold_factor))):
                    return {
                        "breakout_detected": True,
                        "side": "sell",
                        "price": last_4h_close,
                        "atr": atr_val,
                        "limit_price": breakout_limit
                    }
                else:
                    self.logger.debug(f"[Breakout] {symbol} => volume fail => {last_volume} < factor * {avg_volume}")

        return {"breakout_detected": False}

    def _open_position(self, symbol: str, breakout_signal: dict):
        side = breakout_signal["side"]
        price = breakout_signal["price"]
        atr_val = breakout_signal["atr"]

        # meltdown-check
        meltdown_active = self.meltdown_manager.meltdown_active
        if meltdown_active:
            self.logger.warning(f"[Breakout] meltdown => skip open pos => {symbol}")
            return

        # concurrency-check => zie execute_strategy => skip

        # Spot-check: als side=='sell' => controleer of we coin hebben
        if side == "sell" and self.client:
            base_coin = symbol.split("-")[0]
            bal = self.client.get_balance()
            base_balance = Decimal(str(bal.get(base_coin, "0")))
            if base_balance <= 0:
                self.logger.warning(f"[Breakout] {symbol} => 0 {base_coin} => skip short")
                return

        # [CHANGED] => bereken equity, check max_position_pct, max_position_eur
        eq_now = self._get_equity_estimate()
        allowed_eur_pct = eq_now * self.max_position_pct
        allowed_eur = min(allowed_eur_pct, self.max_position_eur)
        if allowed_eur < 5:
            self.logger.warning(f"[Breakout] {symbol} => allowed_eur={allowed_eur:.2f} <5 => skip.")
            return

        # min lot check
        raw_min_lot = self._get_min_lot(symbol)
        real_min_lot = raw_min_lot * self.min_lot_multiplier

        # Bepaal invest tot max "allowed_eur"
        invest_eur = allowed_eur
        amt = invest_eur / price
        if amt < real_min_lot:
            self.logger.warning(f"[Breakout] {symbol} => amt={amt:.4f} < minLot={real_min_lot} => skip.")
            return

        self.logger.info(f"[Breakout] open => side={side}, price={price:.2f}, invest_eur={invest_eur:.2f}, amt={amt:.4f}")

        if self.client:
            try:
                self.client.place_order(side, symbol, float(amt), ordertype="market")
                self.logger.info(f"[Breakout LIVE] => {side.upper()} {symbol}, amt={amt:.4f}, px={price}")
            except Exception as e:
                self.logger.warning(f"[Breakout] place_order error => {e}")
                return
        else:
            self.logger.info(f"[Paper] => {side.upper()} {symbol}, amt={amt:.4f}, px={price}")

        # StopLoss => ATR-based (# [CHANGED])
        sl_dist = atr_val * self.sl_atr_mult
        if side == "buy":
            sl_price = price - sl_dist
        else:
            sl_price = price + sl_dist

        # Sla in self.open_positions
        pos_info = {
            "side": side,
            "entry_price": price,
            "amount": amt,
            "stop_loss": sl_price,
            "atr": atr_val,
            # geen partial => direct
            "highest_price": price if side == "buy" else Decimal("0"),
            "lowest_price": price if side == "sell" else Decimal("999999"),
        }
        self.open_positions[symbol] = pos_info

        # Database => MASTER trade
        fees = 0.0
        pnl_eur = 0.0
        trade_cost = float(amt * price)

        trade_data = {
            "timestamp": int(time.time() * 1000),
            "symbol": symbol,
            "side": side,
            "price": float(price),
            "amount": float(amt),
            "position_id": None,  # kun je genereren bv f"{symbol}-{int(time.time())}"
            "position_type": side,
            "status": "open",
            "pnl_eur": pnl_eur,
            "fees": fees,
            "trade_cost": trade_cost,
            "strategy_name": "breakout",
            "is_master": 1
        }
        self.db_manager.save_trade(trade_data)

        master_id = self.db_manager.cursor.lastrowid
        pos_info["db_id"] = master_id

        self.logger.info(f"[Breakout] Master trade {symbol} => db_id={master_id} opened.")
        self._record_trade_signals(trade_id=master_id, event_type="open", symbol=symbol)

    def _manage_position(self, symbol: str):
        pos = self.open_positions.get(symbol)
        if not pos:
            return

        side = pos["side"]
        sl_price = pos["stop_loss"]
        atr_val = pos["atr"]

        current_price = self._get_latest_price(symbol)
        if current_price <= 0:
            return

        # meltdown-check
        meltdown_active = self.meltdown_manager.meltdown_active
        if meltdown_active:
            self.logger.info(f"[Breakout] meltdown => close pos => {symbol}")
            self._close_position(symbol, reason="Meltdown")
            return

        # StopLoss => ATR-based
        if side == "buy":
            if current_price <= sl_price:
                self.logger.info(f"[Breakout manage] LONG SL => close {symbol}")
                self._close_position(symbol, reason="StopLoss")
                return

            # [CHANGED] Trailing => ATR-based
            if current_price > pos["highest_price"]:
                pos["highest_price"] = current_price
            new_sl = pos["highest_price"] - (atr_val * self.trail_atr_mult)
            if new_sl > pos["stop_loss"]:
                old_sl = pos["stop_loss"]
                pos["stop_loss"] = new_sl
                self.logger.info(f"[Breakout] {symbol} => trailing update => old={old_sl:.2f}, new={new_sl:.2f}")

        else:  # short
            if current_price >= sl_price:
                self.logger.info(f"[Breakout manage] SHORT SL => close {symbol}")
                self._close_position(symbol, reason="StopLoss")
                return

            # trailing for short
            if current_price < pos["lowest_price"]:
                pos["lowest_price"] = current_price
            new_sl = pos["lowest_price"] + (atr_val * self.trail_atr_mult)
            if new_sl < pos["stop_loss"]:
                old_sl = pos["stop_loss"]
                pos["stop_loss"] = new_sl
                self.logger.info(f"[Breakout] {symbol} => trailing update => old={old_sl:.2f}, new={new_sl:.2f}")

    def _close_position(self, symbol: str, reason="ForcedClose"):
        pos = self.open_positions.get(symbol)
        if not pos:
            return
        side = pos["side"]
        amt = pos["amount"]
        entry_price = pos["entry_price"]
        db_id = pos.get("db_id", None)

        current_price = self._get_latest_price(symbol)
        if current_price <= 0:
            self.logger.warning(f"[Breakout] can't close => price=0 => skip.")
            return

        exit_side = "sell" if side == "buy" else "buy"
        if self.client:
            self.client.place_order(exit_side, symbol, float(amt), ordertype="market")
            self.logger.info(f"[Breakout] CLOSE => {exit_side.upper()} {symbol} amt={amt:.4f}, reason={reason}")
        else:
            self.logger.info(f"[Paper] close => {exit_side.upper()} {symbol} amt={amt:.4f}, reason={reason}")

        # Verwijder open pos
        del self.open_positions[symbol]
        self.logger.info(f"[Breakout] Positie {symbol} volledig gesloten. reason={reason}")

        # child-trade in DB
        raw_pnl = (current_price - entry_price) * Decimal(str(amt)) if side=="buy" else (entry_price - current_price)*Decimal(str(amt))
        trade_cost = float(current_price * amt)
        fees = trade_cost * 0.004  # Gewoon beide als float houden
        realized_pnl = float(raw_pnl) - fees

        child_data = {
            "timestamp": int(time.time() * 1000),
            "symbol": symbol,
            "side": exit_side,
            "price": float(current_price),
            "amount": float(amt),
            "position_id": None,
            "position_type": side,
            "status": "closed",
            "pnl_eur": realized_pnl,
            "fees": fees,
            "trade_cost": trade_cost,
            "strategy_name": "breakout",
            "is_master": 0
        }
        self.db_manager.save_trade(child_data)
        child_id = self.db_manager.cursor.lastrowid
        self.logger.info(f"[Breakout] Child trade => id={child_id}, closed => realized_pnl={realized_pnl:.2f}")

        # update master => 'closed'
        if db_id:
            old_row = self.db_manager.execute_query(
                "SELECT fees, pnl_eur FROM trades WHERE id=? LIMIT 1",
                (db_id,)
            )
            if old_row:
                old_fees, old_pnl = old_row[0]
                new_fees = old_fees + fees
                new_pnl = old_pnl + realized_pnl
                self.db_manager.update_trade(db_id, {
                    "status": "closed",
                    "fees": new_fees,
                    "pnl_eur": new_pnl
                })
                self.logger.info(f"[Breakout] Master trade={db_id} => closed => final pnl={new_pnl:.2f}")

            self._record_trade_signals(trade_id=db_id, event_type="closed", symbol=symbol)

    def manage_intra_candle_exits(self):
        """
        [NEW] Methode om intra-candle SL te checken op basis van 'live' (ticker) price.
        Elke 5-10s aanroepen vanuit je executor.
        """
        for sym in list(self.open_positions.keys()):
            self._manage_position(sym)

    def _fetch_and_indicator(self, symbol: str, interval: str, limit=200) -> pd.DataFrame:
        """
        Haalt de candles uit 'candles_kraken' via db_manager.fetch_data("candles_kraken", ...).
        Bereken RSI + Bollinger.
        """
        df = self.db_manager.fetch_data(
            table_name="candles_kraken",
            limit=limit,
            market=symbol,
            interval=interval,
            exchange=None
        )
        if df.empty:
            return pd.DataFrame()

        # Sorteren
        df.sort_values("timestamp", inplace=True)

        # --- Strikt converteren naar float, errors="raise" + logging ---
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                try:
                    df[col] = df[col].astype(float, errors="raise")
                except Exception as exc:
                    unique_vals = df[col].unique()
                    self.logger.error(
                        f"[Breakout] Fout bij float-conversie kolom='{col}'. "
                        f"Symbol={symbol}, interval={interval}, "
                        f"UNIEKE WAARDEN (max 10)={unique_vals[:10]} | Error={exc}"
                    )
                    raise

        # RSI + Bollinger
        df = IndicatorAnalysis.calculate_indicators(df, rsi_window=14)

        # Bollinger (of MACD / etc.)
        bb = IndicatorAnalysis.calculate_bollinger_bands(df, window=20, num_std_dev=2)
        df["bb_upper"] = bb["bb_upper"]
        df["bb_lower"] = bb["bb_lower"]

        return df

    def _get_latest_price(self, symbol: str) -> Decimal:
        """
        Haalt de recentste prijs uit 'ticker_kraken'.
        Fallback => 1m in 'candles_kraken'.
        """
        df_ticker = self.db_manager.fetch_data(
            table_name="ticker_kraken",
            limit=1,
            market=symbol
        )
        if not df_ticker.empty:
            best_bid = df_ticker.get("best_bid", [0]).iloc[0] if "best_bid" in df_ticker.columns else 0
            best_ask = df_ticker.get("best_ask", [0]).iloc[0] if "best_ask" in df_ticker.columns else 0
            if best_bid > 0 and best_ask > 0:
                return Decimal(str((best_bid + best_ask) / 2))

        # Fallback => candles_kraken (1m)
        df_1m = self.db_manager.fetch_data(
            table_name="candles_kraken",
            limit=1,
            market=symbol,
            interval="1m"
        )
        if not df_1m.empty and "close" in df_1m.columns:
            return Decimal(str(df_1m["close"].iloc[0]))

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
        """
        Eenvoudige schatting: EUR-bal + waarde open posities.
        """
        if not self.client:
            return self.initial_capital

        bal = self.client.get_balance()
        eur_balance = Decimal(str(bal.get("EUR", "0")))

        total_positions = Decimal("0")
        for sym, pos in self.open_positions.items():
            amt = pos["amount"]
            px = self._get_latest_price(sym)
            total_positions += (Decimal(str(amt)) * px)
        return eur_balance + total_positions

    def _record_trade_signals(self, trade_id: int, event_type: str, symbol: str):
        """
        Logt de beslisindicatoren (daily RSI, 4h RSI, MACD, volume, meltdown-state, etc.)
        in de 'trade_signals' via self.db_manager.save_trade_signals(...).

        Met oog op ML: we slaan hier zoveel mogelijk relevante kolommen op, net als
        in de oorspronkelijke code.
        """
        try:
            # 1) Haal daily RSI
            df_daily = self._fetch_and_indicator(symbol, self.daily_timeframe, limit=60)
            if not df_daily.empty:
                rsi_daily = float(df_daily["rsi"].iloc[-1])
            else:
                rsi_daily = None

            # 2) Haal 4h => RSI, MACD, volume
            df_4h = self._fetch_and_indicator(symbol, self.main_timeframe, limit=60)
            if not df_4h.empty:
                rsi_4h = float(df_4h["rsi"].iloc[-1])
                macd_val = float(df_4h["macd"].iloc[-1]) if "macd" in df_4h.columns else 0.0
                macd_sig = float(df_4h["macd_signal"].iloc[-1]) if "macd_signal" in df_4h.columns else 0.0
                vol_4h = float(df_4h["volume"].iloc[-1])
            else:
                rsi_4h = None
                macd_val = 0.0
                macd_sig = 0.0
                vol_4h = 0.0

            # 3) Als je de ATR(4h) ook wilt loggen voor ML, kun je deze berekenen:
            atr_4h = None
            if not df_4h.empty:
                from ta.volatility import AverageTrueRange
                if len(df_4h) >= 14:  # of self.atr_window
                    atr_calc = AverageTrueRange(
                        high=df_4h["high"], low=df_4h["low"], close=df_4h["close"], window=14
                    )
                    atr_val = atr_calc.average_true_range().iloc[-1]
                    if not pd.isna(atr_val):
                        atr_4h = float(atr_val)

            # 4) meltdown-state
            meltdown_active = self.meltdown_manager.meltdown_active

            # 5) Build signals_data dict
            signals_data = {
                "trade_id": trade_id,
                "event_type": event_type,
                "symbol": symbol,
                "strategy_name": "breakout",

                # daily
                "rsi_daily": rsi_daily,

                # 4h
                "rsi_h4": rsi_4h,
                "macd_val": macd_val,
                "macd_signal": macd_sig,
                "volume_4h": vol_4h,  # expliciete kolom voor volume
                "atr_value": atr_4h,  # 4h-ATR als je wilt

                # breakout gebruikt geen 15m, maar we laten 'm op None voor consistentie
                "rsi_15m": None,

                # Overige placeholders
                "depth_score": 0.0,  # breakout doet niks met depth
                "ml_signal": 0.0,  # ML niet geactiveerd hier
                "meltdown_active": meltdown_active,  # als je dat ook in DB wilt loggen

                "timestamp": int(time.time() * 1000)
            }

            # Sla het signaal op
            self.db_manager.save_trade_signals(signals_data)
            self.logger.info(f"[BreakoutStrategy] _record_trade_signals => trade_id={trade_id}, event={event_type}")

        except Exception as e:
            self.logger.error(f"[BreakoutStrategy] _record_trade_signals fout: {e}")
