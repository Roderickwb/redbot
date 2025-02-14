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
      1) Bepaal 'range': highest high / lowest low (laatste X candles) + Bollinger-limieten
      2) Candle sluit >= 0.5% boven/onder die limiet (buffer)
      3) Volume-check (moet > 'volume_threshold_factor' x gemiddeld volume)
      4) Open positie =>
         * Stoploss = ATR * sl_atr_mult
         * TrailingStop = ATR * trailing_atr_mult
         * ~~Geen partial-TP in deze versie~~
    - Max 90% van je kapitaal in posities, etc.
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

        # Belangrijk: we gebruiken ATR i.p.v. sl_pct/ trailing_pct
        self.atr_window = int(self.strategy_config.get("atr_window", 14))
        self.sl_atr_mult = Decimal(str(self.strategy_config.get("sl_atr_mult", "1.0")))
        self.trail_atr_mult = Decimal(str(self.strategy_config.get("trailing_atr_mult", "1.0")))

        # ~~ partial TP / R-risk-based => uitcommentarieerd ~~
        # self.partial_tp_r = Decimal(str(self.strategy_config.get("partial_tp_r", "1.0")))
        # self.partial_tp_pct = Decimal(str(self.strategy_config.get("partial_tp_pct", "0.25")))

        self.position_size_pct = Decimal(str(self.strategy_config.get("position_size_pct", "0.10")))
        self.max_positions_equity_pct = Decimal(str(self.strategy_config.get("max_positions_equity_pct", "0.90")))
        self.initial_capital = Decimal(str(self.strategy_config.get("initial_capital", "125")))

        # Eventueel logger opnieuw instellen
        self.logger = setup_logger("breakout_strategy", self.log_file, logging.DEBUG)
        if config_path:
            self.logger.info("[BreakoutStrategy] init with config_path=%s", config_path)
        else:
            self.logger.info("[BreakoutStrategy] init (no config_path)")

        # data-structuur om open positions bij te houden
        self.open_positions = {}

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
        2) daily RSI => bull/bear/neutral
        3) detect breakout => open
        4) manage open positions
        """
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

        # (nogmaals meltdown-check, indien gewenst)
        meltdown_active = self.meltdown_manager.update_meltdown_state(strategy=self, symbol=symbol)
        if meltdown_active:
            self.logger.info("[Breakout] meltdown => skip trades.")
            return

        # 1) daily RSI filter
        trend = self._check_daily_rsi(symbol)
        if trend == "neutral":
            self.logger.info(f"[Breakout] Trend = neutral => geen trades ({symbol})")
            return

        # 2) check breakout
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
            # (CHANGED) log skip
            self.logger.info(f"[Breakout] {symbol} => geen breakout detected => skip")

        # 3) manage open pos
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

        if not self._can_open_new_position():
            self.logger.warning(f"[Breakout] {symbol} => Max belegd => skip open pos")
            return

        eur_balance = self._get_eur_balance()
        trade_capital = eur_balance * self.position_size_pct
        if trade_capital < 5:
            self.logger.warning(f"[Breakout] {symbol} => te weinig capital => skip")
            return

        amount = (trade_capital / price) if (price > 0) else Decimal("0")
        if amount <= 0:
            self.logger.warning(f"[Breakout] {symbol} => amount<=0 => skip")
            return

        # [NEW] => Check of deze amount >= minLot
        min_lot = self._get_min_lot(symbol)
        if amount < min_lot:
            self.logger.warning(
                f"[Breakout] {symbol} => berekend amount={amount:.6f} < minLot={min_lot} => skip open pos."
            )
            return

        self.logger.info(
            f"[Breakout] _open_position => side={side}, price={price}, invest={trade_capital:.2f}, amt={amount:.4f}"
        )

        if self.client:
            self.client.place_order(side, symbol, float(amount), order_type="market")
            self.logger.info(f"[LIVE] {side.upper()} {symbol}, amt={amount:.4f}, price={price:.2f}")
        else:
            self.logger.info(f"[Paper] {side.upper()} {symbol}, amt={amount:.4f}, price={price:.2f}")

        # StopLoss => ATR-based
        sl_dist = atr_val * self.sl_atr_mult
        if side == "buy":
            sl_price = price - sl_dist
        else:
            sl_price = price + sl_dist

        # pos-structuur
        pos_info = {
            "side": side,
            "entry_price": price,
            "amount": amount,
            "stop_loss": sl_price,
            "atr": atr_val,
            # "tp1_done": False,  # partial TP uitgecommentarieerd
            "highest_price": price if side == "buy" else Decimal("0"),
            "lowest_price": price if side == "sell" else Decimal("999999")
        }
        self.open_positions[symbol] = pos_info

        # ---------------------------------------------------
        # [NIEUW] => log "open" signals in trade_signals table
        # ---------------------------------------------------
        # We maken net als bij de andere strategieën een "trade row".
        # In de code hierboven hebben we géén ID direct. We loggen in 'trades'
        #   via "save_trade(...)". Dat doen we normaliter in je code,
        #   maar zie hier is het (nog) niet zichtbaar. We imiteren dat even:
        trade_data = {
            "timestamp": int(time.time() * 1000),
            "symbol": symbol,
            "side": side,
            "price": float(price),
            "amount": float(amount),
            "position_id": None,   # breakouts hebben we niet expliciet
            "position_type": side if side in ("buy","sell") else None,
            "status": "open",
            "pnl_eur": 0.0,
            "fees": 0.0,
            "trade_cost": float(amount * price),
            "strategy_name": "breakout",
            "is_master": 1
        }
        self.db_manager.save_trade(trade_data)

        # Om het ID te weten (zodat we in 'trade_signals' de foreign key (trade_id) kunnen zetten),
        #   pakken we de laatst ingevoerde rowid:
        new_trade_id = self.db_manager.cursor.lastrowid
        self.logger.info(f"[Breakout] Master trade {symbol} => db_id={new_trade_id} is open.")

        # Sla DB-id op in pos_info, zodat we 'm bij closing kunnen updaten
        pos_info["db_id"] = new_trade_id

        # signaal loggen:
        self._record_trade_signals(trade_id=new_trade_id, event_type="open", symbol=symbol)
        self.logger.info(f"[Breakout] OPEN {side.upper()} {symbol}@{price:.2f}, SL={sl_price:.2f}")

    def _manage_position(self, symbol: str):
        pos = self.open_positions.get(symbol)
        if not pos:
            return

        side = pos["side"]
        entry = pos["entry_price"]
        amt = pos["amount"]
        sl_price = pos["stop_loss"]
        atr_val = pos["atr"]

        current_price = self._get_latest_price(symbol)
        if current_price <= 0:
            return

        # [CHANGED] - Extra debug
        self.logger.debug(
            f"[Breakout-manage] symbol={symbol}, side={side}, entry={entry}, curr={current_price}, SL={sl_price}"
        )

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

        # partial TP bij 1R
        #if not pos["tp1_done"]:
        #    if side == "buy":
        #        risk = entry - sl_price
        #        tp1 = entry + risk
        #        self.logger.debug(f"[Breakout-manage] {symbol} => LONG risk={risk:.4f}, tp1={tp1:.4f}")
        #        if current_price >= tp1:
        #            self.logger.info(f"[Breakout] {symbol} => 1R => partial LONG => {pos['partial_qty']:.4f}")
        #            self._take_partial_profit(symbol, pos["partial_qty"], "sell")
        #            pos["tp1_done"] = True
        #    else:
        #        risk = sl_price - entry
        #        tp1 = entry - risk
        #        self.logger.debug(f"[Breakout-manage] {symbol} => SHORT risk={risk:.4f}, tp1={tp1:.4f}")
        #        if current_price <= tp1:
        #            self.logger.info(f"[Breakout] {symbol} => 1R => partial SHORT => {pos['partial_qty']:.4f}")
        #            self._take_partial_profit(symbol, pos["partial_qty"], "buy")
        #            pos["tp1_done"] = True

        # Trailing Stop => ATR-based (trailing_atr_mult)
        if side == "buy":
            # highest_price updaten
            if current_price > pos["highest_price"]:
                pos["highest_price"] = current_price
            new_sl = pos["highest_price"] - (atr_val * self.trail_atr_mult)
            if new_sl > pos["stop_loss"]:
                old_sl = pos["stop_loss"]
                pos["stop_loss"] = new_sl
                self.logger.info(
                    f"[Breakout] {symbol} => update trailing SL => old={old_sl:.2f}, new={new_sl:.2f}"
                )
        else:
            if current_price < pos["lowest_price"]:
                pos["lowest_price"] = current_price
            new_sl = pos["lowest_price"] + (atr_val * self.trail_atr_mult)
            if new_sl < pos["stop_loss"]:
                old_sl = pos["stop_loss"]
                pos["stop_loss"] = new_sl
                self.logger.info(
                    f"[Breakout] {symbol} => update trailing SL => old={old_sl:.2f}, new={new_sl:.2f}"
                )

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

        # Verwijder uit open_positions
        del self.open_positions[symbol]
        self.logger.info(f"[Breakout] Positie {symbol} volledig gesloten.")

        # def _take_partial_profit(self, symbol: str, qty: Decimal, exit_side: str):
        #    exit_side="sell" => partial exit (LONG)
        #    exit_side="buy"  => partial exit (SHORT)
        #    """
        #    if symbol not in self.open_positions:
        #        return

        #    pos = self.open_positions[symbol]
        #    current_price = self._get_latest_price(symbol)

        #    if self.client:
        #        self.client.place_order(exit_side, symbol, float(qty), order_type="market")
        #        self.logger.info(f"[LIVE] PARTIAL EXIT => {exit_side.upper()} {qty:.4f} {symbol} @ ~{current_price:.2f}")
        #    else:
        #        self.logger.info(f"[Paper] PARTIAL EXIT => {exit_side.upper()} {qty:.4f} {symbol} @ ~{current_price:.2f}")

        #    pos["amount"] -= qty
        #    if pos["amount"] <= Decimal("0"):
        #        self.logger.info(f"[Breakout] Positie uitgeput => {symbol} closed")
        #        del self.open_positions[symbol]

        # -----------------------------------------
        # [NIEUW] => signals loggen (event='closed')
        # -----------------------------------------
        # Stel dat je in je 'trades' net ook een close-row opslaat (met status='closed'),
        # dan kunnen we hier het 'trade_id' pakken (als je dat trackte).
        # Hier doen we het net als bij open_position: we maken ad-hoc wat info.
        # Normaliter heb je 'db_manager.save_trade(...)' voor de close, dus daarna:

        # (A) Child–trade: 1 row => 'closed', is_master=0
        current_price = self._get_latest_price(symbol)
        raw_pnl = (current_price - pos["entry_price"]) * Decimal(str(amt))
        trade_cost = float(current_price * amt)
        fees = float(trade_cost * Decimal("0.0025"))
        realized_pnl = float(raw_pnl) - fees

        child_data = {
            "timestamp": int(time.time() * 1000),
            "symbol": symbol,
            "side": "sell" if side == "buy" else "buy",  # tegengestelde order
            "price": float(current_price),
            "amount": float(amt),
            "position_id": None,
            "position_type": side,  # "buy"/"sell"
            "status": "closed",
            "pnl_eur": realized_pnl,
            "fees": fees,
            "trade_cost": trade_cost,
            "strategy_name": "breakout",
            "is_master": 0  # Child-trade
        }
        self.db_manager.save_trade(child_data)
        child_trade_id = self.db_manager.cursor.lastrowid
        self.logger.info(f"[Breakout] Child trade => id={child_trade_id}, closed => realized_pnl={realized_pnl:.2f}")

        # (B) Updaten van je master-trade => status='closed'
        master_id = pos.get("db_id", None)
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
                    "status": "closed",
                    "fees": new_fees,
                    "pnl_eur": new_pnl
                })
                self.logger.info(f"[Breakout] Master trade={master_id} updated => closed + PnL={new_pnl:.2f}")

            # Signaal
            self._record_trade_signals(trade_id=master_id, event_type="closed", symbol=symbol)

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
                    # Log de problematische waarden (max 10)
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
            total_positions += (amt * px)
        return eur_balance + total_positions

    def _get_eur_balance(self) -> Decimal:
        """
        Als client=None => return self.initial_capital als fallback
        """
        if not self.client:
            return self.initial_capital

        bal = self.client.get_balance()
        return Decimal(str(bal.get("EUR", "0")))

    def _can_open_new_position(self) -> bool:
        """
        Check of we onder self.max_positions_equity_pct zitten.
        """
        total_equity = self._get_equity_estimate()
        eur_bal = self._get_eur_balance()
        invested = total_equity - eur_bal
        ratio = invested / total_equity if total_equity > 0 else Decimal("0")
        return ratio < self.max_positions_equity_pct

    def manage_intra_candle_exits(self):
        """
        [NEW] Methode om intra-candle SL te checken op basis van 'live' (ticker) price.
        Elke 5-10s aanroepen vanuit je executor.
        """
        # Loop over alle open posities en roep _manage_position() aan
        for sym in list(self.open_positions.keys()):
            curr_price = self._get_latest_price(sym)
            if curr_price > 0:
                self._manage_position(sym)

    # --------------------------------------------------------------------------
    # [NEW] Hulpmethode voor het loggen van indicatoren in 'trade_signals'
    # --------------------------------------------------------------------------
    def _record_trade_signals(self, trade_id: int, event_type: str, symbol: str):
        """
        Logt de beslisindicatoren (bv. RSI daily/h4, MACD op 4h, volume op 4h)
        in de 'trade_signals' tabel via db_manager.save_trade_signals(...).

        :param trade_id:   ID van de zojuist weggeschreven trade-row in 'trades'
        :param event_type: "open", "closed", ...
        :param symbol:     "BTC-EUR"
        """
        try:
            # 1) Haal daily RSI
            df_daily = self._fetch_and_indicator(symbol, self.daily_timeframe, limit=60)
            rsi_daily = float(df_daily["rsi"].iloc[-1]) if (not df_daily.empty) else None

            # 2) Haal 4h => RSI + MACD + volume
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

            meltdown_active = self.meltdown_manager.meltdown_active

            # Bouw dict
            signals_data = {
                "trade_id": trade_id,
                "event_type": event_type,
                "symbol": symbol,
                "strategy_name": "breakout",
                "rsi_daily": rsi_daily,
                "rsi_h4": rsi_4h,
                "rsi_15m": None,        # in breakout niet gebruikt
                "macd_val": macd_val,
                "macd_signal": macd_sig,
                "atr_value": None,      # evt. kun je hier 4h-ATR opslaan
                "depth_score": 0.0,     # breakout gebruikt geen depth-check
                "ml_signal": 0.0,       # ML niet geactiveerd hier
                "timestamp": int(time.time() * 1000),
                # evt. meltdown_active=meltdown_active, ...
                #  volume => we loggen in macd_val/macd_signal, of apart 'vol_4h'
            }

            # we kunnen 'volume' in 4h in een eigen kolom opslaan (bvb. 'extra_float1'):
            # of je past je schema van trade_signals-tabel aan om 'volume' direct op te nemen.
            # Stel dat je extra kolom 'volume_4h' hebt, dan:
            # signals_data["volume_4h"] = vol_4h

            self.db_manager.save_trade_signals(signals_data)
            self.logger.info(f"[BreakoutStrategy] _record_trade_signals => trade_id={trade_id}, event={event_type}")
        except Exception as e:
            self.logger.error(f"[BreakoutStrategy] _record_trade_signals fout: {e}")
