import logging
import time
from decimal import Decimal
from typing import Dict
from datetime import datetime, timedelta, timezone

import pandas as pd


class KrakenAltcoinScannerStrategy:
    """
    Small/Mid-cap Altcoin Momentum/Rotation Strategy (Scanner) – voor Kraken
    -----------------------------------------------------------------------
    - Periodiek scannen we de (kleinere) alts op Kraken (behalve excludes).
    - Criteria:
        1) Price change >= X% in de laatste N candles
        2) Volume spike >= factor * gem. volume
        3) Genoeg baseVolume om illiquide coins te weren
    - Detecteer signaal => open short-term positie (LONG):
      * position_size_pct van 'free' EUR
      * SL = sl_pct% onder entry
      * TP = tp_pct% boven entry (of trailing)
    - max_positions_equity_pct => max % van je kapitaal in deze strategie
    """

    def __init__(self, kraken_client, db_manager, config: Dict, logger=None):
        """
        :param kraken_client:    KrakenMixedClient (of FakeClient) voor orders/balances
        :param db_manager:       DatabaseManager
        :param config:           dict uit config.yaml => config["altcoin_scanner_strategy"]
        :param logger:           (optioneel) eigen logger, anders maken we file+console
        """

        self.client = kraken_client
        self.db_manager = db_manager
        self.config = config

        self.enabled = bool(config.get("enabled", True))
        self.log_file = config.get("log_file", "logs/altcoin_scanner_strategy.log")

        # Welke coins uitsluiten
        self.exclude_symbols = config.get("exclude_symbols", [])
        # timeframe om te scannen
        self.timeframe = config.get("timeframe", "15m")
        # lookback = #candles
        self.lookback = int(config.get("lookback", 6))
        # thresholds
        self.price_change_threshold = Decimal(str(config.get("price_change_threshold", 5.0)))
        self.volume_threshold_factor = Decimal(str(config.get("volume_threshold_factor", 2.0)))
        self.min_base_volume = Decimal(str(config.get("min_base_volume", 5000)))

        # pos-management
        self.position_size_pct = Decimal(str(config.get("position_size_pct", "0.03")))
        self.max_positions_equity_pct = Decimal(str(config.get("max_positions_equity_pct", "0.50")))
        self.sl_pct = Decimal(str(config.get("sl_pct", 2.0)))
        self.tp_pct = Decimal(str(config.get("tp_pct", 5.0)))
        self.trailing_enabled = bool(config.get("trailing_enabled", False))
        self.trailing_pct = Decimal(str(config.get("trailing_pct", 2.0)))

        self.partial_tp_enabled = bool(config.get("partial_tp_enabled", False))
        self.partial_tp_pct = Decimal(str(config.get("partial_tp_pct", 0.25)))

        if logger:
            self.logger = logger
        else:
            self.logger = self._setup_logger("kraken_altcoin_scanner", self.log_file)

        # open_positions => { "DOGE-EUR": {...}, ... }
        self.open_positions = {}

        # initieel kapitaal (indien client=None => paper)
        self.initial_capital = Decimal("1000")

        self.logger.info("[KrakenAltcoinScanner] init met config OK.")

    def _setup_logger(self, name, log_file, level=logging.DEBUG):
        logger = logging.getLogger(name)
        logger.setLevel(level)

        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        return logger

    # =================================================
    # Hoofd-functie => elke X minuten aanroepen
    # =================================================
    def execute_strategy(self):
        if not self.enabled:
            self.logger.info("[KrakenAltcoinScanner] Strategy disabled => skip.")
            return

        # 1) Haal markten op => veronderstel dat db_manager.get_all_kraken_markets()
        #    of get_all_markets() => lijst van dicts:
        #    [{"market":"DOGE-EUR", "baseVolume":"12345.6"}, ...]
        markets = self.db_manager.get_all_kraken_markets()  # pas aan aan jouw implementatie
        if not markets:
            self.logger.warning("[KrakenAltcoinScanner] geen markten => stop.")
            return

        tradable_symbols = []
        for m in markets:
            sym = m.get("market", "")
            if sym in self.exclude_symbols:
                continue
            base_vol_str = m.get("baseVolume", "0")
            base_vol = Decimal(str(base_vol_str))
            if base_vol < self.min_base_volume:
                continue
            tradable_symbols.append(sym)

        self.logger.info(f"[KrakenAltcoinScanner] scanning {len(tradable_symbols)} symbols, timeframe={self.timeframe}.")

        # 2) Voor elk symbool => check momentum
        for symbol in tradable_symbols:
            # skip als reeds open
            if symbol in self.open_positions:
                continue

            # check equity-limiet
            if not self._can_open_new_position():
                self.logger.info("[KrakenAltcoinScanner] max positions => skip scanning.")
                break

            # fetch candles
            df = self._fetch_candles(symbol, self.timeframe, limit=(self.lookback + 10))
            if df.empty or len(df) < self.lookback:
                continue

            old_close = Decimal(str(df["close"].iloc[-self.lookback]))
            new_close = Decimal(str(df["close"].iloc[-1]))
            if old_close <= 0:
                continue

            price_change_pct = (new_close - old_close) / old_close * Decimal("100")

            # volume spike check
            recent_vol = df["volume"].iloc[-1]
            avg_vol = df["volume"].tail(self.lookback).mean()
            if avg_vol > 0:
                vol_factor = Decimal(str(recent_vol)) / Decimal(str(avg_vol))
            else:
                vol_factor = Decimal("0")

            # beslis: is momentum >= threshold?
            if price_change_pct >= self.price_change_threshold and vol_factor >= self.volume_threshold_factor:
                self.logger.info(f"[Scanner] {symbol} => {price_change_pct:.2f}% up in last {self.lookback}, "
                                 f"volume x{vol_factor:.2f}")
                # open positie
                self._open_position(symbol, new_close)

        # 3) Manage posities
        for sym in list(self.open_positions.keys()):
            self._manage_position(sym)

    # =================================================
    # Open positie => LONG
    # =================================================
    def _open_position(self, symbol: str, current_price: Decimal):
        eur_balance = self._get_eur_balance()
        trade_cap = eur_balance * self.position_size_pct
        if trade_cap < 5:
            self.logger.info(f"[Scanner] te weinig balance om {symbol} te traden => skip.")
            return

        amt = trade_cap / current_price
        if self.client:
            self.client.place_order("buy", symbol, float(amt), order_type="market")
            self.logger.info(f"[LIVE] BUY {symbol}, amt={amt:.4f} @ ~{current_price:.4f}")
        else:
            self.logger.info(f"[Paper] BUY {symbol}, amt={amt:.4f} @ ~{current_price:.4f}")

        # Simple SL/TP
        sl_price = current_price * (Decimal("1") - (self.sl_pct/Decimal("100")))
        tp_price = current_price * (Decimal("1") + (self.tp_pct/Decimal("100")))

        self.open_positions[symbol] = {
            "side": "buy",
            "entry_price": current_price,
            "amount": amt,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "partial_tp_done": False,
            "highest_price": current_price
        }
        self.logger.info(f"[Scanner] OPEN LONG {symbol} => SL={sl_price:.4f}, TP={tp_price:.4f}")

    # =================================================
    # Manage positie => check SL/TP/partial/trailing
    # =================================================
    def _manage_position(self, symbol: str):
        pos = self.open_positions.get(symbol)
        if not pos:
            return

        side = pos["side"]  # "buy"
        curr_price = self._get_latest_price(symbol)
        if curr_price <= Decimal("0"):
            return

        # check stop
        sl_price = pos["stop_loss"]
        if curr_price <= sl_price:
            self.logger.info(f"[Scanner] {symbol} => SL geraakt => close pos")
            self._close_position(symbol)
            return

        # check take-profit
        tp_price = pos["take_profit"]
        if curr_price >= tp_price:
            self.logger.info(f"[Scanner] {symbol} => TP geraakt => close pos")
            self._close_position(symbol)
            return

        # partial TP halverwege
        if self.partial_tp_enabled and (not pos["partial_tp_done"]):
            half_target = pos["entry_price"] + (tp_price - pos["entry_price"]) / Decimal("2")
            if curr_price >= half_target:
                part_qty = pos["amount"] * self.partial_tp_pct
                self.logger.info(f"[Scanner] {symbol} => partial TP => sell {part_qty:.4f}")
                if self.client:
                    self.client.place_order("sell", symbol, float(part_qty), order_type="market")
                pos["amount"] -= part_qty
                pos["partial_tp_done"] = True

        # trailing
        if self.trailing_enabled:
            if curr_price > pos["highest_price"]:
                pos["highest_price"] = curr_price
            # trailing SL = highest_price * (1 - trailing%/100)
            trail_sl = pos["highest_price"] * (Decimal("1") - (self.trailing_pct/Decimal("100")))
            if trail_sl > pos["stop_loss"]:
                old_sl = pos["stop_loss"]
                pos["stop_loss"] = trail_sl
                self.logger.info(f"[Scanner] update trailing SL: old={old_sl:.4f}, new={trail_sl:.4f} for {symbol}")

    # =================================================
    # Close positie
    # =================================================
    def _close_position(self, symbol: str):
        pos = self.open_positions.get(symbol)
        if not pos:
            return
        amt = pos["amount"]
        if self.client:
            self.client.place_order("sell", symbol, float(amt), order_type="market")
            self.logger.info(f"[LIVE] CLOSE LONG => SELL {symbol} amt={amt:.4f}")
        else:
            self.logger.info(f"[Paper] CLOSE LONG => SELL {symbol} amt={amt:.4f}")

        del self.open_positions[symbol]
        self.logger.info(f"[Scanner] Positie {symbol} volledig gesloten.")

    # =================================================
    # Helpers
    # =================================================
    def _fetch_candles(self, symbol: str, interval: str, limit=50) -> pd.DataFrame:
        """
        Haal candles van de DB. Zorg dat 'get_candlesticks' data van Kraken opvraagt.
        Retouneert een df met kolommen: "timestamp","market","interval","open","high","low","close","volume"
        """
        # Voorbeeld:
        df = self.db_manager.get_candlesticks(symbol, interval=interval, limit=limit, exchange="Kraken")
        if df.empty:
            return df
        # Zorg dat kolommen kloppen
        df = df.sort_values("timestamp")
        df = df[["timestamp","market","interval","open","high","low","close","volume"]].copy()
        return df

    def _get_latest_price(self, symbol: str) -> Decimal:
        """
        Eenvoudige 'ticker' fallback: bestBid/bestAsk of 1m candle
        """
        tk = self.db_manager.get_ticker(symbol, exchange="Kraken")
        if tk:
            best_bid = Decimal(str(tk.get("best_bid", 0)))
            best_ask = Decimal(str(tk.get("best_ask", 0)))
            if best_bid>0 and best_ask>0:
                return (best_bid+best_ask)/Decimal("2")
        # fallback => 1m candle
        cdf = self._fetch_candles(symbol, "1m", limit=1)
        if not cdf.empty:
            return Decimal(str(cdf["close"].iloc[-1]))
        return Decimal("0")

    def _get_eur_balance(self) -> Decimal:
        if not self.client:
            return self.initial_capital
        bals = self.client.get_balance()
        # bv. bals["EUR"]
        return Decimal(str(bals.get("EUR", "0")))

    def _get_equity_estimate(self) -> Decimal:
        if not self.client:
            return self.initial_capital
        eur_bal = self._get_eur_balance()
        total_val = Decimal("0")
        for sym, pos in self.open_positions.items():
            amt = pos["amount"]
            px = self._get_latest_price(sym)
            total_val += (amt * px)
        return eur_bal + total_val

    def _can_open_new_position(self) -> bool:
        total_eq = self._get_equity_estimate()
        eur_bal = self._get_eur_balance()
        invested = total_eq - eur_bal
        ratio = invested / total_eq if total_eq>0 else Decimal("0")
        if ratio >= self.max_positions_equity_pct:
            return False
        return True
