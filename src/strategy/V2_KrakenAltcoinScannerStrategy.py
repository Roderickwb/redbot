import logging
import time
import requests
import pandas as pd
from decimal import Decimal
from typing import Dict
from datetime import datetime, timedelta, timezone

from src.meltdown_manager.meltdown_manager import MeltdownManager
from src.logger.logger import setup_logger

# Voor RSI/ATR - als je eigen indicator code hebt, kun je die importeren.
from src.indicator_analysis.indicators import IndicatorAnalysis
from ta.volatility import AverageTrueRange


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
      * ~~TP = tp_pct% boven entry (of trailing)~~ (UITGECOMMENTARIEERD)
    - max_positions_equity_pct => max % van je kapitaal in deze strategie
    """

    def __init__(self, kraken_client, db_manager, config: Dict, logger=None):
        self.client = kraken_client
        self.db_manager = db_manager  # Wordt (optioneel) gebruikt door meltdown_manager
        self.config = config

        self.enabled = bool(config.get("enabled", True))
        self.log_file = config.get("log_file", "logs/altcoin_scanner_strategy.log")

        if logger:
            self.logger = logger
        else:
            self.logger = self._setup_logger("kraken_altcoin_scanner", self.log_file)

        # meltdown-config
        meltdown_cfg = config.get("meltdown_manager", {})
        self.meltdown_manager = MeltdownManager(meltdown_cfg, db_manager=db_manager, logger=self.logger)

        # excl. symbols
        self.exclude_symbols = config.get("exclude_symbols", [])
        self.timeframe = config.get("timeframe", "15m")
        self.lookback = int(config.get("lookback", 6))

        # thresholds
        self.price_change_threshold = Decimal(str(config.get("price_change_threshold", 5.0)))
        self.volume_threshold_factor = Decimal(str(config.get("volume_threshold_factor", 2.0)))
        self.min_base_volume = Decimal(str(config.get("min_base_volume", 5000)))

        # pos-management
        self.position_size_pct = Decimal(str(config.get("position_size_pct", "0.03")))
        self.max_positions_equity_pct = Decimal(str(config.get("max_positions_equity_pct", "0.50")))
        self.sl_pct = Decimal(str(config.get("sl_pct", 2.0)))
        self.trailing_enabled = bool(config.get("trailing_enabled", False))
        self.trailing_pct = Decimal(str(config.get("trailing_pct", 2.0)))

        # Nieuwe keys
        self.min_lot_multiplier = Decimal(str(config.get("min_lot_multiplier", "1.1")))
        self.sl_atr_mult = Decimal(str(config.get("sl_atr_mult", "1.0")))
        self.trailing_atr_mult = Decimal(str(config.get("trailing_atr_mult", "1.0")))

        # open_positions => { "DOGE-EUR": {...}, ... }
        self.open_positions = {}
        self.initial_capital = Decimal(str(config.get("initial_capital", "100")))

        # extra "pump-limit" en RSI(15m)
        self.pump_min_pct = Decimal(str(config.get("pump_min_pct", "5")))
        self.pump_max_pct = Decimal(str(config.get("pump_max_pct", "10")))
        self.pump_lookback_candles = int(config.get("pump_lookback_candles", 4))
        self.rsi_15m_threshold = float(config.get("rsi_15m_threshold", 70.0))

        # ATR
        self.atr_window = int(config.get("atr_window", 14))

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

    # ---------------------------------------------------------------------
    # [NEW] _get_min_lot(...) => fallback dictionary
    # ---------------------------------------------------------------------
    def _get_min_lot(self, symbol: str) -> Decimal:
        kraken_minlots = {
            "XBT-EUR": Decimal("0.0002"),
            "ETH-EUR": Decimal("0.001"),
            "XRP-EUR": Decimal("10"),
            "ADA-EUR": Decimal("10"),
            "DOGE-EUR": Decimal("50"),
            "SOL-EUR": Decimal("0.1"),
            "DOT-EUR": Decimal("0.2"),
        }

        if self.client and hasattr(self.client, "get_min_lot"):
            try:
                return self.client.get_min_lot(symbol)
            except:
                pass

        return kraken_minlots.get(symbol, Decimal("1.0"))

    # =================================================
    # Hoofd-functie => elke X minuten aanroepen
    # =================================================
    def execute_strategy(self):
        self.logger.debug("[AltcoinScanner] execute_strategy called.")

        # meltdown-check
        meltdown_active = self.meltdown_manager.update_meltdown_state(strategy=self, symbol=None)
        if meltdown_active:
            self.logger.warning("[AltcoinScanner] meltdown => skip scanning & close pos.")
            return

        if not self.enabled:
            self.logger.info("[KrakenAltcoinScanner] Strategy disabled => skip.")
            return

        self.logger.debug("[AltcoinScanner] Fetching kraken markets (dynamic REST) ...")
        dynamic_symbols = self._fetch_all_eur_pairs()
        if not dynamic_symbols:
            self.logger.warning("[AltcoinScanner] geen dynamische EUR-paren => stop.")
            return

        # [NIEUW] Filter None/lege symbolen eruit
        valid_symbols = []
        for i, sym in enumerate(dynamic_symbols):
            if not sym:  # sym is None of ""
                self.logger.warning(f"[KrakenAltcoinScanner] symbol is None/empty => skip. index={i}")
                continue
            valid_symbols.append(sym)

        # Filter excludes e.d.
        self.logger.debug(
            f"[AltcoinScanner] Found {len(valid_symbols)} EUR pairs. Filtering exclude/min_baseVol etc."
        )
        tradable_symbols = []
        for sym in valid_symbols:
            if sym in self.exclude_symbols:
                self.logger.debug(f"[AltcoinScanner] symbol={sym} is in exclude_symbols => skip.")
                continue
            tradable_symbols.append(sym)

        self.logger.info(
            f"[KrakenAltcoinScanner] scanning {len(tradable_symbols)} symbols, timeframe={self.timeframe}."
        )

        # 2) Voor elk symbool => check momentum (pump-limit + RSI-check + volume)
        for symbol in tradable_symbols:
            if symbol in self.open_positions:
                self.logger.debug(f"[AltcoinScanner] symbol={symbol} already open => skip scanning.")
                continue

            if not self._can_open_new_position():
                self.logger.info("[KrakenAltcoinScanner] max positions => skip scanning.")
                break

            df = self._fetch_candles(symbol, self.timeframe, limit=(self.pump_lookback_candles + 5))
            if df.empty or len(df) < self.pump_lookback_candles:
                self.logger.debug(f"[AltcoinScanner] symbol={symbol} => not enough candles => skip.")
                continue

            old_close = Decimal(str(df["close"].iloc[-self.pump_lookback_candles]))
            new_close = Decimal(str(df["close"].iloc[-1]))
            self.logger.debug(f"[AltcoinScanner] symbol={symbol}, old_close={old_close}, new_close={new_close}")

            if old_close <= 0:
                self.logger.debug(f"[AltcoinScanner] symbol={symbol}, old_close <= 0 => skip.")
                continue

            price_change_pct = (new_close - old_close) / old_close * Decimal("100")

            if price_change_pct < self.pump_min_pct:
                self.logger.debug(f"[AltcoinScanner] {symbol}: +{price_change_pct:.2f}% < {self.pump_min_pct}% => skip.")
                continue
            if price_change_pct > self.pump_max_pct:
                self.logger.debug(f"[AltcoinScanner] {symbol}: +{price_change_pct:.2f}% > {self.pump_max_pct}% => overshoot => skip.")
                continue

            recent_vol = Decimal(str(df["volume"].iloc[-1]))
            avg_vol = Decimal(str(df["volume"].tail(self.pump_lookback_candles).mean())) if self.pump_lookback_candles>0 else Decimal("0")
            if avg_vol <= 0:
                continue
            vol_factor = recent_vol / avg_vol
            if vol_factor < self.volume_threshold_factor:
                self.logger.debug(
                    f"[AltcoinScanner] {symbol} => vol_factor={vol_factor:.2f} < {self.volume_threshold_factor} => skip"
                )
                continue

            # RSI-check => RSI(15m) < rsi_15m_threshold
            rsi_val = self._get_rsi_15m(symbol)
            if rsi_val is None:
                self.logger.debug(f"[AltcoinScanner] {symbol} => RSI(15m)=None => skip.")
                continue
            if rsi_val >= self.rsi_15m_threshold:
                self.logger.debug(f"[AltcoinScanner] {symbol} => RSI(15m)={rsi_val:.1f} >= {self.rsi_15m_threshold} => skip.")
                continue

            self.logger.info(
                f"[AltcoinScanner] {symbol} => +{price_change_pct:.2f}% in last {self.pump_lookback_candles}, vol x{vol_factor:.2f}, RSI={rsi_val:.1f} => opening pos"
            )
            self._open_position(symbol, new_close)

        # Manage open positions
        for sym in list(self.open_positions.keys()):
            self.logger.debug(f"[AltcoinScanner] Manage existing position => {sym}")
            self._manage_position(sym)


    def _open_position(self, symbol: str, current_price: Decimal):
        self.logger.debug(f"[AltcoinScanner] _open_position called for {symbol} @ {current_price}")
        eur_balance = self._get_eur_balance()
        trade_cap = eur_balance * self.position_size_pct
        if trade_cap < 5:
            self.logger.info(f"[AltcoinScanner] symbol={symbol} => te weinig balance => skip.")
            return

        amt = trade_cap / current_price

        min_lot = self._get_min_lot(symbol)
        needed_amt = min_lot * self.min_lot_multiplier
        if amt < needed_amt:
            self.logger.warning(
                f"[AltcoinScanner] symbol={symbol} => amt={amt:.6f} < needed={needed_amt} (minLot*mult) => skip open pos."
            )
            return

        if self.client:
            self.client.place_order("buy", symbol, float(amt), order_type="market")
            self.logger.info(f"[LIVE] BUY {symbol}, amt={amt:.4f} @ ~{current_price:.4f}")
        else:
            self.logger.info(f"[Paper] BUY {symbol}, amt={amt:.4f} @ ~{current_price:.4f}")

        # Log in DB => 'trades'
        trade_data = {
            "timestamp": int(time.time() * 1000),
            "symbol": symbol,
            "side": "buy",
            "price": float(current_price),
            "amount": float(amt),
            "position_id": f"{symbol}-{int(time.time())}",
            "position_type": "long",
            "status": "open",
            "pnl_eur": 0.0,
            "fees": 0.0,
            "trade_cost": float(amt * current_price),
            "strategy_name": "scanner"
        }
        self.db_manager.save_trade(trade_data)
        new_trade_id = self.db_manager.cursor.lastrowid

        # ATR-based StopLoss
        atr_val = self._calculate_atr(symbol, self.timeframe, window=self.atr_window)
        if atr_val > 0:
            sl_price = current_price - (atr_val * self.sl_atr_mult)
        else:
            sl_price = current_price * Decimal("0.98")
        self.logger.info(
            f"[AltcoinScanner] OPEN LONG {symbol} => ATR={atr_val}, SL={sl_price:.4f}"
        )

        # RSI(15m) => voor logging
        rsi_val = self._get_rsi_15m(symbol)

        self._record_trade_signals(
            trade_id=new_trade_id,
            event_type="open",
            symbol=symbol,
            current_price=current_price,
            note="Scanner => Momentum LONG open",
            atr_value=atr_val,
            rsi_value=rsi_val
        )

        self.open_positions[symbol] = {
            "side": "buy",
            "entry_price": current_price,
            "amount": amt,
            "stop_loss": sl_price,
            "highest_price": current_price
        }
        self.logger.info(f"[AltcoinScanner] Position opened => side=BUY, symbol={symbol}, SL={sl_price:.4f}")


    def _manage_position(self, symbol: str):
        pos = self.open_positions.get(symbol)
        if not pos:
            return

        self.logger.debug(f"[AltcoinScanner] _manage_position => {symbol}, side={pos['side']}")
        curr_price = self._get_latest_price(symbol)
        if curr_price <= Decimal("0"):
            self.logger.debug(f"[AltcoinScanner] symbol={symbol} => current_price=0 => skip manage.")
            return

        sl_price = pos["stop_loss"]
        if curr_price <= sl_price:
            self.logger.info(f"[AltcoinScanner] {symbol} => SL geraakt => close pos")
            self._close_position(symbol)
            return

        if self.trailing_enabled:
            if curr_price > pos["highest_price"]:
                pos["highest_price"] = curr_price
            atr_trail = self._calculate_atr(symbol, self.timeframe, self.atr_window)
            if atr_trail > 0:
                new_sl = pos["highest_price"] - (atr_trail * self.trailing_atr_mult)
            else:
                new_sl = pos["highest_price"] * Decimal("0.98")

            if new_sl > pos["stop_loss"]:
                old_sl = pos["stop_loss"]
                pos["stop_loss"] = new_sl
                self.logger.info(
                    f"[AltcoinScanner] trailing SL updated => old={old_sl:.4f}, new={new_sl:.4f} for {symbol}"
                )


    def _close_position(self, symbol: str):
        pos = self.open_positions.pop(symbol, None)
        if not pos:
            return
        amt = pos["amount"]
        if self.client:
            self.client.place_order("sell", symbol, float(amt), order_type="market")
            self.logger.info(f"[LIVE] CLOSE LONG => SELL {symbol} amt={amt:.4f}")
        else:
            self.logger.info(f"[Paper] CLOSE LONG => SELL {symbol} amt={amt:.4f}")
        self.logger.info(f"[AltcoinScanner] Positie {symbol} volledig gesloten.")

        current_price = self._get_latest_price(symbol)
        trade_data = {
            "timestamp": int(time.time() * 1000),
            "symbol": symbol,
            "side": "sell",
            "price": float(current_price),
            "amount": float(amt),
            "position_id": None,
            "position_type": "long",
            "status": "closed",
            "pnl_eur": 0.0,
            "fees": 0.0,
            "trade_cost": float(amt * current_price),
            "strategy_name": "scanner"
        }
        self.db_manager.save_trade(trade_data)
        closed_trade_id = self.db_manager.cursor.lastrowid

        atr_val = self._calculate_atr(symbol, self.timeframe, self.atr_window)
        rsi_val = self._get_rsi_15m(symbol)

        self._record_trade_signals(
            trade_id=closed_trade_id,
            event_type="closed",
            symbol=symbol,
            current_price=current_price,
            note="Scanner => Momentum LONG closed",
            atr_value=atr_val,
            rsi_value=rsi_val
        )


    def _fetch_candles(self, symbol: str, interval: str, limit=50) -> pd.DataFrame:
        self.logger.debug(f"[AltcoinScanner] _fetch_candles => symbol={symbol}, interval={interval}, limit={limit}")
        int_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60, "4h": 240, "1d": 1440}
        iv = int_map.get(interval, 15)

        # [NIEUW] check of symbol leeg is
        if not symbol:
            self.logger.warning(f"[AltcoinScanner] _fetch_candles => symbol is None/empty => return empty DF.")
            return pd.DataFrame()

        df = self._get_kraken_ohlc(symbol, iv, limit)
        if df.empty:
            self.logger.debug(f"[AltcoinScanner] symbol={symbol}, interval={interval}, => empty DF.")
            return df

        if "timestamp" not in df.columns:
            self.logger.warning(f"[AltcoinScanner] symbol={symbol} => NO 'timestamp' column => skip => empty DF.")
            return pd.DataFrame()

        df.sort_values("timestamp", inplace=True)
        return df


    def _get_kraken_ohlc(self, symbol: str, iv_int: int, limit=50) -> pd.DataFrame:
        # [NIEUW] check of symbol leeg is
        if not symbol:
            self.logger.error("[AltcoinScanner] _get_kraken_ohlc => symbol=None/empty => return empty DF.")
            return pd.DataFrame()

        pair_rest = symbol.replace("-", "/")
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": pair_rest, "interval": iv_int}

        try:
            rr = requests.get(url, params=params, timeout=5)
            rr.raise_for_status()
            data = rr.json()
            if data.get("error"):
                self.logger.debug(f"[AltcoinScanner] _get_kraken_ohlc => error => {data['error']}")
                return pd.DataFrame()

            result = data.get("result", {})
            found_key = None
            for k in result.keys():
                if pair_rest in k:
                    found_key = k
                    break
            if not found_key:
                return pd.DataFrame()

            rows = result[found_key]
            outlist = []
            for row in rows:
                if len(row) < 8:
                    continue
                try:
                    t_s = float(row[0])
                    open_val = float(row[1])
                    high_val = float(row[2])
                    low_val = float(row[3])
                    close_val = float(row[4])
                    vol_val = float(row[6])
                except Exception as e:
                    self.logger.error(f"[AltcoinScanner] parse-error => symbol={symbol}, row={row}, err={e}")
                    continue

                outlist.append({
                    "timestamp": t_s * 1000,
                    "open": open_val,
                    "high": high_val,
                    "low": low_val,
                    "close": close_val,
                    "volume": vol_val
                })

            df = pd.DataFrame(outlist)
            if df.empty:
                return df

            if len(df) > limit:
                df = df.iloc[-limit:]

            cols = ["open", "high", "low", "close", "volume"]
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df.dropna(subset=cols, inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"[AltcoinScanner] _get_kraken_ohlc error => {e}")
            return pd.DataFrame()


    def _fetch_all_eur_pairs(self) -> list:
        try:
            url = "https://api.kraken.com/0/public/AssetPairs"
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            j = r.json()
            if j.get("error"):
                self.logger.debug(f"[AltcoinScanner] _fetch_all_eur_pairs => error => {j['error']}")
                return []
            result = j.get("result", {})
            out = []
            for restname, info in result.items():
                ws = info.get("wsname", "")
                if ws.endswith("/EUR"):
                    sym = ws.replace("/", "-")
                    out.append(sym)
            return out
        except Exception as ex:
            self.logger.error(f"[AltcoinScanner] fetch_all_eur_pairs => {ex}")
            return []


    def _get_latest_price(self, symbol: str) -> Decimal:
        if hasattr(self.client, "get_latest_ws_price"):
            px = self.client.get_latest_ws_price(symbol)
            if px > 0:
                return Decimal(str(px))

        df_1m = self._fetch_candles(symbol, "1m", limit=1)
        if not df_1m.empty and "close" in df_1m.columns:
            last_close = df_1m["close"].iloc[0]
            return Decimal(str(last_close))
        return Decimal("0")


    def _get_eur_balance(self) -> Decimal:
        if not self.client:
            return self.initial_capital
        bals = self.client.get_balance()
        return Decimal(str(bals.get("EUR", "0")))


    def _get_equity_estimate(self) -> Decimal:
        if not self.client:
            return self.initial_capital
        eur_bal = self._get_eur_balance()
        total_val = Decimal("0")
        for sym, pos in self.open_positions.items():
            px = self._get_latest_price(sym)
            total_val += (pos["amount"] * px)
        return eur_bal + total_val


    def _can_open_new_position(self) -> bool:
        tot_eq = self._get_equity_estimate()
        bal = self._get_eur_balance()
        invested = tot_eq - bal
        ratio = invested / tot_eq if tot_eq > 0 else Decimal("0")
        self.logger.debug(
            f"[AltcoinScanner] _can_open_new_position => ratio={ratio:.2f}, max={self.max_positions_equity_pct}"
        )
        return ratio < self.max_positions_equity_pct


    def manage_intra_candle_exits(self):
        for sym in list(self.open_positions.keys()):
            pos = self.open_positions[sym]
            curr_price = self._get_latest_price(sym)
            if curr_price > 0:
                self._manage_position(sym)


    def _fetch_and_indicator(self, symbol: str, interval: str, limit=200) -> pd.DataFrame:
        df = self._fetch_candles(symbol, interval, limit=limit)
        return df


    def _record_trade_signals(self, trade_id: int, event_type: str, symbol: str,
                              current_price: Decimal, note: str = "",
                              atr_value: Decimal = None,
                              rsi_value: float = None):
        try:
            df = self._fetch_candles(symbol, self.timeframe, limit=(self.lookback + 5))
            price_change_pct = 0.0
            vol_factor = 0.0

            if not df.empty and len(df) >= self.lookback:
                old_close = Decimal(str(df["close"].iloc[-self.lookback]))
                new_close = Decimal(str(df["close"].iloc[-1]))
                if old_close > 0:
                    price_change_pct = float((new_close - old_close) / old_close * Decimal("100"))

                recent_vol = Decimal(str(df["volume"].iloc[-1]))
                avg_vol = Decimal(str(df["volume"].tail(self.lookback).mean())) if self.lookback > 0 else Decimal("0")
                if avg_vol > 0:
                    vol_factor = float(recent_vol / avg_vol)

            meltdown_active = self.meltdown_manager.meltdown_active
            signals_data = {
                "trade_id": trade_id,
                "event_type": event_type,
                "symbol": symbol,
                "strategy_name": "scanner",
                "rsi_daily": price_change_pct,   # legacy naming
                "rsi_h4": vol_factor,           # legacy naming
                "rsi_15m": float(rsi_value) if rsi_value else 0.0,
                "macd_val": float(current_price),
                "macd_signal": None,
                "atr_value": float(atr_value) if atr_value else 0.0,
                "depth_score": float(999) if meltdown_active else 0.0,
                "ml_signal": 0.0,
                "timestamp": int(time.time() * 1000)
            }

            self.db_manager.save_trade_signals(signals_data)
            self.logger.info(
                f"[_record_trade_signals] trade_id={trade_id}, event={event_type}, symbol={symbol}, "
                f"ATR={atr_value}, RSI={rsi_value}, note={note}"
            )
        except Exception as e:
            self.logger.error(f"[_record_trade_signals] Fout: {e}")


    def _get_rsi_15m(self, symbol: str, rsi_window=14) -> float:
        df_rsi = self._fetch_candles(symbol, "15m", limit=30)
        if df_rsi.empty or len(df_rsi) < rsi_window:
            return None

        df_rsi.sort_values("timestamp", inplace=True)
        df_rsi = IndicatorAnalysis.calculate_indicators(df_rsi, rsi_window=rsi_window)
        if "rsi" not in df_rsi.columns:
            return None
        return float(df_rsi["rsi"].iloc[-1])


    def _calculate_atr(self, symbol: str, interval: str, window=14) -> Decimal:
        df_atr = self._fetch_candles(symbol, interval, limit=50)
        if df_atr.empty or len(df_atr) < window:
            return Decimal("0")

        df_atr.sort_values("timestamp", inplace=True)
        atr_obj = AverageTrueRange(
            high=df_atr["high"], low=df_atr["low"], close=df_atr["close"], window=window
        )
        series_atr = atr_obj.average_true_range()
        val = series_atr.iloc[-1]
        if pd.isna(val):
            return Decimal("0")
        return Decimal(str(val))
