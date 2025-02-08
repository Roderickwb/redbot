import logging
import time
import requests
import pandas as pd
from decimal import Decimal
from typing import Dict
from datetime import datetime, timedelta, timezone

from src.meltdown_manager.meltdown_manager import MeltdownManager
from src.logger.logger import setup_logger


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
        :param db_manager:       DatabaseManager (voornamelijk voor meltdown_manager,
                                 scanning gebruikt 'm niet meer)
        :param config:           dict uit config.yaml => config["altcoin_scanner_strategy"]
        :param logger:           (optioneel) eigen logger, anders maken we file+console
        """

        self.client = kraken_client
        self.db_manager = db_manager  # Wordt (optioneel) gebruikt door meltdown_manager
        self.config = config

        self.enabled = bool(config.get("enabled", True))
        self.log_file = config.get("log_file", "logs/altcoin_scanner_strategy.log")

        if logger:
            self.logger = logger
        else:
            self.logger = self._setup_logger("kraken_altcoin_scanner", self.log_file)

        # (2) Haal meltdown-config op
        meltdown_cfg = config.get("meltdown_manager", {})
        # (3) Maak meltdown_manager
        self.meltdown_manager = MeltdownManager(meltdown_cfg, db_manager=db_manager, logger=self.logger)

        # Welke coins uitsluiten
        self.exclude_symbols = config.get("exclude_symbols", [])
        # timeframe om te scannen (bv. "15m")
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

        # open_positions => { "DOGE-EUR": {...}, ... }
        self.open_positions = {}

        # initieel kapitaal (indien client=None => paper)
        # [CHANGED] lees uit YAML (default=100):
        self.initial_capital = Decimal(str(config.get("initial_capital", "100")))

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
        # Voorbeeld-lokale minima (fallback)
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
                pass  # als het faalt, val terug

        # Fallback => local dict
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
        # We halen alle EUR-paren van Kraken => /0/public/AssetPairs
        dynamic_symbols = self._fetch_all_eur_pairs()
        if not dynamic_symbols:
            self.logger.warning("[KrakenAltcoinScanner] geen dynamische EUR-paren => stop.")
            return

        # Filter excludes e.d.
        self.logger.debug(f"[AltcoinScanner] Found {len(dynamic_symbols)} EUR pairs. Filtering exclude/min_baseVol etc.")
        tradable_symbols = []
        for sym in dynamic_symbols:
            if sym in self.exclude_symbols:
                self.logger.debug(f"[AltcoinScanner] symbol={sym} is in exclude_symbols => skip.")
                continue
            # Eventueel extra checks
            tradable_symbols.append(sym)

        self.logger.info(f"[KrakenAltcoinScanner] scanning {len(tradable_symbols)} symbols, timeframe={self.timeframe}.")

        # 2) Voor elk symbool => check momentum
        for symbol in tradable_symbols:
            # skip als reeds open
            if symbol in self.open_positions:
                self.logger.debug(f"[AltcoinScanner] symbol={symbol} already open => skip scanning.")
                continue

            # check equity-limiet
            if not self._can_open_new_position():
                self.logger.info("[KrakenAltcoinScanner] max positions => skip scanning.")
                break

            # fetch candles (direct van REST => OHLC)
            df = self._fetch_candles(symbol, self.timeframe, limit=(self.lookback + 10))
            if df.empty or len(df) < self.lookback:
                self.logger.debug(f"[AltcoinScanner] symbol={symbol} => not enough candles => skip.")
                continue

            old_close = Decimal(str(df["close"].iloc[-self.lookback]))
            new_close = Decimal(str(df["close"].iloc[-1]))
            self.logger.debug(f"[AltcoinScanner] symbol={symbol}, old_close={old_close}, new_close={new_close}")

            if old_close <= 0:
                self.logger.debug(f"[AltcoinScanner] symbol={symbol}, old_close <= 0 => skip.")
                continue

            price_change_pct = (new_close - old_close) / old_close * Decimal("100")

            # volume spike check
            recent_vol = Decimal(str(df["volume"].iloc[-1]))
            avg_vol = Decimal(str(df["volume"].tail(self.lookback).mean())) if self.lookback > 0 else Decimal("0")
            if avg_vol > 0:
                vol_factor = recent_vol / avg_vol
            else:
                vol_factor = Decimal("0")

            self.logger.debug(
                f"[AltcoinScanner] symbol={symbol}, price_change={price_change_pct:.2f}%, "
                f"volume factor={vol_factor:.2f}, thresholds => {self.price_change_threshold}% & {self.volume_threshold_factor}x"
            )

            if (price_change_pct >= self.price_change_threshold) and (vol_factor >= self.volume_threshold_factor):
                self.logger.info(
                    f"[AltcoinScanner] {symbol} => {price_change_pct:.2f}% up in last {self.lookback}, "
                    f"volume x{vol_factor:.2f}, => opening position"
                )
                self._open_position(symbol, new_close)
            else:
                self.logger.debug(f"[AltcoinScanner] symbol={symbol} => no momentum => skip")

        # Manage open positions
        for sym in list(self.open_positions.keys()):
            self.logger.debug(f"[AltcoinScanner] Manage existing position => {sym}")
            self._manage_position(sym)

    # =================================================
    # Open positie => LONG
    # =================================================
    def _open_position(self, symbol: str, current_price: Decimal):
        self.logger.debug(f"[AltcoinScanner] _open_position called for {symbol} @ {current_price}")
        eur_balance = self._get_eur_balance()
        trade_cap = eur_balance * self.position_size_pct
        if trade_cap < 5:
            self.logger.info(f"[AltcoinScanner] symbol={symbol} => te weinig balance => skip.")
            return

        amt = trade_cap / current_price

        # check min-lot
        min_lot = self._get_min_lot(symbol)
        if amt < min_lot:
            self.logger.warning(
                f"[AltcoinScanner] symbol={symbol} => calculated amt={amt:.6f} < minLot={min_lot} => skip open pos."
            )
            return

        if self.client:
            self.client.place_order("buy", symbol, float(amt), order_type="market")
            self.logger.info(f"[LIVE] BUY {symbol}, amt={amt:.4f} @ ~{current_price:.4f}")
        else:
            self.logger.info(f"[Paper] BUY {symbol}, amt={amt:.4f} @ ~{current_price:.4f}")

        # Log in DB
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

        # Simple SL/TP
        sl_price = current_price * (Decimal("1") - (self.sl_pct / Decimal("100")))
        tp_price = current_price * (Decimal("1") + (self.tp_pct / Decimal("100")))

        self.open_positions[symbol] = {
            "side": "buy",
            "entry_price": current_price,
            "amount": amt,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "partial_tp_done": False,
            "highest_price": current_price
        }
        self.logger.info(f"[AltcoinScanner] OPEN LONG {symbol} => SL={sl_price:.4f}, TP={tp_price:.4f}")

    # =================================================
    # Manage positie => check SL/TP/partial/trailing
    # =================================================
    def _manage_position(self, symbol: str):
        pos = self.open_positions.get(symbol)
        if not pos:
            return

        self.logger.debug(f"[AltcoinScanner] _manage_position => {symbol}, side={pos['side']}")
        curr_price = self._get_latest_price(symbol)
        if curr_price <= Decimal("0"):
            self.logger.debug(f"[AltcoinScanner] symbol={symbol} => current_price=0 => skip manage.")
            return

        # check stop
        sl_price = pos["stop_loss"]
        if curr_price <= sl_price:
            self.logger.info(f"[AltcoinScanner] {symbol} => SL geraakt => close pos")
            self._close_position(symbol)
            return

        # check take-profit
        tp_price = pos["take_profit"]
        if curr_price >= tp_price:
            self.logger.info(f"[AltcoinScanner] {symbol} => TP geraakt => close pos")
            self._close_position(symbol)
            return

        # partial TP halverwege
        if self.partial_tp_enabled and (not pos["partial_tp_done"]):
            half_target = pos["entry_price"] + (tp_price - pos["entry_price"]) / Decimal("2")
            if curr_price >= half_target:
                part_qty = pos["amount"] * self.partial_tp_pct
                self.logger.info(f"[AltcoinScanner] partial TP => {symbol}, sell {part_qty:.4f}")
                if self.client:
                    self.client.place_order("sell", symbol, float(part_qty), order_type="market")

                # DB-log van partial exit
                trade_data = {
                    "timestamp": int(time.time() * 1000),
                    "symbol": symbol,
                    "side": "sell",
                    "price": float(curr_price),
                    "amount": float(part_qty),
                    "position_id": None,
                    "position_type": "long",
                    "status": "partial",
                    "pnl_eur": 0.0,
                    "fees": 0.0,
                    "trade_cost": float(part_qty * curr_price),
                    "strategy_name": "scanner"
                }
                self.db_manager.save_trade(trade_data)

                pos["amount"] -= part_qty
                pos["partial_tp_done"] = True

        # trailing
        if self.trailing_enabled:
            if curr_price > pos["highest_price"]:
                pos["highest_price"] = curr_price
            # trailing SL = highest_price * (1 - trailing%/100)
            trail_sl = pos["highest_price"] * (Decimal("1") - (self.trailing_pct / Decimal("100")))
            if trail_sl > pos["stop_loss"]:
                old_sl = pos["stop_loss"]
                pos["stop_loss"] = trail_sl
                self.logger.info(
                    f"[AltcoinScanner] trailing SL updated => old={old_sl:.4f}, new={trail_sl:.4f} for {symbol}"
                )

    # =================================================
    # Close positie
    # =================================================
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

        # (NIEUW) DB-log van volledige sluiting
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

    # =================================================
    # Helpers
    # =================================================
    def _fetch_candles(self, symbol: str, interval: str, limit=50) -> pd.DataFrame:
        """
        Haalt candles via Kraken REST /0/public/OHLC?pair=..., in pandas DF.
        interval: "1m", "5m", "15m", "60m", etc => we mappen even naar int
        """
        self.logger.debug(f"[AltcoinScanner] _fetch_candles => symbol={symbol}, interval={interval}, limit={limit}")
        int_map = {"1m":1,"5m":5,"15m":15,"30m":30,"60m":60,"1h":60,"4h":240,"1d":1440}
        iv = int_map.get(interval, 15)
        df = self._get_kraken_ohlc(symbol, iv, limit)
        if df.empty:
            self.logger.debug(f"[AltcoinScanner] symbol={symbol}, interval={interval}, => empty DF.")
            return df
        # sort
        df.sort_values("timestamp", inplace=True)
        return df

    def _get_kraken_ohlc(self, symbol: str, iv_int: int, limit=50) -> pd.DataFrame:
        """
        Roept /0/public/OHLC op, mapped => pd.DataFrame(columns=[timestamp, open, high, low, close, volume])
        + skipt evt. "rare" of "corrupt" rows via try/except
        """
        pair_rest = symbol.replace("-","/")
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
            # Zoek de key in result die (meestal) <pair_rest> of <restName> bevat
            found_key = None
            for k in result.keys():
                if pair_rest in k:
                    found_key = k
                    break
            if not found_key:
                return pd.DataFrame()

            rows = result[found_key]
            outlist = []
            # parse => [time, open, high, low, close, vwap, volume, count]
            for row in rows:
                if len(row) < 8:
                    continue
                try:
                    t_s       = float(row[0])
                    open_val  = float(row[1])
                    high_val  = float(row[2])
                    low_val   = float(row[3])
                    close_val = float(row[4])
                    vol_val   = float(row[6])
                except Exception as e:
                    self.logger.error(f"[AltcoinScanner] parse-error => symbol={symbol}, row={row}, err={e}")
                    continue  # skip dit ene record

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
            # Beperk tot 'limit' laatste candles
            if len(df) > limit:
                df = df.iloc[-limit:]

            # Extra safety: to_numeric & dropna
            # (Mocht er toch nog iets corrupt zijn)
            cols = ["open","high","low","close","volume"]
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df.dropna(subset=cols, inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"[AltcoinScanner] _get_kraken_ohlc error => {e}")
            return pd.DataFrame()

    def _fetch_all_eur_pairs(self)->list:
        """
        Haalt alle wsname die op /EUR eindigt, bijv. XDG/EUR => symbol = 'XDG-EUR'
        return list of local symbols
        """
        try:
            url = "https://api.kraken.com/0/public/AssetPairs"
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            j = r.json()
            if j.get("error"):
                self.logger.debug(f"[AltcoinScanner] _fetch_all_eur_pairs => error => {j['error']}")
                return []
            result = j.get("result", {})
            out=[]
            for restname, info in result.items():
                ws = info.get("wsname","")
                if ws.endswith("/EUR"):
                    sym = ws.replace("/","-")
                    out.append(sym)
            return out
        except Exception as ex:
            self.logger.error(f"[AltcoinScanner] fetch_all_eur_pairs => {ex}")
            return []

    def _get_latest_price(self, symbol: str) -> Decimal:
        """
        Eenvoudige 'ticker' fallback => je kunt ook ws-prijs
        via self.client.get_latest_ws_price(...) gebruiken
        """
        # 1) check of client has get_latest_ws_price
        if hasattr(self.client, "get_latest_ws_price"):
            px = self.client.get_latest_ws_price(symbol)
            if px > 0:
                return Decimal(str(px))

        # 2) fallback => 1m candle
        df_1m = self._fetch_candles(symbol, "1m", limit=1)
        if not df_1m.empty:
            last_close = df_1m["close"].iloc[-1]
            return Decimal(str(last_close))
        return Decimal("0")

    def _get_eur_balance(self) -> Decimal:
        if not self.client:
            return self.initial_capital
        bals = self.client.get_balance()
        return Decimal(str(bals.get("EUR","0")))

    def _get_equity_estimate(self)->Decimal:
        if not self.client:
            return self.initial_capital
        eur_bal = self._get_eur_balance()
        total_val = Decimal("0")
        for sym, pos in self.open_positions.items():
            px = self._get_latest_price(sym)
            total_val += (pos["amount"] * px)
        return eur_bal + total_val

    def _can_open_new_position(self)->bool:
        tot_eq = self._get_equity_estimate()
        bal = self._get_eur_balance()
        invested = tot_eq - bal
        ratio = invested / tot_eq if tot_eq > 0 else Decimal("0")
        self.logger.debug(f"[AltcoinScanner] _can_open_new_position => ratio={ratio:.2f}, max={self.max_positions_equity_pct}")
        return ratio < self.max_positions_equity_pct

    # [NEW] Methode om intra-candle exits te checken voor ALLE open posities
    #       (Elke 5-10s vanuit de executor oproepen.)
    def manage_intra_candle_exits(self):
        """
        [NEW] Aanroepen vanuit je executor-loop (bv. elke 5s),
        om SL/TP semi-live te checken voor alle altcoin-scanner posities.
        """
        for sym in list(self.open_positions.keys()):
            pos = self.open_positions[sym]
            # Haal 'live' price
            curr_price = self._get_latest_price(sym)
            if curr_price > 0:
                # We hergebruiken _manage_position
                self._manage_position(sym)
