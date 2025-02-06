# ============================================================
# src/my_websocket/client.py (Bitvavo-only variant + REST-poll)
# ============================================================

import queue
import pandas as pd
import websocket
import json
import hmac
import hashlib
import time
import threading
from decimal import Decimal
from datetime import datetime, timedelta, timezone

from src.logger.logger import setup_websocket_logger, log_error
from src.indicator_analysis.indicators import IndicatorAnalysis
from src.config.config import WEBSOCKET_LOG_FILE, PULLBACK_CONFIG, yaml_config
from python_bitvavo_api.bitvavo import Bitvavo
import requests
import logging

###############################################################################
# Globale logger-instance: TimedRotatingFileHandler
###############################################################################
main_logger = setup_websocket_logger(
    log_file=WEBSOCKET_LOG_FILE,
    level=10,          # 10 = logging.DEBUG
    when="midnight",   # roteer 1x per dag
    interval=1,
    backup_count=5,
    use_json=False
)
main_logger.info("WebSocket-client gestart.")


def safe_get(url, params=None, max_retries=3, sleep_seconds=1):
    """
    Voer een GET-request uit met retries bij ConnectionError.
    """
    attempts = 0
    while attempts < max_retries:
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError as ce:
            main_logger.warning(f"[safe_get] ConnectionError => {ce}, retry {attempts+1}/{max_retries}...")
            time.sleep(sleep_seconds)
            attempts += 1
        except requests.exceptions.HTTPError as he:
            main_logger.warning(f"[safe_get] HTTPError => {he}.")
            return None
        except Exception as ex:
            main_logger.error(f"[safe_get] Onverwachte fout => {ex}")
            return None
    main_logger.error(f"[safe_get] Max retries={max_retries} overschreden => None.")
    return None


def interval_str_to_minutes(interval_str: str) -> int:
    """Zet een interval-string (zoals '1m', '5m', '1h', '4h', '1d') om in aantal minuten."""
    if interval_str.endswith('m'):
        return int(interval_str[:-1])
    elif interval_str.endswith('h'):
        return int(interval_str[:-1]) * 60
    elif interval_str.endswith('d'):
        return int(interval_str[:-1]) * 1440
    else:
        # Fallback: probeer als integer te interpreteren
        try:
            return int(interval_str)
        except ValueError:
            return 0


def is_candle_closed(candle_timestamp_ms: int, interval_str: str) -> bool:
    """
    Bepaalt of een candle volledig is afgesloten.
    """
    interval_minutes = interval_str_to_minutes(interval_str)
    if interval_minutes <= 0:
        # Onbekend interval => ga uit van gesloten
        return True
    candle_start = datetime.fromtimestamp(candle_timestamp_ms / 1000, tz=timezone.utc)
    candle_end = candle_start + timedelta(minutes=interval_minutes)
    return datetime.now(timezone.utc) >= candle_end


###############################################################################
# WebSocketClient
###############################################################################
class WebSocketClient:
    def __init__(self, ws_url, db_manager, api_key, api_secret):
        """
        Client voor Bitvavo WebSocket + (nu uitgebreid met) REST-candle-poll.
        """
        self.ws_url = ws_url
        self.db_manager = db_manager
        self.api_key = api_key
        self.api_secret = api_secret

        self.logger = main_logger  # TimeRotatingFileHandler logger
        self.logger.debug(f"[INIT] api_key='{api_key}' (len={len(api_key)})")
        self.logger.debug(f"[INIT] api_secret='{api_secret}' (len={len(api_secret)})")

        # Laad bitvavo-config, bv. "pairs": ["BTC-EUR","ETH-EUR"], "intervals": ["5m", "15m"] etc.
        bitvavo_cfg = yaml_config.get("bitvavo", {})
        self.markets = bitvavo_cfg.get("pairs", [])
        # Welke intervals wil je periodiek pollen?
        self.poll_intervals = bitvavo_cfg.get("poll_intervals", ["5m", "15m"])  # Pas aan naar wens, bijv ["1m","5m","15m"]

        self.logger.info(f"[WebSocketClient] Using dynamic markets: {self.markets}")
        self.logger.info(f"[WebSocketClient] Poll intervals: {self.poll_intervals}")

        # Maak Bitvavo-instance (REST + WS)
        self.bitvavo = Bitvavo({
            'APIKEY': self.api_key,
            'APISECRET': self.api_secret,
            'RESTURL': 'https://api.bitvavo.com/v2',
            'WSURL': 'wss://ws.bitvavo.com/v2/',
            'ACCESSWINDOW': 10000
        })

        # Houd WS-prijzen bij + timestamps
        self.latest_prices = {}
        self.latest_update_timestamps = {}

        # WebSocket-attributen
        self.ws = None
        self.order_updates_queue = queue.Queue()

        # Rate-limit attributen
        self.calls_this_minute = 0
        self.last_reset_ts = time.time()
        self.limit_per_minute = 100

        self.subscribed = False
        self.running = False
        self._thread = None

        # Lock + cache voor REST fallback
        self.fallback_lock = threading.Lock()
        self.fallback_price_cache = {}

        # Stop-event voor poll-thread
        self._poll_stop_event = threading.Event()
        self._poll_thread = None


    # -------------------------------------------------------------
    # Rate-limit
    # -------------------------------------------------------------
    def _check_rate_limit(self):
        now = time.time()
        if now - self.last_reset_ts >= 60:
            self.calls_this_minute = 0
            self.last_reset_ts = now
        if self.calls_this_minute >= self.limit_per_minute:
            self.logger.warning("REST rate limit dreigt overschreden te worden. Slaap 5s...")
            time.sleep(5)

    def _increment_call(self):
        self.calls_this_minute += 1

    # -------------------------------------------------------------
    # REST-calls (PAPER-mode)
    # -------------------------------------------------------------
    def get_balance(self):
        """
        In paper-mode doen we geen echte call, maar simuleren we.
        """
        self._check_rate_limit()
        self._increment_call()
        # Fake / paper logic
        yaml_budget = PULLBACK_CONFIG.get("initial_capital", 100)
        budget_decimal = Decimal(str(yaml_budget))
        self.logger.info(f"Simuleer get_balance(): return {{'EUR': {budget_decimal}}}")
        return {"EUR": budget_decimal}

    def place_order(self, side, symbol, amount, _order_type=None):
        """
        LET OP: In 'paper' mode doen we hier géén echte call naar Bitvavo.
        """
        self._check_rate_limit()
        self._increment_call()
        self.logger.info(f"Simuleer place_order({side}, {symbol}, {amount}) (Geen echte call)")
        # Geen echte order-executie in paper mode

    # -------------------------------------------------------------
    # REST-fallback (prijs opvragen) met safe_get
    # -------------------------------------------------------------
    def _fetch_rest_price(self, symbol: str) -> Decimal:
        """
        Vraagt de tickerPrice op via GET en parse 'price' als Decimal.
        Gebruikt safe_get() met retries.
        """
        with self.fallback_lock:
            now_ts = time.time()
            cached_entry = self.fallback_price_cache.get(symbol, None)
            if cached_entry is not None:
                if not isinstance(cached_entry, tuple) or len(cached_entry) != 2:
                    self.logger.error(f"[BUG] fallback_price_cache[{symbol}] bevat: {cached_entry}")
                    del self.fallback_price_cache[symbol]
                    return Decimal("0")
                (cached_price, cached_ts) = cached_entry
                if (now_ts - cached_ts) < 2:
                    # Gebruik cache
                    self.logger.debug(f"[REST-Fallback] Gebruik cache voor {symbol} => {cached_price}")
                    return cached_price

            self._check_rate_limit()
            self._increment_call()

            # safe_get ipv direct requests
            url = f"https://api.bitvavo.com/v2/ticker/price?market={symbol}"
            resp = safe_get(url, max_retries=3, sleep_seconds=2)
            if not resp:
                self.logger.error(f"[REST-Fallback] Kan {symbol} niet ophalen na 3 retries.")
                return Decimal("0")

            try:
                data = resp.json()
            except Exception as e:
                self.logger.error(f"[REST-Fallback] JSON decode error => {e}")
                return Decimal("0")

            self.logger.debug(f"[REST-Fallback RAW] {symbol} => {data}")

            if isinstance(data, dict) and "error" in data:
                self.logger.error(f"[REST-Fallback] Bitvavo error: {data}")
                return Decimal("0")

            elif isinstance(data, dict) and "price" in data:
                rest_str_price = data["price"]
                rest_price = Decimal(str(rest_str_price))
                self.fallback_price_cache[symbol] = (rest_price, now_ts)
                self.logger.debug(f"[REST-Fallback] {symbol} => {rest_price} (single dict from REST)")
                return rest_price

            elif isinstance(data, list) and len(data) > 0 and "price" in data[0]:
                rest_str_price = data[0]["price"]
                rest_price = Decimal(str(rest_str_price))
                self.fallback_price_cache[symbol] = (rest_price, now_ts)
                self.logger.debug(f"[REST-Fallback] {symbol} => {rest_price} (list of tickers)")
                return rest_price
            else:
                self.logger.warning(f"[REST-Fallback] Onverwachte return tickerPrice({symbol}): {data}")
                return Decimal("0")

    # -------------------------------------------------------------
    # get_price_with_fallback
    # -------------------------------------------------------------
    def get_price_with_fallback(self, symbol: str, max_age=10) -> Decimal:
        """
        Check WS-price age, anders fallback.
        """
        now_ts = time.time()
        last_price = self.latest_prices.get(symbol, 0.0)
        last_ts = self.latest_update_timestamps.get(symbol, 0.0)
        age = now_ts - last_ts

        if age <= max_age and last_price > 0:
            return Decimal(str(last_price))

        self.logger.warning(f"[WebSocketClient] {symbol} => WS price is {age:.1f}s oud => fallback to REST")
        rest_price = self._fetch_rest_price(symbol)

        # Na fallback => update timestamp
        if rest_price > 0:
            self.latest_prices[symbol] = float(rest_price)
            self.latest_update_timestamps[symbol] = now_ts

        return rest_price if rest_price > 0 else Decimal("0")

    # -------------------------------------------------------------
    # WebSocket signing
    # -------------------------------------------------------------
    def generate_signature(self, timestamp, method, endpoint):
        """
        HMAC-SHA256 over f"{timestamp}{method}{endpoint}", met secret als utf-8
        """
        message = f"{timestamp}{method}{endpoint}".encode('utf-8')
        secret_bytes = self.api_secret.encode('utf-8')
        raw_hmac = hmac.new(secret_bytes, message, hashlib.sha256)
        return raw_hmac.hexdigest()

    # -------------------------------------------------------------
    # WebSocket Handlers
    # -------------------------------------------------------------
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            self.logger.error(f"Ongeldig JSON-bericht: {message}. Fout: {e}")
            return

        if data.get("event") == "pong":
            self.logger.info("JSON-pong ontvangen!")
            return

        self.logger.info(f"Ontvangen bericht: {data}")

        # Auth bevestigd?
        if data.get("event") == "authenticate" and data.get("authenticated") is True:
            self.logger.info("Authenticatie succesvol.")
            self.subscribe_to_channels(ws)
            return

        event_type = data.get("event")

        # Order/fill
        if event_type == "order":
            self.logger.info("Eigen order geüpdatet => in queue")
            self.order_updates_queue.put(data)
            return
        elif event_type == "fill":
            self.logger.info("Eigen order fill => in queue")
            self.order_updates_queue.put(data)
            return

        # Publieke + account data
        if event_type == "trade":
            self.process_trade_data(data)
        elif event_type == "candle":
            self.process_candle_data(data)
        elif event_type == "ticker":
            self.process_ticker_data(data)
        elif event_type == "book":
            self.process_orderbook_data(data)
        elif event_type == "account":
            self.logger.info(f"Account-update: {data}")
        else:
            self.logger.debug(f"Onbekend event_type: {event_type}, data={data}")

    def subscribe_to_channels(self, ws):
        if self.subscribed:
            self.logger.warning("Al gesubscribed, skip.")
            return
        self.subscribed = True

        # Stel de 'markets' in en abonneer
        subscribe_payload = {
            "action": "subscribe",
            "channels": [
                {
                    "name": "candles",
                    "interval": ["1m", "5m", "15m", "1h", "4h", "1d"],
                    "markets": self.markets
                },
                {"name": "trades", "markets": self.markets},
                {"name": "ticker", "markets": self.markets},
                {"name": "book", "markets": self.markets},
                {"name": "account", "markets": self.markets}
            ]
        }

        self.logger.info(f"Verzenden abonnementspayload: {subscribe_payload}")
        ws.send(json.dumps(subscribe_payload))
        self.logger.info("Abonnementen verzonden na authenticatie.")

    # -------------------------------------------------------------
    # process_* - datahandlers
    # -------------------------------------------------------------
    def process_trade_data(self, data):
        """
        Verwerkt een 'trade'-event van Bitvavo.
          - Soms stuurt Bitvavo een 'trades' array (oude of andere beurzen-stijl).
          - Soms stuurt Bitvavo direct { 'price','amount','timestamp','side' } etc. zonder 'trades' array.

        Voor beide gevallen updaten we self.latest_prices en self.latest_update_timestamps,
        en loggen de 'laatste' of 'single' trade.
        """
        market_symbol = data.get("market")
        if not market_symbol:
            self.logger.debug("[Trades] Geen market_symbol in trade-event.")
            return

        # 1) Check of er een 'trades' array in data zit
        if "trades" in data:
            trades_array = data.get("trades", [])
            if not trades_array:
                self.logger.debug(f"[Trades] wel event maar geen trades[] voor {market_symbol}.")
                return

            last_trade = trades_array[-1]
            now_ts = time.time()

            price_str = last_trade.get("price", "0")
            try:
                price_float = float(price_str)
            except ValueError:
                price_float = 0.0

            if price_float > 0:
                self.latest_prices[market_symbol] = price_float
                self.latest_update_timestamps[market_symbol] = now_ts
                self.logger.debug(
                    f"[Trades] {market_symbol} => last trade price={price_float}, total trades={len(trades_array)}"
                )
            else:
                self.logger.debug(f"[Trades] {market_symbol} => ongeldige price in {last_trade}")

        else:
            # 2) Bitvavo stuurt soms 'amount','price','timestamp','side' direct in data
            now_ts = time.time()

            price_str = data.get("price", "0")
            try:
                price_float = float(price_str)
            except ValueError:
                price_float = 0.0

            if price_float > 0:
                self.latest_prices[market_symbol] = price_float
                self.latest_update_timestamps[market_symbol] = now_ts
                self.logger.debug(
                    f"[Trades] {market_symbol} => single trade price={price_float}, "
                    f"id={data.get('id', '')}, side={data.get('side', 'unknown')}"
                )
            else:
                self.logger.debug(f"[Trades] {market_symbol} => ongeldige price in single trade => {data}")

    def process_candle_data(self, data):
        """
        AANPASSING: i.p.v. opslaan in "candles" => "candles_bitvavo".
                    i.p.v. save_indicators(...) => save_indicators_bitvavo(...).
        """
        try:
            market = data.get("market")
            interval = data.get("interval")
            candles = data.get("candle", [])
            self.logger.info(f"Binnenkomende candle: {market}, {interval}, count={len(candles)}")

            if not candles:
                self.logger.debug(f"Geen candles voor {market} - {interval}.")
                return

            self.logger.debug(f"Raw candles data: {candles}")

            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            df_with_indicators = IndicatorAnalysis.calculate_indicators(df)
            self.logger.info(f"df_with_indicators shape: {df_with_indicators.shape}")

            for _, row in df_with_indicators.iterrows():
                # Bepaal ts_ms
                ts_ms = int(row["timestamp"].timestamp() * 1000)
                # Check of candle closed
                if not is_candle_closed(ts_ms, interval):
                    self.logger.debug(f"Candle voor {market} ts={ts_ms}, interval={interval} nog niet gesloten.")
                    continue

                formatted_candle = (
                    ts_ms,
                    market,
                    interval,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"])
                )
                self.db_manager.save_candles_bitvavo([formatted_candle])

                self.logger.info(
                    f"DEBUG Candle => ts={formatted_candle[0]}, market={formatted_candle[1]}, "
                    f"interval={formatted_candle[2]}, open={formatted_candle[3]}, close={formatted_candle[6]}"
                )

                df_test = self.db_manager.fetch_data("candles_bitvavo", limit=10, market=market, interval=interval)
                self.logger.info(f"Na insert -> fetch {market}, {interval}, got {len(df_test)} records")

            # Indicators opslaan
            df_with_indicators["market"] = market
            df_with_indicators["interval"] = interval
            self.db_manager.save_indicators_bitvavo(df_with_indicators)

            self.logger.info(f"{len(df_with_indicators)} candles verwerkt voor {market}-{interval}.")

        except Exception as e:
            self.logger.error(f"Fout bij verwerken van candle-data: {e}")

    # -------------------------------------------------------------
    # process_ticker_data
    # -------------------------------------------------------------
    def process_ticker_data(self, data):
        """
        AANPASSING: i.p.v. save_ticker(...) => save_ticker_bitvavo(...).
        """
        try:
            self.logger.info(f"Ontvangen ticker-data: {data}")
            best_bid = float(data.get("bestBid", 0))
            best_ask = float(data.get("bestAsk", 0))
            spread = best_ask - best_bid if best_bid and best_ask else 0.0

            ticker_data = {
                'market': data.get("market"),
                'bestBid': best_bid,
                'bestAsk': best_ask,
                'spread': spread,
                'exchange': 'Bitvavo'
            }
            self.db_manager.save_ticker_bitvavo(ticker_data)
            self.logger.info(f"Ticker verwerkt (DB): {ticker_data}")

        except Exception as e:
            log_error(self.logger, f"Fout bij verwerken van ticker-data: {e}")

    def process_orderbook_data(self, data):
        """
        AANPASSING: i.p.v. save_orderbook(...) => save_orderbook_bitvavo(...).
        """
        try:
            orderbook_data = {
                'market': data.get("market"),
                'bids': data.get("bids", []),
                'asks': data.get("asks", []),
                'exchange': 'Bitvavo'
            }
            self.logger.info(f"Orderbook ontvangen: {orderbook_data}")
            self.db_manager.save_orderbook_bitvavo(orderbook_data)
        except Exception as e:
            self.logger.error(f"Fout bij verwerken van orderbook-data: {e}")

    # -------------------------------------------------------------
    # START/STOP
    # -------------------------------------------------------------
    def start_websocket(self):
        """
        Start de WebSocket-client in een aparte thread. Als hij verbroken wordt,
        herstart hij na 5s (tenzij self.running=False).
        """
        self.running = True

        def _run_forever():
            while self.running:
                try:
                    self.ws = websocket.WebSocketApp(
                        self.ws_url,
                        on_open=self.on_open,
                        on_message=self.on_message,
                        on_error=self.on_error,
                        on_close=self.on_close
                    )
                    self.ws.run_forever(ping_interval=30, ping_timeout=10)
                except Exception as e:
                    self.logger.error(f"Fout in _run_forever-lus: {e}")

                if self.running:
                    self.logger.warning("WebSocket verbroken. Herstart over 5s...")
                    time.sleep(5)
                else:
                    self.logger.info("self.running=False => definitief stoppen.")
                    break

            self.logger.info("Einde _run_forever-lus => thread stopt.")

        self._thread = threading.Thread(target=_run_forever, daemon=True)
        self._thread.start()
        self.logger.info("WebSocket client is gestart in aparte thread.")

    def on_open(self, ws):
        self.logger.info("WebSocket on_open triggered.")
        self.logger.info(f"[DEBUG] Key={self.api_key}, len={len(self.api_key)}")
        self.logger.info(f"[DEBUG] Secret start= {self.api_secret[:10]}..., len={len(self.api_secret)}")

        # Authenticatie
        timestamp = int(time.time() * 1000)
        method = "GET"
        endpoint = "/v2/websocket"
        sig_value = self.generate_signature(timestamp, method, endpoint)
        self.logger.debug(f"[AUTH-PAYLOAD] signature={sig_value}, len={len(sig_value)}")
        auth_payload = {
            "action": "authenticate",
            "key": self.api_key,
            "signature": sig_value,
            "timestamp": timestamp,
            "window": 10000
        }
        self.logger.debug(f"[AUTH-PAYLOAD] auth_payload={auth_payload}")
        ws.send(json.dumps(auth_payload))
        self.logger.info("Authenticatie verzonden.")
        self.subscribed = False

    @staticmethod
    def on_error(_ws, error):
        main_logger.error(f"WebSocket Error: {error}")
        try:
            _ws.close()
        except Exception as e:
            main_logger.warning(f"Fout bij sluiten in on_error: {e}")

    @staticmethod
    def on_close(_ws, close_status_code, close_msg):
        main_logger.warning(f"WebSocket-verbinding gesloten. status_code={close_status_code}, msg={close_msg}")

    def stop_websocket(self):
        """
        Stop de WebSocket-loop, poll-thread en alle lopende threads.
        """
        self.logger.info("Stop websocket aangevraagd.")
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                self.logger.error(f"Fout bij sluiten in stop_websocket: {e}")

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        # Ook de poll-thread stoppen
        self.logger.info("Stop poll-thread aangevraagd.")
        self._poll_stop_event.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=5)

        self.logger.info("WebSocket client is volledig gestopt.")

    # -------------------------------------------------------------
    # ORDER/FILL updates
    # -------------------------------------------------------------
    def handle_order_update(self, data: dict):
        try:
            self.logger.info(f"[_handle_order_update] {data}")
            order_id = data.get("orderId")
            market = data.get("market")
            side = data.get("side")
            status = data.get("status", "unknown")
            amount = data.get("amount", "0")
            filled_amount = data.get("filledAmount", "0")
            price = data.get("price", "0")

            order_row = {
                "order_id": order_id,
                "market": market,
                "side": side,
                "status": status,
                "amount": float(amount),
                "filled_amount": float(filled_amount),
                "price": float(price),
                "timestamp": int(time.time() * 1000)
            }
            # (Je kunt bv. self.db_manager.save_order_bitvavo(order_row) gebruiken
            #  als je aparte bitvavo-order table hebt.)
            self.db_manager.save_order(order_row)
            self.logger.info(
                f"[_handle_order_update] order {order_id} => status={status}, filled={filled_amount}/{amount}"
            )
        except Exception as e:
            self.logger.error(f"Fout in _handle_order_update: {e}")

    def handle_fill_update(self, data: dict):
        try:
            self.logger.info(f"[_handle_fill_update] {data}")

            order_id = data.get("orderId")
            market = data.get("market")
            side = data.get("side")
            fill_amount = data.get("amount", "0")
            fill_price = data.get("price", "0")
            fee_amount = data.get("fee", "0")

            fill_row = {
                "order_id": order_id,
                "market": market,
                "side": side,
                "fill_amount": float(fill_amount),
                "fill_price": float(fill_price),
                "fee_amount": float(fee_amount),
                "timestamp": int(time.time() * 1000)
            }
            # of self.db_manager.save_fill_bitvavo(fill_row) als aparte 'fills_bitvavo'
            self.db_manager.save_fill(fill_row)
            self.logger.info(
                f"[_handle_fill_update] fill => amt={fill_amount} @ price={fill_price}, fee={fee_amount}"
            )
        except Exception as e:
            self.logger.error(f"Fout in _handle_fill_update: {e}")


    # -------------------------------------------------------------
    # Nieuw: periodieke REST-candle-poll
    # -------------------------------------------------------------
    def poll_rest_candles(self, market: str, interval: str) -> bool:
        """
        Haalt via de Bitvavo REST '/candles/{market}/{interval}' endpoint
        de nieuwste candles op, en slaat deze in de DB op.
        Retourneert True als er (tenminste 1) nieuwe candle is toegevoegd,
        anders False.
        """
        self._check_rate_limit()
        self._increment_call()

        # Stel de querystring samen. Je kunt ook 'start' en 'end' meegeven
        # als je wilt filteren. Hieronder enkel 'limit' (max 1000 candles).
        url = f"{self.bitvavo.RESTURL}/{market}/candles"
        params = {
            "interval": interval,
            "limit": 20  # bijvoorbeeld de laatste 20 candles
        }
        resp = safe_get(url, params=params, max_retries=3, sleep_seconds=2)
        if not resp:
            self.logger.warning(f"[poll_rest_candles] Kan geen data ophalen voor {market} - {interval}.")
            return False

        try:
            candle_data = resp.json()
        except Exception as e:
            self.logger.error(f"[poll_rest_candles] JSON-error => {e}")
            return False

        if not isinstance(candle_data, list):
            self.logger.warning(f"[poll_rest_candles] Onverwachte response (geen lijst): {candle_data}")
            return False

        # Parse en verwerk net zoals in process_candle_data
        df = pd.DataFrame(
            candle_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df_ind = IndicatorAnalysis.calculate_indicators(df)

        new_candles = 0
        for _, row in df_ind.iterrows():
            ts_ms = int(row["timestamp"].timestamp() * 1000)
            if not is_candle_closed(ts_ms, interval):
                continue

            formatted = (
                ts_ms,
                market,
                interval,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"])
            )
            self.db_manager.save_candles_bitvavo([formatted])
            new_candles += 1

        if new_candles > 0:
            df_ind["market"] = market
            df_ind["interval"] = interval
            self.db_manager.save_indicators_bitvavo(df_ind)

        self.logger.info(f"[poll_rest_candles] {market}-{interval}: {new_candles} candles opgeslagen (REST).")
        return (new_candles > 0)

    def _poll_around_boundary(self, interval="5m", max_attempts=3, sleep_sec=10):
        """
        Korte cycli poll-logic om 'vers afgesloten' 5m-candle
        zo snel mogelijk binnen te halen.
        """
        # We proberen enkele keren; als de candle vertraagd is, komt-ie dan meestal alsnog
        for attempt in range(1, max_attempts + 1):
            if not self.running or self._poll_stop_event.is_set():
                return
            success = False
            for m in self.markets:
                # Als er tenminste 1 candle is toegevoegd
                # (denk aan de net afgesloten candle), zien we dat als succes.
                got_new = self.poll_rest_candles(m, interval)
                if got_new:
                    success = True
            if success:
                self.logger.info(f"[_poll_around_boundary] {interval} candle(s) ontvangen na {attempt} attempts.")
                break
            else:
                self.logger.info(
                    f"[_poll_around_boundary] geen verse candle(s) attempt={attempt}/{max_attempts}, sleep {sleep_sec}s..."
                )
                time.sleep(sleep_sec)

    def _poll_loop(self):
        """
        Hoofdloop die elke X minuten bij de timeframe-boundary de 5m (of andere) candles
        kortcyclisch opvraagt.
        """
        while not self._poll_stop_event.is_set() and self.running:
            now = datetime.utcnow()
            # Zoek de eerstvolgende 5-minutenboundary. We pakken alleen
            # intervals uit self.poll_intervals, maar als '5m' er tussen staat
            # kun je deze logica gebruiken. Voor '15m' is het vergelijkbaar.
            # Hier illustreren we het met '5m'; je kunt per interval werken
            # in 1 loop, of aparte planningen per interval maken.

            # We checken voor elk interval in self.poll_intervals.
            # Voor elk interval ronden we de tijd op die boundary.
            for interval in self.poll_intervals:
                boundary_dt = self._round_up_to_next_interval(now, interval, offset_sec=3)
                sleep_seconds = (boundary_dt - datetime.utcnow()).total_seconds()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

                if self._poll_stop_event.is_set() or (not self.running):
                    return

                # Als we hier zijn, is het net NA de boundary => poll in korte cycli
                self._poll_around_boundary(interval, max_attempts=3, sleep_sec=10)

            # Als je meerdere intervals na elkaar doet, dan ga je hier door.
            # Eventueel kun je daarna weer 'time.sleep(...)' tot de volgende boundary,
            # maar in dit voorbeeld ronden we elke poll_intervals-lus weer af
            # en berekenen we de boundary opnieuw in de volgende iteratie.

    def _round_up_to_next_interval(self, dt: datetime, interval_str: str, offset_sec=3) -> datetime:
        """
        Rond 'dt' af naar de eerstvolgende 'interval_str' boundary,
        bv. 5m, 15m, etc. en voeg 'offset_sec' toe als marge.
        """
        n_minutes = interval_str_to_minutes(interval_str)
        if n_minutes <= 0:
            # Fallback, direct
            return dt + timedelta(seconds=offset_sec)

        # Stel, dt=14:31:12, n_minutes=5 => volgende boundary=14:35:00 (UTC)
        # daarna + offset_sec => 14:35:03
        dt_zero_sec = dt.replace(second=0, microsecond=0)
        minute_mod = dt_zero_sec.minute % n_minutes
        if minute_mod == 0:
            # We zitten al op een boundary => pak deze + offset
            boundary = dt_zero_sec
        else:
            to_add = n_minutes - minute_mod
            boundary = dt_zero_sec + timedelta(minutes=to_add)
        return boundary + timedelta(seconds=offset_sec)

    def start_periodic_candle_polling(self):
        """
        Start de thread die in _poll_loop regelmatig de verse candles ophaalt
        (eventueel in korte cycli rond de boundary).
        """
        def _poll_thread_fn():
            self.logger.info("[start_periodic_candle_polling] poll-loop gestart.")
            self._poll_loop()
            self.logger.info("[start_periodic_candle_polling] poll-loop gestopt.")

        self._poll_stop_event.clear()
        self._poll_thread = threading.Thread(target=_poll_thread_fn, daemon=True)
        self._poll_thread.start()
        self.logger.info("[start_periodic_candle_polling] poll-thread is gestart.")
