import queue
import pandas as pd
import websocket
import json
import hmac
import hashlib
import time
import threading
from decimal import Decimal

from src.logger.logger import setup_logger, log_error
from src.indicator_analysis.indicators import IndicatorAnalysis
from src.config.config import WEBSOCKET_LOG_FILE, PAIRS_CONFIG, PULLBACK_CONFIG
from python_bitvavo_api.bitvavo import Bitvavo

# Globale logger-instance
main_logger = setup_logger('websocket_client', WEBSOCKET_LOG_FILE)
main_logger.info("WebSocket-client gestart.")


class WebSocketClient:
    def __init__(self, ws_url, db_manager, api_key, api_secret):
        ...

        self.logger = main_logger  # bijvoorbeeld
        self.logger.debug(f"[INIT] api_key='{api_key}' (len={len(api_key)})")
        self.logger.debug(f"[INIT] api_secret='{api_secret}' (len={len(api_secret)})")
        ...

        """
        WebSocketClient - Beheert WebSocket-verbindingen en interactie met Bitvavo.
        """
        self.ws_url = ws_url
        self.db_manager = db_manager
        self.api_key = api_key
        self.api_secret = api_secret

        # Interne logger voor deze class (verwijst naar dezelfde file)
        self.logger = main_logger

        # Initialiseer Bitvavo (REST + WS)
        self.bitvavo = Bitvavo({
            'APIKEY': self.api_key,
            'APISECRET': self.api_secret,
            'RESTURL': 'https://api.bitvavo.com/v2',
            'WSURL': 'wss://ws.bitvavo.com/v2/',
            'ACCESSWINDOW': 10000
        })

        self.latest_prices = {}
        self.latest_update_timestamps = {}

        self.ws = None

        self.order_updates_queue = queue.Queue()

        self.calls_this_minute = 0
        self.last_reset_ts = time.time()
        self.limit_per_minute = 100

        self.subscribed = False
        self.running = False
        self._thread = None

        self.markets = PAIRS_CONFIG
        self.logger.info(f"[WebSocketClient] Using dynamic markets: {self.markets}")

        # Lock + cache voor REST fallback
        self.fallback_lock = threading.Lock()
        self.fallback_price_cache = {}

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
    # REST-calls (voor PAPER)
    # -------------------------------------------------------------
    def get_balance(self):
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
        # Dus geen bitvavo.placeOrder(...)

    # -------------------------------------------------------------
    # REST-fallback (prijs opvragen)
    # -------------------------------------------------------------
    def _fetch_rest_price(self, symbol: str) -> Decimal:
        with self.fallback_lock:
            now_ts = time.time()
            cached_entry = self.fallback_price_cache.get(symbol, None)
            if cached_entry is not None:
                if not isinstance(cached_entry, tuple) or len(cached_entry) != 2:
                    self.logger.error(
                        f"[BUG] fallback_price_cache[{symbol}] bevat: {cached_entry} "
                        f"(type={type(cached_entry)})"
                    )
                    del self.fallback_price_cache[symbol]
                    return Decimal("0")

                (cached_price, cached_ts) = cached_entry
                if (now_ts - cached_ts) < 2:
                    self.logger.debug(f"[REST-Fallback] Gebruik cache voor {symbol} => {cached_price}")
                    return cached_price

            self._check_rate_limit()
            self._increment_call()

            tickers = self.bitvavo.tickerPrice({"market": symbol})
            self.logger.debug(f"[REST-Fallback RAW] {symbol} => {tickers}")

            if isinstance(tickers, dict) and "error" in tickers:
                self.logger.error(f"[REST-Fallback] Bitvavo error: {tickers}")
                return Decimal("0")

            elif isinstance(tickers, dict) and "price" in tickers:
                rest_str_price = tickers["price"]
                rest_price = Decimal(str(rest_str_price))
                self.fallback_price_cache[symbol] = (rest_price, now_ts)
                self.logger.debug(f"[REST-Fallback] {symbol} => {rest_price} (single dict from REST)")
                return rest_price

            elif isinstance(tickers, list) and len(tickers) > 0 and "price" in tickers[0]:
                rest_str_price = tickers[0]["price"]
                rest_price = Decimal(str(rest_str_price))
                self.fallback_price_cache[symbol] = (rest_price, now_ts)
                self.logger.debug(f"[REST-Fallback] {symbol} => {rest_price} (list of tickers)")
                return rest_price
            else:
                self.logger.warning(f"[REST-Fallback] Onverwachte return tickerPrice({symbol}): {tickers}")
                return Decimal("0")

    # -------------------------------------------------------------
    # get_price_with_fallback
    # -------------------------------------------------------------
    def get_price_with_fallback(self, symbol: str, max_age=10) -> Decimal:
        now_ts = time.time()
        last_price = self.latest_prices.get(symbol, 0.0)
        last_ts = self.latest_update_timestamps.get(symbol, 0.0)
        age = now_ts - last_ts

        if age <= max_age and last_price > 0:
            return Decimal(str(last_price))

        self.logger.warning(f"[WebSocketClient] {symbol} => WS price is {age:.1f}s oud => fallback to REST")
        rest_price = self._fetch_rest_price(symbol)
        return rest_price if rest_price > 0 else Decimal("0")

    # -------------------------------------------------------------
    # WebSocket
    # -------------------------------------------------------------
    def generate_signature(self, timestamp, method, endpoint):
        """
        Generate the HMAC-SHA256 signature needed for Bitvavo WS/REST auth.
        Met 'bytes.fromhex(...)' omdat de secret in hex is.
        """
        # 1) Bouw de sign-string
        message = f"{timestamp}{method}{endpoint}".encode('utf-8')

        # 2) Zet je 128-char hexsecret om in ruwe bytes
        secret_bytes = bytes.fromhex(self.api_secret)

        # 3) HMAC-SHA256
        raw_hmac = hmac.new(secret_bytes, message, hashlib.sha256)
        return raw_hmac.hexdigest()  # hex-string als resultaat

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
            self.subscribe_to_channels(ws)  # direct ALLE channels
            return

        event_type = data.get("event")

        # Order/fill events
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

        subscribe_payload = {
            "action": "subscribe",
            "channels": [
                {
                    "name": "candles",
                    "interval": ["1m", "5m", "15m", "1h", "4h", "1d"],
                    "markets": self.markets
                },
                {
                    "name": "trades",
                    "markets": self.markets
                },
                {
                    "name": "ticker",
                    "markets": self.markets
                },
                {
                    "name": "book",
                    "markets": self.markets
                },
                {
                    "name": "account",
                    "markets": self.markets
                }
            ]
        }

        self.logger.info(f"Verzenden abonnementspayload: {subscribe_payload}")
        ws.send(json.dumps(subscribe_payload))
        self.logger.info("Abonnementen verzonden na authenticatie.")

    # -------------------------------------------------------------
    # process_* - datahandlers
    # -------------------------------------------------------------
    def process_trade_data(self, data):
        market_symbol = data.get("market")
        trades_array = data.get("trades", [])
        if not market_symbol:
            self.logger.debug("[Trades] Geen market_symbol in trade-event.")
            return
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
                f"[Trades] {market_symbol} => last trade price={price_float}, "
                f"total trades in event={len(trades_array)}"
            )
        else:
            self.logger.debug(
                f"[Trades] {market_symbol} => ongeldige price in {last_trade}"
            )

    def process_candle_data(self, data):
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
                formatted_candle = (
                    int(row["timestamp"].timestamp() * 1000),
                    market,
                    interval,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"])
                )
                self.logger.info(
                    f"DEBUG Candle => ts={formatted_candle[0]}, "
                    f"market={formatted_candle[1]}, interval={formatted_candle[2]}, "
                    f"open={formatted_candle[3]}, close={formatted_candle[6]}"
                )
                self.db_manager.save_candles([formatted_candle])
                df_test = self.db_manager.fetch_data("candles", limit=10, market=market, interval=interval)
                self.logger.info(f"Na insert -> fetch {market}, {interval}, got {len(df_test)} records")

            df_with_indicators["market"] = market
            df_with_indicators["interval"] = interval
            self.db_manager.save_indicators(df_with_indicators)

            self.logger.info(
                f"{len(df_with_indicators)} candles verwerkt voor {market}-{interval}. "
                f"Voorbeeld: {df_with_indicators.head(1).to_dict('records')}"
            )
        except Exception as e:
            self.logger.error(f"Fout bij verwerken van candle-data: {e}")

    # -------------------------------------------------------------
    # process_ticker_data
    # -------------------------------------------------------------
    def process_ticker_data(self, data):
        try:
            self.logger.info(f"Ontvangen ticker-data: {data}")

            best_bid = float(data.get("bestBid", 0))
            best_ask = float(data.get("bestAsk", 0))
            spread = best_ask - best_bid if best_bid and best_ask else 0.0

            ticker_data = {
                'market': data.get("market"),
                'bestBid': best_bid,
                'bestAsk': best_ask,
                'spread': spread
            }
            self.db_manager.save_ticker(ticker_data)
            self.logger.info(f"Ticker verwerkt (DB): {ticker_data}")

        except Exception as e:
            log_error(self.logger, f"Fout bij verwerken van ticker-data: {e}")

    # -------------------------------------------------------------
    # process_orderbook_data
    # -------------------------------------------------------------
    def process_orderbook_data(self, data):
        try:
            orderbook_data = {
                'market': data.get("market"),
                'bids': data.get("bids", []),
                'asks': data.get("asks", [])
            }
            self.logger.info(f"Orderbook ontvangen: {orderbook_data}")
            self.db_manager.save_orderbook(orderbook_data)
        except Exception as e:
            self.logger.error(f"Fout bij verwerken van orderbook-data: {e}")

    # -------------------------------------------------------------
    # START/STOP
    # -------------------------------------------------------------
    def start_websocket(self):
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

        # Onmiddellijk authenticate:
        timestamp = int(time.time() * 1000)
        method = "GET"
        endpoint = "/v2/websocket"
        sig_value = self.generate_signature(timestamp, method, endpoint)
        self.logger.debug(f"[AUTH-PAYLOAD] signature={sig_value}, len={len(sig_value)}")
        auth_payload = {
            "action": "authenticate",
            "key": self.api_key,
            "signature": sig_value,  # hier maak je gebruik van sig_value
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
        self.logger.info("Stop websocket aangevraagd.")
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                self.logger.error(f"Fout bij sluiten in stop_websocket: {e}")

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
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
            self.db_manager.save_order(order_row)

            self.logger.info(
                f"[_handle_order_update] order {order_id} => status={status}, "
                f"filled={filled_amount}/{amount}"
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
            self.db_manager.save_fill(fill_row)

            self.logger.info(
                f"[_handle_fill_update] fill => amt={fill_amount} @ price={fill_price}, fee={fee_amount}"
            )
        except Exception as e:
            self.logger.error(f"Fout in _handle_fill_update: {e}")
