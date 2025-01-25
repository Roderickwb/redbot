import os

from src.logger.logger import setup_logger, log_error
import queue
import pandas as pd
import websocket
import json
import hmac
import hashlib
import time
import threading
from decimal import Decimal
from dotenv import load_dotenv
from src.indicator_analysis.indicators import IndicatorAnalysis
from src.config.config import DB_FILE, WEBSOCKET_LOG_FILE, PAIRS_CONFIG
from src.config.config import PULLBACK_CONFIG

# Dit is de (officiële) python-bitvavo-api import
from python_bitvavo_api.bitvavo import Bitvavo

load_dotenv()

logger = setup_logger('websocket_client', WEBSOCKET_LOG_FILE)
logger.info("WebSocket-client gestart.")

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
WS_URL = os.getenv("WS_URL", "wss://ws.bitvavo.com/v2/")


class WebSocketClient:
    def __init__(self, ws_url, db_manager, api_key, api_secret):
        """
        WebSocketClient - Beheert WebSocket-verbindingen en interactie met Bitvavo.
        """
        self.ws_url = ws_url
        self.db_manager = db_manager
        self.api_key = api_key
        self.api_secret = api_secret

        # Initialiseer Bitvavo (REST + WS)
        self.bitvavo = Bitvavo({
            'APIKEY': self.api_key,
            'APISECRET': self.api_secret,
            'RESTURL': 'https://api.bitvavo.com/v2',
            'WSURL': 'wss://ws.bitvavo.com/v2/',
            'ACCESSWINDOW': 10000
        })

        # Dicts voor live prijzen en timestamps
        self.latest_prices = {}
        self.latest_update_timestamps = {}

        self.ws = None

        # (B) NIEUW: Hier maken we een queue voor orderupdates
        self.order_updates_queue = queue.Queue()

        # Rate-limit tracker
        self.calls_this_minute = 0
        self.last_reset_ts = time.time()
        self.limit_per_minute = 100

        self.subscribed = False
        self.running = False
        self._thread = None

        self.markets = PAIRS_CONFIG
        logger.info(f"[WebSocketClient] Using dynamic markets: {self.markets}")
        self.logger = logger  # ← Zet deze regel erbij

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
            logger.warning("REST rate limit dreigt overschreden te worden. Slaap 5s...")
            time.sleep(5)

    def _increment_call(self):
        self.calls_this_minute += 1

    # -------------------------------------------------------------
    # REST-calls
    # -------------------------------------------------------------
    def get_balance(self):
        self._check_rate_limit()
        self._increment_call()
        # Nog steeds 'fake' call / paper
        yaml_budget = PULLBACK_CONFIG.get("initial_capital", 100)
        budget_decimal = Decimal(str(yaml_budget))
        logger.info(f"Simuleer get_balance(): return {{'EUR': {budget_decimal}}}")
        return {"EUR": budget_decimal}

    def place_order(self, side, symbol, amount, order_type=None):
        self._check_rate_limit()
        self._increment_call()
        logger.info(f"Simuleer place_order({side}, {symbol}, {amount}) (Geen echte call)")

    # -------------------------------------------------------------
    # Nieuwe fallback-call
    # -------------------------------------------------------------
    def _fetch_rest_price(self, symbol: str) -> Decimal:
        with self.fallback_lock:
            now_ts = time.time()
            cached_entry = self.fallback_price_cache.get(symbol, None)
            if cached_entry is not None:
                if not isinstance(cached_entry, tuple) or len(cached_entry) != 2:
                    logger.error(
                        f"[BUG] fallback_price_cache[{symbol}] bevat: {cached_entry} "
                        f"(type={type(cached_entry)})"
                    )
                    # evt. herstellen/weggooien om crash te voorkomen
                    del self.fallback_price_cache[symbol]
                    return Decimal("0")

                (cached_price, cached_ts) = cached_entry
                if (now_ts - cached_ts) < 2:
                    logger.debug(f"[REST-Fallback] Gebruik cache voor {symbol} => {cached_price}")
                    return cached_price

            self._check_rate_limit()
            self._increment_call()

            tickers = self.bitvavo.tickerPrice({"market": symbol})
            logger.debug(f"[REST-Fallback RAW] {symbol} => {tickers}")

            # == CASE 1: Dict met error ==
            if isinstance(tickers, dict) and "error" in tickers:
                logger.error(f"[REST-Fallback] Bitvavo error: {tickers}")
                return Decimal("0")

            # == CASE 2: Dict met 'price' (en dus 1 market) ==
            elif isinstance(tickers, dict) and "price" in tickers:
                rest_str_price = tickers["price"]
                rest_price = Decimal(str(rest_str_price))
                self.fallback_price_cache[symbol] = (rest_price, now_ts)
                logger.debug(f"[REST-Fallback] {symbol} => {rest_price} (single dict from REST)")
                return rest_price

            # == CASE 3: List met minstens 1 item dat 'price' heeft ==
            elif isinstance(tickers, list) and len(tickers) > 0 and "price" in tickers[0]:
                rest_str_price = tickers[0]["price"]
                rest_price = Decimal(str(rest_str_price))
                self.fallback_price_cache[symbol] = (rest_price, now_ts)
                logger.debug(f"[REST-Fallback] {symbol} => {rest_price} (list of tickers from REST)")
                return rest_price

            else:
                logger.warning(f"[REST-Fallback] Onverwachte return van tickerPrice({symbol}): {tickers}")
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

        # Te oud => fallback
        logger.warning(f"[WebSocketClient] {symbol} => WS price is {age:.1f}s oud => fallback to REST")
        rest_price = self._fetch_rest_price(symbol)
        return rest_price if rest_price > 0 else Decimal("0")

    # -------------------------------------------------------------
    # WebSocket-functies
    # -------------------------------------------------------------
    def generate_signature(self, timestamp, method, endpoint):
        message = f"{timestamp}{method}{endpoint}"
        signature = hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha256).hexdigest()
        return signature

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            logger.error(f"Ongeldig JSON-bericht ontvangen: {message}. Fout: {e}")
            return

        if data.get("event") == "pong":
            logger.info("JSON-pong ontvangen!")
            return

        logger.info(f"Ontvangen bericht: {data}")

        # Verwerk authenticatiebevestiging
        if data.get("event") == "authenticate" and data.get("authenticated") == True:
            logger.info("Authenticatie succesvol.")
            self.subscribe_to_channels(ws)
            return

        event_type = data.get("event")

        # Order/fill events
        if event_type == "order":
            logger.info("Eigen order geüpdatet => in queue")
            self.order_updates_queue.put(data)
            return
        elif event_type == "fill":
            logger.info("Eigen order fill => in queue")
            self.order_updates_queue.put(data)
            return

        # ============== DE KEY-CHANGE: trades!! ==============
        if event_type == "trade":
            self.process_trade_data(data)

        elif event_type == "candle":
            self.process_candle_data(data)
        elif event_type == "ticker":
            self.process_ticker_data(data)
        elif event_type == "book":
            self.process_orderbook_data(data)
        elif event_type == "account":
            logger.info(f"Account-update: {data}")
            # event. Je kunt hier bv. balances updaten, als je wilt
            pass

        else:
            logger.debug(f"Onbekend event_type: {event_type}, data={data}")

    def subscribe_to_channels(self, ws):
        if self.subscribed:
            logger.warning("Al gesubscribed, skip subscription.")
            return
        self.subscribed = True

        # Vergeet niet "trades" toe te voegen! (staat er al)
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

        logger.info(f"Verzenden abonnementspayload: {subscribe_payload}")
        ws.send(json.dumps(subscribe_payload))
        logger.info("Abonnementen verzonden na authenticatie.")

    # -------------------------------------------------------------
    # process_trade_data = HOOFDPUNT: Price komt hieruit
    # -------------------------------------------------------------

    def process_trade_data(self, data):
        """
        We halen de laatste verhandelde prijs uit data["trades"],
        want daar staat meestal een lijst van 1..N trades.
        """
        market_symbol = data.get("market")
        trades_array = data.get("trades", [])

        if not market_symbol:
            logger.debug("[Trades] Geen market_symbol in trade-event.")
            return

        if not trades_array:
            logger.debug(f"[Trades] trade-event maar geen trades[] voor {market_symbol}.")
            return

        # We nemen de laatste trade uit de lijst (meest recente)
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
            logger.debug(
                f"[Trades] {market_symbol} => last trade price={price_float}, total trades in event={len(trades_array)}"
            )
        else:
            logger.debug(
                f"[Trades] {market_symbol} => kon geen geldige price parsen uit {last_trade}"
            )

    # -------------------------------------------------------------
    # CANDLES
    # -------------------------------------------------------------
    def process_candle_data(self, data):
        try:
            market = data.get("market")
            interval = data.get("interval")
            candles = data.get("candle", [])
            logger.info(f"Binnenkomende candle: {market}, {interval}, count={len(candles)}")

            if not candles:
                logger.debug(f"Geen candles ontvangen voor {market} - {interval}.")
                return

            # Voeg hier extra logging toe
            logger.debug(f"Raw candles data: {candles}")

            # 1. Maak een DataFrame van de ontvangen candles
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # 2. Bereken de indicatoren via IndicatorAnalysis (optioneel)
            df_with_indicators = IndicatorAnalysis.calculate_indicators(df)
            logger.info(f"df_with_indicators shape: {df_with_indicators.shape}")

            # 3. Sla de candle data op in DB
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
                # AANPASSING 1: Debug print
                logger.info(
                    f"DEBUG Candle => ts={formatted_candle[0]}, "
                    f"market={formatted_candle[1]}, interval={formatted_candle[2]}, "
                    f"open={formatted_candle[3]}, close={formatted_candle[6]}"
                )
                # (A) save_candles kan bv. in 'candles'-tabel
                self.db_manager.save_candles([formatted_candle])
                df_test = self.db_manager.fetch_data("candles", limit=10, market=market, interval=interval) # tijdelijk
                logger.info(f"Na insert -> fetch {market}, {interval}, got {len(df_test)} records") # tijdelijk

            df_with_indicators["market"] = market
            df_with_indicators["interval"] = interval
            self.db_manager.save_indicators(df_with_indicators)

            logger.info(
                f"{len(df_with_indicators)} candles verwerkt voor {market}-{interval}. "
                f"Voorbeeld: {df_with_indicators.head(1).to_dict('records')}"
            )
        except Exception as e:
            logger.error(f"Fout bij verwerken van candle-data: {e}")

    # -------------------------------------------------------------
    # TICKER => We slaan 'ticker' nog op in DB, maar overschrijven
    #           NIET meer onze self.latest_prices (zie comment).
    # -------------------------------------------------------------
    def process_ticker_data(self, data):
        try:
            logger.info(f"Ontvangen ticker-data: {data}")

            best_bid = float(data.get("bestBid", 0))
            best_ask = float(data.get("bestAsk", 0))
            spread = best_ask - best_bid if best_bid and best_ask else 0.0

            # Sla op in DB
            ticker_data = {
                'market': data.get("market"),
                'bestBid': best_bid,
                'bestAsk': best_ask,
                'spread': spread
            }
            self.db_manager.save_ticker(ticker_data)
            logger.info(f"Ticker verwerkt (DB): {ticker_data}")

            # ===== Belangrijk: we commenten de prijs-update uit =====
            # We willen NIET de mid-price meer opslaan als 'live' price.
            # Als je dat wel wilt, uncomment de volgende regels.
            #
            # market_symbol = data.get("market")
            # if best_ask > 0 and best_bid > 0:
            #     live_price = (best_ask + best_bid) / 2.0
            # else:
            #     live_price = best_bid if best_bid > 0 else best_ask
            #
            # now_ts = time.time()
            # self.latest_prices[market_symbol] = live_price
            # self.latest_update_timestamps[market_symbol] = now_ts

        except Exception as e:
            log_error(logger, f"Fout bij verwerken van ticker-data: {e}")

    # -------------------------------------------------------------
    # ORDERBOOK
    # -------------------------------------------------------------
    def process_orderbook_data(self, data):
        try:
            orderbook_data = {
                'market': data.get("market"),
                'bids': data.get("bids", []),
                'asks': data.get("asks", [])
                # Hier kun je evt. 'timestamp' of 'nonce' meenemen als Bitvavo dat meestuurt
            }
            logger.info(f"Orderbook ontvangen: {orderbook_data}")
            self.db_manager.save_orderbook(orderbook_data)
        except Exception as e:
            logger.error(f"Fout bij verwerken van orderbook-data: {e}")

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
                    # Geen protocol ping/pong
                    self.ws.run_forever(ping_interval=30, ping_timeout=10)
                except Exception as e:
                    logger.error(f"Fout in _run_forever-lus: {e}")

                if self.running:
                    logger.warning("WebSocket verbroken. Herstart over 5s...")
                    time.sleep(5)
                else:
                    logger.info("self.running=False => definitief stoppen.")
                    break

            logger.info("Einde _run_forever-lus => thread stopt.")

        self._thread = threading.Thread(target=_run_forever, daemon=True)
        self._thread.start()
        logger.info("WebSocket client is gestart in aparte thread. (geen protocol ping/pong)")

    def on_open(self, ws):
        logger.info("WebSocket on_open triggered.")
        timestamp = int(time.time() * 1000)
        method = "GET"
        endpoint = "/websocket"
        signature = self.generate_signature(timestamp, method, endpoint)
        auth_payload = {
            "action": "authenticate",
            "key": self.api_key,
            "signature": signature,
            "timestamp": timestamp
        }
        ws.send(json.dumps(auth_payload))
        logger.info("Authenticatie verzonden.")
        self.subscribed = False

        # self._start_json_ping(ws)

    #def _start_json_ping(self, ws):
    #    def ping_loop():
    #        while self.running:
    #           time.sleep(25)
    #            if not self.running:
    #                break
    #            try:
    #                # check of socket nog open is
    #                if ws.sock and ws.sock.connected:
    #                    logger.debug("Stuur JSON-ping naar Bitvavo.")
    #                    ws.send(json.dumps({"action": "ping"}))
    #                else:
    #                    logger.debug("WS niet connected, skip ping.")
    #            except Exception as e:
    #                logger.error(f"Fout bij JSON-ping: {e}")
    #                break

    #    t = threading.Thread(target=ping_loop, daemon=True)
    #    t.start()

    def on_error(self, ws, error):
        logger.error(f"WebSocket Error: {error}")
        try:
            ws.close()
        except Exception as e:
            logger.warning(f"Fout bij sluiten in on_error: {e}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"WebSocket-verbinding gesloten. status_code={close_status_code}, msg={close_msg}")

    def stop_websocket(self):
        logger.info("Stop websocket aangevraagd.")
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.error(f"Fout bij sluiten in stop_websocket: {e}")

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("WebSocket client is volledig gestopt.")

    def handle_order_update(self, data: dict):
        """
        Verwerkt 'order' event (status, partial fill, etc.) en slaat in DB op.
        """
        try:
            self.logger.info(f"[_handle_order_update] {data}")
            order_id = data.get("orderId")
            market = data.get("market")
            side = data.get("side")
            status = data.get("status", "unknown")
            amount = data.get("amount", "0")
            filled_amount = data.get("filledAmount", "0")
            price = data.get("price", "0")

            # Opslaan in DB (vereist dat je in db_manager save_order hebt)
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
                f"[_handle_order_update] order {order_id} => status={status}, filled={filled_amount}/{amount}")

        except Exception as e:
            self.logger.error(f"Fout in _handle_order_update: {e}")

    def handle_fill_update(self, data: dict):
        """
        Verwerkt 'fill' event (executed amount, fill price, fee, etc.) en slaat in DB op.
        """
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

            self.logger.info(f"[_handle_fill_update] fill => amt={fill_amount} @ price={fill_price}, fee={fee_amount}")

        except Exception as e:
            self.logger.error(f"Fout in _handle_fill_update: {e}")
