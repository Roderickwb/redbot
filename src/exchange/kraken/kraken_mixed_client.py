# ============================================================
# src/exchange/kraken/kraken_mixed_client.py
# ============================================================

import json
import logging
import time
import threading
import websocket

websocket.enableTrace(True)  # Laat debug-logs van de websocket-library zien

import hashlib
import hmac
import base64
import urllib.parse
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal  # [ADDED for minLot usage]

from src.logger.logger import setup_kraken_logger

import requests

# [QUEUE CHANGE START]
import queue  # <-- Toevoegen voor queue.Queue
# [QUEUE CHANGE END]

logger = setup_kraken_logger(logfile="logs/kraken_client.log", level=logging.DEBUG)


def safe_get(url, params=None, max_retries=3, sleep_seconds=1, headers=None):
    attempts = 0
    while attempts < max_retries:
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError as ce:
            logger.warning(f"[safe_get] ConnectionError => {ce}, retry {attempts + 1}/{max_retries}...")
            time.sleep(sleep_seconds)
            attempts += 1
        except requests.exceptions.HTTPError as he:
            logger.warning(f"[safe_get] HTTPError => {he}.")
            return None
        except Exception as ex:
            logger.error(f"[safe_get] Onverwachte fout => {ex}")
            return None
    logger.error(f"[safe_get] Max retries={max_retries} overschreden => None.")
    return None


def safe_post(url, data=None, headers=None, max_retries=3, sleep_seconds=1):
    attempts = 0
    while attempts < max_retries:
        try:
            resp = requests.post(url, data=data, headers=headers, timeout=10)
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError as ce:
            logger.warning(f"[safe_post] ConnectionError => {ce}, retry {attempts + 1}/{max_retries}...")
            time.sleep(sleep_seconds)
            attempts += 1
        except requests.exceptions.HTTPError as he:
            logger.warning(f"[safe_post] HTTPError => {he}.")
            return None
        except Exception as ex:
            logger.error(f"[safe_post] Onverwachte fout => {ex}")
            return None
    logger.error(f"[safe_post] Max retries={max_retries} overschreden => None.")
    return None


def build_kraken_mapping() -> dict:
    """
    Haalt dynamisch alle paren op van Kraken via /0/public/AssetPairs
    en bouwt een mapping:
      {
        "BTC-EUR": {"wsname": "XBT/EUR", "restname": "XXBTZEUR"},
        "ETH-EUR": {"wsname": "ETH/EUR", "restname": "XETHZEUR"},
        ...
      }
    We filteren alleen paren die eindigen op /EUR.
    """
    url = "https://api.kraken.com/0/public/AssetPairs"
    try:
        resp = safe_get(url, max_retries=3, sleep_seconds=2)
        if not resp:
            logger.error("[DynamicMapping] failed after retries.")
            return {}
        data = resp.json()
        if data.get("error"):
            logger.error(f"[DynamicMapping] Kraken errors: {data['error']}")
            return {}
        all_pairs = data.get("result", {})
        dynamic_map = {}
        for rest_name, info in all_pairs.items():
            ws = info.get("wsname", "")
            if not ws:
                continue
            # filter op EUR
            if not ws.endswith("/EUR"):
                continue
            local = ws.replace("/", "-")
            dynamic_map[local] = {
                "wsname": ws,
                "restname": rest_name
            }
        logger.info(f"[DynamicMapping] build_kraken_mapping => found {len(dynamic_map)} EUR pairs.")
        return dynamic_map
    except Exception as e:
        logger.error(f"[DynamicMapping] exception => {e}")
        return {}


def interval_to_hours(interval_str: str) -> float:
    """
    Converteer een interval-string (bijv. "1m", "15m", "1h", "4h", "1d") naar uren.
    """
    if interval_str.endswith("m"):
        return int(interval_str[:-1]) / 60.0
    elif interval_str.endswith("h"):
        return float(interval_str[:-1])
    elif interval_str.endswith("d"):
        return int(interval_str[:-1]) * 24.0
    else:
        return 0.0

def is_candle_closed(candle_timestamp_ms: int, timeframe: str) -> bool:
    # We gaan ervan uit dat candle_timestamp_ms de EINDtijd van de candle is.
    # => Candle is closed als now >= candle_timestamp_ms

    now_ms = int(time.time() * 1000)
    return now_ms >= candle_timestamp_ms

class KrakenMixedClient:
    """
    Combineert:
      1) Publieke WebSocket-subscripties voor 'ohlc' (realtime intervals).
      2) REST-polling voor grotere intervals (1h, 4h, 1d).
      3) (Optioneel) Private WebSocket voor ownTrades/openOrders

    Wordt ook als "client" gebruikt in je strategie, dus we kunnen hier
    een 'get_balance()' toevoegen zodat meltdown_manager of strategies
    kunnen aanroepen: client.get_balance().

    [NEW] We voegen ook een live-ticker feed toe, zodat je SL/TP
    intra-candle kunt checken via get_latest_ws_price(symbol).
    """

    def __init__(self, db_manager, kraken_cfg: dict, use_private_ws=False):
        """
        :param db_manager:  DatabaseManager
        :param kraken_cfg:  dict met keys:
           {
             "pairs": [...],
             "intervals_realtime": [...],
             "intervals_poll": [...],
             "poll_interval_seconds": ...,
             "apiKey": "...",
             "apiSecret": "..."
           }
        :param use_private_ws: True => start private WS (ownTrades, etc.) als apiKey+Secret zijn gezet
        """
        self.db_manager = db_manager
        self.calls_this_minute = 0
        self.last_reset_ts = time.time()
        self.rest_limit_per_minute = 250  # [CHANGED => 250]

        # 1) Dynamische mapping
        all_mapping = build_kraken_mapping()
        # 2) Bepaal paren
        desired_locals = kraken_cfg.get("pairs", [])
        if not desired_locals:
            logger.warning("[KrakenMixedClient] Geen paren in config 'pairs'.")

        self.pairs = []
        self.kraken_ws_map = {}
        self.kraken_rest_map = {}

        for loc in desired_locals:
            if loc in all_mapping:
                ws_ = all_mapping[loc]["wsname"]
                r_ = all_mapping[loc]["restname"]
                self.kraken_ws_map[loc] = ws_
                self.kraken_rest_map[ws_] = r_
                self.pairs.append(loc)
                logger.info(f"[KrakenMixedClient] Toegevoegd pair: {loc} -> ws: {ws_}, rest: {r_}")
            else:
                logger.warning(f"[KrakenMixedClient] pair {loc} not found => skip")

        self.intervals_realtime = kraken_cfg.get("intervals_realtime", [15])
        self.intervals_poll = kraken_cfg.get("intervals_poll", [60, 240, 1440])
        self.poll_interval = kraken_cfg.get("poll_interval_seconds", 300)

        self.api_key = kraken_cfg.get("apiKey", "")
        self.api_secret = kraken_cfg.get("apiSecret", "")

        self.use_private_ws = use_private_ws
        self.ws_url = "wss://ws.kraken.com/"
        self.ws_auth_url = "wss://ws-auth.kraken.com"

        self.ws = None
        self.ws_running = False
        self.ws_thread = None

        self.ws_private_token = None
        self.ws_private = None
        self.ws_private_running = False
        self.ws_private_thread = None

        # [QUEUE CHANGE START]
        # Maak een nieuwe Queue voor fills
        self.trade_fills_queue = queue.Queue()
        # [QUEUE CHANGE END]

        # channel_id => (local_pair, interval_str, iv_int)
        self.channel_id_map = {}

        # [ADDED] Dictionary voor live tickerprijzen
        self.live_ticker_prices = {}  # key=symbol("BTC-EUR"), value=float price

        logger.info("[KrakenMixedClient] init => pairs=%s", self.pairs)
        logger.info("[KrakenMixedClient] intervals_realtime=%s, intervals_poll=%s, poll_interval=%ds",
                    self.intervals_realtime, self.intervals_poll, self.poll_interval)
        logger.info("[KrakenMixedClient] use_private_ws=%s (key_len=%d)",
                    self.use_private_ws, len(self.api_key))

        # [ADDED for minLot]
        self._min_lot_dict = {}

        # Poll thread
        self.poll_running = False
        self.poll_thread = None

    # [LIVE TRADE CHANGE START]
    def get_balance(self) -> dict:
        """
        Voor meltdown_manager of strategies:
        Deze versie probeert ECHT /0/private/Balance op te vragen
        als self.api_key en self.api_secret niet leeg zijn.
        Anders fallback => {"EUR":"100"} (paper).
        """
        if not self.api_key or not self.api_secret:
            logger.debug("[KrakenMixedClient] get_balance => no key/secret => fallback 350 EUR.")
            return {"EUR": "350"}  # fallback als paper

        path = "/0/private/Balance"
        url = "https://api.kraken.com" + path
        nonce = str(int(time.time() * 1000))

        payload = {"nonce": nonce}
        postdata_str = urllib.parse.urlencode(payload)

        # HMAC signing
        sha256_digest = hashlib.sha256((nonce + postdata_str).encode("utf-8")).digest()
        hmac_key = base64.b64decode(self.api_secret)
        to_sign = path.encode('utf-8') + sha256_digest
        signature = hmac.new(hmac_key, to_sign, hashlib.sha512)
        sigdigest = base64.b64encode(signature.digest()).decode()

        headers = {
            "API-Key": self.api_key,
            "API-Sign": sigdigest,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        self._check_rate_limit()
        self._increment_call()
        resp = safe_post(url, data=payload, headers=headers)
        if not resp:
            logger.error("[KrakenMixedClient] get_balance => geen response => return {}")
            return {}

        j = resp.json()
        logger.debug(f"[KrakenMixedClient] get_balance => raw JSON response: {j}")
        err = j.get("error", [])
        if err:
            logger.error(f"[KrakenMixedClient] get_balance => error={err}")
            return {}

        result = j.get("result", {})
        logger.debug(f"[KrakenMixedClient] get_balance => raw Kraken result => {result}")
        # Bv. {"ZEUR": "100.123", "XXBT": "0.01", ...}
        # event. parse => "EUR": ...
        newdict = {}
        for k, v in result.items():
            if k == "ZEUR":
                newdict["EUR"] = v
            elif k.startswith("X"):
                # "XXBT" => "XBT", "XETH" => "ETH"? net wat je wilt
                sym = k[1:]
                newdict[sym] = v
            else:
                newdict[k] = v

        logger.debug(f"[KrakenMixedClient] get_balance => {newdict}")
        return newdict
    # [LIVE TRADE CHANGE END]

    # [LIVE TRADE CHANGE START]
    def place_order(self, side: str, symbol: str, volume: float,
                    ordertype="market", price=None) -> dict:
        """
        Plaatst een echte order via /0/private/AddOrder op Kraken.
        :param side: 'buy' of 'sell'
        :param symbol: bv. "BTC-EUR" (local)
        :param volume: float (aantal coins)
        :param ordertype: 'market' (of 'limit')
        :param price: alleen bij 'limit'
        :return: {"order_id": "...", "description": {...}}
        """
        if not self.api_key or not self.api_secret:
            # Als geen keys => paper => return pseudo-result
            logger.warning("[KrakenMixedClient] place_order => no key => paper fallback.")
            return {"order_id": "FAKE_ORDER", "description": {}}

        # Vind de REST name
        ws_ = self.kraken_ws_map.get(symbol)
        rest_ = self.kraken_rest_map.get(ws_, "")
        if not rest_:
            logger.error(f"[KrakenMixedClient] place_order => no rest mapping for {symbol} => skip.")
            return {}

        path = "/0/private/AddOrder"
        url = "https://api.kraken.com" + path
        nonce = str(int(time.time() * 1000))

        payload = {
            "nonce": nonce,
            "pair": rest_,
            "type": side,
            "ordertype": ordertype,
            "volume": str(volume)
        }
        if ordertype == "limit" and price is not None:
            payload["price"] = str(price)

        postdata_str = urllib.parse.urlencode(payload)

        # Sign
        sha256_digest = hashlib.sha256((nonce + postdata_str).encode("utf-8")).digest()
        hmac_key = base64.b64decode(self.api_secret)
        to_sign = path.encode('utf-8') + sha256_digest
        signature = hmac.new(hmac_key, to_sign, hashlib.sha512)
        sigdigest = base64.b64encode(signature.digest()).decode()

        headers = {
            "API-Key": self.api_key,
            "API-Sign": sigdigest,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        self._check_rate_limit()
        self._increment_call()
        resp = safe_post(url, data=payload, headers=headers)
        if not resp:
            logger.error("[KrakenMixedClient] place_order => no response => return {}")
            return {}

        j = resp.json()
        err = j.get("error", [])
        if err:
            logger.error(f"[KrakenMixedClient] place_order => error={err}")
            return {}

        r = j.get("result", {})
        txids = r.get("txid", [])
        if not txids:
            logger.warning("[KrakenMixedClient] place_order => geen txid in result => return {}")
            return {}

        order_id = txids[0]
        descr = r.get("descr", {})
        logger.info(f"[KrakenMixedClient] Placed {ordertype} order => side={side}, symbol={symbol}, vol={volume}, order_id={order_id}")
        return {"order_id": order_id, "description": descr}
    # [LIVE TRADE CHANGE END]

    def _check_rate_limit(self):
        now = time.time()
        # Reset de teller elke 60 seconden
        if now - self.last_reset_ts >= 60:
            self.calls_this_minute = 0
            self.last_reset_ts = now
        # Als je limiet bereikt of overschreden is, wacht dan even
        if self.calls_this_minute >= self.rest_limit_per_minute:
            logger.warning("REST rate limit dreigt overschreden. Slaap 5s...")
            time.sleep(5)

    def _increment_call(self):
        self.calls_this_minute += 1

    # ===========================================
    # START / STOP
    # ===========================================
    def start(self):
        """
        Start WS en poll (en private WS indien gewenst).
        """
        logger.info("[KrakenMixedClient] start => launching WS + poll + (maybe) private.")

        # 1) Start WS
        if self.intervals_realtime:
            self._start_ws()

        # 2) Start poll-thread (voor alle intervals behalve 15m)
        if self.intervals_poll:
            self._start_poll_thread()

        # [ADDED for minLot] => bouw min_lot_info
        self.build_min_lot_info()

        # 3) Start private WS
        if self.use_private_ws and self.api_key and self.api_secret:
            tok = self._fetch_kraken_token()
            if tok:
                self.ws_private_token = tok
                self._start_private_ws()
            else:
                logger.warning("[KrakenMixedClient] no private WS token => skip")
        elif self.use_private_ws:
            logger.warning("[KrakenMixedClient] missing key/secret => skip private ws")

    def stop(self):
        """
        Stop WS + poll + private
        """
        logger.info("[KrakenMixedClient] stop => stopping WS + poll + private.")
        # 1) Stop publieke WS
        if self.intervals_realtime:
            self._stop_ws()
        # 2) Stop poll
        if self.intervals_poll:
            self._stop_poll_thread()

        if self.use_private_ws:
            self._stop_private_ws()
        logger.info("[KrakenMixedClient] Stopped everything.")

    # [ADDED for minLot]
    def build_min_lot_info(self):
        """
        Haal van /0/public/AssetPairs alle data op,
        parse 'ordermin' (of lot_decimals) en sla dit
        in self._min_lot_dict op, keyed door rest_name.
        """
        logger.info("[KrakenMixedClient] build_min_lot_info => start polling AssetPairs for minLot.")
        url = "https://api.kraken.com/0/public/AssetPairs"
        try:
            resp = safe_get(url, max_retries=3, sleep_seconds=1)
            if not resp:
                logger.error("[build_min_lot_info] no response => skip.")
                return
            data = resp.json()
            if data.get("error"):
                logger.error(f"[build_min_lot_info] error => {data['error']}")
                return
            results = data.get("result", {})
            for pair_name, info in results.items():
                # We check of 'ordermin' in info, anders fallback
                ordermin_str = None
                if "ordermin" in info:
                    ordermin_str = info["ordermin"]
                elif "lot_decimals" in info:
                    # lot_decimals => b.v. 8 => min lot=10^-8
                    dec = info["lot_decimals"]
                    ordermin_str = str(Decimal("1") / Decimal(str(10**dec)))
                else:
                    ordermin_str = "1.0"
                try:
                    self._min_lot_dict[pair_name] = Decimal(ordermin_str)
                except:
                    self._min_lot_dict[pair_name] = Decimal("1.0")

            logger.info(f"[build_min_lot_info] done => {len(self._min_lot_dict)} pairs in dict.")
        except Exception as e:
            logger.error(f"[build_min_lot_info] Unexpected => {e}")

    def get_min_lot(self, local_symbol: str) -> Decimal:
        """
        Geef minLot terug voor local_symbol ("XBT-EUR", enz).
        We zoeken in self.kraken_rest_map => rest_name => self._min_lot_dict[rest_name].
        Fallback=Decimal("1.0")
        """
        if not hasattr(self, "_min_lot_dict") or not self._min_lot_dict:
            return Decimal("1.0")

        ws_ = self.kraken_ws_map.get(local_symbol, "")
        rest_ = self.kraken_rest_map.get(ws_, "")
        if rest_ in self._min_lot_dict:
            return self._min_lot_dict[rest_]
        else:
            logger.warning(f"[get_min_lot] {local_symbol} => rest_name={rest_} not found => fallback=1.0")
            return Decimal("1.0")

    # -------------------------------------------
    # (A) Publieke WS code
    # -------------------------------------------
    # (NIET AANGEPAST, behalve debug)
    # -------------------------------------------

    def _start_ws(self):
        logger.info("[KrakenMixedClient] _start_ws => intervals=%s", self.intervals_realtime)
        self.ws_running = True

        def _ws_loop():
            while self.ws_running:
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_ws_open,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close
                )
                # Probeer met aangepaste ping_interval en ping_timeout
                self.ws.run_forever(ping_interval=20, ping_timeout=15)
                logger.warning("[KrakenMixedClient] Public WS ended; retry in 5s if still running.")
                time.sleep(5)

        self.ws_thread = threading.Thread(target=_ws_loop, daemon=True)
        self.ws_thread.start()

    def _stop_ws(self):
        logger.info("[KrakenMixedClient] _stop_ws")
        self.ws_running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.error("[KrakenMixedClient] error closing public WS => %s", e)
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5)

    def _on_ws_open(self, ws):
        logger.info("[KrakenMixedClient] _on_ws_open => subscribe intervals=%s", self.intervals_realtime)
        # A) Abonneer op OHLC
        for local in self.pairs:
            if local not in self.kraken_ws_map:
                logger.warning(f"[PubWS] no ws_map for {local}, skip.")
                continue
            ws_name = self.kraken_ws_map[local]
            for iv_int in self.intervals_realtime:
                sub_msg = {
                    "event": "subscribe",
                    "pair": [ws_name],
                    "subscription": {
                        "name": "ohlc",
                        "interval": iv_int
                    }
                }
                ws.send(json.dumps(sub_msg))
                logger.info(f"[PubWS] Subscribe => {ws_name}, interval={iv_int}")

        # B) Ticker feed
        for local in self.pairs:
            if local not in self.kraken_ws_map:
                continue
            ws_name = self.kraken_ws_map[local]
            sub_msg_ticker = {
                "event": "subscribe",
                "pair": [ws_name],
                "subscription": {
                    "name": "ticker"
                }
            }
            ws.send(json.dumps(sub_msg_ticker))
            logger.info(f"[PubWS] Ticker Subscribe => {ws_name}")

    def _on_ws_message(self, ws, message):
        try:
            data = json.loads(message)
        except Exception as e:
            logger.error("[KrakenMixedClient] public WS invalid JSON => %s", e)
            return

        if isinstance(data, dict):
            ev = data.get("event")
            if ev == "systemStatus":
                logger.info("[KrakenMixedClient] systemStatus => %s", data)
            elif ev == "subscriptionStatus":
                self._handle_subscription_status(data)
            elif ev == "heartbeat":
                logger.debug("[KrakenMixedClient] public ws heartbeat.")
            else:
                logger.debug(f"[KrakenMixedClient] public ws unknown => {data}")
        elif isinstance(data, list):
            # Let op: data kan OHLC-data zijn, of Ticker-data
            if len(data) >= 4:
                msg_type = data[-2]  # bv. "ohlc-15" of "ticker"
                if isinstance(msg_type, str) and msg_type.startswith("ohlc"):
                    self._process_ohlc_data(data)
                elif msg_type == "ticker":
                    self._process_ticker_data(data)
                else:
                    logger.debug(f"[KrakenMixedClient] Array msg => {data}")
            else:
                logger.debug(f"[KrakenMixedClient] Short array => {data}")
        else:
            logger.warning(f"[KrakenMixedClient] unexpected ws msg => {type(data)}")

    def _on_ws_error(self, ws, error):
        logger.error("[KrakenMixedClient] public WS error => %s", error)

    def _on_ws_close(self, ws, code, msg):
        current_time = datetime.utcnow().isoformat()
        logger.warning("[KrakenMixedClient] public WS closed at %s => code=%s, msg=%s", current_time, code, msg)

    def _handle_subscription_status(self, data):
        status = data.get("status")
        if status != "subscribed":
            logger.warning("[KrakenMixedClient] subscriptionStatus => not subscribed => %s", data)
            return

        channel_id = data.get("channelID")
        ws_pair = data.get("pair", "?")
        sub = data.get("subscription", {})
        name = sub.get("name")
        iv_int = sub.get("interval", None)

        local_found = None
        for local, ws_ in self.kraken_ws_map.items():
            if ws_ == ws_pair:
                local_found = local
                break

        if not local_found:
            logger.warning(f"[KrakenMixedClient] subscription for {ws_pair} => no local found??")
            return

        if name == "ohlc" and iv_int is not None:
            interval_str = self._iv_int_to_str(iv_int)
            self.channel_id_map[channel_id] = (local_found, interval_str, iv_int)
            logger.info("[KrakenMixedClient] channel_id=%d => local_pair=%s, interval=%s (ohlc)",
                        channel_id, local_found, interval_str)
        elif name == "ticker":
            # Voor ticker hoef je geen intervals bij te houden, maar wel channel_id => local_pair
            self.channel_id_map[channel_id] = (local_found, "ticker", None)
            logger.info("[KrakenMixedClient] channel_id=%d => local_pair=%s (ticker)", channel_id, local_found)
        else:
            logger.info(f"[KrakenMixedClient] subscriptionStatus => sub={name}, {ws_pair}")

    def _process_ticker_data(self, data_list):
        """
        Kraken Ticker data komt in de vorm:
        [channel_id, {
          "b": ["bestBid","lot volume"],
          "a": ["bestAsk","lot volume"],
          ...
        }, "ticker", <pair>]
        """
        chan_id = data_list[0]
        if chan_id not in self.channel_id_map:
            logger.warning("[Ticker] Onbekende channel_id => %d", chan_id)
            return

        local_pair, sub_name, _ = self.channel_id_map[chan_id]
        if len(data_list) < 4:
            logger.warning("[Ticker] Invalid data_list => %s", data_list)
            return

        payload = data_list[1]
        if not isinstance(payload, dict):
            logger.warning("[Ticker] payload is not a dict => %s", payload)
            return

        best_bid_arr = payload.get("b", [])
        best_ask_arr = payload.get("a", [])
        if len(best_bid_arr) < 1 or len(best_ask_arr) < 1:
            logger.debug("[Ticker] incomplete b/a => %s", payload)
            return

        try:
            best_bid = float(best_bid_arr[0])
            best_ask = float(best_ask_arr[0])
            mid_px = (best_bid + best_ask) / 2
            self.live_ticker_prices[local_pair] = mid_px
            logger.debug(f"[Ticker] {local_pair} => bid={best_bid:.2f}, ask={best_ask:.2f}, mid={mid_px:.2f}")
        except Exception as e:
            logger.error("[Ticker] parsing error => %s", e)

    def _iv_int_to_millis(self, iv_int: int) -> int:
        """
        Zet de integer-interval (1, 5, 15, 60, 240, 1440) om in milliseconden.
        15 => 15 minuten => 900_000 ms
        60 => 60 min => 3_600_000 ms
        1440 => 1 dag => 86_400_000 ms
        """
        return iv_int * 60_000

    def _process_ohlc_data(self, data_list):
        """
        Verwerkt de live OHLC-berichten van Kraken's public WebSocket.
        Let op: Kraken stuurt in 'time_s' de START van de candle (bv. 13:30).
        De candle loopt dus tot 13:45, en is pas dan 'closed'.
        """

        if len(data_list) < 4:
            logger.debug("[KrakenMixedClient] ontvangen data_list te kort: %s", data_list)
            return

        chan_id = data_list[0]
        payload = data_list[1]
        msg_type = data_list[2]

        if chan_id not in self.channel_id_map:
            logger.warning("[KrakenMixedClient] onbekende channel_id => %d", chan_id)
            return

        # channel_id_map[chan_id] => (local_pair, interval_str, iv_int)
        local_pair, interval_str, iv_int = self.channel_id_map[chan_id]

        if not isinstance(payload, list) or len(payload) < 8:
            logger.warning("[KrakenMixedClient] ongeldige ohlc payload => %s", payload)
            return

        try:
            # time_s => STARTtijd van de candle (in seconden)
            time_s = float(payload[0])
            open_p = float(payload[2])
            high_p = float(payload[3])
            low_p = float(payload[4])
            close_p = float(payload[5])
            volume = float(payload[7])
        except Exception as e:
            logger.error("[KrakenMixedClient] Fout bij conversie van ohlc data: %s", e)
            return

        start_ms = int(time_s * 1000)
        interval_ms = self._iv_int_to_millis(iv_int)  # bv. 15 => 900_000
        end_ms = start_ms + interval_ms  # Candle eindigt hier

        # We slaan de candle op met 'timestamp = end_ms' (de eindtijd),
        # en alleen als hij echt voorbij is:
        now_ms = int(time.time() * 1000)
        if now_ms >= end_ms:
            logger.info(f"[Candle CLOSED] WS => {local_pair} {interval_str}, end_ms={end_ms}")
            candle_tuple = (end_ms, local_pair, interval_str, open_p, high_p, low_p, close_p, volume)
            self._save_candle_kraken(candle_tuple)
        else:
            # Candle nog niet klaar => skip
            logger.debug("[KrakenMixedClient] Candle voor %s %s is nog niet gesloten. skip.",
                         local_pair, interval_str)

            # Tijdelijk niet gebruiken. Voor 15M een poll ingebouwd voor de rest vertrouwen we op WS.
            #current_time_ms = int(time.time() * 1000)
            # [CHANGED] => 20sec skip
            #if (current_time_ms - ts_ms) < 20000:
            #    logger.debug("[KrakenMixedClient] Candle is nog vers => skip fallback for now.")
            #    return

            #ws_pair = self.kraken_ws_map.get(local_pair)
            #if not ws_pair:
            #    logger.error("[KrakenMixedClient] Geen ws_pair voor %s", local_pair)
            #    return

            # fallback_candle = self._fetch_latest_candle_rest(ws_pair, iv_int)
            # if fallback_candle:
            #    (fb_ts, fb_o, fb_h, fb_l, fb_c, fb_vol) = fallback_candle
            #    if is_candle_closed(fb_ts, interval_str):
            #        logger.info(
            #            f"[Candle CLOSED] REST-fallback => {local_pair} {interval_str}, fb_ts={fb_ts}, recognized closed"
            #        )
            #        fallback_tuple = (fb_ts, local_pair, interval_str, fb_o, fb_h, fb_l, fb_c, fb_vol)
            #        self._save_candle_kraken(fallback_tuple)
            #    else:
            #        logger.debug("[KrakenMixedClient] REST-fallback candle voor %s is nog niet afgesloten.", local_pair)

    # ===========================================
    # (A.1) Opslaan candle in candles_kraken
    # ===========================================
    def _save_candle_kraken(self, candle_tuple):
        """
        Voorheen:
          self.db_manager.connection.execute(...)
          self.db_manager.connection.commit()

        Nu vervangen door self.db_manager.execute_query(...).
        """
        try:
            (ts, mkt, iv, o, h, l, c, vol) = candle_tuple
            dt_utc = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

            sql = """
            INSERT OR REPLACE INTO candles_kraken
            (timestamp, datetime_utc, market, interval, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (ts, dt_utc, mkt, iv, o, h, l, c, vol)

            # => Door "INSERT" begint de DB-manager in de write-tak (BEGIN IMMEDIATE + COMMIT).
            self.db_manager.execute_query(sql, params)

            logger.info(
                "[Store Candle] => [candles_kraken] market=%s, interval=%s, timestamp=%d => open=%.5f, close=%.5f (UTC=%s)",
                mkt, iv, ts, o, c, dt_utc
            )
        except Exception as e:
            logger.error("[KrakenMixedClient] error saving candle => %s", e)

    # ===========================================
    # (B) Private WS
    # ===========================================
    def _fetch_kraken_token(self):
        if not self.api_key or not self.api_secret:
            logger.warning("[PrivateWS] no api_key/secret => can't fetch token.")
            return None
        logger.info("[PrivateWS] Fetching Kraken WS token...")

        nonce = str(int(time.time() * 1000))
        path = "/0/private/GetWebSocketsToken"
        url = "https://api.kraken.com" + path
        postdata_str = urllib.parse.urlencode({"nonce": nonce})
        encoded = (nonce + postdata_str).encode("utf-8")
        sha256_digest = hashlib.sha256(encoded).digest()
        hmac_key = base64.b64decode(self.api_secret)
        to_sign = path.encode('utf-8') + sha256_digest
        signature = hmac.new(hmac_key, to_sign, hashlib.sha512)
        sigdigest = base64.b64encode(signature.digest()).decode()
        headers = {
            "API-Key": self.api_key,
            "API-Sign": sigdigest,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        try:
            resp = safe_post(url, data={"nonce": nonce}, headers=headers, max_retries=3)
            if not resp:
                logger.error("[PrivateWS] fail => no response after retries.")
                return None
            if resp.status_code == 200:
                jj = resp.json()
                err = jj.get("error", [])
                if err:
                    logger.error(f"[PrivateWS] error => {err}")
                    return None
                token = jj["result"]["token"]
                logger.info(f"[PrivateWS] got token => {token[:8]}...redacted")
                return token
            else:
                logger.error("[PrivateWS] fail => %d => %s", resp.status_code, resp.text)
                return None
        except Exception as e:
            logger.error("[PrivateWS] exception => %s", e)
            return None

    def _start_private_ws(self):
        """
        Start de auth-ws => wss://ws-auth.kraken.com
        """
        self.ws_private_running = True

        def _priv_loop():
            while self.ws_private_running:
                self.ws_private = websocket.WebSocketApp(
                    self.ws_auth_url,
                    on_open=self._on_ws_private_open,
                    on_message=self._on_ws_private_message,
                    on_error=self._on_ws_private_error,
                    on_close=self._on_ws_private_close
                )
                self.ws_private.run_forever()
                logger.warning("[PrivateWS] loop ended, retry in 5s if still running.")
                time.sleep(5)

        self.ws_private_thread = threading.Thread(target=_priv_loop, daemon=True)
        self.ws_private_thread.start()

    def _stop_private_ws(self):
        logger.info("[KrakenMixedClient] _stop_private_ws")
        self.ws_private_running = False
        if self.ws_private:
            try:
                self.ws_private.close()
            except Exception as e:
                logger.error("[PrivateWS] closing => %s", e)
        if self.ws_private_thread and self.ws_private_thread.is_alive():
            self.ws_private_thread.join(timeout=5)

    def _on_ws_private_open(self, ws):
        if not self.ws_private_token:
            logger.warning("[PrivateWS] no token => skip subscribe ownTrades.")
            return
        sub_msg = {
            "event": "subscribe",
            "subscription": {
                "name": "ownTrades",
                "token": self.ws_private_token
            }
        }
        ws.send(json.dumps(sub_msg))
        logger.info("Abonnement voor ownTrades verzonden via private WS.")

        # Eventueel openOrders:
        # sub_msg_orders = {
        #     "event": "subscribe",
        #     "subscription": {
        #         "name": "openOrders",
        #         "token": self.ws_private_token
        #     }
        # }
        # ws.send(json.dumps(sub_msg_orders))

    def _on_ws_private_message(self, ws, message):
        try:
            data = json.loads(message)
            if isinstance(data, dict):
                ev = data.get("event")
                if ev == "heartbeat":
                    return
                elif ev == "subscriptionStatus":
                    st = data.get("status")
                    logger.info(f"[PrivateWS] subscriptionStatus => {st}, {data}")
                else:
                    logger.debug(f"[PrivateWS] dict => {data}")
            elif isinstance(data, list):
                if len(data) >= 3 and data[-1] == "ownTrades":
                    own_trades_arr = data[0]
                    for tx_info in own_trades_arr:
                        for txid, fill_data in tx_info.items():
                            self._handle_own_trade(txid, fill_data)
                else:
                    logger.debug(f"[PrivateWS] array => {data}")
            else:
                logger.debug(f"[PrivateWS] ??? => {data}")
        except Exception as e:
            logger.error("[PrivateWS] on_message => %s", e, exc_info=True)

    def _handle_own_trade(self, txid, fill_data):
        side = fill_data.get("type", "").lower()
        vol_s = fill_data.get("vol", "0")
        vol = float(vol_s)
        cost_s = fill_data.get("cost", "0")
        cost = float(cost_s)
        fee_s = fill_data.get("fee", "0")
        fee = float(fee_s)
        pair = fill_data.get("pair", "XBT/USD")
        fill_time = fill_data.get("time", 0.0)
        ms_ = int(float(fill_time) * 1000) if fill_time else int(time.time() * 1000)
        price = cost / vol if vol > 0 else float(fill_data.get("price", "0"))
        logger.info(f"[PrivateWS] ownTrade => txid={txid}, side={side}, vol={vol}, price={price}, fee={fee}")

        # [QUEUE CHANGE START]
        # In plaats van direct DB-manager calls, push in self.trade_fills_queue
        fill_event = {
            "txid": txid,
            "side": side,
            "vol": vol,
            "cost": cost,
            "fee": fee,
            "pair": pair,
            "timestamp_ms": ms_,
            "price": price
        }
        self.trade_fills_queue.put(fill_event)
        # [QUEUE CHANGE END]

        # 1) fill => in 'fills' met exchange="Kraken"
        #fill_row = {
        #    "order_id": txid,
        #    "market": pair,
        #    "side": side,
        #    "fill_amount": vol,
        #    "fill_price": price,
        #    "fee_amount": fee,
        #    "timestamp": ms_,
        #    "exchange": "Kraken"
        #}
        #self.db_manager.save_fill(fill_row)

        # 2) trade => in 'trades' met exchange="Kraken"
        #trade_data = {
        #    "symbol": pair,
        #    "side": side,
        #    "amount": vol,
        #    "price": price,
        #    "timestamp": ms_,
        #    "position_id": txid,
        #    "position_type": "unknown",
        #    "status": "closed",
        #    "pnl_eur": 0.0,
        #    "fees": fee,
        #    "trade_cost": cost,
        #    "exchange": "Kraken"
        #}
        #self.db_manager.save_trade(trade_data)
        # Indien van toepassing, roep hier ook een functie aan voor position-updates.

    def _on_ws_private_error(self, ws, error):
        logger.error("[PrivateWS] => %s", error)

    def _on_ws_private_close(self, ws, code, msg):
        logger.warning("[PrivateWS] closed => code=%s, msg=%s", code, msg)

    # ===========================================
    # (C) Poll-Thread (REST) - origineel
    # ===========================================
    def _start_poll_thread(self):
        logger.info("[KrakenMixedClient] _start_poll_thread => intervals=%s, poll_int=%ds",
                    self.intervals_poll, self.poll_interval)
        self.poll_running = True

        def _poll_loop():
            while self.poll_running:
                self._poll_intervals()
                time.sleep(self.poll_interval)

        self.poll_thread = threading.Thread(target=_poll_loop, daemon=True)
        self.poll_thread.start()

    def _stop_poll_thread(self):
        logger.info("[KrakenMixedClient] _stop_poll_thread")
        self.poll_running = False
        if self.poll_thread and self.poll_thread.is_alive():
            self.poll_thread.join(timeout=5)

    def _poll_intervals(self):
        """
        Poll alle intervals (zoals [60, 240, 1440]) behalve 15m,
        want we doen 15m in een aparte poll-thread met korter interval (30s).

        Als je wél 15m in intervals_poll hebt, kun je het hier uitcommenten of checken.
        """
        for loc in self.pairs:
            if loc not in self.kraken_ws_map:
                logger.warning(f"[POLL] local={loc} => skip (no ws_map).")
                continue
            ws_ = self.kraken_ws_map[loc]
            for iv_int in self.intervals_poll:
                # [COMMENTED] => skip if iv_int==15
                if iv_int == 15:
                    # We skip 15m here, do it in _start_poll_15m_thread
                    continue

                try:
                    rows = self._fetch_ohlc_rest(ws_, iv_int)
                    if rows:
                        self._save_ohlc_rows(loc, iv_int, rows)
                except Exception as e:
                    logger.error("poll_intervals error => pair=%s, iv_int=%d => %s", loc, iv_int, e)

    def _fetch_ohlc_rest(self, ws_pair, iv_int):
        # Voer eerst rate limit check uit:
        self._check_rate_limit()
        self._increment_call()
        rest_name = self.kraken_rest_map.get(ws_pair)
        if not rest_name:
            logger.warning(f"[REST] ws_pair={ws_pair} => no rest_name => skip.")
            return None

        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": rest_name, "interval": iv_int}
        resp = safe_get(url, params=params, max_retries=3, sleep_seconds=2)
        if not resp:
            logger.error(f"[REST] fail => can't fetch {ws_pair} at interval={iv_int} after retries.")
            return None
        if resp.status_code != 200:
            logger.error(f"[REST] fail => {resp.text}")
            return None

        j = resp.json()
        if j.get("error"):
            logger.warning(f"[REST] error => {j['error']}")
            return None
        result = j.get("result", {})
        if not result:
            return None
        key = list(result.keys())[0]
        rows = result[key]
        return rows

    def _save_ohlc_rows(self, local_pair, iv_int, rows):
        """
        Voorheen: `with self.db_manager.connection: for row in rows: ...`

        Nu: maken we een final_list en gebruiken we _executemany(...) in de DB-manager,
        zodat we niet in conflict komen met “cannot start a transaction within a transaction.”
        """
        interval_str = self._iv_int_to_str(iv_int)
        final_list = []
        insert_sql = """
            INSERT OR REPLACE INTO candles_kraken
            (timestamp, datetime_utc, market, interval, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)            
        """

        now_ms = int(time.time() * 1000)
        interval_ms = self._iv_int_to_millis(iv_int)  # <-- AANGEPAST: hulpfunctie

        for row in rows:
            if len(row) < 8:
                continue

            start_sec = float(row[0])  # <-- Candle start
            o_ = float(row[1])
            h_ = float(row[2])
            l_ = float(row[3])
            c_ = float(row[4])
            vol = float(row[6])

            start_ms = int(start_sec * 1000)
            end_ms = start_ms + interval_ms  # <-- AANGEPAST

            # Check of candle gesloten is
            if now_ms < end_ms:
                # Candle is nog niet klaar => skip
                continue

            # datetime_utc alleen voor logging / easy reading
            dt_utc = datetime.fromtimestamp(end_ms / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            params = (end_ms, dt_utc, local_pair, interval_str, o_, h_, l_, c_, vol)
            final_list.append(params)

        if final_list:
            try:
                self.db_manager._executemany(insert_sql, final_list)
                logger.info("[KrakenMixedClient] poll => pair=%s, interval=%s => inserted %d rows",
                            local_pair, interval_str, len(final_list))
            except Exception as e:
                logger.error("[KrakenMixedClient] _save_ohlc_rows error => %s", e)

    # ===========================================
    # (D) Extra poll-thread specifically for 15m
    # ===========================================

    def _poll_15m_only(self):
        """
        Poll alleen de 15m interval (on-demand). Verwijder de sleep-lus, dus
        roep dit enkel aan vanuit Executor of elders wanneer je wilt.
        """
        if 15 not in self.intervals_poll:
            return
        for loc in self.pairs:
            if loc not in self.kraken_ws_map:
                logger.warning(f"[poll_15m_only] local={loc} => skip (no ws_map).")
                continue
            ws_ = self.kraken_ws_map[loc]
            iv_int = 15
            try:
                rows = self._fetch_ohlc_rest(ws_, iv_int)
                if rows:
                    self._save_ohlc_rows(loc, iv_int, rows)
            except Exception as e:
                logger.error("poll_15m_only => pair=%s => %s", loc, e)

    def _iv_int_to_str(self, iv_int):
        mapping = {
            1: "1m", 5: "5m", 15: "15m", 30: "30m",
            60: "1h", 240: "4h", 1440: "1d"
        }
        return mapping.get(iv_int, f"{iv_int}m")

    # ===========================================
    # PUBLIC HELPER: get_latest_ws_price(...)
    # ===========================================
    def get_latest_ws_price(self, symbol: str) -> float:
        """
        Geeft de meest recente (live) prijs terug, zoals ontvangen via
        de 'ticker'-websocket feed. Returnt 0 als er niets bekend is.
        """
        return float(self.live_ticker_prices.get(symbol, 0.0))
