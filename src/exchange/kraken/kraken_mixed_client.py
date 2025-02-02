# ============================================================
# src/exchange/kraken/kraken_mixed_client.py
# ============================================================

import json
import logging
import time
import threading
import websocket

websocket.enableTrace(True)  # Activeer gedetailleerde WS-debug logging

import hashlib
import hmac
import base64
import urllib.parse
import os
from datetime import datetime, timedelta, timezone

from src.logger.logger import setup_kraken_logger

import requests

logger = setup_kraken_logger(logfile="logs/kraken_client.log", level=logging.DEBUG)


def safe_get(url, params=None, max_retries=3, sleep_seconds=1, headers=None):
    attempts = 0
    while attempts < max_retries:
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError as ce:
            logger.warning(f"[safe_get] ConnectionError => {ce}, retry {attempts+1}/{max_retries}...")
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
            logger.warning(f"[safe_post] ConnectionError => {ce}, retry {attempts+1}/{max_retries}...")
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


# -------------------------------
# Helper functies voor candles
# -------------------------------

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
    """
    Check of de candle (start=candle_timestamp_ms) voor 'timeframe' al
    definitief is afgelopen.
    """
    unit = timeframe[-1]
    try:
        value = int(timeframe[:-1])
    except ValueError:
        return False

    if unit == "m":
        duration_ms = value * 60 * 1000
    elif unit == "h":
        duration_ms = value * 60 * 60 * 1000
    elif unit == "d":
        duration_ms = value * 24 * 60 * 60 * 1000
    else:
        duration_ms = 0

    candle_start = datetime.fromtimestamp(candle_timestamp_ms / 1000, tz=timezone.utc)
    candle_end = candle_start + timedelta(milliseconds=duration_ms)
    current_time = datetime.now(timezone.utc)
    return current_time >= candle_end


class KrakenMixedClient:
    """
    Combineert:
      1) Publieke WebSocket-subscripties voor 'ohlc' (realtime intervals).
      2) REST-polling voor grotere intervals (bv. 1h, 4h, 1d).
      3) (Optioneel) Private WebSocket voor ownTrades/openOrders
    """

    def __init__(self, db_manager, kraken_cfg: dict, use_private_ws=False):
        """
        kraken_cfg:
          {
            "pairs": ["BTC-EUR", "ETH-EUR", ...],
            "intervals_realtime": [15],
            "intervals_poll": [60, 240, 1440],
            "poll_interval_seconds": 300,
            "apiKey": "...",
            "apiSecret": "..."
          }
        """
        self.db_manager = db_manager
        self.calls_this_minute = 0
        self.last_reset_ts = time.time()
        self.rest_limit_per_minute = 100

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

        # channel_id => (local_pair, interval_str, iv_int)
        self.channel_id_map = {}

        logger.info("[KrakenMixedClient] init => pairs=%s", self.pairs)
        logger.info("[KrakenMixedClient] intervals_realtime=%s, intervals_poll=%s, poll_interval=%ds",
                    self.intervals_realtime, self.intervals_poll, self.poll_interval)
        logger.info("[KrakenMixedClient] use_private_ws=%s (key_len=%d)",
                    self.use_private_ws, len(self.api_key))

        self._ensure_exchange_column()

        # Poll thread
        self.poll_running = False
        self.poll_thread = None

    def _check_rate_limit(self):
        now = time.time()
        # Reset de teller elke 60 seconden
        if now - self.last_reset_ts >= 60:
            self.calls_this_minute = 0
            self.last_reset_ts = now
        # Als je limiet bereikt of overschreden is, wacht dan even
        if self.calls_this_minute >= self.rest_limit_per_minute:
            logger.warning("REST rate limit dreigt overschreden te worden. Slaap 5s...")
            time.sleep(5)

    def _increment_call(self):
        self.calls_this_minute += 1

    def _ensure_exchange_column(self):
        try:
            rows = self.db_manager.execute_query("PRAGMA table_info(candles)")
            existing = [r[1] for r in rows]
            if 'exchange' not in existing:
                self.db_manager.execute_query("ALTER TABLE candles ADD COLUMN exchange TEXT")
                logger.info("[KrakenMixedClient] Added 'exchange' col to 'candles'.")
        except Exception as e:
            logger.error("[KrakenMixedClient] ensure_exchange => %s", e)

    # ===========================================
    # START / STOP
    # ===========================================
    def start(self):
        """
        Start WS en poll (en private WS indien gewenst).
        """
        logger.info("[KrakenMixedClient] start => launching WS + poll + (maybe) private.")
        if self.intervals_realtime:
            self._start_ws()

        # 2) Start poll
        if self.intervals_poll:
            self._start_poll_thread()

        # 3) Start private WS (optioneel)
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
        # 3) Stop private WS
        if self.use_private_ws:
            self._stop_private_ws()
        logger.info("[KrakenMixedClient] Stopped everything.")

    # ===========================================
    # (A) Publieke WS
    # ===========================================
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
            self._process_ohlc_data(data)
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
        iv_int = sub.get("interval", 1)

        local_found = None
        for local, ws_ in self.kraken_ws_map.items():
            if ws_ == ws_pair:
                local_found = local
                break
        if not local_found:
            logger.warning(f"[KrakenMixedClient] subscription for {ws_pair} => no local found??")
            return

        interval_str = self._iv_int_to_str(iv_int)
        # Sla nu (local_found, interval_str, iv_int) op in de mapping
        self.channel_id_map[channel_id] = (local_found, interval_str, iv_int)
        logger.info("[KrakenMixedClient] channel_id=%d => local_pair=%s, interval=%s",
                    channel_id, local_found, interval_str)

    def _process_ohlc_data(self, data_list):
        if len(data_list) < 4:
            logger.debug("[KrakenMixedClient] ontvangen data_list te kort: %s", data_list)
            return
        chan_id = data_list[0]
        payload = data_list[1]
        msg_type = data_list[2]
        logger.debug("[KrakenMixedClient] berichttype ontvangen: %s", repr(msg_type))
        if not msg_type.startswith("ohlc"):
            logger.debug("[KrakenMixedClient] ontvangen niet-ohlc bericht: %s", data_list)
            return

        if chan_id not in self.channel_id_map:
            logger.warning("[KrakenMixedClient] onbekende channel_id => %d", chan_id)
            return

        # Haal de tuple op: (local_pair, interval_str, iv_int)
        local_pair, interval_str, iv_int = self.channel_id_map[chan_id]
        if not isinstance(payload, list) or len(payload) < 8:
            logger.warning("[KrakenMixedClient] ongeldige ohlc payload voor %s, interval=%s => %s",
                           local_pair, interval_str, payload)
            return

        try:
            time_s = float(payload[0])
            open_p = float(payload[2])
            high_p = float(payload[3])
            low_p = float(payload[4])
            close_p = float(payload[5])
            volume = float(payload[7])
        except Exception as e:
            logger.error("[KrakenMixedClient] Fout bij conversie van ohlc data: %s", e)
            return

        ts_ms = int(time_s * 1000)
        candle = (
            ts_ms,
            local_pair,
            interval_str,
            open_p,
            high_p,
            low_p,
            close_p,
            volume,
            "Kraken"
        )

        # Controleer of closed
        if is_candle_closed(ts_ms, interval_str):
            logger.info(
                f"[Candle CLOSED] WS => {local_pair} {interval_str}, start_ts={ts_ms}, recognized closed at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
            )
            self._save_candle(candle)
        else:
            logger.debug("[KrakenMixedClient] Candle voor %s op ts=%d nog niet gesloten; probeer REST-fallback.",
                         local_pair, ts_ms)
            # Bepaal ws_pair via de mapping van de lokale naam
            ws_pair = self.kraken_ws_map.get(local_pair)
            if ws_pair is None:
                logger.error("[KrakenMixedClient] Geen ws_pair voor %s", local_pair)
                return

            fallback_candle = self._fetch_latest_candle_rest(ws_pair, iv_int)
            if fallback_candle:
                # fallback_candle is een tuple: (ts, o, h, l, c, vol)
                fb_ts, fb_o, fb_h, fb_l, fb_c, fb_vol = fallback_candle
                # Extra check: controleer of deze candle wel afgesloten is
                if is_candle_closed(fb_ts, interval_str):
                    logger.info(
                        f"[Candle CLOSED] REST-fallback => {local_pair} {interval_str}, fb_ts={fb_ts}, recognized closed at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
                    )
                    fallback_candle_full = (fb_ts, local_pair, interval_str, fb_o, fb_h, fb_l, fb_c, fb_vol, "Kraken")
                    self._save_candle(fallback_candle_full)
                else:
                    logger.debug("[KrakenMixedClient] REST-fallback candle voor %s is nog niet afgesloten.", local_pair)
            else:
                logger.debug("[KrakenMixedClient] REST-fallback geen candle voor %s.", local_pair)

    def _save_candle(self, candle_tuple):
        try:
            (ts, mkt, iv, o, h, l, c, vol, exch) = candle_tuple
            dt_utc = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            sql = """
            INSERT OR REPLACE INTO candles
            (timestamp, datetime_utc, market, interval,
             open, high, low, close, volume, exchange)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (ts, dt_utc, mkt, iv, o, h, l, c, vol, exch)
            self.db_manager.connection.execute(sql, params)

            logger.info(
                "[Store Candle] market=%s, interval=%s, timestamp=%d => open=%.5f, close=%.5f (UTC=%s), stored at local=%s",
                mkt, iv, ts, o, c, dt_utc, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
        # Indien gewenst, kun je ook openOrders abonneren:
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

        # 1) fill
        fill_row = {
            "order_id": txid,
            "market": pair,
            "side": side,
            "fill_amount": vol,
            "fill_price": price,
            "fee_amount": fee,
            "timestamp": ms_,
            "exchange": "Kraken"
        }
        self.db_manager.save_fill(fill_row)

        # 2) trade
        trade_data = {
            "symbol": pair,
            "side": side,
            "amount": vol,
            "price": price,
            "timestamp": ms_,
            "position_id": txid,
            "position_type": "unknown",
            "status": "closed",
            "pnl_eur": 0.0,
            "fees": fee,
            "trade_cost": cost,
            "exchange": "Kraken"
        }
        self.db_manager.save_trade(trade_data)
        # Indien van toepassing, roep hier ook een functie aan voor position-updates.

    def _on_ws_private_error(self, ws, error):
        logger.error("[PrivateWS] => %s", error)

    def _on_ws_private_close(self, ws, code, msg):
        logger.warning("[PrivateWS] closed => code=%s, msg=%s", code, msg)

    # ===========================================
    # (C) Poll-Thread (REST)
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
        for loc in self.pairs:
            if loc not in self.kraken_ws_map:
                logger.warning(f"[POLL] local={loc} => skip (no ws_map).")
                continue
            ws_ = self.kraken_ws_map[loc]
            for iv_int in self.intervals_poll:
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
        interval_str = self._iv_int_to_str(iv_int)
        count = 0

        with self.db_manager.connection:
            for row in rows:
                if len(row) < 8:
                    continue
                t_s = float(row[0])
                o_ = float(row[1])
                h_ = float(row[2])
                l_ = float(row[3])
                c_ = float(row[4])
                vol = float(row[6])
                ts_ms = int(t_s * 1000)
                # Converteer timestamp naar UTC-string in Python
                dt_utc = datetime.fromtimestamp(ts_ms / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

                sql = """
                INSERT OR REPLACE INTO candles
                (timestamp, datetime_utc, market, interval,
                 open, high, low, close, volume, exchange)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                params = (ts_ms, dt_utc, local_pair, interval_str, o_, h_, l_, c_, vol, "Kraken")
                self.db_manager.connection.execute(sql, params)
                count += 1

                # Als je specifiek de 4h/1d wilt loggen, doe:
                if interval_str in ("4h", "1d"):
                    logger.info(
                        f"[Poll Candle] {local_pair} {interval_str}, start_ts={ts_ms} => open={o_:.5f}, close={c_:.5f}, dt_utc={dt_utc}"
                    )

        logger.info("[KrakenMixedClient] poll => pair=%s, interval=%s => inserted %d rows",
                    local_pair, interval_str, count)

    # Rest call voor 15m candle als WS openbaar niet levert.
    def _fetch_latest_candle_rest(self, ws_pair, iv_int):
        """
        Haalt de laatste candle op voor ws_pair/iv_int via REST.
        Retourneert (ts, open, high, low, close, volume) of None.
        """
        self._check_rate_limit()
        self._increment_call()
        rest_name = self.kraken_rest_map.get(ws_pair)
        if not rest_name:
            logger.warning(f"[REST] ws_pair={ws_pair} => geen rest_name => overslaan.")
            return None

        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": rest_name, "interval": iv_int}
        resp = safe_get(url, params=params, max_retries=3, sleep_seconds=2)
        if not resp:
            logger.error(f"[REST] fout => geen candle na retries voor {ws_pair}/{iv_int}")
            return None
        if resp.status_code != 200:
            logger.error(f"[REST] fout bij ophalen candle: {resp.text}")
            return None
        data = resp.json()
        if data.get("error"):
            logger.warning(f"[REST] error => {data['error']}")
            return None
        result = data.get("result", {})
        if not result:
            return None
        key = list(result.keys())[0]
        candles = result[key]
        if not candles:
            return None

        last_candle = candles[-1]
        if len(last_candle) < 7:
            return None

        ts = int(float(last_candle[0]) * 1000)
        o = float(last_candle[1])
        h = float(last_candle[2])
        l = float(last_candle[3])
        c = float(last_candle[4])
        vol = float(last_candle[6])
        return (ts, o, h, l, c, vol)

    def _iv_int_to_str(self, iv_int):
        mapping = {
            1: "1m", 5: "5m", 15: "15m", 30: "30m",
            60: "1h", 240: "4h", 1440: "1d"
        }
        return mapping.get(iv_int, f"{iv_int}m")
