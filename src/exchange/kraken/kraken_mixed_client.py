# src/exchange/kraken/kraken_mixed_client.py

import json
import logging
import time
import threading
import requests
import websocket

logger = logging.getLogger("kraken_client")
logger.setLevel(logging.DEBUG)

# [CHANGED] - Unused import verwijderd
# from decimal import Decimal

# Mapping van "BTC-EUR" => "XBT/EUR", etc.
PAIR_MAPPING = {
    "BTC-EUR": "XBT/EUR",
    "ETH-EUR": "ETH/EUR",
    "XRP-EUR": "XRP/EUR",
    "DOGE-EUR": "XDG/EUR",
    # etc. tot 20
}

# Voor intervals in WS (kraken int -> string)
INTERVAL_INT_2_STR = {
    1: "1m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    240: "4h",
    1440: "1d"
}

from src.logger.logger import setup_kraken_logger

logger = setup_kraken_logger(logfile="logs/kraken_client.log", level=logging.DEBUG)
# Nu kun je logger.info("Hallo, ik ben de Kraken-lus") etc.


class KrakenMixedClient:
    """
    Combineert:
      - WebSocket-subscripties voor 'ohlc' op intervals_realtime (zodat <=25 subscripties)
      - REST-polling voor intervals_poll (geen sub-limit, maar poll-based).
    Resultaat: je hebt live data voor "snelle" TF (bv. 15m),
               en polled data voor grote TF (1h,4h,1d).
    pairs = [ "BTC-EUR","ETH-EUR", ... ]   (max ~20)
    intervals_realtime = [15]   # e.g. 15m
    intervals_poll     = [60,240,1440]  # e.g. 1h,4h,1d
    poll_interval_seconds = 300 => om de 5m polling

    We slaan data in "candles" (timestamp,datetime_utc,market,interval,open,high,low,close,volume,exchange="Kraken").
    """

    def __init__(self, db_manager, pairs, intervals_realtime, intervals_poll, poll_interval_seconds=300):
        self.db_manager = db_manager
        self.ws_url = "wss://ws.kraken.com/"
        self.ws = None
        self.ws_running = False
        self.ws_thread = None

        self.pairs = pairs if pairs else []
        self.intervals_realtime = intervals_realtime if intervals_realtime else []
        self.intervals_poll = intervals_poll if intervals_poll else []
        self.poll_interval = poll_interval_seconds

        # channel_id_map => channel_id -> (local_pair, interval_str)
        self.channel_id_map = {}

        logger.info("[KrakenMixedClient] init => pairs=%s, realtime=%s, poll=%s, poll_interval=%ds",
                    self.pairs, self.intervals_realtime, self.intervals_poll, self.poll_interval)

        self._ensure_exchange_column()  # controleer dat candles.exchange bestaat

        # start poll-thread voor grotere intervals
        self.poll_running = False
        self.poll_thread = None

    def _ensure_exchange_column(self):
        """
        Zorgt dat de 'exchange' kolom in 'candles' bestaat.
        (Voor andere tabellen doen we dat in database_manager.)
        """
        try:
            rows = self.db_manager.execute_query("PRAGMA table_info(candles)")
            existing = [r[1] for r in rows]
            if 'exchange' not in existing:
                self.db_manager.execute_query("ALTER TABLE candles ADD COLUMN exchange TEXT")
                logger.info("[KrakenMixedClient] Added 'exchange' col to 'candles'.")
        except Exception as e:
            logger.error("[KrakenMixedClient] ensure_exchange => %s", e)

    ##########################################################
    # START/STOP
    ##########################################################

    def start(self):
        """
        Start both WS thread (for intervals_realtime) and poll thread (for intervals_poll).
        """
        logger.info("[KrakenMixedClient] start => launching WS (if intervals_realtime) + poll thread.")
        if self.intervals_realtime:
            self._start_ws()
        if self.intervals_poll:
            self._start_poll_thread()

    def stop(self):
        """
        Stop both threads.
        """
        logger.info("[KrakenMixedClient] stop => stopping WS + poll.")
        if self.intervals_realtime:
            self._stop_ws()
        if self.intervals_poll:
            self._stop_poll_thread()
        logger.info("[KrakenMixedClient] stopped everything.")

    ##########################################################
    # WebSocket PART (for realtime intervals)
    ##########################################################

    def _start_ws(self):
        logger.info("[KrakenMixedClient] _start_ws => intervals_realtime=%s", self.intervals_realtime)
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
                self.ws.run_forever()
                logger.warning("[KrakenMixedClient] WS loop ended, retry in 5s if still ws_running.")
                time.sleep(5)

        self.ws_thread = threading.Thread(target=_ws_loop, daemon=True)
        self.ws_thread.start()

    def _stop_ws(self):
        logger.info("[KrakenMixedClient] _stop_ws.")
        self.ws_running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.error("Error closing WS => %s", e)
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5)

    def _on_ws_open(self, _):
        logger.info("[KrakenMixedClient] _on_ws_open => subscribe OHLC for intervals: %s", self.intervals_realtime)
        # for each pair, for each interval
        for local_pair in self.pairs:
            kraken_pair = PAIR_MAPPING.get(local_pair, "XBT/EUR")
            for iv_int in self.intervals_realtime:
                sub_msg = {
                    "event": "subscribe",
                    "pair": [kraken_pair],
                    "subscription": {
                        "name": "ohlc",
                        "interval": iv_int
                    }
                }
                self.ws.send(json.dumps(sub_msg))
                logger.info("Subscribe => %s (interval=%d)", kraken_pair, iv_int)

    def _on_ws_message(self, _, message):
        try:
            data = json.loads(message)
        except Exception as e:
            logger.error("[KrakenMixedClient] invalid JSON => %s", e)
            return

        if isinstance(data, dict):
            ev = data.get("event")
            if ev == "systemStatus":
                logger.info("[KrakenMixedClient] systemStatus => %s", data)
            elif ev == "subscriptionStatus":
                self._handle_subscription_status(data)
            elif ev == "heartbeat":
                logger.debug("[KrakenMixedClient] heartbeat ws.")
            else:
                logger.debug("[KrakenMixedClient] other ws => %s", data)
        elif isinstance(data, list):
            self._process_ohlc_data(data)
        else:
            logger.warning("[KrakenMixedClient] unexpected ws msg => %s", type(data))

    # [CHANGED] - Hernoemd 'ws' parameter naar '_' zodat PyCharm niet klaagt
    def _on_ws_error(self, _, error):
        logger.error("[KrakenMixedClient] on_ws_error => %s", error)

    # [CHANGED] - Hernoemd 'ws' parameter naar '_' zodat PyCharm niet klaagt
    def _on_ws_close(self, _, code, msg):
        logger.warning("[KrakenMixedClient] on_ws_close => code=%s, msg=%s", code, msg)

    def _handle_subscription_status(self, data):
        status = data.get("status")
        if status != "subscribed":
            logger.warning("[KrakenMixedClient] subscriptionStatus => not subscribed => %s", data)
            return

        channel_id = data.get("channelID")
        sub = data.get("subscription", {})
        kraken_pair = data.get("pair", "?")
        iv_int = sub.get("interval", 1)
        local_pair = self._inverse_map_pair(kraken_pair)
        interval_str = self._iv_int_to_str(iv_int)
        self.channel_id_map[channel_id] = (local_pair, interval_str)
        logger.info("[KrakenMixedClient] channel_id=%d => %s, interval=%s", channel_id, local_pair, interval_str)

    def _process_ohlc_data(self, data_list):
        # [channelID, [time,etime,open,high,low,close,vwap,volume,count], "ohlc", "XBT/EUR"]
        if len(data_list) < 4:
            return
        chan_id = data_list[0]
        payload = data_list[1]
        msg_type = data_list[2]
        if msg_type != "ohlc":
            return
        if chan_id not in self.channel_id_map:
            logger.warning("[KrakenMixedClient] unknown channel_id => %d", chan_id)
            return
        (local_pair, interval_str) = self.channel_id_map[chan_id]

        if not isinstance(payload, list) or len(payload) < 8:
            logger.warning("[KrakenMixedClient] invalid ohlc => %s", payload)
            return
        time_s = float(payload[0])
        open_p = payload[2]
        high_p = payload[3]
        low_p = payload[4]
        close_p = payload[5]
        volume = payload[7]

        ts_ms = int(time_s * 1000)
        candle_record = (
            ts_ms,
            local_pair,
            interval_str,
            float(open_p),
            float(high_p),
            float(low_p),
            float(close_p),
            float(volume),
            "Kraken"  # [CHANGED] - Altijd 'Kraken'
        )
        self._save_candle(candle_record)

    def _save_candle(self, candle_tuple):
        try:
            with self.db_manager.connection:
                sql = """
                INSERT OR REPLACE INTO candles
                (timestamp, datetime_utc, market, interval, open, high, low, close, volume, exchange)
                VALUES(
                  ?,
                  datetime(?/1000,'unixepoch'),
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?
                )
                """
                (ts, mkt, iv, o, h, l, c, vol, exch) = candle_tuple
                params = (ts, ts, mkt, iv, o, h, l, c, vol, exch)
                self.db_manager.connection.execute(sql, params)
        except Exception as e:
            logger.error("[KrakenMixedClient] error saving candle => %s", e)

    ##########################################################
    # Poll-Thread PART (for intervals_poll) => REST
    ##########################################################

    def _start_poll_thread(self):
        logger.info("[KrakenMixedClient] _start_poll_thread => intervals=%s, every %ds",
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
        Elke poll_interval_seconds:
         - For each pair in self.pairs
           - for each interval in self.intervals_poll
             - call /0/public/OHLC?interval=...
             - parse => store recent candles
        """
        for local_pair in self.pairs:
            kraken_pair = PAIR_MAPPING.get(local_pair, "XBT/EUR")
            for iv_int in self.intervals_poll:
                try:
                    rows = self._fetch_ohlc_rest(kraken_pair, iv_int)
                    if rows:
                        self._save_ohlc_rows(local_pair, iv_int, rows)
                except Exception as e:
                    logger.error("poll_intervals error => pair=%s, iv_int=%d => %s", local_pair, iv_int, e)

    def _fetch_ohlc_rest(self, kraken_pair, iv_int):
        """
        GET https://api.kraken.com/0/public/OHLC?pair=XXBTZEUR&interval=...
        Convert "XBT/EUR" => "XXBTZEUR", etc.
        """
        rest_pair = self._convert_kraken_pair_for_rest(kraken_pair)
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": rest_pair, "interval": iv_int}
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            logger.error("_fetch_ohlc_rest fail => %s", resp.text)
            return None
        data = resp.json()
        if data.get("error"):
            logger.warning("_fetch_ohlc_rest error => %s", data["error"])
            return None
        result = data.get("result", {})
        key = list(result.keys())[0] if result else ""
        rows = result.get(key, [])
        return rows

    def _save_ohlc_rows(self, local_pair, iv_int, rows):
        """
        rows => each => [time, open, high, low, close, vwap, volume, count]
        """
        interval_str = self._iv_int_to_str(iv_int)
        count_inserted = 0
        with self.db_manager.connection:
            for r in rows:
                if len(r) < 8:
                    continue
                time_s = float(r[0])
                open_p = r[1]
                high_p = r[2]
                low_p = r[3]
                close_p = r[4]
                volume = r[6]  # r[6] => volume
                ts_ms = int(time_s * 1000)
                sql = """
                INSERT OR REPLACE INTO candles
                (timestamp, datetime_utc, market, interval, open, high, low, close, volume, exchange)
                VALUES(
                  ?,
                  datetime(?/1000,'unixepoch'),
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?,
                  ?
                )
                """
                params = (
                    ts_ms,
                    ts_ms,
                    local_pair,
                    interval_str,
                    float(open_p),
                    float(high_p),
                    float(low_p),
                    float(close_p),
                    float(volume),
                    "Kraken"  # [CHANGED] - We slaan 'Kraken' op
                )
                self.db_manager.connection.execute(sql, params)
                count_inserted += 1
        logger.info("[KrakenMixedClient] poll => pair=%s, interval=%s => inserted %d rows",
                    local_pair, interval_str, count_inserted)

    ##########################################################
    # HELPERS
    ##########################################################

    def _convert_kraken_pair_for_rest(self, kraken_pair):
        """
        "XBT/EUR" => "XXBTZEUR"
        "ETH/EUR" => "XETHZEUR", etc.
        """
        if kraken_pair == "XBT/EUR":
            return "XXBTZEUR"
        elif kraken_pair == "ETH/EUR":
            return "XETHZEUR"
        elif kraken_pair == "XRP/EUR":
            return "XXRPZEUR"
        elif kraken_pair == "XDG/EUR":
            return "XXDGZEUR"
        # fallback
        return "XXBTZEUR"

    def _inverse_map_pair(self, kp):
        for loc, k in PAIR_MAPPING.items():
            if kp == k:
                return loc
        return kp

    def _iv_int_to_str(self, iv_int):
        return INTERVAL_INT_2_STR.get(iv_int, f"{iv_int}m")
