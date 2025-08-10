import logging
from decimal import Decimal
import pandas as pd

from ta.trend import ADXIndicator, MACD

# Lokale imports (zelfde patronen als je pullback)
from src.config.config import load_config_file
from src.logger.logger import setup_logger
from src.indicator_analysis.indicators import IndicatorAnalysis


class TrendStrategy4H:
    """
    Trend Strategy (4h context, 1h timing) — SKELETON
    - Geen orders, alleen logging.
    - Gebruikt je DB (candles_kraken) en bestaande IndicatorAnalysis.
    """

    def __init__(self, data_client, order_client, db_manager, config_path=None):
        self.data_client = data_client
        self.order_client = order_client
        self.db_manager = db_manager

        if config_path:
            full_cfg = load_config_file(config_path)
            self.cfg = full_cfg.get("trend_strategy_4h", {})
        else:
            self.cfg = {}

        # Flags & params uit config.yaml
        self.enabled = bool(self.cfg.get("enabled", False))
        self.log_file = self.cfg.get("log_file", "logs/trend_strategy_4h.log")

        self.trend_timeframe = self.cfg.get("trend_timeframe", "4h")
        self.entry_timeframe = self.cfg.get("entry_timeframe", "1h")

        # EMA-structuur
        self.ema_fast = int(self.cfg.get("ema_fast", 20))
        self.ema_slow = int(self.cfg.get("ema_slow", 50))
        self.require_trend_stack = bool(self.cfg.get("require_trend_stack", True))

        # ADX filters
        self.use_adx_filter = bool(self.cfg.get("use_adx_filter", True))
        self.adx_entry_tf_threshold = float(self.cfg.get("adx_entry_tf_threshold", 20))
        self.use_adx_multitimeframe = bool(self.cfg.get("use_adx_multitimeframe", True))
        self.adx_high_tf_threshold = float(self.cfg.get("adx_high_tf_threshold", 20))
        self.use_adx_directional_filter = bool(self.cfg.get("use_adx_directional_filter", True))
        self.adx_window = int(self.cfg.get("adx_window", 14))  # optioneel in yaml, default 14

        # Risk params (voor later, nu alleen loggen)
        self.atr_window = int(self.cfg.get("atr_window", 14))
        self.sl_atr_mult = float(self.cfg.get("sl_atr_mult", 1.5))
        self.tp1_atr_mult = float(self.cfg.get("tp1_atr_mult", 1.5))
        self.tp1_portion_pct = float(self.cfg.get("tp1_portion_pct", 0.50))
        self.trailing_atr_mult = float(self.cfg.get("trailing_atr_mult", 1.0))

        # Logger
        self.logger = setup_logger("trend_strategy_4h", self.log_file, logging.INFO)
        self.logger.info("[TrendStrategy4H] initialised (enabled=%s)", self.enabled)

    # ---------- Public API ----------
    def execute_strategy(self, symbol: str):
        """
        SKELETON: Geen orders. Alleen data ophalen, filters evalueren en loggen.
        """
        if not self.enabled:
            return

        # 4h (trend-context)
        df_h4 = self._fetch_and_indicator(symbol, self.trend_timeframe, limit=300)
        if df_h4.empty:
            self.logger.warning("[4h] no data for %s", symbol)
            return

        # 1h (entry-timing)
        df_entry = self._fetch_and_indicator(symbol, self.entry_timeframe, limit=300)
        if df_entry.empty:
            self.logger.warning("[EntryTF=%s] no data for %s", self.entry_timeframe, symbol)
            return

        # ====== 4h trendstructuur ======
        ema_fast_h4 = self._last_valid(df_h4, f"ema_{self.ema_fast}")
        ema_slow_h4 = self._last_valid(df_h4, f"ema_{self.ema_slow}")
        price_h4 = self._last_valid(df_h4, "close")
        adx_h4 = self._last_valid(df_h4, "adx")

        trend_state = "range"
        if ema_fast_h4 and ema_slow_h4 and price_h4:
            if price_h4 > ema_fast_h4 > ema_slow_h4:
                trend_state = "bull"
            elif price_h4 < ema_fast_h4 < ema_slow_h4:
                trend_state = "bear"

        # ADX 4h gate (optioneel)
        if self.use_adx_multitimeframe:
            if adx_h4 is None or adx_h4 < self.adx_high_tf_threshold:
                self.logger.info(
                    "[%s] 4h adx=%.2f<th=%.1f => skip context",
                    symbol, adx_h4 if adx_h4 is not None else float("nan"),
                    self.adx_high_tf_threshold
                )
                return

        # ====== Entry-TF checks ======
        ema_fast_e = self._last_valid(df_entry, f"ema_{self.ema_fast}")
        ema_slow_e = self._last_valid(df_entry, f"ema_{self.ema_slow}")
        price_e = self._last_valid(df_entry, "close")
        adx_e = self._last_valid(df_entry, "adx")
        di_pos_e = self._last_valid(df_entry, "di_pos")
        di_neg_e = self._last_valid(df_entry, "di_neg")
        rsi_e = self._last_valid(df_entry, "rsi")
        macd_e = self._last_valid(df_entry, "macd")

        # ADX entry gate
        if self.use_adx_filter:
            if adx_e is None or adx_e < self.adx_entry_tf_threshold:
                self.logger.info("[%s] entry ADX=%.2f<th=%.1f => skip",
                                 symbol, adx_e if adx_e is not None else float("nan"),
                                 self.adx_entry_tf_threshold)
                return

        # Richtingsfilter met DI
        if self.use_adx_directional_filter and di_pos_e is not None and di_neg_e is not None:
            if trend_state == "bull" and not (di_pos_e > di_neg_e):
                self.logger.info("[%s] bull maar +DI<=-DI (%.2f<=%.2f) => skip", symbol, di_pos_e, di_neg_e)
                return
            if trend_state == "bear" and not (di_neg_e > di_pos_e):
                self.logger.info("[%s] bear maar -DI<=+DI (%.2f<=%.2f) => skip", symbol, di_neg_e, di_pos_e)
                return

        # EMA stack (optioneel afdwingen)
        if self.require_trend_stack and ema_fast_e and ema_slow_e and price_e:
            if trend_state == "bull" and not (price_e > ema_fast_e > ema_slow_e):
                self.logger.info("[%s] bull maar geen EMA stack op entry-TF => skip", symbol)
                return
            if trend_state == "bear" and not (price_e < ema_fast_e < ema_slow_e):
                self.logger.info("[%s] bear maar geen EMA stack op entry-TF => skip", symbol)
                return

        # ---- Setup logging (nog geen orders) ----
        self.logger.info(
            "[SETUP][%s] trend=%s | 4h(adx=%.2f, ema%u=%.2f, ema%u=%.2f) | %s(adx=%.2f, +DI=%.2f, -DI=%.2f, rsi=%.1f, macd=%.3f)",
            symbol,
            trend_state,
            adx_h4 if adx_h4 is not None else float("nan"),
            self.ema_fast, ema_fast_h4 if ema_fast_h4 is not None else float("nan"),
            self.ema_slow, ema_slow_h4 if ema_slow_h4 is not None else float("nan"),
            self.entry_timeframe,
            adx_e if adx_e is not None else float("nan"),
            di_pos_e if di_pos_e is not None else float("nan"),
            di_neg_e if di_neg_e is not None else float("nan"),
            rsi_e if rsi_e is not None else float("nan"),
            macd_e if macd_e is not None else float("nan"),
        )
        # Volgende stap (Stap 5) haken we hier een _open_position in — nu nog NIET.

    # ---------- Helpers ----------
    def _fetch_and_indicator(self, symbol: str, interval: str, limit=300) -> pd.DataFrame:
        """
        Leest candles uit je DB en verrijkt met RSI/EMA/ADX/MACD.
        Robuust: alleen ADX als er genoeg candles zijn.
        """
        try:
            df = self.db_manager.fetch_data(
                table_name="candles_kraken",
                limit=limit,
                market=symbol,
                interval=interval
            )
            if not isinstance(df, pd.DataFrame) or df.empty:
                return pd.DataFrame()

            # Maak kolommen consistent
            for col in ['datetime_utc', 'exchange']:
                if col in df.columns:
                    df.drop(columns=col, inplace=True, errors='ignore')

            df.columns = ["timestamp_ms", "market", "interval", "open", "high", "low", "close", "volume"]
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float, errors="raise")

            df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df.set_index("datetime_utc", inplace=True, drop=False)
            df.sort_index(inplace=True)

            # RSI via jouw helper
            df = IndicatorAnalysis.calculate_indicators(df, rsi_window=14)

            # EMA's
            df[f"ema_{self.ema_fast}"] = df["close"].ewm(span=self.ema_fast).mean()
            df[f"ema_{self.ema_slow}"] = df["close"].ewm(span=self.ema_slow).mean()

            # MACD (optioneel nuttig voor logging)
            macd_ind = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = macd_ind.macd()
            df['macd_signal'] = macd_ind.macd_signal()

            # ADX + DI (alleen bij voldoende lengte)
            needed = self.adx_window + 5
            if len(df) >= needed:
                adx_obj = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=self.adx_window)
                df["adx"] = adx_obj.adx()
                df["di_pos"] = adx_obj.adx_pos()
                df["di_neg"] = adx_obj.adx_neg()

            return df
        except Exception as e:
            self.logger.error("[TrendStrategy4H][_fetch] error: %s", e)
            return pd.DataFrame()

    @staticmethod
    def _last_valid(df: pd.DataFrame, col: str):
        if col not in df.columns:
            return None
        s = df[col].dropna()
        if s.empty:
            return None
        try:
            return float(s.iloc[-1])
        except Exception:
            return None

