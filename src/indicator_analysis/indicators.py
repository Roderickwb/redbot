# ============================================================
# src/indicator_analysis/indicators.py
# ============================================================

import logging
import pandas as pd
import streamlit as st  # Indien je Streamlit gebruikt

from src.config.config import DB_FILE  # (eventueel gebruikt in tests)
from datetime import datetime, timedelta, timezone

# Externe TA-bibliotheken
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# Maak een dedicated logger voor de indicator module
logger = logging.getLogger("indicator_analysis")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # Voeg een console handler toe als er nog geen handlers zijn
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logger.debug("Imports voltooid in indicators.py.")


# ============================================================
# 1) MARKET-KLASSE (voor data-ophalen)
# ============================================================
class Market:
    def __init__(self, symbol, db_manager):
        """
        Market-klasse leest candles uit de database voor een gegeven symbool.
        """
        self.symbol = symbol
        self.db_manager = db_manager

    def fetch_candles(self, interval, limit=200):
        """
        Haal candles op uit de database (zonder indicatorberekening).
        """
        candles = self.db_manager.get_candlesticks(
            market=self.symbol,
            interval=interval,
            limit=limit
        )
        logger.debug(f"[Market] {self.symbol} ({interval}): {len(candles)} candles opgehaald.")
        return candles


# ============================================================
# 2) INDICATORANALYSIS-KLASSE (statische methodes voor TA)
# ============================================================
class IndicatorAnalysis:
    @staticmethod
    def calculate_rsi(df, window=14):
        """
        Handmatige RSI-berekening.
        """
        try:
            delta = df["close"].diff()
            if delta.empty:
                raise ValueError("Delta is leeg. Controleer de input DataFrame.")
            logger.debug(f"[RSI] Delta (head): {delta.head()}")

            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

            logger.debug(f"[RSI] Gain (head): {gain.head()}")
            logger.debug(f"[RSI] Loss (head): {loss.head()}")

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"[RSI] Berekening mislukt: {e}")
            raise

    @staticmethod
    def calculate_moving_average(df, window=20):
        """Bereken een eenvoudig voortschrijdend gemiddelde (MA)."""
        return df['close'].rolling(window=window).mean()

    @staticmethod
    def calculate_bollinger_bands(df, window=20, num_std_dev=2):
        """
        Retourneer een dictionary met {'bb_upper': ..., 'bb_lower': ...}.
        """
        rolling_mean = df["close"].rolling(window=window).mean()
        rolling_std = df["close"].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)
        return {
            "bb_upper": upper_band,
            "bb_lower": lower_band
        }

    @staticmethod
    def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
        """Bereken MACD en de signaallijn."""
        fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd_val = fast_ema - slow_ema
        signal = macd_val.ewm(span=signal_period, adjust=False).mean()
        return macd_val, signal

    @staticmethod
    def calculate_ema(df, short_win=9, long_win=21):
        """Bereken EMA-indicatoren met de TA-bibliotheek."""
        ema_short = EMAIndicator(df['close'], window=short_win).ema_indicator()
        ema_long = EMAIndicator(df['close'], window=long_win).ema_indicator()
        df[f'ema_{short_win}'] = ema_short
        df[f'ema_{long_win}'] = ema_long
        return df

    @staticmethod
    def calculate_atr(df, window=14):
        """
        Bereken ATR (Average True Range) met ta.volatility.AverageTrueRange.
        Geeft een pandas Series terug.
        """
        if len(df) < window:
            return pd.Series([None] * len(df), index=df.index)
        atr_obj = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=window
        )
        return atr_obj.average_true_range()

    @staticmethod
    def calculate_indicators(df, rsi_window=14):
        """
        Voeg RSI, moving average, Bollinger Bands, MACD, EMA(9,21) en ATR(14) toe
        als kolommen aan de DataFrame.
        """
        if df.empty:
            logger.warning("[calculate_indicators] Input DataFrame is leeg.")
            return df

        # RSI
        df['rsi'] = IndicatorAnalysis.calculate_rsi(df, window=rsi_window)
        # Moving average
        df['moving_average'] = IndicatorAnalysis.calculate_moving_average(df, window=20)
        # Bollinger Bands
        bb = IndicatorAnalysis.calculate_bollinger_bands(df, window=20, num_std_dev=2)
        df['bollinger_upper'] = bb['bb_upper']
        df['bollinger_lower'] = bb['bb_lower']
        # MACD
        macd_val, macd_sig = IndicatorAnalysis.calculate_macd(df, fast_period=12, slow_period=26, signal_period=9)
        df['macd'] = macd_val
        df['macd_signal'] = macd_sig
        # EMA(9,21)
        df = IndicatorAnalysis.calculate_ema(df, short_win=9, long_win=21)
        # ATR(14)
        df['atr14'] = IndicatorAnalysis.calculate_atr(df, window=14)

        logger.debug(f"[calculate_indicators] Berekening afgerond. Kolommen: {df.columns.tolist()}")
        return df

    @staticmethod
    def analyze(market, interval, limit):
        """
        Voorbeeldfunctie voor Streamlit: laad candles, bereken indicatoren en toon resultaat.
        """
        logger.info(f"Analyzing indicators for {market.symbol}, interval={interval}, limit={limit}")
        st.write(f"🔍 **IndicatorAnalysis.analyze** -> Market={market.symbol}, Interval={interval}, Limit={limit}")

        # Haal candles op via de Market-klasse
        df = market.fetch_candles(interval=interval, limit=max(limit, 200))
        if df.empty:
            st.warning("⚠️ Geen candlestick data.")
            logger.warning("Geen candlestick data.")
            return pd.DataFrame()

        # Zorg dat de data gesorteerd is op timestamp (oplopend)
        df.sort_values("timestamp", inplace=True)
        logger.debug(f"Ruwe data (gesorteerd) voor {market.symbol}: {df.tail(3)}")

        # Bereken indicatoren
        df = IndicatorAnalysis.calculate_indicators(df)
        st.success("✅ Candlestick data opgehaald + indicators berekend.")
        st.write("📋 Voorbeeld (head):", df.head())
        logger.debug(f"Kolommen na indicator-berekening: {df.columns.tolist()}")
        return df


# ============================================================
# 3) LOSSE FUNCTIE process_indicators (OPTIONEEL)
# ============================================================
def process_indicators(db_manager):
    """
    Voorbeeld: haal candles op uit de 'candles'-tabel, bereken indicatoren en sla ze op in de 'indicators'-tabel.
    """
    df = db_manager.fetch_data(
        table_name="candles",
        limit=200,
        market="XRP-EUR",
        interval="1m"
    )
    if df.empty:
        logger.warning("Geen candles in 'process_indicators'.")
        return

    df_with_indic = IndicatorAnalysis.calculate_indicators(df)
    db_manager.save_indicators(df_with_indic)
    logger.info("[process_indicators] Indicatoren berekend en opgeslagen.")


# ============================================================
# 4) TESTCODE ALS STANDALONE SCRIPT
# ============================================================
if __name__ == "__main__":
    from src.database_manager.database_manager import DatabaseManager

    # Maak een test DatabaseManager aan (gebruik eventueel DB_FILE)
    test_db_manager = DatabaseManager(DB_FILE)
    logger.info(f"[__main__] Verbonden met {DB_FILE} (test mode)")

    # Test de Market-klasse en indicatoranalyse voor XRP-EUR
    my_market = Market(symbol="XRP-EUR", db_manager=test_db_manager)
    df_res = IndicatorAnalysis.analyze(my_market, interval="1m", limit=100)
    if not df_res.empty:
        logger.debug(f"Analyze-resultaat shape: {df_res.shape}")

    # Test process_indicators (optioneel)
    process_indicators(test_db_manager)
