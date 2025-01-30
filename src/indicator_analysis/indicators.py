# src/indicator_analysis/indicators.py

import logging
import pandas as pd
import streamlit as st

from src.config.config import DB_FILE  # Importeer het pad naar de database

# Externe TA-bibliotheken
### CHANGE 1 ###
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

logger = logging.getLogger("main")
logger.debug("Imports voltooid in indicators.py.")

# =========================================================================
# 1) MARKET-KLASSE (ALLEEN VOOR DATA OPHALEN)
# =========================================================================
class Market:
    def __init__(self, symbol, db_manager):
        """
        Initialiseer de Market-klasse met een symbol (bijv. 'XRP-EUR') en een database manager.
        Deze klasse is nu alleen verantwoordelijk voor het ophalen van candles.
        """
        self.symbol = symbol
        self.db_manager = db_manager

    def fetch_candles(self, interval, limit=200):
        """
        Haal candles op uit de database, zonder indicatorberekeningen.
        """
        candles = self.db_manager.get_candlesticks(
            market=self.symbol,
            interval=interval,
            limit=limit
                )
        return candles


# =========================================================================
# 2) INDICATORANALYSIS-KLASSE (STATISCHE METHODE VOOR INDICATOREN)
# =========================================================================
class IndicatorAnalysis:
    @staticmethod
    def calculate_rsi(df, window=14):
        try:
            delta = df["close"].diff()
            if delta.empty:
                raise ValueError("Delta is leeg. Controleer de input DataFrame.")
            print(f"[DEBUG] Delta: {delta.head()}")

            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

            # Debug: Controleer gain en loss
            print(f"[DEBUG] Gain (eerste 5): {gain.head()}")
            print(f"[DEBUG] Loss (eerste 5): {loss.head()}")

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            print(f"[ERROR] RSI-berekening faalde: {e}")
            raise

    @staticmethod
    def calculate_moving_average(df, window=20):
        """Bereken het voortschrijdend gemiddelde (MA)."""
        return df['close'].rolling(window=window).mean()

    @staticmethod
    def calculate_bollinger_bands(df, window=20, num_std_dev=2):
        """Bereken Bollinger Bands."""
        rolling_mean = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
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
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    @staticmethod
    def calculate_ema(df, short_win=9, long_win=21):
        """Bereken EMA-indicators met TA-lib in plaats van handmatig."""
        ema_short = EMAIndicator(df['close'], window=short_win).ema_indicator()
        ema_long = EMAIndicator(df['close'], window=long_win).ema_indicator()
        df[f'ema_{short_win}'] = ema_short
        df[f'ema_{long_win}'] = ema_long
        return df

    @staticmethod
    def calculate_atr(df, window=14):
        """
        Bereken ATR (Average True Range), maar sla crash over
        als df minder dan 'window' rows heeft.
        """
        if len(df) < window:
            # Te weinig data => zet None in deze kolom
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
        Bereken alle indicatoren (RSI, MACD, Bollinger Bands, MA, EMA, ATR)
        en voeg ze toe in de DataFrame.
        """
        # gebruik 'rsi_window' voor RSI
        df['rsi'] = IndicatorAnalysis.calculate_rsi(df, window=rsi_window)

        if df.empty:
            return df  # als df geen rows heeft, gewoon returnen

        # moving_average
        df['moving_average'] = IndicatorAnalysis.calculate_moving_average(df, window=20)

        # Bollinger
        upper, lower = IndicatorAnalysis.calculate_bollinger_bands(df, window=20, num_std_dev=2)
        df['bollinger_upper'] = upper
        df['bollinger_lower'] = lower

        # RSI
        df['rsi'] = IndicatorAnalysis.calculate_rsi(df, window=14)

        # MACD
        macd_val, macd_sig = IndicatorAnalysis.calculate_macd(df, fast_period=12, slow_period=26, signal_period=9)
        df['macd'] = macd_val
        df['macd_signal'] = macd_sig

        # EMA(9,21)
        df = IndicatorAnalysis.calculate_ema(df, short_win=9, long_win=21)

        # ATR(14)
        df['atr14'] = IndicatorAnalysis.calculate_atr(df, window=14)

        return df

    @staticmethod
    def analyze(market, interval, limit):
        """
        Analyseer (via Streamlit) de candles van 'market' en bereken indicatoren,
        waarna het resultaat in een DataFrame wordt getoond.
        """
        logger.info(f"Analyzing indicators for market: {market.symbol}, interval: {interval}, limit: {limit}")
        st.write(f"🔍 **IndicatorAnalysis.analyze Debugging**")
        st.write(f"Markt: {market.symbol}, Interval: {interval}, Limiet: {limit}")

        # Haal candles op via de Market-klasse
        df = market.fetch_candles(interval=interval, limit=max(limit, 200))

        if df.empty:
            st.warning("⚠️ Geen candlestick data beschikbaar.")
            st.write("🔍 Controleer de query-parameters:")
            st.write(f"Markt: {market.symbol}, Interval: {interval}, Limiet: {limit}")
            logger.warning("Geen candlestick data beschikbaar voor de opgegeven parameters.")
            return pd.DataFrame()

        # Statisch indicatoren berekenen
        df = IndicatorAnalysis.calculate_indicators(df)

        # Toon resultaten in Streamlit
        st.success("✅ Candlestick data succesvol opgehaald en indicatoren berekend.")
        st.write("📋 Voorbeeld data:", df.head())
        logger.debug(f"Kolomnamen na indicator-berekening: {df.columns}")

        return df


# =========================================================================
# 3) LOSSE FUNCTIE process_indicators (OPTIONEEL)
# =========================================================================
def process_indicators(db_manager):
    df = db_manager.fetch_data(
        table_name="candles",
        limit=200,
        market="XRP-EUR",  # of wat jij nodig hebt
        interval="1m"
    )
    if df.empty:
        logger.warning("Geen candles beschikbaar in 'process_indicators' functie.")
        return

    df_with_indic = IndicatorAnalysis.calculate_indicators(df)

    # In plaats van rsi per row opslaan:
    db_manager.save_indicators(df_with_indic)

    logger.info("Indicators processed successfully (via process_indicators).")

# =========================================================================
# 4) TESTCODE ALS STANDALONE SCRIPT
# =========================================================================
if __name__ == "__main__":
    from src.database_manager.database_manager import DatabaseManager

    # Lokale test DB-manager, alleen als je dit script direct runt:
    test_db_manager = DatabaseManager(DB_FILE)
    logger.info(f"[__main__] Verbonden met database: {DB_FILE} (test mode)")

    my_market = Market(symbol="XRP-EUR", db_manager=test_db_manager)
    df_res = IndicatorAnalysis.analyze(my_market, interval="1m", limit=100)
    if not df_res.empty:
        logger.debug(f"Analyze-result shape: {df_res.shape}")

    process_indicators(test_db_manager)




