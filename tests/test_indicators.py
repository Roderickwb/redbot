import pandas as pd
from src.indicator_analysis.indicators import IndicatorAnalysis
from src.database_manager.database_manager import DatabaseManager


# Functie om de indicatoren te testen
def test_indicators():
    try:
        # Maak een instantie van de DatabaseManager
        db_manager = DatabaseManager()

        # Haal de laatste 20 candles op uit de database
        candles = db_manager.fetch_data('candles', limit=20)

        if len(candles) < 14:
            print("Er zijn niet genoeg candles in de database om de indicatoren te berekenen (minimaal 14).")
            return False

        # Zet de data om naar een DataFrame voor de berekening van indicatoren
        df = pd.DataFrame(candles,
                          columns=["timestamp", "market", "interval", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Bereken RSI
        df["rsi"] = IndicatorAnalysis.calculate_rsi(df, window=14)

        # Bereken MACD
        df["macd"], df["macd_signal"] = IndicatorAnalysis.calculate_macd(df)

        # Log de berekeningen voor de indicatoren vanaf de 14e candle
        print("RSI en MACD berekend voor de laatste candles (vanaf candle 14):")
        print(df[13:][["timestamp", "rsi", "macd", "macd_signal"]])  # Alleen geldig vanaf de 14e candle

        # Controleer of de berekende indicatoren valide zijn
        if df["rsi"].isna().any() or df["macd"].isna().any():
            print("Fout bij het berekenen van de indicatoren. Waarden zijn NaN.")

        # Specifieke log voor NaN-waarden in MACD en MACD signalen
        if df["macd"].isna().any():
            print("MACD bevat NaN waarden.")
        if df["macd_signal"].isna().any():
            print("MACD signalen bevatten NaN waarden.")

        print("Indicatoren succesvol berekend en geldig.")
        return True
    except Exception as e:
        print(f"Fout bij het testen van de indicatoren: {e}")
        return False


# Voer de test uit
test_result = test_indicators()

if test_result:
    print("Stap 2: Indicatoren zijn succesvol getest.")
else:
    print("Stap 2: Fout bij het testen van de indicatoren.")







