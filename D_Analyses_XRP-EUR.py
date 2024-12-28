import pandas as pd
import logging
import numpy as np


# Logger instellen
def setup_logger(log_file="app.log"):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()


logger = setup_logger()


# Functies voor indicatoren
def calculate_moving_average(data, window=50):
    return data['close'].rolling(window=window).mean()


def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal


def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band


# Functie voor het genereren van handelsignalen
def trading_signal(ticker_df):
    ticker_df['moving_average'] = calculate_moving_average(ticker_df)
    ticker_df['rsi'] = calculate_rsi(ticker_df)
    ticker_df['macd'], ticker_df['macd_signal'] = calculate_macd(ticker_df)
    ticker_df['upper_band'], ticker_df['lower_band'] = calculate_bollinger_bands(ticker_df)

    # Koopsignaal: RSI onder 30, prijs onder de lower Bollinger Band en MACD boven de Signal Line
    buy_signal = (
            ticker_df['rsi'].iloc[-1] < 30 and
            ticker_df['close'].iloc[-1] < ticker_df['lower_band'].iloc[-1] and
            ticker_df['macd'].iloc[-1] > ticker_df['macd_signal'].iloc[-1]
    )

    # Verkoopsignaal: RSI boven 70, prijs boven de upper Bollinger Band en MACD onder de Signal Line
    sell_signal = (
            ticker_df['rsi'].iloc[-1] > 70 and
            ticker_df['close'].iloc[-1] > ticker_df['upper_band'].iloc[-1] and
            ticker_df['macd'].iloc[-1] < ticker_df['macd_signal'].iloc[-1]
    )

    return buy_signal, sell_signal


# Functie om het trading-signaal te loggen
def log_trade_signal(buy_signal, sell_signal):
    if buy_signal:
        logger.info("Koopsignaal gegenereerd.")
    elif sell_signal:
        logger.info("Verkoopsignaal gegenereerd.")
    else:
        logger.info("Geen signaal gegenereerd.")


# Functie om een handel uit te voeren
def execute_trade(buy_signal, sell_signal):
    if buy_signal:
        logger.info("Plaats kooporder!")
        # Plaats kooporder via API-aanroep of andere logica
    elif sell_signal:
        logger.info("Plaats verkooporder!")
        # Plaats verkooporder via API-aanroep of andere logica


# Functie voor het ophalen van data voor XRP-EUR (je kunt hier de gegevens ophalen uit je database of API)
def fetch_xrp_eur_data():
    # Simulatie van data voor het voorbeeld (je zou echte data hier moeten ophalen)
    data = {
        'timestamp': [1609459200, 1609459260, 1609459320, 1609459380, 1609459440],
        'close': [0.22, 0.215, 0.225, 0.23, 0.22]
    }
    return pd.DataFrame(data)


# Hoofdfunctie om alles te draaien
def main():
    # Haal de XRP-EUR gegevens op
    ticker_df = fetch_xrp_eur_data()

    # Genereer handelsignalen
    buy_signal, sell_signal = trading_signal(ticker_df)

    # Log het trading-signaal
    log_trade_signal(buy_signal, sell_signal)

    # Voer de handel uit op basis van het signaal
    execute_trade(buy_signal, sell_signal)


if __name__ == "__main__":
    main()


