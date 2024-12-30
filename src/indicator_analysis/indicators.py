import pandas as pd

class IndicatorAnalysis:
    @staticmethod
    def calculate_moving_average(data, window=50):
        """Bereken het voortschrijdend gemiddelde."""
        return data['close'].rolling(window=window).mean()

    @staticmethod
    def calculate_rsi(data, window=14):
        """Bereken de Relative Strength Index (RSI)."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
        """Bereken MACD en de signaallijn."""
        fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    @staticmethod
    def calculate_bollinger_bands(data, window=20, num_std_dev=2):
        """Bereken Bollinger Bands."""
        rolling_mean = data['close'].rolling(window=window).mean()
        rolling_std = data['close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)
        return upper_band, lower_band