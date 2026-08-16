"""
Vectorized Technical Indicators Engine Module.
Calculates high-precision moving averages, momentum oscillators, volatility bands, and volume ratios using NumPy & Pandas.
Avoids external C-library dependency issues and guarantees deterministic mathematical outputs.
"""

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Vectorized mathematical indicator computation engine."""

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculates Exponential Moving Average (EMA)."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """Calculates Simple Moving Average (SMA)."""
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculates Relative Strength Index (RSI) using Wilder's exponential smoothing method.
        """
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        # Wilder's smoothing: alpha = 1 / period
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series]:
        """
        Calculates Average True Range (ATR) and ATR% (ATR / Close * 100).
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        atr_pct = (atr / close) * 100.0
        return atr.fillna(0.0), atr_pct.fillna(0.0)

    @staticmethod
    def calculate_macd(
        close: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculates MACD line, signal line, and histogram.
        """
        fast_ema = close.ewm(span=fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculates Average Directional Index (ADX), +DI, and -DI.
        """
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        plus_dm = (high - prev_high).clip(lower=0.0)
        minus_dm = (prev_low - low).clip(lower=0.0)

        # When +DM <= -DM, +DM is 0; when -DM <= +DM, -DM is 0
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        plus_di = 100.0 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr)
        minus_di = 100.0 * (pd.Series(minus_dm, index=high.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr)

        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))
        adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        return adx.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)

    @staticmethod
    def calculate_bollinger_bands(
        close: pd.Series, period: int = 20, num_std: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculates Bollinger Middle, Upper, Lower bands, and Bandwidth%.
        """
        sma = close.rolling(window=period, min_periods=1).mean()
        std = close.rolling(window=period, min_periods=1).std().fillna(0.0)
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        bandwidth_pct = ((upper - lower) / sma.replace(0.0, np.nan)) * 100.0
        return upper, sma, lower, bandwidth_pct.fillna(0.0)

    @staticmethod
    def calculate_relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculates Relative Volume (RVol) against 20-day Volume SMA.
        """
        vol_sma = volume.rolling(window=period, min_periods=1).mean()
        rvol = volume / vol_sma.replace(0.0, np.nan)
        return rvol.fillna(1.0)

    @staticmethod
    def calculate_distance_from_52w_high(high: pd.Series, close: pd.Series, period: int = 250) -> pd.Series:
        """
        Calculates percentage distance from 52-week High ((52W_High - Close) / 52W_High * 100).
        """
        high_52w = high.rolling(window=period, min_periods=20).max()
        dist_pct = ((high_52w - close) / high_52w.replace(0.0, np.nan)) * 100.0
        return dist_pct.fillna(0.0)

    @classmethod
    def compute_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches a raw OHLCV DataFrame with all standard quantitative indicators.
        """
        if df.empty or len(df) < 5:
            return df

        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Moving Averages
        df["ema_9"] = cls.calculate_ema(close, 9)
        df["ema_20"] = cls.calculate_ema(close, 20)
        df["ema_50"] = cls.calculate_ema(close, 50)
        df["ema_200"] = cls.calculate_ema(close, 200)
        df["sma_20"] = cls.calculate_sma(close, 20)
        df["sma_50"] = cls.calculate_sma(close, 50)
        df["sma_200"] = cls.calculate_sma(close, 200)

        # Momentum Oscillators
        df["rsi_14"] = cls.calculate_rsi(close, 14)
        df["macd"], df["macd_signal"], df["macd_hist"] = cls.calculate_macd(close, 12, 26, 9)
        df["adx_14"], df["plus_di"], df["minus_di"] = cls.calculate_adx(high, low, close, 14)

        # Volatility
        df["atr_14"], df["atr_pct"] = cls.calculate_atr(high, low, close, 14)
        df["bb_upper"], df["bb_middle"], df["bb_lower"], df["bb_bandwidth"] = cls.calculate_bollinger_bands(close, 20, 2.0)

        # Volume Profile
        df["volume_sma_20"] = cls.calculate_sma(volume, 20)
        df["rvol_20"] = cls.calculate_relative_volume(volume, 20)

        # Structure
        df["distance_52w_high_pct"] = cls.calculate_distance_from_52w_high(high, close, 250)

        return df
