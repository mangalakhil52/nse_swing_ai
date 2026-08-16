"""
Unit tests for vectorized technical indicator computation engine.
"""

import numpy as np
import pandas as pd
import pytest

from src.quant.indicators import TechnicalIndicators


@pytest.fixture
def mock_price_df():
    np.random.seed(42)
    n = 100
    close = np.cumprod(1.0 + np.random.normal(0.001, 0.015, n)) * 1000.0
    high = close * (1.0 + np.random.uniform(0.005, 0.02, n))
    low = close * (1.0 - np.random.uniform(0.005, 0.02, n))
    open_p = (high + low) / 2.0
    volume = np.random.randint(100000, 500000, n)

    return pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_ema_and_sma_calculations(mock_price_df):
    close = mock_price_df["close"]
    ema_20 = TechnicalIndicators.calculate_ema(close, 20)
    sma_20 = TechnicalIndicators.calculate_sma(close, 20)

    assert len(ema_20) == len(close)
    assert len(sma_20) == len(close)
    assert not ema_20.isnull().any()
    assert not sma_20.isnull().any()
    assert ema_20.iloc[-1] > 0.0


def test_rsi_bounds(mock_price_df):
    close = mock_price_df["close"]
    rsi = TechnicalIndicators.calculate_rsi(close, 14)

    assert len(rsi) == len(close)
    assert (rsi >= 0.0).all() and (rsi <= 100.0).all()


def test_atr_calculation(mock_price_df):
    high = mock_price_df["high"]
    low = mock_price_df["low"]
    close = mock_price_df["close"]

    atr, atr_pct = TechnicalIndicators.calculate_atr(high, low, close, 14)
    assert (atr >= 0.0).all()
    assert (atr_pct >= 0.0).all()
    assert (atr_pct <= 20.0).all()


def test_macd_and_adx(mock_price_df):
    high = mock_price_df["high"]
    low = mock_price_df["low"]
    close = mock_price_df["close"]

    macd, signal, hist = TechnicalIndicators.calculate_macd(close)
    assert len(macd) == len(close)
    assert len(hist) == len(close)

    adx, plus_di, minus_di = TechnicalIndicators.calculate_adx(high, low, close, 14)
    assert (adx >= 0.0).all()
    assert (plus_di >= 0.0).all()
    assert (minus_di >= 0.0).all()


def test_compute_all_indicators(mock_price_df):
    enriched = TechnicalIndicators.compute_all_indicators(mock_price_df)
    expected_cols = [
        "ema_20", "ema_50", "ema_200", "rsi_14", "atr_14", "atr_pct",
        "macd", "adx_14", "rvol_20", "distance_52w_high_pct"
    ]
    for col in expected_cols:
        assert col in enriched.columns
