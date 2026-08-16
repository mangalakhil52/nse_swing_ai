"""
Unit tests for deterministic pattern recognition engine.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.types import PatternType
from src.quant.indicators import TechnicalIndicators
from src.quant.patterns import PatternRecognizer


def test_vcp_pattern_detection():
    # Build synthetic VCP price series (3 contracting waves: 12% -> 5% -> 2%)
    n = 60
    base_price = 1000.0

    prices = []
    # Wave 1: 1000 -> 1120 -> 1000 (12% depth)
    prices.extend(np.linspace(1000, 1120, 10))
    prices.extend(np.linspace(1120, 1010, 10))
    # Wave 2: 1010 -> 1070 -> 1020 (5% depth)
    prices.extend(np.linspace(1010, 1070, 10))
    prices.extend(np.linspace(1070, 1025, 10))
    # Wave 3: 1025 -> 1050 -> 1035 (2.5% depth - tight pivot)
    prices.extend(np.linspace(1025, 1050, 10))
    prices.extend(np.linspace(1050, 1045, 10))

    close = np.array(prices)
    high = close * 1.01
    low = close * 0.99
    open_p = (high + low) / 2.0
    # Volume contracts over time
    volume = np.concatenate([
        np.linspace(500000, 300000, 20),
        np.linspace(300000, 200000, 20),
        np.linspace(200000, 100000, 20)
    ])

    df = pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume})
    df = TechnicalIndicators.compute_all_indicators(df)

    res = PatternRecognizer.detect_vcp(df)
    assert res.is_matched is True
    assert res.pattern_type == PatternType.VOLATILITY_CONTRACTION_PATTERN
    assert res.quality_score >= 70.0


def test_flat_base_breakout_detection():
    # Build flat base: tight range between 1000 and 1040 (4% range) for 15 days, then breakout to 1055 with 2.5x volume
    n_base = 20
    close_base = np.random.uniform(1000, 1035, n_base)
    high_base = close_base + 3.0
    low_base = close_base - 3.0
    open_base = close_base - 1.0
    vol_base = np.full(n_base, 100000)

    # Breakout bar
    close_bo = [1055.0]
    high_bo = [1058.0]
    low_bo = [1038.0]
    open_bo = [1040.0]
    vol_bo = [300000]

    df = pd.DataFrame({
        "open": np.concatenate([open_base, open_bo]),
        "high": np.concatenate([high_base, high_bo]),
        "low": np.concatenate([low_base, low_bo]),
        "close": np.concatenate([close_base, close_bo]),
        "volume": np.concatenate([vol_base, vol_bo]),
    })
    df = TechnicalIndicators.compute_all_indicators(df)

    res = PatternRecognizer.detect_flat_base_breakout(df, base_window=12)
    assert res.is_matched is True
    assert res.pattern_type == PatternType.FLAT_BASE_BREAKOUT
    assert res.breakout_price > 1030.0


def test_ema_pullback_detection():
    # Strong uptrend with pullback test of 20 EMA and bullish rejection
    close = np.linspace(100, 200, 50)
    high = close + 2.0
    low = close - 2.0
    open_p = close - 1.0
    volume = np.full(50, 150000)

    df = pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume})
    df = TechnicalIndicators.compute_all_indicators(df)

    # Pull last bar low to exactly touch EMA 20 and close near top
    ema_20_val = df["ema_20"].iloc[-1]
    df.loc[49, "low"] = ema_20_val * 0.998
    df.loc[49, "open"] = ema_20_val * 1.002
    df.loc[49, "close"] = ema_20_val * 1.02
    df.loc[49, "high"] = ema_20_val * 1.025

    res = PatternRecognizer.detect_ema_pullback_reversal(df)
    assert res.is_matched is True
    assert res.pattern_type == PatternType.EMA_PULLBACK_REVERSAL
