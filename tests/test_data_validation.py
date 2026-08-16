"""
Unit tests for strict market data validation and integrity engine.
"""

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import DataIntegrityError
from src.core.types import DataFreshness
from src.data.validation import DataValidator


@pytest.fixture
def sample_valid_ohlcv():
    dates = [datetime.utcnow() - timedelta(days=i) for i in range(100, 0, -1)]
    data = {
        "timestamp": dates,
        "open": np.linspace(100, 150, 100),
        "high": np.linspace(102, 153, 100),
        "low": np.linspace(99, 148, 100),
        "close": np.linspace(101, 151, 100),
        "volume": np.random.randint(100000, 500000, size=100),
        "delivery_volume": np.random.randint(50000, 250000, size=100),
        "delivery_pct": np.random.uniform(40.0, 70.0, size=100),
    }
    return pd.DataFrame(data)


def test_valid_ohlcv_dataframe(sample_valid_ohlcv):
    validator = DataValidator(min_required_bars=60)
    res = validator.validate_ohlcv_dataframe(sample_valid_ohlcv, "TRENT")
    assert res.is_valid is True
    assert len(res.errors) == 0
    assert res.bars_checked == 100
    assert res.freshness in [DataFreshness.LIVE, DataFreshness.RECENT]


def test_invalid_ohlc_geometry(sample_valid_ohlcv):
    # Break high < low on bar 10
    sample_valid_ohlcv.loc[10, "high"] = sample_valid_ohlcv.loc[10, "low"] - 5.0
    validator = DataValidator()
    res = validator.validate_ohlcv_dataframe(sample_valid_ohlcv, "TRENT")
    assert res.is_valid is False
    assert any("High < max" in e for e in res.errors)


def test_negative_price_detection(sample_valid_ohlcv):
    sample_valid_ohlcv.loc[5, "close"] = -10.0
    validator = DataValidator()
    res = validator.validate_ohlcv_dataframe(sample_valid_ohlcv, "TRENT")
    assert res.is_valid is False
    assert any("non-positive prices" in e for e in res.errors)


def test_insufficient_bars_rejection(sample_valid_ohlcv):
    short_df = sample_valid_ohlcv.iloc[:30].copy()
    validator = DataValidator(min_required_bars=60)
    res = validator.validate_ohlcv_dataframe(short_df, "TRENT")
    assert res.is_valid is False
    assert any("Insufficient historical bars" in e for e in res.errors)


def test_enforce_valid_dataframe_raises(sample_valid_ohlcv):
    sample_valid_ohlcv.loc[0, "open"] = np.nan
    validator = DataValidator()
    with pytest.raises(DataIntegrityError):
        validator.enforce_valid_dataframe(sample_valid_ohlcv, "TRENT")
