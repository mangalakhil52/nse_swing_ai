import numpy as np
import pandas as pd

from src.core.types import MarketRegime
from src.quant.advanced_alpha import compute_alpha_features, cross_sectional_zscores
from src.quant.meta_label import empirical_meta_label, wilson_lower_bound


def _ohlcv(n=180, drift=0.001):
    close = 100 * np.exp(np.cumsum(np.full(n, drift)))
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="B"),
        "open": close * .999,
        "high": close * 1.01,
        "low": close * .99,
        "close": close,
        "volume": np.full(n, 100000.0),
    })


def test_advanced_alpha_is_finite():
    features = compute_alpha_features(_ohlcv())
    assert np.isfinite(features.alpha_score)
    assert features.momentum_63 > 0


def test_cross_sectional_zscores_winsorize_extremes():
    z = cross_sectional_zscores({"A": 1, "B": 2, "C": 3, "D": 4, "E": 1000})
    assert all(abs(v) <= 3 for v in z.values())


def test_meta_label_requires_enough_history():
    label = empirical_meta_label(np.ones(20), 5, 2)
    assert label.status == "UNAVAILABLE"
    assert label.probability is None


def test_meta_label_requires_conservative_positive_ev():
    outcomes = np.array([1] * 80 + [-1] * 20)
    label = empirical_meta_label(outcomes, 4, 2)
    assert label.sample_size == 100
    assert label.probability == 0.8
    assert label.status == "TRADEABLE"
    assert wilson_lower_bound(80, 100) < 0.8
