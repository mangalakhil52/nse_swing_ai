import numpy as np
import pandas as pd

from src.quant.regime_transition import compute_regime_transition


def test_regime_transition_is_bounded():
    n = 120
    close = 100 * np.exp(np.cumsum(np.full(n, 0.001)))
    benchmark = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=n, freq="B"), "close": close})
    breadth = pd.Series(np.linspace(40, 70, n))
    vix = pd.Series(np.linspace(20, 15, n))
    result = compute_regime_transition(benchmark, breadth, vix)
    assert -1 <= result.transition_score <= 1
