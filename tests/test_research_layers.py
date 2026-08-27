import pandas as pd

from src.quant.labeling import triple_barrier_label
from src.quant.monte_carlo import simulate_equity_paths
from src.quant.drift_monitor import compare_distributions
from src.risk.position_sizing import size_position


def test_triple_barrier_conservative_both_touch_is_loss():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="B"),
        "high": [105, 110],
        "low": [95, 90],
    })
    result = triple_barrier_label(frame, 100, 5, 5, 2)
    assert result["label"] == "LOSS"


def test_monte_carlo_requires_history():
    try:
        simulate_equity_paths([0.01] * 10)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected insufficient trade history to fail")


def test_drift_detects_large_distribution_shift():
    report = compare_distributions([0.0] * 100 + [1.0] * 20, [5.0] * 120)
    assert report.status == "DRIFT"


def test_position_size_respects_cap():
    sized = size_position(1_000_000, 100, 95, risk_budget_fraction=0.01, max_position_fraction=0.10)
    assert sized.capital_rupees <= 100_000
    assert sized.risk_rupees <= 10_000
