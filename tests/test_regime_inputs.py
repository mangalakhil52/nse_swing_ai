import pandas as pd

from src.quant.regime_inputs import compute_breadth, latest_vix


def _series(closes):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=len(closes), freq="D"),
        "close": closes,
    })


def test_compute_breadth_is_point_in_time_and_deterministic():
    up = _series([100] * 50 + [101])
    down = _series([100] * 50 + [99])
    result = compute_breadth({"UP": up, "DOWN": down}, "2026-02-21")
    assert result == (1.0, 100.0)


def test_latest_vix_uses_as_of_boundary():
    vix = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "close": [12.0, 14.0, 25.0],
    })
    assert latest_vix(vix, "2026-01-02") == 14.0


def test_latest_vix_rejects_invalid_value():
    vix = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01"]), "close": [0.0]})
    try:
        latest_vix(vix, "2026-01-01")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid VIX observation to fail closed")
