import numpy as np
import pandas as pd

from src.backtest.walk_forward import WalkForwardConfig
from src.research.strategy_comparison import ExperimentConfig, walk_forward_compare


def test_oos_comparison_has_strict_windows_even_when_no_setup_fires():
    n = 900
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    close = 100 + np.cumsum(np.full(n, 0.01))
    frame = pd.DataFrame({
        "timestamp": dates,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.full(n, 100000.0),
    })
    result = walk_forward_compare(
        {"TEST": frame},
        frame[["timestamp", "close"]],
        WalkForwardConfig(train_days=504, validation_days=126, test_days=126, step_days=126),
        ExperimentConfig(),
    )
    assert result["leakage_contract"] == "STRICT_CHRONOLOGICAL_OOS"
    assert len(result["windows"]) >= 1
    assert result["windows"][0]["test_start"] > result["windows"][0]["train_end"]
