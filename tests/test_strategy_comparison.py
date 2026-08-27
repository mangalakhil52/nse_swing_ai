import pandas as pd

from src.research.strategy_comparison import ExperimentConfig, generate_signal_sets


def test_strategy_comparison_module_has_explicit_experiment_contract():
    cfg = ExperimentConfig()
    assert cfg.min_history_bars == 130
    assert cfg.min_pattern_quality == 75.0
    assert cfg.min_alpha_score == 0.10


def test_strategy_comparison_fails_closed_when_history_is_insufficient():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="B"),
        "open": [100.0] * 20,
        "high": [101.0] * 20,
        "low": [99.0] * 20,
        "close": [100.0] * 20,
        "volume": [100000.0] * 20,
    })
    baseline, enhanced = generate_signal_sets({"TEST": frame}, frame[["timestamp", "close"]], ExperimentConfig())
    assert baseline == []
    assert enhanced == []
