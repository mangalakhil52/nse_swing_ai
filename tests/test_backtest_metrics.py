from src.quant.backtest_metrics import max_drawdown, profit_factor, sharpe, sortino


def test_metrics_basic():
    r = [0.02, -0.01, 0.03, -0.005, 0.01]
    assert max_drawdown(r) < 0
    assert sharpe(r) > 0
    assert sortino(r) > 0
    assert profit_factor(r) > 1
