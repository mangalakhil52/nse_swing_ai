"""
Unit tests for Phase 5: Backtest Engine and Indian Friction Model.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.friction import IndianFrictionModel


def test_indian_friction_model():
    costs = IndianFrictionModel.calculate_round_trip(
        entry_price=1000.0,
        exit_price=1200.0,
        shares=100,
    )
    assert costs.total_cost_rupees > 0
    assert costs.entry_stt > 0
    assert costs.exit_stt > 0
    # Total cost should be < 1% of entry value for delivery equity
    assert costs.total_cost_pct < 1.0


def test_backtest_engine_winning_trade():
    n = 60
    close = np.linspace(1000, 1300, n)
    high = close + 10.0
    low = close - 10.0
    open_p = close - 5.0
    volume = np.full(n, 500000)

    df = pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume})

    trade = BacktestEngine.simulate_trade(
        symbol="TRENT",
        df=df,
        entry_date_idx=5,
        entry_price=1050.0,
        stop_loss=990.0,  # 5.7% stop
        target_1=1158.0,
        target_2=1218.0,
        target_3=1320.0,
        shares=20,
    )

    assert trade.exit_reason is not None
    assert trade.pnl_rupees is not None
    assert trade.holding_sessions >= 1


def test_backtest_stop_loss_hit():
    n = 60
    # Downtrend — price falls below stop
    close = np.linspace(1000, 700, n)
    high = close + 5.0
    low = close - 15.0
    open_p = close - 2.0
    volume = np.full(n, 500000)

    df = pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume})

    trade = BacktestEngine.simulate_trade(
        symbol="WEAKCO",
        df=df,
        entry_date_idx=2,
        entry_price=980.0,
        stop_loss=940.0,
        target_1=1050.0,
        target_2=1100.0,
        target_3=1200.0,
        shares=10,
    )

    assert "STOP" in (trade.exit_reason or "")
    assert (trade.pnl_rupees or 0.0) < 0.0


def test_full_strategy_backtest():
    np.random.seed(99)
    n = 200
    close = np.cumprod(1.0 + np.random.normal(0.0005, 0.015, n)) * 500.0
    high = close * 1.01
    low = close * 0.99
    open_p = (high + low) / 2.0

    df = pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": np.full(n, 500000)})

    signal_df = pd.DataFrame([
        {"entry_idx": 10, "entry_price": close[10], "stop_loss": close[10] * 0.94,
         "target_1": close[10] * 1.10, "target_2": close[10] * 1.17, "target_3": close[10] * 1.28, "shares": 20},
        {"entry_idx": 50, "entry_price": close[50], "stop_loss": close[50] * 0.94,
         "target_1": close[50] * 1.10, "target_2": close[50] * 1.17, "target_3": close[50] * 1.28, "shares": 20},
        {"entry_idx": 100, "entry_price": close[100], "stop_loss": close[100] * 0.94,
         "target_1": close[100] * 1.10, "target_2": close[100] * 1.17, "target_3": close[100] * 1.28, "shares": 20},
    ])

    result = BacktestEngine.run_strategy_backtest(signal_df, df, "TEST_STRAT")
    assert result.total_trades == 3
    assert 0.0 <= result.win_rate_pct <= 100.0
    assert result.profit_factor >= 0.0
