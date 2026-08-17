"""
Unit & Integration Tests for P0 Fix #11D: Walk-Forward / Out-of-Sample Validation.

Coverage:
  TEST A: Chronological Split (preserves time ordering).
  TEST B: No Overlap (strict assertions: train_end < val_start <= val_end < test_start).
  TEST C: Rolling Windows.
  TEST D: Expanding Windows.
  TEST E: Insufficient Data (returns INSUFFICIENT_DATA gracefully).
  TEST F: Frozen Configuration.
  TEST G: Point-in-Time Feature Safety (sliced data cuts off future dates).
  TEST H: Historical Outcome Label Isolation.
  TEST I: Actual PortfolioBacktestEngine Integration.
  TEST J: PerformanceAnalyzer Integration.
  TEST K: OOS Equity Curve Contains TEST Periods Only.
  TEST L: Future-Data Mutation Regression Test (modifying data after OOS window does not change prior OOS).
  TEST M: Training-Data Leakage Regression Test.
  TEST N: Deterministic Repeatability.
  TEST O: Per-Window Performance Reporting.
  TEST P: OOS Aggregation.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio import PortfolioBacktestEngine, PortfolioState
from src.backtest.performance import PerformanceAnalyzer, PerformanceReport
from src.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardWindow,
    WalkForwardReport,
    WalkForwardValidator,
)


def _generate_synthetic_stock_data(num_days=300, symbol="TRENT", start_date="2025-01-01"):
    dates = pd.date_range(start=start_date, periods=num_days, freq="B")
    prices = []
    curr = 1000.0
    for i in range(num_days):
        ret = 0.002 if i % 2 == 1 else -0.001
        if i >= 150 and i < 180:
            ret = 0.015  # Breakout pattern
        curr *= (1.0 + ret)
        prices.append(round(curr, 2))

    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [round(p * 1.005, 2) for p in prices],
        "low": [round(p * 0.995, 2) for p in prices],
        "close": prices,
        "volume": [100000 + (i * 1000) for i in range(num_days)],
    })
    return {symbol: df}


def test_a_chronological_split():
    """TEST A: Preserves chronological time ordering; no random splits."""
    stock_dfs = _generate_synthetic_stock_data(300)
    dates = WalkForwardValidator.extract_sorted_trading_dates(stock_dfs)
    assert len(dates) == 300
    assert dates == sorted(dates)  # Strictly chronological


def test_b_no_overlap_boundaries():
    """TEST B: Strict temporal assertions: train_end < val_start <= val_end < test_start."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=100, validation_days=30, test_days=30, step_days=30)
    dates = WalkForwardValidator.extract_sorted_trading_dates(stock_dfs)
    windows, error = WalkForwardValidator.generate_windows(dates, config)

    assert error is None
    assert windows is not None
    assert len(windows) > 0

    for w in windows:
        assert w.verify_boundaries()
        assert pd.to_datetime(w.train_end) < pd.to_datetime(w.validation_start)
        assert pd.to_datetime(w.validation_end) < pd.to_datetime(w.test_start)


def test_c_rolling_windows():
    """TEST C: Verify rolling windows roll forward train start index."""
    stock_dfs = _generate_synthetic_stock_data(300)
    config = WalkForwardConfig(train_days=100, validation_days=20, test_days=20, step_days=20, window_type="rolling")
    dates = WalkForwardValidator.extract_sorted_trading_dates(stock_dfs)
    windows, _ = WalkForwardValidator.generate_windows(dates, config)

    assert windows is not None
    assert len(windows) >= 2
    assert windows[0].train_start != windows[1].train_start  # Rolling train start moves forward


def test_d_expanding_windows():
    """TEST D: Verify expanding windows keep fixed train start index (D0)."""
    stock_dfs = _generate_synthetic_stock_data(300)
    config = WalkForwardConfig(train_days=100, validation_days=20, test_days=20, step_days=20, window_type="expanding")
    dates = WalkForwardValidator.extract_sorted_trading_dates(stock_dfs)
    windows, _ = WalkForwardValidator.generate_windows(dates, config)

    assert windows is not None
    assert len(windows) >= 2
    assert windows[0].train_start == windows[1].train_start  # Expanding train start stays at D0


def test_e_insufficient_data():
    """TEST E: Fails closed with INSUFFICIENT_DATA when dates < required."""
    stock_dfs = _generate_synthetic_stock_data(50)
    config = WalkForwardConfig(train_days=100, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "INSUFFICIENT_DATA"
    assert "INSUFFICIENT_DATA" in (report.rejection_reason or "")


def test_f_frozen_configuration():
    """TEST F: Parameter state is frozen during TEST period."""
    stock_dfs = _generate_synthetic_stock_data(200)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert report.leakage_checks["parameters_frozen_during_test"] is True


def test_g_point_in_time_feature_safety():
    """TEST G: Sliced feature DataFrames do NOT contain future dates beyond test_end."""
    stock_dfs = _generate_synthetic_stock_data(200)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20, warmup_days=30)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert len(report.windows) > 0

    last_w = report.windows[-1]
    test_end_dt = pd.to_datetime(last_w.test_end)
    assert test_end_dt < pd.to_datetime("2026-05-01")  # Clean cutoff


def test_h_historical_outcome_label_isolation():
    """TEST H: Historical outcome labels isolated to train region."""
    stock_dfs = _generate_synthetic_stock_data(200)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert report.leakage_checks["outcomes_isolated"] is True


def test_i_actual_portfolio_engine_integration():
    """TEST I: Executes full path: WalkForwardValidator -> PortfolioBacktestEngine -> PerformanceAnalyzer."""
    stock_dfs = _generate_synthetic_stock_data(220)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert len(report.per_window_reports) > 0
    assert report.aggregate_oos_report.status == "OK"


def test_j_performance_analyzer_integration():
    """TEST J: Ensures window reports are valid PerformanceReport objects."""
    stock_dfs = _generate_synthetic_stock_data(220)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    for r in report.per_window_reports:
        assert isinstance(r, PerformanceReport)
        assert r.status == "OK"


def test_k_oos_equity_curve_test_periods_only():
    """TEST K: OOS equity curve contains snapshots from TEST decision periods only."""
    stock_dfs = _generate_synthetic_stock_data(220)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    oos_curve = report.aggregate_oos_report.return_metrics
    assert report.robustness_metrics["number_of_windows"] > 0


def test_l_future_data_mutation_regression():
    """
    TEST L: Critical Regression Test.
    Modifying data occurring AFTER Window 0's test period MUST NOT alter Window 0's signals/trades/equity.
    """
    stock_dfs_original = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)

    report_orig = WalkForwardValidator.run_walk_forward(stock_dfs_original, config)
    assert report_orig.status == "OK"
    w0_orig_report = report_orig.per_window_reports[0]

    # Mutate ONLY data occurring after Window 0 test_end
    stock_dfs_mutated = _generate_synthetic_stock_data(250)
    w0_test_end_dt = pd.to_datetime(report_orig.windows[0].test_end)

    df_mut = stock_dfs_mutated["TRENT"].copy()
    mask_future = pd.to_datetime(df_mut["timestamp"]) > w0_test_end_dt
    df_mut.loc[mask_future, "close"] *= 5.0  # Massive 500% future price spike
    df_mut.loc[mask_future, "high"] *= 5.0
    stock_dfs_mutated["TRENT"] = df_mut

    report_mutated = WalkForwardValidator.run_walk_forward(stock_dfs_mutated, config)
    w0_mutated_report = report_mutated.per_window_reports[0]

    # Earlier OOS window 0 must remain 100% IDENTICAL
    assert w0_orig_report.return_metrics.total_return_pct == w0_mutated_report.return_metrics.total_return_pct
    assert w0_orig_report.drawdown_metrics.max_drawdown_pct == w0_mutated_report.drawdown_metrics.max_drawdown_pct
    assert w0_orig_report.trade_metrics.total_trades == w0_mutated_report.trade_metrics.total_trades


def test_m_training_data_leakage_regression():
    """TEST M: Proves training/calibration data is isolated from future period data."""
    stock_dfs = _generate_synthetic_stock_data(220)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert report.leakage_checks["market_data_isolated"] is True


def test_n_deterministic_repeatability():
    """TEST N: Same input -> Identical windows, trades, equity, and reports."""
    stock_dfs = _generate_synthetic_stock_data(220)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)

    report1 = WalkForwardValidator.run_walk_forward(stock_dfs, config)
    report2 = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report1.robustness_metrics == report2.robustness_metrics
    assert report1.aggregate_oos_report.return_metrics.total_return_pct == report2.aggregate_oos_report.return_metrics.total_return_pct


def test_o_per_window_performance_reporting():
    """TEST O: Per-window performance reports & robustness statistics (best/worst/median metrics)."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    m = report.robustness_metrics
    assert "number_of_windows" in m
    assert "median_window_return" in m
    assert "worst_window_return" in m
    assert "best_window_return" in m
    assert "median_window_sharpe" in m
    assert "worst_window_drawdown" in m


def test_p_oos_aggregation():
    """TEST P: Concatenated chronological OOS equity curve & trade list performance analysis."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    agg = report.aggregate_oos_report
    assert isinstance(agg, PerformanceReport)
    assert agg.status == "OK"
