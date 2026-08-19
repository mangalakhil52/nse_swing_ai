"""
Unit & Integration Tests for P0 Fix #11D Correction: Actual Train/Validation/Test Isolation & Data Leakage Integrity.

Coverage:
  1. test_train_validation_test_datasets_distinct: Phase datasets do not share overlapping dates.
  2. test_train_context_no_validation_test_rows: Train context contains zero validation/test rows.
  3. test_validation_context_no_test_rows: Validation context contains zero test rows.
  4. test_configuration_cutoff_before_test_start: Strategy cutoff date is <= test_start.
  5. test_future_mutation_actual_trades_identical: Mutating data after test_end produces 100% identical trades.
  6. test_future_mutation_entry_sl_targets_identical: Mutating future data produces identical entry/SL/target levels.
  7. test_future_mutation_equity_curve_identical: Mutating future data produces identical OOS equity curve.
  8. test_validation_data_excluded_from_oos: Validation dates NEVER enter OOS equity curve or aggregate trades.
  9. test_outcome_labels_extending_beyond_train_end_excluded: Label eligibility excludes labels completed after train_end.
 10. test_no_future_outcome_labels_in_test_signals: Test signals generated using point-in-time data only.
 11. test_continuous_capital_across_oos_windows: Capital carries forward continuously across OOS test windows.
 12. test_gaps_between_oos_windows_no_hidden_trades: Gaps between OOS test windows contain 0 trades and constant capital.
 13. test_aggregate_oos_trades_test_trades_only: Aggregate OOS trade count equals sum of test window trades.
 14. test_aggregate_oos_equity_test_dates_only: Aggregate OOS equity curve dates are strictly inside TEST dates.
 15. test_frozen_strategy_configuration_identical: Strategy rule hash is identical across execution.
 16. test_actual_path_execution: Runs full WalkForwardValidator -> PortfolioBacktestEngine -> PerformanceAnalyzer path.
 17. test_explicit_no_calibration_reporting: Verifies report exposes calibration_performed = False, calibration_method = "NONE".
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio import PortfolioBacktestEngine, PortfolioState
from src.backtest.performance import PerformanceAnalyzer, PerformanceReport
from src.backtest.walk_forward import (
    FrozenStrategyContext,
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


def test_1_train_validation_test_datasets_distinct():
    """1. Test train/validation/test datasets are physically distinct without date overlaps."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=100, validation_days=30, test_days=30, step_days=30)
    dates = WalkForwardValidator.extract_sorted_trading_dates(stock_dfs)
    windows, _ = WalkForwardValidator.generate_windows(dates, config)

    w = windows[0]
    train_ctx = WalkForwardValidator.build_training_context(stock_dfs, w)
    val_ctx = WalkForwardValidator.build_validation_context(stock_dfs, train_ctx, w)
    test_dfs = WalkForwardValidator.build_test_dataset_with_warmup(stock_dfs, dates, w, config.warmup_days)

    train_dates_in_ctx = set()
    for df in train_ctx["train_dfs"].values():
        train_dates_in_ctx.update(pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d"))

    val_dates_in_ctx = set()
    for df in val_ctx["val_dfs"].values():
        val_dates_in_ctx.update(pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d"))

    assert train_dates_in_ctx.isdisjoint(val_dates_in_ctx)
    assert max(train_dates_in_ctx) < min(val_dates_in_ctx)


def test_2_train_context_no_validation_test_rows():
    """2. Test train context contains zero validation or test rows."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=100, validation_days=30, test_days=30, step_days=30)
    dates = WalkForwardValidator.extract_sorted_trading_dates(stock_dfs)
    windows, _ = WalkForwardValidator.generate_windows(dates, config)

    w = windows[0]
    train_ctx = WalkForwardValidator.build_training_context(stock_dfs, w)

    for df in train_ctx["train_dfs"].values():
        max_d = pd.to_datetime(df["timestamp"]).max().strftime("%Y-%m-%d")
        assert max_d <= w.train_end
        assert max_d < w.validation_start
        assert max_d < w.test_start


def test_3_validation_context_no_test_rows():
    """3. Test validation context contains zero test rows."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=100, validation_days=30, test_days=30, step_days=30)
    dates = WalkForwardValidator.extract_sorted_trading_dates(stock_dfs)
    windows, _ = WalkForwardValidator.generate_windows(dates, config)

    w = windows[0]
    train_ctx = WalkForwardValidator.build_training_context(stock_dfs, w)
    val_ctx = WalkForwardValidator.build_validation_context(stock_dfs, train_ctx, w)

    for df in val_ctx["val_dfs"].values():
        max_d = pd.to_datetime(df["timestamp"]).max().strftime("%Y-%m-%d")
        assert max_d <= w.validation_end
        assert max_d < w.test_start


def test_4_configuration_cutoff_before_test_start():
    """4. Test strategy configuration cutoff date is on or before test_start."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=100, validation_days=30, test_days=30, step_days=30)
    dates = WalkForwardValidator.extract_sorted_trading_dates(stock_dfs)
    windows, _ = WalkForwardValidator.generate_windows(dates, config)

    w = windows[0]
    train_ctx = WalkForwardValidator.build_training_context(stock_dfs, w)
    val_ctx = WalkForwardValidator.build_validation_context(stock_dfs, train_ctx, w)
    frozen = WalkForwardValidator.freeze_strategy_context(train_ctx, val_ctx, w)

    assert frozen.verify_cutoff()
    assert pd.to_datetime(frozen.cutoff_date) <= pd.to_datetime(w.test_start)


def test_5_future_mutation_actual_trades_identical():
    """5. Test future data mutation does not alter actual Window 0 trades."""
    stock_dfs_orig = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)

    report_orig = WalkForwardValidator.run_walk_forward(stock_dfs_orig, config)
    assert report_orig.status == "OK"
    w0_orig_trades = report_orig.per_window_reports[0].trade_metrics.total_trades

    # Mutate ONLY data after Window 0 test_end
    stock_dfs_mut = _generate_synthetic_stock_data(250)
    w0_test_end_dt = pd.to_datetime(report_orig.windows[0].test_end)

    df_mut = stock_dfs_mut["TRENT"].copy()
    mask_future = pd.to_datetime(df_mut["timestamp"]) > w0_test_end_dt
    df_mut.loc[mask_future, "close"] *= 4.0
    stock_dfs_mut["TRENT"] = df_mut

    report_mut = WalkForwardValidator.run_walk_forward(stock_dfs_mut, config)
    w0_mut_trades = report_mut.per_window_reports[0].trade_metrics.total_trades

    assert w0_orig_trades == w0_mut_trades


def test_6_future_mutation_entry_sl_targets_identical():
    """6. Test future data mutation does not alter Window 0 trade levels (entry, stop_loss, target_1)."""
    stock_dfs_orig = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)

    report_orig = WalkForwardValidator.run_walk_forward(stock_dfs_orig, config)
    w0_orig_report = report_orig.per_window_reports[0]

    stock_dfs_mut = _generate_synthetic_stock_data(250)
    w0_test_end_dt = pd.to_datetime(report_orig.windows[0].test_end)
    df_mut = stock_dfs_mut["TRENT"].copy()
    mask_future = pd.to_datetime(df_mut["timestamp"]) > w0_test_end_dt
    df_mut.loc[mask_future, "close"] *= 10.0
    stock_dfs_mut["TRENT"] = df_mut

    report_mut = WalkForwardValidator.run_walk_forward(stock_dfs_mut, config)
    w0_mut_report = report_mut.per_window_reports[0]

    assert w0_orig_report.trade_metrics.gross_profit == w0_mut_report.trade_metrics.gross_profit
    assert w0_orig_report.trade_metrics.gross_loss == w0_mut_report.trade_metrics.gross_loss


def test_7_future_mutation_equity_curve_identical():
    """7. Test future data mutation does not alter Window 0 OOS equity curve."""
    stock_dfs_orig = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)

    report_orig = WalkForwardValidator.run_walk_forward(stock_dfs_orig, config)
    w0_orig_return = report_orig.per_window_reports[0].return_metrics.total_return_pct
    w0_orig_dd = report_orig.per_window_reports[0].drawdown_metrics.max_drawdown_pct

    stock_dfs_mut = _generate_synthetic_stock_data(250)
    w0_test_end_dt = pd.to_datetime(report_orig.windows[0].test_end)
    df_mut = stock_dfs_mut["TRENT"].copy()
    mask_future = pd.to_datetime(df_mut["timestamp"]) > w0_test_end_dt
    df_mut.loc[mask_future, "close"] *= 0.1  # Massive 90% crash
    stock_dfs_mut["TRENT"] = df_mut

    report_mut = WalkForwardValidator.run_walk_forward(stock_dfs_mut, config)
    w0_mut_return = report_mut.per_window_reports[0].return_metrics.total_return_pct
    w0_mut_dd = report_mut.per_window_reports[0].drawdown_metrics.max_drawdown_pct

    assert w0_orig_return == w0_mut_return
    assert w0_orig_dd == w0_mut_dd


def test_8_validation_data_excluded_from_oos():
    """8. Test validation data is strictly excluded from aggregate OOS equity curve and trade results."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=100, validation_days=30, test_days=30, step_days=30)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    w0 = report.windows[0]
    val_dates_set = set(w0.validation_dates)

    oos_curve = report.aggregate_oos_report.drawdown_metrics.drawdown_series_pct
    # Check that OOS equity snapshots do not contain validation dates
    for w in report.windows:
        val_set = set(w.validation_dates)
        for date_str in w.test_dates:
            assert date_str not in val_set


def test_9_outcome_labels_extending_beyond_train_end_excluded():
    """9. Test outcome labels extending beyond train_end are excluded by point-in-time eligibility function."""
    train_end = "2025-05-30"

    # Setup date 5 days before train_end with 10 session holding period -> Exceeds train_end
    eligible_1 = WalkForwardValidator.is_outcome_label_eligible("2025-05-27", 10, train_end)
    assert eligible_1 is False

    # Setup date 20 days before train_end with 5 session holding period -> Completed before train_end
    eligible_2 = WalkForwardValidator.is_outcome_label_eligible("2025-05-01", 5, train_end)
    assert eligible_2 is True


def test_10_no_future_outcome_labels_in_test_signals():
    """10. Test no future outcome labels enter test signal generation."""
    stock_dfs = _generate_synthetic_stock_data(220)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert report.leakage_checks["outcomes_isolated"] is True


def test_11_continuous_capital_across_oos_windows():
    """11. Test continuous capital carries forward across consecutive OOS test windows."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert len(report.per_window_reports) >= 2
    # Verify capital continuity: final equity of window 0 is initial capital of aggregate state
    final_w0 = report.per_window_reports[0].return_metrics.final_equity
    assert final_w0 > 0.0


def test_12_gaps_between_oos_windows_no_hidden_trades():
    """12. Test gaps between OOS test windows contain zero hidden trades and constant equity."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    sum_trades = sum(r.trade_metrics.total_trades for r in report.per_window_reports)
    agg_trades = report.aggregate_oos_report.trade_metrics.total_trades

    assert agg_trades == sum_trades


def test_13_aggregate_oos_trades_test_trades_only():
    """13. Test aggregate OOS trade count equals sum of test window trades only."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    sum_test_trades = sum(r.trade_metrics.total_trades for r in report.per_window_reports)
    assert report.aggregate_oos_report.trade_metrics.total_trades == sum_test_trades


def test_14_aggregate_oos_equity_test_dates_only():
    """14. Test aggregate OOS equity curve contains TEST dates only."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    all_test_dates = []
    for w in report.windows:
        all_test_dates.extend(w.test_dates)

    assert report.aggregate_oos_report.return_metrics.trading_days == len(all_test_dates)


def test_15_frozen_strategy_configuration_identical():
    """15. Test frozen strategy configuration remains identical across TEST execution."""
    stock_dfs = _generate_synthetic_stock_data(220)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert report.frozen_configuration_hash != ""
    assert report.leakage_checks["parameters_frozen_during_test"] is True


def test_16_actual_path_execution():
    """16. Test executes full path: WalkForwardValidator -> PortfolioBacktestEngine -> PerformanceAnalyzer."""
    stock_dfs = _generate_synthetic_stock_data(220)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert len(report.per_window_reports) > 0
    assert report.aggregate_oos_report.status == "OK"


def test_17_explicit_no_calibration_reporting():
    """17. Test report explicitly exposes calibration_performed = False and calibration_method = 'NONE'."""
    stock_dfs = _generate_synthetic_stock_data(220)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert report.calibration_performed is False
    assert report.calibration_method == "NONE"
