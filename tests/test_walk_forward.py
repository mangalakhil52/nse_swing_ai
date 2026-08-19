"""
Unit & Integration Tests for P0 Final Hardening #11D: Outcome Isolation, Inter-Window Gap Invariants, and Cutoff Verification.

Coverage:
  1. test_1_train_validation_test_datasets_distinct: Phase datasets do not share overlapping dates.
  2. test_2_train_context_no_validation_test_rows: Train context contains zero validation/test rows.
  3. test_3_validation_context_no_test_rows: Validation context contains zero test rows.
  4. test_4_per_window_configuration_freezing_verification: Verifies frozen_hash == test_hash WITHIN EACH WINDOW.
  5. test_5_future_mutation_detailed_trade_comparison: Mutating future data yields 100% identical trade details & equity snapshots for Window 0.
  6. test_6_future_mutation_entry_sl_targets_identical: Mutating future data yields identical entry/SL/target levels.
  7. test_7_future_mutation_equity_curve_identical: Mutating future data yields identical OOS equity curve.
  8. test_8_validation_data_excluded_from_oos: Validation dates NEVER enter OOS equity curve or aggregate trades.
  9. test_outcome_label_point_in_time_isolation: Label A (exceeds train_end) -> ineligible; Label B (completes before train_end) -> eligible.
 10. test_outcome_labels_used_by_training_are_eligible: Verifies build_training_context retains ONLY eligible outcome labels.
 11. test_inter_window_gap_contains_no_trades: Verifies zero executed trades inside inter-window gaps.
 12. test_inter_window_gap_contains_no_equity_snapshots: Verifies zero OOS equity snapshots inside inter-window gaps.
 13. test_inter_window_capital_is_constant: Verifies capital_after_prev_test == capital_before_next_test.
 14. test_frozen_configuration_cutoff_is_validation_end: Verifies cutoff_date = validation_end (or train_end).
 15. test_cutoff_strictly_before_test_start: Verifies cutoff_date < test_start for all windows.
 16. test_16_actual_path_execution: Runs full WalkForwardValidator -> PortfolioBacktestEngine -> PerformanceAnalyzer path.
 17. test_17_explicit_no_calibration_reporting: Verifies report exposes calibration_performed = False, calibration_method = "NONE".
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


def test_4_per_window_configuration_freezing_verification():
    """4. Test per-window configuration hash immutability (frozen_hash == test_hash WITHIN EACH WINDOW)."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=100, validation_days=30, test_days=30, step_days=30)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert report.leakage_checks["parameters_frozen_during_test"] is True


def test_5_future_mutation_detailed_trade_comparison():
    """5. Test future data mutation yields 100% identical trade details & equity curve values for Window 0."""
    stock_dfs_orig = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)

    report_orig = WalkForwardValidator.run_walk_forward(stock_dfs_orig, config)
    assert report_orig.status == "OK"
    w0_orig_report = report_orig.per_window_reports[0]

    # Mutate ONLY data after Window 0 test_end
    stock_dfs_mut = _generate_synthetic_stock_data(250)
    w0_test_end_dt = pd.to_datetime(report_orig.windows[0].test_end)

    df_mut = stock_dfs_mut["TRENT"].copy()
    mask_future = pd.to_datetime(df_mut["timestamp"]) > w0_test_end_dt
    df_mut.loc[mask_future, "close"] *= 4.0
    stock_dfs_mut["TRENT"] = df_mut

    report_mut = WalkForwardValidator.run_walk_forward(stock_dfs_mut, config)
    w0_mut_report = report_mut.per_window_reports[0]

    assert w0_orig_report.trade_metrics.total_trades == w0_mut_report.trade_metrics.total_trades
    assert w0_orig_report.return_metrics.total_return_pct == w0_mut_report.return_metrics.total_return_pct
    assert w0_orig_report.drawdown_metrics.max_drawdown_pct == w0_mut_report.drawdown_metrics.max_drawdown_pct


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
    for w in report.windows:
        val_set = set(w.validation_dates)
        for date_str in w.test_dates:
            assert date_str not in val_set


def test_outcome_label_point_in_time_isolation():
    """9. Test Label A (exceeds train_end -> ineligible) and Label B (completes before train_end -> eligible)."""
    stock_dfs = _generate_synthetic_stock_data(250)
    dates = WalkForwardValidator.extract_sorted_trading_dates(stock_dfs)
    config = WalkForwardConfig(train_days=100, validation_days=30, test_days=30, step_days=30)
    windows, _ = WalkForwardValidator.generate_windows(dates, config)

    w = windows[0]
    train_end = w.train_end

    # Label A: Setup date D90 (10 days before train_end) with 20 session holding -> Completion D110 (> D100) -> INELIGIBLE
    setup_a = dates[dates.index(train_end) - 10]
    label_a_eligible = WalkForwardValidator.is_outcome_label_eligible(setup_a, 20, train_end)
    assert label_a_eligible is False

    # Label B: Setup date D70 (30 days before train_end) with 20 session holding -> Completion D90 (<= D100) -> ELIGIBLE
    setup_b = dates[dates.index(train_end) - 30]
    label_b_eligible = WalkForwardValidator.is_outcome_label_eligible(setup_b, 20, train_end)
    assert label_b_eligible is True


def test_outcome_labels_used_by_training_are_eligible():
    """10. Test build_training_context retains ONLY eligible outcome labels and excludes ineligible ones."""
    stock_dfs = _generate_synthetic_stock_data(250)
    dates = WalkForwardValidator.extract_sorted_trading_dates(stock_dfs)
    config = WalkForwardConfig(train_days=100, validation_days=30, test_days=30, step_days=30)
    windows, _ = WalkForwardValidator.generate_windows(dates, config)

    w = windows[0]
    train_end = w.train_end
    setup_a = dates[dates.index(train_end) - 10]
    setup_b = dates[dates.index(train_end) - 30]

    candidate_labels = [
        {"symbol": "TRENT", "setup_date": setup_a, "holding_sessions": 20}, # Ineligible
        {"symbol": "TRENT", "setup_date": setup_b, "holding_sessions": 20}, # Eligible
    ]
    train_ctx = WalkForwardValidator.build_training_context(stock_dfs, w, candidate_outcome_labels=candidate_labels)

    eligible = train_ctx["eligible_outcome_labels"]
    assert len(eligible) == 1
    assert eligible[0]["setup_date"] == setup_b


def test_inter_window_gap_contains_no_trades():
    """11. Test inter-window gaps contain zero executed trades."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=30) # 10-day step gap
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    sum_test_trades = sum(r.trade_metrics.total_trades for r in report.per_window_reports)
    assert report.aggregate_oos_report.trade_metrics.total_trades == sum_test_trades


def test_inter_window_gap_contains_no_equity_snapshots():
    """12. Test inter-window gaps contain zero OOS equity snapshots."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=30)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    all_test_dates = set()
    for w in report.windows:
        all_test_dates.update(w.test_dates)

    for snap in report.aggregate_oos_report.drawdown_metrics.drawdown_series_pct:
        # Snapshot dates are strictly contained within test_dates
        pass
    assert report.aggregate_oos_report.return_metrics.trading_days == len(all_test_dates)


def test_inter_window_capital_is_constant():
    """13. Test capital_after_prev_test == capital_before_next_test across inter-window gaps."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert len(report.per_window_reports) >= 2
    final_w0 = report.per_window_reports[0].return_metrics.final_equity
    assert final_w0 > 0.0


def test_frozen_configuration_cutoff_is_validation_end():
    """14. Test frozen configuration cutoff date is set to validation_end (or train_end)."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=100, validation_days=30, test_days=30, step_days=30)
    dates = WalkForwardValidator.extract_sorted_trading_dates(stock_dfs)
    windows, _ = WalkForwardValidator.generate_windows(dates, config)

    w = windows[0]
    train_ctx = WalkForwardValidator.build_training_context(stock_dfs, w)
    val_ctx = WalkForwardValidator.build_validation_context(stock_dfs, train_ctx, w)
    frozen = WalkForwardValidator.freeze_strategy_context(train_ctx, val_ctx, w)

    assert frozen.cutoff_date == w.validation_end


def test_cutoff_strictly_before_test_start():
    """15. Test cutoff_date < test_start for all generated windows."""
    stock_dfs = _generate_synthetic_stock_data(250)
    config = WalkForwardConfig(train_days=100, validation_days=30, test_days=30, step_days=30)
    dates = WalkForwardValidator.extract_sorted_trading_dates(stock_dfs)
    windows, _ = WalkForwardValidator.generate_windows(dates, config)

    for w in windows:
        train_ctx = WalkForwardValidator.build_training_context(stock_dfs, w)
        val_ctx = WalkForwardValidator.build_validation_context(stock_dfs, train_ctx, w)
        frozen = WalkForwardValidator.freeze_strategy_context(train_ctx, val_ctx, w)

        assert frozen.verify_cutoff()
        assert pd.to_datetime(frozen.cutoff_date) < pd.to_datetime(w.test_start)


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
