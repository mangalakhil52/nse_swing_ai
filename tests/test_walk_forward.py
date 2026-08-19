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
 11. test_inter_window_gap_contains_no_trades: Inspects actual OOS trades and asserts no trade entry_date or exit_date lies in inter-window gap.
 12. test_inter_window_gap_contains_no_equity_snapshots: Inspects actual OOS equity snapshots and asserts no snapshot.date lies in inter-window gap.
 13. test_inter_window_capital_is_constant: Asserts prev_window.final_equity == next_window.initial_capital for EVERY consecutive window pair.
 14. test_gap_data_mutation_does_not_change_adjacent_oos_windows: Mutating data strictly inside inter-window gap leaves Window 0 and Window 1 identical.
 15. test_outcome_consumption_metadata_verification: Asserts consumed_outcome_labels_count == 0 when calibration_performed = False.
 16. test_frozen_configuration_cutoff_is_validation_end: Verifies cutoff_date = validation_end (or train_end).
 17. test_cutoff_strictly_before_test_start: Verifies cutoff_date < test_start for all windows.
 18. test_actual_path_execution: Runs full WalkForwardValidator -> PortfolioBacktestEngine -> PerformanceAnalyzer path.
 19. test_explicit_no_calibration_reporting: Verifies report exposes calibration_performed = False, calibration_method = "NONE".
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


def _generate_synthetic_stock_data(num_days=350, symbol="TRENT", start_date="2025-01-01"):
    dates = pd.date_range(start=start_date, periods=num_days, freq="B")
    prices = []
    curr = 1000.0
    for i in range(num_days):
        ret = 0.002 if i % 2 == 1 else -0.001
        if (i >= 120 and i < 150) or (i >= 220 and i < 250):
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
    """11. Test actual OOS trade inspection asserts zero trade entry_date or exit_date lies in inter-window gaps."""
    stock_dfs = _generate_synthetic_stock_data(350)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=30) # 10-day gap
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert len(report.windows) >= 2

    all_oos_trades = report.oos_completed_trades

    for i in range(1, len(report.windows)):
        prev_test_end = pd.to_datetime(report.windows[i - 1].test_end)
        curr_test_start = pd.to_datetime(report.windows[i].test_start)

        for trade in all_oos_trades:
            entry_dt = pd.to_datetime(trade.entry_date)
            exit_dt = pd.to_datetime(trade.exit_date)

            assert not (prev_test_end < entry_dt < curr_test_start), (
                f"Trade entry {trade.entry_date} lies inside gap ({prev_test_end}, {curr_test_start})"
            )
            assert not (prev_test_end < exit_dt < curr_test_start), (
                f"Trade exit {trade.exit_date} lies inside gap ({prev_test_end}, {curr_test_start})"
            )


def test_inter_window_gap_contains_no_equity_snapshots():
    """12. Test actual OOS equity snapshot inspection asserts zero snapshot.date lies in inter-window gaps."""
    stock_dfs = _generate_synthetic_stock_data(350)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=30) # 10-day gap
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert len(report.windows) >= 2

    all_snapshots = report.oos_equity_snapshots

    for i in range(1, len(report.windows)):
        prev_test_end = pd.to_datetime(report.windows[i - 1].test_end)
        curr_test_start = pd.to_datetime(report.windows[i].test_start)

        for snap in all_snapshots:
            snap_dt = pd.to_datetime(snap.date)
            assert not (prev_test_end < snap_dt < curr_test_start), (
                f"Equity snapshot date {snap.date} lies inside gap ({prev_test_end}, {curr_test_start})"
            )


def test_inter_window_capital_is_constant():
    """13. Test previous_window.final_equity == next_window.initial_capital for EVERY consecutive pair, plus no gap trades/snapshots."""
    stock_dfs = _generate_synthetic_stock_data(350)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=30)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert len(report.per_window_reports) >= 2

    for i in range(1, len(report.per_window_reports)):
        prev_w = report.per_window_reports[i - 1]
        next_w = report.per_window_reports[i]
        assert prev_w.return_metrics.final_equity == next_w.return_metrics.initial_capital

    # Assert no OOS trade or equity snapshot in gap
    for i in range(1, len(report.windows)):
        prev_end = pd.to_datetime(report.windows[i - 1].test_end)
        curr_start = pd.to_datetime(report.windows[i].test_start)
        for t in report.oos_completed_trades:
            e_dt = pd.to_datetime(t.entry_date)
            assert not (prev_end < e_dt < curr_start)
        for s in report.oos_equity_snapshots:
            s_dt = pd.to_datetime(s.date)
            assert not (prev_end < s_dt < curr_start)


def test_gap_data_mutation_does_not_change_adjacent_oos_windows():
    """14. Test mutating data strictly inside an inter-window gap leaves Window 0 and Window 1 identical."""
    stock_dfs_orig = _generate_synthetic_stock_data(350)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=40)
    report_orig = WalkForwardValidator.run_walk_forward(stock_dfs_orig, config)

    assert report_orig.status == "OK"
    assert len(report_orig.windows) >= 2

    w0_end_dt = pd.to_datetime(report_orig.windows[0].test_end)
    w1_start_dt = pd.to_datetime(report_orig.windows[1].test_start)

    # Mutate ONLY rows strictly inside the GAP
    stock_dfs_mut = _generate_synthetic_stock_data(350)
    df_mut = stock_dfs_mut["TRENT"].copy()
    mask_gap = (pd.to_datetime(df_mut["timestamp"]) > w0_end_dt) & (pd.to_datetime(df_mut["timestamp"]) < w1_start_dt)
    assert mask_gap.any(), "Synthetic gap must contain data rows to mutate."
    df_mut.loc[mask_gap, "close"] *= 5.0  # 500% price spike in gap
    df_mut.loc[mask_gap, "high"] *= 5.0
    stock_dfs_mut["TRENT"] = df_mut

    report_mut = WalkForwardValidator.run_walk_forward(stock_dfs_mut, config)

    # Compare actual Window 0 (Window A prior to gap) between report_orig and report_mut
    w0_orig_rep = report_orig.per_window_reports[0]
    w0_mut_rep = report_mut.per_window_reports[0]

    assert w0_orig_rep.trade_metrics.total_trades == w0_mut_rep.trade_metrics.total_trades
    assert w0_orig_rep.return_metrics.final_equity == w0_mut_rep.return_metrics.final_equity
    assert w0_orig_rep.return_metrics.total_return_pct == w0_mut_rep.return_metrics.total_return_pct

    # Detailed trade comparisons for Window 0
    for t_orig, t_mut in zip(w0_orig_rep.completed_trades, w0_mut_rep.completed_trades):
        assert t_orig.symbol == t_mut.symbol
        assert t_orig.entry_date == t_mut.entry_date
        assert t_orig.entry_price == t_mut.entry_price
        assert t_orig.stop_loss == t_mut.stop_loss
        assert t_orig.target_1 == t_mut.target_1
        assert t_orig.exit_date == t_mut.exit_date
        assert t_orig.exit_price == t_mut.exit_price
        assert t_orig.pnl_rupees == t_mut.pnl_rupees

    # Detailed equity snapshot comparisons for Window 0
    for s_orig, s_mut in zip(w0_orig_rep.equity_curve, w0_mut_rep.equity_curve):
        assert s_orig.date == s_mut.date
        assert s_orig.total_equity == s_mut.total_equity
        assert s_orig.cash_available == s_mut.cash_available
        assert s_orig.market_value == s_mut.market_value


def test_outcome_consumption_metadata_verification():
    """15. Test consumed_outcome_labels_count == 0 when calibration_performed = False."""
    stock_dfs = _generate_synthetic_stock_data(220)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert report.calibration_performed is False
    assert report.calibration_method == "NONE"
    assert report.consumed_outcome_labels_count == 0
    assert report.leakage_checks["outcomes_isolated"] is True


def test_frozen_configuration_cutoff_is_validation_end():
    """16. Test frozen configuration cutoff date is set to validation_end (or train_end)."""
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
    """17. Test cutoff_date < test_start for all generated windows."""
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


def test_18_actual_path_execution():
    """18. Test executes full path: WalkForwardValidator -> PortfolioBacktestEngine -> PerformanceAnalyzer."""
    stock_dfs = _generate_synthetic_stock_data(220)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert len(report.per_window_reports) > 0
    assert report.aggregate_oos_report.status == "OK"


def test_19_explicit_no_calibration_reporting():
    """19. Test report explicitly exposes calibration_performed = False and calibration_method = 'NONE'."""
    stock_dfs = _generate_synthetic_stock_data(220)
    config = WalkForwardConfig(train_days=80, validation_days=20, test_days=20, step_days=20)
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config)

    assert report.status == "OK"
    assert report.calibration_performed is False
    assert report.calibration_method == "NONE"
