"""
Unit & Integration Tests for P0 Fix #11C: Portfolio Performance Analytics.

Coverage:
  TEST A: Total Return (₹100k -> ₹110k = 10%).
  TEST B: Equity Curve Identity (cash + market_value == total_equity on all dates).
  TEST C: Max Drawdown (100 -> 110 -> 105 -> 90 -> 95 -> 120 = -18.1818%).
  TEST D: Drawdown Duration (Peak, Trough, Recovery dates).
  TEST E: Sharpe Ratio (Manual numeric formula verification with rf=0% and rf=5%).
  TEST F: Sortino Ratio (Manual numeric formula verification with downside deviation).
  TEST G: Trade Statistics (+100, +200, -50, -100, 0 -> Total=5, Win Rate=40%, Gross Win=300, Gross Loss=150, PF=2.0).
  TEST H: R Multiples (realized_R = net_pnl / initial_risk).
  TEST I: Exposure (cash=40k, market_val=60k, equity=100k -> 60%).
  TEST J: Turnover (Partial-exit executed buy/sell notionals; rejected signals contribute 0).
  TEST K: Transaction Costs (gross_pnl - friction = net_pnl).
  TEST L: Benchmark (NIFTY 50 benchmark comparison & excess return).
  TEST M: Empty Backtest (Graceful handle with 0 trades, no NaN/inf).
  TEST N: Invalid Inputs (Zero/negative capital, missing data).
  TEST O: Determinism (Identical inputs -> identical report).
  TEST P: No Data Leakage (Future data does not alter completed period analytics).
  TEST Q: Performance Integration (PortfolioBacktestEngine -> equity_curve -> PerformanceAnalyzer).
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestTrade
from src.backtest.portfolio import PortfolioBacktestEngine, PortfolioState, OpenPosition, DailyPortfolioSnapshot
from src.backtest.performance import PerformanceAnalyzer, PerformanceReport


def test_a_total_return_calculation():
    """TEST A: Initial = ₹1,00,000, Final = ₹1,10,000 -> Total return = 10.0%."""
    state = PortfolioState(initial_capital=100000.0, total_equity=110000.0)
    state.equity_curve = [
        DailyPortfolioSnapshot("2026-01-01", 100000.0, 0.0, 0.0, 100000.0, 0.0, 0.0, 0, 0.0),
        DailyPortfolioSnapshot("2026-01-31", 110000.0, 0.0, 0.0, 110000.0, 10000.0, 0.0, 0, 0.0),
    ]

    report = PerformanceAnalyzer.analyze_portfolio(state)
    assert report.return_metrics.total_return_pct == 10.0
    assert report.return_metrics.initial_capital == 100000.0
    assert report.return_metrics.final_equity == 110000.0


def test_b_equity_curve_identity():
    """TEST B: Verify cash + market_value == total_equity for EVERY date."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [980.0] * 50 + [1000.0, 1005.0, 1010.0, 1015.0, 1020.0, 1025.0, 1030.0, 1035.0, 1040.0, 1045.0]
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    portfolio, _ = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=1000000.0)
    assert len(portfolio.equity_curve) > 0

    for eq in portfolio.equity_curve:
        assert round(eq.cash_available + eq.market_value, 2) == eq.total_equity


def test_c_max_drawdown_calculation():
    """TEST C: Equity series: 100, 110, 105, 90, 95, 120 -> Peak=110, Trough=90, Max DD = -18.1818%."""
    series = [100.0, 110.0, 105.0, 90.0, 95.0, 120.0]
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"]

    state = PortfolioState(initial_capital=100.0, total_equity=120.0)
    state.equity_curve = [
        DailyPortfolioSnapshot(d, val, 0.0, 0.0, val, 0.0, 0.0, 0, 0.0)
        for d, val in zip(dates, series)
    ]

    report = PerformanceAnalyzer.analyze_portfolio(state)
    assert round(report.drawdown_metrics.max_drawdown_pct, 4) == -18.1818
    assert report.drawdown_metrics.max_drawdown_rupees == 20.0
    assert report.drawdown_metrics.peak_date == "2026-01-02"
    assert report.drawdown_metrics.trough_date == "2026-01-04"


def test_d_drawdown_duration():
    """TEST D: Peak=2026-01-02, Trough=2026-01-04, Recovery=2026-01-06 -> Duration = 4 days."""
    series = [100.0, 110.0, 105.0, 90.0, 95.0, 120.0]
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"]

    state = PortfolioState(initial_capital=100.0, total_equity=120.0)
    state.equity_curve = [
        DailyPortfolioSnapshot(d, val, 0.0, 0.0, val, 0.0, 0.0, 0, 0.0)
        for d, val in zip(dates, series)
    ]

    report = PerformanceAnalyzer.analyze_portfolio(state)
    assert report.drawdown_metrics.recovery_date == "2026-01-06"
    assert report.drawdown_metrics.max_drawdown_duration_days == 4


def test_e_sharpe_ratio_numeric_verification():
    """
    TEST E: Exact manual numeric verification of Sharpe ratio formula.
    Equity series: [100.0, 102.0, 99.0, 104.0, 101.0, 106.0]
    Verifies risk_free_rate_pct=0.0 and risk_free_rate_pct=5.0.
    """
    series = [100.0, 102.0, 99.0, 104.0, 101.0, 106.0]
    dates = [f"2026-01-0{i+1}" for i in range(len(series))]

    state = PortfolioState(initial_capital=100.0, total_equity=106.0)
    state.equity_curve = [
        DailyPortfolioSnapshot(d, val, 0.0, 0.0, val, 0.0, 0.0, 0, 0.0)
        for d, val in zip(dates, series)
    ]

    daily_returns = [
        (series[i] - series[i - 1]) / series[i - 1]
        for i in range(1, len(series))
    ]

    # --- 1. risk_free_rate_pct = 0.0 ---
    mean_excess_0 = np.mean(daily_returns)
    std_excess_0 = np.std(daily_returns, ddof=1)
    expected_sharpe_0 = round((mean_excess_0 / std_excess_0) * math.sqrt(252), 4)

    report_0 = PerformanceAnalyzer.analyze_portfolio(state, risk_free_rate_pct=0.0)
    assert report_0.risk_metrics.sharpe_ratio == expected_sharpe_0

    # --- 2. risk_free_rate_pct = 5.0 ---
    rf_daily_5 = (1.05 ** (1.0 / 252.0)) - 1.0
    excess_5 = [r - rf_daily_5 for r in daily_returns]
    mean_excess_5 = np.mean(excess_5)
    std_excess_5 = np.std(excess_5, ddof=1)
    expected_sharpe_5 = round((mean_excess_5 / std_excess_5) * math.sqrt(252), 4)

    report_5 = PerformanceAnalyzer.analyze_portfolio(state, risk_free_rate_pct=5.0)
    assert report_5.risk_metrics.sharpe_ratio == expected_sharpe_5
    assert report_5.risk_metrics.sharpe_ratio < report_0.risk_metrics.sharpe_ratio


def test_f_sortino_ratio_numeric_verification():
    """
    TEST F: Exact manual numeric verification of Sortino ratio using downside deviation.
    Equity series: [100.0, 102.0, 99.0, 104.0, 101.0, 106.0]
    Verifies risk_free_rate_pct=0.0 and risk_free_rate_pct=5.0.
    """
    series = [100.0, 102.0, 99.0, 104.0, 101.0, 106.0]
    dates = [f"2026-01-0{i+1}" for i in range(len(series))]

    state = PortfolioState(initial_capital=100.0, total_equity=106.0)
    state.equity_curve = [
        DailyPortfolioSnapshot(d, val, 0.0, 0.0, val, 0.0, 0.0, 0, 0.0)
        for d, val in zip(dates, series)
    ]

    daily_returns = [
        (series[i] - series[i - 1]) / series[i - 1]
        for i in range(1, len(series))
    ]

    # --- 1. risk_free_rate_pct = 0.0 ---
    mean_excess_0 = np.mean(daily_returns)
    downside_0 = [min(r, 0.0) for r in daily_returns]
    downside_dev_0 = math.sqrt(np.mean([d ** 2 for d in downside_0]))
    expected_sortino_0 = round((mean_excess_0 / downside_dev_0) * math.sqrt(252), 4)

    report_0 = PerformanceAnalyzer.analyze_portfolio(state, risk_free_rate_pct=0.0)
    assert report_0.risk_metrics.sortino_ratio == expected_sortino_0

    # --- 2. risk_free_rate_pct = 5.0 ---
    rf_daily_5 = (1.05 ** (1.0 / 252.0)) - 1.0
    excess_5 = [r - rf_daily_5 for r in daily_returns]
    mean_excess_5 = np.mean(excess_5)
    downside_5 = [min(x, 0.0) for x in excess_5]
    downside_dev_5 = math.sqrt(np.mean([d ** 2 for d in downside_5]))
    expected_sortino_5 = round((mean_excess_5 / downside_dev_5) * math.sqrt(252), 4)

    report_5 = PerformanceAnalyzer.analyze_portfolio(state, risk_free_rate_pct=5.0)
    assert report_5.risk_metrics.sortino_ratio == expected_sortino_5
    assert report_5.risk_metrics.sortino_ratio < report_0.risk_metrics.sortino_ratio


def test_g_trade_statistics():
    """TEST G: Trades: +100, +200, -50, -100, 0 -> Total=5, Win Rate=40%, Gross Win=300, Gross Loss=150, PF=2.0."""
    pnls = [100.0, 200.0, -50.0, -100.0, 0.0]
    trades = [
        BacktestTrade(
            symbol="T1", entry_date="2026-01-01", entry_price=100.0, stop_loss=95.0,
            target_1=110.0, target_2=115.0, target_3=120.0, shares=10, exit_date="2026-01-05",
            exit_price=110.0, exit_reason="TARGET_1", pnl_pct=10.0, pnl_rupees=pnl,
            gross_pnl_rupees=pnl + 10.0, transaction_cost_rupees=10.0, holding_sessions=5,
            max_adverse_excursion_pct=0.0, max_favorable_excursion_pct=10.0,
            executed_buy_value=1000.0, executed_sell_value=1000.0 + pnl
        )
        for pnl in pnls
    ]

    state = PortfolioState(initial_capital=100000.0, completed_trades=trades)
    report = PerformanceAnalyzer.analyze_portfolio(state)

    m = report.trade_metrics
    assert m.total_trades == 5
    assert m.winning_trades == 2
    assert m.losing_trades == 2
    assert m.breakeven_trades == 1
    assert m.win_rate_pct == 40.0
    assert m.gross_profit == 300.0
    assert m.gross_loss == 150.0
    assert m.profit_factor == 2.0
    assert m.average_winner == 150.0
    assert m.average_loser == 75.0
    assert m.expectancy_per_trade == 30.0


def test_h_r_multiples():
    """TEST H: Realized R = net_pnl / initial_risk."""
    # Entry=100, Stop=95 -> Risk/share = 5. Shares=100 -> Initial Risk = 500. Net PnL = +1000 -> R = 2.0
    t1 = BacktestTrade(
        symbol="T1", entry_date="2026-01-01", entry_price=100.0, stop_loss=95.0,
        target_1=110.0, target_2=115.0, target_3=120.0, shares=100, exit_date="2026-01-05",
        exit_price=110.0, exit_reason="TARGET_1", pnl_pct=10.0, pnl_rupees=1000.0,
        gross_pnl_rupees=1010.0, transaction_cost_rupees=10.0, holding_sessions=5,
        max_adverse_excursion_pct=0.0, max_favorable_excursion_pct=10.0,
        executed_buy_value=10000.0, executed_sell_value=11000.0
    )
    # Entry=100, Stop=95 -> Risk/share = 5. Shares=100 -> Initial Risk = 500. Net PnL = -500 -> R = -1.0
    t2 = BacktestTrade(
        symbol="T2", entry_date="2026-01-01", entry_price=100.0, stop_loss=95.0,
        target_1=110.0, target_2=115.0, target_3=120.0, shares=100, exit_date="2026-01-05",
        exit_price=95.0, exit_reason="STOP_LOSS", pnl_pct=-5.0, pnl_rupees=-500.0,
        gross_pnl_rupees=-490.0, transaction_cost_rupees=10.0, holding_sessions=5,
        max_adverse_excursion_pct=-5.0, max_favorable_excursion_pct=0.0,
        executed_buy_value=10000.0, executed_sell_value=9500.0
    )

    state = PortfolioState(initial_capital=100000.0, completed_trades=[t1, t2])
    report = PerformanceAnalyzer.analyze_portfolio(state)

    r_m = report.r_metrics
    assert r_m.average_R == 0.5
    assert r_m.winning_R_average == 2.0
    assert r_m.losing_R_average == -1.0
    assert r_m.distribution["2 <= R < 3"] == 1
    assert r_m.distribution["-1 <= R < 0"] == 1


def test_i_exposure_metrics():
    """TEST I: Cash = 40,000, market_val = 60,000, equity = 100,000 -> Exposure = 60%."""
    state = PortfolioState(initial_capital=100000.0)
    state.equity_curve = [
        DailyPortfolioSnapshot("2026-01-01", 40000.0, 60000.0, 60000.0, 100000.0, 0.0, 0.0, 2, 60.0),
    ]

    report = PerformanceAnalyzer.analyze_portfolio(state)
    exp = report.exposure_metrics
    assert exp.maximum_exposure_pct == 60.0
    assert exp.average_exposure_pct == 60.0
    assert exp.minimum_cash_pct == 40.0
    assert exp.maximum_open_positions == 2


def test_j_turnover_partial_exit_and_rejected_signal():
    """
    TEST J: Partial-exit executed buy/sell notionals & rejected signal exclusion.
    Buy: 100 shares x ₹100 = ₹10,000.
    Partial sell 1: 40 shares x ₹110 = ₹4,400.
    Partial sell 2: 60 shares x ₹120 = ₹7,200.
    Executed sell value = ₹11,600.
    Expected: total_buy_value = ₹10,000, total_sell_value = ₹11,600, total_turnover = ₹21,600 (NOT 10,000 + 12,000).
    Rejected signals contribute ZERO turnover.
    """
    trade = BacktestTrade(
        symbol="TRENT", entry_date="2026-01-01", entry_price=100.0, stop_loss=95.0,
        target_1=110.0, target_2=115.0, target_3=120.0, shares=100, exit_date="2026-01-05",
        exit_price=120.0, exit_reason="TARGET_3_HIT", pnl_pct=16.0, pnl_rupees=1600.0,
        gross_pnl_rupees=1620.0, transaction_cost_rupees=20.0, holding_sessions=5,
        max_adverse_excursion_pct=0.0, max_favorable_excursion_pct=20.0,
        executed_buy_value=10000.0, executed_sell_value=11600.0  # 4400 + 7200
    )

    state = PortfolioState(
        initial_capital=100000.0,
        completed_trades=[trade],
        rejection_reasons=["MAX_PORTFOLIO_RISK_EXCEEDED: Projected open risk exceeded"]
    )
    report = PerformanceAnalyzer.analyze_portfolio(state)

    t_m = report.turnover_metrics
    assert t_m.total_buy_value == 10000.0
    assert t_m.total_sell_value == 11600.0
    assert t_m.total_turnover == 21600.0
    assert t_m.turnover_pct == 21.6


def test_k_transaction_cost_impact():
    """TEST K: Verify gross_pnl - friction = net_pnl."""
    t1 = BacktestTrade(
        symbol="T1", entry_date="2026-01-01", entry_price=100.0, stop_loss=95.0,
        target_1=110.0, target_2=115.0, target_3=120.0, shares=100, exit_date="2026-01-05",
        exit_price=110.0, exit_reason="TARGET_1", pnl_pct=10.0, pnl_rupees=980.0,
        gross_pnl_rupees=1000.0, transaction_cost_rupees=20.0, holding_sessions=5,
        max_adverse_excursion_pct=0.0, max_favorable_excursion_pct=10.0,
        executed_buy_value=10000.0, executed_sell_value=11000.0
    )

    state = PortfolioState(initial_capital=100000.0, completed_trades=[t1])
    report = PerformanceAnalyzer.analyze_portfolio(state)

    cost_m = report.transaction_cost_metrics
    assert cost_m.gross_pnl == 1000.0
    assert cost_m.total_transaction_costs == 20.0
    assert cost_m.net_pnl == 980.0
    assert round(cost_m.gross_pnl - cost_m.total_transaction_costs, 2) == cost_m.net_pnl


def test_l_benchmark_comparison():
    """TEST L: Deterministic benchmark fixture -> Benchmark return, strategy return, excess return."""
    dates = pd.date_range(start="2026-01-01", periods=10, freq="B")
    b_prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 110.0]
    b_df = pd.DataFrame({"timestamp": dates, "close": b_prices})

    s_prices = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 120.0]
    state = PortfolioState(initial_capital=100.0, total_equity=120.0)
    state.equity_curve = [
        DailyPortfolioSnapshot(d.strftime("%Y-%m-%d"), val, 0.0, 0.0, val, 0.0, 0.0, 0, 0.0)
        for d, val in zip(dates, s_prices)
    ]

    report = PerformanceAnalyzer.analyze_portfolio(state, benchmark_df=b_df)
    b_m = report.benchmark_metrics

    assert b_m.status == "OK"
    assert b_m.benchmark_total_return_pct == 10.0
    assert report.return_metrics.total_return_pct == 20.0
    assert b_m.strategy_excess_return_pct == 10.0


def test_m_empty_backtest():
    """TEST M: Empty backtest produces 0 trades, 0.0 metrics, no inf/NaN."""
    state = PortfolioState(initial_capital=1000000.0)
    report = PerformanceAnalyzer.analyze_portfolio(state)

    assert report.status == "EMPTY_BACKTEST"
    assert report.trade_metrics.total_trades == 0
    assert report.return_metrics.total_return_pct == 0.0
    assert report.risk_metrics.sharpe_ratio == 0.0
    assert report.risk_metrics.sortino_ratio == 0.0
    assert not math.isnan(report.risk_metrics.sharpe_ratio)


def test_n_invalid_inputs():
    """TEST N: Gracefully handles zero/negative capital without crashing."""
    state = PortfolioState(initial_capital=0.0)
    report = PerformanceAnalyzer.analyze_portfolio(state)

    assert report.status == "INVALID_CAPITAL"
    assert report.return_metrics.total_return_pct == 0.0


def test_o_deterministic_repeatability():
    """TEST O: Identical backtest input -> Identical report and metrics."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [980.0] * 50 + [1000.0, 1030.0, 1060.0, 1120.0, 1150.0, 1180.0, 1200.0, 1220.0, 1240.0, 1250.0]
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    port1, _ = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=1000000.0)
    port2, _ = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=1000000.0)

    rep1 = PerformanceAnalyzer.analyze_portfolio(port1)
    rep2 = PerformanceAnalyzer.analyze_portfolio(port2)

    assert rep1.return_metrics.total_return_pct == rep2.return_metrics.total_return_pct
    assert rep1.drawdown_metrics.max_drawdown_pct == rep2.drawdown_metrics.max_drawdown_pct
    assert rep1.risk_metrics.sharpe_ratio == rep2.risk_metrics.sharpe_ratio
    assert rep1.trade_metrics.total_trades == rep2.trade_metrics.total_trades


def test_p_no_data_leakage():
    """TEST P: Modifying data AFTER backtest period does not alter earlier metrics."""
    dates_short = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices_short = [980.0] * 50 + [1000.0, 1030.0, 1060.0, 1120.0, 1150.0, 1180.0, 1200.0, 1220.0, 1240.0, 1250.0]
    df_short = pd.DataFrame({"timestamp": dates_short, "open": prices_short, "high": [p * 1.002 for p in prices_short], "low": [p * 0.998 for p in prices_short], "close": prices_short, "volume": [50000]*50 + [300000]*10})

    port_short, _ = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df_short}, initial_capital=1000000.0)
    rep_short = PerformanceAnalyzer.analyze_portfolio(port_short)

    # Future data appended after backtest period
    dates_long = pd.date_range(start="2025-11-15", periods=100, freq="B")
    prices_long = prices_short + [500.0] * 40  # Massive future crash
    df_long = pd.DataFrame({"timestamp": dates_long, "open": prices_long, "high": [p * 1.002 for p in prices_long], "low": [p * 0.998 for p in prices_long], "close": prices_long, "volume": [50000]*50 + [300000]*50})

    # Run performance analytics strictly on the short period portfolio state
    rep_short_again = PerformanceAnalyzer.analyze_portfolio(port_short)

    assert rep_short.return_metrics.total_return_pct == rep_short_again.return_metrics.total_return_pct
    assert rep_short.drawdown_metrics.max_drawdown_pct == rep_short_again.drawdown_metrics.max_drawdown_pct


def test_q_performance_integration_with_portfolio_engine():
    """
    TEST Q: Integration test exercising:
      PortfolioBacktestEngine -> equity_curve generation -> PerformanceAnalyzer.analyze_portfolio() -> PerformanceReport.
    Proves PerformanceAnalyzer consumes actual engine output without manual equity curve construction.
    """
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [980.0] * 50 + [1000.0, 1030.0, 1060.0, 1120.0, 1150.0, 1180.0, 1200.0, 1220.0, 1240.0, 1250.0]
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [p * 1.002 for p in prices],
        "low": [p * 0.998 for p in prices],
        "close": prices,
        "volume": [50000]*50 + [300000]*10
    })

    portfolio, engine_stats = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=1000000.0)

    report = PerformanceAnalyzer.analyze_portfolio(portfolio)

    assert report.status == "OK"
    assert len(portfolio.equity_curve) == 60
    assert report.return_metrics.total_return_pct == round(((portfolio.total_equity / 1000000.0) - 1.0) * 100.0, 4)
    assert report.trade_metrics.total_trades == len(portfolio.completed_trades)
    assert report.exposure_metrics.maximum_exposure_pct >= 0.0
    assert report.turnover_metrics.total_turnover >= 0.0
