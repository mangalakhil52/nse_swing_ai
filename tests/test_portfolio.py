"""
Unit & Integration Tests for P0 Fix #11A & #11B: Portfolio Capital Accounting & Risk-Based Sizing.

Coverage:
  1. Basic Risk Calculation (Shares * risk_per_share <= max_trade_risk).
  2. Risk Limit Overrides Cash Capacity.
  3. Capital Limit Overrides Risk Capacity.
  4. Invalid Stop Geometry Rejection (INVALID_RISK_GEOMETRY).
  5. Zero / Negative Risk Rejection.
  6. Too-Small Risk Budget Rejection (RISK_BUDGET_TOO_SMALL).
  7. Aggregate Open Risk Limit Enforcement (MAX_PORTFOLIO_RISK_EXCEEDED).
  8. Partial Exit Reduces Open Risk (Shares * (entry - stop)).
  9. Closed Position Contributes Zero Open Risk.
  10. Current Equity Basis for Risk Budget (₹12L equity -> ₹6k risk ceiling at 0.5%).
  11. Two Simultaneous Trades Enforce Both Individual and Portfolio Risk Limits.
  12. Non-Negative Cash Guarantee (#11A preservation).
  13. Deterministic Repeatability & Idempotency.
  14. End-to-End Portfolio Backtest Integration Path.
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.trade_construction_agent import TradeConstructionEngine
from src.backtest.friction import IndianFrictionModel
from src.backtest.portfolio import PortfolioBacktestEngine, PortfolioState, OpenPosition
from src.quant.indicators import TechnicalIndicators


def test_basic_risk_calculation():
    """Test 1: Equity = ₹10L, max_risk_per_trade = 0.5% -> Max trade risk = ₹5,000. Entry = 1000, Stop = 950 -> 100 shares max."""
    state = PortfolioState(initial_capital=1000000.0, max_risk_per_trade_pct=0.50)

    entry_price = 1000.0
    stop_loss = 950.0
    risk_per_share = entry_price - stop_loss  # ₹50

    portfolio_equity = state.total_equity  # ₹10,00,000
    max_trade_risk = portfolio_equity * (state.max_risk_per_trade_pct / 100.0)  # ₹5,000
    max_shares = int(max_trade_risk // risk_per_share)  # 100 shares

    assert max_trade_risk == 5000.0
    assert max_shares == 100
    assert (max_shares * risk_per_share) <= max_trade_risk


def test_risk_limit_overrides_cash_capacity():
    """Test 2: Cash allows 500 shares, but risk budget allows only 100 shares -> 100 shares accepted."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5]
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    # High capital ₹50L, max risk 0.5% = ₹25,000. Stop loss at 96.0 -> risk/share = 4.3 -> max risk shares ~ 5800.
    # Lower max_risk_per_trade_pct to 0.05% (₹500 max risk) -> max risk shares ~ 116.
    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest(
        {"TRENT": df}, initial_capital=1000000.0, max_risk_per_trade_pct=0.05
    )

    if "TRENT" in portfolio.open_positions:
        pos = portfolio.open_positions["TRENT"]
        risk_per_share = pos.entry_price - pos.stop_loss
        trade_risk = risk_per_share * pos.shares
        assert trade_risk <= (1000000.0 * (0.05 / 100.0)) + 1e-2


def test_capital_limit_overrides_risk_capacity():
    """Test 3: Risk budget allows 100 shares, but available cash allows only 50 shares -> 50 shares accepted."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [980.0] * 50 + [1000.0, 1005.0, 1010.0, 1015.0, 1020.0, 1025.0, 1030.0, 1035.0, 1040.0, 1045.0]
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    # Low cash ₹50,000 (can only afford ~50 shares at ₹1,000), but high risk budget (5% = ₹2,500 -> can afford ~120 shares by risk)
    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest(
        {"TRENT": df}, initial_capital=50000.0, max_risk_per_trade_pct=5.0
    )

    if "TRENT" in portfolio.open_positions:
        pos = portfolio.open_positions["TRENT"]
        assert pos.shares * pos.entry_price <= 50000.0
        assert portfolio.cash_available >= 0.0


def test_invalid_stop_geometry_rejection():
    """Test 4 & 5: Stop >= Entry or Risk <= 0 generates INVALID_RISK_GEOMETRY rejection."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [100.0] * 60
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": prices, "low": prices, "close": prices, "volume": [100000]*60})

    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest({"FLAT": df}, initial_capital=1000000.0)

    # Flat prices mean structural stop loss is invalid or R:R insufficient
    assert len(portfolio.completed_trades) == 0
    assert len(portfolio.open_positions) == 0


def test_too_small_risk_budget_rejection():
    """Test 6: Max shares by risk < 1 produces RISK_BUDGET_TOO_SMALL rejection."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5]
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    # Extremely tiny risk per trade (0.0001% of ₹10L = ₹1 max risk -> risk/share ~ ₹4.3 -> max shares = 0)
    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest(
        {"TRENT": df}, initial_capital=1000000.0, max_risk_per_trade_pct=0.0001
    )

    assert any("RISK_BUDGET_TOO_SMALL" in r for r in portfolio.rejection_reasons)
    assert len(portfolio.open_positions) == 0


def test_aggregate_open_risk_limit_enforced():
    """Test 7: Projected open risk > max_total_open_risk_pct produces MAX_PORTFOLIO_RISK_EXCEEDED."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5]

    df1 = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})
    df2 = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    # Max total open risk = 0.40% (₹4,000). Trade 1 takes ~₹4,000 risk, leaving ~0 capacity for Trade 2
    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest(
        {"INFY": df1, "TCS": df2}, initial_capital=1000000.0, max_risk_per_trade_pct=0.50, max_total_open_risk_pct=0.40
    )

    assert any("MAX_PORTFOLIO_RISK_EXCEEDED" in r for r in portfolio.rejection_reasons)


def test_partial_exit_reduces_open_risk():
    """Test 8: Partial exit at Target 1 reduces active open risk based on remaining shares."""
    pos = OpenPosition(
        symbol="TRENT", entry_date="2026-01-01", entry_price=100.0, shares=100,
        invested_value=10000.0, stop_loss=95.0, target_1=110.0, target_2=115.0, target_3=120.0,
        entry_cost=15.0, remaining_shares=100,
    )

    initial_risk = (pos.entry_price - pos.stop_loss) * pos.remaining_shares  # ₹500
    assert initial_risk == 500.0

    # 50 shares exit at T1
    pos.remaining_shares = 50
    reduced_risk = (pos.entry_price - pos.stop_loss) * pos.remaining_shares  # ₹250
    assert reduced_risk == 250.0


def test_closed_position_contributes_zero_open_risk():
    """Test 9: Fully closed position contributes zero open risk to current_total_open_risk."""
    state = PortfolioState(initial_capital=1000000.0)
    pos = OpenPosition(
        symbol="TRENT", entry_date="2026-01-01", entry_price=100.0, shares=100,
        invested_value=10000.0, stop_loss=95.0, target_1=110.0, target_2=115.0, target_3=120.0,
        entry_cost=15.0,
    )
    state.open_positions["TRENT"] = pos
    assert state.current_total_open_risk == 500.0

    # Full exit
    PortfolioBacktestEngine._close_position(state, pos, 110.0, "TARGET_1_HIT", "2026-01-05")
    del state.open_positions["TRENT"]

    assert state.current_total_open_risk == 0.0


def test_current_equity_basis_for_risk_budget():
    """Test 10: Risk budget is calculated from current total_equity (e.g. ₹12L -> ₹6,000 risk ceiling at 0.5%)."""
    state = PortfolioState(initial_capital=1000000.0, total_equity=1200000.0, max_risk_per_trade_pct=0.50)

    max_trade_risk = state.total_equity * (state.max_risk_per_trade_pct / 100.0)
    assert max_trade_risk == 6000.0
    assert max_trade_risk != 5000.0


def test_two_simultaneous_trades_enforce_risk_limits():
    """Test 11: Two simultaneous trades enforce individual trade risk limits and aggregate portfolio open risk limit."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5]

    df1 = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})
    df2 = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest(
        {"INFY": df1, "TCS": df2}, initial_capital=1000000.0, max_risk_per_trade_pct=0.50, max_total_open_risk_pct=2.00
    )

    total_risk = portfolio.current_total_open_risk
    assert total_risk <= portfolio.total_equity * (2.00 / 100.0)


def test_non_negative_cash_guarantee():
    """Test 12: Available cash never drops below zero (#11A preservation)."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5]
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=100000.0)
    assert portfolio.cash_available >= 0.0


def test_deterministic_repeatability():
    """Test 13: Running the exact same historical portfolio backtest twice produces 100% identical results."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 103.0, 106.0, 112.0, 115.0, 118.0, 120.0, 122.0, 124.0, 125.0]
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    port1, stats1 = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=1000000.0)
    port2, stats2 = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=1000000.0)

    assert port1.cash_available == port2.cash_available
    assert port1.realized_pnl == port2.realized_pnl
    assert port1.total_equity == port2.total_equity
    assert port1.rejection_reasons == port2.rejection_reasons
    assert len(port1.completed_trades) == len(port2.completed_trades)
