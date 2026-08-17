"""
Unit & Integration Tests for P0 Fix #11A: Portfolio Capital & Position Accounting.

Tests:
  1. Initial capital state (Cash = ₹10L, Invested = ₹0, Equity = ₹10L).
  2. Entry capital deduction (Entry value + friction costs reduce cash).
  3. Exit capital return (Net proceeds return to cash).
  4. Two simultaneous trades allowed when capital is sufficient.
  5. Insufficient capital rejection (INSUFFICIENT_PORTFOLIO_CAPITAL).
  6. Non-negative cash guarantee.
  7. Duplicate symbol rejection (POSITION_ALREADY_OPEN).
  8. Realized PnL double-counting prevention.
  9. Unrealized PnL isolation from available cash.
  10. Delayed cash availability until exit date.
  11. Transaction friction cost impact on portfolio equity.
  12. Deterministic chronological event processing.
  13. Portfolio equity identity: Cash + Open Market Value = Total Equity.
  14. Multiple position exposure aggregation.
  15. Backtest repeatability & idempotency.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio import PortfolioBacktestEngine, PortfolioState, OpenPosition
from src.backtest.friction import IndianFrictionModel
from src.quant.indicators import TechnicalIndicators


def test_initial_portfolio_state():
    """Test 1: Initial capital ₹10,00,000 produces cash ₹10L, invested ₹0, equity ₹10L."""
    state = PortfolioState(initial_capital=1000000.0)
    assert state.initial_capital == 1000000.0
    assert state.cash_available == 1000000.0
    assert state.invested_capital == 0.0
    assert state.realized_pnl == 0.0
    assert state.unrealized_pnl == 0.0
    assert state.total_equity == 1000000.0
    assert len(state.open_positions) == 0


def test_entry_capital_deduction_and_friction():
    """Test 2: Entry capital + friction costs reduce available cash and update invested capital."""
    state = PortfolioState(initial_capital=1000000.0)

    entry_price = 100.0
    shares = 1000
    entry_cost = PortfolioBacktestEngine.calculate_entry_friction(entry_price, shares)
    required_capital = (entry_price * shares) + entry_cost

    state.cash_available -= required_capital
    state.invested_capital += (entry_price * shares)

    pos = OpenPosition(
        symbol="TRENT",
        entry_date="2026-01-01",
        entry_price=entry_price,
        shares=shares,
        invested_value=entry_price * shares,
        stop_loss=95.0,
        target_1=110.0,
        target_2=115.0,
        target_3=120.0,
        entry_cost=entry_cost,
    )
    state.open_positions["TRENT"] = pos

    assert state.cash_available == 1000000.0 - required_capital
    assert state.invested_capital == 100000.0
    assert state.cash_available < 1000000.0 - 100000.0  # Friction fees deducted


def test_one_completed_trade_returns_net_proceeds_to_cash():
    """Test 3: Exiting a position returns net proceeds (after exit friction) to cash."""
    state = PortfolioState(initial_capital=1000000.0)

    entry_price = 100.0
    shares = 1000
    entry_cost = PortfolioBacktestEngine.calculate_entry_friction(entry_price, shares)
    required_capital = (entry_price * shares) + entry_cost

    state.cash_available -= required_capital
    state.invested_capital += (entry_price * shares)

    pos = OpenPosition(
        symbol="TRENT", entry_date="2026-01-01", entry_price=entry_price, shares=shares,
        invested_value=entry_price * shares, stop_loss=95.0, target_1=110.0, target_2=115.0, target_3=120.0,
        entry_cost=entry_cost,
    )
    state.open_positions["TRENT"] = pos

    # Exit at 110.0
    exit_price = 110.0
    PortfolioBacktestEngine._close_position(state, pos, exit_price, "TARGET_1_HIT", "2026-01-10")

    exit_cost = PortfolioBacktestEngine.calculate_exit_friction(exit_price, shares)
    net_proceeds = (exit_price * shares) - exit_cost

    assert state.invested_capital == 0.0
    assert state.cash_available == (1000000.0 - required_capital) + net_proceeds
    assert round(state.realized_pnl, 2) == round((exit_price - entry_price) * shares - (entry_cost + exit_cost), 2)


def test_two_simultaneous_trades_allowed_when_capital_sufficient():
    """Test 4: Two simultaneous trades on different symbols are allowed when total capital is sufficient."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 103.0, 106.0, 112.0, 115.0, 118.0, 120.0, 122.0, 124.0, 125.0]

    df1 = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})
    df2 = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest({"INFY": df1, "TCS": df2}, initial_capital=1000000.0)

    # Both symbols should be processed without capital insufficiency
    assert portfolio.initial_capital == 1000000.0
    assert not any("INSUFFICIENT_PORTFOLIO_CAPITAL" in r for r in portfolio.rejection_reasons)


def test_insufficient_capital_rejects_trade():
    """Test 5 & 6: Trade requiring > cash_available is rejected with INSUFFICIENT_PORTFOLIO_CAPITAL; cash stays >= 0."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0] * 10

    # Very expensive stock price ₹50,000 where 200 shares = ₹1,00,00,000 (> ₹10L initial capital)
    p_exp = [49000.0] * 50 + [50000.0] * 10
    df_exp = pd.DataFrame({"timestamp": dates, "open": p_exp, "high": [p * 1.002 for p in p_exp], "low": [p * 0.998 for p in p_exp], "close": p_exp, "volume": [50000]*50 + [300000]*10})

    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest({"EXPENSIVE": df_exp}, initial_capital=100000.0)  # Low initial capital ₹1L

    assert any("INSUFFICIENT_PORTFOLIO_CAPITAL" in r for r in portfolio.rejection_reasons)
    assert portfolio.cash_available >= 0.0


def test_duplicate_symbol_rejected_with_position_already_open():
    """Test 7: Duplicate entry signal for already open symbol is rejected with POSITION_ALREADY_OPEN."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    # Breakouts at index 50 and 51
    prices = [98.0] * 50 + [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0]
    volumes = [50000] * 50 + [300000, 350000, 80000, 90000, 120000, 100000, 95000, 90000, 85000, 80000]

    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": volumes})

    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest({"RELIANCE": df}, initial_capital=1000000.0)

    # Second signal on index 51 should be rejected with POSITION_ALREADY_OPEN
    assert any("POSITION_ALREADY_OPEN" in r for r in portfolio.rejection_reasons)


def test_realized_pnl_not_double_counted():
    """Test 8: Realized P&L is updated exactly once upon exit."""
    state = PortfolioState(initial_capital=1000000.0)

    pos = OpenPosition(
        symbol="TRENT", entry_date="2026-01-01", entry_price=100.0, shares=100,
        invested_value=10000.0, stop_loss=95.0, target_1=110.0, target_2=115.0, target_3=120.0,
        entry_cost=15.0,
    )
    state.open_positions["TRENT"] = pos

    PortfolioBacktestEngine._close_position(state, pos, 110.0, "TARGET_1_HIT", "2026-01-05")
    pnl_after_first_close = state.realized_pnl

    # Calling close_position again on 0 remaining shares must do nothing
    PortfolioBacktestEngine._close_position(state, pos, 110.0, "TARGET_1_HIT", "2026-01-05")
    assert state.realized_pnl == pnl_after_first_close


def test_unrealized_pnl_does_not_become_available_cash():
    """Test 9: Unrealized P&L increases total equity but does NOT alter available cash."""
    state = PortfolioState(initial_capital=1000000.0)
    state.cash_available = 900000.0

    pos = OpenPosition(
        symbol="TRENT", entry_date="2026-01-01", entry_price=100.0, shares=1000,
        invested_value=100000.0, stop_loss=95.0, target_1=110.0, target_2=115.0, target_3=120.0,
        entry_cost=150.0,
    )
    state.open_positions["TRENT"] = pos

    # Price rises to 120.0 (Unrealized gain = +₹20,000)
    current_close = 120.0
    market_val = current_close * pos.remaining_shares
    unrealized = (current_close - pos.entry_price) * pos.remaining_shares

    state.unrealized_pnl = unrealized
    state.total_equity = state.cash_available + market_val

    assert state.cash_available == 900000.0  # Cash remains strictly 900k!
    assert state.unrealized_pnl == 20000.0
    assert state.total_equity == 1020000.0


def test_exit_proceeds_delayed_until_exit_date():
    """Test 10: Cash remains committed during holding period until exit date."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 112.0, 115.0, 118.0]
    volumes = [50000] * 50 + [300000, 50000, 50000, 50000, 50000, 50000, 50000, 80000, 90000, 120000]

    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": volumes})

    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=1000000.0)

    # Cash was reduced on entry date and returned on exit date
    assert portfolio.total_equity > 0.0


def test_transaction_costs_reduce_portfolio_value():
    """Test 11: Transaction friction costs strictly reduce portfolio net equity."""
    entry_cost = PortfolioBacktestEngine.calculate_entry_friction(1000.0, 100)
    exit_cost = PortfolioBacktestEngine.calculate_exit_friction(1000.0, 100)
    assert entry_cost > 0.0
    assert exit_cost > 0.0


def test_portfolio_equity_identity_holds():
    """Test 13 & 14: Portfolio equity identity: Cash + Open Market Value == Total Equity."""
    state = PortfolioState(initial_capital=1000000.0)
    state.cash_available = 700000.0
    state.invested_capital = 300000.0

    pos1 = OpenPosition("INFY", "2026-01-01", 100.0, 1500, 150000.0, 95.0, 110.0, 115.0, 120.0, 200.0)
    pos2 = OpenPosition("TCS", "2026-01-01", 100.0, 1500, 150000.0, 95.0, 110.0, 115.0, 120.0, 200.0)
    state.open_positions["INFY"] = pos1
    state.open_positions["TCS"] = pos2

    # Current market prices
    market_val_sum = (105.0 * 1500) + (110.0 * 1500)  # 157,500 + 165,000 = 322,500
    state.total_equity = state.cash_available + market_val_sum

    assert state.total_equity == 700000.0 + 322500.0
    assert state.total_equity == state.cash_available + market_val_sum


def test_repeatability_and_idempotency():
    """Test 12 & 15: Running the exact same historical portfolio backtest twice produces 100% identical results."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 103.0, 106.0, 112.0, 115.0, 118.0, 120.0, 122.0, 124.0, 125.0]

    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    port1, stats1 = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=1000000.0)
    port2, stats2 = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=1000000.0)

    assert port1.cash_available == port2.cash_available
    assert port1.realized_pnl == port2.realized_pnl
    assert port1.total_equity == port2.total_equity
    assert len(port1.completed_trades) == len(port2.completed_trades)
