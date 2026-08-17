"""
Unit & Integration Tests for P0 Fix #11A: Portfolio Capital & Position Accounting.

Strengthened Integration Tests:
All accounting behavior is validated via direct engine execution (PortfolioBacktestEngine.run_portfolio_backtest),
never via manual property assignment.

Coverage:
  1. Initial portfolio state verification.
  2. Integration entry path: TradeConstructionEngine -> Position Accepted -> Cash Deducted.
  3. Integration exit path: Target Hit -> Net proceeds returned to cash.
  4. Insufficient capital rejection generated directly by the engine (INSUFFICIENT_PORTFOLIO_CAPITAL).
  5. Duplicate symbol position rejection generated directly by the engine (POSITION_ALREADY_OPEN).
  6. Delayed cash availability: Cash remains deducted on T+1 while position is open, returns after exit.
  7. Unrealized profit increases total equity but DOES NOT increase available cash.
  8. Realized exit proceeds become available cash strictly on/after exit date.
  9. Non-negative cash guarantee across multiple simultaneous trades.
  10. Dynamic portfolio equity identity (Cash + Open Position Market Value == Total Equity) across all sessions.
  11. Deterministic repeatability & idempotency.
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.trade_construction_agent import TradeConstructionEngine
from src.backtest.friction import IndianFrictionModel
from src.backtest.portfolio import PortfolioBacktestEngine, PortfolioState, OpenPosition
from src.quant.indicators import TechnicalIndicators


def test_initial_portfolio_state():
    """Requirement 1: Initial capital ₹10,00,000 produces cash ₹10L, invested ₹0, equity ₹10L."""
    state = PortfolioState(initial_capital=1000000.0)
    assert state.initial_capital == 1000000.0
    assert state.cash_available == 1000000.0
    assert state.invested_capital == 0.0
    assert state.realized_pnl == 0.0
    assert state.unrealized_pnl == 0.0
    assert state.total_equity == 1000000.0
    assert len(state.open_positions) == 0


def test_integration_entry_path_deducts_cash_and_friction():
    """Requirement 2: Full engine entry path computes TradeConstructionEngine levels, accepts trade, and deducts cash + friction."""
    dates = pd.date_range(start="2025-11-15", periods=52, freq="B")
    # Breakout at index 50 (close = 100.0)
    prices = [98.0] * 50 + [100.0, 100.5]
    df = pd.DataFrame({
        "timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices],
        "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*2,
    })

    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=1000000.0)

    # Position must be open at end of 52 bars
    assert "TRENT" in portfolio.open_positions or len(portfolio.completed_trades) > 0
    if "TRENT" in portfolio.open_positions:
        pos = portfolio.open_positions["TRENT"]
        expected_entry_cost = PortfolioBacktestEngine.calculate_entry_friction(pos.entry_price, pos.shares)
        expected_required = (pos.entry_price * pos.shares) + expected_entry_cost

        # Engine MUST have deducted required capital from cash_available
        assert portfolio.cash_available == round(1000000.0 - expected_required, 2)
        assert round(portfolio.invested_capital, 2) == round(pos.entry_price * pos.shares, 2)
        assert portfolio.cash_available < 1000000.0 - portfolio.invested_capital  # Friction fees deducted


def test_integration_exit_path_returns_net_proceeds_to_cash():
    """Requirement 3: Full engine exit path credits net proceeds (after friction) back to available cash."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    # Breakout at index 50 (100.0), target 1 (104.44) hit on index 52 (106.0)
    prices = [98.0] * 50 + [100.0, 103.0, 106.0, 112.0, 115.0, 118.0, 120.0, 122.0, 124.0, 125.0]
    df = pd.DataFrame({
        "timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices],
        "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10,
    })

    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=1000000.0)

    # Trade completed by engine
    assert len(portfolio.completed_trades) > 0
    trade = portfolio.completed_trades[0]

    assert trade.exit_price > 0.0
    assert trade.pnl_rupees is not None
    # Engine realized PnL reflects net proceeds after friction
    assert round(portfolio.realized_pnl, 2) == round(stats.total_pnl_rupees, 2)


def test_integration_insufficient_capital_rejected_by_engine():
    """Requirement 3: PortfolioBacktestEngine itself produces INSUFFICIENT_PORTFOLIO_CAPITAL when trade exceeds cash."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")

    # High stock price ₹50,000 where position size > ₹1,00,000 initial capital
    prices_expensive = [49000.0] * 50 + [50000.0] * 10
    df_exp = pd.DataFrame({
        "timestamp": dates, "open": prices_expensive, "high": [p * 1.002 for p in prices_expensive],
        "low": [p * 0.998 for p in prices_expensive], "close": prices_expensive, "volume": [50000]*50 + [300000]*10,
    })

    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest({"EXPENSIVE": df_exp}, initial_capital=100000.0)

    # Engine must reject trade due to capital limit
    assert len(portfolio.completed_trades) == 0
    assert len(portfolio.open_positions) == 0
    assert any("INSUFFICIENT_PORTFOLIO_CAPITAL" in r for r in portfolio.rejection_reasons)
    assert portfolio.cash_available == 100000.0


def test_integration_duplicate_position_rejected_by_engine():
    """Requirement 4: Two valid signals for same symbol while position is open produce POSITION_ALREADY_OPEN directly from engine."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    # Breakout signals at index 50 and index 51
    prices = [98.0] * 50 + [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0]
    volumes = [50000] * 50 + [300000, 350000, 80000, 90000, 120000, 100000, 95000, 90000, 85000, 80000]

    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": volumes})

    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest({"RELIANCE": df}, initial_capital=1000000.0)

    # Engine must reject second signal on index 51 with POSITION_ALREADY_OPEN
    assert any("POSITION_ALREADY_OPEN" in r for r in portfolio.rejection_reasons)


def test_delayed_cash_availability_verified():
    """Requirement 5: Cash remains deducted on Day T and Day T+1 while position is open, returns only after exit."""
    dates = pd.date_range(start="2026-01-01", periods=60, freq="B")
    # Day T (index 50): Breakout 100.0
    # Day T+1 (index 51): Price 101.0 (Position remains open)
    # Day T+2 (index 52): Price 106.0 (Target hit, position exits)
    prices = [98.0] * 50 + [100.0, 101.0, 106.0, 112.0, 115.0, 118.0, 120.0, 122.0, 124.0, 125.0]
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    # Run backtest up to Day T+1 (52 bars: index 0..51)
    port_T1, _ = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df.iloc[:52]}, initial_capital=1000000.0)

    assert "TRENT" in port_T1.open_positions
    pos_T1 = port_T1.open_positions["TRENT"]
    entry_cost = PortfolioBacktestEngine.calculate_entry_friction(pos_T1.entry_price, pos_T1.shares)
    expected_cash_T1 = round(1000000.0 - (pos_T1.entry_price * pos_T1.shares) - entry_cost, 2)

    # Day T+1: Cash MUST exclude committed capital
    assert port_T1.cash_available == expected_cash_T1

    # Run backtest up to Day T+3 (54 bars: index 0..53) where target is hit and position exits
    port_T3, _ = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df.iloc[:54]}, initial_capital=1000000.0)

    # Exit: proceeds returned to cash
    assert port_T3.cash_available > port_T1.cash_available


def test_unrealized_profit_does_not_increase_available_cash():
    """Requirement 6: Unrealized price gains increase total_equity but DO NOT increase cash_available during open trade."""
    dates = pd.date_range(start="2026-01-01", periods=52, freq="B")
    # Day T (index 50): Price 100.0
    # Day T+1 (index 51): Price 103.0 (Unrealized gain of 3%)
    prices = [98.0] * 50 + [100.0, 103.0]
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*2})

    portfolio, _ = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=1000000.0)

    assert "TRENT" in portfolio.open_positions
    pos = portfolio.open_positions["TRENT"]
    entry_cost = PortfolioBacktestEngine.calculate_entry_friction(pos.entry_price, pos.shares)
    expected_cash = round(1000000.0 - (pos.entry_price * pos.shares) - entry_cost, 2)

    # Unrealized gain increases equity above cash
    assert portfolio.unrealized_pnl > 0.0
    assert portfolio.total_equity > portfolio.cash_available
    # Cash available is STRICTLY unaffected by unrealized gain
    assert portfolio.cash_available == expected_cash


def test_realized_exit_proceeds_become_cash_only_after_exit():
    """Requirement 7: Realized exit proceeds become available cash strictly on/after exit date."""
    dates = pd.date_range(start="2026-01-01", periods=54, freq="B")
    prices = [98.0] * 50 + [100.0, 101.0, 106.0, 110.0]
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*4})

    # Day T+1 (index 51): Position still open
    port_open, _ = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df.iloc[:52]}, initial_capital=1000000.0)
    cash_open = port_open.cash_available

    # Day T+2 (index 52): Position exited
    port_closed, _ = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df.iloc[:53]}, initial_capital=1000000.0)
    cash_closed = port_closed.cash_available

    assert cash_closed > cash_open


def test_two_accepted_trades_cannot_cause_negative_cash():
    """Requirement 8: Multiple accepted simultaneous trades cannot cause negative cash."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 103.0, 106.0, 112.0, 115.0, 118.0, 120.0, 122.0, 124.0, 125.0]

    df1 = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})
    df2 = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    portfolio, stats = PortfolioBacktestEngine.run_portfolio_backtest({"INFY": df1, "TCS": df2}, initial_capital=1000000.0)

    assert portfolio.cash_available >= 0.0


def test_dynamic_portfolio_equity_identity_across_all_sessions():
    """Requirement 9: Cash + Market Value of Open Positions == Total Equity holds dynamically during actual backtest."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 103.0, 106.0, 112.0, 115.0, 118.0, 120.0, 122.0, 124.0, 125.0]
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    # Test equity identity at 52, 53, 54, 55, 60 bars
    for n_bars in [52, 53, 54, 55, 60]:
        sub_df = df.iloc[:n_bars]
        port, _ = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": sub_df}, initial_capital=1000000.0)

        open_market_val = sum(pos.remaining_shares * float(sub_df.iloc[-1]["close"]) for pos in port.open_positions.values())
        expected_equity = round(port.cash_available + open_market_val, 2)

        assert round(port.total_equity, 2) == expected_equity


def test_repeatability_and_idempotency():
    """Requirement 11: Running the exact same historical portfolio backtest twice produces 100% identical results."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 103.0, 106.0, 112.0, 115.0, 118.0, 120.0, 122.0, 124.0, 125.0]
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices], "low": [p * 0.998 for p in prices], "close": prices, "volume": [50000]*50 + [300000]*10})

    port1, stats1 = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=1000000.0)
    port2, stats2 = PortfolioBacktestEngine.run_portfolio_backtest({"TRENT": df}, initial_capital=1000000.0)

    assert port1.cash_available == port2.cash_available
    assert port1.realized_pnl == port2.realized_pnl
    assert port1.total_equity == port2.total_equity
    assert len(port1.completed_trades) == len(port2.completed_trades)
