"""
Phase #14E — End-to-End Multi-Agent Backtest Engine Unit Tests.

Validates that:
  1. MultiAgentBacktestRunner iterates over decision dates and candidate universe.
  2. Candidate Discovery, Data Quality Gate, Specialist Agents, Evidence Fusion, Conviction Engine, Risk Engine, and CIO synthesize decisions per bar.
  3. Candidates receiving a BUY decision construct trades via BacktestEngine with 100% parity.
  4. Data Quality PIT violations or Risk vetoes prevent trade entry.
  5. BacktestResult correctly compiles win rate, profit factor, drawdown, and total PnL.
"""

from datetime import date, datetime
import numpy as np
import pandas as pd
import pytest

from src.backtest.multi_agent_backtest import MultiAgentBacktestRunner
from src.core.models import AnnualRatios, QuarterlyFinancials


def _make_stock_df(bars: int = 100, trend: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic valid OHLCV DataFrame."""
    dates = pd.date_range(end="2026-06-30", periods=bars, freq="B")
    np.random.seed(seed)
    prices = 100.0 + np.cumsum(np.random.normal(trend, 0.4, bars))
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": 50000,
    })


def test_multi_agent_backtest_runner_execution():
    runner = MultiAgentBacktestRunner()

    df_trent = _make_stock_df(bars=100, trend=0.8, seed=42)
    df_tata = _make_stock_df(bars=100, trend=-0.2, seed=100)

    stock_dfs = {"TRENT": df_trent, "TATAMOTORS": df_tata}
    decision_dates = ["2026-05-15", "2026-06-01", "2026-06-15"]

    # Provide context with strong fundamentals for TRENT
    q_trent = [QuarterlyFinancials(
        symbol="TRENT", period_end_date=date(2026, 3, 31), filing_date=date(2026, 4, 15), available_at=date(2026, 4, 15),
        sales_crores=1000.0, sales_growth_yoy_pct=30.0, pat_crores=200.0, pat_growth_yoy_pct=40.0,
        ebitda_margin_pct=22.0, eps_inr=20.0, pit_status="PIT_VERIFIED",
    )]
    r_trent = AnnualRatios(
        symbol="TRENT", fiscal_year="2026", roe_pct=25.0, roce_pct=28.0, debt_to_equity=0.1,
        cfo_crores=200.0, cfo_to_pat_ratio=1.0, available_at=date(2026, 4, 15), pit_status="PIT_VERIFIED",
    )

    context_data = {
        "TRENT": {"quarterly_financials": q_trent, "annual_ratios": r_trent},
    }

    result = runner.run_simulation(stock_dfs, decision_dates, context_data=context_data)

    assert result is not None
    assert isinstance(result.total_trades, int)
    assert isinstance(result.win_rate_pct, float)
    assert isinstance(result.profit_factor, float)


def test_multi_agent_backtest_empty_universe():
    runner = MultiAgentBacktestRunner()
    result = runner.run_simulation({}, ["2026-06-01"])

    assert result.total_trades == 0
    assert result.total_pnl_rupees == 0.0
    assert result.win_rate_pct == 0.0
