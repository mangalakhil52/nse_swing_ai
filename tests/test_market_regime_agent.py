"""Tests for #14F MarketRegimeAgent."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.agents.market_regime_agent import MarketRegimeAgent
from src.core.models import SymbolMetadata
from src.core.types import AgentStatus, MarketRegime, SignalType, TradingStance


def _nifty(n: int = 150) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    close = np.linspace(22000.0, 25000.0, n)
    return pd.DataFrame({
        "timestamp": dates,
        "open": close * 0.998,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": np.full(n, 2_000_000),
    })


@pytest.mark.asyncio
async def test_market_regime_agent_returns_pit_safe_contract():
    agent = MarketRegimeAgent()
    decision_time = date(2026, 6, 30)
    out = await agent.analyze_contract(
        SymbolMetadata(symbol="TRENT", company_name="Trent Ltd"),
        pd.DataFrame(),
        decision_time,
        "TEST-14F",
        {
            "nifty_df": _nifty(),
            "advance_decline_ratio": 1.6,
            "pct_above_50_sma": 70.0,
            "india_vix": 14.0,
        },
    )
    assert out.status == AgentStatus.SUCCESS
    assert out.pit_safe is True
    assert out.signal in (SignalType.BULLISH, SignalType.NEUTRAL, SignalType.BEARISH)
    assert out.evidence


@pytest.mark.asyncio
async def test_missing_regime_input_fails_closed():
    agent = MarketRegimeAgent()
    out = await agent.analyze_contract(
        SymbolMetadata(symbol="TRENT", company_name="Trent Ltd"),
        pd.DataFrame(),
        date(2026, 6, 30),
        "TEST-14F",
        {"nifty_df": _nifty(), "advance_decline_ratio": 1.6},
    )
    assert out.status == AgentStatus.DATA_UNAVAILABLE
    assert out.signal == SignalType.UNKNOWN
    assert out.pit_safe is False


@pytest.mark.asyncio
async def test_future_nifty_mutation_cannot_change_decision():
    agent = MarketRegimeAgent()
    decision_time = date(2026, 6, 30)
    base = _nifty()
    mutated = base.copy()
    mask = pd.to_datetime(mutated["timestamp"]).dt.date > decision_time
    mutated.loc[mask, "close"] *= 10.0
    mutated.loc[mask, "high"] *= 10.0

    kwargs = {
        "advance_decline_ratio": 1.6,
        "pct_above_50_sma": 70.0,
        "india_vix": 14.0,
    }
    a = await agent.analyze_contract(SymbolMetadata(symbol="TRENT", company_name="Trent Ltd"), pd.DataFrame(), decision_time, "A", {"nifty_df": base, **kwargs})
    b = await agent.analyze_contract(SymbolMetadata(symbol="TRENT", company_name="Trent Ltd"), pd.DataFrame(), decision_time, "B", {"nifty_df": mutated, **kwargs})
    assert a.signal == b.signal
    assert a.score == b.score
    assert a.pit_safe == b.pit_safe
    # StructuredEvidence uses `observation` as its canonical human-readable field.
    assert a.evidence[0].observation == b.evidence[0].observation


@pytest.mark.asyncio
async def test_no_benchmark_fails_closed():
    agent = MarketRegimeAgent()
    out = await agent.analyze_contract(
        SymbolMetadata(symbol="TRENT", company_name="Trent Ltd"),
        pd.DataFrame(),
        date(2026, 6, 30),
        "TEST-14F",
        {
            "nifty_df": None,
            "advance_decline_ratio": 1.6,
            "pct_above_50_sma": 70.0,
            "india_vix": 14.0,
        },
    )
    assert out.status == AgentStatus.DATA_UNAVAILABLE
    assert out.signal == SignalType.UNKNOWN
    assert out.pit_safe is False
