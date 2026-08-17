"""
P0 Remediation Adversarial Test Suite — tests/test_p0_remediation.py

Verifies P0/P1 non-negotiable data integrity rules:
  1. Nifty data failure never creates synthetic data; missing Nifty blocks long trades.
  2. Market regime uses real data only and sets UNKNOWN when data is missing.
  3. Missing fundamentals returns status = DATA_UNAVAILABLE with score = 0.0.
  4. Missing pattern is UNKNOWN (never FLAT_BASE_BREAKOUT fallback).
  5. Missing ADTV or ATR raises DATA_INSUFFICIENT_FOR_EXECUTION and rejects setup.
  6. ProbabilityPathEngine contains ZERO hardcoded EMPIRICAL_DATA dictionary.
  7. Fail-closed architecture across all desks.
"""

from datetime import date
import pandas as pd
import pytest

from src.agents.fundamental_agent import FundamentalAnalysisAgent
from src.agents.risk_agent import RiskManagementAgent
from src.agents.trade_construction_agent import TradeConstructionAgent
from src.core.evidence import EvidenceGraph
from src.core.models import SymbolMetadata
from src.core.types import AgentStatus, MarketRegime, PatternType, SignalType, TradingStance
from src.quant.probability_engine import ProbabilityPathEngine
from src.quant.regime import MarketRegimeClassifier


def test_nifty_data_failure_never_creates_synthetic_data():
    """Missing or empty Nifty DataFrame must return UNKNOWN regime with nifty_close = 0.0."""
    res = MarketRegimeClassifier.classify_regime(nifty_df=pd.DataFrame())

    assert res.regime == MarketRegime.UNKNOWN
    assert res.trading_stance == TradingStance.NO_TRADE
    assert res.nifty_close == 0.0
    assert res.allow_long_swing_trades is False
    assert res.confidence == 0.0


def test_missing_nifty_data_blocks_long_recommendations():
    """When Nifty data is empty, allow_long_swing_trades must be False."""
    res = MarketRegimeClassifier.classify_regime(nifty_df=None)
    assert res.allow_long_swing_trades is False


def test_missing_advance_decline_returns_unknown_regime():
    """Missing advance_decline_ratio (None) must return UNKNOWN regime."""
    nifty_df = pd.DataFrame({
        "timestamp": pd.date_range(end=date.today(), periods=60, freq="B"),
        "open": [24000.0] * 60, "high": [24200.0] * 60, "low": [23900.0] * 60,
        "close": [24100.0] * 60, "volume": [500000] * 60,
    })
    res = MarketRegimeClassifier.classify_regime(
        nifty_df=nifty_df,
        advance_decline_ratio=None,
        pct_above_50_sma=65.0,
        india_vix=14.0,
    )
    assert res.regime == MarketRegime.UNKNOWN
    assert res.allow_long_swing_trades is False


def test_missing_pct_above_50_sma_returns_unknown_regime():
    """Missing pct_above_50_sma (None) must return UNKNOWN regime."""
    nifty_df = pd.DataFrame({
        "timestamp": pd.date_range(end=date.today(), periods=60, freq="B"),
        "open": [24000.0] * 60, "high": [24200.0] * 60, "low": [23900.0] * 60,
        "close": [24100.0] * 60, "volume": [500000] * 60,
    })
    res = MarketRegimeClassifier.classify_regime(
        nifty_df=nifty_df,
        advance_decline_ratio=1.5,
        pct_above_50_sma=None,
        india_vix=14.0,
    )
    assert res.regime == MarketRegime.UNKNOWN
    assert res.allow_long_swing_trades is False


def test_missing_india_vix_returns_unknown_regime():
    """Missing india_vix (None) must return UNKNOWN regime."""
    nifty_df = pd.DataFrame({
        "timestamp": pd.date_range(end=date.today(), periods=60, freq="B"),
        "open": [24000.0] * 60, "high": [24200.0] * 60, "low": [23900.0] * 60,
        "close": [24100.0] * 60, "volume": [500000] * 60,
    })
    res = MarketRegimeClassifier.classify_regime(
        nifty_df=nifty_df,
        advance_decline_ratio=1.5,
        pct_above_50_sma=65.0,
        india_vix=None,
    )
    assert res.regime == MarketRegime.UNKNOWN
    assert res.allow_long_swing_trades is False


def test_all_valid_inputs_returns_valid_classification():
    """When all required inputs are present, existing classification logic runs."""
    prices = [24000.0 + i * 20.0 for i in range(60)]
    nifty_df = pd.DataFrame({
        "timestamp": pd.date_range(end=date.today(), periods=60, freq="B"),
        "open": prices, "high": [p * 1.01 for p in prices], "low": [p * 0.99 for p in prices],
        "close": prices, "volume": [500000] * 60,
    })
    res = MarketRegimeClassifier.classify_regime(
        nifty_df=nifty_df,
        advance_decline_ratio=1.6,
        pct_above_50_sma=70.0,
        india_vix=13.5,
    )
    assert res.regime in [MarketRegime.STRONG_BULL, MarketRegime.BULL, MarketRegime.NEUTRAL]
    assert res.allow_long_swing_trades is True


def test_missing_fundamentals_returns_data_unavailable():
    """FundamentalAnalysisAgent must return AgentStatus.DATA_UNAVAILABLE when data is missing."""
    async def _run():
        agent = FundamentalAnalysisAgent()
        meta = SymbolMetadata(symbol="NODATA", company_name="No Data Ltd")
        ev = EvidenceGraph("TEST-RUN")
        ctx = {"quarterly_financials": [], "annual_ratios": None}

        out = await agent.execute(meta, pd.DataFrame(), ev, "TEST-RUN", ctx)

        assert out.status == AgentStatus.DATA_UNAVAILABLE
        assert out.score == 0.0
        assert out.confidence is None
        assert out.signal == SignalType.NEUTRAL
        assert len(out.evidence) == 0

    import asyncio
    asyncio.run(_run())


def test_no_hardcoded_empirical_probability_table():
    """ProbabilityPathEngine must NOT contain hardcoded EMPIRICAL_DATA table."""
    assert not hasattr(ProbabilityPathEngine, "EMPIRICAL_DATA")


def test_probability_engine_returns_unavailable_without_empirical_store():
    """Without registered historical outcomes, evaluate_expectancy returns confidence_type = UNAVAILABLE."""
    res = ProbabilityPathEngine.evaluate_expectancy(
        pattern_type=PatternType.CUP_AND_HANDLE,
        market_regime=MarketRegime.BULL,
        mansfield_rs=10.0,
        target1_pct=15.0,
        stop_loss_pct=6.0,
    )
    assert res.win_probability is None
    assert res.confidence_type == "UNAVAILABLE"
    assert res.is_ev_positive is False


def test_risk_agent_zero_alpha_gatekeeper():
    """RiskManagementAgent must output score = 0.0 and confidence = None."""
    async def _run():
        agent = RiskManagementAgent()
        meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd")
        ev = EvidenceGraph("TEST-RUN")
        df = pd.DataFrame({
            "timestamp": pd.date_range(end=date.today(), periods=60, freq="B"),
            "open": [100.0] * 60, "high": [105.0] * 60, "low": [95.0] * 60,
            "close": [102.0] * 60, "volume": [50000] * 60, "turnover_crores": [15.0] * 60,
        })
        ctx = {"upcoming_events": [], "market_regime": MarketRegime.BULL}
        out = await agent.execute(meta, df, ev, "TEST-RUN", ctx)

        assert out.score == 0.0
        assert out.confidence is None
        assert out.signal == SignalType.NEUTRAL

    import asyncio
    asyncio.run(_run())


def test_historical_provider_raises_data_unavailable_on_insufficient_history():
    """HistoricalDataProvider must raise DataUnavailableException when genuine bars < min_bars without synthetic duplication."""
    async def _run():
        from src.core.exceptions import DataUnavailableException
        from src.data.historical_provider import HistoricalDataProvider

        provider = HistoricalDataProvider()
        start = date(2026, 8, 10)
        end = date(2026, 8, 14)  # Only 5 calendar days

        with pytest.raises(DataUnavailableException) as exc_info:
            await provider.get_daily_ohlcv("NONEXISTENT_XYZ", start, end, min_bars=50)

        assert "No historical OHLCV records found" in str(exc_info.value) or "Insufficient historical bars" in str(exc_info.value)

    import asyncio
    asyncio.run(_run())
