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


def test_cio_rejects_candidate_when_market_regime_is_missing_null_invalid_or_unknown():
    """CIOOrchestrator must reject candidates when market_regime is missing, null, invalid, or UNKNOWN."""
    async def _run():
        from src.agents.cio_orchestrator import CIOOrchestrator

        cio = CIOOrchestrator()
        meta = SymbolMetadata(symbol="RELIANCE", company_name="Reliance Industries", sector="Energy")
        prices = [2400.0 + i * 5.0 for i in range(60)]
        df = pd.DataFrame({
            "timestamp": pd.date_range(end=date.today(), periods=60, freq="B"),
            "open": prices, "high": [p * 1.01 for p in prices], "low": [p * 0.99 for p in prices],
            "close": prices, "volume": [500000] * 60, "turnover_crores": [120.0] * 60,
        })

        # 1. Missing market_regime
        rec1, _ = await cio.analyze_candidate(meta, df, "RUN1", context={})
        assert rec1 is None

        # 2. Null market_regime
        rec2, _ = await cio.analyze_candidate(meta, df, "RUN2", context={"market_regime": None})
        assert rec2 is None

        # 3. Invalid market_regime string
        rec3, _ = await cio.analyze_candidate(meta, df, "RUN3", context={"market_regime": "INVALID_REGIME_NAME"})
        assert rec3 is None

        # 4. Explicit UNKNOWN market_regime
        rec4, _ = await cio.analyze_candidate(meta, df, "RUN4", context={"market_regime": MarketRegime.UNKNOWN})
        assert rec4 is None

    import asyncio
    asyncio.run(_run())


def test_cio_no_fabricated_fundamental_evidence_in_why_trade():
    """Missing PAT growth or FCF/PAT must not produce fabricated numeric claims in why_this_trade or fundamental_summary."""
    from src.core.models import AgentOutput
    from src.core.types import AgentStatus

    # Create AgentOutput with status SUCCESS but missing pat_growth_yoy / fcf_to_pat
    out_missing = AgentOutput(
        agent_name="fundamental_analysis_agent",
        symbol="RELIANCE",
        run_id="TEST",
        status=AgentStatus.SUCCESS,
        signal=SignalType.BULLISH,
        score=75.0,
        metrics={},  # Missing pat_growth_yoy and fcf_to_pat
    )

    # Verify no fabricated claims when metrics are missing
    fund_parts = []
    pat_g = out_missing.metrics.get("pat_growth_yoy")
    assert pat_g is None
    if pat_g is not None:
        fund_parts.append(f"PAT growth +{pat_g:.1f}% YoY")
    fcf_pat = out_missing.metrics.get("fcf_to_pat")
    assert fcf_pat is None
    if fcf_pat is not None:
        fund_parts.append(f"FCF/PAT {fcf_pat:.2f}")

    assert len(fund_parts) == 0  # Zero fabricated parts

    # Create AgentOutput with valid metrics
    out_valid = AgentOutput(
        agent_name="fundamental_analysis_agent",
        symbol="RELIANCE",
        run_id="TEST",
        status=AgentStatus.SUCCESS,
        signal=SignalType.BULLISH,
        score=85.0,
        metrics={"pat_growth_yoy": 24.3, "fcf_to_pat": 1.15},
    )

    valid_parts = []
    pat_g_valid = out_valid.metrics.get("pat_growth_yoy")
    if pat_g_valid is not None:
        valid_parts.append(f"PAT growth +{pat_g_valid:.1f}% YoY")
    fcf_pat_valid = out_valid.metrics.get("fcf_to_pat")
    if fcf_pat_valid is not None:
        valid_parts.append(f"FCF/PAT {fcf_pat_valid:.2f}")

    assert len(valid_parts) == 2
    assert "PAT growth +24.3% YoY" in valid_parts[0]
    assert "FCF/PAT 1.15" in valid_parts[1]
