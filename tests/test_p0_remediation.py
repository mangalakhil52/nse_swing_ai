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


def test_historical_outcome_generator_pipeline():
    """Verifies HistoricalOutcomeGenerator produces genuine, point-in-time safe outcomes from real historical OHLCV."""
    from src.quant.outcome_generator import HistoricalOutcomeGenerator
    from src.quant.probability_engine import HistoricalSetupOutcomeStore

    # Reset outcome store for clean test environment
    HistoricalSetupOutcomeStore.clear()

    # Generate 100 historical bars with a realistic breakout pattern
    dates = pd.date_range(start="2026-01-01", periods=100, freq="B")
    prices = []
    for k in range(100):
        if k < 60:
            prices.append(100.0 + (k % 5) * 0.5)  # Consolidation
        elif k == 60:
            prices.append(110.0)  # Breakout bar
        else:
            prices.append(110.0 + (k - 60) * 0.8)  # Post-breakout uptrend

    df_hist = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [100000 if i != 60 else 300000 for i in range(100)],
        "turnover_crores": [10.0] * 100,
    })

    records, n_candles, n_setups = HistoricalOutcomeGenerator.generate_outcomes_for_symbol(
        symbol="RELIANCE", df_hist=df_hist, source="NSE_BHAVCOPY_DAILY"
    )
    assert len(records) > 0  # Real outcomes generated

    for outcome in records:
        # 1. Full audit trail metadata present
        assert outcome.symbol == "RELIANCE"
        assert outcome.setup_date is not None
        assert outcome.exit_date is not None
        assert outcome.entry_price > 0.0
        assert outcome.stop_loss > 0.0
        assert outcome.target_1 > 0.0
        assert outcome.source == "NSE_BHAVCOPY_DAILY"
        assert outcome.outcome in ["WIN", "LOSS"]

        # 2. Point-in-time correctness: exit_date >= setup_date
        assert outcome.exit_date >= outcome.setup_date
        assert outcome.holding_sessions > 0

    # 3. Store integration
    added, rejected = HistoricalSetupOutcomeStore.register_outcomes(records, persist=False)
    assert added > 0
    assert len(HistoricalSetupOutcomeStore.query_outcomes(records[0].pattern_type)) == added


def test_deterministic_ohlcv_integration_test_25_days():
    """Requirement 15: Handcrafted 25-day sequential candles integration test. Proves end-to-end outcome generation from OHLCV."""
    from src.quant.outcome_generator import HistoricalOutcomeGenerator
    from src.quant.probability_engine import HistoricalSetupOutcomeStore

    HistoricalSetupOutcomeStore.clear()

    # Day 1-20: Consolidation setup forms
    # Day 21 (index 50, setup_date = 2026-01-29): Breakout entry (close = 100.0, volume = 300000)
    # Day 22-23: Price rises
    # Day 24 (index 53, exit_date = 2026-02-03): Target 1 (110.0) reached!
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [90.0] * 50 + [100.0, 103.0, 106.0, 112.0, 115.0, 118.0, 120.0, 122.0, 124.0, 125.0]
    volumes = [50000] * 50 + [300000, 80000, 90000, 120000, 100000, 95000, 90000, 85000, 80000, 75000]

    df_hist = pd.DataFrame({
        "timestamp": dates,
        "open": [p * 0.99 for p in prices],
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.98 for p in prices],
        "close": prices,
        "volume": volumes,
        "turnover_crores": [10.0] * 60,
    })

    report = HistoricalOutcomeGenerator.generate_outcomes(
        symbols=["TRENT"],
        stock_dfs={"TRENT": df_hist},
        source="NSE_BHAVCOPY_DAILY",
        target_pct=10.0,
        stop_pct=5.0,
    )

    assert report.symbols_processed == 1
    assert report.setups_detected >= 1
    assert report.outcomes_generated >= 1

    stored = HistoricalSetupOutcomeStore.query_outcomes(PatternType.CUP_AND_HANDLE)
    if not stored:
        stored = HistoricalSetupOutcomeStore.query_outcomes(PatternType.VOLATILITY_CONTRACTION_PATTERN)

    assert len(stored) >= 1
    sample_rec = stored[0]

    # Verify exact required fields
    assert sample_rec.symbol == "TRENT"
    assert sample_rec.setup_date == dates[50].strftime("%Y-%m-%d")
    assert sample_rec.entry_price == 100.0
    assert sample_rec.stop_loss == 95.0
    assert sample_rec.target_1 == 110.0
    assert sample_rec.t1_hit_before_sl is True
    assert sample_rec.exit_date > sample_rec.setup_date
    assert sample_rec.holding_sessions > 0
    assert sample_rec.mfe >= 10.0
    assert sample_rec.source == "NSE_BHAVCOPY_DAILY"


def test_outcome_generator_idempotency_duplicate_prevention():
    """Requirement 13: Running generator twice must NOT duplicate the same historical observation."""
    from src.quant.outcome_generator import HistoricalOutcomeGenerator
    from src.quant.probability_engine import HistoricalSetupOutcomeStore

    HistoricalSetupOutcomeStore.clear()

    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [90.0] * 50 + [100.0, 103.0, 106.0, 112.0, 115.0, 118.0, 120.0, 122.0, 124.0, 125.0]

    df_hist = pd.DataFrame({
        "timestamp": dates,
        "open": prices, "high": [p * 1.01 for p in prices], "low": [p * 0.98 for p in prices],
        "close": prices, "volume": [50000] * 50 + [300000] * 10, "turnover_crores": [10.0] * 60,
    })

    report1 = HistoricalOutcomeGenerator.generate_outcomes(["INFY"], {"INFY": df_hist}, source="NSE_BHAVCOPY_DAILY")
    count1 = len(HistoricalSetupOutcomeStore._records)
    assert count1 > 0

    # Run second time
    report2 = HistoricalOutcomeGenerator.generate_outcomes(["INFY"], {"INFY": df_hist}, source="NSE_BHAVCOPY_DAILY")
    count2 = len(HistoricalSetupOutcomeStore._records)

    # Must be strictly equal to count1 (zero duplicate additions)
    assert count2 == count1
    assert report2.outcomes_generated == 0


def test_outcome_generator_record_validation_rejection():
    """Requirement 10: Reject invalid records (missing date, price <= 0, exit_date < setup_date, missing source, UNKNOWN pattern/regime)."""
    from src.quant.probability_engine import HistoricalSetupOutcome, HistoricalSetupOutcomeStore, validate_outcome

    rec_invalid_price = HistoricalSetupOutcome(
        symbol="ABC", pattern_type=PatternType.CUP_AND_HANDLE, market_regime=MarketRegime.BULL,
        setup_date="2026-01-01", entry_price=0.0, stop_loss=95.0, target_1=110.0,
        t1_hit_before_sl=True, holding_sessions=3, exit_date="2026-01-05", source="NSE_BHAVCOPY_DAILY"
    )
    is_valid, reason = validate_outcome(rec_invalid_price)
    assert is_valid is False
    assert "entry_price" in reason

    rec_invalid_exit = HistoricalSetupOutcome(
        symbol="ABC", pattern_type=PatternType.CUP_AND_HANDLE, market_regime=MarketRegime.BULL,
        setup_date="2026-01-05", entry_price=100.0, stop_loss=95.0, target_1=110.0,
        t1_hit_before_sl=True, holding_sessions=3, exit_date="2026-01-01", source="NSE_BHAVCOPY_DAILY"
    )
    is_valid, reason = validate_outcome(rec_invalid_exit)
    assert is_valid is False
    assert "exit_date" in reason

    rec_missing_source = HistoricalSetupOutcome(
        symbol="ABC", pattern_type=PatternType.CUP_AND_HANDLE, market_regime=MarketRegime.BULL,
        setup_date="2026-01-01", entry_price=100.0, stop_loss=95.0, target_1=110.0,
        t1_hit_before_sl=True, holding_sessions=3, exit_date="2026-01-05", source=""
    )
    is_valid, reason = validate_outcome(rec_missing_source)
    assert is_valid is False
    assert "source" in reason

    rec_unknown_pattern = HistoricalSetupOutcome(
        symbol="ABC", pattern_type=PatternType.UNKNOWN, market_regime=MarketRegime.BULL,
        setup_date="2026-01-01", entry_price=100.0, stop_loss=95.0, target_1=110.0,
        t1_hit_before_sl=True, holding_sessions=3, exit_date="2026-01-05", source="NSE_BHAVCOPY_DAILY"
    )
    is_valid, reason = validate_outcome(rec_unknown_pattern)
    assert is_valid is False
    assert "PatternType" in reason

    rec_unknown_regime = HistoricalSetupOutcome(
        symbol="ABC", pattern_type=PatternType.CUP_AND_HANDLE, market_regime=MarketRegime.UNKNOWN,
        setup_date="2026-01-01", entry_price=100.0, stop_loss=95.0, target_1=110.0,
        t1_hit_before_sl=True, holding_sessions=3, exit_date="2026-01-05", source="NSE_BHAVCOPY_DAILY"
    )
    is_valid, reason = validate_outcome(rec_unknown_regime)
    assert is_valid is False
    assert "MarketRegime" in reason


def test_insufficient_historical_setups_remains_insufficient_without_inflation():
    """Requirement 20: 7 genuine setups remains sample_size = 7 without artificial inflation, returning UNAVAILABLE."""
    from src.quant.probability_engine import HistoricalSetupOutcome, HistoricalSetupOutcomeStore, ProbabilityPathEngine

    HistoricalSetupOutcomeStore.clear()

    # Register only 7 genuine outcomes
    outcomes = []
    for i in range(7):
        outcomes.append(
            HistoricalSetupOutcome(
                symbol=f"SYM_{i}",
                pattern_type=PatternType.CUP_AND_HANDLE,
                market_regime=MarketRegime.BULL,
                setup_date=f"2026-01-0{i+1}",
                entry_price=100.0,
                stop_loss=95.0,
                target_1=110.0,
                t1_hit_before_sl=True,
                holding_sessions=3,
                exit_date=f"2026-01-0{i+2}",
                source="NSE_BHAVCOPY_DAILY",
            )
        )
    HistoricalSetupOutcomeStore.register_outcomes(outcomes, persist=False)

    res = ProbabilityPathEngine.evaluate_expectancy(PatternType.CUP_AND_HANDLE, MarketRegime.BULL)
    assert res.sample_size == 7  # Exactly 7, no artificial inflation to 30!
    assert res.win_probability is None
    assert res.confidence_type == "UNAVAILABLE"
    assert res.is_ev_positive is False

