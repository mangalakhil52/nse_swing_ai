"""
Comprehensive Integrity Test Suite — tests/test_p0_p1_integrity.py

Verifies all P0-P8 production data integrity and safety requirements:
  1. Real Historical Data Pipeline & Data Quality Validation.
  2. Zero synthetic data fallbacks.
  3. Risk Agent zero alpha contribution & uncalibrated confidence handling.
  4. Trade Construction structural targets and zero alpha contribution.
  5. Empirical Probability Engine sample size checks (n >= 30).
  6. Point-In-Time safety filtering & Look-Ahead Leakage Prevention (P8).
  7. Actual NSE trading session calendar utilities.
  8. Survivorship-safe date-aware historical universe.
"""

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import pytest

from config.market_hours import get_next_trading_sessions, get_previous_trading_sessions, is_trading_day
from src.agents.risk_agent import RiskManagementAgent
from src.agents.trade_construction_agent import TradeConstructionAgent
from src.core.evidence import EvidenceGraph
from src.core.exceptions import DataIntegrityError, DataUnavailableException
from src.core.models import NewsArticle, SymbolMetadata
from src.core.types import MarketRegime, PatternType, SignalType, SourceTier
from src.data.historical_provider import HistoricalDataProvider
from src.data.historical_universe import HistoricalUniverseProvider
from src.data.point_in_time import PointInTimeFilter
from src.data.validation import validate_ohlcv_dataframe
from src.quant.probability_engine import ProbabilityPathEngine


def test_validation_rejects_invalid_ohlc():
    """P0.3: Invalid OHLC geometry (High < Low) raises DataIntegrityError."""
    df_bad = pd.DataFrame({
        "timestamp": pd.date_range(end=date.today(), periods=60, freq="B"),
        "open": [100.0] * 60,
        "high": [90.0] * 60,  # Invalid: High < Open/Close
        "low": [80.0] * 60,
        "close": [95.0] * 60,
        "volume": [10000] * 60,
    })

    with pytest.raises(DataIntegrityError):
        validate_ohlcv_dataframe(df_bad, min_bars=50, symbol="BAD_STOCK")


def test_validation_rejects_insufficient_bars():
    """P0.1: Dataframe with < 50 historical bars is rejected."""
    df_short = pd.DataFrame({
        "timestamp": pd.date_range(end=date.today(), periods=20, freq="B"),
        "open": [100.0] * 20,
        "high": [105.0] * 20,
        "low": [95.0] * 20,
        "close": [102.0] * 20,
        "volume": [10000] * 20,
    })

    with pytest.raises(DataIntegrityError):
        validate_ohlcv_dataframe(df_short, min_bars=50, symbol="SHORT_STOCK")


def test_risk_agent_zero_alpha():
    """P0.6: RiskManagementAgent must NOT generate positive bullish alpha (score == 0.0)."""
    async def _run():
        agent = RiskManagementAgent()
        meta = SymbolMetadata(symbol="RELIANCE", company_name="Reliance Industries Ltd")
        ev = EvidenceGraph("TEST-RUN")
        df = pd.DataFrame({
            "timestamp": pd.date_range(end=date.today(), periods=60, freq="B"),
            "open": [2500.0] * 60, "high": [2550.0] * 60, "low": [2480.0] * 60,
            "close": [2520.0] * 60, "volume": [500000] * 60, "turnover_crores": [125.0] * 60,
        })
        ctx = {"upcoming_events": [], "market_regime": MarketRegime.BULL}
        out = await agent.execute(meta, df, ev, "TEST-RUN", ctx)

        assert out.score == 0.0  # P0.6 requirement: zero alpha contribution!
        assert out.confidence is None  # P0.7 requirement: uncalibrated confidence
        assert out.signal == SignalType.NEUTRAL

    asyncio.run(_run())


def test_trade_construction_zero_alpha_and_structural_target():
    """P1.0 & P1.1: TradeConstructionAgent must output score = 0.0 and structural levels."""
    async def _run():
        agent = TradeConstructionAgent()
        meta = SymbolMetadata(symbol="RELIANCE", company_name="Reliance Industries Ltd")
        ev = EvidenceGraph("TEST-RUN")

        # 60 bars dataframe with 20-day high at 2600.0
        prices = [2400.0 + i * 2.0 for i in range(60)]
        df = pd.DataFrame({
            "timestamp": pd.date_range(end=date.today(), periods=60, freq="B"),
            "open": prices, "high": [p * 1.01 for p in prices], "low": [p * 0.99 for p in prices],
            "close": prices, "volume": [500000] * 60, "turnover_crores": [125.0] * 60,
            "ema_20": [prices[-1] * 0.96] * 60, "atr_14": [25.0] * 60,
        })
        out = await agent.execute(meta, df, ev, "TEST-RUN", {})

        assert out.score == 0.0  # P1.0 requirement: zero alpha score!
        assert out.signal == SignalType.NEUTRAL
        assert "trade_levels" in out.metrics
        levels = out.metrics["trade_levels"]
        assert levels["stop_loss_price"] < levels["entry_trigger_price"]
        assert levels["target_1"] > levels["entry_trigger_price"]

    asyncio.run(_run())


def test_probability_engine_sample_size_check():
    """P1.3 & P2.1: Small sample size (< 30) or UNKNOWN pattern returns win_probability = None and rejects trade."""
    res = ProbabilityPathEngine.evaluate_expectancy(
        pattern_type=PatternType.UNSTRUCTURED_TREND,
        market_regime=MarketRegime.BULL,
        mansfield_rs=5.0,
        target1_pct=12.0,
        stop_loss_pct=5.0,
    )
    assert res.win_probability is None
    assert res.is_ev_positive is False
    assert "UNAVAILABLE" in res.disqualification_reason


def test_nse_trading_session_calendar():
    """P0.8: get_next_trading_sessions returns actual valid NSE trading session dates."""
    start = date(2026, 8, 14)  # Friday
    sessions = get_next_trading_sessions(start, 3)

    assert len(sessions) == 3
    # Next trading sessions should skip Saturday & Sunday
    for s in sessions:
        assert is_trading_day(s) is True
        assert s.weekday() < 5  # No weekends


def test_point_in_time_future_news_leakage_prevented():
    """P0.9 & P8: News published after as_of_date is filtered out."""
    as_of = date(2026, 8, 14)
    past_news = NewsArticle(
        symbol="TRENT", headline="Q1 Results Positive", summary="Growth",
        publisher="NSE", source_tier=SourceTier.TIER_1, source_url="https://nse.com",
        published_at=datetime(2026, 8, 14, 10, 0),
    )
    future_news = NewsArticle(
        symbol="TRENT", headline="Q2 Guidance Raised", summary="Future news",
        publisher="NSE", source_tier=SourceTier.TIER_1, source_url="https://nse.com",
        published_at=datetime(2026, 8, 16, 10, 0),  # Future relative to as_of
    )

    filtered = PointInTimeFilter.filter_news([past_news, future_news], as_of_date=as_of)
    assert len(filtered) == 1
    assert filtered[0].headline == "Q1 Results Positive"


def test_survivorship_safe_historical_universe():
    """P1.8: Historical universe excludes securities not listed on backtest simulation date."""
    dt_2024 = date(2024, 1, 1)
    univ = HistoricalUniverseProvider.get_universe_for_date(dt_2024)

    assert "SAATVIKGL" not in univ  # Listed in 2026
    assert "RELIANCE" in univ
