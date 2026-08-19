"""
P0 #13 — Data Quality & Evidence Control Layer Unit & Integration Tests.

Validates that:
  1. Fully valid deterministic dataset returns DataQualityStatus.VALID and is_trade_eligible = True.
  2. Broken OHLC geometry (high < low) returns DataQualityStatus.INVALID and is_trade_eligible = False.
  3. Missing required OHLC column or null values returns DataQualityStatus.INVALID.
  4. Future data (available_at > decision_time or timestamp > decision_time) triggers a hard failure (PIT_VIOLATION) and blocks trade eligibility.
  5. Missing fundamentals are reported as DataQualityStatus.UNAVAILABLE, NOT treated as positive/100% quality.
  6. Missing news is reported as DataQualityStatus.UNAVAILABLE, NOT interpreted as negative or positive news.
  7. Stale data (timestamp age > staleness threshold) returns DataQualityStatus.DEGRADED.
  8. PIT violation cannot be hidden by a high quality score; overall_status is forced to PIT_VIOLATION and is_trade_eligible = False.
  9. Multi-source evidence statuses are reported independently (OHLCV=VALID, NEWS=UNAVAILABLE).
 10. Rejection output contains machine-readable reason strings (e.g. OHLC_INVALID_GEOMETRY, PIT_VIOLATION).
"""

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.core.models import QuarterlyFinancials, NewsArticle
from src.core.types import SentimentType, SourceTier
from src.data.data_quality import (
    DataQualityGate,
    DataQualityResult,
    DataQualityStatus,
    SourceQualityResult,
)


def _generate_valid_ohlcv_df(num_bars: int = 60, end_date_str: str = "2026-06-30") -> pd.DataFrame:
    """Helper generating valid OHLCV DataFrame."""
    dates = pd.date_range(end=end_date_str, periods=num_bars, freq="B")
    np.random.seed(42)
    prices = 500.0 + np.cumsum(np.random.normal(0, 2, num_bars))
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": 50000,
    })


def test_valid_data_is_accepted():
    """1. Test fully valid deterministic dataset returns DataQualityStatus.VALID and is_trade_eligible = True."""
    as_of = date(2026, 6, 30)
    df = _generate_valid_ohlcv_df(60, "2026-06-30")

    q1 = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2026, 3, 31),
        filing_date=date(2026, 5, 15),
        available_at=date(2026, 5, 15),
        sales_crores=1000.0,
        sales_growth_yoy_pct=20.0,
        pat_crores=150.0,
        pat_growth_yoy_pct=25.0,
        ebitda_margin_pct=18.0,
        eps_inr=15.0,
        pit_status="VERIFIED",
    )

    res = DataQualityGate.evaluate_evidence_quality(
        symbol="TRENT",
        df=df,
        as_of_date=as_of,
        fundamentals=[q1],
        regime_df=df,
        benchmark_df=df,
    )

    assert res.overall_status == DataQualityStatus.VALID
    assert res.pit_safe is True
    assert res.is_trade_eligible is True
    assert res.overall_quality_score == 100.0
    assert len(res.blocking_reasons) == 0


def test_invalid_ohlc_is_rejected():
    """2. Test broken OHLC geometry (high < low) returns DataQualityStatus.INVALID and is_trade_eligible = False."""
    as_of = date(2026, 6, 30)
    df = _generate_valid_ohlcv_df(60, "2026-06-30")

    # Corrupt bar at index 30: High (400) < Low (500)
    df.loc[30, "high"] = 400.0
    df.loc[30, "low"] = 500.0

    res = DataQualityGate.evaluate_evidence_quality(symbol="TRENT", df=df, as_of_date=as_of)

    assert res.overall_status == DataQualityStatus.INVALID
    assert res.is_trade_eligible is False
    assert "OHLC_INVALID_GEOMETRY" in res.blocking_reasons


def test_missing_ohlc_is_rejected():
    """3. Test missing required OHLC column or null values returns DataQualityStatus.INVALID."""
    as_of = date(2026, 6, 30)
    df = _generate_valid_ohlcv_df(60, "2026-06-30")

    # Drop required column 'close'
    df_missing = df.drop(columns=["close"])

    res = DataQualityGate.evaluate_evidence_quality(symbol="TRENT", df=df_missing, as_of_date=as_of)

    assert res.overall_status == DataQualityStatus.INVALID
    assert res.is_trade_eligible is False
    assert "OHLC_MISSING_COLUMN" in res.blocking_reasons


def test_future_data_is_hard_failure():
    """4. Test future data (timestamp > decision_time) triggers a hard failure (PIT_VIOLATION)."""
    as_of = date(2026, 6, 15)  # Decision date is June 15
    df_future = _generate_valid_ohlcv_df(60, "2026-06-30")  # Data extends to June 30 > June 15

    res = DataQualityGate.evaluate_evidence_quality(symbol="TRENT", df=df_future, as_of_date=as_of)

    assert res.overall_status == DataQualityStatus.PIT_VIOLATION
    assert res.pit_safe is False
    assert res.is_trade_eligible is False
    assert "PIT_VIOLATION" in res.blocking_reasons


def test_missing_fundamentals_are_not_treated_as_positive():
    """5. Test missing fundamentals are reported as DataQualityStatus.UNAVAILABLE, NOT treated as positive/100% quality."""
    as_of = date(2026, 6, 30)
    df = _generate_valid_ohlcv_df(60, "2026-06-30")

    res = DataQualityGate.evaluate_evidence_quality(
        symbol="TRENT",
        df=df,
        as_of_date=as_of,
        fundamentals=None,  # Missing fundamentals
    )

    fund_source = res.sources["FUNDAMENTALS"]
    assert fund_source.status == DataQualityStatus.UNAVAILABLE
    assert fund_source.quality_score == 0.0
    assert "FUNDAMENTAL_UNAVAILABLE" in fund_source.reasons


def test_missing_news_is_not_negative_news():
    """6. Test missing news is reported as DataQualityStatus.UNAVAILABLE, NOT interpreted as negative or positive news."""
    as_of = date(2026, 6, 30)
    df = _generate_valid_ohlcv_df(60, "2026-06-30")

    res = DataQualityGate.evaluate_evidence_quality(
        symbol="TRENT",
        df=df,
        as_of_date=as_of,
        news=None,  # Missing news
    )

    news_source = res.sources["NEWS"]
    assert news_source.status == DataQualityStatus.UNAVAILABLE
    assert news_source.quality_score == 0.0
    assert "NEWS_UNAVAILABLE" in news_source.reasons


def test_stale_data_is_detected():
    """7. Test stale OHLCV data (latest bar age > max staleness days) returns DataQualityStatus.DEGRADED."""
    as_of = date(2026, 6, 30)
    # Latest bar is June 20 -> 10 days stale relative to June 30
    df_stale = _generate_valid_ohlcv_df(60, "2026-06-20")

    res = DataQualityGate.evaluate_evidence_quality(
        symbol="TRENT",
        df=df_stale,
        as_of_date=as_of,
        max_staleness_days=4,
    )

    ohlcv_source = res.sources["OHLCV"]
    assert ohlcv_source.status == DataQualityStatus.DEGRADED
    assert "OHLC_STALE_DATA" in ohlcv_source.reasons


def test_pit_violation_cannot_be_hidden_by_high_quality_score():
    """8. Test PIT violation cannot be hidden by a high quality score; overall_status is forced to PIT_VIOLATION."""
    as_of = date(2026, 6, 15)
    # High-quality clean OHLCV data extending past decision_time to June 30
    df_future = _generate_valid_ohlcv_df(60, "2026-06-30")

    res = DataQualityGate.evaluate_evidence_quality(symbol="TRENT", df=df_future, as_of_date=as_of)

    assert res.overall_status == DataQualityStatus.PIT_VIOLATION
    assert res.overall_quality_score == 0.0
    assert res.is_trade_eligible is False
    assert "PIT_VIOLATION" in res.blocking_reasons


def test_multi_source_quality_is_reported_independently():
    """9. Test multi-source evidence statuses are reported independently (OHLCV=VALID, FUNDAMENTALS=VALID, NEWS=UNAVAILABLE)."""
    as_of = date(2026, 6, 30)
    df = _generate_valid_ohlcv_df(60, "2026-06-30")

    q1 = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2026, 3, 31),
        filing_date=date(2026, 5, 15),
        available_at=date(2026, 5, 15),
        sales_crores=1000.0,
        sales_growth_yoy_pct=20.0,
        pat_crores=150.0,
        pat_growth_yoy_pct=25.0,
        ebitda_margin_pct=18.0,
        eps_inr=15.0,
        pit_status="VERIFIED",
    )

    res = DataQualityGate.evaluate_evidence_quality(
        symbol="TRENT",
        df=df,
        as_of_date=as_of,
        fundamentals=[q1],
        news=None,  # Missing news
    )

    assert res.sources["OHLCV"].status == DataQualityStatus.VALID
    assert res.sources["FUNDAMENTALS"].status == DataQualityStatus.VALID
    assert res.sources["NEWS"].status == DataQualityStatus.UNAVAILABLE
    assert res.sources["OHLCV"].pit_safe is True
    assert res.sources["FUNDAMENTALS"].pit_safe is True


def test_rejection_contains_machine_readable_reason():
    """10. Test rejection output contains machine-readable reason strings."""
    as_of = date(2026, 6, 30)
    df = _generate_valid_ohlcv_df(60, "2026-06-30")
    df.loc[10, "high"] = -10.0  # Invalid non-positive price

    res = DataQualityGate.evaluate_evidence_quality(symbol="TRENT", df=df, as_of_date=as_of)

    assert res.is_trade_eligible is False
    assert len(res.blocking_reasons) > 0
    assert any(r in res.blocking_reasons for r in ["OHLC_NON_POSITIVE_PRICE", "OHLC_INVALID_GEOMETRY"])
