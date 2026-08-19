"""
P0 #13 CORRECTION — Data Quality & Evidence Control Layer Unit & Integration Tests.

Validates:
  1. Same-day intraday future OHLCV timestamp (> decision_time) is a hard PIT_VIOLATION and is_trade_eligible = False.
  2. Same-day intraday future regime timestamp (> decision_time) is a hard PIT_VIOLATION.
  3. Same-day intraday future benchmark timestamp (> decision_time) is a hard PIT_VIOLATION.
  4. Fundamental record with pit_status="PIT_UNVERIFIED" or available_at=None is NOT pit_safe (pit_safe = False, status = PIT_VIOLATION).
  5. Future fundamental record (available_at > decision_time) is a hard PIT_VIOLATION.
  6. Valid fundamental record (available_at <= decision_time, pit_status="VERIFIED") returns VALID and pit_safe = True.
  7. Missing fundamentals (fundamentals = None) returns status = UNAVAILABLE, pit_safe = True (distinct from unverified fundamentals).
  8. PIT_VIOLATION forces overall_quality_score = 0.0 and is_trade_eligible = False regardless of high scores in other sources.
  9. Multi-source quality status independence is maintained (OHLCV=VALID, FUNDAMENTALS=PIT_VIOLATION, NEWS=UNAVAILABLE).
"""

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.core.models import QuarterlyFinancials, NewsArticle
from src.data.data_quality import (
    DataQualityGate,
    DataQualityResult,
    DataQualityStatus,
    SourceQualityResult,
)


def _generate_intraday_ohlcv_df() -> pd.DataFrame:
    """Generates intraday OHLCV DataFrame for 2026-08-19."""
    timestamps = [
        datetime(2026, 8, 19, 9, 30),
        datetime(2026, 8, 19, 10, 0),
        datetime(2026, 8, 19, 15, 0),
    ]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0, 101.0, 105.0],
        "high": [102.0, 103.0, 106.0],
        "low": [99.0, 100.0, 104.0],
        "close": [101.0, 102.0, 105.5],
        "volume": [10000, 15000, 20000],
    })


def _generate_valid_daily_df(num_bars: int = 60, end_date_str: str = "2026-06-30") -> pd.DataFrame:
    """Generates daily OHLCV DataFrame."""
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


def test_same_day_future_ohlcv_timestamp_is_pit_violation():
    """1. Test same-day OHLCV row at 15:00 relative to decision_time 10:00 is a PIT_VIOLATION."""
    decision_time = datetime(2026, 8, 19, 10, 0)
    df = _generate_intraday_ohlcv_df()  # Max timestamp is 15:00 > 10:00

    res = DataQualityGate.evaluate_evidence_quality(
        symbol="TRENT",
        df=df,
        as_of_date=decision_time,
        min_required_bars=1,
    )

    assert res.overall_status == DataQualityStatus.PIT_VIOLATION
    assert res.pit_safe is False
    assert res.is_trade_eligible is False
    assert "PIT_VIOLATION" in res.blocking_reasons


def test_same_day_future_regime_timestamp_is_pit_violation():
    """2. Test same-day market regime row at 15:00 relative to decision_time 10:00 is a PIT_VIOLATION."""
    decision_time = datetime(2026, 8, 19, 10, 0)
    df_valid_ohlcv = pd.DataFrame({
        "timestamp": [datetime(2026, 8, 19, 9, 30), datetime(2026, 8, 19, 10, 0)],
        "open": [100.0, 101.0], "high": [102.0, 103.0], "low": [99.0, 100.0],
        "close": [101.0, 102.0], "volume": [10000, 15000],
    })
    df_future_regime = _generate_intraday_ohlcv_df()  # Max timestamp 15:00

    res = DataQualityGate.evaluate_evidence_quality(
        symbol="TRENT",
        df=df_valid_ohlcv,
        as_of_date=decision_time,
        regime_df=df_future_regime,
        min_required_bars=1,
    )

    assert res.sources["MARKET_REGIME"].status == DataQualityStatus.PIT_VIOLATION
    assert res.sources["MARKET_REGIME"].pit_safe is False
    assert res.overall_status == DataQualityStatus.PIT_VIOLATION
    assert res.is_trade_eligible is False


def test_same_day_future_benchmark_timestamp_is_pit_violation():
    """3. Test same-day benchmark row at 15:00 relative to decision_time 10:00 is a PIT_VIOLATION."""
    decision_time = datetime(2026, 8, 19, 10, 0)
    df_valid_ohlcv = pd.DataFrame({
        "timestamp": [datetime(2026, 8, 19, 9, 30), datetime(2026, 8, 19, 10, 0)],
        "open": [100.0, 101.0], "high": [102.0, 103.0], "low": [99.0, 100.0],
        "close": [101.0, 102.0], "volume": [10000, 15000],
    })
    df_future_bench = _generate_intraday_ohlcv_df()  # Max timestamp 15:00

    res = DataQualityGate.evaluate_evidence_quality(
        symbol="TRENT",
        df=df_valid_ohlcv,
        as_of_date=decision_time,
        benchmark_df=df_future_bench,
        min_required_bars=1,
    )

    assert res.sources["BENCHMARK"].status == DataQualityStatus.PIT_VIOLATION
    assert res.sources["BENCHMARK"].pit_safe is False
    assert res.overall_status == DataQualityStatus.PIT_VIOLATION
    assert res.is_trade_eligible is False


def test_unverified_fundamental_is_not_pit_safe():
    """4. Test fundamental record with filing_date=None, available_at=None, pit_status='PIT_UNVERIFIED' has pit_safe=False."""
    decision_date = date(2026, 6, 30)
    df = _generate_valid_daily_df(60, "2026-06-30")

    unverified_q = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2026, 3, 31),
        filing_date=None,
        available_at=None,
        sales_crores=1000.0,
        sales_growth_yoy_pct=20.0,
        pat_crores=150.0,
        pat_growth_yoy_pct=25.0,
        ebitda_margin_pct=18.0,
        eps_inr=15.0,
        pit_status="PIT_UNVERIFIED",
    )

    res = DataQualityGate.evaluate_evidence_quality(
        symbol="TRENT",
        df=df,
        as_of_date=decision_date,
        fundamentals=[unverified_q],
    )

    fund_res = res.sources["FUNDAMENTALS"]
    assert fund_res.pit_safe is False
    assert fund_res.status != DataQualityStatus.VALID
    assert res.pit_safe is False
    assert res.is_trade_eligible is False


def test_future_fundamental_is_hard_pit_failure():
    """5. Test fundamental record with available_at (2026-08-20) > decision_date (2026-08-19) is a PIT_VIOLATION."""
    decision_date = date(2026, 8, 19)
    df = _generate_valid_daily_df(60, "2026-08-19")

    future_q = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2026, 6, 30),
        filing_date=date(2026, 8, 20),
        available_at=date(2026, 8, 20),
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
        as_of_date=decision_date,
        fundamentals=[future_q],
    )

    fund_res = res.sources["FUNDAMENTALS"]
    assert fund_res.status == DataQualityStatus.PIT_VIOLATION
    assert fund_res.pit_safe is False
    assert res.overall_status == DataQualityStatus.PIT_VIOLATION
    assert res.is_trade_eligible is False


def test_valid_fundamental():
    """6. Test valid fundamental record (available_at <= decision_date, pit_status='VERIFIED') returns VALID and pit_safe=True."""
    decision_date = date(2026, 6, 30)
    df = _generate_valid_daily_df(60, "2026-06-30")

    valid_q = QuarterlyFinancials(
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
        as_of_date=decision_date,
        fundamentals=[valid_q],
    )

    fund_res = res.sources["FUNDAMENTALS"]
    assert fund_res.status == DataQualityStatus.VALID
    assert fund_res.pit_safe is True
    assert fund_res.quality_score == 100.0


def test_missing_fundamentals():
    """7. Test fundamentals=None returns status=UNAVAILABLE and pit_safe=True (distinct from unverified fundamentals)."""
    decision_date = date(2026, 6, 30)
    df = _generate_valid_daily_df(60, "2026-06-30")

    res = DataQualityGate.evaluate_evidence_quality(
        symbol="TRENT",
        df=df,
        as_of_date=decision_date,
        fundamentals=None,
    )

    fund_res = res.sources["FUNDAMENTALS"]
    assert fund_res.status == DataQualityStatus.UNAVAILABLE
    assert fund_res.pit_safe is True
    assert fund_res.quality_score == 0.0
    assert "FUNDAMENTAL_UNAVAILABLE" in fund_res.reasons


def test_quality_score_cannot_hide_pit():
    """8. Test PIT_VIOLATION forces overall_quality_score = 0.0 and is_trade_eligible = False regardless of other sources."""
    as_of = date(2026, 6, 15)
    df_future = _generate_valid_daily_df(60, "2026-06-30")

    res = DataQualityGate.evaluate_evidence_quality(symbol="TRENT", df=df_future, as_of_date=as_of)

    assert res.overall_status == DataQualityStatus.PIT_VIOLATION
    assert res.overall_quality_score == 0.0
    assert res.is_trade_eligible is False
    assert "PIT_VIOLATION" in res.blocking_reasons


def test_source_status_independence():
    """9. Test source status independence (OHLCV=VALID, FUNDAMENTALS=PIT_VIOLATION, NEWS=UNAVAILABLE)."""
    decision_date = date(2026, 6, 30)
    df = _generate_valid_daily_df(60, "2026-06-30")

    unverified_q = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2026, 3, 31),
        filing_date=None,
        available_at=None,
        sales_crores=1000.0,
        sales_growth_yoy_pct=20.0,
        pat_crores=150.0,
        pat_growth_yoy_pct=25.0,
        ebitda_margin_pct=18.0,
        eps_inr=15.0,
        pit_status="PIT_UNVERIFIED",
    )

    res = DataQualityGate.evaluate_evidence_quality(
        symbol="TRENT",
        df=df,
        as_of_date=decision_date,
        fundamentals=[unverified_q],
        news=None,
    )

    assert res.sources["OHLCV"].status == DataQualityStatus.VALID
    assert res.sources["FUNDAMENTALS"].status == DataQualityStatus.PIT_VIOLATION
    assert res.sources["NEWS"].status == DataQualityStatus.UNAVAILABLE
    assert res.sources["OHLCV"].pit_safe is True
    assert res.sources["FUNDAMENTALS"].pit_safe is False
    assert res.sources["NEWS"].pit_safe is True
