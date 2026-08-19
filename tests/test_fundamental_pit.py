"""
P0 #12B — Fundamental Data Point-in-Time Integrity & Leakage Prevention Unit & Integration Tests.

Validates that:
  1. filing_date / available_at strictly controls visibility.
  2. period_end_date <= as_of_date NEVER grants early visibility if available_at > as_of_date.
  3. Records with missing filing_date / available_at fail closed (PIT_UNVERIFIED).
  4. Future fundamental mutations (available_at > T) cannot change score / output at T.
  5. Latest quarterly record is selected strictly by available_at <= as_of_date.
  6. Revisions are invisible before their available_at date.
  7. Historical query filtered by PointInTimeFilter rejects synthetic fallback data.
  8. Provider fallback for QuarterlyFinancials is rejected by PointInTimeFilter.
  9. AnnualRatios fallback without available_at is rejected by PointInTimeFilter.
 10. ShareholdingPattern fallback without available_at is rejected by PointInTimeFilter.
"""

import asyncio
from datetime import date, datetime
import pandas as pd
import pytest

from src.agents.fundamental_agent import FundamentalAnalysisAgent
from src.core.evidence import EvidenceGraph
from src.core.models import AnnualRatios, QuarterlyFinancials, ShareholdingPattern, SymbolMetadata
from src.data.fundamental_provider import ScreenerFundamentalProvider
from src.data.point_in_time import PointInTimeFilter


def test_fundamental_publication_date_controls_visibility():
    """1. Test filing_date / available_at strictly controls fundamental record visibility."""
    rec_a = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2026, 3, 31),
        filing_date=date(2026, 5, 15),
        available_at=date(2026, 5, 15),
        sales_crores=1200.0,
        sales_growth_yoy_pct=20.0,
        pat_crores=180.0,
        pat_growth_yoy_pct=25.0,
        ebitda_margin_pct=18.0,
        eps_inr=15.0,
        pit_status="VERIFIED",
    )

    assert len(PointInTimeFilter.filter_quarterly_financials([rec_a], date(2026, 4, 30))) == 0
    assert len(PointInTimeFilter.filter_quarterly_financials([rec_a], date(2026, 5, 14))) == 0
    assert len(PointInTimeFilter.filter_quarterly_financials([rec_a], date(2026, 5, 15))) == 1
    assert len(PointInTimeFilter.filter_quarterly_financials([rec_a], date(2026, 5, 16))) == 1


def test_period_end_does_not_grant_early_visibility():
    """2. Test period_end_date <= as_of_date does NOT grant early visibility if available_at > as_of_date."""
    rec = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2026, 3, 31),
        filing_date=date(2026, 5, 15),
        available_at=date(2026, 5, 15),
        sales_crores=1200.0,
        sales_growth_yoy_pct=20.0,
        pat_crores=180.0,
        pat_growth_yoy_pct=25.0,
        ebitda_margin_pct=18.0,
        eps_inr=15.0,
        pit_status="VERIFIED",
    )

    # period_end_date (2026-03-31) <= as_of_date (2026-04-01), but available_at is 2026-05-15
    filtered = PointInTimeFilter.filter_quarterly_financials([rec], date(2026, 4, 1))
    assert len(filtered) == 0


def test_missing_fundamental_availability_fails_closed():
    """3. Test record with missing filing_date and available_at (PIT_UNVERIFIED) is rejected for historical PIT use."""
    rec_unverified = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2026, 3, 31),
        filing_date=None,
        available_at=None,
        sales_crores=1200.0,
        sales_growth_yoy_pct=20.0,
        pat_crores=180.0,
        pat_growth_yoy_pct=25.0,
        ebitda_margin_pct=18.0,
        eps_inr=15.0,
        pit_status="PIT_UNVERIFIED",
    )

    filtered = PointInTimeFilter.filter_quarterly_financials([rec_unverified], date(2026, 6, 1))
    assert len(filtered) == 0


def test_future_fundamental_mutation_does_not_change_result_at_T():
    """4. Test mutating future fundamental records (available_at > T) leaves FundamentalAnalysisAgent score at T identical."""
    t_date = date(2026, 5, 10)

    q1_baseline = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2025, 12, 31),
        filing_date=date(2026, 2, 10),
        available_at=date(2026, 2, 10),
        sales_crores=1000.0,
        sales_growth_yoy_pct=15.0,
        pat_crores=150.0,
        pat_growth_yoy_pct=20.0,
        ebitda_margin_pct=18.0,
        eps_inr=12.0,
        pit_status="VERIFIED",
    )

    q2_future_orig = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2026, 3, 31),
        filing_date=date(2026, 5, 25),  # Available after T (2026-05-10)
        available_at=date(2026, 5, 25),
        sales_crores=1200.0,
        sales_growth_yoy_pct=20.0,
        pat_crores=180.0,
        pat_growth_yoy_pct=25.0,
        ebitda_margin_pct=19.0,
        eps_inr=15.0,
        pit_status="VERIFIED",
    )

    q2_future_mut = q2_future_orig.model_copy(update={
        "sales_growth_yoy_pct": -80.0,
        "pat_growth_yoy_pct": -90.0,
        "pat_crores": 10.0,
    })

    agent = FundamentalAnalysisAgent()
    meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail")
    dummy_df = pd.DataFrame()

    # Baseline run at T
    ctx_base = {"quarterly_financials": [q1_baseline, q2_future_orig], "as_of_date": t_date}
    out_base = asyncio.run(agent._analyze(meta, dummy_df, EvidenceGraph(), "run1", ctx_base))

    # Mutated run at T
    ctx_mut = {"quarterly_financials": [q1_baseline, q2_future_mut], "as_of_date": t_date}
    out_mut = asyncio.run(agent._analyze(meta, dummy_df, EvidenceGraph(), "run2", ctx_mut))

    assert out_base.score == out_mut.score
    assert out_base.signal == out_mut.signal
    assert out_base.metrics == out_mut.metrics


def test_latest_available_fundamental_selected_by_availability():
    """5. Test latest quarterly record is selected strictly by available_at <= as_of_date, not period_end_date."""
    q1 = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2025, 12, 31),
        filing_date=date(2026, 2, 15),
        available_at=date(2026, 2, 15),
        sales_crores=1000.0,
        sales_growth_yoy_pct=15.0,
        pat_crores=150.0,
        pat_growth_yoy_pct=15.0,
        ebitda_margin_pct=17.0,
        eps_inr=12.0,
        pit_status="VERIFIED",
    )

    q2 = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2026, 3, 31),
        filing_date=date(2026, 5, 15),
        available_at=date(2026, 5, 15),
        sales_crores=1200.0,
        sales_growth_yoy_pct=25.0,
        pat_crores=200.0,
        pat_growth_yoy_pct=30.0,
        ebitda_margin_pct=19.0,
        eps_inr=16.0,
        pit_status="VERIFIED",
    )

    records = [q1, q2]

    # D5 (2026-01-01): neither available
    assert len(PointInTimeFilter.filter_quarterly_financials(records, date(2026, 1, 1))) == 0

    # D10 (2026-02-15): Q1 available
    f_d10 = PointInTimeFilter.filter_quarterly_financials(records, date(2026, 2, 15))
    assert len(f_d10) == 1
    assert f_d10[0].period_end_date == date(2025, 12, 31)

    # D25 (2026-04-30): Q1 available only (Q2 period ended 03-31 but filing is 05-15!)
    f_d25 = PointInTimeFilter.filter_quarterly_financials(records, date(2026, 4, 30))
    assert len(f_d25) == 1
    assert f_d25[0].period_end_date == date(2025, 12, 31)

    # D30 (2026-05-15): both Q1 and Q2 available
    f_d30 = PointInTimeFilter.filter_quarterly_financials(records, date(2026, 5, 15))
    assert len(f_d30) == 2


def test_future_revision_is_not_visible_before_availability():
    """6. Test a revised filing is NOT visible before its revision available_at date."""
    orig_filing = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2026, 3, 31),
        filing_date=date(2026, 5, 10),
        available_at=date(2026, 5, 10),
        sales_crores=1000.0,
        sales_growth_yoy_pct=15.0,
        pat_crores=150.0,
        pat_growth_yoy_pct=20.0,
        ebitda_margin_pct=17.0,
        eps_inr=12.0,
        pit_status="VERIFIED",
    )

    revised_filing = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2026, 3, 31),
        filing_date=date(2026, 5, 25),  # Revision filed on May 25
        available_at=date(2026, 5, 25),
        sales_crores=1050.0,
        sales_growth_yoy_pct=20.0,
        pat_crores=160.0,
        pat_growth_yoy_pct=25.0,
        ebitda_margin_pct=18.0,
        eps_inr=13.0,
        pit_status="VERIFIED",
    )

    records = [orig_filing, revised_filing]

    # At 2026-05-15: Original filing visible, revision NOT visible
    f_may15 = PointInTimeFilter.filter_quarterly_financials(records, date(2026, 5, 15))
    assert len(f_may15) == 1
    assert f_may15[0].pat_growth_yoy_pct == 20.0

    # At 2026-05-25: Revision becomes visible
    f_may25 = PointInTimeFilter.filter_quarterly_financials(records, date(2026, 5, 25))
    assert len(f_may25) == 2


def test_historical_query_cannot_return_current_fallback_data():
    """7. Test historical query filtered by PointInTimeFilter rejects synthetic fallback data without available_at."""
    provider = ScreenerFundamentalProvider()
    fallback_records = asyncio.run(provider.get_quarterly_financials("UNKNOWN_SYM"))

    assert len(fallback_records) == 1
    assert fallback_records[0].pit_status == "PIT_UNVERIFIED"
    assert fallback_records[0].available_at is None

    # Filtering for historical date must return 0 records
    historical_pit_records = PointInTimeFilter.filter_quarterly_financials(fallback_records, date(2025, 6, 1))
    assert len(historical_pit_records) == 0


def test_provider_fallback_rejected_by_pit_filter():
    """8. Test provider fallback quarterly record passed into PointInTimeFilter is rejected."""
    provider = ScreenerFundamentalProvider()
    fallback = asyncio.run(provider.get_quarterly_financials("FALLBACK_TEST"))
    pit_filtered = PointInTimeFilter.filter_quarterly_financials(fallback, date(2026, 1, 1))
    assert len(pit_filtered) == 0


def test_annual_ratios_fallback_rejected_by_pit_filter():
    """9. Test provider fallback AnnualRatios without available_at is rejected by filter_annual_ratios."""
    provider = ScreenerFundamentalProvider()
    ratios = asyncio.run(provider.get_annual_ratios("FALLBACK_TEST"))

    assert ratios is not None
    assert ratios.available_at is None

    filtered = PointInTimeFilter.filter_annual_ratios([ratios], date(2026, 1, 1))
    assert len(filtered) == 0


def test_shareholding_fallback_rejected_by_pit_filter():
    """10. Test provider fallback ShareholdingPattern without available_at is rejected by filter_shareholding_patterns."""
    provider = ScreenerFundamentalProvider()
    shp = asyncio.run(provider.get_shareholding_pattern("FALLBACK_TEST"))

    assert shp is not None
    assert shp.available_at is None

    filtered = PointInTimeFilter.filter_shareholding_patterns([shp], date(2026, 1, 1))
    assert len(filtered) == 0
