"""
P0 #12E — Historical Universe & Survivorship-Bias Control Unit & Integration Tests.

Validates that:
  1. Securities are NOT eligible prior to their listing_date (IPO).
  2. Historically active securities (active at T, delisted in future > T) remain eligible at T.
  3. Securities delisted on or before T are excluded.
  4. Today's current survivor list CANNOT be used as a proxy for historical universe at T.
  5. Future IPOs cannot enter historical signal pipeline inputs.
  6. Cross-sectional ranking respects historical universe eligibility at T.
"""

from datetime import date
import pytest

from src.core.models import SymbolMetadata
from src.data.historical_universe import HistoricalUniverseProvider
from src.quant.relative_strength import RelativeStrengthEngine


def test_stock_not_eligible_before_listing():
    """1. Test stock is NOT eligible prior to its listing_date."""
    sec = SymbolMetadata(
        symbol="EPACKPEB",
        company_name="EPACK Durable Ltd",
        listing_date=date(2026, 8, 1),
    )

    # At 2026-07-31: Listing date (2026-08-01) > as_of_date -> Excluded
    res_before = HistoricalUniverseProvider.filter_universe_by_date([sec], date(2026, 7, 31))
    assert len(res_before) == 0

    # At 2026-08-01: Eligible
    res_on = HistoricalUniverseProvider.filter_universe_by_date([sec], date(2026, 8, 1))
    assert len(res_on) == 1
    assert res_on[0].symbol == "EPACKPEB"


def test_historically_active_stock_can_exist_after_later_delisting():
    """2. Test a stock active at T remains eligible at T even if delisted in the future (> T)."""
    sec = SymbolMetadata(
        symbol="DECOMM_CORP",
        company_name="Decommissioned Corp Ltd",
        listing_date=date(2020, 1, 1),
        delisting_date=date(2026, 5, 15),  # Delisted in May 2026
    )

    # At 2026-04-01: Delisting is in the future (May 2026 > April 2026) -> Eligible at T
    res_t = HistoricalUniverseProvider.filter_universe_by_date([sec], date(2026, 4, 1))
    assert len(res_t) == 1

    # At 2026-05-15: Delisted -> Excluded
    res_after = HistoricalUniverseProvider.filter_universe_by_date([sec], date(2026, 5, 15))
    assert len(res_after) == 0


def test_current_survivor_list_cannot_be_used_as_historical_universe():
    """3. Test current survivor list is NOT used directly as historical proxy when listing dates differ."""
    survivor_a = SymbolMetadata(symbol="RELIANCE", company_name="Reliance Industries", listing_date=date(1995, 1, 1))
    survivor_b_future = SymbolMetadata(symbol="FUTURE_IPO", company_name="Future Tech Ltd", listing_date=date(2026, 8, 1))
    survivor_c = SymbolMetadata(symbol="TCS", company_name="Tata Consultancy Services", listing_date=date(2004, 8, 25))

    current_universe = [survivor_a, survivor_b_future, survivor_c]

    # Query historical universe at 2026-05-01
    hist_symbols = HistoricalUniverseProvider.get_universe_for_date(date(2026, 5, 1), current_universe)

    assert "RELIANCE" in hist_symbols
    assert "TCS" in hist_symbols
    assert "FUTURE_IPO" not in hist_symbols
    assert len(hist_symbols) == 2


def test_historical_universe_fails_closed_when_membership_unknown():
    """4. Test missing or future listing date fails closed by excluding security."""
    unlisted_sec = SymbolMetadata(
        symbol="UNLISTED_TECH",
        company_name="Unlisted Tech Ltd",
        listing_date=date(2030, 1, 1),
    )

    filtered = HistoricalUniverseProvider.filter_universe_by_date([unlisted_sec], date(2026, 6, 1))
    assert len(filtered) == 0


def test_cross_sectional_ranking_uses_historical_universe():
    """5. Test cross-sectional percentile ranking ranks only eligible historical securities at T."""
    sec_a = SymbolMetadata(symbol="STOCK_A", company_name="Stock A Ltd", listing_date=date(2020, 1, 1))
    sec_b = SymbolMetadata(symbol="STOCK_B", company_name="Stock B Ltd", listing_date=date(2026, 9, 1))  # Not listed at T

    # Filter universe at T = 2026-05-01
    as_of = date(2026, 5, 1)
    eligible = HistoricalUniverseProvider.filter_universe_by_date([sec_a, sec_b], as_of)
    eligible_symbols = {s.symbol for s in eligible}

    raw_scores = {"STOCK_A": 15.2, "STOCK_B": 99.0}  # STOCK_B has high score in raw data
    filtered_scores = {k: v for k, v in raw_scores.items() if k in eligible_symbols}

    percentiles = RelativeStrengthEngine.calculate_universe_percentile_ranks(filtered_scores)

    assert "STOCK_A" in percentiles
    assert "STOCK_B" not in percentiles
    assert percentiles["STOCK_A"] == 0.0  # Sole eligible stock


def test_future_ipo_cannot_enter_historical_signal_pipeline():
    """6. Test future IPO symbol cannot be returned by get_universe_for_date at T."""
    ipo_sec = SymbolMetadata(
        symbol="NEXGEN_AI",
        company_name="NexGen AI Ltd",
        listing_date=date(2026, 12, 1),
    )

    symbols = HistoricalUniverseProvider.get_universe_for_date(date(2026, 5, 15), [ipo_sec])
    assert "NEXGEN_AI" not in symbols
    assert len(symbols) == 0
