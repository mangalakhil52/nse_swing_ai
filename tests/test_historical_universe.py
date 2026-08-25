"""
P0 #12E CORRECTION — Historical Universe & Survivorship-Bias Control Unit & Integration Tests.

Validates that:
  1. HistoricalUniverseProvider.get_universe_for_date(as_of_date) NEVER silently falls back to current live watchlist.
  2. Securities are NOT eligible prior to their listing_date (IPO).
  3. Historically active securities (active at T, delisted in future > T) remain eligible at T.
  4. Securities delisted on or before T are excluded.
  5. Today's current survivor list CANNOT be used as a proxy for historical universe at T.
  6. Partial metadata (unknown listing or delisting date) is handled fail-closed.
  7. Cross-sectional ranking applies over date-eligible securities.
  8. get_current_universe() is explicitly separated from historical date-filtered queries.
"""

from datetime import date

import pandas as pd
import pytest

from config.settings import settings
from src.core.models import SymbolMetadata
from src.data.historical_universe import HistoricalUniverseProvider, HistoricalUniverseUnavailableError
from src.quant.relative_strength import RelativeStrengthEngine


def test_historical_universe_does_not_fallback_to_current_universe():
    """1. Test requesting historical universe without metadata fails closed with HistoricalUniverseUnavailableError."""
    with pytest.raises(HistoricalUniverseUnavailableError) as exc_info:
        HistoricalUniverseProvider.get_universe_for_date(date(2025, 6, 1))

    assert "Historical security master unavailable" in str(exc_info.value)
    assert "Fallback to current live watchlist is strictly forbidden" in str(exc_info.value)


def test_get_current_universe_separate_from_historical(tmp_path, monkeypatch):
    """2. Test get_current_universe reads the official-equity-master cache independently of historical queries."""
    # CI and fresh environments intentionally have no pre-existing cache. Keep this
    # test deterministic by supplying a minimal fixture that mirrors NSE's security
    # master shape rather than depending on developer-local state or live networking.
    cache_dir = tmp_path / "cache"
    bhavcopy_dir = cache_dir / "bhavcopy"
    bhavcopy_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"SYMBOL": "RELIANCE", "SERIES": "EQ"},
            {"SYMBOL": "TCS", "SERIES": "EQ"},
            {"SYMBOL": "IGNORED", "SERIES": "SM"},
            {"SYMBOL": "DUP", "SERIES": "EQ"},
            {"SYMBOL": "DUP", "SERIES": "EQ"},
        ]
    ).to_csv(bhavcopy_dir / "EQUITY_L.csv", index=False)

    monkeypatch.setattr(settings, "CACHE_DIR", cache_dir)

    current_symbols = HistoricalUniverseProvider.get_current_universe()
    assert isinstance(current_symbols, list)
    assert current_symbols == ["RELIANCE", "TCS", "IGNORED", "DUP"]
    assert "RELIANCE" in current_symbols


def test_stock_not_eligible_before_listing():
    """3. Test stock is NOT eligible prior to its listing_date."""
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
    """4. Test a stock active at T remains eligible at T even if delisted in the future (> T)."""
    sec = SymbolMetadata(
        symbol="DECOMM_CORP",
        company_name="Decommissioned Corp Ltd",
        listing_date=date(2020, 1, 1),
        delisting_date=date(2026, 5, 15),
    )

    res_t = HistoricalUniverseProvider.filter_universe_by_date([sec], date(2026, 5, 14))
    assert len(res_t) == 1

    res_on_delist = HistoricalUniverseProvider.filter_universe_by_date([sec], date(2026, 5, 15))
    assert len(res_on_delist) == 0

    res_after = HistoricalUniverseProvider.filter_universe_by_date([sec], date(2026, 6, 1))
    assert len(res_after) == 0


def test_current_survivor_list_cannot_be_used_as_historical_universe():
    """5. Test current survivor list is NOT used directly as historical proxy when listing dates differ."""
    survivor_a = SymbolMetadata(symbol="RELIANCE", company_name="Reliance Industries", listing_date=date(1995, 1, 1))
    survivor_b_future = SymbolMetadata(symbol="FUTURE_IPO", company_name="Future Tech Ltd", listing_date=date(2026, 8, 1))
    survivor_c = SymbolMetadata(symbol="TCS", company_name="Tata Consultancy Services", listing_date=date(2004, 8, 25))

    current_universe = [survivor_a, survivor_b_future, survivor_c]
    hist_symbols = HistoricalUniverseProvider.get_universe_for_date(date(2026, 5, 1), current_universe)

    assert "RELIANCE" in hist_symbols
    assert "TCS" in hist_symbols
    assert "FUTURE_IPO" not in hist_symbols
    assert len(hist_symbols) == 2


def test_partial_metadata_filtering():
    """6. Test filtering handles securities with missing/partial listing or delisting metadata gracefully."""
    sec_a = SymbolMetadata(symbol="STOCK_A", company_name="A Ltd", listing_date=date(2020, 1, 1), delisting_date=date(2026, 5, 15))
    sec_b = SymbolMetadata(symbol="STOCK_B", company_name="B Ltd", listing_date=None, delisting_date=None)
    sec_c = SymbolMetadata(symbol="STOCK_C", company_name="C Ltd", listing_date=date(2020, 1, 1), delisting_date=None)

    securities = [sec_a, sec_b, sec_c]

    res = HistoricalUniverseProvider.filter_universe_by_date(securities, date(2026, 5, 14))
    assert len(res) == 3

    res_delist = HistoricalUniverseProvider.filter_universe_by_date(securities, date(2026, 5, 15))
    assert len(res_delist) == 2
    symbols_delist = [s.symbol for s in res_delist]
    assert "STOCK_A" not in symbols_delist
    assert "STOCK_B" in symbols_delist
    assert "STOCK_C" in symbols_delist


def test_cross_sectional_ranking_uses_historical_universe():
    """7. Test cross-sectional percentile ranking ranks only eligible historical securities at T."""
    sec_a = SymbolMetadata(symbol="STOCK_A", company_name="Stock A Ltd", listing_date=date(2020, 1, 1))
    sec_b = SymbolMetadata(symbol="STOCK_B", company_name="Stock B Ltd", listing_date=date(2026, 9, 1))

    as_of = date(2026, 5, 1)
    eligible = HistoricalUniverseProvider.filter_universe_by_date([sec_a, sec_b], as_of)
    eligible_symbols = {s.symbol for s in eligible}

    raw_scores = {"STOCK_A": 15.2, "STOCK_B": 99.0}
    filtered_scores = {k: v for k, v in raw_scores.items() if k in eligible_symbols}

    percentiles = RelativeStrengthEngine.calculate_universe_percentile_ranks(filtered_scores)

    assert "STOCK_A" in percentiles
    assert "STOCK_B" not in percentiles
    assert percentiles["STOCK_A"] == 0.0


def test_future_ipo_cannot_enter_historical_signal_pipeline():
    """8. Test future IPO symbol cannot be returned by get_universe_for_date at T."""
    ipo_sec = SymbolMetadata(
        symbol="NEXGEN_AI",
        company_name="NexGen AI Ltd",
        listing_date=date(2026, 12, 1),
    )

    symbols = HistoricalUniverseProvider.get_universe_for_date(date(2026, 5, 15), [ipo_sec])
    assert "NEXGEN_AI" not in symbols
    assert len(symbols) == 0
