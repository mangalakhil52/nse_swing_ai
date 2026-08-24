"""#14L market universe normalization tests."""
from datetime import date
from src.data.market_universe import MarketUniverseService, UniverseSymbol


def test_normalize_deduplicates_and_normalizes_symbols():
    out = MarketUniverseService.normalize(["trent", "TRENT", "infy", "", None])
    assert [x.symbol for x in out] == ["INFY", "TRENT"]


def test_historical_listing_and_delisting_are_point_in_time_safe():
    raw = [
        UniverseSymbol("ACTIVE", listing_date=date(2020, 1, 1)),
        UniverseSymbol("FUTURE", listing_date=date(2027, 1, 1)),
        UniverseSymbol("DELISTED", listing_date=date(2020, 1, 1), delisting_date=date(2026, 6, 30)),
    ]
    out = MarketUniverseService.normalize(raw, date(2026, 6, 30))
    assert [x.symbol for x in out] == ["ACTIVE"]


def test_non_nse_symbols_are_excluded():
    out = MarketUniverseService.normalize([{"symbol": "ABC", "exchange": "BSE"}, {"symbol": "XYZ", "exchange": "NSE"}])
    assert [x.symbol for x in out] == ["XYZ"]
