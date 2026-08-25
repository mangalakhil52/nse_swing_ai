"""#14W controlled screening smoke tests."""
from datetime import date
from unittest.mock import patch
import pandas as pd
from src.runtime.nse_screen_smoke import run


def _df(symbol):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-04-01", periods=60, freq="B"),
        "open": [100.0] * 60, "high": [101.0] * 60, "low": [99.0] * 60,
        "close": [100.0] * 60, "volume": [100000.0] * 60,
    })


def test_screen_smoke_stops_at_candidate_discovery():
    raw = [{"symbol": "AAA", "exchange": "NSE"}, {"symbol": "BBB", "exchange": "NSE"}]
    with patch("src.runtime.nse_screen_smoke.NSEOfficialUniverseSource.fetch", return_value=raw), \
         patch("src.runtime.nse_screen_smoke.NSEHistoricalOHLCVSource.fetch", side_effect=lambda s: _df(s)):
        summary, results = run(date(2026, 6, 30), max_workers=2)
    assert summary.universe_count == 2
    assert summary.errors == 0
    assert len(results) == 2


def test_screen_smoke_supports_controlled_limit():
    raw = [{"symbol": "AAA", "exchange": "NSE"}, {"symbol": "BBB", "exchange": "NSE"}]
    with patch("src.runtime.nse_screen_smoke.NSEOfficialUniverseSource.fetch", return_value=raw), \
         patch("src.runtime.nse_screen_smoke.NSEHistoricalOHLCVSource.fetch", side_effect=lambda s: _df(s)):
        summary, _ = run(date(2026, 6, 30), limit=1)
    assert summary.universe_count == 2
    assert summary.normalized_count == 1
