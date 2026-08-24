"""#14M NSE universe adapter tests."""
from datetime import date
from unittest.mock import MagicMock, patch
import pytest
from src.data.nse_universe_adapter import NSEUniverseAdapter

CSV = "symbol,exchange\nTRENT,NSE\nINFY,NSE\nBSEONLY,BSE\nTRENT,NSE\n"


def test_fetch_validates_and_normalizes_snapshot():
    with patch("src.data.nse_universe_adapter.urlopen") as open_mock:
        response = open_mock.return_value.__enter__.return_value
        response.read.return_value = CSV.encode()
        snap = NSEUniverseAdapter("https://example.test/universe.csv").fetch(date(2026, 6, 30))
    assert [x.symbol for x in snap.symbols] == ["INFY", "TRENT"]
    assert snap.source_url.endswith("universe.csv")


def test_missing_symbol_column_fails_closed():
    with patch("src.data.nse_universe_adapter.urlopen") as open_mock:
        response = open_mock.return_value.__enter__.return_value
        response.read.return_value = b"name,exchange\nTrent,NSE\n"
        with pytest.raises(ValueError, match="required columns"):
            NSEUniverseAdapter("https://example.test/universe.csv").fetch()


def test_empty_source_fails_closed():
    with patch("src.data.nse_universe_adapter.urlopen") as open_mock:
        response = open_mock.return_value.__enter__.return_value
        response.read.return_value = b"symbol,exchange\n"
        with pytest.raises(ValueError, match="no rows"):
            NSEUniverseAdapter("https://example.test/universe.csv").fetch()


def test_transient_download_failure_retries():
    successful_response = MagicMock()
    successful_response.__enter__.return_value.read.return_value = CSV.encode()
    with patch("src.data.nse_universe_adapter.urlopen", side_effect=[OSError("temporary"), successful_response]) as open_mock:
        with patch("src.data.nse_universe_adapter.time.sleep") as sleep:
            snap = NSEUniverseAdapter("https://example.test/universe.csv", retries=2).fetch()
    assert len(snap.symbols) == 2
    assert open_mock.call_count == 2
    sleep.assert_called_once_with(1)
