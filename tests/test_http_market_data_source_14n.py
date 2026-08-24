"""Tests for configurable HTTP market-data source."""
from unittest.mock import Mock, patch
import pytest
from src.data.http_market_data_source import HTTPMarketDataSource


def test_http_source_fetches_csv():
    response = Mock(status_code=200, text="timestamp,open,high,low,close,volume\n2026-06-30,1,2,0.5,1.5,100\n")
    with patch("src.data.http_market_data_source.httpx.get", return_value=response):
        out = HTTPMarketDataSource("https://provider.test/data").fetch("TRENT")
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_http_source_propagates_http_errors():
    response = Mock()
    response.raise_for_status.side_effect = RuntimeError("503")
    with patch("src.data.http_market_data_source.httpx.get", return_value=response):
        with pytest.raises(RuntimeError, match="503"):
            HTTPMarketDataSource("https://provider.test/data").fetch("TRENT")


def test_http_source_rejects_invalid_csv():
    response = Mock(status_code=200, text="not,a,valid,market,data")
    with patch("src.data.http_market_data_source.httpx.get", return_value=response):
        out = HTTPMarketDataSource("https://provider.test/data").fetch("TRENT")
    assert list(out.columns) == ["not", "a", "valid", "market", "data"]
