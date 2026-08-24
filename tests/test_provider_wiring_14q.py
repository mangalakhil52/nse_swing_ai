"""#14Q configured provider wiring tests."""
import pytest
from src.runtime.provider_wiring import build_provider_layer


def test_provider_layer_requires_both_sources():
    with pytest.raises(ValueError, match="NSE_UNIVERSE_URL"):
        build_provider_layer("", "https://market.test")
    with pytest.raises(ValueError, match="MARKET_DATA_BASE_URL"):
        build_provider_layer("https://universe.test", "")


def test_provider_layer_builds_configured_components():
    universe, market_data = build_provider_layer("https://universe.test", "https://market.test")
    assert universe.adapter.source_url == "https://universe.test"
    assert market_data.source == "https://market.test"
