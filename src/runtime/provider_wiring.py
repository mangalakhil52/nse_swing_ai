"""#14Q production provider wiring.

Builds configured universe and market-data providers without embedding vendor
URLs. Analysis callbacks remain injected because their contracts belong to the
existing application layer.
"""
from __future__ import annotations
from datetime import date
from src.data.nse_universe_adapter import NSEUniverseAdapter
from src.data.market_data_adapter import MarketDataAdapter
from src.data.http_market_data_source import HTTPMarketDataSource


class ConfiguredUniverseService:
    def __init__(self, adapter: NSEUniverseAdapter):
        self.adapter = adapter

    def source(self, as_of_date: date):
        return self.adapter.fetch(as_of_date).symbols

    def normalize(self, raw_symbols, as_of_date: date):
        from src.data.market_universe import MarketUniverseService
        return MarketUniverseService.normalize(raw_symbols, as_of_date)


def build_provider_layer(universe_url: str, market_data_url: str, timeout_seconds: float = 20.0,
                        retries: int = 3):
    if not universe_url:
        raise ValueError("NSE_UNIVERSE_URL is required")
    if not market_data_url:
        raise ValueError("MARKET_DATA_BASE_URL is required")
    universe = ConfiguredUniverseService(NSEUniverseAdapter(universe_url, timeout_seconds=int(timeout_seconds), retries=retries))
    source = HTTPMarketDataSource(market_data_url, timeout_seconds=timeout_seconds)
    market_data = MarketDataAdapter(source.fetch, source=market_data_url, retries=retries)
    return universe, market_data
