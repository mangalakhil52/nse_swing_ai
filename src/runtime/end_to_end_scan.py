"""#14R end-to-end runtime wiring.

Keeps provider construction separate from the analytical decision pipeline.
The pipeline is injected because its concrete composition belongs to the
existing architecture, not the runtime shell.
"""
from __future__ import annotations
from datetime import date
from src.architecture.market_scan_orchestrator import MarketScanOrchestrator
from src.data.nse_universe_adapter import NSEUniverseAdapter
from src.data.market_data_adapter import MarketDataAdapter
from src.data.http_market_data_source import HTTPMarketDataSource
from src.data.market_universe import MarketUniverseService


class ConfiguredUniverseService:
    def __init__(self, adapter: NSEUniverseAdapter):
        self.adapter = adapter

    def source(self, as_of_date: date):
        return self.adapter.fetch(as_of_date).symbols

    def normalize(self, raw, as_of_date: date):
        return MarketUniverseService.normalize(raw, as_of_date)


def build_scan_orchestrator(config: dict, candidate_discovery, decision_pipeline):
    if not config.get("universe_url"):
        raise RuntimeError("NSE_UNIVERSE_URL is required")
    if not config.get("market_data_url"):
        raise RuntimeError("MARKET_DATA_BASE_URL is required")
    universe = ConfiguredUniverseService(NSEUniverseAdapter(config["universe_url"], timeout_seconds=config.get("timeout_seconds", 20), retries=config.get("retries", 3)))
    source = HTTPMarketDataSource(config["market_data_url"], timeout_seconds=config.get("timeout_seconds", 20))
    market = MarketDataAdapter(source.fetch, retries=config.get("retries", 3))
    return MarketScanOrchestrator(universe, market, candidate_discovery, decision_pipeline, max_workers=config.get("max_workers", 8))
