"""#14U controlled live NSE runtime using official public files."""
from __future__ import annotations
from datetime import date
from src.data.nse_official_source import NSEOfficialUniverseSource, NSEOfficialBhavcopySource
from src.data.market_universe import MarketUniverseService
from src.data.market_data_adapter import MarketDataAdapter
from src.architecture.market_scan_orchestrator import MarketScanOrchestrator


class OfficialNSEUniverseService:
    def __init__(self, source):
        self.source_adapter = source

    def source(self, as_of_date):
        return self.source_adapter.fetch()

    def normalize(self, raw, as_of_date):
        return MarketUniverseService.normalize(raw, as_of_date)


def build_official_nse_orchestrator(as_of_date: date, candidate_discovery, decision_pipeline, timeout_seconds=20, max_workers=4):
    universe_source = NSEOfficialUniverseSource(timeout_seconds=timeout_seconds)
    universe = OfficialNSEUniverseService(universe_source)
    bhavcopy = NSEOfficialBhavcopySource(as_of_date, timeout_seconds=timeout_seconds)
    market = MarketDataAdapter(bhavcopy.fetch, retries=2, backoff_seconds=1)
    return MarketScanOrchestrator(universe, market, candidate_discovery, decision_pipeline, max_workers=max_workers)
