"""#14O whole-market scan orchestration.

The orchestrator deliberately keeps acquisition, screening, intelligence, and
final-decision stages separate. One symbol failing must not abort the scan.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ScanItem:
    symbol: str
    stage: str
    status: str
    result: object | None = None
    error: str | None = None


@dataclass(frozen=True)
class MarketScanResult:
    as_of_date: date
    items: tuple[ScanItem, ...]

    @property
    def decisions(self):
        return tuple(x for x in self.items if x.stage == "DECISION" and x.status == "SUCCESS")


class MarketScanOrchestrator:
    def __init__(self, universe_service, market_data_adapter, candidate_discovery, decision_pipeline,
                 max_workers: int = 8):
        self.universe_service = universe_service
        self.market_data_adapter = market_data_adapter
        self.candidate_discovery = candidate_discovery
        self.decision_pipeline = decision_pipeline
        self.max_workers = max(1, max_workers)

    def scan(self, as_of_date: date) -> MarketScanResult:
        symbols = self.universe_service.normalize(self.universe_service.source(as_of_date), as_of_date)
        items: list[ScanItem] = []

        def process(symbol):
            try:
                market = self.market_data_adapter.fetch(symbol.symbol, as_of_date)
                discovery = self.candidate_discovery(symbol.symbol, market.frame, as_of_date)
                if not getattr(discovery, "eligible", False):
                    return ScanItem(symbol.symbol, "DISCOVERY", "FILTERED", discovery)
                decision = self.decision_pipeline(symbol, market.frame, as_of_date)
                return ScanItem(symbol.symbol, "DECISION", "SUCCESS", decision)
            except Exception as exc:
                return ScanItem(symbol.symbol, "ERROR", "FAILED", error=f"{type(exc).__name__}: {exc}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(process, symbol): symbol.symbol for symbol in symbols}
            for future in as_completed(futures):
                items.append(future.result())
        items.sort(key=lambda x: x.symbol)
        return MarketScanResult(as_of_date=as_of_date, items=tuple(items))
