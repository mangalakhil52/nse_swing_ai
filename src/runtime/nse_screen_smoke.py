"""#14W controlled NSE market-data screening smoke runner.

Historical files are fetched once per trading day and then fanned out to all
symbols. This keeps a 2,500-symbol scan at roughly O(trading-days) downloads,
not O(symbols * trading-days).
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import date
import pandas as pd
from src.data.nse_official_source import NSEOfficialUniverseSource
from src.data.nse_historical_source import NSEHistoricalOHLCVSource
from src.data.market_universe import MarketUniverseService
from src.candidate_discovery import CandidateDiscoveryEngine, CandidateDiscoveryConfig


@dataclass
class ScreenSummary:
    as_of_date: str
    universe_count: int
    normalized_count: int
    data_available: int
    data_unavailable: int
    eligible: int
    ineligible: int
    errors: int
    historical_diagnostics: dict


def run(as_of_date: date, lookback_calendar_days: int = 140, max_workers: int = 4, limit: int | None = None) -> tuple[ScreenSummary, list]:
    raw = NSEOfficialUniverseSource().fetch()
    universe = MarketUniverseService.normalize(raw, as_of_date)
    if limit:
        universe = universe[:limit]

    source = NSEHistoricalOHLCVSource(as_of_date, lookback_calendar_days=lookback_calendar_days)
    symbol_list = [item.symbol for item in universe]
    market_data = source.fetch_many(symbol_list)
    diagnostic = asdict(source.diagnostics)
    config = CandidateDiscoveryConfig()

    def screen(item):
        df = market_data.get(item.symbol, pd.DataFrame())
        return CandidateDiscoveryEngine.discover_candidates(
            universe=[item], as_of_date=as_of_date, market_data_map={item.symbol: df}, config=config
        )[0]

    results = []
    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(screen, item) for item in universe]
        for future in futures:
            try:
                results.append(future.result())
            except Exception:
                errors += 1

    data_available = sum(r.symbol in market_data for r in results)
    eligible = sum(bool(r.eligible) for r in results)
    summary = ScreenSummary(
        as_of_date=as_of_date.isoformat(),
        universe_count=len(raw),
        normalized_count=len(universe),
        data_available=data_available,
        data_unavailable=len(universe) - data_available,
        eligible=eligible,
        ineligible=len(results) - eligible,
        errors=errors,
        historical_diagnostics=diagnostic,
    )
    return summary, sorted(results, key=lambda r: (-(r.discovery_score if r.discovery_score is not None else float("-inf")), r.symbol))
