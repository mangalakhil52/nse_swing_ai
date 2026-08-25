"""#14W controlled NSE market-data screening smoke runner.

Runs universe + historical OHLCV + Candidate Discovery only. It deliberately
stops before CIO/AI analysis so data coverage can be validated independently.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
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


def run(as_of_date: date, lookback_calendar_days: int = 140, max_workers: int = 4, limit: int | None = None) -> tuple[ScreenSummary, list]:
    raw = NSEOfficialUniverseSource().fetch()
    universe = MarketUniverseService.normalize(raw, as_of_date)
    if limit:
        universe = universe[:limit]
    source = NSEHistoricalOHLCVSource(as_of_date, lookback_calendar_days=lookback_calendar_days)
    config = CandidateDiscoveryConfig()
    results, errors = [], 0

    def screen(item):
        try:
            df = source.fetch(item.symbol)
            return CandidateDiscoveryEngine.discover_candidates(
                universe=[item], as_of_date=as_of_date, market_data_map={item.symbol: df}, config=config
            )[0]
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(screen, item): item for item in universe}
        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, Exception):
                errors += 1
            else:
                results.append(result)

    data_available = sum(bool(getattr(r, "pit_safe", False)) for r in results)
    eligible = sum(bool(r.eligible) for r in results)
    summary = ScreenSummary(
        as_of_date=as_of_date.isoformat(), universe_count=len(raw), normalized_count=len(universe),
        data_available=data_available, data_unavailable=len(universe) - data_available,
        eligible=eligible, ineligible=len(results) - eligible, errors=errors,
    )
    return summary, sorted(results, key=lambda r: (-r.discovery_score, r.symbol))
