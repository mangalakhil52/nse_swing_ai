"""#14W controlled NSE market-data screening smoke runner.

Historical files are fetched once per trading day and then fanned out to all
symbols. This keeps a 2,500-symbol scan at roughly O(trading-days) downloads,
not O(symbols * trading-days).
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import date
import pandas as pd
from src.data.nse_official_source import NSEOfficialUniverseSource
from src.data.nse_historical_source import NSEHistoricalOHLCVSource
from src.data.market_universe import MarketUniverseService
from src.candidate_discovery import CandidateDiscoveryEngine, CandidateDiscoveryConfig
from src.runtime.telemetry import scan_progress, agent, alert


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
    scan_progress(status="LOADING_UNIVERSE", processed=0, universe=0, filtered=0, candidates=0, intel=0, final=0)
    raw = NSEOfficialUniverseSource().fetch()
    universe = MarketUniverseService.normalize(raw, as_of_date)
    if limit:
        universe = universe[:limit]
    scan_progress(status="LOADING_HISTORY", universe=len(universe), processed=0)

    source = NSEHistoricalOHLCVSource(as_of_date, lookback_calendar_days=lookback_calendar_days)
    symbol_list = [item.symbol for item in universe]
    market_data = source.fetch_many(symbol_list)
    diagnostic = asdict(source.diagnostics)
    available_symbols = set(market_data)
    data_available = sum(item.symbol in available_symbols for item in universe)
    scan_progress(status="TECHNICAL_SCREEN", universe=len(universe), processed=data_available, filtered=0)
    agent("TECHNICAL", status="SCANNING", progress=0, processed=0, decision="Evaluating technical gates", log=["Historical data loaded", "Running trend / momentum / volume gates"])
    config = CandidateDiscoveryConfig()

    def screen(item):
        symbol = item.symbol
        df = market_data.get(symbol, pd.DataFrame())
        return CandidateDiscoveryEngine.discover_candidates(
            universe=[symbol], as_of_date=as_of_date, market_data_map={symbol: df}, config=config, mode="LIVE"
        )[0]

    results = []
    errors = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(screen, item) for item in universe]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors += 1
                alert(f"Candidate discovery error: {type(exc).__name__}", "red")
            completed += 1
            if completed % max(1, len(universe) // 20) == 0 or completed == len(universe):
                scan_progress(status="TECHNICAL_SCREEN", universe=len(universe), processed=completed)
                agent("TECHNICAL", status="SCANNING", progress=int(completed * 100 / max(1, len(universe))), processed=completed, decision="Candidate discovery in progress")

    eligible = sum(bool(r.eligible) for r in results)
    scan_progress(status="CANDIDATE_DISCOVERY_COMPLETE", universe=len(universe), processed=len(universe), filtered=len(results), candidates=eligible)
    agent("TECHNICAL", status="COMPLETE", progress=100, processed=len(universe), decision=f"{eligible} eligible candidates")
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
    return summary, sorted(results, key=lambda r: (-(r.discovery_score if r.discovery_score is not None else float("-inf")), str(r.symbol)))
