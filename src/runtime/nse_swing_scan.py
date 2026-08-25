"""Dynamic whole-NSE swing scanner.

Pipeline:
  1. Fetch the current NSE equity universe dynamically.
  2. Download historical NSE bhavcopies once per trading day.
  3. Run the cheap Candidate Discovery filters across the whole universe.
  4. Run the existing deterministic Technical Analysis specialist on every
     eligible candidate.
  5. Rank candidates by technical score and return the shortlist.

No stock symbols are hardcoded. This module intentionally stops before the
expensive multi-source CIO pipeline; the shortlist is the input to that stage.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date

import pandas as pd

from src.agents.technical_agent import TechnicalAnalysisAgent
from src.candidate_discovery import CandidateDiscoveryConfig, CandidateDiscoveryEngine
from src.core.evidence import EvidenceGraph
from src.core.models import SymbolMetadata
from src.data.market_universe import MarketUniverseService
from src.data.nse_historical_source import NSEHistoricalOHLCVSource
from src.data.nse_official_source import NSEOfficialUniverseSource


@dataclass
class TechnicalShortlistRow:
    symbol: str
    discovery_score: float | None
    technical_score: float
    signal: str
    rsi_14: float | None
    adx_14: float | None
    rvol_20: float | None
    pattern: str | None
    pattern_quality: float | None
    pit_safe: bool
    discovery_reasons: list[str]
    technical_risks: list[str]


@dataclass
class SwingScanSummary:
    as_of_date: str
    universe_count: int
    candidate_count: int
    technical_success_count: int
    technical_failure_count: int
    shortlist_count: int
    historical_diagnostics: dict


async def _analyze_technical(symbol: str, df: pd.DataFrame, as_of: date):
    meta = SymbolMetadata(symbol=symbol, company_name=symbol, exchange="NSE")
    agent = TechnicalAnalysisAgent()
    graph = EvidenceGraph()
    output = await agent.execute(
        meta,
        df,
        graph,
        run_id=f"NSE-SCAN-{as_of.isoformat()}",
        context={"decision_time": as_of},
    )
    return output


def run(
    as_of_date: date,
    lookback_calendar_days: int = 260,
    max_workers: int = 8,
    shortlist_size: int = 50,
    min_price: float | None = None,
    min_adtv_crores: float | None = None,
) -> tuple[SwingScanSummary, list[TechnicalShortlistRow]]:
    """Run a dynamic full-NSE candidate-to-technical shortlist scan."""
    raw_universe = NSEOfficialUniverseSource().fetch()
    universe = MarketUniverseService.normalize(raw_universe, as_of_date)

    source = NSEHistoricalOHLCVSource(
        as_of_date,
        lookback_calendar_days=lookback_calendar_days,
    )
    market_data = source.fetch_many([item.symbol for item in universe])
    diagnostics = asdict(source.diagnostics)

    config = CandidateDiscoveryConfig()
    if min_price is not None:
        config.min_price = min_price
    if min_adtv_crores is not None:
        config.min_average_turnover_crores = min_adtv_crores

    # Cheap Stage 1 screening across the complete dynamically fetched universe.
    discovery_results = CandidateDiscoveryEngine.discover_candidates(
        universe=[item.symbol for item in universe],
        as_of_date=as_of_date,
        market_data_map=market_data,
        config=config,
        mode="LIVE",
    )
    eligible = [r for r in discovery_results if r.eligible and r.pit_safe]

    # Stage 2: existing deterministic technical engine, parallelized only after
    # the cheap funnel has reduced the universe.
    async def run_one(result):
        df = market_data.get(result.symbol, pd.DataFrame())
        return result, await _analyze_technical(result.symbol, df, as_of_date)

    async def run_all():
        sem = asyncio.Semaphore(max(1, max_workers))

        async def bounded(result):
            async with sem:
                return await run_one(result)

        return await asyncio.gather(*(bounded(r) for r in eligible), return_exceptions=True)

    paired = asyncio.run(run_all())
    rows: list[TechnicalShortlistRow] = []
    technical_failures = 0

    for item in paired:
        if isinstance(item, Exception):
            technical_failures += 1
            continue
        discovery, output = item
        if output.status.value != "SUCCESS":
            technical_failures += 1
            continue
        rows.append(
            TechnicalShortlistRow(
                symbol=discovery.symbol,
                discovery_score=discovery.discovery_score,
                technical_score=float(output.score),
                signal=output.signal.value,
                rsi_14=output.metrics.get("rsi_14"),
                adx_14=output.metrics.get("adx_14"),
                rvol_20=output.metrics.get("rvol_20"),
                pattern=output.metrics.get("pattern_detected"),
                pattern_quality=output.metrics.get("pattern_quality"),
                pit_safe=bool(output.metrics.get("pit_safe", False)),
                discovery_reasons=discovery.reasons,
                technical_risks=list(output.risks_identified or []),
            )
        )

    rows.sort(key=lambda r: (-r.technical_score, -(r.discovery_score or 0.0), r.symbol))
    shortlist = rows[: max(1, shortlist_size)]

    summary = SwingScanSummary(
        as_of_date=as_of_date.isoformat(),
        universe_count=len(universe),
        candidate_count=len(eligible),
        technical_success_count=len(rows),
        technical_failure_count=technical_failures,
        shortlist_count=len(shortlist),
        historical_diagnostics=diagnostics,
    )
    return summary, shortlist
