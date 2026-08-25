"""Stage-2 NSE technical-intelligence funnel.

Reuses the existing Candidate Discovery contract and TechnicalAnalysisAgent.
No new trading rules or final-conviction logic are introduced here.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd

from src.agents.technical_agent import TechnicalAnalysisAgent
from src.candidate_discovery import CandidateDiscoveryConfig, CandidateDiscoveryEngine
from src.core.models import SymbolMetadata
from src.data.market_universe import MarketUniverseService
from src.data.nse_historical_source import NSEHistoricalOHLCVSource
from src.data.nse_official_source import NSEOfficialUniverseSource


@dataclass
class TechnicalScreenSummary:
    as_of_date: str
    universe_count: int
    normalized_count: int
    stage1_eligible: int
    technical_analyzed: int
    technical_success: int
    technical_data_unavailable: int
    technical_errors: int
    bullish: int
    neutral: int
    bearish: int
    pit_safe: int
    historical_diagnostics: dict[str, Any]


def _technical_one(agent: TechnicalAnalysisAgent, symbol: str, df: pd.DataFrame, as_of: date):
    metadata = SymbolMetadata(symbol=symbol, company_name=symbol, exchange="NSE")
    return asyncio.run(agent.analyze_contract(metadata, df, as_of, run_id=f"TECH-{as_of.isoformat()}-{symbol}"))


def run(as_of_date: date, lookback_calendar_days: int = 140, max_workers: int = 4, limit: int | None = None):
    raw = NSEOfficialUniverseSource().fetch()
    universe = MarketUniverseService.normalize(raw, as_of_date)
    if limit:
        universe = universe[:limit]

    source = NSEHistoricalOHLCVSource(as_of_date, lookback_calendar_days=lookback_calendar_days)
    symbols = [item.symbol for item in universe]
    market_data = source.fetch_many(symbols)

    discovery = CandidateDiscoveryEngine.discover_candidates(
        universe=symbols,
        as_of_date=as_of_date,
        market_data_map=market_data,
        config=CandidateDiscoveryConfig(),
        mode="LIVE",
    )
    stage1 = {result.symbol: result for result in discovery if result.eligible}

    agent = TechnicalAnalysisAgent()
    results = []
    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_technical_one, agent, symbol, market_data[symbol], as_of_date) for symbol in stage1 if symbol in market_data]
        for future in futures:
            try:
                results.append(future.result())
            except Exception:
                errors += 1

    technical_success = sum(r.status.value == "SUCCESS" for r in results)
    unavailable = sum(r.status.value == "DATA_UNAVAILABLE" for r in results)
    bullish = sum(r.signal.value == "BULLISH" for r in results)
    neutral = sum(r.signal.value == "NEUTRAL" for r in results)
    bearish = sum(r.signal.value == "BEARISH" for r in results)
    pit_safe = sum(bool(r.pit_safe) for r in results)

    summary = TechnicalScreenSummary(
        as_of_date=as_of_date.isoformat(),
        universe_count=len(raw),
        normalized_count=len(universe),
        stage1_eligible=len(stage1),
        technical_analyzed=len(results),
        technical_success=technical_success,
        technical_data_unavailable=unavailable,
        technical_errors=errors,
        bullish=bullish,
        neutral=neutral,
        bearish=bearish,
        pit_safe=pit_safe,
        historical_diagnostics=asdict(source.diagnostics),
    )
    rows = []
    for result in sorted(results, key=lambda item: (-item.score, item.symbol)):
        rows.append({
            "symbol": result.symbol,
            "score": result.score,
            "signal": result.signal.value,
            "confidence": result.confidence,
            "status": result.status.value,
            "pit_safe": result.pit_safe,
            "reasons": result.reasons,
            "risks": result.risks,
            "evidence": [item.model_dump(mode="json") for item in result.evidence],
        })
    return summary, rows
