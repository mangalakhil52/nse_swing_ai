"""Deep NSE scan: dynamic universe -> technical shortlist -> existing CIO pipeline.

This layer deliberately does not invent a second scoring system. It reuses the
existing technical shortlist and the production CIO orchestrator, while limiting
expensive multi-agent research to a small, high-quality set.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import uuid

import pandas as pd

from config.market_hours import get_latest_trading_day
from src.agents.cio_orchestrator import CIOOrchestrator
from src.data.historical_provider import HistoricalDataProvider
from src.data.market_universe import MarketUniverseService
from src.data.nse_historical_source import NSEHistoricalOHLCVSource
from src.data.nse_official_source import NSEOfficialUniverseSource
from src.quant.regime import MarketRegimeClassifier
from src.quant.screener import QuantScreener, ScreenerCandidate
from src.runtime.nse_swing_scan import run as run_swing_scan


@dataclass
class DeepScanSummary:
    as_of_date: str
    technical_shortlist_count: int
    deep_candidates_count: int
    deep_rejections_count: int
    recommendations_count: int
    regime: str
    trading_stance: str
    historical_diagnostics: dict
    recommendation_symbols: list[str]


def _run_cio(
    candidates: list[ScreenerCandidate],
    stock_dfs: dict[str, pd.DataFrame],
    universe: dict,
    regime_result,
    run_id: str,
):
    return asyncio.run(
        CIOOrchestrator().run_daily_scan(
            candidates=candidates,
            stock_dfs=stock_dfs,
            universe=universe,
            regime_result=regime_result,
            run_id=run_id,
        )
    )


def run(
    as_of_date: date | None = None,
    lookback_calendar_days: int = 260,
    max_workers: int = 8,
    technical_shortlist_size: int = 50,
) -> tuple[DeepScanSummary, list]:
    """Run the expensive CIO pipeline only on the validated technical shortlist."""
    as_of_date = as_of_date or get_latest_trading_day(date.today())

    swing_summary, technical_rows = run_swing_scan(
        as_of_date=as_of_date,
        lookback_calendar_days=lookback_calendar_days,
        max_workers=max_workers,
        shortlist_size=technical_shortlist_size,
    )

    symbols = [row.symbol for row in technical_rows]
    raw_universe = NSEOfficialUniverseSource().fetch()
    universe_items = MarketUniverseService.normalize(raw_universe, as_of_date)
    universe = {item.symbol: item for item in universe_items}

    source = NSEHistoricalOHLCVSource(as_of_date, lookback_calendar_days=lookback_calendar_days)
    market_data = source.fetch_many(symbols)

    hist_provider = HistoricalDataProvider()
    start = as_of_date - timedelta(days=lookback_calendar_days)

    async def load_regime():
        try:
            nifty = await hist_provider.get_daily_ohlcv("NIFTY 50", start, as_of_date, min_bars=50)
        except Exception:
            nifty = pd.DataFrame()
        return MarketRegimeClassifier.classify_regime(nifty_df=nifty)

    try:
        regime_result = asyncio.run(load_regime())
    finally:
        asyncio.run(hist_provider.close())

    if not regime_result.allow_long_swing_trades:
        summary = DeepScanSummary(
            as_of_date=as_of_date.isoformat(),
            technical_shortlist_count=len(technical_rows),
            deep_candidates_count=0,
            deep_rejections_count=len(technical_rows),
            recommendations_count=0,
            regime=regime_result.regime.value,
            trading_stance=regime_result.trading_stance.value,
            historical_diagnostics=asdict(source.diagnostics),
            recommendation_symbols=[],
        )
        return summary, []

    screener = QuantScreener(min_adtv_crores=5.0, min_price=20.0)

    async def load_nifty():
        provider = HistoricalDataProvider()
        try:
            return await provider.get_daily_ohlcv("NIFTY 50", start, as_of_date, min_bars=50)
        except Exception:
            return pd.DataFrame()
        finally:
            await provider.close()

    nifty_df = asyncio.run(load_nifty())

    deep_candidates: list[ScreenerCandidate] = []
    for symbol in symbols:
        meta = universe.get(symbol)
        df = market_data.get(symbol)
        if meta is None or df is None or df.empty:
            continue
        candidate = screener.screen_single_stock(meta, df, nifty_df)
        if candidate is not None:
            deep_candidates.append(candidate)

    run_id = f"DEEP-NSE-{as_of_date:%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"
    recommendations = _run_cio(
        deep_candidates,
        market_data,
        universe,
        regime_result,
        run_id,
    ) if deep_candidates else []

    summary = DeepScanSummary(
        as_of_date=as_of_date.isoformat(),
        technical_shortlist_count=len(technical_rows),
        deep_candidates_count=len(deep_candidates),
        deep_rejections_count=max(0, len(technical_rows) - len(deep_candidates)),
        recommendations_count=len(recommendations),
        regime=regime_result.regime.value,
        trading_stance=regime_result.trading_stance.value,
        historical_diagnostics=asdict(source.diagnostics),
        recommendation_symbols=[r.symbol for r in recommendations],
    )
    return summary, recommendations
