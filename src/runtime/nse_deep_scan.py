"""Deep NSE scan: dynamic universe -> intelligence gate -> CIO pipeline."""
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
from src.quant.indicators import TechnicalIndicators
from src.quant.intelligence_gate import IntelligenceGateConfig, select_normal, select_recent
from src.quant.regime import MarketRegimeClassifier
from src.quant.relative_strength import RelativeStrengthEngine
from src.quant.screener import QuantScreener, ScreenerCandidate
from src.runtime.ipo_radar import RecentIPORadar
from src.runtime.nse_swing_scan import run as run_swing_scan


@dataclass
class DeepScanSummary:
    as_of_date: str
    technical_shortlist_count: int
    intelligence_normal_count: int
    intelligence_recent_count: int
    deep_candidates_count: int
    deep_rejections_count: int
    recommendations_count: int
    regime: str
    trading_stance: str
    historical_diagnostics: dict
    recommendation_symbols: list[str]


def _run_cio(candidates: list[ScreenerCandidate], stock_dfs: dict[str, pd.DataFrame], universe: dict, regime_result, run_id: str):
    return asyncio.run(CIOOrchestrator().run_daily_scan(
        candidates=candidates, stock_dfs=stock_dfs, universe=universe,
        regime_result=regime_result, run_id=run_id,
    ))


def _recent_candidate(symbol: str, df: pd.DataFrame, meta, nifty_df: pd.DataFrame) -> ScreenerCandidate | None:
    """Adapt recent-listing history to the CIO input contract without long-history filters."""
    if df is None or len(df) < 30 or meta is None:
        return None
    enriched = TechnicalIndicators.compute_all_indicators(df)
    close = float(enriched["close"].iloc[-1])
    if close < 20.0:
        return None
    turnover = enriched["turnover_crores"] if "turnover_crores" in enriched.columns else enriched["close"] * enriched["volume"] / 1e7
    adtv = float(turnover.tail(min(20, len(enriched))).mean())
    if adtv < 1.0:
        return None
    rs = 0.0
    if nifty_df is not None and not nifty_df.empty:
        rs_series = RelativeStrengthEngine.calculate_mansfield_rs(enriched["close"], nifty_df["close"], period=20)
        if not rs_series.empty and pd.notna(rs_series.iloc[-1]):
            rs = float(rs_series.iloc[-1])
    trend = 50.0
    if close > float(enriched["ema_200"].iloc[-1]): trend += 15.0
    if float(enriched["rsi_14"].iloc[-1]) >= 55.0: trend += 15.0
    if float(enriched["distance_52w_high_pct"].iloc[-1]) <= 8.0: trend += 10.0
    if float(enriched["rvol_20"].iloc[-1]) >= 1.5: trend += 10.0
    return ScreenerCandidate(
        symbol=symbol.upper(), company_name=meta.company_name or symbol, sector=meta.sector or "General",
        current_price=round(close, 2), adtv_crores=round(adtv, 2),
        rsi_14=round(float(enriched["rsi_14"].iloc[-1]), 2),
        adx_14=round(float(enriched["adx_14"].iloc[-1]), 2),
        atr_pct=round(float(enriched["atr_pct"].iloc[-1]), 2),
        rvol_20=round(float(enriched["rvol_20"].iloc[-1]), 2),
        mansfield_rs=round(rs, 2),
        distance_52w_high_pct=round(float(enriched["distance_52w_high_pct"].iloc[-1]), 2),
        trend_score=round(trend, 1), is_fno=meta.is_fno_eligible, enriched_df=enriched,
    )


def run(as_of_date: date | None = None, lookback_calendar_days: int = 260, max_workers: int = 8, technical_shortlist_size: int = 50) -> tuple[DeepScanSummary, list]:
    """Run the expensive CIO pipeline only on intelligence-gated candidates."""
    as_of_date = as_of_date or get_latest_trading_day(date.today())
    swing_summary, technical_rows = run_swing_scan(
        as_of_date=as_of_date, lookback_calendar_days=lookback_calendar_days,
        max_workers=max_workers, shortlist_size=technical_shortlist_size,
    )

    normal_symbols = [row.symbol for row in technical_rows]
    recent_symbols = [row["symbol"] for row in swing_summary.recent_listing_shortlist]
    fetch_symbols = list(dict.fromkeys(normal_symbols + recent_symbols))
    raw_universe = NSEOfficialUniverseSource().fetch()
    universe_items = MarketUniverseService.normalize(raw_universe, as_of_date)
    universe = {item.symbol: item for item in universe_items}
    source = NSEHistoricalOHLCVSource(as_of_date, lookback_calendar_days=lookback_calendar_days)
    market_data = source.fetch_many(fetch_symbols)
    start = as_of_date - timedelta(days=lookback_calendar_days)

    provider = HistoricalDataProvider()
    async def load_regime():
        try: return await provider.get_daily_ohlcv("NIFTY 50", start, as_of_date, min_bars=50)
        except Exception: return pd.DataFrame()
    try:
        nifty_df = asyncio.run(load_regime())
    finally:
        asyncio.run(provider.close())
    regime_result = MarketRegimeClassifier.classify_regime(nifty_df=nifty_df)

    if not regime_result.allow_long_swing_trades:
        summary = DeepScanSummary(
            as_of_date=as_of_date.isoformat(), technical_shortlist_count=len(technical_rows),
            intelligence_normal_count=0, intelligence_recent_count=0,
            deep_candidates_count=0, deep_rejections_count=len(technical_rows), recommendations_count=0,
            regime=regime_result.regime.value, trading_stance=regime_result.trading_stance.value,
            historical_diagnostics=asdict(source.diagnostics), recommendation_symbols=[],
        )
        return summary, []

    screener = QuantScreener(min_adtv_crores=5.0, min_price=20.0)
    normal_candidates: dict[str, ScreenerCandidate] = {}
    for symbol in normal_symbols:
        meta, df = universe.get(symbol), market_data.get(symbol)
        if meta is None or df is None or df.empty:
            continue
        candidate = screener.screen_single_stock(meta, df, nifty_df)
        if candidate is not None:
            normal_candidates[symbol] = candidate

    recent_radar = RecentIPORadar(as_of_date=as_of_date)
    recent_rows = recent_radar.scan({s: market_data[s] for s in recent_symbols if s in market_data}, limit=25)
    recent_candidates: dict[str, ScreenerCandidate] = {}
    for row in recent_rows:
        candidate = _recent_candidate(row.symbol, market_data.get(row.symbol), universe.get(row.symbol), nifty_df)
        if candidate is not None:
            recent_candidates[row.symbol] = candidate

    gate_cfg = IntelligenceGateConfig()
    normal_gate = select_normal(technical_rows, normal_candidates, gate_cfg)
    recent_gate = select_recent(recent_rows, recent_candidates, gate_cfg)
    combined: list[ScreenerCandidate] = []
    seen: set[str] = set()
    for candidate in normal_gate + recent_gate:
        if candidate.symbol not in seen:
            combined.append(candidate)
            seen.add(candidate.symbol)

    run_id = f"DEEP-NSE-{as_of_date:%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"
    recommendations = _run_cio(combined, market_data, universe, regime_result, run_id) if combined else []
    summary = DeepScanSummary(
        as_of_date=as_of_date.isoformat(), technical_shortlist_count=len(technical_rows),
        intelligence_normal_count=len(normal_gate), intelligence_recent_count=len(recent_gate),
        deep_candidates_count=len(combined),
        deep_rejections_count=max(0, len(technical_rows) + len(recent_rows) - len(combined)),
        recommendations_count=len(recommendations), regime=regime_result.regime.value,
        trading_stance=regime_result.trading_stance.value,
        historical_diagnostics=asdict(source.diagnostics),
        recommendation_symbols=[r.symbol for r in recommendations],
    )
    return summary, recommendations
