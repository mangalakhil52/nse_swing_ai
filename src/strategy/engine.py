"""
Unified Strategy Engine Module — src/strategy/engine.py (Part 22 Live/Backtest Parity)

Provides ONE shared strategy implementation used identically by the Live Scanner and Backtester.
Guarantees 100% parity between live signal generation and historical backtests.
"""

from datetime import date
import logging
from typing import Any
import pandas as pd

from src.agents.cio_orchestrator import CIOOrchestrator
from src.core.models import SymbolMetadata, TradeRecommendation
from src.core.types import MarketRegime, TradingStance
from src.data.historical_provider import HistoricalDataProvider
from src.data.point_in_time import PointInTimeFilter
from src.quant.regime import MarketRegimeClassifier, RegimeAnalysisResult
from src.quant.screener import Stage1Screener

logger = logging.getLogger(__name__)


class StrategyEngine:
    """Unified strategy engine providing parity between live scans and backtests."""

    def __init__(self, hist_provider: HistoricalDataProvider | None = None):
        self.hist_provider = hist_provider or HistoricalDataProvider()
        self.screener = Stage1Screener()
        self.cio = CIOOrchestrator()

    async def run_strategy_scan(
        self,
        as_of_date: date,
        eligible_symbols: list[str],
        run_id: str,
        nifty_df: pd.DataFrame | None = None,
        india_vix: float | None = None,
    ) -> list[TradeRecommendation]:
        """
        Executes complete strategy pipeline as of a specific date (live or backtest).
        """
        start_date = as_of_date - pd.Timedelta(days=120)

        # 1. Fetch & Validate Market Data for symbols
        stock_dfs: dict[str, pd.DataFrame] = {}
        universe_meta: dict[str, SymbolMetadata] = {}

        for sym in eligible_symbols:
            try:
                df_raw = await self.hist_provider.get_daily_ohlcv(sym, start_date, as_of_date, min_bars=50)
                # Apply Point-In-Time safety filter
                df_pit = PointInTimeFilter.filter_market_data(df_raw, as_of_date)
                if len(df_pit) >= 50:
                    stock_dfs[sym] = df_pit
                    universe_meta[sym] = SymbolMetadata(symbol=sym, company_name=sym, sector="General")
            except Exception as e:
                logger.debug(f"[{sym}] Excluded from scan on {as_of_date}: {e}")

        if not stock_dfs:
            logger.warning(f"No valid stock DataFrames found on {as_of_date}.")
            return []

        # 2. Market Regime Classification (Real Nifty OHLC & VIX - Part 9)
        if nifty_df is None or nifty_df.empty:
            try:
                nifty_df = await self.hist_provider.get_daily_ohlcv("NIFTY 50", start_date, as_of_date, min_bars=50)
            except Exception:
                # If Nifty data unavailable, set UNKNOWN regime (no new long trades)
                regime_res = RegimeAnalysisResult(
                    regime=MarketRegime.UNKNOWN,
                    trading_stance=TradingStance.NO_TRADE,
                    risk_multiplier=0.0,
                    allow_long_swing_trades=False,
                    reason="NIFTY market regime data unavailable.",
                )
                logger.warning(f"Market regime UNKNOWN on {as_of_date}. Scan aborted.")
                return []

        regime_res = MarketRegimeClassifier.classify_regime(
            nifty_df=nifty_df,
            india_vix=india_vix or 14.5,
            advance_decline_ratio=1.5,
        )

        if not regime_res.allow_long_swing_trades:
            logger.info(f"Market stance {regime_res.trading_stance.value} on {as_of_date}. Long trades disallowed.")
            return []

        # 3. Stage-1 Screener Filtering
        candidates = self.screener.screen_universe(stock_dfs, universe_meta)
        if not candidates:
            logger.info(f"No Stage-1 screener candidates passed on {as_of_date}.")
            return []

        # 4. CIO Specialist Research Pipeline
        recommendations = await self.cio.run_daily_scan(
            candidates=candidates,
            stock_dfs=stock_dfs,
            universe=universe_meta,
            regime_result=regime_res,
            run_id=run_id,
        )

        return recommendations
