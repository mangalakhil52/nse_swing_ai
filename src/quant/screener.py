"""
Stage-1 Quantitative Pre-Screener Module.
Deterministically filters the entire ~2,200 NSE universe down to 40–100 high-potential swing candidates.
Evaluates liquidity (ADTV >= ₹5 Cr), structural trend (EMAs), ATR volatility bounds, 52W high proximity, and relative volume.
"""

import logging
from typing import Any
import pandas as pd
from pydantic import BaseModel, Field

from config.settings import settings
from src.core.models import SymbolMetadata
from src.quant.indicators import TechnicalIndicators
from src.quant.relative_strength import RelativeStrengthEngine

logger = logging.getLogger(__name__)


class ScreenerCandidate(BaseModel):
    symbol: str
    company_name: str
    sector: str
    current_price: float
    adtv_crores: float
    rsi_14: float
    adx_14: float
    atr_pct: float
    rvol_20: float
    mansfield_rs: float
    distance_52w_high_pct: float
    trend_score: float
    is_fno: bool
    enriched_df: Any = Field(default=None, exclude=True)


class QuantScreener:
    """High-speed Stage-1 quantitative filtration engine."""

    def __init__(
        self,
        min_adtv_crores: float | None = None,
        min_price: float | None = None,
        max_dist_52w_high_pct: float = 20.0,
        min_atr_pct: float = 1.5,
        max_atr_pct: float = 7.0,
        min_rvol: float = 1.0,
    ):
        self.min_adtv = min_adtv_crores or settings.MIN_ADTV_CRORES
        self.min_price = min_price or settings.MIN_STOCK_PRICE
        self.max_dist_52w_high = max_dist_52w_high_pct
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct
        self.min_rvol = min_rvol

    def screen_single_stock(
        self,
        symbol_meta: SymbolMetadata,
        ohlcv_df: pd.DataFrame,
        nifty_df: pd.DataFrame | None = None,
    ) -> ScreenerCandidate | None:
        """
        Evaluates a single stock against Stage-1 quantitative rules.
        Returns a ScreenerCandidate if it passes, else None.
        """
        if ohlcv_df is None or len(ohlcv_df) < 50:
            return None

        # 1. Price check
        latest_close = float(ohlcv_df["close"].iloc[-1])
        if latest_close < self.min_price:
            return None

        # 2. Liquidity check (20-day ADTV in Crores)
        if "turnover_crores" in ohlcv_df.columns:
            adtv = float(ohlcv_df["turnover_crores"].tail(20).mean())
        else:
            # Approximate Turnover = Close * Volume / 1e7
            turnover_approx = (ohlcv_df["close"] * ohlcv_df["volume"]) / 1e7
            adtv = float(turnover_approx.tail(20).mean())

        if adtv < self.min_adtv:
            return None

        # Compute indicators
        df = TechnicalIndicators.compute_all_indicators(ohlcv_df)

        close = df["close"].iloc[-1]
        ema_20 = df["ema_20"].iloc[-1]
        ema_50 = df["ema_50"].iloc[-1]
        ema_200 = df["ema_200"].iloc[-1]
        rsi = df["rsi_14"].iloc[-1]
        adx = df["adx_14"].iloc[-1]
        atr_pct = df["atr_pct"].iloc[-1]
        rvol = df["rvol_20"].iloc[-1]
        dist_52w = df["distance_52w_high_pct"].iloc[-1]

        # 3. Trend criteria: Price > 20 EMA and 20 EMA > 50 EMA
        if close < ema_20 or ema_20 < ema_50:
            return None

        # 4. Volatility bounds (Filters stagnant dead stocks < 1.5% and hyper-erratic lottery stocks > 7%)
        if not (self.min_atr_pct <= atr_pct <= self.max_atr_pct):
            return None

        # 5. Distance from 52-week High (Must be within 20%)
        if dist_52w > self.max_dist_52w_high:
            return None

        # 6. Relative Volume minimum check
        if rvol < self.min_rvol:
            return None

        # 7. Relative Strength vs Nifty 50
        mansfield_rs = 0.0
        if nifty_df is not None and not nifty_df.empty:
            rs_series = RelativeStrengthEngine.calculate_mansfield_rs(df["close"], nifty_df["close"], period=50)
            mansfield_rs = float(rs_series.iloc[-1])
            if mansfield_rs < 0.0:
                # Underperforming benchmark, filter out
                return None

        # Calculate composite trend score
        trend_score = 50.0
        if close > ema_200:
            trend_score += 15.0
        if rsi >= 55.0:
            trend_score += 15.0
        if dist_52w <= 8.0:
            trend_score += 10.0
        if rvol >= 1.5:
            trend_score += 10.0

        return ScreenerCandidate(
            symbol=symbol_meta.symbol,
            company_name=symbol_meta.company_name,
            sector=symbol_meta.sector or "General",
            current_price=round(close, 2),
            adtv_crores=round(adtv, 2),
            rsi_14=round(rsi, 2),
            adx_14=round(adx, 2),
            atr_pct=round(atr_pct, 2),
            rvol_20=round(rvol, 2),
            mansfield_rs=round(mansfield_rs, 2),
            distance_52w_high_pct=round(dist_52w, 2),
            trend_score=round(trend_score, 1),
            is_fno=symbol_meta.is_fno_eligible,
            enriched_df=df,
        )

    def screen_universe(
        self,
        universe: list[SymbolMetadata],
        stock_dfs: dict[str, pd.DataFrame],
        nifty_df: pd.DataFrame | None = None,
    ) -> list[ScreenerCandidate]:
        """
        Screens the full universe and returns top candidates sorted by trend and momentum score.
        """
        candidates: list[ScreenerCandidate] = []

        for sec in universe:
            sym = sec.symbol
            df = stock_dfs.get(sym)
            if df is not None and not df.empty:
                cand = self.screen_single_stock(sec, df, nifty_df)
                if cand:
                    candidates.append(cand)

        # Sort by trend score descending
        candidates.sort(key=lambda c: (c.trend_score, c.mansfield_rs), reverse=True)
        logger.info(f"Quant screener filtered {len(universe)} symbols down to {len(candidates)} candidates.")
        return candidates
