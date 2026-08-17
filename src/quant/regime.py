"""
Market Regime and Breadth Classifier Module.
Evaluates macro trend structure of NIFTY 50, market breadth (% above 50 SMA, A/D ratio), and India VIX.
Determines system-wide risk posture (AGGRESSIVE, NORMAL, SELECTIVE, DEFENSIVE, NO_TRADE).
"""

from typing import Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.types import MarketRegime, TradingStance
from src.quant.indicators import TechnicalIndicators


class RegimeAnalysisResult(BaseModel):
    regime: MarketRegime
    trading_stance: TradingStance
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    nifty_close: float
    trend_description: str
    advance_decline_ratio: float
    pct_above_50_sma: float
    india_vix: float
    allow_long_swing_trades: bool
    risk_multiplier: float = Field(default=1.0, description="Multiplier for position sizing risk (e.g. 1.0, 0.75, 0.5, 0.0)")
    summary: str


class MarketRegimeClassifier:
    """Classifies Indian market regime using multi-factor breadth and price action."""

    @classmethod
    def classify_regime(
        cls,
        nifty_df: pd.DataFrame,
        advance_decline_ratio: float = 1.2,
        pct_above_50_sma: float = 62.0,
        india_vix: float = 14.5,
    ) -> RegimeAnalysisResult:
        """
        Evaluates Nifty 50 trend, market participation, and volatility.
        """
        if nifty_df is None or nifty_df.empty or len(nifty_df) < 50:
            # P0 Fail-Closed: Return UNKNOWN regime and disallow long trades
            return RegimeAnalysisResult(
                regime=MarketRegime.UNKNOWN,
                trading_stance=TradingStance.NO_TRADE,
                confidence=0.0,
                nifty_close=0.0,
                trend_description="Insufficient or missing Nifty index data.",
                advance_decline_ratio=0.0,
                pct_above_50_sma=0.0,
                india_vix=0.0,
                allow_long_swing_trades=False,
                risk_multiplier=0.0,
                summary="Market regime set to UNKNOWN due to missing Nifty index data. Long trades blocked.",
            )

        df = TechnicalIndicators.compute_all_indicators(nifty_df)
        close = df["close"].iloc[-1]
        ema_20 = df["ema_20"].iloc[-1]
        ema_50 = df["ema_50"].iloc[-1]
        ema_200 = df["ema_200"].iloc[-1]
        adx = df["adx_14"].iloc[-1]

        # Trend evaluations
        is_above_20_ema = close > ema_20
        is_above_50_ema = close > ema_50
        is_above_200_ema = close > ema_200
        is_ema_aligned = ema_20 > ema_50 > ema_200

        # Score-based classification (0 to 100)
        regime_score = 50.0  # Base Neutral

        # Price vs EMAs (+- 30 pts)
        if is_above_20_ema:
            regime_score += 10.0
        if is_above_50_ema:
            regime_score += 10.0
        if is_above_200_ema:
            regime_score += 10.0
        if is_ema_aligned:
            regime_score += 10.0
        if close < ema_50:
            regime_score -= 15.0
        if close < ema_200:
            regime_score -= 25.0

        # Breadth (+- 20 pts)
        if pct_above_50_sma >= 65.0:
            regime_score += 15.0
        elif pct_above_50_sma >= 50.0:
            regime_score += 5.0
        elif pct_above_50_sma < 35.0:
            regime_score -= 20.0

        if advance_decline_ratio >= 1.5:
            regime_score += 5.0
        elif advance_decline_ratio < 0.7:
            regime_score -= 10.0

        # Volatility / VIX (+- 10 pts)
        if india_vix < 15.0:
            regime_score += 5.0
        elif india_vix > 22.0:
            regime_score -= 20.0
        elif india_vix > 18.0:
            regime_score -= 5.0

        # Determine Final Regime and Stance
        if regime_score >= 80.0:
            regime = MarketRegime.STRONG_BULL
            stance = TradingStance.AGGRESSIVE
            allow_longs = True
            risk_mult = 1.0
            trend_desc = "Strong Bull Market: Nifty trading above all major EMAs with broad participation."
        elif regime_score >= 65.0:
            regime = MarketRegime.BULL
            stance = TradingStance.NORMAL
            allow_longs = True
            risk_mult = 0.75
            trend_desc = "Bullish Trend: Index supported above 50 EMA with healthy advance/decline."
        elif regime_score >= 45.0:
            regime = MarketRegime.NEUTRAL
            stance = TradingStance.SELECTIVE
            allow_longs = True
            risk_mult = 0.50
            trend_desc = "Consolidation / Range-bound: Mixed breadth, higher conviction required."
        elif regime_score >= 25.0:
            regime = MarketRegime.BEAR
            stance = TradingStance.DEFENSIVE
            allow_longs = False
            risk_mult = 0.25
            trend_desc = "Bear Market Correction: Index below 50 EMA, weak breadth. Capital preservation priority."
        else:
            regime = MarketRegime.STRONG_BEAR
            stance = TradingStance.NO_TRADE
            allow_longs = False
            risk_mult = 0.0
            trend_desc = "Severe Bear Market / Panic: Index breakdown below 200 EMA with elevated VIX. NO LONG TRADES."

        summary = (
            f"Market Regime: {regime.value} | Stance: {stance.value} | Nifty Close: ₹{close:,.1f} | "
            f"Breadth (>50 SMA): {pct_above_50_sma:.1f}% | VIX: {india_vix:.1f}"
        )

        return RegimeAnalysisResult(
            regime=regime,
            trading_stance=stance,
            confidence=0.92,
            nifty_close=round(float(close), 2),
            trend_description=trend_desc,
            advance_decline_ratio=advance_decline_ratio,
            pct_above_50_sma=pct_above_50_sma,
            india_vix=india_vix,
            allow_long_swing_trades=allow_longs,
            risk_multiplier=risk_mult,
            summary=summary,
        )
