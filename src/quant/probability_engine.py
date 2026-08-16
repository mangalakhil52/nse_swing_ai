"""
Probability-of-Path & Expected Value Quantitative Engine.
Calculates historical path probability P(Win), Expected Value (EV), and Reward/Risk expectancy
before trade level approval.
"""

import logging
from dataclasses import dataclass
from src.core.types import MarketRegime, PatternType

logger = logging.getLogger(__name__)


@dataclass
class ProbabilityPathResult:
    win_probability: float  # e.g. 0.68 (68%)
    expected_value: float   # Net expected return %
    risk_reward_ratio: float
    is_ev_positive: bool
    disqualification_reason: str | None = None


class ProbabilityPathEngine:
    """Calculates win rate probability and Expected Value (EV) for candidates."""

    BASE_WIN_RATES = {
        PatternType.VOLATILITY_CONTRACTION_PATTERN: 0.68,
        PatternType.FLAT_BASE_BREAKOUT: 0.64,
        PatternType.EMA_PULLBACK_REVERSAL: 0.61,
        PatternType.CUP_AND_HANDLE: 0.65,
        PatternType.HIGH_TIGHT_FLAG: 0.70,
        PatternType.UNSTRUCTURED_TREND: 0.52,
    }

    @classmethod
    def evaluate_expectancy(
        cls,
        pattern_type: PatternType,
        market_regime: MarketRegime,
        mansfield_rs: float,
        target1_pct: float,
        stop_loss_pct: float,
        fcf_pat_ratio: float = 0.90,
    ) -> ProbabilityPathResult:
        """
        Calculates P(Win) and Expected Value (EV):
            EV = P(Win) * Target1_pct - (1 - P(Win)) * StopLoss_pct
        """
        base_p = cls.BASE_WIN_RATES.get(pattern_type, 0.55)

        # 1. Regime Adjustment
        if market_regime in [MarketRegime.STRONG_BULL, MarketRegime.BULL]:
            regime_adj = 0.08
        elif market_regime == MarketRegime.CORRECTION:
            regime_adj = -0.10
        elif market_regime in [MarketRegime.BEAR, MarketRegime.HIGH_VOLATILITY_BEAR]:
            regime_adj = -0.25
        else:
            regime_adj = 0.0

        # 2. Relative Strength Adjustment
        rs_adj = 0.05 if mansfield_rs > 0.0 else -0.05

        # 3. Cash Quality Adjustment
        cash_adj = 0.03 if fcf_pat_ratio >= 0.80 else -0.04

        win_prob = round(min(0.85, max(0.35, base_p + regime_adj + rs_adj + cash_adj)), 2)

        # Expected Value calculation
        ev = (win_prob * target1_pct) - ((1.0 - win_prob) * stop_loss_pct)
        ev = round(ev, 2)
        rr = round(target1_pct / max(0.1, stop_loss_pct), 2)

        is_valid = ev > 0.0 and win_prob >= 0.55
        reason = None
        if not is_valid:
            if win_prob < 0.55:
                reason = f"Insufficient Win Rate Probability ({win_prob*100:.0f}% < 55% min threshold)."
            elif ev <= 0.0:
                reason = f"Negative Expected Value (EV: {ev:.2f}% <= 0.0%)."

        return ProbabilityPathResult(
            win_probability=win_prob,
            expected_value=ev,
            risk_reward_ratio=rr,
            is_ev_positive=is_valid,
            disqualification_reason=reason,
        )
