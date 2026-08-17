"""
Empirical Probability-of-Path & Expected Value Engine — src/quant/probability_engine.py

Enforces P0 Fix #9, #10, and #11 Compliance:
  1. Complete removal of hardcoded EMPIRICAL_DATA tables and arbitrary percentage adjustments.
  2. Probabilities derived strictly from real historical setup outcomes.
  3. Minimum sample size (n >= 30) required for CALIBRATED status; returns UNAVAILABLE otherwise.
  4. Net Expected Value (Net EV) incorporating slippage and transaction costs.
"""

from dataclasses import dataclass
import logging
from typing import Any
import pandas as pd

from src.core.types import MarketRegime, PatternType

logger = logging.getLogger(__name__)


@dataclass
class HistoricalSetupOutcome:
    symbol: str
    pattern_type: PatternType
    market_regime: MarketRegime
    setup_date: str
    entry_price: float
    stop_loss: float
    target_1: float
    t1_hit_before_sl: bool
    holding_sessions: int


class HistoricalSetupOutcomeStore:
    """Stores and queries verified historical setup outcomes for empirical probability calculation."""

    _records: list[HistoricalSetupOutcome] = []

    @classmethod
    def register_outcomes(cls, outcomes: list[HistoricalSetupOutcome]) -> None:
        cls._records.extend(outcomes)

    @classmethod
    def query_outcomes(
        cls,
        pattern_type: PatternType,
        market_regime: MarketRegime | None = None,
    ) -> list[HistoricalSetupOutcome]:
        results = [r for r in cls._records if r.pattern_type == pattern_type]
        if market_regime is not None and market_regime != MarketRegime.UNKNOWN:
            results = [r for r in results if r.market_regime == market_regime]
        return results


@dataclass
class ProbabilityPathResult:
    win_probability: float | None
    sample_size: int
    confidence_interval: str
    confidence_type: str  # EMPIRICAL, MODEL_CALIBRATED, or UNAVAILABLE
    gross_ev: float
    net_ev: float  # Net Expected Value after friction
    risk_reward_ratio: float
    is_ev_positive: bool
    disqualification_reason: str | None = None

    @property
    def expected_value(self) -> float:
        return self.net_ev


class ProbabilityPathEngine:
    """Calculates empirical win probability P(Win), sample size, and Net EV from real observations."""

    MIN_SAMPLE_SIZE = 30

    @classmethod
    def evaluate_expectancy(
        cls,
        pattern_type: PatternType,
        market_regime: MarketRegime,
        mansfield_rs: float = 0.0,
        target1_pct: float = 10.0,
        stop_loss_pct: float = 5.0,
        fcf_pat_ratio: float | None = None,
        estimated_slippage_pct: float = 0.15,
    ) -> ProbabilityPathResult:
        """
        Calculates empirical win probability and Net EV from stored historical observations.
        If pattern is UNKNOWN or sample size < 30, returns UNAVAILABLE without hardcoded guesses.
        """
        if pattern_type == PatternType.UNKNOWN or pattern_type == PatternType.UNSTRUCTURED_TREND:
            return ProbabilityPathResult(
                win_probability=None,
                sample_size=0,
                confidence_interval="UNAVAILABLE",
                confidence_type="UNAVAILABLE",
                gross_ev=0.0,
                net_ev=0.0,
                risk_reward_ratio=0.0,
                is_ev_positive=False,
                disqualification_reason="UNAVAILABLE: Technical pattern is UNKNOWN or unstructured. Long trades blocked.",
            )

        # Query empirical historical observations matching pattern & regime
        outcomes = HistoricalSetupOutcomeStore.query_outcomes(pattern_type, market_regime)
        if len(outcomes) < cls.MIN_SAMPLE_SIZE:
            # Broaden to all market regimes for pattern if regime-specific sample is insufficient
            outcomes = HistoricalSetupOutcomeStore.query_outcomes(pattern_type, None)

        sample_size = len(outcomes)

        if sample_size < cls.MIN_SAMPLE_SIZE:
            return ProbabilityPathResult(
                win_probability=None,
                sample_size=sample_size,
                confidence_interval="UNAVAILABLE",
                confidence_type="UNAVAILABLE",
                gross_ev=0.0,
                net_ev=0.0,
                risk_reward_ratio=0.0,
                is_ev_positive=False,
                disqualification_reason=f"UNAVAILABLE: Insufficient empirical historical observations ({sample_size} < 30 min required).",
            )

        # Compute empirical win rate from real observations
        wins = sum(1 for o in outcomes if o.t1_hit_before_sl)
        win_prob = round(wins / sample_size, 3)

        # Wilson score confidence interval calculation
        z = 1.96  # 95% confidence
        denom = 1 + z**2 / sample_size
        centre = (win_prob + z**2 / (2 * sample_size)) / denom
        margin = z * ((win_prob * (1 - win_prob) / sample_size + z**2 / (4 * sample_size**2)) ** 0.5) / denom
        ci_lower = max(0.0, round((centre - margin) * 100.0, 1))
        ci_upper = min(100.0, round((centre + margin) * 100.0, 1))
        confidence_interval = f"[{ci_lower}%, {ci_upper}%]"

        # Net Expected Value computation
        risk_reward = target1_pct / max(0.1, stop_loss_pct)
        gross_ev = (win_prob * target1_pct) - ((1.0 - win_prob) * stop_loss_pct)
        net_ev = gross_ev - estimated_slippage_pct

        is_ev_positive = net_ev > 0.0 and win_prob >= 0.55

        disqualification_reason = None
        if not is_ev_positive:
            if win_prob < 0.55:
                disqualification_reason = f"Empirical win probability ({win_prob*100:.1f}%) is below 55.0% threshold."
            else:
                disqualification_reason = f"Net Expected Value ({net_ev:.2f}%) after friction is <= 0.0%."

        return ProbabilityPathResult(
            win_probability=win_prob,
            sample_size=sample_size,
            confidence_interval=confidence_interval,
            confidence_type="EMPIRICAL",
            gross_ev=round(gross_ev, 2),
            net_ev=round(net_ev, 2),
            risk_reward_ratio=round(risk_reward, 2),
            is_ev_positive=is_ev_positive,
            disqualification_reason=disqualification_reason,
        )
