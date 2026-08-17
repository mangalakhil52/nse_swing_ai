"""
Empirical Probability-of-Path & Expected Value Engine — src/quant/probability_engine.py

Enforces P0 Compliance:
  1. Complete removal of hardcoded EMPIRICAL_DATA tables and arbitrary percentage adjustments.
  2. Probabilities derived strictly from real historical setup outcomes.
  3. Minimum sample size (n >= 30) required for EMPIRICAL status; returns UNAVAILABLE otherwise.
  4. Net Expected Value (Net EV) incorporating slippage and transaction costs.
  5. Disk persistence & idempotent registration for HistoricalSetupOutcomeStore.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any
import pandas as pd

from config.settings import settings
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
    exit_date: str
    source: str
    mfe: float = 0.0  # Maximum Favorable Excursion (%)
    mae: float = 0.0  # Maximum Adverse Excursion (%)

    @property
    def outcome(self) -> str:
        return "WIN" if self.t1_hit_before_sl else "LOSS"

    @property
    def stop_price(self) -> float:
        return self.stop_loss

    @property
    def target_price(self) -> float:
        return self.target_1

    @property
    def holding_period(self) -> int:
        return self.holding_sessions

    @property
    def unique_key(self) -> str:
        return f"{self.symbol.upper().strip()}:{self.setup_date}:{self.pattern_type.value}"


def validate_outcome(outcome: HistoricalSetupOutcome) -> tuple[bool, str | None]:
    """
    Validates a HistoricalSetupOutcome record prior to registration.
    Returns (True, None) if valid, or (False, reason) if invalid.
    """
    if not outcome.setup_date or not isinstance(outcome.setup_date, str):
        return False, "Missing or invalid setup_date"
    if outcome.entry_price <= 0.0:
        return False, f"Invalid entry_price ({outcome.entry_price} <= 0)"
    if outcome.stop_loss <= 0.0:
        return False, f"Invalid stop_loss ({outcome.stop_loss} <= 0)"
    if outcome.target_1 <= 0.0:
        return False, f"Invalid target_1 ({outcome.target_1} <= 0)"
    if not outcome.exit_date or outcome.exit_date < outcome.setup_date:
        return False, f"Invalid exit_date ({outcome.exit_date} < setup_date {outcome.setup_date})"
    if outcome.holding_sessions < 0:
        return False, f"Invalid holding_sessions ({outcome.holding_sessions} < 0)"
    if not outcome.source or not outcome.source.strip():
        return False, "Missing source data provenance"
    if outcome.pattern_type == PatternType.UNKNOWN:
        return False, "PatternType cannot be UNKNOWN"
    if outcome.market_regime == MarketRegime.UNKNOWN:
        return False, "MarketRegime cannot be UNKNOWN"
    return True, None


class HistoricalSetupOutcomeStore:
    """Stores, persists, and queries verified historical setup outcomes for empirical probability calculation."""

    _records: list[HistoricalSetupOutcome] = []
    _cache_file: Path | None = None

    @classmethod
    def get_cache_file(cls) -> Path:
        if cls._cache_file is None:
            cache_dir = settings.CACHE_DIR / "probability"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cls._cache_file = cache_dir / "historical_setup_outcomes.json"
        return cls._cache_file

    @classmethod
    def clear(cls) -> None:
        """Clears in-memory records (useful for testing)."""
        cls._records.clear()

    @classmethod
    def load_from_disk(cls) -> None:
        """Loads verified historical outcomes from persistent JSON cache if present."""
        fpath = cls.get_cache_file()
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                recs = []
                existing_keys = set()
                for item in data:
                    rec = HistoricalSetupOutcome(
                        symbol=item["symbol"],
                        pattern_type=PatternType(item["pattern_type"]),
                        market_regime=MarketRegime(item["market_regime"]),
                        setup_date=item["setup_date"],
                        entry_price=item["entry_price"],
                        stop_loss=item["stop_loss"],
                        target_1=item["target_1"],
                        t1_hit_before_sl=item["t1_hit_before_sl"],
                        holding_sessions=item.get("holding_sessions", 0),
                        exit_date=item.get("exit_date", item["setup_date"]),
                        source=item.get("source", "NSE_BHAVCOPY_DAILY"),
                        mfe=item.get("mfe", 0.0),
                        mae=item.get("mae", 0.0),
                    )
                    is_valid, _ = validate_outcome(rec)
                    if is_valid and rec.unique_key not in existing_keys:
                        recs.append(rec)
                        existing_keys.add(rec.unique_key)
                cls._records = recs
                logger.info(f"Loaded {len(cls._records)} empirical historical setup outcomes from persistent store.")
            except Exception as e:
                logger.warning(f"Failed to load historical outcomes store from disk: {e}")

    @classmethod
    def persist_to_disk(cls) -> None:
        """Persists stored historical setup outcomes to disk."""
        fpath = cls.get_cache_file()
        try:
            items = []
            for r in cls._records:
                items.append({
                    "symbol": r.symbol,
                    "pattern_type": r.pattern_type.value,
                    "market_regime": r.market_regime.value,
                    "setup_date": r.setup_date,
                    "entry_price": r.entry_price,
                    "stop_loss": r.stop_loss,
                    "target_1": r.target_1,
                    "t1_hit_before_sl": r.t1_hit_before_sl,
                    "holding_sessions": r.holding_sessions,
                    "exit_date": r.exit_date,
                    "source": r.source,
                    "mfe": r.mfe,
                    "mae": r.mae,
                })
            fpath.write_text(json.dumps(items, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to persist historical outcomes store to disk: {e}")

    @classmethod
    def register_outcomes(cls, outcomes: list[HistoricalSetupOutcome], persist: bool = True) -> tuple[int, int]:
        """
        Idempotently validates and registers historical setup outcomes.
        Returns (registered_count, rejected_count).
        """
        existing_keys = {r.unique_key for r in cls._records}
        added_count = 0
        rejected_count = 0

        for r in outcomes:
            is_valid, reason = validate_outcome(r)
            if not is_valid:
                logger.debug(f"Rejected HistoricalSetupOutcome for {r.symbol} on {r.setup_date}: {reason}")
                rejected_count += 1
                continue

            if r.unique_key in existing_keys:
                # Skip duplicate
                continue

            cls._records.append(r)
            existing_keys.add(r.unique_key)
            added_count += 1

        if added_count > 0 and persist:
            cls.persist_to_disk()

        return added_count, rejected_count

    @classmethod
    def query_outcomes(
        cls,
        pattern_type: PatternType,
        market_regime: MarketRegime | None = None,
    ) -> list[HistoricalSetupOutcome]:
        if market_regime == MarketRegime.UNKNOWN:
            return []
        results = [r for r in cls._records if r.pattern_type == pattern_type]
        if market_regime is not None:
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
        If pattern or regime is UNKNOWN or sample size < 30, returns UNAVAILABLE without hardcoded guesses.
        """
        if market_regime == MarketRegime.UNKNOWN:
            return ProbabilityPathResult(
                win_probability=None,
                sample_size=0,
                confidence_interval="UNAVAILABLE",
                confidence_type="UNAVAILABLE",
                gross_ev=0.0,
                net_ev=0.0,
                risk_reward_ratio=0.0,
                is_ev_positive=False,
                disqualification_reason="UNAVAILABLE: Market regime is UNKNOWN.",
            )

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

        # Ensure persistent store is loaded
        if not HistoricalSetupOutcomeStore._records:
            HistoricalSetupOutcomeStore.load_from_disk()

        # Query empirical historical observations matching pattern & regime strictly
        outcomes = HistoricalSetupOutcomeStore.query_outcomes(pattern_type, market_regime)
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
                disqualification_reason=f"UNAVAILABLE: Insufficient regime-specific empirical observations ({sample_size} < 30 min required).",
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
