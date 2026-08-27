"""Uncertainty-aware decision utilities.

A high point estimate is not enough. This layer makes sample size and confidence
explicit so the decision engine can reject fragile signals.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BernoulliUncertainty:
    estimate: float
    lower_95: float
    upper_95: float
    sample_size: int
    effective_sample_size: float


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> BernoulliUncertainty:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Invalid Bernoulli observations")
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return BernoulliUncertainty(
        estimate=p,
        lower_95=max(0.0, centre - half),
        upper_95=min(1.0, centre + half),
        sample_size=trials,
        effective_sample_size=float(trials),
    )


def decay_weighted_effective_sample_size(weights: list[float]) -> float:
    """Effective N = (sum w)^2 / sum(w^2), useful for recency-weighted evidence."""
    clean = [float(w) for w in weights if math.isfinite(float(w)) and float(w) > 0]
    if not clean:
        return 0.0
    total = sum(clean)
    return total * total / sum(w * w for w in clean)
