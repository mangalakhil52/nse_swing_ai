"""Meta-labeling utilities for conditional trade selection.

A base setup says *what* pattern exists. Meta-labeling estimates whether taking
that setup under the current feature state is worth trading. Training data must
come from strictly historical, point-in-time setup outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MetaLabel:
    probability: float
    expected_value: float
    sample_size: int
    confidence: float
    status: str


def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    """Conservative lower confidence bound for a Bernoulli success rate."""
    if total <= 0 or wins < 0 or wins > total:
        return 0.0
    p = wins / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) / total) + z * z / (4 * total * total)) / denom
    return max(0.0, centre - margin)


def empirical_meta_label(
    outcomes: np.ndarray,
    reward_pct: float,
    risk_pct: float,
    friction_pct: float = 0.15,
    min_samples: int = 50,
) -> MetaLabel:
    """Return a conservative empirical label; never converts insufficient data into a probability."""
    clean = np.asarray(outcomes, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = len(clean)
    if n < min_samples or reward_pct <= 0 or risk_pct <= 0:
        return MetaLabel(None, None, n, 0.0, "UNAVAILABLE")

    wins = int(np.sum(clean > 0))
    p = wins / n
    lower = wilson_lower_bound(wins, n)
    ev = p * reward_pct - (1 - p) * risk_pct - friction_pct
    conservative_ev = lower * reward_pct - (1 - lower) * risk_pct - friction_pct
    confidence = max(0.0, min(1.0, (lower / max(p, 1e-12))))
    status = "TRADEABLE" if ev > 0 and conservative_ev > 0 else "REJECT"
    return MetaLabel(round(p, 4), round(ev, 4), n, round(confidence, 4), status)
