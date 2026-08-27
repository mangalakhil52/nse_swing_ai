"""Live feature/alpha drift monitoring."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DriftReport:
    population_shift: float
    mean_shift: float
    variance_shift: float
    status: str


def compare_distributions(reference: list[float], current: list[float], threshold: float = 0.25) -> DriftReport:
    r = np.asarray([x for x in reference if math.isfinite(x)], dtype=float)
    c = np.asarray([x for x in current if math.isfinite(x)], dtype=float)
    if len(r) < 30 or len(c) < 30:
        return DriftReport(0.0, 0.0, 0.0, "INSUFFICIENT_DATA")
    mean_shift = abs(float(c.mean() - r.mean())) / max(float(r.std()), 1e-9)
    variance_shift = abs(float(c.var() - r.var())) / max(float(r.var()), 1e-9)
    # Symmetric bounded shift proxy; use as an operational alarm, not a p-value.
    population_shift = min(1.0, 0.5 * mean_shift / (1 + mean_shift) + 0.5 * variance_shift / (1 + variance_shift))
    status = "DRIFT" if population_shift >= threshold else "STABLE"
    return DriftReport(round(population_shift, 6), round(mean_shift, 6), round(variance_shift, 6), status)
