"""Standardized performance metrics for the research pipeline."""
from __future__ import annotations
import math
import numpy as np


def max_drawdown(returns: list[float]) -> float:
    equity = np.cumprod(1 + np.asarray(returns, dtype=float))
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity / peaks - 1)) if len(equity) else 0.0


def sharpe(returns: list[float], periods: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods))


def sortino(returns: list[float], periods: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    downside = r[r < 0]
    if len(r) < 2 or len(downside) == 0 or downside.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / downside.std(ddof=1) * math.sqrt(periods))


def profit_factor(returns: list[float]) -> float:
    r = np.asarray(returns, dtype=float)
    losses = abs(float(r[r < 0].sum()))
    return float(r[r > 0].sum() / losses) if losses > 0 else float("inf")
