"""Continuous market-regime transition score.

Instead of treating regimes as instantaneous labels, this estimates transition
intensity from normalized trend, breadth and volatility changes. It is a state
feature for downstream agents, not a trade recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeTransition:
    transition_score: float
    trend_impulse: float
    breadth_impulse: float
    volatility_impulse: float
    state: str


def _z(x: float, mean: float, std: float) -> float:
    return float((x - mean) / max(std, 1e-9))


def compute_regime_transition(
    benchmark_df: pd.DataFrame,
    breadth_series: pd.Series,
    vix_series: pd.Series,
) -> RegimeTransition:
    """Compute a bounded transition intensity from point-in-time market series."""
    if len(benchmark_df) < 80 or len(breadth_series) < 20 or len(vix_series) < 20:
        raise ValueError("Insufficient observations for regime transition analysis")

    b = benchmark_df.sort_values("timestamp") if "timestamp" in benchmark_df.columns else benchmark_df
    close = pd.to_numeric(b["close"], errors="coerce").dropna()
    if len(close) < 80:
        raise ValueError("Insufficient benchmark observations")

    ret20 = float(close.iloc[-1] / close.iloc[-21] - 1)
    ret60 = float(close.iloc[-1] / close.iloc[-61] - 1)
    hist_ret = close.pct_change().dropna()
    trend_impulse = _z(ret20, float(hist_ret.tail(60).mean() * 20), float(hist_ret.tail(60).std() * np.sqrt(20))) + _z(ret60, 0.0, max(float(hist_ret.tail(120).std() * np.sqrt(60)), 1e-6))

    br = pd.to_numeric(breadth_series, errors="coerce").dropna()
    breadth_impulse = _z(float(br.iloc[-1] - br.iloc[-11]), 0.0, max(float(br.tail(60).std()), 1e-6))

    vx = pd.to_numeric(vix_series, errors="coerce").dropna()
    volatility_impulse = _z(float(vx.iloc[-1] - vx.iloc[-11]), 0.0, max(float(vx.tail(60).std()), 1e-6))

    raw = 0.45 * trend_impulse + 0.35 * breadth_impulse - 0.20 * volatility_impulse
    score = float(np.tanh(raw / 2.0))
    if score > 0.35:
        state = "IMPROVING"
    elif score < -0.35:
        state = "DETERIORATING"
    else:
        state = "TRANSITION"
    return RegimeTransition(
        transition_score=round(score, 6),
        trend_impulse=round(trend_impulse, 6),
        breadth_impulse=round(breadth_impulse, 6),
        volatility_impulse=round(volatility_impulse, 6),
        state=state,
    )
