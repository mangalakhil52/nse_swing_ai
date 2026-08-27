"""Regime-conditioned cross-sectional alpha and risk diagnostics.

This module intentionally avoids a single weighted indicator score. It builds
orthogonal, volatility-normalized features and converts them into a compact
cross-sectional alpha signal. All calculations are point-in-time when the
caller supplies data cut at the decision timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlphaFeatures:
    momentum_21: float
    momentum_63: float
    momentum_126: float
    trend_quality: float
    volatility: float
    downside_volatility: float
    volume_surprise: float
    range_compression: float
    breakout_distance: float
    relative_strength: float
    alpha_score: float


def _safe_return(close: pd.Series, periods: int) -> float:
    if len(close) <= periods or close.iloc[-periods - 1] <= 0:
        return 0.0
    return float(close.iloc[-1] / close.iloc[-periods - 1] - 1.0)


def _rolling_slope(values: pd.Series) -> float:
    if len(values) < 20:
        return 0.0
    y = np.log(np.maximum(values.astype(float).to_numpy(), 1e-12))
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope * 252.0)


def compute_alpha_features(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None) -> AlphaFeatures:
    """Compute robust, scale-free features from a PIT OHLCV frame."""
    df = stock_df.copy().sort_values("timestamp") if "timestamp" in stock_df.columns else stock_df.copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    valid = pd.concat([close, high, low, volume], axis=1).dropna()
    if len(valid) < 130:
        raise ValueError("At least 130 OHLCV observations are required for advanced alpha features")
    close, high, low, volume = [valid.iloc[:, i] for i in range(4)]

    returns = close.pct_change().dropna()
    vol20 = float(returns.tail(20).std(ddof=1) * math.sqrt(252))
    downside = returns[returns < 0].tail(60)
    downside_vol = float(downside.std(ddof=1) * math.sqrt(252)) if len(downside) >= 5 else vol20

    log_ret = np.log(close).diff()
    trend_quality = float(abs(log_ret.tail(63).mean()) / max(log_ret.tail(63).std(ddof=1), 1e-9))
    volume_ratio = float(volume.tail(5).mean() / max(volume.tail(60).median(), 1.0))

    true_range = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr14 = float(true_range.rolling(14).mean().iloc[-1])
    atr60 = float(true_range.rolling(60).mean().iloc[-1])
    range_compression = float(atr14 / max(atr60, 1e-9))

    prior_63_high = float(close.iloc[-64:-1].max())
    breakout_distance = float((close.iloc[-1] - prior_63_high) / max(atr14, 1e-9))

    relative_strength = 0.0
    if benchmark_df is not None and not benchmark_df.empty:
        b = benchmark_df.copy().sort_values("timestamp") if "timestamp" in benchmark_df.columns else benchmark_df.copy()
        bclose = pd.to_numeric(b["close"], errors="coerce").dropna()
        if len(bclose) >= 64:
            relative_strength = _safe_return(close, 63) - _safe_return(bclose, 63)

    m21 = _safe_return(close, 21)
    m63 = _safe_return(close, 63)
    m126 = _safe_return(close, 126)

    # Robust cross-sectional ingredients. The score is deliberately not a
    # probability; calibration belongs to the empirical outcome engine.
    raw = (
        0.25 * np.tanh(m21 / max(vol20 * math.sqrt(21), 1e-6))
        + 0.30 * np.tanh(m63 / max(vol20 * math.sqrt(63), 1e-6))
        + 0.20 * np.tanh(m126 / max(vol20 * math.sqrt(126), 1e-6))
        + 0.15 * np.tanh(relative_strength / max(vol20, 0.01))
        + 0.10 * np.tanh((volume_ratio - 1.0))
    )
    alpha_score = float(raw)

    return AlphaFeatures(
        momentum_21=round(m21, 6),
        momentum_63=round(m63, 6),
        momentum_126=round(m126, 6),
        trend_quality=round(trend_quality, 6),
        volatility=round(vol20, 6),
        downside_volatility=round(downside_vol, 6),
        volume_surprise=round(volume_ratio, 6),
        range_compression=round(range_compression, 6),
        breakout_distance=round(breakout_distance, 6),
        relative_strength=round(relative_strength, 6),
        alpha_score=round(alpha_score, 6),
    )


def cross_sectional_zscores(values: dict[str, float], winsorize: float = 3.0) -> dict[str, float]:
    """Winsorized cross-sectional z-scores for ranking an NSE universe."""
    if len(values) < 5:
        return {k: 0.0 for k in values}
    s = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 5 or float(s.std(ddof=1)) == 0.0:
        return {k: 0.0 for k in values}
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(lo, hi)
    z = (s - s.mean()) / max(float(s.std(ddof=1)), 1e-9)
    z = z.clip(-winsorize, winsorize)
    return {k: float(z.get(k, 0.0)) for k in values}
