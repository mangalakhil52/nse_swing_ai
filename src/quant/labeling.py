"""Triple-barrier style forward-path labeling for research only."""

from __future__ import annotations

import pandas as pd


def triple_barrier_label(
    frame: pd.DataFrame,
    entry_price: float,
    upper_pct: float,
    lower_pct: float,
    horizon_bars: int,
) -> dict[str, float | int | str | None]:
    """Label the first barrier touched, preserving MFE/MAE over the horizon."""
    if entry_price <= 0 or upper_pct <= 0 or lower_pct <= 0 or horizon_bars <= 0:
        raise ValueError("Invalid triple-barrier configuration")
    future = frame.copy().sort_values("timestamp").head(horizon_bars)
    if future.empty:
        return {"label": None, "bars_held": 0, "mfe_pct": 0.0, "mae_pct": 0.0}
    high = pd.to_numeric(future["high"], errors="coerce")
    low = pd.to_numeric(future["low"], errors="coerce")
    upper = entry_price * (1 + upper_pct / 100)
    lower = entry_price * (1 - lower_pct / 100)
    label = "TIMEOUT"
    bars = len(future)
    for i, (h, l) in enumerate(zip(high, low), start=1):
        if pd.isna(h) or pd.isna(l):
            continue
        # Conservative same-bar ambiguity rule: if both barriers are touched,
        # assume the adverse barrier was hit first.
        if l <= lower and h >= upper:
            label = "LOSS"
            bars = i
            break
        if l <= lower:
            label = "LOSS"
            bars = i
            break
        if h >= upper:
            label = "WIN"
            bars = i
            break
    mfe = float((high.max() / entry_price - 1) * 100) if high.notna().any() else 0.0
    mae = float((low.min() / entry_price - 1) * 100) if low.notna().any() else 0.0
    return {"label": label, "bars_held": bars, "mfe_pct": mfe, "mae_pct": mae}
