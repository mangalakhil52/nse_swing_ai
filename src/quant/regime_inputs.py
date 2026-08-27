"""Point-in-time market-regime input construction."""

from __future__ import annotations

import pandas as pd


def compute_breadth(stock_dfs: dict[str, pd.DataFrame], as_of_date=None) -> tuple[float, float]:
    """Return (advance/decline ratio, % stocks above 50-SMA) from supplied PIT series."""
    advances = 0
    declines = 0
    participation_total = 0
    above_50 = 0

    for df in stock_dfs.values():
        if df is None or df.empty:
            continue
        frame = df.copy()
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            if as_of_date is not None:
                frame = frame[frame["timestamp"].dt.date <= pd.Timestamp(as_of_date).date()]
            frame = frame.sort_values("timestamp")
        if len(frame) < 51:
            continue

        close = pd.to_numeric(frame["close"], errors="coerce").dropna()
        if len(close) < 51:
            continue
        latest = float(close.iloc[-1])
        previous = float(close.iloc[-2])
        if latest > previous:
            advances += 1
        elif latest < previous:
            declines += 1

        sma50 = float(close.rolling(50).mean().iloc[-1])
        if pd.notna(sma50):
            participation_total += 1
            if latest > sma50:
                above_50 += 1

    if advances + declines == 0 or participation_total == 0:
        raise ValueError("Insufficient point-in-time breadth observations")

    ad_ratio = round(advances / max(declines, 1), 3)
    pct_above_50 = round(above_50 / participation_total * 100.0, 2)
    return ad_ratio, pct_above_50


def latest_vix(vix_df: pd.DataFrame, as_of_date=None) -> float:
    """Extract the latest point-in-time India VIX close."""
    if vix_df is None or vix_df.empty:
        raise ValueError("India VIX data unavailable")
    frame = vix_df.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        if as_of_date is not None:
            frame = frame[frame["timestamp"].dt.date <= pd.Timestamp(as_of_date).date()]
        frame = frame.sort_values("timestamp")
    if frame.empty:
        raise ValueError("No India VIX observation at or before as-of date")
    value = float(frame["close"].iloc[-1])
    if value <= 0:
        raise ValueError("Invalid India VIX value")
    return value
