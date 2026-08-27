"""Point-in-time NSE research dataset builder.

Uses the complete NSE UDiFF bhavcopy for each requested trading day. Because
historical NSE security-master snapshots are not guaranteed to be available,
this module reconstructs the *observed trading universe* from the symbols that
actually appear in each daily bhavcopy. It never substitutes today's universe
for a historical date.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.data.nse_historical_source import NSEHistoricalOHLCVSource


@dataclass(frozen=True)
class HistoricalResearchDataset:
    stock_dfs: dict[str, pd.DataFrame]
    observed_universe_by_date: dict[str, list[str]]
    diagnostics: dict


def load_observed_universe(
    start_date: date,
    end_date: date,
    timeout_seconds: float = 20.0,
) -> HistoricalResearchDataset:
    """Download one complete bhavcopy per trading day and build PIT stock histories."""
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")
    source = NSEHistoricalOHLCVSource(end_date, lookback_calendar_days=(end_date - start_date).days, timeout_seconds=timeout_seconds)
    # Passing no symbols to fetch_many intentionally returns no buckets, so the
    # research builder reads the complete daily source directly via _day(). This
    # is kept in one module to make the transport boundary explicit.
    buckets: dict[str, list[pd.DataFrame]] = {}
    universe_by_date: dict[str, list[str]] = {}
    day = start_date
    while day <= end_date:
        if day.weekday() < 5:
            try:
                frame = source._day(day)  # complete bhavcopy, not a symbol-filtered request
                symbols = sorted(frame["symbol"].dropna().astype(str).str.upper().unique().tolist())
                universe_by_date[day.isoformat()] = symbols
                for symbol, group in frame.groupby("symbol"):
                    buckets.setdefault(symbol, []).append(group[["timestamp", "open", "high", "low", "close", "volume"]].copy())
            except (FileNotFoundError, OSError, IOError, ValueError):
                # Missing historical sessions are recorded by diagnostics and are
                # never replaced with current-universe data.
                universe_by_date.setdefault(day.isoformat(), [])
        day += timedelta(days=1)

    stock_dfs: dict[str, pd.DataFrame] = {}
    for symbol, frames in buckets.items():
        out = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        stock_dfs[symbol] = out.drop_duplicates("timestamp").reset_index(drop=True)
    return HistoricalResearchDataset(stock_dfs, universe_by_date, {
        "requested_start": start_date.isoformat(),
        "requested_end": end_date.isoformat(),
        "observed_sessions": len([d for d, symbols in universe_by_date.items() if symbols]),
        "symbols_observed": len(stock_dfs),
        "source_diagnostics": source.diagnostics.__dict__,
    })
