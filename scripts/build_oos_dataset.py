#!/usr/bin/env python3
"""Build a point-in-time NSE daily OHLCV dataset for walk-forward OOS validation.

The dataset is reconstructed from the official daily NSE bhavcopies. The union
of symbols observed on each historical trading day is retained, so IPOs enter
only from their first observed trading day and delisted names naturally stop
producing observations. No current-universe fallback is used.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.nse_historical_source import NSEHistoricalOHLCVSource


def _trading_days(start: date, end: date):
    day = start
    while day <= end:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


def build(start: date, end: date, output_dir: Path, timeout: float = 30.0) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = NSEHistoricalOHLCVSource(end, lookback_calendar_days=(end - start).days, timeout_seconds=timeout)
    buckets: dict[str, list[pd.DataFrame]] = {}
    attempted = successful = failed = 0

    for day in _trading_days(start, end):
        attempted += 1
        try:
            frame = source._day(day)  # one official NSE archive per trading day
            successful += 1
            for symbol, group in frame.groupby("symbol"):
                if not symbol or symbol == "NAN":
                    continue
                buckets.setdefault(symbol, []).append(group[["timestamp", "open", "high", "low", "close", "volume"]].copy())
        except Exception as exc:
            failed += 1
            print(f"WARNING {day}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not buckets:
        raise RuntimeError("No historical NSE observations were downloaded")

    symbols_written = 0
    rows_written = 0
    for symbol, frames in sorted(buckets.items()):
        df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        df = df.drop_duplicates("timestamp", keep="last")
        df.to_csv(output_dir / f"{symbol}.csv", index=False)
        symbols_written += 1
        rows_written += len(df)

    diagnostics = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "attempted_days": attempted,
        "successful_days": successful,
        "failed_days": failed,
        "symbols": symbols_written,
        "rows": rows_written,
        "source": "NSE official daily UDiFF bhavcopy",
    }
    (output_dir / "_manifest.json").write_text(__import__("json").dumps(diagnostics, indent=2), encoding="utf-8")
    print(__import__("json").dumps(diagnostics, indent=2))
    if successful == 0:
        raise RuntimeError("All historical NSE downloads failed")
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", default="artifacts/oos_data")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    build(date.fromisoformat(args.start), date.fromisoformat(args.end), Path(args.output), args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
