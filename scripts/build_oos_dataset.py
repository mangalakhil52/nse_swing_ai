#!/usr/bin/env python3
"""Build a point-in-time NSE daily OHLCV dataset for walk-forward OOS validation.

The dataset is reconstructed from official daily NSE bhavcopies. The union of
symbols observed on each historical trading day is retained, so IPOs enter only
from their first observed trading day and delisted names naturally stop
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

# NSE Capital Market holiday calendars. These are intentionally explicit and
# versioned: an exchange holiday is not a failed data download. Sources:
# NSE CM holiday circulars for 2023, 2024, 2025 and 2026.
NSE_CM_HOLIDAYS = {
    date(2023, 1, 26), date(2023, 3, 7), date(2023, 3, 30), date(2023, 4, 4),
    date(2023, 4, 7), date(2023, 4, 14), date(2023, 5, 1), date(2023, 6, 28),
    date(2023, 8, 15), date(2023, 9, 19), date(2023, 10, 2), date(2023, 10, 24),
    date(2023, 11, 14), date(2023, 11, 27), date(2023, 12, 25),
    date(2024, 1, 26), date(2024, 3, 8), date(2024, 3, 25), date(2024, 3, 29),
    date(2024, 4, 11), date(2024, 4, 17), date(2024, 5, 1), date(2024, 6, 17),
    date(2024, 7, 17), date(2024, 8, 15), date(2024, 10, 2), date(2024, 11, 1),
    date(2024, 11, 15), date(2024, 12, 25),
    date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31), date(2025, 4, 10),
    date(2025, 4, 14), date(2025, 4, 18), date(2025, 5, 1), date(2025, 8, 15),
    date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 21), date(2025, 10, 22),
    date(2025, 11, 5), date(2025, 12, 25),
    date(2026, 1, 26), date(2026, 3, 3), date(2026, 3, 26), date(2026, 3, 31),
    date(2026, 4, 3), date(2026, 4, 14), date(2026, 5, 1), date(2026, 5, 28),
    date(2026, 6, 26), date(2026, 9, 14), date(2026, 10, 2), date(2026, 10, 20),
    date(2026, 11, 10), date(2026, 11, 24), date(2026, 12, 25),
}


def _trading_days(start: date, end: date):
    day = start
    while day <= end:
        if day.weekday() < 5 and day not in NSE_CM_HOLIDAYS:
            yield day
        day += timedelta(days=1)


def build(start: date, end: date, output_dir: Path, timeout: float = 30.0) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = NSEHistoricalOHLCVSource(end, lookback_calendar_days=(end - start).days, timeout_seconds=timeout)
    buckets: dict[str, list[pd.DataFrame]] = {}
    attempted = successful = failed = expected_holidays = 0

    day = start
    while day <= end:
        if day.weekday() >= 5:
            day += timedelta(days=1)
            continue
        if day in NSE_CM_HOLIDAYS:
            expected_holidays += 1
            day += timedelta(days=1)
            continue
        attempted += 1
        try:
            frame = source._day(day)
            successful += 1
            for symbol, group in frame.groupby("symbol"):
                if not symbol or symbol == "NAN":
                    continue
                buckets.setdefault(symbol, []).append(group[["timestamp", "open", "high", "low", "close", "volume"]].copy())
        except Exception as exc:
            failed += 1
            print(f"WARNING {day}: {type(exc).__name__}: {exc}", file=sys.stderr)
        day += timedelta(days=1)

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
        "expected_holidays": expected_holidays,
        "symbols": symbols_written,
        "rows": rows_written,
        "source": "NSE official daily CM bhavcopy; legacy archive pre-2024-07-08, UDiFF thereafter",
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
