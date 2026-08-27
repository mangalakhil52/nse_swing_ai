"""Efficient universe-wide historical loader built on official NSE Bhavcopies.

The key optimization is temporal batching: one Bhavcopy request supplies the
entire equity universe for a trading day. Never perform N-symbol x M-day HTTP
requests when the exchange already publishes a universe-wide daily file.
"""

from __future__ import annotations

from datetime import date, timedelta
import logging

import pandas as pd

from src.core.exceptions import DataUnavailableException
from src.data.nse_provider import NseDataProvider

logger = logging.getLogger(__name__)


class BulkHistoricalLoader:
    """Load many NSE symbols with one exchange-file fetch per trading day."""

    def __init__(self, provider: NseDataProvider | None = None):
        self.provider = provider or NseDataProvider()

    async def load(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        min_bars: int = 50,
    ) -> dict[str, pd.DataFrame]:
        wanted = {str(s).strip().upper() for s in symbols if str(s).strip()}
        if not wanted:
            return {}

        daily_frames: list[pd.DataFrame] = []
        current = start_date
        attempted_days = 0
        successful_days = 0
        while current <= end_date:
            if current.weekday() < 5:
                attempted_days += 1
                try:
                    day = await self.provider.fetch_bhavcopy_for_date(current)
                    if not day.empty:
                        day = day[day["symbol"].isin(wanted)].copy()
                        if not day.empty:
                            daily_frames.append(day)
                            successful_days += 1
                except DataUnavailableException as exc:
                    logger.warning("Historical Bhavcopy unavailable for %s: %s", current, exc)
            current += timedelta(days=1)

        logger.info(
            "Bulk history: %d trading days attempted, %d loaded, %d requested symbols",
            attempted_days,
            successful_days,
            len(wanted),
        )
        if not daily_frames:
            raise DataUnavailableException("No official NSE historical Bhavcopy data available for requested window")

        all_rows = pd.concat(daily_frames, ignore_index=True)
        all_rows["timestamp"] = pd.to_datetime(all_rows["timestamp"])
        all_rows["symbol"] = all_rows["symbol"].astype(str).str.strip().str.upper()
        all_rows = (
            all_rows.sort_values(["symbol", "timestamp"])
            .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
            .reset_index(drop=True)
        )

        result: dict[str, pd.DataFrame] = {}
        for symbol, group in all_rows.groupby("symbol", sort=False):
            frame = group.sort_values("timestamp").reset_index(drop=True)
            if len(frame) >= min_bars:
                result[symbol] = frame

        logger.info("Bulk history produced validated series for %d/%d symbols", len(result), len(wanted))
        return result

    async def close(self) -> None:
        await self.provider.close()
