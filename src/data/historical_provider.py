"""
Real Historical Data Provider Module — src/data/historical_provider.py

Provides clean, date-by-date historical OHLCV data from official cached NSE Bhavcopies and DB tables.
Enforces P0 data integrity:
  1. Oldest to newest chronological ordering.
  2. Rejects symbols with insufficient history (< min_bars).
  3. ZERO synthetic fallback generation (no linspace/random numbers).
"""

from datetime import date, datetime, timedelta
import io
import json
import logging
from pathlib import Path
from typing import Any
import pandas as pd

from config.settings import settings
from src.core.exceptions import DataUnavailableException
from src.data.validation import validate_ohlcv_dataframe

logger = logging.getLogger(__name__)


class HistoricalDataProvider:
    """Fetches and validates real historical OHLCV data for NSE equities."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or settings.CACHE_DIR / "historical"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bhavcopy_dir = settings.CACHE_DIR / "bhavcopy"
        self.bhavcopy_dir.mkdir(parents=True, exist_ok=True)

    def _get_symbol_cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.upper()}_history.parquet"

    async def get_daily_ohlcv(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        min_bars: int = 50,
    ) -> pd.DataFrame:
        """
        Fetches genuine historical OHLCV observations for a symbol.
        Returns a validated pandas DataFrame ordered oldest -> newest.
        Raises DataUnavailableException if insufficient bars or data fails validation.
        """
        symbol = symbol.upper().strip()
        cache_path = self._get_symbol_cache_path(symbol)

        # 1. Try reading from parquet cache if present and covers date range
        if cache_path.exists():
            try:
                df_cached = pd.read_parquet(cache_path)
                df_cached["timestamp"] = pd.to_datetime(df_cached["timestamp"])
                df_filtered = df_cached[
                    (df_cached["timestamp"].dt.date >= start_date)
                    & (df_cached["timestamp"].dt.date <= end_date)
                ].sort_values("timestamp")

                if len(df_filtered) >= min_bars:
                    validate_ohlcv_dataframe(df_filtered, min_bars=min_bars)
                    return df_filtered.reset_index(drop=True)
            except Exception as e:
                logger.debug(f"Cache miss or error reading {cache_path}: {e}")

        # 2. Build multi-day history from available cached daily bhavcopies
        records = []
        curr = start_date
        while curr <= end_date:
            bhav_file = self.bhavcopy_dir / f"sec_bhavdata_full_{curr.strftime('%d%m%Y')}.csv"
            if bhav_file.exists():
                try:
                    bhav_df = pd.read_csv(bhav_file)
                    bhav_df.columns = [c.strip().upper() for c in bhav_df.columns]
                    sym_row = bhav_df[bhav_df["SYMBOL"].str.strip() == symbol]
                    if not sym_row.empty:
                        row = sym_row.iloc[0]
                        series_type = str(row.get("SERIES", "EQ")).strip()
                        if series_type == "EQ":
                            records.append({
                                "timestamp": pd.Timestamp(curr),
                                "symbol": symbol,
                                "open": float(row.get("OPEN_PRICE", row.get("CLOSE_PRICE"))),
                                "high": float(row.get("HIGH_PRICE", row.get("CLOSE_PRICE"))),
                                "low": float(row.get("LOW_PRICE", row.get("CLOSE_PRICE"))),
                                "close": float(row.get("CLOSE_PRICE")),
                                "volume": int(row.get("TTL_TRD_QNT", 0)),
                                "turnover_crores": float(row.get("TURNOVER_LACS", 0.0)) / 100.0,
                                "delivery_pct": float(row.get("DELIV_PER", 0.0)) if "DELIV_PER" in row else 0.0,
                            })
                except Exception as e:
                    logger.debug(f"Error parsing bhavcopy for {curr}: {e}")
            curr += timedelta(days=1)

        if not records:
            raise DataUnavailableException(
                f"No historical OHLCV records found for symbol '{symbol}' between {start_date} and {end_date}."
            )

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        if len(df) < min_bars:
            raise DataUnavailableException(
                f"Insufficient historical bars for '{symbol}': found {len(df)} bars < {min_bars} required."
            )

        validate_ohlcv_dataframe(df, min_bars=min_bars)

        # Save clean cache
        try:
            df.to_parquet(cache_path)
        except Exception as e:
            logger.debug(f"Error caching parquet for {symbol}: {e}")

        return df
