"""
Market Data Validation & Integrity Engine Module.
Enforces zero-trust data quality controls on raw OHLCV bars before ingestion or quantitative calculations.
Detects missing candles, broken OHLC geometry, split anomalies, zero volume, and timestamp staleness.
"""

from datetime import date, datetime, timedelta
import logging
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.exceptions import DataIntegrityError
from src.core.types import DataFreshness

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    is_valid: bool
    symbol: str
    bars_checked: int
    freshness: DataFreshness
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    latest_bar_date: date | None = None


class DataValidator:
    """Rigorous validator for financial time-series data."""

    def __init__(self, min_required_bars: int = 60, max_staleness_days: int = 4):
        self.min_required_bars = min_required_bars
        self.max_staleness_days = max_staleness_days

    def validate_ohlcv_dataframe(self, df: pd.DataFrame, symbol: str) -> ValidationResult:
        """
        Runs comprehensive integrity checks on an OHLCV DataFrame.
        """
        symbol = symbol.upper().strip()
        errors: list[str] = []
        warnings: list[str] = []

        if df is None or df.empty:
            return ValidationResult(
                is_valid=False,
                symbol=symbol,
                bars_checked=0,
                freshness=DataFreshness.UNKNOWN,
                errors=["DataFrame is empty or None."],
            )

        # 1. Check required columns
        required_cols = {"open", "high", "low", "close", "volume"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            return ValidationResult(
                is_valid=False,
                symbol=symbol,
                bars_checked=len(df),
                freshness=DataFreshness.UNKNOWN,
                errors=[f"Missing required columns: {missing_cols}"],
            )

        # 2. Check sufficient history
        if len(df) < self.min_required_bars:
            errors.append(
                f"Insufficient historical bars: {len(df)} available, minimum {self.min_required_bars} required."
            )

        # 3. Check for null or NaN values
        for col in ["open", "high", "low", "close", "volume"]:
            if df[col].isnull().any():
                errors.append(f"Column '{col}' contains {df[col].isnull().sum()} null/NaN values.")

        # 4. Check OHLC price geometry
        # High must be >= Open, Close, Low
        invalid_high = df[
            (df["high"] < df["open"] - 1e-4)
            | (df["high"] < df["close"] - 1e-4)
            | (df["high"] < df["low"] - 1e-4)
        ]
        if not invalid_high.empty:
            errors.append(
                f"Found {len(invalid_high)} bars where High < max(Open, Close, Low)."
            )

        # Low must be <= Open, Close, High
        invalid_low = df[
            (df["low"] > df["open"] + 1e-4)
            | (df["low"] > df["close"] + 1e-4)
            | (df["low"] > df["high"] + 1e-4)
        ]
        if not invalid_low.empty:
            errors.append(
                f"Found {len(invalid_low)} bars where Low > min(Open, Close, High)."
            )

        # Prices must be strictly positive
        non_positive_prices = df[
            (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
        ]
        if not non_positive_prices.empty:
            errors.append(f"Found {len(non_positive_prices)} bars with non-positive prices.")

        # Volume must be non-negative
        negative_vol = df[df["volume"] < 0]
        if not negative_vol.empty:
            errors.append(f"Found {len(negative_vol)} bars with negative volume.")

        # Check for abnormal consecutive zero volume (illiquid/suspended)
        zero_vol_count = (df["volume"] == 0).sum()
        if zero_vol_count > 5:
            errors.append(f"Stock has {zero_vol_count} bars with zero volume (illiquid).")
        elif zero_vol_count > 0:
            warnings.append(f"Stock has {zero_vol_count} isolated bars with zero volume.")

        # 5. Check for unadjusted stock splits or crazy price spikes (> 60% overnight move)
        if len(df) > 1:
            close_arr = df["close"].values
            pct_changes = np.abs(np.diff(close_arr) / close_arr[:-1])
            spike_indices = np.where(pct_changes > 0.60)[0]
            if len(spike_indices) > 0:
                warnings.append(
                    f"Found {len(spike_indices)} bars with >60% price jump. Possible unadjusted split or anomaly."
                )

        # 6. Timestamp Freshness Classification
        latest_date: date | None = None
        freshness = DataFreshness.UNKNOWN

        if "timestamp" in df.columns:
            ts_series = pd.to_datetime(df["timestamp"])
            latest_ts = ts_series.max()
            latest_date = latest_ts.date() if hasattr(latest_ts, "date") else latest_ts

            days_diff = (datetime.utcnow().date() - latest_date).days
            if days_diff <= 1:
                freshness = DataFreshness.LIVE
            elif days_diff <= 2:
                freshness = DataFreshness.RECENT
            elif days_diff <= self.max_staleness_days:
                freshness = DataFreshness.DELAYED
            else:
                freshness = DataFreshness.STALE
                errors.append(f"Data is STALE: latest bar is from {latest_date} ({days_diff} days old).")

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            symbol=symbol,
            bars_checked=len(df),
            freshness=freshness,
            errors=errors,
            warnings=warnings,
            latest_bar_date=latest_date,
        )

    def enforce_valid_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validates the dataframe and raises DataIntegrityError if invalid.
        Returns cleaned and sorted DataFrame if valid.
        """
        res = self.validate_ohlcv_dataframe(df, symbol)
        if not res.is_valid:
            err_msg = f"Data validation failed for {symbol}: " + "; ".join(res.errors)
            logger.error(err_msg)
            raise DataIntegrityError(err_msg)

        if "timestamp" in df.columns:
            df = df.sort_values(by="timestamp").reset_index(drop=True)

        return df
