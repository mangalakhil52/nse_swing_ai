"""#14N source-agnostic OHLCV acquisition and PIT-safe normalization."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import time
import pandas as pd


@dataclass(frozen=True)
class MarketDataSnapshot:
    symbol: str
    as_of_date: date
    frame: pd.DataFrame
    source: str


class MarketDataAdapter:
    REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}

    def __init__(self, fetcher, source="configured-market-data", retries=3, backoff_seconds=1):
        self.fetcher = fetcher
        self.source = source
        self.retries = max(1, retries)
        self.backoff_seconds = max(0, backoff_seconds)

    def fetch(self, symbol: str, as_of_date: date) -> MarketDataSnapshot:
        if not symbol or not as_of_date:
            raise ValueError("symbol and as_of_date are required")
        last_error = None
        for attempt in range(self.retries):
            try:
                raw = self.fetcher(symbol)
                frame = self._normalize(raw, as_of_date)
                return MarketDataSnapshot(symbol=symbol, as_of_date=as_of_date, frame=frame, source=self.source)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff_seconds * (2 ** attempt))
        raise RuntimeError(f"Unable to fetch market data for {symbol} after {self.retries} attempts: {last_error}") from last_error

    def _normalize(self, raw, as_of_date: date) -> pd.DataFrame:
        if not isinstance(raw, pd.DataFrame):
            raise ValueError("market data fetcher must return a pandas DataFrame")
        frame = raw.copy()
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        missing = self.REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Market data missing required columns: {sorted(missing)}")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=list(self.REQUIRED_COLUMNS))
        frame = frame[frame["timestamp"].dt.date <= as_of_date]
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        if frame.empty:
            raise ValueError("No valid point-in-time market data available")
        return frame
