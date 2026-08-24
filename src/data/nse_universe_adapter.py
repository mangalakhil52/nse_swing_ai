"""#14M NSE universe adapter.

Fetches a security-master CSV from a configured source, validates its schema,
and delegates normalization/PIT filtering to MarketUniverseService.
"""
from __future__ import annotations
import csv
import io
import time
from dataclasses import dataclass
from datetime import date
from urllib.request import Request, urlopen
from src.data.market_universe import MarketUniverseService, UniverseSymbol


@dataclass(frozen=True)
class UniverseSnapshot:
    symbols: list[UniverseSymbol]
    source_url: str
    fetched_at: date


class NSEUniverseAdapter:
    REQUIRED_COLUMNS = {"symbol"}

    def __init__(self, source_url: str, timeout_seconds: int = 20, retries: int = 3):
        if not source_url:
            raise ValueError("source_url is required")
        self.source_url = source_url
        self.timeout_seconds = timeout_seconds
        self.retries = max(1, retries)

    def fetch(self, as_of_date: date | None = None) -> UniverseSnapshot:
        payload = self._download()
        rows = list(csv.DictReader(io.StringIO(payload)))
        if not rows:
            raise ValueError("NSE universe source returned no rows")
        columns = {str(c).strip().lower() for c in rows[0].keys() if c is not None}
        if not self.REQUIRED_COLUMNS.issubset(columns):
            raise ValueError(f"Universe source missing required columns: {self.REQUIRED_COLUMNS - columns}")
        normalized_rows = []
        for row in rows:
            clean = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
            normalized_rows.append({"symbol": clean.get("symbol", ""), "exchange": clean.get("exchange", "NSE")})
        symbols = MarketUniverseService.normalize(normalized_rows, as_of_date)
        if not symbols:
            raise ValueError("NSE universe source produced no valid NSE symbols")
        return UniverseSnapshot(symbols=symbols, source_url=self.source_url, fetched_at=date.today())

    def _download(self) -> str:
        last_error = None
        for attempt in range(self.retries):
            try:
                req = Request(self.source_url, headers={"User-Agent": "nse-swing-ai/1.0"})
                with urlopen(req, timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8-sig")
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Unable to fetch NSE universe after {self.retries} attempts: {last_error}") from last_error
