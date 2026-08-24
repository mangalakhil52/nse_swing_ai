"""Concrete HTTP market-data source for #14N.

The source URL is configuration-driven. It deliberately does not hardcode a
vendor endpoint; the adapter remains reusable with NSE-compatible providers.
"""
from __future__ import annotations
import io
import pandas as pd
import httpx


class HTTPMarketDataSource:
    def __init__(self, base_url: str, timeout_seconds: float = 20.0):
        if not base_url:
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def fetch(self, symbol: str) -> pd.DataFrame:
        response = httpx.get(
            f"{self.base_url}/{symbol}.csv",
            timeout=self.timeout_seconds,
            headers={"User-Agent": "nse-swing-ai/1.0"},
        )
        response.raise_for_status()
        try:
            return pd.read_csv(io.StringIO(response.text))
        except Exception as exc:
            raise ValueError(f"Market-data source returned invalid CSV for {symbol}") from exc
