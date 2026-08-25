"""Optional TrueData market-data client.

TrueData is a secondary market-data provider. It is deliberately disabled
unless credentials are configured, and it must not replace the official NSE
universe or deterministic EOD screening source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class TrueDataConfig:
    username: str
    password: str
    base_url: str = "https://api.truedata.in"
    timeout_seconds: int = 15


class TrueDataClient:
    """Small transport layer; endpoint-specific mapping stays isolated here."""

    def __init__(self, config: TrueDataConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = self.session.get(
            f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}",
            params=params or {},
            auth=(self.config.username, self.config.password),
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def historical(self, symbol: str, start_date: str, end_date: str, interval: str = "eod") -> Any:
        return self._get("historical", {
            "symbol": symbol,
            "startdate": start_date,
            "enddate": end_date,
            "interval": interval,
        })

    def snapshot(self, symbol: str) -> Any:
        return self._get("snapshot", {"symbol": symbol})

    def close(self) -> None:
        self.session.close()
