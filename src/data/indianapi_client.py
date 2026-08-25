"""Optional IndianAPI enrichment client.

IndianAPI is treated as secondary evidence, never as the source of truth for
NSE universe membership or deterministic technical screening.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class IndianAPIConfig:
    api_key: str
    base_url: str = "https://analyst.indianapi.in"
    timeout_seconds: int = 15


class IndianAPIClient:
    def __init__(self, config: IndianAPIConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": config.api_key, "Accept": "application/json"})

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        response = self.session.get(f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}", params=params, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def stock(self, symbol: str) -> Any:
        return self._get("stock", {"name": symbol})

    def historical_stats(self, symbol: str, stats: str) -> Any:
        return self._get("historical_stats", {"stock_name": symbol, "stats": stats})

    def historical_data(self, symbol: str, period: str = "1yr", filter_name: str = "price") -> Any:
        return self._get("historical_data", {"stock_name": symbol, "period": period, "filter": filter_name})

    def forecasts(self, symbol: str, measure_code: str, period_type: str = "Annual", data_type: str = "Actuals", age: str = "Current") -> Any:
        return self._get("stock_forecasts", {"stock_id": symbol, "measure_code": measure_code, "period_type": period_type, "data_type": data_type, "age": age})

    def target_price(self, symbol: str) -> Any:
        return self._get("stock_target_price", {"stock_id": symbol})

    def recent_announcements(self, symbol: str) -> Any:
        return self._get("recent_announcements", {"stock_id": symbol})

    def corporate_actions(self, symbol: str) -> Any:
        return self._get("corporate_actions", {"stock_id": symbol})

    def close(self) -> None:
        self.session.close()
