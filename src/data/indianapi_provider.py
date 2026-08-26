"""IndianAPI adapter for fundamental, news, corporate and IPO intelligence.

IndianAPI is intentionally an intelligence provider, not the authoritative EOD
price source. Every response is retained with source and retrieval timestamp;
callers must apply the point-in-time filter before using it for a decision.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from config.settings import settings
from src.core.exceptions import DataUnavailableException


class IndianAPIProvider:
    """Strict adapter around the documented Indian Stock Market API."""

    def __init__(self) -> None:
        if not settings.INDIANAPI_API_KEY:
            raise DataUnavailableException("INDIANAPI_API_KEY is not configured")
        self.base_url = settings.INDIANAPI_BASE_URL.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _client_for_request(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=settings.INDIANAPI_TIMEOUT_SECONDS,
                headers={"X-Api-Key": settings.INDIANAPI_API_KEY, "Accept": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
        client = await self._client_for_request()
        response = await client.get(f"{self.base_url}/{endpoint.lstrip('/')}", params=params)
        if response.status_code != 200:
            raise DataUnavailableException(
                f"INDIANAPI_HTTP_{response.status_code}: {response.text[:300]}"
            )
        payload = response.json()
        if payload is None or payload == {} or payload == []:
            raise DataUnavailableException(f"INDIANAPI_EMPTY_RESPONSE: {endpoint}")
        return payload

    async def get_stock_bundle(self, symbol: str) -> dict[str, Any]:
        """Return the documented /stock bundle for one NSE symbol."""
        payload = await self._get("stock", {"name": symbol.upper().strip()})
        if not isinstance(payload, dict):
            raise DataUnavailableException(f"INDIANAPI_INVALID_STOCK_RESPONSE: {symbol}")
        return {"source": "INDIANAPI", "retrieved_at": datetime.now(timezone.utc).isoformat(), "data": payload}

    async def get_news(self, symbol: str) -> dict[str, Any] | list[Any]:
        """Return documented company/market news payload."""
        # The public documentation exposes /news but does not guarantee a stable
        # symbol parameter in the published schema; pass symbol only when supplied.
        return await self._get("news", {"stock_name": symbol.upper().strip()})

    async def get_recent_announcements(self) -> dict[str, Any] | list[Any]:
        return await self._get("recent_announcements")

    async def get_corporate_actions(self, symbol: str) -> dict[str, Any] | list[Any]:
        return await self._get("corporate_actions", {"stock_name": symbol.upper().strip()})

    async def get_ipo(self) -> dict[str, Any] | list[Any]:
        return await self._get("ipo")

    async def get_historical_data(self, symbol: str, period: str = "5yr", data_filter: str = "price") -> dict[str, Any] | list[Any]:
        return await self._get(
            "historical_data",
            {"stock_name": symbol.upper().strip(), "period": period, "filter": data_filter},
        )

    async def get_historical_stats(self, symbol: str, stats: str) -> dict[str, Any] | list[Any]:
        return await self._get(
            "historical_stats",
            {"stock_name": symbol.upper().strip(), "stats": stats},
        )

    async def get_trending(self) -> dict[str, Any] | list[Any]:
        return await self._get("trending")
