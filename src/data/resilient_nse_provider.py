"""CI/runtime-resilient NSE provider session bootstrap.

The base provider remains the source of truth for data parsing and endpoints.
This subclass only hardens session initialization against transient NSE 403/
challenge responses seen on hosted runners.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from config.settings import settings
from src.core.exceptions import DataUnavailableException
from src.data.nse_provider import NseDataProvider

logger = logging.getLogger(__name__)


class ResilientNseDataProvider(NseDataProvider):
    """NSE provider with browser-like session bootstrap and bounded retries."""

    async def _get_client(self) -> httpx.AsyncClient:
        if self._session is not None and not self._session.is_closed:
            return self._session

        headers = {
            **self.headers,
            "User-Agent": settings.USER_AGENT,
            "Referer": f"{self.base_url}/option-chain",
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
        }
        client = httpx.AsyncClient(
            headers=headers,
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        bootstrap_urls = (
            f"{self.base_url}/option-chain",
            f"{self.base_url}/market-data/live-equity-market",
            f"{self.base_url}/",
        )
        last_error: Exception | None = None

        for attempt in range(3):
            for bootstrap_url in bootstrap_urls:
                try:
                    response = await client.get(
                        bootstrap_url,
                        headers={**headers, "Referer": f"{self.base_url}/"},
                    )
                    if response.status_code < 400:
                        self._session = client
                        logger.debug("Initialized NSE session via %s", bootstrap_url)
                        return client
                    last_error = RuntimeError(
                        f"NSE bootstrap HTTP {response.status_code} at {bootstrap_url}"
                    )
                except Exception as exc:
                    last_error = exc
            if attempt < 2:
                await asyncio.sleep(1.5 * (attempt + 1))

        await client.aclose()
        raise DataUnavailableException(
            f"Unable to initialize NSE session after bounded retries: {last_error}"
        ) from last_error
