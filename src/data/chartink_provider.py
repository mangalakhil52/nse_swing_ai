"""
Chartink Custom Scanner API Provider Module.
Interfaces directly with Chartink's screener engine (chartink.com/screener/process) for real-time technical condition scans.
Handles CSRF token extraction, custom query execution, retries, and rate limiting.
"""

import asyncio
import logging
import re
from typing import Any
from bs4 import BeautifulSoup
import httpx

from config.settings import settings
from src.data.base import ChartinkScannerProvider

logger = logging.getLogger(__name__)


# Standard Pre-Built Chartink Scan Clauses for High-Probability Swing Setups
PREBUILT_SCAN_CLAUSES = {
    "CONSOLIDATION_BREAKOUT_SURGE": (
        "( {33489} ( ( ( [0] 15 minute volume > [0] 15 minute sma( volume,20 ) * 2 ) "
        "and ( [0] 15 minute close > [-1] 15 minute high ) ) "
        "and ( [0] daily close > [0] daily open ) ) )"
    ),
    "VCP_COMPRESSION_DAILY": (
        "( {33489} ( ( [0] daily close > [0] daily ema( close,20 ) ) "
        "and ( [0] daily ema( close,20 ) > [0] daily ema( close,50 ) ) "
        "and ( [0] daily ema( close,50 ) > [0] daily ema( close,200 ) ) "
        "and ( [0] daily atr( 14 ) / [0] daily close * 100 < 4.5 ) "
        "and ( [0] daily close >= [0] daily max( 20, [0] daily close ) * 0.96 ) ) )"
    ),
    "BULLISH_EMA_PULLBACK_DAILY": (
        "( {33489} ( ( [0] daily close > [0] daily ema( close,20 ) ) "
        "and ( [0] daily low <= [0] daily ema( close,20 ) * 1.01 ) "
        "and ( [0] daily close > [0] daily open ) "
        "and ( [0] daily ema( close,20 ) > [0] daily ema( close,50 ) ) ) )"
    ),
    "RELATIVE_STRENGTH_OUTPERFORMER": (
        "( {33489} ( ( [0] daily close / [0] daily max( 250, [0] daily high ) >= 0.85 ) "
        "and ( [0] daily volume > 100000 ) "
        "and ( [0] daily close > 50 ) "
        "and ( [0] daily rsi( 14 ) >= 58 ) ) )"
    ),
}


class ChartinkProvider(ChartinkScannerProvider):
    """Client for executing queries against Chartink screener engine."""

    def __init__(self):
        self.base_url = "https://chartink.com"
        self.screener_url = f"{self.base_url}/screener/process"
        self.headers = {
            "User-Agent": settings.USER_AGENT,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.9",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": self.base_url,
            "Referer": f"{self.base_url}/screener/",
        }
        self._csrf_token: str | None = None
        self._cookies: dict[str, str] = {}
        self._session: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Initializes client and extracts CSRF token from Chartink."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(
                headers=self.headers,
                timeout=settings.REQUEST_TIMEOUT_SECONDS,
                follow_redirects=True,
            )

        if not self._csrf_token:
            await self._refresh_csrf_token()

        return self._session

    async def _refresh_csrf_token(self) -> None:
        """Scrapes the latest CSRF token and session cookies from Chartink."""
        if self._session is None:
            self._session = httpx.AsyncClient(headers=self.headers, timeout=settings.REQUEST_TIMEOUT_SECONDS)

        try:
            resp = await self._session.get(f"{self.base_url}/screener/")
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                meta_csrf = soup.find("meta", attrs={"name": "csrf-token"})
                if meta_csrf and meta_csrf.get("content"):
                    self._csrf_token = str(meta_csrf["content"])
                    self._session.headers.update({"X-CSRF-TOKEN": self._csrf_token})
                    logger.debug("Successfully acquired Chartink CSRF token.")
                    return

                # Fallback: search regex in text
                match = re.search(r'name="csrf-token"\s+content="([^"]+)"', resp.text)
                if match:
                    self._csrf_token = match.group(1)
                    self._session.headers.update({"X-CSRF-TOKEN": self._csrf_token})
                    return
        except Exception as e:
            logger.warning(f"Failed to acquire Chartink CSRF token: {e}")

    async def run_scanner_query(self, scan_clause: str) -> list[str]:
        """
        Executes a technical scan query against Chartink and returns a list of matching NSE symbols.
        """
        client = await self._get_client()
        payload = {"scan_clause": scan_clause}

        for attempt in range(3):
            try:
                resp = await client.post(self.screener_url, data=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("data", [])
                    symbols = [
                        item.get("nsecode", item.get("stock_name", "")).strip().upper()
                        for item in results
                        if item.get("nsecode") or item.get("stock_name")
                    ]
                    logger.info(f"Chartink query matched {len(symbols)} symbols.")
                    return [s for s in symbols if s]
                elif resp.status_code in [419, 403]:
                    # CSRF expired, refresh token and retry
                    logger.debug("Chartink CSRF token expired. Refreshing...")
                    await self._refresh_csrf_token()
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for Chartink query: {e}")
                await asyncio.sleep(1.0)

        logger.error("All attempts to query Chartink API failed.")
        return []

    async def run_prebuilt_scan(self, scan_name: str) -> list[str]:
        """Runs one of the pre-configured high-probability swing scans."""
        clause = PREBUILT_SCAN_CLAUSES.get(scan_name)
        if not clause:
            raise ValueError(f"Unknown prebuilt scan name: {scan_name}. Available: {list(PREBUILT_SCAN_CLAUSES.keys())}")
        return await self.run_scanner_query(clause)

    async def close(self) -> None:
        """Closes HTTP session."""
        if self._session and not self._session.is_closed:
            await self._session.aclose()
