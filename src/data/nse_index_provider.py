"""Official NSE benchmark and India VIX historical data client."""

from __future__ import annotations

import asyncio
from datetime import date
from urllib.parse import urlencode

import httpx
import pandas as pd

from config.settings import settings
from src.core.exceptions import DataUnavailableException


class NseIndexDataProvider:
    """Small isolated client for NSE's historical index/VIX endpoints."""

    def __init__(self):
        self.base_url = settings.NSE_BASE_URL.rstrip("/")
        self.headers = {
            "User-Agent": settings.USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{self.base_url}/",
            "Connection": "keep-alive",
            "DNT": "1",
        }
        self._session: httpx.AsyncClient | None = None
        self.max_attempts = 5

    async def _client(self) -> httpx.AsyncClient:
        """Bootstrap an NSE session through official pages."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(
                headers=self.headers,
                timeout=settings.REQUEST_TIMEOUT_SECONDS,
                follow_redirects=True,
            )
            bootstrap_urls = [
                f"{self.base_url}/option-chain",
                f"{self.base_url}/",
                f"{self.base_url}/reports-indices-historical-index-data",
            ]
            errors: list[str] = []
            for url in bootstrap_urls:
                for attempt in range(1, 4):
                    try:
                        response = await self._session.get(
                            url,
                            headers={**self.headers, "Referer": f"{self.base_url}/"},
                        )
                        if response.status_code < 400:
                            return self._session
                        errors.append(f"{url} -> HTTP {response.status_code}")
                    except Exception as exc:
                        errors.append(f"{url} -> {type(exc).__name__}: {exc}")
                    await asyncio.sleep(min(0.5 * attempt, 2.0))
            await self._session.aclose()
            self._session = None
            raise DataUnavailableException(
                "Unable to initialize NSE index session after official NSE bootstrap attempts: "
                + "; ".join(errors[-6:])
            )
        return self._session

    async def _get_json(self, url: str, referer: str) -> dict | list:
        """Fetch an NSE JSON endpoint and reject HTTP-200 HTML/error pages.

        NSE can intermittently return an HTML anti-bot/error page with HTTP 200.
        Treat that as transient data unavailability, retry with bounded backoff,
        and fail closed instead of passing malformed data downstream.
        """
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            client = await self._client()
            try:
                response = await client.get(
                    url,
                    headers={
                        **self.headers,
                        "Accept": "application/json,text/plain,*/*",
                        "Referer": referer,
                    },
                )
                if response.status_code >= 400:
                    raise DataUnavailableException(
                        f"NSE API HTTP {response.status_code} for {url.split('?')[0]}"
                    )

                content_type = response.headers.get("content-type", "").lower()
                body = response.text.lstrip()
                if "json" not in content_type and not body.startswith(("{", "[")):
                    preview = " ".join(body[:160].split())
                    raise DataUnavailableException(
                        "NSE returned HTTP 200 but non-JSON content "
                        f"(content-type={content_type or 'missing'}, preview={preview!r})"
                    )

                try:
                    payload = response.json()
                except ValueError as exc:
                    preview = " ".join(body[:160].split())
                    raise DataUnavailableException(
                        f"NSE returned invalid JSON (preview={preview!r})"
                    ) from exc

                if not isinstance(payload, (dict, list)):
                    raise DataUnavailableException("NSE JSON payload has an invalid top-level type")
                return payload
            except DataUnavailableException as exc:
                last_error = exc
            except httpx.HTTPError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc

            if attempt < self.max_attempts:
                await asyncio.sleep(min(1.5 * attempt, 6.0))

        raise DataUnavailableException(
            f"NSE JSON endpoint unavailable after {self.max_attempts} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _index_rows(payload: dict | list, index_name: str) -> list:
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            data = payload.get("data", {})
            if isinstance(data, dict):
                rows = data.get("indexCloseOnlineRecords", [])
                if not rows:
                    rows = data.get("indexCloseOnlineRecordsNew", [])
            elif isinstance(data, list):
                rows = data
            else:
                rows = []
        else:
            rows = []
        if not isinstance(rows, list) or not rows:
            raise DataUnavailableException(f"No NSE index observations for {index_name}")
        return rows

    @staticmethod
    def _normalise_index_rows(rows: list, name: str) -> pd.DataFrame:
        df = pd.DataFrame(rows).rename(columns={
            "EOD_TIMESTAMP": "timestamp",
            "EOD_OPEN_INDEX_VAL": "open",
            "EOD_HIGH_INDEX_VAL": "high",
            "EOD_LOW_INDEX_VAL": "low",
            "EOD_CLOSE_INDEX_VAL": "close",
        })
        required = ["timestamp", "open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataUnavailableException(f"NSE index schema missing {missing} for {name}")
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
        for col in required[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required).sort_values("timestamp").reset_index(drop=True)
        if df.empty:
            raise DataUnavailableException(f"NSE index response contained no valid rows for {name}")
        df["volume"] = 0.0
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    async def close(self) -> None:
        if self._session and not self._session.is_closed:
            await self._session.aclose()

    async def get_index_history(self, index_name: str, start_date: date, end_date: date) -> pd.DataFrame:
        params = urlencode({
            "indexType": "Index Data",
            "from_date": start_date.strftime("%d-%m-%Y"),
            "to_date": end_date.strftime("%d-%m-%Y"),
            "index": index_name.strip(),
        })
        url = f"{self.base_url}/api/historical/indicesHistory?{params}"
        referer = f"{self.base_url}/reports-indices-historical-index-data"
        payload = await self._get_json(url, referer)
        return self._normalise_index_rows(self._index_rows(payload, index_name), index_name)

    async def get_india_vix_history(self, start_date: date, end_date: date) -> pd.DataFrame:
        params = urlencode({
            "from_date": start_date.strftime("%d-%m-%Y"),
            "to_date": end_date.strftime("%d-%m-%Y"),
        })
        url = f"{self.base_url}/api/historical/vixhistory?{params}"
        referer = f"{self.base_url}/reports-indices-historical-vix"
        payload = await self._get_json(url, referer)
        if isinstance(payload, dict):
            rows = payload.get("data", [])
        else:
            rows = payload
        if not isinstance(rows, list) or not rows:
            raise DataUnavailableException("No NSE India VIX observations returned")
        return self._normalise_index_rows(rows, "INDIA VIX")
