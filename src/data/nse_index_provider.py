"""Official NSE benchmark and India VIX historical data client."""

from __future__ import annotations

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
        }
        self._session: httpx.AsyncClient | None = None

    async def _client(self) -> httpx.AsyncClient:
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(
                headers=self.headers,
                timeout=settings.REQUEST_TIMEOUT_SECONDS,
                follow_redirects=True,
            )
            try:
                r = await self._session.get(self.base_url)
                r.raise_for_status()
            except Exception as exc:
                await self._session.aclose()
                self._session = None
                raise DataUnavailableException(f"Unable to initialize NSE index session: {exc}") from exc
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.is_closed:
            await self._session.aclose()

    async def get_index_history(self, index_name: str, start_date: date, end_date: date) -> pd.DataFrame:
        client = await self._client()
        params = urlencode({
            "indexType": index_name,
            "from": start_date.strftime("%d-%m-%Y"),
            "to": end_date.strftime("%d-%m-%Y"),
        })
        url = f"{self.base_url}/api/historical/indicesHistory?{params}"
        try:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            rows = data.get("indexCloseOnlineRecords", []) if isinstance(data, dict) else data
            if not rows:
                raise DataUnavailableException(f"No NSE index observations for {index_name}")
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
                raise DataUnavailableException(f"NSE index schema missing {missing}")
            df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
            for col in required[1:]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=required).sort_values("timestamp").reset_index(drop=True)
            df["volume"] = 0.0
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except DataUnavailableException:
            raise
        except Exception as exc:
            raise DataUnavailableException(f"Unable to fetch NSE index history for {index_name}: {exc}") from exc

    async def get_india_vix_history(self, start_date: date, end_date: date) -> pd.DataFrame:
        client = await self._client()
        params = urlencode({
            "from": start_date.strftime("%d-%m-%Y"),
            "to": end_date.strftime("%d-%m-%Y"),
        })
        url = f"{self.base_url}/historicalOR/vixhistory?{params}"
        try:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("data", []) if isinstance(payload, dict) else payload
            if not rows:
                raise DataUnavailableException("No NSE India VIX observations returned")
            df = pd.DataFrame(rows).rename(columns={
                "EOD_TIMESTAMP": "timestamp",
                "EOD_OPEN_INDEX_VAL": "open",
                "EOD_HIGH_INDEX_VAL": "high",
                "EOD_LOW_INDEX_VAL": "low",
                "EOD_CLOSE_INDEX_VAL": "close",
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
            return df
        except DataUnavailableException:
            raise
        except Exception as exc:
            raise DataUnavailableException(f"Unable to fetch India VIX history: {exc}") from exc
