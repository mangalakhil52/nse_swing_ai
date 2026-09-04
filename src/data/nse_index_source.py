"""Official NSE historical index data source with CI-safe session handling."""
from __future__ import annotations

from datetime import date
import time

import httpx
import pandas as pd


class NSEHistoricalIndexSource:
    """Fetch verified historical NSE index observations with session resilience."""

    API_URL = "https://www.nseindia.com/api/historical/indicesHistory"
    HOME_URL = "https://www.nseindia.com/option-chain"
    REFERER = "https://www.nseindia.com/reports-indices-historical-index-data"

    def __init__(self, timeout_seconds: float = 30.0, max_retries: int = 5):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def fetch(self, index: str, start_date: date, end_date: date) -> pd.DataFrame:
        if start_date > end_date:
            raise ValueError("start_date must be <= end_date")
        if not index or not index.strip():
            raise ValueError("index must be non-empty")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/151.0.0.0 Safari/537.36"
            ),
            "Referer": self.REFERER,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

        # Keep the exact NSE API parameter schema used by the working provider.
        # `indexType` is the report type; the actual index name is supplied as
        # `index`. `from_date` / `to_date` are the current endpoint names.
        params = {
            "indexType": "Index Data",
            "from_date": start_date.strftime("%d-%m-%Y"),
            "to_date": end_date.strftime("%d-%m-%Y"),
            "index": index.strip(),
        }

        last_error: Exception | None = None

        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers=headers,
        ) as client:
            for attempt in range(self.max_retries):
                try:
                    bootstrap = client.get(self.HOME_URL)
                    if bootstrap.status_code >= 400:
                        raise RuntimeError(
                            f"NSE session bootstrap failed with HTTP {bootstrap.status_code}"
                        )

                    response = client.get(self.API_URL, params=params)
                    response.raise_for_status()

                    content_type = response.headers.get("content-type", "").lower()
                    if "json" not in content_type:
                        preview = response.text[:240].replace("\n", " ").strip()
                        raise RuntimeError(
                            "NSE historical index returned non-JSON content "
                            f"(content-type={content_type!r}, preview={preview!r})"
                        )

                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise ValueError("Unexpected NSE historical index payload type")

                    data = payload.get("data", [])
                    if isinstance(data, dict):
                        rows = data.get("indexCloseOnlineRecords", [])
                    else:
                        rows = data

                    if not isinstance(rows, list) or not rows:
                        raise ValueError(
                            f"No historical index data returned for {index} "
                            f"between {start_date} and {end_date}"
                        )

                    frame = pd.DataFrame(rows)
                    date_col = next(
                        (c for c in ("EOD_TIMESTAMP", "TIMESTAMP") if c in frame.columns),
                        None,
                    )
                    close_col = next(
                        (
                            c
                            for c in ("EOD_CLOSE_INDEX_VAL", "CLOSE_INDEX_VAL")
                            if c in frame.columns
                        ),
                        None,
                    )
                    if date_col is None or close_col is None:
                        raise ValueError(
                            "Unexpected NSE index response schema: "
                            f"{list(frame.columns)}"
                        )

                    out = pd.DataFrame(
                        {
                            "timestamp": pd.to_datetime(
                                frame[date_col], dayfirst=True, errors="coerce"
                            ),
                            "close": pd.to_numeric(
                                frame[close_col]
                                .astype("string")
                                .str.replace(",", "", regex=False),
                                errors="coerce",
                            ),
                        }
                    ).dropna(subset=["timestamp", "close"])

                    out = (
                        out.sort_values("timestamp")
                        .drop_duplicates("timestamp")
                        .reset_index(drop=True)
                    )
                    if out.empty:
                        raise ValueError(
                            f"NSE returned no valid observations for {index} "
                            f"between {start_date} and {end_date}"
                        )
                    return out

                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < self.max_retries:
                        time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(
            f"Unable to retrieve verified NSE index data for {index} "
            f"between {start_date} and {end_date} after {self.max_retries} attempts"
        ) from last_error
