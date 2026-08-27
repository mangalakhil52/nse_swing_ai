"""Official NSE historical index data source with CI-safe session handling."""
from __future__ import annotations

from datetime import date
import time
import httpx
import pandas as pd


class NSEHistoricalIndexSource:
    # NSE's historical-index UI is backed by this route. The older
    # /api/historical/indicesHistory route can return an HTML challenge on
    # hosted runners even after a homepage request.
    URL = "https://www.nseindia.com/historicalOR/indicesHistory"
    HOME_URL = "https://www.nseindia.com/option-chain"

    def __init__(self, timeout_seconds: float = 30.0, max_retries: int = 4):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def fetch(self, index: str, start_date: date, end_date: date) -> pd.DataFrame:
        if start_date > end_date:
            raise ValueError("start_date must be <= end_date")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:118.0) Gecko/20100101 Firefox/118.0",
            "Referer": "https://www.nseindia.com/reports-indices-historical-index-data",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        }
        params = {
            "indexType": index,
            "fromDate": start_date.strftime("%d-%m-%Y"),
            "toDate": end_date.strftime("%d-%m-%Y"),
        }
        last_error: Exception | None = None

        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True, headers=headers) as client:
            for attempt in range(self.max_retries):
                try:
                    # NSE commonly issues the API cookies from this page. Keep
                    # the same session for the subsequent historical request.
                    bootstrap = client.get(self.HOME_URL)
                    bootstrap.raise_for_status()
                    response = client.get(self.URL, params=params)
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if "json" not in content_type:
                        preview = response.text[:160].replace("\n", " ")
                        raise RuntimeError(f"NSE historical index returned non-JSON content: {preview}")

                    payload = response.json()
                    rows = payload.get("data", {}) if isinstance(payload, dict) else {}
                    if isinstance(rows, dict):
                        rows = rows.get("indexCloseOnlineRecords", [])
                    if not isinstance(rows, list) or not rows:
                        raise ValueError(
                            f"No historical index data returned for {index} between {start_date} and {end_date}"
                        )

                    frame = pd.DataFrame(rows)
                    date_col = "EOD_TIMESTAMP" if "EOD_TIMESTAMP" in frame.columns else "TIMESTAMP"
                    close_col = "EOD_CLOSE_INDEX_VAL" if "EOD_CLOSE_INDEX_VAL" in frame.columns else "CLOSE_INDEX_VAL"
                    if date_col not in frame.columns or close_col not in frame.columns:
                        raise ValueError(f"Unexpected NSE index response schema: {list(frame.columns)}")

                    out = pd.DataFrame({
                        "timestamp": pd.to_datetime(frame[date_col], errors="coerce"),
                        "close": pd.to_numeric(
                            frame[close_col].astype(str).str.replace(",", "", regex=False), errors="coerce"
                        ),
                    }).dropna().sort_values("timestamp")
                    return out.drop_duplicates("timestamp").reset_index(drop=True)
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < self.max_retries:
                        time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"Unable to retrieve verified NSE index data for {index}") from last_error
