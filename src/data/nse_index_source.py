"""Official NSE historical index data source for research benchmarks."""
from __future__ import annotations

from datetime import date
import httpx
import pandas as pd


class NSEHistoricalIndexSource:
    URL = "https://www.nseindia.com/api/historical/indicesHistory"

    def __init__(self, timeout_seconds: float = 20.0):
        self.timeout_seconds = timeout_seconds

    def fetch(self, index: str, start_date: date, end_date: date) -> pd.DataFrame:
        if start_date > end_date:
            raise ValueError("start_date must be <= end_date")
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nseindia.com/",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        params = {
            "indexType": index,
            "from": start_date.strftime("%d-%m-%Y"),
            "to": end_date.strftime("%d-%m-%Y"),
        }
        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True, headers=headers) as client:
            client.get("https://www.nseindia.com/", headers={"Accept": "text/html,application/xhtml+xml"})
            response = client.get(self.URL, params=params)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data", {}).get("indexCloseOnlineRecords", [])
        if not rows:
            raise ValueError(f"No historical index data returned for {index} between {start_date} and {end_date}")
        frame = pd.DataFrame(rows)
        date_col = "EOD_TIMESTAMP" if "EOD_TIMESTAMP" in frame.columns else "TIMESTAMP"
        close_col = "EOD_CLOSE_INDEX_VAL" if "EOD_CLOSE_INDEX_VAL" in frame.columns else "CLOSE_INDEX_VAL"
        if date_col not in frame.columns or close_col not in frame.columns:
            raise ValueError(f"Unexpected NSE index response schema: {list(frame.columns)}")
        out = pd.DataFrame({
            "timestamp": pd.to_datetime(frame[date_col], errors="coerce"),
            "close": pd.to_numeric(frame[close_col], errors="coerce"),
        }).dropna().sort_values("timestamp")
        return out.drop_duplicates("timestamp").reset_index(drop=True)
