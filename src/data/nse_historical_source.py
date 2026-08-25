"""Historical NSE daily OHLCV acquisition from official UDiFF bhavcopies.

Downloads only the requested date range, caches each trading-day archive in
memory, and exposes per-symbol history suitable for PIT-safe screening.
"""
from __future__ import annotations
import io
import zipfile
from datetime import date, timedelta
from urllib.request import Request, urlopen
import pandas as pd


class NSEHistoricalOHLCVSource:
    URL_TEMPLATE = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"
    REQUIRED = {"TckrSymb", "TradDt", "OpnPric", "HghPric", "LwPric", "ClsPric", "TtlTradgVol"}

    def __init__(self, as_of_date: date, lookback_calendar_days: int = 140, timeout_seconds: float = 20.0, fetcher=None):
        self.as_of_date = as_of_date
        self.lookback_calendar_days = max(1, lookback_calendar_days)
        self.timeout_seconds = timeout_seconds
        self.fetcher = fetcher or self._download
        self._cache: dict[date, pd.DataFrame] = {}

    def fetch(self, symbol: str) -> pd.DataFrame:
        start = self.as_of_date - timedelta(days=self.lookback_calendar_days)
        frames = []
        day = start
        while day <= self.as_of_date:
            if day.weekday() < 5:
                try:
                    frame = self._day(day)
                    row = frame[frame["symbol"].eq(symbol.upper())]
                    if not row.empty:
                        frames.append(row)
                except (OSError, IOError, ValueError, zipfile.BadZipFile):
                    # Missing archive can be a weekend/holiday or an unavailable date;
                    # continue so one missing day does not destroy the history window.
                    pass
            day += timedelta(days=1)
        if not frames:
            raise ValueError(f"No historical NSE OHLCV found for {symbol} through {self.as_of_date}")
        out = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        out = out[out["timestamp"].dt.date <= self.as_of_date]
        return out[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates("timestamp").reset_index(drop=True)

    def _day(self, day: date) -> pd.DataFrame:
        if day not in self._cache:
            raw = self.fetcher(day)
            self._cache[day] = self._parse(raw, day)
        return self._cache[day]

    def _download(self, day: date) -> bytes:
        url = self.URL_TEMPLATE.format(date=day.strftime("%Y%m%d"))
        req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/zip,application/octet-stream"})
        with urlopen(req, timeout=self.timeout_seconds) as response:
            return response.read()

    def _parse(self, payload: bytes, day: date) -> pd.DataFrame:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"No CSV in NSE bhavcopy for {day}")
            with archive.open(names[0]) as handle:
                raw = pd.read_csv(handle)
        missing = self.REQUIRED - set(raw.columns)
        if missing:
            raise ValueError(f"NSE bhavcopy missing columns: {sorted(missing)}")
        return pd.DataFrame({
            "symbol": raw["TckrSymb"].astype(str).str.strip().str.upper(),
            "timestamp": pd.to_datetime(raw["TradDt"], errors="coerce"),
            "open": pd.to_numeric(raw["OpnPric"], errors="coerce"),
            "high": pd.to_numeric(raw["HghPric"], errors="coerce"),
            "low": pd.to_numeric(raw["LwPric"], errors="coerce"),
            "close": pd.to_numeric(raw["ClsPric"], errors="coerce"),
            "volume": pd.to_numeric(raw["TtlTradgVol"], errors="coerce"),
        }).dropna()
