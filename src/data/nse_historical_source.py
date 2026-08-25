"""Historical NSE daily OHLCV acquisition from official UDiFF bhavcopies."""
from __future__ import annotations
import io
import zipfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from urllib.request import Request, urlopen
import pandas as pd


@dataclass
class HistoricalFetchDiagnostics:
    requested_days: int = 0
    attempted_days: int = 0
    successful_days: int = 0
    missing_archives: int = 0
    download_failures: int = 0
    parse_failures: int = 0
    matching_days: int = 0
    errors: list[str] = field(default_factory=list)


class NSEHistoricalOHLCVSource:
    URL_TEMPLATE = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"
    REQUIRED = {"TckrSymb", "TradDt", "OpnPric", "HghPric", "LwPric", "ClsPric", "TtlTradgVol"}

    def __init__(self, as_of_date: date, lookback_calendar_days: int = 140, timeout_seconds: float = 20.0, fetcher=None):
        self.as_of_date = as_of_date
        self.lookback_calendar_days = max(1, lookback_calendar_days)
        self.timeout_seconds = timeout_seconds
        self.fetcher = fetcher or self._download
        self._cache: dict[date, pd.DataFrame] = {}
        self.diagnostics = HistoricalFetchDiagnostics()

    @staticmethod
    def _normalize_columns(columns) -> dict[str, str]:
        normalized = {str(col).replace("\ufeff", "").strip().lower(): col for col in columns}
        aliases = {"tckrsymb":"TckrSymb", "traddt":"TradDt", "opnpric":"OpnPric", "hghpric":"HghPric", "lwpric":"LwPric", "clspric":"ClsPric", "ttltradgvol":"TtlTradgVol"}
        return {canonical: normalized[key] for key, canonical in aliases.items() if key in normalized}

    def fetch(self, symbol: str) -> pd.DataFrame:
        start = self.as_of_date - timedelta(days=self.lookback_calendar_days)
        days = (self.as_of_date - start).days + 1
        self.diagnostics.requested_days = days
        frames = []
        day = start
        while day <= self.as_of_date:
            if day.weekday() < 5:
                self.diagnostics.attempted_days += 1
                try:
                    frame = self._day(day)
                    self.diagnostics.successful_days += 1
                    row = frame[frame["symbol"].eq(symbol.upper())]
                    if not row.empty:
                        self.diagnostics.matching_days += 1
                        frames.append(row)
                except FileNotFoundError:
                    self.diagnostics.missing_archives += 1
                except (OSError, IOError) as exc:
                    self.diagnostics.download_failures += 1
                    self.diagnostics.errors.append(f"{day}: download: {exc}")
                except (ValueError, zipfile.BadZipFile) as exc:
                    self.diagnostics.parse_failures += 1
                    self.diagnostics.errors.append(f"{day}: parse: {exc}")
            day += timedelta(days=1)
        if not frames:
            detail = "; ".join(self.diagnostics.errors[-3:])
            raise ValueError(f"No historical NSE OHLCV found for {symbol} through {self.as_of_date}; diagnostics={self.diagnostics}{(' errors='+detail) if detail else ''}")
        out = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        out = out[out["timestamp"].dt.date <= self.as_of_date]
        if out.empty:
            raise ValueError(f"No point-in-time NSE OHLCV found for {symbol} through {self.as_of_date}")
        return out[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates("timestamp").reset_index(drop=True)

    def _day(self, day: date) -> pd.DataFrame:
        if day not in self._cache:
            self._cache[day] = self._parse(self.fetcher(day), day)
        return self._cache[day]

    def _download(self, day: date) -> bytes:
        url = self.URL_TEMPLATE.format(date=day.strftime("%Y%m%d"))
        req = Request(url, headers={"User-Agent":"Mozilla/5.0", "Accept":"application/zip,application/octet-stream,*/*", "Referer":"https://www.nseindia.com/"})
        with urlopen(req, timeout=self.timeout_seconds) as response:
            return response.read()

    def _parse(self, payload: bytes, day: date) -> pd.DataFrame:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"No CSV in NSE bhavcopy for {day}")
            with archive.open(names[0]) as handle:
                raw = pd.read_csv(handle)
        mapping = self._normalize_columns(raw.columns)
        missing = self.REQUIRED - set(mapping)
        if missing:
            raise ValueError(f"NSE bhavcopy missing columns: {sorted(missing)}; received={list(raw.columns)}")
        return pd.DataFrame({"symbol":raw[mapping["TckrSymb"]].astype(str).str.strip().str.upper(),"timestamp":pd.to_datetime(raw[mapping["TradDt"]],errors="coerce"),"open":pd.to_numeric(raw[mapping["OpnPric"]],errors="coerce"),"high":pd.to_numeric(raw[mapping["HghPric"]],errors="coerce"),"low":pd.to_numeric(raw[mapping["LwPric"]],errors="coerce"),"close":pd.to_numeric(raw[mapping["ClsPric"]],errors="coerce"),"volume":pd.to_numeric(raw[mapping["TtlTradgVol"]],errors="coerce")}).dropna()
