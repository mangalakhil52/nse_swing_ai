"""Historical NSE daily OHLCV acquisition from official UDiFF bhavcopies.

Uses NSE's public report API as the primary transport and the documented archive
URL as a fallback. A trading-day file is downloaded at most once per source.
"""
from __future__ import annotations
import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from urllib.parse import quote
from urllib.request import Request, urlopen
import httpx
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
    rows_matched: int = 0
    errors: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.requested_days = self.attempted_days = self.successful_days = 0
        self.missing_archives = self.download_failures = self.parse_failures = 0
        self.matching_days = self.rows_matched = 0
        self.errors.clear()


class NSEHistoricalOHLCVSource:
    ARCHIVE_URL_TEMPLATE = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"
    REPORT_API = "https://www.nseindia.com/api/reports"
    REPORT_NAME = "CM-UDiFF Common Bhavcopy Final (zip)"
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
        aliases = {
            "tckrsymb": "TckrSymb", "traddt": "TradDt", "opnpric": "OpnPric",
            "hghpric": "HghPric", "lwpric": "LwPric", "clspric": "ClsPric",
            "ttltradgvol": "TtlTradgVol",
        }
        return {canonical: normalized[key] for key, canonical in aliases.items() if key in normalized}

    def fetch(self, symbol: str) -> pd.DataFrame:
        data = self.fetch_many([symbol.upper()])
        out = data.get(symbol.upper())
        if out is None or out.empty:
            detail = "; ".join(self.diagnostics.errors[-5:])
            raise ValueError(f"No historical NSE OHLCV found for {symbol} through {self.as_of_date}; diagnostics={self.diagnostics}{(' errors='+detail) if detail else ''}")
        return out

    def fetch_many(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Build all requested symbols from one download per trading day."""
        self.diagnostics.reset()
        wanted = {str(s).strip().upper() for s in symbols if str(s).strip()}
        start = self.as_of_date - timedelta(days=self.lookback_calendar_days)
        self.diagnostics.requested_days = (self.as_of_date - start).days + 1
        buckets: dict[str, list[pd.DataFrame]] = {s: [] for s in wanted}
        day = start
        while day <= self.as_of_date:
            if day.weekday() < 5:
                self.diagnostics.attempted_days += 1
                try:
                    frame = self._day(day)
                    self.diagnostics.successful_days += 1
                    matched = frame[frame["symbol"].isin(wanted)]
                    if not matched.empty:
                        self.diagnostics.matching_days += 1
                        self.diagnostics.rows_matched += len(matched)
                        for symbol, group in matched.groupby("symbol"):
                            buckets[symbol].append(group.copy())
                except FileNotFoundError:
                    self.diagnostics.missing_archives += 1
                except (OSError, IOError, httpx.HTTPError) as exc:
                    self.diagnostics.download_failures += 1
                    self.diagnostics.errors.append(f"{day}: download: {exc}")
                except (ValueError, zipfile.BadZipFile) as exc:
                    self.diagnostics.parse_failures += 1
                    self.diagnostics.errors.append(f"{day}: parse: {exc}")
            day += timedelta(days=1)

        result: dict[str, pd.DataFrame] = {}
        for symbol, frames in buckets.items():
            if not frames:
                continue
            out = pd.concat(frames, ignore_index=True).sort_values("timestamp")
            out = out[out["timestamp"].dt.date <= self.as_of_date]
            if not out.empty:
                result[symbol] = out[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates("timestamp").reset_index(drop=True)
        return result

    def _day(self, day: date) -> pd.DataFrame:
        if day not in self._cache:
            self._cache[day] = self._parse(self.fetcher(day), day)
        return self._cache[day]

    def _download(self, day: date) -> bytes:
        """Primary: NSE report API; fallback: documented archive URL."""
        try:
            return self._download_report_api(day)
        except (OSError, IOError, ValueError, httpx.HTTPError):
            return self._download_archive(day)

    def _download_report_api(self, day: date) -> bytes:
        archives = [{"name": self.REPORT_NAME, "type": "daily-reports", "category": "capital-market", "section": "equities"}]
        params = {"archives": json.dumps(archives, separators=(",", ":")), "date": day.strftime("%d-%b-%Y"), "type": "equities", "mode": "single"}
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/zip,application/octet-stream,*/*", "Referer": "https://www.nseindia.com/all-reports/"}
        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True, headers=headers) as client:
            client.get("https://www.nseindia.com/", headers={"Accept": "text/html,application/xhtml+xml"})
            response = client.get(self.REPORT_API, params=params)
        if response.status_code == 404:
            raise FileNotFoundError(str(day))
        response.raise_for_status()
        content = response.content
        if content[:2] == b"PK":
            return content
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            payload = response.json()
            url = self._extract_download_url(payload)
            if url:
                with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True, headers=headers) as client:
                    download = client.get(url)
                if download.status_code == 404:
                    raise FileNotFoundError(str(day))
                download.raise_for_status()
                return download.content
        raise ValueError(f"NSE report API returned non-ZIP content for {day}: content_type={content_type}")

    @staticmethod
    def _extract_download_url(payload) -> str | None:
        if isinstance(payload, str) and payload.startswith("http"):
            return payload
        if isinstance(payload, dict):
            for key in ("url", "fileUrl", "downloadUrl", "path", "filePath"):
                value = payload.get(key)
                if isinstance(value, str) and value.startswith("http"):
                    return value
            for value in payload.values():
                found = NSEHistoricalOHLCVSource._extract_download_url(value)
                if found:
                    return found
        if isinstance(payload, list):
            for value in payload:
                found = NSEHistoricalOHLCVSource._extract_download_url(value)
                if found:
                    return found
        return None

    def _download_archive(self, day: date) -> bytes:
        url = self.ARCHIVE_URL_TEMPLATE.format(date=day.strftime("%Y%m%d"))
        req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/zip,application/octet-stream,*/*", "Referer": "https://www.nseindia.com/"})
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
        return pd.DataFrame({
            "symbol": raw[mapping["TckrSymb"]].astype(str).str.strip().str.upper(),
            "timestamp": pd.to_datetime(raw[mapping["TradDt"]], errors="coerce"),
            "open": pd.to_numeric(raw[mapping["OpnPric"]], errors="coerce"),
            "high": pd.to_numeric(raw[mapping["HghPric"]], errors="coerce"),
            "low": pd.to_numeric(raw[mapping["LwPric"]], errors="coerce"),
            "close": pd.to_numeric(raw[mapping["ClsPric"]], errors="coerce"),
            "volume": pd.to_numeric(raw[mapping["TtlTradgVol"]], errors="coerce"),
        }).dropna()
