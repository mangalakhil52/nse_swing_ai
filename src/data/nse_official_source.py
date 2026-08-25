"""Official NSE public-file sources without an extra HTTP dependency."""
from __future__ import annotations
import io
import zipfile
from datetime import date
from urllib.request import Request, urlopen
import pandas as pd


class _RequestsCompat:
    """Small compatibility surface retained for existing tests/mocks."""
    class Session:
        def __init__(self):
            self.headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def get(self, url, timeout=None):
            req = Request(url, headers=self.headers)
            return _UrlopenResponse(urlopen(req, timeout=timeout))


class _UrlopenResponse:
    def __init__(self, response):
        self._response = response
        self.content = response.read()
    def raise_for_status(self):
        return None


requests = _RequestsCompat()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize harmless CSV header formatting differences from NSE exports."""
    out = df.copy()
    out.columns = [str(c).replace("\ufeff", "").strip().upper() for c in out.columns]
    return out


class NSEOfficialUniverseSource:
    URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

    def __init__(self, timeout_seconds: float = 20.0):
        self.timeout_seconds = timeout_seconds

    def fetch(self) -> list[dict]:
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,application/octet-stream"}
        with requests.Session() as session:
            session.headers.update(headers)
            response = session.get(self.URL, timeout=self.timeout_seconds)
            response.raise_for_status()
        df = _normalize_columns(pd.read_csv(io.BytesIO(response.content)))
        required = {"SYMBOL", "SERIES"}
        if not required.issubset(df.columns):
            raise ValueError(f"NSE equity master missing columns: {sorted(required - set(df.columns))}")
        rows = []
        for row in df.itertuples(index=False):
            if str(getattr(row, "SERIES", "")).strip().upper() in {"EQ", "BE", "BZ"}:
                symbol = str(getattr(row, "SYMBOL", "")).strip().upper()
                if symbol:
                    rows.append({"symbol": symbol, "exchange": "NSE"})
        if not rows:
            raise ValueError("NSE equity master returned no eligible equity symbols")
        return rows


class NSEOfficialBhavcopySource:
    URL_TEMPLATE = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"

    def __init__(self, as_of_date: date, timeout_seconds: float = 20.0):
        self.as_of_date = as_of_date
        self.timeout_seconds = timeout_seconds
        self._frame: pd.DataFrame | None = None

    def fetch(self, symbol: str) -> pd.DataFrame:
        if self._frame is None:
            self._frame = self._download_frame()
        df = self._frame[self._frame["symbol"].eq(symbol.upper())].copy()
        if df.empty:
            raise ValueError(f"No NSE bhavcopy row for {symbol} on {self.as_of_date}")
        return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    def _download_frame(self) -> pd.DataFrame:
        url = self.URL_TEMPLATE.format(date=self.as_of_date.strftime("%Y%m%d"))
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/zip,application/octet-stream"}
        with requests.Session() as session:
            session.headers.update(headers)
            response = session.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("NSE bhavcopy ZIP contains no CSV")
            with archive.open(csv_names[0]) as handle:
                raw = pd.read_csv(handle)
        raw = _normalize_columns(raw)
        required = {"TCKRSYMB", "TRADDT", "OPNPRIC", "HGHPric".upper(), "LWPRIC", "CLSPRIC", "TTLTRADGVOL"}
        if not required.issubset(raw.columns):
            raise ValueError(f"NSE UDiFF bhavcopy missing columns: {sorted(required - set(raw.columns))}")
        out = pd.DataFrame({
            "symbol": raw["TCKRSYMB"].astype(str).str.strip().str.upper(),
            "timestamp": pd.to_datetime(raw["TRADDT"], errors="coerce"),
            "open": pd.to_numeric(raw["OPNPRIC"], errors="coerce"),
            "high": pd.to_numeric(raw["HGHPric".upper()], errors="coerce"),
            "low": pd.to_numeric(raw["LWPRIC"], errors="coerce"),
            "close": pd.to_numeric(raw["CLSPRIC"], errors="coerce"),
            "volume": pd.to_numeric(raw["TTLTRADGVOL"], errors="coerce"),
        }).dropna()
        out = out[out["timestamp"].dt.date <= self.as_of_date]
        if out.empty:
            raise ValueError("NSE bhavcopy contains no valid point-in-time rows")
        return out
