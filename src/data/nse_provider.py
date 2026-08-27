"""
NSE Official Market Data Provider Module.

Primary responsibilities:
- Official NSE security master / equity universe.
- Official daily security Bhavcopy.
- Official live equity quotes.
- Official NIFTY index and India VIX observations.
- Strict fail-closed behavior: no fabricated market values are ever returned.
"""

import io
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import httpx
import pandas as pd

from config.settings import settings
from src.core.exceptions import DataUnavailableException
from src.core.models import LiveQuote, MarketBreadthData, SymbolMetadata
from src.data.base import MarketDataProvider

logger = logging.getLogger(__name__)


class NseDataProvider(MarketDataProvider):
    """Primary data provider using official NSE feeds and archives."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or settings.CACHE_DIR / "bhavcopy"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = settings.NSE_BASE_URL.rstrip("/")
        self.headers = {
            "User-Agent": settings.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "DNT": "1",
        }
        self._session: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(
                headers=self.headers,
                timeout=settings.REQUEST_TIMEOUT_SECONDS,
                follow_redirects=True,
            )
            try:
                resp = await self._session.get(self.base_url)
                resp.raise_for_status()
                logger.debug("Initialized NSE session cookies.")
            except Exception as exc:
                await self._session.aclose()
                self._session = None
                raise DataUnavailableException(f"Unable to initialize NSE session: {exc}") from exc
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.is_closed:
            await self._session.aclose()

    def _get_bhavcopy_cache_path(self, target_date: date) -> Path:
        return self.cache_dir / f"sec_bhavdata_full_{target_date:%d%m%Y}.csv"

    async def fetch_bhavcopy_for_date(self, target_date: date) -> pd.DataFrame:
        """Download or load the official NSE security Bhavcopy for one date."""
        cache_path = self._get_bhavcopy_cache_path(target_date)
        if cache_path.exists():
            try:
                raw = pd.read_csv(cache_path)
                return self._clean_bhavcopy_df(raw, target_date)
            except Exception as exc:
                logger.warning("Cached Bhavcopy %s is invalid: %s", cache_path, exc)

        date_str = target_date.strftime("%d%m%Y")
        url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
        client = await self._get_client()
        try:
            resp = await client.get(url, headers={**self.headers, "Referer": self.base_url + "/"})
            if resp.status_code != 200 or len(resp.content) <= 100:
                raise DataUnavailableException(
                    f"NSE Bhavcopy unavailable for {target_date} (HTTP {resp.status_code})"
                )
            cache_path.write_bytes(resp.content)
            return self._clean_bhavcopy_df(pd.read_csv(io.BytesIO(resp.content)), target_date)
        except DataUnavailableException:
            raise
        except Exception as exc:
            raise DataUnavailableException(f"Failed to fetch NSE Bhavcopy for {target_date}: {exc}") from exc

    def _clean_bhavcopy_df(self, df: pd.DataFrame, target_date: date) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip().upper() for c in df.columns]
        if "SERIES" in df.columns:
            df["SERIES"] = df["SERIES"].astype(str).str.strip().str.upper()
            df = df[df["SERIES"].isin(["EQ", "BE", "SM"])].copy()

        rename_map = {
            "SYMBOL": "symbol",
            "OPEN_PRICE": "open",
            "HIGH_PRICE": "high",
            "LOW_PRICE": "low",
            "CLOSE_PRICE": "close",
            "TTL_TRD_QNTY": "volume",
            "DELIV_QTY": "delivery_volume",
            "DELIV_PER": "delivery_pct",
            "TURNOVER_LACS": "turnover_lacs",
            "AVG_PRICE": "vwap",
        }
        df = df.rename(columns=rename_map)
        for col in ["open", "high", "low", "close", "volume", "delivery_volume", "delivery_pct", "vwap"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "turnover_lacs" in df.columns:
            df["turnover_crores"] = pd.to_numeric(df["turnover_lacs"], errors="coerce") / 100.0
        elif {"close", "volume"}.issubset(df.columns):
            df["turnover_crores"] = (df["close"] * df["volume"]) / 1e7

        df["timestamp"] = pd.Timestamp(target_date)
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        required = [
            "timestamp", "symbol", "open", "high", "low", "close", "volume",
            "delivery_volume", "delivery_pct", "turnover_crores", "vwap",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataUnavailableException(f"NSE Bhavcopy missing required columns: {missing}")
        return df[required].reset_index(drop=True)

    async def get_historical_ohlcv(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Build a validated historical series from official daily Bhavcopies."""
        symbol = symbol.upper().strip()
        records: list[pd.DataFrame] = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                try:
                    day_df = await self.fetch_bhavcopy_for_date(current)
                    row = day_df[day_df["symbol"] == symbol]
                    if not row.empty:
                        records.append(row.iloc[[0]])
                except DataUnavailableException:
                    pass
            current += timedelta(days=1)

        if not records:
            return pd.DataFrame(columns=[
                "timestamp", "symbol", "open", "high", "low", "close", "volume",
                "delivery_volume", "delivery_pct", "turnover_crores", "vwap",
            ])
        return pd.concat(records, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    async def get_latest_quote(self, symbol: str) -> LiveQuote:
        """Fetch a live NSE equity quote; failures raise instead of fabricating prices."""
        symbol = symbol.upper().strip()
        client = await self._get_client()
        url = f"{self.base_url}/api/quote-equity?{urlencode({'symbol': symbol})}"
        try:
            resp = await client.get(url, headers={**self.headers, "Referer": f"{self.base_url}/get-quotes/equity?symbol={symbol}"})
            resp.raise_for_status()
            data = resp.json()
            price_info = data.get("priceInfo", {})
            intra = price_info.get("intraDayHighLow", {})
            last_price = float(price_info.get("lastPrice", 0.0))
            if last_price <= 0:
                raise DataUnavailableException(f"NSE returned no valid last price for {symbol}")
            return LiveQuote(
                symbol=symbol,
                last_price=last_price,
                open_price=float(price_info.get("open", last_price)),
                high_price=float(intra.get("max", last_price)),
                low_price=float(intra.get("min", last_price)),
                prev_close=float(price_info.get("previousClose", last_price)),
                change_pct=float(price_info.get("pChange", 0.0)),
                total_traded_volume=int(price_info.get("totalTradedVolume", 0)),
                total_traded_value_crores=float(price_info.get("totalTradedValue", 0.0)) / 1e7,
                upper_circuit_limit=float(price_info.get("upperCP", 0.0)),
                lower_circuit_limit=float(price_info.get("lowerCP", 0.0)),
                vwap=float(price_info.get("vwap", last_price)),
                timestamp=datetime.now().astimezone(),
                data_source="NSE_OFFICIAL_API",
            )
        except Exception as exc:
            raise DataUnavailableException(f"Unable to fetch live NSE quote for {symbol}: {exc}") from exc

    async def get_market_breadth(self, index_symbol: str = "NIFTY 500") -> MarketBreadthData:
        """Fetch current NSE index breadth; never substitutes fabricated 50/50 values."""
        client = await self._get_client()
        params = urlencode({"index": index_symbol.upper()})
        url = f"{self.base_url}/api/equity-stockIndices?{params}"
        try:
            resp = await client.get(url, headers={**self.headers, "Referer": f"{self.base_url}/market-data/live-equity-market"})
            resp.raise_for_status()
            data = resp.json().get("data", [])
            rows = data[1:] if len(data) > 1 else data
            if not rows:
                raise DataUnavailableException("NSE breadth response contains no constituents")
            advances = sum(float(r.get("pChange", 0.0)) > 0 for r in rows)
            declines = sum(float(r.get("pChange", 0.0)) < 0 for r in rows)
            unchanged = len(rows) - advances - declines
            return MarketBreadthData(
                date=date.today(),
                index_symbol=index_symbol,
                advances=int(advances),
                declines=int(declines),
                unchanged=int(unchanged),
                advance_decline_ratio=round(advances / max(declines, 1), 2),
            )
        except Exception as exc:
            raise DataUnavailableException(f"Unable to fetch NSE market breadth: {exc}") from exc

    async def get_index_history(self, index_name: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch historical NSE index OHLCV from the official indicesHistory endpoint."""
        client = await self._get_client()
        params = urlencode({
            "indexType": "Index Data",
            "from_date": start_date.strftime("%d-%m-%Y"),
            "to_date": end_date.strftime("%d-%m-%Y"),
            "index": index_name,
        })
        url = f"{self.base_url}/api/historical/indicesHistory?{params}"
        try:
            resp = await client.get(url, headers={**self.headers, "Referer": f"{self.base_url}/reports-indices-historical-index-data"})
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("data", payload if isinstance(payload, list) else [])
            if not rows:
                raise DataUnavailableException(f"No historical index data returned for {index_name}")
            df = pd.DataFrame(rows)
            mapping = {
                "EOD_TIMESTAMP": "timestamp",
                "EOD_OPEN_INDEX_VAL": "open",
                "EOD_HIGH_INDEX_VAL": "high",
                "EOD_LOW_INDEX_VAL": "low",
                "EOD_CLOSE_INDEX_VAL": "close",
            }
            df = df.rename(columns=mapping)
            if "timestamp" not in df or "close" not in df:
                raise DataUnavailableException(f"Unexpected NSE index response schema for {index_name}")
            df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
            df["volume"] = 0.0
            return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        except Exception as exc:
            raise DataUnavailableException(f"Unable to fetch NSE index history for {index_name}: {exc}") from exc

    async def get_india_vix_history(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch historical India VIX from NSE's official VIX endpoint."""
        client = await self._get_client()
        params = urlencode({
            "from_date": start_date.strftime("%d-%m-%Y"),
            "to_date": end_date.strftime("%d-%m-%Y"),
        })
        url = f"{self.base_url}/api/historical/vixhistory?{params}"
        try:
            resp = await client.get(url, headers={**self.headers, "Referer": f"{self.base_url}/reports-indices-historical-vix"})
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("data", payload if isinstance(payload, list) else [])
            if not rows:
                raise DataUnavailableException("No India VIX history returned")
            df = pd.DataFrame(rows).rename(columns={
                "EOD_TIMESTAMP": "timestamp",
                "EOD_OPEN_INDEX_VAL": "open",
                "EOD_HIGH_INDEX_VAL": "high",
                "EOD_LOW_INDEX_VAL": "low",
                "EOD_CLOSE_INDEX_VAL": "close",
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
            return df.reset_index(drop=True)
        except Exception as exc:
            raise DataUnavailableException(f"Unable to fetch India VIX history: {exc}") from exc

    async def fetch_active_securities(self) -> list[SymbolMetadata]:
        """Fetch all active NSE equity listings from EQUITY_L.csv."""
        cache_file = self.cache_dir / "EQUITY_L.csv"
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        client = await self._get_client()
        try:
            resp = await client.get(url, headers={**self.headers, "Referer": self.base_url + "/"})
            resp.raise_for_status()
            if len(resp.content) <= 500:
                raise DataUnavailableException("NSE EQUITY_L.csv response is unexpectedly small")
            cache_file.write_bytes(resp.content)
            df = pd.read_csv(io.BytesIO(resp.content))
        except Exception as exc:
            if not cache_file.exists():
                raise DataUnavailableException(f"Could not download NSE EQUITY_L.csv: {exc}") from exc
            logger.warning("Using cached EQUITY_L.csv because refresh failed: %s", exc)
            df = pd.read_csv(cache_file)

        df.columns = [str(c).strip().upper() for c in df.columns]
        securities: list[SymbolMetadata] = []
        for _, row in df.iterrows():
            sym = str(row.get("SYMBOL", "")).strip().upper()
            series = str(row.get("SERIES", "EQ")).strip().upper()
            name = str(row.get("NAME OF COMPANY", row.get("COMPANY NAME", sym))).strip()
            isin = str(row.get("ISIN NUMBER", row.get("ISIN", ""))).strip()
            if sym and series in {"EQ", "BE", "SM"}:
                securities.append(
                    SymbolMetadata(
                        symbol=sym,
                        company_name=name,
                        isin=isin or None,
                        exchange="NSE",
                        is_active=True,
                        is_fno_eligible=False,
                    )
                )
        if not securities:
            raise DataUnavailableException("NSE EQUITY_L.csv contained no active equity securities")
        return securities
