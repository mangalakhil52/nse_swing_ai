"""
NSE Official Market Data Provider Module.
Directly ingests official NSE Bhavcopy, Sec_Bhav delivery data, live equity quotes, and index metrics.
No synthetic market-data fallback is permitted.
"""

import io
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
import httpx
import pandas as pd

from config.settings import settings
from src.core.models import LiveQuote, MarketBreadthData, SymbolMetadata
from src.core.exceptions import DataUnavailableException
from src.data.base import MarketDataProvider

logger = logging.getLogger(__name__)


class NseDataProvider(MarketDataProvider):
    """Primary data provider using official National Stock Exchange of India feeds."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or settings.CACHE_DIR / "bhavcopy"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://www.nseindia.com"
        self.headers = {
            "User-Agent": settings.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
        self._session: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(headers=self.headers, timeout=settings.REQUEST_TIMEOUT_SECONDS, follow_redirects=True)
            try:
                await self._session.get(self.base_url)
            except Exception as exc:
                logger.warning("Could not initialize NSE session: %s", exc)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.is_closed:
            await self._session.aclose()

    def _get_bhavcopy_cache_path(self, target_date: date) -> Path:
        return self.cache_dir / f"sec_bhavdata_full_{target_date.strftime('%d%m%Y')}.csv"

    async def fetch_bhavcopy_for_date(self, target_date: date) -> pd.DataFrame:
        cache_path = self._get_bhavcopy_cache_path(target_date)
        if cache_path.exists():
            try:
                return self._clean_bhavcopy_df(pd.read_csv(cache_path), target_date)
            except Exception as exc:
                logger.warning("Error reading cached bhavcopy %s: %s", cache_path, exc)
        date_str = target_date.strftime("%d%m%Y")
        url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
        client = await self._get_client()
        try:
            resp = await client.get(url)
            if resp.status_code == 200 and len(resp.content) > 100:
                cache_path.write_bytes(resp.content)
                return self._clean_bhavcopy_df(pd.read_csv(io.BytesIO(resp.content)), target_date)
        except Exception as exc:
            logger.error("Failed to fetch Bhavcopy %s: %s", target_date, exc)
        return pd.DataFrame()

    def _clean_bhavcopy_df(self, df: pd.DataFrame, target_date: date) -> pd.DataFrame:
        df.columns = [c.strip().upper() for c in df.columns]
        if "SERIES" in df.columns:
            df["SERIES"] = df["SERIES"].astype(str).str.strip()
            df = df[df["SERIES"].isin(["EQ", "BE", "SM"])].copy()
        df = df.rename(columns={
            "SYMBOL": "symbol", "OPEN_PRICE": "open", "HIGH_PRICE": "high", "LOW_PRICE": "low",
            "CLOSE_PRICE": "close", "TTL_TRD_QNTY": "volume", "DELIV_QTY": "delivery_volume",
            "DELIV_PER": "delivery_pct", "TURNOVER_LACS": "turnover_lacs", "AVG_PRICE": "vwap",
        })
        for col in ["open", "high", "low", "close", "volume", "delivery_volume", "delivery_pct", "vwap"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce").fillna(0.0)
        if "turnover_lacs" in df.columns:
            df["turnover_crores"] = pd.to_numeric(df["turnover_lacs"], errors="coerce").fillna(0.0) / 100.0
        df["timestamp"] = pd.to_datetime(target_date)
        df["symbol"] = df["symbol"].astype(str).str.strip()
        required = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "delivery_volume", "delivery_pct", "turnover_crores", "vwap"]
        return df[[c for c in required if c in df.columns]]

    async def get_historical_ohlcv(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        symbol = symbol.upper().strip()
        current_date, records = start_date, []
        while current_date <= end_date:
            if current_date.weekday() < 5:
                day_df = await self.fetch_bhavcopy_for_date(current_date)
                if not day_df.empty and "symbol" in day_df.columns:
                    row = day_df[day_df["symbol"] == symbol]
                    if not row.empty:
                        records.append(row)
            current_date += timedelta(days=1)
        return pd.concat(records, ignore_index=True).sort_values("timestamp").reset_index(drop=True) if records else pd.DataFrame()

    async def get_latest_quote(self, symbol: str) -> LiveQuote:
        symbol = symbol.upper().strip()
        client = await self._get_client()
        try:
            resp = await client.get(f"{self.base_url}/api/quote-equity?symbol={symbol}", headers={**self.headers, "Referer": f"{self.base_url}/get-quotes/equity?symbol={symbol}"})
            if resp.status_code != 200:
                raise DataUnavailableException(f"NSE quote HTTP {resp.status_code} for {symbol}")
            info = resp.json().get("priceInfo", {})
            last = float(info.get("lastPrice", 0.0))
            if last <= 0:
                raise DataUnavailableException(f"NSE quote has invalid last price for {symbol}")
            return LiveQuote(symbol=symbol, last_price=last, open_price=float(info.get("open", last)),
                high_price=float(info.get("intraDayHighLow", {}).get("max", last)), low_price=float(info.get("intraDayHighLow", {}).get("min", last)),
                prev_close=float(info.get("previousClose", last)), change_pct=float(info.get("pChange", 0.0)),
                total_traded_volume=int(info.get("totalTradedVolume", 0)), total_traded_value_crores=float(info.get("totalTradedValue", 0.0))/1e7,
                upper_circuit_limit=float(info.get("upperCP", 0.0)), lower_circuit_limit=float(info.get("lowerCP", 0.0)),
                vwap=float(info.get("vwap")) if info.get("vwap") is not None else None,
                timestamp=datetime.now().astimezone(), data_source="NSE_OFFICIAL_API")
        except DataUnavailableException:
            raise
        except Exception as exc:
            raise DataUnavailableException(f"NSE quote unavailable for {symbol}: {exc}") from exc

    async def get_market_breadth(self, index_symbol: str = "NIFTY 500") -> MarketBreadthData:
        client = await self._get_client()
        try:
            resp = await client.get(f"{self.base_url}/api/equity-stockIndices", params={"index": index_symbol}, headers={**self.headers, "Referer": f"{self.base_url}/market-data/live-equity-market"})
            if resp.status_code != 200:
                raise DataUnavailableException(f"NSE breadth HTTP {resp.status_code}")
            items = resp.json().get("data", [])
            if len(items) <= 1:
                raise DataUnavailableException(f"NSE breadth returned no constituents for {index_symbol}")
            advances = sum(1 for item in items[1:] if float(item.get("pChange", 0.0)) > 0)
            declines = sum(1 for item in items[1:] if float(item.get("pChange", 0.0)) < 0)
            unchanged = sum(1 for item in items[1:] if float(item.get("pChange", 0.0)) == 0)

            vix_resp = await client.get(f"{self.base_url}/api/allIndices", headers={**self.headers, "Referer": self.base_url + "/"})
            if vix_resp.status_code != 200:
                raise DataUnavailableException(f"NSE allIndices HTTP {vix_resp.status_code}")
            vix_value = None
            for item in vix_resp.json().get("data", []):
                name = str(item.get("index", item.get("indexSymbol", ""))).upper().replace(" ", "")
                if name in {"INDIAVIX", "INDIAVIXINDEX"}:
                    raw = item.get("last", item.get("lastPrice", item.get("indexClose")))
                    if raw is not None:
                        vix_value = float(raw)
                    break
            if vix_value is None or vix_value <= 0:
                raise DataUnavailableException("NSE India VIX observation unavailable")

            return MarketBreadthData(date=date.today(), index_symbol=index_symbol, advances=advances, declines=declines,
                unchanged=unchanged, advance_decline_ratio=round(advances / max(declines, 1), 2), india_vix=vix_value)
        except DataUnavailableException:
            raise
        except Exception as exc:
            raise DataUnavailableException(f"NSE breadth unavailable: {exc}") from exc

    async def fetch_active_securities(self) -> list[SymbolMetadata]:
        cache_file = self.cache_dir / "EQUITY_L.csv"
        client = await self._get_client()
        df = None
        try:
            resp = await client.get("https://archives.nseindia.com/content/equities/EQUITY_L.csv")
            if resp.status_code == 200 and len(resp.content) > 500:
                cache_file.write_bytes(resp.content)
                df = pd.read_csv(io.BytesIO(resp.content))
        except Exception as exc:
            logger.warning("Could not download fresh EQUITY_L.csv: %s", exc)
        if df is None and cache_file.exists():
            df = pd.read_csv(cache_file)
        if df is None or df.empty:
            return []
        df.columns = [c.strip().upper() for c in df.columns]
        result = []
        for _, row in df.iterrows():
            sym = str(row.get("SYMBOL", "")).strip()
            series = str(row.get("SERIES", "EQ")).strip()
            if sym and series in ["EQ", "BE", "SM"]:
                result.append(SymbolMetadata(symbol=sym, company_name=str(row.get("NAME OF COMPANY", sym)).strip(),
                    isin=str(row.get("ISIN NUMBER", "")).strip() or None, exchange="NSE", is_active=True, is_fno_eligible=False))
        return result
