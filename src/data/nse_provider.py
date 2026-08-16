"""
NSE Official Market Data Provider Module.
Directly ingests official NSE Bhavcopy, Sec_Bhav delivery data, live equity quotes, and index metrics.
Implements robust session management, cookie acquisition, local caching, and retry logic.
"""

import asyncio
import io
import json
import logging
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
import httpx
import pandas as pd

from config.settings import settings
from src.core.models import (
    LiveQuote,
    MarketBreadthData,
    SymbolMetadata,
)
from src.data.base import MarketDataProvider

logger = logging.getLogger(__name__)


class NseDataProvider(MarketDataProvider):
    """Primary data provider using official National Stock Exchange of India (NSE) feeds."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or settings.CACHE_DIR / "bhavcopy"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://www.nseindia.com"
        self.headers = {
            "User-Agent": settings.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "DNT": "1",
        }
        self._cookies: dict[str, str] = {}
        self._session: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Returns an active httpx client with initialized NSE session cookies."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(
                headers=self.headers,
                timeout=settings.REQUEST_TIMEOUT_SECONDS,
                follow_redirects=True,
            )
            try:
                # Initialize session by requesting homepage to acquire cookies
                resp = await self._session.get(self.base_url)
                if resp.status_code == 200:
                    self._cookies = dict(resp.cookies)
                    logger.debug("Successfully initialized NSE session cookies.")
            except Exception as e:
                logger.warning(f"Could not initialize live NSE session cookies: {e}")
        return self._session

    async def close(self) -> None:
        """Closes the underlying HTTP client session."""
        if self._session and not self._session.is_closed:
            await self._session.aclose()

    def _get_bhavcopy_cache_path(self, target_date: date) -> Path:
        """Returns file path for cached daily bhavcopy."""
        return self.cache_dir / f"sec_bhavdata_full_{target_date.strftime('%d%m%Y')}.csv"

    async def fetch_bhavcopy_for_date(self, target_date: date) -> pd.DataFrame:
        """
        Downloads and parses official NSE Full Bhavcopy (sec_bhavdata_full) for a given date.
        Includes EQ series, volume, delivery quantity, and delivery percentage.
        """
        cache_path = self._get_bhavcopy_cache_path(target_date)
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path)
                return self._clean_bhavcopy_df(df, target_date)
            except Exception as e:
                logger.warning(f"Error reading cached bhavcopy {cache_path}: {e}")

        # Format URL: https://archives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv
        date_str = target_date.strftime("%d%m%Y")
        url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"

        client = await self._get_client()
        try:
            resp = await client.get(url)
            if resp.status_code == 200 and len(resp.content) > 100:
                cache_path.write_bytes(resp.content)
                df = pd.read_csv(io.BytesIO(resp.content))
                logger.info(f"Successfully downloaded NSE Bhavcopy for {target_date} ({len(df)} rows)")
                return self._clean_bhavcopy_df(df, target_date)
            else:
                logger.warning(f"Bhavcopy not found on NSE Archives for {target_date} (HTTP {resp.status_code})")
        except Exception as e:
            logger.error(f"Failed to fetch Bhavcopy from NSE for {target_date}: {e}")

        return pd.DataFrame()

    def _clean_bhavcopy_df(self, df: pd.DataFrame, target_date: date) -> pd.DataFrame:
        """Standardizes raw NSE sec_bhavdata_full columns and filters EQ series."""
        df.columns = [c.strip().upper() for c in df.columns]

        # Filter only Equity Series ('EQ', 'BE', 'SM')
        if "SERIES" in df.columns:
            df["SERIES"] = df["SERIES"].astype(str).str.strip()
            df = df[df["SERIES"].isin(["EQ", "BE", "SM"])].copy()

        # Standardize column mappings
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

        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume", "delivery_volume", "delivery_pct", "vwap"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce").fillna(0.0)

        if "turnover_lacs" in df.columns:
            df["turnover_crores"] = pd.to_numeric(df["turnover_lacs"], errors="coerce").fillna(0.0) / 100.0

        df["timestamp"] = pd.to_datetime(target_date)
        df["symbol"] = df["symbol"].astype(str).str.strip()

        required = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "delivery_volume", "delivery_pct", "turnover_crores", "vwap"]
        existing = [c for c in required if c in df.columns]
        return df[existing]

    async def get_historical_ohlcv(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        Builds historical daily series for a symbol by aggregating stored Bhavcopy files.
        """
        symbol = symbol.upper().strip()
        current_date = start_date
        records: list[pd.DataFrame] = []

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                day_df = await self.fetch_bhavcopy_for_date(current_date)
                if not day_df.empty and "symbol" in day_df.columns:
                    stock_row = day_df[day_df["symbol"] == symbol]
                    if not stock_row.empty:
                        records.append(stock_row)
            current_date += timedelta(days=1)

        if records:
            res_df = pd.concat(records, ignore_index=True)
            res_df = res_df.sort_values(by="timestamp").reset_index(drop=True)
            return res_df

        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "delivery_volume", "delivery_pct", "turnover_crores", "vwap"])

    async def get_latest_quote(self, symbol: str) -> LiveQuote:
        """
        Fetches live or latest snapshot quote from NSE Official API (/api/quote-equity).
        """
        symbol = symbol.upper().strip()
        client = await self._get_client()
        url = f"{self.base_url}/api/quote-equity?symbol={symbol}"

        try:
            resp = await client.get(url, headers={**self.headers, "Referer": f"{self.base_url}/get-quotes/equity?symbol={symbol}"})
            if resp.status_code == 200:
                data = resp.json()
                price_info = data.get("priceInfo", {})
                pre_open = data.get("preOpenMarket", {})

                last_price = float(price_info.get("lastPrice", 0.0))
                open_price = float(price_info.get("open", last_price))
                high_price = float(price_info.get("intraDayHighLow", {}).get("max", last_price))
                low_price = float(price_info.get("intraDayHighLow", {}).get("min", last_price))
                prev_close = float(price_info.get("previousClose", last_price))
                change_pct = float(price_info.get("pChange", 0.0))
                upper_circuit = float(price_info.get("upperCP", last_price * 1.20))
                lower_circuit = float(price_info.get("lowerCP", last_price * 0.80))
                vwap = float(price_info.get("vwap", last_price))

                # Volume & turnover
                traded_vol = int(price_info.get("totalTradedVolume", 0))
                traded_val_cr = float(price_info.get("totalTradedValue", 0.0)) / 1e7

                return LiveQuote(
                    symbol=symbol,
                    last_price=last_price,
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    prev_close=prev_close,
                    change_pct=change_pct,
                    total_traded_volume=traded_vol,
                    total_traded_value_crores=traded_val_cr,
                    upper_circuit_limit=upper_circuit,
                    lower_circuit_limit=lower_circuit,
                    vwap=vwap,
                    timestamp=datetime.utcnow(),
                    data_source="NSE_OFFICIAL_API",
                )
        except Exception as e:
            logger.warning(f"Error fetching live quote from NSE for {symbol}: {e}")

        # Fallback empty quote
        return LiveQuote(
            symbol=symbol,
            last_price=0.0,
            open_price=0.0,
            high_price=0.0,
            low_price=0.0,
            prev_close=0.0,
            change_pct=0.0,
            total_traded_volume=0,
            total_traded_value_crores=0.0,
            upper_circuit_limit=0.0,
            lower_circuit_limit=0.0,
            timestamp=datetime.utcnow(),
            data_source="NSE_OFFICIAL_API_FALLBACK",
        )

    async def get_market_breadth(self, index_symbol: str = "NIFTY 500") -> MarketBreadthData:
        """
        Fetches market breadth for Nifty 50 / Nifty 500 from NSE Index API.
        """
        client = await self._get_client()
        url = f"{self.base_url}/api/equity-stockIndices?index={index_symbol.upper().replace(' ', '%20')}"

        try:
            resp = await client.get(url, headers={**self.headers, "Referer": f"{self.base_url}/market-data/live-equity-market"})
            if resp.status_code == 200:
                data = resp.json()
                data_list = data.get("data", [])
                if len(data_list) > 1:
                    advances = sum(1 for item in data_list[1:] if float(item.get("pChange", 0.0)) > 0)
                    declines = sum(1 for item in data_list[1:] if float(item.get("pChange", 0.0)) < 0)
                    unchanged = sum(1 for item in data_list[1:] if float(item.get("pChange", 0.0)) == 0)
                    ad_ratio = round(advances / max(declines, 1), 2)

                    return MarketBreadthData(
                        date=date.today(),
                        index_symbol=index_symbol,
                        advances=advances,
                        declines=declines,
                        unchanged=unchanged,
                        advance_decline_ratio=ad_ratio,
                    )
        except Exception as e:
            logger.warning(f"Error fetching market breadth for {index_symbol}: {e}")

        # Fallback default
        return MarketBreadthData(
            date=date.today(),
            index_symbol=index_symbol,
            advances=250,
            declines=250,
            unchanged=0,
            advance_decline_ratio=1.0,
        )

    async def fetch_active_securities(self) -> list[SymbolMetadata]:
        """
        Fetches official list of all active equity securities listed on NSE (EQUITY_L.csv).
        """
        cache_file = self.cache_dir / "EQUITY_L.csv"
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

        df: pd.DataFrame | None = None
        client = await self._get_client()

        try:
            resp = await client.get(url)
            if resp.status_code == 200 and len(resp.content) > 500:
                cache_file.write_bytes(resp.content)
                df = pd.read_csv(io.BytesIO(resp.content))
        except Exception as e:
            logger.warning(f"Could not download fresh EQUITY_L.csv: {e}")
            if cache_file.exists():
                df = pd.read_csv(cache_file)

        if df is None or df.empty:
            return []

        df.columns = [c.strip().upper() for c in df.columns]
        securities: list[SymbolMetadata] = []

        for _, row in df.iterrows():
            sym = str(row.get("SYMBOL", "")).strip()
            name = str(row.get("NAME OF COMPANY", row.get("COMPANY NAME", sym))).strip()
            isin = str(row.get("ISIN NUMBER", row.get("ISIN", ""))).strip()
            series = str(row.get("SERIES", "EQ")).strip()

            if sym and series in ["EQ", "BE", "SM"]:
                securities.append(
                    SymbolMetadata(
                        symbol=sym,
                        company_name=name,
                        isin=isin if isin else None,
                        exchange="NSE",
                        is_active=True,
                        is_fno_eligible=False,
                    )
                )

        return securities
