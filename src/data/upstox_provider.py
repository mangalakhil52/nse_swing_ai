"""Authenticated Upstox provider for live quotes, instrument master and candles.

Failed API calls raise DataUnavailableException instead of manufacturing prices,
breadth, or candles. NSE official Bhavcopy remains available as the EOD/archive
provider; Upstox supplies authenticated live/intraday data.
"""
from __future__ import annotations

import gzip
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import pandas as pd

from config.settings import settings
from src.core.exceptions import DataUnavailableException
from src.core.models import LiveQuote, SymbolMetadata

logger = logging.getLogger(__name__)


class UpstoxDataProvider:
    """Strict authenticated Upstox REST provider."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        if not settings.UPSTOX_ACCESS_TOKEN:
            raise DataUnavailableException("UPSTOX_ACCESS_TOKEN is not configured")
        self.cache_dir = cache_dir or settings.CACHE_DIR / "upstox"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = settings.UPSTOX_BASE_URL.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._instruments: list[dict[str, Any]] | None = None
        self._by_symbol: dict[str, dict[str, Any]] = {}
        self._by_index: dict[str, dict[str, Any]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=settings.UPSTOX_TIMEOUT_SECONDS,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.UPSTOX_ACCESS_TOKEN}",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        client = await self._get_client()
        response = await client.get(f"{self.base_url}{path}", params=params)
        if response.status_code != 200:
            raise DataUnavailableException(f"UPSTOX_HTTP_{response.status_code}: {response.text[:300]}")
        payload = response.json()
        if payload.get("status") not in (None, "success"):
            raise DataUnavailableException(f"UPSTOX_API_ERROR: {payload}")
        return payload

    async def load_nse_equity_master(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        """Load the current Upstox NSE instrument master; never use a hardcoded symbol list."""
        cache_path = self.cache_dir / "NSE.json.gz"
        if self._instruments is not None and not force_refresh:
            return self._instruments
        if cache_path.exists() and not force_refresh:
            raw = cache_path.read_bytes()
        else:
            client = await self._get_client()
            response = await client.get(settings.UPSTOX_INSTRUMENT_MASTER_URL)
            if response.status_code != 200 or len(response.content) < 1000:
                raise DataUnavailableException(f"UPSTOX_INSTRUMENT_MASTER_HTTP_{response.status_code}")
            raw = response.content
            cache_path.write_bytes(raw)
        try:
            data = gzip.decompress(raw)
        except OSError:
            data = raw
        instruments = json.loads(data.decode("utf-8"))
        if not isinstance(instruments, list):
            raise DataUnavailableException("Invalid Upstox instrument master payload")

        self._instruments = instruments
        self._by_symbol = {}
        self._by_index = {}
        for item in instruments:
            segment = item.get("segment")
            instrument_type = item.get("instrument_type")
            if segment == "NSE_EQ" and instrument_type in {"EQ", "BE"}:
                symbol = str(item.get("trading_symbol", "")).strip().upper()
                if symbol:
                    self._by_symbol[symbol] = item
            elif segment == "NSE_INDEX" and instrument_type == "INDEX":
                symbol = str(item.get("trading_symbol", "")).strip().upper()
                if symbol:
                    self._by_index[symbol] = item
        if not self._by_symbol:
            raise DataUnavailableException("Upstox NSE equity master contains no NSE_EQ EQ/BE instruments")
        return instruments

    async def fetch_active_securities(self) -> list[SymbolMetadata]:
        instruments = await self.load_nse_equity_master()
        fno_symbols = {
            str(x.get("underlying_symbol", "")).strip().upper()
            for x in instruments
            if x.get("segment") == "NSE_FO" and x.get("underlying_type") == "EQUITY"
        }
        return [
            SymbolMetadata(
                symbol=symbol,
                company_name=str(item.get("name") or item.get("short_name") or symbol),
                isin=item.get("isin") or None,
                exchange="NSE",
                sector="Unknown",
                industry="Unknown",
                is_fno_eligible=symbol in fno_symbols,
                is_active=True,
                lot_size=int(item.get("lot_size") or 1),
            )
            for symbol, item in self._by_symbol.items()
        ]

    async def resolve_instrument_key(self, symbol: str) -> str:
        await self.load_nse_equity_master()
        item = self._by_symbol.get(symbol.upper().strip())
        if not item:
            raise DataUnavailableException(f"UPSTOX_SYMBOL_NOT_FOUND: {symbol}")
        return str(item["instrument_key"])

    async def resolve_index_key(self, symbol: str) -> str:
        await self.load_nse_equity_master()
        item = self._by_index.get(symbol.upper().strip())
        if not item:
            raise DataUnavailableException(f"UPSTOX_INDEX_NOT_FOUND: {symbol}")
        return str(item["instrument_key"])

    async def _get_daily_by_key(self, key: str, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        encoded_key = quote(key, safe="")
        path = f"/v3/historical-candle/{encoded_key}/days/1/{end_date.isoformat()}/{start_date.isoformat()}"
        payload = await self._get(path)
        candles = payload.get("data", {}).get("candles", [])
        if not candles:
            raise DataUnavailableException(f"No Upstox daily candles for {symbol} between {start_date} and {end_date}")
        rows = []
        for candle in candles:
            if len(candle) < 6:
                continue
            rows.append({
                "timestamp": pd.to_datetime(candle[0]), "symbol": symbol.upper().strip(),
                "open": float(candle[1]), "high": float(candle[2]), "low": float(candle[3]),
                "close": float(candle[4]), "volume": int(candle[5]),
                "delivery_volume": None, "delivery_pct": None,
                "turnover_crores": (float(candle[4]) * int(candle[5])) / 1e7,
                "vwap": None, "data_source": "UPSTOX_HISTORICAL_V3",
            })
        df = pd.DataFrame(rows).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        if df.empty:
            raise DataUnavailableException(f"Invalid empty candle set for {symbol}")
        return df

    async def get_daily_ohlcv(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        key = await self.resolve_instrument_key(symbol)
        return await self._get_daily_by_key(key, symbol, start_date, end_date)

    async def get_index_daily_ohlcv(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        key = await self.resolve_index_key(symbol)
        return await self._get_daily_by_key(key, symbol, start_date, end_date)

    async def get_intraday_ohlcv(self, symbol: str, unit: str = "minutes", interval: int = 5) -> pd.DataFrame:
        key = await self.resolve_instrument_key(symbol)
        encoded_key = quote(key, safe="")
        payload = await self._get(f"/v3/historical-candle/intraday/{encoded_key}/{unit}/{interval}")
        candles = payload.get("data", {}).get("candles", [])
        if not candles:
            raise DataUnavailableException(f"No current-day intraday candles for {symbol}")
        rows = [
            {"timestamp": pd.to_datetime(c[0]), "symbol": symbol.upper().strip(), "open": float(c[1]),
             "high": float(c[2]), "low": float(c[3]), "close": float(c[4]), "volume": int(c[5])}
            for c in candles if len(c) >= 6
        ]
        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    async def get_full_market_quotes(self, instrument_keys: list[str]) -> dict[str, dict[str, Any]]:
        if not instrument_keys:
            return {}
        result: dict[str, dict[str, Any]] = {}
        batch_size = max(1, min(int(settings.UPSTOX_BATCH_SIZE), 500))
        for start in range(0, len(instrument_keys), batch_size):
            batch = instrument_keys[start:start + batch_size]
            payload = await self._get("/v2/market-quote/quotes", params={"instrument_key": ",".join(batch)})
            for key, quote_data in payload.get("data", {}).items():
                result[str(quote_data.get("instrument_token", key))] = quote_data
        return result

    async def get_latest_quote(self, symbol: str) -> LiveQuote:
        key = await self.resolve_instrument_key(symbol)
        quotes = await self.get_full_market_quotes([key])
        q = quotes.get(key)
        if not q:
            raise DataUnavailableException(f"No live quote returned for {symbol}")
        last = float(q.get("last_price", 0.0))
        if last <= 0:
            raise DataUnavailableException(f"Invalid live price returned for {symbol}")
        prev_close = float(q.get("prev_close", q.get("ohlc", {}).get("close", 0.0)))
        change_pct = ((last / prev_close) - 1.0) * 100.0 if prev_close > 0 else 0.0
        return LiveQuote(
            symbol=symbol.upper().strip(), last_price=last,
            open_price=float(q.get("ohlc", {}).get("open", last)),
            high_price=float(q.get("ohlc", {}).get("high", last)),
            low_price=float(q.get("ohlc", {}).get("low", last)),
            prev_close=prev_close, change_pct=change_pct,
            total_traded_volume=int(q.get("volume", 0)), total_traded_value_crores=0.0,
            upper_circuit_limit=float(q.get("upper_circuit_limit", 0.0)),
            lower_circuit_limit=float(q.get("lower_circuit_limit", 0.0)),
            vwap=float(q.get("average_price")) if q.get("average_price") is not None else None,
            timestamp=datetime.now().astimezone(), data_source="UPSTOX_FULL_MARKET_QUOTE",
        )
