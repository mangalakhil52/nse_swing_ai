"""
Unit tests for dynamic NSE Universe Discovery and Security Master.
"""

import asyncio
from pathlib import Path
import pytest

from src.core.models import SymbolMetadata
from src.data.base import MarketDataProvider
from src.data.universe import UniverseDiscoveryEngine


class MockMarketDataProvider(MarketDataProvider):
    async def get_historical_ohlcv(self, symbol, start_date, end_date):
        return None

    async def get_latest_quote(self, symbol):
        return None

    async def get_market_breadth(self, index_symbol="NIFTY 500"):
        return None

    async def fetch_active_securities(self):
        return [
            SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", is_active=True, asm_gsm_stage=0),
            SymbolMetadata(symbol="RELIANCE", company_name="Reliance Industries", is_active=True, asm_gsm_stage=0),
            SymbolMetadata(symbol="RISKY_ASM", company_name="Risky Penny", is_active=True, asm_gsm_stage=2),
            SymbolMetadata(symbol="SUSPENDED_CO", company_name="Suspended Co", is_active=False, asm_gsm_stage=0),
        ]


def test_universe_discovery_filtering(tmp_path: Path):
    async def _run():
        mock_provider = MockMarketDataProvider()
        engine = UniverseDiscoveryEngine(market_data_provider=mock_provider, cache_dir=tmp_path)

        universe = await engine.build_universe(force_refresh=True)

        symbols = [s.symbol for s in universe]
        assert "TRENT" in symbols
        assert "RELIANCE" in symbols
        # ASM stage >= 2 must be excluded
        assert "RISKY_ASM" not in symbols
        trent = next(s for s in universe if s.symbol == "TRENT")
        assert trent.is_fno_eligible is True  # standard FNO symbol

    asyncio.run(_run())
