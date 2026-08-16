"""
Unit tests for data providers (NSE Provider, Chartink Provider, Fundamentals, News).
"""

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import pytest

from src.data.chartink_provider import ChartinkProvider
from src.data.fundamental_provider import ScreenerFundamentalProvider
from src.data.news_provider import FinancialNewsProvider
from src.data.nse_provider import NseDataProvider


def test_screener_fundamental_provider(tmp_path: Path):
    async def _run():
        provider = ScreenerFundamentalProvider(cache_dir=tmp_path)

        # Test quarterly financials
        quarterly = await provider.get_quarterly_financials("TRENT")
        assert len(quarterly) > 0
        assert quarterly[0].symbol == "TRENT"
        assert quarterly[0].sales_growth_yoy_pct > 0

        # Test annual ratios
        ratios = await provider.get_annual_ratios("TRENT")
        assert ratios is not None
        assert ratios.roe_pct > 0.0
        assert ratios.debt_to_equity >= 0.0

        # Test shareholding
        shp = await provider.get_shareholding_pattern("TRENT")
        assert shp is not None
        assert shp.promoter_pct > 0.0
        assert shp.promoter_pledged_pct <= 20.0

    asyncio.run(_run())


def test_news_provider(tmp_path: Path):
    async def _run():
        provider = FinancialNewsProvider(cache_dir=tmp_path)
        # Uncached feed returns empty list (no fake neutral news fabricated)
        articles_empty = await provider.fetch_news_feed("TRENT", lookback_days=7)
        assert len(articles_empty) == 0

        # Cached payload feed returns articles
        provider.cache_news_payload(
            "TRENT",
            announcements=[],
            articles=[{
                "symbol": "TRENT",
                "headline": "Trent Q1 Net Profit up 40%",
                "summary": "Strong retail sales momentum",
                "publisher": "Economic Times",
                "source_tier": 2,
                "source_url": "https://economictimes.com/trent",
                "published_at": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "sentiment": "POSITIVE",
                "materiality_score": 0.85,
                "is_catalyst": True,
                "catalyst_type": "EARNINGS_ANNOUNCEMENT",
            }]
        )
        articles = await provider.fetch_news_feed("TRENT", lookback_days=7)
        assert len(articles) == 1
        assert articles[0].symbol == "TRENT"
        assert articles[0].publisher == "Economic Times"

    asyncio.run(_run())


def test_nse_provider_cleaning():
    async def _run():
        provider = NseDataProvider()
        raw_df = pd.DataFrame({
            "SYMBOL": ["TRENT", "RELIANCE"],
            "SERIES": ["EQ", "EQ"],
            "OPEN_PRICE": ["7,000.00", "2,950.00"],
            "HIGH_PRICE": ["7,200.00", "2,980.00"],
            "LOW_PRICE": ["6,950.00", "2,940.00"],
            "CLOSE_PRICE": ["7,180.00", "2,975.00"],
            "TTL_TRD_QNTY": [1500000, 4500000],
            "DELIV_QTY": [900000, 2800000],
            "DELIV_PER": ["60.00", "62.22"],
            "AVG_PRICE": ["7,120.00", "2,965.40"],
        })
        cleaned = provider._clean_bhavcopy_df(raw_df, date(2026, 8, 14))
        assert len(cleaned) == 2
        assert cleaned.iloc[0]["open"] == 7000.0
        assert cleaned.iloc[0]["close"] == 7180.0
        assert cleaned.iloc[0]["delivery_pct"] == 60.0
        await provider.close()

    asyncio.run(_run())
