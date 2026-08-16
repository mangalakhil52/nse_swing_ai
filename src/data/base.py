"""
Abstract Provider Interfaces for Data Ingestion.
Decouples application logic from external data APIs and ensures testability through mocks.
"""

from abc import ABC, abstractmethod
from datetime import date
import pandas as pd

from src.core.models import (
    LiveQuote,
    MarketBreadthData,
    QuarterlyFinancials,
    AnnualRatios,
    ShareholdingPattern,
    NewsArticle,
    CorporateAnnouncement,
    CorporateEvent,
    SplitBonusAdjustment,
    SymbolMetadata,
)


class MarketDataProvider(ABC):
    """Abstract interface for historical and live market price/volume data."""

    @abstractmethod
    async def get_historical_ohlcv(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        Fetches historical daily OHLCV dataframe.
        Must return columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'delivery_volume', 'delivery_pct', 'turnover_crores']
        """
        pass

    @abstractmethod
    async def get_latest_quote(self, symbol: str) -> LiveQuote:
        """Fetches real-time or recent quote including circuit limits, VWAP, and spread."""
        pass

    @abstractmethod
    async def get_market_breadth(self, index_symbol: str = "NIFTY 500") -> MarketBreadthData:
        """Fetches advance/decline ratio and market participation statistics."""
        pass

    @abstractmethod
    async def fetch_active_securities(self) -> list[SymbolMetadata]:
        """Fetches all active equity listings from the exchange."""
        pass


class ChartinkScannerProvider(ABC):
    """Abstract interface for Chartink custom query engine."""

    @abstractmethod
    async def run_scanner_query(self, scan_clause: str) -> list[str]:
        """
        Executes a Chartink Atlas scan query string and returns a list of matching NSE symbols.
        """
        pass


class FundamentalProvider(ABC):
    """Abstract interface for financial statements, profitability metrics, and shareholding."""

    @abstractmethod
    async def get_quarterly_financials(self, symbol: str) -> list[QuarterlyFinancials]:
        """Returns quarterly sales, PAT, EBITDA, and EPS for recent quarters."""
        pass

    @abstractmethod
    async def get_annual_ratios(self, symbol: str) -> AnnualRatios | None:
        """Returns ROE, ROCE, Debt/Equity, and CFO ratios."""
        pass

    @abstractmethod
    async def get_shareholding_pattern(self, symbol: str) -> ShareholdingPattern | None:
        """Returns promoter holding, promoter pledging %, FII %, DII %, and public %."""
        pass


class NewsProvider(ABC):
    """Abstract interface for company announcements and curated financial news."""

    @abstractmethod
    async def fetch_company_announcements(
        self, symbol: str, lookback_days: int = 14
    ) -> list[CorporateAnnouncement]:
        """Returns official exchange announcements filed by the company."""
        pass

    @abstractmethod
    async def fetch_news_feed(
        self, symbol: str, lookback_days: int = 7
    ) -> list[NewsArticle]:
        """Returns financial news articles with source tier, sentiment, and materiality."""
        pass


class CorporateActionsProvider(ABC):
    """Abstract interface for corporate event schedules and split/bonus adjustments."""

    @abstractmethod
    async def get_upcoming_events(self, symbol: str) -> list[CorporateEvent]:
        """Returns upcoming board meetings (earnings), dividends, and record dates."""
        pass

    @abstractmethod
    async def get_historical_adjustments(self, symbol: str) -> list[SplitBonusAdjustment]:
        """Returns historical stock splits and bonus issues."""
        pass
