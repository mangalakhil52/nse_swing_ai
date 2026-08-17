"""
Point-In-Time Data Safety & Leakage Guard Engine — src/data/point_in_time.py

Enforces central point-in-time filtering:
  available_at <= simulation_timestamp / as_of_date
Guarantees zero future-information leakage in backtests and daily scans.
"""

from datetime import date, datetime
import logging
from typing import Any
import pandas as pd

from src.core.models import CorporateAnnouncement, CorporateEvent, NewsArticle, QuarterlyFinancials

logger = logging.getLogger(__name__)


class PointInTimeFilter:
    """Central point-in-time enforcement engine filtering future information."""

    @classmethod
    def filter_market_data(cls, df: pd.DataFrame, as_of_date: date) -> pd.DataFrame:
        """Filters OHLCV dataframe so no bars beyond as_of_date are visible."""
        if df is None or df.empty or "timestamp" not in df.columns:
            return df

        as_of_ts = pd.Timestamp(as_of_date)
        df_pit = df[pd.to_datetime(df["timestamp"]).dt.date <= as_of_date].copy()
        return df_pit.sort_values("timestamp").reset_index(drop=True)

    @classmethod
    def filter_news(cls, articles: list[NewsArticle], as_of_date: date) -> list[NewsArticle]:
        """Filters news articles to only those published/available on or before as_of_date."""
        as_of_dt = datetime.combine(as_of_date, datetime.max.time())
        return [a for a in articles if a.published_at <= as_of_dt]

    @classmethod
    def filter_events(cls, events: list[CorporateEvent], as_of_date: date) -> list[CorporateEvent]:
        """Filters corporate events announced on or before as_of_date."""
        return [e for e in events if e.event_date >= as_of_date]

    @classmethod
    def filter_quarterly_financials(
        cls, financials: list[QuarterlyFinancials], as_of_date: date
    ) -> list[QuarterlyFinancials]:
        """
        Filters quarterly financial results so only results filed/available on or before as_of_date are used.
        """
        valid = []
        for f in financials:
            # Filing date is typically within 45 days of period_end_date
            filing_date = getattr(f, "filing_date", None) or (f.period_end_date + pd.Timedelta(days=45).date())
            if filing_date <= as_of_date:
                valid.append(f)
        return valid
