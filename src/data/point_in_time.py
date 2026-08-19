"""
Point-In-Time Data Safety & Leakage Guard Engine — src/data/point_in_time.py (P0 #12A)

Enforces central point-in-time filtering:
  available_at / publication_date <= decision_time / as_of_date

Guarantees zero future-information leakage in backtests and daily scans.
Fails closed with PIT_UNVERIFIED when availability timestamp is missing.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
import logging
from typing import Any, Callable
import pandas as pd

from src.core.models import (
    AnnualRatios,
    CorporateAnnouncement,
    CorporateEvent,
    NewsArticle,
    QuarterlyFinancials,
    ShareholdingPattern,
)

logger = logging.getLogger(__name__)


class PITViolationError(ValueError):
    """Raised when a data payload violates point-in-time boundaries (contains future rows)."""
    pass


@dataclass
class PITContract:
    source_name: str
    event_date: str | date
    available_at: str | date | datetime | None
    pit_status: str  # "VERIFIED", "UNVERIFIED", "FAIL_CLOSED"
    leakage_risk: str  # "NONE", "LOW", "MEDIUM", "HIGH"

    def is_available(self, as_of_date: date) -> bool:
        """Determines if the record is point-in-time safe and available at as_of_date."""
        if self.available_at is None:
            return False
        avail_d = self.available_at if isinstance(self.available_at, date) else pd.to_datetime(self.available_at).date()
        return avail_d <= as_of_date


class PointInTimeFilter:
    """Central point-in-time enforcement engine filtering future information."""

    @classmethod
    def filter_market_data(cls, df: pd.DataFrame, as_of_date: date | str | pd.Timestamp) -> pd.DataFrame:
        """Filters OHLCV dataframe so no bars beyond as_of_date are visible."""
        if df is None or df.empty or "timestamp" not in df.columns:
            return df

        as_of_dt = pd.to_datetime(as_of_date).date()
        df_copy = df.copy()
        df_copy["_dt"] = pd.to_datetime(df_copy["timestamp"]).dt.date
        df_pit = df_copy[df_copy["_dt"] <= as_of_dt].drop(columns=["_dt"]).copy()
        return df_pit.sort_values("timestamp").reset_index(drop=True)

    @classmethod
    def enforce_pit_boundary(cls, df: pd.DataFrame, as_of_date: date | str | pd.Timestamp) -> pd.DataFrame:
        """
        Enforces the SignalGeneration PIT Contract:
          input_data.max_timestamp <= decision_time
        Fails closed (raises PITViolationError) if any future row > as_of_date is present.
        """
        if df is None or df.empty:
            return df

        as_of_dt = pd.to_datetime(as_of_date).date()
        if "timestamp" in df.columns:
            max_dt = pd.to_datetime(df["timestamp"]).max().date()
        else:
            max_dt = pd.to_datetime(df.index).max().date()

        if max_dt > as_of_dt:
            raise PITViolationError(
                f"PIT Violation: Input DataFrame max_timestamp ({max_dt}) exceeds decision_time ({as_of_dt})."
            )

        return df

    @classmethod
    def filter_news(cls, articles: list[NewsArticle], as_of_date: date | datetime) -> list[NewsArticle]:
        """
        Filters news articles to only those published/available on or before as_of_date/datetime.
        Supports intraday timestamp precision when as_of_date is a datetime object.
        Fails closed (excludes item) if published_at / available_at is missing.
        """
        valid = []
        for a in articles:
            pub = getattr(a, "available_at", None) or getattr(a, "published_at", None)
            if pub is None:
                logger.warning(f"PIT_UNVERIFIED: NewsArticle for {a.symbol} lacks published_at/available_at.")
                continue
            if isinstance(as_of_date, datetime):
                pub_dt = pub if isinstance(pub, datetime) else datetime.combine(pub, datetime.min.time())
                if pub_dt <= as_of_date:
                    valid.append(a)
            else:
                as_of_dt = datetime.combine(as_of_date, datetime.max.time())
                pub_dt = pub if isinstance(pub, datetime) else datetime.combine(pub, datetime.min.time())
                if pub_dt <= as_of_dt:
                    valid.append(a)
        return valid

    @classmethod
    def filter_events(cls, events: list[CorporateEvent], as_of_date: date) -> list[CorporateEvent]:
        """
        Filters corporate events based on explicit announcement_date / available_at.
        Fails closed (excludes item) if availability timestamp is missing.
        """
        valid = []
        for e in events:
            avail = getattr(e, "available_at", None) or getattr(e, "announcement_date", None)
            if avail is None:
                logger.warning(f"PIT_UNVERIFIED: CorporateEvent for {e.symbol} on {e.event_date} lacks available_at/announcement_date.")
                continue
            avail_d = avail if isinstance(avail, date) else pd.to_datetime(avail).date()
            if avail_d <= as_of_date:
                valid.append(e)
        return valid

    @classmethod
    def filter_quarterly_financials(
        cls, financials: list[QuarterlyFinancials], as_of_date: date
    ) -> list[QuarterlyFinancials]:
        """
        Filters quarterly financial results so only results filed/available on or before as_of_date are used.
        Fails closed (excludes item) if filing_date / available_at is missing. Does NOT guess or add 45 days.
        """
        valid = []
        for f in financials:
            avail = getattr(f, "available_at", None) or getattr(f, "filing_date", None)
            if avail is None:
                logger.warning(f"PIT_UNVERIFIED: QuarterlyFinancials for {f.symbol} period_end {f.period_end_date} lacks filing_date/available_at.")
                continue
            avail_d = avail if isinstance(avail, date) else pd.to_datetime(avail).date()
            if avail_d <= as_of_date:
                valid.append(f)
        return valid

    @classmethod
    def filter_annual_ratios(
        cls, ratios: list[AnnualRatios], as_of_date: date
    ) -> list[AnnualRatios]:
        """
        Filters annual ratio records based on explicit available_at timestamp.
        Fails closed (excludes item) if available_at is missing.
        """
        valid = []
        for r in ratios:
            avail = getattr(r, "available_at", None)
            if avail is None:
                logger.warning(f"PIT_UNVERIFIED: AnnualRatios for {r.symbol} fiscal_year {r.fiscal_year} lacks available_at.")
                continue
            avail_d = avail if isinstance(avail, date) else pd.to_datetime(avail).date()
            if avail_d <= as_of_date:
                valid.append(r)
        return valid

    @classmethod
    def filter_shareholding_patterns(
        cls, patterns: list[ShareholdingPattern], as_of_date: date
    ) -> list[ShareholdingPattern]:
        """
        Filters shareholding pattern records based on explicit available_at timestamp.
        Fails closed (excludes item) if available_at is missing.
        """
        valid = []
        for s in patterns:
            avail = getattr(s, "available_at", None)
            if avail is None:
                logger.warning(f"PIT_UNVERIFIED: ShareholdingPattern for {s.symbol} quarter {s.quarter_date} lacks available_at.")
                continue
            avail_d = avail if isinstance(avail, date) else pd.to_datetime(avail).date()
            if avail_d <= as_of_date:
                valid.append(s)
        return valid


class PITRegressionHelper:
    """
    Reusable regression helper that runs a function on baseline data vs future-mutated data
    and asserts exact output identity for session as_of_date.
    """

    @classmethod
    def verify_future_mutation_safety(
        cls,
        target_fn: Callable[[dict[str, pd.DataFrame], str], Any],
        stock_dfs: dict[str, pd.DataFrame],
        as_of_date_str: str,
        price_multiplier: float = 5.0,
    ) -> tuple[bool, Any, Any]:
        """
        Runs target_fn(stock_dfs, as_of_date_str).
        Mutates ONLY rows strictly > as_of_date_str.
        Runs target_fn(mutated_dfs, as_of_date_str).
        Returns (is_identical, baseline_result, mutated_result).
        """
        as_of_dt = pd.to_datetime(as_of_date_str)

        # 1. Run baseline
        baseline_result = target_fn(stock_dfs, as_of_date_str)

        # 2. Build mutated dataset
        mutated_dfs: dict[str, pd.DataFrame] = {}
        for sym, df in stock_dfs.items():
            df_mut = df.copy()
            if "timestamp" in df_mut.columns:
                mask_future = pd.to_datetime(df_mut["timestamp"]) > as_of_dt
            else:
                mask_future = pd.to_datetime(df_mut.index) > as_of_dt

            if mask_future.any():
                df_mut.loc[mask_future, "close"] *= price_multiplier
                df_mut.loc[mask_future, "high"] *= price_multiplier
            mutated_dfs[sym] = df_mut

        # 3. Run mutated
        mutated_result = target_fn(mutated_dfs, as_of_date_str)

        is_identical = (baseline_result == mutated_result)
        return is_identical, baseline_result, mutated_result
