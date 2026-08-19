"""
Survivorship-Safe Historical Universe Snapshots Module — src/data/historical_universe.py

Provides date-aware stock universe filtering (universe(date)) for backtesting and historical scans.
Enforces survivorship-bias safety by excluding newly listed (IPO > as_of_date) or delisted (delisting <= as_of_date)
securities prior to their historical eligibility dates.
"""

from datetime import date
import logging
from typing import Any

from config.settings import settings
from src.core.models import SymbolMetadata

logger = logging.getLogger(__name__)


class HistoricalUniverseUnavailableError(ValueError):
    """Raised when historical security master data is requested for as_of_date but no historical membership source exists."""
    pass


class HistoricalUniverseProvider:
    """Provides date-filtered historical universe snapshots of securities eligible at as_of_date."""

    @classmethod
    def get_current_universe(cls) -> list[str]:
        """
        Returns the current live active focus watchlist for real-time scanning.
        MUST NOT be used as a proxy for historical universe at date T.
        """
        if hasattr(settings, "FOCUS_WATCHLIST") and settings.FOCUS_WATCHLIST:
            return list(settings.FOCUS_WATCHLIST)
        return [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BHARTIARTL", "SBIN",
            "LTIM", "TRENT", "DIXON", "HAL", "BEL", "MAZDOCK", "COALINDIA", "NTPC",
            "POWERGRID", "TATAMOTORS", "M&M", "SUNPHARMA", "CIPLA", "APOLLOHOSP",
        ]

    @classmethod
    def filter_universe_by_date(
        cls, securities: list[SymbolMetadata], as_of_date: date
    ) -> list[SymbolMetadata]:
        """
        Filters supplied SymbolMetadata objects by listing_date and delisting_date as of as_of_date.
        - Excludes security if listing_date > as_of_date (future IPO / not yet listed).
        - Excludes security if delisting_date is set and delisting_date <= as_of_date (already delisted).
        """
        eligible = []
        for sec in securities:
            if sec.listing_date and sec.listing_date > as_of_date:
                logger.debug(f"Excluding {sec.symbol}: listed on {sec.listing_date} > as_of_date {as_of_date}")
                continue
            if sec.delisting_date and sec.delisting_date <= as_of_date:
                logger.debug(f"Excluding {sec.symbol}: delisted on {sec.delisting_date} <= as_of_date {as_of_date}")
                continue
            eligible.append(sec)
        return eligible

    @classmethod
    def get_universe_for_date(
        cls, as_of_date: date, securities: list[SymbolMetadata] | None = None
    ) -> list[str]:
        """
        Returns list of symbol strings eligible as of as_of_date.
        Fails closed with HistoricalUniverseUnavailableError if securities metadata is None.
        Does NOT silently fall back to today's FOCUS_WATCHLIST.
        """
        if securities is not None:
            filtered_metadata = cls.filter_universe_by_date(securities, as_of_date)
            return [s.symbol for s in filtered_metadata]

        # Fail closed: No complete historical security master exists yet for unsupplied historical queries.
        raise HistoricalUniverseUnavailableError(
            f"Historical security master unavailable for as_of_date={as_of_date}. "
            "Pass explicit securities metadata list or supply HistoricalSecurityMaster provider. "
            "Fallback to current live watchlist is strictly forbidden to prevent survivorship bias."
        )
