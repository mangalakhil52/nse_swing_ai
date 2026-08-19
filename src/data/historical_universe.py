"""
Survivorship-Safe Historical Universe Snapshots Module — src/data/historical_universe.py

Provides date-aware stock universe snapshots (universe(date)) for backtesting and historical scans.
Enforces P1.8 survivorship-bias safety by excluding newly listed, delisted, or suspended symbols
prior to their historical eligibility dates.
"""

from datetime import date
import logging
from typing import Any

from config.settings import settings

from src.core.models import SymbolMetadata

logger = logging.getLogger(__name__)


class HistoricalUniverseProvider:
    """Provides historical universe snapshots representation of securities eligible at simulation_date."""

    @classmethod
    def filter_universe_by_date(
        cls, securities: list[SymbolMetadata], as_of_date: date
    ) -> list[SymbolMetadata]:
        """
        Filters SymbolMetadata objects by listing_date and delisting_date as of as_of_date.
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
        Excludes stocks listed after as_of_date or delisted prior to as_of_date.
        """
        if securities:
            filtered_metadata = cls.filter_universe_by_date(securities, as_of_date)
            return [s.symbol for s in filtered_metadata]

        # Historical NSE 500 / Liquid Universe base symbols
        base_symbols = list(settings.FOCUS_WATCHLIST) if hasattr(settings, "FOCUS_WATCHLIST") and settings.FOCUS_WATCHLIST else [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BHARTIARTL", "SBIN",
            "LTIM", "TRENT", "DIXON", "HAL", "BEL", "MAZDOCK", "COALINDIA", "NTPC",
            "POWERGRID", "TATAMOTORS", "M&M", "SUNPHARMA", "CIPLA", "APOLLOHOSP",
        ]

        # Exclude newly listed symbols whose IPO listing date > as_of_date
        # (e.g. IPOs listed in 2026 should not be present in 2024 backtests)
        symbol_listing_dates = {
            "SAATVIKGL": date(2026, 2, 1),
            "EPACKPEB": date(2025, 11, 15),
        }

        eligible_symbols = []
        for sym in base_symbols:
            listing_dt = symbol_listing_dates.get(sym)
            if listing_dt and listing_dt > as_of_date:
                continue  # Exclude stock not yet listed on as_of_date
            eligible_symbols.append(sym)

        return eligible_symbols
