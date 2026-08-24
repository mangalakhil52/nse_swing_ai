"""
Survivorship-Safe Historical Universe Snapshots Module.

Provides date-aware stock universe filtering for backtests and historical scans.
Historical scans require explicit security-master metadata. Live scans use the
current NSE equity master cache and never fall back to a hardcoded watchlist.
"""

from datetime import date
import logging
from pathlib import Path

from config.settings import settings
from src.core.models import SymbolMetadata

logger = logging.getLogger(__name__)


class HistoricalUniverseUnavailableError(ValueError):
    """Raised when the required security-master universe is unavailable."""


class HistoricalUniverseProvider:
    """Provides date-filtered historical or current NSE equity universes."""

    @classmethod
    def get_current_universe(cls) -> list[str]:
        """Return the current NSE equity universe from the official cached security master."""
        cache_file = Path(settings.CACHE_DIR) / "bhavcopy" / "EQUITY_L.csv"
        if not cache_file.exists():
            raise HistoricalUniverseUnavailableError(
                "Current NSE equity universe unavailable: EQUITY_L.csv has not been cached yet. "
                "Fetch the official NSE equity master before running a live scan."
            )

        try:
            import pandas as pd

            df = pd.read_csv(cache_file)
            df.columns = [str(c).strip().upper() for c in df.columns]
            if "SYMBOL" not in df.columns:
                raise ValueError("EQUITY_L.csv has no SYMBOL column")

            if "SERIES" in df.columns:
                df["SERIES"] = df["SERIES"].astype(str).str.strip().str.upper()
                df = df[df["SERIES"].isin(["EQ", "BE", "SM"])].copy()

            symbols = (
                df["SYMBOL"].astype(str).str.strip().str.upper()
                .loc[lambda s: s.ne("")]
                .drop_duplicates()
                .tolist()
            )
            if not symbols:
                raise ValueError("EQUITY_L.csv contains no eligible equity symbols")
            return symbols
        except Exception as exc:
            raise HistoricalUniverseUnavailableError(
                f"Could not load current NSE equity universe from {cache_file}: {exc}"
            ) from exc

    @classmethod
    def filter_universe_by_date(cls, securities: list[SymbolMetadata], as_of_date: date) -> list[SymbolMetadata]:
        """Filter supplied security metadata using listing/delisting dates."""
        eligible: list[SymbolMetadata] = []
        for sec in securities:
            if sec.listing_date and sec.listing_date > as_of_date:
                logger.debug("Excluding %s: listed on %s > as_of_date %s", sec.symbol, sec.listing_date, as_of_date)
                continue
            if sec.delisting_date and sec.delisting_date <= as_of_date:
                logger.debug("Excluding %s: delisted on %s <= as_of_date %s", sec.symbol, sec.delisting_date, as_of_date)
                continue
            eligible.append(sec)
        return eligible

    @classmethod
    def get_universe_for_date(cls, as_of_date: date, securities: list[SymbolMetadata] | None = None) -> list[str]:
        """Return symbols eligible at ``as_of_date``; fail closed without historical metadata."""
        if securities is not None:
            filtered_metadata = cls.filter_universe_by_date(securities, as_of_date)
            return [s.symbol for s in filtered_metadata]

        raise HistoricalUniverseUnavailableError(
            f"Historical security master unavailable for as_of_date={as_of_date}. "
            "Pass explicit SymbolMetadata with listing/delisting metadata. "
            "Fallback to current live watchlist is strictly forbidden."
        )
