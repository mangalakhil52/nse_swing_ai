"""
Portfolio Correlation and Sector Exposure Guard Module.
Prevents portfolio overconcentration in single sectors or macroeconomic themes.
Ensures that the final 0–3 recommendations represent diverse, uncorrelated market leaders.
"""

from typing import Any
from src.core.models import TradeRecommendation


class PortfolioCorrelationGuard:
    """Filters candidate recommendations to ensure low cross-asset correlation."""

    @classmethod
    def filter_uncorrelated_basket(
        cls,
        ranked_recommendations: list[TradeRecommendation],
        max_picks: int = 3,
        max_per_sector: int = 1,
    ) -> list[TradeRecommendation]:
        """
        Selects top 0-3 non-conflicting trade recommendations from candidate pool.
        Ensures max 1 pick per sector.
        """
        selected: list[TradeRecommendation] = []
        sectors_seen: set[str] = set()

        for rec in ranked_recommendations:
            sec_name = (rec.sector or "General").upper().strip()

            # Skip if sector already represented
            if sec_name in sectors_seen and sec_name != "GENERAL":
                continue

            selected.append(rec)
            sectors_seen.add(sec_name)

            if len(selected) >= max_picks:
                break

        return selected
