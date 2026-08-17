"""
Portfolio Factor Risk & Correlation Guard Module — src/risk/correlation.py

Prevents portfolio overconcentration in single sectors, macro factors, or commodity sensitivities.
Evaluates portfolio exposure at the factor cluster level (P1.9).
"""

import logging
from typing import Any
from src.core.models import TradeRecommendation

logger = logging.getLogger(__name__)


class PortfolioCorrelationGuard:
    """Filters candidate recommendations to ensure factor diversity and low cross-asset correlation."""

    FACTOR_CLUSTERS = {
        "FINANCIALS": {"BANK", "NBFC", "HOUSING_FINANCE", "INSURANCE", "FINANCIAL_SERVICES"},
        "IT_TECH": {"IT", "SOFTWARE", "IT_SERVICES", "TECH"},
        "COMMODITY_METALS": {"METALS", "STEEL", "MINING", "OIL_GAS", "CHEMICALS", "ENERGY"},
        "CONSUMER": {"FMCG", "RETAIL", "CONSUMER_DURABLES", "TEXTILES"},
        "INDUSTRIAL": {"CAPITAL_GOODS", "DEFENCE", "INFRASTRUCTURE", "REALTY", "ENGINEERING"},
        "AUTO": {"AUTO", "AUTO_ANCILLARIES"},
        "PHARMA": {"PHARMA", "HEALTHCARE"},
    }

    @classmethod
    def get_factor_cluster(cls, sector: str) -> str:
        sec_upper = (sector or "").upper().strip()
        for cluster_name, member_sectors in cls.FACTOR_CLUSTERS.items():
            if sec_upper in member_sectors:
                return cluster_name
        return sec_upper

    @classmethod
    def filter_uncorrelated_basket(
        cls,
        ranked_recommendations: list[TradeRecommendation],
        max_picks: int = 2,
        max_per_sector: int = 1,
    ) -> list[TradeRecommendation]:
        """
        Selects top 0-2 non-conflicting trade recommendations from candidate pool.
        Ensures max 1 pick per sector and max 1 pick per macro factor cluster (P1.9).
        """
        selected: list[TradeRecommendation] = []
        sectors_seen: set[str] = set()
        clusters_seen: set[str] = set()

        for rec in ranked_recommendations:
            sec_name = (rec.sector or "General").upper().strip()
            factor_cluster = cls.get_factor_cluster(sec_name)

            # Skip if sector already represented
            if sec_name in sectors_seen and sec_name != "GENERAL":
                continue

            # Skip if macro factor cluster already represented
            if factor_cluster in clusters_seen and factor_cluster != "GENERAL":
                continue

            selected.append(rec)
            sectors_seen.add(sec_name)
            clusters_seen.add(factor_cluster)

            if len(selected) >= max_picks:
                break

        return selected
