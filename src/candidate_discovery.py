"""
Candidate Discovery Engine Module — src/candidate_discovery.py (P0 #14B)

Provides fast, deterministic, point-in-time safe screening of an input equity universe
down to an eligible candidate pool for downstream analysis.

Pipeline:
  Universe -> Eligibility Check -> PIT Slicing -> Data Quality Gate -> Screening Filters -> Candidate Pool

Enforces:
  1. Accepts any stock universe input (NO hardcoded symbol list).
  2. Historical & Live modes share identical screening logic using explicit as_of_date.
  3. Strict PIT safety & DataQualityGate integration (future data & corrupted OHLC hard-fail).
  4. Explainable rejection output with machine-readable reason strings.
  5. NO downstream trade execution, target/SL calculations, position sizing, or AI conviction models.
"""

from datetime import date, datetime
import logging
from typing import Any
import pandas as pd
from pydantic import BaseModel, Field

from src.core.models import SymbolMetadata
from src.data.data_quality import DataQualityGate, DataQualityResult, DataQualityStatus
from src.data.point_in_time import PointInTimeFilter

logger = logging.getLogger(__name__)


class CandidateDiscoveryConfig(BaseModel):
    """Filter parameters for Stage-1 Candidate Discovery Screening."""
    min_price: float = Field(default=20.0, description="Minimum stock price in INR")
    max_price: float = Field(default=50000.0, description="Maximum stock price in INR")
    min_average_volume: float = Field(default=10000.0, description="Minimum 20-day average daily volume")
    min_average_turnover_crores: float = Field(default=0.5, description="Minimum 20-day average turnover in ₹ Crores")
    min_history_length: int = Field(default=50, description="Minimum required historical OHLCV bars")
    require_trend_alignment: bool = Field(default=False, description="Whether to require close >= 20 EMA")


class CandidateDiscoveryResult(BaseModel):
    """Deterministic screening result contract for candidate discovery."""
    symbol: str
    decision_time: datetime | date
    eligible: bool
    passed_filters: list[str] = Field(default_factory=list)
    failed_filters: list[str] = Field(default_factory=list)
    filter_results: dict[str, bool] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    data_quality: DataQualityResult | None = None
    pit_safe: bool = Field(default=True)
    discovery_score: float | None = Field(default=None, description="Purely screening metric (e.g. ADTV in ₹ Cr)")


class CandidateDiscoveryEngine:
    """High-speed Stage-1 Candidate Discovery Engine."""

    @classmethod
    def discover_candidates(
        cls,
        universe: list[SymbolMetadata] | list[str],
        as_of_date: datetime | date,
        market_data_map: dict[str, pd.DataFrame],
        config: CandidateDiscoveryConfig | None = None,
    ) -> list[CandidateDiscoveryResult]:
        """
        Screen an input equity universe against Stage-1 quality and liquidity rules as of as_of_date.
        """
        cfg = config or CandidateDiscoveryConfig()
        results: list[CandidateDiscoveryResult] = []

        for item in universe:
            if isinstance(item, SymbolMetadata):
                sec_meta = item
                symbol = sec_meta.symbol.upper().strip()
            else:
                sec_meta = None
                symbol = item.upper().strip()

            filter_results: dict[str, bool] = {}
            passed_filters: list[str] = []
            failed_filters: list[str] = []
            reasons: list[str] = []
            pit_safe = True

            # 1. Universe Eligibility Filter (Metadata check)
            if sec_meta is not None:
                as_of_d = as_of_date.date() if isinstance(as_of_date, datetime) else as_of_date
                if sec_meta.listing_date and sec_meta.listing_date > as_of_d:
                    filter_results["UNIVERSE_ELIGIBILITY"] = False
                    failed_filters.append("UNIVERSE_ELIGIBILITY")
                    reasons.append("UNIVERSE_INELIGIBLE_NOT_YET_LISTED")
                elif sec_meta.delisting_date and sec_meta.delisting_date <= as_of_d:
                    filter_results["UNIVERSE_ELIGIBILITY"] = False
                    failed_filters.append("UNIVERSE_ELIGIBILITY")
                    reasons.append("UNIVERSE_INELIGIBLE_DELISTED")
                elif sec_meta.asm_gsm_stage >= 2:
                    filter_results["UNIVERSE_ELIGIBILITY"] = False
                    failed_filters.append("UNIVERSE_ELIGIBILITY")
                    reasons.append("UNIVERSE_INELIGIBLE_ASM_GSM_STAGE")
                else:
                    filter_results["UNIVERSE_ELIGIBILITY"] = True
                    passed_filters.append("UNIVERSE_ELIGIBILITY")
            else:
                filter_results["UNIVERSE_ELIGIBILITY"] = True
                passed_filters.append("UNIVERSE_ELIGIBILITY")

            if not filter_results["UNIVERSE_ELIGIBILITY"]:
                results.append(CandidateDiscoveryResult(
                    symbol=symbol,
                    decision_time=as_of_date,
                    eligible=False,
                    passed_filters=passed_filters,
                    failed_filters=failed_filters,
                    filter_results=filter_results,
                    reasons=reasons,
                    pit_safe=False,
                ))
                continue

            # 2. Data Availability
            raw_df = market_data_map.get(symbol)
            if raw_df is None or raw_df.empty:
                filter_results["DATA_AVAILABILITY"] = False
                failed_filters.append("DATA_AVAILABILITY")
                reasons.append("NO_DATA_AVAILABLE")
                results.append(CandidateDiscoveryResult(
                    symbol=symbol,
                    decision_time=as_of_date,
                    eligible=False,
                    passed_filters=passed_filters,
                    failed_filters=failed_filters,
                    filter_results=filter_results,
                    reasons=reasons,
                    pit_safe=True,
                ))
                continue

            # Apply PIT slicing to guarantee no future row is consumed
            sliced_df = PointInTimeFilter.filter_market_data(raw_df, as_of_date)
            if sliced_df.empty:
                filter_results["DATA_AVAILABILITY"] = False
                failed_filters.append("DATA_AVAILABILITY")
                reasons.append("NO_DATA_AVAILABLE_BEFORE_AS_OF_DATE")
                results.append(CandidateDiscoveryResult(
                    symbol=symbol,
                    decision_time=as_of_date,
                    eligible=False,
                    passed_filters=passed_filters,
                    failed_filters=failed_filters,
                    filter_results=filter_results,
                    reasons=reasons,
                    pit_safe=True,
                ))
                continue

            filter_results["DATA_AVAILABILITY"] = True
            passed_filters.append("DATA_AVAILABILITY")

            # 3. Data Quality Gate
            dq_result = DataQualityGate.evaluate_evidence_quality(
                symbol, sliced_df, as_of_date, min_required_bars=cfg.min_history_length
            )

            if not dq_result.pit_safe or dq_result.overall_status == DataQualityStatus.PIT_VIOLATION:
                filter_results["DATA_QUALITY"] = False
                failed_filters.append("DATA_QUALITY")
                reasons.append("PIT_VIOLATION")
                pit_safe = False
            elif dq_result.overall_status == DataQualityStatus.INVALID:
                filter_results["DATA_QUALITY"] = False
                failed_filters.append("DATA_QUALITY")
                reasons.append("DATA_QUALITY_INVALID")
            else:
                filter_results["DATA_QUALITY"] = True
                passed_filters.append("DATA_QUALITY")

            if not filter_results["DATA_QUALITY"]:
                results.append(CandidateDiscoveryResult(
                    symbol=symbol,
                    decision_time=as_of_date,
                    eligible=False,
                    passed_filters=passed_filters,
                    failed_filters=failed_filters,
                    filter_results=filter_results,
                    reasons=reasons,
                    data_quality=dq_result,
                    pit_safe=pit_safe,
                ))
                continue

            # 4. History Sufficiency
            if len(sliced_df) < cfg.min_history_length:
                filter_results["HISTORY_SUFFICIENCY"] = False
                failed_filters.append("HISTORY_SUFFICIENCY")
                reasons.append("INSUFFICIENT_HISTORY")
            else:
                filter_results["HISTORY_SUFFICIENCY"] = True
                passed_filters.append("HISTORY_SUFFICIENCY")

            # 5. Price Filter
            latest_close = float(sliced_df["close"].iloc[-1])
            if latest_close < cfg.min_price:
                filter_results["PRICE_RANGE"] = False
                failed_filters.append("PRICE_RANGE")
                reasons.append("PRICE_BELOW_MINIMUM")
            elif latest_close > cfg.max_price:
                filter_results["PRICE_RANGE"] = False
                failed_filters.append("PRICE_RANGE")
                reasons.append("PRICE_ABOVE_MAXIMUM")
            else:
                filter_results["PRICE_RANGE"] = True
                passed_filters.append("PRICE_RANGE")

            # 6. Liquidity Filter
            avg_volume = float(sliced_df["volume"].tail(20).mean())
            if "turnover_crores" in sliced_df.columns:
                adtv_crores = float(sliced_df["turnover_crores"].tail(20).mean())
            else:
                adtv_crores = float(((sliced_df["close"] * sliced_df["volume"]) / 1e7).tail(20).mean())

            if avg_volume < cfg.min_average_volume or adtv_crores < cfg.min_average_turnover_crores:
                filter_results["LIQUIDITY"] = False
                failed_filters.append("LIQUIDITY")
                reasons.append("INSUFFICIENT_LIQUIDITY")
            else:
                filter_results["LIQUIDITY"] = True
                passed_filters.append("LIQUIDITY")

            # 7. Basic Trend Alignment (Optional screening filter)
            if cfg.require_trend_alignment and len(sliced_df) >= 20:
                ema_20 = float(sliced_df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
                if latest_close < ema_20:
                    filter_results["TREND_ALIGNMENT"] = False
                    failed_filters.append("TREND_ALIGNMENT")
                    reasons.append("SCREEN_TREND_FAILED")
                else:
                    filter_results["TREND_ALIGNMENT"] = True
                    passed_filters.append("TREND_ALIGNMENT")

            eligible = len(failed_filters) == 0

            results.append(CandidateDiscoveryResult(
                symbol=symbol,
                decision_time=as_of_date,
                eligible=eligible,
                passed_filters=passed_filters,
                failed_filters=failed_filters,
                filter_results=filter_results,
                reasons=reasons,
                data_quality=dq_result,
                pit_safe=pit_safe,
                discovery_score=round(adtv_crores, 2),  # Screening metric: ADTV in ₹ Cr
            ))

        return results
