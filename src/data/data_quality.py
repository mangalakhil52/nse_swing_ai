"""
Data Quality & Evidence Control Layer Module — src/data/data_quality.py (P0 #13)

Provides deterministic evidence quality evaluation for OHLCV, fundamentals, news, market regime,
and benchmark data before quantitative calculation or downstream decisions.

Distinguishes:
  - SIGNAL STRENGTH
  - DATA QUALITY
  - DECISION CONFIDENCE

Hard Fails (PIT Violation, Invalid OHLC, Missing Columns) block trade eligibility (is_trade_eligible=False).
Soft Degradations (Missing optional news, stale data) reduce quality/confidence scores without fabricating evidence.
"""

from enum import Enum
from datetime import date, datetime, timedelta
import logging
from typing import Any
import pandas as pd
from pydantic import BaseModel, Field

from src.core.models import QuarterlyFinancials, NewsArticle

logger = logging.getLogger(__name__)


class DataQualityStatus(str, Enum):
    VALID = "VALID"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"
    UNAVAILABLE = "UNAVAILABLE"
    PIT_VIOLATION = "PIT_VIOLATION"


class SourceQualityResult(BaseModel):
    source_name: str
    status: DataQualityStatus
    quality_score: float = Field(..., ge=0.0, le=100.0)
    completeness_score: float = Field(..., ge=0.0, le=100.0)
    freshness_score: float = Field(..., ge=0.0, le=100.0)
    pit_safe: bool
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DataQualityResult(BaseModel):
    symbol: str
    as_of_date: date | datetime | None = None
    overall_status: DataQualityStatus
    overall_quality_score: float = Field(..., ge=0.0, le=100.0)
    pit_safe: bool
    is_trade_eligible: bool
    sources: dict[str, SourceQualityResult] = Field(default_factory=dict)
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DataQualityGate:
    """Deterministic gate evaluating multi-source evidence quality before decision-making."""

    @classmethod
    def evaluate_ohlcv(
        cls,
        df: pd.DataFrame | None,
        symbol: str,
        as_of_date: date | datetime | None = None,
        min_required_bars: int = 50,
        max_staleness_days: int = 4,
    ) -> SourceQualityResult:
        reasons: list[str] = []
        warnings: list[str] = []

        if df is None or df.empty:
            return SourceQualityResult(
                source_name="OHLCV",
                status=DataQualityStatus.INVALID,
                quality_score=0.0,
                completeness_score=0.0,
                freshness_score=0.0,
                pit_safe=True,
                reasons=["OHLC_EMPTY_DATAFRAME"],
            )

        # 1. Required columns
        required_cols = {"open", "high", "low", "close", "volume"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            return SourceQualityResult(
                source_name="OHLCV",
                status=DataQualityStatus.INVALID,
                quality_score=0.0,
                completeness_score=0.0,
                freshness_score=0.0,
                pit_safe=True,
                reasons=["OHLC_MISSING_COLUMN"],
                warnings=[f"Missing columns: {missing_cols}"],
            )

        # 2. Check for null values
        for col in ["open", "high", "low", "close", "volume"]:
            if df[col].isnull().any():
                reasons.append("OHLC_NULL_VALUE")
                break

        # 3. Check duplicate timestamps
        if "timestamp" in df.columns:
            if df["timestamp"].duplicated().any():
                reasons.append("OHLC_DUPLICATE_TIMESTAMP")

        # 4. Check price geometry
        invalid_high = df[
            (df["high"] < df["open"] - 1e-4)
            | (df["high"] < df["close"] - 1e-4)
            | (df["high"] < df["low"] - 1e-4)
        ]
        if not invalid_high.empty:
            reasons.append("OHLC_INVALID_GEOMETRY")

        invalid_low = df[
            (df["low"] > df["open"] + 1e-4)
            | (df["low"] > df["close"] + 1e-4)
            | (df["low"] > df["high"] + 1e-4)
        ]
        if not invalid_low.empty and "OHLC_INVALID_GEOMETRY" not in reasons:
            reasons.append("OHLC_INVALID_GEOMETRY")

        # 5. Non-positive prices
        non_pos = df[
            (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
        ]
        if not non_pos.empty:
            reasons.append("OHLC_NON_POSITIVE_PRICE")

        # 6. Negative volume
        neg_vol = df[df["volume"] < 0]
        if not neg_vol.empty:
            reasons.append("OHLC_NEGATIVE_VOLUME")

        # 7. Insufficient history
        if len(df) < min_required_bars:
            reasons.append("OHLC_INSUFFICIENT_BARS")

        # 8. PIT safety check
        pit_safe = True
        if as_of_date is not None and "timestamp" in df.columns:
            as_of_d = as_of_date.date() if isinstance(as_of_date, datetime) else as_of_date
            max_ts = pd.to_datetime(df["timestamp"]).max().date()
            if max_ts > as_of_d:
                pit_safe = False
                reasons.append("PIT_VIOLATION")

        # 9. Freshness
        freshness_score = 100.0
        if "timestamp" in df.columns and as_of_date is not None:
            latest_ts = pd.to_datetime(df["timestamp"]).max()
            ref_dt = pd.to_datetime(as_of_date)
            days_old = (ref_dt.date() - latest_ts.date()).days
            if days_old > max_staleness_days:
                reasons.append("OHLC_STALE_DATA")
                freshness_score = 30.0
            elif days_old > 2:
                freshness_score = 70.0

        # Determine status
        if "PIT_VIOLATION" in reasons:
            status = DataQualityStatus.PIT_VIOLATION
            quality_score = 0.0
        elif any(r in reasons for r in ["OHLC_MISSING_COLUMN", "OHLC_NULL_VALUE", "OHLC_INVALID_GEOMETRY", "OHLC_NON_POSITIVE_PRICE", "OHLC_NEGATIVE_VOLUME", "OHLC_DUPLICATE_TIMESTAMP"]):
            status = DataQualityStatus.INVALID
            quality_score = 0.0
        elif "OHLC_INSUFFICIENT_BARS" in reasons or "OHLC_STALE_DATA" in reasons:
            status = DataQualityStatus.DEGRADED
            quality_score = 60.0
        else:
            status = DataQualityStatus.VALID
            quality_score = 100.0

        completeness_score = min(100.0, (len(df) / max(min_required_bars, 1)) * 100.0)

        return SourceQualityResult(
            source_name="OHLCV",
            status=status,
            quality_score=quality_score,
            completeness_score=round(completeness_score, 1),
            freshness_score=round(freshness_score, 1),
            pit_safe=pit_safe,
            reasons=reasons,
            warnings=warnings,
        )

    @classmethod
    def evaluate_fundamentals(
        cls,
        financials: list[QuarterlyFinancials] | None,
        as_of_date: date | datetime | None = None,
    ) -> SourceQualityResult:
        if not financials:
            return SourceQualityResult(
                source_name="FUNDAMENTALS",
                status=DataQualityStatus.UNAVAILABLE,
                quality_score=0.0,
                completeness_score=0.0,
                freshness_score=0.0,
                pit_safe=True,
                reasons=["FUNDAMENTAL_UNAVAILABLE"],
            )

        reasons: list[str] = []
        pit_safe = True

        for f in financials:
            avail = getattr(f, "available_at", None) or getattr(f, "filing_date", None)
            if avail is None or getattr(f, "pit_status", "") == "PIT_UNVERIFIED":
                if "FUNDAMENTAL_PIT_UNVERIFIED" not in reasons:
                    reasons.append("FUNDAMENTAL_PIT_UNVERIFIED")
            elif as_of_date is not None:
                as_of_d = as_of_date.date() if isinstance(as_of_date, datetime) else as_of_date
                avail_d = avail if isinstance(avail, date) else pd.to_datetime(avail).date()
                if avail_d > as_of_d:
                    pit_safe = False
                    if "PIT_VIOLATION" not in reasons:
                        reasons.append("PIT_VIOLATION")

        if "PIT_VIOLATION" in reasons:
            status = DataQualityStatus.PIT_VIOLATION
            quality_score = 0.0
        elif "FUNDAMENTAL_PIT_UNVERIFIED" in reasons:
            status = DataQualityStatus.DEGRADED
            quality_score = 40.0
        else:
            status = DataQualityStatus.VALID
            quality_score = 100.0

        return SourceQualityResult(
            source_name="FUNDAMENTALS",
            status=status,
            quality_score=quality_score,
            completeness_score=100.0 if status == DataQualityStatus.VALID else 50.0,
            freshness_score=100.0 if pit_safe else 0.0,
            pit_safe=pit_safe,
            reasons=reasons,
        )

    @classmethod
    def evaluate_news(
        cls,
        articles: list[NewsArticle] | None,
        as_of_date: date | datetime | None = None,
    ) -> SourceQualityResult:
        if not articles:
            return SourceQualityResult(
                source_name="NEWS",
                status=DataQualityStatus.UNAVAILABLE,
                quality_score=0.0,
                completeness_score=0.0,
                freshness_score=0.0,
                pit_safe=True,
                reasons=["NEWS_UNAVAILABLE"],
            )

        reasons: list[str] = []
        pit_safe = True

        for a in articles:
            pub = getattr(a, "available_at", None) or getattr(a, "published_at", None)
            if pub is None:
                if "NEWS_TIMESTAMP_MISSING" not in reasons:
                    reasons.append("NEWS_TIMESTAMP_MISSING")
            elif as_of_date is not None:
                if isinstance(as_of_date, datetime):
                    pub_dt = pub if isinstance(pub, datetime) else datetime.combine(pub, datetime.min.time())
                    if pub_dt > as_of_date:
                        pit_safe = False
                        if "PIT_VIOLATION" not in reasons:
                            reasons.append("PIT_VIOLATION")
                else:
                    pub_d = pub.date() if isinstance(pub, datetime) else pub
                    if pub_d > as_of_date:
                        pit_safe = False
                        if "PIT_VIOLATION" not in reasons:
                            reasons.append("PIT_VIOLATION")

        if "PIT_VIOLATION" in reasons:
            status = DataQualityStatus.PIT_VIOLATION
            quality_score = 0.0
        elif "NEWS_TIMESTAMP_MISSING" in reasons:
            status = DataQualityStatus.DEGRADED
            quality_score = 50.0
        else:
            status = DataQualityStatus.VALID
            quality_score = 100.0

        return SourceQualityResult(
            source_name="NEWS",
            status=status,
            quality_score=quality_score,
            completeness_score=100.0 if status == DataQualityStatus.VALID else 50.0,
            freshness_score=100.0 if pit_safe else 0.0,
            pit_safe=pit_safe,
            reasons=reasons,
        )

    @classmethod
    def evaluate_regime(
        cls,
        nifty_df: pd.DataFrame | None,
        as_of_date: date | datetime | None = None,
    ) -> SourceQualityResult:
        if nifty_df is None or nifty_df.empty:
            return SourceQualityResult(
                source_name="MARKET_REGIME",
                status=DataQualityStatus.UNAVAILABLE,
                quality_score=0.0,
                completeness_score=0.0,
                freshness_score=0.0,
                pit_safe=True,
                reasons=["REGIME_UNAVAILABLE"],
            )

        pit_safe = True
        reasons: list[str] = []
        if as_of_date is not None and "timestamp" in nifty_df.columns:
            as_of_d = as_of_date.date() if isinstance(as_of_date, datetime) else as_of_date
            max_ts = pd.to_datetime(nifty_df["timestamp"]).max().date()
            if max_ts > as_of_d:
                pit_safe = False
                reasons.append("PIT_VIOLATION")

        status = DataQualityStatus.PIT_VIOLATION if not pit_safe else DataQualityStatus.VALID
        return SourceQualityResult(
            source_name="MARKET_REGIME",
            status=status,
            quality_score=100.0 if pit_safe else 0.0,
            completeness_score=100.0 if len(nifty_df) >= 50 else 50.0,
            freshness_score=100.0 if pit_safe else 0.0,
            pit_safe=pit_safe,
            reasons=reasons,
        )

    @classmethod
    def evaluate_benchmark(
        cls,
        benchmark_df: pd.DataFrame | None,
        as_of_date: date | datetime | None = None,
    ) -> SourceQualityResult:
        if benchmark_df is None or benchmark_df.empty:
            return SourceQualityResult(
                source_name="BENCHMARK",
                status=DataQualityStatus.UNAVAILABLE,
                quality_score=0.0,
                completeness_score=0.0,
                freshness_score=0.0,
                pit_safe=True,
                reasons=["BENCHMARK_UNAVAILABLE"],
            )

        pit_safe = True
        reasons: list[str] = []
        if as_of_date is not None and "timestamp" in benchmark_df.columns:
            as_of_d = as_of_date.date() if isinstance(as_of_date, datetime) else as_of_date
            max_ts = pd.to_datetime(benchmark_df["timestamp"]).max().date()
            if max_ts > as_of_d:
                pit_safe = False
                reasons.append("PIT_VIOLATION")

        status = DataQualityStatus.PIT_VIOLATION if not pit_safe else DataQualityStatus.VALID
        return SourceQualityResult(
            source_name="BENCHMARK",
            status=status,
            quality_score=100.0 if pit_safe else 0.0,
            completeness_score=100.0,
            freshness_score=100.0 if pit_safe else 0.0,
            pit_safe=pit_safe,
            reasons=reasons,
        )

    @classmethod
    def evaluate_evidence_quality(
        cls,
        symbol: str,
        df: pd.DataFrame | None,
        as_of_date: date | datetime | None = None,
        fundamentals: list[QuarterlyFinancials] | None = None,
        news: list[NewsArticle] | None = None,
        regime_df: pd.DataFrame | None = None,
        benchmark_df: pd.DataFrame | None = None,
        min_required_bars: int = 50,
        max_staleness_days: int = 4,
    ) -> DataQualityResult:
        """
        Runs comprehensive multi-source quality evaluation.
        HARD FAILS (PIT violation, Invalid OHLC, Empty OHLC) set is_trade_eligible = False.
        """
        symbol = symbol.upper().strip()

        ohlcv_res = cls.evaluate_ohlcv(df, symbol, as_of_date, min_required_bars, max_staleness_days)
        fund_res = cls.evaluate_fundamentals(fundamentals, as_of_date)
        news_res = cls.evaluate_news(news, as_of_date)
        regime_res = cls.evaluate_regime(regime_df, as_of_date)
        bench_res = cls.evaluate_benchmark(benchmark_df, as_of_date)

        sources = {
            "OHLCV": ohlcv_res,
            "FUNDAMENTALS": fund_res,
            "NEWS": news_res,
            "MARKET_REGIME": regime_res,
            "BENCHMARK": bench_res,
        }

        blocking_reasons: list[str] = []
        warnings: list[str] = []

        pit_safe = all(s.pit_safe for s in sources.values())

        # Determine overall status and trade eligibility
        if not pit_safe or any(s.status == DataQualityStatus.PIT_VIOLATION for s in sources.values()):
            overall_status = DataQualityStatus.PIT_VIOLATION
            is_trade_eligible = False
            blocking_reasons.append("PIT_VIOLATION")
        elif ohlcv_res.status == DataQualityStatus.INVALID:
            overall_status = DataQualityStatus.INVALID
            is_trade_eligible = False
            blocking_reasons.extend(ohlcv_res.reasons)
        elif ohlcv_res.status == DataQualityStatus.UNAVAILABLE:
            overall_status = DataQualityStatus.UNAVAILABLE
            is_trade_eligible = False
            blocking_reasons.append("OHLC_UNAVAILABLE")
        elif any(s.status == DataQualityStatus.DEGRADED for s in sources.values()):
            overall_status = DataQualityStatus.DEGRADED
            is_trade_eligible = True  # Degraded optional sources can proceed with reduced score
            warnings.extend([r for s in sources.values() for r in s.reasons if r != "PIT_VIOLATION"])
        else:
            overall_status = DataQualityStatus.VALID
            is_trade_eligible = True

        # Calculate deterministic overall quality score
        if overall_status == DataQualityStatus.PIT_VIOLATION or overall_status == DataQualityStatus.INVALID:
            overall_quality_score = 0.0
        else:
            valid_scores = [s.quality_score for s in sources.values() if s.status != DataQualityStatus.UNAVAILABLE]
            overall_quality_score = round(sum(valid_scores) / max(len(valid_scores), 1), 1)

        return DataQualityResult(
            symbol=symbol,
            as_of_date=as_of_date,
            overall_status=overall_status,
            overall_quality_score=overall_quality_score,
            pit_safe=pit_safe,
            is_trade_eligible=is_trade_eligible,
            sources=sources,
            blocking_reasons=list(set(blocking_reasons)),
            warnings=list(set(warnings)),
        )
