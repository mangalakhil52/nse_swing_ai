"""Deterministic Stage-1 candidate discovery with strict PIT/survivorship controls."""

from __future__ import annotations

from datetime import date, datetime
from typing import Sequence

import pandas as pd
from pydantic import BaseModel, Field

from config.settings import settings
from src.core.models import SymbolMetadata
from src.data.data_quality import DataQualityGate, DataQualityResult, DataQualityStatus
from src.data.historical_universe import HistoricalUniverseProvider
from src.data.point_in_time import PointInTimeFilter


class CandidateDiscoveryConfig(BaseModel):
    min_price: float = Field(default=settings.MIN_STOCK_PRICE, ge=0.0)
    min_average_volume: float = Field(default=0.0, ge=0.0)
    min_average_turnover_crores: float = Field(default=settings.MIN_ADTV_CRORES, ge=0.0)
    min_history_length: int = Field(default=50, ge=1)
    liquidity_window: int = Field(default=20, ge=1)
    require_trend_alignment: bool = Field(default=False)


class CandidateDiscoveryResult(BaseModel):
    symbol: str
    decision_time: datetime | date
    eligible: bool
    passed_filters: list[str] = Field(default_factory=list)
    failed_filters: list[str] = Field(default_factory=list)
    filter_results: dict[str, bool] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    data_quality: DataQualityResult | None = None
    pit_safe: bool = False
    discovery_score: float | None = Field(default=None, description="Screening metric only; not conviction.")


class CandidateDiscoveryEngine:
    """Fast, deterministic universe-to-candidate funnel."""

    @classmethod
    def _resolve_mode(cls, universe: Sequence[str] | Sequence[SymbolMetadata], mode: str | None) -> str:
        if mode is not None:
            normalized = mode.upper()
            if normalized not in {"HISTORICAL", "LIVE"}:
                raise ValueError("mode must be 'HISTORICAL' or 'LIVE'")
            return normalized
        return "HISTORICAL" if universe and isinstance(universe[0], SymbolMetadata) else "LIVE"

    @classmethod
    def _resolve_universe(cls, universe: Sequence[str] | Sequence[SymbolMetadata] | None, as_of_date: datetime | date, mode: str) -> list[str]:
        as_of = as_of_date.date() if isinstance(as_of_date, datetime) else as_of_date
        if universe is None:
            if mode == "HISTORICAL":
                raise TypeError("Historical candidate discovery requires explicit SymbolMetadata with listing/delisting metadata")
            return HistoricalUniverseProvider.get_current_universe()
        if mode == "HISTORICAL":
            metadata = list(universe)
            if not metadata or not all(isinstance(item, SymbolMetadata) for item in metadata):
                raise TypeError("Historical candidate discovery requires SymbolMetadata with listing/delisting metadata")
            return HistoricalUniverseProvider.get_universe_for_date(as_of, securities=metadata)
        return [item.symbol.strip().upper() if isinstance(item, SymbolMetadata) else str(item).strip().upper() for item in universe]

    @staticmethod
    def _empty_result(symbol: str, as_of_date: datetime | date, reason: str, data_quality: DataQualityResult | None = None, pit_safe: bool = False) -> CandidateDiscoveryResult:
        return CandidateDiscoveryResult(
            symbol=symbol,
            decision_time=as_of_date,
            eligible=False,
            failed_filters=[reason],
            filter_results={reason: False},
            reasons=[reason],
            data_quality=data_quality,
            pit_safe=pit_safe,
        )

    @classmethod
    def discover_candidates(cls, universe: Sequence[SymbolMetadata] | Sequence[str] | None, as_of_date: datetime | date, market_data_map: dict[str, pd.DataFrame], config: CandidateDiscoveryConfig | None = None, mode: str | None = None) -> list[CandidateDiscoveryResult]:
        cfg = config or CandidateDiscoveryConfig()
        resolved_mode = (mode or ("HISTORICAL" if universe and isinstance(universe[0], SymbolMetadata) else "LIVE")).upper()
        if resolved_mode not in {"HISTORICAL", "LIVE"}:
            raise ValueError("mode must be 'HISTORICAL' or 'LIVE'")
        resolved_universe = cls._resolve_universe(universe, as_of_date, resolved_mode)
        results: list[CandidateDiscoveryResult] = []

        for symbol in resolved_universe:
            symbol = symbol.strip().upper()
            raw_df = market_data_map.get(symbol)
            if raw_df is None:
                raw_df = market_data_map.get(symbol.lower())
            if raw_df is None or raw_df.empty:
                results.append(cls._empty_result(symbol, as_of_date, "NO_DATA_AVAILABLE", pit_safe=False))
                continue

            sliced_df = PointInTimeFilter.filter_market_data(raw_df, as_of_date)
            if sliced_df is None or sliced_df.empty:
                results.append(cls._empty_result(symbol, as_of_date, "NO_DATA_AVAILABLE_BEFORE_AS_OF_DATE", pit_safe=True))
                continue

            dq_source = DataQualityGate.evaluate_ohlcv(sliced_df, symbol, as_of_date=as_of_date, min_required_bars=cfg.min_history_length)
            if dq_source.pit_safe is False or dq_source.status == DataQualityStatus.PIT_VIOLATION:
                dq_result = DataQualityResult(
                    symbol=symbol, as_of_date=as_of_date, overall_status=DataQualityStatus.PIT_VIOLATION,
                    overall_quality_score=0.0, pit_safe=False, is_trade_eligible=False,
                    sources={"OHLCV": dq_source}, blocking_reasons=["PIT_VIOLATION"], warnings=list(dq_source.warnings),
                )
                results.append(cls._empty_result(symbol, as_of_date, "PIT_VIOLATION", dq_result, False))
                continue

            if dq_source.status == DataQualityStatus.INVALID:
                dq_result = DataQualityResult(
                    symbol=symbol, as_of_date=as_of_date, overall_status=DataQualityStatus.INVALID,
                    overall_quality_score=0.0, pit_safe=dq_source.pit_safe, is_trade_eligible=False,
                    sources={"OHLCV": dq_source}, blocking_reasons=list(dq_source.reasons), warnings=list(dq_source.warnings),
                )
                results.append(CandidateDiscoveryResult(
                    symbol=symbol, decision_time=as_of_date, eligible=False,
                    passed_filters=["DATA_AVAILABILITY"], failed_filters=["DATA_QUALITY"],
                    filter_results={"DATA_AVAILABILITY": True, "DATA_QUALITY": False},
                    reasons=list(dq_source.reasons) or ["DATA_QUALITY_INVALID"], data_quality=dq_result, pit_safe=dq_source.pit_safe,
                ))
                continue

            dq_result = DataQualityResult(
                symbol=symbol, as_of_date=as_of_date, overall_status=dq_source.status,
                overall_quality_score=dq_source.quality_score, pit_safe=dq_source.pit_safe,
                is_trade_eligible=dq_source.pit_safe, sources={"OHLCV": dq_source}, blocking_reasons=[], warnings=list(dq_source.warnings),
            )
            passed = ["DATA_AVAILABILITY", "DATA_QUALITY"]
            failed: list[str] = []
            reasons: list[str] = []
            filter_results = {"DATA_AVAILABILITY": True, "DATA_QUALITY": True}

            history_ok = len(sliced_df) >= cfg.min_history_length
            filter_results["HISTORY_SUFFICIENCY"] = history_ok
            (passed if history_ok else failed).append("HISTORY_SUFFICIENCY")
            if not history_ok:
                reasons.append("INSUFFICIENT_HISTORY")

            latest = sliced_df.sort_values("timestamp").iloc[-1] if "timestamp" in sliced_df.columns else sliced_df.iloc[-1]
            latest_close = float(latest["close"])
            price_ok = latest_close >= cfg.min_price
            filter_results["PRICE_RANGE"] = price_ok
            (passed if price_ok else failed).append("PRICE_RANGE")
            if not price_ok:
                reasons.append("PRICE_BELOW_MINIMUM")

            window = sliced_df.sort_values("timestamp").tail(cfg.liquidity_window) if "timestamp" in sliced_df.columns else sliced_df.tail(cfg.liquidity_window)
            avg_volume = float(window["volume"].astype(float).mean())
            adtv_crores = float(window["turnover_crores"].astype(float).mean()) if "turnover_crores" in window.columns else float((window["close"].astype(float) * window["volume"].astype(float)).mean() / 1e7)
            liquidity_ok = avg_volume >= cfg.min_average_volume and adtv_crores >= cfg.min_average_turnover_crores
            filter_results["LIQUIDITY"] = liquidity_ok
            (passed if liquidity_ok else failed).append("LIQUIDITY")
            if not liquidity_ok:
                reasons.append("INSUFFICIENT_LIQUIDITY")

            if cfg.require_trend_alignment:
                if len(sliced_df) < 20:
                    trend_ok = False
                    reasons.append("INSUFFICIENT_HISTORY_FOR_TREND")
                else:
                    closes = sliced_df["close"].astype(float)
                    ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
                    trend_ok = latest_close >= ema20
                    if not trend_ok:
                        reasons.append("SCREEN_TREND_FAILED")
                filter_results["TREND_ALIGNMENT"] = trend_ok
                (passed if trend_ok else failed).append("TREND_ALIGNMENT")

            results.append(CandidateDiscoveryResult(
                symbol=symbol, decision_time=as_of_date, eligible=not failed,
                passed_filters=passed, failed_filters=failed, filter_results=filter_results,
                reasons=reasons, data_quality=dq_result, pit_safe=dq_result.pit_safe,
                discovery_score=round(adtv_crores, 2),
            ))
        return results
